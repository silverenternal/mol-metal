"""Metric 18 — Protein-Metal Binding prediction head, target AUC ≥ 0.85.

Implements the multimodal protein-metal binding prediction head from
:mod:`molmetal.molmetal_lam.sbdd_env.protein_metal_binding` on top of the
MetalCytoToxDB-derived CSV ``molmetal/data/protein_metal_binding.csv``.

Pipeline
--------
1. Load the CSV. If the file is missing, build it via
   :func:`protein_metal_binding.build_protein_metal_csv` (filters to
   metals Ru/Ir/Rh/Os/Re, 24 h ≤ Time, 0.01 µM ≤ IC50 ≤ 1000 µM).
2. 70/10/20 scaffold split (Bemis-Murcko) — same protocol as Metric 1
   so we can compare apples-to-apples.
3. Train the multimodal fusion head (cell-line + metal + D-MPNN ligand +
   optional pocket) with focal loss for 50 epochs.
4. Evaluate ROC-AUC + PR-AUC on the held-out test split. Aggregate
   across N seeds (default 3). Write to ``metrics/by_metric/metric18_protein_metal.json``.

Verification gate
-----------------
AUC ≥ 0.85 ⇒ ✅; AUC ≥ 0.92 ⇒ 🌟 (matches Tier 1 SOTA per METRICS_TARGETS §2).

Usage
-----
    source env.sh
    /home/hugo/codes/mol-metal/.venv/bin/python -u molmetal/scripts/train_protein_metal.py \
        --data molmetal/data/protein_metal_binding.csv \
        --out-dir models/protein_metal_binding --epochs 50
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from molmetal.molmetal_lam.sbdd_env.protein_metal_binding import (
    PMBConfig,
    PMBDataset,
    PMBVocab,
    ProteinMetalBindingHead,
    collate_pmb,
    focal_loss,
    roc_auc_safe,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "train_protein_metal.py needs scikit-learn: uv pip install scikit-learn"
    ) from exc


METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
DEFAULT_DATA = PROJECT_ROOT / "molmetal" / "data" / "protein_metal_binding.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train protein-metal binding head (Metric 18)."
    )
    p.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="CSV with smiles/metal/cell_line/active columns.",
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric18_protein_metal",
        help="Directory for per-seed checkpoints & metrics.",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help="Path to write aggregate metric JSON. Default: metrics/by_metric/metric18_protein_metal.json",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Path to write markdown report. Default: molmetal/reports/wf_protein_metal_binding/final.md",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--build-data",
        action="store_true",
        help="Force-rebuild the data CSV from MetalCytoToxDB before training.",
    )
    p.add_argument("--quick", action="store_true", help="Use 1000 rows for smoke run.")
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on val AUC (0 disables).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Scaffold split (Bemis-Murcko) on the *ligand* SMILES
# ---------------------------------------------------------------------------
def scaffold_split_indices(
    smiles: np.ndarray,
    seed: int,
    fractions: Tuple[float, float, float] = (0.7, 0.1, 0.2),
):
    """Return (train, val, test) row indices via scaffold split.

    We leverage :mod:`molmetal.data.splits.ScaffoldSplitter` when it's
    importable, else fall back to a deterministic hashed scaffold bucket.
    """
    try:
        from molmetal.data.splits import ScaffoldSplitter

        splitter = ScaffoldSplitter(
            strategy="largest_first", fractions=fractions, seed=seed
        )
        # The splitter expects a SMILES list and returns index partitions.
        idx_train, idx_val, idx_test = splitter.split(smiles.tolist())
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)
    except Exception:
        # Fallback: deterministic permutation by Murcko scaffold hashing.
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            # No scaffold available → random split.
            rs = np.random.RandomState(seed)
            perm = rs.permutation(len(smiles))
            n = len(smiles)
            n_train = int(fractions[0] * n)
            n_val = int(fractions[1] * n)
            return (
                perm[:n_train],
                perm[n_train : n_train + n_val],
                perm[n_train + n_val :],
            )
        scaffolds = []
        for smi in smiles:
            try:
                core = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi))
                scaffolds.append(Chem.MolToSmiles(core) if core else smi)
            except Exception:
                scaffolds.append(smi)
        # Largest-first: bucket by scaffold, shuffle buckets by seed.
        from collections import defaultdict

        buckets = defaultdict(list)
        for i, sc in enumerate(scaffolds):
            buckets[sc].append(i)
        keys = list(buckets.keys())
        rs = np.random.RandomState(seed)
        rs.shuffle(keys)
        n = len(smiles)
        n_train = int(fractions[0] * n)
        n_val = int(fractions[1] * n)
        train, val, test = [], [], []
        acc = 0
        for k in keys:
            ids = buckets[k]
            if acc < n_train:
                train.extend(ids)
            elif acc < n_train + n_val:
                val.extend(ids)
            else:
                test.extend(ids)
            acc += len(ids)
        return np.array(train), np.array(val), np.array(test)


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------
def train_one(
    df: pd.DataFrame,
    seed: int,
    cfg: PMBConfig,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Train protein-metal binding head with one RNG seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.seed = seed
    smiles = df["smiles"].astype(str).to_numpy()
    idx_train, idx_val, idx_test = scaffold_split_indices(smiles, seed)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    vocab = PMBVocab.from_df(df)
    cfg.device = args.device
    head = ProteinMetalBindingHead(
        n_cell_lines=vocab.n_cell_lines(),
        n_metals=vocab.n_metals(),
        cfg=cfg,
    ).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    train_ds = PMBDataset(df_train, vocab)
    val_ds = PMBDataset(df_val, vocab)
    test_ds = PMBDataset(df_test, vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pmb,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pmb,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pmb,
        num_workers=0,
    )

    def _run_eval(loader) -> Tuple[float, float]:
        head.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch in loader:
                logits = head(
                    batch["smiles"],
                    batch["cell_line_id"].to(args.device),
                    batch["metal_id"].to(args.device),
                )
                all_logits.append(logits.cpu().numpy())
                all_y.append(batch["y"].cpu().numpy())
        y = np.concatenate(all_y)
        logits = np.concatenate(all_logits, axis=0)
        scores = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
        return roc_auc_safe(y, scores), float(
            average_precision_score(y, scores)
        ) if len(set(y.tolist())) > 1 else 0.5

    best_val_auc = -1.0
    best_test = (0.5, 0.5)
    no_improve = 0
    for ep in range(args.epochs):
        head.train()
        ep_loss = 0.0
        nb = 0
        for batch in train_loader:
            opt.zero_grad()
            logits = head(
                batch["smiles"],
                batch["cell_line_id"].to(args.device),
                batch["metal_id"].to(args.device),
            )
            labels = batch["y"].to(args.device)
            loss = focal_loss(
                logits,
                labels,
                gamma=args.focal_gamma,
                alpha=cfg.focal_alpha,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        val_auc, val_pr = _run_eval(val_loader)
        test_auc, test_pr = _run_eval(test_loader)
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_test = (test_auc, test_pr)
            no_improve = 0
        else:
            no_improve += 1
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(
                f"[seed {seed} ep {ep:3d}] loss={ep_loss / max(nb, 1):.4f} "
                f"val_auc={val_auc:.4f} val_pr={val_pr:.4f} "
                f"test_auc={test_auc:.4f} test_pr={test_pr:.4f}"
            )
        if args.patience > 0 and no_improve >= args.patience:
            print(
                f"[seed {seed}] early stop at ep {ep}; best val AUC {best_val_auc:.4f}"
            )
            break

    test_auc, test_pr = best_test
    return {
        "best_val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "test_pr_auc": float(test_pr),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "n_cell_lines": vocab.n_cell_lines(),
        "n_metals": vocab.n_metals(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = PMBConfig()

    # Optionally rebuild the data CSV.
    if args.build_data or not Path(args.data).exists():
        if args.build_data or not Path(args.data).exists():
            from molmetal.molmetal_lam.sbdd_env.protein_metal_binding import (
                build_protein_metal_csv,
            )

            print(f"[metric18] building CSV at {args.data}")
            stats = build_protein_metal_csv(dst_csv=str(args.data))
            print(f"[metric18] build stats: {stats}")

    df = pd.read_csv(args.data)
    if args.quick:
        df = df.sample(n=min(1000, len(df)), random_state=0).reset_index(drop=True)
    print(
        f"[metric18] loaded {len(df)} rows | "
        f"{df['active'].sum()} positives | "
        f"{(df['active'] == 0).sum()} negatives | "
        f"{df['cell_line'].nunique()} cell lines | "
        f"{df['metal'].nunique()} metals"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    t0 = time.time()
    for seed in args.seeds:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        result = train_one(df, seed, cfg, args)
        per_seed.append(result)
        ck = out_dir / f"seed{seed}_metrics.json"
        ck.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - t0

    aucs = [r["test_auc"] for r in per_seed]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    prs = [r["test_pr_auc"] for r in per_seed]
    mean_pr = float(np.mean(prs))

    agg = {
        "metric": "metric18_protein_metal_binding_auc",
        "seeds": list(args.seeds),
        "epochs": int(args.epochs),
        "test_auc_mean": mean_auc,
        "test_auc_std": std_auc,
        "test_auc_per_seed": aucs,
        "test_pr_auc_mean": mean_pr,
        "test_pr_auc_per_seed": prs,
        "verdict": _verdict(mean_auc),
        "elapsed_sec": round(elapsed, 1),
        "n_rows": int(len(df)),
        "n_cell_lines": int(df["cell_line"].nunique()),
        "n_metals": int(df["metal"].nunique()),
    }

    out_metric = args.out_metric or str(METRICS_DIR / "metric18_protein_metal.json")
    Path(out_metric).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metric).write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"[metric18] wrote metric to {out_metric} (elapsed {elapsed:.1f} s)")
    return 0


def _verdict(auc: float) -> str:
    if auc < 0.7:
        return "below_baseline"
    if auc < 0.85:
        return "baseline"
    if auc < 0.92:
        return "good (≥ 0.85 target met)"
    return "matches_tier1_sota (≥ 0.92)"


if __name__ == "__main__":
    sys.exit(main())
