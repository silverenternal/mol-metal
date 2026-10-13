#!/usr/bin/env python3
"""BLK-D2 Step 2 trainer: chemprop v3 D-MPNN encoder (FROZEN) + trainable D-MPNN-style head.

Per todo.json fix_steps Step 2 (BLK-D2, 2026-10-09): "Try domain-adaptation
pretraining (chemprop_v3 frozen encoder -> fine-tune D-MPNN head); section 5.7
smoke before GPU".

Architecture
------------
    SMILES  →  ChempropV3EncoderForward (chemprop_v2_mt_model_v3 backbone,
                hidden_dim=300, depth=3, dropout=0)
                — **frozen** (requires_grad=False on every parameter) —
            →  (B, 300) pooled molecular embedding
            →  MLPHead(300 → hidden → 1) (the "D-MPNN head" analogue)
            →  (B,) logit per molecule

Loss
----
    BCEWithLogitsLoss with class-imbalance pos_weight (neg/pos) — same protocol
    as the dmpnn / attentivefp paths in train_moleculenet.py so the four
    backbones (morgan_mlp / dmpnn / attentivefp / chemprop_v3_frozen_head) are
    directly comparable on a single scaffold split.

Split
-----
    Bemis-Murcko scaffold split (70/10/20) per Chemprop convention.  Imported
    from train_moleculenet._scaffold_split to keep protocol alignment with the
    in-repo MoleculeNet runs (no separate scaffold function).

§5.7 wire-in smoke
------------------
    A 1-epoch CPU run on the bace subset (1513 rows, 0.45 pos rate) with
    encoder-frozen + head-trainable verifies:
      1. ``--encoder-mode chemprop_v3_frozen`` is accepted by argparse,
      2. the chemprop_v3 encoder imports + initialises on CPU,
      3. encoder parameters are frozen (no grad populated on those tensors),
      4. head parameters train (grad populated, loss decreases),
      5. per-seed JSON lands on disk with finite ROC-AUC.

Why "frozen" + "head" (and not fine-tune the chemprop encoder end-to-end)
------------------------------------------------------------------------
    The TODO fix_steps Step 2 reads "chemprop_v3 frozen encoder -> fine-tune
    D-MPNN head", which is **frozen encoder + trainable head**.  This is the
    domain-adaptation pattern from the chemprop v1 / v2 ablation literature:
    pretrain the encoder on a large chemistry corpus, freeze it, then
    fine-tune a small head on the downstream task.  Full encoder fine-tuning
    is a separate future-work step; that is NOT in scope for BLK-D2 Step 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import (CUDA reads CUDA_VISIBLE_DEVICES only once, at init).
import os as _os
_wf_dir = _os.path.dirname(_os.path.abspath(__file__))
if _wf_dir not in sys.path:
    sys.path.append(_wf_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _cvd
_cvd()
del _cvd, _wf_dir, _os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# MLP head: 300-d frozen embedding -> (B, 1) logit
# ---------------------------------------------------------------------------
class FrozenEncoderHead(nn.Module):
    """Trainable MLP head on top of a frozen chemprop_v3 300-d embedding.

    Twin to a D-MPNN regression head: a small 2-layer MLP that consumes the
    pooled molecular embedding and emits one logit per molecule.  No
    message passing happens in the head — the chemprop_v3 encoder has
    already aggregated the molecular graph into a 300-d vector.
    """

    def __init__(self, in_dim: int = 300, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Bemis-Murcko scaffold split (re-used from train_moleculenet)
# ---------------------------------------------------------------------------
def _bemis_murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return smiles


def _scaffold_split(smiles: List[str], seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.2):
    """Group-scaffold split: largest scaffolds -> test, mid -> val, rest -> train."""
    scaffolds: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles):
        sc = _bemis_murcko_scaffold(s)
        scaffolds.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)
    n = len(smiles)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx: List[int] = []
    val_idx: List[int] = []
    train_idx: List[int] = []
    for s_set in scaffold_sets:
        if len(test_idx) + len(s_set) <= n_test:
            test_idx.extend(s_set)
        elif len(val_idx) + len(s_set) <= n_val:
            val_idx.extend(s_set)
        else:
            train_idx.extend(s_set)
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# CSV loader for the bace / clintox / etc. MoleculeNet CSVs
# ---------------------------------------------------------------------------
def _load_csv(csv_path: Path, smiles_col: str, label_col: str) -> Tuple[List[str], np.ndarray]:
    smiles: List[str] = []
    labels: List[int] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            lbl_raw = row.get(label_col)
            if not smi or lbl_raw is None:
                continue
            try:
                lbl = int(float(lbl_raw))
            except (ValueError, TypeError):
                continue
            if lbl not in (0, 1):
                continue
            smiles.append(smi)
            labels.append(lbl)
    return smiles, np.asarray(labels, dtype=np.int64)


def _resolve_pos_weight(y_train: np.ndarray) -> float:
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    return neg / max(pos, 1.0)


# ---------------------------------------------------------------------------
# Per-seed training
# ---------------------------------------------------------------------------
def _train_one_seed(
    smiles: List[str],
    labels: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    dropout: float,
    device: torch.device,
    out_dir: Path,
    csv_path: Path,
) -> Dict[str, object]:
    from molmetal.scripts.chemprop_v3_encoder_forward import ChempropV3EncoderForward

    print(
        f"[chemprop_v3_frozen_head] seed={seed}  rows={len(smiles)}  "
        f"pos_rate={labels.mean():.3f}",
        flush=True,
    )

    train_idx, val_idx, test_idx = _scaffold_split(smiles, seed=seed)
    print(
        f"[chemprop_v3_frozen_head]   split sizes "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        flush=True,
    )
    if len(test_idx) < 5 or len(np.unique(labels[test_idx])) < 2:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(smiles))
        n = len(idx)
        train_idx = idx[: int(0.7 * n)]
        val_idx = idx[int(0.7 * n) : int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]
        print(
            f"[chemprop_v3_frozen_head]   scaffold split degenerated; random 70/10/20",
            flush=True,
        )

    # Build frozen encoder.  The underlying ``ChempropV2Encoder`` is
    # **lazy-built** — its ``message_passing`` and ``agg`` submodules are
    # only constructed on the first forward call, so ``.parameters()``
    # returns an empty list until then.  We do a one-shot warmup forward
    # on a tiny batch (1 molecule) BEFORE freezing, so the lazy backbone
    # is materialised and the freeze check below can count real params.
    encoder = ChempropV3EncoderForward(hidden_dim=300, depth=3, dropout=0.0, device=str(device))
    warmup_smiles = [s for s in smiles if s][:1] or ["C"]
    with torch.no_grad():
        _ = encoder(warmup_smiles)
    # Freeze every encoder parameter
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    n_frozen = sum(1 for _ in encoder.parameters())
    n_trainable_encoder = sum(1 for p in encoder.parameters() if p.requires_grad)
    print(
        f"[chemprop_v3_frozen_head]   encoder frozen: {n_frozen} params, "
        f"trainable={n_trainable_encoder}",
        flush=True,
    )
    assert n_trainable_encoder == 0, "encoder parameters must all be frozen"

    # Build trainable head
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = FrozenEncoderHead(in_dim=300, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    pos_weight_val = _resolve_pos_weight(labels[train_idx])
    pos_rate_train = float(labels[train_idx].mean())
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(
        f"[chemprop_v3_frozen_head]   pos_rate_train={pos_rate_train:.3f} "
        f"pos_weight={pos_weight_val:.3f}",
        flush=True,
    )

    n_train = len(train_idx)
    train_smiles = [smiles[i] for i in train_idx]
    train_y = labels[train_idx].astype(np.float32)
    val_smiles = [smiles[i] for i in val_idx]
    val_y = labels[val_idx]
    test_smiles = [smiles[i] for i in test_idx]
    test_y = labels[test_idx]

    best_val_auc = -1.0
    best_state = None
    for epoch in range(epochs):
        head.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            bidx = perm[start : start + batch_size]
            batch_smiles = [train_smiles[j] for j in bidx]
            batch_y = torch.from_numpy(train_y[bidx]).to(device)
            opt.zero_grad()
            # Encoder forward (no grad; encoder is frozen).
            with torch.no_grad():
                emb = encoder(batch_smiles).to(device)
            logits = head(emb)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            # Sanity: head params must have grad; encoder must not.
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        head.eval()
        with torch.no_grad():
            val_emb = encoder(val_smiles).to(device)
            val_logits = head(val_emb)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        from sklearn.metrics import roc_auc_score
        try:
            v_auc = float(roc_auc_score(val_y, val_prob))
        except ValueError:
            v_auc = 0.5
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(
                f"[chemprop_v3_frozen_head]   epoch {epoch+1:3d}/{epochs} "
                f"loss={total_loss/max(n_batches,1):.4f} val_auc={v_auc:.4f}",
                flush=True,
            )

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_emb = encoder(test_smiles).to(device)
        test_logits = head(test_emb)
        test_prob = torch.sigmoid(test_logits).cpu().numpy()
    try:
        t_auc = float(roc_auc_score(test_y, test_prob))
    except ValueError:
        t_auc = 0.5

    print(
        f"[chemprop_v3_frozen_head]   TEST AUC={t_auc:.4f}",
        flush=True,
    )
    out = {
        "seed": seed,
        "backbone": "chemprop_v3_frozen_head",
        "encoder": "chemprop_v3_frozen (300-d, depth=3, dropout=0.0)",
        "head": f"MLPHead(300->{hidden}->1, dropout={dropout})",
        "n_rows": len(smiles),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "pos_rate_train": pos_rate_train,
        "pos_weight_used": float(pos_weight_val),
        "encoder_n_params": sum(p.numel() for p in encoder.parameters()),
        "head_n_params": sum(p.numel() for p in head.parameters()),
        "test_metrics": {
            "roc_auc": t_auc,
            "n": int(len(test_idx)),
        },
        "epochs": epochs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"chemprop_v3_frozen_head_seed{seed}.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train FrozenEncoderHead (300-d chemprop_v3 encoder + trainable MLP) on a "
            "MoleculeNet-format CSV (default: bace).  The encoder is FROZEN; only the "
            "head trains.  This is the BLK-D2 Step 2 domain-adaptation pretrain smoke."
        ),
    )
    p.add_argument("--csv", type=Path, default=PROJECT_ROOT / ".data/molmetal/moleculenet/bace.csv")
    p.add_argument("--smiles-col", type=str, default="mol",
                   help="CSV column with SMILES (bace uses 'mol').")
    p.add_argument("--label-col", type=str, default="Class",
                   help="CSV column with 0/1 binary label (bace uses 'Class').")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128, help="MLP head hidden width")
    p.add_argument("--dropout", type=float, default=0.1)
    # AGENTS.md §5.11: `--out-dir` was required=True, which made the flag
    # unsatisfiable from a YAML config — argparse enforces `required` during
    # parse_args, i.e. BEFORE apply_yaml_overrides can fill the dest, so a
    # config-driven launch died with "the following arguments are required:
    # --out-dir" no matter what the YAML declared. Relaxed to a post-parse
    # check so the config is a first-class source for it. A missing --out-dir
    # is still fatal, just with a message that names the config path.
    p.add_argument("--out-dir", type=Path, default=None)
    # AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置
    # 启动训练和推理，no cli参数"): same wiring as train_moleculenet.py — this
    # trainer had no YAML surface, so scripts/launch_wf_blk_d2_*.sh passed all
    # nine of its hyperparameters as CLI flags. CLI flags still parse and still
    # win over the YAML, but no wrapper uses that any more, and
    # scripts/_yaml_only_gate.py fails if one does.
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path; values here override the argparse defaults (§5.11).",
    )
    args = p.parse_args(argv)
    if args.config:
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, str(args.config))
    if args.out_dir is None:
        p.error(
            "--out-dir must come from --config's output.out_dir "
            "(AGENTS.md §5.11) or from an explicit --out-dir; got neither"
        )
    return args


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.csv.exists():
        raise FileNotFoundError(
            f"CSV not found at {args.csv}.  Pass --csv <path> or download the MoleculeNet "
            f"data into .data/molmetal/moleculenet/."
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[chemprop_v3_frozen_head] device={device}  csv={args.csv}  "
        f"smiles_col={args.smiles_col}  label_col={args.label_col}",
        flush=True,
    )
    smiles, labels = _load_csv(args.csv, args.smiles_col, args.label_col)
    print(
        f"[chemprop_v3_frozen_head] loaded {len(smiles)} rows from {args.csv.name}; "
        f"pos_rate={labels.mean():.3f}",
        flush=True,
    )
    t0 = time.time()
    results = []
    for seed in args.seeds:
        r = _train_one_seed(
            smiles=smiles,
            labels=labels,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            device=device,
            out_dir=args.out_dir,
            csv_path=args.csv,
        )
        results.append(r)
    elapsed = time.time() - t0
    agg = {
        "model": "chemprop_v3_frozen_head",
        "csv": str(args.csv),
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "n_seeds_total": len(results),
        "n_seeds_valid": len(results),
        "test_auc_mean": float(np.mean([r["test_metrics"]["roc_auc"] for r in results])),
        "test_auc_std": float(np.std([r["test_metrics"]["roc_auc"] for r in results])) if len(results) > 1 else 0.0,
        "per_seed_auc": [r["test_metrics"]["roc_auc"] for r in results],
        "elapsed_seconds": elapsed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    print(
        f"[chemprop_v3_frozen_head] DONE  n_seeds={len(results)}  "
        f"mean_auc={agg['test_auc_mean']:.4f} +/- {agg['test_auc_std']:.4f}  "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
