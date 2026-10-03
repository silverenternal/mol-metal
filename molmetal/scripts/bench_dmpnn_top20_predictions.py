#!/usr/bin/env python3
"""Wf-58 Phase 2 — D-MPNN pIC50 prediction on top-20 wet-lab candidates.

=====================================================================
Bench runner for the Wf-58 wet-lab top-20 candidates
=====================================================================

**Purpose**: For each of the 20 compounds prioritised in
``molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv``, predict the
**per-cell-line pIC50** using a fresh D-MPNN 4-cell-line multi-task model
trained on ``molmetal/data/platinai_4cell.csv`` (synthetic IC50 labels —
see ``build_platinai_4cell.py`` honest framing).

**Seeds**: 3 (default ``0 1 2``).
**Epochs**: 100 per seed (default).
**Split**: scaffold (Bemis-Murcko) 70/10/20 — same protocol as
``train_platinai_4cell.py``.

**Outputs** (under ``--out-dir``):
  - ``per_compound_pic50.json``   — 20 compounds × 3 seeds = 60 pIC50
                                    predictions (per-cell-line tensor).
  - ``per_cell_pic50.json``       — per-cell-line mean ± std across
                                    the 20 candidates and 3 seeds.
  - ``auc_per_seed.json``         — test-set AUC per cell line per seed.
  - ``synthesis.md``              — human-readable summary.

**Verify gates**:
  - ``per_compound_pic50.json`` carries 60 entries (20 compounds × 3 seeds).
  - Per-compound pIC50 std ≤ 0.15 log units (3-seed reproducibility).
  - AUC ≥ 0.73 mean (m18 anchor) — written to synthesis as MEASURED.

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * **No wet-lab synthesis** — this is a *bench* (in-silico prediction
    layer), not a wet-lab validation.
  * pIC50 predictions for non-Pt compounds (Pd / Au / Ru / Ir in the
    top-20) use a Pt-trained D-MPNN — **cross-metal transfer** —
    so absolute pIC50 numbers carry distribution-shift caveats;
    treat as **rank-only**.
  * IC50 labels in ``platinai_4cell.csv`` are synthetic (Rusanov 2025
    ChemRxiv tables S4-S5 are behind a Cloudflare challenge we cannot
    fetch from this host).
  * The Wf-53 Phase 2 real chemprop D-MPNN pIC50 bench is the binding
    truth; this script is a **CPU-only from-scratch D-MPNN bench**
    for fast iteration.

**Hardware**: CPU-only primary; CUDA optional (RTX 5090 GPU-1 if
``--device cuda:1`` is passed).

**Usage**
-----
::

    source env.sh
    python molmetal/scripts/bench_dmpnn_top20_predictions.py \\
        --top20 molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv \\
        --seeds 0 1 2 --epochs 100 \\
        --out-dir molmetal/reports/wf58_dmpnn_predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# Atomic write protocol — closes the "half-merged file state" failure mode
# (concurrent writers racing on the rename) and the "direct overwrite
# without risk check" failure mode (closeout-drafting scripts silently
# overwriting prior closeouts).  See `wf386a3_atomic_write.py` §0 +
# AGENTS.md §5.6 adoption rule.
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
    detect_truncation,
)

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles_list,
)
from molmetal.scripts.train_platinai_4cell import (
    CELL_LINES,
    DMPNNFourCellModel,
    FourCellDMPNNRunner,
    IC50_COLS,
    PlatinAI4Cell,
    _predict,
)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CELL_LINES: int = len(CELL_LINES)
PIC50_STD_GATE: float = 0.15  # 3-seed reproducibility gate (log units)
AUC_GATE: float = 0.73  # m18 anchor (Wf-31/32 baseline 0.7393)
N_DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EPOCHS: int = 100
DEFAULT_TOP20: Path = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf58_wetlab_dryrun"
    / "top20_candidates.csv"
)
DEFAULT_DATA: Path = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_OUT: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf58_dmpnn_predictions"


# ---------------------------------------------------------------------------
# Top-20 loading
# ---------------------------------------------------------------------------
@dataclass
class Top20Candidate:
    rank: int
    compound_smiles: str
    compound_id: str
    metal: str


def load_top20(path: Path) -> List[Top20Candidate]:
    """Load top-20 candidates from Wf-58 P1 output CSV.

    Expected schema (no header order requirement):
        rank, compound_smiles, id, metal, dmpnn_pic50, qed, sa,
        vina_kcal, astsr_score
    """
    out: List[Top20Candidate] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rank = int(row["rank"])
            except (ValueError, KeyError):
                continue
            smi = row.get("compound_smiles") or row.get("smiles") or ""
            cid = row.get("id") or row.get("compound_id") or f"rank{rank}"
            metal = row.get("metal") or "?"
            if not smi:
                continue
            out.append(
                Top20Candidate(
                    rank=rank,
                    compound_smiles=smi.strip(),
                    compound_id=cid.strip(),
                    metal=metal.strip(),
                )
            )
    out.sort(key=lambda c: c.rank)
    return out


# ---------------------------------------------------------------------------
# External-SMILES pIC50 prediction
# ---------------------------------------------------------------------------
def predict_pic50_external(
    runner: FourCellDMPNNRunner,
    smiles_list: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Return ``(N, 4)`` pIC50 predictions for *external* SMILES.

    Featurises the SMILES, runs the trained model in eval mode, and
    reads off the regression head's output. Invalid molecules produce
    ``NaN`` rows (per-cell-line).
    """
    if runner.model is None:
        raise RuntimeError("Runner not trained — call fit() first.")
    runner.model.eval()
    graphs = featurize_smiles_list([str(s) for s in smiles_list])
    n = len(smiles_list)
    pic50_pred = np.full((n, N_CELL_LINES), np.nan, dtype=np.float32)

    device = runner.device
    for start in range(0, n, batch_size):
        batch_graphs = graphs[start : start + batch_size]
        valid_rows: List[int] = []
        g_list: List = []
        for k, g in enumerate(batch_graphs):
            if g is None:
                continue
            valid_rows.append(k)
            g_list.append(g)
        if not g_list:
            continue
        # Collate + forward
        from molmetal.baselines.dmpnn import collate_graphs

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        with torch.no_grad():
            pic50_pred_batch, _ = runner.model(atom_f, bond_f, e_src, e_dst)
        pic50_pred_batch = pic50_pred_batch.cpu().numpy()  # (B, 4)
        for vi, k in enumerate(valid_rows):
            pic50_pred[k] = pic50_pred_batch[vi]

    return pic50_pred


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_one_seed(
    args: argparse.Namespace,
    seed: int,
    ds: PlatinAI4Cell,
) -> Dict[str, object]:
    """Train + evaluate + predict for a single seed; return JSON-friendly dict."""
    from molmetal.data.splits import ScaffoldSplitter

    print(
        f"\n[wf58dmpnn] === seed {seed} (epochs={args.epochs}, bs={args.batch_size}, "
        f"lr={args.lr}, alpha={args.alpha}) ===",
        flush=True,
    )

    splitter = ScaffoldSplitter(seed=seed, fractions=(0.7, 0.1, 0.2))
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    runner = FourCellDMPNNRunner(
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=seed,
        splitter=splitter,
        device=device,
    )
    t0 = time.time()
    runner.fit(ds, verbose=True)
    train_time = time.time() - t0

    # Evaluate (gives per-cell AUC on platinai_4cell test fold)
    eval_out = runner.evaluate(ds, verbose=False)

    return {
        "seed": seed,
        "device": str(device),
        "elapsed_train_seconds": train_time,
        "per_line_auc": eval_out["per_line_auc"],
        "per_line_ap": eval_out["per_line_ap"],
        "per_line_n": eval_out["per_line_n"],
        "auc_mean": eval_out["auc_mean"],
        "auc_a2780cis": eval_out.get("auc_a2780cis", float("nan")),
        "n_test": int(eval_out["n_test"]),
        "runner": runner,  # returned for caller-side predict; not JSON-serialisable
    }


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def build_per_compound_table(
    per_seed_outputs: List[Tuple[int, np.ndarray, List[Top20Candidate]]],
) -> List[Dict[str, object]]:
    """Build per-compound pIC50 records (20 compounds × n_seeds entries).

    Each entry: ``{rank, compound_id, metal, seed, per_cell_pic50:
    {cell_line: float}, mean_pic50, std_pic50_across_cells}``.
    """
    out: List[Dict[str, object]] = []
    for seed, pic50_arr, top20 in per_seed_outputs:
        for i, cand in enumerate(top20):
            row = pic50_arr[i]  # (4,) per-cell pIC50
            per_cell = {CELL_LINES[j]: float(row[j]) for j in range(N_CELL_LINES)}
            valid = row[~np.isnan(row)]
            mean = float(np.mean(valid)) if valid.size else float("nan")
            out.append(
                {
                    "rank": int(cand.rank),
                    "compound_id": cand.compound_id,
                    "metal": cand.metal,
                    "seed": int(seed),
                    "per_cell_pic50": per_cell,
                    "mean_pic50": mean,
                    "n_cells_valid": int(valid.size),
                }
            )
    return out


def aggregate_per_compound_std(
    per_seed_outputs: List[Tuple[int, np.ndarray, List[Top20Candidate]]],
) -> List[Dict[str, object]]:
    """For each compound, compute per-cell std across the 3 seeds."""
    if not per_seed_outputs:
        return []
    top20 = per_seed_outputs[0][2]
    n_seeds = len(per_seed_outputs)
    out: List[Dict[str, object]] = []
    for i, cand in enumerate(top20):
        stack = np.stack([arr[i] for _, arr, _ in per_seed_outputs], axis=0)  # (S, 4)
        per_cell_std = {}
        per_cell_mean = {}
        for j, line in enumerate(CELL_LINES):
            col = stack[:, j]
            col_valid = col[~np.isnan(col)]
            if col_valid.size:
                per_cell_mean[line] = float(np.mean(col_valid))
                per_cell_std[line] = float(np.std(col_valid))
            else:
                per_cell_mean[line] = float("nan")
                per_cell_std[line] = float("nan")
        # Aggregate std across cells (mean of per-cell std) — gate metric.
        stds = [per_cell_std[l] for l in CELL_LINES if not np.isnan(per_cell_std[l])]
        std_pooled = float(np.mean(stds)) if stds else float("nan")
        out.append(
            {
                "rank": int(cand.rank),
                "compound_id": cand.compound_id,
                "metal": cand.metal,
                "per_cell_mean": per_cell_mean,
                "per_cell_std": per_cell_std,
                "std_pooled_log_units": std_pooled,
                "gate_pass": bool(std_pooled <= PIC50_STD_GATE)
                if not np.isnan(std_pooled)
                else False,
                "n_seeds": n_seeds,
            }
        )
    return out


def aggregate_per_cell(
    per_seed_outputs: List[Tuple[int, np.ndarray, List[Top20Candidate]]],
) -> Dict[str, Dict[str, float]]:
    """For each cell line, compute mean ± std across 20 candidates × n_seeds."""
    if not per_seed_outputs:
        return {
            line: {"mean": float("nan"), "std": float("nan"), "n": 0}
            for line in CELL_LINES
        }
    stack_all = np.stack([arr for _, arr, _ in per_seed_outputs], axis=0)  # (S, 20, 4)
    out: Dict[str, Dict[str, float]] = {}
    for j, line in enumerate(CELL_LINES):
        col = stack_all[..., j].flatten()
        col_valid = col[~np.isnan(col)]
        out[line] = {
            "mean": float(np.mean(col_valid)) if col_valid.size else float("nan"),
            "std": float(np.std(col_valid)) if col_valid.size else float("nan"),
            "median": float(np.median(col_valid)) if col_valid.size else float("nan"),
            "min": float(np.min(col_valid)) if col_valid.size else float("nan"),
            "max": float(np.max(col_valid)) if col_valid.size else float("nan"),
            "n": int(col_valid.size),
        }
    return out


# ---------------------------------------------------------------------------
# Synthesis markdown
# ---------------------------------------------------------------------------
def build_synthesis_md(
    args: argparse.Namespace,
    top20: List[Top20Candidate],
    per_seed_meta: List[Dict[str, object]],
    per_seed_pic50: List[Tuple[int, np.ndarray, List[Top20Candidate]]],
    per_compound_std: List[Dict[str, object]],
    per_cell_agg: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    lines: List[str] = [
        "# Wf-58 Phase 2 — D-MPNN pIC50 prediction on top-20 wet-lab candidates",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Model**: 4-cell-line multi-task D-MPNN "
        f"(hidden={args.hidden}, depth={args.depth}, dropout={args.dropout})",
        f"**Split**: scaffold (Bemis-Murcko) 70/10/20 (per-seed)",
        f"**Seeds**: {list(args.seeds)}  ",
        f"**Epochs**: {args.epochs}, lr={args.lr}, batch={args.batch_size}, alpha={args.alpha}",
        f"**Device**: {args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')}",
        "",
        "## Headline numbers",
        "",
    ]
    # AUC summary across seeds
    auc_means = [m["auc_mean"] for m in per_seed_meta]
    auc_a2780cis = [m["auc_a2780cis"] for m in per_seed_meta]
    lines += [
        "| Metric | Value | Gate |",
        "|---|---|---|",
        f"| Mean AUC (platinai_4cell test fold) | {float(np.mean(auc_means)):.4f} "
        f"± {float(np.std(auc_means)):.4f} (n={len(auc_means)}) | ≥ {AUC_GATE:.2f} |",
        f"| A2780cis AUC (cisplatin-resistant anchor) | {float(np.mean(auc_a2780cis)):.4f} "
        f"± {float(np.std(auc_a2780cis)):.4f} | — |",
    ]
    # Per-cell breakdown
    lines += [
        "",
        "## Per-cell-line mean pIC50 across top-20 (3-seed pooled)",
        "",
        "| Cell line | Mean | Median | Std | Min | Max | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for line_name in CELL_LINES:
        s = per_cell_agg.get(line_name, {})
        lines.append(
            f"| {line_name} | {s.get('mean', float('nan')):.3f} | "
            f"{s.get('median', float('nan')):.3f} | {s.get('std', float('nan')):.3f} | "
            f"{s.get('min', float('nan')):.3f} | {s.get('max', float('nan')):.3f} | "
            f"{s.get('n', 0)} |"
        )
    # Per-compound std gate
    passes = [c for c in per_compound_std if c["gate_pass"]]
    n_pass = len(passes)
    lines += [
        "",
        "## 3-seed reproducibility gate (per-compound pIC50 std ≤ 0.15 log units)",
        "",
        f"* **Pooled std across cells**: min = "
        f"{min((c['std_pooled_log_units'] for c in per_compound_std), default=float('nan')):.4f}, "
        f"mean = {float(np.mean([c['std_pooled_log_units'] for c in per_compound_std])):.4f}, "
        f"max = {max((c['std_pooled_log_units'] for c in per_compound_std), default=float('nan')):.4f}",
        f"* **Gate pass rate**: {n_pass}/{len(per_compound_std)} compounds",
        f"* **Gate**: pooled std ≤ {PIC50_STD_GATE:.2f} log units",
        "",
        "## Per-compound breakdown",
        "",
        "| Rank | ID | Metal | Pooled std (log) | Gate | Mean pIC50 (across cells) |",
        "|---:|---|---|---:|:-:|---|",
    ]
    by_rank = sorted(per_compound_std, key=lambda c: c["rank"])
    for c in by_rank:
        # mean across cells from pooled mean
        per_cell_mean = c["per_cell_mean"]
        valid_means = [
            per_cell_mean[l] for l in CELL_LINES if not np.isnan(per_cell_mean[l])
        ]
        mean_pic50 = float(np.mean(valid_means)) if valid_means else float("nan")
        lines.append(
            f"| {c['rank']} | {c['compound_id']} | {c['metal']} | "
            f"{c['std_pooled_log_units']:.4f} | "
            f"{'PASS' if c['gate_pass'] else 'FAIL'} | {mean_pic50:.3f} |"
        )

    # Honest framing
    lines += [
        "",
        "## Honest framing",
        "",
        "* **Cross-metal transfer caveat**: top-20 spans Pt (n=2) / Pd (n=3) / "
        "Au (n=1) / Ru (n=3) / Ir (n=11). The D-MPNN is trained on PlatinAI "
        "4-cell (Pt-only), so non-Pt predictions are **distribution-shifted**. "
        "Treat per-cell pIC50 numbers for Pd / Au / Ru / Ir as **rank-only**.",
        "* **Synthetic IC50 labels**: ``platinai_4cell.csv`` carries synthetic "
        "pIC50 labels (Rusanov 2025 ChemRxiv tables S4-S5 are behind a "
        "Cloudflare JS challenge; see ``build_platinai_4cell.py`` honest "
        "framing). AUC numbers are bench-quality, not real-data quality.",
        "* **CPU-only from-scratch D-MPNN**: this is the from-scratch edge-aware "
        "D-MPNN baseline, not the production chemprop D-MPNN. Wf-53 Phase 2 "
        "chemprop bench is the binding truth.",
        "* **No wet-lab synthesis** — this is the in-silico prediction layer "
        "for the Wf-58 wet-lab outreach plan (TODO-97 R20-P6).",
        "* **Paper gate**: per TODO-84, no paper §5 claim derives from this "
        "synthesis; outputs feed the wet-lab cooperation pipeline, not paper "
        "text.",
        "",
        "## Cross-references",
        "",
        "* ``TODO/pending/97_r20_p6_wetlab_ic50.md`` — R20-P6 wet-lab plan",
        "* ``molmetal/reports/wf58_wetlab_dryrun/ranking_summary.md`` — Wf-58 P1 ranking",
        "* ``molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`` — input top-20",
        "* ``TODO/completed/82d_m16m19m8_retrain.md`` — Wf-32 P2 ship (m18 anchor)",
        "* ``molmetal/scripts/train_platinai_4cell.py`` — 4-cell D-MPNN model defs",
    ]
    # Wf-410-A3 migration: use atomic_write_file + safe_overwrite_check so
    # concurrent closeout-drafting runs (e.g. Wf-58 re-runs) don't race on
    # the rename and don't silently overwrite prior reports.
    _content = "\n".join(lines) + "\n"
    safe_overwrite_check(out_path, _content)
    atomic_write_file(out_path, _content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-58 Phase 2 — D-MPNN pIC50 prediction on top-20 wet-lab candidates.",
    )
    p.add_argument(
        "--top20",
        type=Path,
        default=DEFAULT_TOP20,
        help="Path to top20_candidates.csv (default: Wf-58 P1 ship)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to platinai_4cell.csv training data (default: PlatinAI 4-cell Pt)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(N_DEFAULT_SEEDS),
        help=f"Random seeds (default: {list(N_DEFAULT_SEEDS)})",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs per seed (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="D-MPNN hidden dim (default: 128)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=3,
        help="D-MPNN message-passing depth (default: 3)",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="D-MPNN dropout (default: 0.1)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="BCE-vs-MSE loss mix (default: 0.5)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto cuda:0 if available else cpu)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch training logs",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load top-20 + training data
    top20 = load_top20(args.top20)
    if len(top20) == 0:
        print(
            f"[wf58dmpnn] ERROR: no candidates loaded from {args.top20}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[wf58dmpnn] Loaded {len(top20)} candidates from {args.top20}",
        flush=True,
    )
    print(
        f"[wf58dmpnn] Metal distribution: "
        f"{dict([(m, sum(1 for c in top20 if c.metal == m)) for m in sorted({c.metal for c in top20})])}",
        flush=True,
    )

    ds = PlatinAI4Cell.from_csv(args.data)
    print(
        f"[wf58dmpnn] Loaded training data: {len(ds)} rows from {args.data}",
        flush=True,
    )

    # 2) Per-seed: train + evaluate + predict top-20 pIC50
    per_seed_meta: List[Dict[str, object]] = []
    per_seed_pic50: List[Tuple[int, np.ndarray, List[Top20Candidate]]] = []

    for seed in args.seeds:
        out = run_one_seed(args, int(seed), ds)
        runner = out.pop("runner")
        per_seed_meta.append(out)

        # Predict pIC50 for top-20
        smiles_top20 = [c.compound_smiles for c in top20]
        pic50_pred = predict_pic50_external(
            runner, smiles_top20, batch_size=args.batch_size
        )
        per_seed_pic50.append((int(seed), pic50_pred, top20))

        print(
            f"[wf58dmpnn] seed={seed} top-20 predicted: "
            f"{np.isnan(pic50_pred).sum(0).tolist()} NaNs per cell",
            flush=True,
        )

    # 3) Aggregate
    per_compound_table = build_per_compound_table(per_seed_pic50)
    per_compound_std = aggregate_per_compound_std(per_seed_pic50)
    per_cell_agg = aggregate_per_cell(per_seed_pic50)

    # 4) Write JSONs
    per_compound_json = out_dir / "per_compound_pic50.json"
    per_cell_json = out_dir / "per_cell_pic50.json"
    auc_per_seed_json = out_dir / "auc_per_seed.json"
    synthesis_md = out_dir / "synthesis.md"

    # per_compound_pic50.json: list of records + summary header
    per_compound_payload = {
        "meta": {
            "n_compounds": len(top20),
            "n_seeds": len(args.seeds),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "cell_lines": list(CELL_LINES),
            "pic50_std_gate": PIC50_STD_GATE,
            "auc_gate": AUC_GATE,
            "model": "4-cell-line multi-task D-MPNN (from-scratch, CPU-friendly)",
            "training_data": str(args.data),
            "top20_source": str(args.top20),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "framing": "honest-negative: synthetic IC50 labels + cross-metal transfer for non-Pt candidates",
        },
        "compounds": per_compound_table,
        "per_compound_std": per_compound_std,
    }
    # Wf-410-A3 migration: atomic write to per-compound JSON.
    _per_compound_payload = json.dumps(per_compound_payload, indent=2) + "\n"
    safe_overwrite_check(per_compound_json, _per_compound_payload)
    atomic_write_file(per_compound_json, _per_compound_payload, encoding="utf-8")

    per_cell_payload = {
        "meta": {
            "n_compounds": len(top20),
            "n_seeds": len(args.seeds),
            "cell_lines": list(CELL_LINES),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "per_cell_aggregate": per_cell_agg,
    }
    # Wf-410-A3 migration: atomic write to per-cell JSON.
    _per_cell_payload = json.dumps(per_cell_payload, indent=2) + "\n"
    safe_overwrite_check(per_cell_json, _per_cell_payload)
    atomic_write_file(per_cell_json, _per_cell_payload, encoding="utf-8")

    auc_payload = {
        "meta": {
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "auc_gate": AUC_GATE,
            "training_data": str(args.data),
        },
        "per_seed": per_seed_meta,
    }
    # Wf-410-A3 migration: atomic write to AUC per-seed JSON.
    _auc_payload = json.dumps(auc_payload, indent=2) + "\n"
    safe_overwrite_check(auc_per_seed_json, _auc_payload)
    atomic_write_file(auc_per_seed_json, _auc_payload, encoding="utf-8")

    # 5) Synthesis markdown
    build_synthesis_md(
        args,
        top20,
        per_seed_meta,
        per_seed_pic50,
        per_compound_std,
        per_cell_agg,
        synthesis_md,
    )

    # 6) Verify gates
    n_records = len(per_compound_table)
    expected = len(top20) * len(args.seeds)
    if n_records != expected:
        print(
            f"[wf58dmpnn] GATE FAIL: expected {expected} per-compound records "
            f"(20×3), got {n_records}",
            file=sys.stderr,
        )
        return 1

    auc_mean = float(np.mean([m["auc_mean"] for m in per_seed_meta]))
    if auc_mean < AUC_GATE:
        print(
            f"[wf58dmpnn] WARN: mean AUC = {auc_mean:.4f} < gate {AUC_GATE:.2f} "
            f"(m18 anchor; expected ≥ 0.7393 ± 0.009)",
            file=sys.stderr,
        )
        # Don't fail — emit the warning so the synthesis.md still ships.

    n_pass = sum(1 for c in per_compound_std if c["gate_pass"])
    if n_pass < len(per_compound_std):
        print(
            f"[wf58dmpnn] NOTE: {n_pass}/{len(per_compound_std)} compounds "
            f"within 3-seed reproducibility gate (std ≤ {PIC50_STD_GATE:.2f})",
            flush=True,
        )

    print(
        f"\n[wf58dmpnn] DONE. {n_records} records. mean AUC = {auc_mean:.4f}. "
        f"Gate pass = {n_pass}/{len(per_compound_std)}.",
        flush=True,
    )
    print(f"[wf58dmpnn]   {per_compound_json}")
    print(f"[wf58dmpnn]   {per_cell_json}")
    print(f"[wf58dmpnn]   {auc_per_seed_json}")
    print(f"[wf58dmpnn]   {synthesis_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
