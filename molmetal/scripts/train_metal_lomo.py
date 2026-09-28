"""Metric 19 — Multi-metal leave-one-metal-out (LOMO) transfer evaluation.

For each held-out metal M ∈ {Ru, Ir, Os, Rh, Re}, train the multi-task
D-MPNN on all rows where metal != M, then evaluate AUC on the held-out M
subset. 5 folds × 3 seeds = 15 runs; report mean ± std across the 15
test-set ROC-AUC values, and per-metal breakdown.

Gate (per ``TODO/METRICS_TARGETS.md`` §2 / Metric 19):
    mean LOMO test AUC ≥ 0.70 ⇒ PASS

Usage
-----
    cd /home/hugo/codes/mol-metal && source env.sh && uv run --no-sync python -u \\
        molmetal/scripts/train_metal_lomo.py \\
        --metals Ru Ir Os Rh Re --seeds 0 1 2 \\
        --out-dir logs/metric19_lomo
"""

from __future__ import annotations

# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

# Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults to project-local data dir
# (overridable via env var; previously hard-coded to /mnt/storage which does not exist
#  on this host, causing Wf-60 P1 m1/m2/m19 FileNotFoundError 5/5 seeds).
import os as _wf50_os
from pathlib import Path as _wf50_Path
_wf50_proj_root = _wf50_Path(__file__).resolve().parent.parent.parent
_wf50_default_data = _wf50_proj_root / ".data" / "molmetal"
if not _wf50_os.environ.get("MOLMETAL_DATA_DIR") and _wf50_default_data.exists():
    _wf50_os.environ["MOLMETAL_DATA_DIR"] = str(_wf50_default_data)
del _wf50_os, _wf50_Path, _wf50_proj_root, _wf50_default_data


import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    MT_ALPHA,
    MT_BATCH_SIZE,
    MT_DEPTH,
    MT_DROPOUT,
    MT_HIDDEN,
    MT_LR,
)
from molmetal.baselines.eval_utils import compute_metrics
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import SplitResult

METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
DEFAULT_LOMO_CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"

GATE_AUC = 0.70  # TODO/METRICS_TARGETS.md §2 / Metric 19


# ---------------------------------------------------------------------------
# LOMO splitter — train on metal != M, val on a small subset of train metal,
# test on metal == M.  We pick val from the same train pool (random subset
# of the 4 held-in metals) so we never leak the test metal into validation.
# ---------------------------------------------------------------------------
@dataclass
class LOMOSplitter:
    held_out_metal: str
    val_fraction: float = 0.10
    seed: int = 42

    def __call__(self, dataset: MetalCytotoxDataset) -> SplitResult:
        metals = dataset.df["Metal"].astype(str).to_numpy()
        test_mask = metals == self.held_out_metal
        test_idx = np.where(test_mask)[0].astype(np.int64)
        train_pool = np.where(~test_mask)[0].astype(np.int64)
        if self.val_fraction > 0 and len(train_pool) >= 10:
            train_idx, val_idx = train_test_split(
                train_pool,
                test_size=self.val_fraction,
                random_state=self.seed,
                stratify=None,  # active ratio is heterogeneous across metals
            )
        else:
            train_idx = train_pool
            val_idx = np.array([], dtype=np.int64)
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Single-seed runner — drop-in replacement using our LOMO splitter
# ---------------------------------------------------------------------------
class LOMODMPNNRunner:
    """D-MPNN runner with a *fixed* multi-metal dataset, varying splitter."""

    def __init__(
        self,
        ds: MetalCytotoxDataset,
        held_out_metal: str,
        hidden: int = MT_HIDDEN,
        depth: int = MT_DEPTH,
        dropout: float = MT_DROPOUT,
        lr: float = MT_LR,
        epochs: int = 30,
        batch_size: int = MT_BATCH_SIZE,
        alpha: float = MT_ALPHA,
        seed: int = 42,
        device: torch.device | None = None,
    ) -> None:
        self.ds = ds
        self.held_out_metal = held_out_metal
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.splitter = LOMOSplitter(held_out_metal=held_out_metal, seed=self.seed)
        self.model: DMPNNMultiTaskBaseline | None = None

    def run(self, verbose: bool = False) -> Dict[str, object]:
        """Train and return per-split metrics + index arrays."""
        # Drive the baseline via its existing interface — we replace _load_dataset
        # so the runner consumes our pre-built multi-metal dataset.
        runner = DMPNNMultiTaskBaseline(
            metal=self.held_out_metal,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            alpha=self.alpha,
            seed=self.seed,
            splitter=self.splitter,
            device=self.device,
        )
        runner._load_dataset = lambda: self.ds  # type: ignore[assignment]
        t0 = time.time()
        result = runner.run(verbose=verbose)
        elapsed = time.time() - t0
        return {
            "n_train": int(result.n_train),
            "n_val": int(result.n_val),
            "n_test": int(result.n_test),
            "pos_rate_train": float(result.pos_rate_train),
            "pos_rate_test": float(result.pos_rate_test),
            "test_metrics": dict(result.test_metrics),
            "val_metrics": dict(result.val_metrics),
            "elapsed_seconds": float(elapsed),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric 19 — multi-metal LOMO transfer AUC.")
    p.add_argument(
        "--metals",
        nargs="+",
        default=["Ru", "Ir", "Os", "Rh", "Re"],
        help="Metal centres to leave-one-out (default: Ru Ir Os Rh Re)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds per fold (default: 0 1 2)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epochs per fold×seed (default: 30 — smaller than Metric 1 because LOMO"
        " train set is larger but per-fold is one model only).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=MT_BATCH_SIZE,
        help=f"Mini-batch size (default: {MT_BATCH_SIZE})",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=MT_LR,
        help=f"Learning rate (default: {MT_LR})",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=MT_HIDDEN,
        help=f"D-MPNN hidden dim (default: {MT_HIDDEN})",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=MT_DEPTH,
        help=f"D-MPNN message passing depth (default: {MT_DEPTH})",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=MT_ALPHA,
        help=f"BCE/MSE weight (default: {MT_ALPHA})",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_LOMO_CSV),
        help="Path to the metal_lomo.csv produced by Step (1).",
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric19_lomo",
        help="Directory for per-seed JSON output (default: logs/metric19_lomo)",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help="Path to write aggregate metrics JSON (default: metrics/by_metric/metal_lomo_auc.json)",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Path to write markdown report (default: molmetal/reports/wf_tier2_lomo/final.md)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="If set, sub-sample to 1000 rows for a smoke run.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Build the multi-metal dataset once (shared across all folds)
# ---------------------------------------------------------------------------
def _build_dataset(args: argparse.Namespace) -> MetalCytotoxDataset:
    """Load metal_lomo.csv via the standard filter pipeline (no whitelist)."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"metal_lomo.csv not found at {csv_path}. "
            "Run Step (1) first to build it from MetalCytoToxDB.csv."
        )
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=None,  # include ALL 5 precious metals
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(path=csv_path, filters=flt)
    if args.quick and len(ds) > 1000:
        rng = np.random.default_rng(0)
        keep = rng.permutation(len(ds))[:1000]
        df_small = ds.df.iloc[keep].reset_index(drop=True).copy()
        ds = MetalCytotoxDataset(df=df_small, cache_dir=ds.cache_dir)
    print(
        f"[metric19] dataset loaded: n={len(ds)}  "
        f"actives={int(ds.active.sum())} ({100*ds.active.mean():.1f}%)  "
        f"metals={list(pd_series(ds.df['Metal']).value_counts().to_dict().items())}",
        flush=True,
    )
    return ds


def pd_series(x):
    """Tiny helper so we don't import pandas at top-level scope clumsily."""
    import pandas as pd

    return pd.Series(x)


# ---------------------------------------------------------------------------
# Single fold+seed
# ---------------------------------------------------------------------------
def _run_one(args: argparse.Namespace, ds: MetalCytotoxDataset, metal: str, seed: int) -> Dict[str, object]:
    print(
        f"\n[metric19] === fold held_out={metal}  seed={seed}  (epochs={args.epochs}) ===",
        flush=True,
    )
    runner = LOMODMPNNRunner(
        ds=ds,
        held_out_metal=metal,
        hidden=args.hidden,
        depth=args.depth,
        dropout=MT_DROPOUT,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=seed,
    )
    res = runner.run(verbose=False)
    tm = res["test_metrics"]
    vm = res["val_metrics"]
    print(
        f"[metric19]   n_train={res['n_train']}  n_test={res['n_test']}  "
        f"pos_rate_test={res['pos_rate_test']:.3f}  "
        f"val_AUC={vm['roc_auc']:.4f}  test_AUC={tm['roc_auc']:.4f}  "
        f"(elapsed={res['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return {
        "fold": metal,
        "seed": seed,
        **res,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _aggregate(per_run: List[Dict[str, object]]) -> Dict[str, object]:
    aucs = [float(d["test_metrics"]["roc_auc"]) for d in per_run if "test_metrics" in d]
    per_fold: Dict[str, List[float]] = {}
    for d in per_run:
        if "test_metrics" not in d:
            continue
        per_fold.setdefault(d["fold"], []).append(float(d["test_metrics"]["roc_auc"]))
    per_fold_mean = {m: float(statistics.mean(v)) if v else float("nan") for m, v in per_fold.items()}
    per_fold_std = {m: float(statistics.pstdev(v)) if len(v) > 1 else 0.0 for m, v in per_fold.items()}
    return {
        "n_runs": len(aucs),
        "n_folds": len(per_fold),
        "test_auc_mean": float(statistics.mean(aucs)) if aucs else float("nan"),
        "test_auc_std": float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "per_fold_auc_mean": per_fold_mean,
        "per_fold_auc_std": per_fold_std,
        "per_run_auc": aucs,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    aggregate: Dict[str, object],
    per_run: List[Dict[str, object]],
    gate_pass: bool,
) -> None:
    auc_mean = aggregate["test_auc_mean"]
    auc_std = aggregate["test_auc_std"]
    lines = [
        "# Metric 19 — Multi-metal leave-one-metal-out (LOMO) transfer AUC",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Metals:** {', '.join(args.metals)}  ",
        f"**Seeds:** {args.seeds}  ",
        f"**Model:** Multi-task D-MPNN (BCE activity + MSE pIC50, alpha={args.alpha})",
        f"**Epochs:** {args.epochs}, lr={args.lr}, batch={args.batch_size}",
        "",
        "## Goal",
        "",
        "Train on **all rows from 4 of the 5 precious metals** in the LOMO subset, "
        "evaluate on the **held-out metal** as an OOD transfer test.  Repeat for every"
        " metal × seed combination and report the **mean ± std across the 15 runs**.",
        "",
        "## Per-fold results",
        "",
        "| Held-out metal | Mean ROC-AUC | Std | n_runs |",
        "|---|---|---|---|",
    ]
    fold_means = aggregate["per_fold_auc_mean"]
    fold_stds = aggregate["per_fold_auc_std"]
    for m in args.metals:
        if m in fold_means:
            n_runs = sum(1 for d in per_run if d.get("fold") == m and "test_metrics" in d)
            lines.append(
                f"| {m} | {fold_means[m]:.4f} | {fold_stds[m]:.4f} | {n_runs} |"
            )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- **Mean LOMO test AUC** = **{auc_mean:.4f}** ± {auc_std:.4f}  "
        f"(n={aggregate['n_runs']} runs across {aggregate['n_folds']} folds)",
        f"- Gate threshold (TODO/METRICS_TARGETS.md §2 / Metric 19): mean ≥ 0.70",
        f"- **Verdict:** "
        + ("PASS — mean LOMO AUC ≥ 0.70 (transfer to held-out metal is non-trivial)"
           if gate_pass else
           "BELOW GATE — mean LOMO AUC < 0.70 (transfer is weak)"),
        "",
        "## Per-run detail",
        "",
        "| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in per_run:
        if "test_metrics" not in d:
            lines.append(f"| {d['fold']} | {d['seed']} | ERROR {d.get('error','?')} | — | — | — | — | — |")
            continue
        tm = d["test_metrics"]
        vm = d["val_metrics"]
        lines.append(
            f"| {d['fold']} | {d['seed']} | {d['n_train']} | {d['n_test']} | "
            f"{d['pos_rate_test']:.3f} | {vm['roc_auc']:.4f} | {tm['roc_auc']:.4f} | "
            f"{d['elapsed_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Honest caveats",
        "",
        "* The smallest folds (Os/Re/Rh) have few test rows (≤166) so per-metal AUC is",
        "  noisy; the aggregate mean is the most reliable number.",
        "* We do **not** include the metal token as an input feature — the model can only",
        "  transfer if the chemistry learned on the other 4 metals generalises to the",
        "  held-out metal's ligands.  This is a *strict* OOD by metal identity.",
        "* pIC50 is regressed jointly with activity (alpha=0.5) as in Metric 1 for",
        "  comparability.  Splitting strategy here is **metal**, not scaffold, so the",
        "  scaffold split is *within* the train pool only.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _build_dataset(args)

    per_run: List[Dict[str, object]] = []
    for metal in args.metals:
        for seed in args.seeds:
            try:
                d = _run_one(args, ds, metal, seed)
            except Exception as exc:  # pragma: no cover — keep going on failures
                print(f"[metric19] fold {metal} seed {seed} FAILED: {exc!r}", flush=True)
                d = {"fold": metal, "seed": seed, "error": repr(exc)}
            per_run.append(d)
            run_path = out_dir / f"fold-{metal}-seed{seed}.json"
            run_path.write_text(json.dumps(d, indent=2, default=str))
            print(f"[metric19] wrote {run_path}", flush=True)

    agg = _aggregate(per_run)
    auc_mean = float(agg["test_auc_mean"])
    auc_std = float(agg["test_auc_std"])
    gate_pass = bool(auc_mean >= GATE_AUC)

    payload = {
        "metric": "metal_lomo_auc",
        "description": (
            "Mean test ROC-AUC across 5 folds × N seeds of multi-metal "
            "leave-one-metal-out transfer on MetalCytoToxDB precious-metal subset."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_sbdd_ood_lomo",
        "source": "molmetal/scripts/train_metal_lomo.py",
        "gate_target": GATE_AUC,
        "value": auc_mean,
        "value_std": auc_std,
        "per_fold_auc_mean": agg["per_fold_auc_mean"],
        "per_fold_auc_std": agg["per_fold_auc_std"],
        "per_run_auc": agg["per_run_auc"],
        "n_runs": agg["n_runs"],
        "n_folds": agg["n_folds"],
        "metals": args.metals,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "model": "dmpnn_multitask",
        "data_source": "molmetal/data/metal_lomo.csv",
        "status": "MEASURED" if gate_pass else "MEASURED_BELOW_GATE",
        "verdict": "PASS" if gate_pass else "BELOW_GATE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metric_path = (
        Path(args.out_metric)
        if args.out_metric
        else METRICS_DIR / "metal_lomo_auc.json"
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[metric19] wrote {metric_path}", flush=True)

    full_payload = {**payload, "per_run": per_run, "aggregate": agg}
    aggregate_path = out_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(full_payload, indent=2, default=str))
    print(f"[metric19] wrote {aggregate_path}", flush=True)

    print(
        f"\n[metric19] === AGGREGATE ===  "
        f"mean LOMO test AUC = {auc_mean:.4f} ± {auc_std:.4f}  "
        f"(n={agg['n_runs']}, gate ≥ {GATE_AUC})  "
        f"verdict={'PASS' if gate_pass else 'BELOW_GATE'}",
        flush=True,
    )

    report_path = (
        Path(args.out_report)
        if args.out_report
        else REPORTS_DIR / "wf_tier2_lomo" / "final.md"
    )
    _write_report(report_path, args, agg, per_run, gate_pass)
    print(f"[metric19] wrote {report_path}", flush=True)

    return 0 if gate_pass else 2  # exit 2 so CI / ultracode can flag "below gate"


if __name__ == "__main__":
    raise SystemExit(main())
