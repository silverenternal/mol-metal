"""Metric 1 — Mol-Metal on MetalCytoToxDB Ru subset, target ≥ 0.86 AUC.

Baseline (Krasnov 2026, J. Med. Chem.): Ru subset binary classification
5-fold CV AUC = 0.81 with Morgan FP + GBDT.  Our V4 multi-task D-MPNN
(with atomic features + metal-aware featurisation) should beat that.

Pipeline:
  1. Load MetalCytoToxDB, filter to ``metal_whitelist=[metal]`` (default Ru)
     with the standard preprocessing (time ≤ 24 h, IC50 ≥ 0.01 µM, pIC50 +
     active derived).
  2. Scaffold split (Bemis-Murcko) per the task spec: 70/10/20 train/val/test
     when ``--scaffold-split`` is set; falls back to a pre-2024 / post-2024
     temporal split when ``--temporal-split`` is set.
  3. Train multi-task D-MPNN for ``--epochs`` (default 50) at lr=1e-3,
     batch=32, alpha=0.5 (BCE activity + MSE pIC50 jointly).
  4. Evaluate ROC-AUC, PR-AUC, Hit@5% on the test split.  Aggregate
     per-seed AUC → mean ± std, write to ``metrics/by_metric/<key>.json``.

Usage
-----
    source env.sh
    python -m molmetal.scripts.train_metal_cytotox --metal Ru \
        --seeds 0 1 2 --scaffold-split \
        --out-dir logs/metric1_metalcytotox_ru
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os

_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from molmetal.baselines.dmpnn import (
    DMPNN_BATCH_SIZE,
    DMPNN_LR,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    MT_ALPHA,
    MT_HIDDEN,
    MT_DEPTH,
    MT_DROPOUT,
)
from molmetal.baselines.eval_utils import compute_metrics, per_cell_line_auc
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]

from molmetal.data.splits import (
    ScaffoldSplitter,
    TemporalSplitter,
)

METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _wf416_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Mol-Metal on MetalCytoToxDB Ru subset."
    )
    p.add_argument(
        "--metal", default="Ru", help="Metal centre to filter on (default: Ru)"
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="RNG seeds for the per-seed runs (default: 0 1 2)",
    )
    p.add_argument(
        "--split",
        choices=["scaffold", "temporal"],
        default=None,
        help="Splitting strategy. Default: scaffold (set --scaffold-split or --temporal-split)",
    )
    p.add_argument(
        "--scaffold-split",
        dest="split",
        action="store_const",
        const="scaffold",
        help="Shortcut: --split=scaffold (default)",
    )
    p.add_argument(
        "--temporal-split",
        dest="split",
        action="store_const",
        const="temporal",
        help="Shortcut: --split=temporal (pre-2024 train / post-2024 test)",
    )
    p.add_argument(
        "--epochs", type=int, default=50, help="Epochs per seed (default: 50)"
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32)"
    )
    p.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    p.add_argument(
        "--alpha", type=float, default=MT_ALPHA, help="BCE/MSE weight (default: 0.5)"
    )
    p.add_argument(
        "--hidden", type=int, default=MT_HIDDEN, help="D-MPNN hidden dim (default: 128)"
    )
    p.add_argument(
        "--depth",
        type=int,
        default=MT_DEPTH,
        help="D-MPNN message passing depth (default: 3)",
    )
    p.add_argument(
        "--time-threshold",
        type=float,
        default=24.0,
        help="Drop rows with Time(h) > t (default: 24 h). 0 disables.",
    )
    p.add_argument(
        "--ic50-min",
        type=float,
        default=0.01,
        help="Drop rows with IC50_Dark_value < t (default: 0.01 µM). 0 disables.",
    )
    p.add_argument(
        "--out-dir",
        default="logs/metric1_metalcytotox_ru",
        help="Directory for per-seed JSON output (default: logs/metric1_metalcytotox_ru)",
    )
    p.add_argument(
        "--out-metric",
        default=None,
        help=(
            "Path to write the aggregate metrics JSON. "
            "Default: metrics/by_metric/metalcytotox_<metal>_auc.json"
        ),
    )
    p.add_argument(
        "--out-report",
        default=None,
        help=(
            "Path to write a markdown report. "
            "Default: molmetal/reports/wf_tier1_metric1_metalcytotox/final.md"
        ),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="If set, use 1000 rows for a smoke run (handy when GPU is busy).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Splitter factory
# ---------------------------------------------------------------------------
def _build_splitter(split: str, seed: int):
    if split == "scaffold":
        return ScaffoldSplitter(
            strategy="largest_first",
            fractions=(0.7, 0.1, 0.2),  # task spec: 70/10/20
            seed=seed,
        )
    if split == "temporal":
        return TemporalSplitter(
            cutoff_year=2024,
            val_fraction=0.1,
            train_fraction_of_pre=0.8,
        )
    raise ValueError(f"Unknown split strategy: {split}")


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------
def _run_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    print(
        f"\n[metric1] === seed {seed} (metal={args.metal}, split={args.split}, "
        f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}) ===",
        flush=True,
    )
    flt = CytotoxFilter(
        time_threshold=args.time_threshold if args.time_threshold > 0 else None,
        ic50_min=args.ic50_min if args.ic50_min > 0 else None,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    if args.quick and len(ds) > 1000:
        # quick mode: subsample to 1000 rows for sanity
        keep = np.random.default_rng(seed).permutation(len(ds))[:1000]
        df_small = ds.df.iloc[keep].reset_index(drop=True).copy()
        ds = MetalCytotoxDataset(df=df_small, cache_dir=ds.cache_dir)
    print(
        f"[metric1]   rows={len(ds)} "
        f"actives={int(ds.active.sum())} ({100*ds.active.mean():.1f}%)",
        flush=True,
    )
    splitter = _build_splitter(args.split, seed)
    runner = DMPNNMultiTaskBaseline(
        metal=args.metal,
        hidden=args.hidden,
        depth=args.depth,
        dropout=MT_DROPOUT,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=seed,
        splitter=splitter,
    )
    # Inject our (possibly quick-mode) dataset into the runner so we don't re-load.
    runner._load_dataset = lambda: ds  # type: ignore[assignment]
    t0 = time.time()
    result = runner.run(verbose=False)
    elapsed = time.time() - t0
    tm = result.test_metrics
    vm = result.val_metrics
    train_score, _ = runner.predict(ds.smiles, runner._train_idx)
    y_train = ds.active[runner._train_idx].astype(int)
    train_metrics = compute_metrics(y_train, train_score)
    print(
        f"[metric1]   train AUC={train_metrics['roc_auc']:.4f}  "
        f"val AUC={vm['roc_auc']:.4f}  test AUC={tm['roc_auc']:.4f}  "
        f"(elapsed={elapsed:.1f}s)",
        flush=True,
    )
    return {
        "seed": seed,
        "metal": args.metal,
        "split": args.split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "n_rows": int(len(ds)),
        "n_train": int(result.n_train),
        "n_val": int(result.n_val),
        "n_test": int(result.n_test),
        "pos_rate_train": float(result.pos_rate_train),
        "pos_rate_test": float(result.pos_rate_test),
        "test_metrics": tm,
        "val_metrics": vm,
        "train_metrics": train_metrics,
        "test_pic50_metrics": result.test_pic50_metrics,
        "per_cell_line_auc": result.per_cell_line,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _aggregate(per_seed: List[Dict[str, object]]) -> Dict[str, object]:
    aucs = [float(d["test_metrics"]["roc_auc"]) for d in per_seed]
    aps = [float(d["test_metrics"]["pr_auc"]) for d in per_seed]
    hits = [float(d["test_metrics"]["hit_rate_top5pct"]) for d in per_seed]
    return {
        "n_seeds": len(per_seed),
        "test_auc_mean": float(statistics.mean(aucs)) if aucs else float("nan"),
        "test_auc_std": float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "test_pr_auc_mean": float(statistics.mean(aps)) if aps else float("nan"),
        "test_pr_auc_std": float(statistics.pstdev(aps)) if len(aps) > 1 else 0.0,
        "test_hit_top5pct_mean": float(statistics.mean(hits)) if hits else float("nan"),
        "per_seed_auc": aucs,
        "per_seed_pr_auc": aps,
        "per_seed_hit_top5pct": hits,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    aggregate: Dict[str, object],
    per_seed: List[Dict[str, object]],
    gate_pass: bool,
) -> None:
    auc_mean = aggregate["test_auc_mean"]
    auc_std = aggregate["test_auc_std"]
    lines = [
        "# Metric 1 — MetalCytoToxDB Ru subset binary classification AUC",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Metal:** {args.metal}  ",
        f"**Split:** {args.split} (Bemis-Murcko scaffold, 70/10/20)",
        f"**Model:** Multi-task D-MPNN (pIC50 regression + activity classification)",
        f"**Seeds:** {args.seeds}  ",
        f"**Epochs:** {args.epochs}, lr={args.lr}, batch={args.batch_size}, alpha={args.alpha}",
        "",
        "## Krasnov 2026 baseline (Tier 1 reference)",
        "",
        "| Source | Model | Split | AUC |",
        "|---|---|---|---|",
        "| Krasnov 2026 (J. Med. Chem.) | Morgan FP + GBDT | Ru subset 5-fold CV | **0.81** |",
        "| **Mol-Metal (this)** | D-MPNN multi-task (this run) | Ru scaffold 70/10/20 | "
        f"**{auc_mean:.4f} ± {auc_std:.4f}** (n={aggregate['n_seeds']} seeds) |",
        "",
        "## Verdict gate",
        "",
        f"- AUC = {auc_mean:.4f} (gate ≥ 0.86)",
        f"- Verdict: {'✅ PASS — beat Krasnov 0.81 by ≥ +0.05' if gate_pass else '⚠️ BELOW GATE'}",
        "",
        "## Per-seed detail",
        "",
        "| Seed | n_train | n_val | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in per_seed:
        tm = d["test_metrics"]
        lines.append(
            f"| {d['seed']} | {d['n_train']} | {d['n_val']} | {d['n_test']} | "
            f"{d['pos_rate_test']:.3f} | {tm['roc_auc']:.4f} | {tm['pr_auc']:.4f} | "
            f"{tm['hit_rate_top5pct']:.3f} |"
        )
    lines += [
        "",
        "## Methodology comparison",
        "",
        "* **Krasnov 2026** uses Morgan FP (radius 2, 2048 bits) + GBDT with 5-fold CV. We use",
        "  a from-scratch D-MPNN (atom+edge features, 3 message-passing rounds, 128 hidden) with",
        "  a multi-task readout (BCE activity + MSE pIC50) at alpha=0.5. Our featurisation is",
        "  strictly richer than Morgan FP (it captures bond directions and atom-bond graph",
        "  topology beyond circular substructures).",
        "* **Splitting** — Krasnov 2026 uses 5-fold CV on the same Ru subset. We use a",
        "  Bemis-Murcko scaffold split which is *stronger* (OOD by chemical scaffold) — any",
        "  drop relative to CV is therefore expected; any lift is more meaningful.",
        "* **Preprocessing** — both filter to IC50_Dark_value (dark toxicity). We additionally",
        "  filter to Time(h) ≤ 24 h to align with the standard assay protocol.",
        "",
        "## Honest caveats",
        "",
        "* Scaffold split is OOD by construction → AUC will be lower than 5-fold CV. Any",
        "  comparison to Krasnov 2026 is therefore *conservative* in our favour.",
        "* The D-MPNN backbone here does *not* include EGNN / Perceiver fusion — those are",
        "  in `molmetal.models.metal_hybrid_v4` (used by other metrics). This metric is the",
        "  pure-D-MPNN baseline; richer hybrids should match or beat this number on scaffold",
        "  splits.",
        "* Per-cell-line AUC has limited statistical power because most cell lines have < 30",
        "  Ru complexes; see `per_cell_line_auc` in the per-seed JSON for the breakdown.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(out_path, "\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: List[Dict[str, object]] = []
    for seed in args.seeds:
        try:
            d = _run_one_seed(args, seed)
        except Exception as exc:
            print(f"[metric1] seed {seed} FAILED: {exc!r}", flush=True)
            d = {"seed": seed, "error": repr(exc)}
        per_seed.append(d)
        # Write per-seed JSON incrementally so a crash doesn't lose progress.
        seed_path = out_dir / f"seed{seed}.json"
        _wf416_atomic_write_text(seed_path, json.dumps(d, indent=2, default=str))
        print(f"[metric1] wrote {seed_path}", flush=True)

    aggregate = _aggregate([d for d in per_seed if "error" not in d])
    auc_mean = aggregate["test_auc_mean"]
    auc_std = aggregate["test_auc_std"]
    GATE = 0.86
    gate_pass = bool(auc_mean >= GATE)

    # 1) per-seed + aggregate JSON (under logs/)
    full_payload = {
        "metal": args.metal,
        "split": args.split,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "hidden": args.hidden,
        "depth": args.depth,
        "krasnov_2026_baseline_auc": 0.81,
        "gate_auc": GATE,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "verdict": "PASS" if gate_pass else "BELOW_GATE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    aggregate_path = out_dir / "aggregate.json"
    _wf416_atomic_write_text(
        aggregate_path, json.dumps(full_payload, indent=2, default=str)
    )
    print(f"\n[metric1] === AGGREGATE ===", flush=True)
    print(
        f"[metric1] test AUC = {auc_mean:.4f} ± {auc_std:.4f}  "
        f"(n={aggregate['n_seeds']} seeds, gate ≥ {GATE})  "
        f"verdict={'PASS' if gate_pass else 'BELOW_GATE'}",
        flush=True,
    )
    print(f"[metric1] wrote {aggregate_path}", flush=True)

    # 2) metrics/by_metric/<metal>_auc.json
    metric_path = (
        Path(args.out_metric)
        if args.out_metric
        else METRICS_DIR / f"metalcytotox_{args.metal.lower()}_auc.json"
    )
    metric_payload = {
        "metric": f"metalcytotox_{args.metal.lower()}_auc",
        "description": (
            f"Test ROC-AUC on MetalCytoToxDB {args.metal} subset with "
            f"D-MPNN multi-task ({args.split} split, {args.epochs} epochs, "
            f"{len(args.seeds)} seeds)."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier1_precious_metal_anticancer",
        "source": (
            f"molmetal/scripts/train_metal_cytotox.py (seed={args.seeds}, "
            f"epochs={args.epochs})"
        ),
        "krasnov_2026_baseline": 0.81,
        "gate_target": GATE,
        "value": float(auc_mean),
        "value_std": float(auc_std),
        "per_seed_values": aggregate["per_seed_auc"],
        "n_seeds": int(aggregate["n_seeds"]),
        "n_train_avg": (
            float(statistics.mean([d.get("n_train", 0) for d in per_seed]))
            if per_seed
            else 0.0
        ),
        "n_test_avg": (
            float(statistics.mean([d.get("n_test", 0) for d in per_seed]))
            if per_seed
            else 0.0
        ),
        "timestamp": time.strftime("%Y-%m-%d"),
        "data_source": "MetalCytoToxDB.csv",
        "status": "MEASURED" if gate_pass else "MEASURED_BELOW_GATE",
        "split": args.split,
        "model": "dmpnn_multitask",
    }
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(metric_path, json.dumps(metric_payload, indent=2))
    print(f"[metric1] wrote {metric_path}", flush=True)

    # 3) Markdown report
    report_path = (
        Path(args.out_report)
        if args.out_report
        else REPORTS_DIR / "wf_tier1_metric1_metalcytotox" / "final.md"
    )
    _write_report(report_path, args, aggregate, per_seed, gate_pass)
    print(f"[metric1] wrote {report_path}", flush=True)

    return 0 if gate_pass else 2  # exit 2 so CI / ultracode can flag "below gate"


if __name__ == "__main__":
    raise SystemExit(main())
