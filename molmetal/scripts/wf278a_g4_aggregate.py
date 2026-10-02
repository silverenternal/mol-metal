#!/usr/bin/env python3
"""Wf-278A — Aggregate TODO-180 G4 14 (metal, seed) pairs from nested wave layout.

The original Wf-124 P3 trainer writes results to a nested path:

    results/wf124_p3_remaining_metals_g4/{wave}/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json
                              ^^^^^^  ^^^^^^         ^^^^   ^^^^^^

The original Wf-196 aggregator (`wf196_aggregate.py`) only knew the flat layout
`molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{seed}.json`,
so the G4 nested layout produced 0 pairs in the summary. This script walks the
nested tree, collects all 14 OK seeds, and emits:

  - <output>/training_summary.json — per-metal mean ± std + 5-metal aggregate

Usage:
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/wf278a_g4_aggregate.py \\
        --input-dir results/wf124_p3_remaining_metals_g4 \\
        --output   results/wf124_p3_remaining_metals_g4/training_summary.json

Path mismatch fix (vs wf196_aggregate.py):
  - OLD glob (flat):   SWEEP_DIR / metal.lower() / "per_seed_metrics_seed*.json"
  - NEW glob (nested): SWEEP_DIR / "*" / {Metal} / "seed*" / metal.lower()
                       / "per_seed_metrics_seed*.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

FIVE_METALS = ["Ru", "Ir", "Os", "Rh", "Re"]


def collect_seed_metrics_nested(root: Path) -> List[Dict]:
    """Walk the nested {wave}/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json tree.

    Returns list of loaded JSON dicts (only status == "OK").
    """
    return collect_seed_metrics_multi_root([root])


def collect_seed_metrics_multi_root(roots: List[Path]) -> List[Dict]:
    """Multi-root aggregation with last-wins-by-mtime dedup.

    Per Wf-294D / Wf-298B: when cascading G4 (30-epoch) + G5 (50-epoch)
    sweep results, a given (metal, seed) may appear under both roots
    (e.g. Ru/seed0 in G4 wave C and G5). The later-trained checkpoint
    must win. We dedup by (metal_dir_lower, seed_n) tuple, keeping the
    file with the largest mtime across all roots.

    Returns list of loaded JSON dicts (only status == "OK"), sorted by
    (metal, seed) for stable output.
    """
    candidates: Dict[tuple, tuple] = {}  # key -> (mtime, path)
    for root in roots:
        if not root.is_dir():
            print(f"[wf278a-agg] WARN: {root} does not exist; skipping", flush=True)
            continue
        for p in root.glob("**/per_seed_metrics_seed*.json"):
            stem = p.stem  # "per_seed_metrics_seed1"
            try:
                seed_n = int(stem.rsplit("seed", 1)[-1])
            except ValueError:
                continue
            try:
                metal_dir = p.parent.name  # "ru", "ir", ...
            except Exception:
                continue
            key = (metal_dir.lower(), seed_n)
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            prev = candidates.get(key)
            if prev is None or mtime > prev[0]:
                candidates[key] = (mtime, p)

    out: List[Dict] = []
    for key in sorted(candidates):
        mtime, p = candidates[key]
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            print(f"[wf278a-agg] skip {p}: {exc!r}", flush=True)
            continue
        if d.get("status", "OK") != "OK":
            print(f"[wf278a-agg] skip {p}: status={d.get('status')}", flush=True)
            continue
        # Sanity: payload metal/seed must match filename-derived key
        payload_metal = str(d.get("metal", "")).lower()
        if payload_metal and payload_metal != key[0]:
            print(
                f"[wf278a-agg] WARN {p}: payload metal={d.get('metal')} != path metal={key[0]}",
                flush=True,
            )
        if d.get("seed") is not None and d.get("seed") != key[1]:
            print(
                f"[wf278a-agg] WARN {p}: payload seed={d.get('seed')} != filename seed={key[1]}",
                flush=True,
            )
        out.append(d)
    return out


def per_metal_summary(metal: str, seeds: List[Dict]) -> Dict:
    """Per-metal mean ± std + n_seeds + ckpt paths (mirrors wf196_aggregate.py)."""
    if not seeds:
        return {
            "metal": metal,
            "n_seeds": 0,
            "status": "NO_SEEDS",
        }
    aucs = np.array(
        [s["test_metrics"].get("roc_auc", float("nan")) for s in seeds],
        dtype=float,
    )
    pearsons = np.array(
        [s["test_pic50_metrics"].get("pearson", float("nan")) for s in seeds],
        dtype=float,
    )
    rmses = np.array(
        [s["test_pic50_metrics"].get("rmse", float("nan")) for s in seeds],
        dtype=float,
    )
    maes = np.array(
        [s["test_pic50_metrics"].get("mae", float("nan")) for s in seeds],
        dtype=float,
    )
    w_finals = np.array(
        [s["wasserstein_history"][-1] if s.get("wasserstein_history") else float("nan")
         for s in seeds],
        dtype=float,
    )
    val_aucs = np.array(
        [s.get("val_metrics", {}).get("roc_auc", float("nan")) for s in seeds],
        dtype=float,
    )
    elapsed = np.array(
        [s.get("elapsed_seconds", float("nan")) for s in seeds],
        dtype=float,
    )
    aucs_finite = aucs[np.isfinite(aucs)]
    return {
        "metal": metal,
        "n_seeds": len(seeds),
        "test_auc_mean": float(np.nanmean(aucs_finite)) if aucs_finite.size else float("nan"),
        "test_auc_std": float(np.nanstd(aucs_finite)) if aucs_finite.size else float("nan"),
        "test_auc_per_seed": {str(s["seed"]): float(s["test_metrics"]["roc_auc"]) for s in seeds},
        "val_auc_mean": float(np.nanmean(val_aucs)),
        "val_auc_std": float(np.nanstd(val_aucs)),
        "test_pearson_mean": float(np.nanmean(pearsons)),
        "test_pearson_std": float(np.nanstd(pearsons)),
        "test_rmse_mean": float(np.nanmean(rmses)),
        "test_rmse_std": float(np.nanstd(rmses)),
        "test_mae_mean": float(np.nanmean(maes)),
        "test_mae_std": float(np.nanstd(maes)),
        "w_final_mean": float(np.nanmean(w_finals)),
        "w_final_std": float(np.nanstd(w_finals)),
        "elapsed_seconds_mean": float(np.nanmean(elapsed)),
        "elapsed_seconds_total": float(np.nansum(elapsed)),
        "n_train_mean": int(np.mean([s["n_train"] for s in seeds])),
        "n_val_mean": int(np.mean([s["n_val"] for s in seeds])),
        "n_test_mean": int(np.mean([s["n_test"] for s in seeds])),
        "epochs": int(seeds[0].get("epochs", 0)),
        "hidden": int(seeds[0].get("hidden", 0)),
        "depth": int(seeds[0].get("depth", 0)),
        "ckpt_paths": [s["ckpt_path"] for s in seeds],
        "status": "OK",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wf-278A G4 nested-path aggregator")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=False,
        help="Single-root dir with nested wave layout, e.g. results/wf124_p3_remaining_metals_g4",
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=False,
        help=(
            "Multi-root dirs (Wf-298B cascade mode, e.g. G4 + G5). "
            "Per (metal, seed) tuple, the file with the latest mtime wins. "
            "Use this for G4+G5 cascading where a (metal, seed) may appear "
            "under both roots."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write training_summary.json",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=15,
        help="Expected total (metal, seed) pairs. Default 15 for G4 (5 metals × 3 seeds).",
    )
    args = parser.parse_args(argv)

    output_path: Path = args.output

    # Resolve input roots: --input-dir takes precedence if set; otherwise use --input-dirs.
    if args.input_dir is not None:
        input_roots: List[Path] = [args.input_dir]
    elif args.input_dirs is not None:
        input_roots = list(args.input_dirs)
    else:
        print(
            "[wf278a-agg] ERROR: must pass either --input-dir or --input-dirs",
            flush=True,
        )
        return 2

    for r in input_roots:
        if not r.is_dir():
            print(f"[wf278a-agg] ERROR: {r} does not exist", flush=True)
            return 2

    if len(input_roots) == 1:
        all_seeds: List[Dict] = collect_seed_metrics_nested(input_roots[0])
    else:
        all_seeds = collect_seed_metrics_multi_root(input_roots)

    summary_per_metal: Dict[str, Dict] = {}
    for metal in FIVE_METALS:
        metal_seeds = [s for s in all_seeds if s.get("metal") == metal]
        summary_per_metal[metal] = per_metal_summary(metal, metal_seeds)

    aucs_all = np.array(
        [s["test_metrics"].get("roc_auc", float("nan")) for s in all_seeds],
        dtype=float,
    )
    pearsons_all = np.array(
        [s["test_pic50_metrics"].get("pearson", float("nan")) for s in all_seeds],
        dtype=float,
    )
    rmses_all = np.array(
        [s["test_pic50_metrics"].get("rmse", float("nan")) for s in all_seeds],
        dtype=float,
    )
    aucs_all_finite = aucs_all[np.isfinite(aucs_all)]

    summary = {
        "wf_tag": "Wf-278A — TODO-180 G4 nested-path re-aggregation (Wf-276C path-mismatch fix)",
        "input_dir": str(input_roots[0]) if len(input_roots) == 1 else None,
        "input_dirs": [str(r) for r in input_roots],
        "cascade_mode": len(input_roots) > 1,
        "metals": FIVE_METALS,
        "epochs": 30,
        "backbone": {"hidden": 300, "depth": 3},
        "n_pairs_total": len(all_seeds),
        "todo_180_g4_total_pairs": len(all_seeds),
        "expected_pairs": int(args.expected_pairs),
        "per_metal": summary_per_metal,
        "aggregate_5metal": {
            "n_pairs": len(all_seeds),
            "test_auc_mean": float(np.nanmean(aucs_all_finite)) if aucs_all_finite.size else float("nan"),
            "test_auc_std": float(np.nanstd(aucs_all_finite)) if aucs_all_finite.size else float("nan"),
            "test_pearson_mean": float(np.nanmean(pearsons_all)),
            "test_pearson_std": float(np.nanstd(pearsons_all)),
            "test_rmse_mean": float(np.nanmean(rmses_all)),
            "test_rmse_std": float(np.nanstd(rmses_all)),
        },
        "ru_auc_target": 0.86,
        "ru_auc_baseline_wf124_p3_seed0": 0.7613,
        "honest_framing_note": (
            "Wf-278A path-mismatch fix: Wf-276C reported TODO-180 G4 14 pairs "
            "MEASURED COMPLETE but training_summary.json showed 0 pairs. Root "
            "cause: Wf-196 aggregator used flat glob "
            "{metal}/per_seed_metrics_seed*.json while the Wf-124 P3 trainer "
            "writes the nested layout "
            "{wave}/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json. "
            "This aggregator walks the nested tree, keeps the most-specific "
            "(deepest) match per seed number, and de-dups the bare "
            "per_seed_metrics.json aggregator wrapper."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[wf278a-agg] wrote {output_path}", flush=True)

    # Console summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Wf-278A — TODO-180 G4 nested-path re-aggregation", flush=True)
    print("=" * 60, flush=True)
    for metal in FIVE_METALS:
        s = summary_per_metal[metal]
        if s.get("status") != "OK":
            print(f"  {metal}: {s.get('status', 'UNKNOWN')}", flush=True)
            continue
        print(
            f"  {metal}: test_auc = {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f} "
            f"(n={s['n_seeds']})  val_auc={s['val_auc_mean']:.4f}±{s['val_auc_std']:.4f}  "
            f"test_pearson={s['test_pearson_mean']:.3f}±{s['test_pearson_std']:.3f}  "
            f"w_final={s['w_final_mean']:.4f}",
            flush=True,
        )
    print("", flush=True)
    if aucs_all_finite.size:
        print(
            f"  5-metal aggregate: test_auc = {summary['aggregate_5metal']['test_auc_mean']:.4f} "
            f"± {summary['aggregate_5metal']['test_auc_std']:.4f} "
            f"(n={summary['aggregate_5metal']['n_pairs']})",
            flush=True,
        )
    print("", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())