#!/usr/bin/env python3
"""Wf-196 — Aggregate per-metal stats from the 50-epoch sweep.

Reads `molmetal/reports/wf196_50ep_sweep/{metal}/per_seed_metrics_seed{seed}.json`
files and computes per-metal mean ± std test_auc + test_pearson + test_rmse
across all seeds that successfully completed.

Writes:
  - `training_summary.json` (per-metal + aggregate 5-metal)
  - `30ep_vs_50ep_comparison.json` (vs Wf-124 P3 30-epoch Ru seed=0 baseline)

Usage:
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf196_aggregate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf196_50ep_sweep"
WF124_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf124_p3_multitask"

FIVE_METALS = ["Ru", "Ir", "Os", "Rh", "Re"]


def collect_seed_metrics(metal_dir: Path) -> List[Dict]:
    """Collect per-seed metrics JSONs from a metal's directory."""
    out = []
    if not metal_dir.is_dir():
        return out
    for p in sorted(metal_dir.glob("per_seed_metrics_seed*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            print(f"[wf196-agg] skip {p}: {exc!r}", flush=True)
            continue
        if d.get("status", "OK") != "OK":
            continue
        out.append(d)
    return out


def per_metal_summary(metal: str, seeds: List[Dict]) -> Dict:
    """Per-metal mean ± std + n_seeds + ckpt paths."""
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
    w_finals = np.array(
        [s["wasserstein_history"][-1] if s.get("wasserstein_history") else float("nan")
         for s in seeds],
        dtype=float,
    )
    aucs_finite = aucs[np.isfinite(aucs)]
    return {
        "metal": metal,
        "n_seeds": len(seeds),
        "test_auc_mean": float(np.nanmean(aucs_finite)) if aucs_finite.size else float("nan"),
        "test_auc_std": float(np.nanstd(aucs_finite)) if aucs_finite.size else float("nan"),
        "test_auc_per_seed": {str(s["seed"]): float(s["test_metrics"]["roc_auc"]) for s in seeds},
        "test_pearson_mean": float(np.nanmean(pearsons)),
        "test_pearson_std": float(np.nanstd(pearsons)),
        "test_rmse_mean": float(np.nanmean(rmses)),
        "test_rmse_std": float(np.nanstd(rmses)),
        "w_final_mean": float(np.nanmean(w_finals)),
        "w_final_std": float(np.nanstd(w_finals)),
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
    if not SWEEP_DIR.is_dir():
        print(f"[wf196-agg] ERROR: {SWEEP_DIR} does not exist; run sweep first",
              flush=True)
        return 2

    summary_per_metal: Dict[str, Dict] = {}
    all_seeds: List[Dict] = []
    for metal in FIVE_METALS:
        metal_dir = SWEEP_DIR / metal.lower()
        seeds = collect_seed_metrics(metal_dir)
        all_seeds.extend(seeds)
        summary_per_metal[metal] = per_metal_summary(metal, seeds)

    # 5-metal × N-seed aggregate
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
        "wf_tag": (
            "Wf-196 — 50-epoch CPU sweep of 14 missing (metal, seed) pairs "
            "(smaller backbone hidden=64 depth=2)"
        ),
        "out_dir": str(SWEEP_DIR),
        "metals": FIVE_METALS,
        "epochs": 50,
        "backbone": {"hidden": 64, "depth": 2},
        "device": "cpu",
        "n_pairs_total": len(all_seeds),
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
        "ru_auc_baseline_single_task": 0.7123,
        "ru_auc_target_met_50ep": (
            float(summary_per_metal["Ru"]["test_auc_mean"]) >= 0.86
            if summary_per_metal["Ru"]["test_auc_mean"] != "NO_SEEDS" else False
        ),
        "honest_framing_note": (
            "Wf-196 CPU sweep at 50 epochs with smaller backbone (hidden=64, "
            "depth=2) to fit the 8h wall-time budget. AUC numbers are NOT "
            "directly comparable to the Wf-124 P3 30-epoch run at full "
            "backbone (hidden=128, depth=3); smaller backbone trades "
            "capacity for speed. Smaller metals (Rh/Os/Re < 1.2k rows) have "
            "noisier per-seed AUC; per-seed spread is reported unfiltered."
        ),
    }

    # 30ep vs 50ep comparison (for Ru seed=0; Wf-124 P3 30-epoch MEASURED)
    comparison = {}
    wf124_ru_s0_path = WF124_DIR / "ru" / "per_seed_metrics_seed0.json"
    if wf124_ru_s0_path.is_file():
        wf124_s0 = json.loads(wf124_ru_s0_path.read_text())
        s0_30ep = wf124_s0["test_metrics"]["roc_auc"]
        w0_30ep = wf124_s0["wasserstein_history"][-1]
        comparison["30ep_baseline_ru_seed0"] = {
            "test_auc": s0_30ep,
            "w_final": w0_30ep,
            "epochs": wf124_s0["epochs"],
            "hidden": 128,
            "depth": 3,
        }
        s0_50ep = None
        s1_50ep_dir = SWEEP_DIR / "ru" / "per_seed_metrics_seed0.json"
        if s1_50ep_dir.is_file():
            s1 = json.loads(s1_50ep_dir.read_text())
            s0_50ep = s1["test_metrics"]["roc_auc"]
            comparison["50ep_run_ru_seed0"] = {
                "test_auc": s0_50ep,
                "w_final": s1["wasserstein_history"][-1],
                "epochs": s1["epochs"],
                "hidden": s1.get("hidden"),
                "depth": s1.get("depth"),
            }
            comparison["delta_ru_seed0"] = {
                "delta_test_auc": s0_50ep - s0_30ep,
                "test_auc_50ep_minus_30ep": s0_50ep - s0_30ep,
                "note": (
                    "30ep = Wf-124 P3 full backbone (hidden=128); "
                    "50ep = Wf-196 smaller backbone (hidden=64). "
                    "Not apples-to-apples (different backbone capacity)."
                ),
            }

    summary["ru_30ep_vs_50ep_comparison"] = comparison

    out_path = SWEEP_DIR / "training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[wf196-agg] wrote {out_path}", flush=True)

    comparison_path = SWEEP_DIR / "30ep_vs_50ep_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"[wf196-agg] wrote {comparison_path}", flush=True)

    # Console summary
    print("", flush=True)
    print("=" * 60, flush=True)
    print("Wf-196 50-epoch CPU sweep — per-metal summary", flush=True)
    print("=" * 60, flush=True)
    for metal in FIVE_METALS:
        s = summary_per_metal[metal]
        if s.get("status") != "OK":
            print(f"  {metal}: {s.get('status', 'UNKNOWN')}", flush=True)
            continue
        print(
            f"  {metal}: test_auc = {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f} "
            f"(n={s['n_seeds']})  test_pearson={s['test_pearson_mean']:.3f}±{s['test_pearson_std']:.3f}",
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
    print(
        f"  Ru 50ep target (>=0.86 Krasnov): "
        f"{'PASS' if summary['ru_auc_target_met_50ep'] else 'BELOW'} "
        f"(Ru mean = {summary_per_metal['Ru']['test_auc_mean']:.4f})",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())