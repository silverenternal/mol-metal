"""Phase 3 — held-out validation + integration smoke for symbolic regression.

Workflow: wf_deflex_symbolic_regression, Phase 3 of 5.
Author: subagent of workflow orchestration.

This script:
1. Loads the F5 formula from molmetal/models/symbolic_reward/symbolic_reward.pkl
2. Runs held-out 10% validation (10 repeats, seed=42) on the 33-cell
   complete-case subset
3. Compares F5 vs a least-squares linear baseline on the SAME splits
4. Runs a 5-cell integration smoke via compute_symbolic_reward()
5. Writes a JSON verdict + a markdown verdict report

Usage:
    uv run python molmetal/scripts/validate_symbolic_regression.py \
        --output-dir molmetal/reports/wf_deflex_symbolic_regression

Honest framing:
- Held-out R^2 with n_complete_case=33 and n_test=3 per fold is VERY
  high-variance. 10 repeats mitigates but does not eliminate single-cell
  domination. The script reports mean ± std so the variance is visible.
- F5 is the Phase 2 Pareto pick (in-sample R^2 = 1.0 on n=33). The
  expectation is that held-out R^2 will be lower than in-sample, possibly
  negative on some folds. The verdict (`symbolic_non_trivial` vs
  `linear_baseline_sufficient`) is reported *as is*; the report calls
  out any caveats.
- The reward target is a heuristic composite (sa_norm + tau + metal),
  not a held-out oracle, so even a "perfect" regression is fitting the
  data-generating process not a downstream metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.symbolic_regression import (  # noqa: E402
    FORMULA_FAMILIES,
    build_reward_target,
    compare_symbolic_vs_linear,
    compute_symbolic_reward,
    held_out_validation,
    linear_baseline_validation,
    load_measured_cells,
)


def run_smoke(n_cells: int = 5) -> Dict[str, Any]:
    """Compute symbolic_reward on n sample cells; verify range & finiteness.

    Returns dict with per-cell rewards and aggregate stats.
    """
    df = load_measured_cells()
    # Take one cell from each cohort (PathA + Pilot + Novel).
    sample_indices = []
    seen_cohorts: set = set()
    for i, cohort in enumerate(df["cohort"].tolist()):
        if cohort not in seen_cohorts and len(sample_indices) < n_cells:
            sample_indices.append(i)
            seen_cohorts.add(cohort)
    # If < n_cells cohorts, top up from front.
    for i in range(len(df)):
        if len(sample_indices) >= n_cells:
            break
        if i not in sample_indices:
            sample_indices.append(i)

    rows = []
    rewards = []
    for i in sample_indices:
        row = df.iloc[i]
        r = compute_symbolic_reward(row)
        rows.append({
            "cell_id": row["cell_id"],
            "cohort": row["cohort"],
            "diversity_tanimoto": float(row["diversity_tanimoto"])
                if not np.isnan(row["diversity_tanimoto"]) else None,
            "sa_mean_norm": float(row["sa_mean_norm"])
                if not np.isnan(row["sa_mean_norm"]) else None,
            "symbolic_reward": r,
        })
        rewards.append(r)

    rewards_arr = np.array(rewards, dtype=float)
    return {
        "n_cells": len(rewards),
        "rewards": rewards,
        "rewards_min": float(rewards_arr.min()),
        "rewards_max": float(rewards_arr.max()),
        "rewards_mean": float(rewards_arr.mean()),
        "all_finite": bool(np.all(np.isfinite(rewards_arr))),
        "in_reasonable_range": bool(
            np.all((rewards_arr >= 0.0) & (rewards_arr <= 3.0))
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 held-out validation for symbolic-reward regression.")
    parser.add_argument("--output-dir",
                        default="molmetal/reports/wf_deflex_symbolic_regression",
                        help="Where to write phase3_validate.md + JSON verdict.")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Number of held-out repeats (default 10).")
    parser.add_argument("--holdout-frac", type=float, default=0.10,
                        help="Held-out fraction (default 0.10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the held-out splits.")
    parser.add_argument("--family", default="F5",
                        choices=list(FORMULA_FAMILIES),
                        help="Family to validate (default F5 = Phase 2 pick).")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_measured_cells()
    print(f"[validate_symbolic_regression] loaded {len(df)} cells")
    print(f"[validate_symbolic_regression] cohorts: "
          f"{df['cohort'].value_counts().to_dict()}")

    # 1) Held-out validation for F5.
    print(f"\n[validate_symbolic_regression] held-out validation "
          f"(family={args.family}, holdout={args.holdout_frac}, "
          f"n_repeats={args.n_repeats}, seed={args.seed})")
    sym = held_out_validation(
        df, family=args.family, holdout_frac=args.holdout_frac,
        n_repeats=args.n_repeats, seed=args.seed,
    )
    print(f"  F5 held-out R^2 = {sym['held_out_r2_mean']:.4f} "
          f"± {sym['held_out_r2_std']:.4f}")
    print(f"  F5 held-out Pearson r = {sym['held_out_pearson_mean']:.4f} "
          f"± {sym['held_out_pearson_std']:.4f}")
    print(f"  F5 in-sample R^2 = {sym['in_sample_r2']:.4f}  "
          f"(n_complete_case = {sym['n_complete_case']}, "
          f"n_repeats = {sym['n_repeats']}, "
          f"n_skipped = {sym.get('n_skipped', 0)})")

    # 2) Linear baseline (same splits, same seed).
    base = linear_baseline_validation(
        df, holdout_frac=args.holdout_frac, n_repeats=args.n_repeats,
        seed=args.seed,
    )
    print(f"\n[validate_symbolic_regression] linear baseline "
          f"(5 features, full LS)")
    print(f"  linear held-out R^2 = {base['baseline_r2_mean']:.4f} "
          f"± {base['baseline_r2_std']:.4f}")
    print(f"  linear held-out Pearson r = {base['baseline_pearson_mean']:.4f} "
          f"± {base['baseline_pearson_std']:.4f}")

    # 3) Head-to-head.
    cmp = compare_symbolic_vs_linear(
        df, family=args.family, holdout_frac=args.holdout_frac,
        n_repeats=args.n_repeats, seed=args.seed, lift_threshold=0.05,
    )
    print(f"\n[validate_symbolic_regression] verdict = {cmp['verdict']}")
    print(f"  symbolic R^2 = {cmp['symbolic_r2_mean']:.4f} ± "
          f"{cmp['symbolic_r2_std']:.4f}")
    print(f"  linear R^2 = {cmp['linear_r2_mean']:.4f} ± "
          f"{cmp['linear_r2_std']:.4f}")
    print(f"  lift (sym - linear) = {cmp['lift']:.4f} "
          f"(threshold ±{cmp['lift_threshold']:.2f})")

    # 4) Integration smoke: 5 cells.
    smoke = run_smoke(n_cells=5)
    print(f"\n[validate_symbolic_regression] integration smoke "
          f"(n_cells = {smoke['n_cells']})")
    for row in smoke["rows"]:
        print(f"  {row['cell_id']:>22} ({row['cohort']:>22})  "
              f"tau={row['diversity_tanimoto']}, "
              f"sa_norm={row['sa_mean_norm']}, "
              f"symbolic_R = {row['symbolic_reward']:.4f}")
    print(f"  rewards range: [{smoke['rewards_min']:.4f}, "
          f"{smoke['rewards_max']:.4f}], "
          f"mean={smoke['rewards_mean']:.4f}")
    print(f"  all_finite = {smoke['all_finite']}, "
          f"in_reasonable_range = {smoke['in_reasonable_range']}")

    # 5) Build verdict JSON.
    verdict = {
        "workflow": "wf_deflex_symbolic_regression",
        "phase": 3,
        "n_cells_total": int(len(df)),
        "n_complete_case": sym["n_complete_case"],
        "n_repeats": args.n_repeats,
        "n_skipped": sym.get("n_skipped", 0),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "family": args.family,
        "symbolic": {
            "held_out_r2_mean": sym["held_out_r2_mean"],
            "held_out_r2_std": sym["held_out_r2_std"],
            "held_out_pearson_mean": sym["held_out_pearson_mean"],
            "held_out_pearson_std": sym["held_out_pearson_std"],
            "in_sample_r2": sym["in_sample_r2"],
            "formula": sym.get("formula_str", ""),
        },
        "linear_baseline": {
            "r2_mean": base["baseline_r2_mean"],
            "r2_std": base["baseline_r2_std"],
            "pearson_mean": base["baseline_pearson_mean"],
            "pearson_std": base["baseline_pearson_std"],
            "n_skipped": base.get("n_skipped", 0),
            "features": FORMULA_FAMILIES[args.family]["features"],
        },
        "head_to_head": {
            "lift": cmp["lift"],
            "lift_threshold": cmp["lift_threshold"],
            "verdict": cmp["verdict"],
        },
        "smoke": smoke,
    }
    json_path = output_dir / "phase3_validate.json"
    with open(json_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n[validate_symbolic_regression] wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
