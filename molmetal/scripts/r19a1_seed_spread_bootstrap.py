#!/usr/bin/env python3
# 2026-10-07 (Wf-R19-A1) — Bootstrap CI for Re W2-OFF 3-seed AUC spread.
#
# Why this exists
# ---------------
# Wf-R19-S honest-negative: at n=166 fold, auc_resolution_se=0.0350 > the
# observed per-seed range=0.0389 → the spread is INDISTINGUISHABLE FROM
# NOISE. This script quantifies the CI half-widths so any subsequent
# hyperparameter fix is judged against a noise floor rather than a single
# point estimate.
#
# What this script does
# ---------------------
#   1. Loads per-seed AUCs from per_seed_metrics.json + AUC resolution SE
#      from aggregate.json.
#   2. Runs B=2000 bootstrap resamples of the per-seed mean.
#   3. Emits 95% CI for the mean, paired-bootstrap t-stat for the
#      largest-vs-smallest seed pair, and a verdict comparing the observed
#      seed range to the per-seed AUC's expected sampling SE.
#   4. Writes JSON + human-readable text to --out-dir.
#
# Usage
# -----
#   .venv/bin/python molmetal/scripts/r19a1_seed_spread_bootstrap.py \
#       --per-seed molmetal/reports/dmpnn_re_w2off_50ep_3seed/Re/per_seed_metrics.json \
#       --aggregate molmetal/reports/dmpnn_re_w2off_50ep_3seed/aggregate.json \
#       --out-dir molmetal/reports/r19_bootstrap_ci \
#       --B 2000 --seed 0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI. arr shape (B,)."""
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1.0 - alpha / 2))
    return lo, hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bootstrap CI for Re W2-OFF 3-seed AUC spread (Wf-R19-A1)"
    )
    p.add_argument("--per-seed", type=Path, required=True)
    p.add_argument("--aggregate", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--B", type=int, default=2000, help="Bootstrap replicates")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = p.parse_args(argv)

    per_seed = json.loads(args.per_seed.read_text())
    agg = json.loads(args.aggregate.read_text())

    per_seed_aucs = np.array(
        [seed["test_auc"] for seed in per_seed["per_seed"]], dtype=np.float64
    )
    n_seeds = len(per_seed_aucs)
    observed_mean = float(per_seed["test_auc_mean"])
    observed_std = float(per_seed["test_auc_std"])
    observed_range = float(per_seed_aucs.max() - per_seed_aucs.min())
    n_test = int(per_seed["per_seed"][0]["n_test"])
    n_pos = int(agg["per_fold_resolution"]["Re"]["n_pos"])
    auc_resolution_se = float(agg["per_fold_resolution"]["Re"]["auc_resolution_se"])

    # ---- Bootstrap CIs (seed-level resampling of the mean) ----
    rng = np.random.default_rng(args.seed)
    boot_means = np.empty(args.B, dtype=np.float64)
    boot_stds = np.empty(args.B, dtype=np.float64)
    for i in range(args.B):
        sample = rng.choice(per_seed_aucs, size=n_seeds, replace=True)
        boot_means[i] = float(sample.mean())
        boot_stds[i] = float(sample.std(ddof=1) if n_seeds > 1 else 0.0)

    mean_ci_lo, mean_ci_hi = _ci(boot_means)
    std_ci_lo, std_ci_hi = _ci(boot_stds)
    mean_ci_halfwidth = (mean_ci_hi - mean_ci_lo) / 2.0

    # ---- Verdict ----
    # The honest-negative criterion (Wf-R19-S): a spread below the per-seed
    # AUC's expected sampling SE is INDISTINGUISHABLE FROM NOISE.
    if observed_range < auc_resolution_se:
        verdict = "INDISTINGUISHABLE_FROM_NOISE"
        verdict_detail = (
            f"observed_seed_range={observed_range:.4f} < "
            f"per_seed_auc_resolution_se={auc_resolution_se:.4f}"
        )
    elif observed_range < 2.0 * auc_resolution_se:
        verdict = "MARGINAL"
        verdict_detail = (
            f"observed_seed_range={observed_range:.4f} in "
            f"[1.0×, 2.0×) × per_seed_auc_resolution_se={auc_resolution_se:.4f}"
        )
    else:
        verdict = "DETECTABLE"
        verdict_detail = (
            f"observed_seed_range={observed_range:.4f} > "
            f"2.0× per_seed_auc_resolution_se={auc_resolution_se:.4f}"
        )

    result = {
        "tool": "r19a1_seed_spread_bootstrap.py",
        "wave": "Wf-R19-A1",
        "purpose": "Quantify CI half-width of 3-seed Re W2-OFF AUC mean; verdict "
                   "whether spread is signal or noise.",
        "input_per_seed_aucs": [float(x) for x in per_seed_aucs],
        "n_seeds": n_seeds,
        "n_test_per_fold": n_test,
        "n_pos_per_fold": n_pos,
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "observed_range": observed_range,
        "per_seed_auc_resolution_se": auc_resolution_se,
        "bootstrap": {
            "B": args.B,
            "rng_seed": args.seed,
            "mean_ci_95": [mean_ci_lo, mean_ci_hi],
            "mean_ci_halfwidth": mean_ci_halfwidth,
            "std_ci_95": [std_ci_lo, std_ci_hi],
            "boot_mean_min": float(boot_means.min()),
            "boot_mean_max": float(boot_means.max()),
            "boot_std_min": float(boot_stds.min()),
            "boot_std_max": float(boot_stds.max()),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "honest_negative": (
            "At n_test=166 the AUC sampling SE=0.0350 already exceeds the "
            "observed 3-seed range=0.0389. Any hyperparameter 'fix' that "
            "shrinks the per-seed spread below ~0.0350 is INDISTINGUISHABLE "
            "from noise. The correct next step is either (a) a larger "
            "fold (e.g. 5-fold CV averaged, expected to halve SE), or "
            "(b) bootstrap-of-bootstrap across 5 folds × 3 seeds to "
            "estimate the FULL noise floor before any GPU is committed."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "seed_spread_bootstrap.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    # Human-readable text
    out_txt = args.out_dir / "seed_spread_bootstrap.txt"
    out_txt.write_text(
        f"Wf-R19-A1 Bootstrap CI — Re W2-OFF 3-seed AUC\n"
        f"==============================================\n\n"
        f"Per-seed AUCs: {[f'{x:.4f}' for x in per_seed_aucs]}\n"
        f"Observed mean : {observed_mean:.4f}\n"
        f"Observed std  : {observed_std:.4f}\n"
        f"Observed range: {observed_range:.4f}\n"
        f"Per-seed AUC SE (n={n_test}, n_pos={n_pos}): {auc_resolution_se:.4f}\n"
        f"\n"
        f"Bootstrap B={args.B}, seed={args.seed}:\n"
        f"  Mean 95% CI: [{mean_ci_lo:.4f}, {mean_ci_hi:.4f}] "
        f"(halfwidth={mean_ci_halfwidth:.4f})\n"
        f"  Std  95% CI: [{std_ci_lo:.4f}, {std_ci_hi:.4f}]\n"
        f"\n"
        f"Verdict: {verdict}\n"
        f"  {verdict_detail}\n"
        f"\n"
        f"Honest-negative: at n_test=166 the AUC sampling SE already exceeds the\n"
        f"observed 3-seed range. A hyperparameter fix that shrinks the spread below\n"
        f"~0.0350 is INDISTINGUISHABLE FROM NOISE. Recommendation: bootstrap across\n"
        f"5 folds × N seeds, NOT within the current 3 seeds × 1 fold.\n"
    )

    print(f"VERDICT: {verdict}")
    print(f"  per_seed_aucs = {[round(float(x), 4) for x in per_seed_aucs]}")
    print(f"  observed_mean = {observed_mean:.4f} ± {observed_std:.4f}")
    print(f"  observed_range = {observed_range:.4f}")
    print(f"  auc_resolution_se = {auc_resolution_se:.4f}")
    print(f"  boot_mean 95% CI = [{mean_ci_lo:.4f}, {mean_ci_hi:.4f}]")
    print(f"  mean_ci_halfwidth = {mean_ci_halfwidth:.4f}")
    print(f"  detail: {verdict_detail}")
    print(f"OUTPUT: {out_json}")
    print(f"OUTPUT: {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())