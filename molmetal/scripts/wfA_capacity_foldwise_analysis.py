#!/usr/bin/env python3
"""Fold-level follow-up to the device-matched capacity pair (no training).

=====================================================================
Why this exists
=====================================================================
``wfA_capacity_pair_analysis.py`` tests the capacity effect at the seed level:
5 paired deltas per metal, paired t-test, Holm. That is the valid primary test,
but it throws away the per-fold detail that both runs already recorded. Three
questions are answerable from that detail without a single new training run:

1. **Is the gain spread across folds, or carried by a few?** A +0.018 metal mean
   could be +0.018 everywhere or +0.10 on two folds and 0 on the rest. Those have
   different implications for what capacity is doing.
2. **Does capacity lift the worst folds?** If h512 mainly rescues the folds where
   h128 collapsed, that is a different mechanism than a uniform shift.
3. **How does the effect scale with cohort size?** With 5 metals this cannot be
   established as a law, but the sign pattern can be reported honestly next to
   the point estimates.

It also derives the **size-adaptive width policy** number: what the five-metal
column looks like if width is chosen per metal from the measured rule
(``hidden=512`` only where n >= 1000) instead of uniformly.

Statistical caution (recorded in the artifact, not just here)
-------------------------------------------------------------
There are 125 (metal, seed, fold) pairs, but they are NOT 125 independent
observations: the 5 folds within a seed come from one split of one cohort, and
the 5 seeds reuse the same rows. The fold-level t-test is therefore
pseudo-replicated and its nominal p is not trustworthy. The direction of the
error is not fixed either: measured, Ru/Ir are ~0 under both, Rh/Os barely move,
and Re's naive p is LARGER than the clustered one (0.071 vs 0.038). This script
computes it only to show the gap next to the seed-clustered test (the one used
for the verdict). The fold-level numbers are descriptive.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wfA_capacity_foldwise_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import _paired_stats  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.wfA_capacity_pair_analysis import (  # noqa: E402
    _check_invariants,
    _per_seed_auc,
)

WIDE_HIDDEN = 512
NARROW_HIDDEN = 128
# Measured cut: both cohorts with n >= 1000 show a Holm-significant gain; the
# three with n <= 334 do not (Rh -0.0027, Os -0.0029, Re +0.0114 ns). Kept as a
# named constant so the policy is auditable rather than a magic number in a call.
MIN_N_FOR_WIDE = 1000


def _fold_records(metal_block: Dict[str, object]) -> Dict[tuple, Dict[str, float]]:
    out: Dict[tuple, Dict[str, float]] = {}
    for split in metal_block["per_split"]:
        seed = int(split["split_seed"])
        for f in split["per_fold"]:
            out[(seed, int(f["fold"]))] = {
                "dmpnn_auc": float(f["dmpnn_auc"]),
                "xgb_auc": float(f["xgb_auc"]),
                "n_test": int(f["n_test"]),
                "pos_rate_test": float(f["pos_rate_test"]),
            }
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fold-level capacity analysis (no training).")
    p.add_argument("--h128", default="metrics/by_metric/dmpnn_within_metal_cap128_pf_gpu0.json")
    p.add_argument("--h512", default="metrics/by_metric/dmpnn_within_metal_cap512_pf_gpu0.json")
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/dmpnn_within_metal_capacity_foldwise.json",
    )
    args = p.parse_args(argv)

    h128 = json.loads((PROJECT_ROOT / args.h128).read_text())
    h512 = json.loads((PROJECT_ROOT / args.h512).read_text())

    invariants = _check_invariants(h128, h512)
    if invariants["status"] != "PASS":
        print("INVARIANT FAIL:", invariants["mismatches"], file=sys.stderr)
        return 1

    per_metal: Dict[str, object] = {}
    scaling_n: List[float] = []
    scaling_effect: List[float] = []

    for metal in h128["per_metal"]:
        b128 = h128["per_metal"][metal]
        b512 = h512["per_metal"][metal]
        f128 = _fold_records(b128)
        f512 = _fold_records(b512)
        if set(f128) != set(f512):
            print(f"FAIL: {metal} fold keys differ", file=sys.stderr)
            return 1

        keys = sorted(f128)
        xgb_diff = max(abs(f128[k]["xgb_auc"] - f512[k]["xgb_auc"]) for k in keys)
        d = np.array([f512[k]["dmpnn_auc"] - f128[k]["dmpnn_auc"] for k in keys], dtype=float)
        a128 = np.array([f128[k]["dmpnn_auc"] for k in keys], dtype=float)
        a512 = np.array([f512[k]["dmpnn_auc"] for k in keys], dtype=float)

        # Seed-clustered test: average the 5 folds inside each seed first, so the
        # unit of independence is the split, not the fold.
        seeds = sorted({k[0] for k in keys})
        seed_deltas = np.array(
            [np.mean([f512[(s, f)]["dmpnn_auc"] - f128[(s, f)]["dmpnn_auc"]
                      for (ss, f) in keys if ss == s]) for s in seeds],
            dtype=float,
        )
        clustered = _paired_stats(seed_deltas)
        naive = _paired_stats(d)

        # Concentration: share of the total signed delta contributed by the
        # k most positive folds. >1 means the rest are net negative.
        order = np.argsort(d)[::-1]
        total = float(d.sum())
        top5 = float(d[order[:5]].sum())
        top_share = (top5 / total) if total > 0 else None

        # Worst-fold behaviour: does h512 rescue the folds where h128 was worst?
        worst_idx = np.argsort(a128)[: max(1, len(a128) // 5)]  # bottom 20% of folds
        worst_delta = float(d[worst_idx].mean())
        rest_delta = float(np.delete(d, worst_idx).mean())

        n_unique = int(b128["n_unique"])
        effect = float(seed_deltas.mean())
        per_metal[metal] = {
            "n_unique": n_unique,
            "n_fold_pairs": len(keys),
            "fold_level": {
                "mean_delta": round(float(d.mean()), 6),
                "median_delta": round(float(np.median(d)), 6),
                "sd_delta": round(float(d.std(ddof=1)), 6),
                "n_folds_positive": int((d > 0).sum()),
                "frac_folds_positive": round(float((d > 0).mean()), 4),
                "min_delta": round(float(d.min()), 6),
                "max_delta": round(float(d.max()), 6),
                "top5_folds_share_of_total_delta": (round(top_share, 4) if top_share is not None else None),
                "worst_quintile_folds_mean_delta": round(worst_delta, 6),
                "other_folds_mean_delta": round(rest_delta, 6),
                "h128_fold_sd": round(float(a128.std(ddof=1)), 6),
                "h512_fold_sd": round(float(a512.std(ddof=1)), 6),
            },
            "seed_clustered_test": clustered,
            "naive_fold_test_DO_NOT_USE": {
                "p_one_sided_ttest": naive["p_one_sided_ttest"],
                "n": naive["n_seeds"],
                "note": (
                    "125 fold pairs are pseudo-replicates (5 folds share one split, 5 seeds share "
                    "one cohort). The nominal p is not trustworthy and the error is not one-sided: "
                    "measured, Re's naive p (0.071) is LARGER than its clustered p (0.038). "
                    "Reported only to show the gap against the seed-clustered test."
                ),
            },
            "fold_identity": {"max_abs_xgb_diff": xgb_diff, "identical": xgb_diff <= 1e-12},
        }
        scaling_n.append(n_unique)
        scaling_effect.append(effect)

    # Size-adaptive width policy: pick the measured winner per metal.
    policy_rows: Dict[str, object] = {}
    for metal, r in per_metal.items():
        n = r["n_unique"]
        wide = n >= MIN_N_FOR_WIDE
        b128 = h128["per_metal"][metal]
        b512 = h512["per_metal"][metal]
        chosen_block = b512 if wide else b128
        krasnov = b128.get("krasnov_reported_auc")
        policy_rows[metal] = {
            "n_unique": n,
            "chosen_hidden": WIDE_HIDDEN if wide else NARROW_HIDDEN,
            "rule": f"hidden={WIDE_HIDDEN} iff n >= {MIN_N_FOR_WIDE}",
            "chosen_auc": float(chosen_block["dmpnn_auc"]),
            "uniform_h128_auc": float(b128["dmpnn_auc"]),
            "gain_vs_uniform_h128": round(float(chosen_block["dmpnn_auc"]) - float(b128["dmpnn_auc"]), 6),
            "gap_to_xgb": float(chosen_block["delta_dmpnn_minus_xgb"]),
            "beats_xgb": bool(chosen_block["delta_dmpnn_minus_xgb"] > 0),
            "beats_krasnov": (None if krasnov is None else bool(chosen_block["dmpnn_auc"] > krasnov)),
        }

    log_n = np.log10(np.array(scaling_n, dtype=float))
    eff = np.array(scaling_effect, dtype=float)
    slope = None
    if log_n.std() > 0:
        slope = float(np.polyfit(log_n, eff, 1)[0])

    report = {
        "metric": "dmpnn_within_metal_capacity_foldwise",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "Fold-level decomposition of the device-matched h128 vs h512 capacity pair. Derived "
            "entirely from the two MEASURED JSONs; no training."
        ),
        "unit": "ROC-AUC difference (h512 - h128) on identical folds",
        "source": "molmetal/scripts/wfA_capacity_foldwise_analysis.py",
        "inputs": {"h128": args.h128, "h512": args.h512},
        "pairing": invariants,
        "per_metal": per_metal,
        "policy": {
            "name": "size_adaptive_width",
            "rule": f"hidden={WIDE_HIDDEN} iff n_unique >= {MIN_N_FOR_WIDE}, else {NARROW_HIDDEN}",
            "why_this_cut": (
                "Measured, not tuned: both metals above the cut (Ru n=4793, Ir n=1281) have a "
                "Holm-significant capacity gain; all three below it (Rh 334, Os 333, Re 250) have "
                "point estimates between -0.003 and +0.011 and none survives correction. The cut "
                "sits in the empty interval between 334 and 1281; it is not fitted to a boundary."
            ),
            "per_metal": policy_rows,
            "still_does_not_beat_xgb_metals": [m for m, r in policy_rows.items() if not r["beats_xgb"]],
        },
        "scaling": {
            "n_metals": len(scaling_n),
            "pearson_log10n_vs_effect": _pearson(log_n, eff),
            "ols_slope_per_decade": (round(slope, 6) if slope is not None else None),
            "caveat": (
                "n=5. This is a descriptive fit, NOT evidence of a scaling law: the three small "
                "cohorts sit inside noise of zero and Re is positive, so the relation is driven by "
                "two high-n points. Do not quote a slope as a finding."
            ),
        },
        "honest_framing": {
            "do_not_claim": [
                "Fold-level p-values: 125 fold pairs are pseudo-replicates; only the seed-clustered test is valid.",
                "A capacity scaling law from 5 metals.",
                "That the size-adaptive policy beats XGBoost -- it beats it on 0/5 metals.",
                "That the policy is a validated design rule on unseen cohorts; it is a post-hoc cut on 5 metals.",
            ],
            "what_this_says": (
                "Where the capacity gain lives inside each metal, whether it rescues bad folds, and "
                "what the five-metal column would look like under a size-adaptive width choice."
            ),
        },
    }

    out = PROJECT_ROOT / args.out_metric
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))

    print(f"\n{'metal':>5} {'n':>6} {'folds+':>7} {'mean':>8} {'worst20%':>9} {'rest':>8} {'p_clust':>8} {'p_naive':>8}")
    for m, r in per_metal.items():
        fl = r["fold_level"]
        print(
            f"{m:>5} {r['n_unique']:>6} {fl['n_folds_positive']:>3}/{r['n_fold_pairs']:<3} "
            f"{fl['mean_delta']:>+8.4f} {fl['worst_quintile_folds_mean_delta']:>+9.4f} "
            f"{fl['other_folds_mean_delta']:>+8.4f} "
            f"{r['seed_clustered_test']['p_one_sided_ttest']:>8.4f} "
            f"{r['naive_fold_test_DO_NOT_USE']['p_one_sided_ttest']:>8.4f}"
        )
    print(f"\npolicy: {report['policy']['rule']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
