#!/usr/bin/env python3
"""Decision-layer policy: verifiable recommendations (AUD-I1) and abstention (AUD-I2).

The prediction side is no longer the bottleneck.  `msrs_code_levers` took OOF
R2 from 0.2704 to 0.4071 (DOI offset) and still no scorer beats the
``global_best`` baseline on DOI-clustered top-1 regret (0.1198).  So the next
lever is the policy that turns scores into a recommendation.

AUD-I1 -- verifiability
    A pairwise "switch m -> m'" is only checkable if **both** metals were
    measured for the same ligand, cell line, paper and duration.  Cohorts with a
    single metal can be scored but cannot support a pairwise claim.  This script
    counts what the evaluation actually covers and reports regret on the
    verifiable subset only.

AUD-I2 -- abstention
    The current scorer always answers.  A margin rule -- abstain when the
    predicted gap between the top two metals is small -- should raise precision
    at the cost of coverage.  The acceptance is precision >= 0.7 at coverage
    >= 20%; if that operating point does not exist, the answer is NEGATIVE.

Usage::

    .venv/bin/python molmetal/scripts/msrs_decision_policy.py
    .venv/bin/python molmetal/scripts/msrs_decision_policy.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.msrs_code_levers import (  # noqa: E402
    COHORT_KEYS,
    ENDPOINT,
    base_blocks,
    fit_arm,
    folds_for,
    native_codes,
)
from molmetal.scripts.msrs_recommendation import (  # noqa: E402
    build_cohorts,
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
    summarise_cohorts,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    score,
)

#: Minimum metals in a cohort for it to support a pairwise recommendation.
MIN_COHORT_SIZE = 2
#: Coverage floor for the abstention acceptance test.
MIN_COVERAGE = 0.20
#: Precision floor for the abstention acceptance test.
MIN_PRECISION = 0.70


def cohort_table(
    frame: pd.DataFrame, scorers: List[str]
) -> pd.DataFrame:
    """One row per (cohort, metal), plus the observed metal count per cohort."""
    cohorts = build_cohorts(frame, scorers, min_size=MIN_COHORT_SIZE)
    sizes = cohorts.groupby("cohort")["metal"].nunique()
    cohorts["cohort_size"] = cohorts["cohort"].map(sizes).astype(int)
    return cohorts


def verifiability_report(cohorts: pd.DataFrame) -> Dict[str, object]:
    """How much of the recommendation surface is actually checkable."""
    # Accept either a raw (cohort, metal) frame or one already carrying the
    # per-cohort metal count from ``cohort_table``.
    if "cohort_size" in cohorts.columns:
        sizes = cohorts.groupby("cohort")["cohort_size"].first()
    else:
        sizes = cohorts.groupby("cohort")["metal"].nunique()
    counts = sizes.value_counts().sort_index()
    n_pairs = sum(len(list(combinations(range(int(k)), 2))) for k in sizes)
    return {
        "n_cohorts": int(len(sizes)),
        "n_singleton_cohorts": int((sizes < MIN_COHORT_SIZE).sum()),
        "cohort_size_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "n_verifiable_pairs": int(n_pairs),
        "mean_metals_per_cohort": round(float(sizes.mean()), 3),
        "unverifiable_recommendations": 0,
        "why": (
            "Scores are only compared inside a cohort, and a cohort holds only "
            "metals that were measured for that ligand/cell/paper/time. So every "
            "recommendation the evaluator can emit is pairwise verifiable by "
            "construction; the cost is coverage, reported above."
        ),
    }


def margin_table(cohorts: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Per-cohort best score, runner-up score, margin and regret."""
    rows = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        s = np.sort(g[scorer].to_numpy(dtype=float))
        y = g["y"].to_numpy(dtype=float)
        rec = cohort_metrics(y, g[scorer].to_numpy(dtype=float))
        rows.append(
            {
                "cohort": int(cid),
                "doi": g["doi"].iloc[0],
                "fold": int(g["fold"].iloc[0]),
                "k": int(len(g)),
                "margin": float(s[1] - s[0]) if len(s) > 1 else np.inf,
                "regret": rec["regret"],
                "top1_correct": rec["top1_correct"],
            }
        )
    return pd.DataFrame(rows)


def abstention_curve(
    margins: pd.DataFrame, thresholds: Sequence[float]
) -> List[Dict[str, object]]:
    """Precision / coverage as the abstention threshold sweeps."""
    out = []
    n = len(margins)
    for thr in thresholds:
        kept = margins[margins["margin"] >= thr]
        if kept.empty:
            out.append(
                {"threshold": round(float(thr), 4), "coverage": 0.0, "precision": None,
                 "n_kept": 0, "mean_regret_kept": None}
            )
            continue
        out.append(
            {
                "threshold": round(float(thr), 4),
                "coverage": round(float(len(kept) / n), 4),
                "precision": round(float(kept["top1_correct"].mean()), 4),
                "n_kept": int(len(kept)),
                "mean_regret_kept": round(float(kept["regret"].mean()), 4),
            }
        )
    return out


def regret_matrix(cohorts: pd.DataFrame, scorers: Sequence[str]) -> pd.DataFrame:
    """Per-cohort regret for every scorer, so subsets can be compared fairly.

    Comparing scorer A on its own high-margin subset against scorer B on B's
    different subset is not a comparison.  This table lets every scorer be
    scored on the *same* kept cohorts.
    """
    rows = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        rec = {"cohort": int(cid)}
        y = g["y"].to_numpy(dtype=float)
        for s in scorers:
            rec[s] = cohort_metrics(y, g[s].to_numpy(dtype=float))["regret"]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("cohort")


def best_operating_point(curve: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Highest-precision point that still clears the coverage floor."""
    eligible = [
        c
        for c in curve
        if c["coverage"] >= MIN_COVERAGE and c["precision"] is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (c["precision"], c["coverage"]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_decision_policy.json",
    )
    args = parser.parse_args(argv)

    long = pd.read_parquet(args.long)
    descriptors = pd.read_csv(args.metal_desc)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    lig, met, cell = base_blocks(sub, descriptors, cell_vocab)
    fold_id = folds_for(groups)
    codes = native_codes(sub, cell_vocab, "full")

    global_pred, cell_pred, baseline_meta = fold_baseline_scores(
        sub["metal"].to_numpy(), sub["cell_line_key"].to_numpy(), y, fold_id
    )
    model_pred = fit_arm(
        lig, met, cell, None, 0, False, y, fold_id, cat_codes=codes
    )
    preds = {"global_best": global_pred, "cell_best": cell_pred, "cat_cell_full": model_pred}

    frame = sub[COHORT_KEYS + ["metal"]].copy()
    frame["y"] = y
    frame["fold"] = fold_id
    frame["cohort"] = frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    for name, pred in preds.items():
        frame[name] = pred

    cohorts = cohort_table(frame, list(preds))
    verifiability = verifiability_report(cohorts)
    print(
        f"[policy] cohorts={verifiability['n_cohorts']} "
        f"verifiable_pairs={verifiability['n_verifiable_pairs']} "
        f"mean_metals={verifiability['mean_metals_per_cohort']}"
    )

    policy: Dict[str, object] = {}
    regret = regret_matrix(cohorts, list(preds))
    thresholds = sorted(
        {
            round(float(v), 4)
            for v in np.concatenate(
                [
                    np.linspace(0.0, 0.5, 26),
                    np.quantile(
                        margin_table(cohorts, "cat_cell_full")["margin"].replace(
                            np.inf, np.nan
                        ).dropna(),
                        np.linspace(0.05, 0.95, 19),
                    ),
                ]
            )
        }
    )
    for scorer in ("global_best", "cell_best", "cat_cell_full"):
        margins = margin_table(cohorts, scorer)
        curve = abstention_curve(margins, thresholds)
        best = best_operating_point(curve)
        if best is not None:
            # Every scorer on the *same* kept cohorts, so the comparison is
            # like for like rather than each scorer on its own subset.
            kept = margins.loc[margins["margin"] >= best["threshold"], "cohort"]
            same = regret.loc[regret.index.isin(kept)]
            best["regret_by_scorer_on_kept"] = {
                s: round(float(same[s].mean()), 4) for s in same.columns
            }
            best["n_kept"] = int(len(kept))
            if scorer != "global_best":
                kept_rec = []
                for cid, g in cohorts[cohorts["cohort"].isin(set(kept))].groupby(
                    "cohort", sort=False
                ):
                    for s in (scorer, "global_best"):
                        rec = cohort_metrics(g["y"].to_numpy(), g[s].to_numpy())
                        kept_rec.append(
                            {"cohort": int(cid), "scorer": s, "doi": g["doi"].iloc[0], **rec}
                        )
                best["vs_global_best_on_kept"] = doi_clustered_test(
                    pd.DataFrame(kept_rec), scorer, "global_best"
                )
        full = curve[0]
        policy[scorer] = {
            "overall": {
                "coverage": 1.0,
                "precision": round(float(margins["top1_correct"].mean()), 4),
                "mean_regret": round(float(margins["regret"].mean()), 4),
            },
            "best_operating_point": best,
            "meets_acceptance": bool(
                best is not None and best["precision"] >= MIN_PRECISION
            ),
            "curve": curve,
        }
        print(
            f"  {scorer:15s} full precision={policy[scorer]['overall']['precision']:.3f} "
            f"| best@cov>={MIN_COVERAGE:.0%}: "
            + (
                f"cov={best['coverage']:.3f} prec={best['precision']:.3f} thr={best['threshold']}"
                if best
                else "none"
            )
        )

    # DOI-clustered test of the model against the baseline, on the same cohorts.
    records = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        for s in ("global_best", "cat_cell_full"):
            rec = cohort_metrics(g["y"].to_numpy(), g[s].to_numpy())
            records.append({"cohort": int(cid), "scorer": s, "doi": g["doi"].iloc[0], **rec})
    rec_df = pd.DataFrame(records)
    clustered = doi_clustered_test(rec_df, "cat_cell_full", "global_best")

    out = {
        "artifact": "msrs_decision_policy",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "cohort_keys": COHORT_KEYS,
            "min_cohort_size": MIN_COHORT_SIZE,
            "scorer_under_test": "cat_cell_full",
            "abstention_rule": "abstain when score(second best) - score(best) < threshold",
            "min_coverage": MIN_COVERAGE,
            "min_precision": MIN_PRECISION,
        },
        "n_rows": int(len(sub)),
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "baseline_meta": baseline_meta,
        "i1_verifiability": verifiability,
        "i2_abstention": policy,
        "i2_verdict": (
            "PASS"
            if policy["cat_cell_full"]["meets_acceptance"]
            else "NEGATIVE"
        ),
        "model_vs_global_best": clustered,
        "cohort_summary": {
            s: summarise_cohorts(rec_df[rec_df["scorer"] == s])
            for s in ("global_best", "cat_cell_full")
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
