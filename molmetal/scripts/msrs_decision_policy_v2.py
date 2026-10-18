#!/usr/bin/env python3
"""FIX-E2 — the decision layer, re-measured on complex features (fixes NEG-16).

NEG-16 recorded that on the full 424-cohort surface the model scored mean regret
0.1278 against the ``global_best`` baseline's 0.1198 — it *lost*.  Every
abstention result in AUD-I2 was measured on a 27.6% high-margin subset of a
representation that could not tell 5,861 complexes apart from their 2,846 main
fragments.  A policy tuned on that representation is not a policy.

This script re-runs the decision layer on the FIX-L1 complex features and keeps
the old fragment model in the table as the reference.  Both grouping protocols
are reported, because the abstention subset is exactly where the split choice
matters most.

Acceptance (written before the run):

* full-cohort mean regret of the complex model < 0.1198 (the baseline), **and**
* DOI-clustered Holm-adjusted one-sided p < 0.05 against ``global_best``.

Anything else is NEGATIVE, and the abstention subset may not be quoted as the
result.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_decision_policy_v2.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.features.build import build
from molmetal.features.spec import COMPLEX, LEGACY_MAIN
from molmetal.scripts.msrs_code_levers import (
    COHORT_KEYS,
    ENDPOINT,
    fit_arm,
    folds_for,
    native_codes,
)
from molmetal.scripts.msrs_decision_policy import (
    MIN_COHORT_SIZE,
    MIN_COVERAGE,
    MIN_PRECISION,
    abstention_curve,
    best_operating_point,
    cohort_table,
    margin_table,
    regret_matrix,
    verifiability_report,
)
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_recommendation import (
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
    summarise_cohorts,
)
from molmetal.scripts.msrs_surface_model import (
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    holm,
)

#: Baseline mean regret the complex model must beat (NEG-16, AUD-I2).
BASELINE_FULL_REGRET = 0.1198
ALPHA = 0.05
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)


def _fit_model(
    ligand: sparse.csr_matrix,
    codes: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    empty = sparse.csr_matrix((len(y), 0), dtype=np.float32)
    return fit_arm(ligand, empty, empty, None, 0, False, y, fold_id, cat_codes=codes)


def _records(cohorts: pd.DataFrame, scorers: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        for s in scorers:
            rec = cohort_metrics(g["y"].to_numpy(), g[s].to_numpy())
            rows.append({"cohort": int(cid), "scorer": s, "doi": g["doi"].iloc[0], **rec})
    return pd.DataFrame(rows)


def run_split(
    sub: pd.DataFrame,
    y: np.ndarray,
    ligand_complex: sparse.csr_matrix,
    ligand_fragment: sparse.csr_matrix,
    codes: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    fold_id = folds_for(groups)
    global_pred, cell_pred, baseline_meta = fold_baseline_scores(
        sub["metal"].to_numpy(), sub["cell_line_key"].to_numpy(), y, fold_id
    )
    preds = {
        "global_best": global_pred,
        "cell_best": cell_pred,
        "model_complex": _fit_model(ligand_complex, codes, y, fold_id),
        "model_fragment": _fit_model(ligand_fragment, codes, y, fold_id),
    }

    frame = sub[COHORT_KEYS + ["metal"]].copy()
    frame["y"] = y
    frame["fold"] = fold_id
    frame["cohort"] = (
        frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    )
    for name, pred in preds.items():
        frame[name] = pred

    cohorts = cohort_table(frame, list(preds))
    verifiability = verifiability_report(cohorts)
    regret = regret_matrix(cohorts, list(preds))
    full = {
        s: round(float(regret[s].mean()), 4) for s in regret.columns
    }

    # Abstention is reported, but it is not the headline: NEG-16 says a
    # high-margin subset may not stand in for the full surface.
    margins = margin_table(cohorts, "model_complex")
    thresholds = sorted(
        {
            round(float(v), 4)
            for v in np.linspace(0.0, 0.5, 26).tolist()
            + np.quantile(
                margins["margin"].replace(np.inf, np.nan).dropna(),
                np.linspace(0.05, 0.95, 19),
            ).tolist()
        }
    )
    curve = abstention_curve(margins, thresholds)
    best = best_operating_point(curve)

    rec = _records(cohorts, list(preds))
    clustered = {
        "complex_vs_global": doi_clustered_test(rec, "model_complex", "global_best"),
        "complex_vs_fragment": doi_clustered_test(rec, "model_complex", "model_fragment"),
    }
    pvals = {
        k: float(v.get("p_one_sided_doi", 1.0)) for k, v in clustered.items()
    }
    adjusted = holm(pvals)
    for k in clustered:
        clustered[k]["p_holm"] = round(float(adjusted[k]), 6)

    return {
        "n_rows": int(len(sub)),
        "n_cohorts": int(verifiability["n_cohorts"]),
        "verifiability": verifiability,
        "baseline_meta": baseline_meta,
        "full_cohort_regret": full,
        "top1_accuracy": {
            s: round(float(rec.loc[rec["scorer"] == s, "top1_correct"].mean()), 4)
            for s in regret.columns
        },
        "abstention": {
            "curve": curve,
            "best_operating_point": best,
            "note": (
                "report only — the acceptance is on full_cohort_regret, per NEG-16"
            ),
        },
        "clustered_tests": clustered,
        "cohort_summary": {
            s: summarise_cohorts(rec[rec["scorer"] == s]) for s in regret.columns
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_decision_policy_v2.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    long = pd.read_parquet(args.long)
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
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    codes = native_codes(sub, cell_vocab, "full")

    complex_bundle = build(sub, COMPLEX, cell_vocabulary_override=cell_vocab)
    fragment_bundle = build(sub, LEGACY_MAIN, cell_vocabulary_override=cell_vocab)

    splits: Dict[str, object] = {}
    for split, col in (
        (SPLIT_FRAGMENT_GROUPED, "ligand_main_fragment"),
        (SPLIT_CANONICAL_GROUPED, "ligand_canonical"),
    ):
        print(f"[policy-v2] split={split}")
        splits[split] = run_split(
            sub,
            y,
            complex_bundle.X,
            fragment_bundle.X,
            codes,
            sub[col].to_numpy(),
        )
        full = splits[split]["full_cohort_regret"]
        print(
            f"  full regret: global={full['global_best']:.4f} "
            f"fragment={full['model_fragment']:.4f} complex={full['model_complex']:.4f}"
        )

    canonical = splits[SPLIT_CANONICAL_GROUPED]
    headline = canonical["full_cohort_regret"]["model_complex"]
    p_holm = canonical["clustered_tests"]["complex_vs_global"]["p_holm"]
    passed = headline < BASELINE_FULL_REGRET and p_holm < ALPHA

    payload = {
        "artifact": "msrs_decision_policy_v2",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "complex_spec_hash": complex_bundle.spec_hash,
            "fragment_spec_hash": fragment_bundle.spec_hash,
        },
        "design": {
            "cohort_keys": COHORT_KEYS,
            "min_cohort_size": MIN_COHORT_SIZE,
            "scorers": ["global_best", "cell_best", "model_fragment", "model_complex"],
            "headline_arm": "model_complex",
            "arm_ligand_source": {
                "global_best": "ligand_canonical",
                "cell_best": "ligand_canonical",
                "model_fragment": "ligand_main_fragment",
                "model_complex": "ligand_canonical",
            },
            "baseline_full_regret": BASELINE_FULL_REGRET,
            "baseline_source": "NEG-16 / AUD-I2 full-cohort mean regret of global_best",
            "alpha": ALPHA,
            "correction": "Holm over the two clustered tests per split",
            "min_coverage": MIN_COVERAGE,
            "min_precision": MIN_PRECISION,
        },
        "n_rows": int(len(sub)),
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "n_complexes": int(sub["ligand_canonical"].nunique()),
        "splits": splits,
        "headline": {
            "split": SPLIT_CANONICAL_GROUPED,
            "full_cohort_regret_model_complex": headline,
            "p_holm_vs_global_best": p_holm,
        },
        "verdict": "POSITIVE" if passed else "NEGATIVE",
        "reading": (
            f"canonical 分组全量 regret：complex {headline:.4f} vs "
            f"baseline {BASELINE_FULL_REGRET:.4f}；DOI 聚类 Holm p={p_holm:.4f}。"
            "弃权曲线仅作报告：按 NEG-16，高 margin 子集不能当交付。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(
        f"[policy-v2] verdict={payload['verdict']} regret={headline:.4f} "
        f"p_holm={p_holm:.4f}"
    )
    print(f"[policy-v2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
