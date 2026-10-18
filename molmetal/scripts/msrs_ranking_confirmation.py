#!/usr/bin/env python3
"""FIX-E4 — does the centered-target gain survive unseen papers?

NEG-18 left the strongest candidate in the project unconfirmed: within-cohort
centering cut mean regret from 0.1242 to 0.0939 and raised top-1 from 0.566 to
0.653, but the DOI-clustered test could not separate that from between-paper
heterogeneity (Holm p = 0.0598 fragment / 0.6737 canonical, 61 clusters).

Clustering corrects the variance of an estimate fitted on the same papers; it
does not ask whether the rule transfers to a paper the model has never seen.
This script asks exactly that: ``GroupKFold(5)`` **grouped by DOI**, so every
fold holds out entire papers.  A cohort is one ligand and lives inside one DOI,
so a held-out cohort shares nothing with training — which is the deployment
condition.

Arms: ``global_best`` (fold-safe), ``regression``, ``centered``.
Acceptance is unchanged and pre-registered: mean regret < 0.1198 and
DOI-clustered Holm p < 0.05.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_ranking_confirmation.py
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
from molmetal.features.cell import cell_vocabulary, native_codes
from molmetal.features.coordination import coordination_block
from molmetal.features.metal import state_block
from molmetal.scripts.msrs_code_levers import COHORT_KEYS, ENDPOINT, fit_arm, folds_for
from molmetal.scripts.msrs_dual_split import write_single_protocol_report
from molmetal.scripts.msrs_ranking_objective import (
    ALPHA,
    BASELINE_FULL_REGRET,
    LIGAND_ONLY,
    _centered_oof,
    _evaluate,
)
from molmetal.scripts.msrs_recommendation import fold_baseline_scores
from molmetal.scripts.msrs_surface_model import (
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ranking_confirmation.json",
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
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    vocab = cell_vocabulary(
        sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        sub["metal"].astype(str).tolist(),
        sub["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )

    bundle = build(sub, LIGAND_ONLY)
    state, _ = state_block(sub)
    coord, _, coverage = coordination_block(sub, include_geometry=True)
    X = sparse.hstack([bundle.X, sparse.csr_matrix(state), coord], format="csr")
    cohorts = (
        sub[COHORT_KEYS].groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    )

    # One protocol, reported once. Unlike the arm A/Bs there is no second
    # grouping to compare: the question is "does this survive unseen papers",
    # so every fold is a DOI fold. Reporting the same numbers under two split
    # keys would look like two protocols and mean one.
    print("[confirm] folds=GroupKFold(5) by doi")
    fold_id = folds_for(sub["doi"].to_numpy())
    global_pred, _, _ = fold_baseline_scores(
        sub["metal"].to_numpy(), sub["cell_line_key"].to_numpy(), y, fold_id
    )
    empty = sparse.csr_matrix((len(sub), 0), dtype=np.float32)
    preds = {
        "global_best": global_pred,
        "regression": fit_arm(
            X, empty, empty, None, 0, False, y, fold_id, cat_codes=codes
        ),
        "centered": _centered_oof(X, y, fold_id, cohorts, codes),
    }
    evaluation = _evaluate(sub, y, preds, cohorts)
    evaluation["fold_assignment"] = "GroupKFold(5) by doi"
    evaluation["n_doi_folds"] = int(len(np.unique(fold_id)))
    splits: Dict[str, object] = {"doi_grouped": evaluation}
    for name, s_ in evaluation["summary"].items():
        print(f"    {name:12s} regret={s_['mean_regret']:.4f} top1={s_['top1_accuracy']:.4f}")

    canonical = splits["doi_grouped"]["summary"]
    candidates = {k: v for k, v in canonical.items() if k != "global_best"}
    best_arm = min(candidates, key=lambda k: candidates[k]["mean_regret"])
    best = canonical[best_arm]
    p_holm = splits["doi_grouped"]["clustered_tests"][best_arm]["p_holm"]
    passed = best["mean_regret"] < BASELINE_FULL_REGRET and p_holm < ALPHA

    payload = {
        "artifact": "msrs_ranking_confirmation",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": bundle.spec_hash,
        },
        "design": {
            "rows": int(len(sub)),
            "n_doi": int(sub["doi"].nunique()),
            "n_cohorts": int(len(np.unique(cohorts))),
            "folds": "GroupKFold(5) grouped by doi — entire papers held out",
            "split_schema": (
                "single doi_grouped protocol; the fragment/canonical split pair "
                "does not apply here because fold assignment is fixed by DOI"
            ),
            "features": "ECFP4(ligand_canonical) | metal state (4) | coordination",
            "ligand_source": "ligand_canonical",
            "ligand_block": "ECFP4(ligand_canonical), 2048 bits",
            "objectives": {"regression": "y", "centered": "y - mean(y | cohort, train)"},
            "coverage": coverage,
            "baseline_full_regret": BASELINE_FULL_REGRET,
            "alpha": ALPHA,
            "preregistration": (
                "same criterion as FIX-E2/E3, written before the run: "
                "mean regret < 0.1198 and DOI-clustered Holm p < 0.05"
            ),
        },
        "splits": splits,
        "best_arm": best_arm,
        "headline": {
            "split": "doi_grouped",
            "arm": best_arm,
            "mean_regret": best["mean_regret"],
            "p_holm_vs_global_best": p_holm,
        },
        "verdict": "POSITIVE" if passed else "NEGATIVE",
        "reading": (
            f"按 DOI 分折（整篇论文留出）后，{best_arm} 的全量 regret "
            f"{best['mean_regret']:.4f}（基线 {BASELINE_FULL_REGRET:.4f}），Holm p={p_holm:.4f}。"
            "这是分布外确认：若增益消失，说明 FIX-E3 的 centered 结果依赖同一批论文。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_single_protocol_report(args.out, payload, "doi_grouped")
    print(f"[confirm] verdict={payload['verdict']} best={best_arm} regret={best['mean_regret']:.4f}")
    print(f"[confirm] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
