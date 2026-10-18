#!/usr/bin/env python3
"""FIX-M1b — the estimand read on the D-MPNN's OOF predictions.

Why a separate script
=====================
``msrs_dmpnn_surface.py`` answers the acceptance question (absolute R² on the
same folds).  It does **not** answer the project's estimand question: does the
model rank metals correctly *inside a cohort*?  FIX-E2/E3/E4 showed the two are
different failure modes — a model can improve R² and still lose the ranking.

Re-running the GPU job just to compute this would cost another hour, so the
surface script writes row-level OOF to
``results/msrs_dmpnn_surface/oof.npz`` and this script reads it back.  The
cohort construction, baselines and clustered test are imported from the same
modules FIX-E2/E3 used, so the numbers are directly comparable.

Reported per split:
    full_cohort_regret   mean regret of each scorer (lower is better)
    top1_accuracy        fraction of cohorts whose best metal is chosen
    clustered_tests      paired regret delta vs ``global_best`` and vs
                         LightGBM, clustered by DOI, Holm over the model arms

Usage::

    .venv/bin/python molmetal/scripts/msrs_dmpnn_decision.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from molmetal.scripts.msrs_code_levers import COHORT_KEYS, folds_for
from molmetal.scripts.msrs_decision_policy import (
    MIN_COHORT_SIZE,
    cohort_table,
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
from molmetal.scripts.msrs_surface_model import _git_commit, holm

#: Mean regret the FIX-E line established as the bar (NEG-16 / FIX-E4).
BASELINE_FULL_REGRET = 0.1198


def load_oof(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def scorer_names(arrays: Dict[str, np.ndarray], split: str) -> List[str]:
    prefix = f"{split}__"
    return sorted(k[len(prefix) :] for k in arrays if k.startswith(prefix))


def frame_for(arrays: Dict[str, np.ndarray], split: str, scorers: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ligand_canonical": arrays["ligand_canonical"],
            "ligand_main_fragment": arrays["ligand_main_fragment"],
            "cell_line_key": arrays["cell_line_key"],
            "doi": arrays["doi"],
            "time_h": arrays["time_h"],
            "metal": arrays["metal"],
            "y": arrays["y"],
        }
    )
    groups = (
        frame["ligand_main_fragment"].to_numpy()
        if split == SPLIT_FRAGMENT_GROUPED
        else frame["ligand_canonical"].to_numpy()
    )
    frame["fold"] = folds_for(groups)
    for name in scorers:
        frame[name] = arrays[f"{split}__{name}"]
    return frame


def _records(cohorts: pd.DataFrame, scorers: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        for s in scorers:
            rec = cohort_metrics(g["y"].to_numpy(), g[s].to_numpy())
            rows.append({"cohort": int(cid), "scorer": s, "doi": g["doi"].iloc[0], **rec})
    return pd.DataFrame(rows)


def run_split(arrays: Dict[str, np.ndarray], split: str) -> Dict[str, object]:
    models = scorer_names(arrays, split)
    frame = frame_for(arrays, split, models)
    global_pred, cell_pred, baseline_meta = fold_baseline_scores(
        frame["metal"].to_numpy(),
        frame["cell_line_key"].to_numpy(),
        frame["y"].to_numpy(),
        frame["fold"].to_numpy(),
    )
    frame["global_best"] = global_pred
    frame["cell_best"] = cell_pred
    frame["cohort"] = (
        frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    )
    scorers = ["global_best", "cell_best"] + models

    cohorts = cohort_table(frame, scorers)
    regret = regret_matrix(cohorts, scorers)
    full = {s: round(float(regret[s].mean()), 4) for s in scorers}
    rec = _records(cohorts, scorers)

    lgb = next((m for m in models if m.startswith("lgb_")), None)
    dmpnn = [m for m in models if not m.startswith("lgb_")]
    vs_global = {m: doi_clustered_test(rec, m, "global_best") for m in dmpnn}
    p_global = {m: float(v.get("p_one_sided_doi", 1.0)) for m, v in vs_global.items()}
    adj_global = holm(p_global)
    for m in vs_global:
        vs_global[m]["p_holm"] = round(float(adj_global[m]), 6)

    vs_lgb = {}
    if lgb:
        vs_lgb = {m: doi_clustered_test(rec, m, lgb) for m in dmpnn}
        p_lgb = {m: float(v.get("p_one_sided_doi", 1.0)) for m, v in vs_lgb.items()}
        adj_lgb = holm(p_lgb)
        for m in vs_lgb:
            vs_lgb[m]["p_holm"] = round(float(adj_lgb[m]), 6)

    return {
        "n_rows": int(len(frame)),
        "n_cohorts": int(verifiability_report(cohorts)["n_cohorts"]),
        "min_cohort_size": int(MIN_COHORT_SIZE),
        "verifiability": verifiability_report(cohorts),
        "baseline_meta": baseline_meta,
        "full_cohort_regret": full,
        "top1_accuracy": {
            s: round(float(rec.loc[rec["scorer"] == s, "top1_correct"].mean()), 4)
            for s in scorers
        },
        "cohort_summary": {s: summarise_cohorts(rec[rec["scorer"] == s]) for s in scorers},
        "clustered_vs_global_best": vs_global,
        "clustered_vs_lightgbm": vs_lgb,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_dmpnn_surface" / "oof.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_dmpnn_decision.json",
    )
    args = parser.parse_args(argv)

    arrays = load_oof(args.oof)
    splits = {}
    for split in (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED):
        if not scorer_names(arrays, split):
            raise SystemExit(f"{args.oof}: no OOF for split {split!r}")
        print(f"[m1b] split={split}")
        splits[split] = run_split(arrays, split)
        print(f"    regret={splits[split]['full_cohort_regret']}")

    payload = {
        "artifact": "msrs_dmpnn_decision",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof.relative_to(PROJECT_ROOT))
            if args.oof.resolve().is_relative_to(PROJECT_ROOT)
            else str(args.oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "cohort_keys": COHORT_KEYS,
            "ligand_source": "ligand_canonical",
            "min_cohort_size": int(MIN_COHORT_SIZE),
            "baseline_reference_regret": BASELINE_FULL_REGRET,
            "note": (
                "Regret = y(chosen) - min y in cohort, lower is better. "
                "global_best/cell_best are fold-safe (training folds only). "
                "The DOI-clustered t-test is the decision statistic; the "
                "bootstrap CI is descriptive (5-seed bootstrap is known "
                "anti-conservative, see NEG ledger)."
            ),
        },
        "splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[m1b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
