#!/usr/bin/env python
"""MSRS-6 — recommendation utility of the metal-substitution surface.

The decision the project actually cares about is: *given a ligand, which metal
should be made next?*  This script turns the MSRS-2 out-of-fold surface into a
ranking over metals and scores that ranking against three references:

* ``global_best``  — always recommend the metal with the lowest mean potency in
  the training folds.  This is the baseline that must be beaten.
* ``cell_best``    — the same, restricted to the cohort's cell line.
* ``oracle``       — the observed ranking (regret 0 by construction), an upper
  bound, not a competitor.

Metrics: top-1 regret in log10 units (lower is better), hit@τ for
τ ∈ {0.1, noise floor, 0.5}, and nDCG@3 with an exponential gain on the y-scale.

Two cohort definitions are reported:

* ``strict`` — ``ligand_canonical`` (same chemical entity).  This is the primary
  view: a metal swap on one and the same ligand.
* ``loose``  — ``ligand_main_fragment`` (largest organic fragment).  This is the
  view MSRS-2/3/4 used for Δ pairing, but a fragment can pool up to 18 distinct
  canonical compounds, so a "cohort" here is not one molecule.  Reported as a
  sensitivity check with the mixing recorded per cohort.

Acceptance (from todo.json): the model must beat ``global_best`` on paired
regret, Holm-corrected across endpoints.  Otherwise the kill criterion fires and
only the global metal ordering may be reported — no ligand-level recommendation.

Usage::

    .venv/bin/python molmetal/scripts/msrs_recommendation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    N_FOLDS,
    NOISE_FLOOR,
    SEED,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    build_design,
    holm,
)

ENDPOINTS = ("potency", "cisplatin_ratio")
COHORT_KEYS = {
    "strict": ["ligand_canonical", "cell_line_key", "doi", "time_h"],
    "loose": ["ligand_main_fragment", "cell_line_key", "doi", "time_h"],
}
MIN_COHORT_SIZE = 2
HIT_THRESHOLDS = (0.1, NOISE_FLOOR, 0.5)
GAIN_SCALE = NOISE_FLOOR  # a metal one noise-floor worse is half as relevant
NDCG_K = 3
SCORERS = ("additive", "interaction", "global_best", "cell_best", "oracle")
MODEL_SCORERS = ("additive", "interaction")


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def build_cohorts(
    frame: pd.DataFrame,
    scorers: Sequence[str],
    min_size: int = MIN_COHORT_SIZE,
) -> pd.DataFrame:
    """Collapse replicate rows to one (cohort, metal) cell and keep real cohorts.

    ``frame`` must carry ``cohort`` (integer), ``metal``, ``y``, ``doi``,
    ``fold`` and one column per scorer.  Replicates of the same (cohort, metal)
    are pooled by median; ``n_rows`` records how many rows were pooled.
    """
    grouped = frame.groupby(["cohort", "metal"], as_index=False, sort=False)
    agg = grouped.agg(
        y=("y", "median"),
        doi=("doi", "first"),
        fold=("fold", "first"),
        n_rows=("y", "size"),
        **{name: (name, "median") for name in scorers},
    )
    size = agg.groupby("cohort")["metal"].nunique()
    keep = size[size >= min_size].index
    agg = agg[agg["cohort"].isin(keep)].reset_index(drop=True)
    # Mixing diagnostic: how many canonical compounds hide behind one cohort?
    if "n_compounds" in frame.columns:
        mix = frame.groupby("cohort")["n_compounds"].max()
        agg["n_compounds"] = agg["cohort"].map(mix).astype(int)
    return agg


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def cohort_metrics(
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    hit_thresholds: Sequence[float] = HIT_THRESHOLDS,
    gain_scale: float = GAIN_SCALE,
    top_k: int = NDCG_K,
) -> Dict[str, object]:
    """Score one cohort.  Lower ``y`` is better (y = log10 IC50, µM)."""
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    k = int(len(y_obs))
    order = np.argsort(y_pred, kind="stable")
    best = float(np.min(y_obs))
    regret = float(y_obs[order[0]] - best)

    out: Dict[str, object] = {
        "k": k,
        "regret": regret,
        "top1_correct": bool(regret <= 1e-9),
        "chosen": int(order[0]),
    }
    for tau in hit_thresholds:
        out[f"hit@{tau:g}"] = bool(regret <= tau)

    m = min(top_k, k)
    gains = 2.0 ** (-(y_obs - best) / gain_scale)
    dcg = float(sum(gains[order[i]] / np.log2(i + 2) for i in range(m)))
    ideal_order = np.argsort(y_obs, kind="stable")
    idcg = float(sum(gains[ideal_order[i]] / np.log2(i + 2) for i in range(m)))
    out["ndcg"] = float(dcg / idcg) if idcg > 0 else 1.0
    return out


def summarise_cohorts(records: pd.DataFrame) -> Dict[str, object]:
    """Aggregate per-cohort metric records for one scorer."""
    if records.empty:
        return {"n_cohorts": 0, "status": "EMPTY"}
    out: Dict[str, object] = {
        "n_cohorts": int(len(records)),
        "mean_regret": round(float(records["regret"].mean()), 4),
        "median_regret": round(float(records["regret"].median()), 4),
        "top1_accuracy": round(float(records["top1_correct"].mean()), 4),
        "mean_ndcg": round(float(records["ndcg"].mean()), 4),
        "by_cohort_size": {},
    }
    for tau in HIT_THRESHOLDS:
        out[f"hit@{tau:g}"] = round(float(records[f"hit@{tau:g}"].mean()), 4)
    for k, g in records.groupby("k"):
        out["by_cohort_size"][str(int(k))] = {
            "n_cohorts": int(len(g)),
            "mean_regret": round(float(g["regret"].mean()), 4),
            "top1_accuracy": round(float(g["top1_correct"].mean()), 4),
            "mean_ndcg": round(float(g["ndcg"].mean()), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Fold-safe baselines
# ---------------------------------------------------------------------------
def fold_assignments(groups: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    """GroupKFold assignment without fitting anything."""
    from sklearn.model_selection import GroupKFold

    fold_id = np.full(len(groups), -1, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros((len(groups), 1)), groups=groups)):
        fold_id[te] = k
    assert (fold_id >= 0).all()
    return fold_id


def fold_baseline_scores(
    metals: np.ndarray,
    cells: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Per-row scores for ``global_best`` and ``cell_best``.

    Both baselines are estimated on the training folds only.  A metal absent
    from the training folds falls back to that fold's overall mean, which makes
    it non-competitive; the count is reported so the reader can see it.
    """
    pred_global = np.full(len(y), np.nan)
    pred_cell = np.full(len(y), np.nan)
    missing_metal = 0
    missing_cell = 0
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = fold_id == k
        overall = float(np.mean(y[tr]))
        per_metal = pd.Series(y[tr]).groupby(metals[tr]).mean()
        per_cell = pd.Series(y[tr]).groupby([cells[tr], metals[tr]]).mean()

        g = np.array([per_metal.get(m, overall) for m in metals[te]], dtype=float)
        missing_metal += int(sum(1 for m in metals[te] if m not in per_metal.index))
        pred_global[te] = g

        c = np.array(
            [per_cell.get((cell, m), per_metal.get(m, overall)) for cell, m in zip(cells[te], metals[te])],
            dtype=float,
        )
        missing_cell += int(
            sum(1 for cell, m in zip(cells[te], metals[te]) if (cell, m) not in per_cell.index)
        )
        pred_cell[te] = c
    assert not np.isnan(pred_global).any() and not np.isnan(pred_cell).any()
    return pred_global, pred_cell, {
        "rows_with_unseen_metal_in_train_fold": missing_metal,
        "rows_with_unseen_cell_metal_pair": missing_cell,
    }


# ---------------------------------------------------------------------------
# Counterfactual predictions (full five-metal ranking)
# ---------------------------------------------------------------------------
def counterfactual_design(
    cohorts: pd.DataFrame, descriptors: pd.DataFrame, cell_vocabulary: Sequence[str]
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    """One design row per (cohort, metal) for all five metals."""
    rows = []
    for cid, g in cohorts.groupby("cohort", sort=True):
        frag = g["ligand_main_fragment"].iloc[0]
        cell = g["cell_line_key"].iloc[0]
        for m in METALS:
            rows.append({"cohort": int(cid), "metal": m, "ligand_main_fragment": frag, "cell_line_key": cell})
    cf = pd.DataFrame(rows)
    X, _, _ = build_design(cf, descriptors, cell_vocabulary)
    return X, cf


def fold_model_predictions(
    X: sparse.csr_matrix,
    y: np.ndarray,
    fold_id: np.ndarray,
    constraints: Optional[List[List[int]]],
    X_cf: Optional[sparse.csr_matrix] = None,
    cf_fold: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Refit one model per fold; return OOF predictions and optional counterfactuals."""
    from lightgbm import LGBMRegressor

    from molmetal.scripts.msrs_surface_model import LGB_PARAMS

    oof = np.full(len(y), np.nan)
    cf = np.full(X_cf.shape[0], np.nan) if X_cf is not None else None
    if X_cf is not None and cf_fold is None:
        raise ValueError("cf_fold is required when X_cf is given")
    params = dict(LGB_PARAMS)
    if constraints is not None:
        params["interaction_constraints"] = constraints
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = fold_id == k
        model = LGBMRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        if X_cf is not None and cf is not None and cf_fold is not None:
            cf[cf_fold == k] = model.predict(X_cf[cf_fold == k])
    assert not np.isnan(oof).any()
    if cf is not None:
        assert not np.isnan(cf).any()
    return oof, cf


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------
def paired_regret_test(
    records: pd.DataFrame, model: str, baseline: str = "global_best"
) -> Dict[str, object]:
    """One-sided paired tests on per-cohort regret (model better than baseline)."""
    wide = records.pivot(index="cohort", columns="scorer", values="regret")
    if model not in wide.columns or baseline not in wide.columns:
        return {"status": "MISSING_SCORER"}
    diff = (wide[baseline] - wide[model]).dropna()
    if len(diff) < 2 or float(diff.std(ddof=1)) == 0.0:
        return {
            "n_cohorts": int(len(diff)),
            "mean_regret_delta": round(float(diff.mean()), 4) if len(diff) else None,
            "t_cohort": 0.0,
            "p_one_sided_cohort": 1.0,
            "direction": "no difference",
        }
    t, p_two = stats.ttest_1samp(diff.to_numpy(), 0.0)
    p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
    return {
        "n_cohorts": int(len(diff)),
        "mean_regret_delta": round(float(diff.mean()), 4),
        "median_regret_delta": round(float(diff.median()), 4),
        "t_cohort": round(float(t), 4),
        "p_one_sided_cohort": p_one,
        "direction": "model better" if diff.mean() > 0 else "baseline better",
    }


def doi_clustered_test(
    records: pd.DataFrame, model: str, baseline: str = "global_best", n_boot: int = 2000, seed: int = SEED
) -> Dict[str, object]:
    """Cluster the paired regret delta by DOI, then test and bootstrap.

    Cohorts inside one paper share ligand chemotypes, cell lines and assay
    protocol; treating them as independent would overstate the evidence.
    """
    wide = records.pivot(index="cohort", columns="scorer", values="regret")
    if model not in wide.columns or baseline not in wide.columns:
        return {"status": "MISSING_SCORER"}
    doi = records.groupby("cohort")["doi"].first()
    diff = (wide[baseline] - wide[model]).dropna()
    diff.index = diff.index.astype(int)
    per_doi = diff.groupby(doi.reindex(diff.index).to_numpy()).mean()
    per_doi = per_doi.dropna()
    if len(per_doi) < 3 or float(per_doi.std(ddof=1)) == 0.0:
        return {
            "n_dois": int(len(per_doi)),
            "mean_regret_delta": round(float(per_doi.mean()), 4) if len(per_doi) else None,
            "t_doi": 0.0,
            "p_one_sided_doi": 1.0,
        }
    t, p_two = stats.ttest_1samp(per_doi.to_numpy(), 0.0)
    p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)

    rng = np.random.default_rng(seed)
    vals = per_doi.to_numpy()
    boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_dois": int(len(per_doi)),
        "mean_regret_delta": round(float(per_doi.mean()), 4),
        "median_regret_delta": round(float(per_doi.median()), 4),
        "t_doi": round(float(t), 4),
        "p_one_sided_doi": p_one,
        "bootstrap_ci95_doi": [round(float(lo), 4), round(float(hi), 4)],
        "n_boot": n_boot,
        "seed": seed,
    }


def disagreement_decomposition(
    records: pd.DataFrame, model: str = "additive", baseline: str = "global_best"
) -> Dict[str, object]:
    """Where does the model earn or lose its keep relative to the baseline?

    The model can only differ from ``global_best`` on cohorts where it picks a
    different metal.  Splitting the cohorts by agreement shows whether the
    deviations are informative or noise.
    """
    wide_reg = records.pivot(index="cohort", columns="scorer", values="regret")
    wide_pick = records.pivot(index="cohort", columns="scorer", values="chosen")
    if model not in wide_reg.columns or baseline not in wide_reg.columns:
        return {"status": "MISSING_SCORER"}
    idx = wide_reg.index
    agree = wide_pick.loc[idx, model] == wide_pick.loc[idx, baseline]
    out: Dict[str, object] = {
        "n_cohorts": int(len(idx)),
        "agreement_rate": round(float(agree.mean()), 4),
        "regret_agree": round(float(wide_reg.loc[agree, model].mean()), 4) if agree.any() else None,
        "n_disagree": int((~agree).sum()),
        "regret_model_when_disagree": (
            round(float(wide_reg.loc[~agree, model].mean()), 4) if (~agree).any() else None
        ),
        "regret_baseline_when_disagree": (
            round(float(wide_reg.loc[~agree, baseline].mean()), 4) if (~agree).any() else None
        ),
        "model_wins_disagreements": (
            int((wide_reg.loc[~agree, model] < wide_reg.loc[~agree, baseline]).sum()) if (~agree).any() else 0
        ),
    }
    if out["regret_model_when_disagree"] is not None:
        out["net_gain_when_disagree"] = round(
            out["regret_baseline_when_disagree"] - out["regret_model_when_disagree"], 4
        )
    return out


# ---------------------------------------------------------------------------
# Full five-metal ranking
# ---------------------------------------------------------------------------
def full_five_report(
    records: pd.DataFrame, cf: pd.DataFrame, cf_pred: np.ndarray, scorer: str = "additive"
) -> Dict[str, object]:
    """How often does a free five-metal ranking pick a metal that was measured?

    Only the observed metals can be scored, so the rate of unverifiable top-1
    picks is itself the headline: a recommendation the data cannot check is not
    evidence.
    """
    cf = cf.copy()
    cf["pred"] = cf_pred
    observed = {
        int(cid): dict(zip(g["metal"], g["y"])) for cid, g in records.groupby("cohort", sort=False)
    }
    picks, verifiable, regret = [], [], []
    for cid, g in cf.groupby("cohort"):
        g = g.sort_values("pred", kind="stable")
        top = g.iloc[0]
        picks.append(str(top["metal"]))
        obs = observed.get(cid, {})
        if top["metal"] in obs:
            verifiable.append(True)
            regret.append(float(obs[top["metal"]] - min(obs.values())))
        else:
            verifiable.append(False)
    out: Dict[str, object] = {
        "n_cohorts": int(len(verifiable)),
        "top1_is_observed_rate": round(float(np.mean(verifiable)), 4) if verifiable else None,
        "n_verifiable": int(np.sum(verifiable)),
        "mean_regret_verifiable": round(float(np.mean(regret)), 4) if regret else None,
        "top1_metal_distribution": {
            m: int(picks.count(m)) for m in METALS if picks.count(m) > 0
        },
        "caveat": (
            "Cohorts only ever contain 2 metals, so three of the five ranks are "
            "unobserved for every cohort. Regret is computed only where the pick "
            "was measured, and that subset is selected by the model itself."
        ),
    }
    return out


# ---------------------------------------------------------------------------
# One endpoint × one cohort definition
# ---------------------------------------------------------------------------
def evaluate(
    df: pd.DataFrame,
    endpoint: str,
    descriptors: pd.DataFrame,
    cohort_name: str,
    verbose: bool = True,
) -> Dict[str, object]:
    t0 = time.time()
    keys = COHORT_KEYS[cohort_name]
    sub = df[df["endpoint"] == endpoint].copy()
    n_total = len(sub)
    n_censored = int(sub["value_log10"].isna().sum())
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna() & sub[keys[0]].notna()]
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return {"status": "EMPTY"}

    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, _, groups = build_design(sub, descriptors, cell_vocab)
    y = sub["value_log10"].to_numpy(dtype=float)
    ligand_groups = sub["ligand_main_fragment"].to_numpy()
    metals = sub["metal"].to_numpy()
    cells = sub["cell_line_key"].to_numpy()

    sub["cohort"] = sub.groupby(keys, dropna=False, sort=False).ngroup().to_numpy()
    if cohort_name == "loose":
        sub["n_compounds"] = sub.groupby("cohort")["ligand_canonical"].transform("nunique")
    else:
        sub["n_compounds"] = 1

    # Folds are grouped by the loose fragment; a strict cohort shares one
    # fragment, so it still never straddles two folds.
    fold_id = fold_assignments(ligand_groups)
    sub["fold_id"] = fold_id

    pred_global, pred_cell, baseline_notes = fold_baseline_scores(metals, cells, y, fold_id)

    # Cohorts with at least two metals — the only ones where a choice exists.
    counts = sub.groupby("cohort")["metal"].nunique()
    usable = counts[counts >= MIN_COHORT_SIZE].index
    cohort_frame = sub[sub["cohort"].isin(usable)].reset_index(drop=True)
    if cohort_frame.empty:
        return {"status": "NO_COHORTS"}

    X_cf, cf_frame = counterfactual_design(cohort_frame, descriptors, cell_vocab)
    cohort_fold = sub.groupby("cohort")["fold_id"].first()
    cf_frame["fold"] = cf_frame["cohort"].map(cohort_fold).astype(int)
    cf_fold = cf_frame["fold"].to_numpy()

    # Re-run the OOF loop with the counterfactual matrix attached.  Row order of
    # ``X`` matches ``sub`` because nothing was reordered above.
    pred_add, cf_add = fold_model_predictions(X, y, fold_id, groups, X_cf, cf_fold)
    pred_int, _ = fold_model_predictions(X, y, fold_id, None)

    frame = pd.DataFrame(
        {
            "cohort": sub["cohort"].to_numpy(),
            "metal": metals,
            "y": y,
            "doi": sub["doi"].to_numpy(),
            "fold": fold_id,
            "ligand_main_fragment": sub["ligand_main_fragment"].to_numpy(),
            "cell_line_key": cells,
            "n_compounds": sub["n_compounds"].to_numpy(),
            "additive": pred_add,
            "interaction": pred_int,
            "global_best": pred_global,
            "cell_best": pred_cell,
        }
    )
    frame = frame[frame["cohort"].isin(usable)].reset_index(drop=True)
    cohorts = build_cohorts(frame, ("additive", "interaction", "global_best", "cell_best"))

    records = []
    for cid, g in cohorts.groupby("cohort", sort=True):
        y_obs = g["y"].to_numpy()
        for scorer in ("additive", "interaction", "global_best", "cell_best"):
            m = cohort_metrics(y_obs, g[scorer].to_numpy())
            records.append(
                {
                    "cohort": int(cid),
                    "scorer": scorer,
                    "doi": g["doi"].iloc[0],
                    "k": m["k"],
                    "regret": m["regret"],
                    "top1_correct": m["top1_correct"],
                    "ndcg": m["ndcg"],
                    "chosen": m["chosen"],
                    **{f"hit@{t:g}": m[f"hit@{t:g}"] for t in HIT_THRESHOLDS},
                }
            )
        # Oracle is the observed ordering, included for scale.
        m = cohort_metrics(y_obs, y_obs)
        records.append(
            {
                "cohort": int(cid),
                "scorer": "oracle",
                "doi": g["doi"].iloc[0],
                "k": m["k"],
                "regret": m["regret"],
                "top1_correct": m["top1_correct"],
                "ndcg": m["ndcg"],
                "chosen": m["chosen"],
                **{f"hit@{t:g}": m[f"hit@{t:g}"] for t in HIT_THRESHOLDS},
            }
        )
    records_df = pd.DataFrame(records)

    summary = {
        scorer: summarise_cohorts(records_df[records_df["scorer"] == scorer])
        for scorer in SCORERS
    }
    paired = {
        scorer: paired_regret_test(records_df, scorer) for scorer in MODEL_SCORERS
    }
    clustered = {
        scorer: doi_clustered_test(records_df, scorer) for scorer in MODEL_SCORERS
    }
    disagreement = {
        scorer: disagreement_decomposition(records_df, scorer) for scorer in MODEL_SCORERS
    }
    full5 = full_five_report(cohorts, cf_frame, cf_add, "additive")
    invariance = additive_ranking_invariance(cf_frame, cf_add)
    order_data = global_metal_order(sub, y, fold_id)

    # Reference points for a two-metal cohort: a coin flip is wrong half the
    # time, so its expected regret is half the mean |Δ|.
    two = cohorts.groupby("cohort").filter(lambda g: len(g) == 2)
    abs_delta = []
    for _, g in two.groupby("cohort"):
        vals = g["y"].to_numpy()
        if len(vals) == 2:
            abs_delta.append(abs(vals[0] - vals[1]))
    random_regret = round(float(np.mean(abs_delta) / 2.0), 4) if abs_delta else None

    if verbose:
        print(
            f"[{endpoint}/{cohort_name}] cohorts={summary['additive']['n_cohorts']} "
            f"rows={len(sub)} (dropped {n_censored} censored of {n_total})"
        )
        for scorer in SCORERS:
            s = summary[scorer]
            if s.get("n_cohorts"):
                print(
                    f"  {scorer:12s} regret={s['mean_regret']:.4f} "
                    f"top1={s['top1_accuracy']:.4f} ndcg={s['mean_ndcg']:.4f}"
                )
        print(
            f"  additive vs global_best: delta={paired['additive']['mean_regret_delta']} "
            f"p_cohort={paired['additive']['p_one_sided_cohort']:.3g} "
            f"delta_doi={clustered['additive'].get('mean_regret_delta')} "
            f"p_doi={clustered['additive'].get('p_one_sided_doi', float('nan')):.3g}"
        )

    return {
        "status": "OK",
        "n_rows": int(len(sub)),
        "n_dropped_censored": n_censored,
        "censored_share": round(n_censored / n_total, 4) if n_total else None,
        "n_cohorts": int(summary["additive"]["n_cohorts"]),
        "cohort_definition": list(keys),
        "cohort_size_distribution": {
            str(int(k)): int(v) for k, v in cohorts.groupby("cohort")["metal"].nunique().value_counts().sort_index().items()
        },
        "n_dois": int(cohorts["doi"].nunique()),
        "mixing": (
            {
                "max_compounds_per_cohort": int(cohorts["n_compounds"].max()),
                "mean_compounds_per_cohort": round(float(cohorts["n_compounds"].mean()), 2),
                "share_cohorts_with_multiple_compounds": round(
                    float((cohorts["n_compounds"] > 1).mean()), 4
                ),
            }
            if cohort_name == "loose"
            else {"max_compounds_per_cohort": 1, "note": "strict cohorts are one canonical ligand"}
        ),
        "baseline_notes": baseline_notes,
        "random_choice_expected_regret": random_regret,
        "scores": summary,
        "paired_vs_global_best": paired,
        "doi_clustered_vs_global_best": clustered,
        "disagreement_decomposition": disagreement,
        "full_five_metal": full5,
        "additive_ranking_invariance": invariance,
        "global_metal_order_from_data": order_data,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def additive_ranking_invariance(cf: pd.DataFrame, cf_pred: np.ndarray) -> Dict[str, object]:
    """Does the additive surface imply one metal order for every ligand?

    The interaction constraints forbid ligand x metal and metal x cell splits on
    the same root-to-leaf path, so the prediction is f(ligand) + g(metal) +
    h(cell).  Within a cohort (same ligand, same cell) the metal term therefore
    decides the ranking, and that term is a constant.  If the check below holds,
    the model is a *global* metal ordering in disguise and can only beat
    ``global_best`` by estimating that one ordering better.
    """
    frame = cf.assign(pred=cf_pred)
    out: Dict[str, object] = {"by_fold": {}}
    all_constant = True
    for k, g in frame.groupby("fold"):
        wide = g.pivot(index="cohort", columns="metal", values="pred")
        orders = wide.apply(lambda row: tuple(row.sort_values(kind="stable").index), axis=1)
        counts = orders.value_counts()
        constant = bool(len(counts) == 1)
        all_constant = all_constant and constant
        out["by_fold"][str(int(k))] = {
            "n_cohorts": int(len(orders)),
            "n_distinct_orders": int(len(counts)),
            "modal_order": list(counts.index[0]),
            "modal_share": round(float(counts.iloc[0] / len(orders)), 4),
        }
    out["constant_within_fold"] = all_constant
    out["note"] = (
        "additive prediction = f(ligand) + g(metal) + h(cell); a constant order means "
        "the within-cohort ranking carries no ligand information at all"
    )
    return out


def global_metal_order(df: pd.DataFrame, y: np.ndarray, fold_id: np.ndarray) -> Dict[str, object]:
    """Fold-averaged mean y per metal, ascending — the ordering the kill criterion leaves standing."""
    per_fold = []
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        per_fold.append(pd.Series(y[tr]).groupby(df["metal"].to_numpy()[tr]).mean())
    mean_by_metal = pd.concat(per_fold, axis=1).mean(axis=1)
    order = mean_by_metal.sort_values(kind="stable")
    return {
        "mean_y_by_metal": {m: round(float(v), 4) for m, v in mean_by_metal.items()},
        "ascending": list(order.index),
        "note": "lower log10 IC50 = more potent; this is the only recommendation the data supports",
    }


def acceptance_decision(clustered_entry: Dict[str, object]) -> Tuple[bool, str]:
    """Kill criterion from todo.json: beat ``global_best`` after Holm correction."""
    delta = clustered_entry.get("mean_regret_delta")
    p = clustered_entry.get("holm_p_across_primary_tests")
    p = 1.0 if p is None else float(p)
    met = bool(delta is not None and delta > 0 and p < 0.05)
    reason = (
        f"mean regret delta {delta} log10, Holm p {p:.3g}; "
        + ("model beats global_best" if met else "model does not beat global_best")
    )
    return met, reason


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_recommendation.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results: Dict[str, Dict[str, object]] = {"strict": {}, "loose": {}}
    for cohort_name in ("strict", "loose"):
        for endpoint in ENDPOINTS:
            results[cohort_name][endpoint] = evaluate(df, endpoint, descriptors, cohort_name)

    # Holm across the four primary tests (2 endpoints x 2 cohort definitions),
    # on the DOI-clustered p-value, which is the conservative one.
    raw = {}
    for cohort_name, eps in results.items():
        for ep, r in eps.items():
            if r.get("status") != "OK":
                continue
            raw[f"{cohort_name}/{ep}"] = r["doi_clustered_vs_global_best"]["additive"]["p_one_sided_doi"]
    adjusted = holm(raw)
    for key, p in adjusted.items():
        cohort_name, ep = key.split("/")
        entry = results[cohort_name][ep]["doi_clustered_vs_global_best"]["additive"]
        entry["holm_p_across_primary_tests"] = p

    primary = results["strict"].get("potency", {})
    acceptance = False
    reason = "no strict potency result"
    if primary.get("status") == "OK":
        acceptance, reason = acceptance_decision(
            primary["doi_clustered_vs_global_best"]["additive"]
        )

    by_test: Dict[str, object] = {}
    n_met = 0
    for cohort_name, eps in results.items():
        for ep, r in eps.items():
            if r.get("status") != "OK":
                continue
            entry = r["doi_clustered_vs_global_best"]["additive"]
            met, why = acceptance_decision(entry)
            n_met += int(met)
            by_test[f"{cohort_name}/{ep}"] = {
                "met": met,
                "mean_regret_delta": entry.get("mean_regret_delta"),
                "holm_p": entry.get("holm_p_across_primary_tests"),
                "reason": why,
            }

    out = {
        "artifact": "msrs_recommendation",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 decision layer",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "question": "given a ligand and its assay context, which of the five metals is predicted to be most potent?",
            "orientation": "y = log10 IC50 (µM); lower is better; regret = y(chosen) - y(best observed)",
            "cohort": "same ligand + cell line + DOI + time point, >= 2 distinct metals measured",
            "cohort_definitions": {
                "strict": "ligand_canonical — one chemical entity; PRIMARY",
                "loose": "ligand_main_fragment — MSRS-2/3/4 pairing key; pools distinct compounds",
            },
            "scorers": list(SCORERS),
            "baselines_fold_safe": "metal / cell-line means recomputed on each training fold",
            "hit_thresholds": list(HIT_THRESHOLDS),
            "ndcg": {
                "k": NDCG_K,
                "gain": f"2 ** (-(y - y_best) / {GAIN_SCALE})",
                "note": "every cohort has k=2 under the strict key, so nDCG is degenerate there",
            },
            "paired_test": "one-sided t on per-cohort regret delta; primary inference clusters by DOI",
            "seed": SEED,
            "n_folds": N_FOLDS,
        },
        "acceptance": {
            "criterion": "model beats global_best on paired regret, Holm-corrected across endpoints",
            "met": acceptance,
            "reason": reason,
            "n_primary_tests_met": n_met,
            "by_test": by_test,
            "consequence_if_not_met": (
                "kill criterion: report only the global metal ordering; no ligand-level recommendation"
            ),
        },
        "endpoints": results,
        "caveats": [
            "Under the strict key every cohort contains exactly two metals, so the five-metal "
            "ranking is only ever observed in two of its five positions.",
            "The additive arm is separable: prediction = f(ligand) + g(metal) + h(cell), so its "
            "within-cohort metal order is one global order per fold (see "
            "additive_ranking_invariance). It can only beat global_best by estimating that single "
            "order better than the raw fold mean does.",
            "The loose key pools up to 18 distinct canonical compounds into one cohort; its deltas "
            "are confounded by compound identity and are a sensitivity check, not the primary result.",
            "Baselines are fold-safe but still estimated from the same corpus that supplies the "
            "evaluation cohorts; a genuinely prospective test would need held-out papers.",
            "The model's counterfactual five-metal picks are scored only where the pick happens to "
            "have been measured, a subset chosen by the model itself.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    print(f"acceptance met: {acceptance} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
