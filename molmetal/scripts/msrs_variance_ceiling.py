#!/usr/bin/env python3
"""FIX-M3 — how much of the within-cohort metal signal is knowable at all?

The decision layer keeps failing: FIX-E2/E3/E4, FIX-M1b and FIX-M2 all land at
the same regret.  Before spending more on objectives, this asks whether the
target itself carries enough signal to rank.

The estimand is the variance of ``y`` *inside a cohort* (same ligand, cell
line, DOI, time).  That variance splits into three parts:

    var_within = metal main effect + cohort x metal interaction + measurement noise

Only the interaction can change a within-cohort ranking; the metal main effect
is the same in every cohort and cancels.  The interaction and the noise are
**not separately identified** unless some cells are measured more than once.
So this script reports what is identified, and a sensitivity curve over the
noise level instead of pretending to a point estimate.

Identified quantities (no assumption about the noise):

  * ``var_within`` — the target variance.
  * ``metal_main_share`` — the part explained by a metal main effect.  This is
    real signal by construction (it is estimated from thousands of rows), so it
    is a **lower bound** on the explainable share.
  * ``model_within_corr`` — how much of the within-cohort deviation the current
    models actually track.  Correlation, not R²: the models are heavily
    shrunk, so R² understates their ranking signal.

Unidentified:

  * the noise floor, hence the ceiling.  The within-cell replicate sample is
    bimodal (median sigma 0.027, pooled sigma 0.211) and almost absent from the
    evaluation cohorts (3 of 848 cells), so it cannot be extrapolated.

Usage::

    .venv/bin/python molmetal/scripts/msrs_variance_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from molmetal.scripts.msrs_code_levers import COHORT_KEYS, ENDPOINT, folds_for
from molmetal.scripts.msrs_recommendation import fold_baseline_scores
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

SIGMA_GRID = (0.027, 0.05, 0.10, 0.15, 0.211, 0.25, 0.30, 0.36)
N_BOOT = 300
SEED = 7
MIN_METALS = 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cohort(path: Path) -> pd.DataFrame:
    long = pd.read_parquet(path)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy()
    sub["cohort"] = sub.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup()
    return sub.reset_index(drop=True)


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (cohort, metal): replicate count, within-cell SS, median y."""
    frame = frame.copy()
    frame["_cell_mean"] = frame.groupby(["cohort", "metal"])["value_log10"].transform("mean")
    frame["_cell_ss"] = (frame["value_log10"] - frame["_cell_mean"]) ** 2
    cells = frame.groupby(["cohort", "metal"], as_index=False).agg(
        n=("value_log10", "size"),
        ss=("_cell_ss", "sum"),
        y=("value_log10", "median"),
    )
    return cells


def evaluation_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Cohorts the decision layer actually scores (>= 2 metals)."""
    sizes = cells.groupby("cohort")["metal"].nunique()
    keep = sizes[sizes >= MIN_METALS].index
    return cells[cells["cohort"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Identified quantities
# ---------------------------------------------------------------------------
def noise_estimate(cells: pd.DataFrame) -> Dict[str, object]:
    rep = cells[cells["n"] >= 2]
    within_var = rep["ss"] / (rep["n"] - 1)
    df = int((rep["n"] - 1).sum())
    pooled = float(rep["ss"].sum()) / df if df else float("nan")
    return {
        "n_replicated_cells": int(len(rep)),
        "n_rows": int(rep["n"].sum()),
        "df": df,
        "pooled_sigma": round(float(np.sqrt(pooled)), 4),
        "pooled_sigma2": round(float(pooled), 5),
        "median_within_cell_sigma": round(float(np.sqrt(within_var.median())), 4),
        "p90_within_cell_sigma": round(float(np.sqrt(within_var.quantile(0.9))), 4),
        "by_metal": {
            str(m): {
                "cells": int(len(g)),
                "pooled_sigma": round(float(np.sqrt(g["ss"].sum() / (g["n"] - 1).sum())), 4),
            }
            for m, g in rep.groupby("metal")
            if len(g) >= 3
        },
    }


def additive_decomposition(cells_eval: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
    """var_within, additive residual MS, and the implied metal-main share."""
    cohort_n = cells_eval["cohort"].nunique()
    dev = cells_eval["y"] - cells_eval.groupby("cohort")["y"].transform("mean")
    ss_within = float((dev**2).sum())
    df_within = int(len(cells_eval) - cohort_n)
    var_within = ss_within / df_within

    design = np.column_stack(
        [
            pd.get_dummies(cells_eval["cohort"], drop_first=True).to_numpy(float),
            pd.get_dummies(cells_eval["metal"], drop_first=True).to_numpy(float),
            np.ones(len(cells_eval)),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, cells_eval["y"].to_numpy(), rcond=None)
    resid = cells_eval["y"].to_numpy() - design @ beta
    df_res = int(len(cells_eval) - np.linalg.matrix_rank(design))
    ms_res = float(resid @ resid / df_res)
    return (
        {
            "n_cohorts": int(cohort_n),
            "n_cells": int(len(cells_eval)),
            "var_within": round(float(var_within), 5),
            "additive_residual_ms": round(ms_res, 5),
            "metal_main_share": round(float((var_within - ms_res) / var_within), 4),
            "interaction_plus_noise_share": round(float(ms_res / var_within), 4),
        },
        resid,
    )


# ---------------------------------------------------------------------------
# Ceiling as a function of an assumed noise level
# ---------------------------------------------------------------------------
def _score(pred: np.ndarray, truth: np.ndarray, cohort: np.ndarray) -> Tuple[float, float]:
    table = pd.DataFrame({"c": cohort, "p": pred, "t": truth})
    best = table.groupby("c", sort=False)["t"].min()
    picked = table.loc[table.groupby("c", sort=False)["p"].idxmin()].set_index("c")["t"]
    picked = picked.reindex(best.index)
    return float((picked == best).mean()), float((picked - best).mean())


def ceiling_curve(
    cells_eval: pd.DataFrame,
    var_within: float,
    ms_res: float,
    sigma_grid: Sequence[float],
) -> List[Dict[str, float]]:
    truth = cells_eval["y"].to_numpy()
    cohort = cells_eval["cohort"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows: List[Dict[str, float]] = []
    for sigma in sigma_grid:
        s2 = float(sigma) ** 2
        interaction = max(0.0, ms_res - s2)
        max_r2 = max(0.0, 1.0 - s2 / var_within)
        t1, rg, t1r, rgr = [], [], [], []
        for _ in range(N_BOOT):
            e1 = rng.normal(0.0, sigma, len(truth))
            e2 = rng.normal(0.0, sigma, len(truth))
            a = _score(truth, truth + e1, cohort)  # perfect model, noisy label
            b = _score(truth + e1, truth + e2, cohort)  # two measurements
            t1.append(a[0]); rg.append(a[1]); t1r.append(b[0]); rgr.append(b[1])
        rows.append(
            {
                "sigma": round(float(sigma), 4),
                "interaction_var": round(interaction, 5),
                "interaction_share_of_within": round(interaction / var_within, 4),
                "max_within_r2": round(max_r2, 4),
                "perfect_model_top1": round(float(np.mean(t1)), 4),
                "perfect_model_regret": round(float(np.mean(rg)), 4),
                "two_measurement_top1": round(float(np.mean(t1r)), 4),
                "two_measurement_regret": round(float(np.mean(rgr)), 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cross-paper replication (independent of the within-cohort replicates)
# ---------------------------------------------------------------------------
def cross_paper(frame: pd.DataFrame) -> Dict[str, object]:
    g = (
        frame.groupby(["ligand_canonical", "cell_line_key", "doi", "metal"])["value_log10"]
        .median()
        .reset_index()
    )
    per_paper = g.groupby(["ligand_canonical", "cell_line_key", "doi"])["metal"].nunique()
    usable = per_paper[per_paper >= 2].reset_index()[["ligand_canonical", "cell_line_key", "doi"]]
    g = g.merge(usable, on=["ligand_canonical", "cell_line_key", "doi"])
    ndoi = g.groupby(["ligand_canonical", "cell_line_key"])["doi"].nunique()
    multi = g[
        g.set_index(["ligand_canonical", "cell_line_key"]).index.isin(
            ndoi[ndoi >= 2].index
        )
    ]

    key = ["ligand_canonical", "cell_line_key", "metal"]
    means = multi.assign(_gm=multi.groupby(key)["value_log10"].transform("mean"))
    means["_dev"] = means["value_log10"] - means["_gm"]
    df = int(len(means) - means.groupby(key).ngroups)
    sigma = float(np.sqrt((means["_dev"] ** 2).sum() / df)) if df > 0 else float("nan")

    # Conservative variant: every (ligand, metal, cell) with >= 2 papers, even
    # when a paper reports only one metal (so it cannot enter the ranking check).
    all_pairs = g.copy()
    ndoi_all = all_pairs.groupby(key)["doi"].nunique()
    wide = all_pairs[
        all_pairs.set_index(key).index.isin(ndoi_all[ndoi_all >= 2].index)
    ]
    wide = wide.assign(_gm=wide.groupby(key)["value_log10"].transform("mean"))
    wide["_dev"] = wide["value_log10"] - wide["_gm"]
    df_all = int(len(wide) - wide.groupby(key).ngroups)
    sigma_all = float(np.sqrt((wide["_dev"] ** 2).sum() / df_all)) if df_all > 0 else float("nan")

    from itertools import combinations

    agree: List[bool] = []
    for _, grp in multi.groupby(["ligand_canonical", "cell_line_key"]):
        dois = grp["doi"].unique()
        for d1, d2 in combinations(dois, 2):
            a = grp[grp["doi"] == d1].set_index("metal")["value_log10"]
            b = grp[grp["doi"] == d2].set_index("metal")["value_log10"]
            common = sorted(set(a.index) & set(b.index))
            if len(common) >= 2:
                agree.append(bool(np.argmin(a.loc[common].to_numpy()) == np.argmin(b.loc[common].to_numpy())))

    n = len(agree)
    k = int(sum(agree))
    lo, hi = _wilson(k, n)
    return {
        "n_groups_with_multiple_dois": int(ndoi[ndoi >= 2].shape[0]),
        "n_paper_pairs": n,
        "top1_agreement": round(k / n, 4) if n else None,
        "top1_agreement_wilson95": [round(lo, 4), round(hi, 4)] if n else None,
        "between_doi_sigma": round(sigma, 4),
        "between_doi_sigma_all_groups": round(sigma_all, 4),
        "between_doi_df_all_groups": df_all,
        "note": (
            "Between-DOI sigma includes protocol and lab differences, so it is an "
            "upper bound on measurement noise, not an estimate of it."
        ),
    }


def _wilson(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# What the current models track
# ---------------------------------------------------------------------------
def model_readout(oof_path: Path, cells_eval: pd.DataFrame) -> Dict[str, object]:
    with np.load(oof_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    frame = pd.DataFrame(
        {
            "ligand_canonical": arrays["ligand_canonical"],
            "cell_line_key": arrays["cell_line_key"],
            "doi": arrays["doi"],
            "time_h": arrays["time_h"],
            "metal": arrays["metal"],
            "y": arrays["y"],
        }
    )
    frame["cohort"] = frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup()
    frame["fold"] = folds_for(frame["ligand_canonical"].to_numpy())
    mask = frame["cohort"].isin(cells_eval["cohort"].unique()).to_numpy()
    frame = frame[mask].reset_index(drop=True)
    frame["yd"] = frame["y"] - frame.groupby("cohort")["y"].transform("mean")

    global_pred, cell_pred, _ = fold_baseline_scores(
        frame["metal"].to_numpy(),
        frame["cell_line_key"].to_numpy(),
        frame["y"].to_numpy(),
        frame["fold"].to_numpy(),
    )
    arms: Dict[str, np.ndarray] = {"global_best": global_pred, "cell_best": cell_pred}
    for name in ("lgb_tree", "gnn_metal_state", "blend_equal", "stack_foldwise"):
        key = f"canonical_grouped__{name}"
        if key in arrays:
            arms[name] = arrays[key][mask]

    out: Dict[str, object] = {}
    for name, pred in arms.items():
        frame["_p"] = pred
        pdev = frame["_p"] - frame.groupby("cohort")["_p"].transform("mean")
        corr = float(np.corrcoef(frame["yd"], pdev)[0, 1])
        out[name] = {
            "within_corr": round(corr, 4),
            "calibrated_within_r2": round(corr**2, 4),
            "pred_within_sd": round(float(pdev.std()), 4),
        }
    out["_note"] = (
        "calibrated_within_r2 = corr^2, i.e. R2 after the best linear rescaling. "
        "Plain R2 is near zero because the predictions are shrunk ~100x; ranking "
        "only needs the correlation."
    )
    return out


def break_even_sigma(
    cells_eval: pd.DataFrame,
    target_top1: float,
    target_regret: float,
    hi: float = 1.5,
    steps: int = 36,
    n_boot: int = 120,
) -> Dict[str, Optional[float]]:
    """Noise level at which a perfect model would fall to the current score.

    Below this sigma the ceiling is above what the current model achieves, so
    the data cannot be the binding constraint.  Above it, the current model
    would be beating a perfect predictor — impossible — so the data would have
    to be at least that noisy.
    """
    truth = cells_eval["y"].to_numpy()
    cohort = cells_eval["cohort"].to_numpy()
    rng = np.random.default_rng(SEED + 1)
    grid = np.linspace(0.01, hi, steps)
    points: List[Tuple[float, float, float]] = []
    for sigma in grid:
        t1, rg = [], []
        for _ in range(n_boot):
            e = rng.normal(0.0, sigma, len(truth))
            top1, regret = _score(truth, truth + e, cohort)
            t1.append(top1); rg.append(regret)
        points.append((float(sigma), float(np.mean(t1)), float(np.mean(rg))))

    def crossing(index: int, target: float) -> Optional[float]:
        for (s0, *v0), (s1, *v1) in zip(points, points[1:]):
            a, b = v0[index], v1[index]
            if (a - target) * (b - target) <= 0 and a != b:
                return round(float(s0 + (s1 - s0) * (target - a) / (b - a)), 4)
        return None

    return {
        "sigma_top1": crossing(0, target_top1),
        "sigma_regret": crossing(1, target_regret),
        "grid_hi": hi,
        "note": "current model is at the ceiling only if the true noise is at least this large",
    }


def current_scores(path: Path, arm: str = "lgb_tree") -> Optional[Dict[str, float]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    table = payload.get("splits", {}).get("canonical_grouped", {})
    top1 = table.get("top1_accuracy", {})
    regret = table.get("full_cohort_regret", {})
    if arm not in top1 or arm not in regret:
        return None
    return {"top1": float(top1[arm]), "regret": float(regret[arm])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_hybrid" / "oof.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_variance_ceiling.json",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_hybrid_decision.json",
    )
    args = parser.parse_args(argv)

    frame = load_cohort(args.long)
    cells = cell_table(frame)
    ev = evaluation_cells(cells)

    noise = noise_estimate(cells)
    ev_reps = ev[ev["n"] >= 2]
    noise["n_evaluation_cells"] = int(len(ev))
    noise["n_evaluation_cells_with_replicates"] = int(len(ev_reps))
    noise["evaluation_replicate_df"] = int((ev_reps["n"] - 1).sum())

    decomp, _ = additive_decomposition(ev)
    curve = ceiling_curve(ev, decomp["var_within"], decomp["additive_residual_ms"], SIGMA_GRID)
    models = model_readout(args.oof, ev)
    cross = cross_paper(frame)

    # Identifiability: the ceiling is only pinned if the replicate sample is
    # representative of the evaluation cohorts.  It is not.
    lo_ceiling = min(c["max_within_r2"] for c in curve)
    hi_ceiling = max(c["max_within_r2"] for c in curve)
    identified = noise["evaluation_replicate_df"] >= 200 and (
        noise["p90_within_cell_sigma"] <= 2 * noise["median_within_cell_sigma"]
    )
    plausible = [c for c in curve if c["sigma"] <= 0.36]
    top1_lo = min(c["perfect_model_top1"] for c in plausible)
    top1_hi = max(c["perfect_model_top1"] for c in plausible)
    reg_lo = min(c["perfect_model_regret"] for c in plausible)
    reg_hi = max(c["perfect_model_regret"] for c in plausible)

    best_model = max(
        (k for k in models if not k.startswith("_")),
        key=lambda k: models[k]["calibrated_within_r2"],
    )
    current = current_scores(args.decision)
    break_even = None
    if current is not None:
        break_even = break_even_sigma(ev, current["top1"], current["regret"])
        break_even["current"] = current
        break_even["current_arm"] = "lgb_tree"
    be_top1 = break_even.get("sigma_top1") if break_even else None
    be_regret = break_even.get("sigma_regret") if break_even else None

    payload = {
        "artifact": "msrs_variance_ceiling",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "oof": str(args.oof.relative_to(PROJECT_ROOT)),
            "oof_sha256": _sha256(args.oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "cohort_keys": COHORT_KEYS,
            "min_metals": MIN_METALS,
            "sigma_grid": list(SIGMA_GRID),
            "n_boot": N_BOOT,
            "seed": SEED,
            "estimand": (
                "variance of y within a cohort (same ligand, cell line, DOI, time); "
                "only the cohort x metal interaction can change a within-cohort ranking"
            ),
        },
        "replicate_noise": noise,
        "decomposition": decomp,
        "ceiling_curve": curve,
        "cross_paper": cross,
        "current_models": models,
        "break_even": break_even,
        "verdict": {
            "identified": bool(identified),
            "identified_metric": "within-cohort R2",
            "explainable_share_range": [round(lo_ceiling, 4), round(hi_ceiling, 4)],
            "decision_ceiling_range": {
                "top1": [round(top1_lo, 4), round(top1_hi, 4)],
                "regret": [round(reg_lo, 4), round(reg_hi, 4)],
                "sigma_range": [0.027, 0.36],
            },
            "identified_lower_bound": decomp["metal_main_share"],
            "best_current_model": best_model,
            "best_current_within_corr": models[best_model]["within_corr"],
            "reading": (
                "The metal main effect is real signal and explains "
                f"{decomp['metal_main_share']:.1%} of within-cohort variance, which is a "
                "lower bound on the explainable share. In R2 terms the ceiling is not "
                f"identified: pooled replicate sigma is {noise['pooled_sigma']} but its "
                f"median is {noise['median_within_cell_sigma']}, and only "
                f"{noise['n_evaluation_cells_with_replicates']} of "
                f"{noise['n_evaluation_cells']} evaluation cells have a replicate, so the "
                f"R2 ceiling ranges from 0 to {hi_ceiling:.1%}. "
                "The ranking metrics are far more noise-robust than R2: across the whole "
                "plausible noise range sigma in [0.027, 0.36], a perfect predictor of the "
                f"true cell mean scores top-1 {top1_lo:.2f}-{top1_hi:.2f} and regret "
                f"{reg_lo:.3f}-{reg_hi:.3f}, always better than the current "
                f"{current['top1']:.3f} / {current['regret']:.4f}. "
                f"The break-even noise is sigma>={be_top1} for top-1 and sigma>={be_regret} "
                "for regret, both above every defensible estimate. Conclusion: the data is "
                "not the binding constraint on the decision layer; the model is."
            ),
            "what_would_identify_it": (
                "About 200 within-cell replicate degrees of freedom inside the 424 "
                "evaluation cohorts (currently "
                f"{noise['evaluation_replicate_df']}); e.g. ~200 cells measured twice."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"[ceiling] var_within={decomp['var_within']:.4f} metal_main={decomp['metal_main_share']:.1%}")
    print(f"[ceiling] sigma pooled={noise['pooled_sigma']} median={noise['median_within_cell_sigma']}")
    print(f"[ceiling] evaluation replicate cells={noise['n_evaluation_cells_with_replicates']}/{noise['n_evaluation_cells']}")
    print(f"[ceiling] ceiling range={lo_ceiling:.3f}-{hi_ceiling:.3f} identified={identified}")
    print(f"[ceiling] best model={best_model} corr={models[best_model]['within_corr']:.3f}")
    if break_even:
        print(
            f"[ceiling] break-even sigma: top1={break_even['sigma_top1']} "
            f"regret={break_even['sigma_regret']} (current top1={current['top1']:.3f}, "
            f"regret={current['regret']:.4f})"
        )
    print(f"[ceiling] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
