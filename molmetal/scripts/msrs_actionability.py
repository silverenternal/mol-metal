#!/usr/bin/env python3
"""MSRS-3 — can the surface say "I don't know"? Split conformal + an actionable set.

=====================================================================
The question
=====================================================================
MSRS-2 ended with a hard result: the additive surface ranks metals better than
the interaction model, but its delta-sign accuracy is 0.62 on the pairs that
matter. A point prediction with 62% decision accuracy is not a design tool.
The next question is whether the *uncertainty* is usable:

    can we mark the subset of recommendations that are trustworthy, and is
    that subset actually more accurate than the rest?

=====================================================================
Why split conformal and not a Gaussian interval
=====================================================================
The residuals are not homoscedastic: a ligand from a paper with one cell line
is easier than a ligand measured across five, and Rh/Os calibration sets are
tiny. Split conformal gives finite-sample marginal coverage without assuming a
noise distribution. The score is normalised by a learned difficulty sigma(x),
so intervals shrink where the surface is confident and widen where it is not
(Mondrian-style locally adaptive conformal).

=====================================================================
Protocol (nested, so the calibration rows never trained the point model)
=====================================================================
For each outer GroupKFold(5) fold (groups = ligand_main_fragment):

    1. split the outer-train ligands into proper-train / calibration (75/25)
    2. fit the additive point model on proper-train
    3. inner GroupKFold(3) on proper-train -> OOF residuals
    4. fit a difficulty model |residual| ~ features on those OOF residuals
    5. score calibration rows: s = |y - y_hat| / sigma(x)
    6. q = the ceil((n+1)(1-alpha))/n empirical quantile of s
    7. on the outer-test fold: interval half-width = q * sigma(x)

Every row is scored exactly once, by the fold that held its ligand out. The
test ligand is never in proper-train or calibration.

=====================================================================
Actionability
=====================================================================
A metal swap m -> m' is actionable when

    the interval is narrow enough      half_width(m) + half_width(m') < delta
    the predicted gain beats noise     |y_hat(m) - y_hat(m')| > epsilon

with epsilon = 0.2091 (the MSRS-1 noise floor) and delta scanned over a grid.
"Precision" is the delta-sign accuracy on the actionable pairs. The comparison
actionable vs rest is clustered by DOI, because one paper supplies many pairs.

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_actionability.py
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    NOISE_FLOOR,
    SEED,
    TOP_CELL_LINES,
    build_design,
    fit_predict,
    matched_delta_pairs,
    oof_predict,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ENDPOINTS = ("potency", "cisplatin_ratio")
ALPHA = 0.10
CALIBRATION_FRACTION = 0.25
INNER_FOLDS = 3
DELTA_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)
PRIMARY_DELTA = 2.0
N_BOOT = 2000

DIFFICULTY_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Conformal machinery
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: ceil((n+1)(1-alpha))/n."""
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _difficulty_model(X: sparse.csr_matrix, resid: np.ndarray) -> object:
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**DIFFICULTY_PARAMS)
    model.fit(X, resid)
    return model


def _inner_residuals(
    X: sparse.csr_matrix, y: np.ndarray, groups: np.ndarray, constraints: List[List[int]]
) -> np.ndarray:
    """OOF absolute residuals inside proper-train, for the difficulty model."""
    resid = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups):
        pred = fit_predict(X[tr], y[tr], X[te], constraints)
        resid[te] = np.abs(y[te] - pred)
    assert not np.isnan(resid).any()
    return resid


def nested_conformal(
    X: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    metals: np.ndarray,
    constraints: List[List[int]],
    alpha: float = ALPHA,
) -> Dict[str, np.ndarray]:
    """Return per-row y_hat, sigma, half_width, and the metal-conditional width."""
    n = len(y)
    y_hat = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    half_global = np.full(n, np.nan)
    half_metal = np.full(n, np.nan)
    q_global = np.full(n, np.nan)
    q_metal = np.full(n, np.nan)

    outer = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        tr_groups = np.unique(groups[tr])
        splitter = GroupShuffleSplit(n_splits=1, test_size=CALIBRATION_FRACTION, random_state=SEED + k)
        proper_idx, cal_idx = next(splitter.split(tr_groups, groups=tr_groups))
        proper_groups = set(tr_groups[proper_idx])
        cal_groups = set(tr_groups[cal_idx])
        proper = tr[np.isin(groups[tr], list(proper_groups))]
        cal = tr[np.isin(groups[tr], list(cal_groups))]
        assert not (set(groups[proper]) & set(groups[cal]))
        assert not (set(groups[proper]) & set(groups[te]))

        y_hat[te] = fit_predict(X[proper], y[proper], X[te], constraints)
        y_hat_cal = fit_predict(X[proper], y[proper], X[cal], constraints)
        resid = _inner_residuals(X[proper], y[proper], groups[proper], constraints)
        diff = _difficulty_model(X[proper], resid)
        sig_cal = np.clip(diff.predict(X[cal]), 0.05, None)
        sig_te = np.clip(diff.predict(X[te]), 0.05, None)
        sigma[te] = sig_te
        scores = np.abs(y[cal] - y_hat_cal) / sig_cal
        qg = conformal_quantile(scores, alpha)
        q_global[te] = qg
        half_global[te] = qg * sig_te

        # Metal-conditional (Mondrian) quantiles; fall back to the global q when
        # a metal has too few calibration rows to support its own quantile.
        metal_q = {}
        for m in METALS:
            mask = metals[cal] == m
            metal_q[m] = conformal_quantile(scores[mask], alpha) if mask.sum() >= 30 else qg
        q_te = np.array([metal_q[m] for m in metals[te]])
        q_metal[te] = q_te
        half_metal[te] = q_te * sig_te

    out = {
        "y_hat": y_hat,
        "sigma": sigma,
        "half_width_global": half_global,
        "half_width_metal": half_metal,
        "q_global": q_global,
        "q_metal": q_metal,
    }
    assert not any(np.isnan(v).any() for v in out.values())
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def coverage_report(
    y: np.ndarray, y_hat: np.ndarray, half: np.ndarray, keys: np.ndarray, label: str
) -> Dict[str, object]:
    covered = np.abs(y - y_hat) <= half
    widths = 2 * half
    rows = {
        "n": int(len(y)),
        "coverage": round(float(covered.mean()), 4),
        "mean_width": round(float(widths.mean()), 4),
        "median_width": round(float(np.median(widths)), 4),
        "width_p10": round(float(np.percentile(widths, 10)), 4),
        "width_p90": round(float(np.percentile(widths, 90)), 4),
        "by_group": {},
    }
    for k in sorted(set(keys)):
        mask = keys == k
        if mask.sum() == 0:
            continue
        rows["by_group"][str(k)] = {
            "n": int(mask.sum()),
            "coverage": round(float(covered[mask].mean()), 4),
            "mean_width": round(float(2 * half[mask].mean()), 4),
        }
    return rows


def actionability_report(
    df: pd.DataFrame,
    y: np.ndarray,
    y_hat: np.ndarray,
    half: np.ndarray,
    epsilon: float = NOISE_FLOOR,
) -> Dict[str, object]:
    pairs = matched_delta_pairs(df)
    if pairs.empty:
        return {"status": "NO_MATCHED_PAIRS"}

    ia, ib = pairs["idx_a"].to_numpy(), pairs["idx_b"].to_numpy()
    # gain(m_a -> m_b) > 0 means m_b is more potent (lower log10 IC50).
    gain_obs = y[ia] - y[ib]
    gain_hat = y_hat[ia] - y_hat[ib]
    pair_half = half[ia] + half[ib]
    informative = np.abs(gain_obs) >= epsilon

    out: Dict[str, object] = {
        "n_pairs": int(len(pairs)),
        "n_informative": int(informative.sum()),
        "n_dois": int(pairs["doi"].nunique()),
        "epsilon": epsilon,
        "delta_grid": {},
    }

    # Does interval width identify easier decisions at all? Split the informative
    # pairs into width quartiles; a flat accuracy profile means the interval is
    # calibrated but carries no decision value.
    correct = (np.sign(gain_hat) == np.sign(gain_obs)).astype(float)
    inf_half = pair_half[informative]
    inf_correct = correct[informative]
    if len(inf_half) >= 40:
        edges = np.percentile(inf_half, [25, 50, 75])
        quart = np.digitize(inf_half, edges)
        out["accuracy_by_width_quartile"] = {
            str(q + 1): {
                "n": int((quart == q).sum()),
                "mean_pair_width": round(float(2 * inf_half[quart == q].mean()), 4),
                "accuracy": round(float(inf_correct[quart == q].mean()), 4),
            }
            for q in range(4)
            if (quart == q).sum() > 0
        }
        out["point_biserial_width_vs_correct"] = round(
            float(stats.pointbiserialr(inf_correct, inf_half).statistic), 4
        )

    for delta in DELTA_GRID:
        act = (pair_half < delta) & (np.abs(gain_hat) > epsilon) & informative
        rest = (~act) & informative
        acc_all = float((np.sign(gain_hat[informative]) == np.sign(gain_obs[informative])).mean())
        entry = {
            "n_actionable": int(act.sum()),
            "share_actionable": round(float(act.sum() / max(informative.sum(), 1)), 4),
            "accuracy_actionable": round(float((np.sign(gain_hat[act]) == np.sign(gain_obs[act])).mean()), 4)
            if act.sum()
            else None,
            "accuracy_rest": round(float((np.sign(gain_hat[rest]) == np.sign(gain_obs[rest])).mean()), 4)
            if rest.sum()
            else None,
            "accuracy_all_informative": round(acc_all, 4),
            "pair_coverage_actionable": round(float((np.abs(gain_obs[act] - gain_hat[act]) <= pair_half[act]).mean()), 4)
            if act.sum()
            else None,
        }
        if act.sum() >= 10 and rest.sum() >= 10:
            entry["accuracy_diff_actionable_minus_rest"] = round(
                float(entry["accuracy_actionable"] - entry["accuracy_rest"]), 4
            )
        out["delta_grid"][f"{delta:g}"] = entry

    # A single clustered contrast at the primary delta: resample DOIs, recompute
    # both accuracies inside each resample (the actionable set is re-selected).
    act = (pair_half < PRIMARY_DELTA) & (np.abs(gain_hat) > epsilon) & informative
    out["primary_delta"] = PRIMARY_DELTA
    out["primary"] = {
        "n_actionable": int(act.sum()),
        "accuracy_actionable": round(float(correct[act].mean()), 4) if act.sum() else None,
        "accuracy_rest": round(float(correct[informative & ~act].mean()), 4)
        if (informative & ~act).sum()
        else None,
        "accuracy_diff_ci95_doi_clustered": _clustered_contrast(
            correct, act, informative, pairs["doi"].to_numpy()
        ),
        "metal_pair_composition_actionable": _composition(pairs, act),
        "metal_pair_composition_rest": _composition(pairs, informative & ~act),
        "by_metal_pair": _stratified_accuracy(pairs, correct, act, informative),
    }
    return out


def _stratified_accuracy(
    pairs: pd.DataFrame, correct: np.ndarray, act: np.ndarray, informative: np.ndarray
) -> Dict[str, object]:
    """Within each metal pair, compare actionable vs rest (removes composition bias)."""
    out: Dict[str, object] = {}
    for (a, b), g in pairs.groupby(["metal_a", "metal_b"]):
        idx = g.index.to_numpy()
        a_mask = act[idx]
        r_mask = informative[idx] & ~act[idx]
        if a_mask.sum() < 10 or r_mask.sum() < 10:
            continue
        acc_a = float(correct[idx][a_mask].mean())
        acc_r = float(correct[idx][r_mask].mean())
        out[f"{a}-{b}"] = {
            "n_actionable": int(a_mask.sum()),
            "n_rest": int(r_mask.sum()),
            "accuracy_actionable": round(acc_a, 4),
            "accuracy_rest": round(acc_r, 4),
            "diff": round(acc_a - acc_r, 4),
        }
    return out


def _composition(pairs: pd.DataFrame, mask: np.ndarray) -> Dict[str, int]:
    if mask.sum() == 0:
        return {}
    sub = pairs.loc[mask]
    return {
        f"{a}-{b}": int(n)
        for (a, b), n in sub.groupby(["metal_a", "metal_b"]).size().items()
    }


def _clustered_contrast(
    correct: np.ndarray,
    act: np.ndarray,
    informative: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Optional[List[float]]:
    """Bootstrap DOIs and recompute accuracy(actionable) - accuracy(rest)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        inf = informative[idx]
        a = act[idx] & inf
        r = (~act[idx]) & inf
        if a.sum() < 5 or r.sum() < 5:
            continue
        draws.append(correct[idx][a].mean() - correct[idx][r].mean())
    if len(draws) < 100:
        return None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------------------
# One endpoint
# ---------------------------------------------------------------------------
def run_endpoint(
    df: pd.DataFrame, endpoint: str, descriptors: pd.DataFrame, verbose: bool = True
) -> Dict[str, object]:
    t0 = time.time()
    sub = df[df["endpoint"] == endpoint]
    n_total = len(sub)
    n_censored = int(sub["value_log10"].isna().sum())
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return {"status": "EMPTY"}

    vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, _, constraints = build_design(sub, descriptors, vocab)
    y = sub["value_log10"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    metals = sub["metal"].to_numpy()

    conf = nested_conformal(X, y, groups, metals, constraints)
    cov_global = coverage_report(y, conf["y_hat"], conf["half_width_global"], metals, "metal")
    cov_metal = coverage_report(y, conf["y_hat"], conf["half_width_metal"], metals, "metal")
    act = actionability_report(sub, y, conf["y_hat"], conf["half_width_global"])

    if verbose:
        print(
            f"[{endpoint}] n={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
            f"dropped_censored={n_censored}"
        )
        print(
            f"  coverage global={cov_global['coverage']} (target {1-ALPHA:.2f}) "
            f"width={cov_global['median_width']} | metal-conditional={cov_metal['coverage']} "
            f"width={cov_metal['median_width']}"
        )
        p = act.get("primary", {})
        print(
            f"  actionable @delta={PRIMARY_DELTA}: n={p.get('n_actionable')} "
            f"acc={p.get('accuracy_actionable')} vs rest={p.get('accuracy_rest')} "
            f"diff CI={p.get('accuracy_diff_ci95_doi_clustered')}"
        )

    return {
        "status": "OK",
        "n_rows": int(len(sub)),
        "n_dropped_censored": n_censored,
        "censored_share": round(n_censored / n_total, 4) if n_total else None,
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "alpha": ALPHA,
        "target_coverage": 1 - ALPHA,
        "coverage_global": cov_global,
        "coverage_metal_conditional": cov_metal,
        "q_global_median": round(float(np.median(conf["q_global"])), 4),
        "actionability": act,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


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
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_actionability.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results = {ep: run_endpoint(df, ep, descriptors) for ep in ENDPOINTS}

    out = {
        "artifact": "msrs_actionability",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 actionability (split conformal)",
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
            "point_model": "additive LightGBM from MSRS-2 (MSRS-2: better delta ranking than interaction)",
            "outer_cv": "GroupKFold(5) by ligand_main_fragment",
            "inner_split": f"GroupShuffleSplit calibration fraction {CALIBRATION_FRACTION} by ligand",
            "difficulty_model": "LightGBM on |OOF residual| from GroupKFold(3) inside proper-train",
            "nonconformity": "|y - y_hat| / sigma(x), sigma clipped at 0.05",
            "quantile": "ceil((n+1)(1-alpha))/n empirical quantile",
            "alpha": ALPHA,
            "epsilon": NOISE_FLOOR,
            "delta_grid": list(DELTA_GRID),
            "primary_delta": PRIMARY_DELTA,
        },
        "endpoints": results,
        "caveats": [
            "Coverage is marginal over the whole test fold; ligand-grouped CV makes it a "
            "new-ligand coverage statement, not a new-paper one. MSRS-7 tests the paper split.",
            "The difficulty model is fitted on inner-OOF residuals from the proper-train "
            "ligands only, so it never sees calibration or test rows. It is still a learned "
            "quantity and can be optimistic where a ligand is unlike anything in training.",
            "The actionable set is defined on matched pairs (loose key). A pair is actionable "
            "only if both endpoints' intervals are narrow; this is conservative.",
            "Metal-conditional quantiles fall back to the global quantile when a metal has "
            "fewer than 30 calibration rows, which is the case for Rh/Os in several folds.",
            "The actionable-vs-rest contrast resamples DOIs and re-selects the actionable set "
            "inside each resample, so the CI includes selection variability.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
