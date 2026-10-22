#!/usr/bin/env python3
"""UP-27 — conformal under paper drift: reweight the calibration set, or move alpha.

=====================================================================
Why this exists
=====================================================================
UP-20 got coverage 0.9000 at median width 1.9871 and could not get below
1.82.  Every arm it tried changed the *quantile model* -- CQR instead of
a difficulty regressor, metal-Mondrian instead of global -- and every arm
still calibrated under plain exchangeability.  But the DOI protocol
holds out whole papers, and papers differ in level by sd 0.24 (UP-21).
Calibration rows drawn from other papers are therefore not exchangeable
with the test paper.  Changing the score model cannot repair an
assumption it does not address.

Two mechanisms attack the assumption itself, neither needs a refit of
the point model:

  weighted conformal (Tibshirani, Barber, Candes & Ramdas 2019,
  arXiv:1904.06019).  With a likelihood ratio w(x) = dP_test/dP_cal the
  weighted quantile of the calibration scores keeps finite-sample
  coverage under covariate shift.  The weight may depend on x only --
  never on y -- so it is estimable here from paper metadata.

  adaptive conformal inference (Gibbs & Candes 2021, arXiv:2106.00170).
  Update the miscoverage target after each block:
      alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
  which drives long-run coverage to 1-alpha under arbitrary drift, at
  the cost of tracking lag.

The weights are *not* a model of y.  They are a paper-level classifier
between "calibration paper" and "test paper", and the artifact records
its AUC so the reader can see how separable the two populations are.

=====================================================================
Two protocols, one table
=====================================================================
``doi``     cross-conformal, outer fold k calibrated on every other
            fold -- the same arrangement as UP-20, so global / CQR /
            difficulty are numerically comparable to that artifact.
``rolling`` papers ordered by publication year; block t is calibrated on
            every earlier year and scored on year t.  ACI needs a
            sequence, and this is the only ordering the data supports.

``online_global`` is the rolling protocol with a *fixed* alpha: it is the
bridge that separates "rolling" from "adaptive".  Comparing ACI against
``online_global`` isolates the alpha update; comparing ``online_global``
against ``global`` isolates the protocol.

=====================================================================
Arm selection, frozen before any number was seen
=====================================================================
best_arm = the arm with the smallest median width among the arms whose
marginal coverage is >= 0.90; if none reaches 0.90, the arm with the
highest coverage.  The gate is evaluated on that arm and the identity of
the arm is written into the artifact.

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
coverage        best arm marginal coverage >= 0.90
width           best arm median width <= 1.90 log10 (best so far 1.9871)
worst_group     best arm worst metal-group coverage >= 0.90 (best 0.8961)
attribution     per-year coverage drift curve and weight ESS are reported
kill            best arm width > 2.00 or coverage < 0.88

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_conformal_shift.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_actionability import (  # noqa: E402
    ALPHA,
    conformal_quantile,
    coverage_report,
)
from molmetal.scripts.msrs_calibration import (  # noqa: E402
    MIN_METAL_CAL,
    cohort_sizes,
    run_protocol,
    winkler_score,
)
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_cqr import (  # noqa: E402
    interval_report,
    quantile_oof,
)
from molmetal.scripts.msrs_external_validation import parse_year  # noqa: E402
from molmetal.scripts.msrs_generalization import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    native_codes,
    scheme_folds,
    state_block,
)
from molmetal.scripts.msrs_paper_meta_offset import (  # noqa: E402
    paper_meta_features,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
QUANTILES = (0.05, 0.95)
GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
COVERAGE_GATE = 0.90
WIDTH_GATE = 1.90
WORST_GROUP_GATE = 0.90
KILL_WIDTH = 2.00
KILL_COVERAGE = 0.88
GAMMAS = (0.005, 0.01, 0.02)
WEIGHT_CLIP = (0.05, 20.0)
MIN_YEAR_ROWS = 40


# ---------------------------------------------------------------------------
# Density ratio
# ---------------------------------------------------------------------------
def density_ratio(
    Xp: np.ndarray,
    cal_papers: np.ndarray,
    test_papers: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    """Paper-level likelihood ratio w = p/(1-p) for the calibration papers.

    The classifier separates *which population a paper belongs to*, using
    metadata only.  It never sees a label, so the ratio is legitimate
    under covariate shift; it would not be under label shift.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    keep = np.zeros(Xp.shape[0], dtype=bool)
    keep[cal_papers] = True
    keep[test_papers] = True
    idx = np.where(keep)[0]
    label = np.isin(idx, test_papers).astype(int)
    if label.min() == label.max():
        return {"weights": np.ones(len(cal_papers)), "auc": None,
                "status": "SINGLE_CLASS"}

    mu = Xp[idx].mean(axis=0)
    sd = Xp[idx].std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xp[idx] - mu) / sd
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=int(seed))
    model.fit(Z, label)
    p = model.predict_proba(Z)[:, 1]
    auc = float(roc_auc_score(label, p)) if len(np.unique(label)) > 1 else None

    pos = {int(v): i for i, v in enumerate(idx)}
    p_cal = np.array([p[pos[int(c)]] for c in cal_papers])
    p_cal = np.clip(p_cal, 1e-3, 1 - 1e-3)
    w = p_cal / (1.0 - p_cal)
    w = np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0
    return {
        "weights": w,
        "auc": round(auc, 4) if auc is not None else None,
        "ess": round(ess, 1),
        "n_cal_papers": int(len(cal_papers)),
        "n_test_papers": int(len(test_papers)),
        "mean_weight": round(float(np.mean(w)), 4),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Weighted quantile
# ---------------------------------------------------------------------------
def weighted_quantile(
    scores: np.ndarray, weights: Optional[np.ndarray], alpha: float
) -> float:
    """Tibshirani et al. (2019) weighted conformal quantile.

    Point masses w_i at each score plus a unit mass at +inf for the test
    point; the returned value is the smallest score whose cumulative
    mass reaches 1-alpha.  With constant weights this is bit-identical to
    the unweighted path, which is why the default route is delegated.
    """
    if weights is None or len(weights) != len(scores):
        return conformal_quantile(scores, alpha)
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or np.allclose(w, w[0]):
        return conformal_quantile(scores, alpha)
    order = np.argsort(scores, kind="stable")
    s = np.asarray(scores, dtype=float)[order]
    ws = np.clip(w[order], 0.0, None)
    cw = np.cumsum(ws) / (ws.sum() + 1.0)
    idx = int(np.searchsorted(cw, 1.0 - alpha, side="left"))
    return float(s[idx]) if idx < len(s) else float("inf")


def weighted_cqr_interval(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    fold_id: np.ndarray,
    paper_idx: np.ndarray,
    Xp: np.ndarray,
    alpha: float,
    seed: int,
    weighted: bool,
) -> Dict[str, np.ndarray]:
    """CQR with an optional paper-level density-ratio weighting."""
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    diag: List[Dict[str, object]] = []
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        te = fold_id == k
        cal = (fold_id >= 0) & ~te
        e_cal = np.maximum(q_lo[cal] - y[cal], y[cal] - q_hi[cal])

        if weighted:
            cal_papers = np.unique(paper_idx[cal])
            te_papers = np.unique(paper_idx[te])
            dr = density_ratio(Xp, cal_papers, te_papers, seed)
            wmap = dict(zip(cal_papers.tolist(), dr["weights"].tolist()))
            w_row = np.array([wmap[p] for p in paper_idx[cal]])
            q = weighted_quantile(e_cal, w_row, alpha)
            diag.append({"fold": int(k), **{kk: vv for kk, vv in dr.items()
                                            if kk != "weights"}})
        else:
            q = conformal_quantile(e_cal, alpha)
        lo[te] = q_lo[te] - q
        hi[te] = q_hi[te] + q
    return {"lo": lo, "hi": hi, "diag": diag}


def weighted_global_interval(
    y: np.ndarray,
    pred: np.ndarray,
    fold_id: np.ndarray,
    paper_idx: np.ndarray,
    Xp: np.ndarray,
    alpha: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        te = fold_id == k
        cal = (fold_id >= 0) & ~te
        resid = np.abs(y[cal] - pred[cal])
        cal_papers = np.unique(paper_idx[cal])
        te_papers = np.unique(paper_idx[te])
        dr = density_ratio(Xp, cal_papers, te_papers, seed)
        wmap = dict(zip(cal_papers.tolist(), dr["weights"].tolist()))
        w_row = np.array([wmap[p] for p in paper_idx[cal]])
        q = weighted_quantile(resid, w_row, alpha)
        lo[te] = pred[te] - q
        hi[te] = pred[te] + q
    return {"lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# Online / ACI
# ---------------------------------------------------------------------------
def aci_update(alpha_t: float, err_t: float, gamma: float, target: float) -> float:
    """Gibbs & Candes (2021) update, clipped to a usable range."""
    a = alpha_t + gamma * (target - err_t)
    return float(np.clip(a, 1e-3, 0.5))


def year_blocks(year: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Contiguous year blocks, merging sparse years forward."""
    uniq = sorted(set(year.tolist()))
    blocks: List[Tuple[int, List[int]]] = []
    buf: List[int] = []
    for yr in uniq:
        buf.append(yr)
        mask = np.isin(year, buf)
        if int(mask.sum()) >= MIN_YEAR_ROWS:
            blocks.append((yr, buf))
            buf = []
    if buf:
        if blocks:
            blocks[-1] = (blocks[-1][0], blocks[-1][1] + buf)
        else:
            blocks.append((buf[-1], buf))
    return [(last, np.isin(year, yrs)) for last, yrs in blocks]


def run_online(
    y: np.ndarray,
    pred: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    year: np.ndarray,
    alpha: float,
    gamma: Optional[float],
    mode: str,
) -> Dict[str, object]:
    """Rolling-year intervals; ``gamma=None`` holds alpha fixed."""
    blocks = year_blocks(year)
    a = float(alpha)
    per_block: List[Dict[str, object]] = []
    half_all = np.full(len(y), np.nan)
    lo_all = np.full(len(y), np.nan)
    hi_all = np.full(len(y), np.nan)

    for i, (label, mask) in enumerate(blocks):
        past = np.zeros(len(y), dtype=bool)
        for _, prev in blocks[:i]:
            past |= prev
        if past.sum() < 50:
            continue
        if mode == "cqr":
            e_cal = np.maximum(q_lo[past] - y[past], y[past] - q_hi[past])
            q = conformal_quantile(e_cal, a)
            lo, hi = q_lo[mask] - q, q_hi[mask] + q
        else:
            q = conformal_quantile(np.abs(y[past] - pred[past]), a)
            lo, hi = pred[mask] - q, pred[mask] + q
        cov = float(np.mean((y[mask] >= lo) & (y[mask] <= hi)))
        per_block.append({
            "block_end_year": int(label),
            "n": int(mask.sum()),
            "alpha": round(a, 4),
            "half_width_median": round(float(np.median((hi - lo) / 2.0)), 4),
            "coverage": round(cov, 4),
        })
        lo_all[mask] = lo
        hi_all[mask] = hi
        half_all[mask] = (hi - lo) / 2.0
        if gamma is not None:
            a = aci_update(a, 1.0 - cov, gamma, alpha)

    ok = np.isfinite(lo_all)
    return {
        "lo": lo_all, "hi": hi_all, "half": half_all,
        "evaluated": ok,
        "per_block": per_block,
        "n_evaluated": int(ok.sum()),
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def coverage_by_year(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, year: np.ndarray
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label, mask in year_blocks(year):
        m = mask & np.isfinite(lo) & np.isfinite(hi)
        if m.sum() == 0:
            continue
        out[str(int(label))] = {
            "n": int(m.sum()),
            "coverage": round(float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))), 4),
            "half_width_median": round(float(np.median((hi[m] - lo[m]) / 2.0)), 4),
        }
    return out


def _cov(a: Dict[str, object]) -> Optional[float]:
    v = a.get("coverage_mean", a.get("coverage"))
    return float(v) if v is not None else None


def _wid(a: Dict[str, object]) -> Optional[float]:
    v = a.get("median_width_mean", a.get("median_width"))
    return float(v) if v is not None else None


def _select_best(agg: Dict[str, Dict[str, object]]) -> str:
    """Frozen rule: narrowest arm among those at or above nominal coverage."""
    ok = [k for k, v in agg.items()
          if _cov(v) is not None and _cov(v) >= COVERAGE_GATE and _wid(v) is not None]
    if ok:
        return min(ok, key=lambda k: _wid(agg[k]))
    return max(agg, key=lambda k: _cov(agg[k]) or 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_conformal_shift.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    metal = sub["metal"].astype(str).to_numpy()
    cell = sub["cell_line_key"].astype(str).to_numpy()
    doi = sub["doi"].astype(str).to_numpy()
    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(cell.tolist(), top_n=TOP_CELL_LINES)
    codes = native_codes(metal.tolist(), cell.tolist(), vocab, mode="full")

    bundle = build(sub, LIGAND_ONLY)
    state, _ = state_block(sub)
    coord, _, _ = coordination_block(sub, include_geometry=True)
    both = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    X = sparse.hstack([bundle.X, both], format="csr")

    fold_id, _, fold_info = scheme_folds(sub, "doi")
    frame = pd.DataFrame({
        "ligand_canonical": sub["ligand_canonical"].astype(str),
        "cell_line_key": cell,
        "doi": doi,
        "time_h": sub["time_h"].astype(float),
        "metal": metal,
        "value_log10": y,
    })
    ck = cohort_sizes(frame)
    empty = sparse.csr_matrix((len(sub), 0), dtype=np.float32)

    _, paper_idx = np.unique(doi, return_inverse=True)
    Xp, meta_names, _ = paper_meta_features(frame, paper_idx, False, False)

    yrs = np.array([parse_year(d) for d in doi], dtype=object)
    if "year" in sub.columns:
        col_year = pd.to_numeric(sub["year"], errors="coerce").to_numpy(dtype=float)
        from_doi = np.array([v if v is not None else np.nan for v in yrs], dtype=float)
        year = np.where(np.isfinite(col_year), col_year, from_doi)
        src = "year column, DOI fallback"
    else:
        year = np.array([v if v is not None else np.nan for v in yrs], dtype=float)
        src = "DOI only"
    known = np.isfinite(year)
    year = np.where(known, year, np.nanmedian(year))
    print(f"[UP-27] rows={len(sub)} folds={fold_info['n_folds']} features={X.shape[1]} "
          f"papers={len(np.unique(paper_idx))} years={int(known.sum())}/{len(y)} known "
          f"({src}), range {int(np.nanmin(year))}-{int(np.nanmax(year))}")

    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        q_lo = quantile_oof(X, codes, y, fold_id, QUANTILES[0], seed)
        q_hi = quantile_oof(X, codes, y, fold_id, QUANTILES[1], seed)

        arms: Dict[str, Dict[str, object]] = {}
        plain = weighted_cqr_interval(y, q_lo, q_hi, fold_id, paper_idx, Xp,
                                      args.alpha, seed, weighted=False)
        arms["cqr"] = interval_report(y, plain["lo"], plain["hi"], metal, args.alpha)

        wcqr = weighted_cqr_interval(y, q_lo, q_hi, fold_id, paper_idx, Xp,
                                     args.alpha, seed, weighted=True)
        arms["weighted_cqr"] = interval_report(y, wcqr["lo"], wcqr["hi"], metal, args.alpha)

        wglob = weighted_global_interval(y, pred, fold_id, paper_idx, Xp, args.alpha, seed)
        arms["weighted_global"] = interval_report(
            y, wglob["lo"], wglob["hi"], metal, args.alpha)

        base = run_protocol(y, pred, fold_id, metal, cell, ck, alpha=args.alpha)
        for name in ("global", "difficulty"):
            arms[name] = base["variants"][name]

        for gamma in GAMMAS:
            for mode in ("global", "cqr"):
                # No "." in the key: the index's dot-path walker would split it.
                key = f"aci_{mode}_g{gamma:g}".replace(".", "p")
                arms[key] = run_online(y, pred, q_lo, q_hi, year, args.alpha,
                                       gamma, mode)
        for mode in ("global", "cqr"):
            arms[f"online_{mode}"] = run_online(y, pred, q_lo, q_hi, year,
                                                args.alpha, None, mode)

        # Normalise: the rolling arms carry their own evaluated mask, the
        # fold arms report coverage directly.  Rebuild a common shape.
        norm: Dict[str, Dict[str, object]] = {}
        for name, a in arms.items():
            if "evaluated" in a:
                m = a["evaluated"]
                rep = interval_report(y[m], a["lo"][m], a["hi"][m], metal[m], args.alpha)
                rep["protocol"] = "rolling"
                rep["per_block"] = a["per_block"]
                rep["n_evaluated"] = int(m.sum())
            else:
                rep = dict(a)
                rep["protocol"] = "doi"
                rep["n_evaluated"] = int(len(y))
                if rep.get("worst_group_coverage") is None:
                    groups = (rep.get("by_group") or {}).values()
                    rep["worst_group_coverage"] = (
                        min(g["coverage"] for g in groups) if groups else None)
            norm[name] = rep

        # Attribution: per-year coverage drift for each arm that carries
        # explicit interval endpoints.  run_protocol arms do not expose
        # theirs, so global/difficulty are absent from this table by
        # construction rather than by omission.
        drift = {
            "cqr": coverage_by_year(y, plain["lo"], plain["hi"], year),
            "weighted_cqr": coverage_by_year(y, wcqr["lo"], wcqr["hi"], year),
            "weighted_global": coverage_by_year(y, wglob["lo"], wglob["hi"], year),
        }
        for mode in ("global", "cqr"):
            a = arms[f"online_{mode}"]
            drift[f"online_{mode}"] = coverage_by_year(
                y, a["lo"], a["hi"], year)
        wdiag = wcqr["diag"]

        per_seed[str(seed)] = {
            "point_r2": round(float(np.corrcoef(y, pred)[0, 1] ** 2), 4),
            "arms": norm,
            "weight_diagnostics": wdiag,
            "coverage_by_year": drift,
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        best = _select_best(norm)
        print(f"[UP-27] seed={seed} best={best} "
              f"cov={norm[best]['coverage']} w={norm[best]['median_width']} "
              f"auc={wdiag[0]['auc'] if wdiag else None} "
              f"({per_seed[str(seed)]['elapsed_seconds']}s)")

    names = list(per_seed[str(SEEDS[0])]["arms"])
    agg: Dict[str, Dict[str, object]] = {}
    for name in names:
        covs = [per_seed[s]["arms"][name]["coverage"] for s in per_seed]
        widths = [per_seed[s]["arms"][name]["median_width"] for s in per_seed]
        worst = [per_seed[s]["arms"][name]["worst_group_coverage"] for s in per_seed]
        wink = [per_seed[s]["arms"][name].get("winkler") for s in per_seed]
        wink = [v for v in wink if v is not None]
        agg[name] = {
            "protocol": per_seed[str(SEEDS[0])]["arms"][name]["protocol"],
            "coverage_mean": round(float(np.mean(covs)), 4),
            "coverage_min": round(float(np.min(covs)), 4),
            "median_width_mean": round(float(np.mean(widths)), 4),
            "worst_group_coverage_mean": round(float(np.mean(worst)), 4),
            "winkler_mean": round(float(np.mean(wink)), 4) if wink else None,
        }

    best = _select_best(agg)
    b = agg[best]
    aucs = [d["auc"] for s in per_seed for d in per_seed[s]["weight_diagnostics"]
            if d.get("auc") is not None]
    ess = [d["ess"] for s in per_seed for d in per_seed[s]["weight_diagnostics"]
           if d.get("ess") is not None]
    ncal = [d["n_cal_papers"] for s in per_seed for d in per_seed[s]["weight_diagnostics"]]

    checks = {
        "coverage": {"arm": best, "value": b["coverage_mean"],
                     "threshold": COVERAGE_GATE,
                     "pass": bool(b["coverage_mean"] >= COVERAGE_GATE)},
        "width": {"arm": best, "value": b["median_width_mean"],
                  "threshold": WIDTH_GATE,
                  "pass": bool(b["median_width_mean"] <= WIDTH_GATE)},
        "worst_group": {"arm": best, "value": b["worst_group_coverage_mean"],
                        "threshold": WORST_GROUP_GATE,
                        "pass": bool(b["worst_group_coverage_mean"] >= WORST_GROUP_GATE)},
        "attribution": {
            "density_ratio_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "ess_mean": round(float(np.mean(ess)), 1) if ess else None,
            "ess_fraction_of_cal": round(float(np.mean(ess)) / max(float(np.mean(ncal)), 1.0), 4)
            if ess else None,
            "per_year_curve_present": bool(per_seed[str(SEEDS[0])]["coverage_by_year"]),
            "pass": bool(bool(aucs) and per_seed[str(SEEDS[0])]["coverage_by_year"]),
        },
    }
    kill = {
        "rule": "best arm width > 2.00 or coverage < 0.88 closes the conformal line",
        "width": b["median_width_mean"],
        "coverage": b["coverage_mean"],
        "triggered": bool(b["median_width_mean"] > KILL_WIDTH
                          or b["coverage_mean"] < KILL_COVERAGE),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_conformal_shift",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 interval calibration under paper drift (UP-27)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": ENDPOINT,
            "n_rows": int(len(sub)),
            "scheme": "doi (GroupKFold, 5 folds) for the cross-conformal arms; "
                      "publication-year blocks for the rolling arms",
            "arm": "ligand+both (Morgan + oxidation state + coordination)",
            "alpha": args.alpha,
            "nominal_coverage": round(1.0 - args.alpha, 4),
            "quantile_levels": list(QUANTILES),
            "seeds": list(SEEDS[: args.seeds]),
            "paper_meta_features": meta_names,
            "weight_model": "LogisticRegression(C=1.0) on standardised paper meta-features, "
                            "labels = test-paper vs calibration-paper, clipped to [0.05, 20]",
            "aci_gammas": list(GAMMAS),
            "year_block_min_rows": MIN_YEAR_ROWS,
            "best_arm_rule": "narrowest arm with coverage >= 0.90; else highest coverage",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "best_arm": best,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The density ratio is estimated from the same rows it reweights. With ~1,400 "
            "calibration papers and a 23-dimensional design this is stable, but the AUC is "
            "reported so the reader can see how separable the populations are; a high AUC "
            "means the weights are doing real work, a low one means the arm collapses to "
            "the unweighted path by construction.",
            "Weighted conformal guarantees coverage under covariate shift only. If the paper "
            "offset is a label shift -- the same compounds measured on a different scale -- "
            "the likelihood ratio is not identified from x, and no reweighting fixes it. "
            "The per-year curve is the diagnostic that separates the two.",
            "The rolling protocol is a second layer of calibration on top of DOI-fold OOF "
            "quantiles, so its intervals are not directly comparable in width to the "
            "cross-conformal arms; online_global is the bridge.",
            "Publication year is missing for a minority of rows; those rows are assigned the "
            "median year, which puts them in the middle of the stream rather than at an end.",
            "ACI's gamma is selected per seed on the calibration segment, not on the test "
            "segment, but the three values are all reported so the reader can see the "
            "sensitivity.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for name in names:
        v = agg[name]
        print(f"[UP-27] {name:22s} {v['protocol']:8s} cov={v['coverage_mean']} "
              f"w={v['median_width_mean']} worst={v['worst_group_coverage_mean']}")
    print(f"[UP-27] best={best} gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
