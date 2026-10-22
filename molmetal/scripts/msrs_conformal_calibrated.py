#!/usr/bin/env python3
"""UP-33 — put the calibration in front of the interval, not behind it.

=====================================================================
Why this exists
=====================================================================
UP-20 and UP-27 spent four interval families (split, CQR, difficulty,
metal-Mondrian) trying to get below 1.99 log10 at 90% coverage and the
best they reached was 1.9871.  Every one of those families calibrated on
the *uncalibrated* point prediction, whose residual carries a per-paper
level that UP-21 measured at sd 0.24 and UP-26 removed with a handful of
measured rows.

A split-conformal half-width is a quantile of |y - y_hat|.  Remove a
per-paper constant from the residual and the quantile moves with the
residual scale; that is arithmetic, not a modelling choice.  UP-26 took
the marginal R^2 from 0.3674 to 0.6839, so the residual sd is ~71% of
what it was and the width has ~0.71 of its old value available to it.

The same argument does *not* apply to CQR, and that is why this script
runs both families.  A CQR interval is [q_lo - E, q_hi + E]; shifting a
paper by a constant moves q_lo and q_hi together and leaves their
difference untouched, so only the conformal correction E can shrink.
CQR is therefore the family where calibration should help least, and it
is reported as a contrast rather than as the primary arm.

=====================================================================
The step that is easy to get wrong
=====================================================================
The rows that estimate a paper's offset must not be the rows that
produce its conformal score.  If they overlap, the score distribution
is computed on residuals that were shrunk toward zero by the very
offset being tested, the quantile is biased low, and coverage falls
below nominal on data that never saw the correction.  Every arm here
therefore draws the offset rows first and scores only the complement,
on both sides of the split: the calibration papers and the test papers
are treated identically.

=====================================================================
Arms
=====================================================================
``split_uncal``        the UP-27 ``global`` arm: |resid| over every
                       calibration row, half-width from that quantile.
                       Reproduces 2.0073.
``split_uncal_matched``the same half-width scored on the calibrated
                       arm's evaluation rows, so the width comparison
                       is paired on identical rows.
``split_cal``          per-paper EB offset, score pool built from other
                       papers' *calibrated* residuals.  PRIMARY.
``split_cal_metal``    the same pool, Mondrian by the metal of the
                       scoring row (>= 30 pool rows per metal).
``split_oracle``       per test paper, the quantile of |resid - mean_p|
                       over all of that paper's rows.  Lower bound for
                       any estimator that only shifts a paper.
``cqr_uncal``          the UP-27 ``cqr`` arm.  Reproduces 1.9871.
``cqr_cal``            CQR correction E recomputed on calibrated
                       scores.  Reported as the contrast described above.

=====================================================================
Gate (todo.json v5, frozen 2026-10-23T03:20+08:00; ``reproduce``
amended at 2026-10-23T03:40+08:00, before any number was produced)
=====================================================================
coverage        split_cal marginal coverage >= 0.90 at alpha = 0.10
median_width    split_cal median width <= 1.50 log10
worst_metal     split_cal worst metal-group coverage >= 0.87
paired          split_cal at least 20% narrower than split_uncal_matched
reproduce       split_uncal = 2.0073 +- 0.02 AND cqr_uncal = 1.9871 +- 0.02
kill            split_cal median width > 1.90 closes the interval line

The amendment exists because the frozen text named 1.9871, which is the
CQR arm's value, while the frozen arm list is split-conformal.  Both
families are now reproduced instead of one, which is the stricter check.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_conformal_calibrated.py
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
from molmetal.scripts.msrs_cqr import interval_report, quantile_oof  # noqa: E402
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
from molmetal.scripts.msrs_paper_calibration import (  # noqa: E402
    eb_offset,
    paper_variances,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
QUANTILES = (0.05, 0.95)
N_CAL = 10
REPEATS = 20
GATE_FROZEN_AT = "2026-10-23T03:20+08:00"
GATE_AMENDED_AT = "2026-10-23T03:40+08:00"
COVERAGE_GATE = 0.90
WIDTH_GATE = 1.50
WORST_METAL_GATE = 0.87
PAIRED_NARROW = 0.20
KILL_WIDTH = 1.90
REPRO_SPLIT = 2.0073
REPRO_CQR = 1.9871
REPRO_TOL = 0.02


# ---------------------------------------------------------------------------
# One paper's offset, drawn so the scored rows never touch the offset rows
# ---------------------------------------------------------------------------
def draw_offset(
    resid: np.ndarray,
    rows: np.ndarray,
    n_cal: int,
    rng: np.random.Generator,
    kappa: float,
) -> Tuple[float, np.ndarray]:
    """Return (offset_hat, evaluation rows) for one paper.

    The permutation is drawn once and split at ``c``: the prefix estimates
    the offset, the suffix is scored.  ``c`` is capped at n-1 so a paper
    always keeps at least one row to be scored on.
    """
    n = len(rows)
    c = min(n_cal, n - 1)
    if c <= 0:
        return 0.0, rows
    perm = rng.permutation(n)
    off = eb_offset(resid[rows[perm[:c]]], kappa)
    return off, rows[perm[c:]]


def pool_scores(
    resid: np.ndarray,
    paper_rows: List[np.ndarray],
    metal: np.ndarray,
    n_cal: int,
    rng: np.random.Generator,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrated |resid| over the complement of every calibration paper."""
    scores: List[np.ndarray] = []
    metals: List[np.ndarray] = []
    for rows in paper_rows:
        off, ev = draw_offset(resid, rows, n_cal, rng, kappa)
        scores.append(np.abs(resid[ev] - off))
        metals.append(metal[ev])
    return np.concatenate(scores), np.concatenate(metals)


def _by_metal(per_seed: Dict[str, object], block: str) -> Dict[str, float]:
    """Average each metal's repeat-mean coverage over seeds."""
    acc: Dict[str, List[float]] = {}
    for seed in per_seed:
        for m, v in per_seed[seed][block]["by_metal_coverage_mean"].items():
            acc.setdefault(m, []).append(float(v))
    return {m: round(float(np.mean(v)), 4) for m, v in sorted(acc.items())}


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
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric"
                    / "msrs_conformal_calibrated.json")
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

    papers, paper_idx = np.unique(doi, return_inverse=True)
    rows_by_paper: List[np.ndarray] = [np.where(paper_idx == p)[0] for p in range(len(papers))]
    folds = sorted(set(fold_id[fold_id >= 0].tolist()))
    print(f"[UP-33] rows={len(sub)} papers={len(papers)} folds={fold_info['n_folds']} "
          f"features={X.shape[1]}")

    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        q_lo = quantile_oof(X, codes, y, fold_id, QUANTILES[0], seed)
        q_hi = quantile_oof(X, codes, y, fold_id, QUANTILES[1], seed)
        resid = y - pred

        kappa_by_fold: Dict[int, float] = {}
        for k in folds:
            tr = np.where((fold_id != k) & (fold_id >= 0))[0]
            kappa_by_fold[k] = paper_variances(resid, paper_idx, tr)["kappa"]

        # Per-row accumulators.  Every arm that needs an interval endpoint
        # writes into one of these; rows that an arm does not evaluate stay
        # NaN and are dropped by coverage_report.
        n = len(y)
        half_uncal = np.full(n, np.nan)
        half_cal = np.full(n, np.nan)
        half_cal_m = np.full(n, np.nan)
        half_oracle = np.full(n, np.nan)
        yhat_cal = np.full(n, np.nan)
        yhat_oracle = np.full(n, np.nan)
        cqr_lo = np.full(n, np.nan)
        cqr_hi = np.full(n, np.nan)
        cqr_c_lo = np.full(n, np.nan)
        cqr_c_hi = np.full(n, np.nan)

        # Per-repeat diagnostics for the two sampling-dependent arms.
        rep_cov_cal: List[float] = []
        rep_wid_cal: List[float] = []
        rep_worst_metal: List[float] = []
        rep_cov_metal: List[float] = []
        rep_wid_metal: List[float] = []
        rep_worst_metal_m: List[float] = []
        # Per-metal coverage across draws.  A single draw can make a metal look
        # repaired when the mean over draws says otherwise, so both are kept.
        metal_cov_cal: Dict[str, List[float]] = {}
        metal_cov_metal: Dict[str, List[float]] = {}

        for k in folds:
            te = fold_id == k
            cal = (fold_id >= 0) & ~te
            cal_papers = [rows_by_paper[p] for p in range(len(papers))
                          if cal[rows_by_paper[p][0]]]
            te_papers = [rows_by_paper[p] for p in range(len(papers))
                         if te[rows_by_paper[p][0]]]

            # -- uncalibrated split conformal (the UP-27 ``global`` arm) --
            q_un = conformal_quantile(np.abs(resid[cal]), args.alpha)
            half_uncal[te] = q_un

            # -- uncalibrated CQR (the UP-27 ``cqr`` arm) --
            e_cal = np.maximum(q_lo[cal] - y[cal], y[cal] - q_hi[cal])
            e_un = conformal_quantile(e_cal, args.alpha)
            cqr_lo[te] = q_lo[te] - e_un
            cqr_hi[te] = q_hi[te] + e_un

            # -- calibrated arms, one draw per repeat --
            for rep in range(args.repeats):
                rng = np.random.default_rng(
                    (int(seed) * 100003 + int(k) * 1009 + rep) % (2 ** 32))
                kappa = kappa_by_fold[k]

                pool_s, pool_m = pool_scores(
                    resid, cal_papers, metal, N_CAL, rng, kappa)
                q_cal = conformal_quantile(pool_s, args.alpha)
                q_metal: Dict[str, float] = {}
                for m in np.unique(pool_m):
                    mask = pool_m == m
                    q_metal[m] = (
                        conformal_quantile(pool_s[mask], args.alpha)
                        if int(mask.sum()) >= MIN_METAL_CAL else q_cal
                    )

                # CQR correction on calibrated scores.  Same offset rows,
                # same disjointness rule: e is computed on the complement.
                e_parts: List[np.ndarray] = []
                for rows in cal_papers:
                    off, ev = draw_offset(resid, rows, N_CAL, rng, kappa)
                    e_parts.append(np.maximum(
                        q_lo[ev] - (y[ev] - off), (y[ev] - off) - q_hi[ev]))
                e_cal_cal = conformal_quantile(np.concatenate(e_parts), args.alpha)

                rep_y: List[np.ndarray] = []
                rep_yhat: List[np.ndarray] = []
                rep_half: List[np.ndarray] = []
                rep_half_m: List[np.ndarray] = []
                rep_metal: List[np.ndarray] = []
                for rows in te_papers:
                    off, ev = draw_offset(resid, rows, N_CAL, rng, kappa)
                    hm = np.array([q_metal.get(m, q_cal) for m in metal[ev]])
                    rep_y.append(y[ev])
                    rep_yhat.append(pred[ev] + off)
                    rep_half.append(np.full(len(ev), q_cal))
                    rep_half_m.append(hm)
                    rep_metal.append(metal[ev])
                    if rep == 0:
                        half_cal[ev] = q_cal
                        half_cal_m[ev] = hm
                        yhat_cal[ev] = pred[ev] + off
                        cqr_c_lo[ev] = q_lo[ev] + off - e_cal_cal
                        cqr_c_hi[ev] = q_hi[ev] + off + e_cal_cal
                        # Oracle floor, on the same evaluation rows.  A paper
                        # with one row has no internal spread to take a
                        # quantile of, so it falls back to the calibrated
                        # global half-width -- which is also the best any
                        # constant-shift method can do for it.
                        r_all = resid[rows]
                        h_or = (conformal_quantile(np.abs(r_all - r_all.mean()),
                                                   args.alpha)
                                if len(rows) >= 2 else q_cal)
                        half_oracle[ev] = h_or
                        yhat_oracle[ev] = pred[ev] + (
                            float(r_all.mean()) if len(rows) >= 2 else 0.0)

                yy = np.concatenate(rep_y)
                yh = np.concatenate(rep_yhat)
                hh = np.concatenate(rep_half)
                hhm = np.concatenate(rep_half_m)
                mm = np.concatenate(rep_metal)

                rep_cov_cal.append(float(np.mean(np.abs(yy - yh) <= hh)))
                rep_wid_cal.append(float(np.median(2 * hh)))
                groups = [float(np.mean(np.abs(yy[mm == m] - yh[mm == m]) <= hh[mm == m]))
                          for m in np.unique(mm) if int((mm == m).sum()) >= 30]
                rep_worst_metal.append(float(min(groups)) if groups else float("nan"))

                # The Mondrian arm needs the same repeat-level treatment: its
                # worst group is the number the group-coverage claim rests on,
                # and a single draw would overstate it.
                rep_cov_metal.append(float(np.mean(np.abs(yy - yh) <= hhm)))
                rep_wid_metal.append(float(np.median(2 * hhm)))
                groups_m = [float(np.mean(np.abs(yy[mm == m] - yh[mm == m]) <= hhm[mm == m]))
                            for m in np.unique(mm) if int((mm == m).sum()) >= 30]
                rep_worst_metal_m.append(
                    float(min(groups_m)) if groups_m else float("nan"))
                for m in np.unique(mm):
                    if int((mm == m).sum()) < 30:
                        continue
                    hit = np.abs(yy[mm == m] - yh[mm == m]) <= hh[mm == m]
                    metal_cov_cal.setdefault(str(m), []).append(float(hit.mean()))
                    hit_m = np.abs(yy[mm == m] - yh[mm == m]) <= hhm[mm == m]
                    metal_cov_metal.setdefault(str(m), []).append(float(hit_m.mean()))

        # ---- reports -------------------------------------------------
        arms: Dict[str, Dict[str, object]] = {}

        ok = np.isfinite(half_uncal)
        rep = coverage_report(y[ok], pred[ok], half_uncal[ok], metal[ok], "split_uncal")
        arms["split_uncal"] = rep

        ok_cal = np.isfinite(half_cal) & np.isfinite(yhat_cal)
        arms["split_uncal_matched"] = coverage_report(
            y[ok_cal], pred[ok_cal], half_uncal[ok_cal], metal[ok_cal],
            "split_uncal_matched")
        arms["split_cal"] = coverage_report(
            y[ok_cal], yhat_cal[ok_cal], half_cal[ok_cal], metal[ok_cal], "split_cal")
        arms["split_cal_metal"] = coverage_report(
            y[ok_cal], yhat_cal[ok_cal], half_cal_m[ok_cal], metal[ok_cal],
            "split_cal_metal")
        ok_or = np.isfinite(half_oracle) & ok_cal
        arms["split_oracle"] = coverage_report(
            y[ok_or], yhat_oracle[ok_or], half_oracle[ok_or], metal[ok_or],
            "split_oracle")

        ok_cq = np.isfinite(cqr_lo) & np.isfinite(cqr_hi)
        arms["cqr_uncal"] = interval_report(y[ok_cq], cqr_lo[ok_cq], cqr_hi[ok_cq],
                                            metal[ok_cq], args.alpha)
        ok_cc = np.isfinite(cqr_c_lo) & np.isfinite(cqr_c_hi)
        arms["cqr_cal"] = interval_report(y[ok_cc], cqr_c_lo[ok_cc], cqr_c_hi[ok_cc],
                                          metal[ok_cc], args.alpha)

        for name in ("split_uncal", "split_uncal_matched", "split_cal",
                     "split_cal_metal", "split_oracle"):
            groups = (arms[name].get("by_group") or {}).values()
            arms[name]["worst_group_coverage"] = (
                round(min(g["coverage"] for g in groups), 4) if groups else None)

        per_seed[str(seed)] = {
            "point_r2": round(float(1.0 - np.sum((y - pred) ** 2)
                                    / np.sum((y - y.mean()) ** 2)), 4),
            "kappa_mean": round(float(np.mean(list(kappa_by_fold.values()))), 4),
            "n_eval_rows_calibrated": int(ok_cal.sum()),
            "n_eval_rows_uncalibrated": int(ok.sum()),
            "arms": arms,
            "primary": {
                "arm": "split_cal",
                "coverage_mean": round(float(np.mean(rep_cov_cal)), 4),
                "median_width_mean": round(float(np.mean(rep_wid_cal)), 4),
                # Worst group = the metal with the lowest coverage *after*
                # averaging over draws.  Taking the per-draw minimum first and
                # averaging those would report the expected minimum of five
                # noisy estimates, which is below the minimum of their means by
                # construction and has nothing to do with a systematic deficit.
                "worst_metal_coverage": round(float(min(
                    np.mean(v) for v in metal_cov_cal.values())), 4),
                "worst_metal_coverage_per_draw_mean": round(
                    float(np.nanmean(rep_worst_metal)), 4),
                "worst_metal_coverage_per_draw_min": round(
                    float(np.nanmin(rep_worst_metal)), 4),
                "n_repeats": args.repeats,
                "definition": "mean over the offset draws; the arms table above "
                              "is a single draw (repeat 0) and exists for the "
                              "group tables",
                "by_metal_coverage_mean": {
                    k: round(float(np.mean(v)), 4)
                    for k, v in sorted(metal_cov_cal.items())},
            },
            "mondrian": {
                "arm": "split_cal_metal",
                "coverage_mean": round(float(np.mean(rep_cov_metal)), 4),
                "median_width_mean": round(float(np.mean(rep_wid_metal)), 4),
                "worst_metal_coverage": round(float(min(
                    np.mean(v) for v in metal_cov_metal.values())), 4),
                "worst_metal_coverage_per_draw_mean": round(
                    float(np.nanmean(rep_worst_metal_m)), 4),
                "worst_metal_coverage_per_draw_min": round(
                    float(np.nanmin(rep_worst_metal_m)), 4),
                "n_repeats": args.repeats,
                "by_metal_coverage_mean": {
                    k: round(float(np.mean(v)), 4)
                    for k, v in sorted(metal_cov_metal.items())},
            },
            "repeat_diagnostics": {
                "coverage_mean": round(float(np.mean(rep_cov_cal)), 4),
                "coverage_min": round(float(np.min(rep_cov_cal)), 4),
                "coverage_max": round(float(np.max(rep_cov_cal)), 4),
                "median_width_mean": round(float(np.mean(rep_wid_cal)), 4),
                "median_width_min": round(float(np.min(rep_wid_cal)), 4),
                "median_width_max": round(float(np.max(rep_wid_cal)), 4),
                "worst_metal_coverage_mean": round(
                    float(np.nanmean(rep_worst_metal)), 4),
                "worst_metal_coverage_min": round(
                    float(np.nanmin(rep_worst_metal)), 4),
                "repeats": args.repeats,
            },
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        a = per_seed[str(seed)]["arms"]
        print(f"[UP-33] seed={seed} split_uncal={a['split_uncal']['median_width']}"
              f"/{a['split_uncal']['coverage']} "
              f"split_cal={a['split_cal']['median_width']}/{a['split_cal']['coverage']} "
              f"cqr_uncal={a['cqr_uncal']['median_width']}/{a['cqr_uncal']['coverage']} "
              f"cqr_cal={a['cqr_cal']['median_width']}/{a['cqr_cal']['coverage']} "
              f"({per_seed[str(seed)]['elapsed_seconds']}s)")

    names = list(per_seed[str(SEEDS[0])]["arms"])
    agg: Dict[str, Dict[str, object]] = {}
    for name in names:
        vals = [per_seed[s]["arms"][name] for s in per_seed]
        agg[name] = {
            "coverage_mean": round(float(np.mean([v["coverage"] for v in vals])), 4),
            "coverage_min": round(float(np.min([v["coverage"] for v in vals])), 4),
            "median_width_mean": round(
                float(np.mean([v["median_width"] for v in vals])), 4),
            "median_width_min": round(
                float(np.min([v["median_width"] for v in vals])), 4),
            "mean_width_mean": round(
                float(np.mean([v["mean_width"] for v in vals])), 4),
            "worst_group_coverage_mean": round(float(np.mean(
                [v["worst_group_coverage"] for v in vals])), 4),
        }

    agg["primary"] = {
        "arm": "split_cal",
        "coverage_mean": round(float(np.mean(
            [per_seed[s]["primary"]["coverage_mean"] for s in per_seed])), 4),
        "median_width_mean": round(float(np.mean(
            [per_seed[s]["primary"]["median_width_mean"] for s in per_seed])), 4),
        "n_seeds": int(len(per_seed)),
        "n_repeats_per_seed": args.repeats,
        "by_metal_coverage_mean": _by_metal(per_seed, "primary"),
    }

    agg["mondrian"] = {
        "arm": "split_cal_metal",
        "coverage_mean": round(float(np.mean(
            [per_seed[s]["mondrian"]["coverage_mean"] for s in per_seed])), 4),
        "median_width_mean": round(float(np.mean(
            [per_seed[s]["mondrian"]["median_width_mean"] for s in per_seed])), 4),
        "n_seeds": int(len(per_seed)),
        "n_repeats_per_seed": args.repeats,
        "by_metal_coverage_mean": _by_metal(per_seed, "mondrian"),
    }

    for _blk in ("primary", "mondrian"):
        _bm = agg[_blk]["by_metal_coverage_mean"]
        agg[_blk]["worst_metal_coverage"] = round(float(min(_bm.values())), 4)
        agg[_blk]["worst_metal"] = min(_bm, key=lambda k: _bm[k])

    cal_w = agg["primary"]["median_width_mean"]
    un_w = agg["split_uncal_matched"]["median_width_mean"]
    narrow = round(float(1.0 - cal_w / un_w), 4)
    checks = {
        "coverage": {
            "value": agg["primary"]["coverage_mean"],
            "threshold": COVERAGE_GATE,
            "pass": bool(agg["primary"]["coverage_mean"] >= COVERAGE_GATE),
        },
        "median_width": {
            "value": cal_w,
            "threshold": WIDTH_GATE,
            "pass": bool(cal_w <= WIDTH_GATE),
        },
        "worst_metal_coverage": {
            "value": agg["primary"]["worst_metal_coverage"],
            "threshold": WORST_METAL_GATE,
            "pass": bool(
                agg["primary"]["worst_metal_coverage"] >= WORST_METAL_GATE),
        },
        "paired_narrowing": {
            "calibrated_width": cal_w,
            "uncalibrated_width_matched": un_w,
            "relative_narrowing": narrow,
            "required": PAIRED_NARROW,
            "pass": bool(narrow >= PAIRED_NARROW),
        },
        "reproduce": {
            "split_uncal": agg["split_uncal"]["median_width_mean"],
            "split_target": REPRO_SPLIT,
            "cqr_uncal": agg["cqr_uncal"]["median_width_mean"],
            "cqr_target": REPRO_CQR,
            "tolerance": REPRO_TOL,
            "pass": bool(
                abs(agg["split_uncal"]["median_width_mean"] - REPRO_SPLIT) <= REPRO_TOL
                and abs(agg["cqr_uncal"]["median_width_mean"] - REPRO_CQR) <= REPRO_TOL),
        },
    }
    kill = {
        "rule": "split_cal median width > 1.90 closes the interval line",
        "value": cal_w,
        "triggered": bool(cal_w > KILL_WIDTH),
    }
    gate_pass = all(v["pass"] for v in checks.values())
    repair = {
        "arm": "split_cal_metal",
        "coverage_mean": agg["mondrian"]["coverage_mean"],
        "median_width_mean": agg["mondrian"]["median_width_mean"],
        "worst_metal_coverage": agg["mondrian"]["worst_metal_coverage"],
        "note": "the pre-registered primary is the global calibrated arm; the "
                "metal-Mondrian variant is reported as the repair for its group "
                "deficit, not as a substitute for the gate result",
    }

    out = {
        "artifact": "msrs_conformal_calibrated",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 deployment interval (UP-33)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_conformal_shift selection: potency, uncensored, ligand "
                      "fragment and oxidation state present",
            "n_rows": int(len(y)),
            "arm": "ligand + state + coordination (the UP-27 arm, so the uncalibrated "
                   "widths are numerically comparable to that artifact)",
            "split": "scheme_folds('doi'), 5 GroupKFold folds on DOI",
            "seeds": list(SEEDS[: args.seeds]),
            "alpha": args.alpha,
            "n_cal": N_CAL,
            "repeats": args.repeats,
            "primary_arm": "split_cal",
            "estimator": "offset = n/(n+kappa) * mean(y - y_hat | n_cal rows of the "
                         "paper); conformal half-width from |y - (y_hat + offset)|",
            "kappa_definition": "sigma2_noise / tau2_paper, method-of-moments on the "
                                "outer fold's training rows",
            "disjointness": "offset rows are removed from the scoring rows on both "
                            "sides of the split (calibration papers and test papers)",
            "oracle_bound": "per test paper, the alpha-quantile of |resid - mean_p| "
                            "over all of its rows: the floor for any constant shift",
            "gate_frozen_at": GATE_FROZEN_AT,
            "gate_amended_at": GATE_AMENDED_AT,
            "amendment": "the frozen reproduce target named 1.9871, the CQR arm's "
                         "value, while the frozen arm list is split-conformal; both "
                         "families are now reproduced",
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "amended_at": GATE_AMENDED_AT,
                 "checks": checks, "gate_pass": bool(gate_pass), "kill": kill,
                 "repair": repair},
        "caveats": [
            "The calibration rows come from the same paper as the evaluation rows and "
            "are then excluded from scoring. That is the deployment regime being "
            "modelled: a plate arrives with its first compounds measured.",
            "kappa is estimated on each outer fold's training rows, so a test paper's "
            "own residual scale never enters its shrinkage factor.",
            "Calibration removes a level, not a slope. Papers whose error is mostly "
            "slope keep it, and the oracle arm measures how much of the width that "
            "leaves behind.",
            "A CQR interval shifted by a constant has the same width, so cqr_cal can "
            "only narrow through the conformal correction E. It is the contrast that "
            "shows the split arm's gain is not a generic property of intervals.",
            "The oracle arm uses every row of a paper to locate that paper's mean, "
            "including rows a real method would spend on calibration. It is a floor, "
            "not a deployable arm.",
            "The metal-Mondrian arm falls back to the global quantile for any metal "
            "with fewer than 30 pool rows, so its worst-group number is a mixture of "
            "Mondrian and pooled behaviour.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for name in names:
        print(f"[UP-33] {name:22s} cov={agg[name]['coverage_mean']:.4f} "
              f"median_w={agg[name]['median_width_mean']:.4f} "
              f"worst={agg[name]['worst_group_coverage_mean']}")
    print(f"[UP-33] narrowing={narrow:.4f} gate_pass={gate_pass} "
          f"kill={kill['triggered']} elapsed={round(time.time() - t0, 1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
