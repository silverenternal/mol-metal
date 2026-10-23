#!/usr/bin/env python3
"""UP-39 — per-paper slope: is the constant offset the ceiling?

=====================================================================
Why this exists
=====================================================================
UP-26 corrected each paper by a constant and reached 0.6839 at n_cal=10,
against a constant-offset oracle of 0.7353.  The gap is 0.0514 and nobody
has asked whether it is estimation error or the model class.  ComBat has
two parameters per batch, a location and a scale; UP-26 only ever used the
location.

This script measures three ceilings on the same held-out rows, then builds
the estimator that would be needed to reach the third:

  oracle_constant   subtract each paper's mean residual (reproduces 0.7353)
  oracle_affine     fit y ~ f inside each paper, in-sample
  oracle_isotonic   fit a monotone y ~ f inside each paper, in-sample

All three are in-sample per paper, which is the same convention UP-26's
constant oracle already used, so they are comparable to each other.  They
are ceilings, not deployable estimators: they see every row of the paper.

=====================================================================
The estimator
=====================================================================
``eb_slope`` generalises UP-26 along the scale parameter only:

    pred_final = pred + off_eb + (b_post - 1) * (pred - mean_cal(pred))

where ``off_eb`` is UP-26's offset and ``b_post`` is the per-paper slope
shrunk toward 1:

    b_ols  = sum(f_c * y_c) / sum(f_c^2)          (centred on the cal rows)
    var_b  = sigma2 / sum(f_c^2)
    b_post = 1 + (b_ols - 1) * tau_b2 / (tau_b2 + var_b)

``tau_b2`` is the between-paper slope variance, estimated by method of
moments on the training folds exactly as ``tau2_paper`` is for the offset.
At ``b_post = 1`` the estimator is bit-identical to ``eb_offset``, so the
comparison isolates the scale term.

=====================================================================
Gate (todo.json, frozen 2026-10-23T05:30+08:00)
=====================================================================
headroom      affine oracle >= 0.75        (constant oracle 0.7353)
estimator     eb_slope at n_cal=10 >= 0.70 (current 0.6839)
no_collapse   eb_slope at n_cal=1  >= 0.55 (the unshrunk affine arm gives 0.4258)
reproduce     oracle_constant 0.7353 and eb 0.6839, both to 5e-5
kill          affine oracle < 0.72 => the constant offset is the ceiling

Amendment (2026-10-23T05:40+08:00, before any run of this script): the
no_collapse check was added after reading the existing UP-26 artifact,
where the unshrunk affine arm scores 0.4258 at n_cal=1 and 0.5210 at
n_cal=2.  The gate numbers above are unchanged; this only states that a
slope that wins at n=10 by collapsing at n=1 does not count.

Usage
=====
    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_paper_scale.py
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
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_paper_calibration import (  # noqa: E402
    SEEDS,
    _pool_repeats,
    _r2,
    assemble,
    eb_offset,
    nested_maxmin,
    nested_random,
    paper_variances,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_GRID = (0, 1, 2, 5, 10, 50)
REPEATS = 20
ARMS = ("eb_random", "affine_unshrunk", "eb_slope")
PRIMARY = "eb_slope"
GATE_FROZEN_AT = "2026-10-23T05:30+08:00"
ORACLE_AFFINE_GATE = 0.75
ORACLE_AFFINE_KILL = 0.72
EST_N10_GATE = 0.70
NO_COLLAPSE_N1 = 0.55
REPRO_TOL = 5e-5
REPRO = {"oracle_constant": 0.7353, "eb_n10": 0.6839, "eb_n0": 0.3674}
SLOPE_CLIP = (0.0, 2.0)
MIN_TRAIN_PAPER_ROWS = 5


def slope_variances(
    resid: np.ndarray,
    pred: np.ndarray,
    paper_idx: np.ndarray,
    rows: np.ndarray,
) -> Dict[str, float]:
    """Between-paper slope variance, with the sampling term subtracted.

    Same method of moments as ``paper_variances``, one level up: per-paper
    OLS slopes are noisy, so ``var(slopes)`` overstates ``tau_b2`` by their
    mean sampling variance.
    """
    idx = paper_idx[rows]
    uniq, inv = np.unique(idx, return_inverse=True)
    n_p = np.bincount(inv).astype(float)
    r = resid[rows]
    f = pred[rows]
    s1r = np.bincount(inv, weights=r)
    s1f = np.bincount(inv, weights=f)
    s2r = np.bincount(inv, weights=r * r)
    s2f = np.bincount(inv, weights=f * f)
    sfr = np.bincount(inv, weights=f * r)
    mean_r = np.divide(s1r, n_p, out=np.zeros_like(s1r), where=n_p > 0)
    mean_f = np.divide(s1f, n_p, out=np.zeros_like(s1f), where=n_p > 0)
    cov = sfr - n_p * mean_f * mean_r
    var_f = s2f - n_p * mean_f * mean_f
    ok = (n_p >= MIN_TRAIN_PAPER_ROWS) & (var_f > 1e-9)
    if int(ok.sum()) < 4:
        return {"tau_b2": 1e-6, "n_papers": int(ok.sum())}
    b_all = np.zeros_like(cov)
    b_all[ok] = cov[ok] / var_f[ok]
    b = b_all[ok]
    # residual variance after the per-paper line, pooled
    ss = (s2r - n_p * mean_r * mean_r) - b_all * cov
    ss = np.where(ok, ss, 0.0)
    dof = np.maximum(n_p - 2.0, 1.0)
    sigma2_b = float(np.sum(ss) / np.sum(np.where(ok, dof, 0.0)))
    sampling = sigma2_b / var_f[ok]
    tau2 = float(np.var(b, ddof=1) - np.mean(sampling))
    return {"tau_b2": max(tau2, 1e-6), "n_papers": int(ok.sum()),
            "mean_sampling_var": float(np.mean(sampling))}


def slope_offset(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    sigma2: float,
    tau_b2: float,
    kappa: float,
) -> Tuple[float, float]:
    """Return (offset, slope) for one paper, both from the calibration rows."""
    c = len(y_cal)
    if c <= 0:
        return 0.0, 1.0
    off = eb_offset(y_cal - pred_cal, kappa)
    if c < 3:
        return off, 1.0
    fc = pred_cal - pred_cal.mean()
    yc = y_cal - y_cal.mean()
    denom = float(np.sum(fc * fc))
    if denom <= 1e-9:
        return off, 1.0
    b = float(np.sum(fc * yc) / denom)
    b = float(np.clip(b, *SLOPE_CLIP))
    var_b = float(sigma2 / denom)
    b_post = 1.0 + (b - 1.0) * tau_b2 / (tau_b2 + var_b)
    return off, float(np.clip(b_post, *SLOPE_CLIP))


def oracle_block(
    y: np.ndarray,
    pred: np.ndarray,
    paper_idx: np.ndarray,
    fold_id: np.ndarray,
    rows_by_paper: List[np.ndarray],
) -> Dict[str, object]:
    """Three in-sample per-paper ceilings on the n_cal=0 evaluation rows."""
    test_papers = sorted(set(paper_idx[fold_id >= 0].tolist()))
    papers = [p for p in test_papers if len(rows_by_paper[p]) >= 2]
    y_parts, c_parts, a_parts, i_parts = [], [], [], []
    n_affine_fail = 0
    for p in papers:
        rows = rows_by_paper[p]
        yy, ff = y[rows], pred[rows]
        y_parts.append(yy)
        c_parts.append(ff + float(np.mean(yy - ff)))
        if len(rows) >= 3 and float(np.var(ff, ddof=1)) > 1e-9:
            b, a = np.polyfit(ff, yy, 1)
            a_parts.append(a + b * ff)
            iso = IsotonicRegression(out_of_bounds="clip").fit(ff, yy)
            i_parts.append(iso.predict(ff))
        else:
            n_affine_fail += 1
            a_parts.append(ff + float(np.mean(yy - ff)))
            i_parts.append(ff + float(np.mean(yy - ff)))
    yy = np.concatenate(y_parts)
    return {
        "oracle_constant": round(float(_r2(yy, np.concatenate(c_parts))), 4),
        "oracle_affine": round(float(_r2(yy, np.concatenate(a_parts))), 4),
        "oracle_isotonic": round(float(_r2(yy, np.concatenate(i_parts))), 4),
        "n_eval_rows": int(len(yy)),
        "n_eval_papers": int(len(papers)),
        "n_papers_too_small_for_affine": int(n_affine_fail),
    }


def sweep_seed(
    y: np.ndarray,
    pred: np.ndarray,
    paper_idx: np.ndarray,
    fold_id: np.ndarray,
    rows_by_paper: List[np.ndarray],
    kappa_by_fold: Dict[int, float],
    sigma2_by_fold: Dict[int, float],
    taub2_by_fold: Dict[int, float],
    repeats: int,
    seed: int,
    sim_by_paper: Dict[int, np.ndarray],
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    test_papers = sorted(set(paper_idx[fold_id >= 0].tolist()))
    papers = [p for p in test_papers if len(rows_by_paper[p]) >= 2]
    resid = y - pred

    # UP-26 draws TWO permutations per (repeat, paper): the random order and a
    # maxmin order.  The second draw advances the generator even when the arm
    # that consumes it is absent here, so it is reproduced unconditionally --
    # otherwise `eb_random` would silently stop being bit-identical to UP-26.
    orders: Dict[Tuple[int, int], np.ndarray] = {}
    for rep in range(repeats):
        for p in papers:
            n = len(rows_by_paper[p])
            orders[(rep, p)] = nested_random(n, rng)
            sim = sim_by_paper.get(p)
            nested_maxmin(sim, rng) if sim is not None else nested_random(n, rng)

    out: Dict[str, object] = {}
    for n_cal in N_GRID:
        per_rep: Dict[str, List[np.ndarray]] = {a: [] for a in ARMS}
        per_rep_y: List[np.ndarray] = []
        per_rep_p0: List[np.ndarray] = []
        for rep in range(repeats):
            cur: Dict[str, List[np.ndarray]] = {a: [] for a in per_rep}
            ybuf: List[np.ndarray] = []
            p0buf: List[np.ndarray] = []
            for p in papers:
                rows = rows_by_paper[p]
                c = min(n_cal, len(rows) - 1)
                perm = orders[(rep, p)]
                ev = np.ones(len(rows), dtype=bool)
                ev[perm[:c]] = False
                k = int(fold_id[rows[0]])
                kap = float(kappa_by_fold[k])
                sig = float(sigma2_by_fold[k])
                taub2 = float(taub2_by_fold[k])
                r = resid[rows]

                ybuf.append(y[rows][ev])
                p0buf.append(pred[rows][ev])
                cur["eb_random"].append(pred[rows][ev] + eb_offset(r[perm[:c]], kap))

                yc, pc = y[rows][perm[:c]], pred[rows][perm[:c]]
                if c >= 2 and float(np.var(pc, ddof=1)) > 1e-2 * float(np.var(pred)):
                    b = float(np.cov(pc, yc, ddof=1)[0, 1] / np.var(pc, ddof=1))
                    b = float(np.clip(b, *SLOPE_CLIP))
                    a = float(np.mean(yc) - b * np.mean(pc))
                elif c >= 1:
                    a, b = float(np.mean(yc - pc)), 1.0
                else:
                    a, b = 0.0, 1.0
                cur["affine_unshrunk"].append(a + b * pred[rows][ev])

                off, bs = slope_offset(yc, pc, sig, taub2, kap)
                centre = float(np.mean(pc)) if c else 0.0
                cur["eb_slope"].append(pred[rows][ev] + off + (bs - 1.0) * (pred[rows][ev] - centre))

            for a in per_rep:
                per_rep[a].append(np.concatenate(cur[a]))
            per_rep_y.append(np.concatenate(ybuf))
            per_rep_p0.append(np.concatenate(p0buf))

        yy = np.concatenate(per_rep_y)
        pp0 = np.concatenate(per_rep_p0)
        row: Dict[str, object] = {}
        for arm, chunks in per_rep.items():
            pp = np.concatenate(chunks)
            r2_reps = np.array([_r2(a, b) for a, b in zip(per_rep_y, chunks)])
            row[arm] = {
                "r2": _r2(yy, pp),
                "mae": round(float(np.mean(np.abs(yy - pp))), 4),
                "r2_matched_zero": _r2(yy, pp0),
                "delta_vs_matched_zero": round(_r2(yy, pp) - _r2(yy, pp0), 4),
                "n_eval_rows_per_repeat": int(len(yy) // repeats),
                "r2_by_repeat": {
                    "mean": round(float(np.mean(r2_reps)), 4),
                    "sd": round(float(np.std(r2_reps, ddof=1)), 4)
                          if len(r2_reps) > 1 else 0.0,
                    "min": round(float(np.min(r2_reps)), 4),
                    "max": round(float(np.max(r2_reps)), 4),
                    "n_repeats": int(len(r2_reps)),
                    "values": [round(float(x), 4) for x in r2_reps],
                },
            }
        out[str(n_cal)] = row
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_paper_scale.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    data = assemble(args.long, args.metal_desc)
    y = data["y"]
    paper_idx = data["paper_idx"]
    fold_id = data["fold_id"]
    rows_by_paper = data["rows_by_paper"]
    sim_by_paper = data["sim_by_paper"]
    X_model = data["X_model"]
    empty = data["empty"]
    codes = data["codes"]

    oracles: Dict[str, object] = {}
    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X_model, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        resid = y - pred
        oracles[str(seed)] = oracle_block(y, pred, paper_idx, fold_id, rows_by_paper)

        kappa_by_fold: Dict[int, float] = {}
        sigma2_by_fold: Dict[int, float] = {}
        taub2_by_fold: Dict[int, float] = {}
        for k in sorted(set(fold_id[fold_id >= 0].tolist())):
            tr = np.where((fold_id != k) & (fold_id >= 0))[0]
            v = paper_variances(resid, paper_idx, tr)
            kappa_by_fold[k] = float(v["kappa"])
            sigma2_by_fold[k] = float(v["sigma2_noise"])
            taub2_by_fold[k] = float(slope_variances(resid, pred, paper_idx, tr)["tau_b2"])

        per_seed[str(seed)] = sweep_seed(
            y, pred, paper_idx, fold_id, rows_by_paper,
            kappa_by_fold, sigma2_by_fold, taub2_by_fold, args.repeats, seed,
            sim_by_paper,
        )
        line = " ".join(f"{c}:{per_seed[str(seed)][str(c)][PRIMARY]['r2']:.4f}"
                        for c in N_GRID)
        print(f"[UP-39] seed={seed} {line} ({round(time.time() - ts, 1)}s)")

    agg: Dict[str, Dict[str, object]] = {}
    for n_cal in N_GRID:
        agg[str(n_cal)] = {}
        for arm in ARMS:
            vals = [per_seed[s][str(n_cal)][arm] for s in per_seed]
            agg[str(n_cal)][arm] = {
                "r2_mean": round(float(np.mean([v["r2"] for v in vals])), 4),
                "r2_min": round(float(np.min([v["r2"] for v in vals])), 4),
                "r2_max": round(float(np.max([v["r2"] for v in vals])), 4),
                "mae_mean": round(float(np.mean([v["mae"] for v in vals])), 4),
                "r2_matched_zero_mean": round(
                    float(np.mean([v["r2_matched_zero"] for v in vals])), 4),
                "delta_vs_matched_zero_mean": round(
                    float(np.mean([v["delta_vs_matched_zero"] for v in vals])), 4),
                "n_eval_rows_mean": int(np.mean([v["n_eval_rows_per_repeat"] for v in vals])),
                "r2_by_repeat": _pool_repeats(vals),
            }

    oracle_summary = {
        k: round(float(np.mean([oracles[s][k] for s in oracles])), 4)
        for k in ("oracle_constant", "oracle_affine", "oracle_isotonic")
    }
    oracle_summary["n_eval_rows"] = int(oracles[list(oracles)[0]]["n_eval_rows"])
    oracle_summary["n_eval_papers"] = int(oracles[list(oracles)[0]]["n_eval_papers"])
    oracle_summary["n_papers_too_small_for_affine"] = int(
        oracles[list(oracles)[0]]["n_papers_too_small_for_affine"])
    oracle_summary["definition"] = (
        "per paper, in-sample: constant = mean residual; affine = OLS y~f; "
        "isotonic = monotone y~f. Same evaluation rows as n_cal=0."
    )

    n1 = agg["1"][PRIMARY]["r2_mean"]
    n10 = agg["10"][PRIMARY]["r2_mean"]
    eb10 = agg["10"]["eb_random"]["r2_mean"]
    eb0 = agg["0"]["eb_random"]["r2_mean"]
    checks = {
        "headroom": {
            "value": oracle_summary["oracle_affine"],
            "threshold": ORACLE_AFFINE_GATE,
            "constant_oracle": oracle_summary["oracle_constant"],
            "pass": bool(oracle_summary["oracle_affine"] >= ORACLE_AFFINE_GATE),
        },
        "estimator_n10": {
            "value": n10,
            "threshold": EST_N10_GATE,
            "pass": bool(n10 >= EST_N10_GATE),
        },
        "no_collapse_n1": {
            "value": n1,
            "threshold": NO_COLLAPSE_N1,
            "pass": bool(n1 >= NO_COLLAPSE_N1),
        },
        "reproduce": {
            "checks": {
                "oracle_constant": {
                    "value": oracle_summary["oracle_constant"],
                    "target": REPRO["oracle_constant"],
                    "pass": bool(abs(oracle_summary["oracle_constant"]
                                     - REPRO["oracle_constant"]) <= REPRO_TOL),
                },
                "eb_n10": {"value": eb10, "target": REPRO["eb_n10"],
                           "pass": bool(abs(eb10 - REPRO["eb_n10"]) <= REPRO_TOL)},
                "eb_n0": {"value": eb0, "target": REPRO["eb_n0"],
                          "pass": bool(abs(eb0 - REPRO["eb_n0"]) <= REPRO_TOL)},
            },
            "pass": True,
        },
    }
    checks["reproduce"]["pass"] = all(v["pass"] for v in checks["reproduce"]["checks"].values())
    gate_pass = all(v["pass"] for v in checks.values())
    kill = {
        "rule": "affine oracle < 0.72 => the constant offset is the ceiling",
        "value": oracle_summary["oracle_affine"],
        "triggered": bool(oracle_summary["oracle_affine"] < ORACLE_AFFINE_KILL),
    }

    out = {
        "artifact": "msrs_paper_scale",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 deployment calibration (UP-39)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_paper_calibration.assemble (same 21,404-row cohort as UP-26)",
            "split": "scheme_folds('doi'), 5 GroupKFold folds on DOI",
            "seeds": list(SEEDS[: args.seeds]),
            "repeats": args.repeats,
            "n_cal_grid": list(N_GRID),
            "primary_arm": PRIMARY,
            "estimator": "pred + eb_offset + (b_post-1)*(pred - mean_cal(pred)); "
                         "b_post shrinks the within-paper OLS slope toward 1 with tau_b2",
            "tau_b2": "between-paper slope variance, method of moments on training folds, "
                      "papers with >=5 rows",
            "oracles": "in-sample per paper, three model classes, same rows as n_cal=0",
            "gate_frozen_at": GATE_FROZEN_AT,
            "amendment": "2026-10-23T05:40+08:00 no_collapse_n1 added before any run, "
                         "because the existing UP-26 affine arm scores 0.4258 at n_cal=1",
        },
        "oracle_by_seed": oracles,
        "oracle": oracle_summary,
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "All three oracles are in-sample per paper: they see every row of the paper "
            "they are scored on. They bound what a per-paper model class can do, not what "
            "a 10-row estimator can do.",
            "The isotonic oracle has as many parameters as the paper has rows, so it is the "
            "loosest of the three and is reported for completeness only.",
            "tau_b2 is estimated from papers with at least 5 training rows. Papers below "
            "that threshold get the pooled value, which is the same choice UP-26 makes "
            "for kappa.",
            "The slope is clipped to [0, 2], matching UP-26's affine arm; without the clip "
            "two calibration rows with near-identical predictions produce an unbounded b.",
            "n_cal<3 returns b_post=1, so the arm is exactly the EB arm there. The "
            "no_collapse check is therefore a statement about the estimator, not the slope.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for n_cal in N_GRID:
        row = agg[str(n_cal)]
        print(f"[UP-39] n_cal={n_cal:2d} " + " ".join(
            f"{a}={row[a]['r2_mean']}" for a in ARMS))
    print(f"[UP-39] oracles constant={oracle_summary['oracle_constant']} "
          f"affine={oracle_summary['oracle_affine']} "
          f"isotonic={oracle_summary['oracle_isotonic']}")
    print(f"[UP-39] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time() - t0, 1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
