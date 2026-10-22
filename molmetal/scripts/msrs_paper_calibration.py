#!/usr/bin/env python3
"""UP-26 — every paper gets its own offset; a handful of measured rows can estimate it.

=====================================================================
Why this exists
=====================================================================
UP-21 measured the paper offset directly: its standard deviation across
held-out papers is 0.24 log10, its mean absolute value 0.2255, and a
ridge on 23 paper-level design features recovers it at R^2 = -0.0259.
The offset is real, it is large relative to the residual, and it is not
a function of what the paper *is*.

What UP-21 did not ask is whether the offset has to be *predicted* at
all.  In deployment a paper arrives with a plate that has already been
run: the first few compounds are measured, the rest are the ones the
model is asked about.  Every round-1..3 evaluation scored the model on
papers with **zero** measured rows, which is a strictly harder problem
than the one the assay pipeline actually poses.

So the estimator here is the empirical-Bayes posterior mean of the
paper offset given n_cal measured rows from that same paper:

    sigma2_noise  within-paper residual variance   (training folds)
    tau2_paper    between-paper variance of offsets (training folds)
    kappa         = sigma2_noise / tau2_paper
    offset_hat    = n_cal / (n_cal + kappa) * mean(y - y_hat | cal)

``kappa`` is estimated once per outer fold on that fold's training rows
and never from the test paper.  The shrinkage is the exact posterior
mean under a normal-normal hierarchy, not a tuned knob.

=====================================================================
Two selection arms, and why they are not the same question
=====================================================================
``random``  uniform sample of the paper's own rows.  Unbiased for the
            paper mean residual, which is the target.  Declared primary.
``maxmin``  greedy farthest-point on Morgan/Tanimoto, so the measured
            rows span the paper's chemistry.  Better for a slope, worse
            for a mean; reported as the design arm.

Both arms draw a *nested* sequence per (paper, repeat): the n_cal=2 set
is a prefix of the n_cal=5 set, so the sweep measures sample size and
not resampling noise.

=====================================================================
What is scored
=====================================================================
For n_cal > 0 the calibration rows are removed from the evaluation, so
the reported R^2 is on rows the correction has not seen.  Because that
row set shrinks with n_cal, every arm also reports ``r2_matched_zero``:
the uncorrected model on the *same* rows.  The paired difference is the
number that means something; the absolute R^2 is the one the gate was
written against.

n_cal = 0 leaves the row set at all test rows, so it must reproduce the
UP-21 base of 0.3670 +- 0.005.  That is the protocol-drift check.

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
doi_r2_n10     >= 0.45 (5 seeds x 20 repeats, primary arm eb_random)
reproduce_n0   |R^2 - 0.3670| <= 0.005
monotone       >= 5 of the 6 transitions in {0,1,2,5,10,20,50} non-decreasing
report         both selection arms and the estimated kappa
kill           R^2 at n_cal=10 < 0.40 closes the per-paper calibration line

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_paper_calibration.py
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

from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_feature_expansion import (  # noqa: E402
    build_arm,
    select_cohort,
)
from molmetal.scripts.msrs_generalization import (  # noqa: E402
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
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    morgan_matrix,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
N_CAL_GRID = (0, 1, 2, 5, 10, 20, 50)
N_REPEATS = 20
FP_BITS = 1024
FP_RADIUS = 2
GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
R2_GATE = 0.45
R2_KILL = 0.40
REPRO_TARGET = 0.3670
REPRO_TOL = 0.005
PRIMARY_ARM = "eb_random"
MIN_MONOTONE_STEPS = 5


# ---------------------------------------------------------------------------
# Variance components
# ---------------------------------------------------------------------------
def paper_variances(
    resid: np.ndarray, paper_idx: np.ndarray, rows: np.ndarray
) -> Dict[str, float]:
    """Method-of-moments split of the residual variance into noise and paper.

    ``var(paper means)`` overstates the between-paper variance by the
    sampling variance of each mean, so that term is subtracted back out.
    The floor on ``tau2`` keeps ``kappa`` finite on a cohort whose papers
    are nearly identical.
    """
    idx = paper_idx[rows]
    r = resid[rows]
    uniq, inv = np.unique(idx, return_inverse=True)
    n_p = np.bincount(inv).astype(float)
    s1 = np.bincount(inv, weights=r)
    s2 = np.bincount(inv, weights=r * r)
    mean = np.divide(s1, n_p, out=np.zeros_like(s1), where=n_p > 0)
    ss_within = s2 - n_p * mean * mean
    dof = np.maximum(n_p - 1.0, 0.0)
    sigma2 = float(ss_within.sum() / dof.sum()) if dof.sum() > 0 else float("nan")

    big = n_p >= 2
    between = float(np.var(mean[big], ddof=1)) if int(big.sum()) > 1 else 0.0
    correction = float(np.mean(sigma2 / n_p[big])) if int(big.sum()) > 0 else 0.0
    tau2 = max(between - correction, 1e-6)
    return {
        "sigma2_noise": sigma2,
        "tau2_paper_raw": between,
        "tau2_paper": tau2,
        "kappa": float(sigma2 / tau2),
        "n_papers": int(len(uniq)),
        "n_rows": int(len(rows)),
    }


def eb_offset(resid_cal: np.ndarray, kappa: float) -> float:
    """Posterior mean of the paper offset under a normal-normal hierarchy."""
    n = float(len(resid_cal))
    if n <= 0:
        return 0.0
    return float(n / (n + kappa) * np.mean(resid_cal))


def js_offsets(
    means: np.ndarray, counts: np.ndarray, sigma2: float
) -> np.ndarray:
    """Positive-part James-Stein shrinkage of a batch of paper means.

    Different mechanism from ``eb_offset``: it borrows strength across the
    *other new papers in the same batch*, so it only exists when several
    papers arrive together.  Reported as a variant, never as the headline.
    """
    p = len(means)
    if p < 4:
        return means.copy()
    ss = float(np.sum(counts * means * means))
    if ss <= 0:
        return np.zeros_like(means)
    shrink = 1.0 - (p - 3.0) * sigma2 / ss
    return np.maximum(shrink, 0.0) * means


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------
def similarity_block(X: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    """Tanimoto similarity inside one paper (binary fingerprints)."""
    B = X[rows].astype(np.float64)
    inter = np.asarray((B @ B.T).todense())
    cnt = np.asarray(B.sum(axis=1)).ravel()
    union = cnt[:, None] + cnt[None, :] - inter
    return inter / np.maximum(union, 1.0)


def nested_random(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def nested_maxmin(sim: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point order; the k-th prefix is the k-point design."""
    n = sim.shape[0]
    order = np.empty(n, dtype=int)
    first = int(rng.integers(n))
    order[0] = first
    best = 1.0 - sim[first]
    best[first] = -np.inf
    for i in range(1, n):
        j = int(np.argmax(best))
        order[i] = j
        best = np.minimum(best, 1.0 - sim[j])
        best[order[: i + 1]] = -np.inf
    return order


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return score(y, p)["r2"] if len(y) > 2 else float("nan")


def sweep_ncal(
    y: np.ndarray,
    pred: np.ndarray,
    paper_idx: np.ndarray,
    fold_id: np.ndarray,
    rows_by_paper: List[np.ndarray],
    sim_by_paper: Dict[int, np.ndarray],
    kappa_by_fold: Dict[int, float],
    sigma2_by_fold: Dict[int, float],
    repeats: int,
    seed: int,
) -> Dict[str, object]:
    """One seed: every (n_cal, repeat) pair for both selection arms."""
    rng = np.random.default_rng(seed)
    test_papers = sorted(set(paper_idx[fold_id >= 0].tolist()))
    papers = [p for p in test_papers if len(rows_by_paper[p]) >= 2]
    resid = y - pred

    # Nested orders are drawn once per (repeat, paper) and reused for every
    # n_cal, so the sweep changes sample size only.
    orders: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for rep in range(repeats):
        for p in papers:
            n = len(rows_by_paper[p])
            sim = sim_by_paper.get(p)
            orders[(rep, p)] = (
                nested_random(n, rng),
                nested_maxmin(sim, rng) if sim is not None else nested_random(n, rng),
            )

    out: Dict[str, object] = {}
    for n_cal in N_CAL_GRID:
        buf: Dict[str, List[np.ndarray]] = {
            a: [] for a in ("eb_random", "eb_maxmin", "js_batch", "affine")
        }
        ybuf: List[np.ndarray] = []
        p0buf: List[np.ndarray] = []

        for rep in range(repeats):
            means: Dict[int, Tuple[float, int, int]] = {}
            for p in papers:
                rows = rows_by_paper[p]
                c = min(n_cal, len(rows) - 1)
                perm, mm = orders[(rep, p)]
                ev = np.ones(len(rows), dtype=bool)
                ev[perm[:c]] = False
                k = int(fold_id[rows[0]])
                kap = kappa_by_fold[k]
                r = resid[rows]

                ybuf.append(y[rows][ev])
                p0buf.append(pred[rows][ev])
                buf["eb_random"].append(pred[rows][ev] + eb_offset(r[perm[:c]], kap))
                buf["eb_maxmin"].append(pred[rows][ev] + eb_offset(r[mm[:c]], kap))

                # affine: level and slope, both from the same calibration rows.
                # The slope is clipped because two points with near-identical
                # predictions give an unbounded b; without the clip the arm
                # produced R^2 = -4162 at n_cal=2.
                yc, pc = y[rows][perm[:c]], pred[rows][perm[:c]]
                if c >= 2 and np.var(pc, ddof=1) > 1e-2 * float(np.var(pred)):
                    b = float(np.cov(pc, yc, ddof=1)[0, 1] / np.var(pc, ddof=1))
                    b = float(np.clip(b, 0.0, 2.0))
                    a = float(np.mean(yc) - b * np.mean(pc))
                elif c >= 1:
                    a, b = float(np.mean(yc - pc)), 1.0
                else:
                    a, b = 0.0, 1.0
                buf["affine"].append(a + b * pred[rows][ev])

                means[p] = (float(np.mean(r[perm[:c]])) if c else 0.0, c, k)

            # James-Stein borrows across the whole batch of new papers, so it
            # needs every paper's mean before any paper can be corrected.
            ps = sorted(means)
            off = js_offsets(
                np.array([means[q][0] for q in ps]),
                np.array([means[q][1] for q in ps], dtype=float),
                sigma2_by_fold[means[ps[0]][2]],
            )
            mapping = dict(zip(ps, off.tolist()))
            for p in ps:
                rows = rows_by_paper[p]
                perm, _ = orders[(rep, p)]
                c = min(n_cal, len(rows) - 1)
                ev = np.ones(len(rows), dtype=bool)
                ev[perm[:c]] = False
                buf["js_batch"].append(pred[rows][ev] + mapping[p])

        yy = np.concatenate(ybuf)
        pp0 = np.concatenate(p0buf)
        ncal_out: Dict[str, object] = {}
        for arm, chunks in buf.items():
            pp = np.concatenate(chunks)
            ncal_out[arm] = {
                "r2": _r2(yy, pp),
                "mae": round(float(np.mean(np.abs(yy - pp))), 4),
                "r2_matched_zero": _r2(yy, pp0),
                "delta_vs_matched_zero": round(_r2(yy, pp) - _r2(yy, pp0), 4),
                "n_eval_rows_per_repeat": int(len(yy) // repeats),
                "n_eval_papers": int(len(papers)),
            }
        out[str(n_cal)] = ncal_out
    return out


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
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_paper_calibration.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    frame = select_cohort(args.long)
    y = frame["value_log10"].to_numpy(dtype=float)
    doi = frame["doi"].astype(str).to_numpy()

    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(frame["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES)
    codes = native_codes(
        frame["metal"].astype(str).tolist(),
        frame["cell_line_key"].astype(str).tolist(),
        vocab, mode="full",
    )
    base = build(frame, LIGAND_ONLY)
    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    state_coord = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    base_ligand = sparse.hstack([base.X, state_coord], format="csr")
    X_model = build_arm(frame, "maccs_rdkit", base_ligand, state_coord)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)

    fold_id, _, fold_info = scheme_folds(frame, "doi")
    papers, paper_idx = np.unique(doi, return_inverse=True)
    n_papers = len(papers)
    print(f"[UP-26] rows={len(frame)} papers={n_papers} folds={fold_info['n_folds']} "
          f"features={X_model.shape[1]}")

    rows_by_paper: List[np.ndarray] = [
        np.where(paper_idx == p)[0] for p in range(n_papers)
    ]
    sizes = np.array([len(r) for r in rows_by_paper])
    print(f"[UP-26] rows/paper: median={int(np.median(sizes))} max={int(sizes.max())} "
          f"singletons={int((sizes == 1).sum())}")

    fp = morgan_matrix(frame["ligand_canonical"].astype(str).tolist(),
                       n_bits=FP_BITS, radius=FP_RADIUS)
    sim_by_paper: Dict[int, np.ndarray] = {}
    for p in range(n_papers):
        rows = rows_by_paper[p]
        if len(rows) >= 2 and len(rows) <= 400:
            sim_by_paper[p] = similarity_block(fp, rows)
    print(f"[UP-26] similarity blocks cached for {len(sim_by_paper)} papers")

    per_seed: Dict[str, object] = {}
    kappa_log: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X_model, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        ev = fold_id >= 0
        base_r2 = score(y[ev], pred[ev])["r2"]

        resid = y - pred
        kappa_by_fold: Dict[int, float] = {}
        sigma2_by_fold: Dict[int, float] = {}
        for k in sorted(set(fold_id[fold_id >= 0].tolist())):
            tr = np.where((fold_id != k) & (fold_id >= 0))[0]
            v = paper_variances(resid, paper_idx, tr)
            kappa_by_fold[k] = v["kappa"]
            sigma2_by_fold[k] = v["sigma2_noise"]
            if seed == SEEDS[0]:
                kappa_log[str(k)] = {kk: round(float(vv), 6) if isinstance(vv, float) else vv
                                     for kk, vv in v.items()}

        sweep = sweep_ncal(
            y, pred, paper_idx, fold_id, rows_by_paper, sim_by_paper,
            kappa_by_fold, sigma2_by_fold, args.repeats, seed,
        )
        per_seed[str(seed)] = {
            "base_r2_all_rows": base_r2,
            "kappa_mean": round(float(np.mean(list(kappa_by_fold.values()))), 4),
            "ncal": sweep,
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        line = " ".join(
            f"{c}:{sweep[str(c)].get(PRIMARY_ARM, {}).get('r2', float('nan')):.4f}"
            for c in N_CAL_GRID
        )
        print(f"[UP-26] seed={seed} base={base_r2:.4f} kappa="
              f"{per_seed[str(seed)]['kappa_mean']} {line} "
              f"({per_seed[str(seed)]['elapsed_seconds']}s)")

    arms = ("eb_random", "eb_maxmin", "js_batch", "affine")
    agg: Dict[str, Dict[str, object]] = {}
    for n_cal in N_CAL_GRID:
        agg[str(n_cal)] = {}
        for arm in arms:
            vals = [per_seed[s]["ncal"][str(n_cal)].get(arm) for s in per_seed]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
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
            }

    n0 = agg["0"][PRIMARY_ARM]["r2_mean"]
    n10 = agg["10"][PRIMARY_ARM]["r2_mean"]
    curve = [agg[str(c)][PRIMARY_ARM]["r2_mean"] for c in N_CAL_GRID]
    steps = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    n_nondec = int(sum(1 for d in steps if d >= 0))

    checks = {
        "doi_r2_n10": {
            "arm": PRIMARY_ARM,
            "value": n10,
            "threshold": R2_GATE,
            "pass": bool(n10 >= R2_GATE),
        },
        "reproduce_n0": {
            "value": n0,
            "target": REPRO_TARGET,
            "tolerance": REPRO_TOL,
            "pass": bool(abs(n0 - REPRO_TARGET) <= REPRO_TOL),
        },
        "monotone": {
            "curve": curve,
            "steps": [round(float(d), 4) for d in steps],
            "non_decreasing_steps": n_nondec,
            "required": MIN_MONOTONE_STEPS,
            "pass": bool(n_nondec >= MIN_MONOTONE_STEPS),
        },
        "report": {
            "arms": list(arms),
            "kappa_mean": round(float(np.mean(
                [per_seed[s]["kappa_mean"] for s in per_seed])), 4),
            "pass": True,
        },
    }
    kill = {
        "rule": "R^2 at n_cal=10 < 0.40 closes the per-paper calibration line",
        "value": n10,
        "triggered": bool(n10 < R2_KILL),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_paper_calibration",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 deployment calibration (UP-26)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_feature_expansion.select_cohort (21,404 rows, potency, uncensored)",
            "base_arm": "maccs_rdkit + state + coordination (the UP-21 arm, 0.3670)",
            "split": "scheme_folds('doi'), 5 GroupKFold folds on DOI",
            "seeds": list(SEEDS[: args.seeds]),
            "repeats": args.repeats,
            "n_cal_grid": list(N_CAL_GRID),
            "primary_arm": PRIMARY_ARM,
            "estimator": "offset = n/(n+kappa) * mean(y - y_hat | calibration rows of the test paper)",
            "kappa_definition": "sigma2_noise / tau2_paper, both method-of-moments on training folds",
            "selection": {"eb_random": "uniform, nested permutation",
                          "eb_maxmin": "greedy Tanimoto farthest-point, nested order"},
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "kappa_by_fold_seed0": kappa_log,
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The calibration rows are drawn from the same paper as the evaluation rows. "
            "That is the deployment regime being modelled, not a leak: the correction uses "
            "their labels, and those rows are then excluded from scoring.",
            "kappa is estimated on the training folds of each outer split, so a test paper's "
            "own residual scale never enters its shrinkage factor.",
            "The affine arm has two free parameters from n_cal rows; at n_cal<=2 it "
            "degenerates to the offset arm and is reported for the decomposition only.",
            "James-Stein borrows strength across the other held-out papers in the same fold, "
            "which is a batch-arrival assumption no single-paper deployment satisfies.",
            "Papers with one row cannot be both calibrated and evaluated, so they leave the "
            "n_cal>0 row sets entirely; n_eval_rows records how much of the cohort is lost.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for n_cal in N_CAL_GRID:
        row = agg[str(n_cal)]
        print(f"[UP-26] n_cal={n_cal:2d} " + " ".join(
            f"{a}={row[a]['r2_mean']}" for a in arms if a in row))
    print(f"[UP-26] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
