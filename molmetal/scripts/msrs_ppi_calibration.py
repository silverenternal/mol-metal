#!/usr/bin/env python3
"""UP-38 — the paper's unlabeled rows are free variance reduction.

=====================================================================
The gap this closes
=====================================================================
UP-26 estimates a paper's offset from its ``n_cal`` measured rows:

    offset_hat = n_cal / (n_cal + kappa) * mean(y - y_hat | calibration rows)

At ``n_cal = 1`` that is one residual, shrunk hard: R^2 = 0.5559.  But the
same paper also has hundreds of rows whose *predictions* are known and whose
labels are not needed for the estimate.  Prediction-Powered Inference
(Angelopoulos et al. 2023) is exactly this setting: a small labeled set plus
a large unlabeled set, where the model output is used only to reduce variance.

=====================================================================
The estimator
=====================================================================
Write ``delta`` for the paper's mean residual, ``f`` for the prediction.
The PPI estimate of the paper mean is

    theta_ppi = mean_unlabeled(f) + mean_calibration(y - f)

and therefore

    delta_ppi = mean_calibration(y - f) + mean_unlabeled(f) - mean_all(f).

Its variance is ``sigma2 / c + Var(f | unlabeled) / (n - c)`` instead of the
naive ``Var(y) / c``.  Since the model explains part of the within-paper
variance, that is strictly smaller.  The shrinkage then uses the *same*
normal-normal posterior mean as UP-26, only with the smaller observation
variance:

    offset = delta_ppi * tau2 / (tau2 + v_ppi).

Substituting ``v = sigma2 / c`` recovers ``eb_offset`` exactly, so the new
estimator is a strict generalisation of the one it replaces, not a different
model.  That is what makes the comparison attributable.

=====================================================================
Arms
=====================================================================
eb_random       UP-26's estimator.  Must reproduce 0.3674/0.5559/0.6094/0.6839.
ppi_eb          PPI observation + the same EB shrinkage.  Declared primary.
ppi_eb_maxmin   the same, with the maxmin calibration design.
ppi_unshrunk    delta_ppi with no shrinkage.  Diagnostic: separates the
                variance gain from the bias the shrinkage was buying.

=====================================================================
Gate (todo.json, frozen 2026-10-23T05:30+08:00)
=====================================================================
n_cal=1   >= 0.60    (current 0.5559)
n_cal=2   >= 0.64    (current 0.6094)
n_cal=10  >= 0.6789  (no trade: the small-n gain must not cost the headline)
reproduce eb_random at 0/1/2/10 to 5e-5
kill      n_cal=1 < 0.5759 closes the small-sample line

Usage
=====
    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_ppi_calibration.py
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
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_GRID = (0, 1, 2, 5, 10, 50)
REPEATS = 20
ARMS = ("eb_random", "ppi_eb", "ppi_eb_maxmin", "ppi_unshrunk")
PRIMARY = "ppi_eb"
GATE_FROZEN_AT = "2026-10-23T05:30+08:00"
GATE_N1 = 0.60
GATE_N2 = 0.64
GATE_N10 = 0.6789
KILL_N1 = 0.5759
REPRO_TOL = 5e-5
REPRO = {0: 0.3674, 1: 0.5559, 2: 0.6094, 10: 0.6839}


def ppi_offset(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_unlab: np.ndarray,
    pred_all: np.ndarray,
    sigma2: float,
    tau2: float,
    shrink: bool,
) -> Tuple[float, float]:
    """Return (offset, observation variance) for one paper.

    ``delta`` is the PPI estimate of the paper's mean residual; ``v`` is its
    variance, which is what the posterior mean divides by.  With ``shrink``
    false the raw estimate is returned so the variance gain can be read
    separately from the bias correction.
    """
    c = len(y_cal)
    if c <= 0:
        return 0.0, 0.0
    delta = float(np.mean(y_cal - pred_cal) + np.mean(pred_unlab) - np.mean(pred_all))
    if len(pred_unlab) > 1:
        v_unlab = float(np.var(pred_unlab, ddof=1)) / float(len(pred_unlab))
    else:
        v_unlab = 0.0
    v = float(sigma2) / c + v_unlab
    if not shrink:
        return delta, v
    return float(delta * tau2 / (tau2 + v)), v


def sweep_seed(
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
    rng = np.random.default_rng(seed)
    test_papers = sorted(set(paper_idx[fold_id >= 0].tolist()))
    papers = [p for p in test_papers if len(rows_by_paper[p]) >= 2]
    resid = y - pred

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
    for n_cal in N_GRID:
        per_rep: Dict[str, List[np.ndarray]] = {a: [] for a in ARMS}
        per_rep_y: List[np.ndarray] = []
        per_rep_p0: List[np.ndarray] = []
        v_ratio: List[float] = []

        for rep in range(repeats):
            cur: Dict[str, List[np.ndarray]] = {a: [] for a in per_rep}
            ybuf: List[np.ndarray] = []
            p0buf: List[np.ndarray] = []
            for p in papers:
                rows = rows_by_paper[p]
                c = min(n_cal, len(rows) - 1)
                perm, mm = orders[(rep, p)]
                ev = np.ones(len(rows), dtype=bool)
                ev[perm[:c]] = False
                k = int(fold_id[rows[0]])
                kap = float(kappa_by_fold[k])
                sig = float(sigma2_by_fold[k])
                tau2 = sig / kap if kap > 0 else 0.0
                r = resid[rows]

                ybuf.append(y[rows][ev])
                p0buf.append(pred[rows][ev])

                # UP-26 arm, unchanged.
                cur["eb_random"].append(pred[rows][ev] + eb_offset(r[perm[:c]], kap))

                for arm, order, shrink in (
                    ("ppi_eb", perm, True),
                    ("ppi_eb_maxmin", mm, True),
                    ("ppi_unshrunk", perm, False),
                ):
                    cal = order[:c]
                    off, v = ppi_offset(
                        y[rows][cal], pred[rows][cal], pred[rows][ev], pred[rows],
                        sig, tau2, shrink,
                    )
                    cur[arm].append(pred[rows][ev] + off)
                    if arm == "ppi_eb" and c > 0:
                        v_eb = sig / c
                        v_ratio.append(float(v / v_eb) if v_eb > 0 else 1.0)

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
                "n_eval_papers": int(len(papers)),
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
        row["v_ppi_over_v_eb"] = round(float(np.mean(v_ratio)), 4) if v_ratio else None
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
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ppi_calibration.json")
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

    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X_model, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        resid = y - pred
        kappa_by_fold: Dict[int, float] = {}
        sigma2_by_fold: Dict[int, float] = {}
        for k in sorted(set(fold_id[fold_id >= 0].tolist())):
            tr = np.where((fold_id != k) & (fold_id >= 0))[0]
            v = paper_variances(resid, paper_idx, tr)
            kappa_by_fold[k] = float(v["kappa"])
            sigma2_by_fold[k] = float(v["sigma2_noise"])

        per_seed[str(seed)] = sweep_seed(
            y, pred, paper_idx, fold_id, rows_by_paper, sim_by_paper,
            kappa_by_fold, sigma2_by_fold, args.repeats, seed,
        )
        line = " ".join(f"{c}:{per_seed[str(seed)][str(c)][PRIMARY]['r2']:.4f}"
                        for c in N_GRID)
        print(f"[UP-38] seed={seed} {line} ({round(time.time() - ts, 1)}s)")

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
        agg[str(n_cal)]["v_ppi_over_v_eb"] = round(
            float(np.mean([per_seed[s][str(n_cal)]["v_ppi_over_v_eb"] for s in per_seed
                           if per_seed[s][str(n_cal)]["v_ppi_over_v_eb"] is not None])), 4
        ) if n_cal > 0 else None

    n1 = agg["1"][PRIMARY]["r2_mean"]
    n2 = agg["2"][PRIMARY]["r2_mean"]
    n10 = agg["10"][PRIMARY]["r2_mean"]
    repro = {
        str(k): {
            "value": agg[str(k)]["eb_random"]["r2_mean"],
            "target": v,
            "pass": bool(abs(agg[str(k)]["eb_random"]["r2_mean"] - v) <= REPRO_TOL),
        }
        for k, v in REPRO.items()
    }
    checks = {
        "n1": {"value": n1, "threshold": GATE_N1, "pass": bool(n1 >= GATE_N1)},
        "n2": {"value": n2, "threshold": GATE_N2, "pass": bool(n2 >= GATE_N2)},
        "n10_no_trade": {"value": n10, "threshold": GATE_N10, "pass": bool(n10 >= GATE_N10)},
        "reproduce": {"checks": repro, "pass": all(v["pass"] for v in repro.values())},
    }
    gate_pass = all(v["pass"] for v in checks.values())
    kill = {
        "rule": "n_cal=1 R^2 < 0.5759 closes the small-sample line",
        "value": n1,
        "triggered": bool(n1 < KILL_N1),
    }

    out = {
        "artifact": "msrs_ppi_calibration",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 deployment calibration (UP-38)",
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
            "estimator": "delta_ppi = mean_cal(y-f) + mean_unlab(f) - mean_all(f); "
                         "offset = delta_ppi * tau2/(tau2 + sigma2/c + Var(f|unlab)/(n-c))",
            "why_strict_generalisation": "setting the unlabeled term to zero recovers "
                                         "eb_offset = c/(c+kappa)*mean(resid|cal)",
            "arms": {
                "eb_random": "UP-26 estimator, reproduction arm",
                "ppi_eb": "PPI observation variance + EB shrinkage (primary)",
                "ppi_eb_maxmin": "same with the maxmin calibration design",
                "ppi_unshrunk": "delta_ppi without shrinkage (diagnostic)",
            },
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The unlabeled rows are the evaluation rows. That is the deployment regime, "
            "not a leak: only their predictions enter the estimate, never their labels.",
            "PPI assumes the calibration rows are a uniform draw from the paper. The "
            "maxmin arm violates that by design and is reported separately for that reason.",
            "v_ppi_over_v_eb is the ratio of observation variances. It is the mechanism "
            "readout: below 1 means the unlabeled rows bought variance.",
            "FAB-PPI (arXiv:2502.02363) was the motivation for pairing a Bayesian "
            "shrinkage with the PPI observation, but its estimator is not implemented "
            "here: the paper text could not be retrieved, and inventing its posterior "
            "would not be reproducible. The EB posterior mean plays that role.",
            "Papers with one row cannot be calibrated and evaluated, so they leave the "
            "n_cal>0 row sets entirely, exactly as in UP-26.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for n_cal in N_GRID:
        row = agg[str(n_cal)]
        print(f"[UP-38] n_cal={n_cal:2d} " + " ".join(
            f"{a}={row[a]['r2_mean']}" for a in ARMS) +
              f" v_ratio={row['v_ppi_over_v_eb']}")
    print(f"[UP-38] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time() - t0, 1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
