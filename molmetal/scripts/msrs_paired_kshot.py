#!/usr/bin/env python3
"""UP-22 — paired k-shot: does the k-point failure come from the points?

=====================================================================
Why this exists
=====================================================================
UP-8 sampled k calibration rows from the new metal's cohort at random and
found that at k=50 only one of five metals reached R^2 > 0.  The
interpretation at the time was "adaptation does not work".  UP-10 then
measured that the information a metal swap needs lives in a *paired* row:
one whose (ligand, cell line) is also measured under a metal the model has
seen.  A random k-point sample contains almost none of those.

So the 1/5 may be a sampling artefact, not a property of adaptation.  This
script makes the number of paired points the independent variable:

    n_paired in {0, 1, 2, 3, 5, 10, 20}, k = 50 fixed

n_paired rows are drawn from the paired pool; the remaining k - n_paired
come from the rest of the cohort.  At n_paired = 0 this is UP-8's design,
which is why that arm doubles as the reproduction check.

=====================================================================
Two sampling units, reported side by side
=====================================================================
UP-8 drew whole DOI groups so that a paper was never split between
calibration and evaluation.  Controlling the paired count forces row-level
selection, so the primary arms sample rows.  The n_paired = 0 cell is run
under both units; if they agree, the change of unit is not what moves the
number, and if they do not, that is a finding rather than a nuisance.

=====================================================================
Ridge in the dual
=====================================================================
UP-8 solves the primal ridge, which is a (d+1) x (d+1) system with
d = 2048 -- about a second per fit.  This sweep needs ~3,500 fits, so the
same estimator is solved in the dual, a k x k system with k = 50.  The two
forms are algebraically identical; `dual_vs_primal_max_abs_diff` in the
artifact verifies it on the first cell.

=====================================================================
Gate (todo.json, frozen 2026-10-22T12:42+08:00)
=====================================================================
primary    k=50, n_paired=3: R^2 > 0 for >= 3 of 5 metals (UP-8: 1/5)
secondary  mean R^2 monotone non-decreasing over n_paired 0 -> 3
report     n and seed count for every (metal, n_paired) cell
kill       fewer than 2 of 5 metals above zero at n_paired=3 closes the
           k-shot line

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_paired_kshot.py
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

from molmetal.scripts.msrs_lomo_error_decomposition import (  # noqa: E402
    build_design,
    r2,
)
from molmetal.scripts.msrs_meta_kshot import adapt_proto, adapt_ridge  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    fit_predict,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

K = 50
N_PAIRED_VALUES = (0, 1, 2, 3, 5, 10, 20)
SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
N_REPEATS = 20
GATE_FROZEN_AT = "2026-10-22T12:42+08:00"
GATE_N_PAIRED = 3
GATE_METALS = 3
KILL_METALS = 2
ARMS = ("offset", "ridge", "proto")


def adapt_ridge_dual(
    Xc: sparse.csr_matrix, resid: np.ndarray, Xt: sparse.csr_matrix, alpha: float = 1.0
) -> np.ndarray:
    """Primal ridge with an unpenalised intercept, solved through the k x k dual.

    w = Xc' (Xc Xc' + alpha I)^-1 yc  is the Woodbury identity for
    (Xc'Xc + alpha I)^-1 Xc' yc, and k << d here.
    """
    A = Xc.toarray()
    B = Xt.toarray()
    xm = A.mean(axis=0)
    ym = float(resid.mean())
    Ac = A - xm
    yc = resid - ym
    Kmat = Ac @ Ac.T + alpha * np.eye(Ac.shape[0])
    a = np.linalg.solve(Kmat, yc)
    w = Ac.T @ a
    return (B - xm) @ w + ym


def paired_mask(
    frame: pd.DataFrame, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Test rows whose (ligand, cell) partner is measured in the training metals."""
    keys = set(zip(
        frame["ligand_canonical"].iloc[train].astype(str),
        frame["cell_line_key"].iloc[train].astype(str),
    ))
    return np.array([
        (str(l), str(c)) in keys
        for l, c in zip(
            frame["ligand_canonical"].iloc[test], frame["cell_line_key"].iloc[test]
        )
    ])


def select_rows(
    pool_paired: np.ndarray, pool_unpaired: np.ndarray, k: int,
    n_paired: int, rng: np.random.Generator,
) -> np.ndarray:
    take_p = min(n_paired, len(pool_paired))
    p = rng.choice(pool_paired, size=take_p, replace=False) if take_p else np.array([], int)
    need = k - take_p
    u = (
        rng.choice(pool_unpaired, size=min(need, len(pool_unpaired)), replace=False)
        if need > 0 and len(pool_unpaired) else np.array([], int)
    )
    return np.concatenate([p, u]).astype(int)


def select_doi_groups(
    dois: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """UP-8's sampler: whole DOI groups until at least k rows are covered."""
    uniq = np.array(sorted(set(dois.tolist())))
    order = rng.permutation(len(uniq))
    chosen, n = [], 0
    for i in order:
        idx = np.where(dois == uniq[i])[0]
        chosen.append(idx)
        n += len(idx)
        if n >= k:
            break
    return np.concatenate(chosen)


def run_cell(
    X, y, frame, metals, groups, holdout, seed, verify_dual: bool
) -> Dict[str, object]:
    train = np.where(metals != holdout)[0]
    test = np.where(metals == holdout)[0]
    if len(train) < 200 or len(test) < 50:
        return {"status": "INSUFFICIENT", "n_train": int(len(train)), "n_test": int(len(test))}

    pred = fit_predict(X[train], y[train], X[test], groups)
    y_te = y[test]
    Xt = X[test]
    base_r2 = r2(pred, y_te)

    pm = paired_mask(frame, train, test)
    idx_paired = np.flatnonzero(pm)
    idx_unpaired = np.flatnonzero(~pm)

    cells: Dict[str, object] = {}
    dual_check = {"status": "SKIPPED"}

    for n_paired in N_PAIRED_VALUES:
        for arm in ARMS:
            scores: List[float] = []
            rng = np.random.default_rng(seed * 1000 + n_paired)
            for _rep in range(N_REPEATS):
                cal = select_rows(idx_paired, idx_unpaired, K, n_paired, rng)
                if len(cal) == 0:
                    continue
                ev = np.ones(len(y_te), dtype=bool)
                ev[cal] = False
                if ev.sum() < 30:
                    continue
                resid = y_te[cal] - pred[cal]
                if arm == "offset":
                    delta = np.full(int(ev.sum()), float(resid.mean()))
                elif arm == "ridge":
                    delta = adapt_ridge_dual(Xt[cal], resid, Xt[ev])
                    if verify_dual and dual_check["status"] == "SKIPPED" and n_paired == 0:
                        primal = adapt_ridge(Xt[cal], resid, Xt[ev])
                        dual_check = {
                            "status": "OK",
                            "n_calibration": int(len(cal)),
                            "max_abs_diff": round(
                                float(np.max(np.abs(primal - delta))), 8),
                        }
                else:
                    delta = adapt_proto(Xt[cal], resid, Xt[ev])
                scores.append(r2(pred[ev] + delta, y_te[ev]))

            key = f"{arm}|{n_paired}"
            arr = np.array(scores)
            cells[key] = {
                "arm": arm,
                "n_paired_requested": n_paired,
                "n_paired_available": int(len(idx_paired)),
                "n_repeats": int(len(arr)),
                "mean_r2": round(float(arr.mean()), 4) if len(arr) else None,
                "median_r2": round(float(np.median(arr)), 4) if len(arr) else None,
                "ci95": [
                    round(float(np.quantile(arr, 0.025)), 4),
                    round(float(np.quantile(arr, 0.975)), 4),
                ] if len(arr) else None,
            }

    # n_paired = 0 under UP-8's DOI-group sampler, for the reproduction check.
    doi_scores: List[float] = []
    dois_te = frame["doi"].astype(str).to_numpy()[test]
    rng = np.random.default_rng(seed * 1000 + 999)
    for _rep in range(N_REPEATS):
        cal = select_doi_groups(dois_te, K, rng)
        ev = np.ones(len(y_te), dtype=bool)
        ev[cal] = False
        if ev.sum() < 30:
            continue
        resid = y_te[cal] - pred[cal]
        delta = np.full(int(ev.sum()), float(resid.mean()))
        doi_scores.append(r2(pred[ev] + delta, y_te[ev]))
    cells["offset_doi_group|0"] = {
        "arm": "offset_doi_group",
        "n_paired_requested": 0,
        "n_paired_available": int(len(idx_paired)),
        "n_repeats": int(len(doi_scores)),
        "mean_r2": round(float(np.mean(doi_scores)), 4) if doi_scores else None,
        "median_r2": round(float(np.median(doi_scores)), 4) if doi_scores else None,
        "ci95": [
            round(float(np.quantile(doi_scores, 0.025)), 4),
            round(float(np.quantile(doi_scores, 0.975)), 4),
        ] if doi_scores else None,
    }

    return {
        "status": "OK",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_paired_rows": int(len(idx_paired)),
        "share_paired_rows": round(float(pm.mean()), 4),
        "zero_shot_r2": round(float(base_r2), 4),
        "cells": cells,
        "dual_vs_primal": dual_check,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_paired_kshot.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    sub = df[(df["endpoint"] == args.endpoint) & df["value_log10"].notna()
             & df["ligand_main_fragment"].notna()].reset_index(drop=True)
    vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, groups = build_design(sub, descriptors, vocabulary)
    y = sub["value_log10"].to_numpy(dtype=float)
    metals = sub["metal"].to_numpy()

    print(f"[UP-22] rows={len(sub)} metals={pd.Series(metals).value_counts().to_dict()} "
          f"k={K} seeds={args.seeds}")

    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        cells = {
            m: run_cell(X, y, sub, metals, groups, m, seed,
                        verify_dual=(seed == SEEDS[0]))
            for m in METALS
        }
        per_seed[str(seed)] = {
            "per_metal": cells,
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        print(f"[UP-22] seed={seed} "
              + " ".join(
                  f"{m}:z={cells[m].get('zero_shot_r2')},"
                  f"np3={cells[m]['cells'].get('ridge|3', {}).get('mean_r2') if cells[m].get('status') == 'OK' else None}"
                  for m in METALS
              )
              + f" ({per_seed[str(seed)]['elapsed_seconds']}s)")

    # Seed-averaged cells.
    agg: Dict[str, object] = {}
    for m in METALS:
        block = {}
        for arm in ARMS:
            for n_paired in N_PAIRED_VALUES:
                key = f"{arm}|{n_paired}"
                vals = [
                    per_seed[s]["per_metal"][m]["cells"].get(key, {}).get("mean_r2")
                    for s in per_seed
                ]
                vals = [v for v in vals if v is not None]
                block[key] = {
                    "mean_r2": round(float(np.mean(vals)), 4) if vals else None,
                    "n_seeds": len(vals),
                }
        vals = [per_seed[s]["per_metal"][m]["cells"]["offset_doi_group|0"]["mean_r2"]
                for s in per_seed]
        block["offset_doi_group|0"] = {
            "mean_r2": round(float(np.mean(vals)), 4),
            "n_seeds": len(vals),
        }
        block["zero_shot_r2"] = round(
            float(np.mean([per_seed[s]["per_metal"][m]["zero_shot_r2"] for s in per_seed])), 4
        )
        block["n_paired_rows"] = per_seed[str(SEEDS[0])]["per_metal"][m]["n_paired_rows"]
        agg[m] = block

    # Gate on the best-performing adaptation arm (ridge), as in UP-8.
    gate_arm = "ridge"
    above = [m for m in METALS
             if (agg[m].get(f"{gate_arm}|{GATE_N_PAIRED}", {}).get("mean_r2") or -9) > 0]
    n_above = len(above)
    monotone_vals = [
        float(np.mean([
            agg[m][f"{gate_arm}|{n}"]["mean_r2"] for m in METALS
            if agg[m][f"{gate_arm}|{n}"]["mean_r2"] is not None
        ]))
        for n in (0, 1, 2, 3)
    ]
    monotone = all(
        monotone_vals[i + 1] >= monotone_vals[i] - 1e-9 for i in range(len(monotone_vals) - 1)
    )

    level_gain = {
        m: round(agg[m][f"{gate_arm}|0"]["mean_r2"] - agg[m]["zero_shot_r2"], 4)
        for m in METALS
        if agg[m][f"{gate_arm}|0"]["mean_r2"] is not None
    }
    interaction_gain = {
        m: round(agg[m][f"{gate_arm}|{GATE_N_PAIRED}"]["mean_r2"]
                 - agg[m][f"{gate_arm}|0"]["mean_r2"], 4)
        for m in METALS
        if agg[m][f"{gate_arm}|0"]["mean_r2"] is not None
        and agg[m][f"{gate_arm}|{GATE_N_PAIRED}"]["mean_r2"] is not None
    }

    checks = {
        "primary": {
            "metric": f"{gate_arm}, k={K}, n_paired={GATE_N_PAIRED}: metals with R2 > 0",
            "value": n_above,
            "metals": above,
            "n_metals": len(METALS),
            "threshold": GATE_METALS,
            "pass": bool(n_above >= GATE_METALS),
        },
        "secondary": {
            "metric": "mean R2 over metals monotone non-decreasing, n_paired 0..3",
            "values": [round(v, 4) for v in monotone_vals],
            "pass": bool(monotone),
        },
    }
    kill = {
        "rule": f"fewer than {KILL_METALS} of 5 metals above zero at n_paired={GATE_N_PAIRED}",
        "value": n_above,
        "triggered": bool(n_above < KILL_METALS),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_paired_kshot",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 few-shot, pairing-controlled (UP-22)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": args.endpoint,
            "n_rows": int(len(sub)),
            "k": K,
            "n_paired_values": list(N_PAIRED_VALUES),
            "n_repeats": N_REPEATS,
            "seeds": list(SEEDS[: args.seeds]),
            "arms": list(ARMS),
            "paired_definition": "test row whose (ligand_canonical, cell_line_key) is "
                                 "measured under a training metal",
            "sampling": "rows; n_paired drawn from the paired pool, the rest from the "
                        "unpaired pool, without replacement",
            "doi_group_arm": "n_paired=0 with UP-8's whole-DOI sampler, as a reproduction check",
            "ridge": "dual form of the primal ridge used by msrs_meta_kshot.adapt_ridge",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "gain_decomposition": {
            "arm": gate_arm,
            "level_gain_n_paired_0": level_gain,
            "interaction_gain_n_paired_0_to_3": interaction_gain,
            "note": "level gain is what a pure offset correction buys with no paired point; "
                    "interaction gain is what the first three paired points add on top.",
        },
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "Re has zero paired rows (UP-19 measured the same fact), so its n_paired > 0 "
            "cells silently fall back to n_paired = 0. The requested count is recorded next "
            "to the available count in every cell.",
            "Calibration rows are sampled at row level, not by DOI group. The "
            "offset_doi_group arm isolates that change at n_paired = 0.",
            "The ridge is solved in the dual (k x k); dual_vs_primal records the maximum "
            "absolute difference against the primal solve on the first cell.",
            "R2 is computed on the evaluation rows of one metal, so it is a within-metal "
            "number and carries no metal-level information.",
            "Twenty repeats per cell give a noisy mean; the seed spread is in per_seed.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for m in METALS:
        print(f"[UP-22] {m:2s} paired={agg[m]['n_paired_rows']:4d} zero={agg[m]['zero_shot_r2']:+.4f} "
              + " ".join(f"np{n}={agg[m][f'{gate_arm}|{n}']['mean_r2']:+.4f}"
                         for n in (0, 1, 2, 3, 5, 10, 20)
                         if agg[m][f'{gate_arm}|{n}']['mean_r2'] is not None))
    print(f"[UP-22] gate_pass={gate_pass} above={n_above}/5 monotone={monotone} "
          f"kill={kill['triggered']} elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
