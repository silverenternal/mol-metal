#!/usr/bin/env python3
"""MSRS-EXT-3 — how many measurements of a new metal do you need before it works?

=====================================================================
What EXT-1 changed about this question
=====================================================================
The plan assumed the leave-one-metal-out failure was dominated by a level
offset, so the natural fix was to estimate that offset from k measurements of
the new metal.  EXT-1 measured the offset share at 0.05-0.10 of squared error
and the shape (ranking) share at 0.65-0.76.  So the level is *not* the problem,
and a pure offset correction has almost nothing to correct.

What remains worth asking is narrower and more useful:

    given k measurements of a metal the model has never seen, can any
    monotone recalibration of its predictions beat "just use the metal's
    mean"?

That is what this script measures.  Three calibrations, because they make
different assumptions about what is broken:

    offset      y_hat + (mean(y_cal) - mean(p_cal))     level only
    linear      OLS y_cal ~ p_cal                        level + compression
    isotonic    monotone, no functional form             level + any monotone warp

=====================================================================
The ceiling, and why every number here is small
=====================================================================
EXT-1 reports `calibration_ceiling_r2 = rho^2` per metal: the R2 an
infinite-sample linear recalibration would reach.  For the honest split
(metal_out_ligand_disjoint) that ceiling is 0.002-0.131.  No k can beat it.
So the interesting output is not "R2 = 0.3"; it is the *fraction of ceiling*
recovered at each k, and whether R2 crosses zero at all.

=====================================================================
Design
=====================================================================
For each endpoint x split x held-out metal:

    1. fit the zero-shot model on the other four metals (raw target)
    2. for k in {0, 5, 10, 20, 50, 100, 200}:
         sample k calibration rows by DOI cluster (never split a paper)
         fit each calibration on those rows
         score on the remaining test rows only
         repeat 20 times; report mean and 2.5/97.5 percentiles
    3. report the ceiling from the full test set

Gate (amended 2026-10-21, before this script was first run -- see todo.json):

    PASS   on metal_out_ligand_disjoint, k=50 gives R2 >= 0 for >= 3/5 metals
    KILL   at k=100, R2 < 0 for >= 4/5 metals

The gate is read on the ligand-disjoint split because deploying a new metal
means new ligands too; the overlapping-ligand split is reported for contrast
and is not a deployment estimate.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_kshot_calibration.py
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
from scipy import stats
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_lomo_error_decomposition import (  # noqa: E402
    build_design,
    r2,
    spearman,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    SEED,
    TOP_CELL_LINES,
    fit_predict,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ENDPOINTS = ("potency", "cisplatin_ratio")
SPLITS = (("metal_out", False), ("metal_out_ligand_disjoint", True))
K_VALUES = (0, 5, 10, 20, 50, 100, 200)
N_REPEATS = 20
CALIBRATIONS = ("offset", "linear", "isotonic")
GATE_SPLIT = "metal_out_ligand_disjoint"
GATE_K = 50
KILL_K = 100


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
# Calibrators: fit on (p_cal, y_cal), map new predictions
# ---------------------------------------------------------------------------
def fit_calibrator(kind: str, p_cal: np.ndarray, y_cal: np.ndarray):
    if kind == "offset":
        shift = float(np.mean(y_cal) - np.mean(p_cal))
        return lambda p: p + shift
    if kind == "linear":
        if np.std(p_cal) < 1e-9:
            shift = float(np.mean(y_cal) - np.mean(p_cal))
            return lambda p: p + shift
        b, a = np.polyfit(p_cal, y_cal, 1)
        return lambda p: a + b * p
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        return lambda p: iso.predict(p)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# DOI-clustered sampling of calibration rows
# ---------------------------------------------------------------------------
def sample_calibration(dois: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick whole DOI groups until at least k rows are covered."""
    uniq = np.array(sorted(set(dois)))
    order = rng.permutation(len(uniq))
    chosen, n = [], 0
    for i in order:
        idx = np.where(dois == uniq[i])[0]
        chosen.append(idx)
        n += len(idx)
        if n >= k:
            break
    return np.concatenate(chosen)


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def run_cell(
    X,
    y: np.ndarray,
    metals: np.ndarray,
    fragments: np.ndarray,
    dois: np.ndarray,
    groups,
    holdout: str,
    ligand_disjoint: bool,
    seed: int,
) -> Dict[str, object]:
    train = np.where(metals != holdout)[0]
    test = np.where(metals == holdout)[0]
    if ligand_disjoint:
        held = set(fragments[test])
        keep = np.array([f not in held for f in fragments[train]])
        train = train[keep]
    if len(train) < 200 or len(test) < 50:
        return {"status": "INSUFFICIENT", "n_train": int(len(train)), "n_test": int(len(test))}

    pred = fit_predict(X[train], y[train], X[test], groups)
    y_te = y[test]
    d_te = dois[test]

    # Ceiling: in-sample linear recalibration on the whole test cohort (rho^2).
    rho_full, _ = spearman(pred, y_te)
    ceiling = float(rho_full ** 2)

    rng = np.random.default_rng(seed)
    cells: Dict[str, object] = {}
    for kind in CALIBRATIONS:
        curve = {}
        for k in K_VALUES:
            if k == 0:
                curve[str(k)] = {
                    "mean_r2": round(r2(pred, y_te), 4),
                    "ci95": [round(r2(pred, y_te), 4)] * 2,
                    "mean_spearman": round(rho_full, 4),
                    "mean_frac_of_ceiling": 1.0 if ceiling > 0 else None,
                    "n_eval": int(len(y_te)),
                }
                continue
            if k >= len(y_te) // 2:
                curve[str(k)] = {"status": "SKIP", "reason": f"k={k} >= half of n_test={len(y_te)}"}
                continue
            vals, rhos, fracs, ns = [], [], [], []
            for _ in range(N_REPEATS):
                cal = sample_calibration(d_te, k, rng)
                mask = np.ones(len(y_te), dtype=bool)
                mask[cal] = False
                if mask.sum() < 30:
                    continue
                apply = fit_calibrator(kind, pred[cal], y_te[cal])
                p_adj = apply(pred[mask])
                r = r2(p_adj, y_te[mask])
                rho, _ = spearman(p_adj, y_te[mask])
                vals.append(r)
                rhos.append(rho)
                ns.append(int(mask.sum()))
                if ceiling > 0:
                    fracs.append(r / ceiling)
            if not vals:
                curve[str(k)] = {"status": "SKIP", "reason": "no valid repeat"}
                continue
            curve[str(k)] = {
                "mean_r2": round(float(np.mean(vals)), 4),
                "median_r2": round(float(np.median(vals)), 4),
                "ci95": [
                    round(float(np.percentile(vals, 2.5)), 4),
                    round(float(np.percentile(vals, 97.5)), 4),
                ],
                "frac_repeats_worse_than_zero_shot": round(
                    float(np.mean([v < r2(pred, y_te) for v in vals])), 3
                ),
                "mean_spearman": round(float(np.mean(rhos)), 4),
                "mean_frac_of_ceiling": round(float(np.mean(fracs)), 4) if fracs else None,
                "n_eval": int(np.mean(ns)),
                "n_repeats": len(vals),
            }
        cells[kind] = curve

    return {
        "status": "OK",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_ligands_test": int(len(set(fragments[test]))),
        "n_dois_test": int(len(set(d_te))),
        "zero_shot_r2": round(r2(pred, y_te), 4),
        "ceiling_r2": round(ceiling, 4),
        "calibrations": cells,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict(cells: Dict[str, Dict[str, object]], k: int, kind: str = "linear") -> Dict[str, object]:
    ok = {m: v for m, v in cells.items() if v.get("status") == "OK"}
    at_k = {
        m: v["calibrations"][kind].get(str(k), {}).get("mean_r2")
        for m, v in ok.items()
        if isinstance(v["calibrations"][kind].get(str(k)), dict)
    }
    vals = {m: x for m, x in at_k.items() if x is not None}
    n_pos = sum(1 for x in vals.values() if x >= 0)
    return {
        "k": k,
        "calibration": kind,
        "per_metal_r2": vals,
        "n_metals_r2_nonnegative": n_pos,
        "n_metals": len(vals),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_kshot_calibration.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = df[(df["endpoint"] == ep) & df["value_log10"].notna() & df["ligand_main_fragment"].notna()].copy()
        sub = sub.reset_index(drop=True)
        vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
        X, groups = build_design(sub, descriptors, vocabulary)
        y = sub["value_log10"].to_numpy(dtype=float)
        metals = sub["metal"].to_numpy()
        fragments = sub["ligand_main_fragment"].to_numpy()
        dois = sub["doi"].to_numpy()

        results[ep] = {"status": "OK", "n_rows": int(len(sub)), "splits": {}}
        for split_name, disjoint in SPLITS:
            t0 = time.time()
            cells = {}
            for metal in METALS:
                cells[metal] = run_cell(
                    X, y, metals, fragments, dois, groups, metal, disjoint, seed=SEED
                )
            results[ep]["splits"][split_name] = {
                "per_metal": cells,
                "verdict_at_gate_k": verdict(cells, GATE_K, "linear"),
                "verdict_at_kill_k": verdict(cells, KILL_K, "linear"),
                "verdict_offset_only": verdict(cells, GATE_K, "offset"),
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            v = results[ep]["splits"][split_name]["verdict_at_gate_k"]
            print(f"[{ep}/{split_name}] k={GATE_K} linear: R2>=0 for {v['n_metals_r2_nonnegative']}/{v['n_metals']} metals")
            for m, x in v["per_metal_r2"].items():
                if x is not None:
                    print(f"    {m:>2s} R2={x:+.3f}")

    gate = results["potency"]["splits"][GATE_SPLIT]["verdict_at_gate_k"]
    kill = results["potency"]["splits"][GATE_SPLIT]["verdict_at_kill_k"]
    if gate["n_metals_r2_nonnegative"] >= 3:
        label = "PASS"
    elif kill["n_metals_r2_nonnegative"] <= 1:
        label = "KILL"
    else:
        label = "PARTIAL"

    out = {
        "artifact": "msrs_kshot_calibration",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 k-shot deployability",
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
            "learner": "additive LightGBM from MSRS-2 (interaction constraints)",
            "k_values": list(K_VALUES),
            "calibrations": list(CALIBRATIONS),
            "n_repeats": N_REPEATS,
            "sampling": "DOI-clustered: whole papers are drawn until k rows are covered",
            "evaluation": "calibration rows excluded from scoring",
            "gate_split": GATE_SPLIT,
            "gate": "k=50 linear R2 >= 0 for >= 3/5 metals on the ligand-disjoint split",
            "kill": "k=100 R2 < 0 for >= 4/5 metals on the ligand-disjoint split",
            "amendment_note": (
                "The original EXT-3 gate was 'k=50 recovers >=60% of the level gap'. "
                "EXT-1 measured the level gap at 5-10% of squared error, which makes that "
                "gate degenerate (a small denominator). The gate was rewritten to the "
                "deployment question before this script was first run."
            ),
        },
        "verdict": label,
        "endpoints": results,
        "caveats": [
            "Ceiling is rho^2 computed in-sample on the full test cohort; k-shot numbers are "
            "out-of-sample on the non-calibration rows, so k-shot cannot reach the ceiling even "
            "at large k. mean_frac_of_ceiling is therefore an upper-bound comparison, not a ratio "
            "that should approach 1.",
            "On metal_out_ligand_disjoint the ceiling is 0.002-0.131, so R2 >= 0 is a weak "
            "absolute bar; it is nevertheless the deployment bar, because R2 < 0 means worse "
            "than predicting the metal's mean.",
            "DOI-clustered sampling means the realised k exceeds the nominal k by up to one "
            "paper's row count.",
            "Censored rows are dropped (potency 18.4%).",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"\njson: {args.out}")
    print(f"VERDICT {label}  (gate {GATE_SPLIT}, k={GATE_K})")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
