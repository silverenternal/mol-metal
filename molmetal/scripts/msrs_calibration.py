#!/usr/bin/env python3
"""MET-4 — is the prediction interval usable, or only nominally calibrated?

The existing conformal report (`msrs_actionability.json`) hits its nominal
coverage (0.9044 at 0.90) but with a median width of 2.2083 log10 — roughly a
162-fold IC50 span.  A 90% interval that wide is calibrated and useless.

This rebuilds the interval on the MET-2 out-of-fold predictions under the
whole-paper-holdout protocol and reports the numbers a reader needs to decide
whether to use it:

  * coverage (marginal, by metal, by cohort size);
  * mean / median / p10 / p90 width;
  * the Winkler interval score (proper, rewards both coverage and sharpness);
  * a PIT histogram (uniform if calibrated, U-shaped if over-confident);
  * point-prediction calibration slope / intercept / R2.

Three interval variants, from cheapest to sharpest:

  ``global``      one conformal quantile on |residual|
  ``metal``       Mondrian: one quantile per metal (fallback to global)
  ``difficulty``  |residual| / sigma(x), where sigma is a small LightGBM fitted
                  on the out-of-fold absolute residuals using (prediction,
                  metal, cell line) — then a metal-conditional quantile

Caveat, stated once and repeated in the artifact: the base predictions are
MET-2's out-of-fold predictions, so for outer fold k the calibration rows come
from models that saw fold k.  Their residuals are mildly optimistic, which
makes q too small and coverage a *lower* bound.  The direction of the bias is
known; it cannot make the interval look better than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_actionability import (  # noqa: E402
    ALPHA,
    conformal_quantile,
    coverage_report,
)
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("doi", "ligand")
ARM = "ligand+both"
MIN_METAL_CAL = 30
MIN_COHORT_CAL = 50
N_BINS = 10
SEED = 20261018
COVERAGE_TOL = 0.02
WIDTH_GATE = 1.0
DIFFICULTY_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Interval metrics
# ---------------------------------------------------------------------------
def _pit(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """PIT = fraction of calibration scores <= the test score (O(n log n))."""
    ordered = np.sort(cal_scores)
    return np.searchsorted(ordered, test_scores, side="right") / len(ordered)


def winkler_score(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA
) -> float:
    """Mean Winkler interval score; lower is better.

    Width plus a 2/alpha penalty for every observation outside the interval.
    Unlike coverage it cannot be improved by making the interval wider.
    """
    width = hi - lo
    penalty = (2.0 / alpha) * (
        np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0)
    )
    return round(float(np.mean(width + penalty)), 4)


def pit_histogram(pit: np.ndarray, n_bins: int = N_BINS) -> Dict[str, object]:
    """PIT histogram; uniform means calibrated, U-shaped means over-confident."""
    counts, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, n_bins + 1))
    frac = counts / max(counts.sum(), 1)
    return {
        "n": int(counts.sum()),
        "bin_fraction": [round(float(x), 4) for x in frac],
        "max_abs_deviation_from_uniform": round(
            float(np.max(np.abs(frac - 1.0 / n_bins))), 4
        ),
        "mean_pit": round(float(np.mean(pit)), 4),
        "ks_stat": round(float(np.max(np.abs(
            np.sort(pit) - np.linspace(0, 1, len(pit))
        ))), 4) if len(pit) else None,
    }


def point_calibration(y: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    slope, intercept = np.polyfit(pred, y, 1)
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # r2_point is the model's own R2 (negative if the point predictions are
    # badly scaled).  r2_fit is the R2 of the linear recalibration, which is
    # just corr(pred, y)^2 — it says how much of the ordering is recoverable
    # after a monotone rescaling, and is the number that separates "wrong
    # scale" from "no signal".
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else 0.0
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2_point": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "r2_fit": round(corr**2, 4),
        "mean_bias": round(float(resid.mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(resid**2))), 4),
        # A Gaussian 90% interval on this residual scale.  The conformal width
        # should land near this number; if it does, the interval is not
        # "too wide" for a fixable reason — the model is that imprecise.
        "gaussian_width_90": round(float(2 * 1.6449 * np.std(resid)), 4),
    }


def cohort_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Number of distinct metals in each row's cohort."""
    key = (
        frame["ligand_canonical"].astype(str)
        + "|"
        + frame["cell_line_key"].astype(str)
        + "|"
        + frame["doi"].astype(str)
        + "|"
        + frame["time_h"].astype(str)
    )
    return frame.assign(_k=key).groupby("_k")["metal"].transform("nunique").to_numpy()


# ---------------------------------------------------------------------------
# Difficulty model
# ---------------------------------------------------------------------------
def difficulty_design(
    pred: np.ndarray, metal: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Features the difficulty model may use: prediction, metal code, cell code."""
    metal_codes = pd.factorize(metal)[0]
    cell_codes = pd.factorize(cell)[0]
    X = np.column_stack([pred, metal_codes, cell_codes]).astype(np.float32)
    return X, [1, 2]


def fit_difficulty(
    pred: np.ndarray, resid: np.ndarray, metal: np.ndarray, cell: np.ndarray
) -> object:
    from lightgbm import LGBMRegressor

    X, cats = difficulty_design(pred, metal, cell)
    model = LGBMRegressor(**DIFFICULTY_PARAMS)
    model.fit(X, resid, categorical_feature=cats)
    return model


def predict_sigma(
    model: object, pred: np.ndarray, metal: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    X, _ = difficulty_design(pred, metal, cell)
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.05, None)


# ---------------------------------------------------------------------------
# Cross-conformal
# ---------------------------------------------------------------------------
def run_protocol(
    y: np.ndarray,
    pred: np.ndarray,
    fold: np.ndarray,
    metal: np.ndarray,
    cell: np.ndarray,
    cohort_k: np.ndarray,
    alpha: float = ALPHA,
    extra_variants: Sequence[str] = (),
) -> Dict[str, object]:
    """Cross-conformal intervals.

    ``extra_variants`` adds interval families without changing the three that
    the published artifact reports.  UP-6 uses it because the coverage failure
    is stratified by **cohort size** (rows whose ligand+cell pair was measured
    with two metals: n=851, coverage 0.8096) and not by metal, which is why the
    existing ``metal`` Mondrian does nothing.
    """
    n = len(y)
    names = ("global", "metal", "difficulty") + tuple(extra_variants)
    half = {name: np.full(n, np.nan) for name in names}
    pit = {name: np.full(n, np.nan) for name in half}

    for k in sorted(set(fold[fold >= 0].tolist())):
        test = fold == k
        cal = (fold >= 0) & ~test
        if test.sum() == 0 or cal.sum() < 50:
            continue
        resid_cal = np.abs(y[cal] - pred[cal])
        resid_te = np.abs(y[test] - pred[test])

        # 1) fixed-width conformal
        q = conformal_quantile(resid_cal, alpha)
        half["global"][test] = q
        pit["global"][test] = _pit(resid_cal, resid_te)

        # 2) Mondrian by metal
        metal_q = {}
        for m in np.unique(metal[cal]):
            mask = metal[cal] == m
            metal_q[m] = (
                conformal_quantile(resid_cal[mask], alpha)
                if mask.sum() >= MIN_METAL_CAL
                else q
            )
        h_metal = np.array([metal_q.get(m, q) for m in metal[test]])
        half["metal"][test] = h_metal
        # PIT for the Mondrian variant is computed against the same per-metal
        # calibration pool that set the width; mixing in the global pool would
        # make the histogram disagree with the interval it is auditing.
        pit_m = np.full(int(test.sum()), np.nan)
        te_metals = metal[test]
        for m in np.unique(te_metals):
            m_te = te_metals == m
            m_cal = metal[cal] == m
            if m_cal.sum() >= MIN_METAL_CAL:
                pit_m[m_te] = _pit(resid_cal[m_cal], resid_te[m_te])
            else:
                pit_m[m_te] = _pit(resid_cal, resid_te[m_te])
        pit["metal"][test] = pit_m

        # 3) difficulty-scaled, then Mondrian on the normalised score
        diff = fit_difficulty(pred[cal], resid_cal, metal[cal], cell[cal])
        sig_cal = predict_sigma(diff, pred[cal], metal[cal], cell[cal])
        sig_te = predict_sigma(diff, pred[test], metal[test], cell[test])
        scores = resid_cal / sig_cal
        score_te = resid_te / sig_te
        q_d = conformal_quantile(scores, alpha)
        q_d_metal = {}
        for m in np.unique(metal[cal]):
            mask = metal[cal] == m
            q_d_metal[m] = (
                conformal_quantile(scores[mask], alpha)
                if mask.sum() >= MIN_METAL_CAL
                else q_d
            )
        q_te = np.array([q_d_metal.get(m, q_d) for m in metal[test]])
        half["difficulty"][test] = q_te * sig_te
        pit["difficulty"][test] = _pit(scores, score_te)

        # 4) Mondrian on cohort size — the stratum that actually fails
        q_k = None
        if "cohort" in extra_variants or "cohort_metal" in extra_variants:
            q_k = {}
            for kk in np.unique(cohort_k[cal]):
                mask = cohort_k[cal] == kk
                q_k[kk] = (
                    conformal_quantile(resid_cal[mask], alpha)
                    if mask.sum() >= MIN_COHORT_CAL
                    else q
                )
        if "cohort" in extra_variants:
            half["cohort"][test] = np.array(
                [q_k.get(kk, q) for kk in cohort_k[test]]
            )
            pit_c = np.full(int(test.sum()), np.nan)
            te_k = cohort_k[test]
            for kk in np.unique(te_k):
                m_te = te_k == kk
                m_cal = cohort_k[cal] == kk
                if m_cal.sum() >= MIN_COHORT_CAL:
                    pit_c[m_te] = _pit(resid_cal[m_cal], resid_te[m_te])
                else:
                    pit_c[m_te] = _pit(resid_cal, resid_te[m_te])
            pit["cohort"][test] = pit_c

        # 5) Mondrian on (cohort size, metal), falling back to cohort then global
        if "cohort_metal" in extra_variants:
            q_cm = {}
            for kk in np.unique(cohort_k[cal]):
                for m in np.unique(metal[cal]):
                    mask = (cohort_k[cal] == kk) & (metal[cal] == m)
                    if mask.sum() >= MIN_METAL_CAL:
                        q_cm[(kk, m)] = conformal_quantile(resid_cal[mask], alpha)
            half["cohort_metal"][test] = np.array(
                [
                    q_cm.get((kk, m), q_k.get(kk, q))
                    for kk, m in zip(cohort_k[test], metal[test])
                ]
            )
            # PIT against the global pool: this variant has no single
            # calibration pool per test row, so the histogram is indicative.
            pit["cohort_metal"][test] = _pit(resid_cal, resid_te)

    evaluated = np.isfinite(half["global"])
    out: Dict[str, object] = {
        "n_evaluated": int(evaluated.sum()),
        "alpha": alpha,
        "point_calibration": point_calibration(y[evaluated], pred[evaluated]),
        "variants": {},
    }
    keys_metal = metal[evaluated]
    keys_k = cohort_k[evaluated]
    for name in half:
        h = half[name][evaluated]
        if not np.isfinite(h).all():
            out["variants"][name] = {"status": "INCOMPLETE"}
            continue
        lo, hi = pred[evaluated] - h, pred[evaluated] + h
        report = coverage_report(
            y[evaluated], pred[evaluated], h, keys_metal, name
        )
        report["winkler"] = winkler_score(y[evaluated], lo, hi, alpha)
        report["pit"] = pit_histogram(pit[name][evaluated])
        report["by_cohort_size"] = coverage_report(
            y[evaluated], pred[evaluated], h, keys_k, name
        )["by_group"]
        out["variants"][name] = report
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_generalization" / "oof.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_calibration.json",
    )
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args(argv)

    data = np.load(args.oof, allow_pickle=False)
    y = data["y"].astype(float)
    metal = data["metal"].astype(str)
    cell = data["cell_line_key"].astype(str)
    frame = pd.DataFrame(
        {
            "ligand_canonical": data["ligand_canonical"].astype(str),
            "cell_line_key": cell,
            "doi": data["doi"].astype(str),
            "time_h": data["time_h"].astype(float),
            "metal": metal,
        }
    )
    ck = cohort_sizes(frame)

    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        key = f"{protocol}__{args.arm}"
        if key not in data:
            protocols[protocol] = {"status": "MISSING_OOF", "key": key}
            continue
        block = run_protocol(
            y,
            data[key].astype(float),
            data[f"{protocol}__fold"].astype(int),
            metal,
            cell,
            ck,
            alpha=args.alpha,
        )
        protocols[protocol] = block
        print(
            f"[met4] {protocol:7s} "
            + " ".join(
                f"{name}: cov={v.get('coverage')} med_w={v.get('median_width')}"
                for name, v in block["variants"].items()
            )
        )

    primary = protocols.get("doi", {}).get("variants", {}).get("difficulty", {})
    cov = primary.get("coverage")
    med_w = primary.get("median_width")
    passed = bool(
        cov is not None
        and abs(cov - (1.0 - args.alpha)) <= COVERAGE_TOL
        and med_w is not None
        and med_w <= WIDTH_GATE
    )
    verdict = "PASS" if passed else "NEGATIVE"

    payload = {
        "artifact": "msrs_calibration",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof),
            "oof_sha256": _sha256(args.oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "split_schema": (
                "consumes MET-2's out-of-fold predictions; the dual-split gate "
                "does not apply"
            ),
            "arm": args.arm,
            "protocols": list(args.protocols),
            "alpha": args.alpha,
            "nominal_coverage": round(1.0 - args.alpha, 4),
            "coverage_tolerance": COVERAGE_TOL,
            "width_gate": WIDTH_GATE,
            "min_metal_calibration_rows": MIN_METAL_CAL,
            "difficulty_model": {
                "features": ["prediction", "metal_code", "cell_line_code"],
                "params": DIFFICULTY_PARAMS,
            },
            "caveat": (
                "base predictions are MET-2 out-of-fold, so for outer fold k the "
                "calibration rows come from models that saw fold k; their "
                "residuals are mildly optimistic, which shrinks q. Coverage is "
                "therefore a lower bound, not an unbiased estimate."
            ),
        },
        "protocols": protocols,
        "verdict": verdict,
        "verdict_rule": (
            "doi protocol, difficulty variant: |coverage - 0.90| <= 0.02 AND "
            "median width <= 1.0 log10"
        ),
        "gate_measurement": {
            "coverage": cov,
            "median_width": med_w,
            "mean_width": primary.get("mean_width"),
            "winkler": primary.get("winkler"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(
        args.out, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[met4] verdict={verdict} coverage={cov} median_width={med_w}")
    print(f"[met4] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
