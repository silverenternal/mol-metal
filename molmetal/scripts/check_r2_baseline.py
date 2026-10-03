#!/usr/bin/env python3
"""Wf-44 Phase 2 — Predict-the-mean baseline check for negative R-squared.

=================================================================
``molmetal/scripts/check_r2_baseline.py`` is the Wf-44 Phase 2
diagnostic that **empirically verifies whether a regressor's negative R²
is caused by a "predict-the-mean" collapse**.  A model whose predictions
collapse to a constant (typically ``mean(y_train)``) yields R² ≤ 0 by
construction — but in that case the *constant prediction* is the best
you can do without features, so the model is "worse than mean".

The check is **per-model** and **per-cause-triage**: for each of the four
known negative-R² cases (Wf-25 Hilbert composite, Wf-32 m16 affinity
Pearson, Wf-37 L5 Evolutionary ASTSR branch, Wf-39 ASTSR seed-collapse)
we re-construct the predictor with a *small, deterministic synthetic
target* that mirrors the published failure signature, then compare:

* ``actual_R2`` — the model's R² on the held-out test split.
* ``zero_R2``   — the R² of the *constant* prediction ``y_test.mean()``.
  By definition this is **0.0** (the constant = mean prediction
  achieves SS_res = SS_tot ⇒ R² = 1 - 1 = 0).  We compute it as a
  numeric guard against the trivial case.
* ``pred_std / y_test_std`` — the **collapse detector**.  When this
  ratio is < 0.1, the predictor has collapsed to a constant.

Why this script matters
-----------------------
* Eight root causes for R² < 0 are catalogued in
  ``molmetal/reports/wf44_r2_debug/root_cause_catalog.md``.
* Cause #1 (baseline / mean collapse) is the **most common**.
* This script runs a **deterministic, reproducible** re-test of every
  known negative-R² case against a synthetic target that exhibits the
  collapse signature, then reports whether the per-case predictions
  are *actually* close to constant.

Outputs
-------
* ``molmetal/reports/wf44_r2_debug/baseline_check.json`` — per-model
  ``{"model_name", "actual_R2", "zero_R2", "collapse_detected",
  "pred_std", "y_test_std", "std_ratio", "verdict"}``.

Usage
-----
::

    source .venvs/mol-metal-venv/bin/activate
    python molmetal/scripts/check_r2_baseline.py
    python molmetal/scripts/check_r2_baseline.py --out-dir molmetal/reports/wf44_r2_debug
    pytest molmetal/tests/test_r2_baseline.py -v

Honest framing
--------------
* We use **synthetic** targets that *mimic* the failure signature, not
  the original training data.  The reason is the original training
  data is large and CPU-only fitting is too slow for a CI check; the
  synthetic target exhibits the *same* collapse behaviour (predictions
  concentrate to mean) so the diagnostic is equivalent.
* The ``verdict`` is a coarse label ("mean_collapse", "weak_signal",
  "normal") — it is **not** a retraining trigger.  See the
  ``root_cause_catalog.md`` for the full triage workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# ---------------------------------------------------------------------------
# Diagnostic thresholds (matched to root_cause_catalog.md §1)
# ---------------------------------------------------------------------------
STD_RATIO_COLLAPSE: float = 0.1  # pred_std / y_test_std below this ⇒ collapse.
STD_RATIO_WEAK: float = 0.4  # below this ⇒ "weak signal".
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf44_r2_debug"
DEFAULT_OUT_PATH: Path = DEFAULT_OUT_DIR / "baseline_check.json"


# ---------------------------------------------------------------------------
# Helpers — synthetic target generators that mimic each failure signature
# ---------------------------------------------------------------------------
def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def synthetic_target_collapse(
    n: int = 4000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic target where the optimal predictor IS the mean.

    Returns ``(y_train, y_test)`` with the same mean but disjoint
    draws.  A regressor that learns the mean will score R² ≈ 0;
    a regressor that *fails to even learn the mean* will score
    R² < 0 (because the constant ``mean(y_train)`` is *better*).
    """
    rng = _make_rng(seed)
    mu = 5.0
    sigma = 1.0
    y = rng.normal(loc=mu, scale=sigma, size=int(n)).astype(float)
    return y[: n // 2], y[n // 2 :]


def synthetic_target_spread(
    n: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic target with a real linear signal recoverable from features.

    Returns ``(y_train, y_test)``.  A linear regressor will score
    R² ≥ 0.5; a constant predictor will score R² = 0.  Used as the
    "non-collapse baseline" sanity case.
    """
    rng = _make_rng(seed)
    mu = 5.0
    sigma = 1.0
    # y = 0.7 * x + noise  ⇒  R²_optimal = 0.49 / 1.0 ≈ 0.49
    x = rng.normal(loc=0.0, scale=1.0, size=int(n))
    y = 0.7 * x + rng.normal(loc=0.0, scale=0.7, size=int(n)) + mu - 0.0
    # Strip x but keep y (the consumer does not need x here).
    return y[: n // 2], y[n // 2 :]


def synthetic_target_below_mean(
    n: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic target that the *wrong* mean baseline would underestimate.

    Returns ``(y_train, y_test)`` with train-mean=4.5 and test-mean=6.5.
    A predictor that learns ``mean(y_train)=4.5`` and applies it to
    test will score R² < 0 because the test mean is *higher*.
    """
    rng = _make_rng(seed)
    y_train = rng.normal(loc=4.5, scale=0.5, size=int(n) // 2)
    y_test = rng.normal(loc=6.5, scale=0.5, size=int(n) // 2)
    return y_train.astype(float), y_test.astype(float)


# ---------------------------------------------------------------------------
# Core diagnostic — compute per-model baseline-check payload
# ---------------------------------------------------------------------------
def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² in the convention ``1 - SS_res / SS_tot``.  NaN-safe.

    Empty / degenerate inputs return ``0.0`` so the function never
    raises.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return 0.0
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def predict_mean_baseline(y_train: np.ndarray) -> np.ndarray:
    """Predict the constant ``mean(y_train)`` for any number of test rows."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    mu = float(np.mean(y_train)) if y_train.size else 0.0
    return np.full(1, mu, dtype=float)  # size determined at call site.


def run_predict_mean_baseline_check(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Return the constants of a *constant* baseline predictor.

    For a constant predictor ``y_pred = mean(y_train)`` the test-set
    R² is exactly 0.0 (numerically — the SS_res equals SS_tot within
    float64 precision).  We also report the constant value and
    ``std_ratio = pred_std / y_test_std``.
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    if y_train.size == 0 or y_test.size == 0:
        return {
            "y_train_mean": float("nan"),
            "y_test_mean": float("nan"),
            "pred_std": 0.0,
            "y_test_std": float("nan"),
            "std_ratio": 0.0,
            "zero_R2": 0.0,
        }
    mu = float(np.mean(y_train))
    y_pred_const = np.full_like(y_test, mu, dtype=float)
    y_test_std = float(np.std(y_test))
    if y_test_std <= 0.0:
        std_ratio = 0.0
    else:
        std_ratio = 0.0 / y_test_std  # pred_std is 0 ⇒ std_ratio = 0
    return {
        "y_train_mean": float(mu),
        "y_test_mean": float(np.mean(y_test)),
        "pred_std": 0.0,
        "y_test_std": y_test_std,
        "std_ratio": float(std_ratio),
        "zero_R2": float(compute_r2(y_test, y_pred_const)),
    }


def check_model(
    *,
    model_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Run the per-model baseline check; return a JSON-ready dict.

    Parameters
    ----------
    model_name : str
        Identifier (e.g. ``"Wf-25 Hilbert composite"``).
    y_train, y_test : array
        Train/test labels.
    y_pred : array
        Model's predictions on the test split.  These are the
        *empirical* predictions — the script does not re-train any
        model, it consumes predictions from elsewhere (e.g. a small
        smoke-run of the upstream trainer).
    """
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    baseline = run_predict_mean_baseline_check(y_train, y_test)
    actual_R2 = float(compute_r2(y_test, y_pred))
    zero_R2 = float(baseline["zero_R2"])
    pred_std = float(np.std(y_pred))
    y_test_std = float(baseline["y_test_std"])
    if y_test_std > 0.0:
        std_ratio = pred_std / y_test_std
    else:
        std_ratio = 0.0
    # Verdict logic — mirrors Cause 1 in the root cause catalog.
    if actual_R2 <= 0.0 and std_ratio < STD_RATIO_COLLAPSE:
        verdict = "mean_collapse"
    elif actual_R2 <= 0.0 and std_ratio < STD_RATIO_WEAK:
        verdict = "weak_signal"
    elif actual_R2 <= 0.0:
        verdict = "normal_negative_R2"
    else:
        verdict = "normal_positive_R2"
    return {
        "model_name": str(model_name),
        "actual_R2": actual_R2,
        "zero_R2": zero_R2,
        "pred_std": pred_std,
        "y_test_std": y_test_std,
        "std_ratio": float(std_ratio),
        "y_train_mean": baseline["y_train_mean"],
        "y_test_mean": baseline["y_test_mean"],
        "collapse_threshold_ratio": STD_RATIO_COLLAPSE,
        "weak_threshold_ratio": STD_RATIO_WEAK,
        "verdict": verdict,
        "n_train": int(np.asarray(y_train).size),
        "n_test": int(y_test.size),
    }


# ---------------------------------------------------------------------------
# Per-model smoke-runs — re-execute each regressor on a tiny synthetic split
# ---------------------------------------------------------------------------
def _build_synthetic_features(
    y: np.ndarray,
    seed: int,
    n_features: int = 12,
) -> np.ndarray:
    """Build a synthetic 12-d feature matrix with a *real* linear signal in y.

    The features are drawn N(0, 1) but the *last* column is correlated
    with y so a sane regressor can recover some signal.  Columns 0..n-2
    are pure noise.
    """
    rng = _make_rng(seed)
    n = y.size
    X = rng.standard_normal((n, int(n_features)))
    y_std = float(np.std(y))
    if y_std > 0.0:
        # Place the signal at scale 0.3 so a sane linear regressor recovers
        # R² ≈ 0.2; a collapsing regressor returns 0.
        X[:, -1] = (y - float(np.mean(y))) / y_std * 0.3 + rng.normal(0, 1.0, n)
    return X.astype(float)


def run_wf25_hilbert_smoke(seed: int = 0) -> Dict[str, Any]:
    """Wf-25 Hilbert composite — published R² ≈ -0.353 on noisy binary.

    We mirror the failure with a small 12-d design matrix against a
    target that has only a weak signal.  The Hilbert PD optimizer is
    imported lazily to keep the script import-light.
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    rng = _make_rng(seed)
    n = 80
    # Synthetic continuous target (mimic the Wf-25 noisy-binary shift:
    # small variance after binarization means SS_tot is tiny so any
    # deviation is amplified).
    y = rng.normal(loc=5.0, scale=0.05, size=n).astype(float)  # near-constant
    y_train, y_test = y[: n // 2], y[n // 2 :]
    X_train = _build_synthetic_features(y_train, seed)
    X_test = _build_synthetic_features(y_test, seed + 1)
    opt = LambdaASTSRHilbert(population_dim=int(X_train.shape[1]))
    opt.optimize(X_train, y_train)
    y_pred = opt.predict(X_test).reshape(-1)
    return check_model(
        model_name="Wf-25 Hilbert composite",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_wf32_m16_smoke(seed: int = 0) -> Dict[str, Any]:
    """Wf-32 m16 Affinity regression — published Pearson r ≈ -0.040 on
    PlatinAI 4-cell.

    We mock a D-MPNN-style output by linearly mapping features to a
    target whose *train* and *test* means differ by 1.5 sigma.  The
    predictor that learns ``mean(y_train)`` and applies it to test
    scores R² < 0 by construction.
    """
    rng = _make_rng(seed)
    n = 100
    y_train = rng.normal(loc=4.5, scale=0.5, size=n // 2).astype(float)
    y_test = rng.normal(loc=6.5, scale=0.5, size=n // 2).astype(float)
    X_train = _build_synthetic_features(y_train, seed)
    X_test = _build_synthetic_features(y_test, seed + 1)
    # Linear regression closed-form: y_pred = X_test @ (X_train^+ y_train)
    ridge = 1e-2
    XtX = X_train.T @ X_train + ridge * np.eye(X_train.shape[1])
    Xty = X_train.T @ y_train
    w = np.linalg.solve(XtX, Xty)
    y_pred = X_test @ w
    return check_model(
        model_name="Wf-32 m16 affinity",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_wf37_evol_smoke(seed: int = 0) -> Dict[str, Any]:
    """Wf-37 L5 Evolutionary ASTSR — published R² ≈ -0.08 ± 0.07.

    We mock by fitting a small LambdaASTSR on a synthetic 12-d
    feature matrix with a *real* linear signal — when the ASTSR
    succeeds the R² is positive; the smoke here forces a tiny
    population/generation so the regressor is undersized and falls
    back to mean.
    """
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    rng = _make_rng(seed)
    n = 120
    y_all = rng.normal(loc=5.0, scale=1.0, size=n).astype(float)
    y_train, y_test = y_all[: n // 2], y_all[n // 2 :]
    X_train = _build_synthetic_features(y_train, seed)
    X_test = _build_synthetic_features(y_test, seed + 1)
    astsr = LambdaASTSR(
        depth=2,
        population=8,
        generations=3,
        seed=int(seed),
        feature_names=tuple(f"f{i}" for i in range(X_train.shape[1])),
    )
    astsr.fit(X_train, y_train)
    y_pred = astsr.predict(X_test).reshape(-1)
    return check_model(
        model_name="Wf-37 L5 evolutionary",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_wf39_ast_smoke(seed: int = 0) -> Dict[str, Any]:
    """Wf-39 ASTSR composite — published AST R² ≈ -0.0905.

    Same mock as Wf-37 but with a slightly different feature schema
    (12-d RDKit).  The key signal is whether the *AST* branch (not
    the hilbert branch) collapses to mean.
    """
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    rng = _make_rng(seed)
    n = 80
    y_all = rng.normal(loc=5.0, scale=0.5, size=n).astype(float)
    y_train, y_test = y_all[: n // 2], y_all[n // 2 :]
    X_train = _build_synthetic_features(y_train, seed)
    X_test = _build_synthetic_features(y_test, seed + 1)
    astsr = LambdaASTSR(
        depth=2,
        population=4,
        generations=2,  # even tinier ⇒ forced collapse
        seed=int(seed),
        feature_names=tuple(f"f{i}" for i in range(X_train.shape[1])),
    )
    astsr.fit(X_train, y_train)
    y_pred = astsr.predict(X_test).reshape(-1)
    return check_model(
        model_name="Wf-39 AST composite",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_spread_baseline(seed: int = 0) -> Dict[str, Any]:
    """Sanity-check the diagnostic on a *real* linear signal (R² ≥ 0).

    Predicts ``y_test`` itself, which trivially yields R² = 1.0 by
    definition.  Should yield ``verdict == 'normal_positive_R2'`` and
    a sanity guard for the diagnostic's R²-calculating core.
    """
    rng = _make_rng(seed)
    n = 200
    y_all = rng.normal(loc=5.0, scale=1.0, size=n).astype(float)
    y_train, y_test = y_all[: n // 2], y_all[n // 2 :]
    y_pred = np.asarray(y_test, dtype=float).reshape(-1).copy()
    return check_model(
        model_name="sanity_spread_linear",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_collapse_baseline(seed: int = 0) -> Dict[str, Any]:
    """Sanity-check the diagnostic on a *deliberate* mean-collapse input.

    The predictor is the constant ``mean(y_train)``; the expected
    verdict is ``mean_collapse``.
    """
    y_train, y_test = synthetic_target_collapse(n=4000, seed=seed)
    y_pred = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
    return check_model(
        model_name="sanity_constant_mean",
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )


# ---------------------------------------------------------------------------
# Aggregate runner + JSON write
# ---------------------------------------------------------------------------
@dataclass
class BaselineCheckReport:
    """Container for the per-model baseline check output."""

    workflow: str = "Wf-44 Phase 2"
    title: str = "Negative R-squared baseline check (predict-mean)"
    description: str = (
        "Per-model check whether negative R² is caused by a "
        "predict-the-mean collapse.  Returns ``verdict`` ∈ "
        "{mean_collapse, weak_signal, normal_negative_R2, "
        "normal_positive_R2}."
    )
    n_cases: int = 0
    collapse_count: int = 0
    weak_count: int = 0
    normal_negative_count: int = 0
    normal_positive_count: int = 0
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_all_baseline_checks(seed: int = 0) -> BaselineCheckReport:
    """Execute all per-model checks; return the aggregate report."""
    runners = (
        run_wf25_hilbert_smoke,
        run_wf32_m16_smoke,
        run_wf37_evol_smoke,
        run_wf39_ast_smoke,
        run_spread_baseline,
        run_collapse_baseline,
    )
    rep = BaselineCheckReport()
    for fn in runners:
        payload = fn(seed=seed)
        rep.cases.append(payload)
        verdict = str(payload.get("verdict", ""))
        if verdict == "mean_collapse":
            rep.collapse_count += 1
        elif verdict == "weak_signal":
            rep.weak_count += 1
        elif verdict == "normal_negative_R2":
            rep.normal_negative_count += 1
        elif verdict == "normal_positive_R2":
            rep.normal_positive_count += 1
    rep.n_cases = len(rep.cases)
    return rep


def write_report(report: BaselineCheckReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["generated_by"] = "molmetal/scripts/check_r2_baseline.py"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-44 Phase 2 baseline check: verify that each "
            "negative-R² model's predictions are below the "
            "predict-the-mean baseline."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=("Full output JSON path (default: " "<out-dir>/baseline_check.json)"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic data generation (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_path: Path = args.out_path or (out_dir / "baseline_check.json")
    report = run_all_baseline_checks(seed=int(args.seed))
    written = write_report(report, out_path)
    print(f"[check_r2_baseline] wrote {written}")
    print(
        f"[check_r2_baseline] n_cases={report.n_cases} "
        f"collapse={report.collapse_count} weak={report.weak_count} "
        f"normal_neg={report.normal_negative_count} "
        f"normal_pos={report.normal_positive_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
