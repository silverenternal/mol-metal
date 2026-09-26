"""Tests for Wf-44 Phase 2 — predict-the-mean baseline check.

================================================================
Validates that ``molmetal/scripts/check_r2_baseline.py`` correctly:

1. **Artefact existence** — ``baseline_check.json`` is written to the
   expected location after a CLI run (1 test).
2. **Schema correctness** — top-level keys + per-case keys, types,
   JSON round-trip (3 tests).
3. **Diagnostic helpers** — ``compute_r2``, ``predict_mean_baseline``,
   ``run_predict_mean_baseline_check``, ``check_model`` (4 tests).
4. **Sanity runners** — spread baseline hits ``normal_positive_R2``,
   collapse baseline hits ``mean_collapse`` (2 tests).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf44_r2_debug"
BASELINE_PATH = REPORT_DIR / "baseline_check.json"
SCRIPT_PATH = _REPO_ROOT / "molmetal/scripts/check_r2_baseline.py"


# ---------------------------------------------------------------------------
# Helpers — import the script module
# ---------------------------------------------------------------------------
def _import_script():
    import importlib.util
    import sys as _sys
    mod_name = "check_r2_baseline"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(_REPO_ROOT / "molmetal/scripts/check_r2_baseline.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register the module BEFORE exec_module so dataclass introspection
    # can find ``cls.__module__`` in ``sys.modules`` (otherwise Python
    # raises ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    # when a dataclass's annotations reference module-level types).
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli_once() -> None:
    """Run the script once with default args so the artefact exists."""
    if BASELINE_PATH.exists():
        return
    out = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"check_r2_baseline CLI failed: {out.stderr or out.stdout}"
        )


# ---------------------------------------------------------------------------
# 1. Artefact existence (CLI run)
# ---------------------------------------------------------------------------
def test_baseline_check_json_exists() -> None:
    _run_cli_once()
    assert BASELINE_PATH.exists(), f"missing: {BASELINE_PATH}"


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline() -> dict:
    _run_cli_once()
    with open(BASELINE_PATH) as fh:
        return json.load(fh)


def test_baseline_top_level_keys(baseline: dict) -> None:
    expected = {
        "workflow", "title", "description", "n_cases",
        "collapse_count", "weak_count",
        "normal_negative_count", "normal_positive_count",
        "cases",
    }
    missing = expected - set(baseline.keys())
    assert not missing, f"missing top-level keys: {missing}"


def test_baseline_cases_have_expected_keys(baseline: dict) -> None:
    expected = {
        "model_name", "actual_R2", "zero_R2",
        "pred_std", "y_test_std", "std_ratio",
        "y_train_mean", "y_test_mean",
        "collapse_threshold_ratio", "weak_threshold_ratio",
        "verdict", "n_train", "n_test",
    }
    for case in baseline["cases"]:
        missing = expected - set(case.keys())
        assert not missing, f"missing case keys: {missing}"


def test_baseline_counts_match_cases(baseline: dict) -> None:
    counts = (
        baseline["collapse_count"]
        + baseline["weak_count"]
        + baseline["normal_negative_count"]
        + baseline["normal_positive_count"]
    )
    assert counts == baseline["n_cases"], (
        f"verdict counts ({counts}) != n_cases ({baseline['n_cases']})"
    )
    assert len(baseline["cases"]) == baseline["n_cases"]


# ---------------------------------------------------------------------------
# 3. Diagnostic helpers — compute_r2 / predict_mean / check_model
# ---------------------------------------------------------------------------
def test_compute_r2_zero_on_constant_prediction() -> None:
    mod = _import_script()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.full_like(y, float(np.mean(y)))
    assert math.isclose(mod.compute_r2(y, y_hat), 0.0, abs_tol=1e-9)


def test_compute_r2_perfect_on_identity() -> None:
    mod = _import_script()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(mod.compute_r2(y, y), 1.0, abs_tol=1e-9)


def test_run_predict_mean_baseline_check_reports_zero_R2() -> None:
    mod = _import_script()
    rng = np.random.default_rng(0)
    # Same distribution for train + test ⇒ train-mean ≈ test-mean so the
    # constant mean(y_train) baseline has SS_res ≈ SS_tot ⇒ zero_R2 ≈ 0.
    # Use n=2000 per split so the means align within 1e-3.
    y_all = rng.normal(5.0, 1.0, 4000)
    y_train, y_test = y_all[:2000], y_all[2000:]
    payload = mod.run_predict_mean_baseline_check(y_train, y_test)
    assert math.isclose(payload["zero_R2"], 0.0, abs_tol=1e-3)
    assert math.isclose(payload["pred_std"], 0.0, abs_tol=1e-12)
    assert math.isclose(payload["std_ratio"], 0.0, abs_tol=1e-12)


def test_check_model_collapse_label_when_constant() -> None:
    mod = _import_script()
    rng = np.random.default_rng(0)
    y_all = rng.normal(5.0, 1.0, 4000)
    y_train, y_test = y_all[:2000], y_all[2000:]
    y_pred = np.full_like(y_test, float(np.mean(y_train)))
    payload = mod.check_model(
        model_name="test_collapse",
        y_train=y_train, y_test=y_test, y_pred=y_pred,
    )
    assert payload["verdict"] == "mean_collapse"
    assert payload["std_ratio"] == 0.0
    # Same-distribution train/test ⇒ zero_R2 ≈ 0 within float64 noise.
    assert math.isclose(payload["zero_R2"], 0.0, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# 4. Sanity runners — spread baseline positive; collapse baseline collapse
# ---------------------------------------------------------------------------
def test_sanity_spread_baseline_is_positive_R2() -> None:
    mod = _import_script()
    payload = mod.run_spread_baseline(seed=0)
    assert payload["verdict"] == "normal_positive_R2", (
        f"spread baseline should yield positive R²; got {payload['verdict']} "
        f"R²={payload['actual_R2']:+.4f}"
    )
    assert payload["actual_R2"] > 0.0


def test_sanity_collapse_baseline_is_mean_collapse() -> None:
    mod = _import_script()
    payload = mod.run_collapse_baseline(seed=0)
    assert payload["verdict"] == "mean_collapse"
    # zero_R2 = R² of (constant = mean(y_train)) on y_test; equals 0.0
    # exactly when y_train_mean = y_test_mean.  The synthetic generator
    # uses the same mean for both splits, so within float64 noise the
    # constant-prediction R² is ~ 0.
    assert math.isclose(payload["zero_R2"], 0.0, abs_tol=1e-3)
    assert math.isclose(payload["pred_std"], 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 5. CLI exit code
# ---------------------------------------------------------------------------
def test_cli_exits_zero() -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--seed", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, (
        f"CLI exited with {out.returncode}: {out.stderr or out.stdout}"
    )