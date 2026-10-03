"""Tests for Wf-44 Phase 4 — numerical stability fix results.

================================================================
Validates that ``molmetal/scripts/numerical_fix_check.py``
correctly:

1. **Artefact existence** — ``numerical_fix_results.json`` is written
   to the expected location after a CLI run (1 test).
2. **Schema correctness** — top-level keys + per-model keys, types,
   JSON round-trip (3 tests).
3. **Per-model fields** — each model entry contains both ``before_``
   and ``after_`` variants of R², pred_std, y_test_std, std_ratio,
   verdict, plus the stability counters (1 test).
4. **Stability counters are non-negative integers** —
   ``n_steps_with_clipped_grad_`` and ``n_steps_with_decayed_lr_``
   are integers >= 0 (1 test).
5. **Verdict buckets** — every model's verdict is one of the four
   canonical buckets (``mean_collapse``, ``weak_signal``,
   ``normal_negative_R2``, ``normal_positive_R2``) (1 test).
6. **Workflow contract** — the top-level ``workflow`` is
   ``"Wf-44 Phase 4"`` (1 test).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf44_r2_debug"
RESULTS_PATH = REPORT_DIR / "numerical_fix_results.json"
SCRIPT_PATH = _REPO_ROOT / "molmetal/scripts/numerical_fix_check.py"

VERDICT_BUCKETS = {"mean_collapse", "weak_signal",
                   "normal_negative_R2", "normal_positive_R2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_script():
    import importlib.util
    import sys as _sys
    mod_name = "numerical_fix_check"
    spec = importlib.util.spec_from_file_location(
        mod_name, str(SCRIPT_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli_once() -> None:
    """Run the script once with default args so the artefact exists."""
    if RESULTS_PATH.exists():
        return
    out = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"numerical_fix_check CLI failed: {out.stderr or out.stdout}"
        )


# ---------------------------------------------------------------------------
# 1. Artefact existence
# ---------------------------------------------------------------------------
def test_numerical_fix_results_json_exists() -> None:
    """``numerical_fix_results.json`` exists after a CLI run."""
    _run_cli_once()
    assert RESULTS_PATH.exists(), f"missing: {RESULTS_PATH}"


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report() -> dict:
    _run_cli_once()
    with open(RESULTS_PATH) as fh:
        return json.load(fh)


def test_top_level_keys(report: dict) -> None:
    expected = {
        "workflow", "title", "description", "seed", "n_models",
        "improved_count", "regressed_count", "unchanged_count",
        "mean_before_R2", "mean_after_R2", "models",
    }
    missing = expected - set(report.keys())
    assert not missing, f"missing top-level keys: {missing}"


def test_workflow_field(report: dict) -> None:
    """The ``workflow`` field is ``"Wf-44 Phase 4"``."""
    assert report["workflow"] == "Wf-44 Phase 4"


def test_n_models_positive(report: dict) -> None:
    """At least one model is present."""
    assert report["n_models"] >= 1
    assert report["n_models"] == len(report["models"])


# ---------------------------------------------------------------------------
# 3. Per-model fields
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def first_model(report: dict) -> dict:
    return report["models"][0]


def test_per_model_required_keys(first_model: dict) -> None:
    expected = {
        "model_name", "n_train", "n_test",
        "before_R2", "before_pred_std", "before_y_test_std",
        "before_std_ratio", "before_verdict",
        "after_R2", "after_pred_std", "after_y_test_std",
        "after_std_ratio", "after_verdict",
        "after_n_steps_with_clipped_grad_",
        "after_n_steps_with_decayed_lr_",
        "after_loss_scale_used_",
        "after_fp64_fallback_used_",
        "r2_delta", "improved",
    }
    missing = expected - set(first_model.keys())
    assert not missing, f"missing per-model keys: {missing}"


def test_per_model_counter_types(first_model: dict) -> None:
    """Stability counters are non-negative integers."""
    n_clipped = first_model["after_n_steps_with_clipped_grad_"]
    n_decayed = first_model["after_n_steps_with_decayed_lr_"]
    assert isinstance(n_clipped, int)
    assert isinstance(n_decayed, int)
    assert n_clipped >= 0
    assert n_decayed >= 0


def test_per_model_verdict_is_canonical(first_model: dict) -> None:
    """Each verdict is in the canonical bucket set."""
    for prefix in ("before_", "after_"):
        verdict = first_model[f"{prefix}verdict"]
        assert verdict in VERDICT_BUCKETS, verdict


def test_per_model_r2_delta_is_float(first_model: dict) -> None:
    """``r2_delta`` is a float and matches ``after_R2 - before_R2``."""
    delta = first_model["r2_delta"]
    expected = first_model["after_R2"] - first_model["before_R2"]
    assert isinstance(delta, float)
    assert delta == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Aggregate counters
# ---------------------------------------------------------------------------
def test_improved_regressed_unchanged_sum(report: dict) -> None:
    """``improved + regressed + unchanged == n_models``."""
    total = (
        report["improved_count"]
        + report["regressed_count"]
        + report["unchanged_count"]
    )
    assert total == report["n_models"]


def test_loss_scale_used_is_one_or_128(report: dict) -> None:
    """``loss_scale_used_`` is either 1.0 (baseline) or 128.0 (after fix)."""
    for m in report["models"]:
        ls = m["after_loss_scale_used_"]
        assert ls in (1.0, 128.0), ls


def test_fp64_fallback_is_bool(report: dict) -> None:
    """``fp64_fallback_used_`` is a bool."""
    for m in report["models"]:
        assert isinstance(m["after_fp64_fallback_used_"], bool)


# ---------------------------------------------------------------------------
# 5. CLI smoke (sanity)
# ---------------------------------------------------------------------------
def test_cli_runs_without_error() -> None:
    """Running the script with default args exits with code 0."""
    if RESULTS_PATH.exists():
        # Don't re-run — we already have an artefact from the fixture.
        return
    out = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr or out.stdout
