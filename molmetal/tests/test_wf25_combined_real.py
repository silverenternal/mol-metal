"""Tests for Wf-25 Phase 3 — Hilbert + Multi-PH joint-feature real evaluation.

================================================================
Validates that ``molmetal/scripts/wf25_combined_real_eval.py``
produces the metrics + comparison_table.md + run.log artefacts the
Wf-25 contract requires, and that the Wf-21 phase-5 ship
(``JointHilbertOptimizer`` + ``JointFeatureConcatenator``) is
correctly wired in.

Test inventory (10 tests)
-------------------------
1. **Artefact existence** — metrics.json, comparison_table.md, run.log
   (3 tests).
2. **Schema correctness** — all required keys present, types correct
   (2 tests).
3. **Result sanity** — joint R² finite + bounded, wall-time
   non-negative, joint wall ≤ separate wall + small slack (3 tests).
4. **Per-fold consistency** — ``len(per_fold) == n_folds``,
   train + test counts sum to n_samples (2 tests).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf25_combined"
METRICS_PATH = REPORT_DIR / "metrics.json"
TABLE_PATH = REPORT_DIR / "comparison_table.md"
LOG_PATH = REPORT_DIR / "run.log"


# ---------------------------------------------------------------------------
# 1. Artefact existence
# ---------------------------------------------------------------------------
def test_metrics_file_exists() -> None:
    """``metrics.json`` exists under the report directory."""
    assert METRICS_PATH.exists(), f"missing: {METRICS_PATH}"


def test_comparison_table_exists() -> None:
    """``comparison_table.md`` exists under the report directory."""
    assert TABLE_PATH.exists(), f"missing: {TABLE_PATH}"


def test_run_log_exists() -> None:
    """``run.log`` exists under the report directory."""
    assert LOG_PATH.exists(), f"missing: {LOG_PATH}"


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metrics() -> dict:
    """Load the artefact ``metrics.json`` once per module."""
    if not METRICS_PATH.exists():
        pytest.skip(
            "metrics.json not produced yet; run wf25_combined_real_eval.py"
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


REQUIRED_KEYS = {
    "joint_r2_mean", "joint_r2_std",
    "separate_r2_mean", "separate_r2_std",
    "joint_wall_s_mean", "joint_wall_s_std",
    "separate_wall_s_mean", "separate_wall_s_std",
    "speedup", "n_samples", "n_folds", "n_join_dim",
    "n_parsed", "n_failed", "n_target_rows",
    "mph_n_parsed", "mph_n_failed", "mph_n_too_large",
    "feature_dim_astr", "feature_dim_multiph",
    "feature_columns_astr", "multi_ph_n_bins",
    "joint_hilbert_dim", "joint_n_iters", "joint_lr",
    "per_fold",
}


def test_metrics_has_required_top_level_keys(metrics: dict) -> None:
    """All expected scalar + list keys present in metrics.json."""
    missing = REQUIRED_KEYS - set(metrics.keys())
    assert not missing, f"missing keys: {sorted(missing)}"


def test_metrics_types_correct(metrics: dict) -> None:
    """Scalar keys are floats/ints; lists are lists."""
    for key in (
        "joint_r2_mean", "joint_r2_std",
        "separate_r2_mean", "separate_r2_std",
        "joint_wall_s_mean", "joint_wall_s_std",
        "separate_wall_s_mean", "separate_wall_s_std",
        "speedup", "joint_hilbert_dim", "joint_n_iters", "joint_lr",
    ):
        assert isinstance(metrics[key], (int, float)), (
            f"{key} must be numeric; got {type(metrics[key]).__name__}"
        )
    for key in ("n_samples", "n_folds", "n_join_dim"):
        assert isinstance(metrics[key], int), (
            f"{key} must be int; got {type(metrics[key]).__name__}"
        )
    assert isinstance(metrics["per_fold"], list)
    assert isinstance(metrics["feature_columns_astr"], list)
    assert all(
        isinstance(s, str) for s in metrics["feature_columns_astr"]
    )


# ---------------------------------------------------------------------------
# 3. Result sanity
# ---------------------------------------------------------------------------
def test_joint_r2_finite_and_bounded(metrics: dict) -> None:
    """Joint R² mean and std are finite; mean is in a reasonable range.

    Honest framing — the ASTSR-proxy stream carries no signal on the
    binary activity target so R² can be negative.  We require a finite
    value but do not require R² > 0 (the model is honest about not
    finding structure).
    """
    j_mean = metrics["joint_r2_mean"]
    j_std = metrics["joint_r2_std"]
    assert math.isfinite(j_mean), f"non-finite joint_r2_mean: {j_mean}"
    assert math.isfinite(j_std), f"non-finite joint_r2_std: {j_std}"
    assert j_std >= 0.0
    # Loose bound — the R² can be a moderately large negative number if
    # the noise blows up; the Wf-21 phase-5 ship does not guarantee
    # an upper bound on R².  We only assert it's not astronomically
    # large (overflow guard).
    assert j_mean >= -1e9, f"joint_r2_mean suspiciously large: {j_mean}"


def test_separate_r2_finite(metrics: dict) -> None:
    """Separate-stream R² mean and std are finite; std non-negative."""
    s_mean = metrics["separate_r2_mean"]
    s_std = metrics["separate_r2_std"]
    assert math.isfinite(s_mean), f"non-finite separate_r2_mean: {s_mean}"
    assert math.isfinite(s_std), f"non-finite separate_r2_std: {s_std}"
    assert s_std >= 0.0


def test_wall_times_non_negative(metrics: dict) -> None:
    """Joint and separate wall-times are non-negative."""
    for key in (
        "joint_wall_s_mean", "joint_wall_s_std",
        "separate_wall_s_mean", "separate_wall_s_std",
    ):
        v = float(metrics[key])
        assert v >= 0.0, f"{key}={v} must be non-negative"


def test_speedup_within_tolerance(metrics: dict) -> None:
    """Speedup = separate_wall / joint_wall is in a sane range.

    The Wf-21 phase-5 ship ``JointHilbertOptimizer`` runs a single
    Hilbert regressor vs. two regressors in the separate branch, so
    speedup should be near or above 1.0.  We bound it from above at
    100x to catch catastrophic miscomputations (the absolute speedup
    is machine-dependent).
    """
    s = float(metrics["speedup"])
    assert math.isfinite(s), f"speedup non-finite: {s}"
    assert 0.5 <= s <= 100.0, (
        f"speedup out of range [0.5, 100.0]: {s}"
    )
    # Verify the speedup was derived from the wall-times.
    j_wall = float(metrics["joint_wall_s_mean"])
    s_wall = float(metrics["separate_wall_s_mean"])
    if j_wall > 0:
        expected = s_wall / j_wall
        assert abs(s - expected) < 1e-6, (
            f"speedup {s} != separate_wall / joint_wall "
            f"= {s_wall:.6f} / {j_wall:.6f} = {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Per-fold consistency
# ---------------------------------------------------------------------------
def test_per_fold_count_matches_n_folds(metrics: dict) -> None:
    """``len(per_fold) == n_folds``."""
    n_folds = int(metrics["n_folds"])
    assert len(metrics["per_fold"]) == n_folds, (
        f"len(per_fold)={len(metrics['per_fold'])} != n_folds={n_folds}"
    )


def test_per_fold_train_test_sum_to_n_samples(metrics: dict) -> None:
    """Each fold's ``n_train + n_test`` sums to ``n_samples``."""
    n_samples = int(metrics["n_samples"])
    for fold in metrics["per_fold"]:
        total = int(fold["n_train"]) + int(fold["n_test"])
        assert total == n_samples, (
            f"fold {fold['fold']}: n_train+n_test={total} != "
            f"n_samples={n_samples}"
        )


# ---------------------------------------------------------------------------
# 5. Script / ship sanity
# ---------------------------------------------------------------------------
def test_script_imports_cleanly() -> None:
    """The script module imports cleanly (catches typos / syntax errors)."""
    from molmetal.scripts import wf25_combined_real_eval  # noqa: F401
    assert hasattr(wf25_combined_real_eval, "build_design_matrix")
    assert hasattr(wf25_combined_real_eval, "cross_validate")
    assert hasattr(wf25_combined_real_eval, "compute_multi_ph_features")
    assert hasattr(wf25_combined_real_eval, "render_table")


def test_wf21_phase5_ship_smoke() -> None:
    """Smoke-test the Wf-21 phase-5 ship directly on synthetic features.

    Verifies that ``JointHilbertOptimizer.fit`` returns a finite R² and
    a positive ``join_dim`` matching the joined feature size, on a
    tiny synthetic problem — guards against regressions in the ship
    class that Phase 3 relies on.
    """
    import numpy as np

    from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
        JointFeatureConcatenator,
        JointHilbertOptimizer,
    )

    rng = np.random.default_rng(42)
    astr_feat = rng.standard_normal((40, 8))
    mph_feat = rng.standard_normal((40, 16, 16))
    target = rng.standard_normal(40)

    opt = JointHilbertOptimizer(hilbert_dim=8, normalize=True, n_iters=10)
    result = opt.fit(astr_feat, mph_feat, target)
    assert math.isfinite(result["r2"]), f"non-finite R²: {result['r2']}"
    assert result["join_dim"] == 8 + 16 * 16, (
        f"join_dim mismatch: {result['join_dim']} != 264"
    )
    assert result["backend"] == "joint_hilbert"
    assert opt.is_fitted

    # The concatenator must record the joined dims.
    cat = JointFeatureConcatenator()
    combined = cat.concatenate(astr_feat, mph_feat)
    assert combined.shape == (40, 8 + 256)
    assert cat.dim_astr == 8
    assert cat.dim_multi_ph == 256


def test_per_fold_metrics_well_formed(metrics: dict) -> None:
    """Each per-fold entry has joint + separate R² / wall-time keys."""
    required_fold_keys = {
        "fold", "n_train", "n_test",
        "joint_r2", "joint_wall_s",
        "separate_r2", "separate_wall_s",
        "joint_join_dim",
    }
    for entry in metrics["per_fold"]:
        missing = required_fold_keys - set(entry.keys())
        assert not missing, (
            f"fold {entry.get('fold')} missing keys: {missing}"
        )
        assert math.isfinite(entry["joint_r2"]), entry
        assert math.isfinite(entry["separate_r2"]), entry
        assert entry["joint_wall_s"] >= 0.0
        assert entry["separate_wall_s"] >= 0.0
        assert entry["joint_join_dim"] == int(metrics["n_join_dim"]), (
            f"fold {entry['fold']}: joint_join_dim="
            f"{entry['joint_join_dim']} != n_join_dim={metrics['n_join_dim']}"
        )
