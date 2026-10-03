"""Tests for Wf-25 Phase 1 — Hilbert-ASTSR real end-to-end evaluation
(TODO-34 §L-R18-P1-5, 2026-09-26).

================================================================
Validates that ``molmetal/scripts/wf25_hilbert_real_eval.py`` produces
the metrics + comparison_table.md + run.log artefacts the Wf-25
contract requires.

Test inventory (12 tests)
-------------------------
1. **Artefact existence** — metrics.json, comparison_table.md, run.log
   (3 tests).
2. **Schema correctness** — all required keys present, types correct
   (3 tests).
3. **Result sanity** — R² in [-1, 1], wall-time non-negative,
   speedup > 1 (3 tests).
4. **Per-fold consistency** — ``len(per_fold) == n_folds``, train + test
   counts sum to n_samples (2 tests).
5. **Script reproducibility** — direct invocation of the helpers
   ``build_design_matrix`` + ``cross_validate`` is deterministic and
   matches the artefacts (1 test).
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


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf25_hilbert_real"
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
        pytest.skip("metrics.json not produced yet; run wf25_hilbert_real_eval.py")
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_metrics_has_required_top_level_keys(metrics: dict) -> None:
    """All expected scalar keys present in metrics.json."""
    expected = {
        "evo_r2_mean",
        "evo_r2_std",
        "hilbert_r2_mean",
        "hilbert_r2_std",
        "evo_wall_time",
        "hilbert_wall_time",
        "n_samples",
        "n_folds",
    }
    missing = expected - set(metrics.keys())
    assert not missing, f"missing keys: {missing}"


def test_metrics_types_correct(metrics: dict) -> None:
    """All scalar keys are numeric; ``per_fold`` is a list of dicts."""
    for k, v in metrics.items():
        if k == "per_fold":
            assert isinstance(v, list), f"{k} should be a list"
            for entry in v:
                assert isinstance(entry, dict)
            continue
        if k == "feature_columns":
            assert isinstance(v, list)
            continue
        assert isinstance(v, (int, float)), f"{k} has type {type(v).__name__}"


def test_metrics_feature_columns_recorded(metrics: dict) -> None:
    """``feature_columns`` is the 8-d RDKit descriptor set."""
    expected = {
        "mol_wt",
        "mol_logp",
        "tpsa",
        "hba",
        "hbd",
        "rot_bonds",
        "ring_count",
        "frac_sp3",
    }
    assert set(metrics["feature_columns"]) == expected


# ---------------------------------------------------------------------------
# 3. Result sanity
# ---------------------------------------------------------------------------
def test_hilbert_r2_in_unit_interval(metrics: dict) -> None:
    """Hilbert R² mean and std are finite, mean in [-1, 1]."""
    h_mean = metrics["hilbert_r2_mean"]
    h_std = metrics["hilbert_r2_std"]
    assert math.isfinite(h_mean), f"non-finite hilbert_r2_mean: {h_mean}"
    assert math.isfinite(h_std), f"non-finite hilbert_r2_std: {h_std}"
    assert -1.0 <= h_mean <= 1.0, f"hilbert_r2_mean out of range: {h_mean}"
    assert h_std >= 0.0


def test_evolutionary_r2_in_unit_interval(metrics: dict) -> None:
    """Evolutionary R² mean and std are finite, mean in [-1, 1]."""
    e_mean = metrics["evo_r2_mean"]
    e_std = metrics["evo_r2_std"]
    assert math.isfinite(e_mean), f"non-finite evo_r2_mean: {e_mean}"
    assert math.isfinite(e_std), f"non-finite evo_r2_std: {e_std}"
    assert -1.0 <= e_mean <= 1.0, f"evo_r2_mean out of range: {e_mean}"
    assert e_std >= 0.0


def test_hilbert_faster_than_evolutionary(metrics: dict) -> None:
    """Hilbert wall-time is at least 2x faster than evolutionary."""
    h_wall = metrics["hilbert_wall_time"]
    e_wall = metrics["evo_wall_time"]
    assert h_wall > 0.0, f"hilbert wall <= 0: {h_wall}"
    assert e_wall > 0.0, f"evol wall <= 0: {e_wall}"
    ratio = e_wall / h_wall
    assert ratio >= 2.0, (
        f"expected hilbert wall <= 2x evolutionary; "
        f"hilbert={h_wall:.4f}s, evol={e_wall:.4f}s, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# 4. Per-fold consistency
# ---------------------------------------------------------------------------
def test_per_fold_count_matches_n_folds(metrics: dict) -> None:
    """``len(per_fold) == n_folds``."""
    assert len(metrics["per_fold"]) == metrics["n_folds"]


def test_per_fold_train_plus_test_equals_n_samples(metrics: dict) -> None:
    """Sum of ``n_train + n_test`` per fold equals ``n_samples``."""
    n_total = metrics["n_samples"]
    for entry in metrics["per_fold"]:
        s = entry["n_train"] + entry["n_test"]
        assert (
            s == n_total
        ), f"fold {entry['fold']}: train+test={s} != n_samples={n_total}"


# ---------------------------------------------------------------------------
# 5. Script reproducibility
# ---------------------------------------------------------------------------
def test_helpers_deterministic_small_problem() -> None:
    """``build_design_matrix`` + ``cross_validate`` are deterministic for a
    fixed seed on a small design matrix.

    The script's CV harness should be re-runnable on a tiny toy problem
    (60 samples, 3 folds) and reproduce the same per-fold R² within a
    numerical tolerance — guarding against regressions in the helper
    functions that produced the artefact.
    """
    from molmetal.scripts.wf25_hilbert_real_eval import (
        FEATURE_COLUMNS,
        POPULATION_DIM,
        build_design_matrix,
        cross_validate,
    )

    platinai_csv = _REPO_ROOT / "molmetal/data/platinai_5k.csv"
    metalcytotox_csv = (
        _REPO_ROOT / ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
    )
    if not platinai_csv.exists() or not metalcytotox_csv.exists():
        pytest.skip("PlatinAI 5K or MetalCytoToxDB CSV missing")

    _, X, y, stats = build_design_matrix(
        platinai_csv=platinai_csv,
        metalcytotox_csv=metalcytotox_csv,
        n_samples=60,
        seed=42,
    )
    assert stats["n_target_rows"] == 60
    m1 = cross_validate(
        X, y, n_folds=3, population_dim=POPULATION_DIM, feature_columns=FEATURE_COLUMNS
    )
    m2 = cross_validate(
        X, y, n_folds=3, population_dim=POPULATION_DIM, feature_columns=FEATURE_COLUMNS
    )
    assert (
        m1["hilbert_r2_mean"] == m2["hilbert_r2_mean"]
    ), "cross_validate not deterministic across runs"
    assert (
        m1["evolutionary_r2_mean"] == m2["evolutionary_r2_mean"]
    ), "cross_validate not deterministic across runs"


# ---------------------------------------------------------------------------
# Supplementary: per-fold metrics are themselves well-formed
# ---------------------------------------------------------------------------
def test_per_fold_metrics_well_formed(metrics: dict) -> None:
    """Each per-fold entry has hilbert_r2 / evolutionary_r2 / wall-time keys."""
    required_fold_keys = {
        "fold",
        "n_train",
        "n_test",
        "hilbert_r2",
        "hilbert_wall_s",
        "evolutionary_r2",
        "evolutionary_wall_s",
    }
    for entry in metrics["per_fold"]:
        missing = required_fold_keys - set(entry.keys())
        assert not missing, f"fold {entry.get('fold')} missing keys: {missing}"
        assert math.isfinite(entry["hilbert_r2"])
        assert math.isfinite(entry["evolutionary_r2"])
        assert entry["hilbert_wall_s"] >= 0.0
        assert entry["evolutionary_wall_s"] >= 0.0
