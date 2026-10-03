"""Tests for Wf-32 Phase 2 — R19 m19 Multi-metal LOMO cross-validation.

These tests are read-only / opt-in.  They verify the Wf-32 Phase 2
deliverables produced by ``molmetal/scripts/train_multimetal_lomo.py``:

* ``metrics/by_metric/multimetal_lomo_auc.json`` — per-fold
  ``test_auc_per_seed`` (5 seeds) + ``per_fold_mean_auc`` +
  ``per_fold_std_auc`` + headline ``mean_test_auc``,
* ``metrics/by_metric/multimetal_lomo_auc_summary.json`` — slim
  final report consumed by ``paper §4 results`` and
  ``R19_PLAN.md``.

All tests skip when the Phase 2 outputs are missing (so CI without
the prior 25-run sweep is not broken).  When present, they assert:

1.  the JSON parses,
2.  the recipe tag is set to ``r19_wf32_m19_multimetal_lomo_*``,
3.  5 folds × 5 seeds = 25 runs recorded,
4.  each fold has a finite ``per_fold_mean_auc`` and ``per_fold_std_auc``,
5.  ``mean_test_auc`` is finite and inside [0, 1],
6.  the headline tag is one of the documented init modes
    (``tmqm`` / ``se3sb`` / ``hilbert``),
7.  the slim summary aligns with the full metrics file.

Count target: ~5 distinct checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "multimetal_lomo_auc.json"
SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "multimetal_lomo_auc_summary.json"
)

LOMO_FOLDS = ("Ru", "Ir", "Rh", "Re", "Os")
N_FOLDS = 5
N_SEEDS = 5
N_RUNS = N_FOLDS * N_SEEDS  # 25
INIT_CHOICES = ("tmqm", "se3sb", "hilbert")


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"WF-32 Phase 2 output missing: {path}")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests — metrics.json shape
# ---------------------------------------------------------------------------
def test_wf32_lomo_metrics_well_formed() -> None:
    """multimetal_lomo_auc.json must parse and declare the Wf-32 recipe."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    assert isinstance(payload, dict)
    recipe = payload.get("recipe", "")
    assert recipe.startswith(
        "r19_wf32_m19_multimetal_lomo"
    ), f"unexpected recipe tag {recipe!r}"


def test_wf32_lomo_metrics_25_runs() -> None:
    """5 folds × 5 seeds = 25 runs must be recorded."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    per_run = payload.get("per_run_auc") or payload.get("test_auc_per_seed")
    # Accept either flat list (N_RUNS elements) or per-fold nested list.
    if isinstance(per_run, list):
        if per_run and isinstance(per_run[0], list):
            flat = [v for sub in per_run for v in sub]
            assert len(flat) == N_RUNS, f"expected {N_RUNS} runs, got {len(flat)}"
        else:
            assert len(per_run) == N_RUNS, f"expected {N_RUNS} runs, got {len(per_run)}"
    elif isinstance(per_run, dict):
        # Per-fold dict of seeds
        assert set(per_run.keys()) == set(
            LOMO_FOLDS
        ), f"per-fold dict keys = {sorted(per_run.keys())}"
        for fold, seeds in per_run.items():
            assert (
                len(seeds) == N_SEEDS
            ), f"fold {fold} has {len(seeds)} seeds (expected {N_SEEDS})"
    else:
        pytest.fail(f"unexpected per_run_auc shape: {type(per_run).__name__}")


def test_wf32_lomo_metrics_per_fold_mean_finite() -> None:
    """Each LOMO fold must have a finite per_fold_mean_auc and std."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    per_fold_mean = payload.get("per_fold_mean_auc", {})
    per_fold_std = payload.get("per_fold_std_auc", {})
    assert set(per_fold_mean.keys()) == set(
        LOMO_FOLDS
    ), f"per_fold_mean keys = {sorted(per_fold_mean.keys())}"
    for fold, mean in per_fold_mean.items():
        assert math.isfinite(mean), f"{fold} mean not finite"
        assert 0.0 <= mean <= 1.0, f"{fold} mean outside [0, 1]: {mean}"
        std = per_fold_std.get(fold)
        assert std is not None and math.isfinite(
            std
        ), f"{fold} std missing or not finite: {std}"
        assert std >= 0, f"{fold} std negative: {std}"


def test_wf32_lomo_metrics_headline_finite() -> None:
    """Headline mean_test_auc must be finite and inside [0, 1]."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    headline = payload.get("mean_test_auc") or payload.get("wf32_headline_value")
    assert headline is not None, "mean_test_auc headline missing"
    assert math.isfinite(headline), f"headline not finite: {headline!r}"
    assert 0.0 <= headline <= 1.0, f"headline outside [0, 1]: {headline!r}"


def test_wf32_lomo_metrics_init_choice_valid() -> None:
    """The reported init must be one of the documented LOMO init modes."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    init = payload.get("init") or payload.get("wf32_init")
    if init is None:
        pytest.skip("init field not recorded in this metrics file")
    assert (
        init in INIT_CHOICES
    ), f"unexpected init choice {init!r} (must be one of {INIT_CHOICES})"


def test_wf32_lomo_summary_well_formed() -> None:
    """multimetal_lomo_auc_summary.json must parse and align with metrics."""
    _skip_if_missing(SUMMARY_PATH)
    summary = _load(SUMMARY_PATH)
    assert isinstance(summary, dict)
    recipe = summary.get("recipe", "")
    assert recipe.startswith(
        "r19_wf32_m19_multimetal_lomo"
    ), f"unexpected summary recipe tag {recipe!r}"
    metrics = _load(METRICS_PATH)
    if metrics is not None:
        for key in ("per_fold_mean_auc",):
            assert summary.get(key) == metrics.get(
                key
            ), f"summary[{key}] does not match metrics[{key}]"
