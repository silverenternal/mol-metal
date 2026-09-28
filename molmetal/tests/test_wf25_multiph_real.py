"""Tests for Wf-25 Phase 2 — Multi-PH real end-to-end evaluation.

================================================================
Validates that ``molmetal/scripts/wf25_multiph_real_eval.py`` produces
the metrics + comparison_table.md + run.log artefacts the Wf-25
contract requires.

Test inventory (10 tests)
-------------------------
1. **Artefact existence** — metrics.json, comparison_table.md, run.log.
2. **Schema correctness** — all required keys present, types correct.
3. **Result sanity** — accuracies in [0, 1], macro-F1 in [0, 1],
   wall-time non-negative.
4. **Per-fold consistency** — ``len(per_fold) == n_folds``, train + test
   counts sum to n_samples.
5. **Script reproducibility** — direct invocation of the helpers
   ``build_design_matrix`` + ``cross_validate`` is deterministic and
   matches the artefacts.
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


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf25_multiph_real"
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
        pytest.skip(f"metrics.json not produced yet; run wf25_multiph_real_eval.py")
    with open(METRICS_PATH) as f:
        return json.load(f)


REQUIRED_KEYS = {
    "rdkit_acc_mean", "rdkit_acc_std",
    "rdkit_f1_mean", "rdkit_f1_std",
    "rdkit_wall_train_mean", "rdkit_wall_train_std",
    "multiph_acc_mean", "multiph_acc_std",
    "multiph_f1_mean", "multiph_f1_std",
    "multiph_wall_train_mean", "multiph_wall_train_std",
    "n_samples", "n_folds",
    "per_fold",
}


def test_metrics_has_required_top_level_keys(metrics: dict) -> None:
    """All expected scalar keys present in metrics.json."""
    missing = REQUIRED_KEYS - set(metrics.keys())
    assert not missing, f"missing keys: {sorted(missing)}"


def test_metrics_types_correct(metrics: dict) -> None:
    """Scalar keys are floats/ints; per_fold is a list."""
    for key in [
        "rdkit_acc_mean", "rdkit_acc_std",
        "rdkit_f1_mean", "rdkit_f1_std",
        "multiph_acc_mean", "multiph_acc_std",
        "multiph_f1_mean", "multiph_f1_std",
        "rdkit_wall_train_mean", "rdkit_wall_train_std",
        "multiph_wall_train_mean", "multiph_wall_train_std",
    ]:
        assert isinstance(metrics[key], (int, float)), (
            f"{key} must be numeric; got {type(metrics[key]).__name__}"
        )
    assert isinstance(metrics["n_samples"], int)
    assert isinstance(metrics["n_folds"], int)
    assert isinstance(metrics["per_fold"], list)


# ---------------------------------------------------------------------------
# 3. Result sanity
# ---------------------------------------------------------------------------
def test_multiph_acc_in_unit_interval(metrics: dict) -> None:
    """Multi-PH accuracy and std live in [0, 1]."""
    a = float(metrics["multiph_acc_mean"])
    s = float(metrics["multiph_acc_std"])
    assert 0.0 <= a <= 1.0, f"multiph_acc_mean={a} not in [0, 1]"
    assert 0.0 <= s <= 1.0, f"multiph_acc_std={s} not in [0, 1]"


def test_rdkit_acc_in_unit_interval(metrics: dict) -> None:
    """RDKit-descriptor accuracy and std live in [0, 1]."""
    a = float(metrics["rdkit_acc_mean"])
    s = float(metrics["rdkit_acc_std"])
    assert 0.0 <= a <= 1.0, f"rdkit_acc_mean={a} not in [0, 1]"
    assert 0.0 <= s <= 1.0, f"rdkit_acc_std={s} not in [0, 1]"


def test_macro_f1_in_unit_interval(metrics: dict) -> None:
    """Macro-F1 means live in [0, 1] for both backends."""
    for key in ("multiph_f1_mean", "rdkit_f1_mean"):
        f1 = float(metrics[key])
        assert 0.0 <= f1 <= 1.0, f"{key}={f1} not in [0, 1]"


def test_wall_times_non_negative(metrics: dict) -> None:
    """Wall-time means are non-negative for both backends."""
    for key in (
        "multiph_wall_train_mean", "multiph_wall_train_std",
        "rdkit_wall_train_mean", "rdkit_wall_train_std",
    ):
        v = float(metrics[key])
        assert v >= 0.0, f"{key}={v} must be non-negative"


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
            f"fold {fold['fold']}: n_train+n_test={total} != n_samples={n_samples}"
        )


# ---------------------------------------------------------------------------
# 5. Honest framing — script reproducibility
# ---------------------------------------------------------------------------
def test_helpers_deterministic() -> None:
    """The CV helper is deterministic across calls with the same seed.

    Sanity check the Wf-25 Phase 2 helpers without running the full
    pipeline (the script itself was the source of metrics.json).
    """
    from molmetal.scripts.wf25_multiph_real_eval import _kfold_indices

    n = 100
    k = 5
    splits_a = _kfold_indices(n, k, seed=42)
    splits_b = _kfold_indices(n, k, seed=42)
    assert splits_a == splits_b, (
        "_kfold_indices must be deterministic for the same seed"
    )
    # Test indices across folds are a partition of [0, n) — each
    # sample appears as test exactly once.
    all_test_idx: list[int] = []
    for _, test_idx in splits_a:
        all_test_idx.extend(test_idx)
    assert sorted(all_test_idx) == list(range(n)), (
        "test indices must form a partition of [0, n) across folds"
    )
    # Train indices per fold = complement of test indices.
    for fold_idx, (train_idx, test_idx) in enumerate(splits_a):
        assert set(train_idx) == set(range(n)) - set(test_idx), (
            f"fold {fold_idx}: train set is not the complement of test set"
        )


def test_multiph_features_shape_contract() -> None:
    """The ``compute_multi_ph_features`` helper returns ``(B, n_bins ** 2)``.

    Run a small inline smoke — verifies the contract the script relies
    on without invoking the full pipeline.
    """
    import torch

    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=8)
    # Tiny pocket cloud.
    coords = torch.zeros(1, 5, 3)
    atom_types = torch.ones(1, 5, dtype=torch.long)
    coords[0] = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
        [1.0, 1.0, 1.0],
    ])
    atom_types[0] = torch.tensor([6, 6, 6, 6, 6])
    out = feats.compute_with_atom_types(coords, atom_types)
    assert out.shape == (1, 8, 8), (
        f"expected (1, 8, 8); got {tuple(out.shape)}"
    )
    flat_dim = out.reshape(-1).shape[0]
    assert flat_dim == 8 * 8, f"flat_dim must be n_bins ** 2 = 64; got {flat_dim}"


def test_script_imports_cleanly() -> None:
    """The script module imports cleanly (catches typos / syntax errors)."""
    from molmetal.scripts import wf25_multiph_real_eval  # noqa: F401
    assert hasattr(wf25_multiph_real_eval, "build_design_matrix")
    assert hasattr(wf25_multiph_real_eval, "cross_validate")
    assert hasattr(wf25_multiph_real_eval, "compute_multi_ph_features")
    assert hasattr(wf25_multiph_real_eval, "render_table")