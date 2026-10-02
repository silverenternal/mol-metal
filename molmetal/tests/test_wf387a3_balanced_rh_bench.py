#!/usr/bin/env python3
"""Tests for Wf-387-A3 balanced Rh benchmark scaffold (CPU-only)."""
from __future__ import annotations

import json
import os
import sys

# Ensure project root on sys.path so we can import the script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from molmetal.scripts.wf387a3_balanced_rh_bench import (  # noqa: E402
    _auc_ranksum,
    _kfold_indices,
    _resolve_repo_path,
    _ridge_fit_predict,
    _synthetic_balanced_smiles,
    balanced_rh_aggregate,
    select_per_cohort_config,
    smoke_run_balanced_rh,
    train_eval_balanced,
)


def test_select_per_cohort_config_tiers() -> None:
    """12-d for n<300, 24-d for n<1000 (per Wf-380-A2 §1 matrix)."""
    assert select_per_cohort_config(200)["n_features_expected"] == 12
    assert select_per_cohort_config(299)["n_features_expected"] == 12
    assert select_per_cohort_config(300)["n_features_expected"] == 24
    assert select_per_cohort_config(999)["n_features_expected"] == 24
    assert select_per_cohort_config(1000)["n_features_expected"] == 24
    # alpha matches per-tier config
    assert select_per_cohort_config(200)["alpha"] == 1.0
    assert select_per_cohort_config(500)["alpha"] == 0.5


def test_auc_ranksum_returns_finite() -> None:
    """AUC is in [0, 1] and 0.5 for constant predictions."""
    import numpy as np
    y = np.asarray([0, 0, 1, 1, 1, 0, 1, 0])
    s = np.asarray([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6, 0.4])
    auc = _auc_ranksum(y, s)
    assert 0.0 <= auc <= 1.0
    # Perfect AUC
    s_perfect = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    assert _auc_ranksum(y, s_perfect) == 1.0
    # Constant class returns 0.5
    assert _auc_ranksum(np.asarray([0, 0, 0]), np.asarray([1.0, 2.0, 3.0])) == 0.5


def test_ridge_fit_predict_predicts_shape() -> None:
    """Ridge prediction shape = (n_test,)."""
    import numpy as np
    X_train = np.random.RandomState(0).randn(50, 5)
    y_train = X_train[:, 0] + np.random.RandomState(0).randn(50) * 0.1
    X_test = np.random.RandomState(1).randn(20, 5)
    preds = _ridge_fit_predict(X_train, y_train, X_test, alpha=1.0)
    assert preds.shape == (20,)


def test_kfold_indices_deterministic() -> None:
    """Same seed produces same folds; n_iterations matches."""
    import numpy as np
    folds1 = _kfold_indices(n=100, k=5, seed=42)
    folds2 = _kfold_indices(n=100, k=5, seed=42)
    folds3 = _kfold_indices(n=100, k=5, seed=43)
    assert len(folds1) == 5
    # Deterministic: same seed = same folds
    for (a, b) in zip(folds1, folds2):
        np.testing.assert_array_equal(a[0], b[0])  # train
        np.testing.assert_array_equal(a[1], b[1])  # val
    # Different seed = different folds
    assert not np.array_equal(folds1[0][1], folds3[0][1])


def test_balanced_rh_aggregate_returns_dict() -> None:
    """Aggregator returns mean + std + verdict + n_seeds."""
    result = balanced_rh_aggregate([0.5, 0.6, 0.7, 0.8])
    assert result["n_seeds"] == 4
    assert 0.5 < result["mean"] < 0.8
    assert result["verdict"] in (
        "PARTIAL_ARTIFACT", "REAL_LIFT", "CEILING_BREAK",
    )
    # Edge case: empty
    empty = balanced_rh_aggregate([])
    assert empty["verdict"] == "INSUFFICIENT_DATA"


def test_smoke_run_produces_valid_output() -> None:
    """Smoke test produces a valid JSON result with verdict_compare set."""
    out = smoke_run_balanced_rh(
        out_dir="molmetal/reports/wf387a3_balanced_rh_smoke/",
        balanced_cohort=None,
        n_seeds=2,
        n_folds=3,
        epochs=30,
    )
    out_path = os.path.join(
        "molmetal/reports/wf387a3_balanced_rh_smoke/", "smoke_result.json",
    )
    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    # Verdict taxonomy (per Wf-387-A3 §0)
    assert blob["verdict_compare"] in (
        "ARTIFACT_LIFT", "PARTIAL_ARTIFACT", "REAL_LIFT",
        "CEILING_BREAK", "INSUFFICIENT_DATA",
    )
    # Aggregate stats present
    assert "mean" in blob["aggregate"]
    assert "std" in blob["aggregate"]
    # Wf-387-A1 reference anchor present
    assert blob["wf387a1_synthetic_max"] == 0.8879
    assert blob["wf387a1_synthetic_min"] == 0.8717
    # Honest framing block populated
    assert "scope" in blob["honest_framing"]


def test_train_eval_balanced_synthetic_smoke() -> None:
    """train_eval_balanced returns Dict with per_seed_auc + verdict."""
    # Build inline cohort
    smiles, labs = _synthetic_balanced_smiles(n_active=20, n_inactive=20, seed=42)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "tiny.csv")
        import csv as csvlib
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csvlib.writer(fh)
            writer.writerow(["smiles", "active"])
            for s, y in zip(smiles, labs):
                writer.writerow([s, int(y)])
        result = train_eval_balanced(
            train_csv=csv_path,
            n_seeds=2,
            n_folds=3,
            epochs=30,
        )
        # Validate shape
        assert "per_seed_auc" in result
        assert "verdict" in result
        # Config used should be n_lt_300 tier (40 SMILES)
        assert result["config"]["tier"] == "n_lt_300"
        assert result["config"]["n_features_expected"] == 12


def test_resolve_repo_path_handles_absolute() -> None:
    """Absolute existing path returned unchanged."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake.json")
        with open(path, "w") as fh:
            fh.write("{}")
        assert _resolve_repo_path(path) == path