#!/usr/bin/env python3
"""Tests for the STR-4b five-seed aggregation (msrs_feature_expansion_ensemble).

The three things that can silently corrupt the headline number:

1. seed averaging that lets an unscored row (fold = -1) leak a NaN into the mean;
2. stacking per-seed OOF files whose identity columns or folds disagree;
3. a sign accuracy that counts pairs the gate was supposed to exclude.

Each test pins one of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD
from molmetal.scripts.msrs_feature_expansion_ensemble import (
    load_seeds,
    seed_average,
    sign_accuracy,
)

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
N = 6


def _write_seed(directory, index: int, pred_shift: float = 0.0, **overrides) -> None:
    y = np.linspace(1.0, 3.0, N)
    arrays = {
        "y": y,
        "metal": np.array(["Pt"] * N, dtype=object),
        "cell_line_key": np.array(["a"] * N, dtype=object),
        "doi": np.array(["10.1/x", "10.1/x", "10.1/y", "10.1/y", "10.1/z", "10.1/z"], dtype=object),
        "time_h": np.array([24.0] * N, dtype=object),
        "ligand_canonical": np.array([f"L{i}" for i in range(N)], dtype=object),
        "ligand_main_fragment": np.array([f"L{i}" for i in range(N)], dtype=object),
    }
    for protocol in PROTOCOLS:
        fold = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        arrays[f"{protocol}__fold"] = overrides.get(f"{protocol}__fold", fold)
        arrays[f"{protocol}__maccs_rdkit"] = y + pred_shift
    arrays.update(overrides.get("extra", {}))
    np.savez(directory / f"oof_seed{index}.npz", **arrays)


def test_seed_average_ignores_nan_rows():
    preds = np.array([[1.0, 1.2], [np.nan, np.nan], [2.0, np.nan]])
    out = seed_average(preds)
    assert out[0] == pytest.approx(1.1)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)


def test_load_seeds_stacks_per_seed_predictions(tmp_path):
    _write_seed(tmp_path, 0, pred_shift=0.0)
    _write_seed(tmp_path, 1, pred_shift=0.5)
    loaded = load_seeds(tmp_path, "maccs_rdkit")
    preds = loaded["doi__preds"]
    assert preds.shape == (N, 2)
    assert preds[:, 1] - preds[:, 0] == pytest.approx(0.5)
    assert loaded["y"].dtype == np.float64


def test_load_seeds_rejects_mismatched_identity(tmp_path):
    _write_seed(tmp_path, 0)
    other_doi = np.array(["10.1/x", "10.1/x", "10.1/y", "10.1/y", "10.1/z", "10.9/other"], dtype=object)
    _write_seed(tmp_path, 1, extra={"doi": other_doi})
    with pytest.raises(ValueError, match="doi"):
        load_seeds(tmp_path, "maccs_rdkit")


def test_load_seeds_rejects_mismatched_fold(tmp_path):
    _write_seed(tmp_path, 0)
    _write_seed(tmp_path, 1, extra={"doi__fold": np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)})
    with pytest.raises(ValueError, match="doi__fold"):
        load_seeds(tmp_path, "maccs_rdkit")


def test_sign_accuracy_drops_pairs_below_gate_and_unscored_rows():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.2, 2.6, 4.0])
    fold = np.array([0, 0, -1, -1])
    # pair (0,1): |d| = 1.0 >= gate, sign agrees  -> counted
    # pair (2,3): |d| = 1.0 >= gate but row 2 unscored -> excluded
    # pair (0,3): |d| = 3.0 >= gate but row 3 unscored -> excluded
    out = sign_accuracy(y, pred, np.array([0, 0, 2]), np.array([1, 3, 3]), fold)
    assert out["n_pairs"] == 1
    assert out["sign_accuracy"] == pytest.approx(1.0)
    assert out["mae_delta"] == pytest.approx(0.2)


def test_sign_accuracy_ignores_subgate_pairs():
    y = np.array([1.0, 1.05, 2.0, 2.6])
    pred = np.array([1.0, 1.10, 2.0, 2.1])
    fold = np.zeros(4, dtype=int)
    out = sign_accuracy(y, pred, np.array([0, 2]), np.array([1, 3]), fold)
    # pair (0,1): |d| = 0.05 < GATE_THRESHOLD -> dropped
    # pair (2,3): |d| = 0.6 >= GATE_THRESHOLD, predicted +0.1 -> sign agrees
    assert out["n_pairs"] == 1
    assert out["sign_accuracy"] == pytest.approx(1.0)
    assert GATE_THRESHOLD > 0.05
