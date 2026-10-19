"""Tests for the STR-2 fusion plumbing (no training, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molmetal.scripts.msrs_fusion_protocols import (
    IDENTITY,
    foldwise,
    load_seed_files,
    paired_r2_delta,
    stack_seeds,
)


def _write(path: Path, y, pred, fold, doi=None, seed_tag=0.0) -> Path:
    n = len(y)
    arrays = {
        "y": np.asarray(y, dtype=float),
        "metal": np.array(["Pt"] * n),
        "cell_line_key": np.array(["A"] * n),
        "doi": np.asarray(doi if doi is not None else [f"10.1/{i}" for i in range(n)]),
        "time_h": np.array([24.0] * n),
        "ligand_canonical": np.array([f"L{i}" for i in range(n)]),
        "ligand_main_fragment": np.array([f"L{i}" for i in range(n)]),
    }
    for proto in ("ligand", "doi", "scaffold", "year"):
        arrays[f"{proto}__fold"] = np.asarray(fold, dtype=np.int16)
        arrays[f"{proto}__ligand+both"] = np.asarray(pred, dtype=float) + seed_tag
        arrays[f"{proto}__gnn_metal_state"] = np.asarray(pred, dtype=float) + seed_tag
    np.savez_compressed(path, **arrays)
    return path


def test_stack_seeds_averages_predictions(tmp_path):
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.1, 0.9, 2.2, 2.8])
    fold = np.array([0, 0, 1, 1])
    paths = [
        _write(tmp_path / "oof_seed0.npz", y, p, fold, seed_tag=0.0),
        _write(tmp_path / "oof_seed1.npz", y, p, fold, seed_tag=0.2),
    ]
    out = stack_seeds(paths, ("doi",), "ligand+both")
    assert np.allclose(out["doi__pred"], p + 0.1)
    assert out["n_seeds"][0] == 2


def test_stack_seeds_rejects_different_labels(tmp_path):
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.zeros(4)
    fold = np.array([0, 0, 1, 1])
    a = _write(tmp_path / "oof_seed0.npz", y, p, fold)
    b = _write(tmp_path / "oof_seed1.npz", y + 1.0, p, fold)
    with pytest.raises(ValueError, match="y"):
        stack_seeds([a, b], ("doi",), "ligand+both")


def test_stack_seeds_rejects_different_folds(tmp_path):
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.zeros(4)
    a = _write(tmp_path / "oof_seed0.npz", y, p, np.array([0, 0, 1, 1]))
    b = _write(tmp_path / "oof_seed1.npz", y, p, np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="fold"):
        stack_seeds([a, b], ("doi",), "ligand+both")


def test_load_seed_files_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_seed_files(tmp_path, "oof_seed*.npz")


def test_paired_r2_delta_is_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.5, size=200)
    clusters = np.array([f"doi{i // 5}" for i in range(200)])
    out = paired_r2_delta(y, p, p, clusters, n_boot=200)
    assert out["delta_r2"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["n_clusters"] == 40


def test_paired_r2_delta_is_positive_for_a_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    bad = y + rng.normal(scale=1.0, size=400)
    good = y + rng.normal(scale=0.2, size=400)
    clusters = np.array([f"doi{i // 4}" for i in range(400)])
    out = paired_r2_delta(y, bad, good, clusters, n_boot=400)
    assert out["delta_r2"] > 0
    assert out["ci_low"] > 0


def test_paired_r2_delta_drops_non_finite_rows():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    a = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = paired_r2_delta(y, a, b, np.arange(5), n_boot=50)
    assert out["n_rows"] == 4


def test_foldwise_linear_recovers_a_known_combination():
    rng = np.random.default_rng(2)
    n = 300
    tree = rng.normal(size=n)
    gnn = rng.normal(size=n)
    y = 0.7 * tree + 0.3 * gnn + 0.1 + rng.normal(scale=1e-3, size=n)
    folds = np.repeat(np.arange(5), n // 5)
    pred, params = foldwise(y, tree, gnn, folds, "linear")
    assert not np.isnan(pred).any()
    assert len(params) == 5
    assert np.corrcoef(pred, y)[0, 1] > 0.999


def test_foldwise_one_weight_stays_in_the_unit_interval():
    rng = np.random.default_rng(3)
    n = 200
    tree = rng.normal(size=n)
    gnn = rng.normal(size=n)
    y = 0.5 * tree + 0.5 * gnn
    folds = np.repeat(np.arange(4), n // 4)
    pred, params = foldwise(y, tree, gnn, folds, "one_weight")
    assert not np.isnan(pred).any()
    assert all(0.0 <= p["w_tree"] <= 1.0 for p in params)


def test_identity_columns_are_the_merge_keys():
    assert "doi" in IDENTITY and "y" in IDENTITY
