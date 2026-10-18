"""Tests for the MSRS-2 CPU surface baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from molmetal.scripts import msrs_surface_model as M


def _descriptors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metal_symbol": ["Ru", "Ir", "Rh", "Os", "Re"],
            "ie1_kj": [710.2, 880.0, 719.7, 840.0, 760.0],
            "ie2_kj": [1620.0, 1600.0, 1740.0, 1600.0, 1260.0],
            "pauling_en": [2.20, 2.20, 2.28, 2.20, 1.90],
        }
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def test_morgan_matrix_shape_and_invalid_smiles():
    X = M.morgan_matrix(["CCO", "not-a-smiles", "c1ccccc1"], n_bits=64, radius=2)
    assert X.shape == (3, 64)
    assert X[1].nnz == 0
    assert X[0].nnz > 0 and X[2].nnz > 0


def test_metal_matrix_layout():
    df = _descriptors()
    X = M.metal_matrix(["Ru", "Rh"], df)
    assert X.shape == (2, 8)
    assert X[0, 0] == 1.0 and X[1, 2] == 1.0
    assert X[0, 5] == pytest.approx(710.2)
    assert X[1, 7] == pytest.approx(2.28)


def test_cell_matrix_unknown_maps_to_other():
    X = M.cell_matrix(["A", "ZZZ", "B"], ["A", "B"])
    assert X.shape == (3, 3)
    assert X[1, 2] == 1.0
    assert X[0, 0] == 1.0 and X[2, 1] == 1.0


def test_build_design_groups_partition_all_features():
    df = pd.DataFrame(
        {
            "ligand_main_fragment": ["CCO", "c1ccccc1"],
            "metal": ["Ru", "Ir"],
            "cell_line_key": ["A549", "OTHER"],
        }
    )
    X, names, groups = M.build_design(df, _descriptors(), ["A549"])
    assert len(names) == X.shape[1]
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(X.shape[1]))
    assert len(flat) == len(set(flat))


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def _pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand_main_fragment": ["L1", "L1", "L2", "L2", "L3"],
            "cell_line_key": ["A", "A", "A", "A", "A"],
            "doi": ["d1", "d1", "d1", "d1", "d1"],
            "time_h": [24.0] * 5,
            "metal": ["Rh", "Ru", "Ru", "Ir", "Ru"],
        }
    )


def test_matched_delta_pairs_are_canonical_and_cross_metal():
    pairs = M.matched_delta_pairs(_pair_frame())
    assert len(pairs) == 2  # L1: Rh-Ru, L2: Ru-Ir; L3 has one metal
    assert set(zip(pairs.metal_a, pairs.metal_b)) == {("Ru", "Rh"), ("Ru", "Ir")}
    for _, r in pairs.iterrows():
        assert r.metal_a != r.metal_b
        assert M.METALS.index(r.metal_a) < M.METALS.index(r.metal_b)


def test_matched_delta_pairs_require_same_cell_and_doi():
    df = _pair_frame()
    df.loc[1, "doi"] = "d2"  # breaks the L1 pair
    df.loc[3, "doi"] = "d3"  # breaks the L2 pair
    pairs = M.matched_delta_pairs(df)
    assert pairs.empty


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_mcnemar_one_sided_known_values():
    assert M._mcnemar_one_sided(0, 0) == 1.0
    assert M._mcnemar_one_sided(0, 10) == 1.0
    assert M._mcnemar_one_sided(10, 0) == pytest.approx(0.5**10)
    assert M._mcnemar_one_sided(5, 5) == pytest.approx(0.6230, abs=1e-4)


def test_holm_is_monotone_and_bounded():
    out = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == pytest.approx(0.03)
    assert out["c"] == pytest.approx(0.06)
    assert out["b"] == pytest.approx(0.06)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_oof_means_do_not_use_test_rows():
    y = np.array([0.0, 2.0, 10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 40.0, 42.0])
    keys = np.array(["A"] * 10)
    groups = np.repeat([f"g{i}" for i in range(5)], 2)
    gmean, kmean = M.oof_means(y, keys, groups)
    expected_first = float(np.mean(y[2:]))  # 26.0
    assert gmean[:2].tolist() == [expected_first, expected_first]
    assert kmean[:2].tolist() == [expected_first, expected_first]
    # No held-out row may be predicted from its own value.
    assert not np.any(np.isclose(gmean, y))


# ---------------------------------------------------------------------------
# Delta-sign report
# ---------------------------------------------------------------------------
def test_delta_sign_report_flags_additive_structure():
    df = _pair_frame()
    pairs = M.matched_delta_pairs(df)
    y = np.array([1.0, 2.0, 2.0, 3.0, 0.0])
    # additive prediction = ligand effect + metal effect, so its delta is constant
    pred_add = np.array([1.0, 2.0, 2.0, 3.0, 0.0])
    pred_int = pred_add.copy()
    fold = np.zeros(len(y), dtype=int)
    out = M.delta_sign_report(y, pred_add, pred_int, pairs, fold, n_boot=50)
    assert out["additive_delta_ligand_independent_given_cell"] is True
    prim = out["by_threshold"]["0.2091"]
    assert prim["additive_accuracy"] == 1.0
    assert prim["interaction_accuracy"] == 1.0
    assert prim["mcnemar_p_one_sided"] == 1.0


def test_delta_sign_report_detects_wrong_interaction_sign():
    df = _pair_frame()
    pairs = M.matched_delta_pairs(df)
    y = np.array([1.0, 2.0, 2.0, 3.0, 0.0])
    pred_add = y.copy()
    pred_int = np.array([2.0, 1.0, 3.0, 2.0, 0.0])  # reverses both deltas
    out = M.delta_sign_report(y, pred_add, pred_int, pairs, np.zeros(len(y), dtype=int), n_boot=50)
    prim = out["by_threshold"]["0.2091"]
    assert prim["additive_accuracy"] == 1.0
    assert prim["interaction_accuracy"] == 0.0
    assert prim["n_additive_right_interaction_wrong"] == 2


# ---------------------------------------------------------------------------
# The structural claim the whole comparison rests on
# ---------------------------------------------------------------------------
def test_interaction_constraints_block_ligand_metal_crosses():
    """A constrained GBDT must not fit a pure ligand x metal interaction."""
    rng = np.random.default_rng(0)
    n, n_bits = 1200, 24
    bits = rng.integers(0, 2, size=(n, n_bits)).astype(np.float32)
    metal = rng.integers(0, 2, size=n)
    X = sparse.hstack(
        [
            sparse.csr_matrix(bits),
            sparse.csr_matrix(np.eye(2, dtype=np.float32)[metal]),
            sparse.csr_matrix(np.ones((n, 1), dtype=np.float32)),
        ],
        format="csr",
    )
    groups = [list(range(n_bits)), [n_bits, n_bits + 1], [n_bits + 2]]
    # y depends only on the cross of bit 0 and metal.
    y = np.where((bits[:, 0] > 0) & (metal == 0), 3.0, 0.0)
    y = np.where((bits[:, 0] > 0) & (metal == 1), -3.0, y)

    pred_add = M.fit_predict(X, y, X, groups)
    pred_int = M.fit_predict(X, y, X, None)
    mae_add = float(np.mean(np.abs(y - pred_add)))
    mae_int = float(np.mean(np.abs(y - pred_int)))
    assert mae_int < 0.5
    # The constrained model can only fit the marginals; it cannot recover the
    # cross, so its error must stay far above the unconstrained model's.
    assert mae_add > 0.5
    assert mae_add > 5.0 * mae_int
