"""Unit tests for the MSRS code levers (AUD-F1 / AUD-D1 / AUD-M1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse, stats

from molmetal.scripts import msrs_code_levers as L


# ---------------------------------------------------------------------------
# AUD-M1: rank-r interaction block
# ---------------------------------------------------------------------------
class _StubSVD:
    def __init__(self, u: np.ndarray):
        self._u = u

    def transform(self, _X):
        return self._u


def test_interaction_block_is_ligand_embedding_times_metal_indicator():
    # two ligands, embeddings [1, 2] and [3, 4]; metal one-hot rows for Ru and Ir
    lig = sparse.csr_matrix(np.zeros((2, 5)))
    met = sparse.csr_matrix(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32))
    block = L.interaction_block(lig, met, _StubSVD(np.array([[1.0, 2.0], [3.0, 4.0]])))
    assert block.shape == (2, 10)  # rank 2 x 5 metals
    dense = block.toarray()
    # row 0: u=[1,2], metal=Ru -> [1,0,0,0,0, 2,0,0,0,0]
    assert dense[0].tolist() == [1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    # row 1: u=[3,4], metal=Ir -> [0,3,0,0,0, 0,4,0,0,0]
    assert dense[1].tolist() == [0, 3, 0, 0, 0, 0, 4, 0, 0, 0]


def test_interaction_block_uses_only_the_metal_onehot_columns():
    lig = sparse.csr_matrix(np.zeros((1, 3)))
    # 5 one-hot + 3 descriptors; the descriptors must not leak into the product
    met = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0, 9, 9, 9]], dtype=np.float32))
    block = L.interaction_block(lig, met, _StubSVD(np.array([[2.0]])))
    assert block.toarray().tolist() == [[0, 2, 0, 0, 0]]


# ---------------------------------------------------------------------------
# AUD-D1: censored-normal objective
# ---------------------------------------------------------------------------
class _StubDataset:
    def __init__(self, y):
        self._y = np.asarray(y, dtype=float)

    def get_label(self):
        return self._y


def test_tobit_gradient_for_exact_rows_is_squared_error():
    sigma = 0.5
    obj = L.tobit_objective(np.array([False, False]), sigma)
    y = np.array([1.0, 1.0])
    pred = np.array([0.5, 1.5])
    grad, hess = obj(pred, _StubDataset(y))
    assert grad.tolist() == pytest.approx([(0.5 - 1.0) / sigma ** 2, (1.5 - 1.0) / sigma ** 2])
    assert hess.tolist() == pytest.approx([1 / sigma ** 2, 1 / sigma ** 2])


def test_tobit_gradient_for_censored_rows_pushes_up_only():
    sigma = 0.5
    obj = L.tobit_objective(np.array([True, True, True]), sigma)
    bound = np.array([2.0, 2.0, 2.0])
    # predicting below the bound, at the bound, and above it
    grad, _ = obj(np.array([1.0, 2.0, 3.0]), _StubDataset(bound))
    # LightGBM steps against the gradient, so a negative gradient raises mu
    assert (grad <= 0).all()
    # the further below the bound, the harder the push up
    assert abs(grad[0]) > abs(grad[1]) > abs(grad[2])


def test_tobit_gradient_vanishes_above_the_bound():
    obj = L.tobit_objective(np.array([True]), 0.5)
    grad, _ = obj(np.array([6.0]), _StubDataset([2.0]))
    assert abs(float(grad[0])) < 1e-3


def test_tobit_gradient_is_finite_far_below_the_bound():
    obj = L.tobit_objective(np.array([True]), 0.5)
    grad, hess = obj(np.array([-50.0]), _StubDataset([2.0]))
    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    assert grad[0] < 0


def test_tobit_matches_the_analytic_inverse_mills_ratio():
    sigma = 0.7
    y, pred = 2.0, 1.4
    z = (y - pred) / sigma
    expected = -stats.norm.pdf(z) / (sigma * stats.norm.sf(z))
    obj = L.tobit_objective(np.array([True]), sigma)
    grad, _ = obj(np.array([pred]), _StubDataset([y]))
    assert grad[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def test_folds_do_not_split_a_ligand():
    groups = np.repeat(np.arange(40), 3)
    fold_id = L.folds_for(groups)
    assert (fold_id >= 0).all()
    assert len(set(fold_id.tolist())) == L.N_FOLDS
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g].tolist())) == 1


# ---------------------------------------------------------------------------
# Decision-layer structure check
# ---------------------------------------------------------------------------
def _cohort_frame(pred_by_ligand):
    rows = []
    for lig, (p_ru, p_ir) in pred_by_ligand.items():
        rows.append({"cohort": lig, "metal": "Ru", "fold": 0, "y": 1.0, "s": p_ru})
        rows.append({"cohort": lig, "metal": "Ir", "fold": 0, "y": 1.1, "s": p_ir})
    return __import__("pandas").DataFrame(rows)


def test_order_diversity_detects_a_constant_order():
    frame = _cohort_frame({0: (0.1, 0.9), 1: (0.2, 0.8), 2: (0.3, 0.7)})
    out = L.order_diversity(frame, "s")
    assert out["min_distinct_orders"] == 1
    assert out["constant_within_fold"] is True


def test_order_diversity_detects_a_ligand_dependent_order():
    frame = _cohort_frame({0: (0.1, 0.9), 1: (0.9, 0.1), 2: (0.3, 0.7)})
    out = L.order_diversity(frame, "s")
    assert out["min_distinct_orders"] == 2
    assert out["constant_within_fold"] is False


def test_order_diversity_ignores_exact_ties():
    # a tie is decided by row order, not by the model, so it must not count
    frame = _cohort_frame({0: (0.5, 0.5), 1: (0.5, 0.5)})
    out = L.order_diversity(frame, "s")
    assert out["n_tied_cohorts"] == 2
    assert out["min_distinct_orders"] == 0


# ---------------------------------------------------------------------------
# RDKit block
# ---------------------------------------------------------------------------
def test_rdkit_block_is_nan_for_unparseable_smiles():
    out = L.rdkit_block(["CCO", "not-a-smiles"])
    assert out.shape == (2, L.RDKIT_DIM)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_rdkit_block_caches_by_smiles(monkeypatch):
    calls = {"n": 0}
    import molmetal.data.rdkit_2d_extended as R

    def fake(smi):
        calls["n"] += 1
        return np.ones(R.RDKIT_2D_DIM, dtype=np.float32)

    monkeypatch.setattr(R, "compute_rdkit_2d_extended", fake)
    L._RDKIT_CACHE.clear()
    L.rdkit_block(["CCO", "CCO", "CCC"])
    assert calls["n"] == 2
