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


# ---------------------------------------------------------------------------
# AUD-D1b: two-part censoring
# ---------------------------------------------------------------------------
def test_truncated_mean_is_always_above_the_bound():
    mu = np.array([-2.0, 0.0, 1.0, 1.9])
    bound = np.full_like(mu, 2.0)
    out = L.truncated_mean(mu, bound, sigma=0.5)
    assert (out >= bound).all()
    # The conditional mean rises with mu.  It does not fall with distance below
    # the bound: as z -> infinity it converges to the bound from above, so a
    # monotonicity claim in the other direction would be wrong.
    assert out[0] < out[1] < out[2] < out[3]


def test_truncated_mean_matches_the_inverse_mills_identity():
    mu, bound, sigma = 1.0, 2.0, 0.7
    z = (bound - mu) / sigma
    expected = mu + sigma * stats.norm.pdf(z) / stats.norm.sf(z)
    assert L.truncated_mean(np.array([mu]), np.array([bound]), sigma)[0] == pytest.approx(expected)


def test_truncated_mean_converges_to_mu_far_above_the_bound():
    out = L.truncated_mean(np.array([10.0]), np.array([2.0]), 0.5)
    assert out[0] == pytest.approx(10.0, abs=1e-6)


def test_native_codes_appends_an_unseen_metal_as_a_new_level():
    import pandas as pd

    df = pd.DataFrame({"metal": ["Ru", "Pt", "Ir"], "cell_line_key": ["A", "A", "B"]})
    codes = L.native_codes(df, ["A", "B"], "top40")
    assert codes[:, 0].tolist() == [0, 5, 1]  # METALS order preserved, Pt appended
    assert codes[:, 1].tolist() == [0, 0, 1]


# ---------------------------------------------------------------------------
# AUD-M3b: Pt auxiliary labels
# ---------------------------------------------------------------------------
def test_platinum_aux_frame_melts_cells_and_converts_nm_to_log10_um(tmp_path):
    import pandas as pd

    csv = tmp_path / "pt.csv"
    pd.DataFrame(
        {
            "smiles": ["c1ccccc1"],
            "ic50_mcf7_nm": [1000.0],   # 1000 nM = 1 uM -> log10 = 0
            "ic50_a549_nm": [10000.0],  # 10 uM -> log10 = 1
            "ic50_a2780cis_nm": [5000.0],  # cell not in vocabulary -> dropped
        }
    ).to_csv(csv, index=False)

    out = L.platinum_aux_frame(csv, ["MCF7", "A549"])
    assert len(out) == 2
    assert set(out["metal"]) == {"Pt"}
    assert set(out["cell_line_key"]) == {"MCF7", "A549"}
    got = dict(zip(out["cell_line_key"], out["value_log10"]))
    assert got["MCF7"] == pytest.approx(0.0)
    assert got["A549"] == pytest.approx(1.0)
    assert out["doi"].unique().tolist() == [L.PT_AUX_DOI]


def test_platinum_aux_frame_drops_smiles_without_a_ligand_fragment(tmp_path):
    import pandas as pd

    csv = tmp_path / "pt.csv"
    # CCO has 3 heavy atoms, below the 6-atom ligand threshold
    pd.DataFrame({"smiles": ["CCO"], "ic50_mcf7_nm": [1000.0]}).to_csv(csv, index=False)
    out = L.platinum_aux_frame(csv, ["MCF7"])
    assert out.empty


# ---------------------------------------------------------------------------
# AUD-T1: nested early stopping
# ---------------------------------------------------------------------------
def test_fit_arm_nested_returns_one_iteration_count_per_fold():
    rng = np.random.default_rng(0)
    n_groups, per = 40, 20
    groups = np.repeat(np.arange(n_groups), per)
    n = len(groups)
    lig = sparse.random(n, 50, density=0.1, random_state=0, format="csr")
    codes = np.column_stack(
        [rng.integers(0, 5, n), rng.integers(0, 10, n)]
    ).astype(np.int32)
    y = rng.normal(size=n)
    fold_id = L.folds_for(groups)

    pred, iters = L.fit_arm_nested(
        lig, codes, y, fold_id, groups, max_rounds=60, patience=10
    )
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    assert len(iters) == L.N_FOLDS
    assert all(1 <= i <= 60 for i in iters)


def test_fit_arm_nested_learns_a_signal_with_few_rounds():
    """A linear signal must be picked up, and the chosen count must be small.

    This is the property the fixed 500 trees cannot guarantee: on a learnable
    but low-dimensional problem the inner split should stop early.
    """
    rng = np.random.default_rng(1)
    n_groups, per = 40, 20
    groups = np.repeat(np.arange(n_groups), per)
    n = len(groups)
    lig = sparse.random(n, 50, density=0.2, random_state=1, format="csr")
    codes = np.column_stack(
        [rng.integers(0, 5, n), rng.integers(0, 10, n)]
    ).astype(np.int32)
    y = np.asarray(lig.sum(axis=1)).ravel() * 0.01
    fold_id = L.folds_for(groups)

    pred, _ = L.fit_arm_nested(
        lig, codes, y, fold_id, groups, max_rounds=200, patience=20
    )
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.3


# ---------------------------------------------------------------------------
# AUD-D2: paper-level offset
# ---------------------------------------------------------------------------
def test_doi_offset_recovers_a_constant_paper_shift():
    rng = np.random.default_rng(2)
    n = 200
    doi = np.array(["d1"] * 100 + ["d2"] * 100)
    fold_id = np.tile([0, 1], 100)
    offset = np.where(doi == "d1", 1.0, -1.0)
    base = rng.normal(size=n)
    y = base + offset

    out, meta = L.doi_offset_calibration(base, y, doi, fold_id, shrink=0.0)
    assert np.allclose(out, y, atol=1e-9)
    assert meta["share_rows_offset_applied"] == 1.0


def test_doi_offset_is_zero_for_an_unseen_paper():
    rng = np.random.default_rng(3)
    n = 200
    # d3 appears only in fold 0, so when fold 0 is the test set its rows have
    # no training residual to borrow from.
    doi = np.array(["d1"] * 100 + ["d3"] * 100)
    fold_id = np.array([0, 1] * 50 + [0] * 100)
    base = rng.normal(size=n)
    y = base + np.where(doi == "d1", 1.0, 5.0)

    out, meta = L.doi_offset_calibration(base, y, doi, fold_id, shrink=0.0)
    assert np.allclose(out[100:], base[100:])
    assert meta["share_rows_offset_applied"] < 1.0


def test_doi_offset_shrinks_small_papers_toward_zero():
    rng = np.random.default_rng(4)
    n = 100
    doi = np.array(["d1"] * 100)
    fold_id = np.tile([0, 1], 50)
    base = rng.normal(size=n)
    y = base + 1.0

    _, no_shrink = L.doi_offset_calibration(base, y, doi, fold_id, shrink=0.0)
    out, heavy = L.doi_offset_calibration(base, y, doi, fold_id, shrink=1e6)
    assert heavy["shrink"] > no_shrink["shrink"]
    # with overwhelming shrinkage the offset is not applied
    assert np.allclose(out, base, atol=1e-4)


# ---------------------------------------------------------------------------
# AUD-D3: duplicate weights
# ---------------------------------------------------------------------------
def test_duplicate_weights_count_each_cohort_once():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "ligand_canonical": ["L1-full", "L1-full", "L1-full", "L2-full"],
            "ligand_main_fragment": ["L1", "L1", "L1", "L2"],
            "cell_line_key": ["A", "A", "A", "A"],
            "metal": ["Ru", "Ru", "Ru", "Ir"],
            "doi": ["d1", "d1", "d1", "d1"],
            "time_h": [48, 48, 48, 48],
        }
    )
    w, counts = L.duplicate_weights(frame)
    assert counts.tolist() == [3.0, 3.0, 3.0, 1.0]
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_duplicate_weights_raises_on_a_missing_key_column():
    """pandas returns all-ones instead of failing; this must not pass silently."""
    import pandas as pd

    frame = pd.DataFrame({"ligand_main_fragment": ["L1"], "metal": ["Ru"]})
    with pytest.raises(KeyError, match="missing key columns"):
        L.duplicate_weights(frame)


# ---------------------------------------------------------------------------
# AUD-M2: per-metal models
# ---------------------------------------------------------------------------
def test_per_metal_fit_predicts_every_row_once():
    rng = np.random.default_rng(5)
    n_groups, metals = 30, np.array(["Ru", "Ir", "Rh", "Os", "Re"])
    groups = np.repeat(np.arange(n_groups), len(metals))
    n = len(groups)
    metal = np.tile(metals, n_groups)
    lig = sparse.random(n, 40, density=0.2, random_state=5, format="csr")
    codes = np.column_stack([np.searchsorted(metals, metal), rng.integers(0, 8, n)]).astype(
        np.int32
    )
    y = rng.normal(size=n)
    fold_id = L.folds_for(groups)

    pred = L.per_metal_fit(lig, codes, y, fold_id, metal)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    # a per-metal model must not produce the same prediction for every metal
    assert len(set(np.round(pred[: len(metals)], 6))) > 1


# ---------------------------------------------------------------------------
# AUD-D2 (nested) and AUD-D4
# ---------------------------------------------------------------------------
def test_rare_cell_analysis_splits_by_row_count():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0, 3.0])
    cell = np.array(["common", "common", "common", "rare"])
    out = L.rare_cell_analysis(y, pred, cell, threshold=2)
    assert out["n_cells"] == 2
    assert out["n_rare_cells"] == 1
    assert out["share_rows_in_rare_cells"] == 0.25
    assert out["mae_common_cells"] == pytest.approx(1.0)
    assert out["mae_rare_cells"] == pytest.approx(3.0)
    assert out["mae_gap_rare_minus_common"] == pytest.approx(2.0)


def test_nested_doi_offset_recovers_a_paper_shift_without_test_labels():
    """The offset must come from an inner split, and still find the shift."""
    rng = np.random.default_rng(6)
    n_groups, per = 30, 6
    groups = np.repeat(np.arange(n_groups), per)
    n = len(groups)
    fold_id = L.folds_for(groups)
    doi = np.where(np.arange(n) % 2 == 0, "d1", "d2")
    offset = np.where(doi == "d1", 1.0, -1.0)

    lig = sparse.random(n, 30, density=0.3, random_state=6, format="csr")
    codes = np.column_stack([np.zeros(n, dtype=int), rng.integers(0, 3, n)]).astype(np.int32)
    y = rng.normal(size=n) + offset

    pred, meta = L.nested_doi_offset(
        lig, codes, y, fold_id, groups, doi, inner_folds=2, shrink=1.0
    )
    assert np.isfinite(pred).all()
    assert meta["inner_folds"] == 2
    assert 0.0 < meta["share_rows_offset_applied"] <= 1.0
    assert pred.shape == (n,)
