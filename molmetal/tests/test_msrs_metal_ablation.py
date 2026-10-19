"""MET-6 tests — every block toggle must change the matrix it claims to."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from molmetal.scripts.msrs_metal_ablation import (
    ARMS,
    arm_matrix,
    group_bootstrap_delta,
    permutation_pvalue,
    permute_metal_block,
    permute_metal_label,
)


def _fixture(n=20, n_complex=7, n_state=4, n_coord=3):
    rng = np.random.default_rng(0)
    complex_block = sparse.csr_matrix(rng.normal(size=(n, n_complex)).astype(np.float32))
    state = rng.normal(size=(n, n_state)).astype(np.float32)
    coord = sparse.csr_matrix(rng.normal(size=(n, n_coord)).astype(np.float32))
    codes = np.column_stack(
        [
            rng.integers(0, 5, size=n),
            rng.integers(0, 3, size=n),
        ]
    ).astype(np.int32)
    return complex_block, state, coord, codes


def test_full_arm_keeps_every_block_and_both_codes():
    cb, st, co, codes = _fixture()
    X, cat = arm_matrix("full", cb, st, co, codes)
    assert X.shape == (20, 7 + 4 + 3)
    assert cat.shape == (20, 2)


def test_metal_blind_drops_all_three_metal_blocks_but_keeps_cell():
    cb, st, co, codes = _fixture()
    X, cat = arm_matrix("metal_blind", cb, st, co, codes)
    assert X.shape == (20, 7)
    assert cat.shape == (20, 1)
    assert np.array_equal(cat[:, 0], codes[:, 1])


def test_ligand_only_has_no_categorical_codes():
    cb, st, co, codes = _fixture()
    X, cat = arm_matrix("ligand_only", cb, st, co, codes)
    assert X.shape == (20, 7)
    assert cat is None


def test_no_complex_keeps_only_the_metal_side():
    cb, st, co, codes = _fixture()
    X, cat = arm_matrix("no_complex", cb, st, co, codes)
    assert X.shape == (20, 4 + 3)
    assert cat.shape == (20, 2)


def test_every_arm_is_declared_with_five_flags():
    for name, flags in ARMS.items():
        assert len(flags) == 5, name


def test_permute_metal_label_preserves_the_metal_marginal():
    _, _, _, codes = _fixture(n=200)
    out = permute_metal_label(codes, np.random.default_rng(1))
    before = np.bincount(codes[:, 0], minlength=5)
    after = np.bincount(out[:, 0], minlength=5)
    assert np.array_equal(before, after)
    # The cell column must not move.
    assert np.array_equal(out[:, 1], codes[:, 1])
    # And the alignment must actually change.
    assert not np.array_equal(out[:, 0], codes[:, 0])


def test_permute_metal_block_uses_one_permutation_for_all_three_blocks():
    _, state, coord, codes = _fixture(n=50)
    # A marker column identifies each original row, so the same permutation is
    # visible across all three outputs.
    marker = np.arange(50, dtype=np.float32).reshape(-1, 1)
    state2 = np.hstack([marker, state])
    coord2 = sparse.hstack([sparse.csr_matrix(marker), coord], format="csr")
    c2, s2, q2 = permute_metal_block(codes, state2, coord2, np.random.default_rng(3))
    order = s2[:, 0].astype(int)
    assert np.array_equal(q2[:, 0].toarray().ravel().astype(int), order)
    # Metal code follows the same order.
    assert np.array_equal(c2[:, 0], codes[order, 0])


def test_permutation_pvalue_uses_the_add_one_rule():
    assert permutation_pvalue(0.9, [0.1, 0.2, 0.3]) == pytest.approx(1 / 4)
    assert permutation_pvalue(0.0, [0.1, 0.2, 0.3]) == pytest.approx(1.0)
    assert permutation_pvalue(0.2, [0.1, 0.2, 0.3]) == pytest.approx(3 / 4)


def test_group_bootstrap_delta_is_positive_when_full_beats_the_arm():
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    y = rng.normal(size=n)
    pred_full = y + rng.normal(scale=0.2, size=n)
    pred_arm = y + rng.normal(scale=0.8, size=n)
    out = group_bootstrap_delta(y, pred_full, pred_arm, groups, n_boot=200)
    assert out["ci95"][0] > 0
    assert out["n_groups"] == 40
