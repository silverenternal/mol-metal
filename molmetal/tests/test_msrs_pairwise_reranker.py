"""Tests for FIX-M4's symmetric pairwise reranker.

The failure modes this pins down are the ones that actually occurred:

* a pair design that is not antisymmetric (FIX-E3 dropped metal B's state),
* same-metal replicate rows entering training as non-zero deltas,
* a sign flip between ``f(b) - f(a)`` and a row's own score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_pairwise_reranker import (  # noqa: E402
    _ridge_fit,
    _ridge_predict,
    pair_accuracy,
    pair_design,
    pair_matrix,
)


def _blocks(n: int = 3) -> dict:
    base = {
        "onehot": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        "state": np.array([[0.0, 0.0], [1.0, 2.0], [0.5, -1.0]]),
        "coord": np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]]),
        "ligand": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    }
    if n == 3:
        return {**base, "_y": np.array([1.0, 2.0, 3.0])}
    reps = int(np.ceil(n / 3))
    return {k: np.tile(v, (reps, 1))[:n] for k, v in base.items()}


@pytest.mark.parametrize("arm", ["pair_identity", "pair_state", "pair_full"])
def test_pair_design_is_antisymmetric(arm):
    blocks = _blocks()
    forward = pair_design(blocks, 0, 1, arm)
    backward = pair_design(blocks, 1, 0, arm)
    assert np.allclose(forward, -backward)


@pytest.mark.parametrize("arm", ["pair_identity", "pair_state", "pair_full"])
def test_pair_design_of_a_row_with_itself_is_zero(arm):
    blocks = _blocks()
    assert np.allclose(pair_design(blocks, 1, 1, arm), 0.0)


def test_pair_matrix_skips_same_metal_and_singleton_cohorts():
    blocks = _blocks(5)
    # cohort 0: Pt, Ru (two metals) -> 2 ordered pairs
    # cohort 1: Pt, Pt (same metal) -> skipped
    # cohort 2: Ru alone -> skipped
    cohorts = np.array([0, 0, 1, 1, 2])
    metals = np.array(["Pt", "Ru", "Pt", "Pt", "Ru"])
    blocks["_y"] = np.array([1.0, 2.0, 5.0, 5.1, 9.0])
    X, t, groups, ab = pair_matrix(blocks, np.arange(5), cohorts, metals, "pair_state")
    assert X.shape[0] == 2
    assert set(groups) == {0}
    assert sorted(map(tuple, ab)) == [(0, 1), (1, 0)]
    assert set(np.round(t, 6)) == {-1.0, 1.0}


def test_ridge_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    truth = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    t = X @ truth
    coef, mu, sd, t_mean = _ridge_fit(X, t, alpha=1e-6)
    pred = _ridge_predict(X, coef, mu, sd, t_mean)
    assert np.allclose(pred, t, atol=1e-4)


def test_pair_accuracy_detects_a_sign_flip():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    cohorts = np.array([0, 0, 1, 1])
    good = np.array([0.0, 1.0, 0.0, 1.0])
    bad = -good
    assert pair_accuracy(good, cohorts, y)["pairwise_accuracy"] == 1.0
    assert pair_accuracy(bad, cohorts, y)["pairwise_accuracy"] == 0.0
