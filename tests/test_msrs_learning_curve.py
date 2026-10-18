"""The learning curve must subsample ligands, never rows."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from molmetal.scripts import msrs_learning_curve as C


class _DummyRegressor:
    fits: list = []

    def __init__(self, **_kw):
        self.n_rows = 0
        self._mean = 0.0

    def fit(self, X, y):
        self.n_rows = int(X.shape[0])
        self._mean = float(np.mean(y))
        _DummyRegressor.fits.append(self.n_rows)
        return self

    def predict(self, X):
        return np.full(int(X.shape[0]), self._mean)


@pytest.fixture()
def dummy(monkeypatch):
    _DummyRegressor.fits = []
    import lightgbm

    monkeypatch.setattr(lightgbm, "LGBMRegressor", _DummyRegressor)
    return _DummyRegressor


def _toy(n_ligands=20, per_ligand=4):
    groups = np.repeat(np.arange(n_ligands), per_ligand)
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(len(groups), 3)))
    y = rng.normal(size=len(groups))
    fold_id = C.folds_for(groups)
    return X, y, groups, fold_id


def test_fraction_one_uses_the_whole_training_fold(dummy):
    X, y, groups, fold_id = _toy()
    _, n_train = C.oof_at_fraction(X, y, groups, fold_id, 1.0, 0)
    expected = [int((fold_id != k).sum()) for k in sorted(set(fold_id.tolist()))]
    assert n_train == expected


def test_smaller_fraction_uses_fewer_rows(dummy):
    X, y, groups, fold_id = _toy()
    _, n10 = C.oof_at_fraction(X, y, groups, fold_id, 0.10, 0)
    _, n50 = C.oof_at_fraction(X, y, groups, fold_id, 0.50, 0)
    assert all(a < b for a, b in zip(n10, n50))


def test_rows_of_a_ligand_are_never_split(dummy):
    X, y, groups, fold_id = _toy()
    C.oof_at_fraction(X, y, groups, fold_id, 0.30, 0)
    # every fit must use a whole number of ligands (per_ligand rows each)
    assert all(n % 4 == 0 for n in _DummyRegressor.fits)


def test_oof_covers_every_row(dummy):
    X, y, groups, fold_id = _toy()
    oof, _ = C.oof_at_fraction(X, y, groups, fold_id, 0.5, 0)
    assert oof.shape == y.shape
    assert not np.isnan(oof).any()
