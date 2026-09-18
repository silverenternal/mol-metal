"""Tests for the standalone symbolic regression wrappers."""

import sys

import numpy as np
import pytest

from molmetal_lam.lam_chem import pysr_wrapper
from molmetal_lam.lam_chem.pysr_wrapper import (
    MetalCoordinationRegressor,
    SymbolicRegressor,
)


def test_instantiate():
    """The wrapper can be constructed even without Julia installed."""
    regressor = SymbolicRegressor()
    assert regressor.n_iterations == 100
    assert regressor.binary_operators == ["+", "*"]


def test_fit_predict(monkeypatch):
    """Fallback fitting provides finite predictions on nonlinear data."""
    monkeypatch.setattr(pysr_wrapper, "_probe_pysr", lambda: False)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    regressor = SymbolicRegressor().fit(X, y)
    predictions = regressor.predict(X[:5])
    assert regressor.backend == "sklearn_ridge"
    assert predictions.shape == (5,)
    assert np.all(np.isfinite(predictions))


def test_fallback(monkeypatch):
    """A missing Julia module selects the Ridge backend cleanly."""
    monkeypatch.setitem(sys.modules, "julia", None)
    monkeypatch.setattr(pysr_wrapper, "_probe_pysr", lambda: False)
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = X[:, 0] - 2 * X[:, 1]
    regressor = SymbolicRegressor().fit(X, y)
    assert regressor.backend == "sklearn_ridge"
    assert regressor.equations() == [
        {"backend": "sklearn_ridge", "equation": "Ridge regression", "loss": 0.0}
    ]
    assert isinstance(regressor.predict(X), np.ndarray)


def test_metal_regressor(monkeypatch):
    """Metal coordination defaults include rational-function support."""
    monkeypatch.setattr(pysr_wrapper, "_probe_pysr", lambda: False)
    regressor = MetalCoordinationRegressor()
    assert regressor.n_iterations == 150
    assert "/" in regressor.binary_operators


@pytest.mark.parametrize("shape", [(3, 2), (8, 3)])
def test_predict_shape(monkeypatch, shape):
    """Predictions preserve one value per sample."""
    monkeypatch.setattr(pysr_wrapper, "_probe_pysr", lambda: False)
    X = np.ones(shape)
    y = np.arange(shape[0], dtype=float)
    model = SymbolicRegressor().fit(X, y)
    assert model.predict(X).shape == (shape[0],)

