"""Unit tests for Wf-249 Fix #2 — MC-Dropout post-hoc ensemble wrapper.

Tests cover:
  1. MCDropoutEnsemble construction + dropout layer discovery.
  2. predict() returns (mean, std) with std > 0 when dropout exists.
  3. predict() with n_passes=1 returns std=0 (deterministic).
  4. Standalone helpers: enable_mc_dropout + mc_dropout_predict.
  5. predict_with_uncertainty: lo < mean < hi for 0.95 CI.
  6. Models with NO dropout: n_dropout_layers=0, but predict still works
     (std=0 because no stochasticity).
  7. Validation: invalid p / n_passes raises ValueError.
  8. CPU-only: no GPU ops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.mc_dropout_wrapper import (
    MCDropoutEnsemble,
    enable_mc_dropout,
    mc_dropout_predict,
)


class TinyModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.drop1(torch.relu(self.fc1(x))))


class TinyModelNoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def test_ensemble_constructs_and_discovers_dropout():
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=10)
    assert ensemble.n_dropout_layers == 1, "Should find 1 dropout layer"
    assert ensemble.dropout_p == 0.2
    assert ensemble.n_passes == 10


def test_ensemble_predict_std_positive():
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=20)
    x = torch.randn(4, 8)
    mean, std = ensemble.predict(x)
    assert mean.shape == (4, 1), f"Expected (4,1), got {tuple(mean.shape)}"
    assert std is not None and std.shape == mean.shape
    # With 20 passes and p=0.2, std should be > 0 (model has dropout).
    assert float(std.mean()) > 0.0, "std should be > 0 with active dropout"


def test_ensemble_n_passes_one_is_deterministic():
    """With n_passes=1, std should be 0 because no aggregation."""
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=1)
    x = torch.randn(4, 8)
    mean, std = ensemble.predict(x, return_std=True)
    # With n_passes=1, the stack has shape (1, ...); std over a single
    # sample is 0 by definition (unbiased=False).
    assert torch.allclose(std, torch.zeros_like(std))


def test_enable_mc_dropout_helper():
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    n = enable_mc_dropout(model, p=0.3)
    assert n == 1
    x = torch.randn(4, 8)
    mean, std = mc_dropout_predict(model, x, n_passes=15)
    assert mean.shape == (4, 1)
    assert float(std.mean()) > 0.0


def test_no_dropout_model_predict_works():
    """A model with no dropout layers should still run; std=0."""
    torch.manual_seed(0)
    model = TinyModelNoDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.1, n_passes=5)
    assert ensemble.n_dropout_layers == 0
    x = torch.randn(4, 4)
    mean, std = ensemble.predict(x)
    assert mean.shape == (4, 1)
    assert std is not None and torch.allclose(std, torch.zeros_like(std))


def test_predict_with_uncertainty_brackets_mean():
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=20)
    x = torch.randn(4, 8)
    mean, lo, hi = ensemble.predict_with_uncertainty(x, confidence=0.95)
    assert torch.all(lo < mean)
    assert torch.all(mean < hi)
    # 0.99 CI should be strictly wider than 0.95 CI on the same data.
    mean_95, lo_95, hi_95 = ensemble.predict_with_uncertainty(x, confidence=0.95)
    mean_99, lo_99, hi_99 = ensemble.predict_with_uncertainty(x, confidence=0.99)
    width_95 = float((hi_95 - lo_95).mean())
    width_99 = float((hi_99 - lo_99).mean())
    assert width_99 > width_95, "0.99 CI must be wider than 0.95 CI"


def test_invalid_p_raises():
    model = TinyModelWithDropout()
    with pytest.raises(ValueError):
        MCDropoutEnsemble(model, dropout_p=1.5, n_passes=10)
    with pytest.raises(ValueError):
        MCDropoutEnsemble(model, dropout_p=-0.1, n_passes=10)


def test_invalid_n_passes_raises():
    model = TinyModelWithDropout()
    with pytest.raises(ValueError):
        MCDropoutEnsemble(model, dropout_p=0.1, n_passes=0)


def test_invalid_confidence_raises():
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.1, n_passes=5)
    x = torch.randn(2, 8)
    with pytest.raises(ValueError):
        ensemble.predict_with_uncertainty(x, confidence=1.5)
    with pytest.raises(ValueError):
        ensemble.predict_with_uncertainty(x, confidence=0.0)


def test_cpu_only():
    """All operations must be CPU-only — no implicit GPU device."""
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=5)
    x = torch.randn(2, 8)
    mean, std = ensemble.predict(x)
    assert mean.device.type == "cpu"
    assert std.device.type == "cpu"


def test_eval_mode_restored_after_predict():
    """After predict(), the parent model should be back in eval mode."""
    torch.manual_seed(0)
    model = TinyModelWithDropout()
    ensemble = MCDropoutEnsemble(model, dropout_p=0.2, n_passes=5)
    x = torch.randn(2, 8)
    _ = ensemble.predict(x)
    # Parent model is in eval (BatchNorm etc.); only dropout layers are
    # flipped to train mode during predict(). Re-flipping is the
    # wrapper's responsibility. After the call, the wrapper leaves
    # dropout layers in train mode (per the MC-dropout recipe); the
    # parent remains eval.
    assert not model.training, "Parent model should be in eval mode"


if __name__ == "__main__":
    test_ensemble_constructs_and_discovers_dropout()
    test_ensemble_predict_std_positive()
    test_ensemble_n_passes_one_is_deterministic()
    test_enable_mc_dropout_helper()
    test_no_dropout_model_predict_works()
    test_predict_with_uncertainty_brackets_mean()
    test_invalid_p_raises()
    test_invalid_n_passes_raises()
    test_invalid_confidence_raises()
    test_cpu_only()
    test_eval_mode_restored_after_predict()
    print("OK - all F2 (MC-Dropout ensemble) tests passed")
