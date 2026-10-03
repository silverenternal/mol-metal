"""Wf-380-A1 §2.7 — Test for consistency_regularization_wf380 SCAFFOLD.

Verifies EDM σ sampling + consistency loss structure (Karras 2022).
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.consistency_regularization_wf380 import (
    consistency_loss_fn,
    sample_edm_sigma,
)


def test_sample_edm_sigma_shape_and_range():
    """σ values have shape (B,) and lie in [sigma_min, sigma_max]."""
    torch.manual_seed(0)
    b = 256
    sigma = sample_edm_sigma(b, sigma_min=0.002, sigma_max=80.0)
    assert sigma.shape == (b,)
    assert (sigma >= 0.002).all()
    assert (sigma <= 80.0).all()


def test_sample_edm_sigma_distribution_approximately_log_normal():
    """Empirical log(σ) has mean near P_mean = -1.2 and std near P_std = 1.2."""
    torch.manual_seed(1)
    b = 4096
    sigma = sample_edm_sigma(b, sigma_min=1e-3, sigma_max=1e6)
    log_sigma = sigma.log()
    # Loose tolerance — large-sample CLT.
    assert abs(log_sigma.mean().item() - (-1.2)) < 0.1
    assert abs(log_sigma.std().item() - 1.2) < 0.1


def test_consistency_loss_zero_for_constant_pred():
    """If pred_fn is constant, the perturbed pred equals the clean pred → loss 0."""
    torch.manual_seed(0)
    b, d = 8, 5
    x_t = torch.randn(b, d, dtype=torch.float32)

    def pred_fn(x):
        return torch.zeros_like(x)

    loss = consistency_loss_fn(pred_fn, x_t, weight=1.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.consistency_regularization_wf380 as mod

    assert hasattr(mod, "consistency_loss_fn")
