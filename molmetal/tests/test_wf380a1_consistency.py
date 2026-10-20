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


# --- FIX-C8 (2026-10-20) -----------------------------------------------------
# The sigma draw defaulted to CPU, so on a CUDA input the multiply raised and
# the trainer's ConsistencyTerm swallowed it and returned exactly zero. The
# default weight is 0.05, so a non-zero term was silently absent from every run.


def test_sigma_draw_follows_the_input_device():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA device")
    x_t = torch.randn(4, 3, device="cuda:0")
    sigma = sample_edm_sigma(4, device=x_t.device)
    assert sigma.device == x_t.device


def test_consistency_loss_does_not_raise_on_cuda():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA device")
    torch.manual_seed(0)
    x_t = torch.randn(4, 3, device="cuda:0", requires_grad=True)
    loss = consistency_loss_fn(lambda z: z * 0.5, x_t, weight=0.05)
    assert loss.device == x_t.device
    assert loss.item() > 0.0


def test_consistency_loss_carries_gradient():
    # A LINEAR pred_fn has zero gradient by construction: v(x+sn) - v(x) = s*n
    # does not depend on x. The term is a smoothness penalty, so it only carries
    # gradient for a nonlinear field.
    torch.manual_seed(0)
    x_t = torch.randn(4, 3, requires_grad=True)
    loss = consistency_loss_fn(lambda z: torch.tanh(z) * 2.0, x_t, weight=1.0)
    loss.backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()
    assert x_t.grad.abs().sum().item() > 0.0
