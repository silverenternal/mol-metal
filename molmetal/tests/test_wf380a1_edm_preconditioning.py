"""Wf-380-A1 §2.1 — Test for edm_preconditioning SCAFFOLD.

Verifies the public API matches Karras 2022 eq. 2 reference values.
"""

from __future__ import annotations

import math

import torch

from molmetal.adapters.flow_matching_lipman.edm_preconditioning import (
    EDMConfig,
    c_in,
    c_noise,
    c_out,
    c_skip,
    preconditioned_loss,
)


def test_edm_preconditioning_factors_match_karras_eq2():
    """Karras 2022 eq. 2: c_skip + c_out + c_in factors.

    At σ = σ_data, the factors reduce to: c_skip = 0.5, c_out = σ_data / sqrt(2),
    c_in = 1 / (σ_data * sqrt(2)).  Verifies the closed-form limits.
    """
    cfg = EDMConfig(sigma_data=1.5)
    sigma = torch.tensor([cfg.sigma_data], dtype=torch.float32)
    s = c_skip(sigma, cfg).item()
    o = c_out(sigma, cfg).item()
    i = c_in(sigma, cfg).item()
    assert abs(s - 0.5) < 1e-6, f"c_skip(σ_data) must be 0.5; got {s}"
    # c_out(σ_data) = σ_data * σ_data / sqrt(2 * σ_data²) = σ_data / sqrt(2)
    expected_o = cfg.sigma_data / math.sqrt(2.0)
    assert abs(o - expected_o) < 1e-5, f"c_out mismatch: {o} vs {expected_o}"
    expected_i = 1.0 / (cfg.sigma_data * math.sqrt(2.0))
    assert abs(i - expected_i) < 1e-6, f"c_in mismatch: {i} vs {expected_i}"


def test_preconditioned_loss_zero_when_pred_matches_target():
    """If F_θ perfectly predicts the conditional velocity, the loss is 0.

    Set pred_raw = (target − c_skip * x_t) / c_out and the preconditioned
    velocity should equal target exactly, so loss = 0.
    """
    cfg = EDMConfig(sigma_data=1.5)
    b, d = 4, 8
    x_t = torch.randn(b, d, dtype=torch.float32)
    target = torch.randn(b, d, dtype=torch.float32)
    sigma = torch.full((b,), 1.0, dtype=torch.float32)
    sigma_b = sigma.view(-1, 1)
    pred_raw = (target - c_skip(sigma_b, cfg) * x_t) / c_out(sigma_b, cfg)
    loss = preconditioned_loss(pred_raw, x_t, target, sigma, cfg)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5), (
        f"loss must be 0 when pred matches target; got {loss.item()}"
    )


def test_c_noise_logarithmic():
    """c_noise(σ) = 0.5 · ln(σ) per Karras 2022 eq. 2."""
    sigma = torch.tensor([1.0, math.e, math.e ** 2], dtype=torch.float32)
    expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    assert torch.allclose(c_noise(sigma), expected, atol=1e-6)


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.edm_preconditioning as mod

    assert hasattr(mod, "preconditioned_loss")
    assert hasattr(mod, "EDMConfig")
