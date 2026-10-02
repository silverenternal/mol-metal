"""Wf-380-A1 §2.9 — Test for log_normal_t SCAFFOLD.

Verifies the log-normal t-sampling distribution shape + range.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.log_normal_t import log_normal_t_sampling


def test_log_normal_t_sampling_shape():
    """Output shape is (batch_size,)."""
    torch.manual_seed(0)
    t = log_normal_t_sampling(128)
    assert t.shape == (128,)


def test_log_normal_t_sampling_in_unit_interval():
    """All sampled t values must lie in (0, 1) (sigmoid output)."""
    torch.manual_seed(1)
    t = log_normal_t_sampling(4096)
    assert (t > 0).all()
    assert (t < 1).all()


def test_log_normal_t_sampling_distribution_mean():
    """The empirical mean of the logit(t) should be near mu."""
    torch.manual_seed(2)
    b = 8192
    t = log_normal_t_sampling(b, mu=-0.5, sigma=1.0)
    # logit(t) ≈ N(mu, sigma²) — verify the empirical mean.
    logit_t = torch.log(t / (1.0 - t))
    assert abs(logit_t.mean().item() - (-0.5)) < 0.1
    assert abs(logit_t.std().item() - 1.0) < 0.1


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.log_normal_t as mod

    assert hasattr(mod, "log_normal_t_sampling")