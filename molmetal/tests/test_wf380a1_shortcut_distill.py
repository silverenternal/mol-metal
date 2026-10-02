"""Wf-380-A1 §2.6 — Test for shortcut_distillation_wf380 SCAFFOLD.

Verifies Frans 2024 §3.2 self-consistency structure.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.shortcut_distillation_wf380 import (
    shortcut_distillation_loss,
    shortcut_pair,
)


def test_shortcut_pair_shape_preserved():
    """The shortcut (x_s, s) has shape compatible with (x_t, t)."""
    torch.manual_seed(0)
    b, d = 8, 5
    x_t = torch.randn(b, d, dtype=torch.float32)
    t = torch.rand(b, dtype=torch.float32)
    target = torch.randn(b, d, dtype=torch.float32)
    x_s, s = shortcut_pair(x_t, t, target, p_self=1.0)
    assert x_s.shape == x_t.shape, f"x_s shape mismatch: {x_s.shape} vs {x_t.shape}"
    assert s.shape == t.shape, f"s shape mismatch: {s.shape} vs {t.shape}"


def test_shortcut_pair_off_when_p_self_zero():
    """p_self=0 ⇒ shortcut pair equals the original (x_t, t) exactly."""
    torch.manual_seed(0)
    b, d = 4, 3
    x_t = torch.randn(b, d, dtype=torch.float32)
    t = torch.rand(b, dtype=torch.float32)
    target = torch.randn(b, d, dtype=torch.float32)
    x_s, s = shortcut_pair(x_t, t, target, p_self=0.0)
    assert torch.allclose(x_s, x_t), "p_self=0 should preserve x_t exactly"
    assert torch.allclose(s, t), "p_self=0 should preserve t exactly"


def test_shortcut_distillation_loss_zero_for_perfect_pred():
    """If pred_fn == target, the loss reduces to the trivial FM term (small)."""
    torch.manual_seed(0)
    b, d = 4, 3
    x_t = torch.randn(b, d, dtype=torch.float32)
    t = torch.rand(b, dtype=torch.float32) * 0.5 + 0.25  # in (0.25, 0.75)
    target = torch.zeros(b, d, dtype=torch.float32)

    def pred_fn(x, t_):
        # Returns target exactly.
        return torch.zeros_like(x)

    loss = shortcut_distillation_loss(pred_fn, x_t, t, target, p_self=0.5)
    assert loss.item() < 1e-3, f"Loss must be ~0 for perfect pred; got {loss.item()}"


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.shortcut_distillation_wf380 as mod

    assert hasattr(mod, "shortcut_distillation_loss")