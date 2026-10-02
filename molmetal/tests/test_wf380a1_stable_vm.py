"""Wf-380-A1 §2.3 — Test for stable_vm SCAFFOLD.

Verifies the multi-scale mixture loss structure matches Wf-340-A3 spec.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.stable_vm import stable_vm_loss


def test_stable_vm_loss_zero_when_perfect():
    """Loss must be 0 when prediction matches target exactly."""
    torch.manual_seed(0)
    pred = torch.randn(8, 4, dtype=torch.float32)
    loss = stable_vm_loss(pred, pred.clone(), n_scales=4)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_stable_vm_loss_scales_monotonic():
    """Larger prediction error → larger loss (monotonicity sanity check)."""
    target = torch.zeros(8, 4, dtype=torch.float32)
    pred_small = torch.full_like(target, 0.1)
    pred_large = torch.full_like(target, 1.0)
    loss_small = stable_vm_loss(pred_small, target, n_scales=4)
    loss_large = stable_vm_loss(pred_large, target, n_scales=4)
    assert loss_large.item() > loss_small.item()


def test_stable_vm_loss_n_scales_validates():
    """n_scales=0 raises ValueError."""
    target = torch.zeros(4, dtype=torch.float32)
    try:
        stable_vm_loss(target, target.clone(), n_scales=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for n_scales=0")


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.stable_vm as mod

    assert hasattr(mod, "stable_vm_loss")