"""Wf-380-A1 §2.4 — Test for so3_average SCAFFOLD.

Verifies SO(3) sampling + equivariance probe (Cao 2025 ICML).
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.so3_average import (
    random_so3_matrix,
    rotate_coords,
    so3_average_loss,
)


def test_random_so3_is_rotation():
    """Sampled matrices must satisfy R @ R.T = I and det(R) = +1."""
    torch.manual_seed(42)
    n = 32
    R = random_so3_matrix(n)
    eye = torch.eye(3).expand(n, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones(n), atol=1e-5)


def test_rotate_coords_round_trip():
    """Apply R, then R.T → identity."""
    torch.manual_seed(0)
    R = random_so3_matrix(1)[0]
    x = torch.randn(8, 3)
    x_round_trip = rotate_coords(rotate_coords(x, R), R.t())
    assert torch.allclose(x, x_round_trip, atol=1e-5)


def test_so3_average_loss_is_finite_and_nonnegative():
    """The SO(3)-averaged loss must be finite and non-negative for any input."""
    torch.manual_seed(7)
    b, n, d = 2, 5, 3
    x_t = torch.randn(b, n, d)
    target = torch.randn(b, n, d)

    def pred_fn(x):
        # Arbitrary pred — constant zero is equivariant by construction.
        return torch.zeros_like(x)

    loss = so3_average_loss(pred_fn, x_t, target, n_rotations=4)
    assert torch.isfinite(loss), f"loss must be finite; got {loss.item()}"
    assert loss.item() >= 0.0
    # For pred=0, the loss equals mean(||R · target||²) = mean(||target||²).
    expected = (target ** 2).mean().item()
    assert abs(loss.item() - expected) < 1e-5, (
        f"For pred=0, loss must equal mean(target²); got {loss.item()} vs {expected}"
    )


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.so3_average as mod

    assert hasattr(mod, "random_so3_matrix")
    assert hasattr(mod, "so3_average_loss")
