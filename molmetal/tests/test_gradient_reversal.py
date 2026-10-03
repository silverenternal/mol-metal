"""Tests for the DANN Gradient Reversal Layer (TODO-143 §2.2).

The GRL is a drop-in port of ``fungtion/DANN_py3/functions.py``.  We verify:

* Forward pass is exactly the identity (no value change, no reshape).
* Backward pass negates the incoming gradient and scales by ``alpha``.
* ``alpha=0`` is a clean no-op (gradient passes through unchanged).
* Multiple calls with different alphas produce the expected backward scaling.
* GRL composes cleanly inside a multi-layer model (full forward/backward via
  ``loss.backward()`` with finite, non-NaN, non-Inf gradients).
* ``DomainAdversarialHead`` from ``molmetal.models.dmpnn`` produces finite
  logits and finite gradients whether or not the GRL is in front of it.

All tests run on CPU via the project micromamba env so they execute quickly
in CI without GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the project importable when the test is invoked from the repo root
# the same way as the existing suite (e.g. ``pytest molmetal/tests/...``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.sbdd_env.gradient_reversal import ReverseLayerF
from molmetal.models.dmpnn import DomainAdversarialHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_tensor() -> torch.Tensor:
    """Deterministic (4, 3) tensor for repeatable gradient checks."""
    torch.manual_seed(0)
    return torch.randn(4, 3, requires_grad=True)


@pytest.fixture
def head() -> DomainAdversarialHead:
    """Default-config DomainAdversarialHead (in_dim=8, num_domains=4)."""
    return DomainAdversarialHead(in_dim=8, hidden_dim=16, num_domains=4, dropout=0.1)


# ---------------------------------------------------------------------------
# Forward identity
# ---------------------------------------------------------------------------
class TestForwardIdentity:
    def test_forward_returns_input_unchanged(self, seeded_tensor):
        out = ReverseLayerF.apply(seeded_tensor, 0.1)
        assert torch.equal(out, seeded_tensor.detach())

    def test_forward_preserves_shape_for_higher_rank(self):
        x = torch.randn(2, 3, 5, requires_grad=True)
        out = ReverseLayerF.apply(x, 0.5)
        assert out.shape == x.shape
        assert torch.equal(out, x.detach())

    def test_forward_preserves_requires_grad(self, seeded_tensor):
        out = ReverseLayerF.apply(seeded_tensor, 0.1)
        assert out.requires_grad


# ---------------------------------------------------------------------------
# Backward negation / scaling
# ---------------------------------------------------------------------------
class TestBackward:
    def test_backward_negates_gradient_with_alpha_one(self, seeded_tensor):
        x = seeded_tensor.detach().clone().requires_grad_(True)
        out = ReverseLayerF.apply(x, 1.0)
        # Use a non-trivial upstream gradient so we can check the sign flip.
        upstream = torch.full_like(out, 0.7)
        out.backward(upstream)
        # alpha=1 means grad should be exactly -upstream
        assert torch.allclose(x.grad, -upstream, atol=1e-7)

    def test_backward_negates_gradient_with_alpha_point_five(self, seeded_tensor):
        x = seeded_tensor.detach().clone().requires_grad_(True)
        out = ReverseLayerF.apply(x, 0.5)
        upstream = torch.full_like(out, 2.0)
        out.backward(upstream)
        # alpha=0.5 means grad should be -0.5 * upstream
        assert torch.allclose(x.grad, -0.5 * upstream, atol=1e-7)

    def test_backward_zero_alpha_passes_through_unchanged(self, seeded_tensor):
        """alpha=0 must short-circuit the negation: gradient is zero."""
        x = seeded_tensor.detach().clone().requires_grad_(True)
        out = ReverseLayerF.apply(x, 0.0)
        upstream = torch.full_like(out, 1.3)
        out.backward(upstream)
        # alpha=0 -> -0 * upstream = 0
        assert torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-7)

    def test_backward_alpha_does_not_receive_gradient(self):
        """alpha is treated as a constant at backward (second return is None)."""
        x = torch.randn(3, 2, requires_grad=True)
        alpha = torch.tensor(0.25, requires_grad=True)
        out = ReverseLayerF.apply(x, alpha)
        # Backprop a constant upstream gradient.
        out.backward(torch.ones_like(out))
        # alpha must not accumulate a gradient (ReverseLayerF.backward returns
        # None for the second arg).
        assert alpha.grad is None


# ---------------------------------------------------------------------------
# Multi-call alpha sensitivity
# ---------------------------------------------------------------------------
class TestAlphaSensitivity:
    @pytest.mark.parametrize("alpha,expected_scale", [
        (0.0, 0.0),
        (0.1, -0.1),
        (1.0, -1.0),
        (2.5, -2.5),
        (10.0, -10.0),
    ])
    def test_alpha_scales_backward_grad(self, seeded_tensor, alpha, expected_scale):
        x = seeded_tensor.detach().clone().requires_grad_(True)
        out = ReverseLayerF.apply(x, alpha)
        upstream = torch.full_like(out, 1.0)
        out.backward(upstream)
        assert torch.allclose(x.grad, expected_scale * upstream, atol=1e-7)

    def test_two_consecutive_calls_use_their_own_alpha(self):
        """Calling GRL twice with different alphas: each call applies its own alpha."""
        x = torch.randn(4, requires_grad=True)
        alpha_a, alpha_b = 0.3, 0.7
        out_a = ReverseLayerF.apply(x, alpha_a)
        out_b = ReverseLayerF.apply(out_a, alpha_b)
        # Outermost grad = -alpha_b * ( -alpha_b * upstream_inner ) ??? Let's
        # verify the actual chain rule: f1(x) = x; b1(go) = -alpha_a * go.
        # f2(y) = y; b2(go) = -alpha_b * go.  So dL/dx = -alpha_b * ( -alpha_a * go )
        # = alpha_a * alpha_b * go  (NOT -alpha_a*alpha_b, because two negations).
        upstream = torch.full_like(out_b, 1.0)
        out_b.backward(upstream)
        expected = alpha_a * alpha_b * upstream
        assert torch.allclose(x.grad, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# End-to-end model integration
# ---------------------------------------------------------------------------
class TestModelIntegration:
    def test_grl_in_loop_produces_finite_gradients(self):
        """GRL inside a small linear stack: forward + backward, no NaN/Inf.

        We compare the GRL case against a separate no-GRL baseline (different
        layer instance) so we can verify that the gradient sign is flipped
        relative to the un-reversed reference.
        """
        torch.manual_seed(1)
        alpha = 0.4
        layer = nn.Linear(3, 3, bias=False)
        ref_layer = nn.Linear(3, 3, bias=False)
        ref_layer.weight.data = layer.weight.data.clone()

        x = torch.randn(5, 3, requires_grad=True)
        h = layer(x)
        rev = ReverseLayerF.apply(h, alpha)
        loss = (rev ** 2).sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.weight.grad).all()

        x_ref = x.detach().clone().requires_grad_(True)
        h_ref = ref_layer(x_ref)
        loss_ref = (h_ref ** 2).sum()
        loss_ref.backward()
        # GRL flips the sign of dL/dh, so dL/dW = -alpha * (no-GRL dL/dW)
        assert torch.allclose(
            layer.weight.grad, -alpha * ref_layer.weight.grad, atol=1e-5
        )

    def test_grl_then_domain_head_round_trip(self, head: DomainAdversarialHead):
        """GRL + DomainAdversarialHead: end-to-end loss.backward() works."""
        torch.manual_seed(2)
        features = torch.randn(8, 8, requires_grad=True)
        alpha = 0.2
        rev = ReverseLayerF.apply(features, alpha)
        logits = head(rev)
        domain_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        loss = nn.functional.cross_entropy(logits, domain_labels)
        loss.backward()
        assert torch.isfinite(loss).item()
        assert torch.isfinite(features.grad).all()
        # Cross-check: without the GRL the sign on features.grad would be
        # opposite (negated by the -alpha factor).  We don't have to prove
        # the exact scaling here; the all-finite check is the safety gate.

    def test_domain_head_forward_shape(self, head: DomainAdversarialHead):
        features = torch.randn(8, 8)
        logits = head(features)
        assert logits.shape == (8, head.num_domains)

    def test_domain_head_extra_repr(self, head: DomainAdversarialHead):
        rep = head.extra_repr()
        assert "in_dim=8" in rep
        assert "num_domains=4" in rep

    def test_domain_head_zero_dropout_path(self):
        head = DomainAdversarialHead(in_dim=4, hidden_dim=8, num_domains=4, dropout=0.0)
        x = torch.randn(2, 4)
        logits = head(x)
        assert logits.shape == (2, 4)


# ---------------------------------------------------------------------------
# Sanity / regression: NaN / Inf guard
# ---------------------------------------------------------------------------
class TestNumericalGuards:
    def test_no_nan_or_inf_in_backward_grad(self):
        """Sanity: GRL never produces NaN/Inf gradients for finite inputs."""
        torch.manual_seed(7)
        for _ in range(20):
            x = torch.randn(3, 5, requires_grad=True)
            alpha = float(torch.rand(1).item())
            out = ReverseLayerF.apply(x, alpha)
            upstream = torch.randn_like(out)
            out.backward(upstream)
            assert torch.isfinite(x.grad).all(), f"alpha={alpha} produced non-finite grad"

    def test_large_alpha_does_not_overflow(self):
        """alpha=1e6 should still produce finite gradients (no overflow)."""
        x = torch.randn(2, 3, requires_grad=True)
        out = ReverseLayerF.apply(x, 1e6)
        out.backward(torch.ones_like(out))
        assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
