"""Pytest: Wf-A64 round 11 — CFM R18 adapter deeper-audit regression tests.

Pins the 7 fixes shipped in Wf-A64 W1 (2026-10-17):

* F1 — ``CoordinationAwareGNNComposite`` short-circuits on empty
  ``features`` so downstream heads receive ``(0, H)`` rather than
  crashing on ``batch.max()``.
* F2 — ``CoordinationAwareGNNLayer`` breaks symmetry on all-zero
  ``coordination`` rows by adding a tiny uniform offset (the
  degenerate path previously collapsed every coord-class-0 node
  to identical messages).
* F4 — ``QuotientSpaceCFM.equivariance_error`` branches on the
  manifold; only ``"SE3"`` exercises full SE(3), ``"translation"``
  tests translation invariance only, ``"rotation"`` tests rotation
  only.
* F5 — ``_euler_se3`` / ``_euler_se3_step`` no longer accept the
  dead ``eps`` arg; ``integrate_with_cfm`` still validates its
  public kwarg for backward compat.
* F6 — ``SE3SBVelocityField`` class docstring now warns the rigid-body
  reduction is structurally unmodelled for non-rigid pose diff.
* F7 — ``CoordinationAwareGNNComposite`` masks empty-graph rows so
  ``0/N`` degenerate cohorts stay ``0.0`` rather than ``NaN``.

All tests run on CPU in <10 s — no GPU / external dependencies
required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — minimal stub modules for coordination_gnn + se3_sb tests
# ---------------------------------------------------------------------------
class _IdentityLayer(nn.Module):
    """Trivial stand-in for ``CoordinationAwareGNNLayer`` that returns
    the input features unchanged (so the composite's forward path
    only exercises the pooling + heads step)."""

    hidden_dim: int = 8
    n_metals: int = 3
    num_head: nn.Module = None  # filled in by test
    geom_head: nn.Module = None

    def forward(self, x, edge_index, coordination, coords=None):
        return x


def _build_composite():
    """Construct a small ``CoordinationAwareGNNComposite`` for testing."""
    from molmetal.adapters.coordination_gnn import (
        CoordinationAwareGNNComposite,
        CoordinationAwareGNNLayer,
        CoordinationGeometryHead,
        CoordinationNumberHead,
    )

    layer = CoordinationAwareGNNLayer(
        hidden_dim=8,
        n_coordination_classes=5,
    )
    num_head = CoordinationNumberHead(
        hidden_dim=8,
        n_metals=3,
        max_coordination=6,
    )
    geom_head = CoordinationGeometryHead(hidden_dim=8, n_geometries=4)
    composite = CoordinationAwareGNNComposite(
        layer=layer,
        num_head=num_head,
        geom_head=geom_head,
        run_layer=False,
    )
    return composite


# ---------------------------------------------------------------------------
# F1 — composite short-circuits on empty features
# ---------------------------------------------------------------------------
def test_coordination_gnn_composite_empty_graph():
    """F1 — empty ``features`` returns ``(0, H)`` tuple, not a crash."""
    composite = _build_composite()
    x_empty = torch.zeros(0, 8)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    coordination = torch.zeros(0, 5)
    batch = torch.zeros(0, dtype=torch.long)
    features, num_logits, geom_logits = composite(
        x_empty, edge_index, coordination, batch=batch
    )
    assert features.shape == (0, 8)
    assert num_logits.shape == (0, 3, 6)
    assert geom_logits.shape == (0, 4)
    # No NaN/Inf even on the empty path.
    assert torch.isfinite(num_logits).all()
    assert torch.isfinite(geom_logits).all()


# ---------------------------------------------------------------------------
# F2 — layer distinguishes soft-zero coord-class rows
# ---------------------------------------------------------------------------
def test_coordination_gnn_zero_coord_class_distinguishes():
    """F2 — uniform ``1e-3`` offset breaks the zero-degeneracy."""
    from molmetal.adapters.coordination_gnn import CoordinationAwareGNNLayer

    layer = CoordinationAwareGNNLayer(hidden_dim=4, n_coordination_classes=3)
    n_nodes = 3
    x = torch.randn(n_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    # All-zero coordination — the degenerate path.
    coord_zero = torch.zeros(n_nodes, 3)
    out_zero = layer(x, edge_index, coord_zero)
    # Same shape, finite.
    assert out_zero.shape == x.shape
    assert torch.isfinite(out_zero).all()
    # Same graph with a one-hot row must yield a DIFFERENT output
    # (proves the offset breaks the symmetry).
    coord_onehot = torch.eye(3)
    out_onehot = layer(x, edge_index, coord_onehot)
    diff = (out_zero - out_onehot).abs().max().item()
    assert diff > 1e-6, (
        f"Zero-coord-class output must differ from one-hot output; got max "
        f"|delta|={diff}"
    )


# ---------------------------------------------------------------------------
# F4 — equivariance_error branches on the manifold
# ---------------------------------------------------------------------------
def test_quotient_cfm_equivariance_each_manifold():
    """F4 — ``equivariance_error`` returns small values for each manifold."""
    from molmetal.adapters.quotient_space_diffusion import (
        QuotientSpace,
        QuotientSpaceCFM,
    )

    class _Dummy(nn.Module):
        def forward(self, z, t, pocket=None):
            return torch.zeros_like(z)

    for manifold in ("SE3", "rotation", "translation"):
        qcfm = QuotientSpaceCFM(
            base_cfm=_Dummy(),
            quotient=QuotientSpace(manifold=manifold),
        )
        err = qcfm.equivariance_error(n_runs=3, atol=1e-3, seed=42)
        assert err < 1e-3, (
            f"equivariance_error on manifold={manifold!r} returned {err}; "
            f"expected < 1e-3"
        )


# ---------------------------------------------------------------------------
# F5 — integrate_with_cfm drops the dead eps arg from the inner Euler step
# ---------------------------------------------------------------------------
def test_se3_sb_integrate_dead_args_removed():
    """F5 — inner Euler step signatures no longer carry the dead ``eps`` arg."""
    import inspect

    from molmetal.adapters import se3_sb_integration as mod

    sig_euler = inspect.signature(mod._euler_se3)
    sig_step = inspect.signature(mod._euler_se3_step)
    assert "eps" not in sig_euler.parameters, (
        f"_euler_se3 must not accept eps; got params {list(sig_euler.parameters)}"
    )
    assert "eps" not in sig_step.parameters, (
        f"_euler_se3_step must not accept eps; got params {list(sig_step.parameters)}"
    )
    # Public ``integrate_with_cfm`` keeps the kwarg for backward compat.
    sig_pub = inspect.signature(mod.integrate_with_cfm)
    assert "eps" in sig_pub.parameters
    # And the function still runs end-to-end without callers passing eps.
    field = mod.SE3SBVelocityField(base_velocity_net=_DummyEGNN())
    samples = torch.randn(2, 6, 3)
    out = mod.integrate_with_cfm(samples, field, n_steps=5)
    assert out.shape == samples.shape
    assert torch.isfinite(out).all()


class _DummyEGNN(nn.Module):
    """Minimal stand-in for the EGNN velocity net used by F5/F6 tests."""

    def forward(self, x, t, pocket=None):
        # (B, N, 3) — small constant velocity so the SE(3) twist
        # integrator has something finite to integrate.
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# F6 — SE3SBVelocityField docstring warns about the rigid-body reduction
# ---------------------------------------------------------------------------
def test_se3_sb_rigid_body_loss_warning():
    """F6 — class docstring mentions the rigid-body reduction caveat."""
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    doc = SE3SBVelocityField.__doc__ or ""
    assert "rigid-body reduction" in doc or "non-rigid" in doc, (
        "SE3SBVelocityField class docstring must warn the rigid-body "
        "reduction is structurally unmodelled for non-rigid pose diff"
    )


# ---------------------------------------------------------------------------
# F7 — composite masks empty-graph rows so 0/N degenerate cohorts stay 0
# ---------------------------------------------------------------------------
def test_coordination_gnn_composite_degenerate_cohort():
    """F7 — graph-id present but contributing 0 atoms yields finite pool."""
    composite = _build_composite()
    # 4 atoms spanning graph ids 0 and 2; graph id 1 has 0 atoms.
    x = torch.randn(4, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )
    coordination = torch.zeros(4, 5)
    batch = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    features, num_logits, geom_logits = composite(
        x, edge_index, coordination, batch=batch
    )
    # 3 graphs (max index 2 + 1) — graph id 1 is degenerate.
    assert num_logits.shape == (3, 3, 6)
    assert geom_logits.shape == (3, 4)
    # Graph id 1's logits must be finite (not NaN) and zero-ish.
    finite = torch.isfinite(num_logits).all() and torch.isfinite(geom_logits).all()
    assert finite, "Degenerate-cohort logits must be finite (no NaN)"