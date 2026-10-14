"""Tests for molmetal.adapters.egnn_extensions.PerLayerPocketInjector.

These tests validate the class itself (the call site in
flow_matching_lipman/__init__.py:1717 currently does not instantiate
the class — that is audit finding #2 / HIGH, the wire-in is Wf-A32
work). The class must:

1. Be importable without the heavy PocketGen deps.
2. Honour its zero-init bit-exact legacy contract (zero input → zero
   output regardless of the bias term).
3. Project a non-zero input to a finite, correctly-shaped tensor.
4. Handle the ``pocket_embed=None`` fallback by returning zeros with
   batch=1.
5. Be deeply cheap to instantiate (CPU-only, no external dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.egnn_extensions import PerLayerPocketInjector  # noqa: E402


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_no_deps():
    """Class importable + instantiable without torch_scatter / esm / etc."""
    inj = PerLayerPocketInjector(pocket_dim=128, hidden_dim=128)
    assert isinstance(inj, torch.nn.Module)


def test_projection_weight_zero_at_init():
    """Zero-init must hold for bit-exact legacy compatibility."""
    inj = PerLayerPocketInjector(pocket_dim=64, hidden_dim=32)
    assert torch.equal(inj.proj.weight, torch.zeros_like(inj.proj.weight))
    if inj.proj.bias is not None:
        assert torch.equal(inj.proj.bias, torch.zeros_like(inj.proj.bias))


# ------------------------------------------------------------------
# Forward — bit-exact legacy contract
# ------------------------------------------------------------------

def test_forward_pocket_none_returns_zeros():
    """When pocket_embed is None, return zeros (batch=1 fallback)."""
    inj = PerLayerPocketInjector(pocket_dim=64, hidden_dim=32)
    out = inj(None, n_max=8)
    assert out.shape == (1, 8, 32)
    assert torch.equal(out, torch.zeros(1, 8, 32))


def test_forward_zero_input_yields_zero_output():
    """Zero pocket input must yield zero output, regardless of bias init.

    The bias is also zero-initialised, so the projection of zero is
    zero — this preserves the bit-exact legacy contract where no
    pocket signal ever enters the EGNN loop body.
    """
    inj = PerLayerPocketInjector(pocket_dim=64, hidden_dim=32)
    pocket = torch.zeros(2, 64)
    out = inj(pocket, n_max=10)
    assert out.shape == (2, 10, 32)
    assert torch.allclose(out, torch.zeros(2, 10, 32))


# ------------------------------------------------------------------
# Forward — non-zero input
# ------------------------------------------------------------------

def test_forward_nonzero_input_yields_nonzero_output_after_nontrivial_init():
    """Sanity: after a non-zero init, a non-zero input gives non-zero output."""
    inj = PerLayerPocketInjector(pocket_dim=64, hidden_dim=32)
    # Replace the zero-init with a random init to verify the projector
    # actually does something when not zeroed.
    with torch.no_grad():
        inj.proj.weight.normal_(mean=0.0, std=0.1)
        if inj.proj.bias is not None:
            inj.proj.bias.normal_(mean=0.0, std=0.1)

    pocket = torch.randn(3, 64)
    out = inj(pocket, n_max=5)
    assert out.shape == (3, 5, 32)
    # The output is broadcast to N atoms per batch element — verify
    # all 5 atoms in each batch element share the same value (this
    # is the per-layer-pulse contract).
    for b in range(3):
        for n1 in range(5):
            for n2 in range(n1 + 1, 5):
                assert torch.allclose(out[b, n1], out[b, n2]), (
                    f"atoms {n1} and {n2} in batch {b} differ — broadcast broken"
                )
    # And that across batches the values differ (the projector output
    # depends on pocket_embed which is per-batch).
    assert not torch.allclose(out[0, 0], out[1, 0]), (
        "per-batch values are identical — projector is broken"
    )


def test_forward_broadcast_against_egnn_h():
    """Simulate the EGNN forward: out has shape (B, N, hidden), so it
    can be added to ``h`` of shape (B, N, hidden) without unsqueeze.
    """
    inj = PerLayerPocketInjector(pocket_dim=128, hidden_dim=64)
    pocket = torch.randn(2, 128)
    n_atoms = 12
    h = torch.randn(2, n_atoms, 64)
    out = inj(pocket, n_max=n_atoms)
    assert out.shape == h.shape  # addable
    h_with_pocket = h + out
    assert h_with_pocket.shape == h.shape


# ------------------------------------------------------------------
# Wf-A32 wire-in contract
# ------------------------------------------------------------------

def test_pocket_dim_can_differ_from_hidden_dim():
    """Wf-A32 contract: pocket_embed_dim from PocketGen may differ from
    the EGNN hidden_dim. The injector should accept any (pocket_dim,
    hidden_dim) pair.
    """
    inj = PerLayerPocketInjector(pocket_dim=256, hidden_dim=64)
    pocket = torch.randn(2, 256)
    out = inj(pocket, n_max=5)
    assert out.shape == (2, 5, 64)
