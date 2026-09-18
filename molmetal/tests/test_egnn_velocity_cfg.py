"""Pytest: Round-10 axis-C — Classifier-Free Guidance (CFG) on the EGNN velocity field.

Five tests:

1. ``test_v_cfg_shapes_match_forward_velocity`` — ``v_cfg`` returns the
   same shape / dtype as ``forward_velocity``.
2. ``test_cfg_scale_one_equals_conditional_fast_path`` —
   ``v_cfg(pocket_embed, cfg_scale=1.0)`` is bit-exactly equal to
   ``forward_velocity(pocket_embed)`` (the fast-path short-circuit).
3. ``test_cfg_combination_is_correct`` — at ``cfg_scale=s``, the returned
   tensor equals ``v_uncond + s * (v_cond - v_uncond)``.
4. ``test_context_dropout_zeros_conditioning`` — after a forward pass in
   ``train()`` mode with ``pocket_embed`` set, at least one batch slice
   has been dropped (so the model learned v_uncond).  With dropout=0
   none are dropped (bit-exact legacy path).
5. ``test_cfg_unconditional_passthrough`` — when ``pocket_embed=None``
   ``v_cfg`` returns ``v_uncond`` regardless of ``cfg_scale``.

All tests use a tiny ``EGNNVelocityField(hidden_dim=32, n_layers=2)`` so
they run in seconds on CPU or ROCm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import EGNNVelocityField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def vfield():
    """A tiny EGNN velocity field on CPU (test runs in <1s)."""
    return EGNNVelocityField(
        hidden_dim=32, n_layers=2, max_atomic_number=20,
    )


@pytest.fixture
def batch_inputs(vfield):
    """A deterministic (B=4, N=6) batch + a (B, H) pocket embedding."""
    torch.manual_seed(0)
    b, n, h = 4, 6, vfield.atom_embed.embedding_dim
    x = torch.randn(b, n, 3)
    atom_types = torch.randint(1, 10, (b, n))
    # Edge index: fully-connected (no-self-loop), shape (B, 2, n*(n-1))
    idx = torch.arange(n)
    src = idx.view(1, n, 1).expand(b, n, n)
    dst = idx.view(1, 1, n).expand(b, n, n)
    mask = src != dst
    src = src[mask].view(b, -1)
    dst = dst[mask].view(b, -1)
    edge_index = torch.stack([src, dst], dim=1)
    t = torch.rand(b)
    pocket_embed = torch.randn(b, h)
    return x, atom_types, edge_index, t, pocket_embed


# ---------------------------------------------------------------------------
# Test 1: shape & dtype contract
# ---------------------------------------------------------------------------
def test_v_cfg_shapes_match_forward_velocity(vfield, batch_inputs):
    """v_cfg returns (B, N, 3) and same dtype as forward_velocity."""
    x, atom_types, edge_index, t, pocket_embed = batch_inputs
    vfield.eval()
    v_ref = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=pocket_embed,
    )
    v_cfg = vfield.v_cfg(
        x, atom_types, edge_index, t, pocket_embed=pocket_embed, cfg_scale=2.0,
    )
    assert v_cfg.shape == v_ref.shape == (4, 6, 3)
    assert v_cfg.dtype == v_ref.dtype
    # Finite values (no NaN / Inf from the linear combination).
    assert torch.isfinite(v_cfg).all()


# ---------------------------------------------------------------------------
# Test 2: cfg_scale=1.0 == conditional forward (fast-path)
# ---------------------------------------------------------------------------
def test_cfg_scale_one_equals_conditional_fast_path(vfield, batch_inputs):
    """When cfg_scale=1.0 the fast-path must be bit-exactly equal to the
    conditional ``forward_velocity`` call (no extra forward pass)."""
    x, atom_types, edge_index, t, pocket_embed = batch_inputs
    vfield.eval()
    v_ref = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=pocket_embed,
    )
    v_cfg = vfield.v_cfg(
        x, atom_types, edge_index, t,
        pocket_embed=pocket_embed, cfg_scale=1.0,
    )
    assert torch.allclose(v_cfg, v_ref, atol=0.0), (
        "cfg_scale=1.0 fast-path drifted from forward_velocity"
    )


# ---------------------------------------------------------------------------
# Test 3: CFG combination v_uncond + s · (v_cond − v_uncond)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_scale", [0.0, 1.5, 2.0, 3.0])
def test_cfg_combination_is_correct(vfield, batch_inputs, cfg_scale):
    """Numerical verification of the CFG linear combination."""
    x, atom_types, edge_index, t, pocket_embed = batch_inputs
    vfield.eval()
    v_cond = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=pocket_embed,
    )
    v_uncond = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=None,
    )
    expected = v_uncond + cfg_scale * (v_cond - v_uncond)
    v_cfg = vfield.v_cfg(
        x, atom_types, edge_index, t,
        pocket_embed=pocket_embed, cfg_scale=cfg_scale,
    )
    # Floating-point tolerance: the CFG path runs two forwards then a
    # linear combination; numerical noise from matmul order may differ
    # by ~1e-5 in float32.
    assert torch.allclose(v_cfg, expected, atol=1e-5), (
        f"CFG mismatch at cfg_scale={cfg_scale}: "
        f"max diff = {(v_cfg - expected).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: context dropout fires in training, disabled in eval / p=0
# ---------------------------------------------------------------------------
def test_context_dropout_fires_in_training(vfield, batch_inputs):
    """When the velocity field is in ``train()`` mode and
    ``context_dropout > 0``, repeated forward passes must show *some*
    drop events (i.e. not every batch slice is conditioned the same way
    every call).  We assert at least one batch index has ``v_drop ==
    v_uncond`` exactly (proving the zero-bias path was taken on that
    sample)."""
    vfield.context_dropout = 0.5  # crank up to make it easy to observe
    vfield.train()
    x, atom_types, edge_index, t, pocket_embed = batch_inputs

    # Compute the "all-dropped" reference (pocket_embed replaced by
    # zero everywhere, which is what happens on a fully-dropped batch).
    vfield.eval()
    v_uncond_ref = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=None,
    )

    # Now sample many forward passes in training mode and look for at
    # least one batch slice whose velocity matches v_uncond_ref exactly
    # (proving that sample was dropped).
    vfield.train()
    saw_drop = False
    for trial in range(64):
        out = vfield(x, atom_types, edge_index, t, pocket_embed=pocket_embed)
        v = out["vel"]
        # A sample was dropped iff ``v[i] == v_uncond_ref[i]`` bit-exactly.
        per_sample_match = (v == v_uncond_ref).all(dim=(-1, -2))
        if per_sample_match.any():
            saw_drop = True
            break
    assert saw_drop, "context_dropout did not fire in 64 training-mode passes"


def test_context_dropout_zero_is_bit_exact(vfield, batch_inputs):
    """With ``context_dropout=0`` the train-mode forward must be
    bit-exactly equal to the eval-mode forward (no random masking)."""
    vfield.context_dropout = 0.0
    x, atom_types, edge_index, t, pocket_embed = batch_inputs

    vfield.eval()
    v_eval = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=pocket_embed,
    )
    vfield.train()
    out = vfield(x, atom_types, edge_index, t, pocket_embed=pocket_embed)
    v_train = out["vel"]
    assert torch.allclose(v_train, v_eval, atol=0.0), (
        "context_dropout=0 broke bit-exact equivalence between train/eval"
    )


# ---------------------------------------------------------------------------
# Test 5: unconditional passthrough (pocket_embed=None)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_scale", [0.5, 1.0, 2.0, 5.0])
def test_cfg_unconditional_passthrough(vfield, batch_inputs, cfg_scale):
    """When ``pocket_embed=None`` v_cfg must return v_uncond regardless
    of cfg_scale (no guidance to apply)."""
    x, atom_types, edge_index, t, _pocket_embed = batch_inputs
    vfield.eval()
    v_uncond = vfield.forward_velocity(
        x, atom_types, edge_index, t, pocket_embed=None,
    )
    v_cfg = vfield.v_cfg(
        x, atom_types, edge_index, t,
        pocket_embed=None, cfg_scale=cfg_scale,
    )
    assert torch.allclose(v_cfg, v_uncond, atol=0.0), (
        f"unconditional v_cfg drifted at cfg_scale={cfg_scale}"
    )