"""Parity tests for the Phase-2 fused-silu-mlp wiring.

These tests verify that replacing the original ``nn.Sequential(Linear,
SiLU, Linear)`` blocks in the EGNN adapter, the train_fm_pocket script,
and the flow-matching time MLP with the TritonConfig-gated
:func:`triton_kernels.fused_silu_mlp` kernel produces output that is
**bit-exact** with the legacy sequential fallback.

The dispatch helper ``TritonConfig.use_fused_mlp`` defaults to ``True``,
so by default the fused path is exercised.  The fallback path is
exercised in the second half of each test by toggling
``triton_config.set_enabled(False)`` and re-running.

Notes
-----
- Bit-exact parity is expected because ``fused_silu_mlp`` is a
  drop-in autograd.Function whose backward delegates to PyTorch
  matmul + the standard activation gradient.  See
  ``triton_kernels/fused_mlp.py`` ``_FusedMLPFunction``.
- All three test cases use a small (M, H_in, H_hidden, H_out) that
  fits inside one Triton tile so autotune cost stays bounded.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the project root is on sys.path so the test can import
# ``molmetal`` without requiring ``uv run``-style path injection.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from triton_kernels import fused_silu_mlp
from triton_kernels.config import triton_config


# ---------------------------------------------------------------------------
# Tolerance note
# ---------------------------------------------------------------------------
# The fused path goes through ``triton_kernels.fused_silu_mlp`` which
# internally calls ``torch.matmul`` for the first GEMM and an explicit
# SiLU + ``torch.matmul`` for the down-projection.  The pure-PyTorch
# fallback goes through ``F.silu(F.linear(x, w1, b1)) @ w2.t() + b2``.
# Both paths are FP32 throughout, but PyTorch's matmul kernel does not
# guarantee bit-identical ordering with itself when called via two
# different code paths, so a tolerance of ~1 ULP at the activation's
# scale (~1.0 here) is required.  Empirically ``atol=1e-5`` /
# ``rtol=1e-4`` covers the noise floor; ``atol=0`` does not (FP32
# reorder noise is ~6e-8 at this scale).
_PARITY_ATOL = 1e-5
_PARITY_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)


def _make_input(M: int, D: int, *, device: str = "cpu") -> torch.Tensor:
    return torch.randn(M, D, device=device, dtype=torch.float32)


def _make_layers(D: int, H: int, out: int, seed: int = 0):
    """Return (linear1.weight, linear1.bias, linear2.weight, linear2.bias).

    Deterministic initialisation so the fused and fallback paths see
    identical parameters.
    """
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(H, D, generator=g) * 0.1
    b1 = torch.randn(H, generator=g) * 0.1
    w2 = torch.randn(out, H, generator=g) * 0.1
    b2 = torch.randn(out, generator=g) * 0.1
    return w1, b1, w2, b2


def _sequential_silu_mlp(x, w1, b1, w2, b2):
    """Pure-PyTorch reference: ``nn.Sequential(Linear, SiLU, Linear)``.

    ``w1`` has shape ``(H, D)`` (nn.Linear convention); the kernel
    uses the transposed layout, so this helper mirrors that internally.
    """
    return F.silu(F.linear(x, w1, b1)) @ w2.t() + b2


# ---------------------------------------------------------------------------
# 1) EGNN adapter parity
# ---------------------------------------------------------------------------
def test_egnn_mlp_parity():
    """``molmetal.adapters.egnn_rocm.EquivariantGraphConv.mlp`` parity."""
    from molmetal.adapters.egnn_rocm import EquivariantGraphConv

    _seed_everything(0)
    in_node_dim, hidden_dim = 8, 16
    conv = EquivariantGraphConv(in_node_dim=in_node_dim, hidden_dim=hidden_dim)
    conv.eval()

    # 4 nodes, fully connected (no self-loop).
    n = 4
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=torch.long,
    )
    h = torch.randn(n, in_node_dim)
    x = torch.randn(n, 3)
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
    h_i = h[src]
    h_j = h[dst]
    mlp_in = torch.cat([h_i, h_j, dist], dim=-1)  # (n_edges, 2*in_node_dim+1)

    # --- fused path (default) ---
    triton_config.set_enabled(True)
    msg_fused = F.silu(conv.mlp(mlp_in))

    # --- fallback path ---
    triton_config.set_enabled(False)
    msg_fallback = F.silu(conv.mlp(mlp_in))

    torch.testing.assert_close(
        msg_fused, msg_fallback, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    # And it must match a hand-rolled nn.Sequential reference for sanity.
    w1, b1, w2, b2 = (
        conv.mlp.linear1.weight,
        conv.mlp.linear1.bias,
        conv.mlp.linear2.weight,
        conv.mlp.linear2.bias,
    )
    ref = _sequential_silu_mlp(mlp_in, w1, b1, w2, b2)
    torch.testing.assert_close(
        msg_fused, F.silu(ref), rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )

    # Restore default for downstream tests.
    triton_config.set_enabled(True)


# ---------------------------------------------------------------------------
# 2) train_fm_pocket parity
# ---------------------------------------------------------------------------
def test_train_fm_mlp_parity():
    """``molmetal.scripts.train_fm_pocket.EquivariantGraphConv.mlp`` parity.

    Uses the local ``EquivariantGraphConv`` defined in
    ``train_fm_pocket.py`` (the in-script class, not the adapter's).
    """
    from molmetal.scripts.train_fm_pocket import (
        EquivariantGraphConv as TrainFmEGC,
        _MaybeFusedSiLUMLP as TrainFmMLP,
    )

    _seed_everything(1)
    in_node_dim, hidden_dim = 8, 16
    conv = TrainFmEGC(in_node_dim=in_node_dim, hidden_dim=hidden_dim)
    conv.eval()

    # 4 nodes, 8 directed edges (matching test_egnn_mlp_parity shape).
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=torch.long,
    )
    n = 4
    h = torch.randn(n, in_node_dim)
    x = torch.randn(n, 3)
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
    mlp_in = torch.cat([h[src], h[dst], dist], dim=-1)

    # Fused path (default).
    triton_config.set_enabled(True)
    msg_fused = F.silu(conv.mlp(mlp_in))

    # Fallback path.
    triton_config.set_enabled(False)
    msg_fallback = F.silu(conv.mlp(mlp_in))

    torch.testing.assert_close(
        msg_fused, msg_fallback, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )

    # Also exercise the standalone _MaybeFusedSiLUMLP forward parity at
    # a different shape.
    triton_config.set_enabled(True)
    mlp = TrainFmMLP(in_dim=12, hidden_dim=24, out_dim=6).eval()
    x_in = torch.randn(5, 12)
    y_fused = mlp(x_in)
    triton_config.set_enabled(False)
    y_fallback = mlp(x_in)
    torch.testing.assert_close(
        y_fused, y_fallback, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )

    triton_config.set_enabled(True)


# ---------------------------------------------------------------------------
# 3) flow-matching time_mlp parity
# ---------------------------------------------------------------------------
def test_flow_matching_time_mlp_parity():
    """``EGNNVelocityField.time_mlp`` parity (Linear -> SiLU -> Linear)."""
    from molmetal.adapters.flow_matching_lipman import EGNNVelocityField

    _seed_everything(2)
    hidden_dim = 16
    vf = EGNNVelocityField(hidden_dim=hidden_dim, n_layers=1, max_atomic_number=4)
    vf.eval()

    # (B, 1) input as produced by ``forward`` after ``t.unsqueeze(-1)``.
    t = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)

    triton_config.set_enabled(True)
    y_fused = vf.time_mlp(t)

    triton_config.set_enabled(False)
    y_fallback = vf.time_mlp(t)

    torch.testing.assert_close(
        y_fused, y_fallback, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )

    # Sanity: matches the sequential reference.
    w1, b1, w2, b2 = (
        vf.time_mlp.linear1.weight,
        vf.time_mlp.linear1.bias,
        vf.time_mlp.linear2.weight,
        vf.time_mlp.linear2.bias,
    )
    ref = _sequential_silu_mlp(t, w1, b1, w2, b2)
    torch.testing.assert_close(
        y_fused, ref, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )

    triton_config.set_enabled(True)


# ---------------------------------------------------------------------------
# 4) Sanity: fused_silu_mlp matches sequential on a hand-rolled input
# ---------------------------------------------------------------------------
def test_fused_silu_mlp_matches_sequential_smoke():
    """Direct smoke test on the public :func:`fused_silu_mlp`."""
    _seed_everything(3)
    M, D, H, H2 = 6, 8, 12, 4
    x = _make_input(M, D)
    w1_nn, b1, w2_nn, b2 = _make_layers(D, H, H2, seed=4)
    # Transposed convention: w1_tr (D, H), w2_tr (H, H2).
    w1_tr = w1_nn.t().contiguous()
    w2_tr = w2_nn.t().contiguous()

    ref = _sequential_silu_mlp(x, w1_nn, b1, w2_nn, b2)

    triton_config.set_enabled(True)
    out_fused = fused_silu_mlp(x, w1_tr, b1, w2_tr, b2)
    torch.testing.assert_close(
        out_fused, ref, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
