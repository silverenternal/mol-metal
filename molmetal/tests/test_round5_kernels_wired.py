"""Tests for the round-5 fused-kernel wire-up into production models.

The four round-5 kernels shipped as opt-in Triton kernels:

- :func:`triton_kernels.fused_dropout_residual`
- :func:`triton_kernels.fused_residual_add`
- :func:`triton_kernels.batched_fused_silu_mlp`
- :func:`triton_kernels.batched_fused_gelu_mlp`

are dispatched in production code paths through
:class:`triton_kernels.config.TritonConfig`'s :meth:`should_use_fused`
helper.  This module pins the wiring contract end-to-end:

1. The :data:`triton_kernels.triton_config` singleton honours the
   default ON-training / OFF-eval policy.
2. ``molmetal.models.dmpnn`` exposes ``_maybe_fused_dropout_only`` and
   ``_maybe_fused_dropout_residual`` helpers that route through the
   fused kernel when the gate is open and fall back to PyTorch
   ``F.dropout`` otherwise.
3. ``molmetal.models.metal_hybrid_v4`` exposes a
   ``_maybe_fused_residual_add`` helper and a
   ``_maybe_batched_silu_mlp`` helper that both honour the gate.
4. ``molmetal.scripts.pretrain_coordination`` exposes a
   ``_maybe_batched_gelu_mlp`` helper.
5. All wrappers degrade cleanly on CPU (no crash when the Triton
   kernel rejects a CPU tensor).

Run with::

    uv run pytest -q molmetal/tests/test_round5_kernels_wired.py --tb=short
"""
from __future__ import annotations

import torch

from triton_kernels.config import triton_config as _triton_config


_PARITY_ATOL = 1e-5
_PARITY_RTOL = 1e-4


# ---------------------------------------------------------------------------
# TritonConfig gates — round-5 dispatch policy
# ---------------------------------------------------------------------------
def test_should_use_fused_default_training_on_eval_off() -> None:
    """Default policy: training ON (with the global flag), eval OFF.

    We simulate "eval" by calling :meth:`should_use_fused` with
    ``training=False`` (override).  In that state the gate should return
    ``False`` unless :attr:`TritonConfig.enabled_eval` is set.
    """
    x = torch.randn(8192, 64, dtype=torch.float32)  # well above MIN_BYTES
    on_train = _triton_config.should_use_fused(x, op="residual_add", training=True)
    on_eval = _triton_config.should_use_fused(x, op="residual_add", training=False)
    # Training path: True when the global flag is on.
    assert on_train is True, (
        f"training-mode should_use_fused gate must be True under the "
        f"default ON-training policy, got {on_train}"
    )
    # Eval mode: default OFF — must be False unless ``enabled_eval`` is on.
    assert on_eval is False, (
        f"eval-mode should_use_fused gate must be False under the "
        f"default OFF-eval policy, got {on_eval}"
    )


def test_should_use_fused_force_off() -> None:
    """``force=True`` always returns ``False`` (force-PyTorch fallback)."""
    x = torch.randn(8192, 64, dtype=torch.float32)
    out = _triton_config.should_use_fused(
        x, op="residual_add", training=True, force=True
    )
    assert out is False


# ---------------------------------------------------------------------------
# molmetal.models.dmpnn — _maybe_fused_dropout_only wiring
# ---------------------------------------------------------------------------
def test_dmpnn_dropout_only_helper() -> None:
    """``molmetal.models.dmpnn._maybe_fused_dropout_only`` exists and
    dispatches to ``F.dropout`` on CPU without raising."""
    from molmetal.models.dmpnn import _maybe_fused_dropout_only

    x = torch.randn(8, 16, dtype=torch.float32)
    # p=0.0 -> identity (no dropout applied).
    out_p0 = _maybe_fused_dropout_only(x, p=0.0, training=True)
    assert torch.equal(out_p0, x), "p=0.0 dropout must be a no-op identity"
    # training=False -> early-out identity (dropout is no-op in eval).
    out_eval = _maybe_fused_dropout_only(x, p=0.5, training=False)
    assert torch.equal(out_eval, x), (
        "training=False dropout must be a no-op identity"
    )


def test_dmpnn_dropout_residual_helper() -> None:
    """``molmetal.models.dmpnn._maybe_fused_dropout_residual`` exists and
    degrades gracefully on CPU."""
    from molmetal.models.dmpnn import _maybe_fused_dropout_residual

    x = torch.randn(8, 16, dtype=torch.float32)
    residual = torch.randn(8, 16, dtype=torch.float32)
    # training=False -> identity (dropout is a no-op in eval); returns x+residual.
    out_eval = _maybe_fused_dropout_residual(x, residual, p=0.5, training=False)
    assert torch.allclose(out_eval, x + residual, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)
    # p=0.0 + training=True on CPU -> x + residual (dropout is no-op when p=0).
    out_p0 = _maybe_fused_dropout_residual(x, residual, p=0.0, training=True)
    assert torch.allclose(out_p0, x + residual, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)


# ---------------------------------------------------------------------------
# molmetal.models.metal_hybrid_v4 — _maybe_fused_residual_add + batched silu mlp
# ---------------------------------------------------------------------------
def test_metal_hybrid_v4_residual_add_helper() -> None:
    """``molmetal.models.metal_hybrid_v4._maybe_fused_residual_add`` exists
    and degrades gracefully on CPU."""
    from molmetal.models.metal_hybrid_v4 import _maybe_fused_residual_add

    x = torch.randn(4, 8, dtype=torch.float32)
    res = torch.randn(4, 8, dtype=torch.float32)
    out = _maybe_fused_residual_add(x, res, training=True)
    # CPU fallback: alpha=1.0, beta=1.0 -> x + res.
    assert torch.allclose(
        out, x + res, atol=_PARITY_ATOL, rtol=_PARITY_RTOL
    ), (
        f"CPU residual_add fallback must equal x + res; "
        f"max diff {(out - (x + res)).abs().max().item()}"
    )

    # Scaled alpha / beta path.
    out_scaled = _maybe_fused_residual_add(
        x, res, alpha=0.5, beta=2.0, training=True
    )
    expected_scaled = 0.5 * x + 2.0 * res
    assert torch.allclose(
        out_scaled, expected_scaled, atol=_PARITY_ATOL, rtol=_PARITY_RTOL
    )


def test_metal_hybrid_v4_batched_silu_mlp_helper() -> None:
    """``molmetal.models.metal_hybrid_v4._maybe_batched_silu_mlp`` exists
    and runs on CPU via the sequential fallback (CPU tensors skip the
    batched kernel)."""
    from molmetal.models.metal_hybrid_v4 import _maybe_batched_silu_mlp

    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 8, 12, 6
    K = 2  # 2 stacked SiLU MLP triplets
    M = 4

    x = torch.randn(M, in_dim, dtype=torch.float32)
    # Build the 2K Linear layers explicitly: each triplet is
    # ``Linear(in_k, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)``
    # for k=0, and ``Linear(hidden_dim, hidden_dim) → SiLU →
    # Linear(hidden_dim, out_dim)`` for k=K-1.  The inter-triplet hidden
    # dim must be ``hidden_dim`` so each triplet's output feeds the
    # next's input.
    linears: list[torch.nn.Linear] = []
    for k in range(K):
        if k == 0:
            linears.append(torch.nn.Linear(in_dim, hidden_dim))
        else:
            linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
        if k == K - 1:
            linears.append(torch.nn.Linear(hidden_dim, out_dim))
        else:
            linears.append(torch.nn.Linear(hidden_dim, hidden_dim))

    # Sequential reference: run the K triplets sequentially.
    h_ref = x
    for k in range(K):
        la = linears[2 * k]
        lb = linears[2 * k + 1]
        h_ref = torch.nn.functional.silu(la(h_ref))
        h_ref = lb(h_ref)

    out = _maybe_batched_silu_mlp(x, linears, training=True)
    assert out.shape == (M, out_dim), (
        f"_maybe_batched_silu_mlp output must be (M, {out_dim}); got "
        f"{tuple(out.shape)}"
    )
    # CPU fallback is the sequential path, so the output must match
    # the reference bit-exactly (no Triton kernel involved on CPU).
    assert torch.allclose(
        out, h_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL
    ), (
        f"_maybe_batched_silu_mlp sequential fallback must match the "
        f"reference; max diff {(out - h_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# molmetal.scripts.pretrain_coordination — _maybe_batched_gelu_mlp wiring
# ---------------------------------------------------------------------------
def test_pretrain_coordination_batched_gelu_mlp_helper() -> None:
    """``molmetal.scripts.pretrain_coordination._maybe_batched_gelu_mlp``
    exists and degrades gracefully on CPU."""
    from molmetal.scripts.pretrain_coordination import _maybe_batched_gelu_mlp

    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 8, 12, 6
    K = 2  # 2 stacked GELU MLP triplets
    M = 4

    x = torch.randn(M, in_dim, dtype=torch.float32)
    # Build the 2K Linear layers explicitly: each triplet is
    # ``Linear(in_k, hidden_dim) → GELU → Linear(hidden_dim, out_k)``
    # where the inter-triplet hidden dim must match.
    linears: list[torch.nn.Linear] = []
    for k in range(K):
        if k == 0:
            linears.append(torch.nn.Linear(in_dim, hidden_dim))
        else:
            linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
        if k == K - 1:
            linears.append(torch.nn.Linear(hidden_dim, out_dim))
        else:
            linears.append(torch.nn.Linear(hidden_dim, hidden_dim))

    # Sequential reference.
    h_ref = x
    for k in range(K):
        la = linears[2 * k]
        lb = linears[2 * k + 1]
        h_ref = torch.nn.functional.gelu(la(h_ref))
        h_ref = lb(h_ref)

    out = _maybe_batched_gelu_mlp(x, linears, training=True)
    assert out.shape == (M, out_dim), (
        f"_maybe_batched_gelu_mlp output must be (M, {out_dim}); got "
        f"{tuple(out.shape)}"
    )
    assert torch.allclose(
        out, h_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL
    ), (
        f"_maybe_batched_gelu_mlp sequential fallback must match the "
        f"reference; max diff {(out - h_ref).abs().max().item()}"
    )


def test_pretrain_coordination_gelu_head_instantiation() -> None:
    """``CoordinationPretrainer(use_gelu_head=True)`` builds the head with
    GELU activations and produces a (B, 2) regression output."""
    from molmetal.scripts.pretrain_coordination import CoordinationPretrainer

    model = CoordinationPretrainer(use_gelu_head=True)
    model.train()
    # Verify the head uses GELU (not ReLU).
    assert isinstance(model.head[1], torch.nn.GELU), (
        f"use_gelu_head=True must swap the head activation to GELU; "
        f"got {type(model.head[1]).__name__}"
    )
    assert isinstance(model.head[4], torch.nn.GELU), (
        f"use_gelu_head=True must swap the second head activation to GELU; "
        f"got {type(model.head[4]).__name__}"
    )
    # Smoke-test forward pass with a tiny D-MPNN-compatible batch.
    n_atoms = 10
    batch = {
        "x": torch.randn(n_atoms, 39, dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.randn(2, 6, dtype=torch.float32),
        "batch": torch.zeros(n_atoms, dtype=torch.long),
        "metal_idx": torch.tensor([0], dtype=torch.long),
        "n_nodes": torch.tensor([[n_atoms]], dtype=torch.float32),
    }
    out_head = model(batch)
    assert out_head.shape == (1, 2), (
        f"CoordinationPretrainer output must be (B=1, 2); got {tuple(out_head.shape)}"
    )
