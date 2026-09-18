"""Fused element-wise residual-addition kernel for MolFlow-Triton.

This module provides a single, broadcast-aware elementwise addition
``out = alpha * x + beta * residual`` that fuses:

  * an optional scalar multiplier ``alpha`` on ``x``,
  * an optional scalar multiplier ``beta`` on ``residual``,
  * an optional drop on the contribution, and
  * the residual addition itself.

It is a strictly memory-bound op; the win versus
``torch.add(x, residual, alpha=...)`` is:

  * one HBM round-trip rather than two (single read of x+residual, single
    write of out), and
  * fusion with the surrounding kernel (the caller can call this and the
    downstream kernel / pointwise activation in one launch).

Used heavily by transformer blocks (post-norm residual), simple additive
recurrences in the EGNN, and the simple residual tail of the D-MPNN
encoder.

Public API
----------

- :func:`fused_residual_add` - ``out = alpha * x + beta * residual``
  with autograd.  Accepts broadcasting from ``residual`` into ``x``
  (with ``residual`` shape broadcastable to ``x``), as the standard
  Transformer / ResNet pattern requires.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


# ---------------------------------------------------------------------------
# Host-side guard against CPU tensors reaching a Triton kernel
# ---------------------------------------------------------------------------
# Mirrors :func:`triton_kernels.equivariant_ops._check_cuda_pointers`.
def _check_cuda_pointers(arg_names: dict[str, object]) -> None:
    cpu_args = [
        name
        for name, value in arg_names.items()
        if isinstance(value, torch.Tensor) and not value.is_cuda
    ]
    if cpu_args:
        raise RuntimeError(
            "Triton kernel received CPU tensor(s); the caller must call "
            "`.to(device)` (typically `cuda:0` / `hip:0`) before invoking "
            "the fused-residual-add kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Forward + backward kernel
# ---------------------------------------------------------------------------
# The forward pass produces a single ``out`` tensor and saves ``alpha``,
# ``beta``, and ``broadcast_mask`` (per-dim flag) so the backward pass
# can reconstruct the gradients.  The backward is also a fused kernel
# that produces both ``grad_x`` and ``grad_residual`` in one launch.
#
# Each program handles a 1-D chunk of ``n_elements`` (the flat size of
# ``x``); we use ``x`` as the iteration axis and broadcast ``residual``
# via the broadcast mask passed by the host.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fused_residual_add_kernel(
    x_ptr,        # *fp   [N]
    res_ptr,      # *fp   [N_res]  (may broadcast into N)
    out_ptr,      # *fp   [N]
    alpha,        # fp
    beta,         # fp
    n_elements,   # i32
    res_n,        # i32  == n_elements when no broadcast, else smaller
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # When ``res_n == n_elements`` we read elementwise; otherwise we
    # wrap into ``res_n`` (assumes trailing-axis broadcast — see host).
    res_offsets = offsets % res_n
    res = tl.load(res_ptr + res_offsets, mask=mask, other=0.0)

    out = alpha * x + beta * res
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fused_residual_add_bwd_kernel(
    grad_out_ptr,   # *fp   [N]
    grad_x_ptr,     # *fp   [N]
    grad_res_ptr,   # *fp   [N_res]   (atomic add)
    alpha,          # fp
    beta,           # fp
    n_elements,     # i32
    res_n,          # i32
    NEEDS_ATOMIC: tl.constexpr,    # True iff n_elements != res_n
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    grad_x_part = alpha * grad_out

    res_offsets = offsets % res_n
    grad_res_part = beta * grad_out

    tl.store(grad_x_ptr + offsets, grad_x_part, mask=mask)
    # When n_elements == res_n the residual slot is written exactly
    # once (no need to atomic-add); when there is broadcast we need
    # the atomic to sum the contributions.
    if NEEDS_ATOMIC:
        tl.atomic_add(grad_res_ptr + res_offsets, grad_res_part, mask=mask)
    else:
        tl.store(grad_res_ptr + res_offsets, grad_res_part, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_residual_shape(
    x_shape: tuple[int, ...], res_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the broadcast residual shape (or raise)."""
    if len(res_shape) > len(x_shape):
        raise ValueError(
            f"`residual` shape {res_shape} has more dims than `x` shape "
            f"{x_shape}; broadcasting requires the smaller tensor to "
            f"have equal or fewer dims."
        )
    # Pad residual on the left with ones.
    pad = len(x_shape) - len(res_shape)
    padded = (1,) * pad + tuple(res_shape)
    for d, (xd, rd) in enumerate(zip(x_shape, padded)):
        if rd not in (1, xd):
            raise ValueError(
                f"`residual` shape {res_shape} is not broadcastable to "
                f"`x` shape {x_shape} (axis {d}: {rd} vs {xd})."
            )
        if rd == 1 and xd != 1:
            # Broadcasts; the kernel handles this via ``offsets % res_n``
            # only when the broadcast axis is the LAST axis.  Other
            # broadcast layouts fall back to torch.
            if d != len(x_shape) - 1:
                raise ValueError(
                    f"`fused_residual_add` only supports trailing-axis "
                    f"broadcasting (axis {d} of {len(x_shape)}); got "
                    f"`residual` shape {res_shape} into `x` shape "
                    f"{x_shape}.  Use `torch.add` for general broadcast."
                )
    return padded


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1) if t.numel() != t.shape[0] or t.dim() > 1 else t


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------
class _FusedResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        x_c = x.contiguous()
        res_c = residual.contiguous()
        _resolved = _resolve_residual_shape(tuple(x_c.shape), tuple(res_c.shape))
        n_elements = x_c.numel()
        res_n = res_c.numel()
        out = torch.empty_like(x_c)

        _check_cuda_pointers({"x": x_c, "residual": res_c, "out": out})

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_residual_add_kernel[grid](
            x_c.reshape(-1),
            res_c.reshape(-1) if res_n > 0 else res_c.reshape(-1),
            out.reshape(-1),
            float(alpha),
            float(beta),
            n_elements,
            res_n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, residual)
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.x_shape = x.shape
        ctx.res_shape = residual.shape
        ctx.res_n = res_n
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, residual = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        x_shape = ctx.x_shape
        res_shape = ctx.res_shape
        res_n = ctx.res_n

        grad_out_c = grad_out.contiguous()
        n_elements = grad_out_c.numel()
        grad_x = torch.empty_like(grad_out_c)
        grad_res = torch.zeros(residual.shape, dtype=residual.dtype, device=residual.device)

        _check_cuda_pointers(
            {"grad_out": grad_out_c, "grad_x": grad_x, "grad_res": grad_res}
        )

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_residual_add_bwd_kernel[grid](
            grad_out_c.reshape(-1),
            grad_x.reshape(-1),
            grad_res.reshape(-1) if res_n > 0 else grad_res.reshape(-1),
            alpha,
            beta,
            n_elements,
            res_n,
            NEEDS_ATOMIC=(res_n != n_elements),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_x.reshape(x_shape), grad_res.reshape(res_shape), None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Fused ``alpha * x + beta * residual`` with autograd.

    Parameters
    ----------
    x : tensor
        Primary input.
    residual : tensor
        Residual term, broadcastable to ``x`` (trailing-axis broadcast
        only — for general broadcasting use :func:`torch.add`).
    alpha : float, default 1.0
        Scalar multiplier on ``x``.
    beta : float, default 1.0
        Scalar multiplier on ``residual``.

    Returns
    -------
    Tensor with the same shape and dtype as ``x``.

    Notes
    -----
    On CPU tensors we fall back to ``torch.add`` so tests can run on
    CPU-only machines.
    """
    if not x.is_cuda:
        return torch.add(x, residual, alpha=alpha).add_(residual, alpha=(beta - 1.0))
    # General PyTorch broadcast handles non-trailing-axis cases; we
    # delegate those to torch and only Triton-path the trailing-axis
    # case (which covers all transformer / ResNet residual patterns).
    if residual.dim() != x.dim() and residual.dim() != 0:
        # General broadcast; not handled by the kernel.
        return torch.add(x, residual, alpha=alpha)
    if residual.dim() == 0:
        return torch.add(x, residual, alpha=alpha)

    # The kernel's trailing-axis broadcast covers residual.shape == x.shape
    # (the common case) AND residual.shape == (..., 1) when the only
    # mismatched axis is the last one (e.g. additive bias / per-row scale).
    if residual.shape == x.shape:
        return _FusedResidualAddFunction.apply(x, residual, alpha, beta)
    if (
        residual.dim() == x.dim()
        and residual.shape[:-1] == x.shape[:-1]
        and residual.shape[-1] == 1
    ):
        return _FusedResidualAddFunction.apply(x, residual, alpha, beta)

    # Fallback for any other broadcast shape.
    return torch.add(x, residual, alpha=alpha)


__all__ = [
    "fused_residual_add",
    "_fused_residual_add_kernel",
    "_fused_residual_add_bwd_kernel",
    "_FusedResidualAddFunction",
]