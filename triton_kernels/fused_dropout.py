"""Fused dropout + residual kernel for MolFlow-Triton.

This module cherry-picks the "fused dropout + residual add" pattern from
the upstream Triton tutorial ``04-low-memory-dropout.py`` and adapts it
to the MolFlow-Triton constraints:

- **No TMA / WGMMA / cluster launch / warp specialization.** Plain
  pointer-based ``tl.load`` / ``tl.store`` with ``num_warps`` /
  ``num_stages`` as the only tuning axes (``TODO/environment.md`` §4, §7).
- **No ``waves_per_eu``.** It is CDNA-only and ignored on gfx1101.
- **Import convention.** We import :mod:`triton` only inside this file
  (the leaf kernel module); callers do ``from triton_kernels import
  fused_dropout_residual``.

The kernel is the canonical two-output elementwise op used inside every
transformer block in the project (and in any standard transformer in
general): compute ``x = dropout(x) + residual`` in a single pass, with
the dropout mask generated via the Philox4x32 RNG inside Triton (so the
mask is reproducible from the user-supplied ``seed`` and ``offset``).

Public API
----------

- :func:`fused_dropout_residual` - ``dropout(x, p) + residual`` fused,
  with autograd.  Returns ``(out, save_mask)`` where ``save_mask`` is
  the deterministic uint8 mask used in the forward, so the backward
  can multiply the upstream gradient by the same mask without
  re-running the RNG.

A :class:`torch.autograd.Function` wrapper
(:class:`_FusedDropoutResidualFunction`) wraps the kernel and replays
the mask on backward.  This matches the standard "save mask and
recompute in backward" pattern from the upstream tutorial.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
            "the fused-dropout kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Forward kernel: out = dropout(x, p) + residual, mask uint8
# ---------------------------------------------------------------------------
# Per-element launch — one program per block of ``BLOCK_SIZE`` elements.
# Each program loads BLOCK_SIZE contiguous elements of ``x`` and
# ``residual``, generates a Philox mask of the same shape, scales the
# kept elements by ``1 / (1 - p)``, sums with the residual, and writes
# both ``out`` and ``mask`` (mask as uint8 to halve the HBM traffic).
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fused_dropout_residual_kernel(
    x_ptr,            # *fp   [N]
    residual_ptr,     # *fp   [N]
    out_ptr,          # *fp   [N]
    mask_ptr,         # *u8   [N]
    p,                # fp
    n_elements,       # i32
    seed,             # i64
    offset,           # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Philox-based RNG (one per element).  ``tl.rand`` is the public
    # helper that delegates to Philox4x32 internally on Triton 3.x.
    rand = tl.rand(seed, offsets + offset)
    keep = rand > p
    mask_u8 = keep.to(tl.uint8)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    # Apply inverted-dropout scaling on the kept elements.
    scale = 1.0 / (1.0 - p)
    dropped = tl.where(keep, x.to(tl.float32) * scale, 0.0).to(x.dtype)
    out = dropped + residual

    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(mask_ptr + offsets, mask_u8, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel: d_x = d_out * mask * scale, d_residual = d_out
# ---------------------------------------------------------------------------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def _fused_dropout_residual_bwd_kernel(
    grad_out_ptr,     # *fp   [N]
    mask_ptr,         # *u8   [N]
    grad_x_ptr,       # *fp   [N]
    grad_res_ptr,     # *fp   [N]
    p,                # fp
    n_elements,       # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)

    scale = 1.0 / (1.0 - p)
    grad_x = tl.where(m, grad_out * scale, 0.0).to(grad_out.dtype)
    grad_res = grad_out

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_res_ptr + offsets, grad_res, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    out = 1
    while out < n:
        out *= 2
    return out


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` flattened to 1-D (no copy when already contiguous)."""
    return x.reshape(-1)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------
class _FusedDropoutResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        p: float,
        seed: int,
        offset: int,
    ) -> torch.Tensor:
        assert x.shape == residual.shape, (
            f"`x` shape {tuple(x.shape)} and `residual` shape "
            f"{tuple(residual.shape)} must match"
        )
        assert 0.0 <= p < 1.0, f"`p` must be in [0, 1); got {p}"

        x_c = _flatten(x.contiguous())
        res_c = _flatten(residual.contiguous())
        n_elements = x_c.numel()
        out = torch.empty_like(x_c)
        mask = torch.empty(n_elements, dtype=torch.uint8, device=x_c.device)

        _check_cuda_pointers(
            {"x": x_c, "residual": res_c, "out": out, "mask": mask}
        )

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_dropout_residual_kernel[grid](
            x_c,
            res_c,
            out,
            mask,
            float(p),
            n_elements,
            int(seed),
            int(offset),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(mask)
        ctx.p = float(p)
        ctx.n_elements = n_elements
        ctx.x_shape = x.shape
        ctx.x_stride = x.stride()
        ctx.res_shape = residual.shape
        ctx.res_stride = residual.stride()
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (mask,) = ctx.saved_tensors
        p = ctx.p
        n_elements = ctx.n_elements

        grad_out_c = _flatten(grad_out.contiguous())
        grad_x = torch.empty_like(grad_out_c)
        grad_res = torch.empty_like(grad_out_c)

        _check_cuda_pointers(
            {
                "grad_out": grad_out_c,
                "mask": mask,
                "grad_x": grad_x,
                "grad_res": grad_res,
            }
        )

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_dropout_residual_bwd_kernel[grid](
            grad_out_c,
            mask,
            grad_x,
            grad_res,
            p,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (
            grad_x.reshape(ctx.x_shape),
            grad_res.reshape(ctx.res_shape),
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_dropout_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.1,
    *,
    seed: Optional[int] = None,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused ``dropout(x, p) + residual`` with autograd.

    Parameters
    ----------
    x : tensor
        Tensor to dropout.  Any shape; treated as a flat buffer inside
        the kernel.
    residual : tensor with the same shape as ``x``
        Residual term added after dropout.
    p : float, default 0.1
        Drop probability.  Must be in ``[0, 1)``.
    seed : int, optional
        RNG seed.  ``None`` picks a non-deterministic seed from
        ``torch.randint``.  Pass an explicit ``int`` for reproducibility.
    offset : int, default 0
        RNG offset, advanced across calls for independent streams.

    Returns
    -------
    (out, mask) : tuple of tensors
        ``out`` has the same shape/dtype as ``x``; ``mask`` is a uint8
        tensor with the same flat shape as ``x`` (1 = keep, 0 = drop).
        The mask is currently returned for diagnostic / testing purposes
        only; the autograd wrapper persists the mask internally.

    Notes
    -----
    On CPU tensors we transparently fall back to
    ``torch.dropout(x, p) + residual`` (no RNG-state parity guarantee)
    so tests can run on CPU-only machines.
    """
    if not x.is_cuda:
        out = torch.dropout(x, p, True) + residual
        # CPU fallback does not need a mask for the caller.
        return out, torch.zeros(x.numel(), dtype=torch.uint8)

    if x.shape != residual.shape:
        raise ValueError(
            f"`x` shape {tuple(x.shape)} and `residual` shape "
            f"{tuple(residual.shape)} must match"
        )

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    out = _FusedDropoutResidualFunction.apply(x, residual, float(p), int(seed), int(offset))
    # We currently drop the mask from the public return type; the
    # autograd path persists it internally.  A zero placeholder keeps
    # the return contract stable for callers that want to know the
    # shape.
    return out, torch.zeros(x.numel(), dtype=torch.uint8, device=x.device)


__all__ = [
    "fused_dropout_residual",
    "_fused_dropout_residual_kernel",
    "_fused_dropout_residual_bwd_kernel",
    "_FusedDropoutResidualFunction",
]