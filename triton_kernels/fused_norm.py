"""Fused LayerNorm / RMSNorm kernels for MolFlow-Triton.

The kernels in this module compute mean, variance (or mean-of-squares
for RMS), normalization and the affine transform in a single Triton
launch.  They are intended for molecular feature tensors of shape
``(B, N_atoms, hidden_dim)`` but the implementation is generic and
works for any 3-D tensor whose last axis is the feature axis.

Both kernels target AMD ROCm GPUs (``gfx1101``) via
``triton-rocm==3.8.0`` and avoid features that 3.8 does not support.
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
            "the fused-norm kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# LayerNorm kernel
# ---------------------------------------------------------------------------
# Autotune over (num_warps, num_stages) keyed on the problem shape.
# The row count drives the launch grid; the feature dim drives the
# in-block reduction width (FEAT_BLOCK).  The autotuner picks the
# best warp / pipeline combination per (n_rows, feat_dim) bucket.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_rows", "feat_dim"])
@triton.jit
def _fused_layer_norm_kernel(
    x_ptr,         # *fp   [..., D]
    weight_ptr,    # *fp   [D]
    bias_ptr,      # *fp   [D]
    out_ptr,       # *fp   [..., D]
    n_rows,        # i32  product of leading dims
    feat_dim,      # i32  D
    eps,           # fp
    FEAT_BLOCK: tl.constexpr,
):
    """One program instance per row; reduces within a single block."""
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return

    row_off = row * feat_dim
    feat_offsets = tl.arange(0, FEAT_BLOCK)
    feat_mask = feat_offsets < feat_dim

    x = tl.load(x_ptr + row_off + feat_offsets, mask=feat_mask, other=0.0)
    x_f = x.to(tl.float32)

    mean = tl.sum(x_f, axis=0) / feat_dim
    centered = tl.where(feat_mask, x_f - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / feat_dim
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + feat_offsets, mask=feat_mask, other=0.0)
    b = tl.load(bias_ptr + feat_offsets, mask=feat_mask, other=0.0)

    normed = (x_f - mean) * rstd
    out = normed * w.to(tl.float32) + b.to(tl.float32)
    tl.store(
        out_ptr + row_off + feat_offsets,
        out.to(x.dtype),
        mask=feat_mask,
    )


# ---------------------------------------------------------------------------
# RMSNorm kernel
# ---------------------------------------------------------------------------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_rows", "feat_dim"])
@triton.jit
def _fused_rms_norm_kernel(
    x_ptr,         # *fp   [..., D]
    weight_ptr,    # *fp   [D]
    out_ptr,       # *fp   [..., D]
    n_rows,        # i32
    feat_dim,      # i32  D
    eps,           # fp
    FEAT_BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return

    row_off = row * feat_dim
    feat_offsets = tl.arange(0, FEAT_BLOCK)
    feat_mask = feat_offsets < feat_dim

    x = tl.load(x_ptr + row_off + feat_offsets, mask=feat_mask, other=0.0)
    x_f = x.to(tl.float32)

    sq = tl.where(feat_mask, x_f * x_f, 0.0)
    ms = tl.sum(sq, axis=0) / feat_dim
    rrms = 1.0 / tl.sqrt(ms + eps)

    w = tl.load(weight_ptr + feat_offsets, mask=feat_mask, other=0.0)
    out = x_f * rrms * w.to(tl.float32)
    tl.store(
        out_ptr + row_off + feat_offsets,
        out.to(x.dtype),
        mask=feat_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    out = 1
    while out < n:
        out *= 2
    return out


def _flatten_leading(x: torch.Tensor) -> tuple[int, int]:
    """Return (n_rows, feat_dim) for a 2-D or 3-D tensor."""
    if x.dim() < 2:
        raise ValueError(
            f"`x` must be at least 2-D (got {x.dim()}-D tensor of shape "
            f"{tuple(x.shape)})"
        )
    feat_dim = x.shape[-1]
    n_rows = 1
    for d in x.shape[:-1]:
        n_rows *= d
    return n_rows, feat_dim


def _contig_2d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).contiguous()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused LayerNorm over the last axis of ``x``.

    Parameters
    ----------
    x : tensor with last dim ``D``
        Shape can be ``(D,)``, ``(N, D)``, ``(B, N, D)``, ...
    weight : ``(D,)`` tensor
        Affine scale ``gamma``.
    bias : ``(D,)`` tensor
        Affine shift ``beta``.
    eps : float
        Numerical stability term added inside the square root.
    out : optional tensor with the same shape as ``x``
        Output buffer.

    Returns
    -------
    Tensor with the same shape and dtype as ``x``.
    """
    x_c = _contig_2d(x)
    n_rows, feat_dim = _flatten_leading(x)
    feat_block = _next_pow2(feat_dim)
    if feat_block > 4096:
        raise ValueError(
            f"Feature dimension {feat_dim} is too large for the in-block "
            f"reduction (max 4096)."
        )

    if weight.shape != (feat_dim,):
        raise ValueError(
            f"`weight` must have shape ({feat_dim},); got {tuple(weight.shape)}"
        )
    if bias.shape != (feat_dim,):
        raise ValueError(
            f"`bias` must have shape ({feat_dim},); got {tuple(bias.shape)}"
        )

    w = weight.contiguous().to(x.dtype)
    b = bias.contiguous().to(x.dtype)

    if out is None:
        out = torch.empty_like(x_c)
    else:
        out_flat = out.reshape(-1, feat_dim)
        if out_flat.shape != x_c.shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} does not match `x` "
                f"shape {tuple(x.shape)}"
            )

    grid = (n_rows,)
    _check_cuda_pointers({"x": x_c, "weight": w, "bias": b, "out": out})
    _fused_layer_norm_kernel[grid](
        x_c,
        w,
        b,
        out,
        n_rows,
        feat_dim,
        float(eps),
        FEAT_BLOCK=feat_block,
    )
    return out.reshape(x.shape)


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused RMSNorm over the last axis of ``x``.

    Parameters
    ----------
    x : tensor with last dim ``D``
    weight : ``(D,)`` tensor
        Affine scale ``gamma``.  No bias is applied.
    eps : float
    out : optional tensor with the same shape as ``x``

    Returns
    -------
    Tensor with the same shape and dtype as ``x``.
    """
    x_c = _contig_2d(x)
    n_rows, feat_dim = _flatten_leading(x)
    feat_block = _next_pow2(feat_dim)
    if feat_block > 4096:
        raise ValueError(
            f"Feature dimension {feat_dim} is too large for the in-block "
            f"reduction (max 4096)."
        )

    if weight.shape != (feat_dim,):
        raise ValueError(
            f"`weight` must have shape ({feat_dim},); got {tuple(weight.shape)}"
        )

    w = weight.contiguous().to(x.dtype)

    if out is None:
        out = torch.empty_like(x_c)
    else:
        out_flat = out.reshape(-1, feat_dim)
        if out_flat.shape != x_c.shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} does not match `x` "
                f"shape {tuple(x.shape)}"
            )

    grid = (n_rows,)
    _check_cuda_pointers({"x": x_c, "weight": w, "out": out})
    _fused_rms_norm_kernel[grid](
        x_c,
        w,
        out,
        n_rows,
        feat_dim,
        float(eps),
        FEAT_BLOCK=feat_block,
    )
    return out.reshape(x.shape)
