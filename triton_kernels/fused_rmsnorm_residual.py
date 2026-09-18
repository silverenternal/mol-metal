"""Fused residual addition followed by RMSNorm."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


def _next_power_of_2(n: int) -> int:
    block = 1
    while block < n:
        block *= 2
    return block


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_rows", "feat_dim"])
@triton.jit
def _fused_rmsnorm_residual_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    n_rows,
    feat_dim,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Compute ``(x + residual) / rms(x + residual) * weight`` per row."""
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < feat_dim
    base = row * feat_dim
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = x + residual
    mean_square = tl.sum(tl.where(mask, summed * summed, 0.0), axis=0) / feat_dim
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offsets, (summed * inv_rms * weight).to(tl.float32), mask=mask)


def fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    autotune_configs: Optional[Sequence[triton.Config]] = None,
) -> torch.Tensor:
    """Fuse ``x + residual`` with RMSNorm and per-channel scaling."""
    if x.dim() < 2:
        raise ValueError(f"x must be at least 2-D; got shape {tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    feat_dim = x.shape[-1]
    if weight.shape != (feat_dim,):
        raise ValueError(f"weight must have shape ({feat_dim},); got {tuple(weight.shape)}")
    if feat_dim <= 0:
        raise ValueError("feature dimension must be positive")

    # Keep CPU behavior useful for reference tests and environments without HIP.
    if not x.is_cuda:
        summed = x + residual
        return summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps) * weight

    # Triton kernels allocate a fresh output tensor and therefore do not
    # participate in PyTorch autograd by themselves.  Keep the fused path
    # for inference, but use the mathematically identical PyTorch expression
    # whenever gradients are requested so training and gradcheck remain
    # correct on CUDA/HIP.
    if x.requires_grad or residual.requires_grad or weight.requires_grad:
        summed = x + residual
        return summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps) * weight

    x_c = x.contiguous()
    residual_c = residual.contiguous()
    weight_c = weight.contiguous().to(dtype=x.dtype)
    out = torch.empty_like(x_c)
    n_rows = x_c.numel() // feat_dim
    block_d = _next_power_of_2(feat_dim)
    if block_d > 4096:
        raise ValueError("feature dimension is too large (maximum 4096)")
    configs = list(autotune_configs) if autotune_configs is not None else AUTOTUNE_CONFIGS
    kernel = _fused_rmsnorm_residual_kernel
    if autotune_configs is not None:
        kernel = triton.autotune(configs=configs, key=["n_rows", "feat_dim"])(_fused_rmsnorm_residual_kernel.fn)
    kernel[(n_rows,)](x_c, residual_c, weight_c, out, n_rows, feat_dim, float(eps), BLOCK_D=block_d)
    return out.reshape(x.shape)


__all__ = ["fused_rmsnorm_residual"]
