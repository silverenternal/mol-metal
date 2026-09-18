"""Fused Triton SwiGLU MLP kernel.

Computes ``(SiLU(x @ W_gate) * (x @ W_up)) @ W_down`` for 2-D inputs.
Weights use the natural matmul layout ``(D_in, D_hidden)`` and
``(D_hidden, D_out)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


def _configs(configs: Optional[Sequence[triton.Config]] = None):
    if configs is not None:
        return list(configs)
    result = []
    for bm, bn, bk in ((32, 32, 32), (64, 32, 32), (128, 32, 32), (64, 64, 32)):
        for nw in (2, 4, 8):
            for ns in (2, 3, 4):
                result.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns))
    return result


@triton.autotune(configs=_configs(), key=["M", "D", "H", "O"])
@triton.jit
def _swiglu_kernel(x, wg, wu, wd, y, M, D, H, O,
                   sxm, sxd, swgm, swgh, swum, swuh, swdh, swdo,
                   sym, syo, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oo = pid_o * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = om < M
    mo = oo < O
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(0, H):
        dot_g = tl.zeros((BLOCK_M,), dtype=tl.float32)
        dot_u = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(D, BLOCK_K)):
            k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            mk = k < D
            xv = tl.load(x + om[:, None] * sxm + k[None, :] * sxd,
                         mask=mm[:, None] & mk[None, :], other=0.0).to(tl.float32)
            gv = tl.load(wg + k * swgm + h * swgh, mask=mk, other=0.0).to(tl.float32)
            uv = tl.load(wu + k * swum + h * swuh, mask=mk, other=0.0).to(tl.float32)
            dot_g += tl.sum(xv * gv[None, :], axis=1)
            dot_u += tl.sum(xv * uv[None, :], axis=1)
        prod = _silu(dot_g) * dot_u
        wdv = tl.load(wd + h * swdh + oo * swdo, mask=mo, other=0.0).to(tl.float32)
        acc += prod[:, None] * wdv[None, :]
    yp = y + om[:, None] * sym + oo[None, :] * syo
    tl.store(yp, acc.to(y.dtype.element_ty), mask=mm[:, None] & mo[None, :])


def fused_swiglu_mlp(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
                     W_down: torch.Tensor, autotune_configs=None) -> torch.Tensor:
    """Compute ``(SiLU(x @ W_gate) * (x @ W_up)) @ W_down``."""
    if x.ndim != 2:
        x2 = x.reshape(-1, x.shape[-1])
    else:
        x2 = x
    if not (x2.is_cuda and W_gate.is_cuda and W_up.is_cuda and W_down.is_cuda):
        raise RuntimeError("fused_swiglu_mlp requires CUDA/HIP tensors")
    M, D = x2.shape
    if W_gate.shape[0] != D or W_up.shape != W_gate.shape or W_down.shape[0] != W_gate.shape[1]:
        raise ValueError("incompatible SwiGLU weight shapes")
    H, O = W_gate.shape[1], W_down.shape[1]
    y = torch.empty((M, O), device=x.device, dtype=x.dtype)
    kernel = _swiglu_kernel if autotune_configs is None else triton.autotune(configs=list(autotune_configs), key=["M", "D", "H", "O"])(_swiglu_kernel)
    kernel[lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(O, META['BLOCK_N']))](x2, W_gate, W_up, W_down, y, M, D, H, O,
        x2.stride(0), x2.stride(1), W_gate.stride(0), W_gate.stride(1), W_up.stride(0), W_up.stride(1), W_down.stride(0), W_down.stride(1), y.stride(0), y.stride(1))
    return y.reshape(*x.shape[:-1], O)


__all__ = ["fused_swiglu_mlp"]
