"""Fused exact-erf GELU MLP for ROCm Triton.

Example::

    mlp = FusedGeluMLP(128, 256).cuda()
    y = mlp(torch.randn(4, 128, device="cuda", dtype=torch.float16))
"""
from __future__ import annotations

import math
import torch
from torch import nn
import triton
import triton.language as tl


def _configs():
    out = []
    for bm, bn, bk in ((32, 32, 32), (64, 32, 32), (32, 64, 32), (64, 64, 32)):
        for nw in (2, 4, 8):
            for ns in (2, 3, 4):
                out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns))
    return out


@triton.jit
def _gelu_erf(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.autotune(configs=_configs(), key=["M", "D", "H", "O"])
@triton.jit
def _kernel(x, w1, b1, w2, b2, y, M, D, H, O,
            sxm, sxd, sw1h, sw1d, sb1, sw2o, sw2h, sb2, sym, syo,
            HAS_B1: tl.constexpr, HAS_B2: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_o = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_o * BLOCK_N + tl.arange(0, BLOCK_N)
    mr = rows < M; mc = cols < O
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(0, H):
        dot = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(D, BLOCK_K)):
            k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            mk = k < D
            xv = tl.load(x + rows[:, None] * sxm + k[None, :] * sxd, mask=mr[:, None] & mk[None, :], other=0.0).to(tl.float32)
            wv = tl.load(w1 + h * sw1h + k * sw1d, mask=mk, other=0.0).to(tl.float32)
            dot += tl.sum(xv * wv[None, :], axis=1)
        if HAS_B1:
            dot += tl.load(b1 + h * sb1).to(tl.float32)
        act = _gelu_erf(dot)
        wv2 = tl.load(w2 + cols * sw2o + h * sw2h, mask=mc, other=0.0).to(tl.float32)
        acc += act[:, None] * wv2[None, :]
    if HAS_B2:
        acc += tl.load(b2 + cols * sb2, mask=mc, other=0.0).to(tl.float32)[None, :]
    tl.store(y + rows[:, None] * sym + cols[None, :] * syo, acc.to(y.dtype.element_ty), mask=mr[:, None] & mc[None, :])


class FusedGeluMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dtype=torch.float16, device="cuda"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.d_model, self.d_hidden = d_model, d_hidden
        self.w1 = nn.Parameter(torch.empty(d_hidden, d_model, dtype=dtype, device=device))
        self.b1 = nn.Parameter(torch.zeros(d_hidden, dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty(d_model, d_hidden, dtype=dtype, device=device))
        self.b2 = nn.Parameter(torch.zeros(d_model, dtype=dtype, device=device))
        nn.init.xavier_uniform_(self.w1); nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[:-1]
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {x.shape[-1]}")
        x2 = x.reshape(-1, self.d_model)
        if not torch.cuda.is_available() or not x2.is_cuda:
            return torch.nn.functional.gelu(torch.nn.functional.linear(x2, self.w1, self.b1), approximate="none").matmul(self.w2.t()).add(self.b2).reshape(*orig, self.d_model)
        y = torch.empty((x2.shape[0], self.d_model), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(x2.shape[0], meta["BLOCK_M"]), triton.cdiv(self.d_model, meta["BLOCK_N"]))
        _kernel[grid](x2, self.w1, self.b1, self.w2, self.b2, y, x2.shape[0], self.d_model, self.d_hidden, self.d_model,
                      x2.stride(0), x2.stride(1), self.w1.stride(0), self.w1.stride(1), self.b1.stride(0), self.w2.stride(0), self.w2.stride(1), self.b2.stride(0), y.stride(0), y.stride(1), True, True)
        return y.reshape(*orig, self.d_model)


__all__ = ["FusedGeluMLP"]
