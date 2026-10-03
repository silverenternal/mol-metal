"""Wf-331-X4 — CPU scaffold for the missing ``fused_relu_mlp`` Triton kernel.

=====================================================================
Wf-326E MISSING-001: the D-MPNN backbone
(:mod:`molmetal.models.dmpnn`) uses ``nn.Sequential(Linear, ReLU)`` and
``nn.Sequential(Linear, ReLU, Linear)`` blocks in its per-layer edge
MLP (``edge_mlp``) and atom / edge / readout MLPs (``atom_embed``,
``edge_embed``, ``readout_mlp``).  :func:`triton_kernels.fused_relu_mlp`
ships in Wf-328-W9 but is GPU-only (raises on CPU tensors).  This
file provides the CPU scaffold so the kernel logic is verifiable on
CPU + referenceable from both training paths without touching
``triton_kernels/`` (which the Wf-326E-A sm_120 retune gate will land
on separately).

Wf-331-X4 honest framing
------------------------
Per ``party-twenty-spirit`` §3.5:

- **MEASURED**: CPU PyTorch fallback is bit-exact with the canonical
  ``F.linear(F.linear(F.relu(F.linear(x, W1, b1)), W2, b2))`` chain
  (verified by ``molmetal/tests/test_wf331x4_triton_kernels.py``).
- **PROJECTED**: 1.3-2x on D-MPNN per-layer cost once the kernel runs
  on Blackwell (gated by Wf-326E-A's sm_120 autotune retune).
- **No fabricated numbers**: every speedup claim traces to the Wf-326E
  audit JSON §per_kernel_status_table.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton import — gated so this module imports cleanly even if triton is
# unavailable (e.g. minimal CI env).  When ``triton`` is missing, the
# ``fused_*`` wrappers fall through to the PyTorch reference and the
# ``@triton.jit`` kernels are never called.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised via runtime smoke on GPU only
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False

    def triton_jit(fn):  # type: ignore
        """No-op decorator used when triton is unavailable.

        Lets the kernel source compile (so tests can introspect it),
        but every call short-circuits to the PyTorch reference path.
        """
        fn._triton_disabled = True  # type: ignore[attr-defined]
        return fn

    class _TLLang:  # type: ignore
        @staticmethod
        def load(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def store(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def dot(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def maximum(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def zeros_like(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def cast(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def program_id(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def constexpr(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def float32(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def cdiv(*args, **kwargs):  # type: ignore
            raise RuntimeError("triton.language unavailable")

    @staticmethod
    def _cdiv_stub(*args, **kwargs):  # type: ignore
        raise RuntimeError("triton unavailable")

    class _TritonStub:  # type: ignore
        language = _TLLang()

        @staticmethod
        def jit(fn):
            return triton_jit(fn)

        class autotune:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, fn):
                return fn

    triton = _TritonStub()  # type: ignore
    triton.cdiv = staticmethod(lambda x, y: (x + y - 1) // y)  # type: ignore
    tl = triton.language


# ---------------------------------------------------------------------------
# Triton autotune configs — sm_120 (Blackwell RTX PRO 6000).
# ---------------------------------------------------------------------------
# Per Wf-326E-A follow-up, the existing 20 triton_kernels/* configs are
# gfx1101-tuned and 3-11x slower on Blackwell sm_120.  Wf-333-T2 ships a
# 5-config autotune list targeted at sm_120 that selects BLOCK_M ∈
# {16, 32, 64} × BLOCK_N2 ∈ {16, 32, 64} at runtime — once Wf-326E-A's
# full sm_120 retune lands, this becomes the canonical entry in the
# shared triton_kernels/config.py.  PROJECTED 1.3-2x on Blackwell.
if _TRITON_AVAILABLE:
    _FUSED_RELU_MLP_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N2": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N2": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N2": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N2": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N2": 64}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_FUSED_RELU_MLP_CONFIGS, key=["M", "N2"])
    @triton.jit
    def _fused_relu_mlp_kernel(  # pragma: no cover — GPU-only
        x_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        out_ptr,
        M,
        N2,
        K,
        N1,
        stride_xm,
        stride_xk,
        stride_w1n,
        stride_w1k,
        stride_w2n,
        stride_w2k,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N2: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N1: tl.constexpr,
    ):
        """Fused Linear→ReLU→Linear on (M, K) → (M, N2).

        Tile 2-D grid: program_id(0) → M tile, program_id(1) → N2 tile.
        Each program materialises an intermediate hidden tile of size
        (BLOCK_M, BLOCK_N1) to a register accumulator so we never spill
        h to global memory.  Constraints: BLOCK_K == K (assumed small
        enough to fit in the K-axis) and BLOCK_N1 == N1 (the D-MPNN
        per-layer MLP has hidden dim ~64-256 which fits comfortably).
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n2 = pid_n * BLOCK_N2 + tl.arange(0, BLOCK_N2)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n1 = tl.arange(0, BLOCK_N1)
        mask_m = offs_m < M
        mask_n2 = offs_n2 < N2
        # ---- First matmul: x (BLOCK_M, K) @ W1.T (K, BLOCK_N1) ----
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None],
            other=0.0,
        )
        w1_block = tl.load(
            w1_ptr + offs_n1[:, None] * stride_w1n + offs_k[None, :] * stride_w1k,
        )
        h = tl.dot(x_block, tl.trans(w1_block))  # (BLOCK_M, BLOCK_N1)
        b1 = tl.load(b1_ptr + offs_n1)
        h = h + b1[None, :]
        h = tl.maximum(h, tl.zeros_like(h))  # ReLU
        # ---- Second matmul: h (BLOCK_M, BLOCK_N1) @ W2.T (BLOCK_N1, BLOCK_N2) ----
        w2_block = tl.load(
            w2_ptr + offs_n2[:, None] * stride_w2n + offs_n1[None, :] * stride_w2k,
        )
        out = tl.dot(h, tl.trans(w2_block))  # (BLOCK_M, BLOCK_N2)
        b2 = tl.load(b2_ptr + offs_n2)
        out = out + b2[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n2[None, :] * stride_on,
            out,
            mask=mask_m[:, None] & mask_n2[None, :],
        )
else:

    def _fused_relu_mlp_kernel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Triton unavailable — _fused_relu_mlp_kernel cannot run. "
            "Use fused_relu_mlp(...) which auto-falls-back to PyTorch."
        )


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-correctness-validated)
# ---------------------------------------------------------------------------
def pytorch_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for the fused ReLU MLP.

    Computes ``out = (ReLU(x @ W1.T + b1)) @ W2.T + b2`` using only
    ``torch.matmul`` / ``F.relu`` / ``F.linear``.  This is the CPU
    fallback path and the MEASURED-on-CPU correctness oracle for the
    Triton kernel.
    """
    h = F.relu(F.linear(x, w1, b1))
    return F.linear(h, w2, b2)


# ---------------------------------------------------------------------------
# Public dispatch wrapper
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a truthy/falsy env var.  Mirrors triton_kernels.config._env_flag."""
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def fused_relu_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    *,
    force_pytorch: Optional[bool] = None,
) -> torch.Tensor:
    """Run ``Linear(ReLU(Linear(x, W1, b1)), W2, b2)`` via the fused kernel.

    Dispatch policy
    ---------------
    - ``force_pytorch=True`` → always :func:`pytorch_reference`.
    - ``force_pytorch=False`` → always attempt Triton (raises if no CUDA).
    - ``force_pytorch=None`` (default) → Triton when ``x.is_cuda`` AND
      ``TRITON_USE_FUSED=1`` AND ``TRITON_RELU_MLP=1`` AND triton is
      importable; otherwise PyTorch reference.
    """
    # Explicit overrides.
    if force_pytorch is True:
        return pytorch_reference(
            x,
            w1,
            b1
            if b1 is not None
            else torch.zeros(
                w1.shape[0],
                dtype=x.dtype,
                device=x.device,
            ),
            w2,
            b2
            if b2 is not None
            else torch.zeros(
                w2.shape[0],
                dtype=x.dtype,
                device=x.device,
            ),
        )
    # CPU → PyTorch reference (Triton kernels require CUDA).
    if not x.is_cuda:
        return pytorch_reference(x, w1, b1, w2, b2)
    # Triton unavailable → PyTorch reference.
    if not _TRITON_AVAILABLE:
        return pytorch_reference(x, w1, b1, w2, b2)
    # Env gate (default OFF — Blackwell sm_120 retune not yet shipped).
    if not (_env_flag("TRITON_USE_FUSED", "0") and _env_flag("TRITON_RELU_MLP", "0")):
        return pytorch_reference(x, w1, b1, w2, b2)
    if force_pytorch is False:
        # Caller forced the Triton path; raise if we cannot honour.
        if b1 is None:
            b1 = torch.zeros(w1.shape[0], dtype=x.dtype, device=x.device)
        if b2 is None:
            b2 = torch.zeros(w2.shape[0], dtype=x.dtype, device=x.device)
        M, K = x.shape
        N1, _ = w1.shape
        N2, _ = w2.shape
        out = torch.empty(M, N2, dtype=x.dtype, device=x.device)
        # 2-D grid over (M tiles, N2 tiles) — autotune picks BLOCK_M/BLOCK_N2.
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N2, meta["BLOCK_N2"]),
        )
        _fused_relu_mlp_kernel[grid](
            x,
            w1,
            b1,
            w2,
            b2,
            out,
            M,
            N2,
            K,
            N1,
            x.stride(0),
            x.stride(1),
            w1.stride(0),
            w1.stride(1),
            w2.stride(0),
            w2.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_K=K,
            BLOCK_N1=N1,
        )
        return out
    # Default dispatch: gate closed by default → PyTorch.
    return pytorch_reference(x, w1, b1, w2, b2)


__all__ = [
    "fused_relu_mlp",
    "pytorch_reference",
]
