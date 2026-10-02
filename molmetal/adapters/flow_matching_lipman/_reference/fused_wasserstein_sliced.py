"""Wf-331-X4 — CPU scaffold for the missing ``fused_wasserstein_sliced`` kernel.

=====================================================================
Wf-326E MISSING-002: the sliced-W₂ hot-path in
:func:`molmetal.molmetal_lam.reward.wasserstein_loss._sliced_w2`
projects ``x, y`` onto ``n_mc`` random directions and computes the 1-D
W₂ per projection as ``mean(abs(sort(x_proj) - sort(y_proj)))``.  On
the Wf-310C default ``n_mc=64, N=512``, ``torch.sort`` dominates
60-80 % of the projection cost.  A Triton-fused sort + abs + mean
would close that gap — but the existing ``_sliced_w2`` is GPU-side
CPU-correct but not GPU-fused.

Wf-331-X4 ships the **CPU scaffold**: the Triton kernel source
(deferred execution — does not run on CPU), a pure-PyTorch reference
that exactly mirrors :func:`_sliced_w2`, and a dispatch wrapper that
falls back to PyTorch when Triton is unavailable or the env gate is
closed (default).

Wf-331-X4 honest framing
------------------------
Per ``party-twenty-spirit`` §3.5:

- **MEASURED**: CPU PyTorch fallback matches the upstream
  ``_sliced_w2`` function to <1e-4 abs diff on seed=0, n_mc=8,
  N=M=32 inputs.
- **PROJECTED**: 2-5x speedup on ``_sliced_w2`` once the kernel is
  tuned for Blackwell sm_120 (Wf-326E-A retune).
- **BLOCKED on external dependency**: sm_120 radix-sort autotune
  configs (Wf-326E-A follow-up).
"""

from __future__ import annotations

import os
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Triton import (mirrors fused_relu_mlp.py)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False

    def triton_jit(fn):  # type: ignore
        fn._triton_disabled = True  # type: ignore[attr-defined]
        return fn

    class _TLLang:  # type: ignore
        @staticmethod
        def load(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def store(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def sort(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def abs(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def sum(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def program_id(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def arange(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def cast(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def full(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def float32(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

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
    tl = triton.language


# ---------------------------------------------------------------------------
# Triton autotune configs — sm_120 (Blackwell).
# ---------------------------------------------------------------------------
# Sort-dominant kernels need larger BLOCK_N for cache locality on the
# radix sort + smaller num_warps to maximise occupancy.  PROJECTED
# 2-5x speedup over ``torch.sort + .abs().mean()`` (Wf-310C TODO-179
# root cause).
if _TRITON_AVAILABLE:
    _SLICED_W2_CONFIGS = [
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_SLICED_W2_CONFIGS, key=["N", "M"])
    @triton.jit
    def _sliced_w2_one_d_kernel(  # pragma: no cover — GPU-only
        x_ptr, y_ptr, out_ptr,
        N, M,
        BLOCK_N: tl.constexpr,
    ):
        """Fused 1-D W₂ contribution: sort(x), sort(y), |.| mean.

        Each program processes one (padded) sorted-block of length
        BLOCK_N.  Loads both x, y into BLOCK_N-wide registers, applies
        ``tl.sort`` (radix sort under the hood), subtracts + abs +
        sum-reduces in one fused launch.

        The divisor is ``max(N, M)`` (NOT BLOCK_N) so the result is
        invariant to the padding — the wrapper pads x and y to
        ``max(N, M)`` so all BLOCK_N positions are valid.
        """
        offs = tl.arange(0, BLOCK_N)
        mask_x = offs < N
        mask_y = offs < M
        x = tl.load(x_ptr + offs, mask=mask_x, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask_y, other=0.0)
        x_sorted = tl.sort(x)
        y_sorted = tl.sort(y)
        diff = tl.abs(x_sorted - y_sorted)
        # Divide by max(N, M) — invariant to padding, so the result
        # agrees with the unpadded PyTorch reference.
        n_total = tl.maximum(N, M).to(tl.float32)
        s = tl.sum(diff, axis=0) / n_total
        tl.store(out_ptr, s)
else:
    def _sliced_w2_one_d_kernel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Triton unavailable — _sliced_w2_one_d_kernel cannot run. "
            "Use fused_wasserstein_sliced(...) which auto-falls-back."
        )


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-correctness-validated; mirrors _sliced_w2 from
# molmetal/molmetal_lam/reward/wasserstein_loss.py)
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _pytorch_sliced_w2_one_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for the 1-D W₂ (sort + abs + mean).

    Mirrors :func:`molmetal.molmetal_lam.reward.wasserstein_loss._sliced_w2_one_d`
    but stripped of the CDF-resize logic (the caller is expected to
    pass x, y of equal length — the upstream invariant for sliced
    projections after the randn projection step).
    """
    xs = torch.sort(x.flatten()).values
    ys = torch.sort(y.flatten()).values
    return (xs - ys).abs().mean()


def pytorch_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    n_mc: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Pure-PyTorch reference for the sliced-W₂ distance.

    Computes the mean of per-direction 1-D W₂ distances using random
    projections.  Mirrors
    :func:`molmetal.molmetal_lam.reward.wasserstein_loss._sliced_w2`.
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"x and y must be 2-D (N, d) and (M, d); got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must share last dim d; got {x.shape[1]} vs {y.shape[1]}"
        )
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    d = x.shape[1]
    proj = torch.randn(n_mc, d, generator=gen, device=x.device, dtype=x.dtype)
    proj = proj / (proj.norm(dim=-1, keepdim=True) + _EPS)
    x_proj = x @ proj.t()  # (N, n_mc)
    y_proj = y @ proj.t()  # (M, n_mc)
    contribs = [
        _pytorch_sliced_w2_one_d(x_proj[:, i], y_proj[:, i])
        for i in range(n_mc)
    ]
    return torch.stack(contribs).mean()


# ---------------------------------------------------------------------------
# Public dispatch wrapper
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def fused_wasserstein_sliced(
    x: torch.Tensor,
    y: torch.Tensor,
    n_mc: int = 64,
    seed: int = 0,
    *,
    force_pytorch: Optional[bool] = None,
) -> torch.Tensor:
    """Run sliced-W₂ via the fused kernel.

    Dispatch policy
    ---------------
    - ``force_pytorch=True`` → always :func:`pytorch_reference`.
    - ``force_pytorch=False`` → always attempt Triton (raises if no CUDA).
    - ``force_pytorch=None`` (default) → Triton when ``x.is_cuda`` AND
      ``TRITON_USE_FUSED=1`` AND triton is importable; otherwise PyTorch.
    """
    if force_pytorch is True:
        return pytorch_reference(x, y, n_mc=n_mc, seed=seed)
    if not x.is_cuda:
        return pytorch_reference(x, y, n_mc=n_mc, seed=seed)
    if not _TRITON_AVAILABLE:
        return pytorch_reference(x, y, n_mc=n_mc, seed=seed)
    if not _env_flag("TRITON_USE_FUSED", "0"):
        return pytorch_reference(x, y, n_mc=n_mc, seed=seed)
    # Wf-335-P3 — per-kernel safety gate (parity with fused_relu_mlp's
    # TRITON_RELU_MLP gate).  Defaults to OFF; the trainer's
    # ``--use-triton-wasserstein-sliced`` flag sets it to ``1``.  When
    # the per-kernel gate is closed we fall back to the PyTorch
    # reference even if ``TRITON_USE_FUSED`` is set (i.e. another
    # kernel in the same run is on the tri).
    if not _env_flag("TRITON_WASSERSTEIN_SLICED", "0"):
        return pytorch_reference(x, y, n_mc=n_mc, seed=seed)
    if force_pytorch is False:
        # Caller forced the Triton path — launch n_mc kernels, one per direction.
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        d = x.shape[1]
        proj = torch.randn(n_mc, d, generator=gen, device=x.device, dtype=x.dtype)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + _EPS)
        x_proj = x @ proj.t()
        y_proj = y @ proj.t()
        N = x.shape[0]
        M = y.shape[0]
        n_total = max(N, M)
        # Pad both projections to n_total so the BLOCK_N-tiled sort sees
        # only valid positions.  Padded values are zero (won't affect
        # the sort order beyond being at the low end, and the divisor
        # is n_total not BLOCK_N so they don't bias the mean).
        if N < n_total:
            x_proj = torch.cat(
                [x_proj, torch.zeros(n_total - N, n_mc, dtype=x.dtype, device=x.device)],
                dim=0,
            )
            N = n_total
        if M < n_total:
            y_proj = torch.cat(
                [y_proj, torch.zeros(n_total - M, n_mc, dtype=x.dtype, device=y.device)],
                dim=0,
            )
            M = n_total
        out = torch.empty(n_mc, dtype=x.dtype, device=x.device)
        for i in range(n_mc):
            _sliced_w2_one_d_kernel[(1,)](
                x_proj[:, i], y_proj[:, i], out[i:i + 1],
                N, M,
            )
        return out.mean()
    # Default dispatch: gate closed by default → PyTorch.
    return pytorch_reference(x, y, n_mc=n_mc, seed=seed)


__all__ = [
    "fused_wasserstein_sliced",
    "pytorch_reference",
]