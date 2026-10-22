"""Reference Triton matmul kernel, tuned per backend.

This module is a port of the canonical Triton tutorial:

    ``triton_tutorials_repo/python/tutorials/03-matrix-multiplication.py``

adapted for the MolFlow-Triton project constraints:

- **No ``waves_per_eu``.** It is a CDNA-only knob and is ignored on
  gfx1101 (RDNA3) and on CUDA.  We tune over ``num_warps`` and
  ``num_stages`` only, plus the per-tile ``BLOCK_*`` sizes.
- **Backend-selected config grid.** The original grid was sized for
  gfx1101's single wave64 per CU, so it capped ``num_warps`` at 8 and
  never used a 128-wide tile.  Blackwell (``sm_120``) has 4 warp slots
  per SM and ~228 KB of shared memory, so the wide tiles and the deeper
  K pipeline are usable; :func:`_matmul_configs` now branches on
  :func:`triton_kernels.backend.active_backend`.
- **FP32 throughout.** The MolFlow-Triton autograd shims operate on
  FP32 hidden states, so we avoid the FP16 cast that the upstream
  tutorial applies at the end of the accumulator.
- **Grouped L2-friendly launch order.** ``GROUP_SIZE_M = 8`` matches
  the upstream default and is well-suited to both targets' L2.
- **Stride-aware ``matmul`` wrapper.** Accepts non-contiguous inputs
  by forwarding per-dimension strides, so callers can avoid an extra
  ``.contiguous()`` copy when downstream layout is already what we
  need.

Public API
----------
- :func:`matmul` - ``(M, K) @ (K, N) -> (M, N)`` Triton matmul.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS
from .backend import active_backend, dot_precision


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
            "the Triton matmul kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Autotune grid for matmul, per backend
# ---------------------------------------------------------------------------
# Cartesian product (modulo the GROUP_SIZE_M=8 constraint) of:
#   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K in {16, 32, 64, 128}
#   num_warps     in {2, 4, 8}
#   num_stages    in {2, 3, 4}
#
# That is 4*4*4 * 3 * 3 = 576 configs in the full grid — too many for
# first-launch latency on a single-GPU dev box.  We therefore trim to
# the 12 configs that have proven robust on gfx1101 in the upstream
# tutorial plus our own sweeps: keep the BLOCK_K=32/64 sweet spots,
# drop the extremes that either starve occupancy (BLOCK_K=16) or
# blow the LDS budget on RDNA3 (BLOCK=128x128 with num_stages=4).
#
# Each entry also fixes ``GROUP_SIZE_M = 8`` (upstream default; the
# L2 super-grouping that promotes A/B reuse across program instances).
def _matmul_configs() -> Sequence[triton.Config]:
    """Autotune grid for :func:`_matmul_kernel`, split by backend.

    ``rocm`` / gfx1101: the 12 tiles below x 3 warps x 3 stages = 108
    configs.  That is already more than the header comment used to claim
    ("the 12 configs that have proven robust on gfx1101"); the grid was
    never trimmed.  It is left alone rather than silently narrowed,
    because narrowing it would change which config an existing gfx1101
    run picks.

    ``cuda`` / Blackwell: 8 tiles x 2 warps x 2 stages = 32 configs.
    Blackwell reaches the 128-wide tiles and the deeper K pipeline that
    RDNA3's LDS budget ruled out, so the grid keeps those and drops the
    16-wide corners that only ever win on a 64-lane wavefront.  Cutting
    108 -> 32 also cuts first-launch autotune latency by ~3x, which
    matters because this kernel is used per-ODE-step in sampling.
    """
    cfgs: list[triton.Config] = []
    if active_backend() == "cuda":
        tiles = [
            (64, 64, 32),
            (64, 128, 32),
            (128, 64, 32),
            (128, 128, 32),
            (64, 64, 64),
            (128, 128, 64),
            (64, 128, 128),
            (128, 128, 128),
        ]
        warps = (4, 8)
        stages = (3, 4)
    else:
        tiles = [
            (32, 32, 32),
            (32, 64, 32),
            (64, 32, 32),
            (64, 64, 32),
            (32, 32, 64),
            (32, 64, 64),
            (64, 32, 64),
            (64, 64, 64),
            (32, 32, 128),
            (64, 32, 128),
            (32, 64, 128),
            (64, 64, 128),
        ]
        warps = (2, 4, 8)
        stages = (2, 3, 4)
    for bm, bn, bk in tiles:
        for num_warps in warps:
            for num_stages in stages:
                cfgs.append(
                    triton.Config(
                        {
                            "BLOCK_SIZE_M": bm,
                            "BLOCK_SIZE_N": bn,
                            "BLOCK_SIZE_K": bk,
                            "GROUP_SIZE_M": 8,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return cfgs


_MATMUL_CONFIGS: Sequence[triton.Config] = _matmul_configs()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Per-dim strides (in elements, not bytes)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters chosen by the autotuner
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """Compute ``C = A @ B`` where ``A`` is ``(M, K)`` and ``B`` is ``(K, N)``.

    Block layout is the canonical L2-friendly grouped order from the
    upstream tutorial — see the L2 Cache Optimizations section there.
    The accumulator is FP32; ``PRECISION`` selects how the multiply
    feeds it (``tf32`` on CUDA by default, ``ieee`` elsewhere).  Leaving
    this to Triton's implicit default silently gave tf32 even when
    ``TRITON_DOT_PRECISION=ieee`` was set, so it is now explicit.
    """
    # ---------------------------------------------------------------
    # Map program ids to the (M, N) block of C this instance owns.
    # ---------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Integer bounds — helps the Triton compiler fold pointer arithmetic.
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ---------------------------------------------------------------
    # Pointers for the first K-block of A and B.
    # ---------------------------------------------------------------
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---------------------------------------------------------------
    # Inner K loop: accumulate into an FP32 tile.
    # ---------------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision=PRECISION)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ---------------------------------------------------------------
    # Write the FP32 result back.  No FP16 cast — MolFlow-Triton uses
    # FP32 throughout its autograd shims.
    # ---------------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    """Reshape ``t`` to exactly 2-D, raising a clear error otherwise."""
    if t.dim() == 2:
        return t
    if t.dim() < 2:
        raise ValueError(
            f"`{name}` must be at least 2-D for matmul; got {t.dim()}-D tensor "
            f"of shape {tuple(t.shape)}"
        )
    return t.reshape(t.shape[0], -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``A @ B`` with a Triton kernel.

    Parameters
    ----------
    a : ``(M, K)`` or batched ``(..., M, K)`` tensor
        Left operand.  Any dtype supported by Triton (``float16``,
        ``bfloat16``, ``float32``) is accepted; the kernel internally
        casts to FP32 for the dot-product accumulator.
    b : ``(K, N)`` or batched ``(..., K, N)`` tensor
        Right operand.  Same dtype/device requirements as ``a``.
    out : optional ``(M, N)`` tensor
        Output buffer.  If ``None``, a new tensor is allocated with the
        same dtype/device as ``a``.

    Returns
    -------
    Tensor of shape ``(M, N)`` with the same dtype as ``a``.
    """
    a2 = _flatten_2d(a, "a")
    b2 = _flatten_2d(b, "b")

    if a2.dim() != 2 or b2.dim() != 2:
        raise ValueError(
            f"`matmul` expects 2-D inputs after flattening; got a={tuple(a2.shape)},"
            f" b={tuple(b2.shape)}"
        )
    if a2.shape[1] != b2.shape[0]:
        raise ValueError(
            f"Inner dimensions mismatch for matmul: a.shape={tuple(a2.shape)}"
            f" vs b.shape={tuple(b2.shape)}"
        )
    if a2.dtype != b2.dtype:
        raise ValueError(
            f"Dtype mismatch for matmul: a.dtype={a2.dtype} vs b.dtype={b2.dtype}"
        )

    M, K = a2.shape
    Kb, N = b2.shape
    assert K == Kb, "K mismatch caught by the check above"

    if out is None:
        out = torch.empty((M, N), dtype=a2.dtype, device=a2.device)
    else:
        if out.shape != (M, N):
            raise ValueError(
                f"`out` shape {tuple(out.shape)} does not match expected "
                f"(M, N) = ({M}, {N})"
            )

    # Defensive check before the autotune probe.
    _check_cuda_pointers({"a": a2, "b": b2, "out": out})

    # 1-D launch grid; the autotuner picks BLOCK_M / BLOCK_N.
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    _matmul_kernel[grid](
        a2,
        b2,
        out,
        M,
        N,
        K,
        a2.stride(0),
        a2.stride(1),
        b2.stride(0),
        b2.stride(1),
        out.stride(0),
        out.stride(1),
        PRECISION=dot_precision(),
    )
    return out


__all__ = ["matmul", "_matmul_kernel", "_MATMUL_CONFIGS"]
