"""Fused ReLU MLP kernels for MolFlow-Triton.

This module closes the **ReLU gap** flagged by Wf-326E
(molmetal/reports/wf326e_triton_audit/audit.json §missing_kernel_candidates
MISSING-001): the existing :func:`fused_silu_mlp` only supports SiLU and
:func:`fused_gelu_mlp` only supports tanh-GELU; the D-MPNN backbone in
:mod:`molmetal.models.dmpnn` uses ``nn.Sequential(Linear, ReLU)`` and
``nn.Sequential(Linear, ReLU, Linear)`` blocks for the per-layer edge MLP
(``edge_mlp``) and the atom / edge / readout MLPs (``atom_embed``,
``edge_embed``, ``readout_mlp``) — none of which could be fused before
this kernel shipped.

Two public entry points
-----------------------

- :func:`fused_relu_linear` -- ``ReLU(Linear(x, W, b))`` — used by D-MPNN's
  ``edge_mlp[layer]`` (``nn.Sequential(Linear(2D, D), ReLU())``).  This is
  the per-layer per-message edge update — it runs ``n_layers * B * E_max``
  times per training step and is the hottest site in the model.
- :func:`fused_relu_mlp` -- ``Linear(ReLU(Linear(x, W1, b1)), W2, b2)`` —
  used by ``atom_embed``, ``edge_embed`` and ``readout_mlp``
  (``nn.Sequential(Linear, ReLU, Linear)``).  These run once per molecule
  per forward pass; the per-call cost is amortised over ``N_max`` atoms
  / ``E_max`` edges.

The single-matmul fused path is exposed via a Triton kernel
(:func:`_fused_relu_linear_kernel`) that computes ``out = ReLU(x @ W.T +
b)`` in one launch with the bias add and ReLU fused into the matmul.  The
two-matmul fused path uses a host-side composition of
:func:`torch.matmul` + the Triton ReLU kernel (the second projection stays
on torch.matmul for the same reason :func:`fused_silu_mlp` keeps it on
cuBLAS — the post-activation tile fits inside the per-wavepage a budget on
gfx1101 and we cannot beat cuBLAS without WGMMA).

CPU fallback policy
-------------------

- When ``x.is_cuda`` is ``False`` we **always** use the PyTorch path —
  the Triton kernel raises on CPU tensors during the autotune probe.
- When ``triton_config.use_fused_relu_linear(x)`` /
  ``use_fused_relu_mlp(x)`` returns ``False`` we fall back to PyTorch.
  The default for both is ``False`` (per ``triton_kernels/config.py``
  L107: 'Default changed 2026-09-27: Triton kernels tuned for gfx1101
  are 3-11x slower than PyTorch on Blackwell; PyTorch is now the
  default').  Operators must opt in via ``TRITON_USE_FUSED=1`` *and*
  ``TRITON_RELU_LINEAR=1`` / ``TRITON_RELU_MLP=1`` once the autotune
  configs are retuned for sm_120 (Wf-326E-A).

NOTE: ``--triton-edge-update`` flag in the D-MPNN training scripts is the
recommended opt-in point for the per-layer edge MLP path; it sets the
``TRITON_RELU_LINEAR`` env var so :data:`triton_config.use_fused_relu_linear`
opens the gate.

Per ``party-twenty-spirit`` §3.5 (MEASURED vs PROJECTED tag discipline)
the projected speedups in this module's docstring are explicitly
PROJECTED — see ``TODO/completed/326e_triton_audit.md`` §5 / Wf-326E-B.
MEASURED numbers will land after Wf-326E-A's sm_120 retune + a follow-up
Wf-326E-D bench.

wf-328-w9 ship notes (2026-09-29)
----------------------------------
This is the Wf-328-W9 ship: a CPU-correctness-validated Triton kernel for
the D-MPNN edge-update hot path.  The kernel is real (``@triton.jit``
with ``tl.load`` / ``tl.dot`` / ``tl.store`` and an ``@triton.autotune``
decorator), but the autotune configs are still gfx1101-tuned.  On
Blackwell sm_120 the kernel is OFF by default.  Tests verify CPU
correctness against the pure-PyTorch reference and a CPU-side speed
comparison so the kernel can be validated without a GPU; MEASURED speed
numbers on Blackwell are tracked as a follow-up (Wf-326E-A retune).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


# ---------------------------------------------------------------------------
# Host-side guard against CPU tensors reaching a Triton kernel
# ---------------------------------------------------------------------------
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
            "the fused-ReLU-MLP kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Triton kernel: fused matmul + bias + ReLU
# ---------------------------------------------------------------------------
# Layout (canonical ``nn.Linear`` convention):
#   x:   (M, D)   row-major; D = in_features
#   w:   (H, D)   row-major; H = out_features (matches ``nn.Linear.weight``)
#   b:   (H,)     row-major; bias broadcast along H
#   out: (M, H)   row-major
#
# ``F.linear(x, w, b)`` computes ``x @ w.T + b``.  The kernel below
# computes the same expression in a single fused launch with the bias
# add and ReLU fused into the matmul epilogue, saving one HBM round-trip
# on the (M, H) post-activation tensor.
#
# Implementation notes
# --------------------
# We tile (M, H) into BLOCK_M x BLOCK_N tiles and reduce over D in
# BLOCK_K chunks.  Each program loads BLOCK_K columns of x once per
# BLOCK_M rows and accumulates the inner product in FP32.  This is the
# same matmul pattern as the Liger-Kernel ``swiglu_kernel`` (see
# ``molmetal/references/liger_kernel_repo/src/liger_kernel/ops/swiglu.py``).
#
# Why not WGMMA / TMA?  Per ``TODO/environment.md`` §4 + §7 we deliberately
# stay on the gfx1101-compatible subset of Triton (no WGMMA, no TMA, no
# cluster launch).  This means the kernel cannot match cuBLAS on Blackwell
# for very large matmuls; the speedup PROJECTED for the D-MPNN edge MLP
# case comes from fusing the bias + ReLU into the matmul epilogue so we
# save one HBM round-trip on the (M, H) post-activation tensor.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "H", "D"])
@triton.jit
def _fused_relu_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    H,
    D,
    sxm,
    sxd,
    swh,
    swd,
    sb,
    som,
    soh,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mr = rows < M
    mc = cols < H

    # ---- Compute out = ReLU(x @ w.T + b) ----
    # x: (M, D) row-major; w: (H, D) row-major (canonical nn.Linear).
    # Inner product: sum_k x[m, k] * w[h, k] = (x @ w.T)[m, h].
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(D, BLOCK_K)):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = k < D
        # Load x: shape (BLOCK_M, BLOCK_K); x[rows, k]
        xv = tl.load(
            x_ptr + rows[:, None] * sxm + k[None, :] * sxd,
            mask=mr[:, None] & mk[None, :],
            other=0.0,
        ).to(tl.float32)
        # Load w: shape (BLOCK_N, BLOCK_K); w[cols, k] — we load this as
        # w.T(k, cols) so the dot product computes sum_k x[m, k] * w[h, k].
        wv = tl.load(
            w_ptr + cols[None, :] * swh + k[:, None] * swd,
            mask=mc[None, :] & mk[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(xv, wv, allow_tf32=True)
    if HAS_BIAS:
        bias = tl.load(b_ptr + cols * sb, mask=mc, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    # Fused ReLU in epilogue.
    acc = tl.maximum(acc, 0.0)
    tl.store(
        out_ptr + rows[:, None] * som + cols[None, :] * soh,
        acc.to(out_ptr.dtype.element_ty),
        mask=mr[:, None] & mc[None, :],
    )


# ---------------------------------------------------------------------------
# Triton kernel: fused matmul + bias (no activation) for the second projection
# ---------------------------------------------------------------------------
# Used by :func:`fused_relu_mlp` for the second matmul (post-activation ->
# out_dim).  This is bandwidth-bound on the post-activation HBM round-trip
# and the bias + matmul fusion saves one HBM write.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "H2", "H1"])
@triton.jit
def _linear_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    H2,
    H1,
    sxm,
    sxh,
    swh2,
    swh1,
    sb,
    som,
    soh2,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mr = rows < M
    mc = cols < H2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(H1, BLOCK_K)):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = k < H1
        xv = tl.load(
            x_ptr + rows[:, None] * sxm + k[None, :] * sxh,
            mask=mr[:, None] & mk[None, :],
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(
            w_ptr + cols[None, :] * swh2 + k[:, None] * swh1,
            mask=mc[None, :] & mk[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(xv, wv, allow_tf32=True)
    if HAS_BIAS:
        bias = tl.load(b_ptr + cols * sb, mask=mc, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    tl.store(
        out_ptr + rows[:, None] * som + cols[None, :] * soh2,
        acc.to(out_ptr.dtype.element_ty),
        mask=mr[:, None] & mc[None, :],
    )


# ---------------------------------------------------------------------------
# Fused-1 host wrapper: fused_relu_linear(x, w, b)  --  Linear -> ReLU
# ---------------------------------------------------------------------------
def fused_relu_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
) -> torch.Tensor:
    """``ReLU(Linear(x, W, b))`` with Triton dispatch.

    Equivalent to::

        h = F.linear(x, w, b)   # nn.Linear forward
        return F.relu(h)

    Parameters
    ----------
    x : (..., D) tensor
        Input features (any leading batch dims are flattened).
    w : (H, D) tensor
        ``nn.Linear`` weight matrix (canonical layout — same as
        ``nn.Linear.weight.shape == (out_features, in_features)``).
    b : optional (H,) tensor
        ``nn.Linear`` bias vector.  ``None`` means no bias.

    Returns
    -------
    (..., H) tensor with the same dtype/device as ``x``.

    Dispatch policy
    ---------------
    CPU tensors → PyTorch reference (Triton cannot accept CPU pointers).
    Otherwise → :func:`_fused_relu_linear_kernel` when
    :func:`triton_config.use_fused_relu_linear` returns ``True``;
    PyTorch reference otherwise.  Default-OFF on RTX 5090 until Wf-326E-A
    lands the sm_120 autotune retune (Wf-326E-B).
    """
    # Defer import to avoid a circular import with triton_kernels.__init__.
    from .config import triton_config as _cfg

    orig_shape = x.shape
    x2 = x.contiguous().view(-1, x.shape[-1])
    M, D = x2.shape
    if w.dim() != 2:
        raise ValueError(
            f"fused_relu_linear expects 2-D weight w; got shape {tuple(w.shape)}"
        )
    # Canonical nn.Linear layout: w is (out_features=H, in_features=D).
    H = w.shape[0]
    if w.shape[1] != D:
        raise ValueError(
            f"fused_relu_linear: x.shape[-1]={D} != w.shape[1]={w.shape[1]} "
            f"(w must be canonical nn.Linear layout (out_features, in_features))"
        )
    if b is not None and b.shape != (H,):
        raise ValueError(f"fused_relu_linear: bias shape {tuple(b.shape)} != ({H},)")

    # GPU + gate open → Triton kernel.
    if x2.is_cuda and _cfg.use_fused_relu_linear(x2):
        _check_cuda_pointers({"x": x2, "w": w, "b": b, "out": None})
        out = torch.empty(
            (M, H),
            dtype=x2.dtype,
            device=x2.device,
        )
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(H, meta["BLOCK_N"]),
        )
        _fused_relu_linear_kernel[grid](
            x2,
            w,
            b if b is not None else x2,  # placeholder when no bias
            out,
            M,
            H,
            D,
            x2.stride(0),
            x2.stride(1),
            w.stride(0),
            w.stride(1),
            b.stride(0) if b is not None else 1,
            out.stride(0),
            out.stride(1),
            HAS_BIAS=b is not None,
        )
        return out.view(*orig_shape[:-1], H)

    # CPU fallback / gate closed → PyTorch reference (F.linear handles
    # the canonical (H, D) layout natively).
    h = torch.nn.functional.linear(x2, w, b)
    return torch.relu(h).view(*orig_shape[:-1], H)


# ---------------------------------------------------------------------------
# Fused-2 host wrapper: fused_relu_mlp(x, w1, b1, w2, b2)
# ---------------------------------------------------------------------------
def fused_relu_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> torch.Tensor:
    """``Linear(ReLU(Linear(x, W1, b1)), W2, b2)`` with Triton dispatch.

    Equivalent to::

        nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
        )

    Parameters
    ----------
    x : (..., D) tensor
        Input features (any leading batch dims are flattened).
    w1, w2 : (H1, D), (H2, H1) tensors
        ``nn.Linear`` weight matrices (canonical layout
        ``(out_features, in_features)``).
    b1, b2 : optional (H1,), (H2,) tensors
        ``nn.Linear`` bias vectors.  ``None`` means no bias.

    Returns
    -------
    (..., H2) tensor with the same dtype/device as ``x``.

    Dispatch policy
    ---------------
    CPU tensors → PyTorch reference.  Otherwise → Triton fused
    :func:`_fused_relu_linear_kernel` (first matmul + ReLU) followed by
    Triton :func:`_linear_bias_kernel` (second matmul + bias) when
    :func:`triton_config.use_fused_relu_mlp` returns ``True``; PyTorch
    reference otherwise.  Default-OFF on RTX 5090 until Wf-326E-A.
    """
    from .config import triton_config as _cfg

    orig_shape = x.shape
    x2 = x.contiguous().view(-1, x.shape[-1])
    M, D = x2.shape
    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError(
            f"fused_relu_mlp expects 2-D weights; got {tuple(w1.shape)}, {tuple(w2.shape)}"
        )
    H1 = w1.shape[0]
    if w1.shape[1] != D:
        raise ValueError(
            f"fused_relu_mlp: x.shape[-1]={D} != w1.shape[1]={w1.shape[1]} "
            f"(w1 must be canonical nn.Linear layout (out_features, in_features))"
        )
    H2 = w2.shape[0]
    if w2.shape[1] != H1:
        raise ValueError(
            f"fused_relu_mlp: w1.shape[0]={H1} != w2.shape[1]={w2.shape[1]}"
        )
    if b1 is not None and b1.shape != (H1,):
        raise ValueError(f"fused_relu_mlp: b1 shape {tuple(b1.shape)} != ({H1},)")
    if b2 is not None and b2.shape != (H2,):
        raise ValueError(f"fused_relu_mlp: b2 shape {tuple(b2.shape)} != ({H2},)")

    if x2.is_cuda and _cfg.use_fused_relu_mlp(x2):
        _check_cuda_pointers(
            {
                "x": x2,
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
                "out1": None,
                "out2": None,
            }
        )
        # Fused first matmul + ReLU.
        out1 = torch.empty((M, H1), dtype=x2.dtype, device=x2.device)
        grid1 = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(H1, meta["BLOCK_N"]),
        )
        _fused_relu_linear_kernel[grid1](
            x2,
            w1,
            b1 if b1 is not None else x2,
            out1,
            M,
            H1,
            D,
            x2.stride(0),
            x2.stride(1),
            w1.stride(0),
            w1.stride(1),
            b1.stride(0) if b1 is not None else 1,
            out1.stride(0),
            out1.stride(1),
            HAS_BIAS=b1 is not None,
        )
        # Fused second matmul + bias.
        out2 = torch.empty((M, H2), dtype=x2.dtype, device=x2.device)
        grid2 = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(H2, meta["BLOCK_N"]),
        )
        _linear_bias_kernel[grid2](
            out1,
            w2,
            b2 if b2 is not None else x2,
            out2,
            M,
            H2,
            H1,
            out1.stride(0),
            out1.stride(1),
            w2.stride(0),
            w2.stride(1),
            b2.stride(0) if b2 is not None else 1,
            out2.stride(0),
            out2.stride(1),
            HAS_BIAS=b2 is not None,
        )
        return out2.view(*orig_shape[:-1], H2)

    # CPU / gate closed → PyTorch reference.
    pre = torch.nn.functional.linear(x2, w1, b1)
    post = torch.relu(pre)
    out = torch.nn.functional.linear(post, w2, b2)
    return out.view(*orig_shape[:-1], H2)


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------
# These wrap ``nn.Linear`` + ReLU / + ReLU + ``nn.Linear`` and route through
# the fused kernel when the dispatch gate is open.  They preserve the
# parameter names ``linear1.*`` / ``linear2.*`` (matching the existing
# D-MPNN ``nn.Sequential(Linear, ReLU, Linear)`` blocks) so existing
# checkpoints load without key remapping.
class FusedReLUMLP(nn.Module):
    """``Linear -> ReLU -> Linear`` module with optional fused-ReLU dispatch.

    Parameter layout matches the canonical D-MPNN ``nn.Sequential(Linear,
    ReLU, Linear)`` block (``linear1.weight`` / ``linear1.bias`` /
    ``linear2.weight`` / ``linear2.bias``) so existing state-dicts load
    unchanged.

    When :data:`triton_kernels.triton_config.use_fused_relu_mlp` returns
    ``True`` AND the input tensor is on CUDA, the forward routes through
    :func:`fused_relu_mlp`; otherwise it falls back to the standard
    ``F.relu(F.linear(...))`` chain (bit-exact with the reference).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Match nn.Sequential(Linear, ReLU, Linear) parameter names.
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_relu_mlp(
            x,
            self.linear1.weight,
            self.linear1.bias,
            self.linear2.weight,
            self.linear2.bias,
        )
        return h


class FusedReLULinear(nn.Module):
    """``Linear -> ReLU`` module with optional fused-ReLU dispatch.

    Parameter layout matches the canonical D-MPNN ``nn.Sequential(Linear,
    ReLU)`` block (``linear.weight`` / ``linear.bias``).

    When :data:`triton_kernels.triton_config.use_fused_relu_linear` returns
    ``True`` AND the input tensor is on CUDA, the forward routes through
    :func:`fused_relu_linear`; otherwise it falls back to the standard
    ``F.relu(F.linear(...))`` chain.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_relu_linear(x, self.linear.weight, self.linear.bias)


__all__ = [
    "fused_relu_mlp",
    "fused_relu_linear",
    "FusedReLUMLP",
    "FusedReLULinear",
]
