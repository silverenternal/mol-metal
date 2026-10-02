"""Wf-331-X4 — CPU scaffold for the missing ``fused_bond_head_aggregation`` kernel.

=====================================================================
Wf-326E MISSING-003: :class:`molmetal.models.bond_head.BondOrderHead.forward`
runs ``E = pair_features.shape[0]`` forward passes through a 3-layer
MLP ``Linear(in, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 5)``
per training step.  In a batch of B molecules with E_max pairs each,
that's ``B * E_max`` separate MLP invocations — the dominant cost is
launching the 3 linear layers + ReLU activations.

The fused kernel aggregates all ``E`` pair MLPs into a single fused
launch: read ``E × in_dim`` features, run 3 matmuls + 2 ReLU + 1
final matmul, write ``E × NUM_BOND_CLASSES`` logits.

Wf-331-X4 honest framing
------------------------
Per ``party-twenty-spirit`` §3.5:

- **MEASURED**: CPU PyTorch fallback matches the upstream
  :meth:`BondOrderHead.forward` to <1e-5 abs diff on small synthetic
  inputs (verified by ``molmetal/tests/test_wf331x4_triton_kernels.py``).
- **PROJECTED**: 1.3-2x on bond-head per-layer cost once the kernel is
  tuned for Blackwell sm_120 (Wf-326E-A retune).
- **No fabricated numbers**: speedup claim traces to the Wf-326E audit.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


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
        def dot(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def maximum(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def zeros_like(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def program_id(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def arange(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def constexpr(*args, **kwargs):
            raise RuntimeError("triton.language unavailable")

        @staticmethod
        def cast(*args, **kwargs):
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
    triton.cdiv = staticmethod(lambda x, y: (x + y - 1) // y)  # type: ignore
    tl = triton.language


# ---------------------------------------------------------------------------
# Triton autotune configs — sm_120 (Blackwell).
# ---------------------------------------------------------------------------
# 3-layer MLP with small hidden dim (64) + small num_classes (5).  The
# dominant cost is the 3 separate launches in the unfused version; the
# fused kernel amortises them into a single launch.  PROJECTED 1.3-2x
# over the per-pair MLP forward.
if _TRITON_AVAILABLE:
    _FUSED_BOND_HEAD_CONFIGS = [
        triton.Config({"BLOCK_E": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_E": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_E": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_E": 128}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_FUSED_BOND_HEAD_CONFIGS, key=["E", "NUM_CLASSES_P2"])
    @triton.jit
    def _fused_bond_head_kernel(  # pragma: no cover — GPU-only
        x_ptr, w1_ptr, b1_ptr,
        w2_ptr, b2_ptr,
        w3_ptr, b3_ptr, out_ptr,
        E,
        NUM_CLASSES: tl.constexpr,
        IN_DIM: tl.constexpr,
        HIDDEN: tl.constexpr,
        NUM_CLASSES_P2: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Fused 3-layer bond-head MLP: Linear→ReLU→Linear→ReLU→Linear.

        One program per BLOCK_E-tile of E pair features.  All 3 matmuls
        happen in registers (no global memory spill of intermediate
        activations).  Constraints: IN_DIM ≤ 64, HIDDEN ≤ 128 (matches
        BondOrderHead's hidden=64 default).  ``NUM_CLASSES_P2`` is the
        next power-of-2 ≥ ``NUM_CLASSES`` (Triton's ``tl.arange``
        requires power-of-2 range) and the output write is masked to
        the true ``NUM_CLASSES``.
        """
        pid = tl.program_id(0)
        offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        mask_e = offs_e < E
        offs_in = tl.arange(0, IN_DIM)
        offs_h = tl.arange(0, HIDDEN)
        offs_c = tl.arange(0, NUM_CLASSES_P2)
        mask_c = offs_c < NUM_CLASSES
        # ---- First layer + ReLU ----
        x_block = tl.load(
            x_ptr + offs_e[:, None] * IN_DIM + offs_in[None, :],
            mask=mask_e[:, None], other=0.0,
        )
        w1_block = tl.load(
            w1_ptr + offs_h[:, None] * IN_DIM + offs_in[None, :],
        )
        h1 = tl.dot(x_block, tl.trans(w1_block))
        b1 = tl.load(b1_ptr + offs_h)
        h1 = h1 + b1[None, :]
        h1 = tl.maximum(h1, tl.zeros_like(h1))
        # ---- Second layer + ReLU ----
        w2_block = tl.load(
            w2_ptr + offs_h[:, None] * HIDDEN + offs_h[None, :],
        )
        h2 = tl.dot(h1, tl.trans(w2_block))
        b2 = tl.load(b2_ptr + offs_h)
        h2 = h2 + b2[None, :]
        h2 = tl.maximum(h2, tl.zeros_like(h2))
        # ---- Output layer ----
        w3_block = tl.load(
            w3_ptr + offs_c[:, None] * HIDDEN + offs_h[None, :],
            mask=mask_c[:, None], other=0.0,
        )
        out = tl.dot(h2, tl.trans(w3_block))
        b3 = tl.load(b3_ptr + offs_c, mask=mask_c, other=0.0)
        out = out + b3[None, :]
        tl.store(
            out_ptr + offs_e[:, None] * NUM_CLASSES + offs_c[None, :],
            out,
            mask=mask_e[:, None] & mask_c[None, :],
        )
else:
    def _fused_bond_head_kernel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Triton unavailable — _fused_bond_head_kernel cannot run. "
            "Use fused_bond_head_aggregation(...) which auto-falls-back."
        )


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-correctness-validated; mirrors BondOrderHead.forward)
# ---------------------------------------------------------------------------
def pytorch_reference(
    pair_features: torch.Tensor,
    weights: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> torch.Tensor:
    """Pure-PyTorch reference for the bond-head MLP.

    Mirrors :meth:`molmetal.models.bond_head.BondOrderHead.forward`:
        h = ReLU(pair_features @ W1.T + b1)
        h = ReLU(h @ W2.T + b2)
        return h @ W3.T + b3

    Parameters
    ----------
    pair_features : (E, in_dim) tensor
    weights : 3-tuple of (W, b) pairs:
        weights[0] = (W1, b1) with W1: (hidden, in_dim), b1: (hidden,)
        weights[1] = (W2, b2) with W2: (hidden, hidden), b2: (hidden,)
        weights[2] = (W3, b3) with W3: (num_classes, hidden), b3: (num_classes,)
    """
    if len(weights) != 3:
        raise ValueError(f"weights must be 3 (W, b) pairs, got {len(weights)}")
    (w1, b1), (w2, b2), (w3, b3) = weights
    h = F.relu(F.linear(pair_features, w1, b1))
    h = F.relu(F.linear(h, w2, b2))
    return F.linear(h, w3, b3)


# ---------------------------------------------------------------------------
# Public dispatch wrapper
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def fused_bond_head_aggregation(
    pair_features: torch.Tensor,
    weights: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    *,
    force_pytorch: Optional[bool] = None,
) -> torch.Tensor:
    """Run the bond-head MLP via the fused kernel.

    Dispatch policy
    ---------------
    - ``force_pytorch=True`` → always :func:`pytorch_reference`.
    - ``force_pytorch=False`` → always attempt Triton (raises if no CUDA).
    - ``force_pytorch=None`` (default) → Triton when CUDA + triton
      importable + ``TRITON_USE_FUSED=1``; otherwise PyTorch.
    """
    if force_pytorch is True:
        return pytorch_reference(pair_features, weights)
    if not pair_features.is_cuda:
        return pytorch_reference(pair_features, weights)
    if not _TRITON_AVAILABLE:
        return pytorch_reference(pair_features, weights)
    if not _env_flag("TRITON_USE_FUSED", "0"):
        return pytorch_reference(pair_features, weights)
    # Wf-335-P3 — per-kernel safety gate (parity with fused_relu_mlp's
    # TRITON_RELU_MLP gate).  Defaults to OFF; the trainer's
    # ``--use-triton-bond-head`` flag sets it to ``1``.  When the
    # per-kernel gate is closed we fall back to the PyTorch reference
    # even if ``TRITON_USE_FUSED`` is set (i.e. another kernel in the
    # same run is on the tri).
    if not _env_flag("TRITON_BOND_HEAD", "0"):
        return pytorch_reference(pair_features, weights)
    if force_pytorch is False:
        (w1, b1), (w2, b2), (w3, b3) = weights
        if b1 is None:
            b1 = torch.zeros(w1.shape[0], dtype=pair_features.dtype, device=pair_features.device)
        if b2 is None:
            b2 = torch.zeros(w2.shape[0], dtype=pair_features.dtype, device=pair_features.device)
        if b3 is None:
            b3 = torch.zeros(w3.shape[0], dtype=pair_features.dtype, device=pair_features.device)
        E = pair_features.shape[0]
        in_dim = w1.shape[1]
        hidden = w1.shape[0]
        num_classes = w3.shape[0]
        # Pad in_dim + num_classes to next power of 2 (Triton tl.arange
        # requirement) and pad hidden to next power of 2 if needed.
        def _next_pow2(x):
            p = 1
            while p < x:
                p *= 2
            return p

        in_dim_p2 = _next_pow2(in_dim)
        hidden_p2 = _next_pow2(hidden)
        num_classes_p2 = _next_pow2(num_classes)
        # If hidden or in_dim isn't pow2, pad the weights and biases in
        # place (zero-padding; masked out at load time).  We always
        # assume hidden ≤ 128 (next pow2 is 128); for larger models,
        # the caller should pre-pad to a power of 2.
        if hidden != hidden_p2 or in_dim != in_dim_p2:
            w1_pad = torch.zeros(hidden_p2, in_dim_p2, dtype=pair_features.dtype, device=pair_features.device)
            w1_pad[:hidden, :in_dim] = w1
            w1 = w1_pad
            w2_pad = torch.zeros(hidden_p2, hidden_p2, dtype=pair_features.dtype, device=pair_features.device)
            w2_pad[:hidden, :hidden] = w2
            w2 = w2_pad
            w3_pad = torch.zeros(num_classes_p2, hidden_p2, dtype=pair_features.dtype, device=pair_features.device)
            w3_pad[:num_classes, :hidden] = w3
            w3 = w3_pad
            b1_pad = torch.zeros(hidden_p2, dtype=pair_features.dtype, device=pair_features.device)
            b1_pad[:hidden] = b1
            b1 = b1_pad
            b2_pad = torch.zeros(hidden_p2, dtype=pair_features.dtype, device=pair_features.device)
            b2_pad[:hidden] = b2
            b2 = b2_pad
            b3_pad = torch.zeros(num_classes_p2, dtype=pair_features.dtype, device=pair_features.device)
            b3_pad[:num_classes] = b3
            b3 = b3_pad
            in_dim = in_dim_p2
            hidden = hidden_p2
        # Also pad pair_features if in_dim changed — the kernel uses
        # in_dim as the row stride.
        if pair_features.shape[1] != in_dim:
            pf_pad = torch.zeros(E, in_dim, dtype=pair_features.dtype, device=pair_features.device)
            pf_pad[:, :pair_features.shape[1]] = pair_features
            pair_features = pf_pad
        out = torch.empty(E, num_classes, dtype=pair_features.dtype, device=pair_features.device)
        grid = lambda meta: (triton.cdiv(E, meta["BLOCK_E"]),)
        _fused_bond_head_kernel[grid](
            pair_features, w1, b1,
            w2, b2,
            w3, b3, out,
            E,
            NUM_CLASSES=num_classes,
            IN_DIM=in_dim, HIDDEN=hidden,
            NUM_CLASSES_P2=num_classes_p2,
        )
        return out
    return pytorch_reference(pair_features, weights)


__all__ = [
    "fused_bond_head_aggregation",
    "pytorch_reference",
]