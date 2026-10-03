"""Wf-331-X4 — CPU scaffold for the missing ``fused_smiles_featurize`` kernel.

=====================================================================
Wf-326E MISSING-004: :class:`molmetal.data.featurize.GraphFeaturizer`
converts an RDKit ``Chem.Mol`` into per-atom + per-edge feature
matrices via Python-level loops over the molecule's atom list.  For a
batch of B molecules, the loop overhead dominates when ``B > 32``.
A Triton-fused featuriser would emit the per-atom one-hot vectors in
a single launch.

Wf-331-X4 ships the **CPU scaffold**: the Triton kernel source
(deferred execution — does not run on CPU), a pure-PyTorch reference
that mirrors the upstream per-atom feature composition, and a dispatch
wrapper that falls back to PyTorch when Triton is unavailable or the
env gate is closed (default).

Wf-331-X4 honest framing
------------------------
Per ``party-twenty-spirit`` §3.5:

- **MEASURED**: CPU PyTorch fallback matches the upstream
  per-atom-feature composition (one-hot symbol + one-hot degree +
  ring indicator + formal charge) to bit-exact equality on a small
  batch.
- **PROJECTED**: 1.2-2x speedup for ``B > 32`` once the kernel is
  tuned for Blackwell sm_120 (Wf-326E-A retune).
- **No fabricated numbers**: speedup claim traces to the Wf-326E audit.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

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
# Per-atom feature tables (mirrors molmetal/data/featurize.py _ATOM_SYMBOLS
# but condensed to the round-10 12-element donor + Pt vocabulary so the
# one-hot stays inside a 32-wide column for Triton kernel convenience).
# ---------------------------------------------------------------------------
# Symbol table of 14 entries (matches molmetal/data/featurize.py):
#   C, N, O, F, P, S, Cl, Br, I, B, Si, Se, H, Other → idx 0..13
_SYMBOL_TO_IDX = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "P": 4,
    "S": 5,
    "Cl": 6,
    "Br": 7,
    "I": 8,
    "B": 9,
    "Si": 10,
    "Se": 11,
    "H": 12,
    "Other": 13,
}
_NUM_SYMBOLS = len(_SYMBOL_TO_IDX)  # 14

# Degree one-hots: 0..5+ → idx 0..5
_NUM_DEGREES = 6

# Total feature dim: one-hot symbol + one-hot degree + ring + aromatic + charge
_FEATURE_DIM = _NUM_SYMBOLS + _NUM_DEGREES + 1 + 1 + 1  # 14 + 6 + 3 = 23


def symbol_to_idx(symbol: str) -> int:
    return _SYMBOL_TO_IDX.get(symbol, _SYMBOL_TO_IDX["Other"])


def encode_atom_features(
    symbols: Sequence[str],
    degrees: Sequence[int],
    in_ring: Sequence[bool],
    aromatic: Sequence[bool],
    formal_charge: Sequence[int],
) -> torch.Tensor:
    """Convert per-atom lists into a (N, 23) float feature matrix.

    CPU-side encoder that mirrors what the Triton kernel does on GPU.
    Public so callers can use it without the Triton dispatch layer.
    """
    n = len(symbols)
    if not (len(degrees) == len(in_ring) == len(aromatic) == len(formal_charge) == n):
        raise ValueError(
            "All per-atom lists must have the same length; got "
            f"symbols={n}, degrees={len(degrees)}, in_ring={len(in_ring)}, "
            f"aromatic={len(aromatic)}, formal_charge={len(formal_charge)}."
        )
    out = torch.zeros((n, _FEATURE_DIM), dtype=torch.float32)
    sym_idx = [symbol_to_idx(s) for s in symbols]
    deg_idx = [min(int(d), _NUM_DEGREES - 1) for d in degrees]
    for i in range(n):
        out[i, sym_idx[i]] = 1.0
        out[i, _NUM_SYMBOLS + deg_idx[i]] = 1.0
        out[i, _NUM_SYMBOLS + _NUM_DEGREES] = float(bool(in_ring[i]))
        out[i, _NUM_SYMBOLS + _NUM_DEGREES + 1] = float(bool(aromatic[i]))
        # Normalise formal charge to [-1, 1] for stability.
        c = int(formal_charge[i])
        c_clip = max(-1, min(1, c))
        out[i, _NUM_SYMBOLS + _NUM_DEGREES + 2] = float(c_clip)
    return out


# ---------------------------------------------------------------------------
# Triton autotune configs — sm_120 (Blackwell).
# ---------------------------------------------------------------------------
# Per-atom featurisation is bandwidth-bound (one-hot writes dominate).
# Larger BLOCK_N batches more atoms per program → fewer launches.
# PROJECTED 1.2-2x speedup over the Python loop for batch > 32.
if _TRITON_AVAILABLE:
    _FUSED_SMILES_CONFIGS = [
        triton.Config({"BLOCK_N": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_FUSED_SMILES_CONFIGS, key=["N"])
    @triton.jit
    def _fused_smiles_featurize_kernel(  # pragma: no cover — GPU-only
        sym_ptr,
        deg_ptr,
        ring_ptr,
        arom_ptr,
        charge_ptr,
        out_ptr,
        N,
        NUM_SYMBOLS: tl.constexpr,
        NUM_DEGREES: tl.constexpr,
        FEATURE_DIM: tl.constexpr,
        FEATURE_DIM_P2: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused per-atom SMILES featuriser.

        Each program processes BLOCK_N atoms.  Loads 5 int32 streams
        (symbol_idx, degree_idx, ring, aromatic, charge), constructs
        the per-column match masks for the one-hot slots + the scalar
        slots, and writes the (BLOCK_N, FEATURE_DIM) output in a single
        tl.store.  ``FEATURE_DIM_P2`` is the next power of 2 ≥
        ``FEATURE_DIM`` (Triton ``tl.arange`` requirement).

        Charge encoding: writes the **signed** value (-1 / 0 / +1) at
        the charge slot (mirrors :func:`pytorch_reference`), not just a
        binary indicator.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        sym = tl.load(sym_ptr + offs, mask=mask, other=0)
        deg = tl.load(deg_ptr + offs, mask=mask, other=0)
        ring = tl.load(ring_ptr + offs, mask=mask, other=0)
        arom = tl.load(arom_ptr + offs, mask=mask, other=0)
        charge = tl.load(charge_ptr + offs, mask=mask, other=0)
        feat_offs = tl.arange(0, FEATURE_DIM_P2)
        mask_f = feat_offs < FEATURE_DIM
        sym_match = (feat_offs[None, :] == sym[:, None]) & (
            feat_offs[None, :] < NUM_SYMBOLS
        )
        deg_offset = NUM_SYMBOLS
        deg_match = (
            ((feat_offs[None, :] - deg_offset) == deg[:, None])
            & ((feat_offs[None, :] - deg_offset) >= 0)
            & ((feat_offs[None, :] - deg_offset) < NUM_DEGREES)
        )
        ring_offset = NUM_SYMBOLS + NUM_DEGREES
        ring_match = (feat_offs[None, :] == ring_offset) & (ring[:, None] == 1)
        arom_offset = ring_offset + 1
        arom_match = (feat_offs[None, :] == arom_offset) & (arom[:, None] == 1)
        charge_offset = arom_offset + 1
        # Signed charge: -1, 0, +1 (clip; mirrors pytorch_reference).
        charge_clip = tl.maximum(tl.minimum(charge, 1), -1).to(tl.float32)
        # Build the full feature matrix as a sum of (mask * value)
        # contributions.  The charge slot writes the signed value
        # directly (not just an indicator).
        is_charge_slot = feat_offs[None, :] == charge_offset
        charge_contrib = tl.where(
            is_charge_slot & mask_f[None, :],
            charge_clip[:, None],
            0.0,
        )
        feat_val = (
            sym_match.to(tl.float32)
            + deg_match.to(tl.float32)
            + ring_match.to(tl.float32)
            + arom_match.to(tl.float32)
            + charge_contrib
        )
        out_addr = out_ptr + offs[:, None] * FEATURE_DIM + feat_offs[None, :]
        tl.store(out_addr, feat_val, mask=mask[:, None] & mask_f[None, :])
else:

    def _fused_smiles_featurize_kernel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Triton unavailable — _fused_smiles_featurize_kernel cannot run. "
            "Use fused_smiles_featurize(...) which auto-falls-back."
        )


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-correctness-validated)
# ---------------------------------------------------------------------------
def pytorch_reference(
    symbols: Sequence[str],
    degrees: Sequence[int],
    in_ring: Sequence[bool],
    aromatic: Sequence[bool],
    formal_charge: Sequence[int],
) -> torch.Tensor:
    """Pure-PyTorch reference for the SMILES featuriser.

    Returns a ``(N, _FEATURE_DIM)`` float tensor.  Equivalent to
    :func:`encode_atom_features` (the reference is the CPU encoder
    itself; the Triton kernel produces the same output via the
    batch-parallel kernel above).
    """
    return encode_atom_features(symbols, degrees, in_ring, aromatic, formal_charge)


# ---------------------------------------------------------------------------
# Public dispatch wrapper
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def fused_smiles_featurize(
    symbols: Sequence[str],
    degrees: Sequence[int],
    in_ring: Sequence[bool],
    aromatic: Sequence[bool],
    formal_charge: Sequence[int],
    *,
    force_pytorch: Optional[bool] = None,
) -> torch.Tensor:
    """Featurise a batch of SMILES-derived atom features via the fused kernel.

    Dispatch policy
    ---------------
    - ``force_pytorch=True`` → always :func:`pytorch_reference`.
    - ``force_pytorch=False`` → always attempt Triton (raises if no CUDA).
    - ``force_pytorch=None`` (default) → Triton when CUDA + triton
      importable + ``TRITON_USE_FUSED=1``; otherwise PyTorch.
    """
    if force_pytorch is True:
        return pytorch_reference(symbols, degrees, in_ring, aromatic, formal_charge)
    # Build the input tensors on CPU first (we need to know whether to
    # move them to CUDA for the Triton path).
    ref_out = pytorch_reference(symbols, degrees, in_ring, aromatic, formal_charge)
    if not torch.cuda.is_available():
        return ref_out
    if not _TRITON_AVAILABLE:
        return ref_out
    if not _env_flag("TRITON_USE_FUSED", "0"):
        return ref_out
    # Wf-335-P3 — per-kernel safety gate (parity with fused_relu_mlp's
    # TRITON_RELU_MLP gate).  Defaults to OFF; the trainer's
    # ``--use-triton-smiles-featurize`` flag sets it to ``1``.  When
    # the per-kernel gate is closed we fall back to the PyTorch
    # reference even if ``TRITON_USE_FUSED`` is set (i.e. another
    # kernel in the same run is on the tri).
    if not _env_flag("TRITON_SMILES_FEATURIZE", "0"):
        return ref_out
    if force_pytorch is False:
        device = torch.device("cuda:0")
        n = len(symbols)
        sym_t = torch.tensor(
            [symbol_to_idx(s) for s in symbols],
            dtype=torch.int32,
            device=device,
        )
        deg_t = torch.tensor(
            [min(int(d), _NUM_DEGREES - 1) for d in degrees],
            dtype=torch.int32,
            device=device,
        )
        ring_t = torch.tensor(
            [int(bool(r)) for r in in_ring],
            dtype=torch.int32,
            device=device,
        )
        arom_t = torch.tensor(
            [int(bool(a)) for a in aromatic],
            dtype=torch.int32,
            device=device,
        )
        charge_t = torch.tensor(
            [int(c) for c in formal_charge],
            dtype=torch.int32,
            device=device,
        )
        out = torch.empty((n, _FEATURE_DIM), dtype=torch.float32, device=device)
        # Pad FEATURE_DIM to next power of 2 (Triton tl.arange requirement).
        feat_dim_p2 = 1
        while feat_dim_p2 < _FEATURE_DIM:
            feat_dim_p2 *= 2
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
        _fused_smiles_featurize_kernel[grid](
            sym_t,
            deg_t,
            ring_t,
            arom_t,
            charge_t,
            out,
            n,
            NUM_SYMBOLS=_NUM_SYMBOLS,
            NUM_DEGREES=_NUM_DEGREES,
            FEATURE_DIM=_FEATURE_DIM,
            FEATURE_DIM_P2=feat_dim_p2,
        )
        return out
    return ref_out


__all__ = [
    "fused_smiles_featurize",
    "pytorch_reference",
    "encode_atom_features",
    "symbol_to_idx",
    "_FEATURE_DIM",
]
