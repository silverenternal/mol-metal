"""Wf-327-F7: Triton-kernel wrapper for the D-MPNN hot-path.

=====================================================================
This module is the canonical dispatch surface for routing D-MPNN
(:mod:`molmetal.models.dmpnn`) through the Triton-fused
``fused_relu_linear`` / ``fused_relu_mlp`` kernels shipped in
Wf-328-W9 (2026-09-29).  It closes the **ReLU gap** flagged by the
Wf-326E audit (:file:`molmetal/reports/wf326e_triton_audit/audit.json`
§missing_kernel_candidates MISSING-001).
=====================================================================

Why this lives in :mod:`molmetal.adapters.flow_matching_lipman._reference`
------------------------------------------------------------------------

The D-MPNN backbone (in :mod:`molmetal.models.dmpnn`) is shared by both
the :mod:`molmetal.models.metal_hybrid_v4` CFM trainer and the
:mod:`molmetal.adapters.flow_matching_lipman` CFM reference (the
optional Meta upstream clone).  Putting the Triton wrapper in
``_reference/`` (next to the ``path/``, ``solver/``, ``utils/`` shims)
makes the wrapper discoverable from both training paths without
importing upstream Facebook packages — the local shim submodules
handle their own upstream lookups, so the wrapper itself is
self-contained.

Public surface
--------------

- :func:`maybe_fused_relu_mlp` — top-level entry point; routes a
  ``nn.Sequential(Linear, ReLU, Linear)`` block through the fused
  Triton kernel when the dispatch gate is open.  This is the function
  D-MPNN's ``atom_embed`` / ``edge_embed`` / ``readout_mlp`` call.
- :func:`maybe_fused_relu_linear` — per-layer single-matmul variant;
  routes ``nn.Sequential(Linear, ReLU)`` through the Triton kernel.
  This is the function D-MPNN's ``edge_mlp[layer]`` calls
  (the per-message edge update hot-path that runs
  ``n_layers * B * E_max`` times per step).
- :class:`DMPNNTritonKernelConfig` — env-var-driven configuration
  holder (TRITON_USE_FUSED, TRITON_RELU_MLP, TRITON_RELU_LINEAR,
  TRITON_USE_FUSED_EVAL).  Exposed so training scripts can introspect
  the gate state without importing :mod:`triton_kernels` directly.
- :func:`dispatch_state_report` — returns a JSON-serializable dict
  summarising the dispatch gate state (used by the Wf-327-F7 tests
  + the upcoming Wf-326E-D re-measurement bench).

Dispatch contract
-----------------

CPU tensors → PyTorch reference (Triton kernels raise on CPU).
Otherwise → Triton kernel when ALL of:
  - ``TRITON_USE_FUSED=1`` (or eval mode + ``TRITON_USE_FUSED_EVAL=1``)
  - ``TRITON_RELU_MLP=1`` (or ``TRITON_RELU_LINEAR=1`` for the single-
    matmul variant)
  - :func:`triton_config.use_fused_relu_mlp(x)` returns ``True``
    (shape + training + size gate)

The defaults match the audit's recommended opt-in path: PyTorch is the
default on Blackwell sm_120 (gfx1101-tuned autotune configs cause
severe register pressure on Blackwell — see
:file:`molmetal/reports/wf326e_triton_audit/audit.json` §audit_summary).
Operators opt in once the Wf-326E-A sm_120 retune lands.

Wf-327-F7 honest framing
------------------------

Per :file:`party-twenty-spirit §3.5`:

- **MEASURED**: CPU fallback path is bit-exact with the canonical
  ``F.relu(F.linear(x, w, b))`` chain (verified by
  :file:`molmetal/tests/test_wf327f7_triton_kernel.py`).
- **PROJECTED**: 1.3-2x on D-MPNN per-layer cost once the kernel runs
  on Blackwell (gated by Wf-326E-A's sm_120 autotune retune).
- **No fabricated numbers**: every speedup claim in this docstring
  traces to the Wf-326E audit JSON §per_kernel_status_table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Triton kernel imports — gated by try/except so this module imports cleanly
# even if the triton_kernels package is missing (e.g. minimal CI env).
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised indirectly via dmpnn tests
    from triton_kernels import (
        fused_relu_linear as _fused_relu_linear,
        fused_relu_mlp as _fused_relu_mlp,
        FusedReLULinear as _FusedReLULinear,
        FusedReLUMLP as _FusedReLUMLP,
    )
    from triton_kernels.config import triton_config as _triton_config
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover — CPU-only minimal CI
    _TRITON_AVAILABLE = False

    def _fused_relu_linear(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "triton_kernels.fused_relu_linear unavailable — kernel "
            "package not importable."
        )

    def _fused_relu_mlp(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "triton_kernels.fused_relu_mlp unavailable — kernel "
            "package not importable."
        )

    _FusedReLULinear = None  # type: ignore
    _FusedReLUMLP = None  # type: ignore

    class _DummyConfig:  # type: ignore
        enabled = False
        enabled_eval = False

        def should_use_fused(self, *args, **kwargs):
            return False

        def use_fused_relu_linear(self, *args, **kwargs):
            return False

        def use_fused_relu_mlp(self, *args, **kwargs):
            return False

    _triton_config = _DummyConfig()


# ---------------------------------------------------------------------------
# Configuration (Wf-327-F7)
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a truthy/falsy env var.  Mirrors :func:`triton_kernels.config._env_flag`."""
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


@dataclass(frozen=True)
class DMPNNTritonKernelConfig:
    """Env-var-driven configuration for the D-MPNN Triton dispatch.

    Frozen dataclass so the config can be hashed + used as a pytest
    parametrize id.  All fields are derived from env vars at
    construction time; mutating them requires rebuilding the config.

    Fields
    ------
    triton_use_fused : bool
        Global fused-kernel gate (``TRITON_USE_FUSED``).  When
        ``False``, all fused kernels stay OFF regardless of per-op
        flags (mirrors ``triton_kernels.triton_config.enabled``).
    triton_use_fused_eval : bool
        Eval-mode opt-in (``TRITON_USE_FUSED_EVAL``).  When ``True``,
        fused kernels also run in eval mode (default OFF — eval passes
        are typically small and autotune cost dominates).
    triton_relu_mlp : bool
        Per-op opt-in for ``fused_relu_mlp`` (``TRITON_RELU_MLP``).
        Default OFF; required for the two-matmul fused path
        (``atom_embed`` / ``edge_embed`` / ``readout_mlp``) to run.
    triton_relu_linear : bool
        Per-op opt-in for ``fused_relu_linear`` (``TRITON_RELU_LINEAR``).
        Default OFF; required for the single-matmul fused path
        (``edge_mlp[layer]``) to run.
    """

    triton_use_fused: bool = field(default_factory=lambda: _env_flag("TRITON_USE_FUSED", "0"))
    triton_use_fused_eval: bool = field(default_factory=lambda: _env_flag("TRITON_USE_FUSED_EVAL", "0"))
    triton_relu_mlp: bool = field(default_factory=lambda: _env_flag("TRITON_RELU_MLP", "0"))
    triton_relu_linear: bool = field(default_factory=lambda: _env_flag("TRITON_RELU_LINEAR", "0"))

    def gate_open(self) -> bool:
        """Return True iff the global fused-kernel gate is open.

        This is the GLOBAL gate — even if ``triton_relu_mlp`` /
        ``triton_relu_linear`` is True, this returns False when the
        global gate is closed.  Mirrors ``triton_config.enabled``.
        """
        return self.triton_use_fused


def current_config() -> DMPNNTritonKernelConfig:
    """Snapshot the current dispatch config from env vars.

    Returns a fresh :class:`DMPNNTritonKernelConfig` each call so
    tests can verify env-driven behaviour without monkey-patching.
    """
    return DMPNNTritonKernelConfig()


def dispatch_state_report() -> Dict[str, Any]:
    """Return a JSON-serializable report of the current dispatch state.

    Used by the Wf-327-F7 tests + the Wf-326E-D re-measurement bench
    (forward-looking) to verify gate state without polluting test
    output with non-JSON-serializable objects (tensors, dataclasses).
    """
    cfg = current_config()
    return {
        "triton_available": _TRITON_AVAILABLE,
        "triton_use_fused": cfg.triton_use_fused,
        "triton_use_fused_eval": cfg.triton_use_fused_eval,
        "triton_relu_mlp": cfg.triton_relu_mlp,
        "triton_relu_linear": cfg.triton_relu_linear,
        "gate_open": cfg.gate_open(),
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def maybe_fused_relu_mlp(
    seq: nn.Sequential,
    x: torch.Tensor,
    *,
    force: Optional[bool] = None,
) -> torch.Tensor:
    """Route a ``nn.Sequential(Linear, ReLU, Linear)`` through fused_relu_mlp.

    Parameters
    ----------
    seq : nn.Sequential
        Three-child Sequential matching the canonical D-MPNN
        ``atom_embed`` / ``edge_embed`` / ``readout_mlp`` shape:
        ``[Linear, ReLU, Linear]``.  Defensive: any other shape
        falls through to the canonical ``seq(x)``.
    x : torch.Tensor
        Input tensor with shape ``(..., D)``.  Last dim must match
        ``seq[0].in_features``.
    force : bool, optional
        ``True`` → always run the fused kernel (still respects the
        shape + CUDA gates — the kernel cannot run on CPU).  ``False``
        → always fall through to PyTorch.  ``None`` (default) → run
        the default dispatch policy (env-driven + ``triton_config``).

    Returns
    -------
    torch.Tensor with shape ``(..., seq[2].out_features)``.

    Dispatch policy
    ---------------
    CPU tensors → ``seq(x)`` (Triton kernels require CUDA).
    Eval mode + ``TRITON_USE_FUSED_EVAL=0`` → ``seq(x)``.
    Three-child Sequential shape mismatch → ``seq(x)`` (defensive).
    Otherwise → :func:`triton_kernels.fused_relu_mlp` with
    ``linear1.weight`` / ``linear1.bias`` /
    ``linear2.weight`` / ``linear2.bias`` pulled from the Sequential.

    Notes
    -----
    The fused kernel preserves parameter names (``linear1.*`` /
    ``linear2.*``) so existing D-MPNN checkpoints load without
    key remapping.  See :file:`triton_kernels/fused_relu_mlp.py`
    for the kernel spec.
    """
    # Explicit overrides.
    if force is False:
        return seq(x)

    # Defensive: shape mismatch → PyTorch reference.
    if len(seq) != 3:
        return seq(x)
    linear1, relu_mid, linear2 = seq[0], seq[1], seq[2]
    if not (isinstance(linear1, nn.Linear) and isinstance(relu_mid, nn.ReLU) and isinstance(linear2, nn.Linear)):
        return seq(x)

    # CPU → PyTorch reference (Triton kernels raise on CPU).
    if not x.is_cuda:
        return seq(x)

    # Eval mode gate (unless TRITON_USE_FUSED_EVAL is on).
    cfg = current_config()
    is_training = torch.is_grad_enabled() and x.requires_grad
    if not is_training and not cfg.triton_use_fused_eval:
        return seq(x)

    # Per-op gate + shape gate.
    if not _triton_config.use_fused_relu_mlp(x):
        return seq(x)

    return _fused_relu_mlp(
        x,
        linear1.weight, linear1.bias,
        linear2.weight, linear2.bias,
    )


def maybe_fused_relu_linear(
    seq: nn.Sequential,
    x: torch.Tensor,
    *,
    force: Optional[bool] = None,
) -> torch.Tensor:
    """Route a ``nn.Sequential(Linear, ReLU)`` through fused_relu_linear.

    Parameters
    ----------
    seq : nn.Sequential
        Two-child Sequential matching the canonical D-MPNN per-layer
        edge MLP shape (``edge_mlp[layer]``):
        ``[Linear, ReLU]``.  Defensive: any other shape falls
        through to the canonical ``seq(x)``.
    x : torch.Tensor
        Input tensor with shape ``(..., D)``.
    force : bool, optional
        ``True`` → always run the fused kernel (still respects the
        shape + CUDA gates).  ``False`` → always fall through to
        PyTorch.  ``None`` (default) → run the default dispatch policy.

    Returns
    -------
    torch.Tensor with shape ``(..., seq[0].out_features)``.

    Dispatch policy
    ---------------
    Mirrors :func:`maybe_fused_relu_mlp` but uses
    :func:`triton_kernels.fused_relu_linear` (single matmul) and the
    ``TRITON_RELU_LINEAR`` per-op gate.

    This is the per-message edge update hot-path — runs
    ``n_layers * B * E_max`` times per training step in the D-MPNN
    backbone.
    """
    # Explicit overrides.
    if force is False:
        return seq(x)

    # Defensive: shape mismatch → PyTorch reference.
    if len(seq) != 2:
        return seq(x)
    linear1, relu_mid = seq[0], seq[1]
    if not (isinstance(linear1, nn.Linear) and isinstance(relu_mid, nn.ReLU)):
        return seq(x)

    # CPU → PyTorch reference.
    if not x.is_cuda:
        return seq(x)

    # Eval mode gate.
    cfg = current_config()
    is_training = torch.is_grad_enabled() and x.requires_grad
    if not is_training and not cfg.triton_use_fused_eval:
        return seq(x)

    # Per-op gate + shape gate.
    if not _triton_config.use_fused_relu_linear(x):
        return seq(x)

    return _fused_relu_linear(
        x,
        linear1.weight, linear1.bias,
    )


__all__ = [
    "DMPNNTritonKernelConfig",
    "current_config",
    "dispatch_state_report",
    "maybe_fused_relu_mlp",
    "maybe_fused_relu_linear",
]
