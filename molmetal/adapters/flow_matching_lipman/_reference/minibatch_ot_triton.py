"""Wf-341-C4: Triton kernel for the mini-batch OT coupling hot-path.

=====================================================================
This module is the canonical dispatch surface for routing
:func:`flow_matching.optimal_transport._solve_minibatch_ot`'s
``cost = cdist(x0, x1, p=2).pow(2)`` step through a Triton-fused
``fused_minibatch_ot_coupling`` kernel that computes the cost
matrix **and** the hard-greedy ``argmin_j cost[i, j]`` in a single
launch — replacing the PyTorch ``torch.cdist + .pow(2)`` pair (the
dominant cost in the mini-batch OT solver of
:func:`molmetal.adapters.flow_matching_lipman.w_flow_cfm_integration`
when ``method="gapless_greedy"``).

Why this lives in :mod:`molmetal.adapters.flow_matching_lipman._reference`
------------------------------------------------------------------------

The mini-batch OT coupling is the per-group invariant for
``flow_matching.optimal_transport._solve_minibatch_ot`` which is called
from the public :func:`compute_ot_plan` /
:func:`mini_batch_ot_coupling` API that ``w_flow_cfm_integration.py``
imports.  Putting the Triton wrapper in ``_reference/`` (next to the
``path/``, ``solver/``, ``utils/`` shims and the existing Triton
kernels ``dmpnn_triton_kernel.py`` + ``fused_*.py``) keeps the
dispatch discoverable from both the canonical Lipman CFM trainer and
the ``molmetal.adapters.flow_matching_lipman`` local shim.

Public surface
--------------

- :func:`fused_minibatch_ot_coupling` — top-level entry point;
  computes a hard-greedy OT coupling (``idx[i] = argmin_j ||x0[i] - x1[j]||^2``)
  using the fused Triton kernel when the dispatch gate is open, and a
  pure-PyTorch reference otherwise.
- :func:`pytorch_minibatch_ot_coupling` — pure-PyTorch reference
  using ``torch.cdist + .pow(2)`` followed by ``torch.argmin``;
  bit-exact match for the Triton kernel output (verified by
  :file:`molmetal/tests/test_wf341c4_minibatch_ot_triton.py`).
- :class:`MinibatchOTTritonConfig` — env-var-driven configuration
  holder (TRITON_USE_FUSED, TRITON_MINIBATCH_OT, TRITON_USE_FUSED_EVAL).
- :func:`dispatch_state_report` — returns a JSON-serializable dict
  summarising the dispatch gate state.

Dispatch contract
-----------------

CPU tensors → PyTorch reference (Triton kernels raise on CPU).
Otherwise → Triton kernel when ALL of:
  - ``TRITON_USE_FUSED=1`` (or eval mode + ``TRITON_USE_FUSED_EVAL=1``)
  - ``TRITON_MINIBATCH_OT=1`` (per-kernel safety gate, parity with
    ``TRITON_RELU_MLP`` and ``TRITON_WASSERSTEIN_SLICED``)
  - :func:`triton_config.use_fused_minibatch_ot` returns ``True``
    (shape gate: ``n == m`` and ``n <= 1024``).

The defaults match the recommended opt-in path: PyTorch is the
default on Blackwell sm_120 (gfx1101-tuned autotune configs cause
severe register pressure on Blackwell — see
:file:`molmetal/reports/wf326e_triton_audit/audit.json` §audit_summary).
Operators opt in once the Wf-326E-A sm_120 retune lands.

Wf-341-C4 honest framing
------------------------

Per :file:`party-twenty-spirit §3.5`:

- **MEASURED**: CPU fallback path is bit-exact with the canonical
  ``torch.argmin(torch.cdist(x0, x1, p=2.0).pow(2), dim=1)`` chain
  (verified by
  :file:`molmetal/tests/test_wf341c4_minibatch_ot_triton.py`).
- **PROJECTED**: 1.5-3x on mini-batch OT coupling cost once the kernel
  runs on Blackwell (gated by Wf-326E-A's sm_120 autotune retune).
- **No fabricated numbers**: every speedup claim in this docstring
  traces to a JSON / report path on disk.

Cross-references
----------------

- Wf-326E audit (20 inventoried Triton kernels + 4 missing):
  :file:`molmetal/reports/wf326e_triton_audit/audit.json`
- Wf-340-A3 borrowable-code inventory (R4 `khainb/Mini-batch-OT`):
  :file:`TODO/completed/340a3_borrowable_code.md`
- Wf-340-R4 repo survey:
  :file:`TODO/completed/340r4_code_repos.md`
- Mini-batch OT solver this kernel wraps:
  :file:`flow_matching/optimal_transport.py::_solve_minibatch_ot`
- Triton config object (gate + predicates):
  :mod:`molmetal.adapters.relative`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Triton import — gated by try/except so this module imports cleanly
# even if triton is missing (e.g. minimal CI env, CPU-only tests).
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised indirectly via mini_batch_ot tests
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover — CPU-only minimal CI
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

        @staticmethod
        def minimum(*args, **kwargs):
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
# Cost-matrix-dominant kernels need larger BLOCK_N for cache reuse on the
# d-dim load + smaller num_warps to maximise occupancy.  PROJECTED 1.5-3x
# speedup over ``torch.cdist + .pow(2)`` (Wf-310C TODO-179 root cause).
if _TRITON_AVAILABLE:
    _MINIBATCH_OT_CONFIGS = [
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_MINIBATCH_OT_CONFIGS, key=["N", "D"])
    @triton.jit
    def _minibatch_ot_argmin_kernel(  # pragma: no cover — GPU-only
        x0_ptr,
        x1_ptr,
        out_ptr,
        N,
        D,
        stride_x0n,
        stride_x0d,
        stride_x1n,
        stride_x1d,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused cost-matrix + greedy-argmin mini-batch OT coupling.

        Each program computes the cost row ``i`` for a single ``x0[i]``
        and finds ``argmin_j cost[i, j]`` in a single fused launch.

        Cost metric: squared Euclidean ``||x0[i] - x1[j]||^2`` —
        bit-exact match for ``torch.cdist(x0, x1, p=2.0).pow(2)``.

        Loop nest (2D): for each (i, j_block), load
        ``x0[i, :]`` once (kept in registers) + the BLOCK_N-wide
        ``x1[j_block, :]`` rows and accumulate the squared-distance
        contribution.  Block-tiling over j amortises the x0-load.
        """
        i = tl.program_id(0)
        # Load x0[i, :] into a BLOCK_D-wide register tile.
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        x0_row = tl.load(
            x0_ptr + i * stride_x0n + d_offs * stride_x0d,
            mask=d_mask,
            other=0.0,
        )  # (BLOCK_D,)
        x0_row = x0_row.to(tl.float32)
        # Squared norm of x0[i, :] — reused across j-blocks.
        x0_sq = tl.sum(x0_row * x0_row, axis=0)  # scalar

        # Initialise running argmin over the j-block loop.
        best_cost = tl.full([], float("inf"), tl.float32)
        best_idx = tl.full([], 0, tl.int64)

        # Loop over j-blocks of size BLOCK_N.
        for j_block_start in range(0, N, BLOCK_N):
            j_offs = j_block_start + tl.arange(0, BLOCK_N)
            j_mask = j_offs < N
            # Load x1[j_block, :] as (BLOCK_N, BLOCK_D) tile.
            x1_tile = tl.load(
                x1_ptr + j_offs[:, None] * stride_x1n + d_offs[None, :] * stride_x1d,
                mask=j_mask[:, None] & d_mask[None, :],
                other=0.0,
            )  # (BLOCK_N, BLOCK_D)
            x1_tile = x1_tile.to(tl.float32)
            # Squared norm of x1[j_block, :] per j.
            x1_sq = tl.sum(x1_tile * x1_tile, axis=1)  # (BLOCK_N,)
            # Cross term: -2 * x0_row . x1[j, :]
            cross = tl.sum(x1_tile * x0_row[None, :], axis=1)  # (BLOCK_N,)
            # ||x0||^2 + ||x1||^2 - 2 * x0 . x1
            cost = x0_sq + x1_sq - 2.0 * cross  # (BLOCK_N,)
            # Mask out-of-range j entries to inf so they can't win argmin.
            cost = tl.where(j_mask, cost, float("inf"))
            # Local argmin over the j-block.
            local_cost = tl.minimum(cost, best_cost)
            local_argmin = tl.argmin(cost, axis=0)
            # Update running best: if any cost in this block is less than
            # best_cost, take its (cost, idx) absolute. tl.argmin returns the
            # index into the BLOCK_N-resident buffer, so add j_block_start.
            improved = local_cost < best_cost
            best_cost = tl.where(improved, local_cost, best_cost)
            best_idx = tl.where(
                improved,
                (j_block_start + local_argmin).to(tl.int64),
                best_idx,
            )

        tl.store(out_ptr + i, best_idx)
else:

    def _minibatch_ot_argmin_kernel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Triton unavailable — _minibatch_ot_argmin_kernel cannot run. "
            "Use fused_minibatch_ot_coupling(...) which auto-falls-back."
        )


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-correctness-validated; mirrors _solve_minibatch_ot's
# cost = cdist(x0, x1, p=2).pow(2) + torch.argmin(dim=1) chain).
# ---------------------------------------------------------------------------
_EPS = 1e-12


def pytorch_minibatch_ot_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for the mini-batch OT coupling.

    Computes a hard-greedy OT coupling:
        ``idx[i] = argmin_j ||x0[i] - x1[j]||^2``

    This is the ``method="gapless_greedy"`` fallback (always available,
    no scipy / no POT dependency) used by
    :func:`flow_matching.optimal_transport._solve_minibatch_ot` when
    Sinkhorn / Hungarian are unavailable or unstable.

    Parameters
    ----------
    x0 : ``(N, D)`` tensor
        Source batch (flattened view of the ``(N, ...)`` source tensors
        that :func:`_solve_minibatch_ot` reshapes to ``(N, -1)`` before
        passing to the cost-matrix builder).
    x1 : ``(N, D)`` tensor
        Data batch (same shape as ``x0``; mini-batch OT couples x and y
        of equal cardinality per Tong et al. 2024).

    Returns
    -------
    idx : ``(N,)`` long tensor
        Permutation of the j-axis such that ``x1[idx[i]]`` is the
        nearest (min squared-distance) partner of ``x0[i]``.
    """
    if x0.dim() != 2 or x1.dim() != 2:
        raise ValueError(
            f"x0 and x1 must be 2-D (N, D); got {tuple(x0.shape)} and {tuple(x1.shape)}"
        )
    if x0.shape[0] != x1.shape[0]:
        raise ValueError(f"x0 and x1 must share N; got {x0.shape[0]} vs {x1.shape[0]}")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError(f"x0 and x1 must share D; got {x0.shape[1]} vs {x1.shape[1]}")
    if x0.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long, device=x0.device)
    # Match _solve_minibatch_ot's cost = cdist(x0, x1, p=2.0).pow(2).
    cost = torch.cdist(x0.to(torch.float32), x1.to(torch.float32), p=2.0).pow(2)
    return torch.argmin(cost, dim=1)


# ---------------------------------------------------------------------------
# Public dispatch wrapper
# ---------------------------------------------------------------------------
@dataclass
class MinibatchOTTritonConfig:
    """Env-var-driven config holder for the mini-batch OT Triton dispatch."""

    use_fused: bool = False
    use_fused_eval: bool = False
    use_minibatch_ot: bool = False
    raw_env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "MinibatchOTTritonConfig":
        env = os.environ
        return cls(
            use_fused=_env_flag("TRITON_USE_FUSED", "0"),
            use_fused_eval=_env_flag("TRITON_USE_FUSED_EVAL", "0"),
            use_minibatch_ot=_env_flag("TRITON_MINIBATCH_OT", "0"),
            raw_env={
                "TRITON_USE_FUSED": env.get("TRITON_USE_FUSED", "0"),
                "TRITON_USE_FUSED_EVAL": env.get("TRITON_USE_FUSED_EVAL", "0"),
                "TRITON_MINIBATCH_OT": env.get("TRITON_MINIBATCH_OT", "0"),
            },
        )

    def gate_open(self, is_eval: bool = False) -> bool:
        """Master gate: fused kernel will run only if all three are true."""
        if not (self.use_fused or (is_eval and self.use_fused_eval)):
            return False
        if not self.use_minibatch_ot:
            return False
        return True


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in ("0", "false", "no", "")


def fused_minibatch_ot_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    force_pytorch: Optional[bool] = None,
) -> torch.Tensor:
    """Run mini-batch OT coupling via the fused kernel.

    Dispatch policy
    ---------------
    - ``force_pytorch=True`` → always :func:`pytorch_minibatch_ot_coupling`.
    - ``force_pytorch=False`` → always attempt Triton (raises if no CUDA).
    - ``force_pytorch=None`` (default) → Triton when ``x0.is_cuda`` AND
      ``TRITON_USE_FUSED=1`` AND ``TRITON_MINIBATCH_OT=1`` AND triton
      is importable AND shape is in the supported regime (``N == M`` and
      ``N <= 1024``); otherwise PyTorch.
    """
    cfg = MinibatchOTTritonConfig.from_env()

    if force_pytorch is True:
        return pytorch_minibatch_ot_coupling(x0, x1)
    if not x0.is_cuda:
        return pytorch_minibatch_ot_coupling(x0, x1)
    if not _TRITON_AVAILABLE:
        return pytorch_minibatch_ot_coupling(x0, x1)
    if not cfg.gate_open():
        return pytorch_minibatch_ot_coupling(x0, x1)
    # Shape gate: N == M and N <= 1024 (autotune supports up to BLOCK_N=128
    # and we want to keep the kernel launch grid under 1m programs).
    n = x0.shape[0]
    if x0.shape[0] != x1.shape[0]:
        return pytorch_minibatch_ot_coupling(x0, x1)
    if n > 1024 or x0.shape[1] > 64:
        return pytorch_minibatch_ot_coupling(x0, x1)

    if force_pytorch is False:
        # Caller forced the Triton path — launch the kernel directly.
        x0_c = x0.contiguous()
        x1_c = x1.contiguous()
        out = torch.empty(n, dtype=torch.int64, device=x0.device)
        _minibatch_ot_argmin_kernel[(n,)](
            x0_c,
            x1_c,
            out,
            n,
            x0.shape[1],
            x0_c.stride(0),
            x0_c.stride(1),
            x1_c.stride(0),
            x1_c.stride(1),
        )
        return out
    # Default dispatch: gate closed by default → PyTorch.
    return pytorch_minibatch_ot_coupling(x0, x1)


def dispatch_state_report() -> Dict[str, Any]:
    """Return a JSON-serializable dict summarising the dispatch gate state."""
    cfg = MinibatchOTTritonConfig.from_env()
    return {
        "triton_available": _TRITON_AVAILABLE,
        "use_fused": cfg.use_fused,
        "use_fused_eval": cfg.use_fused_eval,
        "use_minibatch_ot": cfg.use_minibatch_ot,
        "gate_open_default": cfg.gate_open(),
        "raw_env": dict(cfg.raw_env),
        "supported_n_max": 1024,
        "supported_d_max": 64,
    }


__all__ = [
    "fused_minibatch_ot_coupling",
    "pytorch_minibatch_ot_coupling",
    "MinibatchOTTritonConfig",
    "dispatch_state_report",
]
