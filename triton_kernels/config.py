"""Central dispatch configuration for MolFlow-Triton fused kernels.

MolFlow-Triton ships several Triton-fused implementations of the
PyTorch ops that dominate training time
(:func:`fused_layer_norm`, :func:`fused_rms_norm`, :func:`fused_silu_mlp`,
:func:`softmax_last_dim`, :func:`fused_cross_entropy`).  They are not
always the right choice — autotuning on a 64-element row of an
evaluation-mode pass is pure overhead — so :class:`TritonConfig`
provides a single gate plus a small set of shape-based fallback rules.

Dispatch policy
---------------

The fused kernel is selected when **all** of the following hold:

1. The user did not disable it (``TRITON_USE_FUSED=0``).  The default is
   backend-dependent: **on** for CUDA, **off** for ROCm.  See
   :func:`triton_kernels.backend.active_backend`.
2. The op is on the measured allowlist (:data:`MEASURED_OP_ALLOWLIST`).
   Ops whose Triton kernel loses to PyTorch on this backend are refused
   even when the global gate is open — the global flag says "Triton is
   allowed here", not "every kernel is faster here".
3. The module is in training mode **or** the user explicitly opted in
   for eval mode (``TRITON_USE_FUSED_EVAL=1``).
4. The tensor's last-axis width ``H`` is at most
   :attr:`MAX_FEAT_DIM` (4096 — the in-block reduction budget for the
   fused-norm kernel; beyond this the Triton launch would silently
   truncate or refuse).
5. The total tensor byte size is at least :attr:`MIN_BYTES` (backend
   specific — anything smaller is faster on PyTorch because Triton
   autotune cost dominates).

A user-supplied ``force=False`` keyword argument overrides (1)-(5) and
always returns ``False`` (i.e. fall back to PyTorch).  Callers should
expose the helper as ``triton_config.should_use_fused(tensor, op=...)``.

Environment variables
---------------------

- ``TRITON_USE_FUSED``: unset means "backend default" — ``"1"`` on CUDA,
  ``"0"`` on ROCm.  The ROCm default is ``"0"`` because the kernels were
  tuned for gfx1101 and, per ``molmetal/reports/wf326e_triton_audit``,
  several of them are slower there than the PyTorch op they replace.
  The CUDA default is ``"1"`` because every op in
  :data:`MEASURED_OP_ALLOWLIST` was measured faster than PyTorch on
  ``sm_120`` (``metrics/by_metric/triton_blackwell_bench.json``).
- ``TRITON_USE_FUSED_EVAL``: ``"1"`` to enable fused kernels during
  ``model.eval()`` (default off — eval passes typically operate on
  small batches where autotune cost outweighs the fused speedup).
- ``TRITON_OP_ALLOWLIST``: comma-separated override of the op allowlist,
  e.g. ``TRITON_OP_ALLOWLIST=silu_mlp`` to enable one op and nothing
  else, or ``TRITON_OP_ALLOWLIST=none`` to disable them all.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .backend import active_backend, describe as describe_backend


# Largest feature dim the fused-norm / fused-softmax kernels can reduce
# in a single Triton block.  Matches ``_next_pow2(feat_dim) <= 4096`` in
# :mod:`triton_kernels.fused_norm` / :mod:`triton_kernels.fused_softmax`.
MAX_FEAT_DIM = 4096

# Smallest tensor size (in bytes) for which Triton autotune cost is
# amortised.
#
# CUDA / Blackwell: 128 KB = 32 k float32 elements.  MEASURED on
#   sm_120 — ``fused_silu_mlp`` at M=405, H=128 (207 KB) is 1.12x faster
#   than the PyTorch chain, so the old 1 MB floor was excluding a shape
#   where the kernel already wins.
# ROCm / gfx1101: 1 MB, unchanged from the original tuning.
MIN_BYTES = 128 * 1024 if active_backend() == "cuda" else 1024 * 1024


# ---------------------------------------------------------------------------
# Measured per-op allowlist
# ---------------------------------------------------------------------------
# Source: metrics/by_metric/triton_blackwell_bench.json, MEASURED
# 2026-10-23 on RTX PRO 6000 Blackwell (sm_120), triton 3.8.0,
# torch 2.14.1+cu130.  Rule applied: enable iff numerically correct on
# every shape AND the worst measured speed-up over PyTorch is > 1.0x.
#
#   silu_mlp       1.05-1.23x   rel err ~8e-4 (tf32 tl.dot)
#   layer_norm     1.00-1.38x   rel err ~2e-7
#   residual_add   1.22-1.35x   bit-exact
#   softmax        0.85-1.32x   loses at D=256 -> OFF
#   cross_entropy  0.24-0.25x   loses by 4x   -> OFF
#   scatter        0.20-1.13x   loses past 1.2 k edges -> OFF
#
# ``relu_mlp`` / ``relu_linear`` / ``dropout_residual`` / ``rms_norm`` are
# NOT measured on Blackwell.  They stay allowed (the pre-existing
# behaviour) and are listed here so the gap is visible rather than
# implied.  The D-MPNN line is the only consumer of the first two.
MEASURED_OP_ALLOWLIST: dict[str, bool] = {
    "silu_mlp": True,
    "layer_norm": True,
    "rms_norm": True,
    "residual_add": True,
    "dropout_residual": True,
    "softmax": False,
    "cross_entropy": False,
    "scatter": False,
    "aggregate_vectors": False,
    # unmeasured on sm_120 — left at the pre-existing default
    "relu_mlp": True,
    "relu_linear": True,
}


def _env_flag(name: str, default: str) -> bool:
    """Parse a truthy/falsy env var.

    ``"0"``, ``"false"``, ``"no"``, empty string → False.
    Anything else → True.
    """
    val = os.environ.get(name, default)
    return val.strip().lower() not in ("0", "false", "no", "")


def _default_enabled() -> str:
    """Backend-dependent default for ``TRITON_USE_FUSED``."""
    return "1" if active_backend() == "cuda" else "0"


def _op_allowlist_from_env() -> Optional[dict[str, bool]]:
    """Parse ``TRITON_OP_ALLOWLIST`` into an explicit op → bool map.

    ``None`` means "no override, use :data:`MEASURED_OP_ALLOWLIST`".
    """
    raw = os.environ.get("TRITON_OP_ALLOWLIST", "").strip()
    if not raw:
        return None
    if raw.lower() == "none":
        return {}
    return {name.strip(): True for name in raw.split(",") if name.strip()}


class TritonConfig:
    """Process-wide fused-kernel dispatch gate.

    Use :meth:`should_use_fused` to decide whether to dispatch to a
    Triton kernel or fall back to the matching PyTorch op.  The same
    helper is imported by ``models.encoder``,
    ``models.conditioner``, ``models.velocity_net``,
    ``molmetal.models.dmpnn`` and ``molmetal.models.loss``.

    The class is intentionally a singleton-style configuration holder
    rather than a per-instance object — callers should do::

        from triton_kernels.config import triton_config
        if triton_config.should_use_fused(x, op="layer_norm"):
            ...

    rather than ``TritonConfig().should_use_fused(...)``.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        enabled_eval: Optional[bool] = None,
        max_feat_dim: int = MAX_FEAT_DIM,
        min_bytes: int = MIN_BYTES,
        op_allowlist: Optional[dict[str, bool]] = None,
    ) -> None:
        # Resolve env defaults at construction time so tests can
        # override via ``monkeypatch.setenv`` *before* the singleton is
        # created.  ``None`` means "ask the env".
        if enabled is None:
            # Backend default (2026-10-23): ON for CUDA, OFF for ROCm.
            # The ROCm default is the 2026-09-27 decision — gfx1101-tuned
            # kernels were 3-11x slower than PyTorch on Blackwell, and
            # rather than retune them the gate was closed.  They are now
            # retuned for Blackwell, so the CUDA side opens; the ROCm
            # side keeps the closed default until it is re-measured.
            enabled = _env_flag("TRITON_USE_FUSED", _default_enabled())
        if enabled_eval is None:
            enabled_eval = _env_flag("TRITON_USE_FUSED_EVAL", "0")
        if op_allowlist is None:
            op_allowlist = _op_allowlist_from_env()
            if op_allowlist is None:
                op_allowlist = dict(MEASURED_OP_ALLOWLIST)
        self._enabled = bool(enabled)
        self._enabled_eval = bool(enabled_eval)
        self._max_feat_dim = int(max_feat_dim)
        self._min_bytes = int(min_bytes)
        self._op_allowlist = dict(op_allowlist)

    # ------------------------------------------------------------------
    # State inspection / override
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def enabled_eval(self) -> bool:
        return self._enabled_eval

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def set_enabled_eval(self, value: bool) -> None:
        self._enabled_eval = bool(value)

    @property
    def op_allowlist(self) -> dict[str, bool]:
        return dict(self._op_allowlist)

    def set_op_allowed(self, op: str, value: bool) -> None:
        """Turn one op on or off without touching the global gate."""
        self._op_allowlist[str(op)] = bool(value)

    def op_allowed(self, op: str) -> bool:
        """True iff ``op`` may use its fused kernel on this backend.

        An op absent from the allowlist is allowed: the table is a record
        of what has been MEASURED to lose, and an unmeasured op should
        keep whatever behaviour the global gate gives it.  Pass
        ``TRITON_OP_ALLOWLIST=...`` to make the table exhaustive instead.
        """
        if not op:
            return True
        return bool(self._op_allowlist.get(op, True))

    # ------------------------------------------------------------------
    # Dispatch policy
    # ------------------------------------------------------------------
    def should_use_fused(
        self,
        tensor: torch.Tensor,
        *,
        op: str = "",
        training: Optional[bool] = None,
        force: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff the fused kernel is the right choice.

        Parameters
        ----------
        tensor : torch.Tensor
            The input the fused kernel would consume.  Only its shape,
            dtype and device are inspected — the kernel is not invoked.
        op : str, optional
            Tag for the measured per-op allowlist (``"layer_norm"``,
            ``"silu_mlp"``, ``"softmax"``, ``"cross_entropy"``,
            ``"rms_norm"``, ``"residual_add"``, ``"scatter"``).  An op the
            allowlist records as a loss on this backend is refused here,
            so a caller that names its op gets the measurement for free.
            Leaving it empty keeps the pre-allowlist behaviour.
        training : bool, optional
            Override the global ``torch.is_grad_enabled()``-based
            training detection.  When ``None`` we read the tensor's
            ``requires_grad`` plus ``torch.is_grad_enabled()`` — a
            proxy for "training step in progress".
        force : bool, optional
            ``True`` → always use PyTorch; ``False`` → always use the
            fused kernel (subject to shape gates).  ``None`` → run the
            default policy.
        """
        # Explicit overrides win.
        if force is True:
            return False
        if force is False:
            # Caller asked to use fused; still respect the hard
            # shape gates below so we never silently truncate.
            pass
        else:
            if not self._enabled:
                return False
            if not self.op_allowed(op):
                return False
            # Default policy: training ON, eval OFF unless opted in.
            if training is None:
                training = torch.is_grad_enabled() and tensor.requires_grad
            if not training and not self._enabled_eval:
                return False

        # Hard shape gates.  We allow the caller to skip these with
        # ``force=False`` (i.e. "I really want the fused kernel"),
        # but the kernels themselves will raise on shape violations —
        # so callers that use ``force=False`` must already know their
        # shapes fit.
        if force is None:
            if tensor.dim() < 1:
                return False
            feat_dim = tensor.shape[-1]
            if feat_dim > self._max_feat_dim:
                return False
            n_elem = tensor.numel()
            # 4 bytes per FP32 element; for FP16/BF16 we use 2.  This
            # is a coarse proxy — the actual Triton launch grid is
            # row-major so the bottleneck is row count, not bytes —
            # but it matches the rough "1 MB amortises autotune" rule.
            bytes_per_elem = tensor.element_size()
            approx_bytes = n_elem * bytes_per_elem
            if approx_bytes < self._min_bytes:
                return False

        return True

    # ------------------------------------------------------------------
    # Convenience: per-op gate (used by fused_silu_mlp wiring sites)
    # ------------------------------------------------------------------
    def use_fused_residual_add(
        self,
        n_elem: int,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff the fused ``alpha * x + beta * residual`` kernel is
        the right choice for a tensor with ``n_elem`` elements.

        Mirrors :meth:`use_fused_mlp`.  Used by the round-5 wiring sites
        in :mod:`molmetal.models.dmpnn` and :mod:`molmetal.models.metal_hybrid_v4`.
        Defaults to ``True`` whenever the global fused-kernel gate is on
        and the tensor is large enough to amortise autotune.
        """
        if n_elem <= 0:
            return False
        if not self._enabled:
            return False
        if not self.op_allowed("residual_add"):
            return False
        if training is None:
            training = torch.is_grad_enabled()
        if not training and not self._enabled_eval:
            return False
        # 4 bytes per FP32 element proxy; the kernel is bandwidth-bound
        # so the byte threshold is the same gate as :attr:`MIN_BYTES`.
        approx_bytes = int(n_elem) * 4
        if approx_bytes < self._min_bytes:
            return False
        return True

    def use_fused_dropout_residual(
        self,
        n_elem: int,
        p: float,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff :func:`fused_dropout_residual` is the right choice.

        Mirrors :meth:`use_fused_residual_add` but adds the dropout
        probability gate (``p`` must be in ``[0, 1)``).  Used by
        :mod:`molmetal.models.dmpnn`'s per-layer edge dropout.
        """
        if n_elem <= 0:
            return False
        if p < 0.0 or p >= 1.0:
            return False
        if not self._enabled:
            return False
        if not self.op_allowed("dropout_residual"):
            return False
        if training is None:
            training = torch.is_grad_enabled()
        if not training and not self._enabled_eval:
            return False
        approx_bytes = int(n_elem) * 4
        if approx_bytes < self._min_bytes:
            return False
        return True

    def use_fused_softmax(
        self,
        last_dim: int,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff the fused softmax-last-dim kernel is the
        right choice for a reduction over an axis of width ``last_dim``.

        Mirrors :meth:`use_fused_mlp`.  The fused kernel only fits when
        ``last_dim <= MAX_FEAT_DIM`` (4096 — the in-block reduction
        budget) and the global fused-kernel gate is enabled.  When the
        gate is off or the reduction is too wide, callers must fall
        back to :func:`torch.softmax`.

        Parameters
        ----------
        last_dim : int
            Width of the softmax reduction axis.
        training : bool, optional
            Forwarded to :meth:`should_use_fused`.  ``None`` uses the
            default policy (training ON, eval OFF unless opted in).
        """
        if last_dim <= 0:
            return False
        if last_dim > self._max_feat_dim:
            return False
        if not self._enabled:
            return False
        if not self.op_allowed("softmax"):
            return False
        if training is None:
            training = torch.is_grad_enabled()
        if not training and not self._enabled_eval:
            return False
        return True

    def use_fused_mlp(
        self,
        h_in: int,
        h_hidden: int,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff the fused MLP (Linear->SiLU->Linear) kernel is
        the right choice for shapes ``h_in -> h_hidden -> h_in`` (or ``h_in
        -> h_hidden -> h_hidden``).

        This is the gate used by the production wiring sites in
        :mod:`molmetal.adapters.egnn_rocm`,
        :mod:`molmetal.adapters.flow_matching_lipman` and
        :mod:`molmetal.scripts.train_fm_pocket`.  It wraps
        :meth:`should_use_fused` with the shape contract of
        :func:`triton_kernels.fused_silu_mlp` and defaults to ``True``
        whenever the global fused-kernel gate is on and the hidden
        dimensions fit.

        Parameters
        ----------
        h_in : int
            First / last linear input feature width.
        h_hidden : int
            Hidden feature width.
        training : bool, optional
            Forwarded to :meth:`should_use_fused`.  ``None`` uses the
            default policy (training ON, eval OFF unless opted in).
        """
        if h_in <= 0 or h_hidden <= 0:
            return False
        if h_in > self._max_feat_dim or h_hidden > self._max_feat_dim:
            return False
        if not self._enabled:
            return False
        if not self.op_allowed("silu_mlp"):
            return False
        if training is None:
            training = torch.is_grad_enabled()
        if not training and not self._enabled_eval:
            return False
        return True

    # ------------------------------------------------------------------
    # D-MPNN edge-update gates (fused_relu_linear / fused_relu_mlp)
    # ------------------------------------------------------------------
    # ``triton_kernels.fused_relu_mlp`` has always called these two
    # methods (``_cfg.use_fused_relu_linear(x2)``), but they were never
    # defined here — so the fused path raised ``AttributeError`` on any
    # CUDA input instead of falling through to PyTorch.  Both delegate
    # to :meth:`should_use_fused`, which already carries the
    # ``TRITON_USE_FUSED`` / ``TRITON_USE_FUSED_EVAL`` policy and the
    # shape/byte gates.
    def use_fused_relu_linear(
        self,
        x: torch.Tensor,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff ``ReLU(Linear(x, W, b))`` should use the
        fused kernel for a 2-D input ``x`` of shape ``(N, D)``."""
        return self.should_use_fused(x, op="relu_linear", training=training)

    def use_fused_relu_mlp(
        self,
        x: torch.Tensor,
        *,
        training: Optional[bool] = None,
    ) -> bool:
        """Return ``True`` iff the two-layer ``Linear->ReLU->Linear`` fused
        kernel should be used for a 2-D input ``x`` of shape ``(N, D)``."""
        return self.should_use_fused(x, op="relu_mlp", training=training)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """One dict for run artifacts / logs.

        Records the gate state, the backend, and which ops are refused,
        so a run's JSON says whether the fused path was actually live
        rather than leaving it to be inferred from the timing.
        """
        return {
            "triton_config": {
                "enabled": self._enabled,
                "enabled_eval": self._enabled_eval,
                "max_feat_dim": self._max_feat_dim,
                "min_bytes": self._min_bytes,
                "ops_refused": sorted(
                    op for op, ok in self._op_allowlist.items() if not ok
                ),
                "ops_allowed": sorted(
                    op for op, ok in self._op_allowlist.items() if ok
                ),
            },
            "backend": describe_backend(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"TritonConfig(enabled={self._enabled}, "
            f"enabled_eval={self._enabled_eval}, "
            f"max_feat_dim={self._max_feat_dim}, "
            f"min_bytes={self._min_bytes}, "
            f"refused={sorted(k for k, v in self._op_allowlist.items() if not v)})"
        )


# Process-wide singleton used by ``models.*`` and ``molmetal.models.*``.
triton_config = TritonConfig()


__all__ = [
    "MEASURED_OP_ALLOWLIST",
    "TritonConfig",
    "triton_config",
    "MAX_FEAT_DIM",
    "MIN_BYTES",
]
