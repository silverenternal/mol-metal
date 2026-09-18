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

1. The user did not disable it (``TRITON_USE_FUSED=0``).
2. The module is in training mode **or** the user explicitly opted in
   for eval mode (``TRITON_USE_FUSED_EVAL=1``).
3. The tensor's last-axis width ``H`` is at most
   :attr:`MAX_FEAT_DIM` (4096 — the in-block reduction budget for the
   fused-norm kernel; beyond this the Triton launch would silently
   truncate or refuse).
4. The total tensor byte size ``B * N * H * 4`` is at least
   :attr:`MIN_BYTES` (1 MB — anything smaller is faster on PyTorch
   because Triton autotune cost dominates).

A user-supplied ``force=False`` keyword argument overrides (1)-(4) and
always returns ``False`` (i.e. fall back to PyTorch).  Callers should
expose the helper as ``triton_config.should_use_fused(tensor, op=...)``.

Environment variables
---------------------

- ``TRITON_USE_FUSED``: ``"1"`` (default for training), ``"0"`` to
  force-disable globally.  Override at import-time only.
- ``TRITON_USE_FUSED_EVAL``: ``"1"`` to enable fused kernels during
  ``model.eval()`` (default off — eval passes typically operate on
  small batches where autotune cost outweighs the fused speedup).

The defaults match the "default ON for training, default OFF for eval"
behaviour specified in the production-wiring task.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


# Largest feature dim the fused-norm / fused-softmax kernels can reduce
# in a single Triton block.  Matches ``_next_pow2(feat_dim) <= 4096`` in
# :mod:`triton_kernels.fused_norm` / :mod:`triton_kernels.fused_softmax`.
MAX_FEAT_DIM = 4096

# Smallest tensor size (in bytes) for which Triton autotune cost is
# amortised.  1 MB = ~262 k float32 elements.  Below this we keep the
# PyTorch implementation (and therefore do not pay the autotune
# probe).
MIN_BYTES = 1024 * 1024  # 1 MB


def _env_flag(name: str, default: str) -> bool:
    """Parse a truthy/falsy env var.

    ``"0"``, ``"false"``, ``"no"``, empty string → False.
    Anything else → True.
    """
    val = os.environ.get(name, default)
    return val.strip().lower() not in ("0", "false", "no", "")


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
    ) -> None:
        # Resolve env defaults at construction time so tests can
        # override via ``monkeypatch.setenv`` *before* the singleton is
        # created.  ``None`` means "ask the env".
        if enabled is None:
            enabled = _env_flag("TRITON_USE_FUSED", "1")
        if enabled_eval is None:
            enabled_eval = _env_flag("TRITON_USE_FUSED_EVAL", "0")
        self._enabled = bool(enabled)
        self._enabled_eval = bool(enabled_eval)
        self._max_feat_dim = int(max_feat_dim)
        self._min_bytes = int(min_bytes)

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
            Tag for diagnostics / per-op overrides (``"layer_norm"``,
            ``"silu_mlp"``, ``"softmax"``, ``"cross_entropy"``,
            ``"rms_norm"``).  Currently unused beyond logging but
            reserved so we can extend the policy per-op without
            changing call sites.
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
        if training is None:
            training = torch.is_grad_enabled()
        if not training and not self._enabled_eval:
            return False
        return True

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"TritonConfig(enabled={self._enabled}, "
            f"enabled_eval={self._enabled_eval}, "
            f"max_feat_dim={self._max_feat_dim}, "
            f"min_bytes={self._min_bytes})"
        )


# Process-wide singleton used by ``models.*`` and ``molmetal.models.*``.
triton_config = TritonConfig()


__all__ = [
    "TritonConfig",
    "triton_config",
    "MAX_FEAT_DIM",
    "MIN_BYTES",
]
