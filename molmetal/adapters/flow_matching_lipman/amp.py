"""Mol-Metal CFM Automatic Mixed Precision (AMP) wrapper for ROCm 7.2 + gfx1101.

================================================================
Why this module exists
================================================================
The CFM training loop at :mod:`molmetal.adapters.flow_matching_lipman`
(defaults: ``hidden_dim=128``, ``n_layers=3``, pocket conditioning ON)
pushes the RX 7800 XT 16 GB past its comfort zone — the 5000-step
diagnostic in ``molmetal/reports/wf_gpu_recovery_now`` showed the
adapter trains but burns ~14 GB of HBM at ``batch_size=4`` (see also
``TODO/pending/24_cfm_architecture_redo_plan.md`` for the full audit).

Mixed precision via :func:`torch.autocast` halves the activation
footprint (FP32 -> BF16) and is **the cheapest single lever** to
claw back ~30-40 %% of VRAM.  The wrinkle on ROCm gfx1101 is that:

* ``torch.cuda.amp.GradScaler`` is **not** needed for BF16 (BF16 has
  the same exponent range as FP32 — no overflow / underflow problem
  to scale around).  FP16 *does* need a GradScaler, but the project
  default below is BF16.
* Loss values MUST be promoted to FP32 before summing.  BF16 has
  only 8 mantissa bits so summing N BF16 tensors in a row underflows
  very quickly.  :func:`aggregate_loss` is the single point that
  enforces this.
* ``PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`` (and
  ``TORCH_BLAS_PREFER_HIPBLASLT=1``) must be set **before** ``torch``
  is imported for them to take effect.  :data:`ROCM_GFX1101_ENV_VARS`
  is the canonical set; :func:`check_rocm_env_vars` warns loudly if
  they are missing.

Reference: PyTorch docs, *Automatic Mixed Precision examples*,
https://pytorch.org/docs/stable/amp.html (BF16 autocast section,
2024-10).

Public API
----------
- :class:`CFMAMPContext`         — context-manager wrapper around
                                   :func:`torch.autocast`.
- :func:`aggregate_loss`         — force-loss-to-FP32 aggregator.
- :func:`recommended_amp_kwargs` — gfx1101-safe defaults.
- :func:`check_rocm_env_vars`    — warn if ROCm env vars missing.
- :data:`ROCM_GFX1101_ENV_VARS`  — canonical env-var map.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Recommended ROCm gfx1101 (RX 7800 XT / 7900 XT / 7900 XTX) environment
#: variables that must be set BEFORE ``torch`` is imported.
#:
#: ``PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`` lets the ROCm
#: caching allocator reuse freed blocks instead of holding them in a
#: fixed-size pool — this is the single biggest VRAM reclaim lever on
#: gfx1101 according to the AMD ROCm 7.2 release notes.
#:
#: ``TORCH_BLAS_PREFER_HIPBLASLT=1`` forces PyTorch's matmul dispatch
#: through hipBLASLt, which on gfx1101 is consistently 15-25 %% faster
#: than the legacy rocBLAS path for the (small, batched) GEMM shapes
#: the CFM velocity net uses.
#:
#: These are emit-at-import-time values.  :func:`check_rocm_env_vars`
#: warns when they are missing.
ROCM_GFX1101_ENV_VARS: Dict[str, str] = {
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    "TORCH_BLAS_PREFER_HIPBLASLT": "1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recommended_amp_kwargs() -> Dict[str, Any]:
    """Return gfx1101-safe :func:`torch.autocast` kwargs.

    The defaults match the values used in the smoke-retrain script
    :mod:`molmetal.scripts.metallo_drug_smoke_retrain` (BF16 + cache
    enabled, no GradScaler).  ``cache_enabled=True`` is critical: it
    lets the autocast state cache the per-op dtype decision across
    iterations of the training loop, which the documented
    *Automatic Mixed Precision examples* recipe explicitly recommends
    for stable, fast convergence.
    """
    return {
        "device_type": "cuda",
        "dtype": torch.bfloat16,
        "enabled": True,
        "cache_enabled": True,
    }


def check_rocm_env_vars(warn: bool = True) -> Dict[str, bool]:
    """Return (and optionally warn) about missing gfx1101 env vars.

    Mimics the RDNA3-detect fallback pattern at
    ``models/_scatter.py:84-94`` (lazy capability check, never raises).

    Returns
    -------
    dict
        Mapping ``{var_name: is_set}``.  ``is_set`` is True iff the
        var is present in :data:`os.environ` (we do not compare values
        because some users legitimately override ``expandable_segments``
        with a different policy, e.g. ``max_split_size_mb``).
    """
    status = {
        name: name in os.environ for name in ROCM_GFX1101_ENV_VARS
    }
    if warn:
        missing = [n for n, ok in status.items() if not ok]
        if missing:
            warnings.warn(
                "Missing ROCm gfx1101 env vars (set BEFORE importing torch): "
                + ", ".join(f"{n}={ROCM_GFX1101_ENV_VARS[n]}" for n in missing),
                RuntimeWarning,
                stacklevel=2,
            )
    return status


def aggregate_loss(
    loss_dict: Mapping[str, torch.Tensor],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """Sum a dict of loss tensors, forcing every value to FP32 first.

    Why FP32 here?
    -------------
    BF16 has 8 mantissa bits so summing N tensors in a row accumulates
    rounding error fast.  The canonical PyTorch AMP example
    (https://pytorch.org/docs/stable/amp.html#autocasting) explicitly
    states: "If you need to compute the gradient, do so on a
    FP32-accumulated loss."  Even though the per-tensor outputs from
    BF16 autocast are *upcasted* to FP32 when used with ``F.cross_entropy``
    (see ``wf_cfm_p0_fixes`` F3 in
    ``molmetal/reports/wf_cfm_p0_fixes``), the project convention is to
    defensively upcast here too — it costs nothing and removes a whole
    class of "loss plateau at a spurious floor" bugs.

    Parameters
    ----------
    loss_dict
        ``{name: tensor}`` mapping.  Every tensor MUST be a 0-D scalar
        (use ``.mean()`` or ``.sum()`` upstream).
    weights, optional
        ``{name: scalar}`` mapping of per-component weights.  Defaults
        to uniform 1.0 weights.
    """
    if weights is None:
        weights = {}
    total = None
    for name, value in loss_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"loss_dict[{name!r}] must be a torch.Tensor; "
                f"got {type(value).__name__}"
            )
        v = value.to(dtype=torch.float32)
        w = float(weights.get(name, 1.0))
        contrib = v * w
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("loss_dict is empty; nothing to aggregate")
    return total


# ---------------------------------------------------------------------------
# CFMAMPContext
# ---------------------------------------------------------------------------

class CFMAMPContext:
    """Context-manager wrapper around :func:`torch.autocast`.

    Designed for the CFM training loop:

    .. code-block:: python

        amp_ctx = CFMAMPContext(enabled=True, dtype=torch.bfloat16)
        with amp_ctx:
            loss = aggregate_loss({
                "bond": bond_loss,
                "cfm":  cfm_loss,
                "metal": metal_loss,
            })
        loss.backward()
        optimizer.step()

    Parameters
    ----------
    device
        Torch device the autocast region targets.  ``"cuda"`` and
        ``"cuda:0"`` both work (passed straight through to
        :func:`torch.autocast`).
    dtype
        Autocast dtype.  Defaults to :data:`torch.bfloat16` (no
        GradScaler needed).  Pass :data:`torch.float16` for the
        Nvidia-only path; not recommended on gfx1101.
    enabled
        When ``False``, :meth:`__enter__` returns a no-op context
        manager so the training loop does not have to branch on the
        ``MOLMETAL_CFM_AMP`` env var.
    cache_enabled
        Forwarded to :func:`torch.autocast` so the per-op dtype
        decisions are cached across training iterations.

    Notes
    -----
    The context manager itself is intentionally thin: it does **not**
    own a :class:`torch.cuda.amp.GradScaler` because BF16 autocast does
    not need one.  If you switch to FP16 you must wrap the optimizer
    step yourself (see the canonical PyTorch AMP examples recipe).
    """

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._inner: Optional[torch.autocast] = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "CFMAMPContext":
        if not self.enabled:
            # Hand back a dummy object whose __exit__ is a no-op.
            self._inner = _NullAutocast()
            self._inner.__enter__()
            return self
        self._inner = torch.autocast(
            device_type="cuda",
            dtype=self.dtype,
            enabled=True,
            cache_enabled=self.cache_enabled,
        )
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._inner is not None, (
            "CFMAMPContext.__exit__ called without matching __enter__"
        )
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            self._inner = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def env_summary(self) -> str:
        """One-line human summary of the AMP + ROCm env state.

        Used by :func:`molmetal.scripts.metallo_drug_smoke_retrain` at
        startup to log which autocast path is active.
        """
        bf16 = self.dtype is torch.bfloat16
        fp16 = self.dtype is torch.float16
        fp32 = self.dtype is torch.float32
        tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
        rocm_status = check_rocm_env_vars(warn=False)
        rocm_ok = all(rocm_status.values())
        return (
            f"AMP: enabled={self.enabled} dtype={self.dtype} "
            f"(BF16={bf16} FP16={fp16} FP32={fp32}) | "
            f"cache_enabled={self.cache_enabled} | "
            f"TF32(matmul)={tf32} | "
            f"gfx1101_env_ok={rocm_ok} "
            f"(set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True, "
            f"TORCH_BLAS_PREFER_HIPBLASLT=1 BEFORE importing torch)"
        )


class _NullAutocast:
    """No-op autocast replacement used when ``CFMAMPContext(enabled=False)``.

    Mirrors the :class:`contextlib.nullcontext` API enough to satisfy
    the context-manager protocol; the inner ``torch.autocast`` is
    skipped so the wrapped block runs in its native dtype.
    """

    def __enter__(self) -> "_NullAutocast":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


__all__ = [
    "CFMAMPContext",
    "aggregate_loss",
    "check_rocm_env_vars",
    "recommended_amp_kwargs",
    "ROCM_GFX1101_ENV_VARS",
]
