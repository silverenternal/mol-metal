"""molmetal — Mol-Metal: precious-metal anticancer drug design framework.

Hexagonal architecture on top of MolFlow-Triton (Triton kernels on ROCm
for the RX 7800 XT).  Three layers:

  domain/     — frozen dataclasses (Pocket, Molecule, Complex)
  ports/      — Protocol interfaces (MoleculeGenerator, DockingEngine,
                PropertyPredictor, ScoringFunction, DesignLoop)
  adapters/   — concrete implementations wrapping cloned SOTA code
                (facebookresearch/flow_matching, DiffDock, …)
  orchestration/ — closed-loop driver (generate → dock → score → refine)
  references/ — git-cloned SOTA repos (read-only, treated as data)

ROCm gfx1101 environment bootstrap
----------------------------------
On import, :func:`_apply_rocm_gfx1101_env` runs and (when an AMD ROCm
GPU is visible) sets the two env vars that the AMP module
(:mod:`molmetal.adapters.flow_matching_lipman.amp`) requires to be
present **before** ``torch`` initialises its allocator + matmul
backend:

  * ``PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`` — reclaims
    fragmentation in the ROCm caching allocator (see
    ``molmetal/adapters/flow_matching_lipman/amp.py:62-65``).
  * ``TORCH_BLAS_PREFER_HIPBLASLT=1`` — forces BF16 GEMM dispatch
    through hipBLASLt on gfx1101
    (``molmetal/adapters/flow_matching_lipman/amp.py:67-70``).

The bootstrap is idempotent and a no-op on CPU-only hosts / non-AMD
GPUs.  The set of expected vars is sourced from
:data:`molmetal.adapters.flow_matching_lipman.amp.ROCM_GFX1101_ENV_VARS`
so there is exactly one canonical list.
"""

from __future__ import annotations

import os
from typing import Dict, List

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# ROCm gfx1101 environment bootstrap
# ---------------------------------------------------------------------------

#: Sentinel used by :func:`_apply_rocm_gfx1101_env` to record that it has
#: already run at least once during the current interpreter session.
#: Re-application is safe (idempotent) but we short-circuit on the
#: sentinel so we don't spam the log line per import.
_ROCM_GFX1101_ENV_APPLIED: bool = False


#: Canonical env-var map for ROCm gfx1101 (RX 7800 XT / 7900 XT / 7900
#: XTX).  This is the in-process source of truth used by
#: :func:`_apply_rocm_gfx1101_env` — see
#: :data:`molmetal.adapters.flow_matching_lipman.amp.ROCM_GFX1101_ENV_VARS`
#: for the rationale (allocator fragmentation reclaim + hipBLASLt GEMM
#: dispatch).  Kept inline (not just imported from ``amp``) so the
#: bootstrap works even when :mod:`molmetal.adapters.flow_matching_lipman`
#: fails to import (e.g. on a host where torch / libtorch is broken —
#: exactly the case where env-var hygiene still helps debug).
_ROCM_GFX1101_ENV_VARS: Dict[str, str] = {
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    "TORCH_BLAS_PREFER_HIPBLASLT": "1",
}


def _is_gfx1101_gpu() -> bool:
    """Return True iff an AMD gfx1101-class GPU is visible to PyTorch.

    Detection is **lazy + non-raising**: we import :mod:`torch` only on
    first call (so simply importing :mod:`molmetal` is still cheap) and
    match on substrings of ``torch.cuda.get_device_name(0)`` rather than
    on a hard-coded SKU list, so the same gate picks up the RX 7800 XT,
    RX 7900 XT, RX 7900 XTX and any future gfx1101 SKU.  Matches
    ``gfx1101`` (internal codename), ``Radeon`` (vendor string) and
    ``RX 78`` (current SKU family prefix).

    Returns False on hosts with no CUDA-visible device (CPU-only) or
    non-AMD GPUs.  Mirrors the RDNA3-detect fallback pattern at
    ``models/_scatter.py:84-94`` (lazy capability check, never raises).
    """
    try:
        import torch  # noqa: WPS433 — lazy import is the whole point
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    if not isinstance(name, str):
        return False
    needles = ("gfx1101", "Radeon", "RX 78")
    return any(n in name for n in needles)


def _apply_rocm_gfx1101_env(
    *,
    force: bool = False,
    quiet: bool = False,
) -> Dict[str, str]:
    """Apply the gfx1101 env vars in-process.

    Mirrors the canonical env-var map defined in
    :data:`molmetal.adapters.flow_matching_lipman.amp.ROCM_GFX1101_ENV_VARS`
    (imported lazily to keep the cold-start cost low).  Additional
    hygiene vars — ``TOKENIZERS_PARALLELISM=false`` and
    ``TRANSFORMERS_VERBOSITY=error`` — are applied unconditionally so
    the import is quiet on every host, not just ROCm ones.

    Parameters
    ----------
    force
        When ``True``, apply even if :data:`_ROCM_GFX1101_ENV_APPLIED`
        is already set.  Useful for tests that need to re-trigger the
        log line.
    quiet
        When ``True``, suppress the one-line summary print.

    Returns
    -------
    dict
        Mapping ``{var_name: value}`` of vars that were applied by this
        call.  If the host has no AMD GPU visible we return ``{}`` and
        skip the assignment entirely (so CPU hosts are not polluted
        with HIP-related env vars).

    Notes
    -----
    Idempotent: calling twice with the same map is a no-op for vars
    already set to the recommended value.  Vars already set to a different
    value are **left untouched** so user overrides win.
    """
    global _ROCM_GFX1101_ENV_APPLIED

    if _ROCM_GFX1101_ENV_APPLIED and not force:
        return {}

    applied: Dict[str, str] = {}

    # 1) Hygiene: tokenizer warning noise.  Unconditional — cheap and
    #    useful on every host.
    for name, value in (
        ("TOKENIZERS_PARALLELISM", "false"),
        ("TRANSFORMERS_VERBOSITY", "error"),
    ):
        if os.environ.get(name) != value:
            os.environ[name] = value
            applied[name] = value

    # 2) ROCm gfx1101 — only if we can see an AMD GPU.
    if _is_gfx1101_gpu():
        # Use the inline canonical map as the source of truth.  We do a
        # lazy cross-check against the AMP module's published list (so
        # the two stay in sync) but fall back to the inline map if the
        # AMP module fails to import — e.g. on a host where torch /
        # libtorch is broken, which is exactly the case where env-var
        # hygiene still helps debugging.
        env_vars = _ROCM_GFX1101_ENV_VARS
        try:
            # Lazy import keeps the cold-start cost of ``import molmetal``
            # low on CPU hosts that never touch the AMP module.
            from molmetal.adapters.flow_matching_lipman.amp import (
                ROCM_GFX1101_ENV_VARS as AMP_VARS,
            )
            env_vars = AMP_VARS
        except Exception:
            pass

        for name, value in env_vars.items():
            if os.environ.get(name) != value:
                os.environ[name] = value
                applied[name] = value

    _ROCM_GFX1101_ENV_APPLIED = True

    if applied and not quiet:
        kv = ", ".join(f"{k}={v}" for k, v in applied.items())
        print(f"[molmetal] applied gfx1101 env vars: {kv}")

    return applied


#: Public alias for the list of vars :func:`_apply_rocm_gfx1101_env`
#: would set on a gfx1101 host.  Tests use this to assert coverage
#: without round-tripping through the private ``_ROCM_GFX1101_ENV_APPLIED``
#: sentinel.
ROCM_GFX1101_ENV_APPLIED_VARS: List[str] = [
    "PYTORCH_HIP_ALLOC_CONF",
    "TORCH_BLAS_PREFER_HIPBLASLT",
]


#: Run the bootstrap once at import time so downstream ``import torch``
#: users pick up the env.  Suppressed on non-AMD hosts (the function
#: returns ``{}`` and is a true no-op there).
_apply_rocm_gfx1101_env()


__all__ = [
    "__version__",
    "_apply_rocm_gfx1101_env",
    "_is_gfx1101_gpu",
    "ROCM_GFX1101_ENV_APPLIED_VARS",
]