"""Shared :mod:`triton.autotune` configuration for MolFlow-Triton.

Two tuning grids live here, selected at import time by
:func:`triton_kernels.backend.active_backend`:

``AUTOTUNE_CONFIGS_ROCM``
    The original gfx1101 (RDNA3) grid, written for ``triton-rocm==3.8.0``.
    Per ``TODO/environment.md`` §4, ``waves_per_eu`` is CDNA-only and is
    ignored on gfx1101, so the only legitimate axes are ``num_warps`` and
    ``num_stages``.  gfx1101 has a single wave64 per CU, so ``num_warps``
    above 8 wastes a wave on every program instance; capped at 8.

``AUTOTUNE_CONFIGS_CUDA``
    The Blackwell (``sm_100``/``sm_120``) grid.  A Blackwell SM holds 4
    warp schedulers over 128 FP32 lanes, so 4 warps already fill one
    block and 8/16 are the shapes that let the compiler software-pipeline
    the K loop.  ``num_stages`` above 4 is reachable there because
    Blackwell ships ~228 KB of shared memory per SM; the gfx1101 grid
    stopped at 4 only because RDNA3's LDS budget was smaller.

``AUTOTUNE_CONFIGS``
    Alias for whichever grid the current backend uses.  Every kernel
    module imports this name, so a kernel file does not need to branch on
    the backend to get a sane grid.

Why the split is not cosmetic
-----------------------------
The gfx1101 grid was never retuned for Blackwell.  Per
``molmetal/reports/wf326e_triton_audit/audit.json`` the 24 kernels that
use it ran 3-11x slower than PyTorch on ``sm_120``, which is why
``TRITON_USE_FUSED`` defaulted to ``"0"`` on 2026-09-27.  Wf-331-X3
scaffolded an ``sm_120`` preset but never wired it into this module, so
the kernels kept consuming the ROCm grid.  This file is the wire-in.

Usage
-----
Apply to a Triton JIT function just like the upstream decorator::

    from triton_kernels.autotune import AUTOTUNE_CONFIGS, autotune

    @autotune(key=["M", "N", "K"])
    @triton.jit
    def my_kernel(...): ...

Consumers that need a non-default key list keep the same grid::

    @triton.autotune(AUTOTUNE_CONFIGS, key=["N_ROWS", "FEAT"])
    @triton.jit
    def fused_norm(...): ...
"""

from __future__ import annotations

from typing import Sequence

import triton

from .backend import active_backend


# ---------------------------------------------------------------------------
# gfx1101 / RDNA3 grid (the pre-2026-10 ROCm target)
# ---------------------------------------------------------------------------
# 3 warps x 3 stages = 9 configs.  Kept small so first-call autotune
# latency stays tolerable; expanding the grid is a deliberate action that
# should be justified with a benchmark.
AUTOTUNE_CONFIGS_ROCM: Sequence[triton.Config] = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
]


# ---------------------------------------------------------------------------
# Blackwell grid (sm_100 / sm_120)
# ---------------------------------------------------------------------------
# Mirrors the Wf-326E follow-up-A band that Wf-331-X3 encoded as
# ``SM120_AUTOTUNE_CONFIGS``: num_warps {4, 8, 16} x num_stages {2, 3, 4},
# plus the num_warps=2 column for the memory-bound kernels (residual add,
# dropout-residual) where a wide block only adds scheduling overhead.
# 12 configs total.
AUTOTUNE_CONFIGS_CUDA: Sequence[triton.Config] = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in (4, 8, 16)
    for s in (2, 3, 4)
] + [
    triton.Config({}, num_warps=2, num_stages=s) for s in (2, 3, 4)
]


#: Grid for the backend this process is running on.
AUTOTUNE_CONFIGS: Sequence[triton.Config] = (
    AUTOTUNE_CONFIGS_CUDA if active_backend() == "cuda" else AUTOTUNE_CONFIGS_ROCM
)


#: Smaller grid for eval-mode passes, where autotune latency is not
#: amortised over enough launches.  Four points spread across the band
#: rather than a corner of it, so a narrow-shape kernel still finds a
#: usable config.
AUTOTUNE_CONFIGS_FAST: Sequence[triton.Config] = (
    [
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
    ]
    if active_backend() == "cuda"
    else [
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ]
)


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------
def autotune(
    configs: Sequence[triton.Config] | None = None,
    *,
    key: Sequence[str],
    **kwargs,
):
    """Thin wrapper around :func:`triton.autotune` with our default grid.

    Parameters
    ----------
    configs : sequence of :class:`triton.Config`, optional
        Defaults to :data:`AUTOTUNE_CONFIGS` (backend-selected).
    key : sequence of str
        Required positional-style argument forwarded to
        :func:`triton.autotune`.  Choosing the right key is the most
        important decision a caller makes: keys that are too coarse
        re-tune every launch; keys that are too fine defeat the cache.
    **kwargs
        Forwarded verbatim to :func:`triton.autotune` (e.g.
        ``prune_configs_by``, ``reset_to_zero``).
    """
    if configs is None:
        configs = AUTOTUNE_CONFIGS
    if not key:
        raise ValueError(
            "`key` must be a non-empty sequence of argument names; got an "
            "empty sequence."
        )
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)


__all__ = [
    "AUTOTUNE_CONFIGS",
    "AUTOTUNE_CONFIGS_CUDA",
    "AUTOTUNE_CONFIGS_FAST",
    "AUTOTUNE_CONFIGS_ROCM",
    "autotune",
]
