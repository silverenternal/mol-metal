"""Shared :mod:`triton.autotune` configuration for MolFlow-Triton.

The hardware target is an AMD Radeon RX 7800 XT (``gfx1101``, RDNA3)
running ``triton-rocm==3.8.0``.  Per ``TODO/environment.md`` §4:

- ``waves_per_eu`` is a **CDNA-only** knob (MI300/MI355) and is **ignored
  on gfx1101**.  We therefore never include it in our autotune configs.
- The only legitimate tuning axes on RDNA3 are ``num_warps`` and
  ``num_stages`` (plus, for matmul, the per-tile ``BLOCK_*`` sizes).

The configs exported here are intentionally small.  gfx1101 has a single
wave64 wave per CU, so ``num_warps`` above 8 wastes a wave on every
program instance; we cap at 8.  ``num_stages`` > 4 inflates shared
memory pressure without measurable speed-up on this kernel mix.

Usage
-----
Apply to a Triton JIT function just like the upstream decorator::

    from triton_kernels.autotune import AUTOTUNE_CONFIGS, autotune

    @autotune(key=["M", "N", "K"])
    @triton.jit
    def my_kernel(...): ...

The exported :data:`AUTOTUNE_CONFIGS` is the canonical list — consumers
that need a non-default key list can pass their own ``configs`` argument
while keeping the same list::

    @triton.autotune(AUTOTUNE_CONFIGS, key=["N_ROWS", "FEAT"])
    @triton.jit
    def fused_norm(...): ...
"""

from __future__ import annotations

from typing import Sequence

import triton


# ---------------------------------------------------------------------------
# The canonical config grid for gfx1101
# ---------------------------------------------------------------------------
# Each entry is a :class:`triton.Config` mapping the triton autotuner
# knobs.  We intentionally keep the grid small (9 configs) so first-call
# autotune latency is tolerable; expanding the grid is a deliberate
# action that should be justified with a benchmark.
AUTOTUNE_CONFIGS: Sequence[triton.Config] = [
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
        Defaults to :data:`AUTOTUNE_CONFIGS`.
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


__all__ = ["AUTOTUNE_CONFIGS", "autotune"]
