"""Backend / architecture detection for MolFlow-Triton.

The kernel library was written for an AMD Radeon RX 7800 XT (``gfx1101``,
RDNA3) under ``triton-rocm``.  It now runs on both that target and CUDA
Blackwell (``sm_120``: RTX 5090, RTX PRO 6000 Blackwell).  The two
backends do not share a tuning grid, and at least one kernel that is
correct on CUDA produces corrupted tiles on gfx1101 — so the library has
to know which one it is on rather than assume.

Everything here is computed once at import time.  A process has exactly
one accelerator backend, so there is nothing to invalidate.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch


#: Backend identifiers returned by :func:`active_backend`.
BACKENDS = ("cuda", "rocm", "cpu")


@functools.lru_cache(maxsize=1)
def active_backend() -> str:
    """Return ``"cuda"``, ``"rocm"`` or ``"cpu"``.

    ROCm PyTorch reports ``torch.version.hip`` and exposes its devices
    through ``torch.cuda`` (``torch.cuda.is_available()`` is True on
    HIP), so the HIP check must come first — otherwise every ROCm box
    is misclassified as CUDA and handed the Blackwell tuning grid.
    """
    if getattr(torch.version, "hip", None):
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=1)
def device_capability() -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` for device 0, or ``None`` off CUDA.

    On ROCm this returns ``None``: HIP reports a capability through
    ``torch.cuda.get_device_capability`` too, but it is a GCN arch
    number (``(11, 0)`` for gfx1101), not an SM version, and comparing
    it against ``sm_100``/``sm_120`` would be meaningless.
    """
    if active_backend() != "cuda":
        return None
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001 - no device, driver mismatch, ...
        return None


@functools.lru_cache(maxsize=1)
def is_blackwell() -> bool:
    """True on ``sm_100`` and newer (Blackwell datacentre and workstation)."""
    cap = device_capability()
    return cap is not None and cap[0] >= 10


@functools.lru_cache(maxsize=1)
def device_name() -> str:
    """Human-readable name of device 0, or ``"cpu"``."""
    if active_backend() == "cpu":
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "unknown"


@functools.lru_cache(maxsize=1)
def dot_precision() -> str:
    """Precision for ``tl.dot`` in the fused kernels.

    ``"tf32"`` (default on CUDA) is what every Blackwell GEMM does by
    default and is what the measured speed-ups in
    ``metrics/by_metric/triton_blackwell_bench.json`` were taken at.
    ``"ieee"`` restores full FP32 accumulation at roughly the cost of
    the PyTorch path, which defeats the point but is available for
    bit-exactness checks.  ``TRITON_DOT_PRECISION`` overrides.
    """
    val = os.environ.get("TRITON_DOT_PRECISION", "").strip().lower()
    if val in ("ieee", "tf32", "tf32x3"):
        return val
    return "tf32" if active_backend() == "cuda" else "ieee"


def describe() -> dict:
    """One dict for run artifacts / logs."""
    cap = device_capability()
    return {
        "backend": active_backend(),
        "device_name": device_name(),
        "capability": list(cap) if cap else None,
        "is_blackwell": is_blackwell(),
        "dot_precision": dot_precision(),
    }


__all__ = [
    "BACKENDS",
    "active_backend",
    "device_capability",
    "device_name",
    "describe",
    "dot_precision",
    "is_blackwell",
]
