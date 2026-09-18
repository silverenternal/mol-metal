"""Shared utilities for molmetal adapters.

Currently exposes the ROCm-first device helpers (``get_device``,
``verify_rocm_active``, ``device_guard``).  Re-export them at the
package level so callers can ``from molmetal.utils import get_device``.
"""

from molmetal.utils.device import (
    DEFAULT_DEVICE,
    ROCM_AVAILABLE,
    device_guard,
    get_device,
    verify_rocm_active,
)

__all__ = [
    "ROCM_AVAILABLE",
    "DEFAULT_DEVICE",
    "get_device",
    "verify_rocm_active",
    "device_guard",
]
