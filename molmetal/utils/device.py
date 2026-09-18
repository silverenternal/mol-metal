"""ROCm-first device helpers for molmetal adapters.

On AMD hardware with the ROCm build of PyTorch the CUDA API is
exposed via ``torch.cuda`` — so ``torch.cuda.is_available()`` is the
canonical signal that an AMD GPU is actually visible to torch.

Design rules
------------
1. Adapters must NEVER hard-code ``"cpu"`` as the default device.
   They should call :func:`get_device` which auto-detects ROCm/CUDA
   and falls back to CPU with a one-shot warning.
2. Before any training/inference, call :func:`verify_rocm_active` once
   and store the result — the dict is used by smoke tests to assert
   the GPU is *actually* being used (not just that tensors move
   without error).
3. Use :func:`device_guard` to (re-)move a whole ``nn.Module`` to the
   resolved device at setup time.

iGPU preference policy (WF-iGPU-Switch, 2026-09-14)
---------------------------------------------------
The Radeon 780M (gfx1100) is the healthy integrated GPU; the discrete
RX 7800 XT (gfx1101) is dead with an SMU firmware hang. We *prefer*
the iGPU when it is the only healthy device, but we *never* silently
flip :data:`DEFAULT_DEVICE` based on heuristics — the change is gated
on a real PyTorch ``torch.cuda`` allocation probe in
:func:`detect_active_gpu`. If HSA init fails (which it currently does
on this host because the dGPU pollutes the KFD topology), the probe
returns ``None`` and :data:`DEFAULT_DEVICE` degrades to ``cpu`` with a
warning. This matches the honest-framing mandate from
``molmetal/reports/wf_igpu_switch/probe.md``: do not pretend a GPU is
available when it isn't.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: True iff a ROCm/CUDA device is visible to torch at import time.
#: ROCm PyTorch exposes the CUDA API (``torch.cuda.*``), so this is the
#: canonical detection mechanism — same call works for NVIDIA and AMD.
ROCM_AVAILABLE: bool = torch.cuda.is_available()

#: Tuple of gfx names we currently treat as "healthy integrated Radeon"
#: on the WF-iGPU-Switch policy. Extend cautiously — each new chip must
#: be smoke-tested with a real tensor allocation before being trusted.
_HEALTHY_IGPU_GFX: Tuple[str, ...] = ("gfx1100",)

#: Tuple of gfx names we currently treat as "healthy discrete Radeon".
#: ``gfx1101`` is the RX 7800 XT — it IS in the supported arch list for
#: PyTorch 2.14+rocm7.2 but its SMU firmware is hung on this host, so
#: detection is best-effort only (see ``detect_active_gpu``).
_HEALTHY_DGPU_GFX: Tuple[str, ...] = ("gfx1101",)

#: Module-level flag — warn once, not every time ``get_device`` is called.
_WARNED_CPU_FALLBACK: bool = False


def _resolve_default_device() -> str:
    """Internal helper: pick a default device string.

    Order:
      1. Healthy iGPU visible AND allocatable (preferred — WF-iGPU-Switch)
      2. Healthy discrete GPU visible AND allocatable (legacy default)
      3. ``cpu`` (fallback, with a one-shot warning at first use)

    The probe is wrapped in a try/except so any HSA init error (which
    currently happens on this host because the dGPU pollutes KFD
    topology enumeration) degrades to step (3) without raising.
    """
    info = detect_active_gpu()
    if info["igpu_usable"]:
        return "cuda:0"
    if info["dgpu_usable"]:
        return "cuda:0"
    return "cpu"


# Lazy descriptor for :data:`DEFAULT_DEVICE` so it can call
# :func:`detect_active_gpu` (defined below) without a forward-reference
# NameError. Module-level ``DEFAULT_DEVICE`` is computed on first read
# and cached for the rest of the process.
class _LazyDefaultDevice:
    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: Optional[str] = None

    def __str__(self) -> str:  # type: ignore[override]
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == other

    @property
    def value(self) -> str:
        if self._value is None:
            self._value = _resolve_default_device()
        return self._value


#: Default device string used by every adapter when none is given.
#: Lazy — first read calls :func:`_resolve_default_device`. The
#: preference order (iGPU -> dGPU -> cpu) is observable but never
#: silently wrong.
DEFAULT_DEVICE: Any = _LazyDefaultDevice()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_active_gpu() -> Dict[str, Any]:
    """Honest probe: which GPU is actually usable to PyTorch right now?

    Runs a real 1-element float-tensor allocation on the device PyTorch
    reports (``torch.cuda``), then inspects the device name to decide
    whether it is the iGPU (``gfx1100``), a healthy discrete GPU
    (``gfx1101`` or future), or something else (NVIDIA, AMD CDNA, …).
    The result is the single source of truth for :data:`DEFAULT_DEVICE`
    and for :func:`get_device`.

    Keys
    ----
    torch_version    : str                 — torch.__version__
    hip_version      : str | None          — torch.version.hip
    device_count     : int                 — torch.cuda.device_count()
    device_0_name    : str                 — torch.cuda.get_device_name(0) or 'N/A'
    device_0_gfx     : str                 — best-effort gfx version, e.g. 'gfx1101' or 'unknown'
    torch_reports_cuda : bool              — torch.cuda.is_available()
    alloc_ok         : bool                — 1-elem tensor allocated on cuda:0 without error
    igpu_usable      : bool                — alloc_ok AND gfx in ``_HEALTHY_IGPU_GFX``
    dgpu_usable      : bool                — alloc_ok AND gfx in ``_HEALTHY_DGPU_GFX``
    selected_device  : str                 — 'cuda:0' or 'cpu'
    reason           : str                 — short human-readable verdict

    The function never raises; allocation failures are captured in
    ``alloc_ok`` and reflected in the verdict.
    """
    hip_ver: Optional[str] = None
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        hip_ver = str(torch.version.hip)
    try:
        cuda_visible = bool(torch.cuda.is_available())
    except Exception:
        cuda_visible = False
    try:
        device_count = int(torch.cuda.device_count()) if cuda_visible else 0
    except Exception:
        device_count = 0
        cuda_visible = False
    try:
        device_name: str = (
            torch.cuda.get_device_name(0) if cuda_visible else "N/A"
        )
    except Exception:
        device_name = "N/A"
        cuda_visible = False

    # Best-effort gfx detection from device name (e.g. "Radeon 780M" → gfx1100,
    # "Radeon RX 7800 XT" → gfx1101). Conservative: only set if name matches
    # a known healthy chip.
    device_gfx = "unknown"
    lname = device_name.lower()
    if cuda_visible and "780m" in lname:
        device_gfx = "gfx1100"
    elif cuda_visible and "7800 xt" in lname:
        device_gfx = "gfx1101"

    alloc_ok = False
    if cuda_visible:
        try:
            _t = torch.zeros(1, device="cuda:0")
            del _t
            alloc_ok = True
        except Exception:
            alloc_ok = False

    igpu_usable = bool(alloc_ok and device_gfx in _HEALTHY_IGPU_GFX)
    dgpu_usable = bool(alloc_ok and device_gfx in _HEALTHY_DGPU_GFX)

    if igpu_usable:
        reason = "healthy iGPU (gfx1100) reachable"
    elif dgpu_usable:
        reason = "healthy discrete GPU (gfx1101) reachable"
    elif cuda_visible and not alloc_ok:
        reason = "torch sees a GPU but allocation failed (HSA init broken?)"
    elif not cuda_visible:
        reason = "no ROCm/CUDA device visible to torch"
    else:
        reason = f"unknown GPU {device_name!r} (gfx={device_gfx})"

    return {
        "torch_version": torch.__version__,
        "hip_version": hip_ver,
        "device_count": device_count,
        "device_0_name": device_name,
        "device_0_gfx": device_gfx,
        "torch_reports_cuda": cuda_visible,
        "alloc_ok": alloc_ok,
        "igpu_usable": igpu_usable,
        "dgpu_usable": dgpu_usable,
        "selected_device": "cuda:0" if alloc_ok else "cpu",
        "reason": reason,
    }


def get_device() -> torch.device:
    """Return the canonical torch device (ROCm-first).

    Always returns ``torch.device('cuda:0')`` when :func:`detect_active_gpu`
    confirms a usable GPU (iGPU preferred, then dGPU); otherwise ``cpu``
    and emits a one-shot ``UserWarning``.

    Use this in adapter ``setup()`` methods instead of hard-coding a
    device string.
    """
    global _WARNED_CPU_FALLBACK
    info = detect_active_gpu()
    if info["igpu_usable"] or info["dgpu_usable"]:
        return torch.device("cuda:0")
    if not _WARNED_CPU_FALLBACK:
        warnings.warn(
            "molmetal.utils.device: " + info["reason"] + " — falling "
            "back to CPU.  This will be very slow for training.  "
            "Check `python -c 'from molmetal.utils.device import "
            "detect_active_gpu, ROCM_AVAILABLE, DEFAULT_DEVICE; "
            "print(ROCM_AVAILABLE, DEFAULT_DEVICE); print(detect_active_gpu())'`.",
            UserWarning,
            stacklevel=2,
        )
        _WARNED_CPU_FALLBACK = True
    return torch.device("cpu")


def verify_rocm_active() -> Dict[str, Any]:
    """Return a dict describing whether ROCm is *actually* active.

    The returned dict is what adapters should store on ``self.device_info``
    so that smoke tests can assert ``device_info['roc_active'] is True``.

    This is now a thin wrapper over :func:`detect_active_gpu` plus the
    legacy ``env_visible`` key for backward compatibility with existing
    smoke tests. Adapters are encouraged to call
    :func:`detect_active_gpu` directly for richer diagnostics.

    Keys
    ----
    torch_version    : str        — torch.__version__
    hip_version      : str | None — torch.version.hip (None for CPU/NVIDIA-only builds)
    device_count     : int        — torch.cuda.device_count()
    device_0_name    : str        — torch.cuda.get_device_name(0) or 'N/A'
    device_0_index   : int | None — 0 if any GPU visible
    device_0_gfx     : str        — best-effort gfx version, e.g. 'gfx1100'/'gfx1101'/'unknown'
    roc_active       : bool       — True iff torch.cuda + HIP compile + GPU present
    alloc_ok         : bool       — True iff a 1-elem tensor allocates on cuda:0
    igpu_usable      : bool       — True iff alloc_ok AND device is a healthy iGPU (gfx1100)
    dgpu_usable      : bool       — True iff alloc_ok AND device is a healthy dGPU (gfx1101)
    selected_device  : str        — 'cuda:0' or 'cpu'
    reason           : str        — short human-readable verdict
    env_visible      : bool       — HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES set
    """
    info = detect_active_gpu()

    env_visible = any(
        k in os.environ for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    )

    return {
        "torch_version": info["torch_version"],
        "hip_version": info["hip_version"],
        "device_count": info["device_count"],
        "device_0_name": info["device_0_name"],
        "device_0_index": 0 if info["torch_reports_cuda"] else None,
        "device_0_gfx": info["device_0_gfx"],
        "roc_active": info["alloc_ok"],
        "alloc_ok": info["alloc_ok"],
        "igpu_usable": info["igpu_usable"],
        "dgpu_usable": info["dgpu_usable"],
        "selected_device": info["selected_device"],
        "reason": info["reason"],
        "env_visible": env_visible,
    }


@contextmanager
def device_guard(module: torch.nn.Module, device: Optional[torch.device] = None):
    """Context manager that moves ``module`` to ``device`` (default
    :func:`get_device`) on entry, and back to its original device on
    exit.

    Usage::

        with device_guard(model) as m:
            x = torch.randn(8, 3).to(m.weight.device)
            y = m(x)
    """
    original_device = next(module.parameters()).device
    target = device if device is not None else get_device()
    try:
        module.to(target)
        yield module
    finally:
        try:
            module.to(original_device)
        except Exception:
            # If the original device is no longer visible (e.g. multi-GPU
            # tests), silently leave the module on the new device.
            pass
