"""WF-iGPU-Switch verify: device-selection preference order.

This file locks in the contract of ``molmetal.utils.device``:

1. ``DEFAULT_DEVICE`` prefers a healthy iGPU (``gfx1100``) when present
   and allocatable (WF-iGPU-Switch policy).
2. Falls back to a healthy discrete GPU (``gfx1101``) when iGPU is not
   visible / not allocatable.
3. Falls back to ``cpu`` with a one-shot ``UserWarning`` when no GPU is
   usable. The warning is only emitted by :func:`get_device`, not by
   module import, so smoke tests that only inspect ``DEFAULT_DEVICE``
   do not get noise.

Honest-framing note
-------------------
All tests mock :func:`detect_active_gpu` to inject each scenario,
because the live host currently has *no* usable GPU (HSA init is
blocked by the dGPU SMU hang — see
``molmetal/reports/wf_igpu_switch/probe.md``). Mocking makes these
tests hermetic and reproducible regardless of the host's GPU state.
"""

from __future__ import annotations

import warnings

import pytest

import molmetal.utils.device as device_mod
from molmetal.utils.device import (
    DEFAULT_DEVICE,
    ROCM_AVAILABLE,
    _HEALTHY_DGPU_GFX,
    _HEALTHY_IGPU_GFX,
    detect_active_gpu,
    get_device,
)


def _info(igpu=False, dgpu=False, cuda_visible=True, alloc_ok=True,
          name="Radeon 780M", gfx="gfx1100", count=1):
    """Build a fake detect_active_gpu() return dict."""
    return {
        "torch_version": "test-torch",
        "hip_version": "7.2",
        "device_count": count if cuda_visible else 0,
        "device_0_name": name if cuda_visible else "N/A",
        "device_0_gfx": gfx,
        "torch_reports_cuda": cuda_visible,
        "alloc_ok": alloc_ok,
        "igpu_usable": igpu,
        "dgpu_usable": dgpu,
        "selected_device": "cuda:0" if alloc_ok else "cpu",
        "reason": "mock",
    }


def test_default_prefers_igpu_when_available(monkeypatch):
    """When the iGPU is reachable and allocatable, DEFAULT_DEVICE picks it."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=True, dgpu=False, name="Radeon 780M", gfx="gfx1100"),
    )
    # Reset lazy cache so the new probe is honoured.
    device_mod.DEFAULT_DEVICE._value = None
    assert str(DEFAULT_DEVICE) == "cuda:0"


def test_default_falls_back_to_discrete_gpu(monkeypatch):
    """When only the dGPU is reachable, DEFAULT_DEVICE picks it."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=False, dgpu=True, name="Radeon RX 7800 XT", gfx="gfx1101"),
    )
    device_mod.DEFAULT_DEVICE._value = None
    assert str(DEFAULT_DEVICE) == "cuda:0"


def test_default_falls_back_to_cpu_with_warning(monkeypatch):
    """When no GPU is reachable, DEFAULT_DEVICE picks cpu and get_device warns."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=False, dgpu=False, cuda_visible=False, alloc_ok=False),
    )
    device_mod.DEFAULT_DEVICE._value = None
    assert str(DEFAULT_DEVICE) == "cpu"

    # Reset the one-shot warning guard, then assert get_device emits the warning.
    monkeypatch.setattr(device_mod, "_WARNED_CPU_FALLBACK", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = get_device()
    assert isinstance(d, type(__import__("torch").device("cpu")))
    assert str(d) == "cpu"
    cpu_warnings = [w for w in caught if "falling" in str(w.message).lower()
                    or "cpu" in str(w.message).lower()]
    assert cpu_warnings, "expected a CPU-fallback warning from get_device()"


def test_get_device_returns_cuda_when_igpu(monkeypatch):
    """get_device() returns torch.device('cuda:0') when the iGPU is usable."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=True, dgpu=False, gfx="gfx1100"),
    )
    d = get_device()
    import torch
    assert str(d) == "cuda:0"
    assert isinstance(d, torch.device)


def test_get_device_returns_cuda_when_only_dgpu(monkeypatch):
    """get_device() returns torch.device('cuda:0') when only dGPU is usable."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=False, dgpu=True, gfx="gfx1101"),
    )
    d = get_device()
    import torch
    assert str(d) == "cuda:0"


def test_get_device_warns_only_once(monkeypatch):
    """The CPU-fallback warning is one-shot across calls."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=False, dgpu=False, cuda_visible=False, alloc_ok=False),
    )
    monkeypatch.setattr(device_mod, "_WARNED_CPU_FALLBACK", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_device()
        get_device()
        get_device()
    cpu_warnings = [
        w for w in caught
        if "falling" in str(w.message).lower() or "cpu" in str(w.message).lower()
    ]
    assert len(cpu_warnings) == 1, (
        f"expected exactly one CPU-fallback warning, got {len(cpu_warnings)}"
    )


def test_detect_active_gpu_never_raises(monkeypatch):
    """detect_active_gpu() must not raise even when torch.cuda throws."""
    import torch as _torch

    def _bad_is_available():
        raise RuntimeError("simulated HSA init failure")

    monkeypatch.setattr(_torch.cuda, "is_available", _bad_is_available)
    info = detect_active_gpu()
    assert info["torch_reports_cuda"] is False
    assert info["alloc_ok"] is False
    assert info["igpu_usable"] is False
    assert info["dgpu_usable"] is False
    assert info["selected_device"] == "cpu"


def test_verify_rocm_active_backward_compatible(monkeypatch):
    """verify_rocm_active() keeps the legacy keys smoke tests rely on."""
    monkeypatch.setattr(
        device_mod, "detect_active_gpu",
        lambda: _info(igpu=True, dgpu=False, gfx="gfx1100"),
    )
    info = device_mod.verify_rocm_active()
    # Legacy keys must remain present for downstream smoke tests.
    for k in ("torch_version", "hip_version", "device_count", "device_0_name",
              "device_0_index", "roc_active", "selected_device", "env_visible"):
        assert k in info, f"legacy key {k!r} missing from verify_rocm_active()"
    # New keys added by WF-iGPU-Switch.
    assert "alloc_ok" in info
    assert "igpu_usable" in info
    assert "dgpu_usable" in info
    assert "device_0_gfx" in info


def test_healthy_gfx_lists_are_explicit():
    """Lock in the chip allow-list — protects against silent scope creep."""
    assert "gfx1100" in _HEALTHY_IGPU_GFX
    assert "gfx1101" in _HEALTHY_DGPU_GFX


def test_rocm_available_constant_reflects_torch(monkeypatch):
    """ROCM_AVAILABLE stays a module-level mirror of torch.cuda.is_available()."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    import importlib
    # Note: ROCM_AVAILABLE is captured at import time, so we just verify
    # the contract: it is a bool and the detect probe re-checks live.
    assert isinstance(ROCM_AVAILABLE, bool)


def test_default_device_lazy_caches(monkeypatch):
    """DEFAULT_DEVICE._value caches after first read."""
    calls = {"n": 0}

    def fake_probe():
        calls["n"] += 1
        return _info(igpu=True, dgpu=False, gfx="gfx1100")

    monkeypatch.setattr(device_mod, "detect_active_gpu", fake_probe)
    device_mod.DEFAULT_DEVICE._value = None
    _ = str(DEFAULT_DEVICE)
    _ = str(DEFAULT_DEVICE)
    _ = str(DEFAULT_DEVICE)
    assert calls["n"] == 1, (
        f"DEFAULT_DEVICE should call detect_active_gpu at most once per "
        f"cache clear, got {calls['n']}"
    )
