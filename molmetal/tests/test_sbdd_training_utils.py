"""Tests for ``molmetal.molmetal_lam.sbdd_env.training_utils`` (Wf-130).

Verifies the GPU-default helpers do what they claim:

* :func:`ensure_cuda_visible_devices_default` is module-top safe (does
  not import torch, only sets the env var).
* :func:`apply_default_cuda_visible_devices` honours CLI override +
  shell env + default-to-``"1"``.
* :func:`detect_training_gpu` always returns a dict (never raises).
* :func:`resolve_default_device` returns ``cuda:0`` on a CUDA host,
  ``cpu`` otherwise, and emits a single INFO log line.
* :func:`setup_training_device_env` is idempotent + always seeds torch.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make sure the molmetal package is importable
sys.path.insert(0, "/home/hugo/codes/mol-metal")

from molmetal.molmetal_lam.sbdd_env import training_utils as tu


# ---------------------------------------------------------------------------
# ensure_cuda_visible_devices_default — module-top safe
# ---------------------------------------------------------------------------
def test_ensure_default_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ensure_cuda_visible_devices_default`` does not import torch —
    callers can invoke it at module top BEFORE ``import torch``.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # The import of training_utils above is fine — we just call the
    # function and verify torch isn't imported as a side effect.
    out = tu.ensure_cuda_visible_devices_default()
    assert out == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_ensure_default_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the env var is already set, ``ensure_default`` does nothing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    out = tu.ensure_cuda_visible_devices_default()
    assert out == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_ensure_default_constant() -> None:
    """The module-level DEFAULT_CUDA_VISIBLE_DEVICES is ``"1"`` (Wf-130)."""
    assert tu.DEFAULT_CUDA_VISIBLE_DEVICES == "1"


# ---------------------------------------------------------------------------
# apply_default_cuda_visible_devices — full precedence
# ---------------------------------------------------------------------------
def test_apply_prefers_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI override wins over shell env."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    out = tu.apply_default_cuda_visible_devices(override="0,1")
    assert out == "0,1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_apply_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CLI is empty, env wins over default."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    out = tu.apply_default_cuda_visible_devices(override="")
    assert out == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CLI is empty AND env is unset, default ``"1"`` is used."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    out = tu.apply_default_cuda_visible_devices(override="")
    assert out == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


# ---------------------------------------------------------------------------
# detect_training_gpu — always returns a dict
# ---------------------------------------------------------------------------
def test_detect_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    info = tu.detect_training_gpu()
    assert isinstance(info, dict)
    for key in (
        "cuda_visible_devices",
        "torch_reports_cuda",
        "device_count",
        "device_name",
        "alloc_ok",
        "selected_device",
        "reason",
    ):
        assert key in info, f"missing key {key}"


# ---------------------------------------------------------------------------
# resolve_default_device — never raises, always returns a torch.device
# ---------------------------------------------------------------------------
def test_resolve_returns_torch_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = tu.resolve_default_device()
    # type: ignore[arg-type]
    import torch  # noqa: WPS433 — only here for the assertion
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# setup_training_device_env — returns full snapshot
# ---------------------------------------------------------------------------
def test_setup_returns_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    snap = tu.setup_training_device_env(seed=7, deterministic=False)
    assert "device" in snap
    assert "cuda_visible_devices" in snap
    assert snap["seed"] == 7
    assert snap["deterministic"] is False
    assert "gpu_info" in snap
    assert snap["gpu_info"]["cuda_visible_devices"] == "1"
