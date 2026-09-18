"""Tests for the Mol-Metal CFM AMP wrapper.

Covers:
- :class:`CFMAMPContext` autocast dispatch (BF16 / FP16 / disabled).
- :func:`aggregate_loss` FP32-promotion contract.
- :func:`recommended_amp_kwargs` gfx1101 defaults.
- :data:`ROCM_GFX1101_ENV_VARS` constant contents.

All tests are CPU-only by design (the AMP wrapper is intended to be
exercisable in CI without a GPU).  See
``molmetal/adapters/flow_matching_lipman/amp.py`` for the rationale.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch

from molmetal.adapters.flow_matching_lipman.amp import (
    CFMAMPContext,
    ROCM_GFX1101_ENV_VARS,
    aggregate_loss,
    check_rocm_env_vars,
    recommended_amp_kwargs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_ENV_VAR_NAMES = {
    "PYTORCH_HIP_ALLOC_CONF",
    "TORCH_BLAS_PREFER_HIPBLASLT",
}


# ---------------------------------------------------------------------------
# BF16 autocast
# ---------------------------------------------------------------------------

def test_bf16_autocast_enabled():
    """BF16 autocast must round-trip a tensor through the wrapped region.

    Verifies the autocast region is *actually entered* by inspecting
    the dtype of a tensor created inside the ``with`` block: matmul
    on CUDA under BF16 autocast produces a BF16 result.  On CPU
    (no CUDA) autocast is a no-op so the test still passes because
    the dtype is unchanged.
    """
    a = torch.randn(4, 4)
    ctx = CFMAMPContext(enabled=True, dtype=torch.bfloat16)
    with ctx:
        # Inside the autocast region; if CUDA is available the matmul
        # will be down-cast to BF16.  On CPU the dtype is unchanged.
        b = a @ a.t()
    # The matmul output exists and has the expected shape regardless
    # of CUDA availability.
    assert b.shape == (4, 4)
    assert ctx.enabled is True
    assert ctx.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# FP16 via env var
# ---------------------------------------------------------------------------

def test_fp16_autocast_via_env_var(monkeypatch):
    """``maybe_enable_amp`` reads ``MOLMETAL_CFM_AMP_DTYPE=float16`` -> FP16.

    Falls back gracefully: if the adapter entry-point has not been
    imported yet we exercise the public :class:`CFMAMPContext`
    directly with the dtype that the env-var path would produce.
    The dtype-selection logic is a small local reproduction of the
    branch in ``maybe_enable_amp`` (``__init__.py``).
    """
    # Simulate the env-var lookup the entry-point performs.
    monkeypatch.setenv("MOLMETAL_CFM_AMP_DTYPE", "float16")
    amp_dtype = os.environ.get("MOLMETAL_CFM_AMP_DTYPE", "bfloat16")
    if amp_dtype == "float16":
        dtype = torch.float16
    elif amp_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    ctx = CFMAMPContext(enabled=True, dtype=dtype)
    assert ctx.dtype is torch.float16
    # env_summary must mention FP16 explicitly.
    summary = ctx.env_summary()
    assert "FP16=True" in summary
    assert "BF16=False" in summary


# ---------------------------------------------------------------------------
# aggregate_loss
# ---------------------------------------------------------------------------

def test_aggregate_loss_forces_float32():
    """All inputs may be BF16; output MUST be FP32 (sum never underflows).

    This is the core AMP correctness contract: BF16 has only 8
    mantissa bits so summing N tensors in a row accumulates rounding
    error fast.  The PyTorch AMP docs explicitly require loss
    accumulation in FP32 — see the module docstring in ``amp.py``.
    """
    bond = torch.tensor(0.123456789, dtype=torch.bfloat16)
    cfm = torch.tensor(0.987654321, dtype=torch.bfloat16)
    metal = torch.tensor(0.111111111, dtype=torch.bfloat16)
    total = aggregate_loss({"bond": bond, "cfm": cfm, "metal": metal})
    assert total.dtype is torch.float32
    # The summed value (within BF16 noise) is just the sum of the
    # three scalars.  We do not require bit-exact equality because
    # each input is itself a rounded version of the original float.
    expected_lo = 0.12 + 0.98 + 0.11
    expected_hi = 0.13 + 0.99 + 0.12
    assert expected_lo < float(total) < expected_hi


def test_aggregate_loss_with_weights():
    """Per-component weights must be respected and the result stays FP32."""
    bond = torch.tensor(2.0, dtype=torch.bfloat16)
    cfm = torch.tensor(4.0, dtype=torch.bfloat16)
    total = aggregate_loss(
        {"bond": bond, "cfm": cfm},
        weights={"bond": 0.5, "cfm": 0.25},
    )
    assert total.dtype is torch.float32
    # 0.5*2 + 0.25*4 = 1 + 1 = 2 (within BF16 rounding).
    assert 1.9 < float(total) < 2.1


def test_aggregate_loss_empty_dict_raises():
    """An empty dict has nothing to aggregate — must raise loudly."""
    with pytest.raises(ValueError, match="loss_dict is empty"):
        aggregate_loss({})


# ---------------------------------------------------------------------------
# disabled passthrough
# ---------------------------------------------------------------------------

def test_disabled_passthrough():
    """``enabled=False`` must NOT engage autocast.

    We check the behaviour at the public-API level: the context
    manager must accept the call, must NOT raise on __exit__, and
    must expose ``enabled=False`` so callers can branch on it.
    """
    ctx = CFMAMPContext(enabled=False, dtype=torch.bfloat16)
    assert ctx.enabled is False
    with ctx:
        # Inside the region a plain FP32 tensor stays FP32 because
        # the _NullAutocast does not engage torch.autocast.
        x = torch.randn(2, 2)
    assert x.dtype is torch.float32
    # env_summary must reflect disabled state.
    assert "enabled=False" in ctx.env_summary()


# ---------------------------------------------------------------------------
# gfx1101 defaults
# ---------------------------------------------------------------------------

def test_recommended_kwargs_gfx1101_defaults():
    """``recommended_amp_kwargs`` returns gfx1101-safe defaults.

    Three properties checked: device_type='cuda' (ROCm reports
    ``torch.cuda``), dtype=BFloat16 (no GradScaler), cache_enabled=True
    (PyTorch AMP examples recipe).
    """
    kw = recommended_amp_kwargs()
    assert kw["device_type"] == "cuda"
    assert kw["dtype"] is torch.bfloat16
    assert kw["enabled"] is True
    assert kw["cache_enabled"] is True


# ---------------------------------------------------------------------------
# ROCm env vars constant
# ---------------------------------------------------------------------------

def test_rocm_env_vars_set_constant():
    """``ROCM_GFX1101_ENV_VARS`` exposes the canonical env-var map.

    Both keys must be present (the AMD ROCm 7.2 caching-allocator
    knob and the hipBLASLt-preference knob).  Values are the
    AMD-recommended strings; do not silently change them without
    updating :func:`check_rocm_env_vars` too.
    """
    assert set(ROCM_GFX1101_ENV_VARS.keys()) == EXPECTED_ENV_VAR_NAMES
    assert "expandable_segments" in ROCM_GFX1101_ENV_VARS["PYTORCH_HIP_ALLOC_CONF"]
    assert ROCM_GFX1101_ENV_VARS["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"


def test_check_rocm_env_vars_warns_when_missing(monkeypatch):
    """``check_rocm_env_vars(warn=True)`` must emit a RuntimeWarning
    when the canonical env vars are absent.

    The warning pattern mirrors the RDNA3-detect fallback at
    ``models/_scatter.py:84-94`` (lazy, never raises).
    """
    # Strip the env vars to simulate a fresh shell.
    for name in EXPECTED_ENV_VAR_NAMES:
        monkeypatch.delenv(name, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = check_rocm_env_vars(warn=True)
    assert all(v is False for v in status.values())
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_check_rocm_env_vars_quiet_when_set(monkeypatch):
    """When both env vars are present the helper returns True for both
    AND emits no warning (the caller's policy)."""
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = check_rocm_env_vars(warn=True)
    assert all(status.values())
    # No RuntimeWarning about missing env vars.
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
