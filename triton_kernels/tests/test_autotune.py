"""Regression checks for the per-backend autotuning surface."""

from __future__ import annotations

import triton_kernels as tk
from triton_kernels import autotune as tk_autotune
from triton_kernels.backend import active_backend
from triton_kernels.config import TritonConfig


def test_autotune_configs_are_safe_for_the_active_backend() -> None:
    """Each backend gets the knob range it was tuned for.

    The original assertion pinned ``{2, 4, 8}`` warps, which is the
    gfx1101 (single wave64 per CU) range.  Blackwell's grid reaches
    ``num_warps=16`` and ``num_stages=4``, so the check has to name the
    backend it is checking rather than assume the ROCm one.
    """
    if active_backend() == "cuda":
        # {4, 8, 16} for the compute-bound band plus the num_warps=2
        # column the memory-bound kernels use.
        allowed_warps = {2, 4, 8, 16}
        allowed_stages = {2, 3, 4}
        grid = tk.AUTOTUNE_CONFIGS_CUDA
    else:
        allowed_warps = {2, 4, 8}
        allowed_stages = {2, 3, 4}
        grid = tk.AUTOTUNE_CONFIGS_ROCM
    assert grid
    for config in grid:
        assert config.num_warps in allowed_warps
        assert config.num_stages in allowed_stages


def test_autotune_configs_alias_tracks_the_backend() -> None:
    """``AUTOTUNE_CONFIGS`` must be the active backend's grid, not a constant.

    Every kernel module imports this one name, so if it stopped tracking
    the backend the CUDA path would silently run the gfx1101 grid again —
    which is the exact defect (3-11x slower than PyTorch on sm_120) that
    the 2026-10-23 migration fixed.
    """
    expected = (
        tk_autotune.AUTOTUNE_CONFIGS_CUDA
        if active_backend() == "cuda"
        else tk_autotune.AUTOTUNE_CONFIGS_ROCM
    )
    assert len(tk.AUTOTUNE_CONFIGS) == len(expected)
    assert [
        (c.num_warps, c.num_stages) for c in tk.AUTOTUNE_CONFIGS
    ] == [(c.num_warps, c.num_stages) for c in expected]


def test_autotune_fn_is_callable() -> None:
    assert callable(tk.autotune_fn)


def test_triton_config_is_module_level_dispatch_gate() -> None:
    """The package exposes one process-wide TritonConfig gate."""
    assert isinstance(tk.triton_config, TritonConfig)
    assert (
        tk.triton_config
        is __import__("triton_kernels.config", fromlist=["triton_config"]).triton_config
    )
