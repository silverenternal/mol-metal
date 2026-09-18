"""Regression checks for the gfx1101 autotuning surface."""

from __future__ import annotations

import triton_kernels as tk
from triton_kernels.config import TritonConfig


def test_autotune_configs_are_gfx1101_safe() -> None:
    allowed_warps = {2, 4, 8}
    allowed_stages = {2, 3, 4}
    assert tk.AUTOTUNE_CONFIGS
    for config in tk.AUTOTUNE_CONFIGS:
        assert config.num_warps in allowed_warps
        assert config.num_stages in allowed_stages


def test_autotune_fn_is_callable() -> None:
    assert callable(tk.autotune_fn)


def test_triton_config_is_module_level_dispatch_gate() -> None:
    """The package exposes one process-wide TritonConfig gate."""
    assert isinstance(tk.triton_config, TritonConfig)
    assert tk.triton_config is __import__("triton_kernels.config", fromlist=["triton_config"]).triton_config
