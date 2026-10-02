# -*- coding: utf-8 -*-
"""Tests for precision_policy.py (mirror omegafold/pipeline.py _set_precision)."""
import pytest

from molmetal.scripts.precision_policy import (
    set_precision,
    get_current_precision,
    reset_precision,
    DEFAULT_POLICY,
)


class TestPrecisionPolicy:
    """2+ tests for precision_policy."""

    def test_set_precision_apply_and_readback(self):
        """set_precision applies settings, get_current_precision reads them back."""
        policy = set_precision(allow_tf32=True, cudnn_benchmark=False)
        assert policy["allow_tf32"] is True
        assert policy["cudnn_benchmark"] is False
        # Readback (allow_tf32 + cudnn settings; float32_matmul_precision not read
        # because mixing legacy matmul.allow_tf32 with new set_float32_matmul_precision
        # triggers RuntimeError on readback per PyTorch matmul precision guidance)
        current = get_current_precision()
        assert current["allow_tf32"] is True
        assert current["cudnn_benchmark"] is False
        # Reset
        reset_precision()
        current = get_current_precision()
        assert current["allow_tf32"] is False
        print(f"test_set_precision_apply_and_readback PASS")

    def test_set_precision_float32_matmul_precision(self):
        """set_precision with float32_matmul_precision='high' is accepted (no RuntimeError)."""
        # Note: cannot call torch.get_float32_matmul_precision() after mixing
        # legacy (matmul.allow_tf32) and new (set_float32_matmul_precision) APIs
        policy = set_precision(float32_matmul_precision="high")
        assert policy["float32_matmul_precision"] == "high"
        # Verify the call did not raise
        reset_precision()
        print(f"test_set_precision_float32_matmul_precision PASS")

    def test_reset_precision_returns_to_defaults(self):
        """reset_precision returns torch backends to DEFAULT_POLICY values."""
        import torch
        set_precision(allow_tf32=True, cudnn_benchmark=True, float32_matmul_precision="medium")
        reset_precision()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        print(f"test_reset_precision_returns_to_defaults PASS")
