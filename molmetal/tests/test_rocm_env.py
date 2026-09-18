"""Tests for ``molmetal._apply_rocm_gfx1101_env``.

Coverage
--------
1. :func:`test_env_vars_applied_idempotent` — re-invocation with
   ``force=True`` returns a dict and does not raise; re-invocation with
   ``force=False`` (the default) returns ``{}`` because the sentinel
   short-circuits a second run.
2. :func:`test_pytorch_hip_alloc_conf_set` — after :func:`_apply_rocm_gfx1101_env`
   has run on a synthetic gfx1101 host, the var
   ``PYTORCH_HIP_ALLOC_CONF`` is set to ``expandable_segments:True``.
3. :func:`test_torch_blas_prefer_hipblaslt_set` — same guarantee for
   ``TORCH_BLAS_PREFER_HIPBLASLT=1``.

These tests are CPU-only — they monkey-patch
:func:`molmetal._is_gfx1101_gpu` so they pass on hosts without an AMD
GPU.  No pip deps added (only ``unittest.mock`` from the stdlib).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from molmetal import (
    ROCM_GFX1101_ENV_APPLIED_VARS,
    _apply_rocm_gfx1101_env,
    _is_gfx1101_gpu,
)


def _reset_sentinel() -> None:
    """Clear the global sentinel so each test gets a fresh bootstrap."""
    import molmetal

    molmetal._ROCM_GFX1101_ENV_APPLIED = False


class TestRocmEnv(unittest.TestCase):
    """Verify ``molmetal._apply_rocm_gfx1101_env`` end-to-end."""

    def setUp(self) -> None:
        _reset_sentinel()
        # Always start from a clean slate for the two HIP-related vars
        # — we don't want a stale env (e.g. from ``.bashrc``) masking a
        # regression.
        for var in ROCM_GFX1101_ENV_APPLIED_VARS:
            os.environ.pop(var, None)

    def tearDown(self) -> None:
        _reset_sentinel()
        for var in ROCM_GFX1101_ENV_APPLIED_VARS:
            os.environ.pop(var, None)

    # ------------------------------------------------------------------
    # 1. Idempotency
    # ------------------------------------------------------------------
    def test_env_vars_applied_idempotent(self) -> None:
        """Two ``force=True`` calls run cleanly; sentinel short-circuit
        returns ``{}`` without raising."""
        # Pretend we have a gfx1101 so the env vars actually get set.
        with mock.patch.object(
            __import__("molmetal", fromlist=["_is_gfx1101_gpu"]),
            "_is_gfx1101_gpu",
            return_value=True,
        ):
            # First call: forced, must return at least the HIP vars.
            first = _apply_rocm_gfx1101_env(force=True, quiet=True)
            self.assertIn("PYTORCH_HIP_ALLOC_CONF", first)
            self.assertIn("TORCH_BLAS_PREFER_HIPBLASLT", first)

            # Second forced call: must succeed (no error) and return a
            # dict (even if empty because the values are already correct).
            second = _apply_rocm_gfx1101_env(force=True, quiet=True)
            self.assertIsInstance(second, dict)

            # Non-forced call after the sentinel is set: short-circuits.
            third = _apply_rocm_gfx1101_env(force=False, quiet=True)
            self.assertEqual(third, {})

    # ------------------------------------------------------------------
    # 2. PYTORCH_HIP_ALLOC_CONF
    # ------------------------------------------------------------------
    def test_pytorch_hip_alloc_conf_set(self) -> None:
        """``PYTORCH_HIP_ALLOC_CONF`` is set to ``expandable_segments:True``.

        Matches the canonical value defined in
        :data:`molmetal.adapters.flow_matching_lipman.amp.ROCM_GFX1101_ENV_VARS`
        (``molmetal/adapters/flow_matching_lipman/amp.py:73-76``).
        """
        with mock.patch(
            "molmetal._is_gfx1101_gpu", return_value=True
        ):
            _apply_rocm_gfx1101_env(force=True, quiet=True)

        self.assertEqual(
            os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
            "expandable_segments:True",
        )

    # ------------------------------------------------------------------
    # 3. TORCH_BLAS_PREFER_HIPBLASLT
    # ------------------------------------------------------------------
    def test_torch_blas_prefer_hipblaslt_set(self) -> None:
        """``TORCH_BLAS_PREFER_HIPBLASLT`` is set to ``"1"``.

        Matches the canonical value defined in
        :data:`molmetal.adapters.flow_matching_lipman.amp.ROCM_GFX1101_ENV_VARS`
        (``molmetal/adapters/flow_matching_lipman/amp.py:73-76``).
        """
        with mock.patch(
            "molmetal._is_gfx1101_gpu", return_value=True
        ):
            _apply_rocm_gfx1101_env(force=True, quiet=True)

        self.assertEqual(
            os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
            "1",
        )


class TestGpuDetection(unittest.TestCase):
    """Smoke-check :func:`_is_gfx1101_gpu` returns a bool and never raises."""

    def test_is_gfx1101_gpu_returns_bool(self) -> None:
        # Even on a CPU-only host the helper must return a bool and not
        # blow up — that is the whole point of the lazy + non-raising
        # contract documented in ``molmetal/__init__.py``.
        result = _is_gfx1101_gpu()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()