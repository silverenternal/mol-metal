"""Benchmark utilities for comparing molecular docking engines."""

from .engine_parity import compare_engines

# Public names used by benchmark runners and reporting code.  The parity
# harness contains the Vina baseline and two lightweight mock engines.
ENGINE_NAMES = ("vina", "qvina_mock", "quickvina2_mock")

__all__ = ["compare_engines", "ENGINE_NAMES"]
