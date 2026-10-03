"""Wf-360-A2 — conftest hooks for the cross-Wf integration suite.

Pure CPU-only plumbing. Sets up:

* ``WF360_CPU_ONLY`` env-var default (``1`` so tests skip @gpu-marked).
* ``pytest_collection_modifyitems`` hook that auto-skips GPU-marked
  tests unless the user opts in with ``WF360_CPU_ONLY=0``.

This conftest is additive — it does NOT replace the existing
``molmetal/tests/conftest.py`` (which is the ODESolver shim per
Wf-235). Both conftests coexist.
"""

from __future__ import annotations

import os

import pytest

# Default to CPU-only mode unless user explicitly opts in.
os.environ.setdefault("WF360_CPU_ONLY", "1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip GPU-marked tests when WF360_CPU_ONLY != "0"."""
    if os.environ.get("WF360_CPU_ONLY", "1") == "0":
        return
    skip_gpu = pytest.mark.skip(reason="Wf-360-A2 CPU-only mode (gpu-marked)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
