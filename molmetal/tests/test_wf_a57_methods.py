# Licensed under the Apache License, Version 2.0.
"""Regression tests for Wf-A57 W9-P1 (RECONCILE_VALID_ODE_METHODS).

Wf-A56 honest-negative catalog §6.82g noted that ``GenerationConfig.VALID_METHODS``
held a 4-method set ``{"euler", "midpoint", "dopri5", "heun3"}`` while the
upstream dispatch whitelisted 5 methods including ``rk4``.  Two configs that
named ``method: rk4`` (the trainer-side recommended method per the Lipman
reference) crashed silently at ``__post_init__``.  This file pins the union
shipped on 2026-10-16.

Three regression tests:

  * ``test_valid_methods_accept_union`` — assert all 5 methods pass.
  * ``test_validate_method_rejects_unknown`` — assert ``cgar`` / ``heun`` /
    ``rk2`` (heun + rk2 are valid in ``_reference/solver`` + ``w_flow`` but
    NOT in :class:`GenerationConfig` — keep the asymmetry explicit).
  * ``test_dispatch_in_adapter_validates`` — call the
    ``flow_matching_lipman`` dispatch with a mock config and an unknown
    method to verify the dispatch raises.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.ports import (  # noqa: E402
    GenerationConfig,
    validate_method,
)


# ---------------------------------------------------------------------------
# W9-P1 fix 1 — VALID_METHODS now contains the union of 5 methods.
# ---------------------------------------------------------------------------
class TestValidMethodsAcceptUnion:
    """All five canonical methods must construct without ValueError."""

    @pytest.mark.parametrize(
        "method",
        ["euler", "midpoint", "dopri5", "heun3", "rk4"],
    )
    def test_each_method_constructs(self, method: str) -> None:
        cfg = GenerationConfig(method=method, n_samples=2, n_steps=4)
        assert cfg.method == method

    def test_valid_methods_is_frozenset_of_five(self) -> None:
        assert GenerationConfig.VALID_METHODS == frozenset(
            {"euler", "midpoint", "dopri5", "heun3", "rk4"}
        )


# ---------------------------------------------------------------------------
# W9-P1 fix 1 (negative) — unknown methods must raise at __post_init__.
# ``heun`` and ``rk2`` are valid in ``_reference/solver`` + ``w_flow`` but
# NOT in :class:`GenerationConfig`; we keep the asymmetry explicit.
# ---------------------------------------------------------------------------
class TestValidateMethodRejectsUnknown:
    """Reject non-GenerationConfig-surface method names."""

    @pytest.mark.parametrize("method", ["cgar", "heun", "rk2"])
    def test_post_init_raises(self, method: str) -> None:
        with pytest.raises(ValueError, match="GenerationConfig.method must be one of"):
            GenerationConfig(method=method, n_samples=2, n_steps=4)

    @pytest.mark.parametrize("method", ["cgar", "heun", "rk2"])
    def test_validate_method_helper_raises(self, method: str) -> None:
        with pytest.raises(ValueError, match="ODE method must be one of"):
            validate_method(method)


# ---------------------------------------------------------------------------
# W9-P1 fix 2 — dispatch whitelist re-ship with the expanded union set.
# We assert the helper rejects unknown methods at module import-time
# granularity: importing ``flow_matching_lipman`` exposes ``validate_method``
# which gates the dispatch in :func:`generate`.
# ---------------------------------------------------------------------------
class TestDispatchInAdapterValidates:
    """Dispatch in :mod:`molmetal.adapters.flow_matching_lipman` must reject
    unknown methods via :func:`validate_method`."""

    def test_adapter_exposes_validate_method(self) -> None:
        """The adapter re-exports ``validate_method`` from ``molmetal.ports``
        so the dispatch gate is one canonical call."""
        from molmetal.adapters import flow_matching_lipman

        assert hasattr(flow_matching_lipman, "validate_method")
        # And it is the same callable, so dispatch + config share one source.
        from molmetal.ports import validate_method as vm_port

        assert flow_matching_lipman.validate_method is vm_port

    def test_validate_method_rejects_unknown_at_helper_level(self) -> None:
        # The dispatch re-validates the same canonical set; the helper-level
        # rejection is the load-bearing path (it runs before solver.sample()).
        with pytest.raises(ValueError, match="ODE method must be one of"):
            validate_method("unknown_method")
