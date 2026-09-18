"""Unit tests for the ``SweepConfig`` dataclass.

These tests guard the round-4 SOTA-comparable defaults — a regression
that silently changes the n_candidates/pocket column emitted by
``lambda_100pocket_sweep.py`` is exactly the kind of breakage this
test is here to catch.
"""

from __future__ import annotations

import dataclasses

import pytest

from molmetal.scripts._sweep_helpers import SweepConfig


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


def test_sweep_config_top_k_default_is_20():
    """``top_k=20`` is the round-4 lower edge of the SOTA 50-100 band.

    Phase-0 default was 5 (yielded only ~1.7 candidates/pocket at
    ``best_score=1.7046`` — not SOTA-comparable).
    """
    cfg = SweepConfig()
    assert cfg.top_k == 20


def test_sweep_config_max_depth_default_is_3():
    """``max_depth=3`` gives enough leaves for ``top_k=20``.

    Phase-0 default was 2 (too shallow to populate 20 candidates).
    """
    cfg = SweepConfig()
    assert cfg.max_depth == 3


def test_sweep_config_n_simulations_default_is_200():
    """``n_simulations=200`` is the round-4 bounded-budget default.

    Phase-0 default was 1000 — round-0 measurements showed UCB
    converges within ~30 sims at the 12-tile Phase-0 branching, so
    1000 was wasted compute.  200 is a 5x safety margin.
    """
    cfg = SweepConfig()
    assert cfg.n_simulations == 200


def test_sweep_config_patience_default_is_30():
    """``patience=30`` is the round-0 calibrated early-stop value."""
    cfg = SweepConfig()
    assert cfg.patience == 30


def test_sweep_config_early_stop_default_is_true():
    """``early_stop=True`` is the round-0 default (UCB converges fast)."""
    cfg = SweepConfig()
    assert cfg.early_stop is True


def test_sweep_config_all_defaults_at_once():
    """All five defaults in one assertion — easy to scan in a failure log."""
    cfg = SweepConfig()
    assert cfg.top_k == 20
    assert cfg.max_depth == 3
    assert cfg.n_simulations == 200
    assert cfg.patience == 30
    assert cfg.early_stop is True


# ---------------------------------------------------------------------------
# Override tests (frozen=False)
# ---------------------------------------------------------------------------


def test_sweep_config_is_dataclass():
    """``SweepConfig`` is a ``@dataclass`` (not a plain class)."""
    assert dataclasses.is_dataclass(SweepConfig)


def test_sweep_config_is_not_frozen():
    """The dataclass is intentionally ``frozen=False`` (allow override).

    The class docstring promises callers can do
    ``cfg = SweepConfig(); cfg.top_k = 50``.  This test guards that
    promise — if anyone re-decorates the class with ``@dataclass(frozen=True)``
    this test will fail and the override pattern will break.
    """
    cfg = SweepConfig()
    # frozen dataclasses raise ``FrozenInstanceError`` on assignment.
    cfg.top_k = 50
    assert cfg.top_k == 50


def test_sweep_config_override_each_field():
    """Every field can be overridden in-place after construction."""
    cfg = SweepConfig()
    cfg.top_k = 100
    cfg.max_depth = 5
    cfg.n_simulations = 500
    cfg.patience = 60
    cfg.early_stop = False
    assert cfg.top_k == 100
    assert cfg.max_depth == 5
    assert cfg.n_simulations == 500
    assert cfg.patience == 60
    assert cfg.early_stop is False


def test_sweep_config_kwargs_override_at_construction():
    """Fields can also be overridden at construction time via kwargs."""
    cfg = SweepConfig(top_k=100, n_simulations=500, early_stop=False)
    assert cfg.top_k == 100
    assert cfg.n_simulations == 500
    assert cfg.early_stop is False
    # Untouched fields keep the canonical defaults.
    assert cfg.max_depth == 3
    assert cfg.patience == 30


# ---------------------------------------------------------------------------
# Field-coverage tests — guard against accidentally dropping a field
# ---------------------------------------------------------------------------


def test_sweep_config_has_exactly_five_fields():
    """A new field is a *deliberate* API change; the test must be updated.

    If a future edit adds or removes a field, this test will fail and
    force the author to think about whether the change is intentional.
    """
    field_names = {f.name for f in dataclasses.fields(SweepConfig)}
    assert field_names == {"top_k", "max_depth", "n_simulations", "patience", "early_stop"}


def test_sweep_config_field_types():
    """Field types are stable — guarded by this test."""
    cfg = SweepConfig()
    assert isinstance(cfg.top_k, int)
    assert isinstance(cfg.max_depth, int)
    assert isinstance(cfg.n_simulations, int)
    assert isinstance(cfg.patience, int)
    assert isinstance(cfg.early_stop, bool)
