"""WF-Deflex Wire-up Phase 1 unit tests — F5 learned shaping wire.

Tests for ``RewardAggregator._learned_shaping_contribution`` and
``RewardAggregator.register_learned_shaping_channel``.  The tests
are env-var-driven so the production behaviour (channel OFF) is the
default and the F5 contribution fires only when the gate is on.

Run with::

    uv run pytest tests/test_deflex_wireup_phase1_f5.py -v

Or all deflex-wireup tests at once::

    uv run pytest tests/test_deflex_wireup_*.py -v
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Import the production class under test.  We import lazily so the
# optional ``rdkit`` import in proof_search.py doesn't fail in a
# bare-bones environment.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aggregator_cls():
    try:
        from molmetal_lam.search_alg.proof_search import RewardAggregator
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"proof_search not importable: {exc}")
    return RewardAggregator


@pytest.fixture(autouse=True)
def _reset_env():
    """Reset the LEARNED_SHAPING_ENABLED env-var between tests."""
    old = os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["LEARNED_SHAPING_ENABLED"] = old
        else:
            os.environ.pop("LEARNED_SHAPING_ENABLED", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _DummyState:
    """Minimal stand-in for :class:`MoleculeClosedTerm` for unit tests."""

    def __init__(self, smiles: str = "C") -> None:
        self._smi = smiles

    def canonical_smiles(self) -> str:
        return self._smi


# ---------------------------------------------------------------------------
# 1. Default OFF (no env, no flag)
# ---------------------------------------------------------------------------
def test_f5_contribution_default_off_is_zero(aggregator_cls):
    """Default state: no env-var, no flag → F5 contribution is 0.0."""
    os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    agg = aggregator_cls()
    assert agg.use_learned_shaping is False
    assert agg._learned_shaping_active() is False
    assert agg._learned_shaping_contribution(3.32) == 0.0
    # The aggregator's __call__ with no channels configured is bit-for-bit
    # identical to the legacy behaviour.
    state = _DummyState()
    val = agg(state)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# 2. Programmatic flag enables the channel
# ---------------------------------------------------------------------------
def test_f5_contribution_flag_on_returns_nonzero(aggregator_cls):
    """When ``register_learned_shaping_channel(enabled=True)`` is called,
    the F5 contribution is ``1.0 * (2.5836 - 2.5149 * sa_score)``.
    """
    os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    agg = aggregator_cls()
    agg.register_learned_shaping_channel(enabled=True)
    assert agg.use_learned_shaping is True
    assert agg._learned_shaping_active() is True
    sa_score = 3.32
    expected = 1.0 * (2.5836 - 2.5149 * sa_score)
    assert abs(agg._learned_shaping_contribution(sa_score) - expected) < 1e-6


# ---------------------------------------------------------------------------
# 3. Env-var enables the channel without touching the flag
# ---------------------------------------------------------------------------
def test_f5_contribution_env_var_enables_channel(aggregator_cls):
    """Setting ``LEARNED_SHAPING_ENABLED=1`` enables the channel even
    if ``use_learned_shaping`` is False.
    """
    os.environ["LEARNED_SHAPING_ENABLED"] = "1"
    try:
        agg = aggregator_cls()
        # Flag is still False but the env-var should gate the channel on.
        assert agg.use_learned_shaping is False
        assert agg._learned_shaping_active() is True
        sa_score = 5.0
        expected = 1.0 * (2.5836 - 2.5149 * sa_score)
        assert abs(agg._learned_shaping_contribution(sa_score) - expected) < 1e-6
    finally:
        os.environ.pop("LEARNED_SHAPING_ENABLED", None)


# ---------------------------------------------------------------------------
# 4. The contribution is wired into __call__
# ---------------------------------------------------------------------------
def test_f5_contribution_fires_in_call(aggregator_cls):
    """When the SA channel is set + F5 is enabled, __call__ applies the
    F5 formula on top of the existing channels.  Verify the difference
    between F5-on and F5-off is exactly the F5 formula evaluated at
    the supplied sa_score.
    """
    os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    agg_off = aggregator_cls()
    agg_off.r_sa = lambda state: 3.0  # 1..10 scale
    v_off = agg_off(_DummyState())

    agg_on = aggregator_cls()
    agg_on.r_sa = lambda state: 3.0
    agg_on.register_learned_shaping_channel(enabled=True)
    v_on = agg_on(_DummyState())

    delta = v_on - v_off
    expected_delta = 2.5836 - 2.5149 * 3.0
    assert abs(delta - expected_delta) < 1e-6


# ---------------------------------------------------------------------------
# 5. F5 off + no channels → 0.0 reward (bit-for-bit legacy behaviour)
# ---------------------------------------------------------------------------
def test_f5_off_no_channels_returns_zero(aggregator_cls):
    """Default reward + F5 off = 0.0 (no reward channels, no shaping)."""
    os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    agg = aggregator_cls()
    assert agg(_DummyState()) == 0.0


# ---------------------------------------------------------------------------
# 6. Register channel with custom formula
# ---------------------------------------------------------------------------
def test_f5_register_with_custom_formula(aggregator_cls):
    """Custom formulas are accepted via ``register_learned_shaping_channel``.
    The override is invoked through ``_learned_shaping_contribution``.
    """
    os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    from molmetal_lam.reward.learned_shaping import (
        BestFormula,
        LearnedShaping,
        F5_FEATURE_NAMES,
    )

    def _callable(sa_score: float) -> float:
        return 5.0 + sa_score  # custom intercept + identity slope

    custom = BestFormula(
        name="custom_test",
        formula_str="5.0 + sa_score",
        coefficients={"intercept": 5.0, "sa_mean_norm": 1.0},
        complexity=1,
        r_squared_in_sample=1.0,
        r_squared_loo=0.0,
        lit_anchor="test",
        formula_callable=_callable,
        n_fit=1,
        family="custom",
    )
    agg = aggregator_cls()
    agg.register_learned_shaping_channel(enabled=True, formula=custom)
    assert agg._learned_shaping_contribution(2.0) == 7.0