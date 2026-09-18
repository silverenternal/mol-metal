"""Tests for kinetic aggregation and the sklearn interpolation fallback."""
import logging

import pytest

from molmetal_lam.reactions.kinetic_aggregator import (
    KineticAggregator,
    aggregate_kinetic_score,
)


def test_aggregate_fixed_dict_harmonic_mean():
    rates = {"a": 0.5, "b": 1.0, "c": 0.25}
    expected = 3.0 / (1 / 0.5 + 1 / 1.0 + 1 / 0.25)
    assert aggregate_kinetic_score(rates) == pytest.approx(expected)


def test_custom_weights_are_used():
    aggregator = KineticAggregator(weights={"fast": 2.0, "slow": 1.0})
    expected = 3.0 / (2.0 / 0.8 + 1.0 / 0.2)
    assert aggregator({"fast": 0.8, "slow": 0.2}) == pytest.approx(expected)
    assert aggregator.weights["fast"] == 2.0


def test_missing_reaction_defaults_to_zero(caplog):
    aggregator = KineticAggregator(weights={"CuAAC": 1.0, "Suzuki": 1.0})
    with caplog.at_level(logging.WARNING):
        score = aggregator({"CuAAC": 0.9})
    assert score == 0.0
    assert "Missing reaction rate for Suzuki" in caplog.text


def test_sklearn_fallback_interpolation():
    known = [
        ("CC", "CC", 0.2),
        ("CCO", "CCO", 0.5),
        ("c1ccccc1", "c1ccccc1", 0.9),
    ]
    aggregator = KineticAggregator(known_rate_pairs=known)
    predicted = aggregator.interpolate_rate("CCN", "CCN")
    assert 0.0 <= predicted <= 1.0
    assert aggregator._model is not None


def test_aggregate_score_is_bounded():
    assert 0.0 <= aggregate_kinetic_score({"x": -4, "y": 9}) <= 1.0
    assert 0.0 <= aggregate_kinetic_score({}) <= 1.0


def test_tuple_pair_input_supported():
    aggregator = KineticAggregator()
    pairs = [(('CC', 'CC'), 0.3), (('CCC', 'CCC'), 0.6)]
    assert 0.0 <= aggregator.interpolate_rate('CC', 'CCC', pairs) <= 1.0


def test_default_reaction_names_are_present():
    names = KineticAggregator().weights
    assert {"CuAAC", "SPAAC", "thiol-ene", "Suzuki", "amide-coupling"} <= set(names)


def test_empty_rates_with_defaults_are_zero(caplog):
    with caplog.at_level(logging.WARNING):
        result = KineticAggregator()({})
    assert result == 0.0
    assert caplog.records
