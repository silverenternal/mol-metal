"""MET-8 tests — MDE arithmetic must round-trip through required-n."""

from __future__ import annotations

import math

import pytest

from molmetal.scripts.msrs_power import (
    K,
    describe,
    mde,
    power_curve,
    required_n,
    sd_from_ci,
)


def test_sd_from_ci_is_half_width_over_1_96():
    assert sd_from_ci([-1.0, 1.0]) == pytest.approx(2.0 / (2 * 1.959963984540054))


def test_mde_is_k_times_sd_at_the_reference_n():
    assert mde(0.02, 100) == pytest.approx(K * 0.02)


def test_mde_shrinks_with_the_square_root_of_n():
    base = mde(0.02, 100)
    assert mde(0.02, 100, 400) == pytest.approx(base / 2.0)


def test_required_n_round_trips_the_mde():
    sd, n0 = 0.02, 100
    target = mde(sd, n0)
    assert required_n(sd, target, n0) == pytest.approx(n0, abs=1)


def test_required_n_is_infinite_without_an_effect():
    assert required_n(0.02, 0.0, 100) is None


def test_power_curve_is_monotone_in_n():
    curve = power_curve(0.02, 100, grid=(25, 100, 400))
    assert curve[0]["mde"] > curve[1]["mde"] > curve[2]["mde"]


def test_describe_marks_an_effect_below_the_mde_as_underpowered():
    out = describe("x", observed=0.01, ci=[-0.02, 0.04], n0=50, unit="u")
    assert out["powered_for_observed"] is False
    assert out["n_required_for_observed"] > 50


def test_describe_marks_a_large_effect_as_powered():
    out = describe("x", observed=0.20, ci=[0.10, 0.30], n0=50, unit="u")
    assert out["powered_for_observed"] is True
