"""MET-4 tests — interval metrics must reward sharpness, not just coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.msrs_calibration import (
    _pit,
    cohort_sizes,
    pit_histogram,
    point_calibration,
    run_protocol,
    winkler_score,
)


def test_winkler_equals_width_when_every_point_is_covered():
    y = np.zeros(4)
    lo, hi = np.full(4, -1.0), np.full(4, 1.0)
    assert winkler_score(y, lo, hi, alpha=0.1) == pytest.approx(2.0)


def test_winkler_penalises_misses():
    y = np.array([2.0])
    lo, hi = np.array([-1.0]), np.array([1.0])
    # width 2 + (2/0.1) * 1 = 22
    assert winkler_score(y, lo, hi, alpha=0.1) == pytest.approx(22.0)


def test_winkler_prefers_the_sharper_of_two_calibrated_intervals():
    y = np.zeros(10)
    wide = winkler_score(y, np.full(10, -2.0), np.full(10, 2.0), alpha=0.1)
    narrow = winkler_score(y, np.full(10, -1.0), np.full(10, 1.0), alpha=0.1)
    assert narrow < wide


def test_pit_histogram_is_flat_for_a_uniform_pit():
    pit = np.linspace(0.0, 1.0, 1001)[:-1]
    out = pit_histogram(pit, n_bins=10)
    assert out["max_abs_deviation_from_uniform"] < 0.01


def test_pit_histogram_flags_a_pile_up_at_zero():
    pit = np.concatenate([np.zeros(500), np.linspace(0, 1, 500)])
    out = pit_histogram(pit, n_bins=10)
    assert out["max_abs_deviation_from_uniform"] > 0.2


def test_pit_is_the_fraction_of_calibration_scores_below_the_test_score():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    test = np.array([0.5, 2.0, 4.0, 5.0])
    out = _pit(cal, test)
    assert out.tolist() == [0.0, 0.5, 1.0, 1.0]


def test_point_calibration_recovers_a_linear_relation():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * pred + 1.0
    out = point_calibration(y, pred)
    assert out["slope"] == pytest.approx(2.0, abs=1e-6)
    assert out["intercept"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_fit"] == pytest.approx(1.0, abs=1e-6)
    # The identity line is 2x off, so the raw point R2 is negative.
    assert out["r2_point"] < 0
    # rmse is measured against the identity line, not the fitted one.
    # (The artifact rounds to 4 dp, hence the 1e-4 tolerance.)
    assert out["rmse"] == pytest.approx(float(np.sqrt(np.mean((y - pred) ** 2))), abs=1e-4)


def test_point_calibration_of_a_perfect_identity_is_exact():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    out = point_calibration(pred, pred)
    assert out["slope"] == pytest.approx(1.0, abs=1e-6)
    assert out["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert out["r2_point"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_fit"] == pytest.approx(1.0, abs=1e-6)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_cohort_sizes_counts_distinct_metals_per_cohort():
    frame = pd.DataFrame(
        {
            "ligand_canonical": ["a", "a", "b", "b", "b"],
            "cell_line_key": ["A549"] * 5,
            "doi": ["10.1/x"] * 5,
            "time_h": [48.0] * 5,
            "metal": ["Ru", "Ir", "Ru", "Ru", "Os"],
        }
    )
    assert cohort_sizes(frame).tolist() == [2, 2, 2, 2, 2]


def test_run_protocol_reports_pit_for_every_variant():
    """Regression: the Mondrian arm used to leave its PIT array all-NaN."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    pred = 0.5 * y + rng.normal(scale=0.5, size=n)
    fold = np.arange(n) % 3
    metal = rng.choice(["Ru", "Ir", "Os"], size=n)
    cell = rng.choice(["A549", "HepG2"], size=n)
    out = run_protocol(y, pred, fold, metal, cell, np.ones(n))
    for name, variant in out["variants"].items():
        assert variant["pit"]["n"] == out["n_evaluated"], name
