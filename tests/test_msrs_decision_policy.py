"""Tests for the decision-layer policy (AUD-I1 verifiability / AUD-I2 abstention)."""

from __future__ import annotations

import pandas as pd
import pytest

from molmetal.scripts.msrs_decision_policy import (
    abstention_curve,
    best_operating_point,
    margin_table,
    regret_matrix,
    verifiability_report,
)


def _cohorts(rows):
    """rows: (cohort, metal, y, score, doi)"""
    return pd.DataFrame(
        [
            {"cohort": c, "metal": m, "y": y, "s": s, "doi": d, "fold": 0, "k": 2}
            for c, m, y, s, d in rows
        ]
    )


def test_verifiability_counts_pairs_not_metals():
    frame = _cohorts(
        [
            (0, "Ru", 1.0, 0.9, "d1"),
            (0, "Ir", 1.5, 1.4, "d1"),
            (1, "Ru", 1.0, 0.9, "d1"),
            (1, "Ir", 1.2, 1.1, "d1"),
        ]
    )
    out = verifiability_report(frame)
    assert out["n_cohorts"] == 2
    assert out["cohort_size_counts"] == {"2": 2}
    assert out["n_verifiable_pairs"] == 2
    assert out["unverifiable_recommendations"] == 0
    assert out["mean_metals_per_cohort"] == 2.0


def test_margin_is_second_best_minus_best():
    frame = _cohorts(
        [
            (0, "Ru", 1.0, 0.5, "d1"),
            (0, "Ir", 1.5, 0.8, "d1"),  # margin 0.3, Ru predicted best and correct
            (1, "Ru", 1.0, 0.9, "d1"),
            (1, "Ir", 1.5, 0.2, "d1"),  # margin 0.7, Ir predicted best, wrong
        ]
    )
    out = margin_table(frame, "s").set_index("cohort")
    assert out.loc[0, "margin"] == pytest.approx(0.3)
    assert out.loc[0, "top1_correct"] is True or bool(out.loc[0, "top1_correct"])
    assert out.loc[0, "regret"] == pytest.approx(0.0)
    assert out.loc[1, "margin"] == pytest.approx(0.7)
    assert not bool(out.loc[1, "top1_correct"])
    assert out.loc[1, "regret"] == pytest.approx(0.5)


def test_abstention_curve_trades_coverage_for_precision():
    margins = pd.DataFrame(
        {
            "cohort": [0, 1, 2, 3],
            "margin": [0.05, 0.15, 0.35, 0.75],
            "top1_correct": [False, True, True, True],
            "regret": [1.0, 0.0, 0.0, 0.0],
            "doi": ["d"] * 4,
            "fold": [0] * 4,
            "k": [2] * 4,
        }
    )
    curve = {
        c["threshold"]: c
        for c in abstention_curve(margins, [0.0, 0.1, 0.3, 0.5, 0.9])
    }
    assert curve[0.0]["coverage"] == 1.0
    assert curve[0.0]["precision"] == pytest.approx(0.75)
    assert curve[0.3]["coverage"] == pytest.approx(0.5)
    assert curve[0.3]["precision"] == 1.0
    assert curve[0.5]["coverage"] == pytest.approx(0.25)
    assert curve[0.9]["coverage"] == 0.0


def test_best_operating_point_respects_the_coverage_floor():
    curve = [
        {"threshold": 0.0, "coverage": 1.0, "precision": 0.55, "n_kept": 10, "mean_regret_kept": 0.1},
        {"threshold": 0.3, "coverage": 0.25, "precision": 0.80, "n_kept": 3, "mean_regret_kept": 0.1},
        {"threshold": 0.9, "coverage": 0.05, "precision": 1.00, "n_kept": 1, "mean_regret_kept": 0.0},
    ]
    best = best_operating_point(curve)
    assert best["threshold"] == 0.3  # the 1.0 point is below the 20% floor
    assert best["precision"] == 0.80


def test_best_operating_point_is_none_when_nothing_clears_the_floor():
    curve = [
        {"threshold": 0.9, "coverage": 0.05, "precision": 1.0, "n_kept": 1, "mean_regret_kept": 0.0}
    ]
    assert best_operating_point(curve) is None


def test_regret_matrix_scores_every_cohort_for_every_scorer():
    frame = _cohorts(
        [
            (0, "Ru", 1.0, 0.5, "d1"),
            (0, "Ir", 1.5, 0.8, "d1"),
            (1, "Ru", 1.0, 0.9, "d1"),
            (1, "Ir", 1.5, 0.2, "d1"),
        ]
    )
    frame["t"] = frame["s"]
    out = regret_matrix(frame, ["s", "t"])
    assert list(out.columns) == ["s", "t"]
    assert out.loc[0, "s"] == pytest.approx(0.0)
    assert out.loc[1, "s"] == pytest.approx(0.5)
