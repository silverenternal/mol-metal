"""MET-5 tests — coverage curves must be paired and Holm must be Holm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.msrs_decision_curve import (
    holm,
    net_benefit,
    run_protocol,
    top1_margin,
)


def test_top1_margin_is_the_gap_between_best_and_second_best():
    assert top1_margin([3.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert top1_margin([1.0, 1.0]) == pytest.approx(0.0)


def test_holm_adjusts_in_sorted_order_and_preserves_input_order():
    adj = holm([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max 0.03/0.06/0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_never_exceeds_one():
    assert holm([0.9, 0.8, 0.95]) == pytest.approx([1.0, 1.0, 1.0])


def test_net_benefit_is_one_when_every_cohort_hits():
    hit = np.ones(10, dtype=bool)
    assert net_benefit(hit, [0.5])["0.5"] == pytest.approx(1.0)


def test_net_benefit_is_minus_one_when_none_hit_at_a_half_threshold():
    hit = np.zeros(10, dtype=bool)
    assert net_benefit(hit, [0.5])["0.5"] == pytest.approx(-1.0)


def test_net_benefit_is_zero_at_the_base_rate():
    hit = np.array([True, False, True, False])
    # TP=2, FP=2, p_t=0.5 -> (2 - 2) / 4 = 0
    assert net_benefit(hit, [0.5])["0.5"] == pytest.approx(0.0)


def _synthetic(n_ligands=9, n_metals=2, n_dois=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ligands):
        doi = f"10.1000/{2018 + i % n_dois}.p{i % n_dois}"
        for j, metal in enumerate(["Ru", "Ir"][:n_metals]):
            rows.append(
                {
                    "ligand_canonical": f"L{i}",
                    "cell_line_key": "A549",
                    "doi": doi,
                    "time_h": 48.0,
                    "metal": metal,
                    "y": float(i % 3 + j),
                }
            )
    return pd.DataFrame(rows)


def test_run_protocol_returns_a_paired_curve_at_every_coverage():
    sub = _synthetic()
    y = sub["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    pred = y + rng.normal(scale=0.1, size=len(y))
    out = run_protocol(sub, y, pred, "ligand", seed=0)
    assert out["n_cohorts"] == 9
    assert len(out["curve"]) == 10
    for row in out["curve"]:
        for scorer in ("model", "global_best", "cell_best"):
            assert scorer in row
            assert "mean_regret" in row[scorer]
        assert "p_holm" in row["model_vs_global_best"]
    assert set(out["gate"]) == {"coverage", "mean_regret", "p_holm", "passed"}
