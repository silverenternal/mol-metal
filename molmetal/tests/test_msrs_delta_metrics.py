"""MET-3 tests — delta metrics must measure magnitude, not just sign."""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.msrs_delta_metrics import (
    by_stratum,
    cluster_bootstrap,
    delta_metrics,
)


def test_perfect_prediction_has_zero_error():
    obs = np.array([-1.0, -0.2, 0.0, 0.3, 1.5])
    out = delta_metrics(obs, obs.copy())
    assert out["mae_delta"] == 0.0
    assert out["rmse_delta"] == 0.0
    assert out["sign_accuracy"] == 1.0
    assert out["calibration"]["slope"] == pytest.approx(1.0, abs=1e-6)
    assert out["calibration"]["intercept"] == pytest.approx(0.0, abs=1e-6)


def test_a_two_times_prediction_has_slope_one_half():
    obs = np.array([-1.0, -0.5, 0.5, 1.0])
    out = delta_metrics(obs, 2.0 * obs)
    assert out["calibration"]["slope"] == pytest.approx(0.5, abs=1e-6)
    assert out["calibration"]["slope_through_origin"] == pytest.approx(0.5, abs=1e-6)
    assert out["mae_delta"] == pytest.approx(0.75, abs=1e-6)
    # Sign is still perfect: this is exactly the failure a sign metric misses.
    assert out["sign_accuracy"] == 1.0


def test_reversed_prediction_has_zero_sign_accuracy():
    obs = np.array([-1.0, -0.5, 0.5, 1.0])
    out = delta_metrics(obs, -obs)
    assert out["sign_accuracy"] == 0.0
    assert out["mae_delta"] == pytest.approx(1.5, abs=1e-6)


def test_zero_variance_prediction_reports_status():
    obs = np.array([-1.0, 0.0, 1.0])
    out = delta_metrics(obs, np.zeros(3))
    assert out["calibration"]["status"] == "ZERO_VARIANCE"


def test_empty_input_is_not_silently_zero():
    out = delta_metrics(np.array([]), np.array([]))
    assert out == {"n_pairs": 0, "status": "EMPTY"}


def test_cluster_bootstrap_refuses_too_few_dois():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    pred = y.copy()
    ia = np.array([0, 2])
    ib = np.array([1, 3])
    doi = np.array(["a", "a", "b", "b"])
    out = cluster_bootstrap(y, pred, ia, ib, doi, np.array([True, True]), n_boot=10)
    assert out["status"] == "TOO_FEW_DOIS"


def test_cluster_bootstrap_resamples_dois_not_pairs():
    rng = np.random.default_rng(0)
    n_doi, per_doi = 12, 6
    y = rng.normal(size=n_doi * per_doi)
    pred = y + rng.normal(scale=0.3, size=y.size)
    ia, ib = [], []
    for d in range(n_doi):
        base = d * per_doi
        for j in range(per_doi // 2):
            ia.append(base + 2 * j)
            ib.append(base + 2 * j + 1)
    ia, ib = np.array(ia), np.array(ib)
    # cluster_bootstrap expects the ROW-level DOI array, not a per-pair one.
    row_doi = np.repeat([f"10.1/{d}" for d in range(n_doi)], per_doi)
    out = cluster_bootstrap(
        y, pred, ia, ib, row_doi, np.ones(len(ia), dtype=bool), n_boot=50
    )
    assert out["n_dois"] == n_doi
    lo, hi = out["sign_accuracy"]["ci95"]
    assert lo <= hi


def test_by_stratum_assigns_every_pair_once():
    obs = np.array([0.05, 0.15, 0.3, 0.7, 1.4])
    pred = obs.copy()
    out = by_stratum(obs, pred)
    total = sum(block["n_pairs"] for block in out.values())
    assert total == len(obs)
    assert out["[0,0.1)"]["n_pairs"] == 1
    assert out["[1,inf)"]["n_pairs"] == 1
