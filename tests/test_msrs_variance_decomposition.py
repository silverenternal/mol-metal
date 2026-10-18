"""Tests for the MSRS-1 variance decomposition and interaction test.

The numbers this script emits decide whether the whole recommendation story is
pursued, so the estimators are checked against synthetic data with a known
answer rather than only against the real cohort:

  I1. ``noise_floor`` recovers a planted sigma from replicate groups.
  I2. ``nested_variance_components`` recovers planted DOI / ligand / residual
      variances and returns shares that sum to 1.
  I3. ``interaction_test`` says INSUFFICIENT below the unit threshold, detects a
      planted interaction, and does not detect a pure additive effect.
  I4. ``_min_detectable_share`` shrinks as n grows -- otherwise the power note
      printed next to every null result would be meaningless.

CPU-only.

Run: pytest tests/test_msrs_variance_decomposition.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.msrs_variance_decomposition import (  # noqa: E402
    _min_detectable_share,
    interaction_test,
    nested_variance_components,
    noise_floor,
)


# ---------------------------------------------------------------------------
# I1 -- noise floor
# ---------------------------------------------------------------------------
def test_noise_floor_recovers_planted_sigma():
    rng = np.random.default_rng(0)
    sigma = 0.20
    rows = []
    for lig in range(300):
        for rep in range(2):
            rows.append(
                {
                    "endpoint": "potency",
                    "censoring": "none",
                    "ligand_canonical": f"L{lig}",
                    "metal": "Ru",
                    "cell_line_key": "A549",
                    "time_h": 24.0,
                    "doi": "10.1/x",
                    "value_log10": 1.0 + rng.normal(0, sigma),
                }
            )
    out = noise_floor(pd.DataFrame(rows), "potency")
    assert out["per_metal"]["Ru"]["n_groups"] == 300
    assert out["per_metal"]["Ru"]["sigma"] == pytest.approx(sigma, abs=0.03)


def test_noise_floor_ignores_singletons():
    rows = [
        {"endpoint": "potency", "censoring": "none", "ligand_canonical": f"L{i}",
         "metal": "Ru", "cell_line_key": "A549", "time_h": 24.0, "doi": "d",
         "value_log10": float(i)}
        for i in range(10)
    ]
    out = noise_floor(pd.DataFrame(rows), "potency")
    assert out["n_replicate_groups"] == 0
    assert out["pooled"]["sigma2"] is None


# ---------------------------------------------------------------------------
# I2 -- nested variance components
# ---------------------------------------------------------------------------
def test_nested_components_recover_planted_variances():
    rng = np.random.default_rng(1)
    s_doi, s_lig, s_err = 0.8, 0.5, 0.3
    rows = []
    for d in range(60):
        doi_eff = rng.normal(0, s_doi)
        for lig in range(8):
            lig_eff = rng.normal(0, s_lig)
            for _ in range(3):
                rows.append(
                    {
                        "doi": f"D{d}",
                        "ligand_canonical": f"D{d}L{lig}",
                        "value_log10": doi_eff + lig_eff + rng.normal(0, s_err),
                    }
                )
    out = nested_variance_components(pd.DataFrame(rows), "value_log10")
    got = out["sigma2"]
    assert got["doi"] == pytest.approx(s_doi**2, rel=0.35)
    assert got["ligand_within_doi"] == pytest.approx(s_lig**2, rel=0.45)
    assert got["residual"] == pytest.approx(s_err**2, rel=0.25)
    total = sum(got.values())
    assert sum(out["share_of_random_variance"].values()) == pytest.approx(1.0, abs=1e-3)
    assert total == pytest.approx(s_doi**2 + s_lig**2 + s_err**2, rel=0.3)


# ---------------------------------------------------------------------------
# I3 -- interaction test
# ---------------------------------------------------------------------------
def _pair_frame(n_lig: int, delta: np.ndarray, cell: str = "A549") -> pd.DataFrame:
    rows = []
    for i in range(n_lig):
        base = 1.0
        for metal, y in (("Ru", base), ("Os", base + delta[i])):
            rows.append(
                {
                    "endpoint": "potency",
                    "all_censored": False,
                    "ligand_canonical": f"L{i}",
                    "cell_line_key": cell,
                    "doi": "10.1/pair",
                    "time_h": 24.0,
                    "metal": metal,
                    "value_log10": y,
                }
            )
    return pd.DataFrame(rows)


def test_interaction_insufficient_below_threshold():
    out = interaction_test(_pair_frame(5, np.zeros(5)), "potency", "Ru", "Os", {"Ru": 0.04, "Os": 0.04}, 100)
    assert out["status"] == "INSUFFICIENT"
    assert out["n_units"] == 5


def test_interaction_detects_planted_heterogeneity():
    rng = np.random.default_rng(2)
    n = 150
    delta = rng.normal(0.0, 0.6, size=n)  # true effect varies a lot by ligand
    out = interaction_test(
        _pair_frame(n, delta), "potency", "Ru", "Os", {"Ru": 0.04, "Os": 0.002}, 200
    )
    assert out["status"] == "OK"
    assert out["interaction_share"] > 0.8
    assert out["detected"] is True
    assert out["ci95"][0] if "ci95" in out else out["interaction_share_ci95_clustered_bootstrap"][0] > 0.5


def test_interaction_does_not_detect_additive_effect():
    rng = np.random.default_rng(3)
    n = 150
    noise_sd = 0.21
    # Pure additive shift: the SAME delta for every ligand, plus measurement noise.
    delta = 0.35 + rng.normal(0, noise_sd * np.sqrt(2), size=n)
    out = interaction_test(
        _pair_frame(n, delta), "potency", "Ru", "Os", {"Ru": 0.0441, "Os": 0.0441}, 200
    )
    assert out["status"] == "OK"
    assert out["interaction_share"] < 0.15
    assert out["detected"] is False


def test_interaction_requires_same_doi_and_duration():
    df = _pair_frame(40, np.zeros(40))
    # Move Os measurements to a different DOI: no unit should survive.
    mask = df["metal"] == "Os"
    df.loc[mask, "doi"] = "10.1/other"
    out = interaction_test(df, "potency", "Ru", "Os", {"Ru": 0.04, "Os": 0.04}, 100)
    assert out["status"] == "INSUFFICIENT"
    assert out["n_units"] == 0


# ---------------------------------------------------------------------------
# I4 -- power
# ---------------------------------------------------------------------------
def test_min_detectable_share_shrinks_with_n():
    small = _min_detectable_share(50, 500)
    large = _min_detectable_share(500, 500)
    assert 0.0 < large < small < 1.0


def test_min_detectable_share_is_capped_by_noise_df():
    """Power is limited by how well the noise floor is known, not by n.

    With 100 noise df the critical value stays near 1.24, so a 5% variance
    share is undetectable no matter how many units are collected. This is the
    reason the artifact prints a detection limit next to every null result.
    """
    assert _min_detectable_share(10_000, 100) > 0.2
    assert _min_detectable_share(10_000, 20_000) < 0.05
