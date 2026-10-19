"""Tests for FIX-M3's variance-ceiling analysis (``msrs_variance_ceiling``).

The script answers an identifiability question, so the tests pin the pieces
that could silently produce a number:

* the cell table reproduces replicate counts and within-cell sums of squares,
* only cohorts with >= 2 metals are scored,
* a purely additive dataset yields ~100% metal-main share and ~0 residual,
* the ceiling falls monotonically as the assumed noise grows,
* the Wilson interval stays inside [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_variance_ceiling import (  # noqa: E402
    _wilson,
    additive_decomposition,
    cell_table,
    evaluation_cells,
    noise_estimate,
)


def _frame() -> pd.DataFrame:
    # Two cohorts, two metals each; cohort 1 has a replicate.
    return pd.DataFrame(
        {
            "cohort": [0, 0, 0, 1, 1, 1, 2],
            "metal": ["Pt", "Pt", "Ru", "Pt", "Ru", "Ru", "Pt"],
            "value_log10": [1.0, 1.2, 2.0, 3.0, 4.0, 4.2, 9.0],
        }
    )


def test_cell_table_counts_and_ss():
    cells = cell_table(_frame())
    pt0 = cells[(cells["cohort"] == 0) & (cells["metal"] == "Pt")].iloc[0]
    assert pt0["n"] == 2
    assert pt0["y"] == pytest.approx(1.1)
    assert pt0["ss"] == pytest.approx(0.02)  # (0.1)^2 * 2
    ru1 = cells[(cells["cohort"] == 1) & (cells["metal"] == "Ru")].iloc[0]
    assert ru1["ss"] == pytest.approx(0.02)


def test_evaluation_cells_drops_singletons():
    ev = evaluation_cells(cell_table(_frame()))
    assert set(ev["cohort"]) == {0, 1}  # cohort 2 has one metal


def test_noise_estimate_uses_only_replicated_cells():
    noise = noise_estimate(cell_table(_frame()))
    assert noise["n_replicated_cells"] == 2
    assert noise["df"] == 2  # one df from each 2-replicate cell
    assert noise["pooled_sigma"] == pytest.approx(np.sqrt(0.02), abs=1e-4)


def test_additive_data_has_no_interaction():
    # y = cohort effect + metal effect exactly, one cell per (cohort, metal).
    rows = []
    for c in range(30):
        for m, effect in (("Pt", 0.0), ("Ru", 1.0), ("Ir", -0.5)):
            rows.append({"cohort": c, "metal": m, "value_log10": c * 0.3 + effect})
    cells = pd.DataFrame(rows)
    decomp, resid = additive_decomposition(cell_table(cells))
    assert decomp["metal_main_share"] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(resid, 0.0)


def test_noise_lowers_the_ceiling_monotonically():
    from molmetal.scripts.msrs_variance_ceiling import ceiling_curve

    rows = []
    rng = np.random.default_rng(0)
    for c in range(20):
        for m in ("Pt", "Ru", "Ir"):
            rows.append({"cohort": c, "metal": m, "value_log10": rng.normal()})
    cells = cell_table(pd.DataFrame(rows))
    decomp, _ = additive_decomposition(cells)
    curve = ceiling_curve(cells, decomp["var_within"], decomp["additive_residual_ms"], (0.05, 0.2, 0.5))
    assert curve[0]["max_within_r2"] >= curve[1]["max_within_r2"] >= curve[2]["max_within_r2"]
    assert curve[0]["perfect_model_top1"] >= curve[1]["perfect_model_top1"] >= curve[2]["perfect_model_top1"]


def test_wilson_interval_is_bounded():
    lo, hi = _wilson(13, 15)
    assert 0.0 <= lo <= 13 / 15 <= hi <= 1.0
    assert (lo, hi) == pytest.approx((0.6212, 0.9626), abs=1e-4)
