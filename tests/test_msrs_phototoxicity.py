"""Tests for MSRS-5 phototoxicity classification.

The endpoint decides whether the paper claims a metal-specific light response,
so the censoring logic is checked on the exact cases that occur in the source
(``>100`` dark with an exact light value, and the reverse):

  I1. A right-censored bound above the threshold still decides "inactive"; a
      bound that straddles the threshold decides nothing.
  I2. switch / antagonise are the two directional classes and are not confused.
  I3. Wilson intervals stay inside [0, 1] and widen as n falls.
  I4. The DOI-clustered contrast returns a finite interval and does not silently
      use compounds as independent observations.

CPU-only.

Run: pytest tests/test_msrs_phototoxicity.py -v
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

from molmetal.scripts.msrs_phototoxicity import (  # noqa: E402
    classify,
    clustered_difference,
    is_active,
    wilson,
)


# ---------------------------------------------------------------------------
# I1 -- censoring-aware activity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,censoring,expected",
    [
        (5.0, "none", True),
        (50.0, "none", False),
        (10.0, "none", False),      # mainline uses strict < 10 for active
        (100.0, "right", False),    # ">100" is inactive
        (5.0, "right", None),       # ">5" straddles the 10 uM threshold
        (5.0, "left", True),        # "<5" is active
        (50.0, "left", None),       # "<50" straddles
        (np.nan, "none", None),
    ],
)
def test_is_active_resolves_censoring(value, censoring, expected):
    assert is_active(value, censoring) == expected


# ---------------------------------------------------------------------------
# I2 -- classification
# ---------------------------------------------------------------------------
def test_classify_switch_and_antagonise_are_distinct():
    assert classify(100.0, "right", 1.0, "none") == "switch"
    assert classify(1.0, "none", 100.0, "right") == "antagonise"
    assert classify(1.0, "none", 1.0, "none") == "active_both"
    assert classify(100.0, "right", 100.0, "right") == "inactive_both"


def test_classify_undecidable_when_a_bound_straddles():
    assert classify(5.0, "right", 1.0, "none") == "undecidable"
    assert classify(1.0, "none", 50.0, "left") == "undecidable"


# ---------------------------------------------------------------------------
# I3 -- Wilson interval
# ---------------------------------------------------------------------------
def test_wilson_bounds_and_width():
    lo, hi = wilson(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi < 0.35  # 0/10 is not consistent with a 50% rate
    wide = wilson(2, 4)
    narrow = wilson(200, 400)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# I4 -- clustered contrast
# ---------------------------------------------------------------------------
def _table(rows):
    return pd.DataFrame(rows, columns=["metal", "doi", "cls"])


def test_clustered_difference_returns_interval():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(40):
        doi = f"10.1/{d}"
        for _ in range(5):
            rows.append(("Ru", doi, "switch" if rng.random() < 0.6 else "inactive_both"))
            rows.append(("Ir", doi, "switch" if rng.random() < 0.2 else "inactive_both"))
    out = clustered_difference(_table(rows), "Ru", "Ir", "switch")
    assert out["status"] == "OK"
    assert out["rate_difference"] > 0.2
    lo, hi = out["ci95_doi_clustered_bootstrap"]
    assert lo < out["rate_difference"] < hi
    assert out["n_dois"] == 40


def test_clustered_difference_ignores_undecidable_rows():
    rows = [
        ("Ru", "10.1/a", "switch"),
        ("Ru", "10.1/a", "undecidable"),
        ("Ir", "10.1/a", "inactive_both"),
        ("Ir", "10.1/a", "undecidable"),
    ]
    out = clustered_difference(_table(rows), "Ru", "Ir", "switch")
    assert out["n_units"] == {"Ru": 1, "Ir": 1}
    assert out["rate_difference"] == pytest.approx(1.0)
