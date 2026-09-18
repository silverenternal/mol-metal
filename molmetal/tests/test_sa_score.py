"""Tests for the Ertl-Schuffenhauer SA-score wrapper.

Reference values are taken from the RDKit Contrib SA_Score
``UnitTestSAScore.py`` distribution and the Ertl & Schuffenhauer (2009)
paper.  Aspirin is widely reported at SA ~ 1.5-1.6 (very easy);
polycyclic / saccharide / natural-product SMILES should land in
[5, 10] (harder).
"""

from __future__ import annotations

import math

import pytest

from molmetal_lam.sbdd_env.sa_score import (
    _SASCORER_AVAILABLE,
    sa_score_ertl,
    sa_score_to_unit,
)


pytestmark = pytest.mark.skipif(
    not _SASCORER_AVAILABLE,
    reason="RDKit sascorer not importable in this environment",
)


# ---------------------------------------------------------------------------
# 1. Aspirin -- trivial drug, SA must be in the "very synthesizable" band.
# ---------------------------------------------------------------------------
def test_sa_score_aspirin():
    sa = sa_score_ertl("CC(=O)Oc1ccccc1C(=O)O")
    assert sa == sa  # not NaN
    # Per RDKit UnitTestSAScore + Ertl paper, aspirin ~ 1.5-2.0
    assert 1.0 <= sa <= 5.0, f"aspirin SA out of expected range: {sa}"
    # Tighter bound -- Ertl explicitly highlights aspirin as an "easy"
    # synthesis benchmark.  Use [1.0, 3.0].
    assert 1.0 <= sa <= 3.0, (
        f"aspirin SA expected [1.0, 3.0], got {sa:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. A complex polycyclic + saccharide should land in [5, 10].
#    Use taxol (5.9 reported) -- the canonical hard-to-synthesize marker.
# ---------------------------------------------------------------------------
def test_sa_score_complex():
    # taxol (paclitaxel) -- long polycyclic natural product
    taxol = (
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)"
        "C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)"
        "C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    )
    sa_taxol = sa_score_ertl(taxol)
    assert sa_taxol == sa_taxol  # not NaN
    assert 5.0 <= sa_taxol <= 10.0, (
        f"taxol SA expected in [5, 10], got {sa_taxol:.3f}"
    )

    # Sucrose -- a disaccharide, expected harder than aspirin
    sa_sucrose = sa_score_ertl(
        "OC1C(O)C(O)C(OC1OC1(CO)OC(CO)C(O)C1O)CO"
    )
    assert sa_sucrose == sa_sucrose
    # sucrose SA ~ 4-5 per Ertl: definitely harder than aspirin
    assert 3.0 <= sa_sucrose <= 7.5, (
        f"sucrose SA expected in [3.0, 7.5], got {sa_sucrose:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. Unit mapping -- SA=1 -> 1.0, SA=10 -> 0.0, monotonic, clamped.
# ---------------------------------------------------------------------------
def test_sa_score_unit_map():
    assert sa_score_to_unit(1.0) == pytest.approx(1.0, abs=1e-9)
    assert sa_score_to_unit(10.0) == pytest.approx(0.0, abs=1e-9)
    # Midpoint -- SA=5.5 -> unit = (10-5.5)/9 = 0.5
    assert sa_score_to_unit(5.5) == pytest.approx(0.5, abs=1e-9)
    # Monotonic: as SA grows, unit-score shrinks
    prev = 1.0
    for sa in [1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0]:
        u = sa_score_to_unit(sa)
        assert u < prev + 1e-9, f"unit-score not monotonic at SA={sa}"
        assert 0.0 <= u <= 1.0, f"unit-score out of [0,1] at SA={sa}: {u}"
        prev = u
    # Out-of-range clamp
    assert sa_score_to_unit(-3.0) == pytest.approx(1.0, abs=1e-9)
    assert sa_score_to_unit(99.0) == pytest.approx(0.0, abs=1e-9)
    # NaN -> 0.0
    assert sa_score_to_unit(float("nan")) == 0.0
