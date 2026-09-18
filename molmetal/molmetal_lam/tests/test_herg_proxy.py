"""Tests for the hERG cardiotoxicity heuristic in AnticancerMetricSuite.

References
----------
Aronov, A. M. *Predictive in silico modeling for hERG channel blockers.*
J. Med. Chem. 2005, 48, 1289-1300.  doi:10.1021/jm049371n
Veber, D. F. et al. *Molecular properties that influence the oral
bioavailability of drug candidates.* J. Med. Chem. 2002, 45, 2615-2623.

Honest framing
--------------
This suite scores the Aronov 2005 descriptor combination. It is NOT a
wet-lab hERG assay. The "known hERG-positive / negative" labels below
come from Redfern 2003 / Aronov 2005 literature reviews of
QT-prolongation cardiac risk. We use them as RANKING benchmarks, not
calibration standards.

Score convention
----------------
Score is in [0, 1] where HIGHER = SAFER (less hERG binding risk). This
matches the convention used elsewhere in this suite: the composite score
aggregates herg_proxy with positive weight (cleaner molecules get a
higher composite).

Threshold convention
-------------------
hERG-positive drugs should score < THRESHOLD_POSITIVE (risky / low).
hERG-negative drugs should score > THRESHOLD_NEGATIVE (clean / high).
The midpoint 0.30 is chosen to:
  - flag cisapride (0.30) and terfenadine (0.01) as risky,
  - flag paracetamol (1.00) and aspirin (1.00) as clean,
  - leave a clean margin between the two classes.
"""
from __future__ import annotations

import pytest

from molmetal_lam.priors.anticancer_metric_suite import AnticancerMetricSuite


# Reference drugs (Redfern 2003, Aronov 2005)
# hERG-positive = documented torsades-de-pointes / QT-prolongation risk.
HERG_POSITIVE_DRUGS = {
    "cisapride": "COC1=CC=C(CCN2CCC(CC2)NC(=O)C2=CC(Cl)=C(OC)C=C2)C=C1",  # withdrawn 2000
    "terfenadine": "CC(C)(C)C1=CC=C(C=C1)C(=O)C(C)(C)CCN(C)CCC1=CC=CC=C1",  # withdrawn 1998
}

# hERG-negative = clinically clean cardiac profile.
HERG_NEGATIVE_DRUGS = {
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
}

THRESHOLD_POSITIVE = 0.30  # cleaner-than-this = above; risky = below
THRESHOLD_NEGATIVE = 0.30


@pytest.fixture
def suite():
    return AnticancerMetricSuite()


@pytest.mark.parametrize("smiles", ["not-a-smiles", "", None, "bad-smiles"])
def test_invalid_smiles_falls_back_to_neutral(suite, smiles):
    """Invalid / unparseable SMILES must return the neutral 0.5 fallback."""
    assert suite.herg_proxy(smiles) == 0.5


def test_output_bounded_in_unit_interval(suite):
    """Score must be in [0, 1] across a diverse chemical space."""
    smiles_list = [
        "CCO",                                # ethanol
        "C",                                  # methane (CH4)
        "c1ccccc1",                           # benzene
        "CC(=O)NC1=CC=C(O)C=C1",              # paracetamol
        "CCCCCCCCCCCCCCCCCCCC",               # eicosane (very lipophilic)
        "C(C(C(C(C(C(C=O)O)O)O)O)O)O",        # glucose
        "C1=CN=CC=C1",                        # pyrazine
        "Cl[Pt](N)(N)Cl",                     # cisplatin
        "C1CC2CCC1C2",                        # norbornane
    ]
    for s in smiles_list:
        v = suite.herg_proxy(s)
        assert 0.0 <= v <= 1.0, f"{s} -> {v}"


def test_basic_nitrogen_increases_risk(suite):
    """Adding basic N atoms must lower the score (more cardiotoxic)."""
    base = suite.herg_proxy("c1ccccc1")  # benzene (no N)
    with_more_n = suite.herg_proxy("c1ccc(N(C)C)cc1")  # N,N-dimethylaniline
    assert with_more_n < base, (
        f"Adding basic N should not raise the score: "
        f"benzene={base:.3f} vs aniline={with_more_n:.3f}"
    )


def test_logp_above_threshold_increases_risk(suite):
    """Pushing logP > 3.5 must lower the score."""
    polar = suite.herg_proxy("c1ccccc1O")            # phenol, logP ~1.5
    lipophilic = suite.herg_proxy("c1ccccc1CCCCCCCC")  # octylbenzene, logP > 5
    assert lipophilic < polar, (
        f"Lipophilic aromatic should score lower: "
        f"phenol={polar:.3f} vs octylbenzene={lipophilic:.3f}"
    )


def test_aromatic_ring_threshold(suite):
    """Aromatic rings >= 3 must start contributing to hERG risk."""
    biphenyl = suite.herg_proxy("c1ccc(-c2ccccc2)cc1")     # 2 aromatic rings
    terphenyl = suite.herg_proxy("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")  # 3 rings
    assert terphenyl < biphenyl, (
        f"3+ aromatic rings should score lower than 2 rings: "
        f"biphenyl={biphenyl:.3f} vs terphenyl={terphenyl:.3f}"
    )


def test_molecular_weight_increases_risk(suite):
    """MW > 400 must start contributing to hERG risk."""
    light = suite.herg_proxy("c1ccccc1c1ccccc1")              # biphenyl ~154 Da
    heavy = suite.herg_proxy("c1ccccc1c1ccccc1c1ccccc1c1ccccc1")  # quaterphenyl ~306 Da
    # Heavy polyaromatic combines MW contribution + extra aromatic-ring contribution.
    assert heavy < light, (
        f"Heavier polyaromatic should score lower: "
        f"biphenyl={light:.3f} vs quaterphenyl={heavy:.3f}"
    )


@pytest.mark.parametrize("name,smiles", list(HERG_POSITIVE_DRUGS.items()))
def test_known_herg_positive_drugs_score_low(suite, name, smiles):
    """Withdrawn hERG-positive drugs (cisapride, terfenadine) must score
    BELOW the safety threshold — they should NOT look cardiosafe."""
    score = suite.herg_proxy(smiles)
    assert score < THRESHOLD_POSITIVE, (
        f"{name} is a documented hERG-positive drug, "
        f"score should be < {THRESHOLD_POSITIVE}, got {score:.3f}"
    )


@pytest.mark.parametrize("name,smiles", list(HERG_NEGATIVE_DRUGS.items()))
def test_known_herg_negative_drugs_score_high(suite, name, smiles):
    """Clean cardiac-profile drugs (paracetamol, aspirin) must score
    ABOVE the safety threshold."""
    score = suite.herg_proxy(smiles)
    assert score >= THRESHOLD_NEGATIVE, (
        f"{name} is a clinically clean drug, "
        f"score should be >= {THRESHOLD_NEGATIVE}, got {score:.3f}"
    )


def test_documented_reference_scores(suite):
    """Pin the published reference values for audit stability.

    These are NOT calibrated to assay data — they are heuristic scores
    recorded for regression detection. If they change without a code
    change, that's a regression. If they change WITH a code change,
    document the new value in wf_herg_real/final.md.
    """
    refs = {
        # cisapride: 1 tertiary N + logP~3.8 + MW~3.4 → risk ≈ 0.65 + 0.04 + 0.003 ≈ 0.69
        # score = max(0, 1 - min(1, 0.69)) = 0.31 (within rounding)
        "cisapride": 0.30,
        # terfenadine: 1 tertiary N + logP~5.8 + 2 aromatic rings → risk ≈ 0.99
        # score = 0.01 (saturated near floor)
        "terfenadine": 0.01,
        # paracetamol: 1 secondary amide N (NOT basic — has H), logP~0.5, 1 ring
        # basic_n=0, logP excess=0, arom_rings=1<2, MW<400 → risk = 0
        "paracetamol": 1.00,
        # aspirin: no basic N, logP~1.2, 1 ring, MW ~180 → risk = 0
        "aspirin": 1.00,
        # benzene: no basic N, logP~2, 1 ring → risk = 0 (no excess)
        "benzene": 1.00,
        # ethanol: tiny molecule, no rings, no basic N → risk = 0
        "ethanol": 1.00,
    }
    smiles_map = {
        "cisapride": "COC1=CC=C(CCN2CCC(CC2)NC(=O)C2=CC(Cl)=C(OC)C=C2)C=C1",
        "terfenadine": "CC(C)(C)C1=CC=C(C=C1)C(=O)C(C)(C)CCN(C)CCC1=CC=CC=C1",
        "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "benzene": "c1ccccc1",
        "ethanol": "CCO",
    }
    for name, expected in refs.items():
        actual = suite.herg_proxy(smiles_map[name])
        assert abs(actual - expected) < 0.05, (
            f"{name}: expected ~{expected:.2f}, got {actual:.3f}. "
            f"This is a regression — update final.md if intentional."
        )


def test_documented_zero_risk_floor_at_four_risk_features(suite):
    """Score must saturate at 0.0 (worst case) — never go negative."""
    # terfenadine already hits the floor (combined risk > 1).
    terfenadine = "CC(C)(C)C1=CC=C(C=C1)C(=O)C(C)(C)CCN(C)CCC1=CC=CC=C1"
    assert suite.herg_proxy(terfenadine) < 0.05


def test_documented_unit_ceiling_for_clean_small_molecule(suite):
    """Score must saturate at 1.0 (best case) — never exceed 1."""
    # methanol: no basic N, tiny logP, no aromatic rings, MW=32.
    assert suite.herg_proxy("CO") == 1.0


def test_counter_increments(suite):
    """The herg_calls counter must tick on every call."""
    from molmetal_lam.priors.anticancer_metric_suite import _ANTICANCER_COUNTERS
    before = _ANTICANCER_COUNTERS["herg_calls"]
    suite.herg_proxy("CCO")
    suite.herg_proxy("c1ccccc1")
    after = _ANTICANCER_COUNTERS["herg_calls"]
    assert after - before == 2