"""Tests for :mod:`molmetal_lam.lam_chem.cisplatin_resistance`.

Phase A / Stage 2.2 — chemoinformatic features for the cisplatin-
resistance classifier.

Cisplatin SMILES convention
---------------------------
The task spec writes cisplatin as ``Cl[Pt]([NH3])([NH3])[NH3]Cl``
but RDKit rejects that bracket form (N over-valence).  Two valid
alternatives are used throughout this suite:

* :data:`CISPLATIN_SMILES` = ``"N.N.Cl.Cl.[Pt]"`` — the project's
  canonical multi-component form (see :mod:`molmetal.data.metal_smiles`).
  RDKit parses it cleanly but the Pt has no bonds (each fragment is a
  separate component), so :func:`cis_trans_pattern` returns 0 for it.
* :data:`CISPALTIN_TIGHT_SMILES` = ``"[Pt](N)(N)(Cl)Cl"`` — single-
  component form.  RDKit parses it with all 4 Pt–donor bonds intact,
  so cis_trans detection works and the feature vector is fully populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make project importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem

from molmetal_lam.lam_chem.cisplatin_resistance import (
    CISPLATIN_SMILES,
    RESISTANCE_FEATURE_KEYS,
    compute_resistance_features,
)


# Single-component cisplatin — RDKit-parseable form with all Pt bonds intact.
# Used wherever the test needs the cis/trans detector or the bonded N-donor
# count to behave like a real coordination complex.
CISPLATIN_TIGHT_SMILES = "[Pt](N)(N)(Cl)Cl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_load(smiles: str):
    """Parse SMILES → Chem.Mol, skipping the test if RDKit rejects it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pytest.skip(f"RDKit could not parse {smiles!r}")
    return mol


# ---------------------------------------------------------------------------
# 1. Cisplatin — all features populated + sensible values
# ---------------------------------------------------------------------------
def test_cisplatin_smiles():
    """Cisplatin (single-component) yields the expected feature vector.

    Uses :data:`CISPLATIN_TIGHT_SMILES` because the bracket form
    ``Cl[Pt]([NH3])([NH3])[NH3]Cl`` (RDKit N over-valence) and the
    multi-component ``N.N.Cl.Cl.[Pt]`` (no Pt–ligand bonds) cannot
    both exercise the full feature set; the single-component form
    keeps all 4 Pt–donor bonds intact.

    For cisplatin we expect:

    * logP in [-5, +5] (RDKit reports ≈ +0.6 for the neutral complex).
    * thiol_count == 0  (no S atoms).
    * n_donor_count == 2 (two NH3 nitrogen donors).
    * cis_trans_pattern == +1 (cis square-planar — the active isomer).
    * mw ≈ 300.05 (Pt=195.08 + 2·N + 2·Cl + implicit H ≈ 300.05 g/mol).
    * tpsa, hba, hbd non-negative.
    """
    mol = _safe_load(CISPLATIN_TIGHT_SMILES)
    feats = compute_resistance_features(mol)

    assert -5.0 <= feats["logP"] <= 5.0
    assert feats["thiol_count"] == 0
    assert feats["n_donor_count"] == 2
    assert feats["cis_trans_pattern"] == 1
    assert 280.0 < feats["mw"] < 320.0
    assert 0.0 <= feats["tpsa"] <= 200.0
    assert feats["hba"] >= 2
    assert feats["hbd"] >= 2


def test_cisplatin_multicomponent_form():
    """Multi-component cisplatin parses but Pt has no bonds → cis_trans=0.

    Documents the limitation: when the SMILES splits the metal and
    ligands into disconnected fragments (``N.N.Cl.Cl.[Pt]``), the
    descriptor pipeline still runs (MW, logP, etc.) but
    :func:`compute_resistance_features` cannot detect the Pt
    coordination geometry, so ``cis_trans_pattern`` is 0.
    """
    mol = _safe_load(CISPLATIN_SMILES)
    feats = compute_resistance_features(mol)

    # Multi-component: Pt has 0 bonds in the RDKit graph.
    assert feats["cis_trans_pattern"] == 0
    # Other features remain populated.
    assert feats["thiol_count"] == 0
    assert 280.0 < feats["mw"] < 320.0


# ---------------------------------------------------------------------------
# 2. Feature keys — full set, fixed ordering
# ---------------------------------------------------------------------------
def test_feature_keys():
    """Every expected key is present and only those keys appear."""
    mol = _safe_load(CISPLATIN_SMILES)
    feats = compute_resistance_features(mol)
    assert set(feats.keys()) == set(RESISTANCE_FEATURE_KEYS)
    # All values must be real numbers.
    for k, v in feats.items():
        assert isinstance(v, (int, float)), f"{k} is not numeric"
        assert v == v, f"{k} is NaN"  # NaN != NaN


# ---------------------------------------------------------------------------
# 3. logP within a reasonable range for a small drug-like molecule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,lo,hi",
    [
        ("N.N.Cl.Cl.[Pt]", -5.0, 5.0),  # cisplatin
        ("CC(=O)Oc1ccccc1C(=O)O", -1.0, 5.0),  # aspirin
        ("CCO", -1.0, 1.5),  # ethanol
        ("Cc1ccc(O)cc1", 1.0, 4.0),  # p-cresol
    ],
)
def test_logP_range(smiles, lo, hi):
    """logP lies in a drug-like band for representative small molecules."""
    mol = _safe_load(smiles)
    feats = compute_resistance_features(mol)
    assert lo <= feats["logP"] <= hi, (
        f"logP for {smiles!r} = {feats['logP']:.3f}, " f"expected within [{lo}, {hi}]"
    )


# ---------------------------------------------------------------------------
# 4. Thiol counting
# ---------------------------------------------------------------------------
def test_thiol_count_ethanethiol():
    """Ethanethiol CC[SH] has exactly one -SH group."""
    mol = _safe_load("CCS")
    feats = compute_resistance_features(mol)
    assert feats["thiol_count"] == 1


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCS", 1),  # ethanethiol
        ("CC(S)CO", 1),  # 1-thioglycerol (SH explicit)
        ("CCO", 0),  # ethanol — no S
        ("c1ccc(S)cc1", 1),  # thiophenol
        ("CC(=O)NC(C)(C)S", 1),  # N-acetyl penicillamine thiol form
    ],
)
def test_thiol_count_table(smiles, expected):
    """A small cross-check across thiol / non-thiol molecules."""
    mol = _safe_load(smiles)
    feats = compute_resistance_features(mol)
    assert (
        feats["thiol_count"] == expected
    ), f"{smiles!r}: thiol_count={feats['thiol_count']}, expected {expected}"


# ---------------------------------------------------------------------------
# 5. Robustness — None mol and empty mol yield all-zero vectors
# ---------------------------------------------------------------------------
def test_none_mol_returns_zero_vector():
    """``None`` mol is graceful — never raises, returns zeros."""
    feats = compute_resistance_features(None)
    assert all(v == 0 for v in feats.values())


def test_empty_mol_returns_zero_vector():
    """Empty (atom-less) mol returns zeros without exception."""
    empty = Chem.Mol()
    feats = compute_resistance_features(empty)
    assert all(v == 0 for v in feats.values())


# ---------------------------------------------------------------------------
# 6. cis / trans discrimination on Pt square-planar pairs
# ---------------------------------------------------------------------------
def test_trans_platin_is_negative():
    """trans-Platin (Cl on opposite corners) reports ``-1``."""
    mol = _safe_load("[Pt](N)(Cl)(N)Cl")
    feats = compute_resistance_features(mol)
    assert feats["cis_trans_pattern"] == -1


def test_cis_cisplatin_is_positive():
    """cis-Cisplatin (Cl on adjacent corners) reports ``+1``."""
    mol = _safe_load("[Pt](N)(N)(Cl)Cl")
    feats = compute_resistance_features(mol)
    assert feats["cis_trans_pattern"] == 1


# ---------------------------------------------------------------------------
# 7. Hammett reduction-potential proxy is non-trivial on aromatics
# ---------------------------------------------------------------------------
def test_hammett_proxy_signs():
    """EWG substituents yield positive sum; EDG yield negative sum."""
    # p-nitroaniline: NH2 (-0.66) + NO2 (+0.78) = +0.12
    para_no2_nh2 = _safe_load("Nc1ccc([N+](=O)[O-])cc1")
    feats = compute_resistance_features(para_no2_nh2)
    assert feats["reduction_potential_proxy"] > 0

    # p-toluidine: NH2 + CH3 = -0.83 (EDG dominated)
    para_me_nh2 = _safe_load("Cc1ccc(N)cc1")
    feats2 = compute_resistance_features(para_me_nh2)
    assert feats2["reduction_potential_proxy"] < 0
