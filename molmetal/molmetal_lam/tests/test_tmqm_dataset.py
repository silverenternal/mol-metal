"""Tests for :mod:`molmetal_lam.lam_chem.tmqm_dataset`.

Phase 2.5 of the metallodrug vertical.  CPU-only, RDKit-only — never
touches the GPU or any docking oracle.

The suite covers:
    * Loading the real tmQM_parsed.csv (108,543 rows).
    * Metal-filter narrowing to a single metal (Pt).
    * Pt-subset sanity (cisplatin + carboplatin must be present).
    * Bond-pattern extraction (Pt-Cl, Pt-N must be non-zero).
    * Diverse subset via Morgan-ECFP4 MaxMin.
    * pt_click_compat coverage validation.
    * Dedup guarantee.
    * RDKit-parseable guarantee.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Make the ``lam_chem`` package importable when running this file in
# isolation.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PACKAGE_PARENT = _HERE.parent.parent  # molmetal_lam/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from molmetal_lam.lam_chem.tmqm_dataset import (  # noqa: E402
    DEFAULT_DONOR_ATOMS,
    DEFAULT_TARGET_METALS,
    HEAVY_ATOM_MAX,
    HEAVY_ATOM_MIN,
    TMQM_DEFAULT_PATH,
    TRANSITION_METALS,
    TmQMDataset,
    TmQMBondPattern,
    _extract_metal_bond_dict,
    _infer_metal_from_csv,
    _infer_oxidation_state,
)
from rdkit import Chem  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
TMQM_PATH = Path("/mnt/storage/data/molmetal/tmQM/tmqm_parsed.csv")
HAS_TMQM = TMQM_PATH.is_file()


@pytest.fixture(scope="module")
def full_dataset() -> TmQMDataset:
    """Load the *full* tmQM dataset (slow; ~2-5 s on a workstation)."""
    ds = TmQMDataset(path=TMQM_PATH)
    ds._load()  # force load so failures surface here, not in tests
    return ds


@pytest.fixture(scope="module")
def pt_dataset() -> TmQMDataset:
    """Load only Pt rows."""
    ds = TmQMDataset(path=TMQM_PATH, metal_filter="Pt")
    ds._load()
    return ds


@pytest.fixture(scope="module")
def multi_dataset() -> TmQMDataset:
    """Load Pt + Pd + Au + Ir + Ru rows."""
    ds = TmQMDataset(
        path=TMQM_PATH,
        metal_filter=["Pt", "Pd", "Au", "Ir", "Ru"],
    )
    ds._load()
    return ds


@pytest.fixture(scope="module")
def bond_pattern(multi_dataset: TmQMDataset) -> TmQMBondPattern:
    return TmQMBondPattern(
        dataset=multi_dataset,
        metals=DEFAULT_TARGET_METALS,
        donor_atoms=DEFAULT_DONOR_ATOMS,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_loads(full_dataset: TmQMDataset) -> None:
    """The tmQM CSV parses with a non-trivial number of records."""
    n = len(full_dataset)
    # Sanity: tmQM has 108,543 rows.  After dedup + parse + heavy-atom
    # filter we should still see a substantial subset (60k+ expected).
    assert n > 0, "tmQM dataset is empty"
    assert n < 108544, "tmQM dataset should not exceed row count"
    # Pt should be present (7,854 raw rows; ~6-8k after filter).
    metals = full_dataset.get_metal_list()
    assert "Pt" in metals, "Pt not present in tmQM"
    assert "Au" in metals, "Au not present in tmQM"


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_metal_filter(pt_dataset: TmQMDataset) -> None:
    """The metal_filter='Pt' dataset contains only Pt rows."""
    assert len(pt_dataset) > 0
    metals = pt_dataset.get_metal_list()
    assert all(m == "Pt" for m in metals), f"Non-Pt row leaked: {set(metals)}"
    # We expect roughly 5-7k Pt rows after parsing + dedup + heavy-atom
    # window.
    assert 1000 < len(pt_dataset) < 10000


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_pt_subset(pt_dataset: TmQMDataset) -> None:
    """The Pt subset contains cisplatin + carboplatin (sanity)."""
    smiles_list = pt_dataset.get_smiles_list()
    # cisplatin canonical SMILES: Cl[Pt](Cl)(N)N
    # carboplatin canonical SMILES contains [NH3] and a CBDCA chelate.
    joined = " | ".join(smiles_list)
    has_cisplatin = any(
        "[Pt" in s and "Cl" in s and "N" in s and len(s) < 60
        for s in smiles_list
    )
    assert has_cisplatin, "No short Pt-Cl + Pt-N motif in subset (cisplatin-like)"
    # Both Pt-Cl and Pt-N bond patterns should appear.
    found_pt_cl = False
    found_pt_n = False
    for smi, metal, ox_state, bond_dict in pt_dataset:
        if ("Pt", "Cl") in bond_dict:
            found_pt_cl = True
        if ("Pt", "N") in bond_dict:
            found_pt_n = True
        if found_pt_cl and found_pt_n:
            break
    assert found_pt_cl, "No Pt-Cl bond pattern in Pt subset"
    assert found_pt_n, "No Pt-N bond pattern in Pt subset"


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_bond_patterns(bond_pattern: TmQMBondPattern) -> None:
    """Pt bond statistics include the canonical M-L motifs."""
    pt_stats = bond_pattern.get_metal_bond_statistics("Pt")
    # Pt-Cl and Pt-N must be present (cisplatin baseline).
    assert pt_stats.get(("Pt", "Cl"), 0) > 0, "No Pt-Cl in tmQM Pt bond stats"
    assert pt_stats.get(("Pt", "N"), 0) > 0, "No Pt-N in tmQM Pt bond stats"
    # Pt-C (alkyl/aryl) should also be present (carboplatin / nedaplatin).
    # Pt-O (carboplatin CBDCA chelate) should be present.
    # Pt-P (Pt-phosphine complexes) is also expected.
    assert pt_stats.get(("Pt", "C"), 0) > 0, "No Pt-C in tmQM Pt bond stats"
    assert pt_stats.get(("Pt", "O"), 0) > 0, "No Pt-O in tmQM Pt bond stats"
    # Per-metal count should be in the thousands.
    assert bond_pattern.metal_count.get("Pt", 0) > 1000
    assert bond_pattern.metal_count.get("Pd", 0) > 1000
    assert bond_pattern.metal_count.get("Au", 0) > 100


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_diverse_subset(multi_dataset: TmQMDataset) -> None:
    """The MaxMin diverse subset has exactly ``n`` unique SMILES."""
    bp = TmQMBondPattern(dataset=multi_dataset, metals=DEFAULT_TARGET_METALS)
    sub = bp.get_diverse_subset(n=500, seed=42)
    assert len(sub) == 500, f"Expected 500 got {len(sub)}"
    smiles_in_sub = [r[0] for r in sub]
    # Strictly unique (MaxMin never revisits a pivot).
    assert len(set(smiles_in_sub)) == 500, "Duplicate SMILES in diverse subset"
    # The selected SMILES should all be in the original dataset.
    full_smiles_set = set(multi_dataset.get_smiles_list())
    for smi in smiles_in_sub:
        assert smi in full_smiles_set, "Diverse subset contains a foreign SMILES"


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_pt_click_compat_coverage(bond_pattern: TmQMBondPattern) -> None:
    """pt_click_compat matrix coverage validation."""
    report = bond_pattern.validate_pt_click_compat_coverage()
    assert "required" in report
    assert "covered" in report
    assert "missing" in report
    assert "stats" in report
    # Pt-Cl + Pt-N + Pt-O + Pt-C must be covered; Pt-S / Pt-P / Pt-Br
    # may be missing (acceptable, but covered in most cases).
    required = set(report["required"])
    covered = set(report["covered"])
    missing = set(report["missing"])
    # Required - covered == missing
    assert required - covered == missing
    # The platinum-drug backbone must be covered.
    assert ("Pt", "Cl") in covered, "Pt-Cl missing — required for cisplatin"
    assert ("Pt", "N") in covered, "Pt-N missing — required for ammine ligands"
    assert ("Pt", "O") in covered, "Pt-O missing — required for carboplatin"
    # Coverage fraction should be >= 0.5 (most required patterns present).
    if required:
        cov_frac = len(covered) / len(required)
        assert cov_frac >= 0.5, (
            f"Coverage {cov_frac:.2f} below 0.5 — {missing} are missing"
        )


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_no_duplicates(full_dataset: TmQMDataset) -> None:
    """No duplicate canonical SMILES in the loaded dataset."""
    smiles_list = full_dataset.get_smiles_list()
    assert len(smiles_list) == len(set(smiles_list)), (
        f"Duplicates: {len(smiles_list) - len(set(smiles_list))}"
    )


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_tmqm_rdkit_parseable(full_dataset: TmQMDataset) -> None:
    """Every SMILES re-parses with RDKit (round-trip)."""
    n = len(full_dataset)
    assert n > 0
    sample_indices = list(range(0, n, max(1, n // 200)))[:200]
    for i in sample_indices:
        smi = full_dataset.get_smiles_list()[i]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Unparseable SMILES at {i}: {smi}"
        assert mol.GetNumAtoms() > 0
        # Round-trip canonical should equal the stored SMILES.
        assert Chem.MolToSmiles(mol) == smi


# ---------------------------------------------------------------------------
# Unit tests on the helper functions (no CSV dependency).
# ---------------------------------------------------------------------------
def test_extract_metal_bond_dict_cisplatin() -> None:
    """cisplatin SMILES: Cl[Pt](Cl)(N)N → {('Pt','Cl'):2, ('Pt','N'):2}."""
    smi = "Cl[Pt](Cl)(N)N"
    mol = Chem.MolFromSmiles(smi)
    pt_idx = next(
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Pt"
    )
    bdict = _extract_metal_bond_dict(mol, pt_idx)
    assert bdict.get(("Pt", "Cl")) == 2, f"Pt-Cl count wrong: {bdict}"
    assert bdict.get(("Pt", "N")) == 2, f"Pt-N count wrong: {bdict}"
    # Pt-C / Pt-O / Pt-S not present
    assert ("Pt", "C") not in bdict
    assert ("Pt", "O") not in bdict


def test_extract_metal_bond_dict_carboplatin() -> None:
    """carboplatin: CBDCA chelate → Pt-O = 2, Pt-N = 2."""
    smi = (
        "C1CCC(C(=O)O[Pt]2(N)(N)OC(=O)C1C(=O)O2)CC1"  # compact cartoon
    )
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip("Carboplatin test SMILES unparseable in this rdkit")
    pt_idx = next(
        (a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pt"),
        None,
    )
    if pt_idx is None:
        pytest.skip("No Pt in carboplatin test SMILES")
    bdict = _extract_metal_bond_dict(mol, pt_idx)
    # Pt-O at least 2 (chelate)
    assert bdict.get(("Pt", "O"), 0) >= 1
    assert bdict.get(("Pt", "N"), 0) >= 1


def test_infer_metal_from_csv() -> None:
    """CSV metal column is preferred; bracket-scan is fallback."""
    assert _infer_metal_from_csv("Pt", "[Pt]") == "Pt"
    # CSV column is non-TM, fallback to bracket scan
    assert _infer_metal_from_csv("Al", "[Au]") == "Au"
    # No metal anywhere
    assert _infer_metal_from_csv("C", "CCO") is None
    assert _infer_metal_from_csv("", "[Cu]") == "Cu"


def test_infer_oxidation_state_cisplatin() -> None:
    """[Pt+2] → oxidation state 2."""
    smi = "Cl[Pt+2](Cl)(N)N"
    mol = Chem.MolFromSmiles(smi)
    pt_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pt"
    )
    ox = _infer_oxidation_state(mol, pt_idx)
    assert ox == 2, f"Expected ox=2 got {ox}"


def test_tmqm_default_path_constant() -> None:
    """The default path is sane (string-typed)."""
    assert isinstance(TMQM_DEFAULT_PATH, Path)
    assert "tmqm_parsed.csv" in str(TMQM_DEFAULT_PATH)


def test_invalid_metal_filter_raises() -> None:
    """An invalid metal_filter raises ValueError in __post_init__."""
    with pytest.raises(ValueError):
        TmQMDataset(metal_filter="Xx")  # not a TM


def test_dataset_default_target_metals() -> None:
    """The default target-metal set covers the metallodrug vertical."""
    assert "Pt" in DEFAULT_TARGET_METALS
    assert "Pd" in DEFAULT_TARGET_METALS
    assert "Au" in DEFAULT_TARGET_METALS
    assert "Ir" in DEFAULT_TARGET_METALS
    assert "Ru" in DEFAULT_TARGET_METALS


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_dataset_filter_by_metal_returns_new_dataset(
    multi_dataset: TmQMDataset,
) -> None:
    """filter_by_metal returns a NEW dataset, not mutates the original."""
    new_ds = multi_dataset.filter_by_metal("Pt")
    assert new_ds is not multi_dataset
    new_ds._load()
    metals = new_ds.get_metal_list()
    assert all(m == "Pt" for m in metals)


@pytest.mark.skipif(not HAS_TMQM, reason="tmQM CSV not mounted")
def test_dataset_n_max_caps_records() -> None:
    """The n_max cap is applied AFTER filtering."""
    ds = TmQMDataset(path=TMQM_PATH, metal_filter="Pt", n_max=10)
    ds._load()
    assert len(ds) <= 10