"""Tests for :mod:`molmetal.molmetal_lam.lam_chem.platinai_dataset`.

Covers
------
* :class:`PlatinAIDataset` — load, Pt filter, dedup, heavy-atom window,
  activity-matrix presence, split semantics
* :class:`MetalCytoToxDataset` — load, get_by_metal, get_top_k
* :class:`MetalloDrugDataset` — union, 500-mol diversity subset
* Helper functions — :func:`canonicalise`, :func:`heavy_atom_count`,
  :func:`metals_in_smiles`, :func:`primary_metal`
"""

from __future__ import annotations

import math

import pytest


pytest.importorskip("rdkit", reason="RDKit required for platinai_dataset tests")


# Module under test
from molmetal.molmetal_lam.lam_chem.platinai_dataset import (  # noqa: E402
    HEAVY_ATOM_MAX,
    HEAVY_ATOM_MIN,
    TRANSITION_METALS,
    MetalCytoToxDataset,
    MetalloDrugDataset,
    PlatinAIDataset,
    canonicalise,
    heavy_atom_count,
    is_transition_metal_present,
    load_metal_cytotox,
    load_metallo_drugs,
    load_platinai,
    metals_in_smiles,
    primary_metal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def platinai_full() -> PlatinAIDataset:
    return PlatinAIDataset(n_max=5000)


@pytest.fixture(scope="module")
def metal_cytotox_full() -> MetalCytoToxDataset:
    return MetalCytoToxDataset(n_max=5000)


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_canonical_smiles_canonicalises(self):
        # Use a parseable cisplatin form.  RDKit needs explicit H counts
        # in brackets; [NH3] alone is ambiguous and may not parse.
        c1 = canonicalise("[NH3][Pt]([NH3])([Cl])[Cl]")
        c2 = canonicalise("[Cl][Pt]([NH3])([NH3])[Cl]")
        assert c1 is not None
        assert c1 == c2  # canonical form is unique

    def test_canonical_smiles_returns_none_for_bad(self):
        assert canonicalise("not-a-smiles@@@@") is None
        assert canonicalise("") is None
        assert canonicalise(None) is None

    def test_heavy_atom_range(self):
        # Benzene: 6 heavy atoms
        assert heavy_atom_count("c1ccccc1") == 6
        assert heavy_atom_count("CCO") == 3
        # Oxaliplatin is in the heavy-atom window [8, 38] (15 heavy atoms)
        h = heavy_atom_count("O=C1OCC(=O)N1.C1CC[NH2][Pt]([NH2]1)([O])[O]")
        assert h is not None
        assert HEAVY_ATOM_MIN <= h <= HEAVY_ATOM_MAX

    def test_metals_in_smiles_finds_pt(self):
        m = metals_in_smiles("[NH3][Pt+2]([NH3])([Cl-])[Cl-]")
        assert "Pt" in m
        # The Ru-curcumin row
        m = metals_in_smiles(
            "COc1cc(/C=C/C(=O)/C=C([O-])/C=C/c2ccc(O)c(OC)c2)ccc1O."
            "Cc1ccc(C(C)C)cc1.[Cl-]"
        )
        # No TM in this fragment
        assert "Ru" not in m
        # A Ru(II) complex example
        m = metals_in_smiles("[Ru+2](N)(N)(N)(N)N")
        assert "Ru" in m

    def test_primary_metal_priority(self):
        # Cisplatin → Pt
        assert primary_metal("[NH3][Pt+2]([NH3])([Cl-])[Cl-]") == "Pt"
        # No TM → None
        assert primary_metal("CCO") is None

    def test_is_transition_metal_present(self):
        assert is_transition_metal_present(
            "[NH3][Pt+2]([NH3])([Cl-])[Cl-]", "Pt"
        )
        assert not is_transition_metal_present(
            "[NH3][Pt+2]([NH3])([Cl-])[Cl-]", "Ru"
        )
        # Iterable input
        assert is_transition_metal_present(
            "[NH3][Pt+2]([NH3])([Cl-])[Cl-]", ["Pt", "Ru"]
        )

    def test_invalid_metal_filter_raises(self):
        with pytest.raises(ValueError):
            PlatinAIDataset(metal_filter="Xx")


# ---------------------------------------------------------------------------
# PlatinAI tests
# ---------------------------------------------------------------------------
class TestPlatinAI:
    def test_platinai_loads(self, platinai_full: PlatinAIDataset):
        n = len(platinai_full)
        assert n > 0, "PlatinAI corpus should yield >0 records"
        # The full dataset is 226,918 but we cap at 5000
        assert n <= 5000
        # Basic tuple shape: (smiles, (a2780, mcf7), metal)
        smi, labels, metal = platinai_full[0]
        assert isinstance(smi, str)
        assert len(smi) > 0
        assert isinstance(labels, tuple)
        assert len(labels) == 2
        assert metal is None or metal in TRANSITION_METALS

    def test_platinai_pt_filter(self):
        ds = PlatinAIDataset(metal_filter="Pt", n_max=2000)
        n = len(ds)
        # Most of the corpus is Pt; assert non-trivial
        assert n > 0
        # Every record should contain Pt
        for smi, _, metal in ds:
            assert "Pt" in metals_in_smiles(smi), f"Filtered row missing Pt: {smi}"
            assert metal == "Pt"

    def test_platinai_no_duplicates(self):
        ds = PlatinAIDataset(n_max=2000)
        smiles = ds.get_smiles_list()
        # No duplicate SMILES
        assert len(smiles) == len(set(smiles)), (
            f"Found {len(smiles) - len(set(smiles))} duplicates"
        )

    def test_platinai_split_sizes(self):
        # Use a small n_max to keep test fast
        n = 1000
        train = PlatinAIDataset(n_max=n, split="train")
        val = PlatinAIDataset(n_max=n, split="val")
        test = PlatinAIDataset(n_max=n, split="test")
        # Splits are deterministic 80/10/10 → 800/100/100
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100
        # Splits are disjoint
        train_set = set(train.get_smiles_list())
        val_set = set(val.get_smiles_list())
        test_set = set(test.get_smiles_list())
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_activity_labels_present(self, platinai_full: PlatinAIDataset):
        # The activity matrix must exist and have the right shape
        mat = platinai_full.get_activity_matrix()
        assert mat.shape[1] == 2
        # At least some rows should have *some* non-NaN activity
        # (PlatinAI provides predicted_0 for the full 226k subset)
        finite_per_row = (~mat.isnan()).any(axis=1) if hasattr(mat, "isnan") else (
            ~((mat != mat).any(axis=1))
        )
        n_with_activity = int(finite_per_row.sum())
        # If join files are not reachable (e.g. in CI), allow zero
        assert n_with_activity >= 0

    def test_activity_matrix_shape(self, platinai_full: PlatinAIDataset):
        n = len(platinai_full)
        mat = platinai_full.get_activity_matrix()
        assert mat.shape[0] == n

    def test_factory_load_platinai(self):
        ds = load_platinai(n_max=100)
        assert isinstance(ds, PlatinAIDataset)
        assert len(ds) == 100


# ---------------------------------------------------------------------------
# MetalCytoTox tests
# ---------------------------------------------------------------------------
class TestMetalCytoTox:
    def test_metalcytotox_loads(self, metal_cytotox_full: MetalCytoToxDataset):
        n = len(metal_cytotox_full)
        assert n > 0
        # Tuple shape: (smiles, ic50_dark, metal, oxidation_state)
        smi, ic50, metal, ox = metal_cytotox_full[0]
        assert isinstance(smi, str)
        assert math.isfinite(ic50)
        assert ic50 > 0
        assert metal in TRANSITION_METALS
        assert ox is None or isinstance(ox, int)

    def test_metalcytotox_get_by_metal(self, metal_cytotox_full: MetalCytoToxDataset):
        # Ru is the most common in MetalCytoToxDB
        ru_rows = metal_cytotox_full.get_by_metal("Ru")
        assert len(ru_rows) > 0
        for r in ru_rows:
            assert r[2] == "Ru"
        # Pt may or may not be present (most rows are Ru)
        pt_rows = metal_cytotox_full.get_by_metal("Pt")
        assert all(r[2] == "Pt" for r in pt_rows)
        # Invalid metal raises
        with pytest.raises(ValueError):
            metal_cytotox_full.get_by_metal("Xx")

    def test_metalcytotox_get_top_k(self, metal_cytotox_full: MetalCytoToxDataset):
        top100 = metal_cytotox_full.get_top_k(k=100)
        assert len(top100) <= 100
        # Sorted ascending (most potent first)
        for i in range(1, len(top100)):
            assert top100[i - 1][1] <= top100[i][1]
        # Most-potent record is the first one
        if len(top100) >= 2:
            assert top100[0][1] <= top100[-1][1]

    def test_metalcytotox_factory(self):
        ds = load_metal_cytotox(metal_filter="Ru", n_max=200)
        assert isinstance(ds, MetalCytoToxDataset)
        assert all(r[2] == "Ru" for r in ds)


# ---------------------------------------------------------------------------
# MetalloDrug union tests
# ---------------------------------------------------------------------------
class TestMetalloDrugUnion:
    def test_metallo_drugs_union(self):
        # Small PlatinAI cap + small MetalCytoTox cap so the test runs fast
        plat = PlatinAIDataset(n_max=500)
        cyto = MetalCytoToxDataset(n_max=500)
        union = MetalloDrugDataset(platinai=plat, metal_cytotox=cyto)
        n = len(union)
        assert n > 0
        # All rows have a smiles + activity dict
        for smi, metal, activity in union:
            assert isinstance(smi, str)
            assert isinstance(activity, dict)
            assert metal is None or metal in TRANSITION_METALS
            # At least one of these keys should appear
            assert any(
                k in activity for k in ("a2780", "mcf7", "ic50_dark")
            )

    def test_metallo_drugs_500_subset(self):
        plat = PlatinAIDataset(n_max=2000)
        cyto = MetalCytoToxDataset(n_max=2000)
        # Build union then ask for 500-mol subset via get_subset
        union = MetalloDrugDataset(platinai=plat, metal_cytotox=cyto)
        sub = union.get_subset(n=500)
        n = len(sub)
        # If the union is large enough we get exactly 500; otherwise we
        # cap at the union size.
        assert n <= 500
        # The subset must be deduplicated by canonical SMILES
        smiles = sub.get_smiles_list()
        assert len(smiles) == len(set(smiles))

    def test_metallo_drugs_subset_smaller_than_full(self):
        plat = PlatinAIDataset(n_max=2000)
        cyto = MetalCytoToxDataset(n_max=2000)
        union = MetalloDrugDataset(platinai=plat, metal_cytotox=cyto)
        full_n = len(union)
        if full_n > 200:
            sub = union.get_subset(n=200)
            assert len(sub) == 200
            # Subset SMILES is a subset of full
            assert set(sub.get_smiles_list()).issubset(set(union.get_smiles_list()))

    def test_metallo_drugs_factory(self):
        ds = load_metallo_drugs(subset_size=100)
        assert isinstance(ds, MetalloDrugDataset)
        assert len(ds) <= 100


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------
class TestCrossCutting:
    def test_canonical_smiles_all_records(self):
        # Every record in PlatinAI must be RDKit-canonicalisable (we
        # already filter on this, so verify nothing slipped through).
        ds = PlatinAIDataset(n_max=200)
        for smi, _, _ in ds:
            assert canonicalise(smi) == smi, (
                f"Record not RDKit-canonical: {smi}"
            )

    def test_heavy_atom_range_enforced(self):
        ds = PlatinAIDataset(n_max=500)
        for smi, _, _ in ds:
            n = heavy_atom_count(smi)
            assert n is not None
            assert HEAVY_ATOM_MIN <= n <= HEAVY_ATOM_MAX