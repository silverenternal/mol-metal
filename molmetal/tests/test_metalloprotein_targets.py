"""Tests for the metalloprotein target catalogue (F3 P1).

Two tests, per the task brief:

1. ``METALLOPROTEIN_PDBS`` (i.e. :data:`METALLOPROTEIN_TARGETS`)
   is non-empty and covers at least 8 families (MMP2/MMP9 from
   :mod:`mmp_targets` plus 6+ new families).
2. Each family has at least 3 PDB IDs, and the PDB IDs are valid
   4-character uppercase alphanumeric strings.

The tests are intentionally tolerant of the offline case — they
inspect the catalogue structure but do **not** require the
CrossDocked2020 archive (or PDBbind-CrossDocked-Core) to be present.
"""

from __future__ import annotations

from typing import List

import pytest

from molmetal.data.metalloprotein_targets import (
    ACE_TARGET,
    ADH_TARGET,
    CA2_TARGET,
    CDK2_TARGET,
    CYP3A4_TARGET,
    HDAC2_TARGET,
    METALLOPROTEIN_TARGETS,
    PKA_TARGET,
    SOD1_TARGET,
    MetalloproteinTarget,
    all_targets,
    combined_pdb_ids,
    families_by_metal,
    get_target,
)


# ---------------------------------------------------------------------------
# 1. Catalogue non-empty + ≥ 6 new families beyond MMP2/MMP9
# ---------------------------------------------------------------------------
class TestMetalloproteinCatalogue:
    def test_catalogue_non_empty(self) -> None:
        """Catalogue covers at least 8 families (MMP2/MMP9 + 6 new)."""
        assert isinstance(METALLOPROTEIN_TARGETS, dict)
        assert len(METALLOPROTEIN_TARGETS) >= 8, (
            f"Expected >= 8 metalloprotein families, got {len(METALLOPROTEIN_TARGETS)}: "
            f"{list(METALLOPROTEIN_TARGETS)}"
        )
        # MMP2 + MMP9 from mmp_targets.py are present
        assert "MMP2" in METALLOPROTEIN_TARGETS
        assert "MMP9" in METALLOPROTEIN_TARGETS
        # The 6 new families are present
        new_families = {"CA2", "ACE", "HDAC2", "PKA", "CDK2", "CYP3A4", "ADH1B", "SOD1"}
        missing = new_families - set(METALLOPROTEIN_TARGETS)
        assert not missing, f"Missing new families: {missing}"

    def test_each_family_has_at_least_three_pdbs(self) -> None:
        """Every family exposes ≥ 3 PDB IDs (brief requires ≥ 3)."""
        for name, target in METALLOPROTEIN_TARGETS.items():
            n_pdbs = len(target.pdb_ids)
            assert n_pdbs >= 3, (
                f"{name} has only {n_pdbs} PDB IDs (need >= 3): {target.pdb_ids}"
            )
            # PDB IDs are 4-char uppercase alphanumeric
            for pdb in target.pdb_ids:
                assert isinstance(pdb, str)
                assert len(pdb) == 4, f"{name} PDB {pdb!r} length != 4"
                assert pdb.isalnum(), f"{name} PDB {pdb!r} not alphanumeric"
                assert pdb == pdb.upper(), f"{name} PDB {pdb!r} not uppercase"

    def test_pdbs_2tuple(self) -> None:
        """Brief constraint: at least one family exposes ≥ 2 PDBs as a tuple."""
        # CA2 has 8 PDBs (see molmetal/data/metalloprotein_targets.py)
        assert isinstance(CA2_TARGET, MetalloproteinTarget)
        assert len(CA2_TARGET.pdb_ids) >= 2
        assert isinstance(CA2_TARGET.pdb_ids, tuple)

    def test_combined_pdb_ids_dedup(self) -> None:
        """combined_pdb_ids returns the union without duplicates."""
        pdbs = combined_pdb_ids()
        assert isinstance(pdbs, list)
        assert len(pdbs) == len(set(pdbs)), "combined_pdb_ids contains duplicates"
        assert len(pdbs) >= 40, (
            f"Expected >= 40 PDB IDs across 8 families (rough sanity), got {len(pdbs)}"
        )

    def test_metal_partition(self) -> None:
        """families_by_metal returns the expected partitions."""
        zn = families_by_metal("Zn")
        mg = families_by_metal("Mg")
        fe = families_by_metal("Fe")
        cu = families_by_metal("Cu")
        assert len(zn) >= 5, f"Expected >= 5 Zn families, got {len(zn)}"
        assert len(mg) >= 2, f"Expected >= 2 Mg families (PKA + CDK2), got {len(mg)}"
        assert len(fe) >= 1, f"Expected >= 1 Fe family (CYP3A4), got {len(fe)}"
        assert len(cu) >= 1, f"Expected >= 1 Cu family (SOD1), got {len(cu)}"
        # Family names — check the partition contains what we expect
        zn_names = {t.name for t in zn}
        assert "MMP2" in zn_names and "MMP9" in zn_names and "CA2" in zn_names
        mg_names = {t.name for t in mg}
        assert "PKA" in mg_names and "CDK2" in mg_names
        fe_names = {t.name for t in fe}
        assert "CYP3A4" in fe_names
        cu_names = {t.name for t in cu}
        assert "SOD1" in cu_names

    def test_lookup_and_all_targets(self) -> None:
        """get_target and all_targets are consistent."""
        for name in METALLOPROTEIN_TARGETS:
            t = get_target(name)
            assert t.name == name
            assert t.uniprot_id
            assert t.metal in ("Zn", "Mg", "Fe", "Cu")
            assert t.pdb_ids
        # all_targets returns a fresh dict (not a reference to the global)
        d1 = all_targets()
        d2 = all_targets()
        assert d1 is not d2
        assert d1.keys() == d2.keys() == set(METALLOPROTEIN_TARGETS)


# ---------------------------------------------------------------------------
# 2. Per-family shape checks
# ---------------------------------------------------------------------------
class TestMetalloproteinFamilyShape:
    """Per-family shape checks — short, but cover every new family."""

    @pytest.mark.parametrize(
        "target",
        [
            CA2_TARGET,
            ACE_TARGET,
            HDAC2_TARGET,
            PKA_TARGET,
            CDK2_TARGET,
            CYP3A4_TARGET,
            ADH_TARGET,
            SOD1_TARGET,
        ],
    )
    def test_target_shape(self, target: MetalloproteinTarget) -> None:
        assert isinstance(target, MetalloproteinTarget)
        assert target.name
        assert target.full_name
        assert target.uniprot_id
        assert target.ec_number
        assert target.metal in ("Zn", "Mg", "Fe", "Cu")
        assert target.coordination in (
            "tetrahedral", "octahedral", "trigonal bipyramidal", "square planar"
        )
        # PDB IDs are consistent: each appears in both binding_site_residues
        # and zn_triad_resnums
        for pdb in target.pdb_ids:
            assert pdb in target.binding_site_residues
            assert pdb in target.zn_triad_resnums
        # Each binding_site_residues entry has 6-8 resnums
        for pdb, residues in target.binding_site_residues.items():
            assert len(residues) >= 6, f"{target.name}/{pdb} has {len(residues)} residues"
        # Each zn_triad_resnums entry has 3 (chain, resnum) pairs
        for pdb, triad in target.zn_triad_resnums.items():
            assert len(triad) == 3, f"{target.name}/{pdb} triad length != 3"
            for chain_res in triad:
                assert len(chain_res) == 2
                chain, resnum = chain_res
                assert isinstance(chain, str) and chain
                assert isinstance(resnum, int) and resnum > 0
        # At least one known inhibitor
        assert len(target.known_inhibitors) >= 1
        # Function text is non-empty
        assert target.function


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))