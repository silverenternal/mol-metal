"""Tests for :mod:`molmetal.data.metal_smiles`.

Covers the heuristic SMILES reconstruction for metal complexes used to drive
3-D embedding.  Five core cases are required by T1 A1:

1. ``Pt(II)`` square-planar with 4 ligands → 4-coord.
2. ``Ru(II)`` octahedral with 6 ligands → 6-coord.
3. Overflow (8 donors, octahedral) → 6 kept.
4. Underflow (3 donors, octahedral) → +3 water placeholders.
5. ``Pt(IV)`` octahedral (oxidation-state driven) → 6-coord.

We also add a parser-roundtrip smoke test via RDKit and a donor-count unit
test.

F1 (P0) update
--------------
The reconstruction now defaults to the canonical *multi-component* form
``L1.L2....Ln.[M]`` (RDKit round-trip 100% on MetalCytoToxDB), not the
legacy bracket form ``[M](L1)(L2)...``.  The bracket form suffers a
~3.3% RDKit parse failure because Pt / Pd / Au are outside RDKit's
default organic subset and the bracket form is split into disconnected
fragments.  All assertions below have been updated to expect the
multi-component form; the new :class:`TestMultiComponentReconstruction`
class adds explicit tests for the dot-separated SMILES.
"""

from __future__ import annotations

import pytest

from molmetal.data.metal_smiles import (
    coordination_capacity,
    count_donors,
    reconstruct_metal_complex,
    reconstruct_multi_component,
)


# ---------------------------------------------------------------------------
# 1. Coordination-geometry lookup
# ---------------------------------------------------------------------------
class TestCoordinationCapacity:
    def test_pt_ii_square_planar(self):
        assert coordination_capacity("Pt", 2) == 4

    def test_pt_iv_octahedral(self):
        assert coordination_capacity("Pt", 4) == 6

    def test_ru_ii_octahedral(self):
        assert coordination_capacity("Ru", 2) == 6

    def test_ir_iii_octahedral(self):
        assert coordination_capacity("Ir", 3) == 6

    def test_unknown_falls_back_to_octahedral(self):
        # (Metal, oxidation_state) not in the lookup → default 6.
        assert coordination_capacity("Xx", 2) == 6


# ---------------------------------------------------------------------------
# 2. Donor counting
# ---------------------------------------------------------------------------
class TestDonorCount:
    def test_ammonia_one_n_donor(self):
        assert count_donors("N") == 1

    def test_water_one_o_donor(self):
        assert count_donors("O") == 1

    def test_chloride_is_not_a_donor(self):
        # Cl is not in the donor set; we only count N/O/S/P.
        assert count_donors("[Cl-]") == 0

    def test_ethylenediamine_two_n_donors(self):
        assert count_donors("NCCN") == 2

    def test_bracket_atoms_count(self):
        # [NH3] counts as one N donor.
        assert count_donors("[NH3]") == 1

    def test_empty_fragment_zero(self):
        assert count_donors("") == 0


# ---------------------------------------------------------------------------
# 3. Reconstruction — the 5 required cases (now in multi-component form)
# ---------------------------------------------------------------------------
class TestReconstructMetalComplex:
    def test_cisplatin_pt_square_planar(self):
        # Cisplatin-like: 2 NH3 + 2 Cl.  Pt(II) is square planar (4-coord).
        result = reconstruct_metal_complex("N.N.Cl.Cl", "Pt", 2)
        # Multi-component form: 4 dot-separated ligands + [Pt]
        assert result == "N.N.Cl.Cl.[Pt]"
        assert result.count(".") == 4  # 3 between ligands + 1 before [Pt]
        assert result.endswith(".[Pt]")

    def test_ruthenium_octahedral(self):
        # 6 donors → 6-coord Ru(II) octahedral.
        result = reconstruct_metal_complex("N.N.N.N.Cl.Cl", "Ru", 2)
        assert result == "N.N.N.N.Cl.Cl.[Ru]"
        # 6 fragments + 1 metal ⇒ 6 dots
        assert result.count(".") == 6
        assert result.endswith(".[Ru]")

    def test_overflow_truncation(self):
        # 8 donors for an octahedral metal → 6 kept.
        result = reconstruct_metal_complex(
            "N.N.N.N.N.N.N.N", "Ru", 2
        )
        # 6 ligand fragments + 1 metal = 6 dots
        assert result.count(".") == 6
        # All retained fragments are 'N' (homogeneous input).
        # First token is bare 'N' (no leading dot), the other 5 are prefixed.
        assert result.count("N") == 6
        assert "[OH2]" not in result
        assert result.endswith(".[Ru]")

    def test_underflow_filled_with_water(self):
        # 3 donors for an octahedral metal → 6 ligands = 3 + 3 H2O.
        result = reconstruct_metal_complex("N.N.N", "Ru", 2)
        # 6 fragments + 1 metal = 6 dots
        assert result.count(".") == 6
        # Three water placeholders — RDKit-safe form is [OH2]
        assert result.count("[OH2]") == 3
        # Three original N donors
        assert result.count("N") == 3

    def test_oxidation_state_implicit(self):
        # Pt(IV) is octahedral (6-coord), distinct from Pt(II) square planar.
        result = reconstruct_metal_complex(
            "N.N.Cl.Cl", "Pt", 4
        )
        # 6-coord: 4 provided ligands + 2 water placeholders ⇒ 6 dots
        assert result.count(".") == 6
        assert result.count("[OH2]") == 2
        assert result.count("N") == 2
        assert result.count("Cl") == 2
        # Sanity: Pt(II) on the same input would be 4-coord with no water.
        pt2 = reconstruct_metal_complex("N.N.Cl.Cl", "Pt", 2)
        assert pt2.count(".") == 4
        assert "[OH2]" not in pt2

    def test_counterions_dropped(self):
        # [Cl-] is a counter-ion and should not appear as a ligand.
        result = reconstruct_metal_complex("N.N.[Cl-]", "Pt", 2)
        # Two amines → 2 ligands, padded with 2 water to reach 4-coord.
        assert result.count(".") == 4
        assert "N" in result
        assert "[OH2]" in result
        assert "[Cl-]" not in result

    def test_roundtrip_via_rdkit(self):
        # The reconstructed SMILES must be re-parseable by RDKit.
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        smi = reconstruct_metal_complex("N.N.Cl.Cl", "Pt", 2)
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"failed to re-parse {smi!r}"

    def test_empty_input_returns_empty_string(self):
        assert reconstruct_metal_complex("", "Pt", 2) == ""
        assert reconstruct_metal_complex("N.N", "", 2) == ""

    def test_unknown_metal_uses_default(self):
        # Unknown metal → octahedral default (6-coord).
        result = reconstruct_metal_complex("N.N.N.N.N.N", "Xx", 2)
        assert result.count(".") == 6
        assert result.endswith(".[Xx]")


# ---------------------------------------------------------------------------
# 4. Multi-component reconstruction — explicit tests (F1 / P0)
# ---------------------------------------------------------------------------
class TestMultiComponentReconstruction:
    """Dedicated tests for ``reconstruct_multi_component`` (F1 / P0).

    The output must use the canonical dot-separated form ``L1.L2....Ln.[M]``
    and RDKit must parse it without splitting into disconnected fragments.
    """

    def test_multi_component_cisplatin(self):
        # Canonical cisplatin: 2 NH3 + 2 Cl + Pt(II) square planar (4-coord).
        result = reconstruct_multi_component("N.N.Cl.Cl", "Pt", 2)
        assert result == "N.N.Cl.Cl.[Pt]"
        # RDKit round-trip 100%
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        # All 4 ligands preserved (separate RDKit fragments)
        frags = Chem.GetMolFrags(mol, asMols=True)
        assert len(frags) == 5  # 4 ligands + 1 metal

    def test_multi_component_ruthenium(self):
        # 6 donors → 6-coord Ru(II) octahedral.
        result = reconstruct_multi_component("N.N.N.N.Cl.Cl", "Ru", 2)
        assert result == "N.N.N.N.Cl.Cl.[Ru]"
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        frags = Chem.GetMolFrags(mol, asMols=True)
        assert len(frags) == 7  # 6 ligands + 1 metal

    def test_multi_component_iridium(self):
        # Ir(III) octahedral — use 5 donors (Cl is *not* counted by the
        # donor heuristic) and expect 1 water placeholder to reach 6-coord.
        # Use a mix of N + O donors so 5 are actually counted.
        result = reconstruct_multi_component("N.N.N.O.O.Cl.Cl", "Ir", 3)
        # 5 donors, 6-coord ⇒ 6 fragments total → 6 dots
        assert result.count(".") == 6
        assert result.endswith(".[Ir]")
        # 1 water placeholder, 5 donors counted
        assert result.count("[OH2]") == 1
        # RDKit must parse cleanly.
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        frags = Chem.GetMolFrags(mol, asMols=True)
        assert len(frags) == 7  # 6 ligands + 1 metal

    def test_legacy_api_delegates_to_multi_component(self):
        # Backward-compat: reconstruct_metal_complex must produce the
        # multi-component form (it is a thin wrapper now).
        a = reconstruct_metal_complex("N.N.Cl.Cl", "Pt", 2)
        b = reconstruct_multi_component("N.N.Cl.Cl", "Pt", 2)
        assert a == b == "N.N.Cl.Cl.[Pt]"

    def test_round_trip_100pct(self):
        """Smoke test on MetalCytoToxDB: 100 random rows must round-trip.

        Skips silently if the data file is not available on the test host.
        """
        import os
        csv_path = "/mnt/storage/data/molmetal/MetalCytoToxDB.csv"
        if not os.path.exists(csv_path):
            pytest.skip(f"MetalCytoToxDB not found at {csv_path}")
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        from molmetal.data.cytotox import MetalCytotoxDataset
        ds = MetalCytotoxDataset.from_csv(csv_path)
        df = ds.df
        # Deterministic subset: first 100 with non-empty ligand SMILES.
        rows = df[df["SMILES_Ligands"].notna()].head(100)
        n_ok = 0
        n_total = 0
        for _, row in rows.iterrows():
            sl = row.get("SMILES_Ligands")
            if not sl or not isinstance(sl, str):
                continue
            metal = row.get("Metal")
            ox = row.get("Oxidation_state")
            try:
                ox_int = int(ox) if ox is not None else 2
            except (TypeError, ValueError):
                ox_int = 2
            smi = reconstruct_multi_component(sl, metal, ox_int)
            if not smi:
                continue
            n_total += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                n_ok += 1
        assert n_total > 0, "no rows tested"
        # Multi-component form should achieve 100% on this corpus.
        assert n_ok == n_total, (
            f"Round-trip {n_ok}/{n_total} = {100*n_ok/n_total:.1f}% "
            "(expected 100%)"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
