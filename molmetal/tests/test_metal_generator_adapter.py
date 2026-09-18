"""Tests for the MetalLigandGenerator port + MetalLigandAdapter stub.

Three required test cases (T4):
1. Protocol compliance — the adapter satisfies
   :class:`molmetal.ports.generators.MetalLigandGenerator`.
2. Multi-component round-trip — reconstructed SMILES parses cleanly
   via RDKit with the expected number of fragments.
3. Metal token injection — the reconstructed SMILES contains the
   metal token (``[Pt]`` / ``[Ru]`` …) and the metal is *not* lost
   downstream.
"""
from __future__ import annotations

import pytest
import torch

from molmetal.domain import Complex, Pocket
from molmetal.ports.generators import MetalLigandConfig, MetalLigandGenerator
from molmetal_lam.sbdd_env.metal_generator_adapter import (
    MetalLigandAdapter,
    reconstruct_complex_smiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pocket(n_atoms: int = 4) -> Pocket:
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    atom_types = torch.tensor(
        [6, 7, 8, 16] * (n_atoms // 4 + 1), dtype=torch.long,
    )[:n_atoms]
    residue_ids = torch.zeros(n_atoms, dtype=torch.long)
    chain_ids = torch.zeros(n_atoms, dtype=torch.long)
    mask = torch.ones(n_atoms, dtype=torch.bool)
    center = torch.zeros(3, dtype=torch.float32)
    return Pocket(
        pdb_id="metal_test",
        coords=coords,
        atom_types=atom_types,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        mask=mask,
        center=center,
        radius=6.0,
    )


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------
class TestProtocolCompliance:
    def test_isinstance_metal_ligand_generator(self):
        """MetalLigandAdapter must satisfy the Protocol at runtime."""
        adapter = MetalLigandAdapter()
        # @runtime_checkable Protocol → isinstance works.
        assert isinstance(adapter, MetalLigandGenerator)

    def test_required_attrs_present(self):
        adapter = MetalLigandAdapter()
        assert isinstance(adapter.name, str)
        assert adapter.name == "MetalLigandAdapter_v1"
        assert callable(adapter.setup)
        assert callable(adapter.generate)
        assert callable(adapter.get_metadata)

    def test_setup_and_metadata_no_error(self):
        adapter = MetalLigandAdapter()
        adapter.setup(device="cpu")
        meta = adapter.get_metadata()
        assert meta["name"] == "MetalLigandAdapter_v1"
        assert "engine" in meta
        assert "embed_3d" in meta
        assert "device" in meta

    def test_config_dataclass_construction(self):
        cfg = MetalLigandConfig(
            n_samples=3, oxidation_state=4, seed=7, embed_3d=False,
        )
        assert cfg.n_samples == 3
        assert cfg.oxidation_state == 4
        assert cfg.seed == 7
        assert cfg.embed_3d is False


# ---------------------------------------------------------------------------
# 2. Multi-component round-trip
# ---------------------------------------------------------------------------
class TestMultiComponentRoundTrip:
    def test_reconstruct_helper_str_input(self):
        # Convenience helper: str input → multi-component SMILES.
        smi = reconstruct_complex_smiles("N.N.Cl.Cl", "Pt", 2)
        assert smi == "N.N.Cl.Cl.[Pt]"

    def test_reconstruct_helper_list_input(self):
        # Convenience helper: list input → multi-component SMILES.
        smi = reconstruct_complex_smiles(["N", "N", "Cl", "Cl"], "Pt", 2)
        assert smi == "N.N.Cl.Cl.[Pt]"

    def test_adapter_returns_list_of_complexes(self):
        adapter = MetalLigandAdapter()
        pocket = _make_pocket(n_atoms=4)
        complexes = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=2, oxidation_state=2),
        )
        assert isinstance(complexes, list)
        assert len(complexes) == 2
        for c in complexes:
            assert isinstance(c, Complex)
            assert c.pocket is pocket
            assert c.molecule.smiles == "N.N.Cl.Cl.[Pt]"

    def test_adapter_smiles_round_trips_via_rdkit(self):
        """The reconstructed SMILES must re-parse cleanly via RDKit."""
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        complexes = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=1, oxidation_state=2),
        )
        assert len(complexes) == 1
        full_smi = complexes[0].molecule.smiles
        rdmol = Chem.MolFromSmiles(full_smi)
        assert rdmol is not None, f"RDKit failed to re-parse {full_smi!r}"
        frags = Chem.GetMolFrags(rdmol, asMols=True)
        # 4 ligands (N, N, Cl, Cl) + 1 metal → 5 RDKit fragments.
        assert len(frags) == 5

    def test_empty_ligand_returns_empty_list(self):
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        complexes = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="",
            config=MetalLigandConfig(n_samples=1, oxidation_state=2),
        )
        assert complexes == []


# ---------------------------------------------------------------------------
# 3. Metal token injection
# ---------------------------------------------------------------------------
class TestMetalTokenInjection:
    def test_metal_token_present_pt(self):
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        complexes = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=1, oxidation_state=2),
        )
        assert len(complexes) == 1
        smi = complexes[0].molecule.smiles
        assert "[Pt]" in smi, f"metal token missing from {smi!r}"
        assert smi.endswith(".[Pt]")

    def test_metal_token_present_ru(self):
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        complexes = adapter.generate(
            pocket, metal="Ru",
            ligand_smiles="N.N.N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=1, oxidation_state=2),
        )
        assert len(complexes) == 1
        smi = complexes[0].molecule.smiles
        assert "[Ru]" in smi
        assert smi.endswith(".[Ru]")
        # 6-coord octahedral → 6 fragments + 1 metal = 6 dots.
        assert smi.count(".") == 6

    def test_metal_token_present_ir(self):
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        # Ir(III) is octahedral (6-coord).  4 counted donors + 2 water
        # placeholders gives a clean 6-fragment complex.  Use bracketed
        # donors so they survive the counter-ion / solvent filter.
        complexes = adapter.generate(
            pocket, metal="Ir",
            ligand_smiles="N.N.[OH].[OH]",
            config=MetalLigandConfig(n_samples=1, oxidation_state=3),
        )
        assert len(complexes) == 1
        smi = complexes[0].molecule.smiles
        assert "[Ir]" in smi
        # 4 donors + 2 water placeholders ⇒ 6 fragments + 1 metal.
        assert smi.count("[OH2]") == 2
        assert smi.count(".") == 6

    def test_oxidation_state_drives_coordination(self):
        """Pt(II) is 4-coord square planar; Pt(IV) is 6-coord octahedral."""
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        # Same ligands, different oxidation states.
        pt2 = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=1, oxidation_state=2),
        )[0].molecule.smiles
        pt4 = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles="N.N.Cl.Cl",
            config=MetalLigandConfig(n_samples=1, oxidation_state=4),
        )[0].molecule.smiles
        assert "[Pt]" in pt2 and "[Pt]" in pt4
        # Pt(II): 4-coord → 4 fragments + 1 metal = 4 dots.
        assert pt2.count(".") == 4
        # Pt(IV): 6-coord → 4 fragments + 2 water = 6 fragments = 6 dots.
        assert pt4.count(".") == 6
        assert "[OH2]" in pt4

    def test_metal_token_carries_to_complex(self):
        """The metal token must survive end-to-end into Complex.molecule."""
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        adapter = MetalLigandAdapter()
        pocket = _make_pocket()
        complexes = adapter.generate(
            pocket, metal="Pt",
            ligand_smiles=["N", "N", "Cl", "Cl"],
            config=MetalLigandConfig(n_samples=3, oxidation_state=2),
        )
        assert len(complexes) == 3
        for c in complexes:
            assert c.molecule.smiles.endswith(".[Pt]")
            rdmol = Chem.MolFromSmiles(c.molecule.smiles)
            assert rdmol is not None
            # Confirm a Pt atom is present.
            pt_atoms = [
                a for a in rdmol.GetAtoms() if a.GetSymbol() == "Pt"
            ]
            assert len(pt_atoms) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
