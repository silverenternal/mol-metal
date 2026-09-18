"""Tests for the molmetal domain dataclasses and port Protocols.

Covers:
- Pocket / Molecule / Complex construction
- Molecule.from_smiles('CCO').to_rdkit() round-trip
- Pocket.from_pdb_file on a tiny PDB tempfile with 2 ATOM lines
- All 5 Protocols have the expected attrs
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import List, Optional, Tuple

import pytest
import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import (
    DesignLoop,
    DesignLoopConfig,
    DockingConfig,
    DockingEngine,
    GenerationConfig,
    MoleculeGenerator,
    PropertyPrediction,
    PropertyPredictor,
    ScoredCandidate,
    ScoringFunction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pocket(n_atoms: int = 4) -> Pocket:
    """Tiny Pocket for in-memory tests."""
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    atom_types = torch.tensor([6, 7, 8, 16] * (n_atoms // 4 + 1), dtype=torch.long)[:n_atoms]
    residue_ids = torch.zeros(n_atoms, dtype=torch.long)
    chain_ids = torch.zeros(n_atoms, dtype=torch.long)
    mask = torch.ones(n_atoms, dtype=torch.bool)
    center = torch.zeros(3, dtype=torch.float32)
    return Pocket(
        pdb_id="test",
        coords=coords,
        atom_types=atom_types,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        mask=mask,
        center=center,
        radius=6.0,
    )


def _make_molecule(n_atoms: int = 4) -> Molecule:
    """Tiny Molecule for in-memory tests."""
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    atom_types = torch.tensor([6, 6, 7, 8], dtype=torch.long)[:n_atoms]
    bonds = torch.zeros(2, 0, dtype=torch.long)
    bond_types = torch.zeros(0, dtype=torch.long)
    formal_charges = torch.zeros(n_atoms, dtype=torch.long)
    return Molecule(
        coords=coords,
        atom_types=atom_types,
        bonds=bonds,
        bond_types=bond_types,
        formal_charges=formal_charges,
    )


# ---------------------------------------------------------------------------
# 1. Domain construction
# ---------------------------------------------------------------------------
class TestDomainConstruction:
    def test_pocket_construction(self):
        p = _make_pocket(n_atoms=4)
        assert p.pdb_id == "test"
        assert p.n_atoms == 4
        assert p.coords.shape == (4, 3)
        assert p.atom_types.shape == (4,)
        assert p.center.shape == (3,)
        assert p.radius == 6.0
        assert p.mask.dtype == torch.bool
        # Pocket is frozen
        with pytest.raises(Exception):
            p.radius = 99.0  # type: ignore[misc]

    def test_molecule_construction(self):
        m = _make_molecule(n_atoms=4)
        assert m.n_atoms == 4
        assert m.n_bonds == 0
        assert m.coords.shape == (4, 3)
        assert m.bonds.shape == (2, 0)
        assert m.smiles == ""
        assert m.qed == 0.0

    def test_complex_construction(self):
        p = _make_pocket(n_atoms=2)
        m = _make_molecule(n_atoms=2)
        c = Complex(pocket=p, molecule=m)
        assert c.pose_confidence == 0.0
        assert c.vina_score is None
        assert c.binding_affinity is None
        assert c.rmsd_to_reference is None
        # All three layers are wired
        assert c.pocket is p
        assert c.molecule is m

    def test_complex_with_scores(self):
        p = _make_pocket(n_atoms=2)
        m = _make_molecule(n_atoms=2)
        c = Complex(
            pocket=p,
            molecule=m,
            pose_confidence=0.9,
            vina_score=-7.5,
            binding_affinity=8.2,
            rmsd_to_reference=1.3,
        )
        assert c.pose_confidence == pytest.approx(0.9)
        assert c.vina_score == pytest.approx(-7.5)
        assert c.binding_affinity == pytest.approx(8.2)
        assert c.rmsd_to_reference == pytest.approx(1.3)

    def test_pocket_to_device_round_trip(self):
        p = _make_pocket(n_atoms=4)
        p_cpu = p.to("cpu")
        assert p_cpu.coords.device.type == "cpu"
        assert p_cpu.center.device.type == "cpu"


# ---------------------------------------------------------------------------
# 2. RDKit round-trip
# ---------------------------------------------------------------------------
class TestRDKitRoundTrip:
    def test_smiles_to_rdkit_round_trip(self):
        m = Molecule.from_smiles("CCO", embed_3d=True)
        # Basic invariants on the parsed Molecule
        assert m.n_atoms > 0
        assert m.n_bonds > 0
        assert m.smiles  # canonical SMILES set during parsing
        # Round-trip via RDKit
        mol = m.to_rdkit()
        assert mol is not None
        # SanitizeMol → at least 1 conformer with 3D positions
        assert mol.GetNumConformers() >= 1
        conf = mol.GetConformer()
        assert conf.Is3D()
        # Strip Hs so the SMILES comparison matches the heavy-atom graph
        # of the original input ("CCO" with explicit Hs round-tripped
        # would otherwise compare against the H-explicit canonical form).
        from rdkit import Chem
        mol_no_h = Chem.RemoveHs(mol)
        rdkit_smiles = Chem.MolToSmiles(mol_no_h)
        ref = Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
        assert rdkit_smiles == ref


# ---------------------------------------------------------------------------
# 3. Pocket.from_pdb_file on a tiny tempfile
# ---------------------------------------------------------------------------
class TestPocketFromPDB:
    PDB_2_ATOMS = (
        "HEADER    TEST PDB\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )

    def test_from_pdb_file_two_atoms(self):
        # The 2 ATOM lines are placed within a 6.0 Å sphere of the origin.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as fh:
            fh.write(self.PDB_2_ATOMS)
            tmp_path = fh.name
        try:
            center = torch.zeros(3, dtype=torch.float32)
            pocket = Pocket.from_pdb_file(tmp_path, ligand_center=center, radius=6.0)
            assert pocket.pdb_id == os.path.splitext(os.path.basename(tmp_path))[0]
            assert pocket.n_atoms == 2
            assert pocket.coords.shape == (2, 3)
            assert pocket.mask.shape == (2,)
            assert pocket.center.shape == (3,)
            # Pocket radius preserved
            assert pocket.radius == 6.0
            # Atoms are within radius
            d0 = torch.norm(pocket.coords[0] - pocket.center).item()
            d1 = torch.norm(pocket.coords[1] - pocket.center).item()
            assert d0 < 6.0
            assert d1 < 6.0
        finally:
            os.unlink(tmp_path)

    def test_from_pdb_file_too_far_raises(self):
        # Centre the pocket at (100, 100, 100) so neither atom is within
        # the 1.0 Å sphere around the centroid.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as fh:
            fh.write(self.PDB_2_ATOMS)
            tmp_path = fh.name
        try:
            center = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float32)
            with pytest.raises(ValueError):
                Pocket.from_pdb_file(tmp_path, ligand_center=center, radius=1.0)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 4. All 5 Protocols have the expected attrs
# ---------------------------------------------------------------------------
class _StubGenerator(MoleculeGenerator):
    name = "stub_gen"

    def setup(self, device: str = "cpu") -> None:  # None
        return None

    def generate(self, pocket, config):  # type: ignore[override]
        return []

    def train_step(self, pocket, mols):  # type: ignore[override]
        return 0.0

    def get_metadata(self) -> dict:
        return {}


class _StubDocker(DockingEngine):
    name = "stub_docker"

    def setup(self, device: str = "cpu") -> None:  # None
        return None

    def dock(self, molecule, pocket, config):  # type: ignore[override]
        return []

    def get_metadata(self) -> dict:
        return {}


class _StubPredictor(PropertyPredictor):
    name = "stub_pred"

    def setup(self, device: str = "cpu") -> None:  # None
        return None

    def predict(self, molecule, complex=None):  # type: ignore[override]
        return PropertyPrediction()

    def get_metadata(self) -> dict:
        return {}


class _StubScorer(ScoringFunction):
    name = "stub_scorer"

    def setup(self) -> None:  # None
        return None

    def score(self, candidates):  # type: ignore[override]
        return []

    def get_metadata(self) -> dict:
        return {}


class _StubLoop(DesignLoop):
    def __init__(self, gen, docker, pred, scorer):
        self.generator = gen
        self.docker = docker
        self.predictor = pred
        self.scorer = scorer

    name = "stub_loop"

    def setup(self, device: str = "cpu") -> None:  # None
        return None

    def run(self, pocket, config):  # type: ignore[override]
        return []

    def get_history(self):
        return []


class TestProtocolAttrs:
    def test_molecule_generator_attrs(self):
        gen = _StubGenerator()
        assert hasattr(gen, "name")
        assert hasattr(gen, "setup")
        assert hasattr(gen, "generate")
        assert hasattr(gen, "train_step")
        assert hasattr(gen, "get_metadata")
        assert isinstance(gen.name, str)
        assert callable(gen.setup)
        assert callable(gen.generate)
        assert callable(gen.train_step)
        assert callable(gen.get_metadata)

    def test_docking_engine_attrs(self):
        dock = _StubDocker()
        assert hasattr(dock, "name")
        assert hasattr(dock, "setup")
        assert hasattr(dock, "dock")
        assert hasattr(dock, "get_metadata")
        assert isinstance(dock.name, str)
        assert callable(dock.setup)
        assert callable(dock.dock)
        assert callable(dock.get_metadata)

    def test_property_predictor_attrs(self):
        pred = _StubPredictor()
        assert hasattr(pred, "name")
        assert hasattr(pred, "setup")
        assert hasattr(pred, "predict")
        assert hasattr(pred, "get_metadata")
        assert isinstance(pred.name, str)
        assert callable(pred.setup)
        assert callable(pred.predict)
        assert callable(pred.get_metadata)

    def test_scoring_function_attrs(self):
        sc = _StubScorer()
        assert hasattr(sc, "name")
        assert hasattr(sc, "setup")
        assert hasattr(sc, "score")
        assert hasattr(sc, "get_metadata")
        assert isinstance(sc.name, str)
        assert callable(sc.setup)
        assert callable(sc.score)
        assert callable(sc.get_metadata)

    def test_design_loop_attrs(self):
        gen, dock, pred, sc = _StubGenerator(), _StubDocker(), _StubPredictor(), _StubScorer()
        loop = _StubLoop(gen, dock, pred, sc)
        assert hasattr(loop, "name")
        assert hasattr(loop, "setup")
        assert hasattr(loop, "run")
        assert hasattr(loop, "get_history")
        assert isinstance(loop.name, str)
        assert callable(loop.setup)
        assert callable(loop.run)
        assert callable(loop.get_history)

    def test_config_dataclasses_constructible(self):
        assert GenerationConfig(n_samples=2).n_samples == 2
        assert DockingConfig(n_poses=3).n_poses == 3
        cfg = DesignLoopConfig(n_iterations=2, n_samples_per_round=5, n_top_k=3)
        assert cfg.n_iterations == 2
        assert cfg.n_samples_per_round == 5
        assert cfg.n_top_k == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))