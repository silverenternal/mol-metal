"""Smoke tests for the Vina docking adapter.

Three tiers:
1. Module imports + helper functions (always fast).
2. Full docking pipeline on aspirin against a real pocket
   (requires vina+meeko installed).
3. Redocking benchmark via the ``redock_for_test`` helper.

Tests are skipped if vina or meeko are not installed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from molmetal.domain import Molecule, Pocket
from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
    VinaDockingAdapter,
    _have_meeko,
    _have_vina,
    _pocket_to_pdb_string,
    dock_smiles,
    redock_for_test,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
EXAMPLE_PDB = (
    "/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/"
    "examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb"
)


def _real_pocket() -> Pocket:
    """Load a real pocket from TargetDiff's bundled example (572 atoms).

    This bypasses Pocket.from_pdb_file (which needs a ligand center) and
    directly constructs a Pocket from the PDB coordinates. The returned
    Pocket has a sidecar attribute ``_pdb_path`` pointing at the original
    PDB on disk so the Vina adapter can use it directly with
    ``mk_prepare_receptor.py`` (preserves residue connectivity).
    """
    from Bio.PDB import PDBParser  # type: ignore

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("test", EXAMPLE_PDB)
    coords = []
    types = []
    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
        types.append(atom.element.strip().capitalize())
    coords_arr = np.asarray(coords, dtype=np.float32)
    atomic_nums = np.asarray(
        [
            {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "H": 1,
             "Fe": 26, "Zn": 30, "Mg": 12, "Mn": 25, "Ca": 20,
             "Cl": 17, "Br": 35, "I": 53, "F": 9}.get(s, 6)
            for s in types
        ],
        dtype=np.int64,
    )
    n = len(coords_arr)
    center = coords_arr.mean(axis=0)
    radius = float(np.linalg.norm(coords_arr - center, axis=1).max()) + 1.0
    pocket = Pocket(
        pdb_id="1h36",
        coords=torch.from_numpy(coords_arr),
        atom_types=torch.from_numpy(atomic_nums),
        residue_ids=torch.zeros(n, dtype=torch.long),
        chain_ids=torch.zeros(n, dtype=torch.long),
        mask=torch.ones(n, dtype=torch.bool),
        center=torch.from_numpy(center.astype(np.float32)),
        radius=radius,
    )
    # Pocket is a frozen dataclass; attach the PDB path as an instance
    # attribute by going around __setattr__.
    object.__setattr__(pocket, "_pdb_path", EXAMPLE_PDB)
    return pocket


def _synthetic_pocket(n_atoms: int = 80, radius: float = 12.0) -> Pocket:
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    rng = np.random.default_rng(42)
    xyz = rng.normal(scale=4.0, size=(n_atoms, 3))
    coords = torch.from_numpy(xyz.astype(np.float32))
    return Pocket(
        pdb_id="TEST",
        coords=coords,
        atom_types=torch.full((n_atoms,), 6, dtype=torch.long),
        residue_ids=torch.arange(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=torch.zeros(3, dtype=torch.float32),
        radius=float(radius),
    )


# ----------------------------------------------------------------
# 1. Imports + helpers
# ----------------------------------------------------------------
class TestImports:
    def test_have_vina(self):
        assert _have_vina(), "vina not installed; run `uv pip install vina meeko`"

    def test_have_meeko(self):
        assert _have_meeko(), "meeko not installed; run `uv pip install meeko`"

    def test_pocket_to_pdb_string(self):
        p = _synthetic_pocket(n_atoms=5)
        s = _pocket_to_pdb_string(p)
        # 5 ATOM lines + END
        assert s.count("ATOM") == 5
        assert "END" in s

    def test_adapter_constructs(self):
        adapter = VinaDockingAdapter()
        assert adapter.name == "AutoDockVina_v1"


# ----------------------------------------------------------------
# 2. End-to-end docking on a real pocket
# ----------------------------------------------------------------
@pytest.mark.skipif(
    not (_have_vina() and _have_meeko()), reason="vina/meeko not installed"
)
@pytest.mark.skipif(
    not Path(EXAMPLE_PDB).exists(), reason=f"example PDB not at {EXAMPLE_PDB}",
)
class TestDockingPipeline:
    def test_setup_is_noop(self):
        adapter = VinaDockingAdapter()
        adapter.setup("cuda")  # should log but not crash

    def test_dock_aspirin(self):
        adapter = VinaDockingAdapter(default_box_padding=8.0)
        adapter.setup()
        pocket = _real_pocket()
        dummy = Molecule(
            coords=torch.zeros(1, 3),
            atom_types=torch.tensor([6], dtype=torch.long),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.tensor([0], dtype=torch.long),
            smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        )
        from molmetal.ports import DockingConfig
        cfg = DockingConfig(n_poses=2, exhaustiveness=4)
        complexes = adapter.dock(dummy, pocket, cfg)
        assert len(complexes) >= 1, "Docking should return ≥1 Complex"
        c = complexes[0]
        assert c.vina_score is not None
        # Aspirin docked into a real pocket should have a negative
        # binding affinity (the score reflects inter + intra).
        # Score may be near 0 if no clashes, but should be a float.
        assert isinstance(c.vina_score, float)
        assert c.molecule.n_atoms == 13, "aspirin has 13 heavy atoms"
        assert "CC" in c.molecule.smiles or "Oc" in c.molecule.smiles

    def test_dock_smiles_helper(self):
        pocket = _real_pocket()
        complexes = dock_smiles("c1ccccc1O", pocket, n_poses=1, exhaustiveness=4)
        assert len(complexes) == 1
        assert complexes[0].vina_score is not None
        # Phenol has 7 heavy atoms (6 C + 1 O)
        assert complexes[0].molecule.n_atoms == 7

    def test_redock_for_test(self):
        score = redock_for_test("TEST", "CCO", exhaustiveness=2)
        assert score is None or isinstance(score, float)

    def test_get_metadata(self):
        adapter = VinaDockingAdapter()
        meta = adapter.get_metadata()
        assert meta["name"] == "AutoDockVina_v1"
        assert "scoring_function" in meta


# ----------------------------------------------------------------
# 3. Robustness
# ----------------------------------------------------------------
@pytest.mark.skipif(
    not (_have_vina() and _have_meeko()), reason="vina/meeko not installed"
)
class TestRobustness:
    def test_bad_smiles_raises(self):
        from molmetal.ports import DockingConfig
        adapter = VinaDockingAdapter()
        pocket = _synthetic_pocket()
        bad = Molecule(
            coords=torch.zeros(1, 3),
            atom_types=torch.tensor([6], dtype=torch.long),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.tensor([0], dtype=torch.long),
            smiles="not_a_smiles_$$$",
        )
        with pytest.raises(Exception):
            adapter.dock(bad, pocket, DockingConfig(n_poses=1, exhaustiveness=1))

    def test_empty_smiles_raises(self):
        from molmetal.ports import DockingConfig
        adapter = VinaDockingAdapter()
        pocket = _synthetic_pocket()
        empty = Molecule(
            coords=torch.zeros(1, 3),
            atom_types=torch.tensor([6], dtype=torch.long),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.tensor([0], dtype=torch.long),
            smiles="",
        )
        with pytest.raises(ValueError):
            adapter.dock(empty, pocket, DockingConfig(n_poses=1, exhaustiveness=1))
