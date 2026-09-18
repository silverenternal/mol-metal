"""Tests for the DiffDockAdapter and EquiBindAdapter skeletons.

Phase-1 STUB-only coverage:

* Both adapters can be instantiated with no arguments (default paths).
* ``setup(device='cpu')`` is a no-op for STUB mode and returns ``None``.
* ``dock(...)`` returns exactly ``config.n_poses`` :class:`Complex`
  instances whose ``pocket`` field equals the input pocket.
* Each returned ``vina_score`` is in the realistic ``[-15, 0]`` kcal/mol
  range and ``pose_confidence`` is in ``[0, 1]``.
* Both adapters satisfy the :class:`DockingEngine` Protocol (we use
  ``isinstance`` against the runtime-checkable Protocol from ports).
* The ``name`` property matches the documented version string.
* ``get_metadata`` returns a ``dict`` with the expected keys.

These tests deliberately do NOT exercise the real DiffDock/EquiBind
model paths — those will be added in Phase 2 once checkpoints are
available.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from molmetal.adapters.diffdock import DiffDockAdapter
from molmetal.adapters.equibind import EquiBindAdapter
from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import DockingConfig, DockingEngine


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
# 1. DiffDockAdapter
# ---------------------------------------------------------------------------
class TestDiffDockAdapter:
    """Phase-1 STUB coverage for DiffDockAdapter."""

    def test_default_construction(self):
        adapter = DiffDockAdapter()
        assert adapter.name == "DiffDock_v1"
        # _loaded should be False in STUB mode.
        assert adapter._loaded is False

    def test_setup_runs_in_stub_mode(self):
        adapter = DiffDockAdapter()
        # setup() must return None in STUB mode (no model to load).
        result = adapter.setup(device="cpu")
        assert result is None
        assert adapter._device == "cpu"
        assert adapter._loaded is False

    def test_setup_records_device(self):
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        assert adapter._device == "cpu"

    def test_dock_returns_n_poses_complexes(self):
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=5)
        complexes: List[Complex] = adapter.dock(molecule, pocket, config)
        assert len(complexes) == 5
        for c in complexes:
            assert isinstance(c, Complex)
            # Each Complex must have the input pocket wired through.
            assert c.pocket is pocket
            # Molecule coordinates should be a tensor with the right
            # shape (n_atoms, 3).
            assert c.molecule.coords.shape == (4, 3)

    def test_vina_score_in_realistic_range(self):
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=20)
        complexes = adapter.dock(molecule, pocket, config)
        for c in complexes:
            assert c.vina_score is not None
            # Realistic vina range: very negative ≈ strong binding.
            assert -15.0 <= c.vina_score <= 0.0

    def test_pose_confidence_in_unit_range(self):
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=10)
        complexes = adapter.dock(molecule, pocket, config)
        for c in complexes:
            assert 0.0 <= c.pose_confidence <= 1.0

    def test_dock_changes_molecule_coordinates(self):
        """STUB must perturb the coords; otherwise the test suite would
        not be exercising any randomness and the adapter would silently
        be a no-op."""
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=3)
        complexes = adapter.dock(molecule, pocket, config)
        # Original molecule coords are all zeros; at least one Complex
        # must have non-zero coords because the STUB applies SE(3).
        any_perturbed = any(
            float(c.molecule.coords.abs().sum().item()) > 1e-6
            for c in complexes
        )
        assert any_perturbed, "STUB should perturb at least one pose."

    def test_satisfies_docking_engine_protocol(self):
        adapter = DiffDockAdapter()
        # DockingEngine is a typing.Protocol (PEP 544) but is not
        # @runtime_checkable, so we duck-type by attribute presence
        # (mirrors the convention used in test_ports.py).
        assert hasattr(adapter, "name")
        assert hasattr(adapter, "setup")
        assert hasattr(adapter, "dock")
        assert hasattr(adapter, "get_metadata")
        assert isinstance(adapter.name, str)
        assert callable(adapter.setup)
        assert callable(adapter.dock)
        assert callable(adapter.get_metadata)

    def test_metadata_shape(self):
        adapter = DiffDockAdapter()
        adapter.setup(device="cpu")
        meta = adapter.get_metadata()
        assert isinstance(meta, dict)
        assert meta["model"] == "DiffDock_v1"
        assert meta["type"] == "diffdock"
        assert meta["stub"] is True
        assert meta["device"] == "cpu"
        assert meta["checkpoint_path"] is None
        assert "repo_path" in meta
        assert "paper" in meta
        assert "arxiv" in meta


# ---------------------------------------------------------------------------
# 2. EquiBindAdapter
# ---------------------------------------------------------------------------
class TestEquiBindAdapter:
    """Phase-1 STUB coverage for EquiBindAdapter."""

    def test_default_construction(self):
        adapter = EquiBindAdapter()
        assert adapter.name == "EquiBind_v1"
        assert adapter._loaded is False

    def test_setup_runs_in_stub_mode(self):
        adapter = EquiBindAdapter()
        result = adapter.setup(device="cpu")
        assert result is None
        assert adapter._device == "cpu"
        assert adapter._loaded is False

    def test_setup_records_device(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        assert adapter._device == "cpu"

    def test_dock_returns_n_poses_complexes(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=7)
        complexes = adapter.dock(molecule, pocket, config)
        assert len(complexes) == 7
        for c in complexes:
            assert isinstance(c, Complex)
            assert c.pocket is pocket
            assert c.molecule.coords.shape == (4, 3)

    def test_vina_score_in_realistic_range(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=20)
        complexes = adapter.dock(molecule, pocket, config)
        for c in complexes:
            assert c.vina_score is not None
            assert -15.0 <= c.vina_score <= 0.0

    def test_pose_confidence_in_unit_range(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=10)
        complexes = adapter.dock(molecule, pocket, config)
        for c in complexes:
            assert 0.0 <= c.pose_confidence <= 1.0

    def test_dock_changes_molecule_coordinates(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=4)
        molecule = _make_molecule(n_atoms=4)
        config = DockingConfig(n_poses=3)
        complexes = adapter.dock(molecule, pocket, config)
        any_perturbed = any(
            float(c.molecule.coords.abs().sum().item()) > 1e-6
            for c in complexes
        )
        assert any_perturbed, "STUB should perturb at least one pose."

    def test_satisfies_docking_engine_protocol(self):
        adapter = EquiBindAdapter()
        # DockingEngine is a typing.Protocol (PEP 544) but is not
        # @runtime_checkable, so we duck-type by attribute presence
        # (mirrors the convention used in test_ports.py).
        assert hasattr(adapter, "name")
        assert hasattr(adapter, "setup")
        assert hasattr(adapter, "dock")
        assert hasattr(adapter, "get_metadata")
        assert isinstance(adapter.name, str)
        assert callable(adapter.setup)
        assert callable(adapter.dock)
        assert callable(adapter.get_metadata)

    def test_metadata_shape(self):
        adapter = EquiBindAdapter()
        adapter.setup(device="cpu")
        meta = adapter.get_metadata()
        assert isinstance(meta, dict)
        assert meta["model"] == "EquiBind_v1"
        assert meta["type"] == "equibind"
        assert meta["stub"] is True
        assert meta["device"] == "cpu"
        assert meta["checkpoint_path"] is None
        assert "repo_path" in meta
        assert "paper" in meta
        assert "arxiv" in meta


# ---------------------------------------------------------------------------
# 3. Cross-adapter sanity
# ---------------------------------------------------------------------------
class TestCrossAdapter:
    """Sanity checks that span both adapters."""

    @pytest.mark.parametrize(
        "factory",
        [DiffDockAdapter, EquiBindAdapter],
    )
    def test_each_adapter_dock_writes_pocket_field(self, factory):
        adapter = factory()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=2)
        molecule = _make_molecule(n_atoms=2)
        complexes = adapter.dock(molecule, pocket, DockingConfig(n_poses=2))
        assert all(c.pocket is pocket for c in complexes)

    @pytest.mark.parametrize(
        "factory",
        [DiffDockAdapter, EquiBindAdapter],
    )
    def test_each_adapter_zero_pose_edge_case(self, factory):
        """DockingConfig(n_poses=0) should return an empty list — STUB
        iterates ``range(0)`` which is empty, so this is a no-op."""
        adapter = factory()
        adapter.setup(device="cpu")
        pocket = _make_pocket(n_atoms=2)
        molecule = _make_molecule(n_atoms=2)
        complexes = adapter.dock(molecule, pocket, DockingConfig(n_poses=0))
        assert complexes == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
