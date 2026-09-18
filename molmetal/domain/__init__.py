"""Domain dataclasses for the de novo + SBDD framework.

Three value types flow through every port:
- ``Pocket`` — the protein binding site
- ``Molecule`` — a 3D candidate ligand
- ``Complex`` — a (pocket, molecule) pair with optional score/pose

All types are ``dataclass(frozen=True)`` so that any modification produces a
new instance — useful for functional-style orchestration and easy hashing
in test fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# 1. Pocket
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pocket:
    """A protein binding pocket — the conditioning input for de novo design.

    Atoms are stored in a flat (N_atoms, 3) coordinate tensor.  Sidechain
    flexibility is supported via ``flexibility``; residues marked flexible
    have their sidechain atoms allowed to move during docking.
    """

    pdb_id: str
    coords: torch.Tensor           # (N_atoms, 3)
    atom_types: torch.Tensor       # (N_atoms,)  atomic numbers
    residue_ids: torch.Tensor      # (N_atoms,)  residue index in chain
    chain_ids: torch.Tensor        # (N_atoms,)  chain label
    mask: torch.Tensor             # (N_atoms,)  bool — True for pocket atoms
    center: torch.Tensor           # (3,)  binding site centroid (Å)
    radius: float = 6.0            # pocket sphere radius (Å)

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    def to(self, device: str | torch.device) -> "Pocket":
        return Pocket(
            pdb_id=self.pdb_id,
            coords=self.coords.to(device),
            atom_types=self.atom_types.to(device),
            residue_ids=self.residue_ids.to(device),
            chain_ids=self.chain_ids.to(device),
            mask=self.mask.to(device),
            center=self.center.to(device),
            radius=self.radius,
        )

    @classmethod
    def from_pdb_file(
        cls, path: str | Path, ligand_center: torch.Tensor, radius: float = 6.0
    ) -> "Pocket":
        """Parse a PDB file and extract atoms within ``radius`` Å of ``ligand_center``.

        Implementation lives in ``io/pdb_loader.py`` (Phase 0a).  Stub here so
        the dataclass can be imported without circular dependencies.
        """
        from molmetal.domain.io.pdb_loader import _load_pocket  # type: ignore

        return _load_pocket(path, ligand_center, radius)


# ---------------------------------------------------------------------------
# 2. Molecule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Molecule:
    """A 3D candidate ligand.

    Coordinates are stored as a (N_atoms, 3) tensor.  Bonds are stored as a
    (2, N_bonds) edge index plus per-bond bond-order labels so the structure
    can be re-serialised as RDKit Mol or SMILES downstream.
    """

    coords: torch.Tensor            # (N_atoms, 3)
    atom_types: torch.Tensor        # (N_atoms,)  atomic numbers
    bonds: torch.Tensor            # (2, N_bonds)  int64
    bond_types: torch.Tensor        # (N_bonds,)  int8 (1=single, 2=double, ...)
    formal_charges: torch.Tensor    # (N_atoms,)  int8
    smiles: str = ""                # canonical SMILES (empty until decoded)
    qed: float = 0.0
    sa_score: float = 0.0
    logp: float = 0.0

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.bonds.shape[1])

    def to(self, device: str | torch.device) -> "Molecule":
        return Molecule(
            coords=self.coords.to(device),
            atom_types=self.atom_types.to(device),
            bonds=self.bonds.to(device),
            bond_types=self.bond_types.to(device),
            formal_charges=self.formal_charges.to(device),
            smiles=self.smiles,
            qed=self.qed,
            sa_score=self.sa_score,
            logp=self.logp,
        )

    @classmethod
    def from_rdkit_mol(cls, mol, conformer_id: int = 0) -> "Molecule":
        from molmetal.domain.io.rdkit_io import _mol_to_molecule  # type: ignore

        return _mol_to_molecule(mol, conformer_id)

    def to_rdkit(self):
        from molmetal.domain.io.rdkit_io import _molecule_to_mol  # type: ignore

        return _molecule_to_mol(self)

    @classmethod
    def from_smiles(cls, smiles: str, embed_3d: bool = True) -> "Molecule":
        from molmetal.domain.io.rdkit_io import _smiles_to_molecule  # type: ignore

        return _smiles_to_molecule(smiles, embed_3d)


# ---------------------------------------------------------------------------
# 3. Complex
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Complex:
    """A (pocket, molecule) pair after docking — pose + optional score."""

    pocket: Pocket
    molecule: Molecule
    pose_confidence: float = 0.0         # 0..1
    vina_score: Optional[float] = None    # kcal/mol, lower = better
    binding_affinity: Optional[float] = None  # pIC50 / pKi, higher = better
    rmsd_to_reference: Optional[float] = None  # for benchmarking only


__all__ = ["Pocket", "Molecule", "Complex"]
