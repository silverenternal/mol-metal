"""RDKit ⇄ Molecule dataclass converters.

The Molecule dataclass stores flat tensors (coords, atom_types, bonds,
bond_types, formal_charges).  This module serialises them to/from the
RDKit ``Chem.Mol`` representation so we can use RDKit's 3D embedding,
QED/SA/logP calculators, and SMILES canonicalisation.

Notes
-----
* ETKDGv3 + MMFF94 is the default 3D embedding strategy (Phase 0a).
* Bonds are stored as (2, N_bonds) edge_index with bond-types in
  [1=single, 2=double, 3=triple, 12=aromatic] (RDKit conventions).
* Heavy import of rdkit is lazy so the rest of molmetal/ stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from molmetal.domain import Molecule

if TYPE_CHECKING:
    from rdkit import Chem  # noqa: F401


# ---------------------------------------------------------------------------
# Public converters (callable from Molecule.from_smiles / .from_rdkit_mol)
# ---------------------------------------------------------------------------
def _mol_to_molecule(mol, conformer_id: int = 0) -> Molecule:
    """Convert an RDKit ``Chem.Mol`` (with at least one conformer) into a Molecule.

    Falls back to embedding a 3D conformer with ETKDGv3 + MMFF94 if the
    input conformer is 2D or absent.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    # Sanitise / add Hs if needed
    if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D() is False:
        mol = Chem.Mol(mol)
        mol = _ensure_3d(mol)

    conf = mol.GetConformer(conformer_id)
    n_atoms = mol.GetNumAtoms()
    coords = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(n_atoms)],
        dtype=torch.float32,
    )
    atom_types = torch.tensor(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
    )
    formal_charges = torch.tensor(
        [a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.long
    )

    # Edge index + bond types
    src_list, dst_list, bt_list = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _rdkit_bond_type_to_int(bond.GetBondType())
        # Store both directions so the EGNN edge_index is symmetric.
        src_list.extend([i, j])
        dst_list.extend([j, i])
        bt_list.extend([bt, bt])
    if src_list:
        bonds = torch.tensor([src_list, dst_list], dtype=torch.long)
        bond_types = torch.tensor(bt_list, dtype=torch.long)
    else:
        bonds = torch.zeros(2, 0, dtype=torch.long)
        bond_types = torch.zeros(0, dtype=torch.long)

    smiles = Chem.MolToSmiles(mol) if mol.GetNumAtoms() > 0 else ""

    return Molecule(
        coords=coords,
        atom_types=atom_types,
        bonds=bonds,
        bond_types=bond_types,
        formal_charges=formal_charges,
        smiles=smiles,
    )


def _molecule_to_mol(molecule: Molecule):
    """Convert a Molecule back into an RDKit ``Chem.Mol`` with 3D coords."""
    from rdkit import Chem
    from rdkit import Geometry

    rw = Chem.RWMol()
    n = molecule.n_atoms
    # Add atoms
    for i in range(n):
        z = int(molecule.atom_types[i].item())
        atom = Chem.Atom(z)
        atom.SetFormalCharge(int(molecule.formal_charges[i].item()))
        rw.AddAtom(atom)

    # Add bonds (deduplicate: stored as undirected pairs, both directions)
    bond_set: set[tuple[int, int]] = set()
    if molecule.n_bonds > 0:
        for k in range(molecule.n_bonds):
            i, j = (
                int(molecule.bonds[0, k].item()),
                int(molecule.bonds[1, k].item()),
            )
            key = (min(i, j), max(i, j))
            if key in bond_set or i == j:
                continue
            bond_set.add(key)
            bt = _int_to_rdkit_bond_type(int(molecule.bond_types[k].item()))
            rw.AddBond(i, j, bt)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(n)
    for i in range(n):
        p = Geometry.Point3D(
            float(molecule.coords[i, 0].item()),
            float(molecule.coords[i, 1].item()),
            float(molecule.coords[i, 2].item()),
        )
        conf.SetAtomPosition(i, p)
    mol.AddConformer(conf, assignId=True)
    return mol


def _smiles_to_molecule(smiles: str, embed_3d: bool = True) -> Molecule:
    """Build a Molecule from a SMILES string, embedding 3D if requested."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if embed_3d:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            # Fallback to 2D coords if 3D embedding fails (rare for valid SMILES)
            AllChem.Compute2DCoords(mol)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # MMFF may fail for some organometallics; that's OK, we keep
            # the ETKDGv3 geometry.
            pass
    return _mol_to_molecule(mol, conformer_id=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_3d(mol, embed_3d: bool = True):
    """Add Hs if missing and embed a 3D conformer if none exists."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol.GetNumAtoms() > 0 and mol.GetAtomWithIdx(0).GetNoImplicit():
        # already has explicit Hs (caller's responsibility)
        pass
    if mol.GetNumHs() == 0:
        mol = Chem.AddHs(mol)
    if embed_3d and (mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D()):
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    return mol


# RDKit bond-type -> int (we follow RDKit's order: SINGLE=1, DOUBLE=2, …)
_RD_BOND_TYPE_TO_INT = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 12,
}
_INT_TO_RD_BOND_TYPE = {v: k for k, v in _RD_BOND_TYPE_TO_INT.items()}


def _rdkit_bond_type_to_int(bt) -> int:
    name = bt.name if hasattr(bt, "name") else str(bt)
    return _RD_BOND_TYPE_TO_INT.get(name, 1)


def _int_to_rdkit_bond_type(b: int):
    from rdkit import Chem
    name = _INT_TO_RD_BOND_TYPE.get(int(b), "SINGLE")
    return getattr(Chem.rdchem.BondType, name)
