"""Chemistry discriminator (Wf-A11 / eval wave).

Per-decoded-mol chemistry audit for metallo-drug priors:
  * ``dative_bond_rate``         — fraction of bonds that are dative
    (donor -> metal).  Metal = Pt/Ru/Ir/Au/Rh/Os/Pd/Cu/Zn.
  * ``metal_coord_1_rate``       — fraction of metal centres that are
    under-coordinated (coord <= 1, i.e. unbound or a poor pose).
  * ``is_aromatic_atom_count``   — count of aromatic atoms.

All metrics are computed from RDKit's bond / atom metadata.  Tolerant of
``None`` (parse failure).

Backward-compat
---------------
* :meth:`forward(mol, metal_z=78)` defaults to Pt (Z=78).  Callers can
  override per-run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import Chem

#: Set of metals whose bonds we count as "dative".
_METAL_SET: set = {8, 44, 45, 46, 76, 77, 78, 79}  # Cu, Ru, Rh, Pd, Os, Ir, Pt, Au


class ChemistryDiscriminator:
    """Chemistry audit over one decoded RDKit Mol."""

    name = "chemistry"

    def __init__(self, default_metal_z: int = 78) -> None:
        self.default_metal_z = int(default_metal_z)

    def forward(
        self,
        decoded_mol: Optional[Chem.Mol],
        metal_z: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """Return the chemistry audit dict."""
        if decoded_mol is None:
            return None
        z = int(metal_z) if metal_z is not None else self.default_metal_z
        try:
            n_atoms = decoded_mol.GetNumAtoms()
            if n_atoms <= 0:
                return None
            n_bonds = decoded_mol.GetNumBonds()
            n_dative = 0
            n_metal = 0
            n_coord_le_1 = 0
            for bond in decoded_mol.GetBonds():
                a, b = bond.GetBeginAtom(), bond.GetEndAtom()
                # Dative: any bond whose bond type is "dative" or where a
                # donor (Z=7,8,16) bonds to a metal (Z=z).
                is_dative = (
                    (a.GetAtomicNum() == z and b.GetAtomicNum() in (7, 8, 16))
                    or (b.GetAtomicNum() == z and a.GetAtomicNum() in (7, 8, 16))
                    or bond.GetBondType() == Chem.BondType.DATIVE
                )
                if is_dative:
                    n_dative += 1
            for a in decoded_mol.GetAtoms():
                if a.GetAtomicNum() in _METAL_SET:
                    n_metal += 1
                    n_nbr = sum(
                        1
                        for b in a.GetNeighbors()
                        if b.GetAtomicNum() != a.GetAtomicNum()
                    )
                    if n_nbr <= 1:
                        n_coord_le_1 += 1
            dative_bond_rate = float(n_dative) / float(n_bonds) if n_bonds > 0 else 0.0
            metal_coord_1_rate = float(n_coord_le_1) / float(n_metal) if n_metal > 0 else 0.0
            n_aromatic = sum(1 for a in decoded_mol.GetAtoms() if a.GetIsAromatic())
            return {
                "dative_bond_rate": dative_bond_rate,
                "metal_coord_1_rate": metal_coord_1_rate,
                "is_aromatic_atom_count": float(n_aromatic),
            }
        except Exception:
            return None

    def aggregate(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        """Mean-aggregate a list of chemistry dicts."""
        if not rows:
            return {
                "dative_bond_rate": 0.0,
                "metal_coord_1_rate": 0.0,
                "is_aromatic_atom_count": 0.0,
            }
        n = float(len(rows))
        keys = rows[0].keys()
        return {k: sum(float(r.get(k, 0.0)) for r in rows) / n for k in keys}


__all__ = ["ChemistryDiscriminator"]