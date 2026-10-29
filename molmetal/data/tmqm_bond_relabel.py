# SPDX-License-Identifier: Apache-2.0
"""Relabel ligand bond classes from tmQM's reference SMILES — MEASURED, REFUTED.

WHY THIS EXISTS (Wf-A113 T84)
-----------------------------
``_BO_TYPE_BANDS`` assigns ``BOND_AROMATIC`` one bond at a time from a
continuous Wiberg order, using a fixed 1.25-1.75 window.  Aromaticity is a
property of a RING, not of a bond, and the window does not respect that:
a delocalised six-ring whose C-C bonds sit near 1.4 (inside the window) and
whose C-N bonds sit near 1.2 (outside it) comes out as four aromatic bonds
and two single ones.  RDKit's kekulizer needs every aromatic atom to sit in a
closed conjugated system, so it refuses the whole molecule.

The obvious repair is the one MolDiff uses (Peng et al., ICML 2023, §3.1.1:
"four real bond types (single, double, triple, and aromatic bonds)" taken from
RDKit-parsed GEOM): take the bond classes from a molecule RDKit has already
validated.  tmQM ships such a molecule in ``tmqm_parsed.csv``'s ``smiles``
column for 93.2% of the six-metal cohort, and the atom ORDER does not match
the geometry, so the mapping is found by graph isomorphism on the ligand
subgraph (element-labelled, bond type ignored).

IT DOES NOT WORK, AND THE REASON IS INFORMATIVE
-----------------------------------------------
MEASURED on 900 X1/X2/X3 complexes
(``metrics/by_metric/wfa113_t84_label_relabel.json``):

* 98.5% of the reference SMILES are CHARGE-SEPARATED — carbonyls are written
  ``[C-]#[O+]``, halides ``[Br-]``, the metal ``[Re+]``.  The decoder's
  ``_assemble_mol`` builds every atom with formal charge 0 and no charge
  channel exists anywhere in ``AtomCloud``, so transferring the SMILES's bond
  orders without its charges produces ``C#O`` on neutral atoms: oxygen
  valence 3, ``AtomValenceException``, 372/812 complexes lost.
* On the 85 X1 complexes whose reference SMILES carries no formal charge at
  all, relabelling is not better than the shipped band: 69.4% versus 71.8%
  agreement with the reference's aromatic-ring count.

So the binding constraint is NOT the per-bond aromatic window.  It is that
the decoder cannot represent a formal charge — see T85.  This module is kept
because the isomorphism step is correct and reusable, and because a future
charge channel would need exactly it.  ``relabel_bond_classes_from_smiles``
returns ``None`` rather than a partially relabelled list, so no caller can
silently mix two label models.

Metal bonds are never relabelled: the SMILES writes them as dative or drops
them, and their classes come from the Wiberg order (T77).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "BOND_AROMATIC",
    "BOND_DOUBLE",
    "BOND_SINGLE",
    "BOND_TRIPLE",
    "relabel_bond_classes_from_smiles",
]

BOND_SINGLE = 1
BOND_DOUBLE = 2
BOND_TRIPLE = 3
BOND_AROMATIC = 4

_SYMBOL_OF_Z = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 33: "As", 34: "Se", 35: "Br", 53: "I",
    26: "Fe", 28: "Ni", 29: "Cu", 30: "Zn", 44: "Ru", 45: "Rh", 46: "Pd",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
}


def _rdkit_class(bond_type) -> int:
    from rdkit import Chem

    if bond_type == Chem.BondType.AROMATIC:
        return BOND_AROMATIC
    if bond_type == Chem.BondType.DOUBLE:
        return BOND_DOUBLE
    if bond_type == Chem.BondType.TRIPLE:
        return BOND_TRIPLE
    return BOND_SINGLE


def relabel_bond_classes_from_smiles(
    pairs: Sequence[Tuple[int, int, int]],
    zs: Sequence[int],
    smiles: str,
    metal_zs: Iterable[int],
) -> Optional[List[Tuple[int, int, int]]]:
    """Return ``pairs`` with ligand bond classes taken from ``smiles``.

    Parameters
    ----------
    pairs
        ``(i, j, class)`` per bond, indices into ``zs``.  Classes are the
        ``BOND_LABELS`` ints the loader produces from the Wiberg band.
    zs
        Atomic number per cloud atom, in the cloud's own order.
    smiles
        The corpus's reference SMILES for this complex.
    metal_zs
        Atomic numbers treated as metals.  Their bonds are passed through
        unchanged.

    Returns
    -------
    list or None
        ``None`` when the reference cannot be used — unparsable SMILES, a
        ligand subgraph that is not isomorphic to the reference's, or
        networkx being unavailable.  The caller must then keep the Wiberg
        band; returning a partially relabelled list would mix two models.
    """
    if not smiles or not pairs:
        return None
    from rdkit import Chem

    metals: Set[int] = set(int(z) for z in metal_zs)
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return None
    ref = Chem.RemoveHs(ref)

    lig_atoms = [
        a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() not in metals
    ]
    if not lig_atoms:
        return None
    lig_set = set(lig_atoms)

    cloud_lig = [i for i, z in enumerate(zs) if int(z) not in metals]
    cloud_lig_set = set(cloud_lig)
    cloud_edges = [
        (i, j) for i, j, _c in pairs
        if i in cloud_lig_set and j in cloud_lig_set
    ]
    ref_edges = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in ref.GetBonds()
        if b.GetBeginAtomIdx() in lig_set and b.GetEndAtomIdx() in lig_set
    ]
    if len(cloud_lig) != len(lig_atoms) or len(cloud_edges) != len(ref_edges):
        return None

    try:
        import networkx as nx
    except Exception:
        return None

    g = nx.Graph()
    g.add_nodes_from(
        (i, {"el": _SYMBOL_OF_Z.get(int(zs[i]), str(int(zs[i])))}) for i in cloud_lig
    )
    g.add_edges_from(cloud_edges)
    h = nx.Graph()
    h.add_nodes_from(
        (i, {"el": ref.GetAtomWithIdx(i).GetSymbol()}) for i in lig_atoms
    )
    h.add_edges_from(ref_edges)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        g, h, node_match=lambda a, b: a["el"] == b["el"]
    )
    try:
        mapping = next(matcher.isomorphisms_iter())
    except StopIteration:
        return None

    out: List[Tuple[int, int, int]] = []
    for i, j, cls in pairs:
        if i in cloud_lig_set and j in cloud_lig_set:
            bond = ref.GetBondBetweenAtoms(mapping[i], mapping[j])
            if bond is None:
                return None
            out.append((i, j, _rdkit_class(bond.GetBondType())))
        else:
            out.append((i, j, int(cls)))
    return out
