"""Metal-aware graph construction (I1b) — ligand graph + a connected metal node.

Background / why this exists
============================
``SMILES_Ligands`` carries NO metal centre (G2 MEASURED: 0/19135 Ru rows).  So
every existing model — including chemprop v2 — is metal-blind.  I1 makes the
metal a first-class graph object, but G2 also showed that
``metal_smiles.reconstruct_multi_component`` emits a *disconnected* metal
(``metal_disconnected_rate=1.0``): feeding that to a GNN gives an isolated node
message passing cannot reach.  I1b is the missing wiring step: attach the metal
to the ``cap`` best donor atoms so it participates in message passing.

Design
======
This module produces, for one row:

    ligand graph (as ``molmetal.baselines.dmpnn.mol_to_graph``)  +
    1 appended metal node  +  ``cap`` undirected coordination edges (2 directed)

and a separate ``metal_feats`` vector (``METAL_FEAT_DIM``) the model projects.

The metal node's atom features reuse the FREE trailing slots of the 39-dim
``_atom_features`` layout (indices 30-38 are always 0 for real atoms — the
layout uses 30 dims then zero-pads): slot 30 is an ``is_metal`` marker, 31-35
the 5-metal identity one-hot, 36-38 the oxidation state / charge / coordination
number (scaled).  This keeps ``ATOM_FEATURE_DIM=39`` unchanged (no compat break).

Reuse (H5)
==========
``coordination_capacity`` (metal_smiles) sizes the wiring; ``classify_coord_geometry``
+ ``one_hot_coord_geometry`` (coord_geometry) build the geometry block;
``mol_to_graph`` (baselines.dmpnn) builds the ligand graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    _bond_features,
    _atom_features,
)
from molmetal.data.coord_geometry import (
    COORD_GEOMETRY_VOCAB,
    classify_coord_geometry,
    one_hot_coord_geometry,
)
from molmetal.data.metal_smiles import coordination_capacity

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METALS5: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
METAL_Z = {"Ru": 44, "Rh": 45, "Ir": 77, "Os": 76, "Re": 75}

#: Atom-feature slots reserved for the metal node (free in the 39-dim layout).
_METAL_MARKER_SLOT = 30
_METAL_ID_OFFSET = 31  # 31..35 = identity one-hot (len(METALS5))
_METAL_OS_SLOT = 36
_METAL_CHARGE_SLOT = 37
_METAL_CAP_SLOT = 38

#: Metal-feature vector layout: identity(5) + [os, charge, cap, d_electrons](4)
#: + geometry one-hot(9).
N_GEOM = len(COORD_GEOMETRY_VOCAB)
METAL_FEAT_DIM = len(METALS5) + 4 + N_GEOM

#: I2 — rich transferable-descriptor block appended after ``METAL_FEAT_DIM``.
#: Every field is an RDKit-verifiable periodic/size property (no fabricated
#: literature constants): row, group, scaled atomic weight, n_outer_electrons,
#: covalent radius, van der Waals radius, default valence.  These are the
#: "periodic position + size" axes that let an unseen metal interpolate.
#:
#: NOTE (honest, §5.2): the ligand-field splitting Δo, redox potential and HSAB
#: hardness the I2 plan also names require an *externally citable* per-metal
#: table that does not exist in this repo; hardcoding them from memory would
#: violate the MEASURED-vs-PROJECTED rule.  They are recorded as BLOCKED-pending-
#: source rather than fabricated.  d_electrons (already present) is the one
#: redox-adjacent axis we can derive without a table.
_RICH_SCALES = {
    "row": 5.0,
    "group": 10.0,
    "weight": 200.0,
    "n_outer": 10.0,
    "rcov": 2.0,
    "rvdw": 3.0,
    "valence": 8.0,
}
RICH_METAL_DIM = METAL_FEAT_DIM + len(_RICH_SCALES)

#: I2 (chemistry axis) — externally-citable per-metal redox / electronegativity
#: block.  Values are read from ``metal_chemistry_5.csv`` (NIST ionization
#: energies + Pauling electronegativity), NOT hardcoded from memory, so the
#: MEASURED-vs-PROJECTED rule holds.  This is the redox / HSAB-adjacent axis the
#: I2 plan names; the ligand-field splitting Δo has no citable per-metal table
#: (searched 2026-10-14: Wikipedia CFT / molar-IE pages carry no Δo values) and
#: remains BLOCKED-pending-source.
_CHEM_SCALES = {"ie1": 1000.0, "ie2": 2000.0, "en": 2.5}
CHEM_METAL_DIM = len(_CHEM_SCALES)

_CHEM_TABLE: dict = {}


def _load_chem_table() -> dict:
    """Lazily read the citable per-metal chemistry table (cached)."""
    if _CHEM_TABLE:
        return _CHEM_TABLE
    p = Path(__file__).with_name("metal_chemistry_5.csv")
    if p.exists():
        import csv

        with p.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                _CHEM_TABLE[row["metal_symbol"]] = row
    return _CHEM_TABLE


def chem_metal_block(metal: str) -> np.ndarray:
    """I2 redox/EN block (len ``len(_CHEM_SCALES)``) from the citable CSV."""
    t = _load_chem_table().get(metal)
    if not t:
        return np.zeros(len(_CHEM_SCALES), dtype=np.float32)
    return np.array(
        [
            float(t["ie1_kj"]) / _CHEM_SCALES["ie1"],
            float(t["ie2_kj"]) / _CHEM_SCALES["ie2"],
            float(t["pauling_en"]) / _CHEM_SCALES["en"],
        ],
        dtype=np.float32,
    )

_DONOR_STRENGTH = {"N": 3.0, "O": 2.0, "S": 1.5, "P": 2.0, "Se": 1.0}


def d_electrons(metal: str, oxidation_state: int) -> int:
    """Rough d-electron count: group number - oxidation state (clipped ≥0)."""
    group = {"Ru": 8, "Rh": 9, "Ir": 9, "Os": 8, "Re": 7}.get(metal, 8)
    return max(0, min(10, group - int(oxidation_state)))


def rich_metal_block(metal: str) -> np.ndarray:
    """I2: RDKit-verifiable periodic/size descriptor block (len ``len(_RICH_SCALES)``)."""
    from rdkit.Chem import GetPeriodicTable

    pt = GetPeriodicTable()
    z = METAL_Z.get(metal, 0)
    if z == 0:
        return np.zeros(len(_RICH_SCALES), dtype=np.float32)
    vals = {
        "row": float(pt.GetRow(z)),
        "group": float(pt.GetNOuterElecs(z)),  # outer-electron count ≈ group
        "weight": float(pt.GetAtomicWeight(z)),
        "n_outer": float(pt.GetNOuterElecs(z)),
        "rcov": float(pt.GetRcovalent(z)),
        "rvdw": float(pt.GetRvdw(z)),
        "valence": float(pt.GetDefaultValence(z)),
    }
    return np.array(
        [vals[k] / _RICH_SCALES[k] for k in _RICH_SCALES], dtype=np.float32
    )


def metal_feature_vector(
    metal: str,
    oxidation_state: int,
    charge_complex: int,
    coord_number: int,
    rich: bool = False,
    chem: bool = False,
) -> np.ndarray:
    """Return the metal descriptor vector.

    ``rich=True`` (I2 periodic/size) appends the RDKit-verifiable block;
    ``chem=True`` (I2 chemistry) appends the citable redox/EN block.  With both
    set the vector has length ``RICH_METAL_DIM + CHEM_METAL_DIM``.
    """
    dim = METAL_FEAT_DIM
    if rich:
        dim += len(_RICH_SCALES)
    if chem:
        dim += len(_CHEM_SCALES)
    v = np.zeros(dim, dtype=np.float32)
    if metal in METALS5:
        v[METALS5.index(metal)] = 1.0
    off = len(METALS5)
    v[off + 0] = float(oxidation_state) / 4.0
    v[off + 1] = float(charge_complex) / 4.0
    v[off + 2] = float(coord_number) / 8.0
    v[off + 3] = float(d_electrons(metal, oxidation_state)) / 10.0
    geo = classify_coord_geometry(int(coord_number), METAL_Z.get(metal, 0), int(oxidation_state))
    oh = one_hot_coord_geometry(geo, n_classes=N_GEOM)
    if oh is not None:
        v[off + 4 : off + 4 + N_GEOM] = oh
    block_off = METAL_FEAT_DIM
    if rich:
        v[block_off : block_off + len(_RICH_SCALES)] = rich_metal_block(metal)
        block_off += len(_RICH_SCALES)
    if chem:
        v[block_off : block_off + len(_CHEM_SCALES)] = chem_metal_block(metal)
    return v


def _metal_node_atom_feats(
    metal: str, oxidation_state: int, charge_complex: int, coord_number: int
) -> np.ndarray:
    """39-dim atom-feature vector for the appended metal node (free slots)."""
    f = np.zeros(ATOM_FEATURE_DIM, dtype=np.float32)
    f[_METAL_MARKER_SLOT] = 1.0
    if metal in METALS5:
        f[_METAL_ID_OFFSET + METALS5.index(metal)] = 1.0
    f[_METAL_OS_SLOT] = float(oxidation_state) / 4.0
    f[_METAL_CHARGE_SLOT] = float(charge_complex) / 4.0
    f[_METAL_CAP_SLOT] = float(coord_number) / 8.0
    return f


def _donor_atom_indices(mol: Chem.Mol, cap: int) -> List[int]:
    """Top-``cap`` donor atoms (N/O/S/P/Se) by strength, tie-break by degree."""
    scored = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in _DONOR_STRENGTH:
            scored.append((_DONOR_STRENGTH[sym] + 0.1 * atom.GetDegree(), atom.GetIdx()))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _s, idx in scored[:cap]]


def _coordination_bond_feature() -> np.ndarray:
    """10-dim bond feature for a metal→donor coordination bond.

    All zeros in the organic bond-type block (not single/double/triple/aromatic),
    so it is distinguishable from covalent bonds; a real dative-bond flag could
    be added to a free slot if the bond layout ever gains one.
    """
    return np.zeros(BOND_FEATURE_DIM, dtype=np.float32)


def build_metal_graph(
    ligand_smiles: str,
    metal: str,
    oxidation_state: int,
    charge_complex: int = 0,
    use_metal_node: bool = True,
    strip_metals: bool = True,
    rich_metal: bool = False,
    chem_metal: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build a (possibly metal-augmented) directed-edge graph for one row.

    Parameters
    ----------
    ligand_smiles
        ``SMILES_Ligands`` value (may be multi-component ``.``-separated).
    metal, oxidation_state, charge_complex
        Metal metadata from the row.
    use_metal_node
        ``True`` (I1) appends the metal node + coordination edges; ``False``
        returns the plain ligand graph (metal-blind baseline).
    strip_metals
        Drop any precious-metal atoms already present (so the cohort is
        metal-blind by construction, matching the MEASURED 0/19135 fact).

    Returns
    -------
    ``(atom_feats, bond_feats, edge_src, edge_dst, metal_feats)`` or ``None``
    if the SMILES cannot be parsed.
    """
    smi = str(ligand_smiles or "")
    if strip_metals:
        from molmetal.data.featurize import _strip_metal_atoms

        smi, _n = _strip_metal_atoms(smi)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    V = mol.GetNumAtoms()
    atom_feats = [_atom_features(a) for a in mol.GetAtoms()]
    edge_src: List[int] = []
    edge_dst: List[int] = []
    bond_feats: List[np.ndarray] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_src += [u, v]
        edge_dst += [v, u]
        bond_feats += [bf, bf]

    cap = int(coordination_capacity(metal, int(oxidation_state)))
    mf = metal_feature_vector(
        metal, int(oxidation_state), int(charge_complex), cap,
        rich=rich_metal, chem=chem_metal,
    )

    if use_metal_node:
        m_idx = V  # metal node is appended at the end
        atom_feats.append(_metal_node_atom_feats(metal, int(oxidation_state), int(charge_complex), cap))
        cbf = _coordination_bond_feature()
        for d in _donor_atom_indices(mol, cap):
            edge_src += [m_idx, d]
            edge_dst += [d, m_idx]
            bond_feats += [cbf, cbf]

    af = np.stack(atom_feats, axis=0).astype(np.float32)
    if bond_feats:
        bf_arr = np.stack(bond_feats, axis=0).astype(np.float32)
    else:
        bf_arr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
    return (
        af,
        bf_arr,
        np.array(edge_src, dtype=np.int64),
        np.array(edge_dst, dtype=np.int64),
        mf,
    )


__all__ = [
    "METALS5",
    "METAL_Z",
    "METAL_FEAT_DIM",
    "RICH_METAL_DIM",
    "N_GEOM",
    "d_electrons",
    "metal_feature_vector",
    "rich_metal_block",
    "build_metal_graph",
]
