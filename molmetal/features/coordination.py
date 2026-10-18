"""Coordination environment from the complex SMILES (FIX-L3).

The model sees a metal label and a ligand fingerprint but never which atoms are
actually pointing at the metal.  That is the difference between a bidentate
chelator and two monodentate ligands with the same molecular formula, and it is
the part of the chemistry the fingerprint cannot express: ECFP4 records that an
N is two bonds from a ring, not that the N binds Ru.

This module derives, per row, a small dense block from ``ligand_canonical``:

* donor counts by element (N, O, S, P, Se) and carbene carbon;
* anionic vs neutral vs aromatic donors;
* denticity proxy = donors in the single most donor-rich component;
* estimated coordination number = ``min(donors, capacity(metal, OS))``;
* the geometry class that CN implies for that metal and oxidation state.

The capacity table and the geometry classifier already exist
(``metal_smiles.coordination_capacity``, ``coord_geometry``); this module only
supplies the ligand side that was missing.

Honest limits: donor detection is element + charge, not a real metal-binding
model.  A pyridine N and a pyrrole N are both counted, though only the former
binds without deprotonation.  The block is a *description* of the ligand, and
the A/B decides whether it is worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

#: Elements that can donate a lone pair to a 4d/5d metal centre.
DONOR_ELEMENTS: Tuple[str, ...] = ("N", "O", "S", "P", "Se")


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


@dataclass(frozen=True)
class CoordinationProfile:
    """Per-row ligand-side coordination description."""

    n_donor: int = 0
    n_N: int = 0
    n_O: int = 0
    n_S: int = 0
    n_P: int = 0
    n_Se: int = 0
    n_carbene: int = 0
    n_anionic: int = 0
    n_aromatic: int = 0
    max_component_donors: int = 0
    n_donor_components: int = 0

    @property
    def has_donor(self) -> bool:
        return self.n_donor > 0


def coordination_profile(smiles: object) -> CoordinationProfile:
    """Count donor atoms in every component of a dot-separated SMILES."""
    Chem = _rdkit()
    text = "" if smiles is None else str(smiles)
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return CoordinationProfile()

    counts = {el: 0 for el in DONOR_ELEMENTS}
    carbene = anionic = aromatic = 0
    per_component: List[int] = []

    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        in_frag = 0
        for atom in frag.GetAtoms():
            symbol = atom.GetSymbol()
            if atom.GetFormalCharge() > 0:
                continue  # quaternary / oxonium: no lone pair left to give
            if symbol in counts:
                counts[symbol] += 1
                in_frag += 1
                if atom.GetFormalCharge() < 0:
                    anionic += 1
                if atom.GetIsAromatic():
                    aromatic += 1
            elif _is_carbene_carbon(atom):
                carbene += 1
                in_frag += 1
        per_component.append(in_frag)

    total = sum(counts.values()) + carbene
    return CoordinationProfile(
        n_donor=total,
        n_N=counts["N"],
        n_O=counts["O"],
        n_S=counts["S"],
        n_P=counts["P"],
        n_Se=counts["Se"],
        n_carbene=carbene,
        n_anionic=anionic,
        n_aromatic=aromatic,
        max_component_donors=max(per_component) if per_component else 0,
        n_donor_components=sum(1 for c in per_component if c > 0),
    )


def _is_carbene_carbon(atom) -> bool:
    """NHC-style divalent carbon: two heavy neighbours, no H, neutral."""
    from rdkit import Chem

    return (
        atom.GetSymbol() == "C"
        and atom.GetFormalCharge() == 0
        and atom.GetDegree() == 2
        and atom.GetTotalNumHs() == 0
        and atom.GetTotalValence() == 2
        and atom.GetHybridization() == Chem.HybridizationType.SP2
    )


def estimated_coordination_number(
    profile: CoordinationProfile, metal: str, oxidation_state: int
) -> int:
    """Donors available, capped by the metal's typical capacity."""
    from molmetal.data.metal_smiles import coordination_capacity

    if profile.n_donor == 0:
        return 0
    capacity = int(coordination_capacity(metal, int(oxidation_state)))
    return int(min(profile.n_donor, capacity))


def geometry_vector(
    coord_number: int, metal: str, oxidation_state: int
) -> Tuple[np.ndarray, str]:
    """One-hot geometry class implied by ``(CN, metal, OS)``; unknown → zeros."""
    from molmetal.data.coord_geometry import (
        COORD_GEOMETRY_VOCAB,
        classify_coord_geometry,
        one_hot_coord_geometry,
    )
    from molmetal.data.metal_graph import METAL_Z

    width = len(COORD_GEOMETRY_VOCAB)
    out = np.zeros(width, dtype=np.float32)
    if coord_number <= 0:
        return out, "none"
    cls = classify_coord_geometry(
        int(coord_number), METAL_Z.get(metal, 0), int(oxidation_state)
    )
    hot = one_hot_coord_geometry(cls)
    if hot is not None:
        out[: len(hot)] = np.asarray(hot, dtype=np.float32)
    return out, (cls or "unknown")


#: Dense columns produced by :func:`coordination_block`, in order.
COUNT_COLUMNS: Tuple[str, ...] = (
    "coord_n_donor",
    "coord_n_N",
    "coord_n_O",
    "coord_n_S",
    "coord_n_P",
    "coord_n_Se",
    "coord_n_carbene",
    "coord_n_anionic",
    "coord_n_aromatic",
    "coord_max_component_donors",
    "coord_n_donor_components",
    "coord_has_donor",
    "coord_est_cn",
)


def coordination_block(
    frame: pd.DataFrame,
    *,
    smiles_col: str = "ligand_canonical",
    metal_col: str = "metal",
    os_col: str = "oxidation_state",
    include_geometry: bool = True,
) -> Tuple[sparse.csr_matrix, List[str], Dict[str, float]]:
    """Dense coordination block + column names + coverage stats.

    Coverage is reported on two definitions, because they differ: ``has_donor``
    is "the ligand has at least one donor atom", ``has_geometry`` additionally
    requires that the metal/OS/CN triple maps to a known geometry class.
    """
    from molmetal.data.coord_geometry import COORD_GEOMETRY_VOCAB

    smiles = frame[smiles_col].astype(object).tolist()
    metals = frame[metal_col].astype(str).tolist()
    os_values = frame[os_col].tolist()

    cache: Dict[str, CoordinationProfile] = {}
    profiles: List[CoordinationProfile] = []
    for smi in smiles:
        key = "" if smi is None else str(smi)
        if key not in cache:
            cache[key] = coordination_profile(key)
        profiles.append(cache[key])

    rows = np.zeros((len(frame), len(COUNT_COLUMNS)), dtype=np.float32)
    geoms = np.zeros((len(frame), len(COORD_GEOMETRY_VOCAB)), dtype=np.float32)
    n_geometry = 0
    for i, (prof, metal, os_) in enumerate(zip(profiles, metals, os_values)):
        os_int = int(os_) if pd.notna(os_) else 0
        cn = estimated_coordination_number(prof, metal, os_int)
        rows[i] = [
            prof.n_donor,
            prof.n_N,
            prof.n_O,
            prof.n_S,
            prof.n_P,
            prof.n_Se,
            prof.n_carbene,
            prof.n_anionic,
            prof.n_aromatic,
            prof.max_component_donors,
            prof.n_donor_components,
            1.0 if prof.has_donor else 0.0,
            cn,
        ]
        if include_geometry:
            hot, cls = geometry_vector(cn, metal, os_int)
            geoms[i] = hot
            if cls not in ("none", "unknown"):
                n_geometry += 1

    names = list(COUNT_COLUMNS)
    parts = [rows]
    if include_geometry:
        names += [f"coord_geom_{c}" for c in COORD_GEOMETRY_VOCAB]
        parts.append(geoms)

    X = sparse.csr_matrix(np.hstack(parts).astype(np.float32))
    coverage = {
        "rows": int(len(frame)),
        "has_donor": round(float(np.mean([p.has_donor for p in profiles])), 4),
        "has_geometry": round(float(n_geometry / max(len(frame), 1)), 4),
        "unique_smiles": int(len(cache)),
    }
    return X, names, coverage
