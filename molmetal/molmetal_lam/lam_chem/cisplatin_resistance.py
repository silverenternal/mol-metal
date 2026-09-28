"""Cisplatin-resistance feature module — Phase A / Stage 2.2.

Returns an 9-element chemoinformatic feature dict for the cisplatin-
resistance classifier (Wang & Lippard 2005, Nat Rev Drug Discov 4:307).
For Pt(II) complexes use multi-component SMILES (e.g. ``"N.N.Cl.Cl.[Pt]"``)
because RDKit rejects the bracket form ``Cl[Pt]([NH3])([NH3])[NH3]Cl``
(N over-valence).  See :mod:`molmetal.data.metal_smiles` for the
project-wide convention.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors


CISPLATIN_SMILES: str = "N.N.Cl.Cl.[Pt]"

RESISTANCE_FEATURE_KEYS: Tuple[str, ...] = (
    "logP", "thiol_count", "n_donor_count", "cis_trans_pattern",
    "reduction_potential_proxy", "mw", "tpsa", "hba", "hbd",
)

# Hammett σ_p constants (Hansch, Leo, Taft 1991, Chem Rev 91:165).
_HAMMETT_SIGMA: Dict[str, float] = {
    "NH2": -0.66, "N(CH3)2": -0.83, "OH": -0.37, "OCH3": -0.27,
    "CH3": -0.17, "F": 0.06, "Cl": 0.23, "Br": 0.23, "I": 0.18,
    "COOH": 0.45, "COCH3": 0.50, "CN": 0.66, "NO2": 0.78,
    "CF3": 0.54, "SO3H": 0.50,
}

# Pre-compiled SMARTS — re-built on every call would dominate batch latency.
# N-donor SMARTS use ``[N;H1,H2,H3]`` to cover NH / NH2 / NH3 (cisplatin's
# amine nitrogens have H-count 3 in the multi-component SMILES form
# ``N.N.Cl.Cl.[Pt]`` because RDKit fills implicit valence to 3).
_SH    = Chem.MolFromSmarts("[SH]")
_NH    = Chem.MolFromSmarts("[N;H1,H2,H3]")
_NH2   = Chem.MolFromSmarts("[NH2]")
_NHET  = Chem.MolFromSmarts("[nH]")
_PT    = Chem.MolFromSmarts("[Pt]")


def _count_thiols(mol: Chem.Mol) -> int:
    """Count -SH atoms (thiol + thiophenol)."""
    return 0 if mol is None else len(mol.GetSubstructMatches(_SH))


def _count_n_donors(mol: Chem.Mol) -> int:
    """Count distinct N-H + amine + N-heterocycle donor atoms."""
    if mol is None:
        return 0
    idx: set = set()
    for pat in (_NH, _NH2, _NHET):
        for tup in mol.GetSubstructMatches(pat):
            idx.update(tup)
    return len(idx)


def _detect_cis_trans(mol: Chem.Mol) -> int:
    """Pt(II) square-planar geometry: +1 cis / -1 trans / 0 no Pt.

    For 4-coordinate Pt with two pairs of identical ligands (e.g. cis-
    or trans-``[Pt](N)(N)(Cl)Cl``), SMILES bond order encodes the
    geometry: like-ligands on bonds (1,3) → trans, on (1,2) → cis.
    """
    if mol is None:
        return 0
    pt_idx = mol.GetSubstructMatch(_PT)
    if not pt_idx:
        return 0
    pt = mol.GetAtomWithIdx(pt_idx[0])
    if pt.GetDegree() != 4:
        return 0
    ligands: List[str] = [b.GetOtherAtom(pt).GetSymbol() for b in pt.GetBonds()]
    counts = Counter(ligands)
    if len(counts) != 2 or not all(c == 2 for c in counts.values()):
        return 0
    return -1 if ligands[0] == ligands[2] else 1


def _hammett_proxy(mol: Chem.Mol) -> float:
    """Sum σ_p across aromatic-ring substituents — Pt(II)→Pt(0) proxy.

    More EWG (positive σ) → easier reduction under hypoxia.
    """
    if mol is None:
        return 0.0
    total = 0.0
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        for ai in ring:
            atom = mol.GetAtomWithIdx(ai)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring:
                    continue
                sym, nh = nbr.GetSymbol(), nbr.GetTotalNumHs()
                label = ""
                # Nitro group — cationic N with at least one =O.
                if sym == "N" and nbr.GetFormalCharge() == 1 and any(
                    b.GetBondTypeAsDouble() == 2.0
                    and b.GetOtherAtom(nbr).GetSymbol() == "O"
                    for b in nbr.GetBonds()
                ):
                    label = "NO2"
                elif sym == "N" and nh >= 2:
                    label = "NH2"
                elif sym == "O" and nh == 1:
                    label = "OH"
                elif sym == "O" and nh == 0:
                    # OMe / OEt — first non-ring C neighbour qualifies.
                    for b in nbr.GetBonds():
                        o = b.GetOtherAtom(nbr)
                        if o.GetIdx() not in ring and o.GetSymbol() == "C":
                            label = "OCH3"
                            break
                elif sym == "C" and nh == 3:
                    label = "CH3"
                if label:
                    total += _HAMMETT_SIGMA.get(label, 0.0)
    return total


def compute_resistance_features(mol: Chem.Mol) -> Dict[str, float]:
    """Compute the cisplatin-resistance feature vector.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.  Use multi-component SMILES for Pt complexes.

    Returns
    -------
    dict
        Keys in :data:`RESISTANCE_FEATURE_KEYS`.  Returns all zeros
        on ``None`` mol or any descriptor failure so the downstream
        classifier never crashes on a malformed row.
    """
    empty = {k: 0.0 for k in RESISTANCE_FEATURE_KEYS}
    if mol is None:
        return empty
    try:
        return {
            "logP":                      float(Descriptors.MolLogP(mol)),
            "thiol_count":               float(_count_thiols(mol)),
            "n_donor_count":             float(_count_n_donors(mol)),
            "cis_trans_pattern":         float(_detect_cis_trans(mol)),
            "reduction_potential_proxy": _hammett_proxy(mol),
            "mw":                        float(Descriptors.MolWt(mol)),
            "tpsa":                      float(Descriptors.TPSA(mol)),
            "hba":                       float(Descriptors.NumHAcceptors(mol)),
            "hbd":                       float(Descriptors.NumHDonors(mol)),
        }
    except Exception:
        return empty


__all__ = [
    "CISPLATIN_SMILES",
    "RESISTANCE_FEATURE_KEYS",
    "compute_resistance_features",
]
