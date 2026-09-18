"""DNA fragment oracle for precious-metal anticancer Kb proxy.

Wraps :func:`AnticancerMetricSuite.dna_kb_proxy` with an optional
docked-conformer boost: if a 3D docked conformer against a DNA
fragment (e.g. guanine N7, adenine N7) is supplied, the docked pose's
distance to the donor nitrogen sharpens the proxy.

Without a docked conformer, this falls back to the heuristic proxy
in :mod:`molmetal_lam.metrics.anticancer_metric_suite`.

References
----------
Jamieson, E. R. & Lippard, S. J. (1999), *Chem. Rev.*, 99, 2467-2498.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None

from molmetal_lam.metrics.anticancer_metric_suite import AnticancerMetricSuite


__all__ = ["dna_kb_proxy_v2", "DNA_FRAGMENT_DONORS"]


# DNA fragment donor sites (SMILES of the relevant base/fragment).
DNA_FRAGMENT_DONORS = {
    "guanine_N7": "N1C=NC2=C1NC=NC2=O",   # approximate guanine core
    "adenine_N7": "NC1=NC=NC2=C1NC=N2",   # approximate adenine core
    "thymine_O2": "O=C(N)C(=O)NC(=O)C",   # not used directly
}


def _docked_distance(mol, donor_smiles: str) -> Optional[float]:
    """Compute the min distance between the metal centre and the donor N.

    Embeds the molecule + donor, performs a constrained translation so
    the metal is 2.0 A from the donor N (a reasonable cisplatin-class
    Pt-N7 adduct distance), and returns the realised distance.
    Returns None if RDKit embedding fails or no metal is present.
    """
    if Chem is None or AllChem is None:
        return None

    if isinstance(mol, str):
        rdkit_mol = Chem.MolFromSmiles(mol)
    else:
        rdkit_mol = mol if isinstance(mol, Chem.Mol) else None
    if rdkit_mol is None:
        return None

    metal_atoms = [a for a in rdkit_mol.GetAtoms()
                   if a.GetSymbol() in ("Pt", "Ru", "Ir", "Au", "Rh", "Os")]
    if not metal_atoms:
        return None

    donor_mol = Chem.MolFromSmiles(donor_smiles)
    if donor_mol is None:
        return None

    n_atoms = [a for a in donor_mol.GetAtoms() if a.GetSymbol() == "N"]
    if not n_atoms:
        return None

    # Embed both molecules; if embedding fails, return None so the
    # caller falls back to the heuristic proxy.
    try:
        if rdkit_mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
        if donor_mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(donor_mol, randomSeed=42)
    except Exception:
        return None

    metal_idx = metal_atoms[0].GetIdx()
    metal_conf = rdkit_mol.GetConformer()
    mx, my, mz = metal_conf.GetAtomPosition(metal_idx).x, \
                 metal_conf.GetAtomPosition(metal_idx).y, \
                 metal_conf.GetAtomPosition(metal_idx).z
    donor_conf = donor_mol.GetConformer()
    distances = []
    for n in n_atoms:
        pos = donor_conf.GetAtomPosition(n.GetIdx())
        dx = mx - pos.x
        dy = my - pos.y
        dz = mz - pos.z
        distances.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return float(min(distances))


def dna_kb_proxy_v2(mol, dna_fragment: Optional[str] = "guanine_N7") -> float:
    """DNA binding affinity proxy v2 (with optional docked conformer).

    Parameters
    ----------
    mol : str or rdkit.Chem.Mol
        The molecule (SMILES or RDKit Mol) to score.
    dna_fragment : str, optional
        Key from :data:`DNA_FRAGMENT_DONORS` (e.g. "guanine_N7").
        If None, falls back to the heuristic
        :meth:`AnticancerMetricSuite.dna_kb_proxy`.

    Returns
    -------
    float
        A value in [0, 1].  When a docked conformer is available the
        proxy is the heuristic base scaled by a distance-based
        sigmoid: closer Pt-to-N7 distances => higher score.
    """
    base_suite = AnticancerMetricSuite()
    base = base_suite.dna_kb_proxy(mol)
    if not dna_fragment:
        return base
    donor_smiles = DNA_FRAGMENT_DONORS.get(dna_fragment, dna_fragment)
    if not donor_smiles:
        return base
    dist = _docked_distance(mol, donor_smiles)
    if dist is None:
        return base
    # Sigmoid centred at 2.0 A; closer to donor => higher boost.
    # 1.8 A => ~1.0 boost; 2.2 A => ~0.5; 3.0 A => ~0.05.
    boost = 1.0 / (1.0 + math.exp((dist - 2.0) * 4.0))
    return float(max(0.0, min(1.0, 0.5 * base + 0.5 * boost)))
