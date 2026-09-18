"""Lightweight retrosynthesis feasibility cost estimation.

The estimator is deliberately dependency-tolerant: RDKit provides the preferred
descriptors, while a conservative fallback keeps the public API importable in
minimal environments.  Scores are costs, so zero is easiest and one is hardest.
"""

from __future__ import annotations

import math
from typing import Iterable


def _rdkit_modules():
    """Return RDKit modules, or ``None`` when RDKit is not installed."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
        RDLogger.DisableLog("rdApp.error")
        return Chem, rdMolDescriptors
    except Exception:
        return None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sa_score(mol, descriptors) -> float:
    """Obtain a normalized SA cost (0 easy, 1 difficult).

    The optional RDKit Contrib implementation is used when present.  Its native
    range is approximately 1--10, hence the conversion below.  Descriptor-based
    proxy is intentionally stable across RDKit versions.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        raw = float(sascorer.calculateScore(mol))
        return _clip((raw - 1.0) / 9.0)
    except Exception:
        heavy = mol.GetNumHeavyAtoms()
        rings = descriptors.CalcNumRings(mol)
        complexity = heavy + 2.0 * rings
        # 0 for small fragments, approaching 1 for large/complex molecules.
        return _clip((complexity - 8.0) / 42.0)


def _building_block_cost(mol, smiles: str, blocks: list[str] | None, Chem) -> float:
    if blocks is None:
        return 0.5  # unknown inventory: neutral prior
    if not blocks:
        return 1.0
    target = Chem.MolToSmiles(mol, canonical=True)
    parsed = []
    for item in blocks:
        try:
            block = Chem.MolFromSmiles(str(item))
            if block is not None:
                parsed.append(Chem.MolToSmiles(block, canonical=True))
        except Exception:
            continue
    if target in parsed:
        return 0.0
    # A listed fragment that is a substructure makes retrosynthesis easier.
    for item in blocks:
        try:
            fragment = Chem.MolFromSmiles(str(item))
            if fragment is not None and mol.HasSubstructMatch(fragment):
                return 0.25
        except Exception:
            continue
    return 1.0


def _complexity_cost(mol, descriptors) -> float:
    rings = descriptors.CalcNumRings(mol)
    from rdkit import Chem
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    hetero = descriptors.CalcNumHeteroatoms(mol)
    bond_stereo = sum(1 for bond in mol.GetBonds()
                      if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)
    stereo = descriptors.CalcNumAtomStereoCenters(mol) + bond_stereo
    # Weighted counts, normalized to a practical medicinal-chemistry range.
    raw = 0.9 * rings + 0.8 * chiral + 0.35 * hetero + 0.6 * stereo
    return _clip(raw / 18.0)


def synthesis_cost(mol_smiles: str, available_building_blocks: list[str] | None = None) -> float:
    """Return a retrosynthesis feasibility cost in the inclusive range [0, 1]."""
    modules = _rdkit_modules()
    if modules is None:
        return 1.0
    Chem, descriptors = modules
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return 1.0
        sa = _sa_score(mol, descriptors)
        bb = _building_block_cost(mol, mol_smiles, available_building_blocks, Chem)
        complexity = _complexity_cost(mol, descriptors)
        return _clip(0.5 * sa + 0.3 * bb + 0.2 * complexity)
    except Exception:
        return 1.0


class SynthesisCostEstimator:
    """Configurable wrapper around :func:`synthesis_cost`."""

    def __init__(self, available_building_blocks: list[str] | None = None,
                 sa_weight: float = 0.5, bb_weight: float = 0.3,
                 complexity_weight: float = 0.2):
        weights = (float(sa_weight), float(bb_weight), float(complexity_weight))
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative and have positive sum")
        total = sum(weights)
        self.sa_weight, self.bb_weight, self.complexity_weight = (w / total for w in weights)
        self.available_building_blocks = available_building_blocks

    def cost(self, smiles: str) -> float:
        """Estimate cost using this estimator's inventory and normalized weights."""
        modules = _rdkit_modules()
        if modules is None:
            return 1.0
        Chem, descriptors = modules
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return 1.0
            return _clip(self.sa_weight * _sa_score(mol, descriptors)
                         + self.bb_weight * _building_block_cost(mol, smiles, self.available_building_blocks, Chem)
                         + self.complexity_weight * _complexity_cost(mol, descriptors))
        except Exception:
            return 1.0


__all__ = ["synthesis_cost", "SynthesisCostEstimator"]
