"""Tests for :mod:`molmetal.scripts.check_3d_embed`.

T4 B1 sanity tests:

1. ``test_embed_cisplatin_pt`` — embed Pt(NH3)2Cl2, assert 3-D coords are
   finite, and the Pt–N bond distances are within ``0.30`` Å of the
   literature value of 2.05 Å.
2. ``test_embed_ru_octahedral`` — embed [Ru](N)6 (hexaammineruthenium)
   and assert we get a finite 3-D conformer.
3. ``test_failure_rate_under_10pct`` — running the script's ``_embed_one``
   helper against the synthetic cisplatin pool should yield ``>= 90 %``
   success.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make project importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem

from molmetal.scripts.check_3d_embed import (
    _build_synthetic_pt_smiles,
    _embed_one,
    _metal_bond_distances,
)


DEFAULT_PT_N_DISTANCE = 2.05  # Å, Cambridge Structural Database typical
DEFAULT_BOND_TOL = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_finite_3d(mol: Chem.Mol) -> bool:
    """True iff mol has a conformer and every coordinate is finite."""
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
    return True


def _metal_donor_distances(mol: Chem.Mol, metal_sym: str, donor_sym: str) -> list[float]:
    """Return all metal→donor distances (Å) in ``mol``."""
    dist_map = _metal_bond_distances(mol, metal_sym)
    return dist_map.get(donor_sym, [])


def _embed_simple(smiles: str, seed: int = 42, max_iter: int = 200) -> Chem.Mol:
    """Embed + MMFF-optimise ``smiles`` and return the resulting mol."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse {smiles}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxIterations = int(max_iter)
    # Required for metal centres: RDKit has no ETKDG torsions for Pt/Ru/Ir.
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    assert status == 0, f"EmbedMolecule failed with status {status}"
    AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    assert _is_finite_3d(mol), "Conformer has non-finite coords"
    return mol


# ---------------------------------------------------------------------------
# 1. Cisplatin (Pt(NH3)2Cl2) — Pt–N within tolerance of 2.05 Å
# ---------------------------------------------------------------------------
def test_embed_cisplatin_pt():
    """Pt(NH3)2Cl2 (cisplatin) → 3-D coords finite + Pt–N ≈ 2.05 Å."""
    cisplatin = _embed_simple("[Pt](N)(N)(Cl)(Cl)", seed=42)
    pt_n = _metal_donor_distances(cisplatin, "Pt", "N")
    assert len(pt_n) == 2, f"Expected 2 Pt-N bonds in cisplatin, got {len(pt_n)}: {pt_n}"
    mean_pt_n = sum(pt_n) / len(pt_n)
    assert abs(mean_pt_n - DEFAULT_PT_N_DISTANCE) <= DEFAULT_BOND_TOL, (
        f"Pt-N mean distance {mean_pt_n:.3f} Å not within "
        f"{DEFAULT_BOND_TOL} Å of {DEFAULT_PT_N_DISTANCE} Å"
    )


# ---------------------------------------------------------------------------
# 2. Ru(II) hexaammine — finite 3-D coords for [Ru](N)6
# ---------------------------------------------------------------------------
def test_embed_ru_octahedral():
    """[Ru](N)6 (hexaammineruthenium) → 3-D coords finite."""
    ru = _embed_simple("[Ru](N)(N)(N)(N)(N)(N)", seed=7, max_iter=300)
    # Atoms count check: Ru + 6 N + H on each amine.
    # RDKit's ``AddHs`` adds one H per bare ``N`` (the input ``N`` is an
    # NH2 already encoded), giving 19 atoms total: 1 Ru + 6 N + 12 H.
    assert ru.GetNumAtoms() == 19, (
        f"Expected 19 atoms in [Ru](N)6 + Hs, got {ru.GetNumAtoms()}"
    )
    # Should have 6 Ru-N bonds.
    ru_n = _metal_donor_distances(ru, "Ru", "N")
    assert len(ru_n) == 6, f"Expected 6 Ru-N bonds, got {len(ru_n)}: {ru_n}"


# ---------------------------------------------------------------------------
# 3. Failure rate under 10 % on the synthetic cisplatin Pt pool
# ---------------------------------------------------------------------------
def test_failure_rate_under_10pct():
    """For 100 synthetic Pt rows, embed failure rate < 10 % (>= 90 % success).

    The MetalCytoToxDB CSV contains no Pt rows (Ru/Ir/Rh/Os/Re only), so
    we exercise the script's synthetic-Pt fallback.  This is also the
    pool we use for ``--metal Pt`` runs.
    """
    rows = _build_synthetic_pt_smiles(100)
    success = 0
    for i, (smiles, metal, ox) in enumerate(rows):
        mol = _embed_one(
            smiles_ligands=smiles,
            metal=metal,
            oxidation_state=ox,
            seed=0xC0FFEE + i,
            max_attempts=5,
        )
        if mol is not None:
            success += 1
    success_rate = success / len(rows)
    assert success_rate >= 0.90, (
        f"Pt embedding success rate {success_rate:.2%} < 90 % "
        f"({success}/{len(rows)}); recommend UFF fallback (TODO/07 B2)"
    )
