"""Tests for :mod:`molmetal_lam.lam_chem.conformer_embed`.

Phase 2 / L1 — canonical 3D conformer generation (ETKDGv3 + MMFF94s).
These tests exercise the public surface of
:mod:`molmetal_lam.lam_chem.conformer_embed` against real RDKit
implementations (no mocks).

Test matrix
-----------
1. ``test_generate_conformer_cisplatin``      Pt(II) d4-square-planar
2. ``test_generate_conformer_benzene``        planar 6-ring, no metal
3. ``test_generate_conformer_organic_no_metal``  purely organic SMILES
4. ``test_generate_conformer_invalid_smiles``  empty / unparseable input
5. ``test_extract_coords_shape``               (N_atoms, 3) numpy
6. ``test_multiple_conformers``                n_confs=5 returns 5 distinct
7. ``test_deterministic_seed``                 bit-for-bit same coords

All tests assert finite 3D coordinates + the metal-bond / ring
properties the Vina downstream path needs.  The Pt-N bond tolerance is
0.40 Å — looser than the Cambridge Structural Database typical (2.05 Å)
because ETKDGv3 + MMFF94s do not strictly enforce metal-ligand bond
lengths (RDKit has no MMFF atom-type for Pt in some builds).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make project importable when running pytest from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem

from molmetal_lam.lam_chem.conformer_embed import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_ITERS,
    DEFAULT_SEED,
    ConformerEmbedError,
    extract_coords,
    generate_conformer,
    generate_conformers,
    has_finite_3d,
    to_pdb_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_distances_finite(coords: np.ndarray) -> bool:
    """Return True iff every coord is finite (not NaN, not inf)."""
    return bool(np.isfinite(coords).all())


def _metal_bond_distances(mol, metal_sym: str, donor_sym: str):
    """Return all metal->donor Euclidean distances (Å) in ``mol``.

    Uses the bond table (after AddHs) — the heavy-atom order in
    ``mol.GetAtoms()`` is preserved by RDKit.
    """
    conf = mol.GetConformer(0)
    out = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        sa, sb = a.GetSymbol(), b.GetSymbol()
        if sa == metal_sym and sb == donor_sym:
            pa = conf.GetAtomPosition(a.GetIdx())
            pb = conf.GetAtomPosition(b.GetIdx())
            out.append(
                math.sqrt(
                    (pa.x - pb.x) ** 2
                    + (pa.y - pb.y) ** 2
                    + (pa.z - pb.z) ** 2
                )
            )
        elif sb == metal_sym and sa == donor_sym:
            pa = conf.GetAtomPosition(a.GetIdx())
            pb = conf.GetAtomPosition(b.GetIdx())
            out.append(
                math.sqrt(
                    (pa.x - pb.x) ** 2
                    + (pa.y - pb.y) ** 2
                    + (pa.z - pb.z) ** 2
                )
            )
    return out


def _planarity_score(coords: np.ndarray, ring_indices) -> float:
    """RMS deviation (Å) of a 6-ring from its best-fit plane.

    Returns 0.0 for a perfectly planar ring, ~0.5 Å for chair cyclohexane.
    """
    pts = coords[list(ring_indices)]
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    # Best-fit plane normal via SVD of the centred points.
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    # Distance of each atom to the plane = |(p - c) . n|
    distances = np.abs(centred @ normal)
    return float(np.sqrt(np.mean(distances ** 2)))


def _ring_atoms(mol, size: int = 6):
    """Return list of atom-index lists for all rings of size ``size``."""
    rings = mol.GetRingInfo().AtomRings()
    return [list(r) for r in rings if len(r) == size]


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "[H]N([H])([H])[Pt](N([H])([H])[H])(Cl)Cl"
CISPLATIN_TIGHT_SMILES = "[Pt](N)(N)(Cl)(Cl)"
BENZENE_SMILES = "c1ccccc1"
ETHANOL_SMILES = "CCO"


# ---------------------------------------------------------------------------
# 1. Cisplatin (Pt(II) d4-square-planar) — 4 Pt-bonds + finite coords
# ---------------------------------------------------------------------------
def test_generate_conformer_cisplatin():
    """Cisplatin: 4 Pt-bonds, finite 3D coords, Pt-N ~ 2.05 Å.

    We use ``use_random_coords=True`` because RDKit's ETKDG torsion
    priors do not cover Pt centres.  The Pt-N bond tolerance is
    0.40 Å (loose: MMFF94s is the *published* FF for organics; Pt
    coordinates rely on the UFF-style generic-atom-type fallback which
    can drift up to 0.3 Å from the experimental 2.05 Å).
    """
    smiles = CISPLATIN_TIGHT_SMILES
    mol, coords = generate_conformer(
        smiles=smiles,
        seed=DEFAULT_SEED,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        use_random_coords=True,  # metal centres need random init
    )
    assert mol is not None
    assert mol.GetNumConformers() >= 1
    assert _all_distances_finite(coords), "Non-finite coords in cisplatin"

    # 4 Pt bonds expected (2 Pt-N + 2 Pt-Cl).
    pt_n = _metal_bond_distances(mol, "Pt", "N")
    pt_cl = _metal_bond_distances(mol, "Pt", "Cl")
    assert len(pt_n) == 2, f"Expected 2 Pt-N bonds, got {len(pt_n)}"
    assert len(pt_cl) == 2, f"Expected 2 Pt-Cl bonds, got {len(pt_cl)}"

    # Pt-N mean within 0.40 Å of 2.05 Å (CSD typical).
    mean_pt_n = sum(pt_n) / len(pt_n)
    assert abs(mean_pt_n - 2.05) <= 0.40, (
        f"Pt-N mean {mean_pt_n:.3f} Å not within 0.40 Å of 2.05 Å "
        f"(Pt-N={pt_n})"
    )


# ---------------------------------------------------------------------------
# 2. Benzene — planar 6-ring, no metal
# ---------------------------------------------------------------------------
def test_generate_conformer_benzene():
    """Benzene: planar 6-ring (RMS < 0.10 Å from best-fit plane).

    Aromatic bond delocalisation + ETKDGv3 + MMFF94s should preserve
    the planar geometry — the published MMFF94s RMS error for ring
    angles is 1.2° (Halgren 1996), so a planarity RMS < 0.10 Å is a
    loose gate that catches most broken-embedding regressions.
    """
    mol, coords = generate_conformer(smiles=BENZENE_SMILES, seed=DEFAULT_SEED)
    assert mol is not None
    assert mol.GetNumConformers() >= 1
    assert _all_distances_finite(coords)
    rings = _ring_atoms(mol, size=6)
    assert len(rings) == 1, f"Expected exactly one 6-ring, got {len(rings)}"
    rms = _planarity_score(coords, rings[0])
    assert rms < 0.10, f"Benzene RMS planarity deviation {rms:.4f} Å > 0.10 Å"


# ---------------------------------------------------------------------------
# 3. Purely organic SMILES (ethanol) — works without metal-coord fallback
# ---------------------------------------------------------------------------
def test_generate_conformer_organic_no_metal():
    """Ethanol CCO: 3 heavy atoms + 6 Hs, no metal centres, no ring.

    Smoke test for the most common case in the Vina downstream path —
    organic-only ligands.  No metal fallback needed.
    """
    mol, coords = generate_conformer(smiles=ETHANOL_SMILES, seed=DEFAULT_SEED)
    assert mol is not None
    assert mol.GetNumConformers() >= 1
    assert _all_distances_finite(coords)
    # Ethanol = C-C-O = 3 heavy + 6 H = 9 atoms total after AddHs.
    assert mol.GetNumAtoms() == 9, (
        f"Expected 9 atoms (3 heavy + 6 H), got {mol.GetNumAtoms()}"
    )
    # Returned coords shape matches atom count.
    assert coords.shape == (mol.GetNumAtoms(), 3)


# ---------------------------------------------------------------------------
# 4. Invalid SMILES — must raise ConformerEmbedError
# ---------------------------------------------------------------------------
def test_generate_conformer_invalid_smiles():
    """Empty / unparseable SMILES must raise :class:`ConformerEmbedError`."""
    with pytest.raises(ConformerEmbedError):
        generate_conformer(smiles="")
    with pytest.raises(ConformerEmbedError):
        generate_conformer(smiles="this-is-not-a-smiles!!!@@@###")
    with pytest.raises(ConformerEmbedError):
        generate_conformer(smiles="@@bad@@")


# ---------------------------------------------------------------------------
# 5. extract_coords returns (N_atoms, 3) numpy
# ---------------------------------------------------------------------------
def test_extract_coords_shape():
    """``extract_coords`` returns float64 ``(N_atoms, 3)`` numpy."""
    mol, coords = generate_conformer(smiles=BENZENE_SMILES, seed=DEFAULT_SEED)
    # benzene = 6 heavy + 6 H = 12 atoms after AddHs.
    expected_n = mol.GetNumAtoms()
    assert coords.shape == (expected_n, 3)
    assert coords.dtype == np.float64
    assert _all_distances_finite(coords)


# ---------------------------------------------------------------------------
# 6. n_confs=5 returns 5 distinct conformers
# ---------------------------------------------------------------------------
def test_multiple_conformers():
    """``generate_conformers`` with n_confs=5 returns 5 distinct geometries.

    We compare pairwise RMSD on heavy atoms (Hs have many degenerate
    rotations, so a heavy-atom RMSD gate is the right one).  The gate
    is **at least one pair** with RMSD >= 0.05 Å — a very loose test
    that catches "all 5 conformers collapsed onto the same point" but
    does not require genuine structural diversity.
    """
    n_confs = 5
    mol, coords_list = generate_conformers(
        smiles=BENZENE_SMILES,
        n_confs=n_confs,
        seed=DEFAULT_SEED,
    )
    assert mol is not None
    assert len(coords_list) == n_confs
    for i, c in enumerate(coords_list):
        assert _all_distances_finite(c), f"conformer {i} non-finite"
        assert c.shape[0] == mol.GetNumAtoms(), (
            f"conformer {i} coords has {c.shape[0]} atoms, "
            f"mol has {mol.GetNumAtoms()}"
        )
    # At least one heavy-atom RMSD >= 0.05 Å between distinct conformers.
    # Benzene has 12 atoms; H atoms come after the 6 heavy atoms
    # (RDKit convention) so heavy-atom slice = c[:, :6].
    n_heavy = mol.GetNumHeavyAtoms()
    pairwise_max = 0.0
    for i in range(n_confs):
        for j in range(i + 1, n_confs):
            rmsd = float(
                np.sqrt(
                    np.mean(
                        (coords_list[i][:n_heavy] - coords_list[j][:n_heavy]) ** 2
                    )
                )
            )
            if rmsd > pairwise_max:
                pairwise_max = rmsd
    assert pairwise_max >= 0.05, (
        f"All {n_confs} conformers collapsed to within 0.05 Å; "
        f"max pairwise RMSD = {pairwise_max:.4f} Å"
    )


# ---------------------------------------------------------------------------
# 7. Deterministic seed — bit-for-bit same coords across two calls
# ---------------------------------------------------------------------------
def test_deterministic_seed():
    """Same seed -> bit-for-bit same coords across two calls.

    Riniker 2015 §3.3 documents RDKit's reproducibility guarantee for
    fixed seeds (the random-coord init and the torsion sampling both
    draw from the same seeded RNG).  We assert exact equality to make
    any future RDKit regression visible.
    """
    _, coords_a = generate_conformer(smiles=ETHANOL_SMILES, seed=42)
    _, coords_b = generate_conformer(smiles=ETHANOL_SMILES, seed=42)
    assert np.array_equal(coords_a, coords_b), (
        "Same seed (42) produced different coords — RDKit regression?"
    )
    # And a different seed -> at least one coord differs.
    _, coords_c = generate_conformer(smiles=ETHANOL_SMILES, seed=43)
    assert not np.array_equal(coords_a, coords_c), (
        "Different seeds (42 vs 43) produced identical coords — RNG bug?"
    )


# ---------------------------------------------------------------------------
# 8. (Bonus) has_finite_3d + to_pdb_block sanity
# ---------------------------------------------------------------------------
def test_pdb_block_export():
    """``to_pdb_block`` returns a non-empty PDB string for a valid 3D mol."""
    mol, _ = generate_conformer(smiles=BENZENE_SMILES, seed=DEFAULT_SEED)
    assert has_finite_3d(mol), "mol should have finite 3D"
    block = to_pdb_block(mol)
    assert isinstance(block, str)
    assert "ATOM" in block or "HETATM" in block, (
        f"PDB block missing ATOM/HETATM records: {block[:200]!r}"
    )
    assert len(block) > 100, f"PDB block suspiciously short: {len(block)} bytes"


# ---------------------------------------------------------------------------
# 9. (Bonus) n_confs < 1 raises immediately
# ---------------------------------------------------------------------------
def test_n_confs_must_be_positive():
    """``generate_conformers`` with n_confs=0 must raise."""
    with pytest.raises(ConformerEmbedError):
        generate_conformers(smiles=BENZENE_SMILES, n_confs=0)
    with pytest.raises(ConformerEmbedError):
        generate_conformers(smiles=BENZENE_SMILES, n_confs=-1)