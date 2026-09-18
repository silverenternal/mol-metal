"""Synthetic-accessibility score (Ertl & Schuffenhauer 2009).

Lifted from ``Pocket2Mol/evaluation/sascorer.py`` with the only change being
``from rdkit.six`` → Python-3 builtins (``cPickle`` → ``pickle``,
``iteritems`` → ``.items()``). Modern RDKit (>= 2022) no longer ships
``rdkit.six``. Algorithm and pickled fpscores table are unchanged.

Reference
---------
Ertl, P. & Schuffenhauer, A. "Estimation of Synthetic Accessibility Score
of Drug-like Molecules based on Molecular Complexity and Fragment
Contributions." J. Cheminform. 1:8 (2009).
"""
from __future__ import annotations

import gzip
import math
import os
import pickle
from collections import defaultdict
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# fpscores.pkl.gz is shipped alongside this module so the wrapper is
# self-contained and does not depend on the Pocket2Mol clone being on
# sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FPSCORES_PATH = os.path.join(_HERE, "fpscores.pkl.gz")

_fscores: Optional[dict] = None


def _load_fscores(path: str = _FPSCORES_PATH) -> dict:
    global _fscores
    if _fscores is None:
        with gzip.open(path, "rb") as f:
            _fscores = pickle.load(f, encoding="Latin1")  # noqa: S615
        if isinstance(_fscores, list):
            # Original Pocket2Mol format: list of (fragment_smiles, score)
            d = defaultdict(float)
            for smi, sc in _fscores:
                d[smi] = sc
            _fscores = dict(d)
    return _fscores


def _num_spiro_atoms(mol: Chem.Mol) -> int:
    n = 0
    for atom in mol.GetAtoms():
        ri = mol.GetRingInfo()
        if not ri.IsAtomInRingOfSize(atom.GetIdx(), 3):
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() != atom.GetIdx() and ri.IsAtomInRingOfSize(nbr.GetIdx(), 3):
                n += 1
                break
    return n


def _num_bridgehead_atoms(mol: Chem.Mol) -> int:
    n = 0
    ri = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        in_rings = [r for r in ri.AtomRings() if atom.GetIdx() in r]
        if len(in_rings) > 1:
            sizes = sorted(set(len(r) for r in in_rings))
            if len(sizes) > 1 or sizes[0] > 6:
                n += 1
    return n


def compute_sa_score(mol: Chem.Mol) -> float:
    """Return the synthetic-accessibility score in [1, 10]. Lower = easier."""
    if mol is None:
        return float("nan")
    fscores = _load_fscores()
    try:
        # Use the modern MorganGenerator API (RDKit >= 2024) to avoid
        # the GetMorganFingerprint deprecation warning, but fall back to
        # the legacy API for older RDKit builds.
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_info = gen.GetCountFingerprint(mol)
        nonzero = {}
        for bit_id, count in fp_info.GetNonzeroElements().items():
            # bit_id may be (bit, radius) tuple on newer RDKit; take just the bit.
            bit = bit_id[0] if isinstance(bit_id, tuple) else bit_id
            nonzero[bit] = count
        fps = nonzero
        score1 = 0.0
        nf = 0
        for bit_id in fps.keys():
            if bit_id in fscores:
                score1 += fscores[bit_id]
                nf += 1
        if nf > 0:
            score1 /= nf
        else:
            score1 = 5.0
    except Exception:
        score1 = 5.0

    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_spiro = _num_spiro_atoms(mol)
    n_bridge = _num_bridgehead_atoms(mol)
    n_macro = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) > 8:
            n_macro += 1

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(max(1, n_chiral))
    spiro_penalty = math.log10(max(1, n_spiro))
    bridge_penalty = math.log10(max(1, n_bridge))
    macro_penalty = math.log10(max(1, n_macro))
    score2 = (
        0.0
        - size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macro_penalty
    )

    score3 = 0.0
    n_rings = mol.GetRingInfo().NumRings()
    if n_atoms > 0:
        score3 = math.log10(max(1, n_rings)) / n_atoms
    score3 = -score3

    # score4 = -fsp3 (kept from the original; density 1 - sp3 ratio)
    n_sp3 = 0
    for atom in mol.GetAtoms():
        if str(atom.GetHybridization()) == "SP3":
            n_sp3 += 1
    score4 = 0.0
    if n_atoms > 0:
        score4 = (n_sp3 / n_atoms) - 0.5

    sa = score1 + score2 + score3 + score4
    # normalise to the [1, 10] scale reported by Ertl & Schuffenhauer
    sa = max(1.0, min(10.0, sa))
    return float(sa)


__all__ = ["compute_sa_score"]
