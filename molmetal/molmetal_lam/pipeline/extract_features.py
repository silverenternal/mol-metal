"""8-d feature extraction for :class:`MoleculeClosedTerm`.

============================================================
Why a fixed-length 8-d vector?
============================================================
The MCTS heuristic (a) the symbolic-regression wrapper, and (b) any
downstream classifier (binding-type inhabitants, ADMET filter, …) all
need a *fixed-length* numerical representation of a closed λ-term.

Eight hand-picked descriptors cover the dominant axis of variation in
drug-like molecules:

    idx | name             | source            | meaning
    ----|------------------|-------------------|---------------------------
     0  | n_atoms          | term              | heavy-atom count
     1  | mw               | RDKit Descriptors | molecular weight (Da)
     2  | logp             | RDKit Descriptors | Wildman-Crippen logP
     3  | tpsa             | RDKit Descriptors | topological polar SA (Å²)
     4  | n_clicks_used    | term              | how many reactions fired
     5  | n_aromatic_rings | RDKit             | aromatic ring count
     6  | n_rotatable      | RDKit Descriptors | rotatable-bond count
     7  | n_h_donors       | RDKit             | Lipinski HBD count

The first four are *term-level* descriptors (size + lipophilicity +
polarity), the next four are *chemotype + flexibility* descriptors.
Combined they give the PySR regressor enough handles to discover a
meaningful closed-form heuristic like
``score ≈ 0.4*QED + 0.2*(1 - tpsa/150)``.

The vector is intentionally *fixed-length* — when the input SMILES
fails to parse we return the all-zeros fallback.  This is required by
the PySR/sklearn contract: every sample must have the same shape.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

# Lazy RDKit imports so this module remains importable in pure-MLC envs.
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_RDKIT = False


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "n_atoms",
    "mw",
    "logp",
    "tpsa",
    "n_clicks_used",
    "n_aromatic_rings",
    "n_rotatable",
    "n_h_donors",
]
"""Human-readable feature names.  Index ``i`` of every feature vector
returned by :func:`molecule_to_features` corresponds to
``FEATURE_NAMES[i]``."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def molecule_to_features(m: MoleculeClosedTerm) -> np.ndarray:
    """Return an 8-d feature vector for the given closed λ-term.

    Parameters
    ----------
    m : MoleculeClosedTerm
        Any molecule — the closed-term layer takes care of falling back
        to a zero-vector when SMILES parsing fails (no exceptions).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(8,)`` with the layout described in
        :data:`FEATURE_NAMES`.  Always finite (no NaN/inf), so the
        output can be fed directly into PySR / sklearn.
    """
    n_atoms = float(m.n_atoms) if m is not None else 0.0
    # ``n_clicks_used`` = total β-reductions performed on this term
    # (a coarser proxy for ``n_bonds``).  We use ``n_covalent_bonds``
    # so hydrogen bonds (RDKit bookkeeping) don't dominate.
    n_clicks = float(getattr(m, "n_covalent_bonds", 0) if m is not None else 0)

    if not _HAVE_RDKIT or m is None:  # pragma: no cover - environment dependent
        return np.zeros(8, dtype=np.float32)

    smi = m.canonical_smiles if hasattr(m, "canonical_smiles") else None
    if not smi:
        return np.array([n_atoms, 0.0, 0.0, 0.0, n_clicks, 0.0, 0.0, 0.0], dtype=np.float32)

    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        mol = None
    if mol is None:
        return np.array([n_atoms, 0.0, 0.0, 0.0, n_clicks, 0.0, 0.0, 0.0], dtype=np.float32)

    try:
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        mw = 0.0
    try:
        logp = float(Descriptors.MolLogP(mol))
    except Exception:
        logp = 0.0
    try:
        tpsa = float(Descriptors.TPSA(mol))
    except Exception:
        tpsa = 0.0
    try:
        n_ar = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        n_ar = 0.0
    try:
        n_rot = float(Descriptors.NumRotatableBonds(mol))
    except Exception:
        n_rot = 0.0
    try:
        n_hbd = float(rdMolDescriptors.CalcNumHBD(mol))
    except Exception:
        n_hbd = 0.0

    vec = np.array(
        [n_atoms, mw, logp, tpsa, n_clicks, n_ar, n_rot, n_hbd],
        dtype=np.float32,
    )
    # Guard against NaN/inf from RDKit edge cases on weird input.
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def molecules_to_features(mols: Iterable[MoleculeClosedTerm]) -> np.ndarray:
    """Convenience batch wrapper — returns ``(n_mols, 8)`` array."""
    return np.stack([molecule_to_features(m) for m in mols], axis=0)


__all__ = ["FEATURE_NAMES", "molecule_to_features", "molecules_to_features"]
