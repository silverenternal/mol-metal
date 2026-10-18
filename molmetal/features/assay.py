"""Assay-level blocks: counterion identity and component count (FIX-P1 step 4).

These are not chemistry — they describe the salt form and the stoichiometry of
the sample that was weighed out.  They are kept separate from the ligand
fingerprint so an ablation can drop them without re-deriving the matrix, and so
a reader can tell which part of a score came from the complex and which from
its counterion.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy import sparse


def counterion_block(
    counterions: Sequence[object], vocabulary: Sequence[str]
) -> Tuple[sparse.csr_matrix, List[str]]:
    """One-hot over a fixed counterion vocabulary, plus an ``other`` column.

    ``None`` and ``""`` both mean "no counterion"; they share the same level
    because the cohort CSV spells the absence both ways.
    """
    index = {c: i for i, c in enumerate(vocabulary)}
    other = len(vocabulary)
    out = np.zeros((len(counterions), len(vocabulary) + 1), dtype=np.float32)
    for i, raw in enumerate(counterions):
        key = "" if raw is None else str(raw)
        out[i, index.get(key, other)] = 1.0
    names = [f"counterion_{c}" for c in vocabulary] + ["counterion_other"]
    return sparse.csr_matrix(out), names


def counterion_vocabulary(
    counterions: Sequence[object], *, top_n: int = 32
) -> List[str]:
    """Most frequent counterions; ``""`` is a level, not a missing value."""
    import pandas as pd

    values = pd.Series(["" if c is None else str(c) for c in counterions])
    return values.value_counts().head(top_n).index.tolist()


def n_fragments_block(smiles: Sequence[object]) -> Tuple[sparse.csr_matrix, List[str]]:
    """Number of disconnected components per row.

    A single-component ligand and a five-component complex are different
    objects; the count lets the model see that without naming the components.
    """
    from molmetal.features.ligand import component_smiles

    counts = np.array(
        [len(component_smiles(s)) for s in smiles], dtype=np.float32
    ).reshape(-1, 1)
    return sparse.csr_matrix(counts), ["n_components"]
