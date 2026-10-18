"""Cell-line block and native categorical codes (FIX-P1 step 3).

Two encodings coexist in the project for a reason, and both are preserved here:

* ``one_hot`` — top-N cell lines plus an ``other`` bucket.  This is what
  ``msrs_surface_model.cell_matrix`` does and what every pre-AUD-F5 artifact
  scored.
* ``codes`` — integer codes handed to LightGBM's ``categorical_feature`` so it
  can split on the full level set instead of 41 binary columns.  AUD-F5 measured
  this as the larger of the two levers.

The vocabulary is always supplied by the caller.  Deriving it inside the block
would let a train/test split change the column meaning mid-run.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse

OTHER = "__other__"


def cell_vocabulary(
    cells: Sequence[str], *, top_n: int = 40, mode: str = "top40"
) -> List[str]:
    """Most frequent cell lines, or every level in sorted order for ``full``.

    Delegates to ``Series.value_counts`` so the ordering matches the historical
    ``sub["cell_line_key"].value_counts().head(40)`` exactly, ties included.
    """
    import pandas as pd

    values = pd.Series([str(c) for c in cells])
    if mode == "full":
        return sorted(values.unique().tolist())
    if mode == "none":
        return []
    return values.value_counts().head(top_n).index.tolist()


def cell_block(
    cells: Sequence[str], vocabulary: Sequence[str]
) -> Tuple[sparse.csr_matrix, List[str]]:
    """One-hot with an ``other`` column, matching ``cell_matrix`` exactly."""
    index = {c: i for i, c in enumerate(vocabulary)}
    other = len(vocabulary)
    rows = np.arange(len(cells))
    cols = np.array([index.get(str(c), other) for c in cells], dtype=np.int32)
    matrix = sparse.csr_matrix(
        (np.ones(len(cells), dtype=np.float32), (rows, cols)),
        shape=(len(cells), len(vocabulary) + 1),
        dtype=np.float32,
    )
    names = [f"cell_{c}" for c in vocabulary] + ["cell_other"]
    return matrix, names


def native_codes(
    metals: Sequence[str],
    cells: Sequence[str],
    cell_vocabulary: Sequence[str],
    *,
    mode: str = "top40",
    metal_vocabulary: Sequence[str] = ("Ru", "Ir", "Rh", "Os", "Re"),
) -> np.ndarray:
    """``(n, 2)`` int32 codes: [metal, cell], for LightGBM categorical splits.

    ``full`` sorts every observed level; ``top40`` keeps the supplied vocabulary
    and sends unseen lines to ``__other__``.  Extra metals beyond the cohort
    vocabulary (e.g. the auxiliary ``Pt`` arm) are appended in sorted order so
    codes stay stable across runs.
    """
    extras = sorted(set(map(str, metals)) - set(metal_vocabulary))
    metals_all = tuple(metal_vocabulary) + tuple(extras)
    metal_index = {m: i for i, m in enumerate(metals_all)}

    cell_values = [str(c) for c in cells]
    if mode == "full":
        levels = {c: i for i, c in enumerate(sorted(set(cell_values)))}
    elif mode == "top40":
        levels = {c: i for i, c in enumerate(cell_vocabulary)}
        levels[OTHER] = len(cell_vocabulary)
    else:
        raise ValueError(f"native_codes: unknown mode {mode!r}")

    other = levels.get(OTHER, len(levels))
    metal = np.array([metal_index[m] for m in metals], dtype=np.int32)
    cell = np.array([levels.get(c, other) for c in cell_values], dtype=np.int32)
    return np.column_stack([metal, cell])


def codes_summary(codes: np.ndarray) -> Dict[str, int]:
    """Small provenance record: cardinality per code column."""
    return {
        "n_metals": int(len(np.unique(codes[:, 0]))),
        "n_cells": int(len(np.unique(codes[:, 1]))),
    }
