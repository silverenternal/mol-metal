"""172-d per-fragment descriptor table (FIX-P1 step 5).

``[RDKit-12 | Multi-PH-32 | PLL-128]`` computed once per unique SMILES.  This
used to live inside ``msrs_172d_rebuild.py``; it moved here so the table has one
construction site and a spec hash, like every other block.

Failed streams stay ``NaN``.  ``compute_172d`` zero-fills them, which is
indistinguishable from a real zero to a tree model — LightGBM's native NaN
handling is the honest representation of "not computed".
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

#: Re-exported so callers do not import the builder script directly.
from molmetal.scripts.build_172d_features import (  # noqa: F401
    MULTI_PH_DIM,
    PLL_DIM,
    RDKIT_DIM as RDKIT12_DIM,
    compute_multiph_32d,
    compute_pll_128d,
    compute_rdkit_12d,
)

FEATURE_DIM = RDKIT12_DIM + MULTI_PH_DIM + PLL_DIM  # 172


def fragment_descriptor_row(smiles: str) -> Tuple[np.ndarray, Dict[str, bool]]:
    """One 172-d row with NaN on any stream that failed for this molecule."""
    vec = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    ok = {"rdkit": False, "multiph": False, "pll": False}

    rd = compute_rdkit_12d(smiles)
    if rd is not None and len(rd) == RDKIT12_DIM:
        vec[:RDKIT12_DIM] = np.asarray(rd, dtype=np.float32)
        ok["rdkit"] = True

    mph = compute_multiph_32d(smiles)
    if mph is not None and mph.size == MULTI_PH_DIM:
        vec[RDKIT12_DIM : RDKIT12_DIM + MULTI_PH_DIM] = mph
        ok["multiph"] = True

    pll = compute_pll_128d(smiles)
    if pll is not None and pll.size == PLL_DIM:
        vec[RDKIT12_DIM + MULTI_PH_DIM :] = pll
        ok["pll"] = True

    return vec, ok


def descriptor_table(
    smiles: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Feature matrix + per-stream boolean masks, one row per SMILES."""
    n = len(smiles)
    X = np.full((n, FEATURE_DIM), np.nan, dtype=np.float32)
    masks = {k: np.zeros(n, dtype=bool) for k in ("rdkit", "multiph", "pll")}
    for i, smi in enumerate(smiles):
        vec, ok = fragment_descriptor_row(smi)
        X[i] = vec
        for k, v in ok.items():
            masks[k][i] = v
    return X, masks


def column_names() -> list[str]:
    """Stable names, in block order, for a reader of the matrix."""
    names = [f"rdkit12_{i}" for i in range(RDKIT12_DIM)]
    names += [f"multiph32_{i}" for i in range(MULTI_PH_DIM)]
    names += [f"pll128_{i}" for i in range(PLL_DIM)]
    return names
