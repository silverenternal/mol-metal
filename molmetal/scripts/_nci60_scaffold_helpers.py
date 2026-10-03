"""Wf-319E — Pure-Python NCI-60 scaffold-split helpers.

================================================================
Extracted from ``molmetal/scripts/nci60_9cell_chemprop_driver.py``
(Wf-315F) so the helper logic can be imported and unit-tested
without the chemprop v2 driver import chain (which requires
``cuik_molmaker``).

This module depends only on numpy, pandas, and rdkit (no chemprop
v2 / torch). The driver imports these helpers as
``from molmetal.scripts._nci60_scaffold_helpers import (...)`` so
the public API is unchanged.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np


def _stable_hash_int(text: str, modulus: int) -> int:
    """Process-independent hash of *text* into ``[0, modulus)``.

    Wf-A10 FIX (2026-10-03) — Python salts ``hash()`` of ``str`` per process
    (PYTHONHASHSEED), so the two call sites below produced DIFFERENT values on
    every run.  MEASURED for ``hash("CCO") % 1024`` across three processes:
    519 / 244 / 396.  That silently made (a) the scaffold fallback grouping —
    and therefore the train/val/test split — irreproducible, and (b) the
    reported ``scaffold_ood_group_*`` run label differ run to run, so runs
    could not be compared or tracked by label.

    A SHA-256 digest is fixed, so the same input always yields the same id on
    any process and any machine.  Mirrors ``_stable_hash_int`` in
    ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py`` (same fix class).

    NOTE: this defect was LATENT — verified no shipped metric contains a
    ``scaffold_ood_group_*`` or ``__circular_*`` string, so no published number
    was contaminated.  It is fixed to stop it becoming live.
    """
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulus)


__all__ = [
    "_scaffold_split_indices_nci60",
    "compute_per_cell_n_test_min",
]


def _scaffold_split_indices_nci60(
    *,
    smiles_subset: List[str],
    active_subset: np.ndarray,
    seed: int,
    n_test_min: int = 50,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Hold out one Bemis-Murcko scaffold group per seed for OOD test.

    Mirrors the Wf-312A pattern from
    ``wf219_temporal_split_driver._scaffold_split_indices`` but
    adapted to NCI-60 cells (which lack a temporal Year column). For
    DU145 (161 rows / ~110 scaffolds) or other small cells where no
    scaffold group has >= n_test_min rows, returns ``None`` so the
    caller can fall back to the random 80/10/20 split.

    Returns ``None`` if no scaffold group has >= n_test_min rows.
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffolds: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_subset):
        try:
            sc = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            sc = ""
        if not sc:
            # Wf-A10 FIX: stable hash (was `hash(smi) % 1024`, salted per process).
            sc = f"__circular_{_stable_hash_int(smi, 1024)}__"
        scaffolds.setdefault(sc, []).append(int(i))

    # Sort scaffold groups by size desc, then randomise seed.
    rng = np.random.default_rng(seed)
    groups = sorted(scaffolds.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    # Greedily pick the first group that gives >= n_test_min rows.
    for sc, idx_list in shuffled_groups:
        if len(idx_list) < n_test_min:
            continue
        test_idx = np.asarray(idx_list, dtype=np.int64)
        # 90/10 train/val on the rest, seeded.
        rest = [i for sc_other, idxs in groups if sc_other != sc
                for i in idxs]
        rest_arr = np.asarray(rest, dtype=np.int64)
        rng.shuffle(rest_arr)
        n_val = max(int(round(0.10 * len(rest_arr))), 1)
        val_idx = rest_arr[:n_val]
        train_idx = rest_arr[n_val:]
        # Wf-A10 FIX: stable hash (was `abs(hash(sc)) % 10**8`, salted per process).
        label = f"scaffold_ood_group_{_stable_hash_int(sc, 10**8)}_n={len(idx_list)}"
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
            label,
        )

    return None


def compute_per_cell_n_test_min(
    *,
    smiles_subset: List[str],
    n_test_min_floor: int = 5,
    headroom_factor: float = 0.85,
) -> int:
    """Wf-319E: auto-compute per-cell scaffold threshold.

    Per Wf-315F §3 honest finding, no NCI-60 cell has a Bemis-Murcko
    scaffold group with >= 50 rows; the 3 largest cells (A549/MCF7/
    A2780) have max scaffold sizes 34-37. With default n_test_min=50
    the scaffold helper always returns None -> random fallback. To
    make the scaffold fallback actually trigger for the 3 largest
    cells (and gracefully fall back for the smaller 6), we
    auto-compute per-cell n_test_min = max(round(headroom_factor *
    max_scaffold_size), n_test_min_floor).

    Worked examples (MEASURED max scaffold sizes from Wf-315F §3):
      * A549 (max=37)    -> 0.85 * 37 = 31 (floor 5) -> n_test_min=31
      * MCF7 (max=34)    -> 0.85 * 34 = 28             -> n_test_min=28
      * A2780 (max=37)   -> 0.85 * 37 = 31             -> n_test_min=31
      * A2780cis (max=30)-> 0.85 * 30 = 26             -> n_test_min=26
      * HCT116 (max=20)  -> 0.85 * 20 = 17             -> n_test_min=17
      * SKOV3 (max=18)   -> 0.85 * 18 = 15             -> n_test_min=15
      * HL60 (max=18)    -> 0.85 * 18 = 15             -> n_test_min=15
      * DU145 (max=10)   -> 0.85 * 10 = 9 (floor 5)    -> n_test_min=9
      * PC3 (max=9)      -> 0.85 * 9 = 8 (floor 5)     -> n_test_min=8

    Returns ``n_test_min_floor`` if the SMILES list is empty or all
    scaffolds fail to parse (defensive default).
    """
    if not smiles_subset:
        return int(n_test_min_floor)
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffolds: Dict[str, int] = {}
    for smi in smiles_subset:
        try:
            sc = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            sc = ""
        if not sc:
            continue
        scaffolds[sc] = scaffolds.get(sc, 0) + 1
    if not scaffolds:
        return int(n_test_min_floor)
    max_size = max(scaffolds.values())
    return max(int(round(headroom_factor * max_size)), int(n_test_min_floor))