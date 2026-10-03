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
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# T2-FIX (2026-10-03): machine-readable enum for the split path actually
# taken.  Emitted by the driver per (cell, seed) and by
# ``write_split_manifest`` so a gate can assert, e.g.,
# ``path_taken == "scaffold_ood"`` after requesting a scaffold split.
SPLIT_PATH_SCAFFOLD_OOD = "scaffold_ood"
SPLIT_PATH_RANDOM_FALLBACK = "random_fallback"
SPLIT_PATH_RANDOM_TEMPORAL = "random_temporal"
SPLIT_PATH_UNKNOWN = "unknown"


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
    "write_split_manifest",
    "SPLIT_PATH_SCAFFOLD_OOD",
    "SPLIT_PATH_RANDOM_FALLBACK",
    "SPLIT_PATH_RANDOM_TEMPORAL",
    "SPLIT_PATH_UNKNOWN",
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

    T2-FIX (2026-10-03): the ``None`` fallback is no longer silent.
    MEASURED on ``nci60_precious_9cell.csv``: at the default
    ``n_test_min=50`` *all nine* cells hit this branch (max scaffold
    group sizes 9-37), so every ``--split scaffold`` run produced a
    random 80/10/20 split while run metadata still said "scaffold".
    The caller must now decide with the fallback in plain sight: this
    function emits ``logging.warning`` + ``RuntimeWarning`` naming the
    seed, threshold, row count, and max group size, and the driver
    records ``path_taken="random_fallback"`` per (cell, seed) plus a
    JSON split manifest (``write_split_manifest``).
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

    # T2-FIX: LOUD fallback.  Was: silent ``return None`` with the caller
    # quietly substituting the random 80/10/20 split (which is exactly how
    # 9/9 cells stayed "scaffold" in name only).  Belt-and-braces on both
    # channels: logging (structured, driver logs) + warnings (visible even
    # when a library consumer reconfigured logging).
    max_group_size = max((len(v) for v in scaffolds.values()), default=0)
    msg = (
        f"NCI-60 scaffold split FALLBACK (seed={seed}, n_test_min={n_test_min}): "
        f"no Bemis-Murcko scaffold group has >= n_test_min rows "
        f"(n_rows={len(smiles_subset)}, n_groups={len(scaffolds)}, "
        f"max_group_size={max_group_size}). The caller will use the random "
        f"80/10/20 split — downstream results are NOT scaffold-OOD "
        f"(path_taken=random_fallback)."
    )
    logger.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
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

    Worked examples (max scaffold sizes MEASURED from Wf-315F §3; the
    returned thresholds re-MEASURED T2-FIX 2026-10-03 — note MCF7 and
    DU145 in older comments were off by one; ``round`` is banker's):
      * A549 (max=37)    -> 0.85 * 37 = 31 (floor 5) -> n_test_min=31
      * MCF7 (max=34)    -> 0.85 * 34 = 28.9 -> round -> n_test_min=29
      * A2780 (max=37)   -> 0.85 * 37 = 31             -> n_test_min=31
      * A2780cis (max=30)-> 0.85 * 30 = 26             -> n_test_min=26
      * HCT116 (max=20)  -> 0.85 * 20 = 17             -> n_test_min=17
      * SKOV3 (max=18)   -> 0.85 * 18 = 15             -> n_test_min=15
      * HL60 (max=18)    -> 0.85 * 18 = 15             -> n_test_min=15
      * DU145 (max=10)   -> round(8.5) = 8 (banker's)  -> n_test_min=8
      * PC3 (max=9)      -> round(7.65) = 8 (floor 5)  -> n_test_min=8

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


def write_split_manifest(
    path: Union[str, Path],
    records: List[Dict[str, Any]],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """T2-FIX: write a machine-readable per-(cell, seed) split manifest.

    Cheap trace that makes the scaffold-vs-random path auditable after the
    fact: each record carries ``path_taken`` (one of the
    ``SPLIT_PATH_*`` constants — ``scaffold_ood`` means the run's test
    partition really is a held-out Bemis-Murcko scaffold group;
    ``random_fallback`` means a scaffold split was requested but could
    not be built, so the run silently-equivalent-to-random 80/10/20 was
    used) plus the split label, run status, and n_train/n_val/n_test.

    Format (``nci60_split_manifest_v1``)::

        {"schema": "nci60_split_manifest_v1",
         "n_records": N,
         "records": [{cell, seed, path_taken, split_label, status,
                      n_train, n_val, n_test, ...}, ...],
         "extra": {...}}

    A gate can load this and assert ``path_taken == "scaffold_ood"`` for
    every record when a scaffold-OOD run was requested.  Cost: one small
    JSON file (<1 MB for 9 cells x any seed count); no index hashes are
    included yet (would need the index arrays to leave
    ``run_one_per_cell``), values are ``None`` where unknown, never NaN.

    Returns the path written.
    """
    p = Path(path)
    payload: Dict[str, Any] = {
        "schema": "nci60_split_manifest_v1",
        "n_records": len(records),
        "records": [dict(r) for r in records],
        "extra": dict(extra or {}),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p
