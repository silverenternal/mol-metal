"""CrossDocked2020 filtering helpers — pull MMP-related pairs by receptor PDB.

This module sits next to :mod:`molmetal.data.crossdocked` and exposes a
single targeted operation: **given a list of PDB IDs, return the
CrossDocked pairs whose receptor (pocket) matches one of them.**

How the receptor PDB is recovered from a pair path
--------------------------------------------------
CrossDocked pair paths follow::

    {pocket_relpath}/X{pdb_id}{chain}_rec_{ref_pdb}_{lig_code}_lig_tt_{min|docked}_{n}_pocket10.pdb
    {ligand_relpath}/X{pdb_id}{chain}_rec_{ref_pdb}_{lig_code}_lig_tt_{min|docked}_{n}.sdf

where ``X`` is either an underscore-prefixed lowercase 4-character PDB
id (e.g. ``1hov_A_rec_1hov_xxx_lig_tt_min_0_pocket10.pdb``).  The
receptor PDB is the **first** 4-char lowercase token *before* the
chain letter.

Examples
--------
>>> ds = CrossDockedDataset(...)
>>> from molmetal.data.mmp_targets import MMP9_TARGET
>>> entries = CrossDockedDataset.filter_by_pdb(ds, ["1GKC"])
>>> len(entries)
1
>>> entries[0].keys()
dict_keys(['idx', 'pocket_pdb_path', 'ligand_sdf_path', 'pocket_pdb_relpath', 'ligand_sdf_relpath', 'affinity', 'receptor_pdb'])
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from molmetal.data.crossdocked import CrossDockedDataset


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CrossDockedEntry:
    """One filtered CrossDocked2020 pair, keyed by receptor PDB.

    Attributes
    ----------
    idx : int
        Index in the parent :class:`CrossDockedDataset`.
    pocket_pdb_path : str
        Absolute path to the pocket PDB (after extraction, if available).
    ligand_sdf_path : str
        Absolute path to the ligand SDF.
    pocket_pdb_relpath : str
        Original relative path inside the tarball.
    ligand_sdf_relpath : str
        Original relative path inside the tarball.
    affinity : float
        Reported binding affinity (kcal/mol); ``NaN`` if not parsed
        from the index file.
    receptor_pdb : str
        4-character PDB id of the receptor (uppercase, ``"1GKC"``).
    split : str
        ``"train"`` or ``"test"`` from the parent dataset.
    """

    idx: int
    pocket_pdb_path: str
    ligand_sdf_path: str
    pocket_pdb_relpath: str
    ligand_sdf_relpath: str
    affinity: float
    receptor_pdb: str
    split: str = "train"


@dataclass
class FilterStats:
    """Summary statistics from a :func:`filter_by_pdb` call.

    Attributes
    ----------
    total_pairs : int
        Total pairs in the parent dataset (``len(ds)``).
    matching_pairs : int
        Number of pairs whose receptor PDB is in ``pdb_ids``.
    unique_receptor_pdbs : set of str
        Unique PDB IDs that were matched.
    requested_pdbs : set of str
        PDB IDs the caller asked for.
    missing_pdbs : set of str
        PDB IDs that were asked for but **not** found in the dataset.
    mean_ligand_atoms : float
        Mean heavy-atom count across matching ligands (``0.0`` if no
        matches).
    mean_pocket_atoms : float
        Mean atom count across matching pocket PDBs (``0.0`` if no
        matches).
    """

    total_pairs: int = 0
    matching_pairs: int = 0
    unique_receptor_pdbs: Set[str] = field(default_factory=set)
    requested_pdbs: Set[str] = field(default_factory=set)
    missing_pdbs: Set[str] = field(default_factory=set)
    mean_ligand_atoms: float = 0.0
    mean_pocket_atoms: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_pairs": self.total_pairs,
            "matching_pairs": self.matching_pairs,
            "unique_receptor_pdbs": sorted(self.unique_receptor_pdbs),
            "requested_pdbs": sorted(self.requested_pdbs),
            "missing_pdbs": sorted(self.missing_pdbs),
            "mean_ligand_atoms": self.mean_ligand_atoms,
            "mean_pocket_atoms": self.mean_pocket_atoms,
        }


# ---------------------------------------------------------------------------
# Receptor-PDB recovery
# ---------------------------------------------------------------------------
# Match the 4-char lowercase PDB id (pre-chain) at the start of the
# CrossDocked basename — e.g. ``1hov_A_rec_...`` -> ``1hov``.
_PDB_RE = re.compile(r"^([0-9][a-z0-9]{3})(?=[._])")


def _extract_receptor_pdb(relpath: str) -> Optional[str]:
    """Extract the 4-character PDB id from a CrossDocked pair's relative
    path.  Returns ``None`` if the prefix is unrecognised."""
    basename = os.path.basename(relpath)
    m = _PDB_RE.match(basename)
    if not m:
        return None
    return m.group(1).upper()


# ---------------------------------------------------------------------------
# File-level utilities (count atoms without RDKit-dependency on import)
# ---------------------------------------------------------------------------
def _count_ligand_atoms(sdf_path: str) -> int:
    """Count ligand atoms in an SDF file (count of non-bond ``>  <n_atoms>``
    tags).  Falls back to 0 on error."""
    try:
        if not Path(sdf_path).exists():
            return 0
        with open(sdf_path, "r", errors="replace") as f:
            head = f.read(4096)
        # SDF format: the count line appears before the atom block.
        # We split on newlines and look at the 4th line (0-indexed: 3).
        lines = head.splitlines()
        if len(lines) >= 4:
            try:
                return int(lines[3].split()[0])
            except (ValueError, IndexError):
                pass
        return 0
    except Exception:
        return 0


def _count_pocket_atoms(pdb_path: str) -> int:
    """Count ATOM/HETATM lines in a pocket PDB.  Falls back to 0 on error."""
    try:
        if not Path(pdb_path).exists():
            return 0
        n = 0
        with open(pdb_path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    n += 1
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------
def filter_by_pdb(
    ds: CrossDockedDataset,
    pdb_ids: Sequence[str],
    *,
    compute_stats: bool = True,
    max_entries: Optional[int] = None,
) -> Tuple[List[CrossDockedEntry], FilterStats]:
    """Return the CrossDocked entries whose receptor PDB matches ``pdb_ids``.

    Parameters
    ----------
    ds : CrossDockedDataset
        Loaded CrossDocked2020 dataset (any split).
    pdb_ids : Sequence of str
        Upper- or lowercase PDB IDs to retain (e.g. ``["1GKC", "1HOV"]``).
    compute_stats : bool
        If True, count ligand and pocket atoms on disk to populate
        ``mean_ligand_atoms`` / ``mean_pocket_atoms``.
    max_entries : int or None
        Cap the result list size for fast iteration (handy for tests).

    Returns
    -------
    entries : list of CrossDockedEntry
        Matching pairs in dataset order.
    stats : FilterStats
        Diagnostic summary.

    Notes
    -----
    The filter does **not** load the actual PDB/SDF files into memory —
    it only inspects the pair's relative path.  Atom counting is best-
    effort and returns 0 for entries whose files are not yet extracted.
    """
    wanted = {p.upper() for p in pdb_ids}
    entries: List[CrossDockedEntry] = []
    seen_pdbs: Set[str] = set()
    lig_atom_counts: List[int] = []
    poc_atom_counts: List[int] = []

    n_total = len(ds)
    for idx in range(n_total):
        item = ds[idx]
        receptor = _extract_receptor_pdb(str(item.get("pocket_pdb_relpath", "")))
        if receptor is None or receptor not in wanted:
            continue
        entry = CrossDockedEntry(
            idx=int(idx),
            pocket_pdb_path=str(item.get("pocket_pdb_path", "")),
            ligand_sdf_path=str(item.get("ligand_sdf_path", "")),
            pocket_pdb_relpath=str(item.get("pocket_pdb_relpath", "")),
            ligand_sdf_relpath=str(item.get("ligand_sdf_relpath", "")),
            affinity=float(item.get("affinity", float("nan"))),
            receptor_pdb=receptor,
            split=str(item.get("split", "train")),
        )
        if compute_stats:
            lig_atom_counts.append(_count_ligand_atoms(entry.ligand_sdf_path))
            poc_atom_counts.append(_count_pocket_atoms(entry.pocket_pdb_path))
        entries.append(entry)
        seen_pdbs.add(receptor)
        if max_entries is not None and len(entries) >= max_entries:
            break

    stats = FilterStats(
        total_pairs=int(n_total),
        matching_pairs=int(len(entries)),
        unique_receptor_pdbs=seen_pdbs,
        requested_pdbs=set(wanted),
        missing_pdbs=set(wanted) - seen_pdbs,
        mean_ligand_atoms=float(np.mean(lig_atom_counts)) if lig_atom_counts else 0.0,
        mean_pocket_atoms=float(np.mean(poc_atom_counts)) if poc_atom_counts else 0.0,
    )
    return entries, stats


# ---------------------------------------------------------------------------
# Patch CrossDockedDataset with the filter method for API compatibility
# ---------------------------------------------------------------------------
def _filter_by_pdb_method(
    self: CrossDockedDataset,
    pdb_ids: Iterable[str],
    *,
    compute_stats: bool = True,
    max_entries: Optional[int] = None,
) -> List[CrossDockedEntry]:
    """Instance method form — convenience over :func:`filter_by_pdb`."""
    entries, _stats = filter_by_pdb(
        self,
        list(pdb_ids),
        compute_stats=compute_stats,
        max_entries=max_entries,
    )
    return entries


# Attach as an unbound method so callers can write ``ds.filter_by_pdb(...)``
if not hasattr(CrossDockedDataset, "filter_by_pdb"):
    CrossDockedDataset.filter_by_pdb = _filter_by_pdb_method  # type: ignore[attr-defined]


__all__ = [
    "CrossDockedEntry",
    "FilterStats",
    "filter_by_pdb",
]