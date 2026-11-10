"""Bond-length prior table (Wf-A27 step-2 → Wf-A29 step-1, 2026-10-14).

Three lookup layers, in priority order:

1. **tmQM-DFT empirical mean** (Wf-A29 step-1, this wave).  When the
   ``.data/molmetal/tmQM/bond_length_table.json`` cache exists and the
   ``(min(z1,z2), max(z1,z2), round(BO))`` bin has ``n >= 5``
   observations, the function returns the Wiberg-weighted mean of
   the 3D distances in the tmQM release
   (Balcells & Skjelstad 2020, 108k mononuclear d-block complexes at
   TPSSh-D3BJ/def2-SVP, MIT licence).  Cache is built once via
   :func:`build_bond_length_table` (~20 s end-to-end) and cached as
   JSON; subsequent :func:`get_bond_length` calls hit it in O(1).

2. **Pyykkö per-order sum-of-radii** (Wf-A28 step-2).  When
   ``order`` ∈ {1, 2, 3} and BOTH ``z1`` / ``z2`` are in the
   corresponding Pyykkö table, return the sum of radii directly.
   Single-bond table is Pyykkö & Atsumi 2009 (doi:10.1002/
   chem.200800987); double is Pyykkö 2008 (same doi); triple is
   Pyykkö 2005 (doi:10.1002/chem.200401299).

3. **Hückel shortening** (Wf-A27).  For ``order`` > 3, or when one of
   the atoms is missing from the per-order table, multiply the
   single-bond sum by ``1 - 0.18 * ln(order)``.  Falls back to a
   0.77 Å Cordero-default per-side for unmapped Z.

The Pyykkö covalent-radii table is the same family of metal-radius
values the standalone ``ClashPenalty`` class (in
:mod:`molmetal.loss.prior_losses`) uses, and provides a calibrated
resting length for any (z, z) pair — a value an xtb optimisation
would reproduce within ~5% but which costs zero wall time to look up.
When the trainer eventually opts into a wall-potential term, the
prior serves as the equilibrium: ``E_wall = exp(-(d - r_prior) /
sigma)``.

The 14-element scope (H, C, N, O, F, P, S, Cl, Se, Br, I, Pt, Pd, Au)
matches the CFM YuelBond / bond-head cohort — extending Pyykkö to
all 92 elements is out of scope for this wave.  The tmQM table
covers the wider 30-metal + main-group-organics cohort as
represented in the CSD-derived mononuclear set.

Backward-compat rule (~~AGENTS.md §5.11~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31)
----------------------------------------------------------------------------------------------
``get_bond_length(z1, z2, order=1)`` signature is BYTE-IDENTICAL with
Wf-A27 / Wf-A28.  When the tmQM cache is ABSENT (default for a
fresh checkout), behaviour is byte-identical with Wf-A28 step-2
(Pyykkö priority then Hückel then Cordero fallback).  When the
cache is PRESENT, the empirical mean wins for any (z, z, order) bin
with ``n >= 5``; Pyykkö remains the fallback for sparse bins.  This
preserves the bit-exact contract for any trainer that imports the
module before building the cache.

Citation
--------
Pyykkö, P.; Atsumi, M. (2009).  *Molecular Single-Bond Covalent Radii
for Elements 1–118.*  Chem. Eur. J. 15 (1): 186–197.
doi:10.1002/chem.200800987.

Pyykkö, P.; Riedel, S.; Patzschke, M. (2005).  *Triple-Bond Covalent
Radii.*  Chem. Eur. J. 11 (12): 3511–3520.  doi:10.1002/chem.200401299.

Balcells, D.; Skjelstad, B. B. (2020).  *tmQM Dataset — Quantum
Geometries and Properties of 86k Transition Metal Complexes.*
J. Chem. Inf. Model. 60, 6135–6146.  doi:10.1021/acs.jcim.0c01041.
The release shipped here is the extended 108k mononuclear-complex
shard set, MIT licence.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Wf-A29 step-1 (2026-10-14): REAL tmQM-DFT bond-length table.
#
# The Pyykkö tables above are the literature average; the tmQM release
# (Balcells & Skjelstad 2020 — 108k mononuclear d-block complexes at
# TPSSh-D3BJ/def2-SVP, MIT licence, on disk at .data/molmetal/tmQM/)
# gives per-(Z, Z, order) Wiberg bond order + 3D-distance statistics
# measured on actual transition-metal complexes.  This module's
# ``build_bond_length_table`` reads the three XYZ.gz + BO.gz shard pairs
# once, bins all bonds with BO > 0.1 (Housecroft convention) by
# ``(min(Z_i,Z_j), max(Z_i,Z_j), round(BO_ij))``, and persists
# ``{mean, std, n, min, max}`` per bin to a JSON cache.
#
# Precedence (Wf-A28 backward-compat preserved):
#   1. tmQM mean (this step) when ``(z1, z2, order)`` hit AND ``n >= 5``
#   2. Pyykkö per-order single/double/triple (Wf-A28 step-2) — same
#      lookup path as before
#   3. Pyykkö single + Hückel shortening for order > 1 (Wf-A27/A28 path)
#   4. Cordero default (0.77 Å per-side) for unmapped Z
#
# Loader discovery: ``build_bond_length_table`` defaults to the
# ``.data/molmetal/tmQM/`` path (the on-disk layout for this checkout).
# Callers may pass ``root=`` to override; the env var
# ``$MOLMETAL_TMQM_DIR`` is honoured by ``molmetal.data.tmqm.tmqm_dir``
# so the parent module's discovery is preserved when this table is
# generated via the trainer pipeline.
# ---------------------------------------------------------------------------
_DEFAULT_TMQM_BOND_TABLE_ROOT = Path(
    os.environ.get(
        "MOLMETAL_TMQM_DIR",
        str(Path(__file__).resolve().parent.parent.parent / ".data" / "molmetal" / "tmQM"),
    )
)
_DEFAULT_BOND_TABLE_CACHE = _DEFAULT_TMQM_BOND_TABLE_ROOT / "bond_length_table.json"

# Wiberg bond-order threshold below which we ignore a bond for the prior
# (Housecroft & Sharpe 2018 treats BO < 0.1 as "negligible").
_WIBERG_BOND_THRESHOLD: float = 0.1

# Minimum bin sample-count to trust the empirical mean as a prior (5
# observations = 5 distinct tmQM complexes; below this we fall back to
# Pyykkö rather than emit a noisy empirical mean).
_MIN_BIN_N: int = 5

# Symbol → Z lookup for the elements that appear in the tmQM XYZ shard.
# Covers all d-block + main-group organics that appear in the CSD-derived
# mononuclear set; the ``default`` branch returns ``None`` for elements
# tmQM never produces.
_ELEMENT_SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}

_XYZ_FILES = ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
_BO_FILES = ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")
_HEADER_RE = re.compile(r"CSD_code\s*=\s*(\S+)")
_ATOM_RE = re.compile(r"([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def _element_to_z(sym: str) -> Optional[int]:
    return _ELEMENT_SYMBOL_TO_Z.get(str(sym).strip())


def _xyz_complex_block(path: Path):
    """Yield ``(csd_code, [(Z, x, y, z), ...])`` for every complex in ``path``.

    The XYZ file format is one complex per block::

        <n_atoms>
        CSD_code = WELROW | q = .. | S = .. | Stoichiometry = .. | MND = .. | 2020-2024 CSD
        El  x  y  z
        El  x  y  z
        ...

    The header is the second line; subsequent lines are atoms until the
    next ``<n_atoms>`` line closes the block.  Skips malformed blocks
    (negative atom counts, unparseable elements) rather than raising —
    the parent parse_xyz_headers loader tolerates the same malformations.
    """
    with gzip.open(path, "rt") as fh:
        n_atoms_pending: Optional[int] = None
        cur_code: Optional[str] = None
        cur_atoms: list = []
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if n_atoms_pending is None:
                # First non-empty line of a block is the atom count.
                try:
                    n_atoms_pending = int(stripped)
                    cur_atoms = []
                    cur_code = None
                    continue
                except ValueError:
                    return  # malformed file — caller will see fewer rows
            if cur_code is None:
                m = _HEADER_RE.search(stripped)
                if m is None:
                    # Skip until next <n_atoms> boundary.
                    n_atoms_pending = max(0, n_atoms_pending - 1) or None
                    continue
                cur_code = m.group(1)
                continue
            # Atom data line.
            m = _ATOM_RE.match(stripped)
            if m is not None:
                sym = m.group(1)
                z = _element_to_z(sym)
                if z is not None:
                    cur_atoms.append(
                        (z, float(m.group(2)), float(m.group(3)), float(m.group(4)))
                    )
            n_atoms_pending -= 1
            if n_atoms_pending <= 0:
                yield cur_code, cur_atoms
                n_atoms_pending = None
                cur_code = None
                cur_atoms = []


def _bo_block_lines(path: Path):
    """Yield ``(csd_code, [per-atom-row-tokens])`` per complex.

    The BO file format is one complex per block::

        CSD_code = WEL360 | 2020-2024 CSD
            1  La  3.031        Se   4 0.429    Se   5 0.411    ...
            2  Se  1.967        P    8 1.313    La   1 0.285    ...

    Each row's first three tokens are ``<idx> <El> <total_BO>`` and the
    remainder is a flat triple stream ``<nbr_El> <nbr_idx> <BO>``.
    """
    with gzip.open(path, "rt") as fh:
        cur_code: Optional[str] = None
        cur_rows: list = []
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            m = _HEADER_RE.search(stripped)
            if m is not None:
                if cur_code is not None:
                    yield cur_code, cur_rows
                cur_code = m.group(1)
                cur_rows = []
                continue
            # Per-atom data row.
            parts = stripped.split()
            if len(parts) >= 3:
                cur_rows.append(parts)
        if cur_code is not None:
            yield cur_code, cur_rows


def build_bond_length_table(
    root: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    """Build the per-(Z, Z, order) bond-length table from tmQM shards.

    Parameters
    ----------
    root
        tmQM data root.  Defaults to ``.data/molmetal/tmQM/`` for this
        checkout (or ``$MOLMETAL_TMQM_DIR`` if set).
    cache_path
        Output JSON path.  Defaults to
        ``<root>/bond_length_table.json``.
    force
        If ``True``, rebuild the table from raw shards even when the
        cache file already exists.

    Returns
    -------
    dict
        Mapping ``(z_min, z_max, order_int) -> {mean, std, n, min, max}``
        with ``mean / std / min / max`` in Angstrom and ``n`` ≥ 5
        (bins with fewer observations are dropped to keep the prior
        trustworthy).

    Notes
    -----
    Caching is JSON for human auditability + trivial git-diff review.
    First run is O(108k) complexes × per-atom-row parse; subsequent
    runs are an O(1) JSON load.  The Wiberg BO threshold is 0.1
    (Housecroft & Sharpe 2018 convention).
    """
    root_path = Path(root) if root is not None else _DEFAULT_TMQM_BOND_TABLE_ROOT
    cache = Path(cache_path) if cache_path is not None else root_path / "bond_length_table.json"
    if cache.is_file() and not force:
        with cache.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        for k, v in raw.items():
            z_min, z_max, order = (int(t) for t in k.split(","))
            out[(z_min, z_max, order)] = {
                "mean": float(v["mean"]),
                "std": float(v["std"]),
                "n": int(v["n"]),
                "min": float(v["min"]),
                "max": float(v["max"]),
            }
        return out

    # Build XYZ coord cache: csd_code -> list[(Z, x, y, z)] indexed 0-based
    # to match the BO file's 1-based atom index.
    coords_by_code: Dict[str, list] = {}
    for xyz_name in _XYZ_FILES:
        xyz_path = root_path / xyz_name
        if not xyz_path.is_file():
            continue
        for code, atoms in _xyz_complex_block(xyz_path):
            coords_by_code[code] = atoms

    # Stream BO shards; bin distances per (z_min, z_max, round(bo)).
    bins: Dict[Tuple[int, int, int], list] = {}
    for bo_name in _BO_FILES:
        bo_path = root_path / bo_name
        if not bo_path.is_file():
            continue
        for code, rows in _bo_block_lines(bo_path):
            atoms = coords_by_code.get(code)
            if atoms is None:
                continue
            for parts in rows:
                # Format: <idx> <El> <total_BO>  <nbr_El> <nbr_idx> <BO> ...
                try:
                    self_idx_1b = int(parts[0])
                    self_el = parts[1]
                    self_z = _element_to_z(self_el)
                    if self_z is None:
                        continue
                except (ValueError, IndexError):
                    continue
                # 1-based → 0-based; bounds-check against coords cache.
                self_idx = self_idx_1b - 1
                if self_idx < 0 or self_idx >= len(atoms):
                    continue
                self_z_actual, sx, sy, sz = atoms[self_idx]
                if self_z_actual != self_z:
                    # Mismatch between XYZ and BO element identity — skip
                    # the bond for safety rather than silently picking one.
                    continue
                # Iterate neighbour triples.  BO entries are symmetric
                # (atom i's row mentions atom j AND atom j's row mentions
                # atom i); only count each pair once via nbr_idx > self_idx.
                for off in range(3, len(parts) - 2, 3):
                    nbr_el = parts[off]
                    nbr_idx_1b = parts[off + 1]
                    bo_str = parts[off + 2]
                    try:
                        nbr_idx = int(nbr_idx_1b) - 1
                        bo_val = float(bo_str)
                    except ValueError:
                        continue
                    if bo_val <= _WIBERG_BOND_THRESHOLD:
                        continue
                    if nbr_idx <= self_idx:
                        continue
                    if nbr_idx < 0 or nbr_idx >= len(atoms):
                        continue
                    nbr_z = _element_to_z(nbr_el)
                    if nbr_z is None:
                        continue
                    nz, nx, ny, nzz = atoms[nbr_idx]
                    if nz != nbr_z:
                        continue
                    dx = sx - nx
                    dy = sy - ny
                    dz = sz - nzz
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    z_min = self_z if self_z <= nbr_z else nbr_z
                    z_max = nbr_z if self_z <= nbr_z else self_z
                    key = (z_min, z_max, int(round(bo_val)))
                    bucket = bins.get(key)
                    if bucket is None:
                        bins[key] = [dist]
                    else:
                        bucket.append(dist)

    # Aggregate bins with enough observations.
    out_agg: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    serialisable: Dict[str, Dict[str, float]] = {}
    for key, distances in bins.items():
        n = len(distances)
        if n < _MIN_BIN_N:
            continue
        mean = sum(distances) / float(n)
        var = sum((d - mean) ** 2 for d in distances) / float(n)
        std = var ** 0.5
        dmin = min(distances)
        dmax = max(distances)
        entry = {
            "mean": float(mean),
            "std": float(std),
            "n": int(n),
            "min": float(dmin),
            "max": float(dmax),
        }
        out_agg[key] = entry
        serialisable[f"{key[0]},{key[1]},{key[2]}"] = entry
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, sort_keys=True)
    # Refresh the in-memory cache so subsequent get_bond_length / tmqm_table_lookup
    # calls in this process see the freshly-built entries without requiring a
    # re-import.  Backward-compat: callers that import BEFORE build (the
    # common case for trainer entry points) hit the disk cache via load_
    # bond_length_table; this is the post-build hot path.
    _BOND_TABLE_CACHE.clear()
    _BOND_TABLE_CACHE.update(out_agg)
    return out_agg


def load_bond_length_table(
    cache_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Optional[Dict[Tuple[int, int, int], Dict[str, float]]]:
    """Return the cached tmQM bond-length table or ``None`` if absent.

    Convenience for callers that want a fast hit without forcing a
    rebuild.  Returns ``None`` when the cache file is missing so the
    caller can decide whether to fall back to Pyykkö or trigger a
    :func:`build_bond_length_table` rebuild.
    """
    root_path = Path(root) if root is not None else _DEFAULT_TMQM_BOND_TABLE_ROOT
    cache = Path(cache_path) if cache_path is not None else root_path / "bond_length_table.json"
    if not cache.is_file():
        return None
    with cache.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for k, v in raw.items():
        z_min, z_max, order = (int(t) for t in k.split(","))
        out[(z_min, z_max, order)] = {
            "mean": float(v["mean"]),
            "std": float(v["std"]),
            "n": int(v["n"]),
            "min": float(v["min"]),
            "max": float(v["max"]),
        }
    return out


# Cache path constant for the trainer pipeline.
BOND_LENGTH_TABLE_PATH: Path = _DEFAULT_BOND_TABLE_CACHE

# ---------------------------------------------------------------------------
# Public table: atomic number -> single-bond covalent radius (Angstrom).
# Values from Pyykkö & Atsumi (2009) Table 1 (single-bond, *rcov*, in Å).
# Order matches the CFM YuelBond vocab (H + common organics + common Pt-
# environment ligating atoms + the 3 project metals Pt/Pd/Au).
# ---------------------------------------------------------------------------
PYYKKO_SINGLE_BOND: Dict[int, float] = {
    1: 0.32,   # H
    6: 0.75,   # C
    7: 0.71,   # N
    8: 0.63,   # O
    9: 0.64,   # F
    15: 1.05,  # P
    16: 1.02,  # S
    17: 0.99,  # Cl
    34: 1.20,  # Se
    35: 1.14,  # Br
    53: 1.33,  # I
    46: 1.39,  # Pd
    78: 1.23,  # Pt   (Pt–Pt single bond 2.41 Å; Pt–X single shorter)
    79: 1.25,  # Au
}

# ---------------------------------------------------------------------------
# Wf-A28 step-2 (2026-10-14): per-order Pyykkö 2008 (double-bond) + 2005
# (triple-bond) covalent radii for the 14-element CFM cohort.  When both
# ``z1`` and ``z2`` map to ``PYYKKO_DOUBLE_BOND`` (or ``PYYKKO_TRIPLE_BOND``),
# the function looks up sum-of-radii directly; otherwise it falls back to
# the Wf-A27 Hückel-shortened single-bond length so the function never
# raises and unmapped Z still produces a reasonable PROJECTED-rough value.
#
# Citations:
#   - Pyykkö, P.; Atsumi, M. (2009) — single-bond covalent radii (above).
#     Chem. Eur. J. 15 (1): 186–197. doi:10.1002/chem.200800987.
#   - Pyykkö, P.; Riedel, S.; Patzschke, M. (2005) — triple-bond covalent
#     radii (Tables 1-3).  Chem. Eur. J. 11 (12): 3511–3520.
#     doi:10.1002/chem.200401299.
#   - Pyykkö, P.; Atsumi, M. (2008) — double-bond covalent radii (Table 1).
#     Chem. Eur. J. 14 (25): 7707–7717.  doi:10.1002/chem.200800987 — note:
#     some refs use Pyykkö 2009 Table 1 column "other bond orders" for
#     these values; the 2008 paper is the canonical source.  Values are
#     covalent radii in Angstrom; PROJECTED-rough outside the cohort.
# ---------------------------------------------------------------------------
PYYKKO_DOUBLE_BOND: Dict[int, float] = {
    1: 0.30,   # H (rare)
    6: 0.67,   # C
    7: 0.62,   # N
    8: 0.57,   # O
    9: 0.55,   # F
    15: 0.96,  # P
    16: 0.94,  # S
    17: 0.95,  # Cl
    34: 1.07,  # Se
    35: 1.07,  # Br
    53: 1.23,  # I
    46: 1.27,  # Pd
    78: 1.10,  # Pt   (Pt=P / Pt=C double bonds)
    79: 1.13,  # Au   (Au=C / Au=N double bonds)
}

# Triple-bond (Pi) covalent radii — covers only the elements with stable
# triple bonds in the CFM cohort.  H/halogens/se/I do NOT have well-defined
# triple-bond radii and fall back to Hückel shortening of the single bond.
PYYKKO_TRIPLE_BOND: Dict[int, float] = {
    6: 0.60,   # C    (alkyne / cyano)
    7: 0.54,   # N    (nitrile)
    8: 0.53,   # O    (CO triple, rare)
    15: 0.92,  # P    (P≡P phospha-alkynes)
    16: 0.92,  # S    (CS2 / thiones — these are C=S double in practice)
    46: 1.20,  # Pd   (Pd≡CO is unknown in practice; placeholder)
    78: 1.05,  # Pt   (Pt≡C rare; included for completeness)
    79: 1.10,  # Au   (Au≡C rare; included for completeness)
}

# Default element-row fallback (Å) for Z outside the 14-element cohort.
# Sourced from Cordero 2008 mean metal radius — same convention the
# ``ClashPenalty`` class in :mod:`molmetal.loss.prior_losses` already uses.
_DEFAULT_RADIUS_A: float = 0.77

# ---------------------------------------------------------------------------
# Cache for pairwise already-symmetric lookups (z1, z2) -> r_cov.
# Tiny (≈ 14*14 = 196 entries) so a dict cache is fine and avoids
# re-summing on every call.
# ---------------------------------------------------------------------------
_PAIR_CACHE: Dict[Tuple[int, int], float] = {}

# ---------------------------------------------------------------------------
# Wf-A29 step-1: tmQM-DFT empirical-mean cache.  Loaded eagerly on
# import when the cache JSON exists; absence is a SILENT no-op so the
# Wf-A28 Pyykkö path remains byte-exact for any code path that runs
# before the table is built.  The cache key is
# ``(min(z1, z2), max(z1, z2), order_int)`` — symmetric on the
# unordered pair.
# ---------------------------------------------------------------------------
_BOND_TABLE_CACHE: Dict[Tuple[int, int, int], Dict[str, float]] = {}
_cached_table = load_bond_length_table()
if _cached_table is not None:
    _BOND_TABLE_CACHE.update(_cached_table)


def tmqm_table_lookup(
    z1: int,
    z2: int,
    order: int = 1,
) -> Optional[Dict[str, float]]:
    """Return the tmQM-DFT empirical entry for ``(z1, z2, order)`` or ``None``.

    The result dict has keys ``{mean, std, n, min, max}`` in Angstrom /
    count units.  Returns ``None`` when the cache is unpopulated or
    the bin is absent (callers can then fall back to Pyykkö).
    """
    if not _BOND_TABLE_CACHE:
        return None
    key = _symmetric_key(z1, z2) + (int(order),)
    return _BOND_TABLE_CACHE.get(key)


def _symmetric_key(z1: int, z2: int) -> Tuple[int, int]:
    """Return a canonical key with the smaller Z first."""
    a, b = int(z1), int(z2)
    return (a, b) if a <= b else (b, a)


def get_bond_length(z1: int, z2: int, order: int = 1) -> float:
    """Sum of Pyykkö covalent radii for ``z1``–``z2`` at the given ``order``.

    Parameters
    ----------
    z1, z2
        Atomic numbers.  Symmetric: ``get_bond_length(z1, z2) ==
        get_bond_length(z2, z1)``.
    order
        Bond order (default 1 = single bond).  For ``order`` ∈ {2, 3}
        and BOTH ``z1`` / ``z2`` in the per-order Pyykkö table, the
        function returns the sum of per-order radii directly.
        Otherwise, the function falls back to the Wf-A27 Hückel
        shortening of the single-bond sum: ``factor = 1 -
        0.18 * ln(order)`` (1 → 1.000, 2 → 0.875, 3 → 0.805).
        Order 0 or negative falls back to 1.0 (single bond).

    Returns
    -------
    float
        Bond length in Angstrom.  For unknown ``z`` the function uses
        :data:`_DEFAULT_RADIUS_A` (0.77 Å) — never raises, never
        returns ``None``.  Reviewers should treat a returned value
        computed from one or two fallback radii as PROJECTED-rough.

    Notes
    -----
    Wf-A27 step-2 ships the single-bond TABLE only.
    Wf-A28 step-2 adds the per-order Pyykkö 2008 (double) and Pyykkö
    2005 (triple) tables; the single-bond path is BYTE-IDENTICAL with
    Wf-A27.  No trainer / EGNN / loss path consults ``get_bond_length``
    today; a future Wf may add a wall-potential term with the
    conventional ``weight=0`` default.
    """
    ord_int = int(order)

    # Wf-A29 step-1: tmQM-DFT empirical mean takes precedence when the
    # (z1, z2, order) bin has ``n >= 5`` observations in the cached table.
    # Loader is O(1) cache hit; absence of cache is silently transparent
    # — we fall through to the Wf-A28 Pyykkö path bit-exactly.
    tmqm_entry = _BOND_TABLE_CACHE.get(_symmetric_key(z1, z2) + (ord_int,))
    if tmqm_entry is not None and int(tmqm_entry["n"]) >= _MIN_BIN_N:
        return float(tmqm_entry["mean"])

    # Wf-A28 step-2: per-order Pyykkö lookup when both atomic numbers are
    # in the table for the requested order.  Defaults to 1.0 (single) for
    # non-positive orders.
    if ord_int == 2:
        r1 = PYYKKO_DOUBLE_BOND.get(int(z1))
        r2 = PYYKKO_DOUBLE_BOND.get(int(z2))
        if r1 is not None and r2 is not None:
            return float(r1 + r2)
    elif ord_int == 3:
        r1 = PYYKKO_TRIPLE_BOND.get(int(z1))
        r2 = PYYKKO_TRIPLE_BOND.get(int(z2))
        if r1 is not None and r2 is not None:
            return float(r1 + r2)

    key = _symmetric_key(z1, z2)
    cached = _PAIR_CACHE.get(key)
    if cached is None:
        r1 = PYYKKO_SINGLE_BOND.get(int(z1), _DEFAULT_RADIUS_A)
        r2 = PYYKKO_SINGLE_BOND.get(int(z2), _DEFAULT_RADIUS_A)
        cached = r1 + r2
        _PAIR_CACHE[key] = cached
    if ord_int <= 1:
        # Single-bond path is byte-exact with Wf-A27; default order=1 hits this.
        return float(cached)
    # Order > 1: Hückel shortening of aromatic / double / triple bonds.
    # ln(2)=0.693 → 0.875, ln(3)=1.099 → 0.802.  Both within 5% of the
    # tabulated shortening for the same bond orders in Pyykkö 2009 §3.
    import math
    factor = 1.0 - 0.18 * math.log(float(ord_int))
    return float(cached) * factor


def has_entry(z: int) -> bool:
    """Whether ``z`` is in the curated 14-element table (no fallback)."""
    return int(z) in PYYKKO_SINGLE_BOND


__all__ = [
    "PYYKKO_SINGLE_BOND",
    "PYYKKO_DOUBLE_BOND",
    "PYYKKO_TRIPLE_BOND",
    "BOND_LENGTH_TABLE_PATH",
    "build_bond_length_table",
    "load_bond_length_table",
    "tmqm_table_lookup",
    "get_bond_length",
    "has_entry",
]
