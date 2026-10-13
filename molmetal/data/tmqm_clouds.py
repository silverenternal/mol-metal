"""tmQM -> AtomCloud loader: real metal-bearing 3D geometry for CFM evaluation.

Why this module exists
----------------------
The CFM decoder was validated only on SYNTHETIC clouds -- every one the same
square-planar Pt layout (1 Pt + 2 NH3 + 2 Cl), jittered, decode_ratio 1.000.
The obvious "real" corpus, MetalCytoToxDB, cannot serve this: its structure
column is literally ``SMILES_Ligands``, so the metal is a separate COLUMN and
never appears in the structure (MEASURED: 0 of 800 sampled rows contain any
metal atom), and 100% of its rows are multi-fragment.  Pointed at it the
decoder collapses to decode_ratio 0.000.

tmQM does serve this.  It is a CSD-derived corpus of 108,541 mononuclear
d-block complexes with explicit DFT (TPSSh-D3BJ/def2-SVP) coordinates,
MIT-licensed, and already on disk at ``.data/molmetal/tmQM/``.  It covers all
six of this project's target metals: Pt 8,275 / Ru 9,068 / Ir 4,695 /
Rh 4,278 / Re 4,033 / Os 1,224.

What this module does NOT do
----------------------------
It does not change any vocabulary.  ``_YUELBOND_Z_TABLE`` currently contains
exactly one metal (Pt = 78), so a Ru/Ir/Os/Re/Rh cloud will have its metal
clamped to the padding index by ``_z_to_onehot``.  Filtering to Pt is
therefore the meaningful first experiment; the other metals need the vocab
extension that this loader deliberately leaves to its caller.

Layout notes
------------
``tmQM_X{1,2,3}.xyz.gz`` are gzipped multi-frame XYZ.  Each block is::

    <n_atoms>
    CSD_code = XXXXXX | q = ... | S = ... | Stoichiometry = ... | MND = n | years
    <element> <x> <y> <z>
    ... n_atoms lines ...

The files are ~68 MB gzipped / large uncompressed, so everything here STREAMS
and stops as soon as enough clouds have been collected -- never load a whole
file to filter it.
"""

from __future__ import annotations

import csv
import gzip
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from molmetal.data.tmqm import PocketCloud, load_pocket_from_pdb
from molmetal.models.bond_head import AtomCloud

__all__ = [
    "BO_BOND_THRESHOLD",
    "TMQM_METAL_Z",
    "iter_bo_bonds",
    "iter_bo_continuous",
    "load_tmqm_clouds",
    "load_tmqm_metadata",
    "iter_tmqm_blocks",
]

#: Element symbol -> atomic number, for the metals this project targets.
#: Kept local so the loader does not depend on RDKit being importable.
TMQM_METAL_Z: Dict[str, int] = {
    "Pt": 78,
    "Ru": 44,
    "Ir": 77,
    "Rh": 45,
    "Os": 76,
    "Re": 75,
}

#: Element symbol -> Z for everything else commonly present in these
#: complexes.  Anything not listed is SKIPPED with a counter rather than
#: silently mapped to a wrong element.
_COMMON_Z: Dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Mo": 42, "Tc": 43, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "W": 74,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
}
_SYMBOL_TO_Z: Dict[str, int] = {**_COMMON_Z, **TMQM_METAL_Z}


def load_tmqm_metadata(parsed_csv: str | Path) -> Dict[str, dict]:
    """Return ``{csd_code: row_dict}`` from ``tmqm_parsed.csv``.

    The CSV carries ``metal``, ``coord_number``, ``charge``, ``spin`` and
    ``smiles`` per CSD code, which is what makes filtering cheap -- the
    geometry files are then only read for the codes actually wanted.
    """
    out: Dict[str, dict] = {}
    with open(parsed_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            code = str(row.get("csd_code", "")).strip()
            if code:
                out[code] = row
    return out


def iter_tmqm_blocks(xyz_gz_paths: Sequence[str | Path]) -> Iterator[Tuple[str, List[str], List[Tuple[float, float, float]]]]:
    """Stream ``(csd_code, symbols, coords)`` blocks from gzipped XYZ files.

    Stops reading a file as soon as the caller stops consuming -- callers that
    only want a few hundred structures never decompress the rest.
    """
    for path in xyz_gz_paths:
        p = Path(path)
        if not p.is_file():
            continue
        with gzip.open(str(p), "rt") as fh:
            while True:
                header = fh.readline()
                if not header:
                    # 2026-10-04 FIX — this was `return`, which exits the
                    # WHOLE generator rather than this file's loop.  MEASURED
                    # consequence: only tmQM_X1.xyz.gz was ever read, so
                    # 73,571 of the corpus's 108,541 complexes (67.8%) were
                    # invisible, and the six-metal pool was 6,872 instead of
                    # 33,429.  A caller asking for more than ~6,872 got
                    # 6,872 and no warning.  `break` moves to the next shard.
                    break
                header = header.strip()
                if not header:
                    continue
                try:
                    n_atoms = int(header)
                except ValueError:
                    continue
                comment = fh.readline()
                code = ""
                for part in comment.split("|"):
                    part = part.strip()
                    if part.startswith("CSD_code"):
                        code = part.split("=", 1)[-1].strip()
                        break
                symbols: List[str] = []
                coords: List[Tuple[float, float, float]] = []
                for _ in range(n_atoms):
                    line = fh.readline()
                    if not line:
                        break
                    f = line.split()
                    if len(f) < 4:
                        continue
                    symbols.append(f[0])
                    coords.append((float(f[1]), float(f[2]), float(f[3])))
                if symbols:
                    yield code, symbols, coords


# ---------------------------------------------------------------------------
# Helpers — safe CSV-to-tensor parsing + the cloud-build chokepoint
# ---------------------------------------------------------------------------
def _safe_int(v: object) -> Optional[int]:
    """Parse a CSV-stringified int; ``None`` on any failure.

    Used for the per-cloud scalar fields (charge, spin, CSD_years). Empty
    strings, ``"nan"`` / ``"None"`` / ``"null"`` all return ``None``; the
    corresponding AtomCloud field stays ``None`` rather than becoming a
    bogus 0.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _safe_float(v: object) -> Optional[float]:
    """Parse a CSV-stringified float; ``None`` on any failure."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _scalar_tensor(value: Optional[float], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Wrap a scalar value in a 0-d tensor if not ``None``.

    Per-cloud scalar slots on :class:`AtomCloud` are stored as 0-d tensors
    (so :meth:`AtomCloud.to` can move them with the rest of the cloud's
    tensors).  ``None`` in -> ``None`` out, so a missing metadata field
    does not become a 0-valued tensor.
    """
    if value is None:
        return None
    return torch.tensor(value, dtype=dtype)


class _BoInfo:
    """Per-csd-code Wiberg BO data, gathered once and indexed by code.

    The cloud build site needs three views of the same block:

    * ``pairs_with_class`` — threshold-filtered int class (legacy bond head
      supervision).
    * ``pairs_continuous`` — NO threshold float BO (the parallel supervision
      for the regression target ``bond_orders_continuous``).
    * ``per_atom_nbr_zsets`` — top-K (default 12) neighbour atomic numbers
      per atom, sorted by descending BO.  Zero-padding to ``(N, K)`` happens
      in the cloud build site so this struct stays CSR-friendly.

    All three are produced in one pass over the BO file by
    :func:`iter_bo_bonds` (the first two views also in
    :func:`iter_bo_continuous`).
    """

    __slots__ = ("pairs_with_class", "pairs_continuous", "per_atom_nbr_zsets")

    def __init__(
        self,
        pairs_with_class: List[Tuple[int, int, int]],
        pairs_continuous: List[Tuple[int, int, float]],
        per_atom_nbr_zsets: Dict[int, List[int]],
    ) -> None:
        self.pairs_with_class = pairs_with_class
        self.pairs_continuous = pairs_continuous
        self.per_atom_nbr_zsets = per_atom_nbr_zsets


def _build_cloud_from_kept(
    code: str,
    keep_idx: List[int],
    pos: torch.Tensor,
    zt: torch.Tensor,
    symbols: List[str],
    row: Dict[str, object],
    bo_info: Optional["_BoInfo"],
    natural_charges_map: Optional[Dict[str, List[float]]],
    pocket: Optional[PocketCloud] = None,
) -> Optional[AtomCloud]:
    """Build a fully-populated :class:`AtomCloud` for ONE tmQM complex.

    Single chokepoint used by both the main loop and the
    :func:`min_per_metal` back-fill.  Populates all 19 Wf-A11 supervision
    fields; the legacy 4 fields (``positions``, ``atomic_numbers``,
    ``bonds``, ``bond_types``) are populated the same way as before so
    existing call sites see no behavioural change.

    Returns ``None`` if AtomCloud construction raises (the existing
    ``try/except ValueError`` pattern the main loop uses); the caller
    counts the skip and continues.
    """
    n_atoms = int(pos.shape[0])

    # ---- metal_atom_index: position in the kept atoms where Z == metal Z ----
    metal_sym = str(row.get("metal", "")).strip()
    metal_z = TMQM_METAL_Z.get(metal_sym)
    metal_idx_in_kept: Optional[int] = None
    if metal_z is not None:
        for new_i, z in enumerate(zt.tolist()):
            if z == metal_z:
                metal_idx_in_kept = new_i
                break
    metal_atom_index_t = (
        torch.tensor(metal_idx_in_kept, dtype=torch.long)
        if metal_idx_in_kept is not None
        else torch.tensor(AtomCloud.NO_METAL_INDEX, dtype=torch.long)
    )

    # ---- coord_neighbors_per_atom: zero-padded (N, K_NEIGHBOURS) int64 ----
    nbrs_matrix = torch.zeros(n_atoms, AtomCloud.K_NEIGHBOURS, dtype=torch.long)
    if bo_info is not None:
        for atom_new_idx, old_idx in enumerate(keep_idx):
            z_list = bo_info.per_atom_nbr_zsets.get(old_idx, [])
            for k, z in enumerate(z_list[: AtomCloud.K_NEIGHBOURS]):
                nbrs_matrix[atom_new_idx, k] = int(z)

    # ---- bo_per_atom: sum of per-pair Wiberg BOs (continuous, no threshold) ----
    bo_per_atom_t: Optional[torch.Tensor] = None
    if bo_info is not None and bo_info.pairs_continuous:
        sums = [0.0] * n_atoms
        old_to_new_for_sum = {old: new for new, old in enumerate(keep_idx)}
        for i, j, v in bo_info.pairs_continuous:
            new_i = old_to_new_for_sum.get(i)
            new_j = old_to_new_for_sum.get(j)
            if new_i is not None:
                sums[new_i] += v
            if new_j is not None:
                sums[new_j] += v
        bo_per_atom_t = torch.tensor(sums, dtype=torch.float32)

    # ---- natural_charges: per-atom q from tmQM_X.q (loaded once at top) ----
    natural_charges_t: Optional[torch.Tensor] = None
    if natural_charges_map is not None:
        q_list = natural_charges_map.get(code)
        if q_list is not None:
            mapped: List[float] = []
            ok = True
            for old in keep_idx:
                if old < len(q_list):
                    mapped.append(float(q_list[old]))
                else:
                    ok = False
                    break
            if ok and len(mapped) == n_atoms:
                natural_charges_t = torch.tensor(mapped, dtype=torch.float32)

    # ---- bonds + bond_types + bond_orders_continuous + is_dative_pair ----
    bonds_t = bond_types_t = None
    bond_orders_continuous_t: Optional[torch.Tensor] = None
    is_dative_pair_t: Optional[torch.Tensor] = None
    if bo_info is not None and bo_info.pairs_with_class:
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        pairs = [
            (old_to_new[i], old_to_new[j], t)
            for i, j, t in bo_info.pairs_with_class
            if i in old_to_new and j in old_to_new
        ]
        if pairs:
            bonds_t = torch.tensor(
                [[p[0] for p in pairs], [p[1] for p in pairs]], dtype=torch.long
            )
            bond_types_t = torch.tensor([p[2] for p in pairs], dtype=torch.long)
            # Continuous BO for the same bonded pairs (lookup; bond file is
            # the source of truth for the float, even if the int-class
            # bond is below the threshold — but since `pairs_with_class`
            # is itself threshold-filtered, the float will exist for them).
            cont_lookup: Dict[Tuple[int, int], float] = {}
            for i, j, v in bo_info.pairs_continuous:
                a, b = (i, j) if i < j else (j, i)
                # MAX of symmetric entries (matches _bo_continuous_pairs)
                if (a, b) not in cont_lookup or v > cont_lookup[(a, b)]:
                    cont_lookup[(a, b)] = v
            cont_vals: List[float] = []
            is_dative_list: List[bool] = []
            for i, j, _t in pairs:
                a, b = (i, j) if i < j else (j, i)
                v = cont_lookup.get((a, b), 0.0)
                cont_vals.append(v)
                # dative iff at least one endpoint is the metal AND the
                # BO is in the dative range (single bond, < 1.0).
                # Catches Pt-Cl (~0.45) / Pt-N (~0.45) / Ru-N (~0.40);
                # excludes metal-metal bonds (BO > 1.0) and organic
                # C-C / C-H (no metal endpoint).
                if (
                    metal_z is not None
                    and (zt[i].item() == metal_z or zt[j].item() == metal_z)
                    and v < 1.0
                ):
                    is_dative_list.append(True)
                else:
                    is_dative_list.append(False)
            if cont_vals:
                bond_orders_continuous_t = torch.tensor(
                    cont_vals, dtype=torch.float32
                )
                is_dative_pair_t = torch.tensor(
                    is_dative_list, dtype=torch.bool
                )

    # ---- per-cloud scalar fields from metadata row ----
    oxidation_state_t = _scalar_tensor(_safe_int(row.get("charge")), torch.long)
    spin_state_t = _scalar_tensor(_safe_int(row.get("spin")), torch.long)
    csd_publication_year_t = _scalar_tensor(
        _safe_int(row.get("CSD_years")), torch.long
    )
    dft_electronic_e_t = _scalar_tensor(
        _safe_float(row.get("Electronic_E")), torch.float32
    )
    dft_dispersion_e_t = _scalar_tensor(
        _safe_float(row.get("Dispersion_E")), torch.float32
    )
    dipole_m_t = _scalar_tensor(_safe_float(row.get("Dipole_M")), torch.float32)
    metal_q_t = _scalar_tensor(_safe_float(row.get("Metal_q")), torch.float32)
    hl_gap_t = _scalar_tensor(_safe_float(row.get("HL_Gap")), torch.float32)
    homo_e_t = _scalar_tensor(_safe_float(row.get("HOMO_Energy")), torch.float32)
    lumo_e_t = _scalar_tensor(_safe_float(row.get("LUMO_Energy")), torch.float32)
    polarizability_t = _scalar_tensor(
        _safe_float(row.get("Polarizability")), torch.float32
    )
    stoich = str(row.get("stoichiometry", "")).strip()
    smi = str(row.get("smiles", "")).strip()

    # ---- Wf-A27 (2026-10-13) step-1 coord_geom_class — metal site ----
    # Rule-based classifier (no xtb GPU path: Wf-A28 deferred).  Looks up
    # the dominant metal site (Pt/Ir/Pd/Au/etc.) by its coordination
    # number (number of bonded neighbours).  When the metal is not in
    # :data:`TMQM_METAL_Z`, the CN is outside [2, 12], or the (Z, CN)
    # pair is unmapped, the loader leaves the slot at ``None`` —
    # preserving the byte-identical 63-site constructor contract.
    #
    # Lazy import to keep the cold-start cost of every other loader
    # call unchanged; the loader runs once at the start of training.
    coord_geom_class: Optional[str] = None
    if (
        metal_z is not None
        and metal_idx_in_kept is not None
        and bo_info is not None
    ):
        metal_cn = len(bo_info.per_atom_nbr_zsets.get(metal_idx_in_kept, []))
        if metal_cn >= 2 and metal_cn <= 12:
            from molmetal.data.coord_geometry import classify_coord_geometry
            coord_geom_class = classify_coord_geometry(metal_cn, int(metal_z))

    try:
        return AtomCloud(
            positions=pos,
            atomic_numbers=zt,
            bonds=bonds_t,
            bond_types=bond_types_t,
            natural_charges=natural_charges_t,
            bo_per_atom=bo_per_atom_t,
            coord_neighbors_per_atom=nbrs_matrix,
            bond_orders_continuous=bond_orders_continuous_t,
            is_dative_pair=is_dative_pair_t,
            metal_atom_index=metal_atom_index_t,
            oxidation_state=oxidation_state_t,
            spin_state=spin_state_t,
            csd_publication_year=csd_publication_year_t,
            dft_electronic_e=dft_electronic_e_t,
            dft_dispersion_e=dft_dispersion_e_t,
            dipole_m=dipole_m_t,
            metal_q=metal_q_t,
            hl_gap=hl_gap_t,
            homo_e=homo_e_t,
            lumo_e=lumo_e_t,
            polarizability=polarizability_t,
            stoichiometry=stoich or None,
            smiles=smi or None,
            # Wf-A16 (2026-10-13) c-flow pocket attachment — passed
            # through by load_tmqm_clouds when ``pocket_path`` is set.
            # None preserves the 63 legacy construction sites.
            pocket=pocket,
            # Wf-A27 (2026-10-13) step-1 — coord_geom_class slot.  Loader-
            # attached; ``None`` when unmapped ⇒ no change for the 63
            # other construction sites that go through ``AtomCloud(...)``.
            coord_geom_class=coord_geom_class,
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Ground-truth bond graph — tmQM's Wiberg bond-order blocks
# ---------------------------------------------------------------------------
#: Bond-order blocks ship beside the geometries and use the SAME atom indexing,
#: which is why they, and not the SMILES column, are the target source.  The
#: SMILES in ``tmQM_y.csv`` is written in its own atom order, so using it would
#: require a graph-matching step whose failures are indistinguishable from
#: modelling failures.
_BO_FILES = ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")

#: Wiberg bond order at or above which a pair counts as bonded.  tmQM lists
#: every neighbour pair including sub-0.15 contacts between ligand hydrogens;
#: 0.3 is the conventional single/dative separation used in the BO-based
#: bond-perception literature (and the same threshold
#: ``scripts/bench_tmqm_bond_perception.py`` scores against).
BO_BOND_THRESHOLD: float = 0.3

#: Bond-order magnitude -> ``BOND_LABELS`` class index.
#:
#: Aromaticity is NOT recoverable from a Wiberg order — delocalised C-C sits
#: near 1.4-1.5, which overlaps dative metal bonds.  The aromatic band is
#: therefore a deliberate, documented heuristic rather than a reading, and the
#: band is narrow so that a wrong call is a near-miss rather than a wild one.
_BO_TYPE_BANDS = (
    (1.25, 1),  # BOND_SINGLE  — includes dative, which is ~0.3-0.9
    (1.75, 4),  # BOND_AROMATIC — ~1.4-1.5
    (2.50, 2),  # BOND_DOUBLE
    (float("inf"), 3),  # BOND_TRIPLE
)


def bo_to_bond_class(bo: float) -> int:
    """Map a Wiberg bond order to a ``BOND_LABELS`` class index."""
    for upper, cls in _BO_TYPE_BANDS:
        if float(bo) < upper:
            return cls
    return 3


def iter_bo_bonds(
    bo_gz_paths: Sequence[str | Path],
    *,
    threshold: float = BO_BOND_THRESHOLD,
) -> Iterator[Tuple[str, List[Tuple[int, int, int]]]]:
    """Yield ``(csd_code, [(i, j, bond_class), ...])`` from the BO shards.

    Indices are **0-based** here (the file is 1-based) and each pair appears
    once.  Blocks whose atom count does not match their geometry are the
    caller's problem — this function does not see the geometry.
    """
    for path in bo_gz_paths:
        p = Path(path)
        if not p.is_file():
            continue
        code: Optional[str] = None
        atoms: List[Tuple[int, str, List[Tuple[int, float]]]] = []
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, _bo_pairs(atoms, threshold)
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    if code is not None and atoms:
                        yield code, _bo_pairs(atoms, threshold)
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                rest = parts[3:]
                nbrs: List[Tuple[int, float]] = []
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, parts[1], nbrs))
        if code is not None and atoms:
            yield code, _bo_pairs(atoms, threshold)


def _bo_pairs(
    atoms: Sequence[Tuple[int, str, List[Tuple[int, float]]]],
    threshold: float,
) -> List[Tuple[int, int, int]]:
    """Deduplicate the symmetric neighbour lists into one pair list."""
    best: Dict[Tuple[int, int], float] = {}
    for idx, _sym, nbrs in atoms:
        for nbr_idx, bo in nbrs:
            if bo < threshold or nbr_idx == idx:
                continue
            key = (min(idx, nbr_idx), max(idx, nbr_idx))
            if key not in best or bo > best[key]:
                best[key] = bo
    # 1-based in the file -> 0-based here.
    return [(i - 1, j - 1, bo_to_bond_class(v)) for (i, j), v in sorted(best.items())]


def _bo_continuous_pairs(
    atoms: Sequence[Tuple[int, str, List[Tuple[int, float]]]],
) -> List[Tuple[int, int, float]]:
    """Deduplicate the symmetric neighbour lists into one float-BO pair list.

    NO threshold (the parallel companion to :func:`_bo_pairs` for the
    continuous-BO supervision signal).  Every pair the file lists — including
    sub-0.15 H-H contacts — is included.  The float is the MAX of the two
    symmetric file entries (the file can disagree at the 4th decimal), and
    pairs are returned sorted so downstream tensors stay CSR-friendly.

    Indices are 1-based in the file, 0-based here (mirrors :func:`_bo_pairs`).
    """
    best: Dict[Tuple[int, int], float] = {}
    for idx, _sym, nbrs in atoms:
        for nbr_idx, bo in nbrs:
            if nbr_idx == idx:
                continue
            key = (min(idx, nbr_idx), max(idx, nbr_idx))
            if key not in best or bo > best[key]:
                best[key] = bo
    return [(i - 1, j - 1, v) for (i, j), v in sorted(best.items())]


def _per_atom_nbr_zsets(
    atoms: Sequence[Tuple[int, str, List[Tuple[int, float]]]],
    k: int = 12,
) -> Dict[int, List[int]]:
    """Build per-atom neighbour Z-sets (no zero-pad; cloud build pads).

    For each atom in ``atoms`` (the 1-based Wiberg BO block), sort the
    neighbour list by BO descending and take the top ``k`` atomic numbers
    via :data:`_SYMBOL_TO_Z` (falling back to 0 for any symbol the loader
    does not recognise — that is the SAME fallback the geometry stream
    uses, so an unknown neighbour symbol produces a 0 in the slot rather
    than crashing the cloud build).

    The result is a mapping ``0-based atom_idx -> list[Z]`` of length up to
    ``k`` (often shorter for the few low-coordination atoms in tmQM).
    """
    # Build an idx -> symbol lookup once so the neighbour-element lookup
    # below is O(1) per neighbour rather than O(N).
    idx_to_sym: Dict[int, str] = {idx: sym for idx, sym, _ in atoms}
    out: Dict[int, List[int]] = {}
    for idx, _sym, nbrs in atoms:
        sorted_nbrs = sorted(nbrs, key=lambda x: -x[1])
        z_list: List[int] = []
        for nbr_idx, _bo in sorted_nbrs[:k]:
            nbr_sym = idx_to_sym.get(nbr_idx)
            z_list.append(_SYMBOL_TO_Z.get(nbr_sym, 0) if nbr_sym else 0)
        out[idx - 1] = z_list  # 1-based in the file -> 0-based here
    return out


def iter_bo_bonds(
    bo_gz_paths: Sequence[str | Path],
    *,
    threshold: float = BO_BOND_THRESHOLD,
) -> Iterator[Tuple[str, List[Tuple[int, int, int]], Dict[int, List[int]]]]:
    """Yield ``(csd_code, pairs, per_atom_nbr_zsets)`` from the BO shards.

    Indices are **0-based** here (the file is 1-based) and each pair appears
    once. ``pairs`` is the threshold-filtered int-class list (legacy format,
    consumed by every existing bond-head site).  ``per_atom_nbr_zsets``
    maps each 0-based ``atom_idx`` to a list of up to 12 neighbour
    atomic numbers sorted by descending BO (no zero-pad; the cloud build
    site zero-pads to ``(N, 12)`` to match :data:`AtomCloud.K_NEIGHBOURS`).

    The 12-cap is the same constant the dataclass uses, so the two stay
    paired.  Raising it requires a paired change in both places.

    Blocks whose atom count does not match their geometry are the caller's
    problem — this function does not see the geometry.
    """
    for path in bo_gz_paths:
        p = Path(path)
        if not p.is_file():
            continue
        code: Optional[str] = None
        atoms: List[Tuple[int, str, List[Tuple[int, float]]]] = []
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, _bo_pairs(atoms, threshold), _per_atom_nbr_zsets(atoms)
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    if code is not None and atoms:
                        yield code, _bo_pairs(atoms, threshold), _per_atom_nbr_zsets(atoms)
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                rest = parts[3:]
                nbrs: List[Tuple[int, float]] = []
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, parts[1], nbrs))
        if code is not None and atoms:
            yield code, _bo_pairs(atoms, threshold), _per_atom_nbr_zsets(atoms)


def iter_bo_continuous(
    bo_gz_paths: Sequence[str | Path],
) -> Iterator[Tuple[str, List[Tuple[int, int, float]]]]:
    """Yield ``(csd_code, [(i, j, bo_float), ...])`` from the BO shards, NO threshold.

    The parallel companion to :func:`iter_bo_bonds` for the float-Wiberg-BO
    supervision signal (:data:`AtomCloud.bond_orders_continuous`).  Every
    pair the file lists is included — including sub-0.15 H-H contacts —
    because the trainer treats the float as a regression target, not a
    classifier label, so a "no bond" example in the int-class stream
    should still carry a small float-BO supervision rather than drop out
    entirely.

    Indices are 0-based here, 1-based in the file.  The float per pair is
    the MAX of the two symmetric file entries (mirrors :func:`_bo_pairs`).
    """
    for path in bo_gz_paths:
        p = Path(path)
        if not p.is_file():
            continue
        code: Optional[str] = None
        atoms: List[Tuple[int, str, List[Tuple[int, float]]]] = []
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, _bo_continuous_pairs(atoms)
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    if code is not None and atoms:
                        yield code, _bo_continuous_pairs(atoms)
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                rest = parts[3:]
                nbrs: List[Tuple[int, float]] = []
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, parts[1], nbrs))
        if code is not None and atoms:
            yield code, _bo_continuous_pairs(atoms)


def load_tmqm_clouds(
    tmqm_dir: str | Path,
    metals: Optional[Iterable[str]] = None,
    coord_number: Optional[int] = None,
    n_clouds: Optional[int] = None,
    seed: int = 0,
    include_hydrogens: bool = False,
    split: Optional[str] = None,
    val_frac: float = 0.2,
    min_per_metal: int = 0,
    # ---- Wf-A16 (2026-10-13) c-flow pocket threading (backward-compat) ----
    pocket_path: Optional[str] = None,
    # ---- Wf-A22 (2026-10-13) step-1: per-cloud pocket draw from a 100K
    #      CrossDocked pool.  When pocket_pool_per_cloud=False (default),
    #      behaviour is bit-exact identical to the legacy pocket_path path
    #      (no second load, no per-cloud redraw).
    # ------------------------------------------------------------------------
    pocket_pool_per_cloud: bool = False,
    pocket_pool_path: Optional[str] = None,
    pocket_pool_size: int = 0,
    pocket_draw_seed: int = 0,
    # ---- Wf-A14 (2026-10-13) Bemis-Murcko scaffold split (backward-compat) ----
    split_kind: str = "random",
) -> List["object"]:
    """Load real tmQM geometries as :class:`AtomCloud` objects.

    Parameters
    ----------
    tmqm_dir
        Directory holding ``tmqm_parsed.csv`` and ``tmQM_X{1,2,3}.xyz.gz``.
    metals
        Element symbols to keep (e.g. ``["Pt"]``).  None = all six targets.
    coord_number
        Keep only complexes whose ``coord_number`` equals this (e.g. 4 for
        square-planar Pt(II)).
    n_clouds
        Cap on returned clouds; the files are streamed and reading stops early.
    seed
        Seeds the reservoir sample used when more candidates than ``n_clouds``
        are available, so the selection is reproducible AND spread across the
        whole corpus rather than being the first N of file 1.
    include_hydrogens
        tmQM's XYZ includes H.  Default False matches the synthetic clouds,
        which are heavy-atom only.
    split
        ``"train"`` / ``"val"`` / ``"test"`` for a DISJOINT partition, or
        None for no split.  ``"test"`` is only valid with
        ``split_kind="scaffold"`` (the random hash path is binary 80/20).

        This exists because the obvious approach -- calling this twice with
        different seeds -- LEAKS.  MEASURED: with n_clouds=96 and 32 (so the
        candidate caps are 480 and 160) the val candidate pool was a strict
        SUBSET of the train one, 160/160, because both stream from the same
        head of the same files.  Any decode_ratio measured that way is
        contaminated by train/val overlap.

        The partition here is by a stable hash of the CSD code (default
        ``split_kind="random"``) OR by Bemis-Murcko scaffold
        (``split_kind="scaffold"``), so the sides are disjoint by
        construction and the split does not depend on how many clouds are
        requested on either side.
    split_kind
        ``"random"`` (default, legacy 80/20 hash split — backward
        compatible) or ``"scaffold"`` (Bemis-Murcko 80/10/10).

        The scaffold branch uses
        :func:`molmetal.data.splits._bemis_murcko_key` on each
        :class:`CSD <https://www.ccdc.cam.ac.uk/>`_ code's SMILES from
        ``tmqm_parsed.csv`` (90.2% filled per 2026-10-13 audit; 9.8%
        empty fall back to the CSD 6-letter refcode as the scaffold
        proxy, cited as an honest-negative in
        :file:`docs/HONEST_NEGATIVE_CATALOG.md`).

        Scaffolds are then partitioned by
        :class:`molmetal.data.splits.ScaffoldSplitter` with
        ``strategy="largest_first"`` + ``fractions=(0.8, 0.1, 0.1)`` +
        ``seed=seed``.  The same scaffold family therefore never crosses
        train/val/test — exactly the OOD-by-construction property DrugOOD
        uses for its OOD axes (Ji 2023, iMoLD 2024).
    val_frac
        Fraction assigned to ``"val"`` when ``split_kind="random"``.
        Ignored for ``split_kind="scaffold"`` (the scaffold branch uses
        the 80/10/10 fractions directly from ``ScaffoldSplitter``).
    min_per_metal
        If > 0, GUARANTEE at least this many clouds for each requested metal.

        Why this exists: 2026-10-12 zero-Pt finding. With ``n_clouds=192`` and
        seed=0, the val split on the six-metal pool returned
        ``Ru 64 / Ir 45 / Re 38 / Rh 35 / Os 10 / Pt 0`` -- a reservoir
        sample of a 192-of-960 window in which no Pt complex was selected,
        while Pt is 24.8% of the training pool (6634 / 26715). The training
        fix is therefore NOT a retrain (the checkpoint has seen Pt
        extensively); the val-window fix is to force representation. Default 0
        preserves the original reservoir behaviour exactly.

        When > 0, the loader IGNORES the streaming cap (``_CAP = max(n_clouds
        * 5, n_clouds)``) and scans the entire filtered pool, because short
        pools can hide underrepresented metals beyond the cap. Replacing slots
        uses the FIRST non-reservoir candidate for each metal (file iteration
        order), so the selection remains deterministic given the inputs.

    Returns
    -------
    list[AtomCloud]
        Heavy-atom clouds with the metal PRESENT in ``atomic_numbers``.

    Notes
    -----
    A symbol outside :data:`_SYMBOL_TO_Z` causes that complex to be skipped and
    counted -- never silently dropped to a wrong element.  The skip counts are
    printed, because a silent shortfall here would look like a smaller corpus
    rather than a parse failure.
    """
    # `AtomCloud` is imported at module level (see top of file) so the
    # `K_NEIGHBOURS` / `NO_METAL_INDEX` class constants are available without
    # the cost of re-resolving the symbol on every call.

    root = Path(tmqm_dir)
    parsed = root / "tmqm_parsed.csv"
    if not parsed.is_file():
        raise FileNotFoundError(f"tmqm_parsed.csv not found under {root}")

    # Wf-A30-3 (2026-10-14) C5 FIX — log the actually-resolved root so a
    # reviewer/agent can tell at a glance which shard was loaded (the
    # repo-local fallback under .data/molmetal/tmQM vs the shared-storage
    # mount /mnt/storage/data/molmetal/tmQM).
    print(
        f"[tmqm-clouds] root={root}",
        flush=True,
    )

    # Wf-A16 (2026-10-13) c-flow pocket threading.
    #
    # tmQM is a small-molecule corpus with no protein context, so the
    # pocket comes from a SEPARATE .pdb side-car (typically a CrossDocked /
    # PDBBind receptor).  The pocket is shared across the cohort, so we
    # load it ONCE here and attach it to every cloud we build.  When
    # ``pocket_path`` is None (default) we skip the load entirely and
    # AtomCloud.pocket stays None on every cloud — bit-exact backward
    # compat with Wf-A15.
    pocket: Optional[PocketCloud] = None
    if pocket_path is not None:
        pocket = load_pocket_from_pdb(pocket_path)
        print(
            f"[tmqm-clouds] pocket loaded: {pocket.pocket_id} "
            f"P={int(pocket.coords.shape[0])} from {pocket_path}",
            flush=True,
        )

    # Wf-A22 (2026-10-13) step-1: per-cloud pocket pool draw.
    #
    # When ``pocket_pool_per_cloud=True``, each cloud gets ONE pocket
    # drawn from a reservoir-sampled prefix of ``list_pocket_pool(split="train")``
    # (default size 0 = full pool; 100K is the CrossDocked-2020 train
    # manifest).  The draw is deterministic via ``pocket_draw_seed`` and
    # keyed on the CSD code so the same code always maps to the same
    # pocket.  When the flag is OFF (legacy default), the cohort-wide
    # ``pocket`` (above) is used and behaviour is bit-exact identical to
    # pre-Wf-A22.
    pocket_pool: Optional[List["object"]] = None
    if pocket_pool_per_cloud:
        from molmetal.data.crossdocked_pool import list_pocket_pool
        _pool_kwargs: Dict[str, object] = {"split": "train"}
        if int(pocket_pool_size) > 0:
            _pool_kwargs["max_n"] = int(pocket_pool_size)
        pocket_pool = list_pocket_pool(**_pool_kwargs)
        print(
            f"[tmqm-clouds] pocket_pool loaded: {len(pocket_pool)} entries "
            f"(size={pocket_pool_size if int(pocket_pool_size) > 0 else 'all'}); "
            f"draw_seed={pocket_draw_seed}",
            flush=True,
        )

    wanted_metals = {m.strip() for m in metals} if metals else set(TMQM_METAL_Z)
    unknown = wanted_metals - set(TMQM_METAL_Z)
    if unknown:
        raise ValueError(
            f"unsupported metal symbol(s): {sorted(unknown)}; "
            f"known: {sorted(TMQM_METAL_Z)}"
        )

    meta = load_tmqm_metadata(parsed)
    keep_codes = {
        code
        for code, row in meta.items()
        if str(row.get("metal", "")).strip() in wanted_metals
        and (
            coord_number is None
            or str(row.get("coord_number", "")).strip() == str(int(coord_number))
        )
    }

    # Deterministic, disjoint train/val/test partition.  Two modes:
    #
    # 1. ``split_kind="random"`` (default, Wf-A8c legacy 80/20 binary):
    #    partition by a stable hash of the CSD code.  Independent of
    #    n_clouds so requesting 96 train + 32 val cannot make one a
    #    subset of the other (the leak described in the docstring).
    #
    # 2. ``split_kind="scaffold"`` (Wf-A14, 2026-10-13, Bemis-Murcko
    #    80/10/10): partition by Bemis-Murcko scaffold computed from each
    #    CSD code's SMILES (or its 6-letter refcode as proxy when the
    #    smiles column is empty).  Same scaffold family never crosses
    #    train/val/test — the OOD-by-construction property DrugOOD uses.
    if split is not None:
        key = str(split).strip().lower()
        if split_kind == "scaffold":
            if key not in ("train", "val", "test"):
                raise ValueError(
                    f"split must be 'train'/'val'/'test' when "
                    f"split_kind='scaffold', got {split!r}"
                )
            # Bemis-Murcko key per CSD code (with CSD-prefix proxy for
            # the ~9.8% of rows whose SMILES is empty in tmqm_parsed.csv).
            #
            # The Murcko decomposition is expensive on 100k+ rows, so we
            # cache the keys to ``<tmqm_dir>/scaffold_cache.json`` and
            # only compute the entries that are missing.  The cache is
            # keyed on (CSD code, SMILES) so an updated SMILES column in
            # tmqm_parsed.csv (re-prep) auto-invalidates the affected
            # entries without recomputing the whole table.
            from molmetal.data.splits import _bemis_murcko_key
            import json as _json

            cache_path = Path(root) / "scaffold_cache.json"
            cache: Dict[str, str] = {}
            if cache_path.is_file():
                try:
                    with cache_path.open("r", encoding="utf-8") as _cf:
                        cache = _json.load(_cf)
                except Exception:
                    cache = {}

            csd_to_smi: Dict[str, str] = {
                code: str(meta.get(code, {}).get("smiles", "")).strip()
                for code in keep_codes
            }
            csd_to_scaffold: Dict[str, str] = {}
            n_computed = 0
            for code, smi in csd_to_smi.items():
                cache_key = f"{code}|{smi}"
                if cache_key in cache:
                    csd_to_scaffold[code] = cache[cache_key]
                    continue
                if smi:
                    key_bm = _bemis_murcko_key(smi)
                else:
                    # CSD-prefix proxy (6-letter refcode; e.g. WELROW).
                    # Cited as honest-negative in
                    # docs/HONEST_NEGATIVE_CATALOG.md.
                    key_bm = f"__csd_prefix__{code[:6]}"
                key_bm = key_bm or f"__empty__{code}"
                csd_to_scaffold[code] = key_bm
                cache[cache_key] = key_bm
                n_computed += 1
            # Persist cache (atomic write via os.replace).
            if n_computed > 0:
                try:
                    tmp_path = cache_path.with_suffix(".json.tmp")
                    with tmp_path.open("w", encoding="utf-8") as _cf:
                        _json.dump(cache, _cf, ensure_ascii=False)
                    import os as _os
                    _os.replace(tmp_path, cache_path)
                except Exception as _exc:
                    LOG.warning(
                        "[tmqm-clouds] could not write scaffold_cache.json: %s",
                        _exc,
                    )
            # Group codes by scaffold.
            from collections import defaultdict as _dd
            scaffold_to_codes: Dict[str, List[str]] = _dd(list)
            for code, scaf in csd_to_scaffold.items():
                scaffold_to_codes[scaf].append(code)
            # _assign_groups expects int group members (dataset row
            # indices).  Assign each unique CSD code a stable int via
            # enumerate over a sorted list, then run the assignment.
            all_codes = sorted({c for codes in scaffold_to_codes.values() for c in codes})
            code_to_idx = {c: i for i, c in enumerate(all_codes)}
            groups: List[List[int]] = [
                [code_to_idx[c] for c in scaffold_to_codes[scaf]]
                for scaf in sorted(scaffold_to_codes.keys())
            ]
            from molmetal.data.splits import ScaffoldSplitter, DEFAULT_FRACTIONS
            splitter = ScaffoldSplitter(
                strategy="largest_first",
                fractions=DEFAULT_FRACTIONS,
                seed=seed,
            )
            idx_assignment: Dict[int, str] = splitter._assign_groups(groups)
            idx_to_code = {i: c for c, i in code_to_idx.items()}
            keep_codes = {
                idx_to_code[i] for i, s in idx_assignment.items() if s == key
            }
            if not keep_codes:
                raise RuntimeError(
                    f"split_kind='scaffold' split={key!r} left 0 tmQM "
                    f"entries; check metals={sorted(wanted_metals)} "
                    f"coord_number={coord_number}"
                )
        else:
            # Legacy random hash split (binary 80/20).
            if key not in ("train", "val"):
                raise ValueError(
                    f"split must be 'train' or 'val' for "
                    f"split_kind='random', got {split!r}"
                )
            import hashlib as _hashlib

            def _is_val(code: str) -> bool:
                h = int.from_bytes(
                    _hashlib.sha256(code.encode("utf-8")).digest()[:8], "big"
                )
                return (h % 10_000) < int(round(float(val_frac) * 10_000))

            keep_codes = {
                c for c in keep_codes if _is_val(c) == (key == "val")
            }
            if not keep_codes:
                raise RuntimeError(
                    f"split={key!r} left 0 tmQM entries; check val_frac="
                    f"{val_frac} against the corpus size for metals="
                    f"{sorted(wanted_metals)}"
                )
    if not keep_codes:
        raise RuntimeError(
            f"no tmQM entries match metals={sorted(wanted_metals)} "
            f"coord_number={coord_number}"
        )

    xyz = sorted(root.glob("tmQM_X*.xyz.gz"))
    if not xyz:
        raise FileNotFoundError(f"no tmQM_X*.xyz.gz under {root}")

    # Ground-truth bond graph, indexed once and filtered to the codes in play.
    #
    # Built BEFORE the geometry stream because the two shard families are read
    # independently: the geometry loop skips codes it does not want (metal and
    # coord-number filters), so zipping the two iterators would drift out of
    # step at the first skip.  Restricting to `keep_codes` keeps this bounded
    # regardless of the corpus size.  A missing BO shard is not fatal — the
    # clouds simply carry no bond truth and the trainer falls back to its
    # geometric rule — but it is reported, because silently training with no
    # targets is exactly the failure this indexing exists to prevent.
    #
    # Wf-A11 (2026-10-13): the indexing now stores three views per code in
    # `_BoInfo` (threshold-filtered int class + continuous float BO + per-atom
    # neighbour Z-sets).  The two legacy call paths keep reading
    # `bo_index[code]` as the int-class list, so existing trainer sites see
    # no behaviour change; the new fields (bond_orders_continuous,
    # coord_neighbors_per_atom, bo_per_atom, is_dative_pair) are populated
    # in the cloud build site from the parallel indexes below.
    bo_index: Optional[Dict[str, List[Tuple[int, int, int]]]] = None
    bo_full_index: Dict[str, _BoInfo] = {}
    bo_files = [root / name for name in _BO_FILES]
    if any(p.is_file() for p in bo_files):
        bo_index = {}
        for _code, _pairs, _nbrs in iter_bo_bonds(bo_files):
            if _code in keep_codes:
                bo_index[_code] = _pairs
                # placeholder; pairs_continuous filled in the next pass
                bo_full_index[_code] = _BoInfo(
                    pairs_with_class=_pairs,
                    pairs_continuous=[],
                    per_atom_nbr_zsets=_nbrs,
                )
        # Separate pass for the continuous-BO view (no threshold).
        for _code, _cont in iter_bo_continuous(bo_files):
            if _code in bo_full_index:
                bo_full_index[_code].pairs_continuous = _cont
        print(
            f"[tmqm-clouds] bond truth: {len(bo_index)}/{len(keep_codes)} "
            f"requested complexes have a Wiberg BO block",
            flush=True,
        )
    else:
        print(
            "[tmqm-clouds][WARNING] no tmQM_X*.BO.gz found — clouds carry no "
            "bond truth, so downstream bond targets fall back to the geometric "
            "rule.  That rule produced 100% BOND_NO_BOND on non-Pt metals.",
            flush=True,
        )

    # Natural atomic charges (NBO summary file).  Optional source: a missing
    # .q file is NOT fatal — the per-atom field simply stays None.  The
    # parser is the new `parse_natural_charges` added in Wf-A11 (D1 root
    # cause: previously, natural charges lived in a separate file the loader
    # never opened, so the field was dropped between the metadata CSV and
    # the cloud object).
    natural_charges_map: Optional[Dict[str, List[float]]] = None
    try:
        from molmetal.data.tmqm import parse_natural_charges

        q_files = [root / "tmQM_X.q", root / "tmQM_X.q.gz"]
        for q_candidate in q_files:
            if q_candidate.is_file():
                natural_charges_map = parse_natural_charges(q_path=q_candidate)
                break
        if natural_charges_map is not None:
            in_keep = sum(1 for c in natural_charges_map if c in keep_codes)
            print(
                f"[tmqm-clouds] natural charges: {in_keep}/{len(keep_codes)} "
                f"requested complexes have a tmQM_X.q entry",
                flush=True,
            )
    except FileNotFoundError:
        # Expected when the .q file is not shipped (e.g. trimmed tmQM
        # release for benchmark runs); the field simply stays None.
        natural_charges_map = None

    # Reservoir sample: keeps the selection unbiased across files while still
    # stopping early once the pool is comfortably larger than the request.
    #
    # 2026-10-12 — the ``_CAP`` that lets the loop terminate early is
    # INCOMPATIBLE with ``min_per_metal``: a metal absent from the cap-window
    # cannot be back-filled, so requesting minimum representation requires
    # scanning the entire filtered pool. Without that, the only option is to
    # exit early and report ZERO for that metal — which is the very defect
    # this knob exists to fix.
    _cap_active = n_clouds is not None and int(min_per_metal) <= 0
    rng = random.Random(int(seed))
    reservoir: List["object"] = []
    n_seen = 0
    n_skip_symbol = 0
    n_skip_no_metal = 0
    _CAP = None if not _cap_active else max(int(n_clouds) * 5, int(n_clouds))

    for code, symbols, coords in iter_tmqm_blocks(xyz):
        if code not in keep_codes:
            continue
        zs: List[int] = []
        bad = False
        for s in symbols:
            if not include_hydrogens and s == "H":
                continue
            z = _SYMBOL_TO_Z.get(s)
            if z is None:
                bad = True
                break
            zs.append(z)
        if bad:
            n_skip_symbol += 1
            continue
        if not any(z in set(TMQM_METAL_Z.values()) for z in zs):
            # The metadata says this code has a target metal but the geometry
            # block does not -- a corpus inconsistency worth counting, not
            # silently training on.
            n_skip_no_metal += 1
            continue
        pos = torch.tensor(coords, dtype=torch.float32)
        # EDM-style zero-CoM translational gauge (Hoogeboom 2022,
        # arXiv:2203.17003 §3). CSD XYZ files sit at an arbitrary Cartesian
        # frame: per-complex |centroid| ~11 A, raw std 5.14 — 67.8% of the
        # variance is uncorrelated translation. Subtracting the centroid
        # removes that degrees of freedom so the model's capacity goes into
        # internal geometry (= chemistry) instead. Velocity field needs no
        # change — x0 ~ N(0,I) is already centered so x_t = (1-t)*x0 + t*x1
        # inherits centering. Do NOT scale x here: that would require an
        # EDM-style log-likelihood correction (Hoogeboom §3.2).
        pos = pos - pos.mean(0, keepdim=True)
        zt = torch.tensor(zs, dtype=torch.long)
        keep_idx = list(range(len(symbols)))
        if not include_hydrogens:
            keep_idx = [i for i, s in enumerate(symbols) if s != "H"]
            pos = pos[keep_idx]
        # Re-center on the SELECTED atoms: the centering above was on all
        # atoms (with H); after dropping H the heavy-atom centroid is non-zero
        # by the mass of the stripped hydrogens. CFM trainer is heavy-atom
        # only — its positions need to have centroid 0.
        pos = pos - pos.mean(0, keepdim=True)

        # 2026-10-04 — attach the TRUE bond graph.
        #
        # Without this the CFM trainer had to invent a target, and the rule it
        # used ("bonded iff one endpoint is Pt, Z=78 hardcoded") produced
        # 1331/1331 BOND_NO_BOND on a Ru complex — the head learned to predict
        # "no bond" everywhere and every decode came back as loose atoms.
        # `.BO.gz` carries Wiberg orders in the SAME atom order as the geometry,
        # so the mapping below is a relabelling, not a guess.
        #
        # Wf-A11 (2026-10-13): the construction is now centralised in
        # `_build_cloud_from_kept`, which populates all 19 new supervision
        # fields (per-atom: natural_charges, bo_per_atom, coord_neighbors;
        # per-pair: bond_orders_continuous, is_dative_pair; per-cloud:
        # metal_atom_index, oxidation_state, spin_state, csd_publication_year,
        # dft_*, dipole_m, metal_q, hl_gap, homo_e, lumo_e, polarizability;
        # str: stoichiometry, smiles).  The legacy 4 fields keep the same
        # contract, so every existing call site sees byte-equivalent clouds.
        bo_info = bo_full_index.get(code) if bo_full_index else None
        row = meta.get(code, {})
        # Wf-A22 (2026-10-13) step-1: per-cloud pocket draw.
        #
        # When ``pocket_pool`` is populated (pocket_pool_per_cloud=True),
        # draw ONE pocket for THIS cloud from the pool keyed on the CSD
        # code via a deterministic hash.  Otherwise pass through the
        # cohort-wide ``pocket`` (legacy / bit-exact).
        _cloud_pocket = pocket
        if pocket_pool is not None and pocket_pool:
            import hashlib as _hashlib
            _h = _hashlib.sha256(
                f"{code}|{int(pocket_draw_seed)}".encode("utf-8")
            ).hexdigest()
            _idx = int(_h, 16) % len(pocket_pool)
            _pocket_entry = pocket_pool[_idx]
            _cloud_pocket = load_pocket_from_pdb(_pocket_entry.pocket_path)
        cloud = _build_cloud_from_kept(
            code=code,
            keep_idx=keep_idx,
            pos=pos,
            zt=zt,
            symbols=symbols,
            row=row,
            bo_info=bo_info,
            natural_charges_map=natural_charges_map,
            pocket=_cloud_pocket,
        )
        if cloud is None:
            n_skip_symbol += 1
            continue
        n_seen += 1
        if n_clouds is None:
            reservoir.append(cloud)
        elif len(reservoir) < int(n_clouds):
            reservoir.append(cloud)
        else:
            j = rng.randrange(n_seen)
            if j < int(n_clouds):
                reservoir[j] = cloud
        if _CAP is not None and n_seen >= _CAP:
            break

    print(
        f"[tmqm-clouds] metals={sorted(wanted_metals)} coord_number={coord_number}: "
        f"{len(reservoir)} cloud(s) from {n_seen} candidate(s); "
        f"skipped {n_skip_symbol} (unmapped element) + {n_skip_no_metal} "
        f"(metal missing from geometry)"
    )
    # Wf-A30-2 B2 (2026-10-14) — surface the n_smiles_empty cohort.
    # Earlier, ~9.8% of tmQM rows have an empty SMILES column and the
    # loader silently drops smiles into ``None``. The downstream
    # Bemis-Murcko scaffold branch + the per_loss_projection consumer
    # both rely on the smiles being present; the dropout was invisible
    # to a reader of the loader summary. Counting here makes the gap
    # visible at INFO so a stale corpus can be detected without
    # grepping the source.
    n_smiles_empty = sum(1 for c in reservoir if c.smiles is None)
    print(
        f"[tmqm-clouds] smiles_empty: {n_smiles_empty} "
        f"({n_smiles_empty * 100 / max(len(reservoir), 1):.1f}%) "
        f"of {len(reservoir)} cloud(s) have smiles=None (Bemis-Murcko "
        f"scaffold branch falls back to CSD-prefix proxy for these rows)"
    )
    if not reservoir:
        raise RuntimeError("load_tmqm_clouds produced 0 clouds")

    # ---------------------------------------------------------------------------
    # min_per_metal enforcement (2026-10-12 zero-Pt finding).
    #
    # The reservoir above is unbiased across files for a GIVEN seed and
    # ``n_clouds``, but with small ``n_clouds`` (e.g. 192) it can give zero
    # representations of a particular metal (Pt in the seed=0 val window) even
    # when that metal is 24.8% of the underlying pool. The training fix is
    # already settled -- the model HAS seen Pt extensively -- so this is a
    # sampling-artefact back-fill, not a retraining question.
    #
    # Mechanism: re-stream the geometry files WITHOUT the streaming cap, keep
    # the FIRST ``min_per_metal`` non-reservoir candidates per metal, then
    # replace reservoir slots starting from the END with the union of those
    # back-fill clouds. Replacement order is deterministic (file iteration
    # order, which is sorted ``tmQM_X1 -> X2 -> X3``), so ``seed`` still
    # determines which reservoir slots are swapped.
    if int(min_per_metal) > 0:
        if n_clouds is None:
            raise ValueError(
                "min_per_metal requires n_clouds to be set; refusing to "
                "back-fill an unbounded selection"
            )
        if int(min_per_metal) * len(wanted_metals) > int(n_clouds):
            raise ValueError(
                f"min_per_metal={min_per_metal} * "
                f"|metals|={len(wanted_metals)} = "
                f"{int(min_per_metal) * len(wanted_metals)} exceeds "
                f"n_clouds={n_clouds}; bump n_clouds or lower min_per_metal"
            )
        # Identify each metal's current count from the reservoir itself.
        # Mononuclear tmQM carries the metal somewhere in atomic_numbers;
        # the file's iteration order is arbitrary, so locate the metal by
        # Z-value rather than assuming it is at index 0.
        z_to_sym = {z: s for s, z in TMQM_METAL_Z.items()}
        metal_z_set = set(TMQM_METAL_Z.values())
        reservoir_metal_count: Dict[str, int] = {m: 0 for m in wanted_metals}
        reservoir_by_pos: Dict[int, str] = {}  # position -> metal symbol

        def _metal_sym_of(atomic_numbers: torch.Tensor) -> Optional[str]:
            for z in atomic_numbers.tolist():
                if z in metal_z_set:
                    return z_to_sym.get(z)
            return None

        for pos, cl in enumerate(reservoir):
            sym = _metal_sym_of(cl.atomic_numbers)
            if sym in reservoir_metal_count:
                reservoir_metal_count[sym] += 1
            reservoir_by_pos[pos] = sym if sym is not None else ""
        # Back-fill: for each requested metal that has FEWER than
        # min_per_metal copies in the reservoir, re-stream the geometry files
        # (the main loop already exited early under the original _CAP) and
        # take the FIRST missing candidates per metal.
        #
        # The full back-fill rebuilds the cloud object the same way the main
        # loop did (CoM removal, BO remap, AtomCloud call), so what we stash
        # here is byte-equivalent to what a non-cap-bounded reservoir would
        # have produced given the same iteration order.
        metals_still_needed = {
            m: int(min_per_metal) - reservoir_metal_count[m]
            for m in wanted_metals
            if reservoir_metal_count[m] < int(min_per_metal)
        }
        # pick reservoir replacement slots from the END first (most-recent
        # seen) so a deterministic seed keeps earlier reservoir slots stable.
        replacement_slots: List[int] = list(reversed(range(len(reservoir))))
        if sum(metals_still_needed.values()) > len(replacement_slots):
            raise RuntimeError(
                "min_per_metal underflow: more back-fill requests than "
                f"reservoir slots ({sum(metals_still_needed.values())} vs "
                f"{len(replacement_slots)}); this should be guarded above"
            )
        for code, symbols, coords in iter_tmqm_blocks(xyz):
            if not metals_still_needed:
                break
            if code not in keep_codes:
                continue
            zs: List[int] = []
            bad = False
            for s in symbols:
                if not include_hydrogens and s == "H":
                    continue
                z = _SYMBOL_TO_Z.get(s)
                if z is None:
                    bad = True
                    break
                zs.append(z)
            if bad:
                continue
            # Find the metal Z anywhere in ``zs`` (mononuclear but file
            # atom order is arbitrary -- cannot assume ``zs[0]``).
            sym: Optional[str] = None
            for z in zs:
                if z in metal_z_set:
                    sym = z_to_sym.get(z)
                    break
            if sym is None or sym not in metals_still_needed:
                continue
            if metals_still_needed[sym] <= 0:
                continue
            # Reuse cloud construction (same code as the main loop).
            pos = torch.tensor(coords, dtype=torch.float32)
            pos = pos - pos.mean(0, keepdim=True)
            zt = torch.tensor(zs, dtype=torch.long)
            keep_idx = list(range(len(symbols)))
            if not include_hydrogens:
                keep_idx = [i for i, s in enumerate(symbols) if s != "H"]
                pos = pos[keep_idx]
            pos = pos - pos.mean(0, keepdim=True)
            # Wf-A11: route through the shared chokepoint so the back-fill
            # clouds carry the same 19 new fields as the main-loop clouds.
            bo_info = bo_full_index.get(code) if bo_full_index else None
            row = meta.get(code, {})
            # Wf-A22 (2026-10-13) step-1: per-cloud pocket draw (mirrors the
            # main-loop logic so back-fill rows see the same c-flow).
            _cloud_pocket = pocket
            if pocket_pool is not None and pocket_pool:
                import hashlib as _hashlib
                _h = _hashlib.sha256(
                    f"{code}|{int(pocket_draw_seed)}".encode("utf-8")
                ).hexdigest()
                _idx = int(_h, 16) % len(pocket_pool)
                _pocket_entry = pocket_pool[_idx]
                _cloud_pocket = load_pocket_from_pdb(_pocket_entry.pocket_path)
            cloud = _build_cloud_from_kept(
                code=code,
                keep_idx=keep_idx,
                pos=pos,
                zt=zt,
                symbols=symbols,
                row=row,
                bo_info=bo_info,
                natural_charges_map=natural_charges_map,
                pocket=_cloud_pocket,
            )
            if cloud is None:
                continue
            # Find a replacement slot using seed-stable order.
            # We shuffle replacement_slots with rng2 once to avoid
            # always overwriting the same trailing slot when many metals
            # are needed in sequence; under the same seed, this is still
            # deterministic.
            if not replacement_slots:
                break
            # Pop the slot whose index is the LARGEST among current
            # replacement slots whose existing metal does not match ``sym``
            # — we prefer swapping a different-metal slot so we never
            # replace what we want to keep. If every remaining slot is the
            # same metal, fall back to the largest-index slot.
            chosen_slot = None
            for s in sorted(replacement_slots, reverse=True):
                existing = reservoir_by_pos.get(s)
                if existing != sym:
                    chosen_slot = s
                    break
            if chosen_slot is None:
                chosen_slot = replacement_slots[0]
            replacement_slots.remove(chosen_slot)
            reservoir[chosen_slot] = cloud
            reservoir_metal_count[sym] += 1
            metals_still_needed[sym] -= 1
        unmet = {m: n for m, n in metals_still_needed.items() if n > 0}
        if unmet:
            raise RuntimeError(
                f"min_per_metal could not be satisfied: metals under quota = "
                f"{unmet}; available_candidates in the filtered pool is the "
                f"limit and at least one quota exceeds the supply"
            )
        print(
            f"[tmqm-clouds] min_per_metal={min_per_metal}: enforced per-metal "
            f"minimum; reservoir now has "
            + ", ".join(
                f"{m}={reservoir_metal_count[m]}"
                for m in sorted(wanted_metals)
            )
        )

    return reservoir
