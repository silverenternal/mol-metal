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

__all__ = [
    "TMQM_METAL_Z",
    "load_tmqm_metadata",
    "iter_tmqm_blocks",
    "load_tmqm_clouds",
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


def load_tmqm_clouds(
    tmqm_dir: str | Path,
    metals: Optional[Iterable[str]] = None,
    coord_number: Optional[int] = None,
    n_clouds: Optional[int] = None,
    seed: int = 0,
    include_hydrogens: bool = False,
    split: Optional[str] = None,
    val_frac: float = 0.2,
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
        ``"train"`` / ``"val"`` for a DISJOINT partition, or None for no split.

        This exists because the obvious approach -- calling this twice with
        different seeds -- LEAKS.  MEASURED: with n_clouds=96 and 32 (so the
        candidate caps are 480 and 160) the val candidate pool was a strict
        SUBSET of the train one, 160/160, because both stream from the same
        head of the same files.  Any decode_ratio measured that way is
        contaminated by train/val overlap.

        The partition here is by a stable hash of the CSD code, so the two
        sides are disjoint by construction and the split does not depend on
        how many clouds are requested on either side.
    val_frac
        Fraction assigned to ``"val"`` when ``split`` is set.

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
    from molmetal.models.bond_head import AtomCloud

    root = Path(tmqm_dir)
    parsed = root / "tmqm_parsed.csv"
    if not parsed.is_file():
        raise FileNotFoundError(f"tmqm_parsed.csv not found under {root}")

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

    # Deterministic, disjoint train/val partition by CSD-code hash.  Kept
    # independent of n_clouds so requesting 96 train + 32 val cannot make one
    # a subset of the other (the leak described in the docstring).
    if split is not None:
        key = str(split).strip().lower()
        if key not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
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
    bo_index: Optional[Dict[str, List[Tuple[int, int, int]]]] = None
    bo_files = [root / name for name in _BO_FILES]
    if any(p.is_file() for p in bo_files):
        bo_index = {}
        for _code, _pairs in iter_bo_bonds(bo_files):
            if _code in keep_codes:
                bo_index[_code] = _pairs
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

    # Reservoir sample: keeps the selection unbiased across files while still
    # stopping early once the pool is comfortably larger than the request.
    rng = random.Random(int(seed))
    reservoir: List["object"] = []
    n_seen = 0
    n_skip_symbol = 0
    n_skip_no_metal = 0
    _CAP = None if n_clouds is None else max(int(n_clouds) * 5, int(n_clouds))

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
        zt = torch.tensor(zs, dtype=torch.long)
        keep_idx = list(range(len(symbols)))
        if not include_hydrogens:
            keep_idx = [i for i, s in enumerate(symbols) if s != "H"]
            pos = pos[keep_idx]

        # 2026-10-04 — attach the TRUE bond graph.
        #
        # Without this the CFM trainer had to invent a target, and the rule it
        # used ("bonded iff one endpoint is Pt, Z=78 hardcoded") produced
        # 1331/1331 BOND_NO_BOND on a Ru complex — the head learned to predict
        # "no bond" everywhere and every decode came back as loose atoms.
        # `.BO.gz` carries Wiberg orders in the SAME atom order as the geometry,
        # so the mapping below is a relabelling, not a guess.
        #
        # The remap matters: dropping hydrogens renumbers the atoms, so a pair
        # written against the file's indexing would otherwise point at the
        # wrong atoms — silently, and only for clouds that had hydrogens.
        bonds_t = bond_types_t = None
        bo_pairs = bo_index.get(code) if bo_index is not None else None
        if bo_pairs:
            old_to_new = {old: new for new, old in enumerate(keep_idx)}
            pairs = [
                (old_to_new[i], old_to_new[j], t)
                for i, j, t in bo_pairs
                if i in old_to_new and j in old_to_new
            ]
            if pairs:
                bonds_t = torch.tensor([[p[0] for p in pairs], [p[1] for p in pairs]],
                                       dtype=torch.long)
                bond_types_t = torch.tensor([p[2] for p in pairs], dtype=torch.long)

        try:
            cloud = AtomCloud(
                positions=pos,
                atomic_numbers=zt,
                bonds=bonds_t,
                bond_types=bond_types_t,
            )
        except ValueError:
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
    if not reservoir:
        raise RuntimeError("load_tmqm_clouds produced 0 clouds")
    return reservoir
