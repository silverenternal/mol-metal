"""tmQM loader — 108k mononuclear transition-metal complexes (TODO F2 / P1).

Source
------
Balcells & Skjelstad, *tmQM Dataset — Quantum Geometries and Properties of
86k Transition Metal Complexes*, J. Chem. Inf. Model. 2020, 60, 6135-6146
(the release shipped here is the extended 108k version).
Level of theory: TPSSh-D3BJ/def2-SVP.  Licence: MIT.
Repo: https://github.com/uiocompcat/tmQM

On-disk layout (``/mnt/storage/data/molmetal/tmQM/``)::

    tmQM_y.csv          ';'-separated — CSD_code, 8 DFT properties, SMILES
    tmQM_X{1,2,3}.xyz.gz  XYZ blocks; the comment line carries
                          ``CSD_code = X | q = .. | S = .. |
                            Stoichiometry = .. | MND = n | <years> CSD``
    tmQM_X{1,2,3}.BO.gz   Wiberg bond-order blocks, one line per atom::

                            <idx> <El> <total_BO>  <El> <idx> <BO> ...

    tmQM_X.q            Natural atomic charges (not used here).

Two supervision signals are extracted for D-MPNN pre-training:

``coord_number``
    The **MND** field (Metal Node Degree) from the XYZ comment line — the
    number of ligating atoms bonded to the metal centre.  This is tmQM's
    own connectivity assignment, so it is consistent across the corpus.

``metal_bo_total``
    The metal atom's *total* Wiberg bond order (3rd column of its BO line),
    i.e. the summed bond order over all its bonds.  ``metal_bo_sum`` is the
    sum over the explicitly listed neighbour bond orders (tmQM truncates the
    neighbour list at a small BO threshold, so the two differ slightly).

Notes
-----
* The metal is identified from the ``Stoichiometry`` field, **not** from the
  first atom of the XYZ block — tmQM does not always put the metal first
  (only ~65% of entries do).
* ~7.7k of the 108,541 entries have an empty SMILES field; those rows are
  dropped by default (``require_smiles=True``) because the graph models
  need a parseable molecular graph.
* Parsed results are cached as a plain CSV next to the raw data (or under
  ``molmetal/data/.cache`` if the data directory is read-only) — the full
  parse takes ~15 s, the cached load ~0.5 s.
"""

from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TMQM_DIR = Path("/mnt/storage/data/molmetal/tmQM")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FALLBACK_CACHE_DIR = PROJECT_ROOT / "molmetal" / "data" / ".cache"

#: The 30 d-block metals covered by tmQM (mononuclear complexes only).
TRANSITION_METALS: tuple[str, ...] = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
)
_TM_SET = frozenset(TRANSITION_METALS)

#: The three metals this project's anticancer story is built around.
PAPER_METALS: tuple[str, ...] = ("Pt", "Ru", "Ir")

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
_CACHE_NAME = "tmqm_parsed.csv"

_XYZ_FILES = ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz")
_BO_FILES = ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")
_Y_FILE = "tmQM_y.csv"


def tmqm_dir() -> Path:
    """Root of the raw tmQM release (override with ``$MOLMETAL_TMQM_DIR``)."""
    return Path(os.environ.get("MOLMETAL_TMQM_DIR", str(_DEFAULT_TMQM_DIR)))


def _cache_path(root: Path) -> Path:
    if os.access(root, os.W_OK):
        return root / _CACHE_NAME
    _FALLBACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _FALLBACK_CACHE_DIR / _CACHE_NAME


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------
def metal_from_stoichiometry(stoich: str) -> Optional[str]:
    """Return the single transition metal in a tmQM stoichiometry string.

    ``'C40H36LaN2P3Se6'`` -> ``'La'``.  Returns ``None`` if zero or more than
    one d-block element is present (should not happen for mononuclear tmQM,
    but we stay defensive).
    """
    hits = _TM_SET.intersection(_ELEMENT_RE.findall(stoich))
    if len(hits) != 1:
        return None
    return next(iter(hits))


def _parse_header(line: str) -> Optional[Dict[str, object]]:
    """Parse one XYZ comment line into a record dict."""
    if "CSD_code" not in line:
        return None
    fields = [seg.strip() for seg in line.strip().split("|")]
    rec: Dict[str, object] = {}
    for seg in fields:
        if "=" not in seg:
            continue
        key, _, val = seg.partition("=")
        rec[key.strip()] = val.strip()
    code = rec.get("CSD_code")
    stoich = rec.get("Stoichiometry")
    if code is None or stoich is None:
        return None
    try:
        mnd = int(str(rec.get("MND", "")).strip())
    except (TypeError, ValueError):
        return None
    return {
        "csd_code": str(code),
        "stoichiometry": str(stoich),
        "metal": metal_from_stoichiometry(str(stoich)),
        "charge": int(float(rec.get("q", 0))),
        "spin": int(float(rec.get("S", 0))),
        "coord_number": mnd,
    }


def parse_xyz_headers(root: Optional[Path] = None) -> pd.DataFrame:
    """Stream the three ``.xyz.gz`` shards, keeping only the comment lines.

    Returns a DataFrame with columns
    ``csd_code, stoichiometry, metal, charge, spin, coord_number``.
    """
    root = Path(root) if root is not None else tmqm_dir()
    rows: List[Dict[str, object]] = []
    for name in _XYZ_FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"tmQM shard missing: {path}")
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    rec = _parse_header(line)
                    if rec is not None:
                        rows.append(rec)
    return pd.DataFrame(rows)


def parse_bo_metal(root: Optional[Path] = None) -> pd.DataFrame:
    """Extract the metal-centre Wiberg bond-order line from each BO block.

    Returns a DataFrame with columns
    ``csd_code, metal_bo_total, metal_bo_sum, bo_n_neighbors``.

    ``metal_bo_total`` is the value tmQM prints as the atom's total bond
    order; ``metal_bo_sum`` sums the neighbour bond orders actually listed
    (tmQM truncates small contributions, so ``sum <= total``).
    """
    root = Path(root) if root is not None else tmqm_dir()
    rows: List[Dict[str, object]] = []
    for name in _BO_FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"tmQM BO shard missing: {path}")
        with gzip.open(path, "rt") as fh:
            code: Optional[str] = None
            done = True
            for line in fh:
                if line.startswith("CSD_code"):
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    done = False
                    continue
                if done or code is None:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                element = parts[1]
                if element not in _TM_SET:
                    continue
                try:
                    total = float(parts[2])
                except ValueError:
                    continue
                tail = parts[3:]
                # neighbours come as (element, index, bond_order) triples
                bos = [float(tail[i + 2]) for i in range(0, len(tail) - 2, 3)]
                rows.append(
                    {
                        "csd_code": code,
                        "metal_bo_total": total,
                        "metal_bo_sum": float(sum(bos)),
                        "bo_n_neighbors": len(bos),
                    }
                )
                done = True  # mononuclear: skip the rest of this block
    return pd.DataFrame(rows)


def parse_natural_charges(
    q_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Stream the ``tmQM_X.q`` (NBO natural-charge) file and return per-complex
    per-atom q values.

    Parameters
    ----------
    q_path
        Direct path to the ``.q`` / ``.q.gz`` file.  If ``None`` (the default),
        the function probes the standard tmQM root for ``tmQM_X.q`` /
        ``tmQM_X.q.gz``.  Both plain and gzipped variants are supported; tmQM's
        own release ships plain.
    root
        Raw-data directory.  Only used when ``q_path`` is None; defaults to
        :func:`tmqm_dir`.

    Returns
    -------
    dict[str, list[float]]
        Mapping ``csd_code -> [q_1, q_2, ...]`` (1-based indices become
        0-based list positions; the list is exactly as long as the complex's
        heavy-atom count in the matched geometry block).

    Format
    ------
    The file uses the same ``CSD_code = XXXXXX`` block-header format as
    :func:`parse_bo_metal` (and the XYZ shards).  Per-atom lines are read as
    ``<index> <element> <charge>`` triples — the typical Gaussian NBO summary
    layout tmQM ships.  Anything we cannot parse is logged and SKIPPED; the
    function NEVER silently drops a complex (which would make a missing
    per-atom field indistinguishable from a parse failure downstream — exactly
    the failure mode the natural-charge field exists to fix per SCOUT D1).

    Raises
    ------
    FileNotFoundError
        When ``q_path`` (or the resolved standard path) is absent.  This is
        a FATAL condition: the function has no way to know which complexes
        were intended, so a missing source cannot become a silent empty dict
        (an empty dict would mean ``natural_charges=None`` for every cloud,
        which is the bug we are fixing).
    """
    import logging

    if q_path is None:
        root = Path(root) if root is not None else tmqm_dir()
        for candidate in (root / "tmQM_X.q", root / "tmQM_X.q.gz"):
            if candidate.is_file():
                q_path = candidate
                break
        if q_path is None:
            raise FileNotFoundError(
                f"tmQM natural-charge file not found under {root}: tried "
                f"'tmQM_X.q' and 'tmQM_X.q.gz'"
            )
    q_path = Path(q_path)
    if not q_path.is_file():
        raise FileNotFoundError(f"tmQM natural-charge file missing: {q_path}")

    logger = logging.getLogger(__name__)
    out: Dict[str, List[float]] = {}
    # Stream the file.  The NBO summary file is line-oriented; each line is
    # one atom.  Block headers (CSD_code = XXXXXX) reset the buffer.
    open_fn = gzip.open if q_path.suffix == ".gz" else (lambda p, m: open(p, m))
    code: Optional[str] = None
    buf: List[float] = []
    n_skip = 0
    with open_fn(q_path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("CSD_code"):
                if code is not None and buf:
                    out[code] = buf
                code = line.split("=", 1)[1].split("|", 1)[0].strip()
                buf = []
                continue
            if code is None:
                # Per-atom lines arriving before any block header — the file
                # is malformed; skip but log so a future code review sees it.
                n_skip += 1
                continue
            parts = line.split()
            if len(parts) < 3:
                n_skip += 1
                continue
            try:
                # 1-based index in the file -> we just need the q value
                # (the i-th atom's charge becomes the i-1 slot of the list).
                _idx = int(parts[0])
                q_val = float(parts[2])
            except ValueError:
                n_skip += 1
                continue
            buf.append(q_val)
    if code is not None and buf:
        out[code] = buf
    if n_skip:
        logger.warning(
            "parse_natural_charges: skipped %d unparseable line(s) in %s",
            n_skip,
            q_path,
        )
    return out


def parse_properties(root: Optional[Path] = None) -> pd.DataFrame:
    """Load ``tmQM_y.csv`` (';'-separated DFT properties + SMILES)."""
    root = Path(root) if root is not None else tmqm_dir()
    path = root / _Y_FILE
    if not path.exists():
        raise FileNotFoundError(f"tmQM property table missing: {path}")
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns={"CSD_code": "csd_code", "SMILES": "smiles"})
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_tmqm(
    metals: Optional[Sequence[str]] = None,
    require_smiles: bool = True,
    root: Optional[Path] = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the merged tmQM table.

    Parameters
    ----------
    metals
        Restrict to these metal centres (e.g. ``("Pt", "Ru", "Ir")``).
        ``None`` (default) keeps all 30 d-block metals.
    require_smiles
        Drop rows whose SMILES field is empty (7,692 of 108,541).
    root
        Raw-data directory; defaults to ``$MOLMETAL_TMQM_DIR`` or
        ``/mnt/storage/data/molmetal/tmQM``.
    use_cache / refresh
        Read from (write to) the parsed CSV cache; ``refresh=True`` forces a
        re-parse of the gzipped shards.

    Returns
    -------
    pandas.DataFrame with columns::

        csd_code, metal, coord_number, charge, spin, stoichiometry,
        metal_bo_total, metal_bo_sum, bo_n_neighbors, smiles,
        Electronic_E, Dispersion_E, Dipole_M, Metal_q, HL_Gap,
        HOMO_Energy, LUMO_Energy, Polarizability, CSD_years
    """
    root = Path(root) if root is not None else tmqm_dir()
    cache = _cache_path(root)

    df: Optional[pd.DataFrame] = None
    if use_cache and not refresh and cache.exists():
        df = pd.read_csv(cache)
    if df is None:
        headers = parse_xyz_headers(root)
        bo = parse_bo_metal(root)
        props = parse_properties(root)
        df = headers.merge(bo, on="csd_code", how="left").merge(
            props, on="csd_code", how="left"
        )
        df = df[df["metal"].notna()]
        # One CSD code (IMUJUY) appears twice in the BO shards; the merge would
        # otherwise duplicate that complex.
        df = df.drop_duplicates(subset="csd_code").reset_index(drop=True)
        if use_cache:
            try:
                df.to_csv(cache, index=False)
            except OSError:
                pass

    if require_smiles:
        df = df[df["smiles"].notna() & (df["smiles"].astype(str).str.len() > 0)]
    if metals is not None:
        df = filter_by_metal(df, metals)
    return df.reset_index(drop=True)


def filter_by_metal(df: pd.DataFrame, metals: Sequence[str] | str) -> pd.DataFrame:
    """Keep only rows whose metal centre is in ``metals`` (case-sensitive)."""
    if isinstance(metals, str):
        metals = [metals]
    wanted = list(metals)
    unknown = [m for m in wanted if m not in _TM_SET]
    if unknown:
        raise ValueError(
            f"Not transition metals present in tmQM: {unknown}. "
            f"Valid: {', '.join(TRANSITION_METALS)}"
        )
    return df[df["metal"].isin(wanted)].reset_index(drop=True)


@dataclass
class TmqmStats:
    """Summary of a (filtered) tmQM slice."""

    n_total: int
    per_metal: Dict[str, int] = field(default_factory=dict)
    coord_distribution: Dict[str, Dict[int, int]] = field(default_factory=dict)
    coord_mean: Dict[str, float] = field(default_factory=dict)
    bo_mean: Dict[str, float] = field(default_factory=dict)
    n_missing_bo: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_total": self.n_total,
            "per_metal": self.per_metal,
            "coord_distribution": {
                m: {int(k): int(v) for k, v in d.items()}
                for m, d in self.coord_distribution.items()
            },
            "coord_mean": self.coord_mean,
            "bo_mean": self.bo_mean,
            "n_missing_bo": self.n_missing_bo,
        }


def summarize(df: pd.DataFrame) -> TmqmStats:
    """Compute per-metal counts, coordination histogram and mean metal BO."""
    per_metal = df["metal"].value_counts().to_dict()
    coord_dist: Dict[str, Dict[int, int]] = {}
    coord_mean: Dict[str, float] = {}
    bo_mean: Dict[str, float] = {}
    for metal, sub in df.groupby("metal"):
        coord_dist[str(metal)] = {
            int(k): int(v)
            for k, v in sub["coord_number"].value_counts().sort_index().items()
        }
        coord_mean[str(metal)] = float(sub["coord_number"].mean())
        bo_mean[str(metal)] = float(
            np.nanmean(sub["metal_bo_total"].to_numpy(dtype=float))
        )
    return TmqmStats(
        n_total=int(len(df)),
        per_metal={str(k): int(v) for k, v in per_metal.items()},
        coord_distribution=coord_dist,
        coord_mean=coord_mean,
        bo_mean=bo_mean,
        n_missing_bo=int(df["metal_bo_total"].isna().sum()),
    )


# ---------------------------------------------------------------------------
# Wf-A16 (2026-10-13) — pocket loader (c-flow end-to-end propagation).
#
# tmQM is a small-molecule corpus with no protein context, so the loader
# below reads a SEPARATE protein pocket from a .pdb side-car (typically a
# CrossDocked / PDBBind receptor).  Heavy atoms only; hydrogen-free for
# parity with the tmQM heavy-atom convention; capped at ``max_atoms`` to
# bound the EGNN-Lite message-passing cost in PocketEncoder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PocketCloud:
    """A small frozen protein-pocket point cloud (Wf-A16 protocol).

    Attributes
    ----------
    coords : (P, 3) tensor
        Heavy-atom coordinates in Angstrom.
    atom_types : (P,) int64 tensor
        Atomic numbers (1=H, 6=C, 7=N, 8=O, 16=S, ...).
    mask : (P,) bool tensor
        True for every real atom (always all-True; reserved for future
        padding when callers batch pockets of unequal size).
    pocket_id : str
        CSD code / PDB id / hash of the source file.
    """

    coords: "torch.Tensor"
    atom_types: "torch.Tensor"
    mask: "torch.Tensor"
    pocket_id: str = ""

    def __post_init__(self) -> None:
        # Defer torch import to keep tmqm.py usable without torch if needed.
        import torch as _torch  # noqa: WPS433 (deferred)

        if self.coords.dim() != 2 or int(self.coords.shape[1]) != 3:
            raise ValueError(
                f"PocketCloud.coords must be (P, 3); got {tuple(self.coords.shape)}"
            )
        if int(self.coords.shape[0]) != int(self.atom_types.shape[0]):
            raise ValueError(
                "PocketCloud.coords and atom_types must agree on P="
                f"{int(self.coords.shape[0])} vs {int(self.atom_types.shape[0])}"
            )
        if int(self.mask.shape[0]) != int(self.coords.shape[0]):
            raise ValueError("PocketCloud.mask must match coords on P")
        if self.coords.dtype != _torch.float32:
            raise ValueError(
                f"PocketCloud.coords dtype must be float32; got {self.coords.dtype}"
            )
        if self.atom_types.dtype != _torch.int64:
            raise ValueError(
                f"PocketCloud.atom_types dtype must be int64; got {self.atom_types.dtype}"
            )


def load_pocket_from_pdb(
    pdb_path: str | Path,
    max_atoms: int = 256,
) -> PocketCloud:
    """Load a protein pocket as a :class:`PocketCloud` from a .pdb file.

    Reads via :func:`rdkit.Chem.MolFromPDBFile` (already on the venv per
    :file:`scripts/setup_molmetal_env.sh`).  Hydrogen atoms are skipped for
    parity with the tmQM heavy-atom convention; atoms beyond ``max_atoms``
    are truncated by selecting the residues nearest the centroid (keeps
    the binding-site core, drops the periphery).

    Args
    ----
    pdb_path
        Path to a .pdb file.  Raises :class:`FileNotFoundError` when missing.
    max_atoms
        Cap on returned heavy atoms.  Default 256 matches the pocket
        message-passing budget in :class:`PocketEncoder`.

    Returns
    -------
    PocketCloud
        Frozen dataclass with ``coords``, ``atom_types``, ``mask``, and
        ``pocket_id = Path(pdb_path).stem``.

    Raises
    ------
    ImportError
        When RDKit is unavailable.
    FileNotFoundError
        When ``pdb_path`` does not exist.
    ValueError
        When the PDB has zero heavy atoms or parsing fails.
    """
    import torch as _torch  # noqa: WPS433

    p = Path(pdb_path)
    if not p.is_file():
        raise FileNotFoundError(f"pdb_path not found: {p}")

    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover - RDKit missing
        raise ImportError(
            "load_pocket_from_pdb requires RDKit; install rdkit or run the "
            "protein-pocket prep step on a host with rdkit available."
        ) from exc

    mol = Chem.MolFromPDBFile(str(p), removeHs=True, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit MolFromPDBFile returned None for {p}")

    conf = mol.GetConformer()
    if conf is None:
        raise ValueError(f"No conformer found in {p}")

    coords_list = []
    types_list = []
    for atom in mol.GetAtoms():
        z = int(atom.GetAtomicNum())
        if z == 1:
            continue  # skip hydrogens
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords_list.append((float(pos.x), float(pos.y), float(pos.z)))
        types_list.append(z)

    if not coords_list:
        raise ValueError(f"PDB {p} has zero heavy atoms")

    coords = _torch.tensor(coords_list, dtype=_torch.float32)
    atom_types = _torch.tensor(types_list, dtype=_torch.int64)

    # Truncate to max_atoms by nearest-to-centroid (binding-site core).
    if int(coords.shape[0]) > int(max_atoms):
        centroid = coords.mean(dim=0)
        d = _torch.norm(coords - centroid.unsqueeze(0), dim=-1)
        keep = _torch.argsort(d)[: int(max_atoms)]
        coords = coords[keep].contiguous()
        atom_types = atom_types[keep].contiguous()

    n = int(coords.shape[0])
    mask = _torch.ones(n, dtype=_torch.bool)
    return PocketCloud(
        coords=coords,
        atom_types=atom_types,
        mask=mask,
        pocket_id=str(p.stem),
    )


__all__ = [
    "PAPER_METALS",
    "PocketCloud",
    "TRANSITION_METALS",
    "TmqmStats",
    "filter_by_metal",
    "load_pocket_from_pdb",
    "load_tmqm",
    "metal_from_stoichiometry",
    "parse_bo_metal",
    "parse_natural_charges",
    "parse_properties",
    "parse_xyz_headers",
    "summarize",
    "tmqm_dir",
]
