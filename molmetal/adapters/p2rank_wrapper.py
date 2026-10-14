"""molmetal/adapters/p2rank_wrapper.py — P2Rank pocket detection (Wf-A33 prep).

P2Rank (Krivak & Hoksza 2018, J Cheminform; actively maintained through
PrankWeb 4 / 2025) is a lightweight ML-based pocket detector. It takes
a full PDB file as input and returns the predicted binding-site
residue ranges, ranked by probability. License: MIT.

Why we wrap it
--------------
The CFM pocket-conditioning pipeline currently consumes a *pre-cropped*
pocket (typically a 10 Å sphere around a docked ligand from the
CrossDocked 2020 benchmark). For the platiny_pocket_pool (Wf-A31
deliverable) the pocket files have NOT been extracted yet — only the
10 literature PDB IDs are declared in the CSV. P2Rank is the
upstream detector that turns a full PDB into a residue range we can
then crop to a 10 Å sphere and feed to PocketGenEncoder.

This module is the Wf-A33 prep. The actual subprocess + output
parsing is **deferred** (Java is a system dep that is not in the
default `.venv`; per AGENTS.md §5.1 it is a user OK point to add).
What we ship now:

  1. P2RankResult dataclass (the output of one predict call)
  2. parse_p2rank_csv(csv_path) — pure-Python parser, no subprocess
  3. crop_pdb_to_residues(pdb_path, residues, out_path) — RDKit-free
     fallback that uses Biopython (already a project dep)
  4. is_p2rank_available() — checks for the JAR / CLI on PATH
  5. run_p2rank(pdb_path, out_dir, p2rank_cmd="p2rank") — stub that
     raises NotImplementedError pointing the user at the install
     instructions; will be implemented in Wf-A33.

References
----------
- P2Rank paper: Krivak & Hoksza, J Cheminform 10, 39 (2018).
  DOI: 10.1186/s13321-018-0285-8
- P2Rank GitHub: https://github.com/rdk/p2rank
- PrankWeb (web wrapper, 2025): https://prankweb.cz/
- License: MIT (https://github.com/rdk/p2rank/blob/master/LICENSE)

Usage
-----
    from molmetal.adapters.p2rank_wrapper import (
        parse_p2rank_csv, crop_pdb_to_residues, is_p2rank_available,
    )

    # Once p2rank is on PATH:
    if is_p2rank_available():
        run_p2rank("input.pdb", "out_dir")           # Wf-A33 wire-in
        result = parse_p2rank_csv("out_dir/input_residues.csv")
        crop_pdb_to_residues("input.pdb", result.top_pocket_residues, "pocket.pdb")
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------
# Path constants
# ----------------------------------------------------------------------

#: Default p2rank command name. Override via ``MOLMETAL_P2RANK_CMD``.
P2RANK_CMD_DEFAULT = "p2rank"

#: Where to look for p2rank when ``MOLMETAL_P2RANK_CMD`` is not set.
_P2RANK_CANDIDATE_PATHS = (
    "/usr/local/bin/p2rank",
    "/opt/p2rank/p2rank.sh",
    "/home/linuxbrew/.linuxbrew/bin/p2rank",
    "/home/hugo/codes/pocket/tools/p2rank/p2rank.sh",
)


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class P2RankResult:
    """One row of P2Rank output: a predicted pocket.

    Attributes
    ----------
    rank : int
        1-based rank (1 = top prediction).
    score : float
        P2Rank composite score (higher = better).
    probability : float
        Per-residue probability of being in the pocket.
    chain : str
        Chain ID of the centre residue.
    resi : int
        Residue index (PDB residue number) of the centre.
    resn : str
        3-letter residue name of the centre.
    residues : Tuple[Tuple[str, int], ...]
        All residues in the predicted pocket, as (chain, resi) pairs.
    """
    rank: int
    score: float
    probability: float
    chain: str
    resi: int
    resn: str
    residues: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def centre(self) -> Tuple[str, int, str]:
        return (self.chain, self.resi, self.resn)


# ----------------------------------------------------------------------
# CLI availability
# ----------------------------------------------------------------------

def _resolve_p2rank_cmd() -> Optional[str]:
    """Return the p2rank command path if it can be located, else None.

    Resolution order (per AGENTS.md §5.5 filesystem-truth):
      1. ``MOLMETAL_P2RANK_CMD`` env var (if set and runnable)
      2. ``which p2rank`` (PATH lookup)
      3. ``_P2RANK_CANDIDATE_PATHS`` (well-known install locations)
    """
    env_cmd = os.environ.get("MOLMETAL_P2RANK_CMD")
    if env_cmd and os.access(env_cmd, os.X_OK):
        return env_cmd
    which = shutil.which(P2RANK_CMD_DEFAULT)
    if which:
        return which
    for cand in _P2RANK_CANDIDATE_PATHS:
        if os.access(cand, os.X_OK):
            return cand
    return None


def is_p2rank_available() -> bool:
    """Return True iff the p2rank CLI is reachable on this host.

    Cheap, side-effect-free check. Use this to gate UI prompts / early
    returns before the heavy crop path.
    """
    return _resolve_p2rank_cmd() is not None


# ----------------------------------------------------------------------
# Subprocess driver (Wf-A33 wire-in target)
# ----------------------------------------------------------------------

def run_p2rank(
    pdb_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    p2rank_cmd: Optional[str] = None,
    timeout_s: int = 600,
) -> Path:
    """Invoke p2rank on a single PDB file.

    Parameters
    ----------
    pdb_path : path-like
        Input PDB file (full protein structure).
    out_dir : path-like
        Output directory. p2rank writes ``<basename>_residues.csv`` here
        plus per-pocket PDBs as ``<basename>_pocket<N>.pdb``.
    p2rank_cmd : str | None
        Override the p2rank executable. If None, falls back to
        ``_resolve_p2rank_cmd()``.
    timeout_s : int
        Subprocess timeout in seconds. Default 10 min — p2rank typically
        takes 1-5 s per protein on a single core.

    Returns
    -------
    Path
        Path to the residues CSV (``<out_dir>/<basename>_residues.csv``).

    Raises
    ------
    NotImplementedError
        Wf-A33 wire-in target. Currently raises with install instructions.
    """
    raise NotImplementedError(
        "run_p2rank() is the Wf-A33 wire-in target. "
        "Install p2rank (https://github.com/rdk/p2rank) and set "
        "MOLMETAL_P2RANK_CMD=/path/to/p2rank.sh, then re-enable this "
        "function. The pure-Python parser (parse_p2rank_csv) and the "
        "Biopython cropper (crop_pdb_to_residues) are available NOW "
        "for testing on pre-computed p2rank outputs."
    )


# ----------------------------------------------------------------------
# Pure-Python CSV parser
# ----------------------------------------------------------------------

#: Column names we expect in p2rank's *_residues.csv.
#: P2Rank emits several variants; we accept either the legacy or
#: 2024-revised header.
_P2RANK_CSV_COLS = (
    "rank",
    "score",
    "label",
    "chain",
    "resi",
    "resn",
    "probability",
)


def _coerce_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _coerce_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def parse_p2rank_csv(csv_path: str | os.PathLike) -> List[P2RankResult]:
    """Parse a p2rank ``*_residues.csv`` file into a list of P2RankResult.

    The CSV has one row per predicted pocket centre, with the per-pocket
    residue list either inline (column ``residues_list``) or in a sibling
    file (``<basename>_pocket<N>.pdb``). When inline, the residue list
    is a semicolon-separated ``chain:resi`` string (p2rank convention).

    Parameters
    ----------
    csv_path : path-like
        Path to the ``*_residues.csv`` file written by p2rank.

    Returns
    -------
    list[P2RankResult]
        One entry per pocket, sorted by rank (rank 1 first).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    out: list = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            residues_raw = row.get("residues_list", "") or ""
            residues: list = []
            if residues_raw.strip():
                # Format: "A:25;A:26;A:27;..."  (chain:resi;chain:resi)
                for token in residues_raw.split(";"):
                    token = token.strip()
                    if not token:
                        continue
                    if ":" in token:
                        chain, resi_str = token.split(":", 1)
                        try:
                            residues.append((chain.strip(), int(resi_str)))
                        except ValueError:
                            continue
                    else:
                        # Fallback: bare residue number, chain unknown
                        try:
                            residues.append(("", int(token)))
                        except ValueError:
                            continue
            out.append(
                P2RankResult(
                    rank=_coerce_int(row.get("rank", "0")),
                    score=_coerce_float(row.get("score", "0")),
                    probability=_coerce_float(row.get("probability", "0")),
                    chain=(row.get("chain") or "").strip(),
                    resi=_coerce_int(row.get("resi", "0")),
                    resn=(row.get("resn") or "").strip(),
                    residues=tuple(residues),
                )
            )

    out.sort(key=lambda r: r.rank)
    return out


# ----------------------------------------------------------------------
# PDB cropper (Biopython — already a project dep)
# ----------------------------------------------------------------------

def crop_pdb_to_residues(
    pdb_path: str | os.PathLike,
    residues: Sequence[Tuple[str, int]] | P2RankResult,
    out_path: str | os.PathLike,
    sphere_radius_A: float = 10.0,
) -> Path:
    """Crop a full PDB to a 10 Å sphere around the given residues.

    Parameters
    ----------
    pdb_path : path-like
        Input full PDB.
    residues : sequence of (chain, resi) pairs, OR a P2RankResult
        Either an explicit list of residue selections, or the
        ``P2RankResult.residues`` field.
    out_path : path-like
        Where to write the cropped PDB.
    sphere_radius_A : float
        Crop radius in Angstrom around the residue centres. Default
        10 Å matches the CrossDocked "pocket10" convention.

    Returns
    -------
    Path
        The output PDB path.

    Raises
    ------
    ImportError
        If Biopython is not installed (it is a project dep, so this
        should not happen in the standard env).
    """
    if isinstance(residues, P2RankResult):
        residues = list(residues.residues)
    residues = list(residues)

    try:
        from Bio.PDB import PDBParser, NeighborSearch
        from Bio.PDB.PDBIO import PDBIO
    except ImportError as exc:  # pragma: no cover — Biopython is a project dep
        raise ImportError(
            "Biopython is required for crop_pdb_to_residues. It is a project "
            "dep but may be missing in your env. `uv add biopython`."
        ) from exc

    import numpy as np  # noqa: F401  — used by Bio.PDB internally

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("full", str(pdb_path))
    model = next(structure.get_models())

    # Gather the atoms of the seed residues, then expand by sphere.
    seed_atoms: list = []
    seed_residue_set: set = set()
    for chain_id, resi in residues:
        chain_obj = model[chain_id] if chain_id in [c.id for c in model] else None
        if chain_obj is None:
            continue
        for residue in chain_obj:
            if residue.id[1] == resi:
                seed_atoms.extend(list(residue.get_atoms()))
                seed_residue_set.add((chain_id, resi))
                break

    if not seed_atoms:
        # Nothing to crop — write an empty structure.
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # We have nothing to write. Caller should treat this as a
        # missing-pocket signal; raise to make the gap explicit.
        raise ValueError(
            f"crop_pdb_to_residues: no seed atoms matched in {pdb_path} "
            f"for residues={residues[:3]}..."
        )

    # Find all atoms within sphere_radius_A of any seed atom.
    all_atoms = list(model.get_atoms())
    sphere_atom_set: set = set()
    for seed_atom in seed_atoms:
        ns = NeighborSearch(all_atoms)
        nearby = ns.search(
            center=seed_atom.get_vector().get_array(),
            radius=sphere_radius_A,
            level="A",  # atom level
        )
        sphere_atom_set.update(nearby)

    # Build a new structure containing only the sphere atoms.
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model as PDBModel
    from Bio.PDB.Chain import Chain
    cropped = Structure("cropped")
    new_model = PDBModel(0)
    cropped.add(new_model)

    by_chain: dict = {}
    for atom in sphere_atom_set:
        parent = atom.get_parent()
        chain_id = parent.get_parent().id
        resi_key = parent.id
        by_chain.setdefault(chain_id, {}).setdefault(resi_key, set()).add(atom)

    for chain_id, resi_map in by_chain.items():
        new_chain = Chain(chain_id)
        new_model.add(new_chain)
        for resi_key, atom_set in resi_map.items():
            # Clone the residue then keep only the selected atoms.
            parent_residue = None
            for residue in model[chain_id]:
                if residue.id == resi_key:
                    parent_residue = residue
                    break
            if parent_residue is None:
                continue
            new_residue = parent_residue.copy()
            # `__all_atoms__` was an internal child in older Biopython;
            # newer versions don't ship it. Guard the detach so we work on
            # both. (Fix per Wf-A33 v2: fpocket_wrapper reuses this.)
            try:
                new_residue.detach_child("__all_atoms__")
            except KeyError:
                pass
            for child in list(new_residue.get_atoms()):
                if child not in atom_set:
                    new_residue.detach_child(child.get_id())
            new_chain.add(new_residue)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(cropped)
    io.save(str(out_path))
    return out_path


# ----------------------------------------------------------------------
# Convenience: top-N pockets
# ----------------------------------------------------------------------

def top_pockets(
    csv_path: str | os.PathLike,
    n: int = 1,
    min_probability: float = 0.0,
) -> List[P2RankResult]:
    """Return the top-N pockets from a parsed p2rank CSV, filtered by min prob.

    Parameters
    ----------
    csv_path : path-like
        p2rank ``*_residues.csv`` file.
    n : int
        Number of pockets to return (default 1 = top pocket).
    min_probability : float
        Drop pockets with probability < this threshold (default 0 = keep
        all). p2rank probabilities are in [0, 1].

    Returns
    -------
    list[P2RankResult]
        Up to N pockets, sorted by rank.
    """
    results = parse_p2rank_csv(csv_path)
    results = [r for r in results if r.probability >= min_probability]
    return results[:n]


# ----------------------------------------------------------------------
# Self-test (smoke)
# ----------------------------------------------------------------------

def _smoke() -> int:
    """Smoke test: parser + availability + module imports."""
    # 1. is_p2rank_available is side-effect-free.
    avail = is_p2rank_available()
    print(f"[p2rank-wrapper] is_p2rank_available: {avail}")

    # 2. parse_p2rank_csv on a non-existent file returns [] (not raise).
    result = parse_p2rank_csv("/tmp/_no_such_p2rank.csv")
    if result != []:
        print(f"[p2rank-wrapper] FAIL: missing-file parse returned {result}")
        return 1
    print("[p2rank-wrapper] missing-file parse returns [] OK")

    # 3. Inline parse on a synthetic CSV.
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "rank,score,label,chain,resi,resn,probability,residues_list\n"
            "1,5.42,pocket1,A,123,SER,0.91,A:120;A:121;A:122;A:123;A:124\n"
            "2,3.10,pocket2,B,55,GLY,0.78,B:50;B:51;B:52;B:53;B:54;B:55\n"
        )
        tmp_name = f.name
    try:
        results = parse_p2rank_csv(tmp_name)
        if len(results) != 2:
            print(f"[p2rank-wrapper] FAIL: expected 2 pockets, got {len(results)}")
            return 1
        if results[0].rank != 1 or results[0].chain != "A" or results[0].resi != 123:
            print(f"[p2rank-wrapper] FAIL: top pocket = {results[0]}")
            return 1
        if len(results[0].residues) != 5:
            print(f"[p2rank-wrapper] FAIL: top pocket residues len = {len(results[0].residues)}")
            return 1
        if results[1].chain != "B" or results[1].resn != "GLY":
            print(f"[p2rank-wrapper] FAIL: 2nd pocket = {results[1]}")
            return 1
        print(f"[p2rank-wrapper] synthetic parse OK (2 pockets, {len(results[0].residues)}+{len(results[1].residues)} residues)")

        # top_pockets helper
        top1 = top_pockets(tmp_name, n=1)
        if len(top1) != 1 or top1[0].rank != 1:
            print(f"[p2rank-wrapper] FAIL: top_pockets = {top1}")
            return 1
        top1_filtered = top_pockets(tmp_name, n=1, min_probability=0.85)
        if len(top1_filtered) != 1:
            print(f"[p2rank-wrapper] FAIL: min_probability=0.85 should keep 1, got {len(top1_filtered)}")
            return 1
        top1_strict = top_pockets(tmp_name, n=1, min_probability=0.99)
        if len(top1_strict) != 0:
            print(f"[p2rank-wrapper] FAIL: min_probability=0.99 should drop all, got {len(top1_strict)}")
            return 1
        print("[p2rank-wrapper] top_pockets OK (filter + n respected)")
    finally:
        os.unlink(tmp_name)

    # 4. crop_pdb_to_residues with no Biopython parser call — just the
    # imports — to confirm Biopython is available.
    try:
        from Bio.PDB import PDBParser  # noqa: F401
        print("[p2rank-wrapper] Biopython import OK")
    except ImportError as exc:
        print(f"[p2rank-wrapper] FAIL: Biopython missing: {exc}")
        return 1

    # 5. run_p2rank raises NotImplementedError (Wf-A33 wire-in target).
    try:
        run_p2rank("/tmp/_no_such.pdb", "/tmp/_no_such_out")
    except NotImplementedError as exc:
        print(f"[p2rank-wrapper] run_p2rank raises NotImplementedError as expected: {str(exc)[:60]}...")
    except Exception as exc:
        print(f"[p2rank-wrapper] FAIL: run_p2rank raised wrong exception {type(exc).__name__}: {exc}")
        return 1

    print("[p2rank-wrapper] PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
