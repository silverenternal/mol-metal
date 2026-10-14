"""molmetal/adapters/fpocket_wrapper.py - fpocket detection (Wf-A33 v2, no Java).

Upstream: fpocket 4.2.3 (Le Guilloux et al. 2009, J. Mol. Graph. Model., MIT).
https://github.com/Discngine/fpocket

Why fpocket over P2Rank (no Java):
  - Pure C binary, no Java dep
  - MIT license (vs P2Rank MIT - same, but fpocket is more battle-tested in industry)
  - Output format simpler: <basename>_out/pockets/pocketN_atm.pdb + <basename>_out/<basename>_info.txt
  - Pre-installed on host: /home/hugo/.mamba/pkgs/fpocket-4.2.3-h8d0616a_1/bin/fpocket

Strategy A: center-residue 10Å sphere.
  - For each pocket, find the residue closest to the Voronoi center (centroid of pocketN_atm.pdb).
  - Reuse p2rank_wrapper.crop_pdb_to_residues(pdb_path, [(chain, resi)], out_path, sphere_radius_A=10.0) for the crop.

References
----------
- fpocket paper: Le Guilloux, Schmidtke & Tuffery, J Mol Graph Model 27 (2009) 978-987.
  DOI: 10.1016/j.jmgm.2009.01.003
- fpocket GitHub: https://github.com/Discngine/fpocket
- License: MIT (https://github.com/Discngine/fpocket/blob/master/LICENSE)

Usage
-----
    from molmetal.adapters.fpocket_wrapper import (
        is_fpocket_available, run_fpocket, parse_fpocket_info,
        parse_pocket_atoms, compute_pocket_center, find_center_residue,
        crop_pocket_to_sphere, extract_pocket,
    )

    if is_fpocket_available():
        info_path = run_fpocket("input.pdb", "out_dir")
        pockets = parse_fpocket_info(info_path)
        atoms = parse_pocket_atoms("out_dir/pockets/pocket1_atm.pdb")
        center = compute_pocket_center("out_dir/pockets/pocket1_atm.pdb")
        crop_pocket_to_sphere("input.pdb", ("A", 335), "pocket.pdb")  # reuses p2rank_wrapper
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file


# ----------------------------------------------------------------------
# Path constants
# ----------------------------------------------------------------------

#: Default fpocket command name. Override via ``MOLMETAL_FPOCKET_CMD``.
FPCKT_CMD_DEFAULT = "fpocket"

#: Where to look for fpocket when ``MOLMETAL_FPOCKET_CMD`` is not set.
FPCKT_CANDIDATE_PATHS = (
    "/home/hugo/.mamba/pkgs/fpocket-4.2.3-h8d0616a_1/bin/fpocket",  # host conda
    "/usr/local/bin/fpocket",
    "/opt/fpocket/fpocket",
)


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PocketResult:
    """One pocket detected by fpocket.

    Attributes
    ----------
    pocket_id : int
        1-based pocket ID (matches ``pocketN_atm.pdb`` filename).
    score : float
        fpocket Drug Score (higher = more druggable, [0, 1]).
    drugability : float
        fpocket Druggability Score (higher = more druggable, [0, 1]).
    n_alpha : int
        Number of alpha spheres in the pocket.
    volume : float
        Pocket volume (Monte Carlo estimate, Å³).
    """

    pocket_id: int
    score: float
    drugability: float
    n_alpha: int
    volume: float = 0.0

    @property
    def label(self) -> str:
        return f"pocket{self.pocket_id}"


# ----------------------------------------------------------------------
# CLI availability
# ----------------------------------------------------------------------

def _resolve_fpocket_cmd() -> Optional[str]:
    """Return the fpocket command path if it can be located, else None.

    Resolution order (per AGENTS.md §5.5 filesystem-truth):
      1. ``MOLMETAL_FPOCKET_CMD`` env var (if set and runnable)
      2. ``which fpocket`` (PATH lookup)
      3. ``FPCKT_CANDIDATE_PATHS`` (well-known install locations)
    """
    env_cmd = os.environ.get("MOLMETAL_FPOCKET_CMD")
    if env_cmd and os.access(env_cmd, os.X_OK):
        return env_cmd
    which = shutil.which(FPCKT_CMD_DEFAULT)
    if which:
        return which
    for cand in FPCKT_CANDIDATE_PATHS:
        if os.access(cand, os.X_OK):
            return cand
    return None


def is_fpocket_available() -> bool:
    """Return True iff the fpocket CLI is reachable on this host.

    Cheap, side-effect-free check. Use this to gate UI prompts / early
    returns before the heavy crop path.
    """
    return _resolve_fpocket_cmd() is not None


# ----------------------------------------------------------------------
# Subprocess driver
# ----------------------------------------------------------------------

def run_fpocket(
    pdb_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    fpocket_cmd: Optional[str] = None,
    timeout_s: int = 600,
) -> Path:
    """Invoke fpocket on a single PDB file.

    Parameters
    ----------
    pdb_path : path-like
        Input PDB file (full protein structure).
    out_dir : path-like
        Output directory. fpocket writes ``<basename>_out/pockets/`` here
        plus ``<out_dir>/<basename>_out/<basename>_info.txt``.
    fpocket_cmd : str | None
        Override the fpocket executable. If None, falls back to
        ``_resolve_fpocket_cmd()``.
    timeout_s : int
        Subprocess timeout in seconds. Default 10 min.

    Returns
    -------
    Path
        Path to the per-pocket info file (``<out_dir>/<basename>_out/<basename>_info.txt``).

    Raises
    ------
    FileNotFoundError
        If fpocket is not installed on this host.
    subprocess.CalledProcessError
        If fpocket exits with non-zero status.
    """
    cmd = fpocket_cmd or _resolve_fpocket_cmd()
    if not cmd:
        raise FileNotFoundError(
            "fpocket is not installed on this host. Install via conda "
            "(conda install -c conda-forge fpocket) or download from "
            "https://github.com/Discngine/fpocket and set MOLMETAL_FPOCKET_CMD."
        )

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fpocket CLI: fpocket --file <pdb>
    # It writes <basename>_out/ to the directory of the input file (not cwd).
    # Copy the PDB into out_dir so the fpocket output lands there too.
    basename = pdb_path.stem
    work_pdb = out_dir / pdb_path.name
    if work_pdb.resolve() != pdb_path.resolve():
        # §5.6 atomic_write: copy via atomic_write_file (read source, write dest).
        atomic_write_file(work_pdb, pdb_path.read_bytes(), binary=True)

    cmd_list = [cmd, "--file", work_pdb.name]
    try:
        subprocess.run(
            cmd_list,
            cwd=str(out_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.CalledProcessError as exc:
        raise subprocess.CalledProcessError(
            exc.returncode, exc.cmd, output=exc.output, stderr=exc.stderr,
        ) from exc

    info_path = out_dir / f"{basename}_out" / f"{basename}_info.txt"
    if not info_path.exists():
        raise FileNotFoundError(
            f"fpocket completed but {info_path} not found. Check fpocket output in {out_dir}."
        )
    return info_path


# ----------------------------------------------------------------------
# Pure-Python parser for fpocket _info.txt
# ----------------------------------------------------------------------

#: Map of fpocket info.txt field labels to PocketResult attribute names.
_INFO_FIELD_MAP = {
    "Score": "score",
    "Druggability Score": "drugability",
    "Number of Alpha Spheres": "n_alpha",
    "Volume": "volume",
}

#: Regex matching "Pocket N :" header lines.
_POCKET_HEADER_RE = re.compile(r"^\s*Pocket\s+(\d+)\s*:\s*$")


def _coerce_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s.strip())
    except (TypeError, ValueError):
        return default


def _coerce_int(s: str, default: int = 0) -> int:
    try:
        return int(s.strip())
    except (TypeError, ValueError):
        return default


def parse_fpocket_info(info_path: str | os.PathLike) -> List[PocketResult]:
    """Parse an fpocket ``*_info.txt`` file into a list of PocketResult.

    The fpocket info file format is section-based, e.g.::

        Pocket 1 :
        	Score : 	0.353
        	Druggability Score : 	0.725
        	Number of Alpha Spheres : 	141
        	...
        Pocket 2 :
        	...

    Parameters
    ----------
    info_path : path-like
        Path to the ``<basename>_info.txt`` file written by fpocket.

    Returns
    -------
    list[PocketResult]
        One entry per pocket, sorted by pocket_id (1 first).
    """
    info_path = Path(info_path)
    if not info_path.exists():
        return []

    out: list = []
    current_id: Optional[int] = None
    current_fields: dict = {}

    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            # Match "Pocket N :" header
            m = _POCKET_HEADER_RE.match(line)
            if m:
                # Flush previous pocket
                if current_id is not None:
                    out.append(_build_pocket_result(current_id, current_fields))
                current_id = int(m.group(1))
                current_fields = {}
                continue

            if current_id is None:
                # Skip pre-header lines (file preamble)
                continue

            # Match "Label : value" lines (tab-separated)
            # The colon is the separator; the label is before, the value is after.
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            label = label.strip()
            value = value.strip()
            if not value:
                continue

            if label in _INFO_FIELD_MAP:
                attr = _INFO_FIELD_MAP[label]
                if attr in ("score", "drugability", "volume"):
                    current_fields[attr] = _coerce_float(value)
                elif attr == "n_alpha":
                    current_fields[attr] = _coerce_int(value)

    # Flush final pocket
    if current_id is not None:
        out.append(_build_pocket_result(current_id, current_fields))

    out.sort(key=lambda p: p.pocket_id)
    return out


def _build_pocket_result(pocket_id: int, fields: dict) -> PocketResult:
    """Construct a PocketResult from accumulated fields."""
    return PocketResult(
        pocket_id=pocket_id,
        score=fields.get("score", 0.0),
        drugability=fields.get("drugability", 0.0),
        n_alpha=fields.get("n_alpha", 0),
        volume=fields.get("volume", 0.0),
    )


# ----------------------------------------------------------------------
# Pocket atom extraction + center
# ----------------------------------------------------------------------

def parse_pocket_atoms(atm_pdb_path: str | os.PathLike) -> List[Tuple[str, int]]:
    """Extract unique ``(chain, resi)`` residue identifiers from a pocketN_atm.pdb.

    Parameters
    ----------
    atm_pdb_path : path-like
        fpocket's ``pocketN_atm.pdb`` file (atoms contacting Voronoi vertices).

    Returns
    -------
    list[(chain, resi)]
        Unique residues in the pocket, sorted first by chain then by resi.
    """
    atm_pdb_path = Path(atm_pdb_path)
    if not atm_pdb_path.exists():
        return []

    residues: set = set()
    with open(atm_pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            # Standard PDB ATOM/HETATM record.
            # Columns: 1-6 record, 7-11 serial, 13-16 atom name, 17 altLoc,
            # 18-20 resName, 22 chainID, 23-26 resSeq, 31-38 x, 39-46 y, 47-54 z.
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                chain_id = line[21:22].strip() or "A"
                resi = int(line[22:26].strip())
            except (ValueError, IndexError):
                continue
            residues.add((chain_id, resi))

    return sorted(residues)


def compute_pocket_center(atm_pdb_path: str | os.PathLike) -> Tuple[float, float, float]:
    """Compute the centroid (x, y, z) of all atoms in a pocketN_atm.pdb file.

    Parameters
    ----------
    atm_pdb_path : path-like
        fpocket's ``pocketN_atm.pdb`` file.

    Returns
    -------
    (x, y, z) : tuple of float
        Mean coordinate across all atoms. Returns (0.0, 0.0, 0.0) if
        the file has no parseable atom lines.
    """
    atm_pdb_path = Path(atm_pdb_path)
    if not atm_pdb_path.exists():
        return (0.0, 0.0, 0.0)

    coords: list = []
    with open(atm_pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append((x, y, z))
            except (ValueError, IndexError):
                continue

    if not coords:
        return (0.0, 0.0, 0.0)

    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    return (cx, cy, cz)


def find_center_residue(
    atm_pdb_path: str | os.PathLike,
    pdb_path: str | os.PathLike,
) -> Tuple[str, int]:
    """Find the residue (chain, resi) closest to the pocket centroid.

    Loads the full PDB to find candidate residues, then picks the residue
    whose centroid is closest to the pocket's atom centroid.

    Parameters
    ----------
    atm_pdb_path : path-like
        fpocket's ``pocketN_atm.pdb`` file (atoms contacting the pocket).
    pdb_path : path-like
        Original full PDB.

    Returns
    -------
    (chain, resi) : tuple
        The closest residue to the pocket centroid. Raises ValueError if
        no pocket atoms are found.
    """
    atm_pdb_path = Path(atm_pdb_path)
    pdb_path = Path(pdb_path)

    pocket_residues = parse_pocket_atoms(atm_pdb_path)
    if not pocket_residues:
        raise ValueError(
            f"find_center_residue: no pocket atoms found in {atm_pdb_path}"
        )

    center = compute_pocket_center(atm_pdb_path)

    try:
        from Bio.PDB import PDBParser
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for find_center_residue. It is a project "
            "dep but may be missing in your env. `uv add biopython`."
        ) from exc

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("full", str(pdb_path))
    model = next(structure.get_models())

    # Compute centroid for each candidate residue, pick the closest to pocket center.
    best_residue: Optional[Tuple[str, int]] = None
    best_dist_sq: float = float("inf")

    for chain_id, resi in pocket_residues:
        chain_obj = None
        for c in model:
            if c.id == chain_id:
                chain_obj = c
                break
        if chain_obj is None:
            continue
        for residue in chain_obj:
            if residue.id[1] != resi:
                continue
            atoms = list(residue.get_atoms())
            if not atoms:
                continue
            rcx = sum(a.get_vector().get_array()[0] for a in atoms) / len(atoms)
            rcy = sum(a.get_vector().get_array()[1] for a in atoms) / len(atoms)
            rcz = sum(a.get_vector().get_array()[2] for a in atoms) / len(atoms)
            d_sq = (rcx - center[0]) ** 2 + (rcy - center[1]) ** 2 + (rcz - center[2]) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_residue = (chain_id, resi)
            break

    if best_residue is None:
        raise ValueError(
            f"find_center_residue: no candidate residue found in {pdb_path} "
            f"for pocket residues {pocket_residues[:3]}..."
        )
    return best_residue


# ----------------------------------------------------------------------
# Crop to 10Å sphere (REUSES p2rank_wrapper.crop_pdb_to_residues)
# ----------------------------------------------------------------------

def crop_pocket_to_sphere(
    pdb_path: str | os.PathLike,
    center_residue: Tuple[str, int],
    out_path: str | os.PathLike,
    sphere_radius_A: float = 10.0,
) -> Path:
    """Crop a full PDB to a sphere around the given (chain, resi) center residue.

    This delegates to ``p2rank_wrapper.crop_pdb_to_residues`` (Biopython
    NeighborSearch). Reuse keeps a single canonical crop implementation
    across the Wf-A33 detector stack (fpocket + p2rank).

    Parameters
    ----------
    pdb_path : path-like
        Input full PDB.
    center_residue : (chain, resi)
        Residue to center the sphere on.
    out_path : path-like
        Where to write the cropped PDB.
    sphere_radius_A : float
        Crop radius in Angstrom. Default 10 Å matches CrossDocked pocket10.

    Returns
    -------
    Path
        The output PDB path.
    """
    # Local import to avoid a hard dependency cycle / hidden cost.
    from molmetal.adapters.p2rank_wrapper import crop_pdb_to_residues
    return crop_pdb_to_residues(
        pdb_path,
        [center_residue],
        out_path,
        sphere_radius_A=sphere_radius_A,
    )


# ----------------------------------------------------------------------
# Convenience orchestrator
# ----------------------------------------------------------------------

def extract_pocket(
    pdb_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    sphere_radius_A: float = 10.0,
    fpocket_cmd: Optional[str] = None,
    timeout_s: int = 600,
) -> Path:
    """Run fpocket + crop the top pocket to a 10Å sphere.

    Convenience wrapper that orchestrates the full pipeline:
      1. Run fpocket on the PDB.
      2. Parse the info file for the top pocket.
      3. Find the center residue of the top pocket.
      4. Crop the full PDB to a sphere around that residue.

    Parameters
    ----------
    pdb_path : path-like
        Input full PDB.
    out_dir : path-like
        Output directory.
    sphere_radius_A : float
        Crop radius. Default 10 Å.
    fpocket_cmd : str | None
        Override fpocket executable.
    timeout_s : int
        fpocket subprocess timeout.

    Returns
    -------
    Path
        Path to the cropped pocket PDB (named ``<basename>_pocket1_crop.pdb``).
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info_path = run_fpocket(pdb_path, out_dir, fpocket_cmd=fpocket_cmd, timeout_s=timeout_s)
    pockets = parse_fpocket_info(info_path)
    if not pockets:
        raise ValueError(f"fpocket produced 0 pockets for {pdb_path}")

    # Top pocket = highest Drug Score (more druggable).
    top = max(pockets, key=lambda p: p.drugability)
    atm_pdb = out_dir / f"{pdb_path.stem}_out" / "pockets" / f"{top.label}_atm.pdb"
    if not atm_pdb.exists():
        raise FileNotFoundError(f"fpocket pocket atoms not found: {atm_pdb}")

    center_residue = find_center_residue(atm_pdb, pdb_path)
    out_pdb = out_dir / f"{pdb_path.stem}_{top.label}_crop.pdb"
    return crop_pocket_to_sphere(pdb_path, center_residue, out_pdb, sphere_radius_A=sphere_radius_A)


# ----------------------------------------------------------------------
# Self-test (smoke)
# ----------------------------------------------------------------------

def _smoke() -> int:
    """Smoke test: parser + availability + module imports."""
    # 1. is_fpocket_available is side-effect-free.
    avail = is_fpocket_available()
    print(f"[fpocket-wrapper] is_fpocket_available: {avail}")

    # 2. parse_fpocket_info on a non-existent file returns [] (not raise).
    result = parse_fpocket_info("/tmp/_no_such_fpocket_info.txt")
    if result != []:
        print(f"[fpocket-wrapper] FAIL: missing-file parse returned {result}")
        return 1
    print("[fpocket-wrapper] missing-file parse returns [] OK")

    # 3. parse_fpocket_info on a synthetic info.txt.
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_info.txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "Pocket 1 :\n"
            "\tScore : \t0.353\n"
            "\tDruggability Score : \t0.725\n"
            "\tNumber of Alpha Spheres : \t141\n"
            "\tVolume : \t747.435\n"
            "\n"
            "Pocket 2 :\n"
            "\tScore : \t0.344\n"
            "\tDruggability Score : \t0.872\n"
            "\tNumber of Alpha Spheres : \t51\n"
            "\tVolume : \t569.545\n"
        )
        tmp_name = f.name
    try:
        results = parse_fpocket_info(tmp_name)
        if len(results) != 2:
            print(f"[fpocket-wrapper] FAIL: expected 2 pockets, got {len(results)}")
            return 1
        if results[0].pocket_id != 1 or results[0].drugability < 0.7:
            print(f"[fpocket-wrapper] FAIL: top pocket = {results[0]}")
            return 1
        if results[1].n_alpha != 51:
            print(f"[fpocket-wrapper] FAIL: 2nd pocket n_alpha = {results[1].n_alpha}")
            return 1
        print(f"[fpocket-wrapper] synthetic parse OK (2 pockets)")
    finally:
        os.unlink(tmp_name)

    # 4. parse_pocket_atoms on a non-existent file returns [].
    atoms = parse_pocket_atoms("/tmp/_no_such_atm.pdb")
    if atoms != []:
        print(f"[fpocket-wrapper] FAIL: missing atm.pdb parse returned {atoms}")
        return 1
    print("[fpocket-wrapper] missing atm.pdb parse returns [] OK")

    # 5. compute_pocket_center on a non-existent file returns (0, 0, 0).
    center = compute_pocket_center("/tmp/_no_such_atm.pdb")
    if center != (0.0, 0.0, 0.0):
        print(f"[fpocket-wrapper] FAIL: missing center returned {center}")
        return 1
    print("[fpocket-wrapper] missing center returns (0,0,0) OK")

    # 6. crop_pocket_to_sphere wrapper imports OK.
    try:
        from molmetal.adapters.p2rank_wrapper import crop_pdb_to_residues  # noqa: F401
        print("[fpocket-wrapper] p2rank_wrapper.crop_pdb_to_residues import OK")
    except ImportError as exc:
        print(f"[fpocket-wrapper] FAIL: p2rank_wrapper import missing: {exc}")
        return 1

    print("[fpocket-wrapper] PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
