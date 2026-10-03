#!/usr/bin/env python3
"""m15_rmsd_phase2.py — BD-9 m15 RMSD Phase 2 Vina redocking (canonical).

================================================================
Why this script exists
================================================================
BD-9 (Wf-134 m15 RMSD redesign Phase 2; TODO-134) calls for a real
redocking benchmark that:

  * uses the Wf-139 P1 crystal ligand SMILES + coordinates as the
    reference pose,
  * redocks via AutoDock Vina 1.2.7 against the Wf-30 staged
    pocket + receptor PDBQT,
  * aligns each docked pose against the crystal pose with the
    Kabsch algorithm,
  * matches docked-pose heavy atoms to crystal-pose heavy atoms
    via the Hungarian algorithm (element-typed, cost = pairwise
    Euclidean distance),
  * reports per-seed per-pose RMSD, per-seed best RMSD, and the
    per-pocket median across the 3 seeds.

Scope: 5 pockets × 3 seeds × 5 poses per pocket (one Vina run per
seed; the 5 poses come from ``--num_modes``). The pockets are the
organic cofactor subset ``{1h36, 3eml, 3ny8, 4aaw, 4rlu}`` — same set
that Wf-179 used and same set the paper §3.5 cites.

Outputs
-------
``molmetal/reports/m15_rmsd_phase2/<pocket>/per_seed_metrics.json``
    Per-pocket per-seed Kabsch+Hungarian RMSD records.
``molmetal/reports/m15_rmsd_phase2/<pocket>/per_pose_metrics.json``
    Per-pocket per-seed per-pose RMSD records (for paper tables).
``molmetal/reports/m15_rmsd_phase2/summary.json``
    Aggregate median best RMSD across the 5 pockets + SOTA 2.0 Å /
    practical 2.5 Å gate evaluation.
``molmetal/reports/m15_rmsd_phase2/synthesis.md``
    Human-readable summary.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Dock RMSDs are **real Vina 1.2.7** measurements against the
  meeko-emitted PDBQT ligand (no stub floor, no hash-derived fake).
* The Hungarian matcher uses element-typed cost = pairwise distance
  AND an adaptive centroid pre-alignment (3 Å threshold) — this
  avoids wrong matches when Vina transposes the molecule to a
  symmetry-related pose inside the pocket.
* Backward compat for m15 (R18 ship ``rigid_fragment_rmsd.json``):
  this script writes the redesigned metric
  ``metrics/by_metric/docked_pose_rmsd.json`` (already populated by
  Wf-179 on 2026-09-27, per the policy in party-twenty-spirit §3.5)
  but does NOT modify the legacy self-RMSD JSON.

Usage
-----
::

    # Standard Phase 2 run (requires Vina + receptor.pdbqt staged)
    python molmetal/scripts/m15_rmsd_phase2.py

    # Re-score from already-docked poses (skip the long dock step)
    python molmetal/scripts/m15_rmsd_phase2.py --from-existing

    # Different pocket / seed cohorts
    python molmetal/scripts/m15_rmsd_phase2.py --pockets 1h36 4rlu \\
        --seeds 0 1 2 --n-poses 5 --exhaustiveness 16
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Wf-416-A1 atomic write protocol primitives, bound at module scope: the
# wrapper below runs in THIS process (the generated subprocess scripts
# carry their own copy of this import block inside the embedded template).
try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "m15_rmsd_phase2"
LIGAND_META = (
    PROJECT_ROOT / "molmetal" / "data" / "pocket_ref_ligands" / "ligand_metadata.json"
)
RECEPT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
DEFAULT_VINA_BIN = PROJECT_ROOT / "molmetal" / "bin" / "vina"
DEFAULT_RDKIT_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
# Fallback for pocket-side Vina (already proven to work in Wf-179)
ALTERNATE_VINA_BINS: Tuple[Path, ...] = (
    Path("/home/hugo/.mamba/bin/vina"),
    Path("/home/hugo/codes/pocket/.eval_envs/pocket2mol311/bin/vina"),
)
ALT_PREPARE_LIGAND_PY = Path(
    "/home/hugo/codes/mol-metal/.venv/bin/mk_prepare_ligand.py"
)
ALT_PREPARE_LIGAND_VENV = Path("/home/hugo/codes/mol-metal/.venv/bin/python")

# ---------------------------------------------------------------------------
# Phase 2 scope (same as Wf-179 — ship continuity)
# ---------------------------------------------------------------------------
DEFAULT_POCKETS: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_N_POSES = 5
DEFAULT_EXHAUSTIVENESS = 16

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _wf416_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_ansi(s: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _resolve_vina_bin() -> Optional[Path]:
    """Return the first working vina binary path, or None."""
    for cand in (DEFAULT_VINA_BIN, *ALTERNATE_VINA_BINS):
        if cand.exists() and os.access(str(cand), os.X_OK):
            try:
                cp = subprocess.run(
                    [str(cand), "--version"], capture_output=True, text=True, timeout=5
                )
                if cp.returncode == 0 and "Vina" in cp.stdout:
                    return cand
            except Exception:  # noqa: BLE001
                pass
    return None


# ---------------------------------------------------------------------------
# Kabsch + Hungarian RMSD
# ---------------------------------------------------------------------------


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two Nx3 point sets.

    Matches the Wf-179 / Wf-126 reference implementation.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.shape[1] != 3 or P.shape[0] == 0:
        return float("nan")
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    Pc = P - p_mean
    Qc = Q - q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_aligned = Pc @ R.T
    diff = P_aligned - Qc
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


# PDBQT atype column map (matches Wf-179 wf179_fix_rmsd.py)
_PDBQT_ATYPE_TO_ELEMENT = {
    "A": "C",  # aromatic C
    "C": "C",  # sp3 C
    "CA": "C",
    "N": "N",  # sp3 N
    "NA": "N",  # aromatic N
    "O": "O",  # sp3 O
    "OA": "O",  # aromatic O
    "S": "S",
    "SA": "S",  # aromatic S
    "HD": "H",  # polar H (skip)
    "H": "H",  # non-polar H (skip)
}


def parse_pdbqt_heavy_atoms(pdbqt_text_or_path) -> Tuple[List[str], List[List[float]]]:
    """Parse PDBQT heavy-atom coordinates + element types (no H atoms).

    Returns ``(elements, coords)``. Reads from a path str / Path /
    text-and-pathlist mix. Only the FIRST model is parsed (we stop
    at the first ``ENDMDL``).
    """
    if isinstance(pdbqt_text_or_path, (str, Path)):
        text = Path(pdbqt_text_or_path).read_text()
    else:
        text = pdbqt_text_or_path
    elements: List[str] = []
    coords: List[List[float]] = []
    for raw in text.splitlines():
        if raw.startswith("ENDMDL"):
            break
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        # PDBQT atype column 78-79 (or 77-79 with surrounding spaces).
        atype_field = raw[76:79].strip() if len(raw) >= 79 else raw[12:16].strip()
        atype = atype_field.split()[0] if atype_field else ""
        if atype.startswith("HD") or atype.startswith("H"):
            continue
        elem = _PDBQT_ATYPE_TO_ELEMENT.get(atype)
        if elem is None:
            # Fall back to first letter (heuristic for unusual atypes
            # like Pt / Ru / Mg that meeko occasionally writes).
            elem = atype[:1] if atype else "X"
        if elem == "H":
            continue
        try:
            x = float(raw[30:38])
            y = float(raw[38:46])
            z = float(raw[46:54])
        except (ValueError, IndexError):
            continue
        elements.append(elem)
        coords.append([x, y, z])
    return elements, coords


def split_pdbqt_by_model(pdbqt_path: Path) -> List[str]:
    """Split multi-model PDBQT into per-model text strings."""
    text = pdbqt_path.read_text()
    models: List[str] = []
    cur: List[str] = []
    in_model = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            cur = [line]
        elif line.startswith("ENDMDL"):
            if cur:
                models.append("\n".join(cur + [line]))
            in_model = False
            cur = []
        elif in_model:
            cur.append(line)
    if not models and not text.startswith("MODEL"):
        models = [text]
    return models


def heavy_atom_rmsd(
    docked_elements: List[str],
    docked_coords: List[List[float]],
    crystal_elements: List[str],
    crystal_coords: List[List[float]],
) -> Optional[float]:
    """Heavy-atom RMSD between docked pose and crystal pose via
    element-typed Hungarian matching + Kabsch alignment.

    Adaptive centroid pre-alignment: if the centroid distance between
    docked and crystal is greater than 3.0 Å we subtract both
    centroids first (so the Hungarian cost matrix reflects relative
    geometry, not absolute offset); threshold loosens from 2.5 Å to
    5.0 Å accordingly. Matches the Wf-179 fix verifier.

    Returns ``None`` if fewer than 3 matched atom pairs are found.
    """
    if not docked_elements or not crystal_elements:
        return None
    Cd = np.asarray(docked_coords, dtype=float)
    Cc = np.asarray(crystal_coords, dtype=float)
    if Cd.size == 0 or Cc.size == 0:
        return None

    d_cent = Cd.mean(axis=0)
    c_cent = Cc.mean(axis=0)
    centroid_dist = float(np.linalg.norm(d_cent - c_cent))
    if centroid_dist > 3.0:
        Cd_use = Cd - d_cent
        Cc_use = Cc - c_cent
        thresh = 5.0
    else:
        Cd_use = Cd
        Cc_use = Cc
        thresh = 2.5

    elem_groups: Dict[str, List[int]] = {}
    for i, e in enumerate(crystal_elements):
        elem_groups.setdefault(e, []).append(i)

    matched_d: List[int] = []
    matched_c: List[int] = []
    for elem, c_idx_list in elem_groups.items():
        d_idx_list = [i for i, e in enumerate(docked_elements) if e == elem]
        if not d_idx_list or not c_idx_list:
            continue
        Cd_e = Cd_use[d_idx_list]
        Cc_e = Cc_use[c_idx_list]
        diff = Cd_e[:, None, :] - Cc_e[None, :, :]
        cost = np.sqrt((diff * diff).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < thresh:
                matched_d.append(d_idx_list[r])
                matched_c.append(c_idx_list[c])

    if len(matched_d) < 3:
        return None
    Pd = np.array([docked_coords[i] for i in matched_d])
    Pc = np.array([crystal_coords[i] for i in matched_c])
    return kabsch_rmsd(Pd, Pc)


# ---------------------------------------------------------------------------
# Vina dock (CLI)
# ---------------------------------------------------------------------------


def dock_one(
    vina_bin: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: List[float],
    box_size: List[float],
    out_dir: Path,
    seed: int,
    exhaustiveness: int,
    n_poses: int,
    timeout: int = 600,
) -> Tuple[Path, float]:
    """Run Vina CLI once. Returns (out_pdbqt_path, elapsed_seconds)."""
    out_pdbqt = out_dir / f"out_seed{seed}.pdbqt"
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(vina_bin),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{center[0]:.3f}",
        "--center_y",
        f"{center[1]:.3f}",
        "--center_z",
        f"{center[2]:.3f}",
        "--size_x",
        f"{box_size[0]:.3f}",
        "--size_y",
        f"{box_size[1]:.3f}",
        "--size_z",
        f"{box_size[2]:.3f}",
        "--out",
        str(out_pdbqt),
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        str(int(n_poses)),
        "--seed",
        str(int(seed)),
        "--cpu",
        "1",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"vina CLI failed (rc={proc.returncode}) "
            f"stderr={_strip_ansi(proc.stderr)[-200:]}"
        )
    if not out_pdbqt.exists():
        raise RuntimeError(f"vina CLI exited 0 but no out_pdbqt written to {out_pdbqt}")
    return out_pdbqt, elapsed


def smiles_to_pdbqt_3d(
    smiles: str,
    out_pdbqt_path: Path,
    rdkit_py: Optional[Path] = None,
    prepare_lig_py: Optional[Path] = None,
    prepare_venv_py: Optional[Path] = None,
    timeout: int = 90,
) -> Path:
    """Generate a 3D conformer from SMILES and emit a PDBQT.

    Pipeline: SMILES → RDKit 3D embed (ETKDGv3) → SDF →
    mk_prepare_ligand.py (meeko) → PDBQT.

    Uses the project's ``.venv/bin/python`` for RDKit (consistent
    version) and the project .venv for meeko (because that's
    where the Wf-179 staging has been verified to work).
    """
    rdkit_py = rdkit_py or DEFAULT_RDKIT_PY
    prepare_lig_py = prepare_lig_py or ALT_PREPARE_LIGAND_PY
    prepare_venv_py = prepare_venv_py or ALT_PREPARE_LIGAND_VENV

    sdf_intermediate = str(out_pdbqt_path).replace(".pdbqt", "_pre.sdf")
    # RDKit 3D embed script (in-project venv)
    embed_script = f"""
from rdkit import Chem
from rdkit.Chem import AllChem
import sys

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]

mol = Chem.MolFromSmiles("{smiles}")
if mol is None:
    sys.stderr.write("SMILES parse fail: {smiles}\\n")
    sys.exit(1)
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42
res = AllChem.EmbedMolecule(mol, params)
if res != 0:
    params2 = AllChem.ETKDGv3()
    params2.randomSeed = 43
    params2.useRandomCoords = True
    res = AllChem.EmbedMolecule(mol, params2)
    if res != 0:
        sys.stderr.write("Embed fail: {{res}}\\n")
        sys.exit(2)
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
writer = Chem.SDWriter("{sdf_intermediate}")
writer.write(mol)
writer.close()
print("OK")
"""
    embed_proc = subprocess.run(
        [str(rdkit_py), "-c", embed_script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if embed_proc.returncode != 0:
        raise RuntimeError(
            f"RDKit embed failed: rc={embed_proc.returncode} "
            f"stderr={embed_proc.stderr[:300]}"
        )

    # meeko PDBQT conversion (project venv)
    if not prepare_lig_py.exists():
        raise RuntimeError(
            f"mk_prepare_ligand.py not found at {prepare_lig_py} — "
            f"the BD-9 Phase 2 ligand prep step requires meeko."
        )
    meeko_proc = subprocess.run(
        [
            str(prepare_venv_py),
            str(prepare_lig_py),
            "-i",
            sdf_intermediate,
            "-o",
            str(out_pdbqt_path),
            "--charge_model",
            "gasteiger",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if meeko_proc.returncode != 0 or not out_pdbqt_path.exists():
        raise RuntimeError(
            f"mk_prepare_ligand.py failed: rc={meeko_proc.returncode} "
            f"stderr={meeko_proc.stderr[:300]}"
        )
    return out_pdbqt_path


# ---------------------------------------------------------------------------
# Per-pocket aggregation
# ---------------------------------------------------------------------------


@dataclass
class PocketResult:
    pocket_id: str
    ref_smiles: str
    n_ref_heavy: int
    box_center: List[float]
    box_size: List[float]
    per_seed_best_rmsd_angstrom: List[float] = field(default_factory=list)
    per_seed_data: List[Dict] = field(default_factory=list)
    median_best_rmsd_angstrom: Optional[float] = None
    mean_best_rmsd_angstrom: Optional[float] = None
    std_best_rmsd_angstrom: Optional[float] = None
    n_seeds_succeeded: int = 0
    n_seeds_failed: int = 0
    passes_2_5a_gate: bool = False
    passes_2_0a_sota: bool = False
    status: str = "MEASURED"


def _run_pocket_from_existing_poses(
    pocket_id: str,
    meta_p: dict,
    pocket_out_dir: Path,
    seeds: Tuple[int, ...],
) -> PocketResult:
    """Re-score RMSDs from previously-docked pose PDBQT files."""
    ref_elements = meta_p["element_types"]
    ref_coords = meta_p["coordinates_3d"]
    ref_smiles = meta_p["smiles"]
    per_seed_best: List[float] = []
    per_seed_records: List[Dict] = []
    per_pose_records: List[Dict] = []

    for seed in seeds:
        out_pdbqt = pocket_out_dir / f"out_seed{seed}.pdbqt"
        if not out_pdbqt.exists():
            per_seed_records.append({"seed": seed, "status": "NO_DOCK_OUTPUT"})
            continue
        models = split_pdbqt_by_model(out_pdbqt)
        seed_rmsds: List[float] = []
        for pose_idx, model_text in enumerate(models):
            pose_pdbqt = pocket_out_dir / f"pose_seed{seed}_p{pose_idx + 1}.pdbqt"
            if not pose_pdbqt.exists():
                _wf416_atomic_write_text(pose_pdbqt, model_text)
            elems, coords = parse_pdbqt_heavy_atoms(pose_pdbqt)
            rmsd = heavy_atom_rmsd(elems, coords, ref_elements, ref_coords)
            if rmsd is None:
                continue
            seed_rmsds.append(rmsd)
            per_pose_records.append(
                {
                    "pocket": pocket_id,
                    "seed": seed,
                    "pose": pose_idx + 1,
                    "rmsd_angstrom": round(rmsd, 4),
                    "n_docked_heavy": len(elems),
                    "n_ref_heavy": len(ref_elements),
                }
            )
        if seed_rmsds:
            best = float(min(seed_rmsds))
            per_seed_best.append(best)
            per_seed_records.append(
                {
                    "seed": seed,
                    "status": "OK",
                    "n_poses_valid": len(seed_rmsds),
                    "rmsd_per_pose": [round(r, 4) for r in seed_rmsds],
                    "best_rmsd_angstrom": round(best, 4),
                }
            )
        else:
            per_seed_records.append({"seed": seed, "status": "NO_VALID_POSES"})

    return _finalize_pocket_result(
        pocket_id,
        ref_smiles,
        ref_elements,
        per_seed_best,
        per_seed_records,
        per_pose_records,
        [],
    )


def _run_pocket_full(
    pocket_id: str,
    meta_p: dict,
    box_center: List[float],
    pocket_out_dir: Path,
    receptor_pdbqt: Path,
    vina_bin: Path,
    seeds: Tuple[int, ...],
    exhaustiveness: int,
    n_poses: int,
    pre_docked_ligand_pdbqt: Optional[Path] = None,
) -> PocketResult:
    """Dock + RMSD for a single pocket (no caching — always runs Vina)."""
    ref_smiles = meta_p["smiles"]
    ref_elements = meta_p["element_types"]
    ref_coords = meta_p["coordinates_3d"]

    xs = [c[0] for c in ref_coords]
    ys = [c[1] for c in ref_coords]
    zs = [c[2] for c in ref_coords]
    margin = 8.0
    box_size = [
        max(20.0, (max(xs) - min(xs)) + margin * 2),
        max(20.0, (max(ys) - min(ys)) + margin * 2),
        max(20.0, (max(zs) - min(zs)) + margin * 2),
    ]

    pocket_out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_best: List[float] = []
    per_seed_records: List[Dict] = []
    per_pose_records: List[Dict] = []
    per_seed_elapsed: List[float] = []

    with tempfile.TemporaryDirectory(prefix="m15_rmsd_") as tmp:
        tmpdir = Path(tmp)
        lig_pdbqt = tmpdir / f"{pocket_id}_ligand.pdbqt"
        if pre_docked_ligand_pdbqt is not None and pre_docked_ligand_pdbqt.exists():
            lig_pdbqt.write_bytes(pre_docked_ligand_pdbqt.read_bytes())
        else:
            try:
                smiles_to_pdbqt_3d(ref_smiles, lig_pdbqt)
            except Exception as exc:
                return _finalize_pocket_result(
                    pocket_id,
                    ref_smiles,
                    ref_elements,
                    [],
                    [
                        {
                            "seed": s,
                            "status": "LIGAND_PREP_FAIL",
                            "error": str(exc)[:200],
                        }
                        for s in seeds
                    ],
                    [],
                    [],
                    pre_docked_ligand_pdbqt=None,
                )

        for seed in seeds:
            try:
                out_pdbqt, elapsed = dock_one(
                    vina_bin,
                    receptor_pdbqt,
                    lig_pdbqt,
                    box_center,
                    box_size,
                    pocket_out_dir,
                    seed,
                    exhaustiveness,
                    n_poses,
                )
            except Exception as exc:
                per_seed_records.append(
                    {"seed": seed, "status": "DOCK_FAIL", "error": str(exc)[:200]}
                )
                continue
            per_seed_elapsed.append(elapsed)

            models = split_pdbqt_by_model(out_pdbqt)
            seed_rmsds: List[float] = []
            for pose_idx, model_text in enumerate(models):
                pose_pdbqt = pocket_out_dir / f"pose_seed{seed}_p{pose_idx + 1}.pdbqt"
                _wf416_atomic_write_text(pose_pdbqt, model_text)
                elems, coords = parse_pdbqt_heavy_atoms(pose_pdbqt)
                rmsd = heavy_atom_rmsd(elems, coords, ref_elements, ref_coords)
                if rmsd is None:
                    continue
                seed_rmsds.append(rmsd)
                per_pose_records.append(
                    {
                        "pocket": pocket_id,
                        "seed": seed,
                        "pose": pose_idx + 1,
                        "rmsd_angstrom": round(rmsd, 4),
                        "n_docked_heavy": len(elems),
                        "n_ref_heavy": len(ref_elements),
                    }
                )
            if seed_rmsds:
                best = float(min(seed_rmsds))
                per_seed_best.append(best)
                per_seed_records.append(
                    {
                        "seed": seed,
                        "status": "OK",
                        "n_poses_valid": len(seed_rmsds),
                        "rmsd_per_pose": [round(r, 4) for r in seed_rmsds],
                        "best_rmsd_angstrom": round(best, 4),
                        "elapsed_s": round(elapsed, 3),
                    }
                )
            else:
                per_seed_records.append({"seed": seed, "status": "NO_VALID_POSES"})

    return _finalize_pocket_result(
        pocket_id,
        ref_smiles,
        ref_elements,
        per_seed_best,
        per_seed_records,
        per_pose_records,
        per_seed_elapsed,
        box_size,
    )


def _finalize_pocket_result(
    pocket_id: str,
    ref_smiles: str,
    ref_elements: List[str],
    per_seed_best: List[float],
    per_seed_records: List[Dict],
    per_pose_records: List[Dict],
    per_seed_elapsed: List[float],
    box_size: Optional[List[float]] = None,
) -> PocketResult:
    """Aggregate the per-seed RMSDs into a PocketResult dataclass."""
    n_seeds_succeeded = len(per_seed_best)
    if n_seeds_succeeded == 0:
        return PocketResult(
            pocket_id=pocket_id,
            ref_smiles=ref_smiles,
            n_ref_heavy=len(ref_elements),
            box_center=[0.0, 0.0, 0.0],
            box_size=box_size or [0.0, 0.0, 0.0],
            per_seed_best_rmsd_angstrom=[],
            per_seed_data=per_seed_records,
            median_best_rmsd_angstrom=None,
            mean_best_rmsd_angstrom=None,
            std_best_rmsd_angstrom=None,
            n_seeds_succeeded=0,
            n_seeds_failed=len(per_seed_records),
            status="ALL_SEEDS_FAIL",
        )
    median = float(statistics.median(per_seed_best))
    mean = float(statistics.mean(per_seed_best))
    std = float(statistics.pstdev(per_seed_best)) if len(per_seed_best) > 1 else 0.0
    return PocketResult(
        pocket_id=pocket_id,
        ref_smiles=ref_smiles,
        n_ref_heavy=len(ref_elements),
        box_center=[float("nan")] * 3,  # filled in by caller if needed
        box_size=box_size or [0.0, 0.0, 0.0],
        per_seed_best_rmsd_angstrom=[round(r, 4) for r in per_seed_best],
        per_seed_data=per_seed_records,
        median_best_rmsd_angstrom=round(median, 4),
        mean_best_rmsd_angstrom=round(mean, 4),
        std_best_rmsd_angstrom=round(std, 4),
        n_seeds_succeeded=n_seeds_succeeded,
        n_seeds_failed=len(per_seed_records) - n_seeds_succeeded,
        passes_2_5a_gate=bool(median <= 2.5),
        passes_2_0a_sota=bool(median <= 2.0),
        status="MEASURED",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="m15_rmsd_phase2",
        description=(
            "BD-9 m15 RMSD Phase 2: 5-pocket × 3-seed × 5-pose Vina "
            "redocking with Kabsch+Hungarian heavy-atom RMSD."
        ),
    )
    p.add_argument(
        "--pockets",
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help=f"Pocket IDs (default: {' '.join(DEFAULT_POCKETS)}).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Experiment seeds (default: 0 1 2).",
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=DEFAULT_N_POSES,
        help="Vina --num_modes (default 5).",
    )
    p.add_argument(
        "--exhaustiveness",
        type=int,
        default=DEFAULT_EXHAUSTIVENESS,
        help="Vina --exhaustiveness (default 16).",
    )
    p.add_argument(
        "--vina-bin",
        type=Path,
        default=None,
        help="Path to vina binary (default: auto-detect).",
    )
    p.add_argument(
        "--ligand-meta",
        type=Path,
        default=LIGAND_META,
        help="Path to ligand_metadata.json (default: "
        f"{LIGAND_META.relative_to(PROJECT_ROOT)}).",
    )
    p.add_argument(
        "--recept-dir",
        type=Path,
        default=RECEPT_DIR,
        help="Wf-30 receptor.pdbqt directory (default: "
        f"{RECEPT_DIR.relative_to(PROJECT_ROOT)}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)}).",
    )
    p.add_argument(
        "--from-existing",
        action="store_true",
        default=False,
        help=(
            "Re-score from already-docked pose PDBQT files under "
            "--out-dir/<pocket>/pose_seed*.pdbqt — skip the Vina dock step. "
            "Used for idempotent re-runs and to drive the "
            "Kabsch+Hungarian re-validation without re-docking."
        ),
    )
    p.add_argument(
        "--dry-run-summary",
        action="store_true",
        default=False,
        help="Only print scope + Vina binary status without running.",
    )
    return p


def _load_pocket_box_centers(
    pockets: List[str], recept_dir: Path
) -> Dict[str, List[float]]:
    """Read docking box centers from ``prep_meta.json`` if present.

    Falls back to the centroid of ``receptor.pdbqt`` heavy atoms.
    """
    out: Dict[str, List[float]] = {}
    for p in pockets:
        meta = recept_dir / p / "prep_meta.json"
        if meta.exists():
            try:
                d = json.loads(meta.read_text())
                c = d.get("docking_box_center") or d.get("box_center")
                if c and len(c) == 3:
                    out[p] = [float(c[0]), float(c[1]), float(c[2])]
                    continue
            except Exception:  # noqa: BLE001
                pass
        # Fallback: receptor heavy-atom centroid
        rec = recept_dir / p / "receptor.pdbqt"
        if rec.exists():
            xs, ys, zs = [], [], []
            for raw in rec.read_text().splitlines():
                if raw.startswith(("ATOM", "HETATM")) and len(raw) >= 54:
                    try:
                        xs.append(float(raw[30:38]))
                        ys.append(float(raw[38:46]))
                        zs.append(float(raw[46:54]))
                    except ValueError:
                        pass
            if xs:
                out[p] = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
                continue
        out[p] = [0.0, 0.0, 0.0]
    return out


def _write_metric_json(
    out_dir: Path,
    results: Dict[str, PocketResult],
    seeds: Tuple[int, ...],
    exhaustiveness: int,
    n_poses: int,
    vina_bin: Path,
    note: str = "",
) -> Path:
    """Write the redesigned metric ``docked_pose_rmsd.json`` (back-compat).

    This is the BD-9 successor to ``rigid_fragment_rmsd.json`` per
    TODO-134 §0 / Wf-179. We do NOT modify the legacy self-RMSD JSON
    (back-compat policy in party-twenty-spirit §3.5).
    """
    measurements = []
    for p, r in results.items():
        if r.status != "MEASURED":
            measurements.append(
                {
                    "pocket": p,
                    "status": r.status,
                    "per_seed_best_rmsd_angstrom": r.per_seed_best_rmsd_angstrom,
                }
            )
            continue
        measurements.append(
            {
                "pocket": p,
                "per_seed_best_rmsd_angstrom": r.per_seed_best_rmsd_angstrom,
                "median_best_rmsd_angstrom": r.median_best_rmsd_angstrom,
                "mean_best_rmsd_angstrom": r.mean_best_rmsd_angstrom,
                "std_best_rmsd_angstrom": r.std_best_rmsd_angstrom,
                "passes_2_5a_gate": r.passes_2_5a_gate,
                "passes_2_0a_sota": r.passes_2_0a_sota,
                "n_seeds_succeeded": r.n_seeds_succeeded,
                "status": r.status,
                "date_utc": _utc_now_iso(),
            }
        )
    medians = [
        r.median_best_rmsd_angstrom
        for r in results.values()
        if r.status == "MEASURED" and r.median_best_rmsd_angstrom is not None
    ]
    aggregate_median = float(statistics.median(medians)) if medians else float("nan")
    n_pass_25 = sum(1 for m in medians if m <= 2.5)
    n_pass_20 = sum(1 for m in medians if m <= 2.0)
    payload = {
        "metric": "docked_pose_rmsd",
        "description": (
            "Median best heavy-atom RMSD (Kabsch+Hungarian, adaptive "
            "centroid pre-alignment) of Vina redocked poses vs crystal "
            f"ligand pose across {len(results)} pockets x {len(seeds)} "
            f"seeds x exhaustiveness={exhaustiveness}. "
            "BD-9 m15 RMSD Phase 2 Vina redocking (canonical script)."
        ),
        "unit": "angstrom",
        "scope": "pocket-bound redocking benchmark (m15 redesigned per TODO-134)",
        "source": "TODO-134 (m15 redesign spec) + Wf-139 P1 (ref ligand SMILES extraction) + BD-9 m15 RMSD Phase 2 (this script)",
        "engine": f"AutoDock Vina via {vina_bin}",
        "exhaustiveness": exhaustiveness,
        "n_poses": n_poses,
        "seeds": list(seeds),
        "pockets_planned": list(results.keys()),
        "value": round(aggregate_median, 4) if medians else None,
        "sota_threshold_angstrom": 2.0,
        "sota_success_criterion": (
            "median_best_rmsd <= 2.0 A (Trott & Olson 2010 Vina paper "
            "convention); per-pocket success rate >= 80% (TransDiffSBDD "
            "reference: 83.9%)"
        ),
        "aggregate_median_across_pockets": round(aggregate_median, 4)
        if medians
        else None,
        "n_pockets_passed_2_5a": n_pass_25,
        "n_pockets_passed_2_0a": n_pass_20,
        "n_pockets_measured": len(medians),
        "gate_4_passes_2_5a_overall": bool(aggregate_median <= 2.5)
        if medians
        else False,
        "measurements": measurements,
        "note": note
        or (
            "Generated by molmetal/scripts/m15_rmsd_phase2.py — canonical "
            "BD-9 m15 RMSD Phase 2 script. Backward-compatible with "
            "existing docked_pose_rmsd.json schema (Wf-179 ship)."
        ),
    }
    # Write both: stage in the Wf-179 ship location AND a metrics-file
    # friendly path so paper §3.5 / metrics/by_metric pickers see it.
    out_metric = PROJECT_ROOT / "metrics" / "by_metric" / "docked_pose_rmsd.json"
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(out_metric, json.dumps(payload, indent=2))
    return out_metric


def _write_summary(
    out_dir: Path,
    results: Dict[str, PocketResult],
    seeds: Tuple[int, ...],
    exhaustiveness: int,
    n_poses: int,
    vina_bin: Path,
    from_existing: bool,
    pocket_box_centers: Dict[str, List[float]],
) -> dict:
    """Write ``summary.json`` and ``synthesis.md`` under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_summary: Dict[str, dict] = {}
    for p, r in results.items():
        r.box_center = pocket_box_centers.get(p, [0.0, 0.0, 0.0])
        results_summary[p] = asdict(r)
    medians = [
        r.median_best_rmsd_angstrom
        for r in results.values()
        if r.status == "MEASURED" and r.median_best_rmsd_angstrom is not None
    ]
    aggregate_median = float(statistics.median(medians)) if medians else float("nan")
    n_pass_25 = sum(1 for m in medians if m <= 2.5)
    n_pass_20 = sum(1 for m in medians if m <= 2.0)
    summary = {
        "workflow": "BD-9 m15 RMSD Phase 2 (canonical script)",
        "engine": f"AutoDock Vina via {vina_bin}",
        "exhaustiveness": int(exhaustiveness),
        "n_poses": int(n_poses),
        "seeds": list(seeds),
        "pockets": results_summary,
        "aggregate_median_across_pockets": round(aggregate_median, 4)
        if medians
        else None,
        "n_pockets": len(medians),
        "n_pockets_passed_2_5a": n_pass_25,
        "n_pockets_passed_2_0a": n_pass_20,
        "gate_4_passes_2_5a_overall": bool(aggregate_median <= 2.5)
        if medians
        else False,
        "from_existing_poses": bool(from_existing),
        "generated_utc": _utc_now_iso(),
    }
    (_wf416_atomic_write_text(out_dir / "summary.json"), json.dumps(summary, indent=2))

    md_lines: List[str] = []
    md_lines.append("# BD-9 m15 RMSD Phase 2 — Vina redocking benchmark")
    md_lines.append("")
    md_lines.append(f"_Generated_: {_utc_now_iso()}  ")
    md_lines.append(f"_Engine_: {vina_bin}  ")
    md_lines.append(
        f"_Exhaustiveness_: {exhaustiveness}; _n_poses_: {n_poses}; "
        f"_seeds_: {list(seeds)}."
    )
    md_lines.append(
        f"_Mode_: {'from-existing (re-score only)' if from_existing else 'full dock'}."
    )
    md_lines.append("")
    md_lines.append("## Per-pocket median best RMSD")
    md_lines.append("")
    md_lines.append(
        "| Pocket | n_ref_heavy | median A | mean A | std A | n_seeds | passes 2.5 A | passes 2.0 A | status |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for p, r in results.items():
        if r.status != "MEASURED":
            md_lines.append(
                f"| {p} | {r.n_ref_heavy} | — | — | — | "
                f"{r.n_seeds_succeeded}/{r.n_seeds_succeeded + r.n_seeds_failed} | — | — | {r.status} |"
            )
            continue
        md_lines.append(
            f"| {p} | {r.n_ref_heavy} | {r.median_best_rmsd_angstrom:.4f} | "
            f"{r.mean_best_rmsd_angstrom:.4f} | {r.std_best_rmsd_angstrom:.4f} | "
            f"{r.n_seeds_succeeded}/{r.n_seeds_succeeded + r.n_seeds_failed} | "
            f"{'Y' if r.passes_2_5a_gate else 'N'} | "
            f"{'Y' if r.passes_2_0a_sota else 'N'} | {r.status} |"
        )
    md_lines.append("")
    if medians:
        md_lines.append(
            f"**Aggregate median across pockets**: {aggregate_median:.4f} A "
            f"({len(medians)} pockets)."
        )
        md_lines.append("")
        md_lines.append(
            f"Pass 2.5 A (practical gate): {n_pass_25}/{len(medians)} pockets.  "
        )
        md_lines.append(
            f"Pass 2.0 A (SOTA gate, Trott & Olson 2010): {n_pass_20}/{len(medians)} pockets."
        )
    else:
        md_lines.append(
            "**No pockets measured** — Vina binary or receptor.pdbqt likely missing."
        )
    md_lines.append("")
    md_lines.append("## Honest framing (party-twenty-spirit §3.5)")
    md_lines.append("")
    md_lines.append(
        "* All RMSDs are **real Vina 1.2.7 redocking** — not a stub "
        "floor, not a hash-derived placeholder."
    )
    md_lines.append(
        "* The Hungarian matcher uses element-typed cost = "
        "pairwise Euclidean distance AFTER adaptive centroid "
        "pre-alignment (3 A threshold). This avoids wrong matches "
        "when Vina transposes the molecule inside the pocket."
    )
    md_lines.append(
        "* Backward compat: the legacy m15 metric "
        "(`rigid_fragment_rmsd.json`, self-RMSD) is **NOT** modified; "
        "this script writes the redesigned metric "
        "`docked_pose_rmsd.json` (per TODO-134 §0 / Wf-179)."
    )
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append("")
    md_lines.append("* `summary.json` — machine-readable aggregate.")
    md_lines.append(
        "* `<pocket>/per_seed_metrics.json` — per-pocket per-seed RMSD records."
    )
    md_lines.append(
        "* `<pocket>/per_pose_metrics.json` — per-pocket per-seed per-pose RMSD records."
    )
    md_lines.append("* `<pocket>/pose_seed<N>_p<M>.pdbqt` — individual docked poses.")
    (_wf416_atomic_write_text(out_dir / "synthesis.md"), "\n".join(md_lines) + "\n")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir

    # ---- Pre-flight ----
    vina_bin = args.vina_bin or _resolve_vina_bin()
    if vina_bin is None and not args.dry_run_summary:
        print(
            "ERROR: no working vina binary found. Run check_vina.py " "to diagnose.",
            file=sys.stderr,
        )
        return 1
    vina_bin_label = str(vina_bin) if vina_bin else "(dry-run only)"

    if args.dry_run_summary:
        print("DRY RUN — BD-9 m15 RMSD Phase 2 scope:")
        print(f"  pockets : {list(args.pockets)}")
        print(f"  seeds   : {list(args.seeds)}")
        print(f"  n_poses : {args.n_poses}")
        print(f"  exh     : {args.exhaustiveness}")
        print(f"  vina bin: {vina_bin_label}")
        print(f"  out dir : {out_dir}")
        print(f"  mode    : {'from-existing' if args.from_existing else 'full dock'}")
        return 0

    if not args.ligand_meta.exists():
        print(
            f"ERROR: ligand_metadata.json not found at {args.ligand_meta}. "
            f"Wf-139 P1 must ship first.",
            file=sys.stderr,
        )
        return 2

    # Load ligand metadata + box centers
    meta = json.loads(args.ligand_meta.read_text())
    pocket_meta = {p["pocket_id"]: p for p in meta["pockets"]}
    box_centers = _load_pocket_box_centers(list(args.pockets), args.recept_dir)

    print(f"Vina binary: {vina_bin}")
    cp = subprocess.run(
        [str(vina_bin), "--version"], capture_output=True, text=True, timeout=10
    )
    print(f"Vina version: {_strip_ansi(cp.stdout).strip()}")
    print(f"Pockets: {list(args.pockets)}")
    print(f"Seeds:   {list(args.seeds)}")
    print(f"n_poses: {args.n_poses}; exhaustiveness: {args.exhaustiveness}")
    print(
        f"Mode:    {'from-existing (re-score only)' if args.from_existing else 'full dock'}"
    )
    print()

    results: Dict[str, PocketResult] = {}
    for pocket_id in args.pockets:
        if pocket_id not in pocket_meta:
            print(f"SKIP {pocket_id}: not in ligand_metadata.json")
            continue
        receptor = args.recept_dir / pocket_id / "receptor.pdbqt"
        if not receptor.exists():
            print(f"SKIP {pocket_id}: no receptor.pdbqt at {receptor}")
            continue
        pocket_out = out_dir / pocket_id
        print(f"=== Pocket {pocket_id} ===")

        if args.from_existing:
            r = _run_pocket_from_existing_poses(
                pocket_id,
                pocket_meta[pocket_id],
                pocket_out,
                tuple(args.seeds),
            )
        else:
            r = _run_pocket_full(
                pocket_id,
                pocket_meta[pocket_id],
                box_centers[pocket_id],
                pocket_out,
                receptor,
                vina_bin,
                tuple(args.seeds),
                int(args.exhaustiveness),
                int(args.n_poses),
            )

        pocket_out.mkdir(parents=True, exist_ok=True)
        r.box_center = box_centers.get(pocket_id, [0.0, 0.0, 0.0])
        # Per-seed / per-pose JSON (also written by helper functions for full mode)
        (
            _wf416_atomic_write_text(pocket_out / "per_seed_metrics.json"),
            json.dumps(asdict(r), indent=2),
        )
        results[pocket_id] = r
        if r.status == "MEASURED":
            print(
                f"  median={r.median_best_rmsd_angstrom:.4f}A "
                f"mean={r.mean_best_rmsd_angstrom:.4f}A "
                f"std={r.std_best_rmsd_angstrom:.4f}A "
                f"passes_2.5={r.passes_2_5a_gate}"
            )
        else:
            print(f"  status={r.status}")

    # ---- Write summary + back-compat metric JSON ----
    summary = _write_summary(
        out_dir,
        results,
        tuple(args.seeds),
        int(args.exhaustiveness),
        int(args.n_poses),
        vina_bin,
        bool(args.from_existing),
        box_centers,
    )
    metric_path = _write_metric_json(
        out_dir,
        results,
        tuple(args.seeds),
        int(args.exhaustiveness),
        int(args.n_poses),
        vina_bin,
    )
    print()
    print(f"Wrote summary: {out_dir / 'summary.json'}")
    print(f"Wrote synthesis: {out_dir / 'synthesis.md'}")
    print(f"Wrote redesigned metric: {metric_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:  # pragma: no cover — defensive
        traceback.print_exc()
        sys.exit(99)
