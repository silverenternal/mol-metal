"""Wf-246-D m8 PB DOCK mode steric-clash pipeline.

Runs PoseBusters (PB) dock mode against the 65 docked poses produced
by Wf-179 P2 (5 pockets × 3 seeds × 5 poses, minus 3ny8's 2 missing
seeds = 65 actual poses on disk).  The runner preserves docked
coordinates and runs the full PB ``config=dock`` battery, which adds
**12 protein-aware checks** on top of the 14 chemistry checks:

* ``minimum_distance_to_protein``               — minimum atom-atom
  distance between ligand heavy atoms and protein heavy atoms.
  PB threshold = 2.0 Å (Buttenschoen 2024 §2.3, clash = distance < 2.0).
* ``volume_overlap_with_protein``               — hard steric overlap
  via BHT volume integration (PB threshold = 0 overlap).
* ``protein-ligand_maximum_distance``            — ligand stays near
  pocket (PB threshold ≤ 6.0 Å median ligand-atom-to-protein-atom).
* 6 cofactor / water distance + volume checks (minimum_distance_to_*,
  volume_overlap_with_*, not_too_far_away_*) that surface only when
  the PDB declares cofactor HETATM records (e.g. NAD, heme, SO4).

Output schema (per pose):
    {pocket, seed, pose_idx, pdbqt_path, smiles, n_heavy,
     pb_valid, pb_status, checks: {check_name: bool},
     min_protein_dist_A, mean_protein_dist_A,
     volume_overlap_A3, n_failures, failures}

Aggregation:
    per_pocket[ pocket ] = {clash_rate, mean_dist_to_protein, ...}
    cross_pocket         = {clash_rate_overall, mean_overall, ...}

Honest framing (party-twenty-spirit §3.5):
* The brief asks for "75 Vina jobs" — 75 = 5 × 3 × 5.  On disk we
  have **65** poses because 3ny8 only ran 1 of 3 seeds
  (``per_seed_metrics.json`` for 3ny8 carries a single seed=0
  record).  We report n=65 and DO NOT inflate to 75.
* PB dock mode here runs on Vina-docked heavy-atom coordinates
  (NO hydrogens on the ligand — Vina PDBQT is heavy-atom only).
  PB's clash checks treat heavy atoms only; Hs are added internally
  by PB.  This is the documented PB dock mode contract.
* Pass-rate is computed on the full 26-check dock battery.  We
  additionally surface a **clash-rate** as the fraction of poses
  that fail ``minimum_distance_to_protein`` OR
  ``volume_overlap_with_protein`` (the two protein-aware checks
  that matter for steric clash).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on sys.path so we can import the validation runner
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore

from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (  # type: ignore
    parse_pdbqt_heavy_atoms,
    smiles_from_pdbqt_remark,
)
from molmetal.validation.posebusters_runner import (  # type: ignore
    check_docked_pose,
    posebusters_available,
)

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
WF179_DIR = (
    Path(__file__).resolve().parents[3]
    / "molmetal"
    / "reports"
    / "wf179_bd9_m15_rmsd_p2"
)
OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "molmetal"
    / "reports"
    / "wf246d_m8_pb_dock"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

POCKETS = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]
RECEPTOR_PDBS = {
    p: OUT_DIR / f"{p}_receptor.pdb" for p in POCKETS
}

# Checks we care about for steric clash — protein-aware subset.
PROTEIN_AWARE_KEYS = (
    "minimum_distance_to_protein",
    "volume_overlap_with_protein",
    "protein-ligand_maximum_distance",
)


# --------------------------------------------------------------------------
# Build RDKit Mol from docked PDBQT coordinates
# --------------------------------------------------------------------------
def _build_mol_from_pdbqt(pdbqt_path: Path) -> Tuple[Optional[Chem.Mol], str]:
    """Build a single-conformer RDKit Mol from a docked PDBQT pose.

    Returns ``(mol_or_None, smiles)``.  The Mol preserves the docked
    heavy-atom coordinates exactly (no embed, no optimize).  This is
    the contract that ``check_docked_pose`` expects.

    Strategy:
        1. Pull SMILES from the Vina PDBQT ``REMARK SMILES`` line.
        2. Parse heavy atoms (no H) from the PDBQT ATOM block.
        3. Build an RDKit Mol from SMILES (sanitized so bonds are
           correct), then assign the parsed heavy-atom coordinates
           onto a fresh conformer.
        4. Sanitize the molecule so PB's chemistry checks have a
           valid molecule (but coordinates are NOT touched).
    """
    smiles = smiles_from_pdbqt_remark(pdbqt_path)
    heavy = parse_pdbqt_heavy_atoms(pdbqt_path)
    if not heavy:
        return None, smiles
    # Sanitize-from-SMILES so the bond graph is correct (PB chemistry
    # checks require a valid mol).
    template = Chem.MolFromSmiles(smiles)
    if template is None or template.GetNumAtoms() == 0:
        return None, smiles
    template = Chem.AddHs(template)
    n_template = template.GetNumAtoms()
    if n_template != len(heavy):
        # Template (with explicit Hs) doesn't match heavy-atom count.
        # Drop Hs and try aligning to heavy count.
        template = Chem.RemoveHs(template)
        n_template = template.GetNumAtoms()
        if n_template != len(heavy):
            # Length mismatch — return None so the caller can record
            # a coordinate_alignment failure rather than a fake PASS.
            return None, smiles
    # Build a fresh conformer from the parsed coordinates.
    conf = Chem.Conformer(n_template)
    conf.Set3D(True)
    for i, atom in enumerate(heavy):
        x = float(atom["x"])
        y = float(atom["y"])
        z = float(atom["z"])
        conf.SetAtomPosition(i, (x, y, z))
    template.RemoveAllConformers()
    template.AddConformer(conf, assignId=True)
    # Now template has one conformer with the docked coords.  We do
    # NOT sanitize the geometry here — that would break bond
    # perception if any atoms are too close.  check_docked_pose
    # accepts molecules in any sanitization state; PB itself
    # tolerates non-canonical geom.
    return template, smiles


# --------------------------------------------------------------------------
# Geometry-from-coords fall-back for min/mean protein distance
# --------------------------------------------------------------------------
def _protein_distance_stats(mol: Chem.Mol, receptor_pdb: Path) -> Dict[str, float]:
    """Compute min + mean heavy-atom distance from ligand to protein.

    Used as a cross-check independent of PB's internal clash
    classification (PB's check returns a boolean; we want the actual
    Å figure so the report can compare pockets).
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {"min_dist_A": float("nan"), "mean_dist_A": float("nan")}
    lig_coords = mol.GetConformer().GetPositions()
    if len(lig_coords) == 0:
        return {"min_dist_A": float("nan"), "mean_dist_A": float("nan")}
    prot_coords = _load_protein_coords(receptor_pdb)
    if not prot_coords.size:
        return {"min_dist_A": float("nan"), "mean_dist_A": float("nan")}
    # brute-force min (small ligand × medium receptor — fine for ~30 atoms)
    import numpy as np
    diff = lig_coords[:, None, :] - prot_coords[None, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    min_d = float(np.sqrt(dist2.min()))
    mean_d = float(np.sqrt(dist2.mean()))
    return {"min_dist_A": min_d, "mean_dist_A": mean_d}


_PROT_CACHE: Dict[Path, "np.ndarray"] = {}


def _load_protein_coords(receptor_pdb: Path):
    import numpy as np
    if receptor_pdb in _PROT_CACHE:
        return _PROT_CACHE[receptor_pdb]
    coords = []
    with open(receptor_pdb) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    arr = np.array(coords, dtype=float) if coords else np.empty((0, 3))
    _PROT_CACHE[receptor_pdb] = arr
    return arr


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_one_pose(
    pdbqt_path: Path, receptor_pdb: Path
) -> Dict[str, Any]:
    """Run PB dock mode on a single docked pose."""
    t0 = time.time()
    out: Dict[str, Any] = {
        "pdbqt_path": str(pdbqt_path),
        "receptor_pdb": str(receptor_pdb),
        "smiles": None,
        "n_heavy": 0,
        "pb_valid": False,
        "pb_status": "not_run",
        "checks": {},
        "min_dist_to_protein_A": None,
        "mean_dist_to_protein_A": None,
        "n_failures": 0,
        "failures": [],
        "wall_seconds": 0.0,
        "error": None,
    }
    try:
        mol, smiles = _build_mol_from_pdbqt(pdbqt_path)
        out["smiles"] = smiles
        if mol is None:
            out["pb_status"] = "build_failed"
            out["failures"] = ["build_failed"]
            out["error"] = "build_mol_failed"
            out["wall_seconds"] = round(time.time() - t0, 3)
            return out
        out["n_heavy"] = mol.GetNumAtoms()

        # Independent min/mean distance to receptor (cross-check).
        d = _protein_distance_stats(mol, receptor_pdb)
        out["min_dist_to_protein_A"] = d["min_dist_A"]
        out["mean_dist_to_protein_A"] = d["mean_dist_A"]

        # PoseBusters dock mode.
        pb = check_docked_pose(mol, receptor_pdb)
        out["pb_valid"] = pb.get("pb_valid", False)
        out["pb_status"] = pb.get("status", "unknown")
        out["checks"] = pb.get("checks", {})
        out["failures"] = pb.get("failures", [])
        out["n_failures"] = sum(
            1 for k, v in (out["checks"] or {}).items() if v is False
        )
        out["version"] = pb.get("version")
        out["coordinates_preserved"] = pb.get("coordinates_preserved", None)
    except Exception as exc:
        out["pb_status"] = "error"
        out["error"] = f"{type(exc).__name__}:{exc}"
        out["failures"] = ["runner_exception"]
    out["wall_seconds"] = round(time.time() - t0, 3)
    return out


def is_clash(rec: Dict[str, Any]) -> bool:
    """True iff the pose fails a protein-aware steric clash check.

    PB's protein-aware battery (Buttenschoen 2024 §2.3) flags steric
    clash when **any** of the following is False:

        * minimum_distance_to_protein   — min ligand-protein distance
          must be ≥ 2.0 Å (PB default; matches Vina's clash tolerance)
        * volume_overlap_with_protein   — BHT volume integration must
          report zero overlap (no atoms interpenetrating)
        * protein-ligand_maximum_distance — ligand centroid must be
          within the binding site (not detached in solution)

    A pose fails the **clash_rate** metric iff it fails ANY of these
    three.  This is the protein-aware steric-clash metric requested
    by TODO-130 m8 REDESIGN.
    """
    chk = rec.get("checks") or {}
    for key in PROTEIN_AWARE_KEYS:
        v = chk.get(key)
        if v is False:
            return True
    return False


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-pocket + cross-pocket aggregation.

    Reports TWO pass-rates side by side, following party-twenty-spirit
    §3.5 (honest framing):

        * ``pb_pass_rate_*``  — fraction of poses where the FULL
          26-check PB dock battery passes.  Vina-docked poses
          routinely fail the 14 chemistry checks because (a) Vina
          PDBQT is heavy-atom only (no explicit Hs) and (b) Vina
          does NOT optimise bonded terms — its scoring function
          treats the ligand as a sum of atom-atom pairwise terms
          (Trott 2010, J. Comput. Chem.).  So pb_pass_rate_* on
          raw Vina output is expected to be 0.000.

        * ``clash_rate_*``    — fraction of poses that fail ANY of
          the 3 protein-aware steric clash checks
          (minimum_distance_to_protein / volume_overlap_with_protein
          / protein-ligand_maximum_distance).  This is the
          ``m8 steric clash`` metric the brief asks for; it is
          independent of the Vina heavy-atom / no-Hs issue.

        * ``mean_min_dist_to_protein_A`` — our independent
          brute-force min-distance computation; cross-checks PB's
          own protein-aware boolean.
    """
    per_pocket: Dict[str, Dict[str, Any]] = {}
    for pocket in POCKETS:
        pocket_recs = [r for r in results if POCKETS.index(pocket) == _pocket_of(r)]
        n_total = len(pocket_recs)
        n_clash = sum(1 for r in pocket_recs if is_clash(r))
        n_pass = sum(1 for r in pocket_recs if r.get("pb_valid") is True)
        n_fail = sum(1 for r in pocket_recs if r.get("pb_valid") is False)
        n_skip = n_total - n_pass - n_fail
        dists = [
            r["min_dist_to_protein_A"]
            for r in pocket_recs
            if isinstance(r.get("min_dist_to_protein_A"), (int, float))
            and r["min_dist_to_protein_A"] == r["min_dist_to_protein_A"]
        ]
        mean_d = (
            sum(dists) / len(dists) if dists else float("nan")
        )
        per_pocket[pocket] = {
            "pocket": pocket,
            "n_total": n_total,
            "n_clash": n_clash,
            "clash_rate": (n_clash / n_total) if n_total else 0.0,
            "n_pb_pass": n_pass,
            "n_pb_fail": n_fail,
            "n_pb_skip": n_skip,
            "pb_pass_rate": (n_pass / n_total) if n_total else 0.0,
            "mean_min_dist_to_protein_A": round(mean_d, 4) if dists else None,
        }

    n_total = len(results)
    n_clash = sum(1 for r in results if is_clash(r))
    n_pass = sum(1 for r in results if r.get("pb_valid") is True)
    n_fail = sum(1 for r in results if r.get("pb_valid") is False)
    dists = [
        r["min_dist_to_protein_A"]
        for r in results
        if isinstance(r.get("min_dist_to_protein_A"), (int, float))
        and r["min_dist_to_protein_A"] == r["min_dist_to_protein_A"]
    ]
    mean_d = sum(dists) / len(dists) if dists else float("nan")
    return {
        "n_total_poses": n_total,
        "n_clash": n_clash,
        "clash_rate_overall": (n_clash / n_total) if n_total else 0.0,
        "n_pb_pass": n_pass,
        "n_pb_fail": n_fail,
        "pb_pass_rate_overall": (n_pass / n_total) if n_total else 0.0,
        "mean_min_dist_to_protein_A": round(mean_d, 4) if dists else None,
        "per_pocket": per_pocket,
    }


def _pocket_of(rec: Dict[str, Any]) -> int:
    """Return the index in POCKETS of the pocket this record belongs to."""
    pdb = rec.get("pdbqt_path", "")
    for i, pocket in enumerate(POCKETS):
        if f"/{pocket}/" in pdb:
            return i
    return -1


def discover_poses() -> List[Tuple[str, int, int, Path]]:
    """Yield (pocket, seed, pose_idx, pdbqt_path) for every docked pose."""
    out: List[Tuple[str, int, int, Path]] = []
    for pocket in POCKETS:
        d = WF179_DIR / pocket
        if not d.is_dir():
            continue
        for p in sorted(d.glob("pose_seed*_p*.pdbqt")):
            # parse: pose_seed{s}_p{n}.pdbqt
            stem = p.stem  # pose_seed0_p1
            try:
                seed_part = stem.split("_")[1]  # seed0
                pose_part = stem.split("_")[2]  # p1
                seed = int(seed_part.replace("seed", ""))
                pose_idx = int(pose_part.replace("p", ""))
            except (IndexError, ValueError):
                continue
            out.append((pocket, seed, pose_idx, p))
    return out


def main() -> int:
    if not posebusters_available():
        print("ERROR: posebusters not installed", file=sys.stderr)
        return 1
    poses = discover_poses()
    print(f"[wf246d] discovered {len(poses)} docked poses")
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for pocket, seed, pose_idx, p in poses:
        receptor_pdb = RECEPTOR_PDBS[pocket]
        rec = run_one_pose(p, receptor_pdb)
        rec["pocket"] = pocket
        rec["seed"] = seed
        rec["pose_idx"] = pose_idx
        results.append(rec)
        print(
            f"[wf246d] {pocket:5s} seed={seed} pose={pose_idx} "
            f"smiles_len={len(rec['smiles'] or '')} "
            f"n_heavy={rec['n_heavy']} "
            f"pb_valid={rec['pb_valid']} "
            f"min_d={rec['min_dist_to_protein_A']} "
            f"status={rec['pb_status']} "
            f"wall={rec['wall_seconds']}s"
        )
    wall = round(time.time() - t_start, 3)
    agg = aggregate(results)
    summary = {
        "workflow": "Wf-246-D TODO-130 m8 PB DOCK mode steric clash",
        "engine": "PoseBusters 0.6.5 (config=dock, real conformers)",
        "input_source": str(WF179_DIR),
        "receptor_pdbs": {k: str(v) for k, v in RECEPTOR_PDBS.items()},
        "n_poses_target": 75,        # 5 pockets × 3 seeds × 5 poses
        "n_poses_actual": len(results),  # 65 on disk (3ny8 has 1 seed)
        "wall_seconds": wall,
        "aggregation": agg,
        "results": results,
    }
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[wf246d] wrote {out_json}  (wall={wall}s)")
    print(
        f"[wf246d] clash_rate={agg['clash_rate_overall']:.4f}  "
        f"pb_pass_rate={agg['pb_pass_rate_overall']:.4f}  "
        f"mean_min_dist={agg['mean_min_dist_to_protein_A']} Å"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
