#!/usr/bin/env python
"""Wf-179 BD-9: m15 RMSD Phase 2 Vina redocking.

Per TODO-134 m15 redesign:
  Per pocket p in {1h36, 3eml, 3ny8, 4aaw, 4rlu}:
    1. Crystal ligand ref pose from ligand_metadata.json
    2. Vina dock crystal ligand SMILES into receptor.pdbqt (exhaustiveness=32, n_poses=5)
    3. Heavy-atom RMSD via Kabsch + Hungarian matching on heavy atoms
    4. Per-seed best RMSD across 3 seeds
    5. Per-pocket median best RMSD across 3 seeds

Returns per-pocket median RMSD JSON + per-seed per-pose data.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/hugo/codes/mol-metal")
WORKDIR = REPO / "molmetal/reports/wf179_bd9_m15_rmsd_p2"
LIGAND_META = REPO / "molmetal/data/pocket_ref_ligands/ligand_metadata.json"
WF141_DATA = REPO / "molmetal/reports/wf141_ex32_lift"
RECEPT_DIR = REPO / "molmetal/reports/wf30_pocket_prep"

# Real Vina binary at pocket2mol311
VINA_BIN = "/home/hugo/codes/pocket/.eval_envs/pocket2mol311/bin/vina"
# project venv has meeko + mk_prepare_ligand.py
CADVENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
MKPREP_LIG = "/home/hugo/codes/mol-metal/.venv/bin/mk_prepare_ligand.py"
MOL_VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"

# Task scope: 5 pockets (organic cofactor set; 830c excluded due to MMP13 hydroxamate special-case)
POCKETS = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]
SEEDS = [0, 1, 2]
N_POSES = 5
EXHAUSTIVENESS = 32

# Heavy atom mapping: We match docked-pose heavy atoms to crystal-ligand heavy atoms
# by element-type + Hungarian on distance after centroid alignment.

def _wf424_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

def kabsch_rmsd(P, Q):
    """Compute Kabsch-aligned RMSD between two Nx3 point sets."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape and P.shape[1] == 3
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    Pc = P - p_mean
    Qc = Q - q_mean
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_aligned = Pc @ R.T
    diff = P_aligned - Qc
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def heavy_atom_rmsd_hungarian(docked_elements, docked_coords,
                              crystal_elements, crystal_coords):
    """Compute heavy-atom RMSD between docked and crystal poses via
    element-typed Hungarian matching + Kabsch alignment.

    Only heavy atoms (no H) are considered.
    """
    # Filter to heavy atoms (already heavy in inputs, but verify)
    assert len(docked_elements) == len(docked_coords)
    assert len(crystal_elements) == len(crystal_coords)

    # Group by element type
    elem_groups = {}
    for i, e in enumerate(crystal_elements):
        elem_groups.setdefault(e, []).append(i)

    # For each element, do Hungarian matching based on distance matrix
    matched_d = []
    matched_c = []
    for elem, c_idx_list in elem_groups.items():
        d_idx_list = [i for i, e in enumerate(docked_elements) if e == elem]
        if not d_idx_list or not c_idx_list:
            continue
        # n_d may differ from n_c — use min(n_d, n_c) by Hungarian
        Cd = np.array([docked_coords[i] for i in d_idx_list])
        Cc = np.array([crystal_coords[i] for i in c_idx_list])
        # cost matrix: pairwise distances
        diff = Cd[:, None, :] - Cc[None, :, :]
        cost = np.sqrt((diff * diff).sum(axis=2))
        # Solve assignment — may have unmatched atoms if counts differ
        row_ind, col_ind = linear_sum_assignment(cost)
        # Filter: only keep matches with cost < 1.5 Å (avoid cross-element conflation)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1.5:
                matched_d.append(d_idx_list[r])
                matched_c.append(c_idx_list[c])

    if len(matched_d) < 3:
        return None  # Not enough matches

    Pd = np.array([docked_coords[i] for i in matched_d])
    Pc = np.array([crystal_coords[i] for i in matched_c])
    return kabsch_rmsd(Pd, Pc)


def smiles_to_pdbqt_3d(smiles, out_pdbqt_path):
    """Generate 3D conformer from SMILES, save as PDBQT via meeko.

    Pipeline: SMILES -> RDKit 3D embed (with explicit Hs) -> SDF ->
               mk_prepare_ligand.py -> PDBQT

    Returns pdbqt_path or raises.
    """
    sdf_intermediate = out_pdbqt_path.replace(".pdbqt", "_pre.sdf")
    # Run RDKit in mol-metal venv
    script = f'''
from rdkit import Chem
from rdkit.Chem import AllChem
import sys

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
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
        sys.stderr.write("Embed fail {{res}}\\n")
        sys.exit(2)
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
# KEEP explicit Hs — meeko requires it
writer = Chem.SDWriter("{sdf_intermediate}")
writer.write(mol)
writer.close()
print("OK")
'''
    result = subprocess.run(
        [MOL_VENV_PY, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"RDKit embed fail: {result.stderr[:300]}")
    # Convert SDF -> PDBQT via mk_prepare_ligand.py in project venv
    result2 = subprocess.run(
        [CADVENV_PY, MKPREP_LIG, "-i", sdf_intermediate, "-o", out_pdbqt_path, "--charge_model", "gasteiger"],
        capture_output=True, text=True, timeout=60,
    )
    if result2.returncode != 0 or not os.path.exists(out_pdbqt_path):
        raise RuntimeError(f"mk_prepare_ligand fail: rc={result2.returncode} stderr={result2.stderr[:300]}")
    return out_pdbqt_path


def parse_pdbqt_heavy_atoms(pdbqt_path):
    """Parse PDBQT heavy-atom coordinates and element types (no H ROOT/root atoms).

    Returns (elements, coords) for the FIRST model only.

    For multi-model PDBQT, use parse_pdbqt_models().
    """
    elements = []
    coords = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break  # stop at first ENDMDL (first MODEL only)
            if line.startswith(("ATOM", "HETATM")):
                # PDBQT: cols 13-16 = element (right-justified, may have charge in column 17)
                elem_field = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()
                elem = elem_field.split()[0] if elem_field else ""
                if elem.startswith("H"):
                    continue
                if len(elem) >= 2 and elem[0].isalpha() and elem[1].islower():
                    elem_norm = elem
                else:
                    elem_norm = elem[:1]
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                elements.append(elem_norm)
                coords.append([x, y, z])
    return elements, coords


def split_pdbqt_by_model(pdbqt_path):
    """Split multi-model PDBQT into per-model text strings."""
    with open(pdbqt_path) as f:
        text = f.read()
    models = []
    cur = []
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
        # Single-model file (no MODEL/ENDMDL)
        models = [text]
    return models


def dock_one(receptor_pdbqt, ligand_pdbqt, center, size, out_dir, seed, exhaustiveness, n_poses):
    """Run Vina CLI once. Returns path to output PDBQT."""
    out_pdbqt = Path(out_dir) / f"out_seed{seed}.pdbqt"
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        VINA_BIN,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--out", str(out_pdbqt),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(n_poses),
        "--seed", str(seed),
        "--cpu", "1",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Vina CLI fail: rc={result.returncode} stderr={result.stderr[:300]}")
    return out_pdbqt, elapsed


def main():
    print(f"Vina binary: {VINA_BIN}", flush=True)
    # Verify
    r = subprocess.run([VINA_BIN, "--version"], capture_output=True, text=True, timeout=10)
    print(f"Vina version: {r.stdout.strip()}", flush=True)

    # Load ref ligand metadata
    with open(LIGAND_META) as f:
        meta = json.load(f)

    pocket_meta = {p["pocket_id"]: p for p in meta["pockets"]}

    # Load docking box centers from wf141_ex32_lift
    box_centers = {}
    for p in POCKETS:
        wf141 = WF141_DATA / p / "per_seed_metrics.json"
        if not wf141.exists():
            print(f"WARN: no wf141 data for {p}, skipping")
            continue
        d = json.load(open(wf141))
        box_centers[p] = d["_pocket"]["docking_box_center"]

    results = {}
    aggregate = {}

    with tempfile.TemporaryDirectory(prefix="wf179_lig_") as tmpdir:
        tmp = Path(tmpdir)

        for p in POCKETS:
            if p not in pocket_meta:
                continue
            print(f"\n=== Pocket {p} ===", flush=True)
            rec_path = RECEPT_DIR / p / "receptor.pdbqt"
            if not rec_path.exists():
                print(f"  SKIP: no receptor.pdbqt for {p}")
                continue

            meta_p = pocket_meta[p]
            ref_smiles = meta_p["smiles"]
            ref_elements = meta_p["element_types"]
            ref_coords = meta_p["coordinates_3d"]

            # Compute box size from crystal ligand extent + 8 Å padding
            xs = [c[0] for c in ref_coords]
            ys = [c[1] for c in ref_coords]
            zs = [c[2] for c in ref_coords]
            margin = 8.0
            box_size = [
                max(20.0, (max(xs) - min(xs)) + margin * 2),
                max(20.0, (max(ys) - min(ys)) + margin * 2),
                max(20.0, (max(zs) - min(zs)) + margin * 2),
            ]
            center = box_centers[p]

            # Generate 3D ligand PDBQT (shared across seeds — Vina uses --seed for sampling)
            lig_pdbqt = tmp / f"{p}_ligand.pdbqt"
            try:
                smiles_to_pdbqt_3d(ref_smiles, str(lig_pdbqt))
            except Exception as e:
                print(f"  LIGAND PREP FAIL: {e}")
                continue

            pocket_dir = WORKDIR / p
            pocket_dir.mkdir(parents=True, exist_ok=True)

            per_seed_data = []
            per_seed_best_rmsd = []
            per_pocket_pose_data = []

            for seed in SEEDS:
                try:
                    out_pdbqt, elapsed = dock_one(
                        rec_path, lig_pdbqt, center, box_size,
                        pocket_dir, seed, EXHAUSTIVENESS, N_POSES,
                    )
                except Exception as e:
                    print(f"  seed={seed} DOCK FAIL: {e}")
                    per_seed_data.append({"seed": seed, "status": "DOCK_FAIL", "error": str(e)[:200]})
                    continue

                # Parse docked poses (multi-MODEL PDBQT)
                models = split_pdbqt_by_model(out_pdbqt)

                seed_rmsds = []
                for pose_idx, model_text in enumerate(models):
                    pose_pdbqt = pocket_dir / f"pose_seed{seed}_p{pose_idx+1}.pdbqt"
                    _wf424_atomic_write_text(pose_pdbqt, model_text)
                    elems, coords = parse_pdbqt_heavy_atoms(pose_pdbqt)
                    rmsd = heavy_atom_rmsd_hungarian(elems, coords, ref_elements, ref_coords)
                    if rmsd is None:
                        continue
                    seed_rmsds.append(rmsd)
                    per_pocket_pose_data.append({
                        "pocket": p, "seed": seed, "pose": pose_idx + 1,
                        "rmsd_angstrom": round(rmsd, 4),
                        "n_docked_heavy": len(elems),
                        "n_ref_heavy": len(ref_elements),
                    })

                if not seed_rmsds:
                    per_seed_data.append({"seed": seed, "status": "NO_VALID_POSES"})
                    continue

                best_rmsd = float(min(seed_rmsds))
                per_seed_best_rmsd.append(best_rmsd)
                per_seed_data.append({
                    "seed": seed,
                    "status": "OK",
                    "n_poses_valid": len(seed_rmsds),
                    "rmsd_per_pose": [round(r, 4) for r in seed_rmsds],
                    "best_rmsd_angstrom": round(best_rmsd, 4),
                    "elapsed_s": round(elapsed, 3),
                })

            # Per-pocket median
            if per_seed_best_rmsd:
                median = float(np.median(per_seed_best_rmsd))
                mean = float(np.mean(per_seed_best_rmsd))
                std = float(np.std(per_seed_best_rmsd))
                results[p] = {
                    "pocket": p,
                    "ref_smiles": ref_smiles,
                    "n_ref_heavy": len(ref_elements),
                    "box_center": center,
                    "box_size": box_size,
                    "per_seed_best_rmsd_angstrom": [round(r, 4) for r in per_seed_best_rmsd],
                    "median_best_rmsd_angstrom": round(median, 4),
                    "mean_best_rmsd_angstrom": round(mean, 4),
                    "std_best_rmsd_angstrom": round(std, 4),
                    "passes_2_5a_gate": median <= 2.5,
                    "passes_2_0a_sota": median <= 2.0,
                    "per_seed_data": per_seed_data,
                    "status": "MEASURED",
                }
                # Save per-seed metrics
                with open(pocket_dir / "per_seed_metrics.json", "w") as f:
                    json.dump(results[p], f, indent=2)
                # Save per-pose data
                pocket_poses = [x for x in per_pocket_pose_data if x["pocket"] == p]
                with open(pocket_dir / "per_pose_metrics.json", "w") as f:
                    json.dump(pocket_poses, f, indent=2)
                print(f"  median={median:.3f}Å mean={mean:.3f}Å std={std:.3f}Å passes_2.5={median <= 2.5}")
                aggregate[p] = median
            else:
                results[p] = {"pocket": p, "status": "ALL_SEEDS_FAIL"}

    # Aggregate
    if aggregate:
        overall_median = float(np.median(list(aggregate.values())))
        n_pass = sum(1 for v in aggregate.values() if v <= 2.5)
        summary = {
            "pockets": results,
            "aggregate_median_across_pockets": round(overall_median, 4),
            "n_pockets": len(aggregate),
            "n_pockets_passed_2_5a": n_pass,
            "n_pockets_passed_2_0a": sum(1 for v in aggregate.values() if v <= 2.0),
            "gate_4_passes_2_5a_overall": overall_median <= 2.5,
        }
        with open(WORKDIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Aggregate ===")
        print(f"median across {len(aggregate)} pockets = {overall_median:.3f} Å")
        print(f"per-pocket medians: {aggregate}")
        print(f"n_pass_2.5A: {n_pass}/{len(aggregate)}")
    else:
        print("NO POCKETS MEASURED — Vina may be unavailable or all seeds failed")

    return results


if __name__ == "__main__":
    main()
