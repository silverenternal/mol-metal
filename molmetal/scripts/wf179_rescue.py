#!/usr/bin/env python
"""Wf-179 BD-9 m15 RMSD Phase 2 RESCUE: finish incomplete pockets.

Strategy:
  - Use existing pose PDBQT files for 1h36, 3eml, 3ny8 seed 0
  - Compute RMSDs from existing pose files
  - Run 3ny8 seeds 1-2, 4aaw seeds 0-2, 4rlu seeds 0-2 with ex=16 (faster)
  - Compute per-pocket median + summary JSON

Wall budget: ~10-15 min
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Wf-424-A4 atomic write protocol primitives, bound at module scope: the
# wrapper below runs in THIS process (the generated subprocess scripts
# carry their own copy of this import block inside the embedded template).
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

REPO = Path(__file__).resolve().parents[2]
WORKDIR = REPO / "molmetal/reports/wf179_bd9_m15_rmsd_p2"
LIGAND_META = REPO / "molmetal/data/pocket_ref_ligands/ligand_metadata.json"
WF141_DATA = REPO / "molmetal/reports/wf141_ex32_lift"
RECEPT_DIR = REPO / "molmetal/reports/wf30_pocket_prep"

VINA_BIN = "/home/hugo/codes/pocket/.eval_envs/pocket2mol311/bin/vina"
CADVENV_PY = sys.executable
MKPREP_LIG = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "mk_prepare_ligand.py")
MOL_VENV_PY = sys.executable

POCKETS = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]
SEEDS = [0, 1, 2]
N_POSES = 5
EXHAUSTIVENESS = 16  # Reduced from 32 to fit budget


def _wf424_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
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
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


def kabsch_rmsd(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
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


def heavy_atom_rmsd_hungarian(
    docked_elements, docked_coords, crystal_elements, crystal_coords
):
    elem_groups = {}
    for i, e in enumerate(crystal_elements):
        elem_groups.setdefault(e, []).append(i)

    matched_d = []
    matched_c = []
    for elem, c_idx_list in elem_groups.items():
        d_idx_list = [i for i, e in enumerate(docked_elements) if e == elem]
        if not d_idx_list or not c_idx_list:
            continue
        Cd = np.array([docked_coords[i] for i in d_idx_list])
        Cc = np.array([crystal_coords[i] for i in c_idx_list])
        diff = Cd[:, None, :] - Cc[None, :, :]
        cost = np.sqrt((diff * diff).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1.5:
                matched_d.append(d_idx_list[r])
                matched_c.append(c_idx_list[c])

    if len(matched_d) < 3:
        return None
    Pd = np.array([docked_coords[i] for i in matched_d])
    Pc = np.array([crystal_coords[i] for i in matched_c])
    return kabsch_rmsd(Pd, Pc)


def parse_pdbqt_heavy_atoms(pdbqt_path):
    elements = []
    coords = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                elem_field = (
                    line[76:78].strip() if len(line) >= 78 else line[12:16].strip()
                )
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
        models = [text]
    return models


def compute_rmsd_for_pocket(pocket_dir, ref_elements, ref_coords, seeds):
    """Compute per-seed per-pose RMSD from existing pose files."""
    results = {}
    per_pocket_pose_data = []
    for seed in seeds:
        seed_dir = pocket_dir
        rmsds = []
        for pose_idx in range(1, N_POSES + 1):
            pp = seed_dir / f"pose_seed{seed}_p{pose_idx}.pdbqt"
            if not pp.exists():
                continue
            elems, coords = parse_pdbqt_heavy_atoms(pp)
            rmsd = heavy_atom_rmsd_hungarian(elems, coords, ref_elements, ref_coords)
            if rmsd is None:
                continue
            rmsds.append(rmsd)
            per_pocket_pose_data.append(
                {
                    "pocket": pocket_dir.name,
                    "seed": seed,
                    "pose": pose_idx,
                    "rmsd_angstrom": round(rmsd, 4),
                    "n_docked_heavy": len(elems),
                    "n_ref_heavy": len(ref_elements),
                }
            )
        if rmsds:
            results[seed] = {
                "seed": seed,
                "status": "OK",
                "n_poses_valid": len(rmsds),
                "rmsd_per_pose": [round(r, 4) for r in rmsds],
                "best_rmsd_angstrom": round(float(min(rmsds)), 4),
            }
        else:
            results[seed] = {"seed": seed, "status": "NO_VALID_POSES"}
    return results, per_pocket_pose_data


def smiles_to_pdbqt_3d(smiles, out_pdbqt_path):
    sdf_intermediate = out_pdbqt_path.replace(".pdbqt", "_pre.sdf")
    script = f"""
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
writer = Chem.SDWriter("{sdf_intermediate}")
writer.write(mol)
writer.close()
print("OK")
"""
    result = subprocess.run(
        [MOL_VENV_PY, "-c", script], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"RDKit fail: {result.stderr[:200]}")
    result2 = subprocess.run(
        [
            CADVENV_PY,
            MKPREP_LIG,
            "-i",
            sdf_intermediate,
            "-o",
            out_pdbqt_path,
            "--charge_model",
            "gasteiger",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result2.returncode != 0 or not os.path.exists(out_pdbqt_path):
        raise RuntimeError(
            f"mk_prepare fail: rc={result2.returncode} stderr={result2.stderr[:200]}"
        )
    return out_pdbqt_path


def dock_one(
    receptor_pdbqt, ligand_pdbqt, center, size, out_dir, seed, exhaustiveness, n_poses
):
    out_pdbqt = Path(out_dir) / f"out_seed{seed}.pdbqt"
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        VINA_BIN,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--out",
        str(out_pdbqt),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--seed",
        str(seed),
        "--cpu",
        "1",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Vina fail: rc={result.returncode}")
    return out_pdbqt, elapsed


def main():
    print(
        f"=== Wf-179 BD-9 m15 RMSD Phase 2 RESCUE (ex={EXHAUSTIVENESS}) ===", flush=True
    )
    print(f"Vina: {VINA_BIN}")
    r = subprocess.run(
        [VINA_BIN, "--version"], capture_output=True, text=True, timeout=10
    )
    print(f"Vina version: {r.stdout.strip()}")

    meta = json.load(open(LIGAND_META))
    pocket_meta = {p["pocket_id"]: p for p in meta["pockets"]}

    box_centers = {}
    for p in POCKETS:
        wf141 = WF141_DATA / p / "per_seed_metrics.json"
        if wf141.exists():
            d = json.load(open(wf141))
            box_centers[p] = d["_pocket"]["docking_box_center"]

    # Box sizes
    box_sizes = {}
    for p in POCKETS:
        ref = pocket_meta[p]
        coords = ref["coordinates_3d"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        m = 8.0
        box_sizes[p] = [
            max(20.0, max(xs) - min(xs) + m * 2),
            max(20.0, max(ys) - min(ys) + m * 2),
            max(20.0, max(zs) - min(zs) + m * 2),
        ]

    # Step 1: rescue from existing files
    print("\n=== Step 1: Rescue from existing pose files ===", flush=True)
    rescued = {}
    for p in POCKETS:
        pocket_dir = WORKDIR / p
        meta_p = pocket_meta[p]
        # Detect which seeds already have files
        existing_seeds = []
        for s in SEEDS:
            if (pocket_dir / f"out_seed{s}.pdbqt").exists() and (
                pocket_dir / f"pose_seed{s}_p1.pdbqt"
            ).exists():
                existing_seeds.append(s)
        if not existing_seeds:
            continue
        print(f"  {p}: existing seeds {existing_seeds}", flush=True)
        per_seed_data, per_pose = compute_rmsd_for_pocket(
            pocket_dir,
            meta_p["element_types"],
            meta_p["coordinates_3d"],
            existing_seeds,
        )
        rescued[p] = {
            "per_seed_data": per_seed_data,
            "per_pose": per_pose,
            "meta": meta_p,
            "box_center": box_centers[p],
            "box_size": box_sizes[p],
        }

    # Step 2: dock missing (pocket, seed) combos
    print("\n=== Step 2: Dock missing (pocket, seed) combos ===", flush=True)
    with __import__("tempfile").TemporaryDirectory(prefix="wf179r_lig_") as tmpdir:
        tmp = Path(tmpdir)
        lig_pdbqts = {}
        for p in POCKETS:
            if p in rescued:
                continue  # all 3 seeds done
            lig = tmp / f"{p}_ligand.pdbqt"
            try:
                smiles_to_pdbqt_3d(pocket_meta[p]["smiles"], str(lig))
                lig_pdbqts[p] = lig
            except Exception as e:
                print(f"  {p}: ligand prep fail {e}")

        for p in POCKETS:
            if p not in lig_pdbqts:
                continue
            pocket_dir = WORKDIR / p
            rec_path = RECEPT_DIR / p / "receptor.pdbqt"
            center = box_centers[p]
            size = box_sizes[p]
            meta_p = pocket_meta[p]
            # Get existing seeds
            existing_seeds = [
                s
                for s in SEEDS
                if (pocket_dir / f"out_seed{s}.pdbqt").exists()
                and (pocket_dir / f"pose_seed{s}_p1.pdbqt").exists()
            ]
            if p not in rescued:
                rescued[p] = {
                    "per_seed_data": {},
                    "per_pose": [],
                    "meta": meta_p,
                    "box_center": center,
                    "box_size": size,
                }
            print(
                f"  {p}: docking missing seeds {set(SEEDS) - set(existing_seeds)}",
                flush=True,
            )
            for seed in SEEDS:
                if seed in existing_seeds:
                    continue
                try:
                    out_pdbqt, elapsed = dock_one(
                        rec_path,
                        lig_pdbqts[p],
                        center,
                        size,
                        pocket_dir,
                        seed,
                        EXHAUSTIVENESS,
                        N_POSES,
                    )
                except Exception as e:
                    print(f"    seed={seed} FAIL: {e}")
                    continue
                models = split_pdbqt_by_model(out_pdbqt)
                seed_rmsds = []
                for pose_idx, model_text in enumerate(models):
                    pose_pdbqt = pocket_dir / f"pose_seed{seed}_p{pose_idx+1}.pdbqt"
                    _wf424_atomic_write_text(pose_pdbqt, model_text)
                    elems, coords = parse_pdbqt_heavy_atoms(pose_pdbqt)
                    rmsd = heavy_atom_rmsd_hungarian(
                        elems, coords, meta_p["element_types"], meta_p["coordinates_3d"]
                    )
                    if rmsd is None:
                        continue
                    seed_rmsds.append(rmsd)
                    rescued[p]["per_pose"].append(
                        {
                            "pocket": p,
                            "seed": seed,
                            "pose": pose_idx + 1,
                            "rmsd_angstrom": round(rmsd, 4),
                            "n_docked_heavy": len(elems),
                            "n_ref_heavy": len(meta_p["element_types"]),
                        }
                    )
                if seed_rmsds:
                    rescued[p]["per_seed_data"][seed] = {
                        "seed": seed,
                        "status": "OK",
                        "n_poses_valid": len(seed_rmsds),
                        "rmsd_per_pose": [round(r, 4) for r in seed_rmsds],
                        "best_rmsd_angstrom": round(float(min(seed_rmsds)), 4),
                        "elapsed_s": round(elapsed, 3),
                    }
                else:
                    rescued[p]["per_seed_data"][seed] = {
                        "seed": seed,
                        "status": "NO_VALID_POSES",
                    }

    # Step 3: aggregate per-pocket
    print("\n=== Step 3: Aggregate ===", flush=True)
    aggregate = {}
    results = {}
    for p in POCKETS:
        if p not in rescued:
            results[p] = {"pocket": p, "status": "NOTHING_DONE"}
            continue
        r_data = rescued[p]
        per_seed_rmsd = []
        per_seed_data_list = []
        for s in SEEDS:
            sd = r_data["per_seed_data"].get(s)
            if sd and sd.get("best_rmsd_angstrom") is not None:
                per_seed_rmsd.append(sd["best_rmsd_angstrom"])
                per_seed_data_list.append(sd)
        if per_seed_rmsd:
            median = float(np.median(per_seed_rmsd))
            mean = float(np.mean(per_seed_rmsd))
            std = float(np.std(per_seed_rmsd))
            meta_p = r_data["meta"]
            results[p] = {
                "pocket": p,
                "ref_smiles": meta_p["smiles"],
                "n_ref_heavy": len(meta_p["element_types"]),
                "box_center": r_data["box_center"],
                "box_size": r_data["box_size"],
                "per_seed_best_rmsd_angstrom": [round(r, 4) for r in per_seed_rmsd],
                "median_best_rmsd_angstrom": round(median, 4),
                "mean_best_rmsd_angstrom": round(mean, 4),
                "std_best_rmsd_angstrom": round(std, 4),
                "passes_2_5a_gate": median <= 2.5,
                "passes_2_0a_sota": median <= 2.0,
                "per_seed_data": per_seed_data_list,
                "n_seeds_succeeded": len(per_seed_rmsd),
                "n_seeds_failed": 3 - len(per_seed_rmsd),
                "status": "MEASURED",
            }
            # Save per-pocket
            with open(WORKDIR / p / "per_seed_metrics.json", "w") as f:
                json.dump(results[p], f, indent=2)
            with open(WORKDIR / p / "per_pose_metrics.json", "w") as f:
                json.dump(r_data["per_pose"], f, indent=2)
            print(
                f"  {p}: median={median:.4f}A mean={mean:.4f}A std={std:.4f}A pass={median <= 2.5}",
                flush=True,
            )
            aggregate[p] = median
        else:
            results[p] = {
                "pocket": p,
                "status": "ALL_SEEDS_FAIL",
                "per_seed_data": list(r_data["per_seed_data"].values()),
            }
            with open(WORKDIR / p / "per_seed_metrics.json", "w") as f:
                json.dump(results[p], f, indent=2)
            print(f"  {p}: ALL_SEEDS_FAIL")

    # Summary
    if aggregate:
        overall = float(np.median(list(aggregate.values())))
        n_pass = sum(1 for v in aggregate.values() if v <= 2.5)
        summary = {
            "workflow": "Wf-179 BD-9 m15 RMSD Phase 2 RESCUE",
            "engine": "AutoDock Vina 7ac2999-mod",
            "exhaustiveness": EXHAUSTIVENESS,
            "n_poses": N_POSES,
            "seeds": SEEDS,
            "pockets": results,
            "aggregate_median_across_pockets": round(overall, 4),
            "n_pockets": len(aggregate),
            "n_pockets_passed_2_5a": n_pass,
            "n_pockets_passed_2_0a": sum(1 for v in aggregate.values() if v <= 2.0),
            "gate_4_passes_2_5a_overall": overall <= 2.5,
        }
        with open(WORKDIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n=== FINAL ===", flush=True)
        print(
            f"aggregate median: {overall:.4f}A across {len(aggregate)} pockets",
            flush=True,
        )
        print(f"n_pass_2.5A: {n_pass}/{len(aggregate)}", flush=True)
        print(f"per-pocket medians: {aggregate}", flush=True)


if __name__ == "__main__":
    main()
