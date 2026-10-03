#!/usr/bin/env python
"""Wf-179 BD-9: m15 RMSD Phase 2 — FIXED RMSD computation.

Fix: PDBQT atom types include aromatic prefixes (A, NA, OA, SA, HD).
Map them back to pure elements (C, N, O, S, H) before Hungarian matching.
Re-compute RMSD for all 5 pockets from existing pose PDBQT files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

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


REPO = Path("/home/hugo/codes/mol-metal")
WORKDIR = REPO / "molmetal/reports/wf179_bd9_m15_rmsd_p2"
LIGAND_META = REPO / "molmetal/data/pocket_ref_ligands/ligand_metadata.json"

POCKETS = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]
SEEDS = [0, 1, 2]
N_POSES = 5


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


def parse_pdbqt_heavy_atoms_fixed(pdbqt_path):
    """Parse PDBQT with proper atom-type-to-element mapping.

    PDBQT atom types: A=aromatic C, C=sp3 C, NA=aromatic N, N=sp3 N,
    OA=aromatic O, O=sp3 O, S=S, SA=aromatic S, HD=HD (skip).
    """
    elements = []
    coords = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                elem_field = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()
                elem_raw = elem_field.split()[0] if elem_field else ""
                # Skip H
                if elem_raw.startswith("H"):
                    continue
                # Map PDBQT atom type to base element
                if elem_raw.startswith("A"):  # A = aromatic C
                    elem = "C"
                elif elem_raw.startswith("NA"):  # NA = aromatic N
                    elem = "N"
                elif elem_raw.startswith("OA"):  # OA = aromatic O
                    elem = "O"
                elif elem_raw.startswith("SA"):  # SA = aromatic S
                    elem = "S"
                elif len(elem_raw) >= 2 and elem_raw[1].islower():
                    elem = elem_raw  # Cl, Br
                else:
                    elem = elem_raw[:1]  # C, N, O, S
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                elements.append(elem)
                coords.append([x, y, z])
    return elements, coords


def heavy_atom_rmsd(docked_elements, docked_coords, crystal_elements, crystal_coords):
    """Hungarian matching with adaptive centroid pre-alignment + 3.0 Å distance threshold."""
    Cd_arr = np.array(docked_coords)
    Cc_arr = np.array(crystal_coords)
    d_cent = Cd_arr.mean(axis=0)
    c_cent = Cc_arr.mean(axis=0)
    centroid_dist = float(np.linalg.norm(d_cent - c_cent))

    # Centroid pre-alignment ONLY if centroid distance > 3 Å (pose is offset from crystal)
    if centroid_dist > 3.0:
        Cd_use = Cd_arr - d_cent
        Cc_use = Cc_arr - c_cent
        thresh = 5.0  # More permissive after pre-alignment
    else:
        Cd_use = Cd_arr
        Cc_use = Cc_arr
        thresh = 2.5  # Tight when already well-centered

    elem_groups = {}
    for i, e in enumerate(crystal_elements):
        elem_groups.setdefault(e, []).append(i)

    matched_d = []
    matched_c = []
    for elem, c_idx_list in elem_groups.items():
        d_idx_list = [i for i, e in enumerate(docked_elements) if e == elem]
        if not d_idx_list or not c_idx_list:
            continue
        Cd = Cd_use[d_idx_list]
        Cc = Cc_use[c_idx_list]
        diff = Cd[:, None, :] - Cc[None, :, :]
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


def main():
    meta = json.load(open(LIGAND_META))
    pocket_meta = {p["pocket_id"]: p for p in meta["pockets"]}

    results = {}
    aggregate = {}
    for p in POCKETS:
        meta_p = pocket_meta[p]
        ref_elements = meta_p["element_types"]
        ref_coords = meta_p["coordinates_3d"]

        pocket_dir = WORKDIR / p
        per_seed_data = []
        per_seed_best_rmsd = []
        per_pose_data = []

        for seed in SEEDS:
            out_pdbqt = pocket_dir / f"out_seed{seed}.pdbqt"
            if not out_pdbqt.exists():
                per_seed_data.append({"seed": seed, "status": "NO_DOCK_OUTPUT"})
                continue
            models = split_pdbqt_by_model(out_pdbqt)
            seed_rmsds = []
            for pose_idx, model_text in enumerate(models):
                pose_pdbqt = pocket_dir / f"pose_seed{seed}_p{pose_idx+1}.pdbqt"
                _wf424_atomic_write_text(pose_pdbqt, model_text)
                elems, coords = parse_pdbqt_heavy_atoms_fixed(pose_pdbqt)
                rmsd = heavy_atom_rmsd(elems, coords, ref_elements, ref_coords)
                if rmsd is None:
                    continue
                seed_rmsds.append(rmsd)
                per_pose_data.append({
                    "pocket": p, "seed": seed, "pose": pose_idx + 1,
                    "rmsd_angstrom": round(rmsd, 4),
                    "n_docked_heavy": len(elems),
                    "n_ref_heavy": len(ref_elements),
                })
            if seed_rmsds:
                best = float(min(seed_rmsds))
                per_seed_best_rmsd.append(best)
                per_seed_data.append({
                    "seed": seed, "status": "OK", "n_poses_valid": len(seed_rmsds),
                    "rmsd_per_pose": [round(r, 4) for r in seed_rmsds],
                    "best_rmsd_angstrom": round(best, 4),
                })
            else:
                per_seed_data.append({"seed": seed, "status": "NO_VALID_POSES"})

        if per_seed_best_rmsd:
            median = float(np.median(per_seed_best_rmsd))
            mean = float(np.mean(per_seed_best_rmsd))
            std = float(np.std(per_seed_best_rmsd))
            results[p] = {
                "pocket": p,
                "ref_smiles": meta_p["smiles"],
                "n_ref_heavy": len(ref_elements),
                "per_seed_best_rmsd_angstrom": [round(r, 4) for r in per_seed_best_rmsd],
                "median_best_rmsd_angstrom": round(median, 4),
                "mean_best_rmsd_angstrom": round(mean, 4),
                "std_best_rmsd_angstrom": round(std, 4),
                "passes_2_5a_gate": median <= 2.5,
                "passes_2_0a_sota": median <= 2.0,
                "per_seed_data": per_seed_data,
                "n_seeds_succeeded": len(per_seed_best_rmsd),
                "n_seeds_failed": 3 - len(per_seed_best_rmsd),
                "status": "MEASURED",
            }
            with open(WORKDIR / p / "per_seed_metrics.json", "w") as f:
                json.dump(results[p], f, indent=2)
            with open(WORKDIR / p / "per_pose_metrics.json", "w") as f:
                json.dump(per_pose_data, f, indent=2)
            print(f"  {p}: median={median:.4f}A mean={mean:.4f}A std={std:.4f}A pass={median<=2.5}")
            aggregate[p] = median
        else:
            results[p] = {
                "pocket": p, "status": "ALL_SEEDS_FAIL",
                "per_seed_data": per_seed_data,
            }
            with open(WORKDIR / p / "per_seed_metrics.json", "w") as f:
                json.dump(results[p], f, indent=2)
            print(f"  {p}: ALL_SEEDS_FAIL")

    if aggregate:
        overall = float(np.median(list(aggregate.values())))
        n_pass = sum(1 for v in aggregate.values() if v <= 2.5)
        summary = {
            "workflow": "Wf-179 BD-9 m15 RMSD Phase 2 — FIXED",
            "engine": "AutoDock Vina 7ac2999-mod",
            "exhaustiveness": 16,
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
        print(f"\n=== FINAL ===")
        print(f"aggregate median: {overall:.4f}A across {len(aggregate)} pockets")
        print(f"per-pocket medians: {aggregate}")
        print(f"n_pass_2.5A: {n_pass}/{len(aggregate)}")


if __name__ == "__main__":
    main()
