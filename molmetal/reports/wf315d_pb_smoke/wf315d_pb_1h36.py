#!/usr/bin/env python3
"""Wf-315D PoseBusters smoke retry on 1h36 (CPU-only).

Per WEAK_TO_STRONG §2.6: re-extract protein-aware steric clash metric
(m8) for pocket 1h36 using the on-disk PB runner.

Input:
  /home/hugo/codes/mol-metal/molmetal/reports/wf179_bd9_m15_rmsd_p2/1h36/pose_seed{0,1,2}_p{1..5}.pdbqt
  /home/hugo/codes/mol-metal/molmetal/reports/wf246d_m8_pb_dock/1h36_receptor.pdb

Output:
  /home/hugo/codes/mol-metal/molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
MOLMETAL_ROOT = REPORT_DIR.parent.parent  # /home/hugo/codes/mol-metal/molmetal
WF246D = MOLMETAL_ROOT / "reports" / "wf246d_m8_pb_dock"
POSE_DIR = MOLMETAL_ROOT / "reports" / "wf179_bd9_m15_rmsd_p2" / "1h36"
RECEPTOR_PDB = WF246D / "1h36_receptor.pdb"

# Add molmetal package root to sys.path so we can import molmetal.validation.posebusters_runner
sys.path.insert(0, str(MOLMETAL_ROOT.parent))  # /home/hugo/codes/mol-metal

# Sanity check
assert POSE_DIR.exists(), f"missing pose dir: {POSE_DIR}"
assert RECEPTOR_PDB.exists(), f"missing receptor: {RECEPTOR_PDB}"

# Discover poses
pose_paths = sorted(POSE_DIR.glob("pose_seed*_p*.pdbqt"))
print(f"[wf315d] found {len(pose_paths)} poses in {POSE_DIR}")

# Run PB via the same module Wf-246D used
try:
    from molmetal.validation.posebusters_runner import check_docked_pose
except Exception as e:
    print(f"[wf315d] FATAL: cannot import check_docked_pose: {e}")
    sys.exit(2)

results = []
n_total = 0
n_pb_pass_protein_aware = 0
n_pb_pass_full = 0
wall_total = 0.0
for pdbqt in pose_paths:
    n_total += 1
    t0 = time.time()
    try:
        # Build RDKit Mol from PDBQT coordinates (preserve dock pose)
        # The runner expects (mol, receptor_pdb) where mol is a docked RDKit Mol
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (
            parse_pdbqt_heavy_atoms,
            smiles_from_pdbqt_remark,
        )

        smiles = smiles_from_pdbqt_remark(str(pdbqt))
        heavy_atoms = parse_pdbqt_heavy_atoms(str(pdbqt))
        if not heavy_atoms:
            raise ValueError("no heavy atoms parsed from PDBQT")
        # Sanitize-from-SMILES so the bond graph is correct (PB chemistry checks require a valid mol)
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        template = Chem.AddHs(template)
        n_template = template.GetNumAtoms()
        if n_template != len(heavy_atoms):
            template = Chem.RemoveHs(template)
            n_template = template.GetNumAtoms()
            if n_template != len(heavy_atoms):
                raise ValueError(
                    f"heavy-atom count mismatch: PDBQT={len(heavy_atoms)} SMILES(no H)={n_template}"
                )
        # Build a fresh conformer from the parsed coordinates (preserve dock pose)
        conf = Chem.Conformer(n_template)
        conf.Set3D(True)
        for i, atom in enumerate(heavy_atoms):
            x = float(atom["x"])
            y = float(atom["y"])
            z = float(atom["z"])
            conf.SetAtomPosition(i, (x, y, z))
        template.RemoveAllConformers()
        template.AddConformer(conf, assignId=True)
        mol_for_pb = template

        result = check_docked_pose(mol_for_pb, str(RECEPTOR_PDB))
        wall = time.time() - t0
        wall_total += wall

        # result is a dict with 'checks' (Dict[str, bool]) and 'pb_valid' (bool)
        checks = result.get("checks", {}) if isinstance(result, dict) else {}
        # PB protein-aware checks (the steric clash subset)
        protein_aware_keys = [
            "protein-ligand_maximum_distance",
            "minimum_distance_to_protein",
            "minimum_distance_to_organic_cofactors",
            "minimum_distance_to_inorganic_cofactors",
            "minimum_distance_to_waters",
            "volume_overlap_with_protein",
            "volume_overlap_with_organic_cofactors",
            "volume_overlap_with_inorganic_cofactors",
            "volume_overlap_with_waters",
        ]
        n_pa_pass = sum(bool(checks.get(k, False)) for k in protein_aware_keys)
        all_pa_pass = n_pa_pass == len(protein_aware_keys) and len(protein_aware_keys) > 0
        if all_pa_pass:
            n_pb_pass_protein_aware += 1

        # Full PB pass rate: all checks True
        all_checks = list(checks.values())
        n_pass = sum(1 for v in all_checks if v is True)
        n_total_checks = len(all_checks)
        all_pass = n_pass == n_total_checks and n_total_checks > 0
        if all_pass:
            n_pb_pass_full += 1

        results.append({
            "pdbqt": str(pdbqt),
            "smiles": smiles,
            "n_heavy": len(heavy_atoms),
            "wall_seconds": round(wall, 4),
            "pb_valid": bool(result.get("pb_valid", False)) if isinstance(result, dict) else False,
            "status": result.get("status") if isinstance(result, dict) else None,
            "failures": result.get("failures", []) if isinstance(result, dict) else [],
            "checks": {k: bool(v) for k, v in checks.items()},
            "protein_aware_pass": all_pa_pass,
            "n_protein_aware_pass": n_pa_pass,
            "n_protein_aware_total": len(protein_aware_keys),
            "full_pass": all_pass,
            "n_checks_pass": n_pass,
            "n_checks_total": n_total_checks,
            "version": result.get("version") if isinstance(result, dict) else None,
        })
    except Exception as e:
        wall = time.time() - t0
        wall_total += wall
        results.append({
            "pdbqt": str(pdbqt),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "wall_seconds": round(wall, 4),
        })

# Aggregate
m8_protein_aware_pass_rate = n_pb_pass_protein_aware / max(n_total, 1)
m8_full_pass_rate = n_pb_pass_full / max(n_total, 1)

output = {
    "workflow": "Wf-315D PoseBusters smoke retry on 1h36 (CPU-only, m8 protein-aware subset)",
    "engine": f"PoseBusters 0.6.5 (config=dock, real conformers) + RDKit {__import__('rdkit').__version__}",
    "input_dir": str(POSE_DIR),
    "receptor_pdb": str(RECEPTOR_PDB),
    "n_poses": n_total,
    "n_protein_aware_pass": n_pb_pass_protein_aware,
    "n_full_pass": n_pb_pass_full,
    "m8_protein_aware_pass_rate": m8_protein_aware_pass_rate,
    "m8_full_pass_rate": m8_full_pass_rate,
    "wall_seconds_total": round(wall_total, 2),
    "results": results,
}

OUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "molmetal"
    / "reports"
    / "wf315d_pb_smoke"
    / "m8_prod_1h36.json"
)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"[wf315d] wrote {OUT_PATH}")
print(f"[wf315d] n_poses={n_total} n_pa_pass={n_pb_pass_protein_aware} n_full_pass={n_pb_pass_full}")
print(f"[wf315d] m8_protein_aware_pass_rate={m8_protein_aware_pass_rate:.4f}")
print(f"[wf315d] m8_full_pass_rate={m8_full_pass_rate:.4f}")