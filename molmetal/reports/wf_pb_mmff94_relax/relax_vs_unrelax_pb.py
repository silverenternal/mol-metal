"""Standalone paired-statistic comparison: PB pass rate with vs without MMFF94s.

Reads the docked SDF saved by the baseline 1-pocket WF-PB-Pass-Real-Dock
report, runs PoseBusters ``dock`` mode on:

1. The un-relaxed Vina pose (baseline WF-PB-Pass-Real-Dock behaviour).
2. The MMFF94s-relaxed Vina pose (WF-PB-MMFF94-Relax).

Reports the per-check pass counts and a paired statistic so the spec's
``test_mmff94s_relax_improves_pb_score`` contract has real measured data
rather than only synthetic panel data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Baseline 1-pocket docked SDF (Vina output, no MMFF94s relaxation).
BASELINE_SDF = PROJECT_ROOT / "molmetal" / "reports" / "wf_pb_pass_real_dock" / "r4c_poses" / "533128de8a6c1d0d" / "candidate_0000.sdf"
RECEPTOR_PDB = PROJECT_ROOT / "molmetal" / "references" / "Pocket2Mol" / "example" / "4yhj.pdb"

# Also try to use the same receptor as the baseline.
BASELINE_RECEPTOR = PROJECT_ROOT / "molmetal" / "reports" / "wf_pb_pass_real_dock" / "r4c_poses" / "533128de8a6c1d0d" / "receptor" / "selected" / "receptor.pdb"

OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf_pb_mmff94_relax" / "relax_vs_unrelax.json"


def main():
    if not BASELINE_SDF.exists():
        print(f"baseline SDF missing: {BASELINE_SDF}", file=sys.stderr)
        sys.exit(2)
    # Use the baseline receptor if available; else fall back to bundled 4yhj.
    receptor = str(BASELINE_RECEPTOR if BASELINE_RECEPTOR.exists() else RECEPTOR_PDB)
    print(f"receptor: {receptor}")

    # Load the docked pose.
    suppl = Chem.SDMolSupplier(str(BASELINE_SDF), removeHs=False)
    poses = [m for m in suppl if m is not None]
    print(f"loaded {len(poses)} docked poses from {BASELINE_SDF}")
    if not poses:
        sys.exit(2)
    pose = poses[0]
    n_atoms = pose.GetNumAtoms()
    print(f"  pose[0]: {n_atoms} atoms, 1 conformer={pose.GetNumConformers() == 1}")

    # --- Pre-relax: PoseBusters dock mode on the un-relaxed Vina pose. ---
    from molmetal.validation.posebusters_runner import check_docked_pose
    pre_relax_pb = check_docked_pose(pose, receptor)
    print(f"\nPRE-RELAX PoseBusters dock:")
    print(f"  status: {pre_relax_pb.get('status')}")
    print(f"  pb_valid: {pre_relax_pb.get('pb_valid')}")
    print(f"  failed: {pre_relax_pb.get('failures')}")

    # --- Post-relax: MMFF94s on a clone, then PB on the relaxed pose. ---
    pose_for_relax = Chem.Mol(pose)
    from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import mmff94s_relax_pose
    relax_report = mmff94s_relax_pose(pose_for_relax, max_iters=200)
    print(f"\nMMFF94s relax:")
    print(f"  ok: {relax_report.get('ok')}")
    print(f"  status: {relax_report.get('status')}")
    print(f"  variant: {relax_report.get('variant')}")
    print(f"  error: {relax_report.get('error')}")

    # Compute heavy-atom RMSD pre vs post.
    try:
        rmsd = AllChem.GetBestRMS(pose, pose_for_relax, prbId=0, refId=0)
    except Exception as exc:
        rmsd = None
    print(f"  heavy-atom RMSD pre/post: {rmsd}")

    if relax_report.get("ok"):
        post_relax_pb = check_docked_pose(pose_for_relax, receptor)
        print(f"\nPOST-RELAX PoseBusters dock:")
        print(f"  status: {post_relax_pb.get('status')}")
        print(f"  pb_valid: {post_relax_pb.get('pb_valid')}")
        print(f"  failed: {post_relax_pb.get('failures')}")
    else:
        post_relax_pb = None
        print("\nPOST-RELAX skipped: relax failed")

    out = {
        "baseline_sdf": str(BASELINE_SDF),
        "receptor_pdb": receptor,
        "pre_relax_pb_valid": pre_relax_pb.get("pb_valid"),
        "pre_relax_status": pre_relax_pb.get("status"),
        "pre_relax_failures": pre_relax_pb.get("failures"),
        "pre_relax_n_passed": sum(1 for v in (pre_relax_pb.get("checks") or {}).values() if v is True),
        "pre_relax_n_checks": sum(1 for v in (pre_relax_pb.get("checks") or {}).values() if v is not None),
        "mmff94s_relax": relax_report,
        "heavy_atom_rmsd_pre_post_A": rmsd,
        "post_relax_pb_valid": post_relax_pb.get("pb_valid") if post_relax_pb else None,
        "post_relax_status": post_relax_pb.get("status") if post_relax_pb else None,
        "post_relax_failures": post_relax_pb.get("failures") if post_relax_pb else None,
        "post_relax_n_passed": sum(1 for v in (post_relax_pb.get("checks") or {}).values() if v is True) if post_relax_pb else None,
        "post_relax_n_checks": sum(1 for v in (post_relax_pb.get("checks") or {}).values() if v is not None) if post_relax_pb else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {OUT}")

    # Paired statistic for the spec.
    pre_passed = out["pre_relax_n_passed"]
    post_passed = out["post_relax_n_passed"]
    pre_total = out["pre_relax_n_checks"]
    post_total = out["post_relax_n_checks"]
    print(f"\nPaired statistic (single docked pose):")
    print(f"  pre  pass: {pre_passed}/{pre_total}")
    print(f"  post pass: {post_passed}/{post_total}")
    if pre_passed is not None and post_passed is not None:
        delta = post_passed - pre_passed
        print(f"  delta: {delta:+d}")
        if post_passed >= pre_passed:
            print(f"  PASS: post-relax PB checks >= pre-relax (contract satisfied).")
        else:
            print(f"  FAIL: post-relax PB checks < pre-relax.")


if __name__ == "__main__":
    main()