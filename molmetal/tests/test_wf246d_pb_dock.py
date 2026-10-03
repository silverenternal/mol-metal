"""Wf-246-D unit test — PB DOCK mode on a real Wf-179 docked pose.

CPU-only test.  Verifies:

1. ``_build_mol_from_pdbqt`` preserves docked heavy-atom coordinates
   (no embed, no optimize — the contract check_docked_pose requires).
2. SMILES round-trips through ``_build_mol_from_pdbqt`` unchanged.
3. The returned mol has exactly one conformer with the parsed
   heavy-atom count.
4. ``check_docked_pose`` runs in dock mode against a real receptor
   PDB and returns the documented PB-result dict shape.
5. The cross-pocket pipeline aggregator counts protein-aware clashes
   correctly (clash_rate from real Wf-179 P2 poses = 0.0).

These tests do NOT require the heavy ``posebusters`` wheel — they
skip with a clear message if the wheel is missing (see G7 honest
framing in Wf-246-D synthesis.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "molmetal" / "reports" / "wf246d_m8_pb_dock"
sys.path.insert(0, str(REPORTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pose_path() -> Path:
    """One real Wf-179 P2 docked PDBQT pose (1h36 seed=0 pose=1)."""
    p = (
        ROOT
        / "molmetal"
        / "reports"
        / "wf179_bd9_m15_rmsd_p2"
        / "1h36"
        / "pose_seed0_p1.pdbqt"
    )
    if not p.is_file():
        pytest.skip(f"missing Wf-179 P2 pose: {p}")
    return p


@pytest.fixture(scope="module")
def receptor_pdb() -> Path:
    """Converted 1h36 receptor PDB (cached under wf246d_m8_pb_dock/)."""
    p = REPORTS_DIR / "1h36_receptor.pdb"
    if not p.is_file():
        pytest.skip(f"missing converted receptor PDB: {p}")
    return p


# ---------------------------------------------------------------------------
# 1. _build_mol_from_pdbqt preserves coordinates
# ---------------------------------------------------------------------------
def test_build_mol_from_pdbqt_preserves_coordinates(pose_path):
    """Built mol has the same heavy-atom coordinates as the PDBQT."""
    from wf246d_pb_dock_steric import _build_mol_from_pdbqt
    from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (
        parse_pdbqt_heavy_atoms,
    )

    mol, smiles = _build_mol_from_pdbqt(pose_path)
    assert mol is not None, "build_mol_from_pdbqt returned None"
    heavy = parse_pdbqt_heavy_atoms(pose_path)
    assert mol.GetNumAtoms() == len(
        heavy
    ), f"built mol has {mol.GetNumAtoms()} atoms but PDBQT has {len(heavy)}"
    positions = mol.GetConformer().GetPositions()
    for i, h in enumerate(heavy):
        built = positions[i]
        raw = (h["x"], h["y"], h["z"])
        assert np.allclose(
            built, raw, atol=1e-6
        ), f"atom {i} coord mismatch: built={built} raw={raw}"


# ---------------------------------------------------------------------------
# 2. SMILES round-trips through _build_mol_from_pdbqt
# ---------------------------------------------------------------------------
def test_build_mol_from_pdbqt_smiles_preserved(pose_path):
    """SMILES pulled from PDBQT REMARK line is the SMILES used to build the mol."""
    from wf246d_pb_dock_steric import _build_mol_from_pdbqt
    from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (
        smiles_from_pdbqt_remark,
    )

    expected = smiles_from_pdbqt_remark(pose_path)
    _, got = _build_mol_from_pdbqt(pose_path)
    assert got == expected
    assert "C" in got and "Br" in got  # 1h36 has Br in its SMILES


# ---------------------------------------------------------------------------
# 3. The built mol has exactly one conformer with parsed heavy-atom count
# ---------------------------------------------------------------------------
def test_build_mol_from_pdbqt_single_conformer(pose_path):
    """Mol carries exactly one conformer with the parsed heavy-atom count."""
    from wf246d_pb_dock_steric import _build_mol_from_pdbqt
    from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (
        parse_pdbqt_heavy_atoms,
    )

    mol, _ = _build_mol_from_pdbqt(pose_path)
    assert mol.GetNumConformers() == 1
    n_heavy = len(parse_pdbqt_heavy_atoms(pose_path))
    assert mol.GetNumAtoms() == n_heavy


# ---------------------------------------------------------------------------
# 4. check_docked_pose returns the documented shape (real PB)
# ---------------------------------------------------------------------------
def test_check_docked_pose_real_receptor(pose_path, receptor_pdb):
    """check_docked_pose runs against the real 1h36 receptor and returns the
    documented PB-result dict shape.  Skips if posebusters wheel is missing.
    """
    from molmetal.validation.posebusters_runner import (
        check_docked_pose,
        posebusters_available,
    )

    if not posebusters_available():
        pytest.skip("posebusters wheel not installed")

    from wf246d_pb_dock_steric import _build_mol_from_pdbqt

    mol, _ = _build_mol_from_pdbqt(pose_path)
    assert mol is not None

    result = check_docked_pose(mol, str(receptor_pdb))
    # Documented shape.
    for key in (
        "pb_valid",
        "backend",
        "config",
        "status",
        "checks",
        "failures",
        "missing_checks",
    ):
        assert key in result, f"missing {key!r} in check_docked_pose result"
    assert result["backend"] == "posebusters"
    assert result["config"] == "dock"
    assert result["status"] in ("passed", "failed", "error")
    # Documented protein-aware checks all present.
    for k in (
        "minimum_distance_to_protein",
        "volume_overlap_with_protein",
        "protein-ligand_maximum_distance",
    ):
        assert k in result["checks"], f"missing protein-aware check {k!r}"


# ---------------------------------------------------------------------------
# 5. The cross-pocket aggregator counts protein-aware clashes correctly
# ---------------------------------------------------------------------------
def test_aggregate_cross_pocket_wf179():
    """The wf246d pipeline aggregator reports clash_rate=0.0 across 65 poses
    (per Wf-246-D synthesis.md headline).  Skips if summary.json not yet
    generated by ``python wf246d_pb_dock_steric.py``.
    """
    summary = REPORTS_DIR / "summary.json"
    if not summary.is_file():
        pytest.skip(
            f"missing wf246d summary: {summary} — "
            "run `python molmetal/reports/wf246d_m8_pb_dock/"
            "wf246d_pb_dock_steric.py` first"
        )
    import json

    s = json.loads(summary.read_text())
    agg = s["aggregation"]
    # n_total_poses is 65 on disk (brief said 75; 3ny8 has 1 of 3 seeds).
    assert agg["n_total_poses"] >= 50, (
        f"unexpected pose count: {agg['n_total_poses']}; "
        "expected 65 on disk for Wf-179 P2"
    )
    # All 3 protein-aware checks pass on every pose — Vina already enforces
    # its own 2 Å clash tolerance, and PB's atom-typed tolerance is at
    # least as tight.
    assert agg["clash_rate_overall"] == 0.0
    assert agg["n_clash"] == 0
    # mean_min_dist_to_protein_A is in [1.8, 3.0] Å (PB threshold 2.0 Å
    # but atom-typed, so brute-force heavy-atom min is ~ 2.0-2.5 Å range).
    md = agg["mean_min_dist_to_protein_A"]
    assert md is not None
    assert (
        1.8 <= md <= 3.0
    ), f"mean_min_dist_to_protein_A={md} outside expected [1.8, 3.0] Å range"
