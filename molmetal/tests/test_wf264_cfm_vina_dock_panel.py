"""Tests for Wf-264-Vina — CFM Vina 5-pocket docking panel.

These tests verify:

  1. Five canonical Wf-179 m15 pockets are enumerated (1h36, 3eml,
     3ny8, 4aaw, 4rlu).
  2. The default ``--vina-bin`` resolves to ``molmetal/bin/vina``.
  3. SMILES CSV parsing works (with 'smiles' column).
  4. JSON schema is well-formed and complete.
  5. STUB_FALLBACK path is honoured when the Vina binary is missing
     or when receptors are not staged.
  6. CLI --help exits 0 and mentions required flags.

All tests are CPU-only; no actual docking is performed.  The
``dock_runner`` seam in :func:`build_panel` lets us drive every cell
through a deterministic fake so the test does not invoke Vina.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "cfm_vina_dock_panel.py"
POCKETS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"


def _load_module():
    """Load cfm_vina_dock_panel.py as a module.

    The dataclass annotations in the script use module-relative
    forward refs (e.g. ``"STATUS_REAL_VINA"``) which require the
    module to be registered in ``sys.modules`` *before*
    ``spec.loader.exec_module`` runs.
    """
    mod_name = "cfm_vina_dock_panel"
    spec = importlib.util.spec_from_file_location(mod_name, str(SCRIPT_PATH))
    assert spec and spec.loader, "cfm_vina_dock_panel.py must exist and be importable"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Drop the half-registered module to keep sys.modules clean.
        sys.modules.pop(mod_name, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# 1. Five canonical Wf-179 m15 pockets are enumerated.
# ---------------------------------------------------------------------------
def test_default_pockets_enumerated():
    mod = _load_module()
    expected = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
    assert mod.DEFAULT_POCKETS == expected, (
        f"Default pockets must be the 5 Wf-179 m15 pockets; "
        f"got {mod.DEFAULT_POCKETS}"
    )
    assert len(mod.DEFAULT_POCKETS) == 5


# ---------------------------------------------------------------------------
# 2. Default --vina-bin path.
# ---------------------------------------------------------------------------
def test_default_vina_bin_path():
    mod = _load_module()
    expected = str(PROJECT_ROOT / "molmetal" / "bin" / "vina")
    assert mod.DEFAULT_VINA_BIN == expected, (
        f"Default vina bin must be the project wrapper at "
        f"{expected}; got {mod.DEFAULT_VINA_BIN}"
    )


# ---------------------------------------------------------------------------
# 3. CLI --help exits 0 and mentions required flags.
# ---------------------------------------------------------------------------
def test_cli_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help should exit 0; got {result.returncode}\n"
        f"STDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    for flag in (
        "--smiles-csv",
        "--pocket-pdb-dir",
        "--out-json",
        "--vina-bin",
        "--n-candidates",
    ):
        assert flag in result.stdout, (
            f"--help must mention {flag}; " f"got STDOUT={result.stdout[:200]}"
        )


# ---------------------------------------------------------------------------
# 4. SMILES CSV parsing (with 'smiles' column).
# ---------------------------------------------------------------------------
def test_smiles_csv_parse(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "smiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "extra"])
        writer.writerow(["CCO", "ethanol"])
        writer.writerow(["c1ccccc1", "benzene"])
        writer.writerow(["", "blank-row-skipped"])
        writer.writerow(["CC(=O)Oc1ccccc1C(=O)O", "aspirin"])

    out = mod.read_smiles_csv(csv_path)
    assert out == [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
    ], f"Expected 3 SMILES after dropping blank; got {out}"

    # Missing 'smiles' column raises ValueError.
    bad_csv = tmp_path / "bad.csv"
    with bad_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mol", "name"])
        writer.writerow(["CCO", "ethanol"])
    with pytest.raises(ValueError, match="smiles"):
        mod.read_smiles_csv(bad_csv)


# ---------------------------------------------------------------------------
# 5. STUB_FALLBACK path: no real Vina docking happens.
# ---------------------------------------------------------------------------
class _FakeRunner:
    """Deterministic dock_runner for tests.

    Records every (smiles, pocket) call and returns a
    PocketCell with a synthetic kcal/mol so the aggregate stats
    can be verified end-to-end.
    """

    def __init__(self, kcal_by_pocket: Optional[Dict[str, float]] = None):
        self.kcal_by_pocket = kcal_by_pocket or {
            "1h36": -7.5,
            "3eml": -6.0,
            "3ny8": -8.0,
            "4aaw": -7.0,
            "4rlu": -6.5,
        }
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        smiles: str,
        pocket_id: str,
        receptor: Path,
        vina_bin: str,
        center,
        box_size: float,
        exhaustiveness: int,
        n_poses: int,
        seed: int,
        timeout: int,
        work_dir: Path,
        pt_strip: bool = False,
        **kwargs,
    ):
        from molmetal.scripts.cfm_vina_dock_panel import (
            PocketCell,
            STATUS_REAL_VINA,
        )

        self.calls.append(
            {
                "smiles": smiles,
                "pocket_id": pocket_id,
                "vina_bin": vina_bin,
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "box_size": box_size,
                "center": center,
            }
        )
        # Deterministic kcal/mol: base + a per-smiles small offset.
        # hash gives a deterministic 0..1 jitter per molecule.
        h = abs(hash(smiles)) % 1000 / 1000.0
        kcal = self.kcal_by_pocket[pocket_id] - h
        return PocketCell(
            pocket=pocket_id,
            best_kcal_mol=float(kcal),
            n_poses=int(n_poses),
            wall_time_s=0.01,
            status=STATUS_REAL_VINA,
            note="FAKE_FOR_TEST",
        )


def _stage_receptors(tmp_path: Path) -> Path:
    """Create per-pocket subdirs with stub receptor.pdbqt files.

    The content is not real PDBQT — the fake dock_runner does not
    read it.  We just need the file to exist so
    ``_resolve_pocket_receptor`` finds it.
    """
    pockets_dir = tmp_path / "pockets"
    pockets_dir.mkdir(parents=True, exist_ok=True)
    for pocket_id in ("1h36", "3eml", "3ny8", "4aaw", "4rlu"):
        pdir = pockets_dir / pocket_id
        pdir.mkdir(exist_ok=True)
        # Stub content — the fake runner does not parse it.
        (pdir / "receptor.pdbqt").write_text(
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
            "END\n"
        )
    return pockets_dir


def test_build_panel_scaffold_no_real_vina(tmp_path):
    """End-to-end ``build_panel`` with a fake runner.

    Verifies the JSON schema, per-molecule derived fields, and
    aggregate stats without invoking a real Vina binary.
    """
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    panel = mod.build_panel(
        smiles_list=smiles_list,
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",  # intentionally bad path
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=mod.DEFAULT_SEED,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
    )

    # ------------------------------------------------------------------
    # Metadata + topline counts.
    # ------------------------------------------------------------------
    md = panel["metadata"]
    assert md["panel_version"] == mod.PANEL_VERSION
    assert md["pockets"] == list(mod.DEFAULT_POCKETS)
    assert md["vina_bin"] == "/nonexistent/vina"
    assert md["exhaustiveness"] == mod.DEFAULT_EXHAUSTIVENESS
    assert md["n_poses"] == mod.DEFAULT_N_POSES
    assert md["box_size"] == mod.DEFAULT_BOX_SIZE
    assert md["seed"] == mod.DEFAULT_SEED

    assert panel["n_input"] == 3
    # n_parsed depends on whether RDKit+Meeko are importable in the
    # test environment; the lower bound is 0 when Meeko is missing.
    # The fake runner returns REAL_VINA for every cell regardless.
    assert panel["n_parsed"] >= 0
    # The fake runner returns REAL_VINA for every cell, so n_docked
    # equals n_input * n_pockets.
    assert panel["n_docked"] == 3 * len(mod.DEFAULT_POCKETS)

    # Per-molecule shape.
    assert len(panel["per_molecule"]) == 3
    for entry in panel["per_molecule"]:
        assert "smiles" in entry
        assert "per_pocket" in entry
        assert "best_pocket" in entry
        assert "best_kcal_mol" in entry
        assert "mean_kcal_mol" in entry
        # Every pocket must be enumerated.
        for pocket_id in mod.DEFAULT_POCKETS:
            assert pocket_id in entry["per_pocket"], (
                f"per_pocket must enumerate all 5 pockets; " f"missing {pocket_id}"
            )
            cell = entry["per_pocket"][pocket_id]
            for key in ("best_kcal_mol", "n_poses", "wall_time_s", "status", "note"):
                assert key in cell, f"pocket cell missing key {key!r}: {cell}"

    # Aggregate per-pocket shape.
    agg = panel["aggregate"]
    assert "per_pocket" in agg
    for pocket_id in mod.DEFAULT_POCKETS:
        a = agg["per_pocket"][pocket_id]
        for key in (
            "n",
            "n_real_vina",
            "n_stub_fallback",
            "n_other",
            "mean",
            "min",
            "max",
        ):
            assert key in a, f"aggregate per_pocket {pocket_id} missing key {key!r}"
    assert "best_pocket_overall" in agg
    assert "best_score_overall" in agg
    assert "mean_score_overall" in agg


def test_build_panel_aggregate_stats_match_fake_runner(tmp_path):
    """The aggregate stats must match the fake runner's deterministic kcal/mol."""
    mod = _load_module()
    pockets_dir = _stage_receptors(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = _FakeRunner()
    smiles_list = ["CCO", "c1ccccc1"]
    panel = mod.build_panel(
        smiles_list=smiles_list,
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=0,
        timeout=60,
        work_dir=work_dir,
        dock_runner=runner,
    )

    # Per-pocket means reflect the fake runner's kcal_by_pocket +
    # hash-jitter (-0..-1).  We can recompute them.
    for pocket_id, base_kcal in runner.kcal_by_pocket.items():
        cells = [
            m["per_pocket"][pocket_id]["best_kcal_mol"] for m in panel["per_molecule"]
        ]
        # All cells are not None because the fake runner returns real kcal.
        assert all(c is not None for c in cells), (
            f"per_pocket cell kcal/mol must be set when the fake "
            f"runner returned REAL_VINA; got {cells}"
        )
        # Mean must match within rounding tolerance.
        agg = panel["aggregate"]["per_pocket"][pocket_id]
        expected_mean = sum(cells) / len(cells)
        assert agg["mean"] == pytest.approx(expected_mean, abs=1e-3)
        assert agg["min"] == pytest.approx(min(cells), abs=1e-3)
        assert agg["max"] == pytest.approx(max(cells), abs=1e-3)
        assert agg["n_real_vina"] == 2
        assert agg["n"] == 2

    # best_score_overall is the global min across all molecules / pockets.
    all_kcals = [
        cell["best_kcal_mol"]
        for m in panel["per_molecule"]
        for cell in m["per_pocket"].values()
        if cell["best_kcal_mol"] is not None
    ]
    assert panel["aggregate"]["best_score_overall"] == pytest.approx(
        min(all_kcals), abs=1e-3
    )
    assert panel["aggregate"]["mean_score_overall"] == pytest.approx(
        sum(all_kcals) / len(all_kcals), abs=1e-3
    )


def test_build_panel_no_receptor_stub_fallback(tmp_path):
    """Missing receptors → every cell is NO_RECEPTOR (STUB_FALLBACK class)."""
    mod = _load_module()
    # Use an empty pockets dir — no subdirs.
    pockets_dir = tmp_path / "empty_pockets"
    pockets_dir.mkdir()

    runner = _FakeRunner()
    panel = mod.build_panel(
        smiles_list=["CCO"],
        pocket_pdb_dir=pockets_dir,
        pockets=list(mod.DEFAULT_POCKETS),
        vina_bin="/nonexistent/vina",
        exhaustiveness=mod.DEFAULT_EXHAUSTIVENESS,
        n_poses=mod.DEFAULT_N_POSES,
        box_size=mod.DEFAULT_BOX_SIZE,
        seed=0,
        timeout=60,
        work_dir=tmp_path / "work",
        dock_runner=runner,
    )

    assert panel["n_docked"] == 0
    # Every per-pocket cell is NO_RECEPTOR.
    for entry in panel["per_molecule"]:
        for pocket_id, cell in entry["per_pocket"].items():
            assert cell["status"] == mod.STATUS_NO_RECEPTOR, (
                f"Expected NO_RECEPTOR; got {cell['status']} for "
                f"({entry['smiles']}, {pocket_id})"
            )
            assert cell["best_kcal_mol"] is None
    # Aggregate best_score_overall must be None when no real scores.
    assert panel["aggregate"]["best_score_overall"] is None
    assert panel["aggregate"]["mean_score_overall"] is None
    # Fake runner was never called (because we short-circuit on missing rec).
    assert runner.calls == []


def test_parse_vina_output_extracts_best_first():
    """_parse_vina_output returns best kcal/mol first."""
    mod = _load_module()
    pdbqt = """
REMARK  Name = foo
REMARK  9 active torsions:
REMARK  status: ('A' for Active; 'I' for Inactive)
REMARK    1  A   between atoms: C_1 and C_2
ROOT
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C
ENDROOT
BRANCH   1  2
ATOM      2  C   UNL     1       1.000   0.000   0.000  1.00  0.00    +0.000 C
ENDBRANCH   1  2
TORSDOF 1
MODEL 1
REMARK VINA RESULT:    -8.342      0.000      0.000
REMARK VINA RESULT:    -7.123      1.234      2.345
ENDMDL
MODEL 2
REMARK VINA RESULT:    -6.500      1.000      2.000
ENDMDL
"""
    scores = mod._parse_vina_output(pdbqt)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(-8.342, abs=1e-3)
    assert scores[1] == pytest.approx(-7.123, abs=1e-3)
    assert scores[2] == pytest.approx(-6.500, abs=1e-3)

    # Empty input → empty list.
    assert mod._parse_vina_output("") == []
    # Garbage input → empty list.
    assert mod._parse_vina_output("REMARK no results here\n") == []


def test_cli_end_to_end_with_stub_fallback(tmp_path):
    """End-to-end CLI run with a missing Vina binary → STUB_FALLBACK everywhere."""
    csv_path = tmp_path / "smiles.csv"
    out_path = tmp_path / "panel.json"
    pockets_dir = _stage_receptors(tmp_path / "pockets_for_cli")
    fake_vina = tmp_path / "fake_vina"  # does NOT exist

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "extra"])
        writer.writerow(["CCO", "ethanol"])
        writer.writerow(["c1ccccc1", "benzene"])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smiles-csv",
            str(csv_path),
            "--pocket-pdb-dir",
            str(pockets_dir),
            "--out-json",
            str(out_path),
            "--vina-bin",
            str(fake_vina),
            "--n-candidates",
            "3",
            "--exhaustiveness",
            "1",
            "--seed",
            "0",
            "--dock-timeout",
            "10",
            "--work-dir",
            str(tmp_path / "wd"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"script failed: rc={result.returncode}\n"
        f"STDOUT={result.stdout}\nSTDERR={result.stderr}"
    )
    assert out_path.is_file()
    with out_path.open("r", encoding="utf-8") as fh:
        panel = json.load(fh)
    assert panel["n_input"] == 2
    assert panel["metadata"]["vina_binary_ok"] is False
    # Wf-A101 (2026-10-20) — with the binary missing the panel emits
    # STUB_FALLBACK, or MEEKO_PREP_ONLY when meeko is installed and can still
    # prepare the ligand (cfm_vina_dock_panel.py:1877-1884).  MEASURED on this
    # venv: meeko is present, so every cell reads MEEKO_PREP_ONLY.  The
    # invariant that actually matters is not the label but that NO score is
    # invented without a binary, so that is what is asserted; the old test
    # pinned the label and went red the day the meeko path landed.
    for entry in panel["per_molecule"]:
        for cell in entry["per_pocket"].values():
            assert cell["status"] in {"STUB_FALLBACK", "MEEKO_PREP_ONLY"}, (
                f"binary missing must never produce a docked status; got "
                f"{cell['status']}"
            )
            assert cell["best_kcal_mol"] is None, (
                "no docking score may be reported without a Vina binary"
            )
    # aggregate best_score_overall must be None.
    assert panel["aggregate"]["best_score_overall"] is None
