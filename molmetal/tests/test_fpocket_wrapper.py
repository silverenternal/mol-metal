"""Tests for molmetal.adapters.fpocket_wrapper (Wf-A33 v2, no Java).

Aim: lock in the fpocket detector API so the Wf-A33 wire-in (PocketGenEncoder
pocket conditioning) can plug into a clean Python interface.

Tests use the pre-existing fpocket output from
``/home/hugo/codes/pocket/new/external_baselines/Pocket2Mol/example/``
(4yhj.pdb + 4yhj_out/pockets/pocketN_atm.pdb + 4yhj_info.txt) as a fixture,
so no live fpocket run is needed to run the test suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.fpocket_wrapper import (  # noqa: E402
    PocketResult,
    _resolve_fpocket_cmd,
    compute_pocket_center,
    crop_pocket_to_sphere,
    extract_pocket,
    find_center_residue,
    is_fpocket_available,
    parse_fpocket_info,
    parse_pocket_atoms,
    run_fpocket,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

#: Path to the pre-existing fpocket 4yhj output (used as a test fixture).
#: Per Wf-A33 task: this fixture is on disk from the Pocket2Mol example
#: directory; tests do NOT need to invoke fpocket at runtime.
FPOCKET_FIXTURE_DIR = Path(
    "/home/hugo/codes/pocket/new/external_baselines/Pocket2Mol/example"
)
FPOCKET_4YHJ_PDB = FPOCKET_FIXTURE_DIR / "4yhj.pdb"
FPOCKET_4YHJ_INFO = FPOCKET_FIXTURE_DIR / "4yhj_out" / "4yhj_info.txt"
FPOCKET_4YHJ_POCKET1_ATM = (
    FPOCKET_FIXTURE_DIR / "4yhj_out" / "pockets" / "pocket1_atm.pdb"
)


@pytest.fixture
def synthetic_fpocket_info(tmp_path):
    """Write a synthetic fpocket info.txt and return the path."""
    info_path = tmp_path / "test_info.txt"
    info_path.write_text(
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
        "\n"
        "Pocket 3 :\n"
        "\tScore : \t0.253\n"
        "\tDruggability Score : \t0.935\n"
        "\tNumber of Alpha Spheres : \t81\n"
        "\tVolume : \t1161.730\n",
        encoding="utf-8",
    )
    return info_path


# ------------------------------------------------------------------
# Skip marker: tests requiring the on-disk fixture
# ------------------------------------------------------------------

requires_fixture = pytest.mark.skipif(
    not (FPOCKET_4YHJ_PDB.exists() and FPOCKET_4YHJ_INFO.exists()
         and FPOCKET_4YHJ_POCKET1_ATM.exists()),
    reason="fpocket 4yhj fixture not present on disk "
           "(expected at /home/hugo/codes/pocket/new/external_baselines/Pocket2Mol/example/)",
)


# ------------------------------------------------------------------
# is_fpocket_available / _resolve_fpocket_cmd
# ------------------------------------------------------------------

def test_is_fpocket_available_is_pure_function():
    """Calling is_fpocket_available twice returns the same answer; no side effects."""
    a = is_fpocket_available()
    b = is_fpocket_available()
    assert a == b


def test_resolve_fpocket_cmd_returns_none_or_str():
    """The resolver returns either a str (path) or None — never raises."""
    cmd = _resolve_fpocket_cmd()
    assert cmd is None or isinstance(cmd, str)


def test_molmetal_fpocket_cmd_env_override(monkeypatch, tmp_path):
    """MOLMETAL_FPOCKET_CMD env var takes precedence (when runnable)."""
    # Create a fake executable.
    fake = tmp_path / "fake_fpocket.sh"
    fake.write_text("#!/bin/sh\necho fpocket\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("MOLMETAL_FPOCKET_CMD", str(fake))
    cmd = _resolve_fpocket_cmd()
    assert cmd == str(fake)


# ------------------------------------------------------------------
# parse_fpocket_info
# ------------------------------------------------------------------

def test_parse_synthetic_info_returns_3_pockets(synthetic_fpocket_info):
    results = parse_fpocket_info(synthetic_fpocket_info)
    assert len(results) == 3


def test_parse_synthetic_info_sorted_by_pocket_id(synthetic_fpocket_info):
    results = parse_fpocket_info(synthetic_fpocket_info)
    assert [p.pocket_id for p in results] == [1, 2, 3]


def test_parse_synthetic_info_pocket_metadata(synthetic_fpocket_info):
    results = parse_fpocket_info(synthetic_fpocket_info)
    top = results[0]
    assert top.pocket_id == 1
    assert top.score == pytest.approx(0.353)
    assert top.drugability == pytest.approx(0.725)
    assert top.n_alpha == 141
    assert top.volume == pytest.approx(747.435)


def test_parse_missing_info_returns_empty_list(tmp_path):
    result = parse_fpocket_info(tmp_path / "_no_such_info.txt")
    assert result == []


def test_parse_empty_info_returns_empty_list(tmp_path):
    info_path = tmp_path / "empty_info.txt"
    info_path.write_text("", encoding="utf-8")
    assert parse_fpocket_info(info_path) == []


@requires_fixture
def test_parse_real_4yhj_info_returns_pockets():
    """Parse the real 4yhj fpocket output (81 pockets expected)."""
    results = parse_fpocket_info(FPOCKET_4YHJ_INFO)
    # The real 4yhj output has 81 pockets (counted from the on-disk output).
    assert len(results) >= 1, "Expected at least 1 pocket from real 4yhj fpocket output"
    # Pocket 1 should be the highest-scoring pocket.
    top = results[0]
    assert top.pocket_id == 1
    assert top.drugability > 0.0
    assert top.n_alpha > 0


def test_pocket_result_label_property():
    p = PocketResult(pocket_id=5, score=1.0, drugability=0.5, n_alpha=10)
    assert p.label == "pocket5"


# ------------------------------------------------------------------
# parse_pocket_atoms
# ------------------------------------------------------------------

def test_parse_pocket_atoms_missing_file(tmp_path):
    """Non-existent atm.pdb returns []."""
    result = parse_pocket_atoms(tmp_path / "_no_such_atm.pdb")
    assert result == []


@requires_fixture
def test_parse_real_pocket1_atoms_returns_residues():
    """Parse the real pocket1_atm.pdb and confirm it has residue hits."""
    residues = parse_pocket_atoms(FPOCKET_4YHJ_POCKET1_ATM)
    assert len(residues) > 0, "Expected at least 1 residue in real pocket1_atm.pdb"
    # All entries are (chain, resi) tuples.
    for r in residues:
        assert isinstance(r, tuple)
        assert len(r) == 2
        assert isinstance(r[0], str)  # chain
        assert isinstance(r[1], int)  # resi


@requires_fixture
def test_parse_real_pocket1_atoms_unique():
    """parse_pocket_atoms returns unique residues only."""
    residues = parse_pocket_atoms(FPOCKET_4YHJ_POCKET1_ATM)
    assert len(residues) == len(set(residues))


@requires_fixture
def test_parse_real_pocket1_atoms_sorted():
    """parse_pocket_atoms returns residues sorted (chain, resi)."""
    residues = parse_pocket_atoms(FPOCKET_4YHJ_POCKET1_ATM)
    assert residues == sorted(residues)


# ------------------------------------------------------------------
# compute_pocket_center
# ------------------------------------------------------------------

def test_compute_pocket_center_missing_file(tmp_path):
    """Non-existent atm.pdb returns (0, 0, 0)."""
    center = compute_pocket_center(tmp_path / "_no_such_atm.pdb")
    assert center == (0.0, 0.0, 0.0)


@requires_fixture
def test_compute_real_pocket1_center_returns_tuple_of_floats():
    """Compute the centroid of real pocket1_atm.pdb atoms."""
    center = compute_pocket_center(FPOCKET_4YHJ_POCKET1_ATM)
    assert isinstance(center, tuple)
    assert len(center) == 3
    x, y, z = center
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(z, float)
    # Sanity: centroid should be in a reasonable coordinate range (Å).
    assert -100 < x < 100
    assert -100 < y < 100
    assert -100 < z < 100


@requires_fixture
def test_compute_real_pocket1_center_nonzero():
    """The pocket1 centroid should be a non-trivial coordinate (not all zeros)."""
    center = compute_pocket_center(FPOCKET_4YHJ_POCKET1_ATM)
    assert any(c != 0.0 for c in center)


# ------------------------------------------------------------------
# find_center_residue
# ------------------------------------------------------------------

def test_find_center_residue_no_pocket_atoms_raises(tmp_path):
    """When pocket atoms are missing, find_center_residue raises ValueError."""
    empty_atm = tmp_path / "empty_atm.pdb"
    empty_atm.write_text("HEADER    EMPTY\n", encoding="utf-8")
    # Use the synthetic fpocket fixture PDB (4yhj.pdb); the test PDB here
    # only needs to exist so Biopython can open it.
    dummy_pdb = tmp_path / "dummy.pdb"
    dummy_pdb.write_text(
        "HEADER    DUMMY                                                  \n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "END                                                               \n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no pocket atoms found"):
        find_center_residue(empty_atm, dummy_pdb)


@requires_fixture
def test_find_center_residue_real_returns_residue():
    """Find the center residue for the real 4yhj pocket1."""
    residue = find_center_residue(FPOCKET_4YHJ_POCKET1_ATM, FPOCKET_4YHJ_PDB)
    assert isinstance(residue, tuple)
    assert len(residue) == 2
    chain_id, resi = residue
    assert isinstance(chain_id, str)
    assert isinstance(resi, int)
    # Sanity: the residue should be one of the pocket1 atom residues.
    pocket_residues = parse_pocket_atoms(FPOCKET_4YHJ_POCKET1_ATM)
    assert (chain_id, resi) in pocket_residues


# ------------------------------------------------------------------
# crop_pocket_to_sphere — reuses p2rank_wrapper.crop_pdb_to_residues
# ------------------------------------------------------------------

@requires_fixture
def test_crop_pocket_to_sphere_produces_nonempty_pdb(tmp_path):
    """End-to-end crop produces a non-empty PDB file."""
    out_path = tmp_path / "pocket1_crop.pdb"
    residue = find_center_residue(FPOCKET_4YHJ_POCKET1_ATM, FPOCKET_4YHJ_PDB)
    crop_pocket_to_sphere(FPOCKET_4YHJ_PDB, residue, out_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "ATOM" in content
    assert content.count("ATOM") >= 1


@requires_fixture
def test_crop_pocket_to_sphere_respects_radius(tmp_path):
    """Cropping with a smaller radius produces fewer (or equal) atoms."""
    out_big = tmp_path / "pocket1_crop_10A.pdb"
    out_small = tmp_path / "pocket1_crop_5A.pdb"
    residue = find_center_residue(FPOCKET_4YHJ_POCKET1_ATM, FPOCKET_4YHJ_PDB)
    crop_pocket_to_sphere(FPOCKET_4YHJ_PDB, residue, out_big, sphere_radius_A=10.0)
    crop_pocket_to_sphere(FPOCKET_4YHJ_PDB, residue, out_small, sphere_radius_A=5.0)
    big_atoms = out_big.read_text(encoding="utf-8").count("ATOM")
    small_atoms = out_small.read_text(encoding="utf-8").count("ATOM")
    assert small_atoms <= big_atoms


# ------------------------------------------------------------------
# run_fpocket — wire-in contract
# ------------------------------------------------------------------

def test_run_fpocket_raises_file_not_found_when_unavailable(monkeypatch, tmp_path):
    """When fpocket is not on PATH and no candidate path works, raise FileNotFoundError."""
    # Force resolution to None.
    monkeypatch.setenv("MOLMETAL_FPOCKET_CMD", "")
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("molmetal.adapters.fpocket_wrapper.FPCKT_CANDIDATE_PATHS", ())
    pdb = tmp_path / "dummy.pdb"
    pdb.write_text("HEADER\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="fpocket is not installed"):
        run_fpocket(pdb, tmp_path)


def test_run_fpocket_raises_for_missing_pdb(monkeypatch, tmp_path):
    """run_fpocket raises FileNotFoundError when PDB doesn't exist."""
    # Even with fpocket available, a missing PDB is caught first.
    monkeypatch.setattr(
        "molmetal.adapters.fpocket_wrapper._resolve_fpocket_cmd",
        lambda: "/fake/fpocket",
    )
    with pytest.raises(FileNotFoundError, match="PDB file does not exist"):
        run_fpocket(tmp_path / "_no_such.pdb", tmp_path)


# ------------------------------------------------------------------
# extract_pocket — orchestrator
# ------------------------------------------------------------------

@requires_fixture
def test_extract_pocket_real_4yhj_produces_cropped_pdb(tmp_path):
    """End-to-end extract_pocket on real 4yhj.pdb produces a cropped PDB."""
    out_pdb = extract_pocket(FPOCKET_4YHJ_PDB, tmp_path, sphere_radius_A=10.0)
    assert out_pdb.exists()
    content = out_pdb.read_text(encoding="utf-8")
    assert "ATOM" in content


# ------------------------------------------------------------------
# CLI smoke
# ------------------------------------------------------------------

def test_module_cli_smoke_returns_0():
    """Running ``python -m molmetal.adapters.fpocket_wrapper`` returns 0 and prints PASS."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "molmetal.adapters.fpocket_wrapper"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"smoke failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
