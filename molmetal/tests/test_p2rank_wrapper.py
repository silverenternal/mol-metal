"""Tests for molmetal.adapters.p2rank_wrapper (Wf-A33 prep).

Aim: lock in the parser + availability check + cropper API so the
Wf-A33 wire-in (subprocess call) only needs to add the subprocess
invocation, not reshape the dataclass / parser contract.

No Java / no p2rank CLI required to run these tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.p2rank_wrapper import (  # noqa: E402
    P2RankResult,
    _resolve_p2rank_cmd,
    crop_pdb_to_residues,
    is_p2rank_available,
    parse_p2rank_csv,
    run_p2rank,
    top_pockets,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def synthetic_p2rank_csv(tmp_path):
    """Write a synthetic p2rank *_residues.csv and return the path."""
    csv_path = tmp_path / "test_input_residues.csv"
    csv_path.write_text(
        "rank,score,label,chain,resi,resn,probability,residues_list\n"
        "1,5.42,pocket1,A,123,SER,0.91,A:120;A:121;A:122;A:123;A:124\n"
        "2,3.10,pocket2,B,55,GLY,0.78,B:50;B:51;B:52;B:53;B:54;B:55\n"
        "3,1.20,pocket3,A,7,ALA,0.45,A:5;A:6;A:7;A:8\n",
        encoding="utf-8",
    )
    return csv_path


# ------------------------------------------------------------------
# is_p2rank_available / _resolve_p2rank_cmd
# ------------------------------------------------------------------

def test_is_p2rank_available_is_pure_function():
    """Calling is_p2rank_available twice returns the same answer; no side effects."""
    a = is_p2rank_available()
    b = is_p2rank_available()
    assert a == b


def test_resolve_p2rank_cmd_returns_none_or_str():
    """The resolver returns either a str (path) or None — never raises."""
    cmd = _resolve_p2rank_cmd()
    assert cmd is None or isinstance(cmd, str)


def test_molmetal_p2rank_cmd_env_override(monkeypatch, tmp_path):
    """MOLMETAL_P2RANK_CMD env var takes precedence (when runnable)."""
    # Create a fake executable.
    fake = tmp_path / "fake_p2rank.sh"
    fake.write_text("#!/bin/sh\necho p2rank\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("MOLMETAL_P2RANK_CMD", str(fake))
    cmd = _resolve_p2rank_cmd()
    assert cmd == str(fake)


# ------------------------------------------------------------------
# parse_p2rank_csv
# ------------------------------------------------------------------

def test_parse_synthetic_csv_returns_3_pockets(synthetic_p2rank_csv):
    results = parse_p2rank_csv(synthetic_p2rank_csv)
    assert len(results) == 3


def test_parse_synthetic_csv_sorted_by_rank(synthetic_p2rank_csv):
    results = parse_p2rank_csv(synthetic_p2rank_csv)
    assert [r.rank for r in results] == [1, 2, 3]


def test_parse_synthetic_csv_top_pocket_metadata(synthetic_p2rank_csv):
    results = parse_p2rank_csv(synthetic_p2rank_csv)
    top = results[0]
    assert top.rank == 1
    assert top.score == pytest.approx(5.42)
    assert top.probability == pytest.approx(0.91)
    assert top.chain == "A"
    assert top.resi == 123
    assert top.resn == "SER"


def test_parse_synthetic_csv_residues_list_parsed(synthetic_p2rank_csv):
    results = parse_p2rank_csv(synthetic_p2rank_csv)
    # Top pocket has 5 residues
    assert len(results[0].residues) == 5
    assert results[0].residues[0] == ("A", 120)
    assert results[0].residues[-1] == ("A", 124)
    # 2nd pocket has 6 residues on chain B
    assert len(results[1].residues) == 6
    assert all(chain == "B" for chain, _ in results[1].residues)


def test_parse_missing_csv_returns_empty_list(tmp_path):
    result = parse_p2rank_csv(tmp_path / "_no_such.csv")
    assert result == []


def test_parse_malformed_csv_does_not_crash(tmp_path):
    """P2Rank occasionally emits an empty / header-only CSV. Must not raise."""
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text("rank,score,label,chain,resi,resn,probability,residues_list\n", encoding="utf-8")
    results = parse_p2rank_csv(csv_path)
    assert results == []


def test_parse_garbled_numeric_fields_default_to_zero(tmp_path):
    """A row with garbage in score/probability must coerce to 0.0, not raise."""
    csv_path = tmp_path / "garbled.csv"
    csv_path.write_text(
        "rank,score,label,chain,resi,resn,probability,residues_list\n"
        "1,not_a_number,p1,A,nope,SER,bad_prob,A:1\n",
        encoding="utf-8",
    )
    results = parse_p2rank_csv(csv_path)
    assert len(results) == 1
    assert results[0].score == 0.0
    assert results[0].probability == 0.0
    assert results[0].chain == "A"


def test_p2rank_result_centre_property():
    r = P2RankResult(rank=1, score=1.0, probability=0.5, chain="X", resi=42, resn="HIS")
    assert r.centre == ("X", 42, "HIS")


# ------------------------------------------------------------------
# top_pockets
# ------------------------------------------------------------------

def test_top_pockets_n_1(synthetic_p2rank_csv):
    top1 = top_pockets(synthetic_p2rank_csv, n=1)
    assert len(top1) == 1
    assert top1[0].rank == 1


def test_top_pockets_n_larger_than_pool(synthetic_p2rank_csv):
    top = top_pockets(synthetic_p2rank_csv, n=10)
    assert len(top) == 3  # only 3 pockets in the synthetic CSV


def test_top_pockets_min_probability_filter(synthetic_p2rank_csv):
    """Filter drops low-probability pockets."""
    top = top_pockets(synthetic_p2rank_csv, n=10, min_probability=0.80)
    assert len(top) == 1  # only the top pocket (0.91) passes
    assert top[0].rank == 1


def test_top_pockets_strict_filter_drops_all(synthetic_p2rank_csv):
    top = top_pockets(synthetic_p2rank_csv, n=10, min_probability=0.99)
    assert top == []


# ------------------------------------------------------------------
# crop_pdb_to_residues — input validation
# ------------------------------------------------------------------

def test_crop_pdb_to_residues_no_match_raises(tmp_path):
    """When the seed residues don't match any atom in the PDB, raise (not silent)."""
    # Write a minimal valid PDB with chain A residue 1 (ALA), no residue 999.
    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        "HEADER    TINY                                                  \n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
        "END                                                               \n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pocket.pdb"
    with pytest.raises(ValueError, match="no seed atoms matched"):
        crop_pdb_to_residues(
            pdb_path,
            [("A", 999)],  # not present
            out_path,
        )


def test_crop_pdb_to_residues_with_p2rank_result(tmp_path):
    """P2RankResult input is auto-converted to its .residues field."""
    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        "HEADER    TINY                                                  \n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
        "END                                                               \n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pocket.pdb"
    pr = P2RankResult(
        rank=1, score=1.0, probability=0.5, chain="A", resi=999, resn="ALA",
        residues=(("A", 999),),
    )
    # No match → must raise
    with pytest.raises(ValueError, match="no seed atoms matched"):
        crop_pdb_to_residues(pdb_path, pr, out_path)


# ------------------------------------------------------------------
# run_p2rank — wire-in contract
# ------------------------------------------------------------------

def test_run_p2rank_raises_not_implemented():
    """The subprocess driver is the Wf-A33 wire-in target."""
    with pytest.raises(NotImplementedError, match="Wf-A33 wire-in"):
        run_p2rank("/tmp/_no_such.pdb", "/tmp/_no_such_out")


def test_run_p2rank_does_not_actually_subprocess():
    """The stub must NOT call subprocess — verify by passing a non-existent
    PDB that would cause subprocess to error if it ran."""
    with pytest.raises(NotImplementedError):
        # If the stub were a real subprocess call, this would have raised
        # FileNotFoundError or a similar IOError. NotImplementedError wins.
        run_p2rank("/this/file/does/not/exist.pdb", "/tmp")


# ------------------------------------------------------------------
# CLI smoke
# ------------------------------------------------------------------

def test_module_cli_smoke_returns_0():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "molmetal.adapters.p2rank_wrapper"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"smoke failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
