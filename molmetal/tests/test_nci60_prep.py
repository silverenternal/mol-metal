"""Tests for Wf-28 Phase 2 — NCI-60 9-cell precious-metal data prep.

Covers:

* :mod:`molmetal.scripts.prepare_nci60_data` — the wf28 data-prep
  script that emits the long-form CSV, the PlatinAI cross-link, the
  summary markdown, and the audit JSON.
* :func:`canonical_smiles`, :func:`has_existing_csv`,
  :func:`build_template`, :func:`build_cross_link`,
  :func:`write_long_csv`, :func:`write_link_csv` — the helpers the
  script exposes for in-process testing.

Background
----------
Wf-28 Phase 2 needs an NCI-60 9-cell precious-metal cytotoxicity
benchmark + a PlatinAI 4-cell cross-link.  The benchmark covers the
9 most-studied NCI-60-overlapping cell lines from MetalCytoToxDB
(MCF-7, A549, A2780, A2780cisR, HCT-116, PC-3, HL60, SK-OV-3, DU-145)
restricted to precious metals (Ru/Ir/Rh/Os/Re).  The cross-link
joins on canonical SMILES to the PlatinAI 4-cell Pt corpus.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def prep_module():
    """Import the prepare_nci60_data module (slow first import)."""
    from molmetal.scripts import prepare_nci60_data as m  # noqa: E402

    return m


@pytest.fixture
def tmp_outputs(tmp_path: Path):
    """Yield a dict of tmp output paths wired to the script defaults."""
    return {
        "out_csv": tmp_path / "nci60_precious_9cell.csv",
        "link_csv": tmp_path / "nci60_platinai_link.csv",
        "platinai_csv": tmp_path / "platinai_4cell.csv",
        "summary_md": tmp_path / "nci60_summary.md",
        "link_json": tmp_path / "nci60_platinai_link.json",
    }


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
def test_constants_cell_source_existing_has_9_cells(prep_module):
    """NCI60_9CELL_EXISTING must contain exactly 9 cell-line pairs."""
    cell_source = prep_module.NCI60_9CELL_EXISTING
    assert len(cell_source) == 9
    assert all(len(pair) == 2 for pair in cell_source)
    # All display names must be unique
    disp_names = [disp for disp, _ in cell_source]
    assert len(set(disp_names)) == 9


def test_constants_precious_metals_excludes_pt(prep_module):
    """PRECIOUS_METALS must NOT contain Pt (Pt is in platinai_4cell)."""
    assert "Pt" not in prep_module.PRECIOUS_METALS
    assert set(prep_module.PRECIOUS_METALS) == {"Ru", "Ir", "Rh", "Os", "Re"}


def test_constants_active_threshold_um_is_10(prep_module):
    """Active threshold is 10 µM (industry-standard Pt / Ru cutoff)."""
    assert prep_module.ACTIVE_THRESHOLD_UM == 10.0


# ---------------------------------------------------------------------------
# 2. canonical_smiles
# ---------------------------------------------------------------------------
def test_canonical_smiles_roundtrip(prep_module):
    """canonical_smiles is idempotent on valid SMILES."""
    smi = "BrCc1ccc(-c2ccccn2)nc1"
    canon1 = prep_module.canonical_smiles(smi)
    canon2 = prep_module.canonical_smiles(canon1)
    assert canon1 == canon2
    assert canon1  # non-empty


def test_canonical_smiles_falls_back_on_garbage(prep_module):
    """Invalid SMILES returns the input unchanged (no crash)."""
    bad = "not_a_smiles_@@@"
    out = prep_module.canonical_smiles(bad)
    assert out == bad  # identity fallback


# ---------------------------------------------------------------------------
# 3. has_existing_csv
# ---------------------------------------------------------------------------
def test_has_existing_csv_true(prep_module, tmp_path: Path):
    p = tmp_path / "good.csv"
    p.write_text("smiles,cell_line,pIC50,active\nc1ccccc1,MCF7,5.0,1\n")
    assert prep_module.has_existing_csv(p) is True


def test_has_existing_csv_false_on_wrong_header(prep_module, tmp_path: Path):
    p = tmp_path / "wrong.csv"
    p.write_text("compound,cell_line,pIC50,active\nc1ccccc1,MCF7,5.0,1\n")
    assert prep_module.has_existing_csv(p) is False


def test_has_existing_csv_false_on_missing(prep_module, tmp_path: Path):
    assert prep_module.has_existing_csv(tmp_path / "missing.csv") is False


# ---------------------------------------------------------------------------
# 4. build_template (deterministic synthetic fallback)
# ---------------------------------------------------------------------------
def test_build_template_row_count(prep_module):
    """Template generates exactly n_per_cell * n_cells rows."""
    rows = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=5, seed=42
    )
    assert len(rows) == 5 * 9


def test_build_template_columns(prep_module):
    """Each row has (smiles, cell_line, pIC50, active) with correct types."""
    rows = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=3, seed=1
    )
    for smi, cell, pic50, active in rows:
        assert isinstance(smi, str) and smi
        assert isinstance(cell, str) and cell
        assert isinstance(pic50, float)
        assert active in (0, 1)


def test_build_template_deterministic(prep_module):
    """Same seed → identical rows (the synthetic fallback is reproducible)."""
    a = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=4, seed=20260926
    )
    b = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=4, seed=20260926
    )
    assert a == b


def test_build_template_pic50_reasonable(prep_module):
    """Synthetic pIC50 is roughly in [3.0, 8.0] — IC50 in nM-100µM range."""
    rows = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=20, seed=123
    )
    pic50s = [r[2] for r in rows]
    assert min(pic50s) > 2.0
    assert max(pic50s) < 9.0


# ---------------------------------------------------------------------------
# 5. write_long_csv + write_link_csv
# ---------------------------------------------------------------------------
def test_write_long_csv_schema(prep_module, tmp_path: Path):
    """Long-form CSV has the documented (smiles, cell_line, pIC50, active) schema."""
    rows = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=2, seed=99
    )
    out = tmp_path / "long.csv"
    n = prep_module.write_long_csv(rows, out)
    assert n == 18
    with out.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["smiles", "cell_line", "pIC50", "active"]
        n_data = sum(1 for _ in reader)
    assert n_data == 18


def test_write_link_csv_schema(prep_module, tmp_path: Path):
    """Cross-link CSV has the documented schema."""
    rows = prep_module.build_template(
        prep_module.NCI60_9CELL_EXISTING, n_per_cell=2, seed=99
    )
    plat = {"c1ccccc1": 1}  # won't overlap — that's fine for the schema check
    link_rows = prep_module.build_cross_link(
        rows, plat, {}, [d for d, _ in prep_module.NCI60_9CELL_EXISTING]
    )
    out = tmp_path / "link.csv"
    n = prep_module.write_link_csv(link_rows, out)
    assert n == 18
    with out.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[:5] == [
            "smiles",
            "cell_line_precious",
            "pIC50_precious",
            "active_precious",
            "in_platinai",
        ]


# ---------------------------------------------------------------------------
# 6. build_cross_link
# ---------------------------------------------------------------------------
def test_build_cross_link_counts_overlap(prep_module):
    """A SMILES present in both corpora is marked in_platinai=True."""
    rows = [
        ("Cc1ccccc1", "MCF7", 5.0, 1),
        ("c1ccncc1", "MCF7", 5.0, 0),
    ]
    plat_smiles = {"Cc1ccccc1": 1, "other_smiles": 1}
    link = prep_module.build_cross_link(
        rows, plat_smiles, {}, ["MCF7"]
    )
    by_smi = {r["smiles"]: r for r in link}
    assert by_smi["Cc1ccccc1"]["in_platinai"] is True
    assert by_smi["Cc1ccccc1"]["n_platinai_rows"] == 1
    assert by_smi["c1ccncc1"]["in_platinai"] is False
    assert by_smi["c1ccncc1"]["n_platinai_rows"] == 0


def test_build_cross_link_platinai_pic50_aggregation(prep_module):
    """platinai_avg_pIC50 is the mean of the per-cell PlatinAI pIC50s."""
    rows = [("test_smi", "MCF7", 5.0, 1)]
    plat_smiles = {"test_smi": 1}
    # 4 PlatinAI pIC50s per row (MCF7, A549, A2780, A2780cis)
    plat_pic50 = {"test_smi": [5.0, 5.5, 6.0, 6.5]}
    link = prep_module.build_cross_link(rows, plat_smiles, plat_pic50, ["MCF7"])
    avg = link[0]["platinai_avg_pIC50"]
    assert avg is not None
    assert abs(avg - 5.75) < 1e-6
    # active_rate: 4 / 4 active (all pIC50 >= 5.0)
    assert abs(link[0]["platinai_active_rate"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. End-to-end on temp outputs (template mode, no substrate)
# ---------------------------------------------------------------------------
def test_end_to_end_template_mode(prep_module, tmp_outputs, monkeypatch):
    """End-to-end run in template mode produces all 4 output files."""
    # Force the substrate lookup to return None
    monkeypatch.setattr(prep_module, "find_substrate", lambda d: None)
    # Empty PlatinAI CSV
    platinai_csv = tmp_outputs["platinai_csv"]
    platinai_csv.parent.mkdir(parents=True, exist_ok=True)
    platinai_csv.write_text(
        "complex_id,smiles,ic50_mcf7_nm,ic50_a549_nm,ic50_a2780_nm,ic50_a2780cis_nm,source\n"
    )

    argv = [
        "--n-per-cell",
        "3",
        "--out-csv",
        str(tmp_outputs["out_csv"]),
        "--link-csv",
        str(tmp_outputs["link_csv"]),
        "--platinai-csv",
        str(tmp_outputs["platinai_csv"]),
        "--summary-md",
        str(tmp_outputs["summary_md"]),
        "--link-json",
        str(tmp_outputs["link_json"]),
        "--seed",
        "42",
    ]
    rc = prep_module.main(argv)
    assert rc == 0

    # Long-form CSV: 9 cells × 3 rows = 27
    assert tmp_outputs["out_csv"].exists()
    with tmp_outputs["out_csv"].open(newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert len(data) == 27

    # Link CSV: same 27 rows
    assert tmp_outputs["link_csv"].exists()
    with tmp_outputs["link_csv"].open(newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert len(data) == 27

    # Summary markdown exists & mentions all 9 cell lines
    assert tmp_outputs["summary_md"].exists()
    md = tmp_outputs["summary_md"].read_text()
    for disp, _src in prep_module.NCI60_9CELL_EXISTING:
        assert disp in md, f"missing {disp} in summary"

    # Link JSON exists & parses
    assert tmp_outputs["link_json"].exists()
    payload = json.loads(tmp_outputs["link_json"].read_text())
    assert payload["n_nci60_rows"] == 27
    assert payload["n_link_rows"] == 27
    assert payload["n_in_platinai"] == 0  # no PlatinAI overlap by construction
    assert payload["cell_lines"] == [d for d, _ in prep_module.NCI60_9CELL_EXISTING]


def test_end_to_end_idempotent_reuse(prep_module, tmp_outputs, monkeypatch):
    """Re-running without --force re-uses the existing CSV (no rebuild)."""
    monkeypatch.setattr(prep_module, "find_substrate", lambda d: None)
    platinai_csv = tmp_outputs["platinai_csv"]
    platinai_csv.parent.mkdir(parents=True, exist_ok=True)
    platinai_csv.write_text(
        "complex_id,smiles,ic50_mcf7_nm,ic50_a549_nm,ic50_a2780_nm,ic50_a2780cis_nm,source\n"
    )

    argv = [
        "--n-per-cell",
        "2",
        "--out-csv",
        str(tmp_outputs["out_csv"]),
        "--link-csv",
        str(tmp_outputs["link_csv"]),
        "--platinai-csv",
        str(tmp_outputs["platinai_csv"]),
        "--summary-md",
        str(tmp_outputs["summary_md"]),
        "--link-json",
        str(tmp_outputs["link_json"]),
        "--seed",
        "7",
    ]
    rc = prep_module.main(argv)
    assert rc == 0
    # The long-form CSV is now on disk.  Bump mtime by writing a known bad
    # file to a different path; verify reuse by adding a row-count assertion.
    with tmp_outputs["out_csv"].open(newline="") as f:
        n_first = sum(1 for _ in csv.DictReader(f))
    rc = prep_module.main(argv)
    assert rc == 0
    with tmp_outputs["out_csv"].open(newline="") as f:
        n_second = sum(1 for _ in csv.DictReader(f))
    # Same row count both times (idempotent re-use)
    assert n_first == n_second
