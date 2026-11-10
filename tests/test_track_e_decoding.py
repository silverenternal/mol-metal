"""Regression tests for Track E (Wf-60231cff-03e-15) m14-admet-paper-align.

Verifies the 3 new modules shipped in Track E: decode_gate, admet8,
paper_table.  Each new flag wired into a CLI must round-trip through
its argparse dest + emit a JSON that contains expected keys; this is
the §5.7 trainer-flag wire-in smoke at the test-suite level.

Run from project root::

    pytest -x tests/test_track_e_decoding.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.eval.scoring.admet8 import compute_admet8_one, compute_admet8_panel
from molmetal.eval.scoring.decode_gate import gate_smiles, gate_ratio
from molmetal.eval.scoring.paper_table import (
    COHORT_KEYS,
    PROTOCOL_VERSION,
    build_paper_table,
    write_paper_table_json,
)


PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Per-atom gate (EV1)
# ---------------------------------------------------------------------------
class TestDecodeGate:
    def test_c_dot_c_dot_c_dot_c_rejected(self):
        """``C.C.C.C`` is sanitised by RDKit but is multi-fragment;
        per-atom gate MUST reject at this repo's own 0.80 floor (a REPO-LOCAL
        project-strict choice; MOSES has no fragment term)."""
        r = gate_smiles("C.C.C.C", largest_frac_min=0.80, min_atoms=2)
        assert r.ok is False
        assert r.reason == "multi_fragment"
        assert r.sanitized is True  # RDKit still sanitises 4 CH4

    def test_c_dot_c_dot_c_dot_c_backward_compat(self):
        """Legacy RDKit-strict-only callers pass ``largest_frac_min=0``;
        MUST remain bit-exact accepted."""
        r = gate_smiles("C.C.C.C", largest_frac_min=0.0, min_atoms=0)
        assert r.ok is True

    def test_single_atom_rejected_by_min_atoms(self):
        r = gate_smiles("[He]", largest_frac_min=0.0, min_atoms=2)
        assert r.ok is False
        assert r.reason == "min_atoms"

    def test_unparseable(self):
        r = gate_smiles("nonsense_smiles")
        assert r.ok is False
        assert r.reason == "parse_failed"

    def test_ethanol_ok(self):
        r = gate_smiles("CCO", largest_frac_min=0.80, min_atoms=2)
        assert r.ok is True
        assert r.n_heavy_atoms == 3
        assert r.largest_frac == pytest.approx(1.0)

    def test_gate_ratio_aggregates(self):
        gr = gate_ratio(
            ["CCO", "C.C.C.C", "nonsense", "[He]", "CC"],
            largest_frac_min=0.80, min_atoms=2,
        )
        assert gr["n_total"] == 5
        assert gr["n_ok"] == 2
        assert gr["per_atom_validity"] == pytest.approx(0.4)
        assert gr["multi_frag_rejects"] == 1
        assert gr["min_atoms_rejects"] == 1
        assert gr["parse_failures"] == 1

    def test_gate_ratio_empty(self):
        gr = gate_ratio([], largest_frac_min=0.80)
        assert gr["n_total"] == 0
        assert gr["per_atom_validity"] == 0.0


# ---------------------------------------------------------------------------
# 8 ADMET proxy (EV2)
# ---------------------------------------------------------------------------
class TestADMET8:
    def test_ethanol_full_row(self):
        row = compute_admet8_one("CCO")
        assert row is not None
        assert row.hbd == 1
        assert row.hba == 1
        assert row.rotB == 0
        assert row.aromatic_rings == 0

    def test_salt_split(self):
        """Acetic acid + Na+ counterion: scoring MUST be on the parent."""
        row = compute_admet8_one("CC(=O)O.[Na+]")
        assert row is not None
        assert row.hbd == 1  # acetic acid OH

    def test_panel_empty(self):
        p = compute_admet8_panel([], cohort_tag="empty")
        assert p["n_input"] == 0
        assert p["n_ok"] == 0
        assert "per_metric_mean" in p

    def test_panel_5mol(self):
        cohort = ["CCO", "c1ccccc1", "invalid_smiles", None]
        p = compute_admet8_panel(cohort, cohort_tag="smoke")
        assert p["n_input"] == 4
        assert p["n_ok"] >= 1
        assert "sa" in p["per_metric_mean"]
        assert p["protocol_version"] == "admet8/2026-10-13"


# ---------------------------------------------------------------------------
# Paper table (EV7)
# ---------------------------------------------------------------------------
class TestPaperTable:
    def test_cohort_keys_canonical(self):
        for k in ("platinai_pt_subset_5k", "nci60_9_cell", "drugood"):
            assert k in COHORT_KEYS

    def test_paper_table_empty_cohort(self, tmp_path: Path):
        row = build_paper_table(
            cohort_id="platinai_pt_subset_5k",
            smiles=[],
            metrics_dir=tmp_path,
        )
        assert row.n_smiles == 0
        assert row.protocol_version == PROTOCOL_VERSION
        out = tmp_path / "x.json"
        write_paper_table_json(row, out)
        assert out.exists()
        with out.open() as f:
            rd = json.load(f)
        assert rd["cohort_id"] == "platinai_pt_subset_5k"

    def test_paper_table_with_cohort(self, tmp_path: Path):
        cohort = ["CCO", "c1ccccc1", "C.C.C.C"]
        row = build_paper_table(
            cohort_id="lambda_pilot_5x1",
            smiles=cohort,
            metrics_dir=tmp_path,
        )
        assert row.n_smiles == 3
        # C.C.C.C should be rejected, CCO + benzene accepted
        assert row.m14_decode["per_atom_validity"] < 1.0
        assert row.m14_decode["n_multi_frag_rejects"] == 1
        assert row.admet8["n_ok"] >= 2

    def test_paper_table_atomic_write(self, tmp_path: Path):
        cohort = ["CCO"]
        row = build_paper_table(
            cohort_id="nci60_9_cell",
            smiles=cohort,
            metrics_dir=tmp_path,
        )
        out = tmp_path / "atomic.json"
        write_paper_table_json(row, out)
        assert out.exists()
        # No leftover temp files
        temps = list(tmp_path.glob("*.tmp"))
        assert temps == [], temps


# ---------------------------------------------------------------------------
# CLI wire-in (§5.7)
# ---------------------------------------------------------------------------
class TestCLI:
    def test_bench_paper_table_help(self):
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "molmetal" / "scripts" / "bench_paper_table.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # New flags from the §5.7 wire-in.
        for flag in ("--cohort", "--smiles-csv", "--largest-frac-min", "--min-atoms", "--emit"):
            assert flag in result.stdout, flag

    def test_bench_paper_table_smoke(self):
        """Run the CLI's __main__ smoke (no args)."""
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "molmetal" / "scripts" / "bench_paper_table.py")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "SMOKE PASS" in result.stdout

    def test_metallo_smoke_retrain_flags_wired(self):
        """The §5.7 wire-in: every new flag accepts on --help."""
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "molmetal" / "scripts" / "metallo_drug_smoke_retrain.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        for flag in (
            "--strict-large-frag", "--largest-frac-min", "--min-atoms",
            "--emit-admet8", "--paper-table-cohort",
        ):
            assert flag in result.stdout, flag
