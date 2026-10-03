"""Tests for Wf-264-SA — CFM SA + QED property panel.

These tests verify:
  1. SA + QED values are computed correctly for a small SMILES list.
  2. Invalid SMILES are handled gracefully (sa / qed = None).
  3. Aggregate statistics (mean / std / percentiles / n) are
     numerically correct on a hand-checkable fixture.
  4. CLI --help exits 0 and prints expected strings.

All tests are CPU-only, no GPU required, no checkpoint needed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"


def _load_module():
    """Load cfm_property_panel.py as a module."""
    spec = importlib.util.spec_from_file_location("cfm_property_panel", str(SCRIPT_PATH))
    assert spec and spec.loader, "cfm_property_panel.py must exist and be importable"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sa_qed_toy_smiles():
    """Verify SA + QED for three well-known SMILES are sensible floats."""
    mod = _load_module()
    sascorer = mod._load_sascorer()
    smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    panel = mod.build_panel(smiles_list)
    assert panel["n_input"] == 3
    assert panel["n_parsed"] == 3
    assert panel["n_failed"] == 0
    for entry in panel["per_molecule"]:
        assert isinstance(entry["sa"], float)
        assert isinstance(entry["qed"], float)
        assert 1.0 <= entry["sa"] <= 10.0, f"SA out of range: {entry}"
        assert 0.0 <= entry["qed"] <= 1.0, f"QED out of range: {entry}"


def test_invalid_smiles_handled():
    """Verify unparseable SMILES yield sa / qed = None without crashing."""
    mod = _load_module()
    sascorer = mod._load_sascorer()
    bad = "NOT_A_SMILES@!"
    entry = mod.compute_one(bad, sascorer)
    assert entry["smiles"] == bad
    assert entry["sa"] is None
    assert entry["qed"] is None

    # Also exercise the panel-level code path.
    panel = mod.build_panel(["CCO", bad, "c1ccccc1"])
    assert panel["n_input"] == 3
    assert panel["n_parsed"] == 2
    assert panel["n_failed"] == 1
    # The bad SMILES entry should have None for sa/qed.
    bad_entry = next(m for m in panel["per_molecule"] if m["smiles"] == bad)
    assert bad_entry["sa"] is None
    assert bad_entry["qed"] is None
    # parsed_ratio should reflect 2/3.
    assert panel["aggregate"]["parsed_ratio"] == pytest.approx(2.0 / 3.0)


def test_aggregate_stats_correct():
    """Verify aggregate mean / std / percentiles on a hand-checkable fixture."""
    mod = _load_module()
    # Five identical valid molecules → mean == single value, std == 0.
    panel = mod.build_panel(["CCO"] * 5)
    sa_agg = panel["aggregate"]["sa"]
    qed_agg = panel["aggregate"]["qed"]
    assert sa_agg["n"] == 5
    assert qed_agg["n"] == 5
    assert sa_agg["min"] == pytest.approx(sa_agg["max"], abs=1e-9)
    assert sa_agg["mean"] == pytest.approx(sa_agg["p50"], abs=1e-9)
    assert sa_agg["std"] == pytest.approx(0.0, abs=1e-9)
    assert qed_agg["std"] == pytest.approx(0.0, abs=1e-9)

    # Inject four invalid SMILES so aggregate sees only 1 valid out of 5.
    panel2 = mod.build_panel(["CCO", "bad1", "bad2", "bad3", "bad4"])
    assert panel2["aggregate"]["sa"]["n"] == 1
    assert panel2["aggregate"]["qed"]["n"] == 1
    assert panel2["n_parsed"] == 1
    assert panel2["n_failed"] == 4

    # Percentiles on a known fixed list (5 values) — n=5, even.
    fixed = [1.0, 2.0, 3.0, 4.0, 5.0]
    agg = mod._percentiles(fixed)
    assert agg["n"] == 5
    assert agg["min"] == 1.0
    assert agg["max"] == 5.0
    assert agg["mean"] == pytest.approx(3.0)
    # Linear-interpolation percentile at p=0.50 on [1,2,3,4,5] = 3.0
    assert agg["p50"] == pytest.approx(3.0)
    # p25 on [1,2,3,4,5]: pos = 0.25 * 4 = 1.0 → srt[1] = 2.0
    assert agg["p25"] == pytest.approx(2.0)
    # p75 on [1,2,3,4,5]: pos = 0.75 * 4 = 3.0 → srt[3] = 4.0
    assert agg["p75"] == pytest.approx(4.0)


def test_cli_help():
    """Verify CLI --help exits 0 and mentions the script's purpose."""
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
    assert "--smiles-csv" in result.stdout
    assert "--out-json" in result.stdout


def test_cli_end_to_end(tmp_path):
    """End-to-end CLI run: write a 3-row CSV, run script, inspect JSON."""
    mod = _load_module()
    csv_path = tmp_path / "smiles.csv"
    out_path = tmp_path / "panel.json"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "extra_col"])
        writer.writerow(["CCO", "ethanol"])
        writer.writerow(["CC(=O)Oc1ccccc1C(=O)O", "aspirin"])
        writer.writerow(["c1ccccc1", "benzene"])

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--smiles-csv", str(csv_path),
            "--out-json", str(out_path),
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
    assert panel["n_input"] == 3
    assert panel["n_parsed"] == 3
    assert panel["n_failed"] == 0
    assert "aggregate" in panel
    assert "sa" in panel["aggregate"]
    assert "qed" in panel["aggregate"]
    # Cross-check with in-process build_panel (independent code path).
    panel_ref = mod.build_panel(
        ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    )
    sa_ref = [m["sa"] for m in panel_ref["per_molecule"]]
    sa_out = [m["sa"] for m in panel["per_molecule"]]
    for a, b in zip(sa_ref, sa_out):
        assert a == pytest.approx(b, abs=1e-9)
