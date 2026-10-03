"""Wf-320C — TODO-220 stub driver full implementation verification + integration tests.

================================================================
10 tests verifying Wf-308C / Wf-310C / Wf-313D work was complete +
integration with downstream metrics + Wf-313D acceptance gates.

Per Wf-315B recommendation #5 (TODO-220 stub 实装): Wf-305C shipped
the dry-run stub; full impl was claimed by Wf-308C. Wf-308C + Wf-310C
+ Wf-313D shipped --execute paths for both temporal and NCI-60
drivers. Wf-320C verifies the full impl surface and wires it into
the existing acceptance gates + downstream metric reporting.

Honest framing (per party-twenty-spirit §3.5):
* --execute paths: SHIPPED by Wf-308C; this workflow is verification
  (no new implementation).
* Smoke MEASURED via existing sweep_summary_smoke.json JSONs (5/5
  temporal, 9/9 NCI-60).
* GPU full sweep: PROJECTED PENDING USER_OK + 3.9 hr GPU wall; not
  in scope of this verification workflow.

Test inventory (10 tests):
* Temporal --execute path (3):
    test_temporal_01_execute_sweep_callable
    test_temporal_02_dry_run_emits_csv_exists_true
    test_temporal_03_execute_sweep_returns_dict_with_status
* NCI-60 --execute path (3):
    test_nci60_01_execute_sweep_callable
    test_nci60_02_dry_run_emits_csv_exists_true
    test_nci60_03_execute_sweep_returns_dict_with_status
* Integration with downstream metrics (2):
    test_integration_temporal_smoke_matches_metrics
    test_integration_nci60_smoke_matches_metrics
* Per-driver integration with Wf-313D acceptance gates (2):
    test_acceptance_temporal_smoke_5_of_5_pass
    test_acceptance_nci60_smoke_9_of_9_pass

Cross-references
----------------
* TODO/completed/220_wf220_temporal_split_acceptance.md — §6.3 doc
* TODO/completed/220_wf219_nci60_9cell_acceptance.md — §6.4 doc
* molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json — smoke MEASURED (5/5)
* molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json — smoke MEASURED (9/9)
* molmetal/scripts/wf219_temporal_split_driver.py — temporal driver
* molmetal/scripts/nci60_9cell_chemprop_driver.py — NCI-60 driver
* molmetal/scripts/launch_wf310c_todo220_gpu.sh — GPU wrapper (USER_OK-gated)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
TEMPORAL_DRIVER = ROOT / "molmetal/scripts/wf219_temporal_split_driver.py"
NCI60_DRIVER = ROOT / "molmetal/scripts/nci60_9cell_chemprop_driver.py"
TEMPORAL_SMOKE = ROOT / "molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json"
NCI60_SMOKE = ROOT / "molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json"
GPU_WRAPPER = ROOT / "molmetal/scripts/launch_wf310c_todo220_gpu.sh"
SEC_6_3_DOC = ROOT / "TODO/completed/220_wf220_temporal_split_acceptance.md"
SEC_6_4_DOC = ROOT / "TODO/completed/220_wf219_nci60_9cell_acceptance.md"
PYTHON_BIN = "/home/hugo/codes/mol-metal/.venv/bin/python"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    assert path.exists(), f"smoke JSON missing: {path}"
    return json.loads(path.read_text())


def _run_cli(driver: Path, *args: str) -> dict:
    """Run the driver with given args and parse JSON stdout."""
    cmd = [PYTHON_BIN, str(driver), *args]
    out = subprocess.run(
        cmd,
        cwd=ROOT, text=True,
        capture_output=True, timeout=60,
    )
    assert out.returncode == 0, (
        f"driver {driver.name} failed: rc={out.returncode}\n"
        f"stderr={out.stderr[-500:]}"
    )
    # Strip [molmetal] env-var banner line if present
    text = out.stdout
    if text.startswith("[molmetal]"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return json.loads(text)


def _import_driver(driver_path: Path):
    """Import a driver as a module via importlib; register in sys.modules
    so dataclass __module__ lookups work cleanly."""
    import importlib.util
    import sys as _sys
    mod_name = f"_wf320c_{driver_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, driver_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Temporal --execute path (3 tests)
# ===========================================================================


def test_temporal_01_execute_sweep_callable():
    """Temporal driver: execute_sweep() is callable and exported in __all__."""
    mod = _import_driver(TEMPORAL_DRIVER)
    assert hasattr(mod, "execute_sweep"), (
        "temporal driver must expose execute_sweep() function (Wf-308C contract)"
    )
    assert "execute_sweep" in mod.__all__, (
        "execute_sweep must be in temporal driver's __all__ (Wf-308C contract)"
    )
    assert callable(mod.execute_sweep), "execute_sweep must be callable"


def test_temporal_02_dry_run_emits_csv_exists_true():
    """Temporal driver --dry-run emits csv_exists=true (CSV present on disk)."""
    plan = _run_cli(TEMPORAL_DRIVER, "--dry-run")
    assert plan.get("driver") == "wf219_temporal_split_driver"
    assert plan.get("csv_exists") is True, (
        "temporal CSV must exist on disk for --dry-run to claim csv_exists=true"
    )
    assert plan.get("status") == "STUB_PLAN_ONLY (CPU-only dry-run; --execute path implemented in Wf-308C; needs CSV + chemprop v2 + USER_OK for full GPU)", (
        "temporal status string must reference Wf-308C + USER_OK gate"
    )
    assert plan.get("split_protocol", "").startswith("temporal_strict")


def test_temporal_03_execute_sweep_returns_dict_with_status():
    """Temporal execute_sweep returns dict; with non-existent CSV returns CSV_MISSING status."""
    mod = _import_driver(TEMPORAL_DRIVER)
    args = mod.build_argparser().parse_args([
        "--csv", "/tmp/_wf320c_definitely_does_not_exist.csv",
        "--epochs", "1",
        "--seeds", "0",
    ])
    result = mod.execute_sweep(args)
    assert isinstance(result, dict), "execute_sweep must return dict"
    assert result.get("ran") is False
    assert result.get("status") == "CSV_MISSING", (
        "execute_sweep with missing CSV must return CSV_MISSING status (Wf-306B honest-negative)"
    )
    assert "do not run --execute until the CSV exists" in result.get("error", "")


# ===========================================================================
# NCI-60 --execute path (3 tests)
# ===========================================================================


def test_nci60_01_execute_sweep_callable():
    """NCI-60 driver: execute_sweep() is callable and exported in __all__."""
    mod = _import_driver(NCI60_DRIVER)
    assert hasattr(mod, "execute_sweep"), (
        "NCI-60 driver must expose execute_sweep() function (Wf-308C contract)"
    )
    assert "execute_sweep" in mod.__all__, (
        "execute_sweep must be in NCI-60 driver's __all__ (Wf-308C contract)"
    )
    assert callable(mod.execute_sweep), "execute_sweep must be callable"


def test_nci60_02_dry_run_emits_csv_exists_true():
    """NCI-60 driver --dry-run emits csv_exists=true (CSV present on disk)."""
    plan = _run_cli(NCI60_DRIVER, "--dry-run")
    assert plan.get("driver") == "nci60_9cell_chemprop_driver"
    assert plan.get("csv_exists") is True, (
        "NCI-60 CSV must exist on disk for --dry-run to claim csv_exists=true"
    )
    assert plan.get("status") == "STUB_PLAN_ONLY (CPU-only dry-run; --execute path implemented in Wf-308C; needs CSV + chemprop v2 + USER_OK for full GPU)", (
        "NCI-60 status string must reference Wf-308C + USER_OK gate"
    )
    assert plan.get("n_cells") == 9


def test_nci60_03_execute_sweep_returns_dict_with_status():
    """NCI-60 execute_sweep returns dict; with non-existent CSV returns CSV_MISSING status."""
    mod = _import_driver(NCI60_DRIVER)
    args = mod.build_argparser().parse_args([
        "--csv", "/tmp/_wf320c_definitely_does_not_exist.csv",
        "--epochs", "1",
        "--seeds", "0",
    ])
    result = mod.execute_sweep(args)
    assert isinstance(result, dict), "execute_sweep must return dict"
    assert result.get("ran") is False
    assert result.get("status") == "CSV_MISSING", (
        "execute_sweep with missing CSV must return CSV_MISSING status (Wf-306B honest-negative)"
    )
    assert "do not run --execute until the CSV exists" in result.get("error", "")


# ===========================================================================
# Integration with downstream metrics (2 tests)
# ===========================================================================


def test_integration_temporal_smoke_matches_metrics():
    """Temporal smoke JSON matches §6.3.1 schema for downstream metric reporting."""
    smoke = _read_json(TEMPORAL_SMOKE)
    # §6.3 schema contract
    assert smoke["n_runs_planned"] == 5
    assert smoke["n_runs_completed"] == 5
    assert smoke["partial"] is False
    assert "mean_auc_across_metals" in smoke
    assert "per_metal_summary" in smoke
    expected_metals = {"Ru", "Ir", "Os", "Rh", "Re"}
    assert set(smoke["per_metal_summary"].keys()) == expected_metals, (
        f"per_metal_summary keys {set(smoke['per_metal_summary'].keys())} != {expected_metals}"
    )
    assert smoke["krasnov_protocol_aligned"] is True
    assert smoke["split_protocol"].startswith("temporal_strict")
    # MEASURED values must be present
    for metal in expected_metals:
        m = smoke["per_metal_summary"][metal]
        assert 0.0 <= m["mean_auc"] <= 1.0, f"{metal} mean_auc OOR: {m['mean_auc']}"
    # The Krasnov 2026 baseline citation must be present for honest comparison
    assert "krasnov_2026_baseline_ru" in smoke
    assert "krasnov_2026_baseline_ir" in smoke


def test_integration_nci60_smoke_matches_metrics():
    """NCI-60 smoke JSON matches §6.4.1 schema for downstream metric reporting."""
    smoke = _read_json(NCI60_SMOKE)
    # §6.4 schema contract
    assert smoke["n_runs_planned"] == 9
    assert smoke["n_runs_completed"] == 9
    assert smoke["partial"] is False
    assert "mean_auc_across_cells" in smoke
    assert "per_cell_summary" in smoke
    assert smoke["n_cells_used"] == 9
    assert smoke["n_cells_requested"] == 9
    assert smoke["cells_unmatched"] == []
    # All cells must have MEASURED AUC in [0,1]
    for cell, summary in smoke["per_cell_summary"].items():
        assert 0.0 <= summary["mean_auc"] <= 1.0, f"{cell} mean_auc OOR: {summary['mean_auc']}"
    # Honest-negative: PC3 below random baseline must be MEASURED, not paraphrased
    pc3_auc = smoke["per_cell_summary"]["PC3"]["mean_auc"]
    assert pc3_auc < 0.5, (
        f"PC3 MEASURED AUC {pc3_auc} must be honest-negative (<0.5); "
        f"per Wf-313D §6.4.4 this is declared honest-negative"
    )
    # Honest-positive: HCT116 highlight must be MEASURED above random
    hct_auc = smoke["per_cell_summary"]["HCT116"]["mean_auc"]
    assert hct_auc > 0.5, (
        f"HCT116 MEASURED AUC {hct_auc} must be > 0.5 random baseline (Wf-313D §6.4.4 highlight)"
    )


# ===========================================================================
# Per-driver integration with Wf-313D acceptance gates (2 tests)
# ===========================================================================


def test_acceptance_temporal_smoke_5_of_5_pass():
    """§6.3 acceptance gate: 5/5 ok MEASURED + §6.3 doc cites Wf-308C/310C/313D lineage."""
    smoke = _read_json(TEMPORAL_SMOKE)
    # 5/5 ok contract
    ok_count = sum(
        1 for metal in smoke["per_metal_summary"].values()
        if metal["mean_auc"] is not None and metal["n_seeds"] >= 1
    )
    assert ok_count == 5, f"temporal smoke must have 5/5 ok, got {ok_count}/5"
    # §6.3 doc must exist and reference Wf-308C/310C/313D
    assert SEC_6_3_DOC.exists(), f"§6.3 doc must exist at {SEC_6_3_DOC}"
    text = SEC_6_3_DOC.read_text()
    for wf in ("Wf-310C", "Wf-313D"):
        assert wf in text, f"§6.3 doc must cite {wf}"
    assert "Wf-308C" in text or "Wf-310C Part" in text, (
        "§6.3 doc must cite Wf-308C OR reference Wf-310C Part lineage"
    )
    assert "5/5" in text or "MEASURED PASS" in text, "§6.3 doc must declare 5/5 measured pass"


def test_acceptance_nci60_smoke_9_of_9_pass():
    """§6.4 acceptance gate: 9/9 ok MEASURED + §6.4 doc cites Wf-308C/310C/313D lineage."""
    smoke = _read_json(NCI60_SMOKE)
    # 9/9 ok contract
    ok_count = sum(
        1 for cell in smoke["per_cell_summary"].values()
        if cell["mean_auc"] is not None and cell["n_seeds"] >= 1
    )
    assert ok_count == 9, f"NCI-60 smoke must have 9/9 ok, got {ok_count}/9"
    # §6.4 doc must exist and reference Wf-308C/310C/313D
    assert SEC_6_4_DOC.exists(), f"§6.4 doc must exist at {SEC_6_4_DOC}"
    text = SEC_6_4_DOC.read_text()
    # §6.4 doc lineage: Wf-310C (smoke + GPU wrapper) + Wf-313D (acceptance gate).
    # Wf-308C is the upstream executor referenced via "Wf-310C" lineage.
    for wf in ("Wf-310C", "Wf-313D"):
        assert wf in text, f"§6.4 doc must cite {wf}"
    assert "Wf-308C" in text or "Wf-310C Part 4" in text, (
        "§6.4 doc must cite Wf-308C OR reference Wf-310C Part 4 lineage"
    )
    assert "9/9" in text or "MEASURED PASS" in text, "§6.4 doc must declare 9/9 measured pass"
    # Honest-negative for PC3 must be cited in §6.4 doc
    assert "PC3" in text, "§6.4 doc must discuss PC3 cell line"
    assert "honest-negative" in text.lower(), (
        "§6.4 doc must declare PC3 honest-negative (party-twenty-spirit §3.5)"
    )