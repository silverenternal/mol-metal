"""Wf-353-B2 — verify Wf-343c2 per-wave + Wf-343c3 data quality integration end-to-end.

Per Wf-343c2 + Wf-343c3: per-wave check + data quality check were shipped as separate
scripts. Wf-346-B2 found +2 unique gates (`p3_wf343c2_per_wave_runtime_check` +
`p3_wf343c3_data_quality_check`) registered. Need to verify end-to-end.

Tests:
  1. wf343c2 contract: per-wave script runs to completion, emits valid 7-check JSON
     schema, and reports honest PASS/SKIP/FAIL with non-empty evidence.
  2. wf343c3 contract: data quality script runs to completion on real data
     loaders, emits PASS verdicts across 5 loaders with valid schema.

All tests CPU-only; use existing reports (no GPU launches).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
VENV_PY = sys.executable

REPORT_C2 = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf343c2_per_wave_check"
    / "per_wave_check.json"
)
REPORT_C3 = REPO_ROOT / "molmetal" / "reports" / "wf343c3_data_quality" / "report.json"

EXPECTED_CHECK_IDS_C2 = frozenset(
    {
        "preflight_freshness_24h",
        "preflight_no_fail_gates",
        "hyperparams_config_complete",
        "yuelbond_decoder_registered",
        "wave_budget_remaining_nonneg",
        "gpu_vram_available",
        "disk_free_space",
    }
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess with PYTHONPATH prepended."""
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)


def test_wf343c2_per_wave_check_contract() -> None:
    """Wf-343c2 per-wave check: PASS/SKIP/FAIL contract + 7-check schema.

    Runs `wf343c2_per_wave_check.py --wave 1 --skip-live` and verifies:
      - exit code is 0 or 1 (PASS or FAIL, never crash)
      - JSON output written to REPORT_C2 has n_checks == 7
      - check_id union == EXPECTED_CHECK_IDS_C2
      - each check has status ∈ {PASS, FAIL, SKIP} + non-empty evidence
      - schema_valid == True
      - honest_framing field is non-empty
    """
    proc = _run(
        [
            VENV_PY,
            str(SCRIPTS_DIR / "wf343c2_per_wave_check.py"),
            "--wave",
            "1",
            "--skip-live",
        ]
    )
    assert proc.returncode in (0, 1), (
        f"wf343c2_per_wave_check crashed: rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
    )
    assert REPORT_C2.exists(), (
        f"per_wave_check.json not written: {REPORT_C2}\n"
        f"stderr: {proc.stderr[-500:]}"
    )
    payload = json.loads(REPORT_C2.read_text())
    assert payload["n_checks"] == 7, f"expected 7 checks, got {payload['n_checks']}"
    check_ids = {c["check_id"] for c in payload["checks"]}
    assert check_ids == EXPECTED_CHECK_IDS_C2, (
        f"check_id mismatch: missing={EXPECTED_CHECK_IDS_C2 - check_ids}, "
        f"extra={check_ids - EXPECTED_CHECK_IDS_C2}"
    )
    for c in payload["checks"]:
        assert c["status"] in {
            "PASS",
            "FAIL",
            "SKIP",
        }, f"check {c['check_id']} has invalid status {c['status']}"
        assert c.get("evidence"), f"check {c['check_id']} missing evidence"
    assert (
        payload.get("schema_valid") is True
    ), f"schema_valid must be True, got {payload.get('schema_valid')}"
    assert payload.get("honest_framing"), "honest_framing field missing"


def test_wf343c3_data_quality_check_contract() -> None:
    """Wf-343c3 data quality check: PASS contract on real data loaders.

    Runs `wf343c3_data_quality_check.py` standalone and verifies:
      - exit code is 0 (overall PASS)
      - JSON output written to REPORT_C3 has loader_count == 5
      - each loader report contains PASS checks (real-data quality)
      - schema includes checks_per_loader field
      - all 5 expected loaders present
    """
    proc = _run([VENV_PY, str(SCRIPTS_DIR / "wf343c3_data_quality_check.py")])
    assert proc.returncode == 0, (
        f"wf343c3_data_quality_check failed: rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
    )
    assert REPORT_C3.exists(), (
        f"report.json not written: {REPORT_C3}\n" f"stderr: {proc.stderr[-500:]}"
    )
    payload = json.loads(REPORT_C3.read_text())
    assert (
        payload.get("loader_count") == 5
    ), f"expected 5 loaders, got {payload.get('loader_count')}"
    assert (
        payload.get("checks_per_loader") >= 1
    ), f"checks_per_loader must be ≥ 1, got {payload.get('checks_per_loader')}"
    loader_reports = payload.get("loader_reports", [])
    assert (
        len(loader_reports) == 5
    ), f"expected 5 loader_reports, got {len(loader_reports)}"
    loader_names = {lr.get("loader") for lr in loader_reports}
    expected_loaders = {
        "load_binding_db",
        "load_nci60",
        "load_drugood_train",
        "metalcytotox_5metal_multitask_v4",
        "platinai_4cell",
    }
    assert loader_names == expected_loaders, (
        f"loader mismatch: missing={expected_loaders - loader_names}, "
        f"extra={loader_names - expected_loaders}"
    )
    # Per real-data contract: each loader should have at least 1 PASS check.
    for lr in loader_reports:
        pass_count = sum(1 for c in lr.get("checks", []) if c.get("status") == "PASS")
        assert (
            pass_count >= 1
        ), f"loader {lr.get('loader')} has 0 PASS checks on real data"
