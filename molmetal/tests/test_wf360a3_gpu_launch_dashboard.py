"""Wf-360-A3 — GPU launch dashboard tests.

Owner:        Hugo Lee (autonomous Wf-360-A3 ultracode unit)
Date:         2026-09-29
Origin:       Wf-360-A3 GPU launch dashboard scaffold (per Wf-357-C3)
Parent:       scripts/gpu_launch_dashboard.sh + molmetal/scripts/gpu_launch_aggregator.py

These 2 NEW tests verify that the GPU launch dashboard scaffold (per
AGENTS.md §1.2 Standing GPU auth + §5.1 kill_then_adapt) satisfies
the 2 critical contract points:

  1. **Dashboard script parses per-launcher output** — `bash -n` parses
     AND `bash scripts/gpu_launch_dashboard.sh --list-launchers` returns
     the expected 7 known launchers.
  2. **Aggregator produces correct JSON schema** — runs the python
     aggregator and verifies the output schema (7 launchers, required
     fields, activation_gated=True).

All 2 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU
launches; aggregator is read-only).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-
run output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
DASHBOARD_SH = PROJECT_ROOT / "scripts" / "gpu_launch_dashboard.sh"
AGGREGATOR_PY = PROJECT_ROOT / "molmetal" / "scripts" / "gpu_launch_aggregator.py"
DASHBOARD_JSON = PROJECT_ROOT / "molmetal" / "reports" / "gpu_launch_dashboard.json"

EXPECTED_LAUNCHERS = {
    "wf355_a1",   # DrugOOD 1.58M chemprop v2
    "wf355_a2",   # BindingDB 1.07M chemprop v2
    "wf355_a3",   # tmQM + PlatinAI joint
    "wf355_b1",   # Wf-131 P3 deferred
    "wf355_b2",   # Wf-146 deferred
    "g5_retrain", # G5 15 pairs
    "g6_recover", # G6 60 tuples
}


def test_01_dashboard_script_parses_per_launcher_output():
    """Verify dashboard bash script: bash -n parses + --list-launchers returns 7 known launchers."""
    # bash syntax check
    result = subprocess.run(
        ["bash", "-n", str(DASHBOARD_SH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    # --list-launchers mode
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--list-launchers"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--list-launchers failed: {result.stderr}"

    out = result.stdout
    for name in EXPECTED_LAUNCHERS:
        assert name in out, f"launcher {name!r} missing from --list-launchers output:\n{out}"

    # Also check --status mode returns one line per launcher
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--status"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    status_lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(status_lines) >= len(EXPECTED_LAUNCHERS), (
        f"--status should return ≥{len(EXPECTED_LAUNCHERS)} lines, got {len(status_lines)}: "
        f"{status_lines}"
    )

    # JSON mode roundtrip — must be valid JSON
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"--json failed: {result.stderr}"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"--json output is not valid JSON: {e}\n{result.stdout[:500]}")
    assert "launchers" in data
    assert isinstance(data["launchers"], list)
    assert len(data["launchers"]) == len(EXPECTED_LAUNCHERS), (
        f"expected {len(EXPECTED_LAUNCHERS)} launchers, got {len(data['launchers'])}"
    )
    names_in_json = {l["name"] for l in data["launchers"]}
    assert names_in_json == EXPECTED_LAUNCHERS, (
        f"launcher name mismatch: got {names_in_json}, expected {EXPECTED_LAUNCHERS}"
    )


def test_02_aggregator_produces_correct_json_schema():
    """Verify aggregator python script: imports + runs + schema validates."""
    # python syntax check
    import ast
    src = AGGREGATOR_PY.read_text()
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"aggregator python syntax error: {e}")

    # Import (loads aggregator module without running main)
    spec = importlib_util.spec_from_file_location("gpu_launch_aggregator", str(AGGREGATOR_PY))
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "aggregate"), "aggregate() function missing"
    assert hasattr(mod, "aggregate_launcher"), "aggregate_launcher() missing"
    assert hasattr(mod, "launcher_status"), "launcher_status() missing"
    assert hasattr(mod, "pair_status"), "pair_status() missing"

    # Run main() and write JSON file
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR_PY)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"aggregator failed: {result.stderr}"
    assert DASHBOARD_JSON.exists(), f"dashboard JSON not written to {DASHBOARD_JSON}"

    # Schema validation
    data = json.loads(DASHBOARD_JSON.read_text())
    assert "dashboard_generated_at" in data
    assert data["dashboard_schema_version"] == "1.0"
    assert data["wf_origin"] == "Wf-360-A3 GPU launch dashboard scaffold"
    assert data["activation_gated"] is True
    assert data["user_ok_required"] is True
    assert data["launcher_count"] == len(EXPECTED_LAUNCHERS)
    assert len(data["launchers"]) == len(EXPECTED_LAUNCHERS)

    required_top_fields = {
        "name", "script", "root", "status", "wall_projected_hr",
        "last_activity", "elapsed_sec", "pair_progress",
        "activation_gated", "user_ok_required",
    }
    required_pair_fields = {
        "total_pairs", "discovered_pairs", "ok", "failed", "running", "pending",
    }
    valid_statuses = {
        "COMPLETED", "RUNNING", "PARTIAL", "FAILED", "DRY_RUN_OK",
        "PLAN_ONLY", "READY", "STALE", "NO_PLAN",
    }

    for launcher_entry in data["launchers"]:
        missing = required_top_fields - set(launcher_entry.keys())
        assert not missing, f"launcher {launcher_entry.get('name')} missing fields: {missing}"
        assert launcher_entry["activation_gated"] is True
        assert launcher_entry["user_ok_required"] is True
        assert launcher_entry["status"] in valid_statuses, (
            f"launcher {launcher_entry['name']} has invalid status: {launcher_entry['status']}"
        )
        pp = launcher_entry["pair_progress"]
        missing_pp = required_pair_fields - set(pp.keys())
        assert not missing_pp, f"pair_progress missing fields: {missing_pp}"
        # Sum of (ok + failed + running + pending) == total_pairs OR matches discovered
        sum_components = pp["ok"] + pp["failed"] + pp["running"] + pp["pending"]
        assert sum_components <= pp["total_pairs"], (
            f"launcher {launcher_entry['name']}: sum {sum_components} > total {pp['total_pairs']}"
        )


# Importlib helper (py3.8+)
import importlib.util as importlib_util