"""Wf-364-B2 — GPU launch dashboard UX polish + activation gate tests.

Owner:        Hugo Lee (autonomous Wf-364-B2 ultracode unit)
Date:         2026-09-29
Origin:       Wf-360-A3 GPU launch dashboard scaffold + Wf-363-B1/B2 UX gaps
Parent:       Wf-360-A3 + Wf-363-A1/A2 (DRY_RUN.log polish + USER_OK UX audit)
              + Wf-364-B1 (pre-flight cascade root-cause fix)
              + Wf-364-B2 (UX polish + activation gate, this wave)

Per the user directive 2026-09-29: 之前的workflow (Wf-360-A3 + Wf-363-B1/B2)
没彻底解决 dashboard UX 问题,需要 launch 新的 ultracode 做 follow up。
Wf-364-B2 closes:
  1. `--status-summary` compact view flag (NEW).
  2. `--activate` activation gate (default = READ-ONLY scaffold; LIVE only
     when operator explicitly passes --activate).

These 2 NEW tests verify:
  1. test_01_status_summary_compact_output — bash + python --status-summary
     produces compact TSV with the 4 expected columns and ≥ 7 rows.
  2. test_02_activation_gate_default_readonly — bash + python default mode
     is READ-ONLY_SCAFFOLD; --activate flag toggles to LIVE.

All 2 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU
launches; aggregator + dashboard bash are read-only).
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

EXPECTED_LAUNCHERS = {
    "wf355_a1", "wf355_a2", "wf355_a3", "wf355_b1", "wf355_b2",
    "g5_retrain", "g6_recover",
}

# ---------------------------------------------------------------------------
# Test 1: --status-summary compact output
# ---------------------------------------------------------------------------

def test_01_status_summary_compact_output():
    """Verify --status-summary flag produces compact TSV with 4 columns and ≥ 7 rows.

    Both bash dashboard (scripts/gpu_launch_dashboard.sh --status-summary) and
    python aggregator (molmetal/scripts/gpu_launch_aggregator.py --status-summary
    --stdout) must produce a tab-separated view with header
    "NAME\\tSTATUS\\tPAIRS\\tWALL_PROJ_HR" and one row per known launcher.
    """
    # --- bash dashboard --status-summary ---
    # bash syntax check
    result = subprocess.run(
        ["bash", "-n", str(DASHBOARD_SH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    # bash --status-summary
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--status-summary"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--status-summary failed: {result.stderr}"

    out = result.stdout.strip()
    # Skip mode-banner lines that start with "dashboard_mode=".
    data_lines = [
        l for l in out.splitlines()
        if l.strip() and not l.startswith("dashboard_mode=")
    ]
    # Header + ≥ 7 launcher rows
    assert len(data_lines) >= 1 + len(EXPECTED_LAUNCHERS), (
        f"--status-summary: expected ≥ {1 + len(EXPECTED_LAUNCHERS)} data lines, "
        f"got {len(data_lines)}:\n{out}"
    )
    # First non-banner line is header
    header = data_lines[0]
    assert "\t" in header, f"header should be tab-separated: {header!r}"
    cols = header.split("\t")
    assert len(cols) == 4, f"expected 4 columns in header, got {len(cols)}: {cols}"
    assert cols[0].upper() == "NAME"
    assert cols[1].upper() == "STATUS"
    assert cols[2].upper() == "PAIRS"
    assert cols[3].upper() == "WALL_PROJ_HR"
    # Each data row has 4 tab-separated columns
    for line in data_lines[1:]:
        row_cols = line.split("\t")
        assert len(row_cols) == 4, f"data row should have 4 columns, got {len(row_cols)}: {line!r}"
    # All 7 launchers present in data rows
    body_rows = data_lines[1:]
    launcher_names = {row.split("\t")[0] for row in body_rows}
    assert launcher_names == EXPECTED_LAUNCHERS, (
        f"--status-summary launcher mismatch: got {launcher_names}, expected {EXPECTED_LAUNCHERS}"
    )

    # --- python aggregator --status-summary --stdout ---
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR_PY), "--status-summary", "--stdout"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"aggregator --status-summary failed: {result.stderr}"

    out = result.stdout.strip()
    data_lines = [
        l for l in out.splitlines()
        if l.strip() and not l.startswith("dashboard_mode=")
    ]
    assert len(data_lines) >= 1 + len(EXPECTED_LAUNCHERS), (
        f"aggregator --status-summary: expected ≥ {1 + len(EXPECTED_LAUNCHERS)} data lines, "
        f"got {len(data_lines)}:\n{out}"
    )
    header = data_lines[0]
    cols = header.split("\t")
    assert len(cols) == 4, f"aggregator expected 4 columns, got {len(cols)}: {cols}"
    body_rows = data_lines[1:]
    launcher_names = {row.split("\t")[0] for row in body_rows}
    assert launcher_names == EXPECTED_LAUNCHERS, (
        f"aggregator --status-summary launcher mismatch: got {launcher_names}, "
        f"expected {EXPECTED_LAUNCHERS}"
    )


# ---------------------------------------------------------------------------
# Test 2: activation gate (default = READ-ONLY scaffold)
# ---------------------------------------------------------------------------

def test_02_activation_gate_default_readonly():
    """Verify activation gate: default mode is READ-ONLY_SCAFFOLD, --activate flips to LIVE.

    Both bash dashboard and python aggregator must:
      - DEFAULT (no --activate flag): dashboard_mode = READ-ONLY_SCAFFOLD,
        activation_flag_set = False.
      - --activate flag: dashboard_mode = LIVE, activation_flag_set = True.

    The activation gate is the Wf-364-B2 user-protection: live progress tracking
    is USER_OK-gated per AGENTS.md §5.1; the operator must explicitly opt-in
    by passing --activate.
    """
    # --- bash dashboard: default mode ---
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--status-summary"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"bash default mode failed: {result.stderr}"
    out = result.stdout
    # First line should advertise dashboard_mode
    first_line = out.splitlines()[0]
    assert first_line.startswith("dashboard_mode="), (
        f"expected first line to start with 'dashboard_mode=', got: {first_line!r}"
    )
    assert "READ-ONLY_SCAFFOLD" in first_line, (
        f"default mode must be READ-ONLY_SCAFFOLD, got: {first_line!r}"
    )

    # --- bash dashboard: --activate mode ---
    result = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--status-summary", "--activate"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"bash --activate mode failed: {result.stderr}"
    first_line = result.stdout.splitlines()[0]
    assert "LIVE" in first_line, (
        f"--activate should set mode=LIVE, got: {first_line!r}"
    )
    # LIVE mode should also emit an ACTIVATION log line to stderr
    stderr = result.stderr
    assert "LIVE mode" in stderr or "ACTIVATION" in stderr, (
        f"--activate should emit ACTIVATION log, got stderr: {stderr!r}"
    )

    # --- python aggregator: default mode (no --activate) ---
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR_PY), "--stdout"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"aggregator default mode failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["dashboard_mode"] == "READ-ONLY_SCAFFOLD", (
        f"default aggregator mode should be READ-ONLY_SCAFFOLD, got: {data['dashboard_mode']!r}"
    )
    assert data["activation_flag_set"] is False, (
        f"default activation_flag_set should be False, got: {data['activation_flag_set']!r}"
    )
    # activation_gated invariant
    assert data["activation_gated"] is True
    assert data["user_ok_required"] is True

    # --- python aggregator: --activate mode ---
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR_PY), "--stdout", "--activate"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"aggregator --activate mode failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["dashboard_mode"] == "LIVE", (
        f"--activate should set dashboard_mode=LIVE, got: {data['dashboard_mode']!r}"
    )
    assert data["activation_flag_set"] is True, (
        f"--activate should set activation_flag_set=True, got: {data['activation_flag_set']!r}"
    )
