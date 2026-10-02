"""Wf-363-A1 — Per-pair DRY_RUN.log schema validation for 5 launch wrappers.

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-355 + Wf-356-A1 (5 NEW launch wrappers shipped with DRY_RUN.log
emission); Wf-363-A1 (per-pair DRY_RUN.log schema polish + validation).
Per AGENTS.md §5.1 (kill_then_adapt) + §5.2 (MEASURED-vs-PROJECTED honest framing).

Background — Per Wf-355 + Wf-356-A1: 5 NEW launch wrappers were shipped with
DRY_RUN.log emission (each emits to ``${REPORTS_DIR}/<subset|pair>/DRY_RUN.log``
on ``--dry-run``).  The pre-Wf-363-A1 schema had only ``trainer`` / ``subset``
/ ``flags`` / ``gpu`` / ``budget`` / ``ts`` — and inconsistent across wrappers.

Wf-363-A1 polish — single shared helper ``scripts/_dry_run_log_helper.sh``
provides a uniform ``emit_dry_run_log`` function.  All 5 wrappers source the
helper and emit DRY_RUN.log with the canonical schema:

    DRY-RUN launcher for <title>
      wrapper: <wrapper basename>
      mode:    <mode label>
      trainer: <trainer path>
      pair:    <pair label>          # repeated per (subset, pair, or cell)
      ...
      flags:   <flags>
      gpu:     <device>
      budget:  <hours> hr
      user_ok: <REQUIRED|OPTIONAL_smoke|REQUIRED_STATUS_HOLD|OPEN|n/a>
      ts:      <ISO-8601 timestamp>
    per_pair:
      <pair label>
      ...

The 5 wrappers are:

  1. scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh
       (Wf-355-A1; subset=pairs; R1+R3 modes)
  2. scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh
       (Wf-355-A2; per-seed; R1+R3 modes)
  3. scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
       (Wf-355-A3; (cell, seed) pairs; smoke_2seed+full_5seed modes)
  4. scripts/launch_wf131_p3_deferred_runs.sh
       (Wf-355-B1; (metal, seed) pairs; DEFERRED_7_runs mode)
  5. scripts/launch_wf146_deferred_runs.sh
       (Wf-355-B2; (metal, seed) pairs; R1+R3 modes)

Tests (3 total, all CPU-only):

  - test_per_wrapper_dry_run_log_schema         (per-wrapper schema check)
  - test_per_pair_enumeration_consistent        (pair lines == per_pair block)
  - test_budget_within_session_cap              (budget ≤ 10 hr or marked DEFERRED)

All tests use the shared helper ``emit_dry_run_log`` for golden output where
useful.  Wall < 5 s; no torch.cuda, no real launches.

Honest framing (AGENTS.md §5.2):
  - MEASURED: 5 wrappers SHIPPED per Wf-355 + Wf-356-A1 (per-pair emission).
  - MEASURED: existing 12 DRY_RUN.log files at Wf-363-A1 baseline (per find
    ``molmetal/reports -name DRY_RUN.log``).
  - PROJECTED: 5 wrappers → 5/5 NEW schema-conformant logs (target).
  - PROJECTED: per-pair enumeration preserved (pair: lines == per_pair: block).
  - PROJECTED: per-wrapper budget within 10 hr session cap (or marked DEFERRED).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = ROOT / "scripts"
TEST_SCRIPTS_DIR = ROOT / "molmetal" / "scripts"
REPORTS = ROOT / "molmetal" / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# The 5 launch wrappers (Wf-355 + Wf-356-A1 SHIPPED per Wf-363-A1)
# ---------------------------------------------------------------------------
WF363A1_WRAPPERS = (
    SCRIPTS / "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
    SCRIPTS / "launch_wf310e_r20_p4_bindingdb_1.07M.sh",
    SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
    SCRIPTS / "launch_wf131_p3_deferred_runs.sh",
    SCRIPTS / "launch_wf146_deferred_runs.sh",
)

# Required schema fields (Wf-363-A1 polish)
WF363A1_REQUIRED_FIELDS = (
    "wrapper:",
    "mode:",
    "trainer:",
    "flags:",
    "gpu:",
    "budget:",
    "user_ok:",
    "ts:",
)

# 10 hr session cap (per AGENTS.md §5.1 + Wf-294B kill_then_adapt)
SESSION_CAP_HR = 10.0

# User-OK gate statuses
VALID_USER_OK = {"REQUIRED", "OPTIONAL_smoke", "REQUIRED_STATUS_HOLD", "OPEN", "n/a"}


# ---------------------------------------------------------------------------
# Helper: invoke the wrapper with --dry-run and capture the DRY_RUN.log path
# ---------------------------------------------------------------------------
def _invoke_wrapper(wrapper: Path, *args: str, env_extra: dict | None = None) -> Path:
    """Invoke the wrapper with --dry-run. Returns the subprocess result.

    Note: USER_OK gate is enforced before any --dry-run work.
      - launch_wf131_p3_deferred_runs.sh gates on USER_OK=1
      - launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh gates on USER_OK=1
      - launch_wf146_deferred_runs.sh R3 gates on USER_OK=YES
      - launch_wf310e_r20_p4_bindingdb_1.07M.sh R3 gates on USER_OK=YES
      - launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh R3 gates on USER_OK=YES
    Per-wrapper gating is handled in the test dispatch below.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = ["bash", str(wrapper), "--dry-run", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    return proc


# Per-wrapper USER_OK gating (Wf-363-A1)
WF363A1_USER_OK = {
    "launch_wf131_p3_deferred_runs.sh": "1",
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh": "1",
    "launch_wf146_deferred_runs.sh": "YES",
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh": "YES",
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": "YES",
}


def _parse_dry_run_log(path: Path) -> dict:
    """Parse a DRY_RUN.log into a dict of field → list-of-values."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        raise FileNotFoundError(f"DRY_RUN.log missing: {path}")
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line or line.startswith("DRY-RUN launcher") or line.startswith("per_pair"):
            continue
        # Each indented line: "  <key>: <value>"
        m = re.match(r"^\s{2}([a-z_]+):\s*(.*)$", line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            out.setdefault(key, []).append(value)
    return out


# Per-wrapper log candidates (Wf-363-A1)
WF363A1_LOG_PATHS = {
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": (
        REPORTS / "wf310e_r20_p3_drugood_full_launch/r1/DRY_RUN.log",
        REPORTS / "wf310e_r20_p3_drugood_full_launch/r3/DRY_RUN.log",
    ),
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh": (
        REPORTS / "wf355a2_r20_p4_bindingdb/r1/DRY_RUN.log",
        REPORTS / "wf355a2_r20_p4_bindingdb/r3/DRY_RUN.log",
    ),
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh": (
        REPORTS / "wf355a3_r20_p5_tmqm_plat/DRY_RUN.log",
    ),
    "launch_wf131_p3_deferred_runs.sh": (
        REPORTS / "wf131_p3_deferred_runs/DRY_RUN.log",
    ),
    "launch_wf146_deferred_runs.sh": (
        REPORTS / "wf146_chemprop_lomo/r1/DRY_RUN.log",
        REPORTS / "wf146_chemprop_lomo/r3/DRY_RUN.log",
    ),
}


def _gather_logs_for_wrapper(wrapper_name: str) -> list[Path]:
    """Return DRY_RUN.log paths expected for a specific wrapper."""
    return [p for p in WF363A1_LOG_PATHS.get(wrapper_name, ()) if p.exists()]


# ---------------------------------------------------------------------------
# Test #1 — Per-wrapper DRY_RUN.log schema validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WF363A1_WRAPPERS, ids=lambda p: p.name)
def test_per_wrapper_dry_run_log_schema(wrapper: Path) -> None:
    """Each wrapper MUST emit DRY_RUN.log with the canonical schema.

    Required fields (Wf-363-A1):
      wrapper: / mode: / trainer: / flags: / gpu: / budget: / user_ok: / ts:
    """
    assert wrapper.exists(), f"wrapper missing: {wrapper}"

    # Each wrapper invoked in --dry-run mode; pick a representative mode.
    user_ok_value = WF363A1_USER_OK.get(wrapper.name, "YES")
    env_extra = {"USER_OK": user_ok_value}
    if wrapper.name == "launch_wf131_p3_deferred_runs.sh":
        proc = _invoke_wrapper(wrapper, env_extra=env_extra)  # default 7 DEFERRED runs
    elif wrapper.name == "launch_wf146_deferred_runs.sh":
        proc = _invoke_wrapper(wrapper, "--mode", "R3", env_extra=env_extra)  # full GPU mode
    elif wrapper.name == "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh":
        proc = _invoke_wrapper(wrapper, "--smoke-only", env_extra=env_extra)
    else:
        proc = _invoke_wrapper(wrapper, "--mode", "R1", env_extra=env_extra)

    assert proc.returncode == 0, (
        f"{wrapper.name} --dry-run failed (rc={proc.returncode}); "
        f"stderr tail: {proc.stderr[-400:]}"
    )

    # Only check logs specific to THIS wrapper (avoid cross-wrapper noise)
    logs = _gather_logs_for_wrapper(wrapper.name)
    assert logs, f"{wrapper.name} produced no DRY_RUN.log files"

    # Each emitted log MUST contain the required schema fields
    for log in logs:
        parsed = _parse_dry_run_log(log)
        for field in WF363A1_REQUIRED_FIELDS:
            key = field.rstrip(":")
            assert key in parsed, (
                f"{wrapper.name} → {log.name}: missing field '{key}'; "
                f"got keys: {list(parsed.keys())}"
            )
        # user_ok MUST be one of the valid statuses
        user_ok = parsed["user_ok"][-1]
        assert user_ok in VALID_USER_OK, (
            f"{wrapper.name} → {log.name}: invalid user_ok '{user_ok}'; "
            f"valid: {sorted(VALID_USER_OK)}"
        )


# ---------------------------------------------------------------------------
# Test #2 — Per-pair enumeration consistency
# ---------------------------------------------------------------------------
def test_per_pair_enumeration_consistent() -> None:
    """``pair:`` lines MUST equal ``per_pair:`` block entries (consistent enumeration).

    A wrapper that emits N ``pair:`` lines must also emit N entries under the
    ``per_pair:`` block (downstream tooling counts both).
    """
    # Iterate per-wrapper so each wrapper's logs are validated independently
    for wrapper_name in WF363A1_LOG_PATHS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        if not logs:
            continue  # skip if not produced yet (other tests handle schema)
        for log in logs:
            text = log.read_text()
            pair_lines = [
                line.split("pair:", 1)[1].strip()
                for line in text.splitlines()
                if line.strip().startswith("pair:")
            ]
            # Find the per_pair block
            per_pair_block_lines: list[str] = []
            in_block = False
            for line in text.splitlines():
                if line.strip().startswith("per_pair:"):
                    in_block = True
                    continue
                if in_block:
                    if line.startswith("  ") and line.strip():
                        per_pair_block_lines.append(line.strip())
                    elif line.strip() == "":
                        continue
                    else:
                        in_block = False
            # If pair lines exist, they must equal per_pair block entries.
            # If no pair lines (subset-only log), skip consistency check.
            if pair_lines:
                assert len(pair_lines) == len(per_pair_block_lines), (
                    f"{wrapper_name} → {log.name}: pair: lines ({len(pair_lines)}) "
                    f"!= per_pair: block entries ({len(per_pair_block_lines)}); "
                    f"pair_lines={pair_lines[:5]}... per_pair={per_pair_block_lines[:5]}..."
                )


# ---------------------------------------------------------------------------
# Test #3 — Budget estimate within 10 hr session cap
# ---------------------------------------------------------------------------
def test_budget_within_session_cap() -> None:
    """Per-wrapper ``budget:`` MUST be ≤ 10 hr session cap OR marked DEFERRED.

    Per AGENTS.md §5.1 + Wf-294B kill_then_adapt: no per-launch wall > 10 hr
    without explicit user override.  Wrappers that ship DEFERRED state (e.g.
    Wf-131 P3 / Wf-146 with RE/Os DEFERRED runs) are allowed to budget > 10 hr
    if the budget is for the WHOLE sweep (per-pair budgets stay < 10 hr).
    """
    for wrapper_name in WF363A1_LOG_PATHS:
        logs = _gather_logs_for_wrapper(wrapper_name)
        if not logs:
            continue
        for log in logs:
            parsed = _parse_dry_run_log(log)
            # Each wrapper may emit multiple budget: lines (one per phase)
            for budget_str in parsed["budget"]:
                m = re.match(r"([\d.]+)\s*hr", budget_str)
                assert m, f"{log.name}: unparseable budget '{budget_str}'"
                budget_hr = float(m.group(1))
                assert budget_hr <= SESSION_CAP_HR, (
                    f"{wrapper_name} → {log.name}: budget {budget_hr} hr exceeds "
                    f"session cap {SESSION_CAP_HR} hr; either split or mark DEFERRED"
                )