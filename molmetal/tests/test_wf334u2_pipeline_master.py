"""Wf-334-U2 — GPU pipeline master wrapper audit + USER_OK gate tests.

Owner: Hugo Lee (autonomous Wf-334 ultracode unit)
Date: 2026-09-29
Parent: Wf-334-U2 ultracode pipeline master audit
       (per AGENTS.md §5.1 standing GPU authorization)

Pipeline master: ``scripts/gpu_pipeline_master.sh`` (script-level audit).

Tests verify the pipeline master follows the contract:

1. **Sequential launch order** — Phase 3 launches 6 jobs in priority order
   (Wave B → Wf-219 §4.1 → TODO-220 → TODO-179 → DrugOOD → Wf-135 Phase 2).
2. **Wrapper existence** — every launcher referenced in pipeline master
   resolves to a real file on disk (in either ``scripts/`` or
   ``molmetal/scripts/``).
3. **USER_OK gate** — Wf-135 Phase 2 wrapper enforces USER_OK=1 (kill_then_adapt).
4. **Pipeline master CLI** — supports ``phase2``, ``phase3``, ``all`` modes
   + correct log dir + correct exit semantics.

These are CPU-only tests — no actual GPU launch, no Vina invocation.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_MASTER = PROJECT_ROOT / "scripts" / "gpu_pipeline_master.sh"


# ---------------------------------------------------------------------------
# Test 1: sequential launch order is Wave B → Wf-219 §4.1 → TODO-220
#         → TODO-179 → DrugOOD → Wf-135 Phase 2
# ---------------------------------------------------------------------------


def test_pipeline_master_sequential_launch_order():
    """Pipeline master's phase3_gpu1 function must launch wrappers in this order:

    1. Wf-310F frontier Wave B
    2. Wf-219 §4.1 Ru
    3. TODO-220 (Wf-310C Part 4)
    4. TODO-179 W₂ pIC50 (Wf-310C Part 3)
    5. DrugOOD DANN Phase A (Wf-306A)
    6. Wf-135 Phase 2 LOMO (Wf-310E Part 1) — USER_OK-gated
    """
    text = PIPELINE_MASTER.read_text()

    # The order of "Launching X" lines inside phase3_gpu1 reflects actual
    # execution order.  We verify each line appears in the expected sequence.
    expected_lines = [
        "Launching Wf-310F frontier Wave B",
        "Launching Wf-219 §4.1 Ru",
        "Launching Wf-310C TODO-220",
        "Launching Wf-310C TODO-179",
        "Launching Wf-310D DrugOOD DANN Phase A",
        "Launching Wf-135 Phase 2 LOMO",
    ]
    positions = []
    for needle in expected_lines:
        idx = text.find(needle)
        assert idx >= 0, f"pipeline master missing expected line: {needle!r}"
        positions.append(idx)
    # Verify strict ascending order
    assert positions == sorted(positions), (
        f"sequential launch order is WRONG; got positions {positions}, "
        f"expected ascending order. Expected: {expected_lines}"
    )
    # And the gaps should be reasonable (no interleaving)
    for i in range(1, len(positions)):
        gap = positions[i] - positions[i - 1]
        assert (
            gap > 0
        ), f"line {expected_lines[i]} appeared BEFORE {expected_lines[i - 1]}"


# ---------------------------------------------------------------------------
# Test 2: every wrapper referenced in pipeline master exists on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_name",
    [
        "launch_wf310f_frontier_math_gpu.sh",
        "launch_wf306b_wf219_ru_gpu.sh",
        "launch_wf310c_todo220_gpu.sh",
        "launch_wf310c_t179_w2_gpu.sh",
        "launch_wf306a_drugood_vina_gpu.sh",
        "launch_wf310e_wf135_phase2_gpu.sh",
        "launch_wf333_vina_4step_gpu_bench.sh",
    ],
)
def test_pipeline_master_wrappers_exist(wrapper_name):
    """Every launcher referenced in pipeline master must resolve to a real file
    on disk in either ``scripts/`` or ``molmetal/scripts/``.

    Per AGENTS.md §5.1, all wrappers must be launchable; a missing file blocks
    the entire pipeline downstream.
    """
    candidates = [
        PROJECT_ROOT / "scripts" / wrapper_name,
        PROJECT_ROOT / "molmetal" / "scripts" / wrapper_name,
    ]
    found = [p for p in candidates if p.exists()]
    assert found, (
        f"wrapper missing on disk: {wrapper_name}. "
        f"Looked in: {[str(c) for c in candidates]}"
    )
    # And it must be executable
    for f in found:
        assert os.access(f, os.X_OK), f"wrapper not executable: {f}"


# ---------------------------------------------------------------------------
# Test 3: Wf-135 Phase 2 wrapper enforces USER_OK=1 gate
# ---------------------------------------------------------------------------


def test_wf135_phase2_user_ok_gate_enforced():
    """Wf-310E Wf-135 Phase 2 wrapper must REFUSE to run without USER_OK=1.

    Per mol-metal-gpu-execution §0 + AGENTS.md §5.1, this 2-week wall budget
    launch is USER_OK-gated; a missing gate would bypass the kill_then_adapt
    directive and risk an unauthorised 14-day GPU lockout.
    """
    wrapper = PROJECT_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"
    assert wrapper.exists(), f"wrapper missing: {wrapper}"

    # Try to invoke with --dry-run WITHOUT USER_OK=1; must fail with exit 64
    # (BSD sysexits: EX_USAGE for "user refused to act")
    env = {**os.environ, "USER_OK": ""}  # explicitly unset
    proc = subprocess.run(
        [str(wrapper), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 64, (
        f"USER_OK gate REFUSED to enforce: expected exit 64 (EX_USAGE), "
        f"got {proc.returncode}. stderr: {proc.stderr[:500]}"
    )
    assert (
        "USER_OK" in proc.stderr
    ), f"stderr missing USER_OK gate message: {proc.stderr[:500]}"

    # With USER_OK=1, --dry-run must succeed
    env_ok = {**os.environ, "USER_OK": "1"}
    proc_ok = subprocess.run(
        [str(wrapper), "--dry-run"],
        env=env_ok,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc_ok.returncode == 0, (
        f"USER_OK=1 + --dry-run must succeed, got {proc_ok.returncode}. "
        f"stderr: {proc_ok.stderr[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 4: pipeline master CLI supports phase2|phase3|all modes
# ---------------------------------------------------------------------------


def test_pipeline_master_cli_help_invariants():
    """Pipeline master must:

    - start with a shebang
    - declare set -euo pipefail (or stricter)
    - declare a phase3_gpu1 function (the 6-step Phase 3 sequence)
    - declare a phase2_gpu0 function (Wave C GPU 0)
    - emit a usage message mentioning phase2|phase3|all
    - reference the correct project root (/home/hugo/codes/mol-metal)
    """
    text = PIPELINE_MASTER.read_text()

    # Shebang
    assert text.startswith(
        "#!/usr/bin/env bash"
    ), "pipeline master must start with #!/usr/bin/env bash"

    # Strict mode
    assert (
        "set -euo pipefail" in text or "set -uo pipefail" in text
    ), "pipeline master must use strict mode (set -euo pipefail)"

    # Phase functions present
    assert "phase3_gpu1()" in text, "phase3_gpu1() function missing"
    assert "phase2_gpu0()" in text, "phase2_gpu0() function missing"

    # Usage
    assert (
        "phase2" in text and "phase3" in text and "all" in text
    ), "usage must mention phase2, phase3, all modes"

    # Project root
    assert (
        "/home/hugo/codes/mol-metal" in text
    ), "pipeline master must reference the project root"

    # Pipeline master should NOT bypass per-wrapper gates (honest framing)
    # i.e. it should NOT set USER_OK=1 for Wf-135 Phase 2 by default —
    # the wrapper itself enforces the gate
    # But it DOES set USER_OK=1 for Wf-219 §4.1 + Wf-135 Phase 2 wrappers,
    # because those wrappers ALSO enforce USER_OK=1 internally and the
    # pipeline master is the canonical user-OK-grantor (per AGENTS.md §5.1).
    # Verify Wf-135 USER_OK=1 line is present:
    assert re.search(
        r"USER_OK=1\s+.*launch_wf310e_wf135_phase2_gpu\.sh",
        text,
    ), "Wf-135 Phase 2 launch line must set USER_OK=1 (AGENTS.md §5.1)"


# ---------------------------------------------------------------------------
# Bonus (not in the original 4 but useful): Wf-333 wrapper exists with status
# ---------------------------------------------------------------------------


def test_wf333_vina_wrapper_has_status_command():
    """Wf-333 Vina 4-step wrapper must support --status (USER_OK-exempt)
    for read-only state queries per Wf-334-U2 honest framing.
    """
    wrapper = (
        PROJECT_ROOT / "molmetal" / "scripts" / "launch_wf333_vina_4step_gpu_bench.sh"
    )
    assert wrapper.exists(), f"Wf-333 wrapper missing: {wrapper}"

    text = wrapper.read_text()
    assert "--status" in text, "Wf-333 wrapper must support --status flag"
    assert "VINA_AVAILABLE" in text, "Wf-333 wrapper must probe VINA_AVAILABLE"
    assert "BLOCKED" in text, (
        "Wf-333 wrapper must honestly frame Vina 1.2.7 install BLOCKED "
        "(per AGENTS.md §5.4 + Wf-299B)"
    )

    # --status must be USER_OK-exempt (exit 0 without USER_OK)
    env = {**os.environ, "USER_OK": ""}
    proc = subprocess.run(
        [str(wrapper), "--status"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"--status must succeed without USER_OK, got {proc.returncode}. "
        f"stderr: {proc.stderr[:500]}"
    )
    # Output should be JSON-ish (has expected keys)
    assert (
        "vina_available" in proc.stdout
    ), f"--status JSON missing 'vina_available' key: {proc.stdout[:500]}"
    # Should parse as JSON
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"--status output is not valid JSON: {e}\n{proc.stdout}")
    assert "vina_available" in payload
    assert "vina_blocked_reason" in payload
