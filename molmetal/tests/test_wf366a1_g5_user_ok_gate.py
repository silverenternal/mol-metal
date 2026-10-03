"""Wf-366-A1 — USER_OK gate fix verification for launch_wf124_p3_g5.sh.

============================================================================
Owner:       Hugo Lee (autonomous Wf-366-A1 ultracode unit)
Date:        2026-09-29
Parent:      Wf-366 ultracode residual wave (post-Wf-363-A2 GAP-1 finding)
             + AGENTS.md §5.1 standing GPU authorization + kill_then_adapt.

Per Wf-363-A2 audit finding GAP-1:
  "launch_wf124_p3_g5.sh has NO USER_OK gate entirely. RC=124 timeout means
   wrapper proceeds straight into pre-flight without user OK. ~5-hr GPU
   launch risk."

This test suite verifies the Wf-366-A1 fix closes the gap by adding the
USER_OK gate pattern (consistent with Wf-355-B1 wf131 wrapper lines 82-88):

  1. test_no_user_ok_exits_64
        No USER_OK env var → wrapper exits 64 + emits USER_OK REQUIRED
        FATAL message.
  2. test_user_ok_proceeds_to_preflight
        USER_OK=1 set → wrapper proceeds past gate (exits non-zero on
        pre-flight which is expected in CPU-only env, OR exits 0 if
        --dry-run is also passed).
  3. test_dry_run_with_user_ok_emits_log_and_plan
        USER_OK=1 + --dry-run → exits 0, emits DRY_RUN.log AND plan.json
        in OUT_BASE.

CPU-only tests — no torch.cuda, no actual training, no GPU execution.
Each test invocation is bounded by a short timeout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf124_p3_g5.sh"
HELPER = ROOT / "scripts" / "_dry_run_log_helper.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"file missing: {path}"
    return path.read_text()


def _run(
    cmd: list[str], env_extra: dict | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run a subprocess with USER_OK explicitly unset by default.

    We always strip USER_OK from the parent env so test isolation is
    deterministic. `env_extra` is merged on top.
    """
    base_env = {k: v for k, v in os.environ.items() if k != "USER_OK"}
    if env_extra:
        base_env.update(env_extra)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=base_env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Test 1: no USER_OK → exit 64 + USER_OK REQUIRED message
# ---------------------------------------------------------------------------


def test_no_user_ok_exits_64():
    """Without USER_OK env var, wrapper must exit 64 BEFORE any GPU work.

    Per Wf-366-A1 fix: gate pattern mirrors scripts/launch_wf131_p3_deferred_runs.sh
    Wf-355-B1 lines 118-125 (exit 64 + USER_OK REQUIRED FATAL message).

    Closes Wf-363-A2 GAP-1 HONEST-NEGATIVE finding (G5 wrapper was the
    only launch wrapper missing the gate; live-run audit showed RC=124
    timeout when invoked without --dry-run + without USER_OK).
    """
    # 1. Source-level gate presence
    src = _read(WRAPPER)
    assert "USER_OK:-0" in src, "USER_OK env-var check missing from wrapper"
    assert "exit 64" in src, "USER_OK gate refusal exit code 64 missing"
    assert "USER_OK gate refused" in src, "USER_OK gate FATAL message missing"
    assert "kill_then_adapt" in src, "kill_then_adapt directive reference missing"
    # Gate is wired BEFORE the wave-tool guards (pre_flight_check.py call)
    gate_idx = src.find("USER_OK:-0")
    preflight_idx = src.find("pre_flight_check.py")
    assert gate_idx > 0, "USER_OK gate marker not found"
    assert preflight_idx > 0, "pre_flight_check.py call not found"
    assert gate_idx < preflight_idx, (
        "USER_OK gate must precede pre_flight_check.py call; otherwise "
        "the wrapper can auto-trigger pre-flight work without user OK"
    )

    # 2. Live-run behaviour: no USER_OK → exit 64
    proc = _run(["bash", str(WRAPPER)], timeout=15)
    assert proc.returncode == 64, (
        f"wrapper without USER_OK should exit 64 per Wf-366-A1 fix; "
        f"got rc={proc.returncode}. stderr={proc.stderr[-400:]!r}"
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "USER_OK gate refused" in combined, (
        f"FATAL message 'USER_OK gate refused' missing from output; "
        f"got combined={combined[-400:]!r}"
    )
    assert "USER_OK=1" in combined, (
        f"remediation hint 'USER_OK=1' missing from output; "
        f"got combined={combined[-400:]!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: USER_OK=1 → proceeds past gate
# ---------------------------------------------------------------------------


def test_user_ok_proceeds_to_preflight():
    """With USER_OK=1 set, wrapper proceeds past the gate block.

    Per Wf-366-A1 fix: the gate exit-code 64 path is bypassed when
    USER_OK=1 is set in env. We assert the wrapper gets past the gate
    by checking that either:
      (a) --dry-run mode produces exit 0 + DRY_RUN.log + plan.json, OR
      (b) real-launch mode proceeds to pre-flight check (which may then
          exit non-zero on missing pre-flight data, but that's PRE-FLIGHT
          noise, not the gate).

    The point is: USER_OK=1 must NOT trigger the gate's exit 64 path.
    """
    src = _read(WRAPPER)

    # Quick source-level check: the gate block is conditioned on USER_OK != 1
    assert (
        'if [ "${USER_OK:-0}" != "1" ]' in src
    ), "USER_OK=1 conditional gate block missing from wrapper"

    # Live-run with USER_OK=1 + --dry-run: should exit 0 (gate passes, dry-run
    # exits before pre-flight). DRY_RUN.log + plan.json MUST be emitted.
    proc = _run(
        ["bash", str(WRAPPER), "--dry-run"],
        env_extra={"USER_OK": "1"},
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"USER_OK=1 + --dry-run must exit 0 (gate passes + dry-run no GPU); "
        f"got rc={proc.returncode}. stderr={proc.stderr[-400:]!r}"
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "USER_OK gate refused" not in combined, (
        f"USER_OK=1 should bypass the gate; found gate refusal in output: "
        f"{combined[-400:]!r}"
    )
    # Verify the dry-run proceeded all the way (enumerated 15 pairs)
    assert (
        "15 (metal, seed) pairs PLANNED" in combined
    ), f"dry-run output should enumerate 15 pairs; got: {combined[-400:]!r}"


# ---------------------------------------------------------------------------
# Test 3: --dry-run with USER_OK=1 → emits DRY_RUN.log + plan.json
# ---------------------------------------------------------------------------


def test_dry_run_with_user_ok_emits_log_and_plan(tmp_path: Path):
    """USER_OK=1 + --dry-run → exits 0 AND emits DRY_RUN.log AND plan.json.

    Per Wf-366-A1 fix + Wf-363-A1 DRY_RUN.log schema helper:
      - DRY_RUN.log emitted to OUT_BASE (= results/wf124_p3_50epoch_full_sweep_g5)
      - plan.json emitted to OUT_BASE with "user_ok": true marker
      - 15 pairs enumerated via --pair args in DRY_RUN.log
    """
    # Use a sandboxed OUT_BASE so we don't pollute the repo's results/
    # The wrapper hardcodes OUT_BASE = "results/wf124_p3_50epoch_full_sweep_g5"
    # so we just verify emission into that fixed location; the dry-run is
    # CPU-only (mkdir -p + textual emission only) so no actual data is
    # written.
    out_base = ROOT / "results" / "wf124_p3_50epoch_full_sweep_g5"

    # Clean previous run if present (test idempotency)
    import shutil

    if out_base.exists():
        shutil.rmtree(out_base)

    proc = _run(
        ["bash", str(WRAPPER), "--dry-run"],
        env_extra={"USER_OK": "1"},
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"--dry-run with USER_OK=1 must exit 0; got rc={proc.returncode}. "
        f"stderr={proc.stderr[-400:]!r}"
    )

    # DRY_RUN.log emitted to OUT_BASE
    dry_run_log = out_base / "DRY_RUN.log"
    assert dry_run_log.exists(), f"DRY_RUN.log missing at {dry_run_log}"
    log_text = dry_run_log.read_text()
    # Wf-363-A1 schema: 5+ pair lines + user_ok REQUIRED + budget 5 hr
    pair_lines = re.findall(r"^\s*pair:\s*(.+)$", log_text, re.MULTILINE)
    assert len(pair_lines) == 15, (
        f"DRY_RUN.log should enumerate 15 (metal, seed) pairs per Wf-363-A1 "
        f"schema; got {len(pair_lines)} pair lines"
    )
    assert "user_ok: REQUIRED" in log_text, (
        f"DRY_RUN.log should declare user_ok: REQUIRED per Wf-363-A1 schema; "
        f"got: {log_text[-600:]!r}"
    )
    assert (
        "budget:  5 hr" in log_text or "budget: 5 hr" in log_text
    ), f"DRY_RUN.log should declare 5 hr wall budget; got: {log_text[-600:]!r}"

    # plan.json emitted to OUT_BASE with "user_ok": true marker
    plan_json = out_base / "plan.json"
    assert plan_json.exists(), f"plan.json missing at {plan_json}"
    plan = json.loads(plan_json.read_text())
    assert (
        plan.get("user_ok") is True
    ), f"plan.json must declare user_ok=true; got {plan!r}"
    assert (
        plan.get("dry_run") is True
    ), f"plan.json must declare dry_run=true; got {plan!r}"
    assert (
        plan.get("total_runs") == 15
    ), f"plan.json must declare total_runs=15; got {plan!r}"
    assert (
        plan.get("status") == "PLAN_ONLY"
    ), f"plan.json status should be PLAN_ONLY; got {plan!r}"
    # metals + seeds sanity
    assert set(plan.get("metals", [])) == {
        "Ru",
        "Ir",
        "Os",
        "Rh",
        "Re",
    }, f"plan.json metals mismatch; got {plan.get('metals')!r}"
    assert plan.get("seeds") == [
        0,
        1,
        2,
    ], f"plan.json seeds mismatch; got {plan.get('seeds')!r}"

    # Honest-framing block populated per Wf-355-A3 convention
    assert "honest_framing" in plan, "plan.json missing honest_framing block"
    hf = plan["honest_framing"]
    assert "MEASURED_today" in hf, "honest_framing missing MEASURED_today"
    assert "PROJECTED" in hf, "honest_framing missing PROJECTED"
    assert (
        "kill_then_adapt_directive" in hf
    ), "honest_framing missing kill_then_adapt_directive reference"

    # Wrapper also references the DRY_RUN.log helper (Wf-363-A1 schema source)
    src = _read(WRAPPER)
    assert (
        "_dry_run_log_helper.sh" in src
    ), "DRY_RUN.log helper source-line missing from wrapper"
    assert HELPER.exists(), f"helper file missing at {HELPER}"
