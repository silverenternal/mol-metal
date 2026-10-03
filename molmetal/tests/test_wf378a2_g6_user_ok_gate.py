"""Wf-378-A2 — USER_OK gate parity closure for launch_wf124_p3_g6.sh.

============================================================================
Owner:       Hugo Lee (autonomous Wf-378-A2 ultracode unit)
Date:        2026-09-30
Parent:      Wf-378 ultracode residual wave (post-Wf-368-B1 §7.1
             forward-looking item)
             + AGENTS.md §5.1 standing GPU authorization + kill_then_adapt.

Per Wf-365-B2 final integration test + Wf-368-A4 section 3 + Wf-368-B1
section 7.1: legacy `launch_wf124_p3_g6.sh` predates the Wf-300C USER_OK
gate pattern. While G6 had a USER_OK gate block at original lines
208-218, it was wired AFTER the --dry-run exit block at lines 170-206,
so `--dry-run` silently bypassed the gate (HONEST-NEGATIVE per Wf-363-A2
audit).

This test suite verifies the Wf-378-A2 fix closes the gap by moving the
USER_OK gate BEFORE the --dry-run block per the Wf-366-A2 lesson
(recover_g6_failed_tuples.sh GAP-2 fix), using the canonical Wf-366-A1
pattern (exit 64 + USER_OK gate refused FATAL message), and adding
DRY_RUN.log + plan.json emission in the --dry-run block:

  1. test_user_ok_gate_present_before_dry_run
        Source-inspects the wrapper and verifies:
        - USER_OK gate block uses canonical Wf-366-A1 pattern
        - USER_OK gate line < --dry-run exit line (CORE INVARIANT)
        - USER_OK gate uses `exit 64` (Wf-294B family convention)
        - DRY_RUN.log helper is sourced
        - DRY_RUN.log emission + plan.json emission present
  2. test_no_user_ok_exits_64 (parametrized 3 ways)
        - [no_USER_OK_no_dry_run]: gate fires, rc=64
        - [no_USER_OK_with_dry_run]: gate fires, rc=64 (was rc=0 BEFORE fix)
        - [USER_OK_1_with_dry_run]: gate passes, rc=0 + DRY_RUN.log +
          plan.json emitted
  3. test_dry_run_with_user_ok_emits_log_and_plan
        USER_OK=1 + --dry-run → exits 0 AND emits DRY_RUN.log AND
        plan.json in OUT_BASE.

CPU-only tests — no torch.cuda, no actual training, no GPU execution.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf124_p3_g6.sh"
HELPER = ROOT / "scripts" / "_dry_run_log_helper.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"file missing: {path}"
    return path.read_text()


def _run(
    cmd: list[str],
    env_extra: dict | None = None,
    timeout: int = 60,
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
# Test 1: source-level invariants — USER_OK gate BEFORE --dry-run block
# ---------------------------------------------------------------------------


def test_user_ok_gate_present_before_dry_run() -> None:
    """USER_OK gate must fire BEFORE --dry-run exit block (Wf-366-A2 lesson).

    Per Wf-378-A2 fix: the G6 wrapper's USER_OK gate was originally
    wired AFTER the --dry-run exit block (at lines 208-218), so
    `--dry-run` silently bypassed the gate (HONEST-NEGATIVE per
    Wf-363-A2 audit + Wf-365-B2 + Wf-368-B1 §7.1).

    The fix moves the gate to the top of the script (right after
    `cd /home/hugo/codes/mol-metal`) per the canonical Wf-366-A1 pattern
    used in launch_wf124_p3_g5.sh.

    Source-level invariants verified here:
      1. Canonical Wf-366-A1 gate pattern present.
      2. USER_OK gate block line < --dry-run exit block line.
      3. Gate uses `exit 64` (Wf-294B family convention).
      4. DRY_RUN.log helper is sourced.
      5. DRY_RUN.log emission (emit_dry_run_log) present in --dry-run block.
      6. plan.json emission present in --dry-run block.
      7. Reference to Wf-366-A2 / Wf-378-A2 in source comments.
    """
    src = _read(WRAPPER)

    # 1. Canonical Wf-366-A1 gate pattern present
    assert "USER_OK:-0" in src, "USER_OK env-var check missing from wrapper"
    assert "exit 64" in src, "USER_OK gate refusal exit code 64 missing"
    assert (
        "USER_OK gate refused" in src
    ), "USER_OK gate FATAL message missing from wrapper"
    assert "kill_then_adapt" in src, "kill_then_adapt directive reference missing"

    # 2. CORE INVARIANT: USER_OK gate line < --dry-run exit line
    gate_marker = 'if [ "${USER_OK:-0}" != "1" ]; then'
    dry_run_marker = 'if [[ "${DRY_RUN}" == "1" ]]; then'
    assert (
        gate_marker in src
    ), f"USER_OK gate block marker not found; expected: {gate_marker!r}"
    assert (
        dry_run_marker in src
    ), f"--dry-run exit block marker not found; expected: {dry_run_marker!r}"
    gate_idx = src.find(gate_marker)
    dry_run_idx = src.find(dry_run_marker)
    assert gate_idx < dry_run_idx, (
        f"USER_OK gate (line ~{src[:gate_idx].count(chr(10))}) must precede "
        f"--dry-run block (line ~{src[:dry_run_idx].count(chr(10))}); "
        f"otherwise --dry-run bypasses the gate (the Wf-378-A2 fix)"
    )

    # 3. Gate uses exit 64 (Wf-294B family convention)
    # Find the gate block end — locate "fi" on its own line that closes
    # the gate's `if ... then` block (line-starts-with `fi` + optional whitespace)
    gate_end_idx = None
    for cand_idx in re.finditer(r"^fi\s*$", src[gate_idx:], re.MULTILINE):
        gate_end_idx = gate_idx + cand_idx.end()
        break
    assert (
        gate_end_idx is not None
    ), "Could not locate closing 'fi' of USER_OK gate block"
    gate_block = src[gate_idx:gate_end_idx]
    assert (
        "exit 64" in gate_block
    ), "USER_OK gate must use 'exit 64' per Wf-294B family convention"

    # 4. DRY_RUN.log helper is sourced
    assert (
        "_dry_run_log_helper.sh" in src
    ), "DRY_RUN.log helper source-line missing from wrapper"
    assert HELPER.exists(), f"helper file missing at {HELPER}"

    # 5. DRY_RUN.log emission (emit_dry_run_log) present in --dry-run block
    dry_run_end_idx = None
    for cand_idx in re.finditer(r"^fi\s*$", src[dry_run_idx:], re.MULTILINE):
        dry_run_end_idx = dry_run_idx + cand_idx.end()
        break
    assert (
        dry_run_end_idx is not None
    ), "Could not locate closing 'fi' of --dry-run block"
    dry_run_block = src[dry_run_idx:dry_run_end_idx]
    assert (
        "emit_dry_run_log" in dry_run_block
    ), "emit_dry_run_log call missing from --dry-run block"

    # 6. plan.json emission present in --dry-run block
    assert (
        "PLAN_JSON" in src or "plan.json" in src
    ), "plan.json emission block missing from wrapper"
    assert '"user_ok": true' in src, "plan.json must declare 'user_ok: true' marker"

    # 7. Reference to Wf-366-A2 / Wf-378-A2 in source comments
    assert (
        "Wf-378-A2" in src or "Wf-366-A2" in src
    ), "Wf-378-A2 / Wf-366-A2 reference comment missing from wrapper"


# ---------------------------------------------------------------------------
# Test 2: live-run behaviour — parametrized 3 ways
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, env_extra, args, expected_rc, expect_gate_message",
    [
        (
            "no_USER_OK_no_dry_run",
            None,
            [],
            64,
            True,
        ),
        (
            "no_USER_OK_with_dry_run",
            None,
            ["--dry-run"],
            64,  # was rc=0 BEFORE Wf-378-A2 fix (gate was AFTER --dry-run block)
            True,
        ),
        (
            "USER_OK_1_with_dry_run",
            {"USER_OK": "1"},
            ["--dry-run"],
            0,  # gate passes, dry-run emits DRY_RUN.log + plan.json
            False,
        ),
    ],
)
def test_no_user_ok_exits_64(
    label: str,
    env_extra: dict | None,
    args: list[str],
    expected_rc: int,
    expect_gate_message: bool,
) -> None:
    """Live-run gate enforcement — no USER_OK → exit 64; USER_OK=1 → exit 0.

    Per Wf-378-A2 fix: the gate fires FIRST regardless of --dry-run
    (Wf-366-A2 lesson). The 3 parametrized cases cover:
      - [no_USER_OK_no_dry_run]: real-launch path with no USER_OK → rc=64
      - [no_USER_OK_with_dry_run]: --dry-run with no USER_OK → rc=64
        (was rc=0 BEFORE Wf-378-A2 fix; the CORE gap closure)
      - [USER_OK_1_with_dry_run]: --dry-run with USER_OK=1 → rc=0 +
        DRY_RUN.log + plan.json emitted
    """
    proc = _run(
        ["bash", str(WRAPPER)] + args,
        env_extra=env_extra,
        timeout=60,
    )
    assert proc.returncode == expected_rc, (
        f"{label}: expected rc={expected_rc}; got rc={proc.returncode}. "
        f"stderr={proc.stderr[-400:]!r}"
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if expect_gate_message:
        assert "USER_OK gate refused" in combined, (
            f"{label}: expected 'USER_OK gate refused' FATAL message; "
            f"got combined={combined[-400:]!r}"
        )
        assert "USER_OK=1" in combined, (
            f"{label}: expected 'USER_OK=1' remediation hint; "
            f"got combined={combined[-400:]!r}"
        )
    else:
        assert "USER_OK gate refused" not in combined, (
            f"{label}: USER_OK=1 should bypass gate; found gate refusal: "
            f"{combined[-400:]!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: --dry-run with USER_OK=1 emits DRY_RUN.log + plan.json
# ---------------------------------------------------------------------------


def test_dry_run_with_user_ok_emits_log_and_plan() -> None:
    """USER_OK=1 + --dry-run → exits 0 AND emits DRY_RUN.log AND plan.json.

    Per Wf-378-A2 fix + Wf-363-A1 DRY_RUN.log schema helper:
      - DRY_RUN.log emitted to OUT_BASE = results/wf124_p3_per_innovation_ablation_g6
      - plan.json emitted to OUT_BASE with "user_ok": true marker
      - 60 (cfg, metal, seed) tuples enumerated via --pair args in DRY_RUN.log
      - plan.json carries total_tuples=60 + status=PLAN_ONLY + honest_framing
    """
    out_base = ROOT / "results" / "wf124_p3_per_innovation_ablation_g6"

    # Clean previous run if present (test idempotency)
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
    # Wf-363-A1 schema: 60 pair lines + user_ok REQUIRED + budget 7.5 hr
    pair_lines = re.findall(r"^\s*pair:\s*(.+)$", log_text, re.MULTILINE)
    assert len(pair_lines) == 60, (
        f"DRY_RUN.log should enumerate 60 (cfg, metal, seed) tuples per "
        f"Wf-363-A1 schema; got {len(pair_lines)} pair lines"
    )
    assert "user_ok: REQUIRED" in log_text, (
        f"DRY_RUN.log should declare user_ok: REQUIRED per Wf-363-A1 schema; "
        f"got: {log_text[-600:]!r}"
    )
    assert "budget:  7.5 hr" in log_text or "budget: 7.5 hr" in log_text, (
        f"DRY_RUN.log should declare 7.5 hr wall budget per TODO-182 §3 + "
        f"Wf-260 G6 timing; got: {log_text[-600:]!r}"
    )

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
        plan.get("total_tuples") == 60
    ), f"plan.json must declare total_tuples=60; got {plan!r}"
    assert (
        plan.get("status") == "PLAN_ONLY"
    ), f"plan.json status should be PLAN_ONLY; got {plan!r}"
    # configs + metals + seeds sanity
    assert set(plan.get("configs", [])) == {
        "no_multi_task",
        "no_wass",
        "no_kendall",
        "no_dann",
    }, f"plan.json configs mismatch; got {plan.get('configs')!r}"
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
    assert "Wf-378-A2" in hf.get(
        "wf378_a2_status", ""
    ), "honest_framing wf378_a2_status should reference Wf-378-A2"
