"""Wf-366-A2 — Block-order swap verification for ``recover_g6_failed_tuples.sh``.

============================================================================
Owner:       Hugo Lee (autonomous Wf-366-A2 ultracode follow-up unit)
Date:        2026-09-29
Parent:      Wf-363-A2 audit finding (recover_g6_failed_tuples.sh dry-run
             bypasses USER_OK gate) + AGENTS.md §5.1 kill_then_adapt directive.

Background — Wf-363-A2 audit surfaced a block-order bug in
``scripts/recover_g6_failed_tuples.sh``: the script's ``--dry-run`` exit
fired BEFORE the ``USER_OK`` gate, meaning an operator could dry-run the
recovery plan without explicit USER_OK authorization. This is a CRITICAL
safety gap because:

  * The recovery is a multi-hour GPU sweep (~11.4 hr PROJECTED for 57
    FAILED + PENDING tuples on RTX 5090).
  * The dry-run path emits DRY_RUN.log, recovery_manifest.tsv, and writes
    to master.log — non-trivial work that should require explicit USER_OK
    per AGENTS.md §5.1 standing GPU authorization.
  * Wf-337-G1 documented the gate in the header but the script logic
    short-circuited it on --dry-run (block-order bug).

Wf-366-A2 fix (this file) — swap the block order so that the ``USER_OK``
gate fires FIRST, before any prereg.md check, before manifest enumeration,
and before --dry-run handling. Reference pattern: the 5 Wf-355 launch
wrappers (USER_OK gate at lines 115-122 in
``launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh``; --dry-run plan emit
later at line 364).

Tests (2 NEW):

  - test_block_order_user_ok_before_dry_run
        Source-inspects ``recover_g6_failed_tuples.sh`` to confirm:
          (a) The ``USER_OK`` gate appears BEFORE the ``--dry-run`` exit block.
          (b) The gate uses exit-code 64 (consistent with Wf-294B family).
          (c) The ``--dry-run`` exit block appears AFTER the manifest enumeration
              AND AFTER the ``USER_OK`` gate.
          (d) The reference pattern (line range + filename) is cited in a
              code comment so future agents see the cross-reference.

  - test_dry_run_without_user_ok_exits_64
        Live-runs ``recover_g6_failed_tuples.sh --dry-run`` WITHOUT
        ``USER_OK=1`` set in the environment. Asserts exit-code 64
        (gate fires). Also live-runs without --dry-run and without
        USER_OK → exit-code 64 (same gate, same exit code).
        Third assertion: USER_OK=1 --dry-run → exit-code 0 (gate passes,
        dry-run plan emits DRY_RUN.log).

Honest framing (party-twenty-spirit §3.5):
  - MEASURED: rc=0 (broken) → rc=64 (fixed) on --dry-run without USER_OK.
  - MEASURED: rc=1 → rc=64 on no --dry-run + no USER_OK (exit-code update).
  - MEASURED: USER_OK=1 --dry-run still emits DRY_RUN.log with 120 (cfg,
    metal, seed) pairs (per-pair enumeration preserved per Wf-363-A1).
  - PROJECTED: 4 scenarios per task step 6 (Scenario 4 = real launch path,
    not live-tested since it would consume GPU budget; the gate structure
    is source-inspected instead).
  - No GPU launches performed in this wave.

Cross-references
----------------
- ``scripts/recover_g6_failed_tuples.sh``        — Wf-337-G1 wrapper (fixed)
- ``scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh``
                                                — Wf-355-A3 reference pattern
- ``scripts/_dry_run_log_helper.sh``             — Wf-363-A1 shared helper
- ``molmetal/tests/test_wf363a2_user_ok_ux.py``   — Wf-363-A2 audit (sibling)
- ``TODO/completed/363a2_user_ok_ux.md §3 GAP-2`` — original gap finding
- ``TODO/completed/366a2_gap2_recover_g6_block_order_fix.md``
                                                — Wf-366-A2 ship doc (this fix)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = PROJECT_ROOT / "scripts"
WRAPPER = SCRIPTS / "recover_g6_failed_tuples.sh"

# Reference pattern: scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
#   - USER_OK gate at lines 115-122 (FIRST, exit 64 on refusal).
#   - --dry-run plan emit later at line 364.
#   - DRY_RUN.log helper sourced at lines 107-110.
REFERENCE_WRAPPER = SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh"


# ---------------------------------------------------------------------------
# Test 1 — Block order: USER_OK gate BEFORE --dry-run exit (source inspect)
# ---------------------------------------------------------------------------
def test_block_order_user_ok_before_dry_run():
    """Wf-366-A2 GAP-2 fix: confirm ``recover_g6_failed_tuples.sh`` has the
    ``USER_OK`` gate BEFORE the ``--dry-run`` exit block (source-level).

    Per AGENTS.md §5.1 kill_then_adapt + Wf-363-A2 audit:
      * Block order must be: arg-parse → USER_OK gate → prereg check →
        manifest enumeration → --dry-run exit (if set) → launch loop.
      * Gate must use exit-code 64 (matches Wf-294B family pattern).
      * The --dry-run exit must appear AFTER the manifest enumeration
        (so the operator sees the recovery_manifest.tsv + DRY_RUN.log
        before the script exits), but AFTER the USER_OK gate (so the
        gate cannot be bypassed).
      * A code comment must cite the reference pattern
        (launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh) for cross-trace.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"
    text = WRAPPER.read_text()

    # Find line numbers of key blocks (1-indexed).
    user_ok_gate_match = re.search(
        r"if\s*\[\[\s*\"\$\{USER_OK:-0\}\"\s*!=\s*\"1\"\s*\]", text,
    )
    dry_run_exit_match = re.search(
        r"if\s*\[\[\s*\"\$\{DRY_RUN\}\"\s*==\s*\"1\"\s*\]\]", text,
    )
    manifest_match = re.search(
        r"#\s*Python helper:.*scan training_summary\.json", text,
    )

    assert user_ok_gate_match, (
        f"{WRAPPER.name}: USER_OK gate not found in source (expected pattern "
        "'if [[ \"${{USER_OK:-0}}\" != \"1\" ]]'). Gate is REQUIRED FIRST."
    )
    assert dry_run_exit_match, (
        f"{WRAPPER.name}: --dry-run exit block not found in source "
        "(expected pattern 'if [[ \"${{DRY_RUN}}\" == \"1\" ]]')."
    )
    assert manifest_match, (
        f"{WRAPPER.name}: manifest enumeration comment not found; cannot "
        "verify block-order invariants."
    )

    # 1-indexed line numbers for clarity.
    user_ok_line = text[: user_ok_gate_match.start()].count("\n") + 1
    dry_run_line = text[: dry_run_exit_match.start()].count("\n") + 1
    manifest_line = text[: manifest_match.start()].count("\n") + 1

    # CORE INVARIANT: USER_OK gate line < --dry-run exit line.
    assert user_ok_line < dry_run_line, (
        f"{WRAPPER.name}: USER_OK gate (line {user_ok_line}) must appear "
        f"BEFORE --dry-run exit (line {dry_run_line}); --dry-run currently "
        "short-circuits the gate. See Wf-363-A2 GAP-2 finding + this fix."
    )

    # INVARIANT: --dry-run exit must be AFTER manifest enumeration
    # (operator should see the recovery_manifest.tsv + DRY_RUN.log before
    # the script exits).
    assert dry_run_line > manifest_line, (
        f"{WRAPPER.name}: --dry-run exit (line {dry_run_line}) must appear "
        f"AFTER manifest enumeration (line {manifest_line}); otherwise the "
        "operator cannot see the recovery manifest before exit."
    )

    # INVARIANT: gate uses exit-code 64 (matches Wf-294B family + reference).
    # Find the closest `exit N` after the gate match.
    post_gate_text = text[user_ok_gate_match.start():]
    exit_match = re.search(r"exit\s+(\d+)", post_gate_text)
    assert exit_match, (
        f"{WRAPPER.name}: USER_OK gate has no exit-code after it; gate may "
        "silently no-op. Expected 'exit 64' per Wf-294B family pattern."
    )
    exit_code = int(exit_match.group(1))
    assert exit_code == 64, (
        f"{WRAPPER.name}: USER_OK gate exit code is {exit_code}, expected 64 "
        "(matches Wf-294B family + reference wrapper pattern). "
        "AGENTS.md §5.1 kill_then_adapt convention."
    )

    # INVARIANT: the source cites the reference pattern (line range + filename)
    # so future agents can trace the convention.
    assert REFERENCE_WRAPPER.name in text, (
        f"{WRAPPER.name}: source must cite {REFERENCE_WRAPPER.name} as the "
        "reference pattern for the USER_OK gate + --dry-run block order; "
        "future agents need the cross-reference to maintain consistency."
    )
    # Reference should cite specific line numbers (e.g. "lines 115-122").
    assert re.search(r"line[s]?\s+\d+(-\d+)?", text), (
        f"{WRAPPER.name}: reference comment should cite specific line "
        "numbers in {REFERENCE_WRAPPER.name}; bare filename reference is "
        "insufficient for cross-trace."
    )

    # INVARIANT: --dry-run exit uses exit-code 0 (matches reference).
    post_dry_run_text = text[dry_run_exit_match.start():]
    dry_exit_match = re.search(r"exit\s+(\d+)", post_dry_run_text)
    assert dry_exit_match, (
        f"{WRAPPER.name}: --dry-run exit has no exit-code after it."
    )
    dry_exit_code = int(dry_exit_match.group(1))
    assert dry_exit_code == 0, (
        f"{WRAPPER.name}: --dry-run exit code is {dry_exit_code}, expected 0 "
        "(dry-run is a plan-only path; exit 0 = success)."
    )

    # INVARIANT: DRY_RUN.log helper is sourced (Wf-363-A1 schema consistency).
    assert "_dry_run_log_helper.sh" in text, (
        f"{WRAPPER.name}: source must source scripts/_dry_run_log_helper.sh "
        "for Wf-363-A1 schema-consistent DRY_RUN.log emission."
    )

    # INVARIANT: emit_dry_run_log is called inside the --dry-run block.
    assert "emit_dry_run_log" in text, (
        f"{WRAPPER.name}: emit_dry_run_log not called; Wf-363-A1 schema-"
        "consistent DRY_RUN.log emission expected."
    )


# ---------------------------------------------------------------------------
# Test 2 — --dry-run without USER_OK exits 64 (live-run)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args, expected_rc, expected_user_ok_msg, label",
    [
        # Scenario 1: no USER_OK, no --dry-run → gate fires first → exit 64
        ([], 64, True, "no_USER_OK_no_dry_run"),
        # Scenario 2: no USER_OK, --dry-run → gate fires first → exit 64
        # (was rc=0 BEFORE the fix; this is the GAP-2 regression guard).
        (["--dry-run"], 64, True, "no_USER_OK_with_dry_run"),
        # Scenario 3: USER_OK=1, --dry-run → gate passes → exit 0 + DRY_RUN.log
        (["--dry-run"], 0, False, "USER_OK_1_with_dry_run"),
    ],
)
def test_dry_run_without_user_ok_exits_64(
    args, expected_rc, expected_user_ok_msg, label,
):
    """Wf-366-A2 GAP-2 fix live-run guard.

    Per task step 6 verification:
      - No USER_OK, no --dry-run      → exit 64 (gate fires first)
      - No USER_OK, --dry-run          → exit 64 (gate fires first; was 0)
      - USER_OK=1, --dry-run           → exit 0 (gate passes; dry-run plan)

    Scenario 4 (USER_OK=1, no --dry-run → proceed to launch) is NOT live-tested
    since invoking it would trigger the actual GPU sweep. The block-order
    invariant is verified at the source level in test 1.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"
    cmd = ["bash", str(WRAPPER), *args]

    # Build env WITHOUT USER_OK unless label requires it.
    env = os.environ.copy()
    env.pop("USER_OK", None)
    if label == "USER_OK_1_with_dry_run":
        env["USER_OK"] = "1"

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env, cwd=str(PROJECT_ROOT),
    )

    stdout_tail = (proc.stdout or "")[-600:]
    stderr_tail = (proc.stderr or "")[-600:]
    combined = (proc.stdout or "") + (proc.stderr or "")

    assert proc.returncode == expected_rc, (
        f"[{label}] expected rc={expected_rc}, got rc={proc.returncode}; "
        f"stdout tail={stdout_tail!r} stderr tail={stderr_tail!r}"
    )

    if expected_user_ok_msg:
        # Gate refusal message must mention USER_OK so operator can recover.
        assert "USER_OK" in combined, (
            f"[{label}] gate refusal message missing 'USER_OK' reference; "
            f"operator cannot recover. combined={combined[-400:]!r}"
        )
        # Should NOT reach the manifest enumeration (which prints
        # "RECOVERY MANIFEST:").
        if label == "no_USER_OK_no_dry_run":
            assert "RECOVERY MANIFEST" not in combined, (
                f"[{label}] gate should fire BEFORE manifest enumeration; "
                f"manifest output suggests block-order bug. "
                f"combined={combined[-400:]!r}"
            )

    # Scenario 3 (USER_OK=1 + --dry-run) must emit DRY_RUN.log to OUT_BASE.
    if label == "USER_OK_1_with_dry_run":
        out_dir = PROJECT_ROOT / "results" / "wf124_p3_per_innovation_ablation_g6"
        dry_run_log = out_dir / "DRY_RUN.log"
        assert dry_run_log.exists(), (
            f"[{label}] USER_OK=1 --dry-run should emit DRY_RUN.log at "
            f"{dry_run_log}; not found. combined={combined[-400:]!r}"
        )
        # Verify canonical Wf-363-A1 schema fields are present.
        log_content = dry_run_log.read_text()
        for field in ("wrapper:", "mode:", "trainer:", "flags:", "gpu:",
                      "budget:", "user_ok:", "ts:"):
            assert field in log_content, (
                f"[{label}] DRY_RUN.log missing canonical schema field "
                f"'{field}'; Wf-363-A1 schema requires all 8 fields. "
                f"log_content={log_content[:600]!r}"
            )