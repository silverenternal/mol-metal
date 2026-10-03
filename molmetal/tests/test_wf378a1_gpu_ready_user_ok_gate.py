"""Wf-378-A1 — USER_OK hard entry gate closure for pre_flight_gpu_ready.sh.

============================================================================
Owner:       Hugo Lee (autonomous Wf-378-A1 ultracode unit)
Date:        2026-09-30
Parent:      Wf-363-A2 GAP-3 closure (deferred to follow-up per
             Wf-366-C1 §8 + Wf-368-A4 §3 + Wf-368-B1 §7.1).
             Closes Wf-363-A2 HONEST-NEGATIVE that pre_flight_gpu_ready.sh
             LIVE mode lacked a hard USER_OK entry-point gate.

Per AGENTS.md §5.1 (Standing GPU authorization 2026-09-29 user grant) +
kill_then_adapt directive: any > 1 hr GPU work MUST be gated by USER_OK=1
in env, NOT silently auto-triggered.

This file delivers 5 NEW tests:

  - test_wf378a1_syntax_check
        bash -n syntax check on the modified script (preserves parse-clean).
  - test_wf378a1_hard_gate_source_present
        Source-inspects for the Wf-378-A1 USER_OK hard entry gate pattern:
        DRY_RUN-ne-1 check + USER_OK=1 check + exit 64 + kill_then_adapt ref.
  - test_wf378a1_user_ok_unset_live_exits_64
        Live-runs the script without --dry-run and without USER_OK → expects
        exit 64 (matches Wf-294B / Wf-366-A1 family convention).
  - test_wf378a1_dry_run_skips_gate
        Live-runs the script with --dry-run and without USER_OK → expects
        exit 0 (Gate 5 emits SKIP; hard entry gate is bypassed by --dry-run).
  - test_wf378a1_user_ok_one_live_proceeds_past_hard_gate
        Spawns the script with USER_OK=1, waits for the first 3 log lines
        (~1-3 s of script startup), then kills the process. Verifies the
        hard gate did NOT exit 64 and the script reached the log-emit block.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- USER_OK=1 + LIVE mode test uses subprocess.Popen + readlines + terminate
  pattern because the full pipeline (pre_flight_check.py + wf343c2 +
  wf343c3) takes ~3 min wall. We only need to verify the script reaches
  the log-emit block (proving the hard gate passed), then terminate it.
- The hard gate fires ONLY when not --dry-run (preserves existing semantics
  where --dry-run is a CPU-only verification path with no USER_OK
  requirement).
- Exit code 64 follows the Wf-294B family convention (mirrors
  scripts/launch_wf124_p3_g5.sh:60-67 Wf-366-A1 fix).

Cross-references
----------------
- `scripts/pre_flight_gpu_ready.sh`     — Wf-378-A1 modified script
- `scripts/launch_wf124_p3_g5.sh:60-67` — Wf-294B / Wf-366-A1 reference
                                          USER_OK gate pattern
- `TODO/completed/378a1_gpu_ready_user_ok_gate.md` — Wf-378-A1 ship doc
- `TODO/completed/368b1_user_ok_final.md`          — Wf-368-B1 §7.1
                                                    identifies GAP-3
- `TODO/completed/363a2_user_ok_ux.md`             — Wf-363-A2 audit
                                                    that surfaced GAP-3
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = PROJECT_ROOT / "scripts" / "pre_flight_gpu_ready.sh"


# ---------------------------------------------------------------------------
# Test 1: bash -n syntax check on the modified script
# ---------------------------------------------------------------------------


def test_wf378a1_syntax_check():
    """Wf-378-A1 task contract — modified pre_flight_gpu_ready.sh parses clean.

    Verifies:
      - The wrapper file exists.
      - `bash -n` produces rc=0 (no syntax errors).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"bash -n syntax check failed; rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Wf-378-A1 USER_OK hard entry gate source pattern is present
# ---------------------------------------------------------------------------


def test_wf378a1_hard_gate_source_present():
    """Wf-378-A1 task contract — the hard entry gate is wired into the script.

    Verifies the source contains:
      - DRY_RUN-ne-1 conditional
      - USER_OK=1 check inside that conditional
      - exit 64 (Wf-294B family convention)
      - kill_then_adapt directive reference
      - FATAL / "USER_OK gate refused" message pattern
      - Comment block referencing Wf-378-A1
      - Updated exit-code documentation (64 added)

    All checks are source-only (no subprocess invocation) and run in < 1 s.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"
    text = WRAPPER.read_text()

    # 1. DRY_RUN-ne-1 conditional present
    assert re.search(
        r"DRY_RUN\}\s*-ne\s*1", text
    ), "DRY_RUN-ne-1 conditional missing; hard entry gate is not wired"

    # 2. USER_OK=1 check inside that conditional (with default 0 fallback)
    assert re.search(r'USER_OK:-0[^=]*!=\s*"1"', text), (
        "USER_OK=1 check (with :0 default fallback) missing; "
        "hard entry gate may misfire on unset USER_OK"
    )

    # 3. exit 64 (Wf-294B family convention)
    assert re.search(
        r"exit\s+64", text
    ), "exit 64 missing; hard entry gate does not follow Wf-294B convention"

    # 4. kill_then_adapt directive reference
    assert "kill_then_adapt" in text, (
        "kill_then_adapt directive reference missing; "
        "operator cannot tell why the gate exists"
    )

    # 5. FATAL / USER_OK gate refused message
    assert "USER_OK gate refused" in text, (
        "FATAL / 'USER_OK gate refused' message missing; "
        "operator cannot tell why the launch was refused"
    )

    # 6. Wf-378-A1 cross-reference in the comment block
    assert "Wf-378-A1" in text, (
        "Wf-378-A1 cross-reference missing; "
        "future maintainers cannot trace this gate back to its origin"
    )

    # 7. Updated exit-code documentation (64 added to the comment block)
    assert re.search(
        r"64\s*=\s*USER_OK\s+hard\s+gate\s+refused", text
    ), "exit-code documentation does not list 64 for USER_OK hard gate"

    # 8. Updated Gate 5 comment: PASS/SKIP-only after Wf-378-A1
    assert re.search(
        r"Gate\s+5.*USER_OK.*PASS/SKIP-only",
        text,
        re.DOTALL,
    ) or re.search(
        r"Wf-378-A1\s+hard\s+entry\s+gate\s+pre-filtered",
        text,
    ), "Gate 5 not updated to PASS/SKIP-only after Wf-378-A1 hard entry gate"


# ---------------------------------------------------------------------------
# Test 3: USER_OK unset + LIVE mode → exit 64 (Wf-294B family convention)
# ---------------------------------------------------------------------------


def test_wf378a1_user_ok_unset_live_exits_64():
    """Wf-378-A1 task contract — LIVE mode without USER_OK=1 → exit 64.

    Verifies:
      - `USER_OK="" bash pre_flight_gpu_ready.sh` (no --dry-run) exits 64.
      - The error message references USER_OK so the operator knows what to do.
      - The error message references kill_then_adapt so the operator knows
        why the gate exists.

    This is the same exit-code behaviour as scripts/launch_wf124_p3_g5.sh:66
    (Wf-294B / Wf-366-A1 family).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    # Explicitly unset USER_OK (defends against host env leaking USER_OK).
    env = {k: v for k, v in os.environ.items() if k != "USER_OK"}
    env["USER_OK"] = ""  # present but empty

    proc = subprocess.run(
        ["bash", str(WRAPPER)],  # no --dry-run → LIVE mode
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    rc = proc.returncode
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    assert rc == 64, (
        f"USER_OK unset + LIVE mode should exit 64 (Wf-294B convention); "
        f"got rc={rc} stdout={(proc.stdout or '')[-300:]!r} "
        f"stderr={(proc.stderr or '')[-300:]!r}"
    )
    assert "USER_OK" in combined, (
        "USER_OK gate failure message missing USER_OK reference; "
        "operator cannot recover"
    )
    assert "kill_then_adapt" in combined or "USER_OK gate refused" in combined, (
        "USER_OK gate failure message missing kill_then_adapt reference or "
        "USER_OK gate refused signature; cannot trace gate origin"
    )
    assert "--dry-run" in combined, (
        "USER_OK gate failure message missing --dry-run bypass hint; "
        "operator cannot find the verification path"
    )


# ---------------------------------------------------------------------------
# Test 4: --dry-run skips the hard gate (USER_OK not required for CPU verify)
# ---------------------------------------------------------------------------


def test_wf378a1_dry_run_skips_gate():
    """Wf-378-A1 task contract — --dry-run bypasses the hard gate.

    Verifies:
      - `USER_OK="" bash pre_flight_gpu_ready.sh --dry-run` exits 0
        (not 64 — hard gate is bypassed by --dry-run).
      - The output includes a [SKIP] verdict line for USER_OK_gate.
      - The output includes the Wf-378-A1 cross-reference in the SKIP
        evidence (operator can trace the bypass to this gate).

    Note: full --dry-run wall time is ~3 min because the script invokes
    pre_flight_check.py + wf343c2 + wf343c3. We set timeout=300s.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    # Explicitly unset USER_OK (defends against host env leaking USER_OK).
    env = {k: v for k, v in os.environ.items() if k != "USER_OK"}
    env["USER_OK"] = ""

    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    rc = proc.returncode
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    assert rc == 0, (
        f"--dry-run should bypass hard gate and exit 0; "
        f"got rc={rc} stdout_tail={(proc.stdout or '')[-500:]!r} "
        f"stderr_tail={(proc.stderr or '')[-500:]!r}"
    )

    # Gate 5 emits [SKIP] USER_OK_gate
    assert "[SKIP] USER_OK_gate" in combined, (
        "--dry-run must show [SKIP] USER_OK_gate; "
        f"got combined_tail={combined[-500:]!r}"
    )

    # SKIP evidence references Wf-378-A1
    assert (
        "Wf-378-A1" in combined
    ), "SKIP gate evidence should reference Wf-378-A1 for traceability"

    # Hard gate must NOT have fired (no "USER_OK gate refused" in output)
    assert (
        "USER_OK gate refused" not in combined
    ), "Hard gate fired on --dry-run; bypass semantics broken"


# ---------------------------------------------------------------------------
# Test 5: USER_OK=1 + LIVE mode bypasses hard gate (script proceeds past it)
# ---------------------------------------------------------------------------


def test_wf378a1_user_ok_one_live_proceeds_past_hard_gate():
    """Wf-378-A1 task contract — USER_OK=1 + LIVE mode → hard gate passes.

    Verifies:
      - `USER_OK=1 bash pre_flight_gpu_ready.sh` exits != 64 (hard gate
        fires; subsequent gates may fail per Wf-358-D1 §0 honest-negative
        baseline but the hard gate MUST pass).
      - First log line shows DRY_RUN=0 (proves the script reached the
        log-emit block AFTER the hard gate).
      - No "USER_OK gate refused" message (hard gate did NOT fire).

    We use subprocess.Popen + select + terminate pattern because the full
    pipeline (pre_flight_check.py + wf343c2 + wf343c3) takes ~3 min wall.
    We only need to verify the script reaches the log-emit block (proving
    the hard gate passed), then terminate it.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    env = {**os.environ, "USER_OK": "1"}

    # Spawn the script; collect up to ~30 log lines or 30 s, whichever first.
    proc = subprocess.Popen(
        ["bash", str(WRAPPER)],  # no --dry-run → LIVE mode
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        bufsize=1,
    )

    # Read with timeout via select. We give 30 s to capture startup log lines.
    deadline = time.monotonic() + 30.0
    captured = []
    try:
        while time.monotonic() < deadline:
            # Wait up to 1 s for new output
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break  # EOF
                captured.append(line)
                if len(captured) >= 30:
                    break  # enough lines for our checks
            if proc.poll() is not None:
                # Process exited; drain remaining output
                rest = proc.stdout.read()
                if rest:
                    captured.append(rest)
                break
    finally:
        # Terminate the script if it's still running — full pipeline can
        # take ~3 min; we only need the first few log lines to verify the
        # hard gate passed.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    combined = "".join(captured)

    # Hard gate must NOT have fired
    assert "USER_OK gate refused" not in combined, (
        "USER_OK=1 + LIVE mode: hard gate fired (should have passed); "
        f"combined={combined[-500:]!r}"
    )

    # Script must have reached the log-emit block (after hard gate)
    # The first log line includes the Wf-350-B1 run header
    assert "Wf-350-B1 pre_flight_gpu_ready.sh" in combined, (
        "USER_OK=1 + LIVE mode: script did not reach the log-emit block; "
        f"combined={combined[:500]!r}"
    )

    # DRY_RUN must be 0 (proves we're in LIVE mode, not dry-run)
    assert "DRY_RUN=0" in combined, (
        "USER_OK=1 + LIVE mode: DRY_RUN flag is not 0; " f"combined={combined[:500]!r}"
    )

    # rc may be 0, 1, or other (depending on cascading pre-flight gates),
    # but it MUST NOT be 64 (which would mean hard gate fired despite
    # USER_OK=1).
    final_rc = proc.returncode
    assert final_rc != 64, (
        f"USER_OK=1 + LIVE mode: hard gate unexpectedly exited 64; " f"rc={final_rc}"
    )
