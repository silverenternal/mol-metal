"""Wf-363-A3 — Polish pre_flight_gpu_ready.sh dry-run output + USER_OK gate UX.

Owner:        Hugo Lee (autonomous Wf-363-A3 ultracode unit)
Date:         2026-09-29
Parent:       Wf-350-B1 (pre_flight_gpu_ready.sh SHIPPED 389 LOC with 5 gates
              wired). Wf-363-A3 polishes the dry-run output format and
              USER_OK gate UX for consistency with Wf-355 wrapper convention.

These 2 NEW tests verify that the polished
``scripts/pre_flight_gpu_ready.sh`` satisfies the 2 critical contract points
per AGENTS.md §1.2 (Standing GPU auth) + §5.1 (kill_then_adapt directive) +
Wf-294B / Wf-300C USER_OK pattern:

  1. **Dry-run output format**     — Per-gate verdict (PASS/FAIL/SKIP)
                                    formatted + per-gate evidence path
                                    summary block + total + USER_OK status.
  2. **USER_OK gate consistency**  — Same USER_OK=1 convention as Wf-355
                                    wrappers; SKIP on --dry-run, HARD-REFUSED
                                    with rc=64 + "[FATAL] USER_OK gate refused"
                                    without USER_OK=1 on real launch (the
                                    Wf-378-A1 hard entry gate — see the dated
                                    note in test_wf363a3_user_ok_gate_
                                    consistency_wf355 below), PASS gate with
                                    USER_OK=1.

Both tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU launches).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-run
output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = PROJECT_ROOT / "scripts" / "pre_flight_gpu_ready.sh"
VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"

# The wrapper colourises its refusal line, so the literal bytes on stderr are
# "\x1b[0;31m[FATAL]\x1b[0m USER_OK gate refused: ...".  Strip the SGR
# escapes before asserting on the message text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI SGR escape sequences so message text can be asserted on."""
    return _ANSI_RE.sub("", text)

# Expected gate names (5 gates per Wf-350-B1 spec)
EXPECTED_GATES = [
    "pre_flight_all",
    "compute_manifest",
    "wf343c2_per_wave",
    "wf343c3_data_quality",
    "USER_OK_gate",
]


# ---------------------------------------------------------------------------
# Test 1: dry-run output format (per-gate verdict + evidence paths)
# ---------------------------------------------------------------------------


def test_wf363a3_dry_run_output_format():
    """Wf-363-A3 task contract — dry-run output is polished per the spec:
       - Wf-363-A3 dry-run summary header at the top with per-gate
         expectations enumerated.
       - Per-gate verdict line emitted in [HH:MM:SS] name PASS/FAIL/SKIP format.
       - Wf-363-A3 dry-run summary footer at the bottom with per-gate
         verdict + evidence path enumerated.
       - 5 gates covered.
       - USER_OK gate status reported (SKIP for --dry-run).
       - Total PASS/FAIL/SKIP counts emitted.

    Verifies:
      - The wrapper script exists and `bash -n` parses cleanly.
      - dry-run exits with rc=0 (or non-zero only due to non-Wf-350-B1
        cascading issues; the Wf-350-B1 polish summary header/footer
        MUST be emitted regardless of final verdict).
      - All 5 expected gate names appear in the output.
      - Summary header text "Wf-363-A3 dry-run summary" appears.
      - Summary footer text "Wf-363-A3 dry-run summary footer" appears.
      - "per-gate verdict + evidence path" appears in the footer.
      - "USER_OK gate status" appears.
      - "totals: PASS=" summary line appears.
      - Exit code is 0 for dry-run (no blocking FAILs).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    # bash syntax check
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

    # Run --dry-run and capture output
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    out = proc.stdout
    err = proc.stderr
    combined = out + "\n" + err

    # dry-run should exit 0 (no blocking fails; wf343c2 cascading is non-blocking
    # per Wf-350-A2 §5)
    assert proc.returncode == 0, (
        f"dry-run should exit zero; got rc={proc.returncode} " f"stdout={out[-500:]!r}"
    )

    # Summary header at top
    assert "Wf-363-A3 dry-run summary" in combined, (
        "summary header missing in dry-run output; "
        f"first 500 chars={combined[:500]!r}"
    )
    assert (
        "Per-gate verdicts (PASS/FAIL/SKIP)" in combined
    ), "per-gate verdict enumeration missing in summary header"

    # All 5 expected gate names appear in stdout
    for gate_name in EXPECTED_GATES:
        assert gate_name in combined, (
            f"gate name {gate_name!r} missing from dry-run output; "
            f"output_tail={combined[-500:]!r}"
        )

    # Summary footer at bottom
    assert (
        "Wf-363-A3 dry-run summary footer" in combined
    ), "summary footer missing in dry-run output"
    assert (
        "per-gate verdict + evidence path" in combined
    ), "per-gate evidence path block missing in footer"
    assert "totals: PASS=" in combined, "totals summary line missing in footer"
    assert (
        "USER_OK gate status:" in combined
    ), "USER_OK gate status line missing in footer"
    assert "log file:" in combined, "log file path missing in footer"

    # The footer must show USER_OK status as SKIP on --dry-run
    assert (
        "SKIP (--dry-run" in combined
    ), "USER_OK gate status must report SKIP for --dry-run"

    # The footer must show 5 per-gate verdicts with bracket formatting [PASS]
    for gate_name in EXPECTED_GATES:
        assert (
            f"[PASS] {gate_name}" in combined
            or f"[FAIL] {gate_name}" in combined
            or f"[SKIP] {gate_name}" in combined
        ), (
            f"footer verdict line for {gate_name!r} missing; "
            f"footer={combined[-1500:]!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: USER_OK gate UX consistency with Wf-355 wrappers
# ---------------------------------------------------------------------------


def test_wf363a3_user_ok_gate_consistency_wf355():
    """Wf-363-A3 task contract — USER_OK gate UX is consistent with Wf-355
       wrappers (launch_wf300_cfm_main.sh + launch_wf310b_g6_final.sh
       + launch_wf264_cfm_property_panel.sh pattern):

         * USER_OK=1 env var convention (NOT USER_OK=YES as in some wrappers).
         * --dry-run SKIPs USER_OK gate (rc=0 always).
         * Real launch without USER_OK=1 → HARD-REFUSED: rc=64 +
           "[FATAL] USER_OK gate refused", NO per-gate line at all (the
           Wf-378-A1 hard entry gate fires BEFORE any gate runs).
         * Real launch with USER_OK=0 → same hard refusal (rc=64).
         * Real launch with USER_OK=1 → USER_OK gate PASS (verdict PASS line
           visible in output).

    Verifies:
      - Source code references "USER_OK=1" (Wf-355 convention).
      - --dry-run with no USER_OK exits zero with SKIP gate line.
      - Real launch with no USER_OK exits 64 with the "[FATAL] USER_OK gate
        refused" refusal message (Wf-378-A1 hard entry gate).
      - Real launch with USER_OK=1 emits PASS gate line for USER_OK_gate.

    ======================================================================
    ⚠️ SUPERSEDED RENDERING — 2026-09-29 (Wf-363-A3, the ORIGINAL freeze) —
    PRESERVED VERBATIM per AGENTS.md §5.5 rule #6 (a correction is ADDITIVE
    + DATED, never a silent edit, never a deletion).  DO NOT DELETE.
    ======================================================================

    THE SUPERSEDED CONTRACT, AS WRITTEN ON 2026-09-29 (verbatim):

        * Real launch without USER_OK=1 → USER_OK gate FAIL + rc=1.
        * Real launch with USER_OK=0 → USER_OK gate FAIL + rc=1.

      - Real launch with no USER_OK exits non-zero with FAIL gate line.

    ...and the corresponding assertions, verbatim:

        assert proc.returncode != 0, (
            "real launch without USER_OK=1 should exit non-zero (gate enforced); "
            f"got rc={proc.returncode}"
        )
        combined_no = proc.stdout + "\n" + proc.stderr
        assert "[FAIL] USER_OK_gate" in combined_no, (
            f"real launch without USER_OK must show [FAIL] USER_OK_gate; "
            f"got={combined_no[-500:]!r}"
        )

        assert proc.returncode != 0, (
            "real launch with USER_OK=0 should exit non-zero; " f"got rc={proc.returncode}"
        )
        combined_zero = proc.stdout + "\n" + proc.stderr
        assert "[FAIL] USER_OK_gate" in combined_zero, (
            f"real launch with USER_OK=0 must show [FAIL] USER_OK_gate; "
            f"got={combined_zero[-500:]!r}"
        )

    WHY IT WAS SUPERSEDED (MEASURED 2026-11-04, CPU-only, live wrapper run).
    The LIVE WRAPPER is the SSOT (AGENTS.md §5.5 filesystem-truth: report
    SHIPPED != file on disk; BOTH must hold).  `scripts/pre_flight_gpu_ready.sh`
    no longer implements the 2026-09-29 shape: **Wf-378-A1 (2026-09-30) added a
    HARD ENTRY GATE** at `scripts/pre_flight_gpu_ready.sh:144-150`:

        if [[ ${DRY_RUN} -ne 1 ]]; then
            if [ "${USER_OK:-0}" != "1" ]; then
                printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to run LIVE.\n' ... >&2
                ...
                exit 64
            fi
        fi

    That gate fires **BEFORE any per-gate line is emitted**, so the per-gate
    `USER_OK_gate` verdict (which is PASS/SKIP-only in LIVE mode —
    `scripts/pre_flight_gpu_ready.sh:434-438`) can **NEVER** render
    "[FAIL] USER_OK_gate" on a real launch.  The wrapper's own header documents
    the new exit code at `scripts/pre_flight_gpu_ready.sh:77`:
    "64 = USER_OK hard gate refused (Wf-378-A1 NEW; only in LIVE mode without
    USER_OK=1)".  The exit-64 USER_OK shape is the CANONICAL Wf-294B-family
    contract and is already pinned repo-wide by
    `molmetal/tests/test_wf405a4_user_ok_gaps.py` (WRAPPERS["pre_flight_gpu_ready"]
    → wave "Wf-378-A1", exit_code_doc "64"; USER_OK_GATE_PATTERN + EXIT_64_PATTERN).

    MEASURED 2026-11-04 on this host (CPU-only; the wrapper refuses before it
    touches anything, so no GPU was allocated):
        --dry-run, USER_OK unset   → rc=0   ; "[SKIP] USER_OK_gate" present
        real,    USER_OK unset     → rc=64  ; "[FATAL] USER_OK gate refused: set
                                            USER_OK=1 in env to run LIVE."
        real,    USER_OK=0         → rc=64  ; same FATAL message
        real,    USER_OK=1         → rc=1   ; "[PASS] USER_OK_gate" present
                                            (the rc=1 is the wf343c2 cascading
                                            FAIL, NOT the USER_OK gate)

    ⇒ THE LIVE WRAPPER IS THE SSOT; the 2026-09-29 "[FAIL] USER_OK_gate"
    expectation is the STALE rendering.  This test is re-frozen to the live
    contract (rc=64 + the FATAL refusal string), and the dry-run
    "[SKIP] USER_OK_gate" assertion is UNCHANGED.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()
    # Wf-355 convention: USER_OK=1 (not USER_OK=YES)
    assert (
        "USER_OK=1" in text
    ), "USER_OK=1 env var convention (Wf-355) missing in wrapper"
    # USER_OK gate verdict strings
    assert "USER_OK_gate" in text, "USER_OK_gate name missing in wrapper"
    assert (
        "kill_then_adapt standing directive" in text
    ), "kill_then_adapt reference missing in USER_OK gate comment"
    assert (
        "Wf-355 wrapper convention" in text or "Wf-355" in text
    ), "Wf-355 cross-reference missing in wrapper"

    # (a) --dry-run SKIP gate
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert (
        "[SKIP] USER_OK_gate" in combined
    ), f"dry-run must show [SKIP] USER_OK_gate; got={combined[-500:]!r}"

    # (b) Real launch without USER_OK → HARD-REFUSED by the Wf-378-A1 entry
    #     gate (rc=64 + "[FATAL] USER_OK gate refused"), BEFORE any per-gate
    #     line is emitted.  See the dated SUPERSEDED RENDERING note in this
    #     function's docstring for the 2026-09-29 expectation this replaced.
    env_no_user_ok = {k: v for k, v in os.environ.items() if k != "USER_OK"}
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=env_no_user_ok,
    )
    assert proc.returncode == 64, (
        "real launch without USER_OK=1 must be refused by the Wf-378-A1 hard "
        f"entry gate with exit code 64; got rc={proc.returncode}"
    )
    combined_no = _plain(proc.stdout + "\n" + proc.stderr)
    assert "[FATAL] USER_OK gate refused" in combined_no, (
        "real launch without USER_OK must show the Wf-378-A1 refusal message "
        f"'[FATAL] USER_OK gate refused'; got={combined_no[-500:]!r}"
    )
    assert (
        "USER_OK=1" in combined_no
    ), "failure message should reference USER_OK=1 convention"

    # (c) Real launch with USER_OK=0 → the SAME hard refusal (rc=64).
    env_user_ok_zero = {**os.environ, "USER_OK": "0"}
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=env_user_ok_zero,
    )
    assert proc.returncode == 64, (
        "real launch with USER_OK=0 must be refused by the Wf-378-A1 hard "
        f"entry gate with exit code 64; got rc={proc.returncode}"
    )
    combined_zero = _plain(proc.stdout + "\n" + proc.stderr)
    assert "[FATAL] USER_OK gate refused" in combined_zero, (
        "real launch with USER_OK=0 must show the Wf-378-A1 refusal message "
        f"'[FATAL] USER_OK gate refused'; got={combined_zero[-500:]!r}"
    )

    # (d) Real launch with USER_OK=1 → USER_OK gate PASS line emitted.
    # NOTE: Final exit code depends on other gates (wf343c2 cascading FAIL is
    # non-blocking on this host but we just verify the USER_OK gate verdict
    # line itself is PASS, not the overall verdict).
    env_user_ok_one = {**os.environ, "USER_OK": "1"}
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
        env=env_user_ok_one,
    )
    combined_one = proc.stdout + "\n" + proc.stderr
    assert "[PASS] USER_OK_gate" in combined_one, (
        f"real launch with USER_OK=1 must show [PASS] USER_OK_gate; "
        f"got={combined_one[-500:]!r}"
    )
    assert (
        "USER_OK=1 confirmed" in combined_one
    ), "USER_OK=1 confirmation message missing"
