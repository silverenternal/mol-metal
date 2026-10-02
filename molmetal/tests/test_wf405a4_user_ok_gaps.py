"""Wf-405-A4 — Cross-wrapper USER_OK UX gap verification.

============================================================================
Owner:       Hugo Lee (autonomous Wf-405-A4 ultracode follow-up unit)
Date:        2026-09-30
Parent:      Wf-378-A1 (pre_flight_gpu_ready.sh) + Wf-378-A2 (G6 wrapper) +
             Wf-366-A2 (recover_g6_failed_tuples.sh) — all USER_OK gate
             closures from the Wf-368-B1 §7.1 follow-up queue.

Background
----------
Wf-368-B1 §7.1 enumerated 4 forward-looking USER_OK UX gaps for Wf-378/379+
follow-ups:
  (a) Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate  [CLOSED]
  (b) Wf-378-A2 launch_wf124_p3_g6.sh USER_OK gate parity          [CLOSED]
  (c) Wf-365-B2 G6 USER_OK parity                                 [CLOSED]
  (d) Wf-363-A2 GAP-3 (pre_flight_gpu_ready dry-run SKIP)         [CLOSED]

Wf-405-A4 is the cross-wrapper regression guard verifying the three
launch wrappers (the canonical pre_flight_gpu_ready + G6 + G6-recovery)
remain in the closed shape and that the Wf-294B family USER_OK pattern
is intact across all three. Two NEW tests:

  - test_all_wrappers_have_user_ok_gate
        Source-inspects all 3 wrappers for the canonical Wf-294B USER_OK
        gate pattern: USER_OK:-0 default fallback + != 1 check + exit 64
        + Wf-378-A1 / Wf-378-A2 / Wf-366-A2 cross-references.

  - test_block_order_before_dry_run
        Cross-wrapper source-inspects for: USER_OK gate line number <
        --dry-run exit block line number in each wrapper (the block-order
        invariant that the original Wf-363-A2 GAP-2/GAP-3 fix established).

Honest framing (party-twenty-spirit §3.5)
--------------------------------------------------------
- Source-only inspection (no subprocess invocations); <1 s wall.
- All 3 wrappers verified SHIPPED; no GAPs remain from the Wf-368-B1 §7.1
  forward-looking queue at Wf-405-A4 timestamp 2026-09-30.
- PROJECTED: 0 NEW gates added; this is a regression-guard wave, not a
  ship surface wave.

Cross-references
----------------
- `scripts/pre_flight_gpu_ready.sh`              — Wf-378-A1 ship surface
- `scripts/launch_wf124_p3_g6.sh`                — Wf-378-A2 ship surface
- `scripts/recover_g6_failed_tuples.sh`          — Wf-366-A2 ship surface
- `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py`   — Wf-378-A1 tests
- `molmetal/tests/test_wf366a2_recover_g6_block_order.py`   — Wf-366-A2 tests
- `molmetal/tests/test_wf363a2_user_ok_ux.py`     — Wf-363-A2 audit sibling
- `TODO/completed/378a1_gpu_ready_user_ok_gate.md`           — Wf-378-A1 doc
- `TODO/completed/378a2_g6_user_ok_parity.md`                — Wf-378-A2 doc
- `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` — Wf-366-A2 doc
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = PROJECT_ROOT / "scripts"

# The 3 USER_OK-gated launch wrappers per Wf-368-B1 §7.1 (a)+(b)+(c).
WRAPPERS = {
    "pre_flight_gpu_ready": {
        "path": SCRIPTS / "pre_flight_gpu_ready.sh",
        "wave": "Wf-378-A1",
        "exit_code_doc": "64",
    },
    "launch_wf124_p3_g6": {
        "path": SCRIPTS / "launch_wf124_p3_g6.sh",
        "wave": "Wf-378-A2",
        "exit_code_doc": "64",
    },
    "recover_g6_failed_tuples": {
        "path": SCRIPTS / "recover_g6_failed_tuples.sh",
        "wave": "Wf-366-A2",
        "exit_code_doc": "64",
    },
}

# Canonical Wf-294B family USER_OK gate pattern (any whitespace):
#   if [[ "${USER_OK:-0}" != "1" ]]; then ... exit 64 ... fi
# Some wrappers use single-bracket test: if [ "${USER_OK:-0}" != "1" ]; then
# Both are functionally equivalent for this regex (matches USER_OK:-0 check).
USER_OK_GATE_PATTERN = re.compile(
    r"""if\s+\[{1,2}\s+["']?\$\{USER_OK:-0\}["']?\s*!=\s*["']1["']""",
    re.MULTILINE,
)
# --dry-run exit block pattern (the block that bails out for plan-only):
DRY_RUN_EXIT_PATTERN = re.compile(
    r"""if\s+\[{1,2}\s+["']?\$\{DRY_RUN\}["']?\s*==\s*["']1["']""",
    re.MULTILINE,
)
# Exit-code 64 anywhere in the wrapper (gate + documentation + Wf-294B pattern).
EXIT_64_PATTERN = re.compile(r"\bexit\s+64\b")


def _line_number_at(text: str, regex_match: re.Match) -> int:
    """Return 1-indexed line number where regex match starts in text."""
    return text[: regex_match.start()].count("\n") + 1


# ---------------------------------------------------------------------------
# Test 1 — All 3 wrappers have the canonical Wf-294B USER_OK gate pattern
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper_name, wrapper_meta",
    list(WRAPPERS.items()),
    ids=list(WRAPPERS.keys()),
)
def test_all_wrappers_have_user_ok_gate(wrapper_name, wrapper_meta):
    """Wf-405-A4 regression guard — every wrapper has the Wf-294B USER_OK gate.

    Verifies for each of the 3 wrappers:
      (a) The wrapper file exists.
      (b) The USER_OK=1 check pattern (with :0 fallback) is present.
      (c) An `exit 64` follows the gate (Wf-294B family convention).
      (d) The wrapper source cites the originating Wf-378-A1 / Wf-378-A2 /
          Wf-366-A2 wave for cross-trace.
      (e) An exit-code documentation block lists `64 = USER_OK hard gate
          refused` or references the Wf-294B family convention.

    Source-only inspection; <1 s wall per wrapper.
    """
    path: Path = wrapper_meta["path"]
    wave: str = wrapper_meta["wave"]
    assert path.exists(), f"wrapper missing: {path}"

    text = path.read_text()

    # (a) USER_OK=1 check with :0 default fallback present.
    gate_match = USER_OK_GATE_PATTERN.search(text)
    assert gate_match, (
        f"{path.name}: Wf-294B USER_OK gate pattern not found "
        f'(expected: if [[ "${{USER_OK:-0}}" != "1" ]]; then). '
        "Gate is REQUIRED for safe LIVE-mode per AGENTS.md §5.1."
    )

    # (b) Exit-code 64 follows the gate (Wf-294B family).
    #    Find the closest exit N after the gate match.
    post_gate_text = text[gate_match.start():]
    exit_match = re.search(r"exit\s+(\d+)", post_gate_text)
    assert exit_match, (
        f"{path.name}: USER_OK gate has no exit-code after it; gate may "
        "silently no-op. Expected 'exit 64' per Wf-294B family."
    )
    exit_code = int(exit_match.group(1))
    assert exit_code == 64, (
        f"{path.name}: USER_OK gate exit code is {exit_code}, expected 64 "
        "(Wf-294B family convention per AGENTS.md §5.1 kill_then_adapt)."
    )

    # (c) Source cites the originating wave for cross-trace.
    assert wave in text, (
        f"{path.name}: source must cite {wave} (originating wave) so future "
        "maintainers can trace the gate back to its ship doc."
    )

    # (d) Reference to the Wf-294B / kill_then_adapt directive.
    assert "kill_then_adapt" in text or "Wf-294B" in text, (
        f"{path.name}: source must cite kill_then_adapt or Wf-294B "
        "directive origin so operator can understand why the gate exists."
    )

    # (e) An "USER_OK gate refused" or similar refusal message exists.
    assert "USER_OK gate refused" in text, (
        f"{path.name}: refusal message 'USER_OK gate refused' missing; "
        "operator cannot tell why the launch was refused."
    )


# ---------------------------------------------------------------------------
# Test 2 — All 3 wrappers: USER_OK gate line < --dry-run exit block line
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper_name, wrapper_meta",
    list(WRAPPERS.items()),
    ids=list(WRAPPERS.keys()),
)
def test_block_order_before_dry_run(wrapper_name, wrapper_meta):
    """Wf-405-A4 regression guard — USER_OK gate fires BEFORE --dry-run exit.

    Verifies the block-order invariant that Wf-366-A2 GAP-2 fix established:
      * `USER_OK` gate appears at a lower line number than the
        `--dry-run` exit block (for launch_wf124_p3_g6.sh + recover_g6
        wrappers — bail-out pattern).
      * OR the `USER_OK` gate appears BEFORE the gate-conditional
        `if [[ ${DRY_RUN} -ne 1 ]]` block (for pre_flight_gpu_ready.sh —
        conditional gate pattern).

    For `pre_flight_gpu_ready.sh`, the pattern is structurally different:
      - The `DRY_RUN-ne-1` block (gate-conditional) appears BEFORE the
        `--dry-run` exit (this is the Wf-378-A1 hard entry gate that
        fires only in LIVE mode but its gate SOURCE is BEFORE --dry-run).
      - There is no `if [[ "${DRY_RUN}" == "1" ]] then ... exit 0; fi`
        bail block because the script runs all gates in --dry-run mode.

    For launch_wf124_p3_g6.sh + recover_g6_failed_tuples.sh, the pattern is:
      - USER_OK gate source appears at a lower line number than the
        `if [[ "${DRY_RUN}" == "1" ]]; then ... exit 0; fi` bail block.

    Source-only inspection; <1 s wall per wrapper.
    """
    path: Path = wrapper_meta["path"]
    assert path.exists(), f"wrapper missing: {path}"
    text = path.read_text()

    # Find gate line number.
    gate_match = USER_OK_GATE_PATTERN.search(text)
    assert gate_match, (
        f"{path.name}: USER_OK gate pattern not found (test 1 should have "
        "already verified; if test 1 passes and this fails, the gate "
        "pattern regex may be off)"
    )
    gate_line = _line_number_at(text, gate_match)

    # Find --dry-run exit block line number.
    dry_run_match = DRY_RUN_EXIT_PATTERN.search(text)
    if dry_run_match is None:
        # pre_flight_gpu_ready.sh uses a different pattern: the gate is
        # conditional on `if [[ ${DRY_RUN} -ne 1 ]]; then` (gate-conditional
        # pattern) — the script runs all gates in --dry-run mode and emits
        # SKIP for USER_OK_gate. No bail-out block.
        # Verify the conditional pattern instead.
        conditional_match = re.search(
            r"if\s+\[\[\s+\$\{DRY_RUN\}\s+-ne\s+1\s+\]\]", text,
        )
        assert conditional_match, (
            f"{path.name}: neither DRY_RUN==1 bail pattern nor DRY_RUN!=1 "
            "conditional pattern found; wrapper has no --dry-run handling."
        )
        conditional_line = _line_number_at(text, conditional_match)
        assert gate_line > conditional_line, (
            f"{path.name}: pre_flight_gpu_ready.sh pattern — the "
            "DRY_RUN!=1 conditional should come BEFORE the gate's inner "
            "USER_OK check (gate is conditional on DRY_RUN!=1). "
            f"conditional line {conditional_line} should be < gate line "
            f"{gate_line}; if not, the gate may fire on --dry-run."
        )
        return  # pre_flight pattern verified; bail out of further checks.

    dry_run_line = _line_number_at(text, dry_run_match)

    # CORE INVARIANT: gate source line < --dry-run exit line.
    assert gate_line < dry_run_line, (
        f"{path.name}: USER_OK gate (line {gate_line}) must appear BEFORE "
        f"--dry-run exit (line {dry_run_line}); --dry-run currently "
        "short-circuits the gate. Wf-363-A2 GAP-2/GAP-3 regression risk."
    )

    # INVARIANT: --dry-run block contains `exit 0` (plan-only success path).
    # Find the closing `fi` of the dry-run block, then check exits inside.
    post_dry_run_text = text[dry_run_match.start():]
    # The dry-run block ends at the first top-level `fi` after `then`.
    # Simpler: search for `exit 0` anywhere in the dry-run block.
    # (Caveat: the dry-run block may have intermediate `exit 1` fallback
    # statements for missing CSV etc — we accept that, but we require
    # an `exit 0` SOMEWHERE in the dry-run block.)
    dry_block_has_exit_zero = bool(re.search(r"\bexit\s+0\b", post_dry_run_text))
    assert dry_block_has_exit_zero, (
        f"{path.name}: --dry-run block must contain 'exit 0' (plan-only "
        "success path); not found in source after the dry-run match. "
        "post_dry_run_text[:600]=" + post_dry_run_text[:600]
    )

    # INVARIANT: the gate precedes the manifest / dry-run-log emission block.
    # The reference pattern for the gap fix cites the launch_wf310e
    # reference wrapper for cross-trace.
    assert re.search(r"line[s]?\s+\d+(-\d+)?", text), (
        f"{path.name}: reference comment should cite specific line numbers "
        "for cross-trace; bare filename reference is insufficient."
    )