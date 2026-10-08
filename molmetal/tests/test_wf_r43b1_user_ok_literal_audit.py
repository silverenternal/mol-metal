#!/usr/bin/env python
"""Wf-R43-B1 — regression test for `--user-ok "REQUIRED"` literal bug class.

================================================================
Wf-R43-B1 — scan all launch_wf*.sh wrappers and ensure the
`--user-ok "REQUIRED"` literal is NEVER an active trainer invocation.

Background (post-mortem 2026-10-08, per chemprop_v3 P1.4 wrapper audit)
----------------------------------------------------------------------
The pattern `--user-ok "REQUIRED"` was historically appended to TRAINER
invocations (`${PY} ${SCRIPT} --user-ok "REQUIRED" ...`) as a marker that
the wrapper requires USER_OK=1 to launch. This is a BUG:

  * The 4 canonical trainers (`train_metal_lomo_wasserstein.py`,
    `train_dmpnn_multitask_wasserstein.py`, `train_cfm_yuelbond_wasserstein.py`,
    `train_chemprop_v2_v3.py`) do NOT accept `--user-ok` as a CLI flag.
  * Passing the literal causes `argparse.exit(rc=2)` immediately, BEFORE
    training starts. With pipefail disabled (to avoid DataLoader SIGPIPE
    on `${PY} trainer | tee log`), the trainer's rc=2 was MASKED by `tee`
    returning rc=0 — the wrapper falsely reported success.
  * Fix pattern: USER_OK is enforced via env-var check in the wrapper
    (`[ "${USER_OK:-0}" != "1" ] && exit 64`), NOT via a CLI flag passed
    to the trainer. The `--user-ok` flag IS legitimately accepted by
    `scripts/_dry_run_log_helper.sh::emit_dry_run_log` (Wf-363-A1 DRY-RUN
    emission schema field), but NEVER by any trainer.

What this test guards
---------------------
For every `scripts/launch_*.sh` wrapper:

  1. The string `--user-ok "REQUIRED"` may appear ONLY in:
       a) Comment lines (lines starting with `#`).
       b) As an argument to `emit_dry_run_log` (DRY-RUN emission helper).
  2. The string MUST NOT appear as an active flag passed to a trainer
     invocation (`${PY} ${SCRIPT} ...`, `exec ... python ... ${SCRIPT} ...`,
     etc.).
  3. None of the 4 canonical trainers accept `--user-ok` as an argparse
     flag (so any active occurrence in a wrapper is structurally a bug).

Waves fixed (2026-10-08 audit): launch_wf119_p1_4_chemprop_v3_lomo.sh +
launch_wf299_cfm_main.sh + launch_wf310e_wf219_production_gpu.sh +
launch_wf310e_wf135_phase2_gpu.sh + launch_wf310f_frontier_math_gpu.sh +
launch_wf310b_g6_final.sh + launch_wf388_a1_metal_coord_retrain.sh.

Per AGENTS.md §5.2 (honest framing, MEASURED/PROJECTED) + §5.5
(filesystem-truth) + §5.7 (trainer-flag wire-in smoke). No GPU launches;
all checks are CPU + filesystem only.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# -----------------------------------------------------------------------------
# Project root + constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
HELPER = SCRIPTS_DIR / "_dry_run_log_helper.sh"

# 4 canonical trainers that MUST NOT accept --user-ok
TRAINER_PATHS = [
    "train_metal_lomo_wasserstein.py",
    "train_dmpnn_multitask_wasserstein.py",
    "train_cfm_yuelbond_wasserstein.py",
    "train_chemprop_v2_v3.py",
]

USER_OK_LITERAL = re.compile(r'--user-ok\s+"REQUIRED"')


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _is_comment_line(line: str) -> bool:
    """A line is a comment if its first non-whitespace char is `#`."""
    return line.lstrip().startswith("#")


def _find_active_user_ok_required_violations(wrapper: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, line) where `--user-ok "REQUIRED"` is active.

    "Active" means:
      * the line itself is NOT a comment, AND
      * the line is NOT inside an `emit_dry_run_log` invocation.

    A line is "inside an emit_dry_run_log invocation" if anywhere in the
    preceding 12 lines (a reasonable proxy for a multi-line function call)
    a line contains the bare token `emit_dry_run_log`.
    """
    violations: list[tuple[int, str]] = []
    text = wrapper.read_text(encoding="utf-8")
    lines = text.splitlines()
    emit_dry_window_start = 0  # earliest line in current emit_dry_run_log window
    in_emit_dry = False

    for idx, line in enumerate(lines, start=1):
        # Track entry into / out of an emit_dry_run_log multi-line call.
        # We treat emit_dry_run_log as spanning 14 lines after each invocation
        # start (covers typical 12-15 line multi-flag call).
        if "emit_dry_run_log" in line and not _is_comment_line(line):
            in_emit_dry = True
            emit_dry_window_start = idx
            continue
        if in_emit_dry and idx - emit_dry_window_start > 14:
            in_emit_dry = False

        if not USER_OK_LITERAL.search(line):
            continue
        if _is_comment_line(line):
            continue
        if in_emit_dry:
            continue

        violations.append((idx, line))

    return violations


# -----------------------------------------------------------------------------
# Test 1 — all launch wrappers: no active --user-ok "REQUIRED" in trainer calls
# -----------------------------------------------------------------------------
def test_no_active_user_ok_required_in_launch_wrappers() -> None:
    """Regression: every launch_wf*.sh must not pass --user-ok "REQUIRED" to a trainer."""
    wrappers = sorted(SCRIPTS_DIR.glob("launch_*.sh"))
    assert wrappers, f"No launch_*.sh found under {SCRIPTS_DIR}"

    all_violations: dict[str, list[tuple[int, str]]] = {}
    for wrapper in wrappers:
        bad = _find_active_user_ok_required_violations(wrapper)
        if bad:
            all_violations[str(wrapper.relative_to(PROJECT_ROOT))] = bad

    assert not all_violations, (
        "Found active `--user-ok \"REQUIRED\"` literal in launch wrappers —\n"
        "this flag is NOT accepted by any trainer and triggers argparse rc=2.\n"
        "USER_OK gating is enforced by the wrapper's env-var check, NOT a CLI flag.\n"
        "Violations:\n"
        + "\n".join(
            f"  {path}:{ln}: {line.strip()}"
            for path, items in all_violations.items()
            for ln, line in items
        )
    )


# -----------------------------------------------------------------------------
# Test 2 — sample subset: 4 fixed wrappers have zero violations (MEASURED post-fix)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wrapper_name",
    [
        "launch_wf119_p1_4_chemprop_v3_lomo.sh",
        "launch_wf299_cfm_main.sh",
        "launch_wf310e_wf219_production_gpu.sh",
        "launch_wf310e_wf135_phase2_gpu.sh",
        "launch_wf310f_frontier_math_gpu.sh",
        "launch_wf310b_g6_final.sh",
        "launch_wf388_a1_metal_coord_retrain.sh",
    ],
)
def test_fixed_wrappers_no_violations(wrapper_name: str) -> None:
    """MEASURED post-fix: 7 wrappers fixed on 2026-10-08 must have 0 violations.

    Note: comment-only mentions (e.g. the bug-fix inline comment) are
    intentionally preserved so the historical post-mortem is visible.
    """
    wrapper = SCRIPTS_DIR / wrapper_name
    assert wrapper.exists(), f"{wrapper_name} not on disk (expected post-fix)"
    violations = _find_active_user_ok_required_violations(wrapper)
    assert not violations, (
        f"{wrapper_name} still has active --user-ok \"REQUIRED\" after the 2026-10-08 fix:\n"
        + "\n".join(f"  line {ln}: {line.strip()}" for ln, line in violations)
    )


# -----------------------------------------------------------------------------
# Test 3 — emit_dry_run_log helper: --user-ok IS a legitimate argument
# -----------------------------------------------------------------------------
def test_emit_dry_run_log_helper_accepts_user_ok() -> None:
    """The DRY-RUN emission helper accepts `--user-ok` as a schema field
    (per Wf-363-A1). This is the SAFE context for `--user-ok "REQUIRED"`
    to appear in a wrapper — it is a dry-run record marker, not a CLI flag
    passed to a trainer.
    """
    assert HELPER.exists(), f"helper not on disk: {HELPER}"
    text = HELPER.read_text(encoding="utf-8")
    assert "--user-ok" in text, (
        f"expected `_dry_run_log_helper.sh` to define a --user-ok arg parser "
        f"branch (per Wf-363-A1 schema); not found in {HELPER}"
    )


# -----------------------------------------------------------------------------
# Test 4 — 4 canonical trainers do NOT accept --user-ok as a CLI flag
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("trainer_basename", TRAINER_PATHS)
def test_trainers_do_not_accept_user_ok_flag(trainer_basename: str) -> None:
    """The 4 canonical trainers have no `--user-ok` argparse dest.

    Verification: each trainer's CLI rejects `--user-ok` via argparse
    (exit rc=2), proving it is NOT a trainer-level option. Therefore any
    `--user-ok "REQUIRED"` literal passed to a trainer invocation is
    structurally a bug.
    """
    trainer = PROJECT_ROOT / "molmetal" / "scripts" / trainer_basename
    assert trainer.exists(), f"{trainer_basename} not on disk"
    # Try to invoke trainer with --user-ok; argparse should reject with rc=2.
    # We use --epochs 1 --device cpu to skip heavy imports; the parser check
    # is what we want (argparse runs before any heavy work).
    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(trainer),
            "--user-ok",
            "REQUIRED",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # argparse rejects unknown flags with exit code 2.
    assert result.returncode == 2, (
        f"{trainer_basename} accepts --user-ok (rc={result.returncode}) — "
        f"this would un-guards the regression. Expected rc=2 from argparse.\n"
        f"stderr={result.stderr[-500:]}"
    )