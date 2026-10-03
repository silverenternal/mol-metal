"""Tests for wf_reproduce_R17.sh — the single-command R17 reproducibility driver.

TODO IMPROVEMENT_BACKLOG extension item G1 (2026-09-25): the bash script
``molmetal/scripts/wf_reproduce_R17.sh`` runs 8 verification sections in
sequence and emits a final ``ALL DONE`` marker.  This test:

  1. Verifies the bash script exists, is executable, and is parseable.
  2. Invokes the script via ``subprocess.run`` with a generous timeout
     (1800 s = 30 min wall budget per the script's own docstring).
  3. Asserts the subprocess exit code is 0.
  4. Asserts ``ALL DONE`` appears in stdout.

The test is intentionally slow-friendly — it is marked as the
``reproduce_R17`` pytest marker so callers can opt-out via
``-m "not reproduce_R17"`` for fast inner-loop runs.

Usage::

    cd /home/hugo/codes/mol-metal
    pytest molmetal/tests/test_reproduce_R17.py -v

Or skip on a quick CI lane::

    pytest -m "not reproduce_R17"
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_reproduce_R17.sh"


# ---------------------------------------------------------------------------
# Lightweight static checks (run on every CI; sub-second).
# ---------------------------------------------------------------------------
def test_script_exists():
    """The reproducibility driver must be on disk at the canonical path."""
    assert SCRIPT.is_file(), f"missing script at {SCRIPT}"


def test_script_is_executable():
    """The script must be marked executable so callers can ``bash`` it
    without first chmod-ing."""
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"{SCRIPT} is not executable (mode={oct(mode)}); run " f"`chmod +x {SCRIPT}`"
    )


def test_script_has_all_done_marker():
    """The script must emit ``ALL DONE`` as its final stdout marker."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ALL DONE" in text, (
        "wf_reproduce_R17.sh must emit an `ALL DONE` marker so callers "
        "can grep for it from a dispatcher."
    )


def test_script_has_eight_sections():
    """The script must define 8 numbered sections (per the G1 spec)."""
    import re

    text = SCRIPT.read_text(encoding="utf-8")
    # Sections are announced via ``echo "[wf_reproduce_R17] SECTION N — …"``.
    section_nums = {int(m.group(1)) for m in re.finditer(r"SECTION\s+(\d+)\b", text)}
    assert section_nums == {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    }, f"expected SECTION markers {{1..8}} in the script, found {sorted(section_nums)}"


# ---------------------------------------------------------------------------
# Heavy subprocess invocation — marked so callers can opt out.
# ---------------------------------------------------------------------------
def _have_uv() -> bool:
    """We rely on ``uv run --no-sync`` inside the script; skip if no uv."""
    return shutil.which("uv") is not None


@pytest.mark.reproduce_R17
@pytest.mark.skipif(
    not _have_uv(), reason="uv not on PATH; cannot drive wf_reproduce_R17.sh"
)
@pytest.mark.skipif(
    not os.environ.get("MOLMETAL_RUN_REPRODUCE"),
    reason=(
        "set MOLMETAL_RUN_REPRODUCE=1 to actually invoke the full 25-min "
        "reproducibility driver; otherwise this test only checks the "
        "static invariants above."
    ),
)
def test_script_runs_and_emits_all_done():
    """Invoke wf_reproduce_R17.sh end-to-end and verify the ALL DONE marker.

    Wall budget: ~25 min on CPU per the script docstring; we set a 1800 s
    timeout so pytest surfaces a clear failure rather than hanging.  Set
    the ``MOLMETAL_RUN_REPRODUCE=1`` env var to enable this check (it is
    skipped by default so the inner-loop test suite stays fast).
    """
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"wf_reproduce_R17.sh exited with {proc.returncode}; "
        f"stderr (tail 2000 chars):\n{proc.stderr[-2000:]}"
    )
    assert "ALL DONE" in proc.stdout, (
        f"expected `ALL DONE` in stdout; last 2000 chars:\n" f"{proc.stdout[-2000:]}"
    )
