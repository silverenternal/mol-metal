"""Wf-424-A2 — Wf-409-A3 wrapper --metals/--seeds tokenization fix tests.

================================================================
Background
================================================================
Wf-423-A2 audit (2026-09-30) flagged a CRITICAL tokenization bug on the
Wf-409-A3 launch wrapper:

  `scripts/launch_wf409a3_metal_coord_retrain.sh` line 72 (pre-fix):
    `--metals) METALS="$2"; shift 2`
    `--seeds)  SEEDS="$2"; shift 2`

  Captures only 1 argv token per flag. The SCAFFOLD argparse
  (`molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` line 129)
  declares ``nargs="+"`` for `--metals` + `--seeds`, so the SCAFFOLD
  expects MULTIPLE discrete argv tokens. When a user invokes the
  wrapper with the unquoted multi-token form (e.g.
  ``--metals Ru Ir Rh --seeds 0 1 2``), only the FIRST token goes
  into METALS; the next token (``Ir``) becomes the next $1 and the
  wrapper exits with ``[FATAL] unknown flag: Ir``.

Wf-424-A2 closes this gap with a greedy tokenisation pattern
(nargs="+" equivalent) that consumes all tokens up to the next
``--*`` flag. Both the quoted-string form (``--metals "Ru Ir Rh"``)
AND the unquoted multi-token form (``--metals Ru Ir Rh``) now work.

Verification
================================================================
Tests run on Python extraction of the wrapper's CLI parsing logic
(extracted from the wrapper source via regex) — this sidesteps the
PRE-EXISTING tput/TTY issue at line 229 of the wrapper (which is
unrelated to the tokenization fix; documented in closeout §5).

Honest framing
==============
* Wf-424-A2 fix: greedy tokenisation for `--metals` + `--seeds`.
* Sister finding (NOT Wf-424-A2 scope): `tput bold 2>/dev/null || true`
  at line 229 returns escape sequences when stderr is not a TTY; bash
  then interprets the substituted output as a command line. Pre-existing
  bug, documented separately. Closure path = Wf-424-A3 sister fix or
  simpler rewrite using static ANSI escapes.
* 0 GPU launches performed.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).

Cross-references
================
* Wf-423-A2 wrapper flag-surface audit (CRITICAL tokenization bug)
* Wf-409-A3 wrapper (`scripts/launch_wf409a3_metal_coord_retrain.sh`)
* SCAFFOLD argparse contract (`molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py`)
* AGENTS.md §5.5 filesystem-truth rule (every closeout includes ls -la evidence)
* AGENTS.md §5.6 atomic write protocol (used to write this test file)
* AGENTS.md §5.7 trainer flag wire-in smoke-required rule (Wf-399-A1 NEW)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Repo root (mol-metal/) resolved from this test file location.
_TEST_FILE = Path(__file__).resolve()
# molmetal/tests/test_wf424a2_wrapper_tokenization.py -> molmetal/tests/ -> molmetal/ -> mol-metal/
REPO_ROOT = _TEST_FILE.parent.parent.parent
WRAPPER = REPO_ROOT / "scripts" / "launch_wf409a3_metal_coord_retrain.sh"
# SCAFFOLD argparse contract for --metals + --seeds (nargs="+")
SCAFFOLD = (
    REPO_ROOT / "molmetal" / "scripts" / "wf409a3_metal_coord_retrain_scaffold.py"
)


# -----------------------------------------------------------------------------
# Helper: extract the CLI parsing block from the wrapper source via regex.
# Reads lines 67-95 (CLI parsing + USER_OK gate) so we test the actual
# wrapper logic, not a re-implementation. We use a Python sub-shell that
# sources the extracted block to verify tokenization in isolation.
# -----------------------------------------------------------------------------
def _extract_parser_block() -> str:
    """Extract the CLI parsing block from the wrapper.

    Returns a Python-callable bash snippet that mirrors the wrapper's
    while/case logic. The block defines METALS, SEEDS, EPOCHS, OUT_DIR,
    DRY_RUN, SMOKE, ACTUAL from $1, $2, ... argv tokens.

    Uses indentation tracking to handle the nested `while` blocks inside
    `--metals` / `--seeds` greedy loops (each has its own `done`).
    """
    src = WRAPPER.read_text(encoding="utf-8")
    lines = src.splitlines()
    parser_lines: list[str] = []
    in_parser = False
    outer_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        if not in_parser:
            # Detect the OUTER `while [ $# -gt 0 ]; do` (CLI parser loop)
            if re.match(r"while \[ \$# -gt 0 \]; do$", stripped):
                outer_indent = len(line) - len(stripped)
                in_parser = True
                parser_lines.append(line)
            continue
        # We're in the parser block. Check if this line closes it.
        parser_lines.append(line)
        if stripped == "done":
            indent = len(line) - len(stripped)
            if indent == outer_indent:
                # OUTER done — stop capturing
                break
    return "\n".join(parser_lines)


def _run_parser_block(args: list[str]) -> dict[str, str]:
    """Run the extracted parser block on the given argv list, returning
    METALS, SEEDS, EPOCHS, OUT_DIR, DRY_RUN, SMOKE, ACTUAL as dict.

    Skips the tput/DRY_RUN.log emission that lives in the wrapper
    (post-parser) — focused unit test on parser logic only.
    """
    parser_block = _extract_parser_block()
    # Append the echo of parsed values for inspection.
    test_block = (
        parser_block
        + "\necho METALS=$METALS\necho SEEDS=$SEEDS\necho EPOCHS=$EPOCHS\necho OUT_DIR=$OUT_DIR\necho DRY_RUN=$DRY_RUN\necho SMOKE=$SMOKE\necho ACTUAL=$ACTUAL\n"
    )
    argv_str = " ".join(repr(a) for a in args)
    full_cmd = f"{test_block}"
    proc = subprocess.run(
        ["bash", "-c", full_cmd, "bash"] + args,
        capture_output=True,
        text=True,
        timeout=10,
    )
    parsed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            if k in {
                "METALS",
                "SEEDS",
                "EPOCHS",
                "OUT_DIR",
                "DRY_RUN",
                "SMOKE",
                "ACTUAL",
            }:
                parsed[k] = v
    return parsed


# -----------------------------------------------------------------------------
# Test 1: unquoted multi-token form (THE BUG SCENARIO per Wf-423-A2).
# Pre-fix: METALS=Ru, SEEDS=0 (only first token captured), wrapper dies on
# "Ir" as unknown flag. Post-fix: METALS="Ru Ir Rh", SEEDS="0 1 2".
# -----------------------------------------------------------------------------
def test_metals_unquoted_multi_token() -> None:
    """Verify --metals accepts unquoted multi-token form (THE BUG SCENARIO)."""
    parsed = _run_parser_block(["--metals", "Ru", "Ir", "Rh", "--seeds", "0", "1", "2"])
    assert (
        parsed["METALS"] == "Ru Ir Rh"
    ), f"Expected METALS='Ru Ir Rh', got METALS='{parsed.get('METALS', '<missing>')}'"
    assert (
        parsed["SEEDS"] == "0 1 2"
    ), f"Expected SEEDS='0 1 2', got SEEDS='{parsed.get('SEEDS', '<missing>')}'"


# -----------------------------------------------------------------------------
# Test 2: quoted-string form (pre-fix WORKS, post-fix still works — backward compat).
# -----------------------------------------------------------------------------
def test_metals_quoted_string_form() -> None:
    """Verify --metals accepts quoted-string form (backward compat)."""
    parsed = _run_parser_block(["--metals", "Ru Ir Rh", "--seeds", "0 1 2"])
    assert (
        parsed["METALS"] == "Ru Ir Rh"
    ), f"Expected METALS='Ru Ir Rh', got METALS='{parsed.get('METALS', '<missing>')}'"
    assert parsed["SEEDS"] == "0 1 2"


# -----------------------------------------------------------------------------
# Test 3: 5-metal + 5-seed full Path A (per Wf-390-A1 5-metal subset).
# -----------------------------------------------------------------------------
def test_metals_full_path_a_5_metals_5_seeds() -> None:
    """Verify --metals + --seeds accept 5+5 token lists (full Path A)."""
    parsed = _run_parser_block(
        ["--metals", "Ru", "Ir", "Rh", "Os", "Re", "--seeds", "0", "1", "2", "3", "4"]
    )
    assert (
        parsed["METALS"] == "Ru Ir Rh Os Re"
    ), f"Expected METALS='Ru Ir Rh Os Re', got METALS='{parsed.get('METALS', '<missing>')}'"
    assert (
        parsed["SEEDS"] == "0 1 2 3 4"
    ), f"Expected SEEDS='0 1 2 3 4', got SEEDS='{parsed.get('SEEDS', '<missing>')}'"


# -----------------------------------------------------------------------------
# Test 4: separator distinction (--metals stops at next --epochs flag).
# -----------------------------------------------------------------------------
def test_metals_stops_at_next_flag() -> None:
    """Verify --metals stops capturing when next --* flag is seen."""
    parsed = _run_parser_block(["--metals", "Ru", "Ir", "Rh", "--epochs", "50"])
    assert parsed["METALS"] == "Ru Ir Rh"
    assert (
        parsed["EPOCHS"] == "50"
    ), f"Expected EPOCHS='50', got EPOCHS='{parsed.get('EPOCHS', '<missing>')}'"


# -----------------------------------------------------------------------------
# Optional sanity: bash syntax check on the wrapper itself (independent of parser).
# -----------------------------------------------------------------------------
def test_wrapper_bash_syntax_ok() -> None:
    """Verify the wrapper passes bash -n syntax check."""
    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (
        proc.returncode == 0
    ), f"bash -n failed: rc={proc.returncode} stderr={proc.stderr}"


# -----------------------------------------------------------------------------
# Module-level sanity (pytest collection).
# -----------------------------------------------------------------------------
def test_wrapper_file_present() -> None:
    """Sanity check that the wrapper file exists on disk (filesystem-truth)."""
    assert WRAPPER.exists(), f"wrapper missing on disk: {WRAPPER}"
    assert WRAPPER.stat().st_size > 5000, "wrapper file size < 5 KB — likely corrupted"
    # SCAFFOLD contract verification
    assert SCAFFOLD.exists(), f"SCAFFOLD missing on disk: {SCAFFOLD}"
    scaffold_src = SCAFFOLD.read_text(encoding="utf-8")
    assert 'nargs="+"' in scaffold_src, (
        'SCAFFOLD argparse no longer uses nargs="+" for --metals/--seeds — '
        "the wrapper fix contract may be stale; verify SCAFFOLD."
    )
    assert "--metals" in scaffold_src and "--seeds" in scaffold_src
