# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Wf-A269 wave-10 -- the ``grep -c`` double-zero capture guard.

THE DEFECT CLASS
----------------
``grep -c`` PRINTS the match count **and** exits non-zero when the count is
zero.  A shell capture that guards it with ``|| echo 0`` therefore fires the
fallback *in addition to* the count that ``grep -c`` already printed, and the
captured value becomes the two-line string ``"0\\n0"`` -- no longer a number.

MEASURED on 2026-11-02 (CPU-only, GNU grep 3.12 under ``bash``, which is the
interpreter the repaired script runs under):

    $ out="$(grep -acE 'ZZZ_NEVER' file 2>/dev/null || echo 0)"
    $ printf '%s' "$out" | od -c
    0000000   0  \\n   0
    0000003

    $ out="$(grep -acE 'ZZZ_NEVER' file 2>/dev/null || true)"
    $ printf '%s' "$out" | od -c
    0000000   0
    0000001

The seven affected sites were the count captures in
``molmetal/scripts/final_paper_build.sh`` (three pdflatex-warning counts and
four bibtex/aux counts).  Their values reach only the build log
``molmetal/reports/wf_final_paper_build/build_log.txt``, so the pre-repair
impact was a corrupted count in a log string, not a crash.

THE REPAIR
----------
``|| echo 0`` -> ``|| true``.  ``grep -c`` already prints the single ``"0"``
on no-match, so the count stays a number; the fallback now only swallows the
non-zero exit status.

WHAT THIS GATE COVERS
----------------------
* Every count capture of the form ``VAR="$(grep -c ...)"`` /
  ``VAR="$(grep -ac ...)"`` **in the file itself** -- the sites are parsed out
  of the live script, not copied into this test, so a NEW site added later is
  covered automatically.
* The captured value is executed for real, in a scratch directory, against
  stub artefacts: a no-match grep must yield the single-line string ``"0"``.
* A mutation self-test runs the pre-repair idiom and asserts it yields the
  two-line ``"0\\n0"`` -- i.e. this gate BITES on the old code.

CPU-only, no GPU, no network, nothing is written into the repository (the
scratch artefacts live in pytest's ``tmp_path``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "molmetal" / "scripts" / "final_paper_build.sh"

#: A shell capture of the form ``VAR="$(<command>)"``.  The capture must close
#: on the line, so the command group is delimited by the trailing ``)"``.
CAPTURE_RE = re.compile(
    r'^\s*[A-Za-z_][A-Za-z0-9_]*="\$\((?P<cmd>[^\n]*)\)"\s*$')

#: The short-option cluster of the FIRST grep in a command.  The cluster is
#: matched rather than the literal ``-c`` because the script writes the count
#: flag combined with the others (``-acE``).
FIRST_GREP_RE = re.compile(r"grep\s+(?P<flags>-[A-Za-z]+)(?=\s)")

#: A count-flag grep anywhere on a line (``-c``, ``-ac``, ``-acE``).
COUNT_GREP_RE = re.compile(r"grep\s+-[A-Za-z]*c[A-Za-z]*(?=\s)")

#: The stub artefacts the count captures read, created in the scratch cwd.
STUBS = (".pdflatex_pass1.log", "main.blg", "main.aux")

#: The measured pre-repair census of count-capture sites in the script.
EXPECTED_SITES = 7


def _script_text() -> str:
    assert SCRIPT.is_file(), f"the repaired script is missing: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8", errors="replace")


def _is_count_cmd(cmd: str) -> bool:
    """True when the FIRST grep of ``cmd`` carries the count flag."""
    m = FIRST_GREP_RE.search(cmd)
    return bool(m) and "c" in m.group("flags")


def _count_sites(text: str) -> list[str]:
    """Every count-capture command in the script, in file order."""
    out = []
    for line in text.splitlines():
        m = CAPTURE_RE.match(line)
        if m and _is_count_cmd(m.group("cmd")):
            out.append(m.group("cmd"))
    return out


def _code_lines(text: str) -> list[tuple[int, str]]:
    """The script's non-comment lines, as ``(lineno, line)``.

    Full-line comments are dropped so that a NOTE documenting the repair can
    quote the pre-repair idiom without the gate reading it as a live site.
    """
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((i, line))
    return out


def _run_bash(cmd: str, cwd: Path) -> str:
    """Run ``cmd`` under bash in ``cwd`` and return its captured stdout.

    The captured value is wrapped in ``<<< >>>`` so that a value containing a
    newline is unambiguous -- that is precisely the shape this gate cares
    about.  ``i=1`` is exported because the script's own captures interpolate
    ``${i}`` into the pdflatex log file name.
    """
    if shutil.which("bash") is None:  # pragma: no cover
        raise AssertionError("bash is unavailable in this environment")
    script = 'i=1\nOUT="$(' + cmd + ')"\nprintf \'<<<%s>>>\' "$OUT"\n'
    res = subprocess.run(["bash", "-c", script], cwd=str(cwd),
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, (
        f"the captured command failed under bash (rc={res.returncode}): "
        f"{cmd}\nstderr: {res.stderr}")
    return res.stdout


def _make_stubs(cwd: Path, pdflatex_log: str = "") -> None:
    for name in STUBS:
        (cwd / name).write_text(
            pdflatex_log if name == ".pdflatex_pass1.log" else "",
            encoding="utf-8")


def test_script_is_valid_bash():
    """The repair must not have broken the script's syntax."""
    res = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True,
                         text=True, check=False)
    assert res.returncode == 0, (
        f"bash -n rejected the repaired script: {res.stderr}")


def test_no_grep_c_echo_zero_fallback_remains():
    """THE GATE: no count capture may fall back to ``|| echo``."""
    offenders = []
    for lineno, line in _code_lines(_script_text()):
        if COUNT_GREP_RE.search(line) and "|| echo" in line:
            offenders.append(f"{SCRIPT.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "a count capture still uses an `|| echo` fallback -- `grep -c` prints "
        "its own \"0\" and exits 1 on no-match, so the fallback appends a "
        "SECOND \"0\" and the captured value becomes \"0\\n0\":\n  "
        + "\n  ".join(offenders))


def test_count_capture_sites_are_all_classified():
    """Every count capture in the script is a site this gate exercises.

    The sites are parsed out of the live script, so a NEW count capture added
    later is covered without touching this test; the floor pins that the
    seven measured sites are still present.
    """
    sites = _count_sites(_script_text())
    assert len(sites) >= EXPECTED_SITES, (
        f"expected at least {EXPECTED_SITES} count captures in "
        f"{SCRIPT.name}, found {len(sites)}: {sites}")
    assert all("|| echo" not in cmd for cmd in sites), (
        f"a parsed count capture still carries an `|| echo` fallback: {sites}")
    print(f"\nWf-A269 wave-10 grep -c guard: {len(sites)} count capture(s) "
          f"in {SCRIPT.name}")


def test_no_match_grep_yields_a_single_line_zero(tmp_path):
    """THE BEHAVIOURAL GATE: a no-match count is a single-line ``"0"``.

    Every count capture in the script is executed for real against empty stub
    artefacts, so the value the build log would receive is the value measured
    here.
    """
    _make_stubs(tmp_path)
    sites = _count_sites(_script_text())
    assert sites, "no count capture found -- the parser is broken"
    for cmd in sites:
        out = _run_bash(cmd, tmp_path)
        assert out.startswith("<<<") and out.endswith(">>>"), (
            f"the capture did not round-trip through the harness: {out!r}")
        value = out[3:-3]
        assert value == "0", (
            f"a no-match count capture must be the single value \"0\", got "
            f"{value!r} (repr shows the doubled-zero defect) from: {cmd}")
        assert "\n" not in value, (
            f"the captured count is multi-line: {value!r} from: {cmd}")


def test_match_grep_yields_a_single_line_integer(tmp_path):
    """The match path stays numeric too (one line, digits only)."""
    _make_stubs(tmp_path, pdflatex_log="LaTeX Warning: something\n")
    for cmd in _count_sites(_script_text()):
        out = _run_bash(cmd, tmp_path)
        value = out[3:-3]
        assert value.strip().isdigit(), (
            f"the captured count must be a bare integer, got {value!r} "
            f"from: {cmd}")
        assert "\n" not in value, (
            f"the captured count is multi-line: {value!r} from: {cmd}")


def test_pre_repair_fallback_would_have_produced_two_lines(tmp_path):
    """MUTATION SELF-TEST: the pre-repair idiom really was broken.

    Running the OLD form (``|| echo 0``) must reproduce the doubled zero.
    Without this, the gate above could pass for a reason that has nothing to
    do with the defect it exists to catch.
    """
    _make_stubs(tmp_path)
    old = "grep -c 'ZZZ_NEVER_MATCHES_ANYTHING' " + STUBS[1]
    out = _run_bash(old + " || echo 0", tmp_path)
    assert out == "<<<0\n0>>>", (
        "the pre-repair `|| echo 0` fallback did NOT produce the doubled "
        f"zero -- got {out!r}; the guard's premise no longer holds")
    new = _run_bash(old + " || true", tmp_path)
    assert new == "<<<0>>>", (
        f"the repaired `|| true` fallback must yield a single \"0\", got "
        f"{new!r}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    text = _script_text()
    sites = _count_sites(text)
    print(f"script            : {SCRIPT}")
    print(f"count captures    : {len(sites)}")
    for cmd in sites:
        print(f"  {cmd}")
    sys.exit(0)
