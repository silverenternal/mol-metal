"""Wf-A269 repair 2 — guard against the "die-before-it-is-defined" class.

MEASURED DEFECT (2026-11-02, main thread, on disk): three launch wrappers
called the shell helper ``die()`` BEFORE the helper was defined (and one
wrapper never defined it at all):

    scripts/launch_wf394a1_lambda_cfm_patch.sh    die() CALLED @55/57, DEFINED @113
    scripts/launch_wf409a3_metal_coord_retrain.sh  die() CALLED @82/84, DEFINED @119
    scripts/launch_wf380_scaffold_loss.sh          die() CALLED @65/67, NEVER DEFINED

Bash executes a script line-by-line, so a function definition that appears
LATER in the file is NOT yet in scope at an earlier call site.  The call
therefore resolved to "die: command not found" and returned 127.

MEASURED consequence (bash probe, 2026-11-02, this session) — the brief's
stated consequence ("the || chain then takes the SUCCESS branch") is NOT what
bash does: with ``set -euo pipefail`` the undefined ``die`` is the FINAL
command of the ``||`` list, so `errexit` DOES fire and the wrapper aborts.
What is actually lost is (a) the FATAL message (replaced by "command not
found") and (b) the exit code (127 instead of the documented 1).  The
pre-flight gate still aborts; it just aborts illegibly.  Probe output:

    $ bash /tmp/dieprobe/probe_faithful.sh ; echo "exit=$?"
    dieprobe/probe_faithful.sh: line 7: die: command not found
    exit=127

REPO-WIDE SCAN (this session, 2026-11-02) — the class was NOT limited to the
three assigned wrappers.  84 *.sh files were scanned under scripts/ (git
ls-files + find; never os.walk).  BEFORE the repair, 7 files called a
die-style helper at a line number EARLIER than its definition and 1 file
called a helper it never defined — 8 defective files in total:

    scripts/launch_wf131_p3_deferred_runs.sh          die def@173 call@116
    scripts/launch_wf146_deferred_runs.sh             die def@133 call@92
    scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh  die def@127 call@93
    scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh  die def@141 call@91
    scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh    die def@166 call@126
    scripts/launch_wf380_scaffold_loss.sh              die NEVER DEFINED, call@65
    scripts/launch_wf394a1_lambda_cfm_patch.sh         die def@113 call@55
    scripts/launch_wf409a3_metal_coord_retrain.sh       die def@119 call@82

(Two of the five extra files — launch_wf310e_r20_p4 and launch_wf310e_r20_p5
— acquired a NEW early `die` call in today's own uncommitted Wf-A269 wave-10
edit, so they are part of the same defect family, not a pre-existing class.)

AFTER the repair: 0 defective files, 84 scanned, 24 files use a die-style
helper and every one of them defines it before its first call.

This module is the GUARD.  It asserts the invariant repo-wide (no baseline,
no exception list — AGENTS.md §6.263 moved this repo off baselines), and it
asserts that the detector itself has teeth by running it against a
deliberately broken throwaway snippet.

--------------------------------------------------------------------------------
CORRECTION (Wf-A269 wave-16, 2026-11-03) — ADDITIVE + DATED.  The two
renderings above are PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6;
this block corrects them.

**⇒ THE "24 files use a die-style helper" NUMBER IS STALE.  The live census
is 22.** MEASURED 2026-11-03 by running this module's OWN detector against
the live tree (`scan_repo_scripts()`):

    [wf_a269_die_before_use] scripts scanned: 84 (*.sh under .../scripts);
    files with a code-level die-style helper: 22
    [wf_a269_die_before_use] defective files: 0 (defective helper instances: 0)

The count fell from 24 to 22 because two wrappers dropped their die-style
helper during the Wf-A269 waves; the "0 defective files" claim still holds
and is still enforced by the tests below.

**⇒ AND "repo-wide" IS NOT TRUE.  THE GUARD SCANS `scripts/` ONLY.**
MEASURED 2026-11-03: `git ls-files '*.sh'` ⇒ **113** tracked shell scripts
repo-wide; `git ls-files 'scripts/*.sh'` ⇒ **84**.  The guard's own
`iter_script_paths()` (documented in its docstring as "tracked (git
ls-files) + untracked (find)") therefore covers **84 of 113** tracked
scripts, and **29 tracked `.sh` files OUTSIDE `scripts/` are never
scanned** — a die-before-defined defect in any of them is invisible to this
guard.

**⇒ ⇒ THE CONSEQUENCE, STATED HONESTLY: THIS GUARD IS A `scripts/`-SCOPED
GUARD WHOSE DOCSTRING CLAIMED A REPO-WIDE ONE.** The invariant it enforces
is real for the directory it scans; the scope it claims is wider than the
scope it has. Widening it is OPEN (the guard's own `SCRIPTS_DIR` is already
overridable via `WFA269_SCRIPTS_DIR`, so a second root is a small change) —
but a docstring that overstates its own coverage is the defect class this
whole wave is about, so the claim is corrected HERE rather than left to be
discovered by the next reader.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The directory the guard scans.  Overridable via ``WFA269_SCRIPTS_DIR`` so
#: the guard's own self-test can point it at a throwaway broken tree in /tmp
#: and prove that the repo-wide assertion really does FAIL.
SCRIPTS_DIR = Path(os.environ.get("WFA269_SCRIPTS_DIR", REPO_ROOT / "scripts"))

#: The die-style helper names this guard covers.
DIE_STYLE_NAMES: Tuple[str, ...] = ("die", "fatal", "fail_hard", "abort")

#: Measured on 2026-11-02 (this session) — see the module docstring.
#: BEFORE the repair: 8 defective files out of 84 scanned.
MEASURED_DEFECTIVE_BEFORE = 8
#: AFTER the repair: 0 defective files out of 84 scanned.
MEASURED_DEFECTIVE_AFTER = 0

# `name() {` / `name () {` / `function name {` / `function name() {`
_DEF_RE = re.compile(
    r"^\s*(?:function\s+)?(" + "|".join(DIE_STYLE_NAMES) + r")\s*(?:\(\s*\))?\s*\{"
)
# `name()` / `function name` with the opening brace on the NEXT line.
_DEF_NO_BRACE_RE = re.compile(
    r"^\s*(?:function\s+)?(" + "|".join(DIE_STYLE_NAMES) + r")\s*(?:\(\s*\))?\s*$"
)
# A use of the name as a command.  The lookbehind rejects `x-die`/`--die` and
# the lookahead rejects `die()` (a definition), `die_hard` and `die-cast`.
_CALL_RES = {
    name: re.compile(r"(?<![A-Za-z0-9_.\-])" + re.escape(name) + r"(?![A-Za-z0-9_(\-])")
    for name in DIE_STYLE_NAMES
}
# `source foo.sh` / `. foo.sh`
_SOURCE_RE = re.compile(r"^\s*(?:source|\.)\s+[\"']?([^\"'\s;]+\.sh)")


class Finding(NamedTuple):
    """One defect: a die-style helper used before it was defined."""

    path: str
    name: str
    def_line: Optional[int]
    call_line: Optional[int]
    reason: str


def _strip_code_only(line: str) -> str:
    """Blank out quoted strings and trailing comments, keeping only code.

    Prevents `printf 'die '` and `# see die()` from counting as calls.
    """
    out: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote is not None:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(" ")
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(" ")
            i += 1
            continue
        if ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _definition_names(lines: List[str]) -> Dict[str, int]:
    """Map die-style helper name -> line number of its FIRST definition."""
    defs: Dict[str, int] = {}
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DEF_RE.match(line)
        if match is None:
            match = _DEF_NO_BRACE_RE.match(line)
            if match is not None:
                # `name()` with the brace on a later line — confirm the brace.
                nxt = ""
                for probe in lines[idx:]:
                    if probe.strip():
                        nxt = probe.strip()
                        break
                if not nxt.startswith("{"):
                    match = None
        if match is not None:
            defs.setdefault(match.group(1), idx)
    return defs


def _call_lines(lines: List[str]) -> Dict[str, int]:
    """Map die-style helper name -> line number of its FIRST call."""
    calls: Dict[str, int] = {}
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _DEF_RE.match(line):
            continue  # a definition line, not a call site
        code = _strip_code_only(line)
        for name in DIE_STYLE_NAMES:
            if name in calls:
                continue
            if _CALL_RES[name].search(code):
                calls[name] = idx
    return calls


def _sourced_definers(path: Path, text: str) -> set:
    """Names of die-style helpers defined in a helper this script sources."""
    defined: set = set()
    for match in _SOURCE_RE.finditer(text):
        target = match.group(1)
        if target.startswith("/"):
            candidate = Path(target)
        else:
            candidate = path.parent / target
        try:
            if not candidate.is_file():
                continue
            sub_text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in _definition_names(sub_text.splitlines()):
            defined.add(name)
    return defined


def scan_text(text: str, path: Path, check_undefined: bool = True) -> List[Finding]:
    """Return every die-before-defined defect in one script's text."""
    lines = text.splitlines()
    defs = _definition_names(lines)
    calls = _call_lines(lines)
    findings: List[Finding] = []
    for name in DIE_STYLE_NAMES:
        def_line = defs.get(name)
        call_line = calls.get(name)
        if def_line is not None and call_line is not None:
            if call_line <= def_line:
                findings.append(
                    Finding(
                        path=str(path),
                        name=name,
                        def_line=def_line,
                        call_line=call_line,
                        reason="call-before-definition",
                    )
                )
        elif call_line is not None and def_line is None and check_undefined:
            if name not in _sourced_definers(path, text):
                findings.append(
                    Finding(
                        path=str(path),
                        name=name,
                        def_line=None,
                        call_line=call_line,
                        reason="called-but-never-defined",
                    )
                )
    return findings


def iter_script_paths(root: Path = SCRIPTS_DIR) -> List[Path]:
    """Every *.sh under scripts/ — tracked (git ls-files) + untracked (find).

    NEVER os.walk: this repo is ~201 GB and a walk has OOM-killed a session.
    """
    paths: set = set()
    root_res = root.resolve()
    try:
        rel_root = root_res.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_root = None  # outside the repo (self-test override) — find only
    if rel_root is not None:
        tracked = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", f"{rel_root}/*.sh"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in tracked.splitlines():
            candidate = (REPO_ROOT / line.strip()).resolve()
            if candidate.is_file():
                paths.add(candidate)
    untracked = subprocess.run(
        ["find", str(root), "-maxdepth", "3", "-name", "*.sh", "-type", "f"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in untracked.splitlines():
        candidate = Path(line.strip()).resolve()
        if candidate.is_file():
            paths.add(candidate)
    return sorted(paths)


def scan_repo_scripts(root: Path = SCRIPTS_DIR) -> Tuple[List[Finding], str]:
    """Scan every *.sh under scripts/ and return (findings, human report)."""
    paths = iter_script_paths(root)
    findings: List[Finding] = []
    users = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        defs = _definition_names(lines)
        calls = _call_lines(lines)
        if not defs and not calls:
            continue  # no CODE-level use of a die-style helper
        users += 1
        findings.extend(scan_text(text, path))
    report = "\n".join(
        [
            f"[wf_a269_die_before_use] scripts scanned: {len(paths)} "
            f"(*.sh under {root}); files with a code-level die-style helper: {users}",
            f"[wf_a269_die_before_use] defective files: {len({f.path for f in findings})}"
            f" (defective helper instances: {len(findings)})",
        ]
    )
    return findings, report


# --- fixtures ---------------------------------------------------------------

#: A deliberately BROKEN throwaway snippet: `die` is called before it is
#: defined.  The detector MUST flag this — if it does not, the guard is dead.
BROKEN_SNIPPET = """#!/usr/bin/env bash
set -euo pipefail
MISSING="${1:-/nonexistent/helper.sh}"
[ -f "${MISSING}" ] || die "helper missing: ${MISSING}"
die() {
    printf '[FATAL] %s\\n' "$*" >&2
    exit 1
}
"""

#: The SAME snippet with the definition hoisted above the call — must pass.
GOOD_SNIPPET = """#!/usr/bin/env bash
set -euo pipefail
die() {
    printf '[FATAL] %s\\n' "$*" >&2
    exit 1
}
MISSING="${1:-/nonexistent/helper.sh}"
[ -f "${MISSING}" ] || die "helper missing: ${MISSING}"
"""


# --- tests ------------------------------------------------------------------


def test_detector_flags_die_called_before_definition(tmp_path):
    """The detector MUST have teeth: a broken snippet must be flagged."""
    broken = tmp_path / "launch_broken_fixture.sh"
    broken.write_text(BROKEN_SNIPPET, encoding="utf-8")
    findings = scan_text(BROKEN_SNIPPET, broken)
    assert findings, "detector did NOT flag a deliberately broken snippet"
    finding = findings[0]
    assert finding.name == "die"
    assert finding.reason == "call-before-definition"
    assert finding.call_line is not None and finding.def_line is not None
    assert finding.call_line < finding.def_line, (
        f"fixture is malformed: call@{finding.call_line} def@{finding.def_line}"
    )


def test_detector_passes_when_definition_precedes_call(tmp_path):
    """The hoisted form must NOT be flagged (no false positive)."""
    good = tmp_path / "launch_good_fixture.sh"
    good.write_text(GOOD_SNIPPET, encoding="utf-8")
    assert scan_text(GOOD_SNIPPET, good) == []


def test_repo_scan_over_a_broken_directory_is_not_clean(tmp_path):
    """A whole scripts/ tree containing one broken file MUST be reported.

    This is the end-to-end proof that the repo-wide assertion below can
    actually fail: it points the real scanner at a directory that holds a
    deliberately broken script.
    """
    (tmp_path / "launch_ok.sh").write_text(GOOD_SNIPPET, encoding="utf-8")
    (tmp_path / "launch_broken.sh").write_text(BROKEN_SNIPPET, encoding="utf-8")
    findings, report = scan_repo_scripts(root=tmp_path)
    assert findings, f"repo scan did NOT report the broken fixture\n{report}"
    assert any(f.reason == "call-before-definition" for f in findings)
    assert "defective files: 1" in report


def test_no_die_before_use_in_repo_scripts():
    """Repo-wide assertion: no scripts/*.sh may call a die-style helper
    before (or without) defining it.  Zero tolerance — no baseline."""
    findings, report = scan_repo_scripts()
    print(report)
    for finding in findings:
        print(
            f"[wf_a269_die_before_use] DEFECT {finding.path}: {finding.name} "
            f"call@{finding.call_line} def@{finding.def_line} ({finding.reason})"
        )
    assert not findings, (
        "die-before-definition defects found:\n"
        + "\n".join(
            f"  {f.path}: {f.name} call@{f.call_line} def@{f.def_line} ({f.reason})"
            for f in findings
        )
        + f"\n{report}"
    )


def test_guard_scans_a_non_trivial_number_of_scripts():
    """The guard's own census must be non-trivial (it must actually scan)."""
    if SCRIPTS_DIR.resolve() != (REPO_ROOT / "scripts").resolve():
        pytest.skip("self-test override of WFA269_SCRIPTS_DIR is active")
    paths = iter_script_paths()
    assert len(paths) >= 50, f"only {len(paths)} scripts scanned — census is broken"
    assert all(p.suffix == ".sh" for p in paths)
