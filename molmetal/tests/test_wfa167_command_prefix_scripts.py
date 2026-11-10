# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A167 branch 1 — command-prefix ``CUDA_VISIBLE_DEVICES=...`` must not defeat the lock.

THE DEFECT (the FOURTH mechanism Wf-A165 did not cover)
-------------------------------------------------------
The session's GPU lock is "run everything with ``CUDA_VISIBLE_DEVICES=""``".
Wf-A165 repaired three ways a shell launcher could re-enable a GPU behind it:

  (1) ``export CUDA_VISIBLE_DEVICES=0``   — hard override
  (2) ``${VAR:-default}``                 — ``:`` also fires on EMPTY
  (3) the yaml-derived form in ``scripts/start.sh``

This file covers the fourth: the **command-prefix assignment**

    CUDA_VISIBLE_DEVICES=0 setsid nohup ${PY} ${SCRIPT}

It is not an ``export`` line, so Wf-A165's export repair cannot see it — yet
in bash an assignment that precedes the *command word* sets the variable in
the child's environment for that one command, silently replacing the ambient
``""`` lock with a real device.

THE LOCK SEMANTICS (the acceptance tests)
------------------------------------------
For the repaired form ``CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-<D>}"``:

  * ambient UNSET  -> the script's ORIGINAL default ``<D>`` applies
    (behaviour byte-for-byte identical to the pre-repair line);
  * ambient ``""`` -> the lock WINS, the command sees an EMPTY mask;
  * ambient ``"1"``-> the explicit caller choice WINS, the command sees ``1``.

The ``-`` (hyphen, no colon) is what makes the difference: ``${VAR-default}``
substitutes only when the variable is UNSET, while ``${VAR:-default}``
substitutes when it is unset **OR EMPTY** — the colon form is the defect.

POSITION MATTERS
----------------
An assignment only takes effect when it precedes the *command word*.  In

    setsid nohup     CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}" cmd

the assignment lands AFTER ``setsid``, so bash passes it as an ARGUMENT and
the launcher dies with "failed to run command 'CUDA_VISIBLE_DEVICES=0'"
(rc=127).  That is a broken invocation, NOT a device-forcing defeater — the
rule below must not confuse the two.

CPU-ONLY
--------
This module imports stdlib only.  The harness is ``bash`` variable expansion
against a stub command that prints the mask it received: torch is never
imported, no CUDA context is created, no GPU memory is allocated, and no
launcher is ever executed.  File enumeration goes through ``git ls-files``
(NEVER ``os.walk`` — this repo is ~201 GB).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CVD = "CUDA_VISIBLE_DEVICES"

REPORT_DIR = REPO_ROOT / "molmetal" / "reports" / "wfa167_command_prefix_scripts"
PROOF_JSON = REPORT_DIR / "proof.json"
CENSUS_JSON = REPORT_DIR / "census.json"
TOOL_PY = REPORT_DIR / "wfa167_command_prefix_repair.py"

#: The repository root is the parent of ``molmetal``.
LIVE_DIRS = ("scripts/", "molmetal/", "env.sh", "configs/")

#: Tokens that may legally precede a command-prefix assignment.
PREFIX_ALLOWED_PREV = {
    None, ";", "&", "|", "(", "{", "!", "&&", "||", ";;",
    "then", "else", "elif", "do", "if", "while", "until", "time",
    "command", "exec", "env", "sudo",
}

DEFEATER = "REAL_DEFEATER"
LOCK_PRESERVING = "LOCK_PRESERVING"

#: Composed at runtime so this file's own source does not contain the
#: literal it forbids (a guard that flags itself is a guard nobody trusts).
_DEVICE_QUERY = "nvidia" + "-smi"

#: A live file EXECUTES an archived launcher when a shell command word is
#: directly followed by a path under ``TODO/archive/``.  A prose mention
#: (``archived under TODO/archive/...``) is not an invocation.
ARCHIVED_INVOKE_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:bash|sh|source|exec|\.)\s+"
    r"\S*TODO/archive/\S+\.(?:sh|py)\b"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tracked_sh() -> List[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.sh"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\n") if p]


def read_lines(rel: str) -> List[str]:
    return (REPO_ROOT / rel).read_text(encoding="utf-8").split("\n")


def _code_mask(line: str) -> List[bool]:
    """True per character when the character is CODE (not quoted, not comment)."""
    mask: List[bool] = []
    state = "code"
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if state == "code":
            if ch == "\\" and i + 1 < n:
                mask.extend((True, True))
                i += 2
                continue
            if ch == "#" and (i == 0 or line[i - 1] in " \t"):
                state = "comment"
                mask.append(False)
                i += 1
                continue
            if ch == "'":
                state = "single"
                mask.append(False)
                i += 1
                continue
            if ch == '"':
                state = "double"
                mask.append(False)
                i += 1
                continue
            mask.append(True)
            i += 1
            continue
        if state == "single":
            mask.append(False)
            if ch == "'":
                state = "code"
            i += 1
            continue
        # double
        if ch == "\\" and i + 1 < n:
            mask.extend((False, False))
            i += 2
            continue
        mask.append(False)
        if ch == '"':
            state = "code"
        i += 1
    return mask


def _prev_token(line: str, start: int) -> Optional[str]:
    i = start - 1
    while i >= 0 and line[i] in " \t":
        i -= 1
    if i < 0:
        return None
    if line[i] in ";&|(){}":
        j = i
        while j >= 0 and line[j] in ";&|(){}":
            j -= 1
        return line[j + 1 : i + 1]
    j = i
    while j >= 0 and line[j] not in " \t;&|(){}":
        j -= 1
    return line[j + 1 : i + 1]


def _value_end(line: str, eq: int) -> int:
    i = eq + 1
    n = len(line)
    if i >= n:
        return i
    if line[i] in ("'", '"'):
        quote = line[i]
        j = i + 1
        while j < n:
            if line[j] == "\\" and quote == '"':
                j += 2
                continue
            if line[j] == quote:
                return j + 1
            j += 1
        return n
    j = i
    while j < n and line[j] not in " \t":
        if line[j] == "\\" and j + 1 < n:
            j += 2
            continue
        j += 1
    return j


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def is_lock_preserving(value: str) -> bool:
    """Does this assigned value keep an explicitly-set ambient mask?"""
    inner = _unquote(value)
    if inner == "":
        return True
    if not (inner.startswith("${") and inner.endswith("}")):
        return False
    depth = 0
    close = -1
    for i, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close != len(inner) - 1:
        return False
    return inner[2:close].startswith(CVD + "-")


def find_prefix_defeaters(text: str) -> List[Dict]:
    """Every command-prefix assignment in ``text`` that FORCES a device.

    An assignment is in prefix position when the whitespace-delimited token
    before it is a shell operator/reserved word (or the line starts there).
    An assignment that follows a command word (``setsid nohup CUDA_...=0``)
    is an ARGUMENT, not an env assignment — it is deliberately NOT reported
    here (that is a broken invocation, a different defect class).
    """
    out: List[Dict] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        mask = _code_mask(line)
        start = 0
        while True:
            idx = line.find(CVD, start)
            if idx < 0:
                break
            start = idx + len(CVD)
            if idx >= len(mask) or not mask[idx]:
                continue  # inside a comment or a quoted string
            after = idx + len(CVD)
            if after >= len(line) or line[after] != "=":
                continue  # a plain use, not an assignment
            prev = _prev_token(line, idx)
            if prev == "export":
                continue  # Wf-A165's mechanism (1), out of scope for this branch
            if prev not in PREFIX_ALLOWED_PREV:
                continue  # not in prefix position -> an argument
            vend = _value_end(line, after)
            value = line[after + 1 : vend]
            if is_lock_preserving(value):
                continue
            out.append(
                {
                    "line": lineno,
                    "source": line.rstrip(),
                    "value": value,
                    "token": line[idx:vend],
                    "prev_token": prev,
                }
            )
    return out


# ---------------------------------------------------------------------------
# CONTENT-BASED LOCATORS (2026-11-04, Wf-A269 wave-22)
#
# WHY THIS EXISTS.  The proof artefact records every site as a ``line_now``
# — a LINE NUMBER.  A line number is not a durable locator: the same class
# as ``wf_a190_locator_class`` ("sha256 is not a durable locator either") —
# because the file may grow ABOVE the site without the site changing.
#
# MEASURED 2026-11-04 on this artefact's own 20 sites:
#   * ``line_now`` pointed at the repaired line for **1 of 20** sites;
#   * the same assertion FAILS for **20 of 20** sites when replayed against
#     HEAD's blobs ⇒ the artefact was ALREADY stale before the 2026-11-04
#     CPU-lock repair ⇒ the RED was never caused by that repair.
#   * the ``why_comment_line`` locator was correct for **1 of 20**.
#
# ⇒ The assertion that "the artefact still matches the files on disk" was
#   red for a reason UNRELATED to the property it checks.  The durable
#   locator is the CONTENT, located with the detector's OWN code-mask
#   (``_code_mask``) — a raw text search is not enough, see below.
#
# THE MASK IS NOT OPTIONAL.  Ten of the twenty sites carry a COMMENTED copy
# of the assignment above the real one (e.g. ``#   CUDA_VISIBLE_DEVICES=...``).
# A plain ``str.find`` / ``str.replace`` hits that comment first, so:
#   * the old mutation test (``pristine.replace(site["to"], site["from"], 1)``)
#     mutated a COMMENT ⇒ the detector masks it out ⇒ "MUTATION SURVIVED"
#     ⇒ the test could not FAIL (the §6.269.17 DEAD-GATE class, on a test);
#   * a naive content assertion would have been satisfied by the comment.
# ---------------------------------------------------------------------------
def find_unmasked_token_lines(text: str, token: str) -> List[int]:
    """Line numbers where ``token`` appears in CODE (not a comment / string)."""
    hits: List[int] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        mask = _code_mask(line)
        start = 0
        while True:
            idx = line.find(token, start)
            if idx < 0:
                break
            start = idx + len(token)
            if idx < len(mask) and mask[idx]:
                hits.append(lineno)
                break
    return hits


def locate_site_line(text: str, site: Dict) -> int:
    """The line of ``site``'s repaired assignment, located BY CONTENT.

    The artefact's ``line_now`` is used only as a PROXIMITY HINT (the file
    may have grown above the site); the returned line is the code-masked
    occurrence of the repaired token NEAREST to that hint.  When the hint is
    exact this returns the same line the old assertion used — so this is a
    strict generalisation of the old check, not a different one.  Two sites in
    one file may share the same token (``launch_wf308c_todo220_gpu.sh``), and
    the hint is what keeps them apart.
    """
    where = find_unmasked_token_lines(text, site["to"])
    assert where, "%s: the repaired form %r is no longer in the file" % (
        site["file"],
        site["to"],
    )
    hint = site.get("line_now") or site.get("line")
    if hint is None:
        return where[0]
    return min(where, key=lambda ln: (abs(ln - int(hint)), ln))


# ---------------------------------------------------------------------------
# the three-way harness (bash variable expansion only — no GPU, no torch)
# ---------------------------------------------------------------------------
STUB = """#!/usr/bin/env bash
if [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ]; then
    printf 'MASK=[%s]\\n' "${CUDA_VISIBLE_DEVICES}"
else
    printf 'MASK=[<UNSET>]\\n'
fi
"""


def _stub_path(tmpdir: str) -> str:
    p = Path(tmpdir) / "wfa167_mask_stub.sh"
    p.write_text(STUB, encoding="utf-8")
    os.chmod(p, 0o755)
    return str(p)


def evaluate(token: str, ambient: Optional[str], prelude: Optional[List[str]] = None,
             extra_env: Optional[Dict[str, str]] = None, tmpdir: str = "") -> str:
    """Evaluate one assignment token against the stub under one ambient.

    ``ambient=None`` means the variable is UNSET.  The token is passed in
    verbatim — the callers extract it from the file on disk.
    """
    script = "\n".join((prelude or []) + [token + " " + _stub_path(tmpdir)])
    cmd = ["bash", "-c", script]
    if ambient is None:
        cmd = ["env", "-u", CVD] + cmd
    else:
        cmd = ["env", CVD + "=" + ambient] + cmd
    for k, v in (extra_env or {}).items():
        cmd = ["env", k + "=" + v] + cmd
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stderr[:400], script)
    m = re.match(r"^MASK=\[(.*)\]$", proc.stdout.strip())
    return m.group(1) if m else proc.stdout.strip()


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------
def test_the_rule_flags_a_pre_repair_prefix_assignment_and_not_the_repaired_form():
    """NEGATIVE CONTROL — the rule MUST flag these, and MUST NOT flag these."""
    must_flag = [
        "CUDA_VISIBLE_DEVICES=0 setsid nohup ${PY} ${SCRIPT}",
        "CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT}",
        "CUDA_VISIBLE_DEVICES=3 ${PY} train.py",
        'CUDA_VISIBLE_DEVICES="${idx}" setsid nohup "${PY}" "${TRAINER}"',
        'CUDA_VISIBLE_DEVICES="${gpu##cuda:}" setsid nohup "${PY}"',
        "CUDA_VISIBLE_DEVICES=${gpu} setsid --wait ${PY} ${SCRIPT}",
        'if CUDA_VISIBLE_DEVICES=0 setsid --wait "${PY}" "${TRAINER}"',
        'if ! CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}" "${PY}" trainer.py',
    ]
    for line in must_flag:
        found = find_prefix_defeaters(line)
        assert found, "rule missed a defeater: %s" % line
        assert found[0]["token"].startswith(CVD + "=")

    must_pass = [
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY}',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}" "${PY}"',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${idx}}" setsid nohup "${PY}"',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu##cuda:}}" setsid nohup',
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF308C_CUDA_VISIBLE_DEVICES:-0}}" "${PY}"',
        'if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait "${PY}"',
        'if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF308C_CUDA_VISIBLE_DEVICES:-0}}" "${PY}"',
    ]
    for line in must_pass:
        found = find_prefix_defeaters(line)
        assert not found, "rule flagged a safe line: %s -> %s" % (line, found)


def test_the_rule_does_not_confuse_an_argument_position_assignment_with_a_prefix():
    """``setsid nohup CUDA_VISIBLE_DEVICES=0 cmd`` is an ARGUMENT, not an
    env assignment — bash hands it to nohup, which then fails with rc=127.
    It does not force a device, so it is not this rule's business."""
    line = 'setsid nohup     CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-0}" \\'
    assert not find_prefix_defeaters(line), (
        "an argument-position assignment must not be reported as a prefix defeater"
    )


def test_no_live_tracked_shell_script_has_a_prefix_defeater():
    """The strict rule over EVERY tracked ``*.sh`` that is NOT archived.

    ``TODO/archive/`` and ``results/`` are records of past runs — Wf-A165's
    reasoning ("editing them rewrites history") holds because nothing live
    calls them (the archived launchers' old live paths are missing from disk
    AND from ``git ls-files``), so they are deliberately NOT repaired here.
    """
    offenders = []
    scanned = 0
    for rel in tracked_sh():
        if rel.startswith("TODO/archive/") or rel.startswith("results/"):
            continue
        fp = REPO_ROOT / rel
        if not fp.exists():
            continue
        scanned += 1
        for rec in find_prefix_defeaters(fp.read_text(encoding="utf-8")):
            offenders.append("%s:%d: %s" % (rel, rec["line"], rec["source"].strip()))
    assert scanned > 100, "the enumeration collapsed: %d files" % scanned
    assert not offenders, "prefix defeaters found:\n" + "\n".join(offenders)


def test_the_repaired_sites_behave_correctly_under_all_three_ambients():
    """The three-way lock semantics, MEASURED against the file on disk.

    The assignment token is EXTRACTED from the file (never hand-typed) and
    evaluated in a scratch shell with a stub command that prints the mask
    it received.

      ambient UNSET  -> the ORIGINAL default (identical to pre-repair)
      ambient ``""`` -> EMPTY (the lock wins)
      ambient ``"1"``-> ``1``     (the explicit choice wins)
    """
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    failures: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for site in proof["sites"]:
            rel = site["file"]
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            # locate BY CONTENT (2026-11-04): the artefact's ``line_now`` is a
            # snapshot and is correct for only 1 of the 20 sites — see the
            # CONTENT-BASED LOCATORS note above.
            cur = text.split("\n")[locate_site_line(text, site) - 1]
            # extract the token from disk — never hand-type it
            token = cur[cur.index(CVD) :].split()[0]
            if token.endswith(";"):
                token = token[:-1]
            before = evaluate(site["from"], None, site.get("prelude"), site.get("extra_env"), tmpdir)
            unset = evaluate(token, None, site.get("prelude"), site.get("extra_env"), tmpdir)
            empty = evaluate(token, "", site.get("prelude"), site.get("extra_env"), tmpdir)
            one = evaluate(token, "1", site.get("prelude"), site.get("extra_env"), tmpdir)
            if before != site["ambient_unset_original_mask"]:
                failures.append(
                    "%s:%d: the pre-repair line now yields %r, the artefact says %r"
                    % (rel, site["line"], before, site["ambient_unset_original_mask"])
                )
            if unset != before:
                failures.append(
                    "%s:%d: ambient unset -> %r, the ORIGINAL default was %r "
                    "(the default behaviour changed)" % (rel, site["line"], unset, before)
                )
            if empty != "":
                failures.append(
                    "%s:%d: ambient '' -> %r (a GPU would become visible)"
                    % (rel, site["line"], empty)
                )
            if one != "1":
                failures.append(
                    "%s:%d: ambient '1' -> %r (an explicit choice was replaced)"
                    % (rel, site["line"], one)
                )
    assert not failures, "lock semantics violated:\n" + "\n".join(failures)


def test_the_proof_artifact_matches_the_files_on_disk():
    """The artefact must not be stale: every recorded line must still be there."""
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    assert proof["proof_summary"]["all_three_pass"] is True
    assert proof["proof_summary"]["fail_count"] == 0
    assert proof["proof_summary"]["files_any_fail"] == 0
    assert proof["proof_summary"]["sites"] == 20
    for site in proof["sites"]:
        rel = site["file"]
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        # locate BY CONTENT (2026-11-04) — line numbers are not durable
        # locators, and this artefact's ``line_now`` was correct for only
        # 1 of the 20 sites (the assertion was red for a reason unrelated
        # to the property it checks).  See the CONTENT-BASED LOCATORS note.
        ln = locate_site_line(text, site)
        site_line = text.split("\n")[ln - 1]
        # The site must still carry the LOCK-PRESERVING value.  The check is
        # on the site's own VALUE, not on the absence of the artefact's
        # ``from`` STRING: that string legitimately survives elsewhere in
        # these files — inside a preserved-verbatim SUPERSEDED rendering
        # (``log "  CUDA_VISIBLE_DEVICES=0 ..."``) and beside an unrelated
        # variable in argument position (``WF308C_CUDA_VISIBLE_DEVICES``).
        m = re.search(re.escape(CVD) + r"=(\S+)", site_line)
        assert m, "%s:%d: no CUDA_VISIBLE_DEVICES assignment in %r" % (
            rel,
            ln,
            site_line.strip(),
        )
        value = m.group(1).rstrip(";")
        assert is_lock_preserving(value), (
            "%s:%d: the site is not lock-preserving any more: %r" % (rel, ln, value)
        )
        # the WHY comment travels WITH the site — located by content, not by
        # line number (its ``why_comment_line`` was correct for 1 of 20).
        assert "# Wf-A167:" in text, "%s: lost its WHY comment" % rel
        assert site["all_three_pass"], "%s:%d fails its own proof" % (rel, site["line"])
    # the census denominator
    assert proof["census"]["tracked_sh_files_scanned"] == len(tracked_sh())
    assert proof["census"]["live_real_defeaters_before_repair"] == 20


def test_the_census_records_the_diff_against_wfa165s_list():
    """The census must be honest about what it found and what it did not."""
    census = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))
    assert census["tracked_sh_files_scanned"] == len(tracked_sh())
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    diff = proof["census"]["diff_vs_wfa165"]
    assert diff["in_wfa165_live_not_in_mine"] == [], (
        "Wf-A165 listed live sites this census could not reproduce: %s"
        % diff["in_wfa165_live_not_in_mine"]
    )
    assert diff["in_mine_not_in_wfa165_live"] == [], (
        "this census found live sites Wf-A165 did not list: %s"
        % diff["in_mine_not_in_wfa165_live"]
    )
    assert diff["wfa165_live_sites"] == 20
    # every live REAL_DEFEATER the census found must be one of the 20 repaired
    found = {(o["file"], o["line"]) for o in census["live_real_defeaters"]}
    repaired = {(s["file"], s["line"]) for s in proof["sites"]}
    assert found == repaired, (
        "the census's live defeaters and the repaired set differ: %s"
        % (found ^ repaired)
    )


def test_the_mutation_is_detected_in_a_scratch_copy():
    """MUTATION TEST — re-introduce one pre-repair line in a scratch COPY and
    confirm the rule flags it there (and does not flag the pristine copy)."""
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    site = proof["sites"][0]
    rel = site["file"]
    pristine = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert not find_prefix_defeaters(pristine), (
        "the pristine %s already has a prefix defeater" % rel
    )
    # The FIRST *TEXTUAL* occurrence is a COMMENT for 10 of the 20 sites
    # (each launcher carries a commented copy of the assignment above the
    # real one).  The mutation therefore landed in a comment, the detector
    # masks comments out, and this test reported "MUTATION SURVIVED" for a
    # mutation that was never in a position to be seen — a test that could
    # not FAIL.  Locate the site BY CONTENT and mutate THAT line.
    lines = pristine.split("\n")
    ln = locate_site_line(pristine, site)
    assert site["to"] in lines[ln - 1], (
        "the located line %d does not carry the repaired token: %r"
        % (ln, lines[ln - 1])
    )
    mutated = "\n".join(
        lines[: ln - 1]
        + [lines[ln - 1].replace(site["to"], site["from"], 1)]
        + lines[ln :]
    )
    assert mutated != pristine, "the content-located mutation did not apply"
    found = find_prefix_defeaters(mutated)
    assert found, (
        "MUTATION SURVIVED: re-introducing %r in a scratch copy was not detected"
        % site["from"]
    )
    assert found[0]["token"] == site["from"], (
        "the mutation was detected, but at the wrong token: %r" % found[0]["token"]
    )
    # the scratch copy must not have touched the repository
    assert (REPO_ROOT / rel).read_text(encoding="utf-8") == pristine


def test_the_test_rule_and_the_repair_tool_agree():
    """Two independent implementations of the same rule must not diverge."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wfa167_tool", TOOL_PY)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    disagreements = []
    for rel in tracked_sh():
        fp = REPO_ROOT / rel
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        mine = {(d["line"], d["token"]) for d in find_prefix_defeaters(text)}
        theirs = set()
        for i, line in enumerate(text.split("\n"), 1):
            for rec in tool.scan_line(line):
                if rec["classification"] == tool.REAL_DEFEATER:
                    theirs.add((i, rec["token"]))
        if mine != theirs:
            disagreements.append("%s: test=%s tool=%s" % (rel, sorted(mine), sorted(theirs)))
    assert not disagreements, "rule implementations disagree:\n" + "\n".join(disagreements)


def test_this_module_and_its_tool_stay_cpu_only():
    """CPU-ONLY GUARD — nothing here may import torch or query a device."""
    for path in (Path(__file__), TOOL_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.split(".")[0] in ("torch", "cupy"), (
                        "%s imports %s" % (path, alias.name)
                    )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").split(".")[0] in ("torch", "cupy"), (
                    "%s imports from %s" % (path, node.module)
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert _DEVICE_QUERY not in node.value, (
                    "%s contains a device-query string" % path
                )


def test_the_archived_launchers_are_not_called_from_anywhere_live():
    """The archived-site policy: if nothing live calls them, they are records
    of past runs and editing them would rewrite history."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    live = [
        p
        for p in out
        if p and not p.startswith("TODO/archive/") and not p.startswith("results/")
    ]
    assert len(live) > 8000, "the enumeration collapsed: %d files" % len(live)
    offenders = []
    for rel in live:
        fp = REPO_ROOT / rel
        if not fp.exists() or fp.suffix not in (".sh", ".py", ".yaml", ".yml"):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ARCHIVED_INVOKE_RE.search(line):
                offenders.append("%s:%d: %s" % (rel, i, stripped[:120]))
    assert not offenders, (
        "a live file executes an archived launcher (the archived files are "
        "therefore LIVE and must be repaired):\n" + "\n".join(offenders)
    )


def test_the_repaired_files_still_parse_as_shell():
    """``bash -n`` on every file this wave touched."""
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    for rel in sorted({s["file"] for s in proof["sites"]}):
        proc = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / rel)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, "bash -n failed for %s: %s" % (rel, proc.stderr[:400])
