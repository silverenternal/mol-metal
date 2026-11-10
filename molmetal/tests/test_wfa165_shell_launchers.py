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
"""Wf-A165 branch A — the shell launchers must not hard-override the GPU mask.

THE DEFECT
----------
The session's GPU lock is "run everything with ``CUDA_VISIBLE_DEVICES=""``".
Several tracked shell scripts UNCONDITIONALLY overwrite that variable, so
running one of them silently re-enables a GPU — exactly what the lock exists
to prevent.  Two mechanisms were in scope:

  (1) HARD OVERRIDE    ``export CUDA_VISIBLE_DEVICES=0``   (or ``=1``)
  (2) ``${VAR:-default}`` — ``:-`` substitutes the default when the variable
      is unset **OR EMPTY**, so an ambient ``""`` (the lock) is replaced by
      the default.  POSIX ``${VAR-default}`` (no colon) keeps an explicitly
      empty value.

THE RULE THIS FILE ENFORCES
----------------------------
Every ``export CUDA_VISIBLE_DEVICES=...`` line in every tracked ``*.sh``
must be **lock-preserving**: it may not turn an explicitly-set ambient mask
into a *more permissive* one.  Two ways to satisfy that:

  * the value expression REFERENCES ``CUDA_VISIBLE_DEVICES`` with the
    ``-`` operator (no colon) — ``${CUDA_VISIBLE_DEVICES-0}`` — so an
    explicitly-set ambient value (including the empty string) survives; or
  * the value expression is the empty string — it can only ever TIGHTEN the
    mask, never re-enable a GPU.

The behavioural form of the same rule is checked too: for each export line,
the line is EXTRACTED from the file (never hand-typed), evaluated in a
``bash -c`` harness with the file's own simple assignments in front of it,
and the result must satisfy

  * ambient ``""``   -> the exported value is ``""``   (the lock survives), and
  * ambient ``"1"``  -> the exported value is ``"1"`` or ``""``
    (an explicit choice is honoured, or the mask is tightened — never
    silently swapped for a different device).

EXEMPTIONS
----------
``scripts/start.sh`` is owned by another branch of the same wave: its two
exports live inside ``if [[ "$_AMBIENT_CVD_SET" == "1" ]] ... elif ...``
— a *conditional* whose meaning a line-level rule cannot see (the yaml path
is only reached when the caller expressed no preference).  The exemption is
NOT unconditional: ``test_the_exempt_file_still_distinguishes_unset_from_set_empty``
asserts the file still carries an ambient-presence test, so if that guard is
ever removed the exemption becomes invalid and the strict rule applies again.

CPU-ONLY
--------
This module imports stdlib only.  The harness is ``bash`` variable
expansion: torch is never imported, no CUDA context is created, and no GPU
memory is allocated.  File enumeration goes through ``git ls-files``
(NEVER ``os.walk`` — this repo is ~201 GB).
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CVD = "CUDA_VISIBLE_DEVICES"

#: The repair artefact this test cross-checks against.
PROOF_JSON = REPO_ROOT / "molmetal" / "reports" / "wfa165_shell_launchers" / "proof.json"
#: The repair tool (its classifier is cross-checked against this file's own).
TOOL_PY = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wfa165_shell_launchers"
    / "wfa165_shell_launcher_repair.py"
)

#: Files owned by another branch, with the justification that must still hold.
EXEMPT_FILES: Dict[str, str] = {
    "scripts/start.sh": (
        "owned by branch B (the yaml-derived form). Its exports sit inside an "
        "ambient-presence conditional, which a line-level rule cannot see."
    )
}

EXPORT_RE = re.compile(r"^(?P<indent>\s*)export\s+CUDA_VISIBLE_DEVICES\s*=\s*(?P<rhs>.*)$")
ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=")
#: A test that distinguishes "unset" from "set to the empty string".
AMBIENT_PRESENCE_RE = re.compile(
    r"\$\{CUDA_VISIBLE_DEVICES\+set\}|-v\s+CUDA_VISIBLE_DEVICES|_AMBIENT_CVD_SET"
)

SAFE = "safe_preserves_ambient"
FORCES_BLANK = "safe_forces_blank"
DEFEATER = "defeater_overwrites_ambient"


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


def unquote(rhs: str) -> str:
    rhs = rhs.strip()
    if len(rhs) >= 2 and rhs[0] == rhs[-1] and rhs[0] in ("'", '"'):
        return rhs[1:-1]
    return rhs


def _matching_brace(inner: str) -> int:
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def classify(rhs: str) -> str:
    """The rule: is this export line lock-preserving?

    ``safe_preserves_ambient`` — the value references ``CUDA_VISIBLE_DEVICES``
    with the ``-`` (no colon) operator, so an explicitly-set ambient value
    survives.
    ``safe_forces_blank``      — the value is the empty string.
    ``defeater_overwrites_ambient`` — anything else.
    """
    inner = unquote(rhs)
    if inner == "":
        return FORCES_BLANK
    if inner.startswith("${") and inner.endswith("}"):
        close = _matching_brace(inner)
        if close == len(inner) - 1:
            name = inner[2:close]
            if name.startswith(CVD + "-"):
                return SAFE
    return DEFEATER


def export_sites(rel: str) -> List[Tuple[int, str, str]]:
    """-> [(line_no, full_line, rhs), ...] for one file."""
    out = []
    for i, line in enumerate(read_lines(rel), 1):
        m = EXPORT_RE.match(line)
        if m:
            out.append((i, line, m.group("rhs").strip()))
    return out


def prelude_for(rel: str, upto: int, inner: str) -> List[str]:
    """The file's own simple assignments for the variables the value needs.

    ``launch_cfm_round7_f5f6_retrain.sh`` derives its default from ``GPU``;
    without the launcher's own ``GPU="${GPU:-0}"`` the harness would measure
    the harness instead of the launcher.
    """
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", inner)) - {CVD}
    prelude: List[str] = []
    lines = read_lines(rel)
    for name in sorted(names):
        found = None
        for line in lines[: upto - 1]:
            m = ASSIGN_RE.match(line)
            if m and m.group("name") == name:
                found = line
        if found is not None:
            prelude.append(found)
    return prelude


def evaluate(rel: str, line_no: int, ambient: Optional[str]) -> str:
    """Evaluate the file's export line under one ambient setting.

    ``ambient=None`` means the variable is UNSET.  The export line is
    EXTRACTED from the file; the harness is bash variable expansion only.
    """
    for ln, full, rhs in export_sites(rel):
        if ln != line_no:
            continue
        prelude = prelude_for(rel, ln, unquote(rhs))
        show = 'printf "[%s]\\n" "$' + CVD + '"'
        script = "\n".join(prelude + [full, show])
        cmd = ["bash", "-c", script]
        if ambient is None:
            cmd = ["env", "-u", CVD] + cmd
        else:
            cmd = ["env", CVD + "=" + ambient] + cmd
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True
        )
        assert proc.returncode == 0, (rel, line_no, proc.stderr[:400])
        return proc.stdout.strip()
    raise AssertionError("no export line at %s:%d" % (rel, line_no))


def load_tool_classifier():
    """Import the repair tool's classifier for a cross-check."""
    spec = importlib.util.spec_from_file_location("wfa165_tool", TOOL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------
def test_the_rule_flags_a_synthetic_hard_override():
    """NEGATIVE CONTROL — the rule MUST flag these, and MUST NOT flag these."""
    must_flag = [
        "export CUDA_VISIBLE_DEVICES=0",
        "export CUDA_VISIBLE_DEVICES=1",
        "export CUDA_VISIBLE_DEVICES=3",
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"',
        'export CUDA_VISIBLE_DEVICES="${GPU}"',
        'export CUDA_VISIBLE_DEVICES="${PA17_GPU:-0}"',
        "export CUDA_VISIBLE_DEVICES=$GPU",
    ]
    for line in must_flag:
        rhs = EXPORT_RE.match(line).group("rhs")
        assert classify(rhs) == DEFEATER, "rule missed a defeater: %s" % line

    must_pass = [
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"',
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"',
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"',
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${PA17_GPU:-0}}"',
        'export CUDA_VISIBLE_DEVICES=""',
    ]
    for line in must_pass:
        rhs = EXPORT_RE.match(line).group("rhs")
        assert classify(rhs) != DEFEATER, "rule flagged a safe line: %s" % line


def test_no_tracked_shell_script_hard_overrides_the_gpu_mask():
    """The strict rule over EVERY tracked ``*.sh`` (minus the documented
    exemptions).  A single offender fails the wave."""
    offenders = []
    for rel in tracked_sh():
        if rel in EXEMPT_FILES:
            continue
        for ln, full, rhs in export_sites(rel):
            if classify(rhs) == DEFEATER:
                offenders.append("%s:%d: %s" % (rel, ln, full.strip()))
    assert not offenders, "lock-defeating export lines found:\n" + "\n".join(offenders)


def test_every_export_line_respects_the_lock_behaviourally():
    """The same rule, measured instead of parsed.

    ambient ""  -> the exported value must be "" (the lock survives)
    ambient "1" -> the exported value must be "1" or "" (an explicit choice
                    is honoured, or the mask is tightened — never swapped for
                    a different device)
    """
    failures = []
    for rel in tracked_sh():
        if rel in EXEMPT_FILES:
            continue
        for ln, full, rhs in export_sites(rel):
            empty = evaluate(rel, ln, "")
            one = evaluate(rel, ln, "1")
            if empty != "[]":
                failures.append(
                    "%s:%d: ambient '' -> %s (a GPU would become visible)"
                    % (rel, ln, empty)
                )
            if one not in ("[1]", "[]"):
                failures.append(
                    "%s:%d: ambient '1' -> %s (an explicit choice was replaced)"
                    % (rel, ln, one)
                )
    assert not failures, "lock-defeating export lines found:\n" + "\n".join(failures)


def test_the_exempt_file_still_distinguishes_unset_from_set_empty():
    """The exemption is NOT unconditional: the ambient capture must survive."""
    for rel in EXEMPT_FILES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert AMBIENT_PRESENCE_RE.search(text), (
            "%s is exempt from the strict rule (%s) but no longer carries an "
            "ambient-presence test — the exemption is now unjustified and the "
            "strict rule must be applied to it." % (rel, EXEMPT_FILES[rel])
        )


def find_token_lines(rel: str, token: str) -> List[int]:
    """Line numbers in ``rel`` whose stripped content is exactly ``token``.

    The CONTENT-based replacement for a line-number locator (2026-11-04): a
    line number is not a durable locator — the same class as
    ``wf_a190_locator_class`` ("sha256 is not a durable locator either") —
    because the file may grow above the site without the site changing.
    MEASURED: this artefact's ``line_now`` was correct for 18 of its 23
    sites, and the same assertion fails for 5 of 23 against HEAD's blobs
    ⇒ the drift is PRE-EXISTING, not caused by the 2026-11-04 CPU-lock
    repair.
    """
    return [i + 1 for i, line in enumerate(read_lines(rel)) if line.strip() == token]


def test_the_census_covers_every_tracked_shell_script():
    """The denominator — as an INVARIANT, not as a frozen count.

    2026-11-04 (Wf-A269 wave-22).  The old assertion was
    ``proof["census"]["export_lines_on_disk"] == total`` — a frozen count of a
    DATED SNAPSHOT compared against the LIVE tree.  It is red whenever the
    tree grows, which is not a defect:

      * the live count is **33** against the census's **28**, and the +5 is
        this repo's OWN 2026-11-04 CPU-lock repair adding
        ``export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"`` to the
        five ``molmetal/scripts/smoke/*.sh`` scripts (HEAD has **0** export
        lines in each);
      * the census also no longer lists **6** of the tracked ``*.sh``: they
        were ``git mv``'d from ``TODO/archive/wf437_simplify/`` into
        ``scripts/`` after it ran, so the census's recorded paths are gone
        and the new paths are unrecorded.

    A frozen count is the §6.269.32 defect class (a pin that can only be
    right once).  What is worth asserting, and is drift-proof:

      * the census is INTERNALLY consistent — its per-file counts sum to its
        own headline;
      * the census is a FLOOR, never a ceiling — the live tree may only have
        grown since it ran;
      * every file the census recorded still exists, either at its recorded
        path or under a RENAME (its basename is still tracked).

    The live lock-preservation invariant the count was a proxy for is
    enforced by the two sibling tests above, which scan ``tracked_sh()``
    directly — a count is not that.
    """
    files = tracked_sh()
    assert len(files) > 100, "the enumeration collapsed: %d files" % len(files)
    total = sum(len(export_sites(rel)) for rel in files)
    assert total >= 1
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    census = proof["census"]
    # (a) INTERNAL consistency of the artefact
    summed = sum(entry.get("n_export_lines", 0) for entry in proof["files"])
    assert summed == census["export_lines_on_disk"], (
        "the census's per-file counts do not sum to its own headline: "
        "%d != %d" % (summed, census["export_lines_on_disk"])
    )
    # (b) the census is a FLOOR: the live tree may only have grown since it ran
    assert census["export_lines_on_disk"] <= total, (
        "the live tree has FEWER export lines (%d) than the 2026-11-01 census "
        "recorded (%d) — a census cannot un-count lines"
        % (total, census["export_lines_on_disk"])
    )
    assert census["tracked_sh_files_scanned"] <= len(files), (
        "the live tree has FEWER tracked *.sh (%d) than the 2026-11-01 census "
        "scanned (%d) — the enumeration cannot shrink"
        % (len(files), census["tracked_sh_files_scanned"])
    )
    # (c) every recorded file still exists, at its path or under a rename
    live_basenames = {Path(rel).name for rel in files}
    lost = [
        entry["file"]
        for entry in proof["files"]
        if not (REPO_ROOT / entry["file"]).exists()
        and Path(entry["file"]).name not in live_basenames
    ]
    assert not lost, "the census records files that no longer exist: %s" % lost


def test_the_proof_artifact_matches_the_files_on_disk():
    """The artefact must not be stale: every recorded line must still be there."""
    proof = json.loads(PROOF_JSON.read_text(encoding="utf-8"))
    assert proof["proof_summary"]["all_three_pass"] is True
    assert proof["proof_summary"]["files_any_fail"] == 0
    for site in proof["census"]["repair_sites"]:
        rel = site["file"]
        # locate BY CONTENT (2026-11-04) — see ``find_token_lines``: the
        # artefact's ``line_now`` is a snapshot and is correct for only 18 of
        # the 23 sites; a line number is not a durable locator.
        where = find_token_lines(rel, site["to"])
        assert where, (
            "%s: the repaired export line %r is no longer in the file"
            % (rel, site["to"])
        )
    for entry in proof["files"]:
        if entry["status"] != "applicable":
            continue
        for site in entry["exports"]:
            assert site["all_three_pass"], "%s:%d fails its own proof" % (
                entry["file"],
                site["line"],
            )


def test_the_test_rule_and_the_repair_tool_agree():
    """Two independent implementations of the same rule must not diverge."""
    tool = load_tool_classifier()
    disagreements = []
    for rel in tracked_sh():
        for ln, full, rhs in export_sites(rel):
            mine = classify(rhs)
            theirs = tool.classify_export(rhs)[0]
            if mine != theirs:
                disagreements.append(
                    "%s:%d %r test=%s tool=%s" % (rel, ln, rhs, mine, theirs)
                )
    assert not disagreements, "rule implementations disagree:\n" + "\n".join(
        disagreements
    )
