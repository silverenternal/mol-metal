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

"""Wf-A269 wave-10 -- the DEAD-PROBE class: an epoch probe that cannot match.

THE DEFECT (FINDING C2-7)
-------------------------
``scripts/launch_chemprop_v3_p14_post_cfm_round7.sh`` polls the cfm_round7 run
directory every 60 s while that wrapper is still alive and prints the last
epoch it can see.  The probe read

    run/metrics.json          <- DOES NOT EXIST on disk (MEASURED 2026-11-02)
    pattern '"epoch":[0-9]+'  <- matches 0 lines of the file that DOES exist

The sibling that does exist is ``run/per_epoch_metrics.json``, and it writes
``"epoch": 0`` WITH the JSON indent space -- so the unspaced pattern matched
nothing.  A probe whose pattern matches nothing can never report an epoch:
the success branch is unreachable and the wrapper always fell through to the
train.log fallback.  MEASURED 2026-11-02 on disk: 0 matches for the unspaced
form, 6 for the spaced form; ``run/metrics.json`` does not exist at all.

THE REPAIR
----------
The probe reads ``run/per_epoch_metrics.json`` and its pattern tolerates the
indent space (``'"epoch":[[:space:]]*[0-9]+'``); the ``[cfm-bond] epoch=``
train.log fallback is kept.

THE TESTING METHOD (and its one deliberate blind spot)
------------------------------------------------------
The wrapper preserves the superseded rendering VERBATIM in a comment block
(AGENTS.md §5.5 rule #6), so the dead path and the dead pattern are still on
disk.  A substring test would therefore be satisfied by the comment.  These
tests strip the full-line comments first and assert against the CODE, which is
what actually runs.

The blind spot this method does NOT cover: an inline trailing comment
(``cmd # note``) is not stripped, so a path mentioned only in such a comment
would still be seen as "active".  Every occurrence of the probe in this
wrapper is a full-line comment, so the assertion is exact for it.

THE GROUND TRUTH IS GIT-IGNORED
--------------------------------
Both probe targets live under ``molmetal/reports/**/``, which is git-ignored
(.gitignore:146 ``molmetal/reports/**/train.log``, .gitignore:158
``molmetal/reports/**/*.json``).  On a clean checkout neither file exists, so
the "matches a real line" assertion falls back to a fixture that reproduces
the artifact's exact byte shape (a JSON indent space after the colon).  The
source-level assertions -- which target the probe names, and which pattern it
uses -- hold on a clean checkout either way.

CPU-only: no GPU, no torch, no trainer launch.  The only subprocesses are a
``grep -oE`` over a JSON file on disk and a ``bash -c`` that runs the probe's
own pipeline.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "launch_chemprop_v3_p14_post_cfm_round7.sh"

#: The run artifact the cfm_round7 trainer actually writes (MEASURED 2026-11-02:
#: it exists on disk; the sibling ``run/metrics.json`` does NOT).
EXPECTED_TARGET = "molmetal/reports/cfm_round7_f5f6_retrain/run/per_epoch_metrics.json"

#: The artifact that does not exist -- the pre-repair probe target.
DEAD_TARGET = "molmetal/reports/cfm_round7_f5f6_retrain/run/metrics.json"

#: The pre-repair pattern: no space tolerated after the colon.
DEAD_PATTERN = '"epoch":[0-9]+'

#: The train.log fallback the repair must keep.
FALLBACK_PATTERN = r"\[cfm-bond\] epoch=[0-9]+"
FALLBACK_TARGET = "molmetal/reports/cfm_round7_f5f6_retrain/run/train.log"

#: The §5.5 rule-#6 preservation marker.
RULE6_MARKER = (
    "# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:"
)

#: A probe line: ``grep -oE '<pattern>' "${REPO_ROOT}/<path>" ...``
PROBE_RE = re.compile(
    r"grep\s+-oE\s+'(?P<pattern>[^']+)'\s+"
    r"\"\$\{REPO_ROOT\}/(?P<path>[^\"]+)\""
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _code_only(text: str) -> str:
    """Strip full-line shell comments.

    The wrapper keeps the superseded rendering of the repaired probe in a
    comment block (AGENTS.md §5.5 rule #6), so the dead path and the dead
    pattern are still on disk.  The assertion has to be about the CODE that
    runs, not about the record of what used to run.
    """
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _probes(text: str) -> list[tuple[str, str]]:
    """Return ``[(pattern, repo-relative path), ...]`` for the probe CODE."""
    out: list[tuple[str, str]] = []
    for m in PROBE_RE.finditer(_code_only(text)):
        out.append((m.group("pattern"), m.group("path")))
    return out


def _json_probe(text: str) -> tuple[str, str]:
    """The probe that reads a JSON epoch record: ``(pattern, path)``."""
    for pattern, path in _probes(text):
        if path.endswith(".json"):
            return pattern, path
    raise AssertionError(
        "the wrapper no longer contains a JSON epoch probe -- this test's "
        "universe shrank, so it would be vacuous"
    )


def _fallback_probe(text: str) -> tuple[str, str]:
    """The train.log fallback probe: ``(pattern, path)``."""
    for pattern, path in _probes(text):
        if path.endswith("train.log"):
            return pattern, path
    raise AssertionError(
        "the wrapper no longer contains the train.log epoch fallback"
    )


def _py_regex(ere: str) -> str:
    """Translate the ERE the wrapper uses into a Python regex.

    POSIX character classes (``[[:space:]]``) are NOT supported by Python's
    ``re``: inside a character class it is read as a set of literal
    characters, so ``[[:space:]]*`` would NOT match the JSON indent space.
    The wrapper runs under grep(1), where the class is real; the translation
    exists so the Python-side assertion measures the same thing grep does.
    """
    return ere.replace("[[:space:]]", r"[ \t\n\r\f\v]")


def _grep_matches(pattern: str, path: Path) -> list[str]:
    """Run the wrapper's own ``grep -oE`` over ``path``; return the matches."""
    proc = subprocess.run(
        ["grep", "-oE", pattern, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Test 1 -- the probe must target the artifact that EXISTS
# ---------------------------------------------------------------------------


def test_epoch_probe_targets_the_artifact_that_exists() -> None:
    """The probe must read the file the run actually writes.

    MEASURED before the repair: the probe read ``run/metrics.json``, which
    does not exist on disk, so the branch was unreachable.
    """
    assert WRAPPER.is_file(), f"wrapper missing: {WRAPPER}"
    text = WRAPPER.read_text(encoding="utf-8")
    _pattern, path = _json_probe(text)
    assert path == EXPECTED_TARGET, (
        f"the epoch probe reads {path!r}; the run writes {EXPECTED_TARGET!r} "
        f"(MEASURED 2026-11-02) and {DEAD_TARGET!r} does not exist at all"
    )
    assert path != DEAD_TARGET, (
        f"the epoch probe still reads the absent artifact {DEAD_TARGET!r} "
        "-- the probe can never match"
    )


# ---------------------------------------------------------------------------
# Test 2 -- the pattern must match a REAL line of its target
# ---------------------------------------------------------------------------


def test_epoch_probe_pattern_matches_a_real_line_of_its_target() -> None:
    """The probe's own pattern must match at least one real line.

    MEASURED before the repair: ``'"epoch":[0-9]+'`` matches 0 lines of
    ``run/per_epoch_metrics.json`` (which writes ``"epoch": 0`` WITH the JSON
    indent space), so the probe could never report an epoch.

    The target artifact is git-ignored (``.gitignore:158``), so on a clean
    checkout it is absent; the assertion then runs against a fixture with the
    artifact's exact byte shape, and the ABSENCE must be DOCUMENTED by the
    wrapper -- an undocumented absence is exactly how the dead probe came
    back.
    """
    text = WRAPPER.read_text(encoding="utf-8")
    pattern, path = _json_probe(text)
    target = REPO_ROOT / path

    if target.exists():
        matches = _grep_matches(pattern, target)
        assert matches, (
            f"the epoch probe's pattern {pattern!r} matches NO line of the "
            f"real artifact {path} -- the probe is dead"
        )
        py_hits = [
            ln for ln in target.read_text(encoding="utf-8").splitlines()
            if re.search(_py_regex(pattern), ln)
        ]
        assert py_hits, (
            f"the epoch probe's pattern {pattern!r} matches no line of the "
            f"real artifact {path} under the Python translation "
            f"{_py_regex(pattern)!r} either"
        )
    else:
        assert "git-ignored" in text, (
            f"the probe target {path} is absent from the checkout and the "
            "wrapper does not DOCUMENT that absence -- an undocumented "
            "absence is how the dead probe came back"
        )
        fixture = (
            '{\n'
            '  "best": {\n'
            '    "epoch": 0,\n'
            '    "gate_value": 0.0\n'
            '  }\n'
            '}\n'
        )
        assert re.search(_py_regex(pattern), fixture), (
            f"the epoch probe's pattern {pattern!r} does not match the "
            "artifact's byte shape (\"epoch\": 0, with the JSON indent space) "
            "-- the probe would be dead on a clean checkout too"
        )
        assert not re.search(_py_regex(DEAD_PATTERN), fixture), (
            "the fixture no longer distinguishes the dead pattern from the "
            "repaired one -- the assertion above would be vacuous"
        )


# ---------------------------------------------------------------------------
# Test 3 -- the dead rendering must not be in the ACTIVE code
# ---------------------------------------------------------------------------


def test_the_dead_probe_rendering_is_not_in_the_active_code() -> None:
    """The absent path and the unspaced pattern must be out of the CODE.

    The superseded rendering is preserved VERBATIM in a comment block
    (AGENTS.md §5.5 rule #6), so the assertion has to be against the code
    that runs.
    """
    code = _code_only(WRAPPER.read_text(encoding="utf-8"))
    assert DEAD_TARGET not in code, (
        f"the ACTIVE code still reads the absent artifact {DEAD_TARGET!r} -- "
        "the probe can never match"
    )
    assert DEAD_PATTERN not in code, (
        f"the ACTIVE code still uses the dead pattern {DEAD_PATTERN!r}, which "
        "matches 0 lines of the artifact the run actually writes"
    )


# ---------------------------------------------------------------------------
# Test 4 -- the superseded rendering must be preserved VERBATIM (rule #6)
# ---------------------------------------------------------------------------


def test_the_superseded_rendering_is_preserved_verbatim() -> None:
    """AGENTS.md §5.5 rule #6: a superseded rendering is preserved VERBATIM."""
    text = WRAPPER.read_text(encoding="utf-8")
    assert RULE6_MARKER in text, (
        "the wrapper does not carry the §5.5 rule-#6 preservation marker -- "
        "the superseded probe rendering was deleted instead of preserved"
    )
    # The preserved rendering is the pre-repair probe, byte for byte.
    preserved = (
        "#   epoch_n=$(grep -oE '\"epoch\":[0-9]+' "
        "\"${REPO_ROOT}/molmetal/reports/cfm_round7_f5f6_retrain/run/metrics.json\" "
        "2>/dev/null | tail -1)"
    )
    assert preserved in text, (
        "the pre-repair probe rendering is not preserved verbatim under the "
        "rule-#6 marker"
    )


# ---------------------------------------------------------------------------
# Test 5 -- the train.log fallback must survive the repair
# ---------------------------------------------------------------------------


def test_the_train_log_fallback_is_retained() -> None:
    """The repair must keep the ``[cfm-bond] epoch=`` train.log fallback."""
    text = WRAPPER.read_text(encoding="utf-8")
    pattern, path = _fallback_probe(text)
    assert pattern == FALLBACK_PATTERN, (
        f"the train.log fallback's pattern changed to {pattern!r}"
    )
    assert path == FALLBACK_TARGET, (
        f"the train.log fallback now reads {path!r}"
    )
    code = _code_only(text)
    assert 'if [[ -z "${epoch_n}" ]]; then' in code, (
        "the fallback branch is gone -- the probe no longer degrades to the "
        "train.log epoch when the JSON record is absent"
    )


# ---------------------------------------------------------------------------
# Test 6 -- the probe pipeline really produces a value (end to end)
# ---------------------------------------------------------------------------


def test_the_probe_pipeline_yields_an_epoch_on_this_checkout() -> None:
    """Run the wrapper's own probe pipeline over the real artifact.

    This is the end-to-end proof that the repaired probe can produce a value:
    the very command the wrapper runs, with ``${REPO_ROOT}`` substituted.

    When the git-ignored artifact is absent (a clean checkout) the pipeline
    has nothing to read, so the assertion becomes the fail-closed
    documented-absence check -- it must NOT silently pass, because a test
    that passes vacuously is the same defect class this file guards.
    """
    text = WRAPPER.read_text(encoding="utf-8")
    pattern, path = _json_probe(text)
    target = REPO_ROOT / path
    if not target.exists():
        assert "git-ignored" in text, (
            f"the probe target {path} is absent from the checkout and the "
            "wrapper does not DOCUMENT that absence"
        )
        return
    proc = subprocess.run(
        ["bash", "-c", f"grep -oE '{pattern}' '{target}' 2>/dev/null | tail -1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout.strip()
    assert out, (
        f"the probe pipeline produced NO value for {path} (rc={proc.returncode}) "
        "-- the probe is still dead"
    )
    assert re.search(r"[0-9]+", out), (
        f"the probe pipeline produced {out!r}, which carries no epoch number"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
