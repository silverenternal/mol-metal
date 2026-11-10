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

"""Wf-A269 wave-10 -- the ``|| true`` / ``PIPESTATUS`` rc-capture guard.

THE DEFECT CLASS
----------------
Eight Mol-Metal launchers read the trainer's exit status out of
``${PIPESTATUS[0]}`` immediately after a ``... | tee "<log>" || true``
pipeline::

    setsid nohup "${CMD[@]}" 2>&1 | tee "${LOG_OUT}" || true
    RC=${PIPESTATUS[0]}

``|| true`` runs AFTER the pipeline has already failed.  Bash resets
``PIPESTATUS`` to the status of the LAST command it ran, so the read sees
``(0)`` and every "trainer exited rc != 0 -> exit" branch is DEAD: the
wrapper exits 0 after a crashed trainer.  The mirror image of the wave-10
class "a check that cannot fail" -- here the verdict can never be READ.

MEASURED on 2026-11-02 (CPU-only, GNU bash, the interpreter these wrappers
run under)::

    $ bash -c 'set -uo pipefail; (exit 7) | tee /dev/null || true; echo "${PIPESTATUS[0]}"'
    0
    $ bash -c 'set -uo pipefail; set +o pipefail; (exit 7) | tee /dev/null; echo "${PIPESTATUS[0]}"; set -o pipefail'
    7

THE REPAIR
----------
The repo's own canonical shape (``scripts/_WRAPPER_TEMPLATE.sh:190-193``)::

    set +o pipefail
    setsid --wait ... 2>&1 | tee "<log>"
    rc=${PIPESTATUS[0]}
    set -o pipefail

``set +o pipefail`` is load-bearing (under pipefail a trainer whose forked
DataLoader workers still hold the pipe open takes SIGPIPE and the pipeline
reports 141 instead of the trainer's own status); ``setsid --wait`` is
load-bearing (bare ``setsid`` may fork and return 0 before the trainer has
done anything, which would make ``PIPESTATUS[0]`` report success for a run
that had not started); and the rc read must sit BETWEEN the two ``set``
lines, while ``PIPESTATUS`` still holds the pipeline's statuses.

WHAT THIS GATE COVERS
----------------------
* The eight launchers of the Wf-A269 wave-10 group 3 block: ``bash -n``
  passes, the canonical 4-line shape is present, and no live
  ``| tee ... || true`` survives in front of a ``${PIPESTATUS[0]}`` read.
* A BEHAVIOURAL check: the block is lifted out of the LIVE script, the
  trainer program is swapped for a stub that exits 7, and the captured rc
  must come back as 7.  The pre-repair idiom is run through the SAME harness
  and must come back as 0 -- the dead branch, demonstrated rather than
  asserted.
* The AGENTS.md §5.7 smoke wrapper additionally runs END-TO-END from a copy
  in a scratch tree whose producer stub exits 7: the wrapper must exit 7
  (the FAIL branch), not 0.
* Each repaired file carries the AGENTS.md §5.5 rule #6 marker, so the
  superseded rendering is preserved rather than deleted.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: The eight launchers of Wf-A269 wave-10 group 3.
LAUNCHERS = (
    "launch_variance_puct_smoke.sh",
    "launch_cfm_round7_f5f6_retrain.sh",
    "launch_cfm_tmqm_all6_retrain.sh",
    "launch_dmpnn_clean5_lomo.sh",
    "launch_dmpnn_clean5v3_loso.sh",
    "launch_dmpnn_clean5v4_loso_d1d3.sh",
    "launch_variance_puct_full.sh",
    "launch_wf446_cfm_tmqm_large_cohort.sh",
)

#: The AGENTS.md §5.7 50-step CPU smoke wrapper (finding C1-1).
SMOKE_LAUNCHER = "launch_variance_puct_smoke.sh"

#: AGENTS.md §5.5 rule #6 -- the exact marker a superseded rendering sits under.
SUPERSEDED_MARKER = (
    "# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:"
)

#: The rc read: ``RC=${PIPESTATUS[0]}`` / ``TRAIN_RC=...`` / ``local rc=...``.
RC_READ_RE = re.compile(
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)=\s*\"?\$\{PIPESTATUS\[0\]\}\"?"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _script_lines(name: str) -> list[str]:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8").splitlines()


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _rc_read_index(lines: list[str], name: str) -> int:
    """The index of the LIVE ``${PIPESTATUS[0]}`` read (comments excluded)."""
    hits = [
        i
        for i, ln in enumerate(lines)
        if "PIPESTATUS[0]" in ln and not _is_comment(ln)
    ]
    assert len(hits) == 1, (
        f"{name}: expected exactly one live ${{PIPESTATUS[0]}} read, got {hits}"
    )
    return hits[0]


def _canonical_indices(lines: list[str], name: str) -> tuple[int, int, int]:
    """(disable, read, enable) -- the indices of the canonical 4-line shape."""
    i_read = _rc_read_index(lines, name)
    disables = [i for i, ln in enumerate(lines) if ln.strip() == "set +o pipefail"]
    enables = [i for i, ln in enumerate(lines) if ln.strip() == "set -o pipefail"]
    i_disable = max((i for i in disables if i < i_read), default=None)
    i_enable = min((i for i in enables if i > i_read), default=None)
    assert i_disable is not None, (
        f"{name}: no `set +o pipefail` before the rc read at line {i_read + 1}"
    )
    assert i_enable is not None, (
        f"{name}: no `set -o pipefail` after the rc read at line {i_read + 1}"
    )
    return i_disable, i_read, i_enable


def _pipeline_text(lines: list[str], i_disable: int, i_read: int) -> str:
    """The joined text of the pipeline between the two ``set`` lines."""
    block = lines[i_disable + 1 : i_read]
    assert block, "the canonical block carries no pipeline"
    joined = " ".join(ln.strip() for ln in block)
    joined = re.sub(r"\\\s*", " ", joined)  # drop shell line continuations
    return re.sub(r"\s+", " ", joined).strip()


def _stub_left(left: str, stub: str) -> str:
    """Swap the TRAINER PROGRAM for the stub, keeping the command's shape.

    The stub ignores its arguments and exits 7, so whatever the launcher's
    trainer invocation looked like, the pipeline's left-hand status is 7.
    """
    if re.search(r"(?<![\w-])setsid(?![\w-])", left):
        return re.sub(r"(?<![\w-])setsid(?![\w-])", stub, left, count=1)
    if '"$SMOKE"' in left:
        return left.replace('"$PY"', "bash").replace('"$SMOKE"', stub)
    raise AssertionError(f"unrecognised trainer command shape: {left!r}")


def _synth_script(name: str, tmp_path: Path, old_shape: bool) -> Path:
    """Build a runnable harness around the LIVE block lifted from ``name``.

    ``old_shape=True`` re-appends the pre-repair ``|| true`` to the SAME
    pipeline -- the mutation self-test that demonstrates the dead branch.
    """
    lines = _script_lines(name)
    i_disable, i_read, _ = _canonical_indices(lines, name)
    pipe_text = _pipeline_text(lines, i_disable, i_read)

    match = re.match(r"^(?P<left>.*?)\|\s*tee\s+(?P<target>\S+)$", pipe_text)
    assert match, f"{name}: no `| tee` in the canonical block: {pipe_text!r}"

    stub = tmp_path / "stub_trainer.sh"
    stub.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    stub.chmod(0o755)

    left = _stub_left(match.group("left").strip(), str(stub))
    pipeline = f"{left} | tee /dev/null"
    if old_shape:
        pipeline += " || true"

    rc_line = lines[i_read]
    var_match = RC_READ_RE.search(rc_line)
    assert var_match, f"{name}: cannot parse the rc read line: {rc_line!r}"
    var = var_match.group("var")

    script = tmp_path / f"synth_{'old' if old_shape else 'new'}_{name}"
    script.write_text(
        "#!/usr/bin/env bash\n"
        # `set -u` is deliberately NOT set here: the harness substitutes the
        # trainer command with a stub, so the launcher's own shell variables
        # (${LOG_OUT}, ${OUT_BASE}, ...) are undefined in the harness.  What
        # is under test is the PIPESTATUS capture, not nounset.
        "set -o pipefail\n"
        "__wfa269_probe() {\n"
        f"{pipeline}\n"
        f"{rc_line}\n"
        "set -o pipefail\n"
        f"printf 'CAPTURED_RC=%s\\n' \"${var}\"\n"
        "}\n"
        "__wfa269_probe\n",
        encoding="utf-8",
    )
    return script


def _run(script: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(script.parent),
    )


def _captured(proc: subprocess.CompletedProcess) -> int:
    match = re.search(r"CAPTURED_RC=(-?\d+)", proc.stdout)
    assert match, (
        f"no CAPTURED_RC in stdout={proc.stdout!r} stderr={proc.stderr!r} "
        f"rc={proc.returncode}"
    )
    return int(match.group(1))


# ---------------------------------------------------------------------------
# static gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_parses(name: str) -> None:
    """``bash -n`` must accept every repaired wrapper."""
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPTS_DIR / name)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"{name}: bash -n failed: {proc.stderr}"


@pytest.mark.parametrize("name", LAUNCHERS)
def test_canonical_shape_present(name: str) -> None:
    """set +o pipefail / pipeline / rc read / set -o pipefail, in that order."""
    lines = _script_lines(name)
    i_disable, i_read, i_enable = _canonical_indices(lines, name)
    assert i_disable < i_read < i_enable, (
        f"{name}: canonical shape out of order "
        f"({i_disable}, {i_read}, {i_enable})"
    )
    block = lines[i_disable + 1 : i_read]
    assert any("| tee" in ln for ln in block), (
        f"{name}: the canonical block carries no `| tee` pipeline"
    )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_no_tee_or_true_in_front_of_the_rc_read(name: str) -> None:
    """No live ``| tee ... || true`` may sit in front of a rc read.

    ``|| true`` clobbers PIPESTATUS, so the rc read downstream of it always
    sees 0 -- the defect this gate exists to keep fixed.
    """
    lines = _script_lines(name)
    i_read = _rc_read_index(lines, name)
    preceding = [
        (i, ln)
        for i, ln in enumerate(lines[:i_read])
        if "| tee" in ln and not _is_comment(ln)
    ]
    assert preceding, f"{name}: no live `| tee` pipeline above the rc read"
    for i, ln in preceding:
        assert "|| true" not in ln, (
            f"{name}:{i + 1}: `|| true` still precedes the rc read: {ln!r}"
        )
    # ... and nowhere else in the file either.
    for i, ln in enumerate(lines):
        if _is_comment(ln):
            continue
        if "| tee" in ln:
            assert "|| true" not in ln, (
                f"{name}:{i + 1}: live `| tee ... || true`: {ln!r}"
            )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_superseded_rendering_is_preserved(name: str) -> None:
    """AGENTS.md §5.5 rule #6 -- the old rendering is kept, not deleted."""
    text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
    assert SUPERSEDED_MARKER in text, (
        f"{name}: the superseded rendering was deleted instead of preserved"
    )


# ---------------------------------------------------------------------------
# behavioural gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", LAUNCHERS)
def test_live_block_captures_a_non_zero_trainer_rc(name: str, tmp_path: Path) -> None:
    """A trainer exiting 7 must come back as rc=7 through the LIVE block."""
    script = _synth_script(name, tmp_path, old_shape=False)
    proc = _run(script)
    assert _captured(proc) == 7, (
        f"{name}: the canonical block did NOT capture the trainer's rc: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r} rc={proc.returncode}"
    )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_pre_repair_shape_could_never_see_a_failure(
    name: str, tmp_path: Path
) -> None:
    """Mutation self-test: the pre-repair ``|| true`` idiom reads 0.

    This is the dead branch, demonstrated through the same harness -- the
    pre-repair shape CANNOT report a failed trainer.
    """
    script = _synth_script(name, tmp_path, old_shape=True)
    proc = _run(script)
    assert _captured(proc) == 0, (
        f"{name}: the pre-repair shape unexpectedly reported a failure -- "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_smoke_wrapper_fails_when_its_producer_fails(tmp_path: Path) -> None:
    """END-TO-END: the AGENTS.md §5.7 smoke gate must be able to FAIL.

    The wrapper is copied into a scratch tree whose venv python is a stub
    that exits 7.  Pre-repair the wrapper reported PASS and exited 0; the
    repaired wrapper must take its FAIL branch and exit 7.
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    shutil.copy2(SCRIPTS_DIR / SMOKE_LAUNCHER, root / "scripts" / SMOKE_LAUNCHER)

    stub_py = root / ".venv" / "bin" / "python"
    stub_py.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    stub_py.chmod(0o755)
    (root / "scripts" / "variance_puct_smoke_invocation.py").write_text(
        "# stub producer: the venv python above exits 7 before reading it\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["USER_OK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        ["bash", str(root / "scripts" / SMOKE_LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(root),
    )
    assert proc.returncode == 7, (
        f"the smoke wrapper did not take its FAIL branch: rc={proc.returncode} "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "FAIL: variance PUCT smoke exited 7" in proc.stderr, (
        f"the smoke wrapper's FAIL message is missing: {proc.stderr!r}"
    )
    assert "PASS: variance PUCT smoke" not in proc.stdout, (
        f"the smoke wrapper reported PASS for a failed producer: {proc.stdout!r}"
    )


def test_smoke_wrapper_still_passes_when_the_producer_succeeds(
    tmp_path: Path,
) -> None:
    """The repair must not break the happy path: rc=0 -> PASS, exit 0."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    shutil.copy2(SCRIPTS_DIR / SMOKE_LAUNCHER, root / "scripts" / SMOKE_LAUNCHER)

    stub_py = root / ".venv" / "bin" / "python"
    stub_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub_py.chmod(0o755)
    (root / "scripts" / "variance_puct_smoke_invocation.py").write_text(
        "# stub producer: the venv python above exits 0\n", encoding="utf-8"
    )

    env = dict(os.environ)
    env["USER_OK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        ["bash", str(root / "scripts" / SMOKE_LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(root),
    )
    assert proc.returncode == 0, (
        f"the smoke wrapper failed on a successful producer: rc={proc.returncode} "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "PASS: variance PUCT smoke" in proc.stdout, (
        f"the smoke wrapper did not report PASS: {proc.stdout!r}"
    )
