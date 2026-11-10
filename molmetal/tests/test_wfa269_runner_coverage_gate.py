#!/usr/bin/env python3
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
"""Wf-A269 — regression + mutation tests for the runner-coverage gate.

The gate under test is :mod:`molmetal.scripts.wfa269_runner_coverage_gate`.

WHAT THIS FILE IS FOR
---------------------
The repo's blind spot is that a test file on **no** executing runner's
pytest argv is never collected, never executed, and its assertions are
never evaluated — so a pin written inside it stays wrong forever and
nothing reports it.  The gate makes that class VISIBLE without running
1367 files.

These tests pin the gate's OWN behaviour, including the three ways it must
FAIL:

  M1  a NEW test file that no runner names and that is not in the
      declared baseline  => FAIL (the load-bearing assertion);
  M2  a declared baseline entry that became wired/deleted => FAIL (the
      baseline must be tightened, not left to rot);
  M3  a runner naming a path that does not exist => FAIL (pytest exits 4
      on such a path, so the step can never pass).

Every mutation is performed on a SYNTHETIC REPO BUILT IN ``tmp_path`` — no
file in the repository is ever written, mutated or restored (the
"mutations run on copies" rule, AGENTS.md §6.269.35.7's mechanism applied
to the gate itself).

THE AXIS IS EXERCISED, NOT ASSUMED
-----------------------------------
``test_axis_excludes_comment_and_docstring_mentions`` proves that a
pytest argv inside a YAML comment is NOT an invocation, and
``test_preflight_axis_resolves_through_variables`` proves the pre-flight
extractor resolves ``TESTS_DIR / "x.py"`` and a Name bound to a tuple of
paths.  A gate whose extractor silently returned nothing would report
"0 executed" and, without the floor, could PASS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

import wfa269_runner_coverage_gate as gate  # noqa: E402


# ---------------------------------------------------------------------------
# helpers: build a synthetic repo
# ---------------------------------------------------------------------------
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(
    tmp_path: Path,
    *,
    test_files: list[str],
    ci_argv: list[str] | None = None,
    ci_extra: str = "",
    preflight_text: str | None = None,
    baseline: dict[str, str] | None = None,
    baseline_size: int | None = None,
    executed: list[str] | None = None,
    write_sidecar: bool = True,
) -> Path:
    """Build a minimal synthetic repo the gate can read."""
    root = tmp_path / "repo"
    for rel in test_files:
        _write(root / rel, "# synthetic test file\ndef test_x():\n    assert True\n")

    ci_lines = ["name: ci", "on: [push]", "jobs:", "  t:", "    runs-on: ubuntu-latest",
                "    steps:", "      - name: pytest", "        run: |"]
    for rel in ci_argv or []:
        ci_lines.append(f"          .venv/bin/python -m pytest -q {rel}")
    if ci_extra:
        ci_lines.append(ci_extra)
    _write(root / gate.CI_WORKFLOW, "\n".join(ci_lines) + "\n")

    if preflight_text is None:
        preflight_text = (
            "def build_gates(pipeline):\n"
            "    runners = []\n"
            '    if pipeline == "all":\n'
            "        runners += [\n"
            '            ("g", lambda: gate_pytest("g", "P2", "tests", [\n'
            '                TESTS_DIR / "test_a.py",\n'
            "            ])),\n"
            "        ]\n"
            "    return runners\n"
        )
    _write(root / gate.PREFLIGHT, preflight_text)

    if write_sidecar:
        body = {
            "unexecuted": baseline or {},
            "baseline_size": baseline_size if baseline_size is not None else len(baseline or {}),
            "executed": executed or [],
        }
        _write(root / gate.BASELINE_SIDECAR, json.dumps(body, indent=2, sort_keys=True))
    return root


# ---------------------------------------------------------------------------
# (1) the live repo
# ---------------------------------------------------------------------------
def test_live_repo_gate_passes():
    out = gate.verify(REPO)
    assert out["verdict"] == "PASS", out["failures"]
    assert out["population_n"] >= 1000
    assert out["executed_n"] >= gate.MIN_EXECUTED_FLOOR
    assert out["new_unexecuted"] == []
    assert out["stale_baseline"] == []
    assert out["executed_changed"] == []
    assert out["dead_invocations"] == []


def test_live_baseline_sidecar_is_self_consistent():
    baseline, recorded = gate.load_baseline(REPO)
    assert recorded == len(baseline) == gate.BASELINE_SIZE
    assert gate.BASELINE == baseline
    # every declared entry really is unexecuted
    out = gate.verify(REPO)
    assert not (set(baseline) & set(out["executed"]))
    # and the declared reason is uniform (this is a made-explicit census,
    # not a per-file adjudication — AGENTS.md §6.269.6)
    assert set(baseline.values()) == {gate.BASELINE_REASON}


def test_live_coverage_is_the_measured_number():
    out = gate.verify(REPO)
    # MEASURED 2026-11-09 on the live tree: population 1367, executed 31,
    # unexecuted 1336 (97.73 %).  These are asserted as RELATIONS, not as
    # frozen literals, so a new test file does not break the suite — but
    # the unexecuted fraction must stay visible and non-trivial.
    assert out["population_n"] == out["executed_n"] + out["unexecuted_n"]
    assert 0 < out["unexecuted_n"] < out["population_n"]
    assert out["coverage_pct"] < 50.0


def test_guard_test_file_is_itself_wired_into_ci():
    """THE SELF-APPLICATION ASSERTION.

    A guard that no runner executes is invisible (AGENTS.md §6.269.29).
    This file must therefore be on the CI workflow's pytest argv, or the
    guard it tests would be one more unexecuted file.
    """
    ci_paths = gate.ci_test_paths((REPO / gate.CI_WORKFLOW).read_text())
    assert "molmetal/tests/test_wfa269_runner_coverage_gate.py" in ci_paths, (
        "the runner-coverage guard's own test file is not on any CI job's "
        "pytest argv — the guard would be invisible"
    )


# ---------------------------------------------------------------------------
# (2) the axis: a MENTION is not an INVOCATION
# ---------------------------------------------------------------------------
def test_axis_excludes_comment_and_docstring_mentions():
    text = (
        "jobs:\n"
        "  t:\n"
        "    steps:\n"
        "      - name: pytest\n"
        "        run: |\n"
        "          # .venv/bin/python -m pytest -q molmetal/tests/test_commented.py\n"
        "          .venv/bin/python -m pytest -q molmetal/tests/test_real.py\n"
    )
    paths = gate.ci_test_paths(text)
    assert paths == ["molmetal/tests/test_real.py"]


def test_axis_joins_backslash_continuations():
    text = (
        "jobs:\n"
        "  t:\n"
        "    steps:\n"
        "      - name: pytest\n"
        "        run: |\n"
        "          .venv/bin/python -m pytest -q \\\n"
        "            molmetal/tests/test_one.py \\\n"
        "            molmetal/tests/test_two.py\n"
    )
    assert gate.ci_test_paths(text) == [
        "molmetal/tests/test_one.py",
        "molmetal/tests/test_two.py",
    ]


def test_preflight_axis_resolves_through_variables():
    src = (
        "TESTS_DIR = Path('molmetal/tests')\n"
        "extra = (TESTS_DIR / 'test_b.py', TESTS_DIR / 'test_c.py')\n"
        "def build_gates(pipeline):\n"
        "    runners = []\n"
        '    if pipeline == "all":\n'
        "        runners += [\n"
        '            ("g", lambda: gate_pytest("g", "P2", "tests", [\n'
        '                TESTS_DIR / "test_a.py",\n'
        "            ])),\n"
        "        ]\n"
        '    if pipeline in ("frontier", "all"):\n'
        "        runners += [\n"
        '            ("h", lambda: gate_pytest("h", "P2", "tests", extra)),\n'
        "        ]\n"
        "    return runners\n"
    )
    per = gate.preflight_test_paths(src)
    assert per["all"] == [
        "molmetal/tests/test_a.py",
        "molmetal/tests/test_b.py",
        "molmetal/tests/test_c.py",
    ]
    # the `frontier` pipeline is NEVER run by an automated runner, so it is
    # excluded from the executing set by construction
    assert "frontier" not in gate.EXECUTING_PREFLIGHT_PIPELINES


# ---------------------------------------------------------------------------
# (3) the three FAILURE modes (the mutation proofs)
# ---------------------------------------------------------------------------
def test_m1_new_unwired_test_file_fails(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=[
            "molmetal/tests/test_a.py",
            "molmetal/tests/test_b.py",
            "molmetal/tests/test_brand_new.py",
        ],
        ci_argv=["molmetal/tests/test_a.py"],
        baseline={"molmetal/tests/test_b.py": gate.BASELINE_REASON},
        executed=["molmetal/tests/test_a.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert out["new_unexecuted"] == ["molmetal/tests/test_brand_new.py"]
    assert any("never wired" in f for f in out["failures"])


def test_m2_baseline_entry_that_became_wired_fails(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=[
            "molmetal/tests/test_a.py",
            "molmetal/tests/test_b.py",
        ],
        ci_argv=["molmetal/tests/test_a.py", "molmetal/tests/test_b.py"],
        baseline={"molmetal/tests/test_b.py": gate.BASELINE_REASON},
        executed=["molmetal/tests/test_a.py", "molmetal/tests/test_b.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert out["stale_baseline"] == ["molmetal/tests/test_b.py"]
    assert any("tightened" in f for f in out["failures"])


def test_m3_runner_naming_a_missing_file_fails(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=["molmetal/tests/test_a.py"],
        ci_argv=["molmetal/tests/test_a.py", "molmetal/tests/test_deleted.py"],
        baseline={},
        executed=["molmetal/tests/test_a.py", "molmetal/tests/test_deleted.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert out["dead_invocations"] == ["molmetal/tests/test_deleted.py"]
    assert any("never pass" in f for f in out["failures"])


def test_missing_sidecar_fails_closed(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=["molmetal/tests/test_a.py"],
        ci_argv=["molmetal/tests/test_a.py"],
        write_sidecar=False,
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert out["baseline_error"]
    assert any("could not be read" in f for f in out["failures"])


def test_floor_violation_fails(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=["molmetal/tests/test_a.py"],
        ci_argv=[],
        baseline={"molmetal/tests/test_a.py": gate.BASELINE_REASON},
        executed=[],
    )
    out = gate.verify(root, floor=2)
    assert out["verdict"] == "FAIL"
    assert out["floor_ok"] is False
    assert any("floor" in f for f in out["failures"])


def test_sidecar_size_pin_detects_an_edited_sidecar(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=["molmetal/tests/test_a.py", "molmetal/tests/test_b.py"],
        ci_argv=["molmetal/tests/test_a.py"],
        baseline={"molmetal/tests/test_b.py": gate.BASELINE_REASON},
        baseline_size=99,          # the pin no longer matches the content
        executed=["molmetal/tests/test_a.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert any("without updating the pin" in f for f in out["failures"])


def test_executed_set_drift_is_detected(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=["molmetal/tests/test_a.py", "molmetal/tests/test_b.py"],
        ci_argv=["molmetal/tests/test_a.py"],
        baseline={"molmetal/tests/test_b.py": gate.BASELINE_REASON},
        executed=["molmetal/tests/test_a.py", "molmetal/tests/test_b.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "FAIL"
    assert out["executed_changed"] == ["molmetal/tests/test_b.py"]
    assert any("EXECUTED set differs" in f for f in out["failures"])


def test_a_clean_synthetic_repo_passes(tmp_path):
    root = _make_repo(
        tmp_path,
        test_files=[
            "molmetal/tests/test_a.py",
            "molmetal/tests/test_b.py",
            "molmetal/tests/test_c.py",
        ],
        ci_argv=["molmetal/tests/test_a.py"],
        baseline={
            "molmetal/tests/test_b.py": gate.BASELINE_REASON,
            "molmetal/tests/test_c.py": gate.BASELINE_REASON,
        },
        executed=["molmetal/tests/test_a.py"],
    )
    out = gate.verify(root, floor=0)
    assert out["verdict"] == "PASS", out["failures"]
    assert out["population_n"] == 3
    assert out["executed_n"] == 1
    assert out["unexecuted_n"] == 2


# ---------------------------------------------------------------------------
# (4) the CLI
# ---------------------------------------------------------------------------
def test_cli_exit_codes(tmp_path, capsys):
    # The CLI always uses the LIVE floor (no override), so the synthetic
    # repository must wire at least MIN_EXECUTED_FLOOR files.  The
    # synthetic pre-flight names `test_a.py`, so that file must exist too
    # (otherwise it would be a DEAD INVOCATION and the repo would FAIL).
    floor = gate.MIN_EXECUTED_FLOOR
    wired = ["molmetal/tests/test_a.py"] + [
        f"molmetal/tests/test_f{i:03d}.py" for i in range(floor - 1)
    ]
    root = _make_repo(
        tmp_path,
        test_files=list(wired),
        ci_argv=list(wired),
        baseline={},
        executed=list(wired),
    )
    assert gate.main(["--repo-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out

    bad = _make_repo(
        tmp_path / "bad",
        test_files=list(wired) + ["molmetal/tests/test_z.py"],
        ci_argv=list(wired),
        baseline={},
        executed=list(wired),
    )
    assert gate.main(["--repo-root", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_module_exposes_the_repo_convention():
    """The BASELINE dict + BASELINE_SIZE convention must be importable."""
    assert isinstance(gate.BASELINE, dict)
    assert isinstance(gate.BASELINE_SIZE, int)
    assert gate.BASELINE_SIZE == len(gate.BASELINE)
    assert isinstance(gate.AXIS, str) and "EXECUTING" in gate.AXIS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
