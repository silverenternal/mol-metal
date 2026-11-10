# SPDX-License-Identifier: Apache-2.0
"""Wf-A154 — the two SSOT-document consistency gates are wired into the
pre-flight, and they stay wired.

WHAT THIS GUARDS
----------------
Two audits existed ONLY as pytest modules, so a commit that broke them still
passed the pre-flight:

  * ``molmetal/tests/test_wfa131_repo_path_reference_guard.py`` — a
    repo-relative path cited by an SSOT document must RESOLVE on disk
    (AGENTS.md §5.5 filesystem-truth, read from the other side: the document
    points at nothing).
  * ``molmetal/tests/test_wfa152_census_guard_coverage.py`` — a census
    statistic quoted by a document must be the CURRENT value in
    ``metrics/by_metric/cfm_run_census.json`` (AGENTS.md §5.2
    MEASURED-vs-PROJECTED).  This guard resolves each claim against the
    artifact itself, so a stale number cannot hide behind an uncurated list
    of retired numbers the way the older Wf-A134 guard allowed.

This test file asserts the WIRING (the gates exist, they are registered,
they run, and they pass on the current tree).  The RULES live in the guard
modules themselves — each runner script imports its guard module and calls
that module's OWN assertions, so there is exactly ONE implementation of each
rule and no forked copy to drift.  That "one implementation" property is
asserted here too, because a forked copy is exactly how a guard silently
stops protecting anything.

CPU-only: no torch import, no CUDA call, no checkpoint read.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = REPO / "molmetal" / "scripts"
VENV_PY = REPO / ".venv" / "bin" / "python"

RUNNERS = {
    "p3_ssot_path_reference_check": (
        SCRIPT_DIR / "wf_a131_path_reference_guard_runner.py",
        "molmetal.tests.test_wfa131_repo_path_reference_guard",
    ),
    "p3_ssot_census_claim_check": (
        SCRIPT_DIR / "wf_a152_census_claim_guard_runner.py",
        "molmetal.tests.test_wfa152_census_guard_coverage",
    ),
}


def _build_gates(pipeline: str):
    sys.path.insert(0, str(SCRIPT_DIR))
    from pre_flight_check import build_gates  # noqa: PLC0415

    return build_gates(pipeline)


def _import_runner(filename: str):
    """Import a runner script by filename from ``molmetal/scripts/``."""
    import importlib  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module(filename.removesuffix(".py"))


def _python() -> str:
    """The interpreter a child process must use.

    AGENTS.md §5.9 (single-venv): the canonical interpreter is
    ``/home/hugo/codes/mol-metal/.venv/bin/python``.  ``sys.executable`` is
    the right answer whenever pytest itself runs inside that venv; the
    explicit path is the fallback for the case where it does not, because a
    child that drifts onto another venv is exactly the defect §5.9 forbids.
    """
    if VENV_PY.is_file():
        return str(VENV_PY)
    return sys.executable


def test_both_runner_scripts_exist() -> None:
    for name, (script, _guard) in RUNNERS.items():
        assert script.is_file(), (
            f"{name}: runner script missing: {script} — the gate would FAIL "
            f"closed on every pre-flight run."
        )


def test_both_gates_are_registered_on_the_all_pipeline() -> None:
    names = {name for name, _ in _build_gates("all")}
    for name in RUNNERS:
        assert name in names, (
            f"{name} is not registered on the 'all' pipeline — a commit that "
            f"breaks an SSOT document would still pass the pre-flight."
        )


def test_the_launch_pipeline_inherits_both_gates() -> None:
    """"launch" normalises to "all", so it inherits both for free."""
    names = {name for name, _ in _build_gates("launch")}
    for name in RUNNERS:
        assert name in names, f"{name} is missing from the 'launch' pipeline"


def test_the_other_pipelines_are_unchanged() -> None:
    """The SSOT gates are 'all'-only; they must not leak into the narrow
    pipelines, whose counts are a stability contract of their own."""
    expected = {"cfm": 8, "frontier": 25, "chemprop_v2_v3": 18}
    for pipeline, count in expected.items():
        actual = len(_build_gates(pipeline))
        assert actual == count, (
            f"pipeline {pipeline!r} count drift: {actual} != {count} — the "
            f"Wf-A154 SSOT gates were registered on the wrong pipeline."
        )


def test_each_runner_calls_its_guards_own_assertions() -> None:
    """One implementation of each rule.

    A runner that re-implemented the rule inline would drift from the guard
    module, and the guard module is the thing the repo's tests exercise.
    """
    for name, (script, guard_module) in RUNNERS.items():
        text = script.read_text(encoding="utf-8")
        assert guard_module in text, (
            f"{name}: the runner does not reference {guard_module} — it must "
            f"import the guard module and call its OWN assertions, not "
            f"re-implement the rule."
        )
        assert "def test_" not in text, (
            f"{name}: the runner defines its own test_* function — that is a "
            f"forked copy of the rule, which is what this test forbids."
        )


def _code_symbols(path: Path) -> tuple[set[str], set[str]]:
    """Every identifier / attribute / import name and every NON-docstring
    string literal in a Python file.

    Docstrings are excluded on purpose: the runners DOCUMENT that they do
    not touch the GPU ("no torch import, no CUDA call, no checkpoint
    read"), and a naive substring scan over the raw file would flag its own
    prose — the exact false positive that makes a guard decorative.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_ids.add(id(first.value))

    names: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.lower())
                names.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.lower())
            for alias in node.names:
                names.add(alias.name.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_ids:
                strings.add(node.value.lower())
    return names, strings


def test_the_runners_cannot_touch_the_gpu() -> None:
    """CPU-only by construction (AGENTS.md §5.1 standing grant aside, the
    Wf-A119/A124/A125 chain is under a 0-GPU mandate)."""
    for name, (script, _guard) in RUNNERS.items():
        names, strings = _code_symbols(script)
        for forbidden in ("torch", "cuda", "nvidia_smi", "checkpoint", "ckpt"):
            assert forbidden not in names, (
                f"{name}: the runner's CODE names {forbidden!r} — it must stay "
                f"a pure text/JSON scan with no torch import, no CUDA call and "
                f"no checkpoint read."
            )
        for text in strings:
            assert "nvidia-smi" not in text, (
                f"{name}: the runner shells out to nvidia-smi — it must not "
                f"touch the GPU at all."
            )


def test_each_runner_passes_on_the_current_tree() -> None:
    """The real end-to-end check: run the runner exactly as the pre-flight
    gate does, and require exit code 0 (PASS)."""
    for name, (script, _guard) in RUNNERS.items():
        proc = subprocess.run(
            [_python(), str(script)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, (
            f"{name}: the runner exited {proc.returncode} (expected 0 = PASS).\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
        assert "PASS" in proc.stdout, (
            f"{name}: the runner exited 0 but did not print PASS:\n"
            f"{proc.stdout[-2000:]}"
        )


def test_each_runner_can_actually_fail(monkeypatch) -> None:
    """A gate that cannot FAIL is decoration.

    Inject a synthetic failing assertion into each guard module and require
    the runner to report FAIL (exit code 1).  This proves three things at
    once: the runner really consults the guard module, it really calls the
    guard's own assertions, and it really propagates the verdict instead of
    swallowing it.  Without this test, a runner that returned 0
    unconditionally would look identical to a working one.

    Nothing on disk is modified — the patch lives only for the duration of
    the call.
    """
    import importlib  # noqa: PLC0415

    cases = [
        (
            "wf_a131_path_reference_guard_runner.py",
            "test_wfa131_repo_path_reference_guard",
            "test_the_baseline_matches_the_recorded_measurement",
        ),
        (
            "wf_a152_census_claim_guard_runner.py",
            "test_wfa152_census_guard_coverage",
            "test_the_production_path_catches_a_stale_bolded_count",
        ),
    ]

    for runner_file, guard_name, victim in cases:
        runner = _import_runner(runner_file)
        guard = importlib.import_module(f"molmetal.tests.{guard_name}")
        assert hasattr(guard, victim), (
            f"{guard_name} exposes no {victim!r} — the fail-path probe has "
            f"nothing to break."
        )

        def _boom() -> None:
            raise AssertionError("synthetic: a document quotes a stale value")

        monkeypatch.setattr(guard, victim, _boom)
        try:
            rc = runner.main()
        finally:
            monkeypatch.undo()
        assert rc == 1, (
            f"{runner_file}: the runner returned {rc} when the guard's own "
            f"assertion failed — it must return 1 (FAIL), or the gate can "
            f"never fire."
        )


def test_the_gate_names_are_unique_in_the_all_pipeline() -> None:
    names = [name for name, _ in _build_gates("all")]
    assert len(names) == len(set(names)), "a gate name is duplicated"
    for name in RUNNERS:
        assert names.count(name) == 1, f"{name} is registered more than once"


def test_the_expected_gate_counts_agree_with_the_live_counts() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from pre_flight_check import EXPECTED_GATE_COUNTS  # noqa: PLC0415

    for pipeline in ("all", "launch", "cfm", "frontier", "chemprop_v2_v3"):
        assert len(_build_gates(pipeline)) == EXPECTED_GATE_COUNTS[pipeline], (
            f"pipeline {pipeline!r}: live count "
            f"{len(_build_gates(pipeline))} != EXPECTED_GATE_COUNTS "
            f"{EXPECTED_GATE_COUNTS[pipeline]}"
        )


def test_the_audit_doc_header_agrees_with_the_code_constant() -> None:
    """The count lives in 4 places (catalog §6.236); this asserts the 2nd
    (the SSOT doc header) so a partial bump leaves a RED test behind."""
    doc = REPO / "TODO" / "completed" / "321d_preflight_gate_audit.md"
    assert doc.is_file(), f"audit doc missing: {doc}"
    text = doc.read_text(encoding="utf-8")
    match = re.search(
        r"### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates", text
    )
    assert match is not None, "audit doc §2.1 header not found"
    sys.path.insert(0, str(SCRIPT_DIR))
    from pre_flight_check import EXPECTED_GATE_COUNTS  # noqa: PLC0415

    assert int(match.group(1)) == EXPECTED_GATE_COUNTS["all"], (
        f"audit doc §2.1 says {match.group(1)} but the code constant says "
        f"{EXPECTED_GATE_COUNTS['all']} — bump BOTH in the same edit."
    )
