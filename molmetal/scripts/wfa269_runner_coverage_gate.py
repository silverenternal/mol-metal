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
"""Wf-A269 — the RUNNER-COVERAGE gate: make the unexecuted test class VISIBLE.

================================================================================
WHY THIS FILE EXISTS
================================================================================
The repo's test population is large and its EXECUTED population is tiny.
MEASURED 2026-11-09 on the live tree (this module's own ``verify()``):

    population  (molmetal/tests/**/test_*.py) ....... 1367
    on an executing runner's pytest argv ............ 31
    on NO executing runner ........................... 1336  (97.73 %)

A test file that no runner executes is **invisible**: it is never collected,
never executed, and its own assertions are never evaluated.  A test written
to pin a number that is wrong (a "born-false pin") therefore stays wrong
forever, and nothing reports it.  The repo's own catalogue records this
class twice — AGENTS.md §6.269.29 ("a guard that no runner executes") and
§6.269.32 ("a pin that was born false and was never run") — and §6.269.32
leaves the closure **OPEN**: *"there is still no gate that runs the repo's
test suite, so the next born-false pin will also stay red undetected."*

Running 1367 files is not a remedy (it burns VRAM and hours).  This module
is the minimal remedy that makes the blind spot **visible** instead of
**silent**, without executing anything.

================================================================================
THE AXIS (stated once, used everywhere)
================================================================================
:data:`AXIS`

A path quoted in a COMMENT, a DOCSTRING, a YAML ``name:`` field or a shell
variable is a **MENTATION** and does NOT count.  This module re-implements
the extraction rather than importing the repo's existing
``molmetal/tests/test_wfa269_wave28_published_number_guards.py::_ci_pytest_argvs``
so that the gate does not depend on a test file (which is itself only one
more unexecuted file until a runner names it).

================================================================================
THE BASELINE MECHANISM (the repo's convention: BASELINE dict + BASELINE_SIZE)
================================================================================
The baseline is the **declared UNEXECUTED set**: every test file that is on
no executing runner's argv, mapped to the reason it is tolerated.  It lives
in a JSON sidecar (:data:`BASELINE_SIDECAR`) because it is 1336 entries of
DATA, not source; :func:`load_baseline` returns it as a ``dict`` and the
module exposes it as :data:`BASELINE` / :data:`BASELINE_SIZE` so the
convention is preserved for readers and for consumers.

The baseline is compared **BOTH ways** (the same two-sided pattern the
repo's path-reference guard uses, AGENTS.md §6.269.35.7):

* ``unexecuted - BASELINE`` must be EMPTY — a test file that no runner
  names and that is not declared is a **FAIL** (this is the load-bearing
  assertion: it fires when somebody adds a test file and forgets to wire
  it);
* ``BASELINE - unexecuted`` must be EMPTY — a declared file that is now
  wired (or deleted) means the baseline must be **tightened**, otherwise
  the exemption has silently become a hole;
* ``len(unexecuted) == BASELINE_SIZE`` — the recorded measurement is a live
  reading, not a historical record.

HONEST LIMIT (per AGENTS.md §5.2 / §6.269.6): the baseline's per-entry
reason is **uniform** — every entry reads ``NOT_ON_ANY_EXECUTING_RUNNER_ARGV``.
This baseline is the blind spot made EXPLICIT, not an adjudication of 1336
individual defects.  It does not claim those 1336 files are individually
justified; it claims they are *known*, *named* and *pinned*, so that the
set cannot change without a decision.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CI_WORKFLOW = ".github/workflows/ci.yml"
PREFLIGHT = "molmetal/scripts/pre_flight_check.py"
BASELINE_SIDECAR = "molmetal/scripts/wfa269_runner_coverage_baseline.json"

#: The pre-flight pipelines an automated runner can select.  ``all`` is
#: what the pre-commit hook runs (``.git/hooks/pre-commit``) and what the
#: live launchers under ``scripts/`` run; ``launch`` is the launcher-only
#: variant.  ``frontier`` and ``cfm`` are NEVER run by an automated runner
#: (AGENTS.md §6.269.29.1), so they are excluded on purpose.
EXECUTING_PREFLIGHT_PIPELINES = ("all", "launch")

#: The uniform reason recorded for every baseline entry.
BASELINE_REASON = "NOT_ON_ANY_EXECUTING_RUNNER_ARGV"

AXIS = (
    "A test file counts as EXECUTED iff its repo-relative path appears as a "
    "literal token in a pytest argv that an EXECUTING runner passes to a "
    "pytest process.  The executing runners are: (R1) a CI job's `run:` "
    "block in .github/workflows/ci.yml; (R2) a pre-flight gate registered "
    "in the `all` or `launch` pipeline of molmetal/scripts/pre_flight_check.py. "
    "A path quoted in a COMMENT, a DOCSTRING, a YAML `name:` field or a shell "
    "variable is a MENTION and does NOT count.  The Makefile's `test:` target "
    "names the whole molmetal/tests DIRECTORY and would cover the whole "
    "population, but NO CI job and NO runner calls `make test`, so it is NOT "
    "an executing runner and is excluded."
)

EXIT_PASS = 0
EXIT_FAIL = 1


# ---------------------------------------------------------------------------
# (i) the population
# ---------------------------------------------------------------------------
def test_population(repo_root: Path = REPO_ROOT) -> list[str]:
    """Every test file under ``molmetal/tests/`` — the population, sorted.

    RECURSIVE on purpose: the population is
    ``molmetal/tests/**/test_*.py``, so a test file in a sub-directory
    counts.  A flat glob would under-count the population and make the
    coverage denominator wrong.
    """
    tests_root = repo_root / "molmetal" / "tests"
    if not tests_root.is_dir():
        return []
    return sorted(
        str(p.relative_to(repo_root))
        for p in tests_root.rglob("test_*.py")
        if p.is_file()
    )


# ---------------------------------------------------------------------------
# (ii) the executed set — CI (R1)
# ---------------------------------------------------------------------------
def ci_pytest_argvs(text: str) -> list[list[str]]:
    """The pytest ARGVs of the CI workflow, gathered across line continuations.

    A line whose first non-space character is ``#`` is a COMMENT and is
    skipped outright; a comment line inside a continuation is skipped too.
    This is the "a mention is not an invocation" rule (AGENTS.md §6.269.28
    / the §5.9 single-venv gate's own lesson), re-implemented here so that
    the gate does not depend on a test file.

    CONTINUATION CORRECTNESS (MEASURED 2026-11-09): the logical line is
    ``line`` plus every following line while the LAST LINE ALREADY GATHERED
    ends with a backslash.  Gathering "the following line, if the
    following line ends with a backslash" instead drops the FINAL line of
    every command whose last line carries no trailing backslash — a silent
    token drop on the very axis this gate measures.  The repo's existing
    extractor (``molmetal/tests/test_wfa269_wave28_published_number_guards.py``
    ::``_ci_pytest_argvs``) carries the second, weaker form; on today's
    ``ci.yml`` both extractors return the SAME test-path set (every pytest
    command there ends in ``| tee <log>``), so the published numbers are
    unaffected — but the weaker form is latent.
    """
    argvs: list[list[str]] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if not ("-m pytest" in stripped or stripped.endswith(" pytest")):
            i += 1
            continue
        block = [stripped]
        j = i + 1
        while block[-1].rstrip().endswith("\\") and j < n:
            cont = lines[j].strip()
            j += 1
            if not cont or cont.startswith("#"):
                continue
            block.append(cont)
        tokens: list[str] = []
        for tok in " ".join(block).replace("\\", " ").split():
            tok = tok.strip("'\"")
            if tok:
                tokens.append(tok)
        argvs.append(tokens)
        i = j if j > i else i + 1
    return argvs


def ci_test_paths(text: str) -> list[str]:
    """The test paths named on a CI job's pytest argv."""
    out: set[str] = set()
    for argv in ci_pytest_argvs(text):
        for tok in argv:
            if tok.startswith("molmetal/tests/") and tok.endswith(".py"):
                out.add(tok)
    return sorted(out)


# ---------------------------------------------------------------------------
# (ii) the executed set — the pre-flight (R2)
# ---------------------------------------------------------------------------
_DUNDER_DIRS = {
    "TESTS_DIR": "molmetal/tests",
    "MOLMETAL_ROOT": "molmetal",
    "REPO_ROOT": "",
    "SCRIPTS_DIR": "molmetal/scripts",
    "DATA_DIR": "molmetal/data",
    "DOT_DATA_DIR": "molmetal/data",
}


def _preflight_assignments(tree: ast.AST) -> dict[str, ast.AST]:
    """Every ``Name = <expr>`` assignment in the module (any scope)."""
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.setdefault(node.target.id, node.value)
    return out


def _resolve_path(node: ast.AST | None, assigns: dict[str, ast.AST], depth: int = 0) -> str | None:
    """Resolve a path expression to a repo-relative path string.

    Handles ``TESTS_DIR / "x.py"``, a bare string, and a Name bound (at any
    scope) to either of those — the pre-flight's own test lists are written
    as ``TESTS_DIR / "..."`` lists plus two intermediate variables.
    """
    if node is None or depth > 12:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_path(node.left, assigns, depth + 1)
        right = _resolve_path(node.right, assigns, depth + 1)
        if left is None or right is None:
            return None
        return f"{left}/{right}" if left.endswith("/") else f"{left}/{right}"
    if isinstance(node, ast.Name):
        if node.id in _DUNDER_DIRS:
            return _DUNDER_DIRS[node.id]
        if node.id in assigns:
            return _resolve_path(assigns[node.id], assigns, depth + 1)
        return None
    if isinstance(node, ast.Call):
        for arg in node.args:
            got = _resolve_path(arg, assigns, depth + 1)
            if got is not None:
                return got
        return None
    return None


def _resolve_path_list(node: ast.AST | None, assigns: dict[str, ast.AST], depth: int = 0) -> list[str]:
    if node is None or depth > 12:
        return []
    if isinstance(node, ast.Name):
        if node.id in assigns:
            return _resolve_path_list(assigns[node.id], assigns, depth + 1)
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for elt in node.elts:
            got = _resolve_path(elt, assigns, depth + 1)
            if got is not None:
                out.append(got)
        return out
    got = _resolve_path(node, assigns, depth + 1)
    return [got] if got is not None else []


def _pipeline_constants(test: ast.AST) -> set[str]:
    """The pipeline names a guard condition selects.

    Handles both shapes the pre-flight uses::

        if pipeline == "all": ...
        if pipeline in ("cfm", "all"): ...

    Returns an empty set for any other condition.
    """
    if not isinstance(test, ast.Compare):
        return set()
    if not (isinstance(test.left, ast.Name) and test.left.id == "pipeline"):
        return set()
    out: set[str] = set()
    for cmp in test.comparators:
        if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
            out.add(cmp.value)
        elif isinstance(cmp, (ast.Tuple, ast.List, ast.Set)):
            for elt in cmp.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _enclosing_pipeline(tree: ast.AST) -> dict[int, set[str]]:
    """Map ``id(node)`` -> the set of pipelines whose guard block contains it.

    A node may sit under several nested pipeline guards (``if pipeline in
    ("cfm", "all")`` inside ``if pipeline == "all"``), so the value is a
    SET, not a single name — a gate registered under ``("cfm", "all")``
    runs in BOTH pipelines and must be counted in both.
    """
    out: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        pipelines = _pipeline_constants(node.test)
        if not pipelines:
            continue
        for sub in ast.walk(node):
            out.setdefault(id(sub), set()).update(pipelines)
    return out


def preflight_test_paths(text: str, pipelines: tuple[str, ...] = EXECUTING_PREFLIGHT_PIPELINES) -> dict[str, list[str]]:
    """The test paths each pre-flight pipeline executes, via its pytest gates.

    Returns ``{pipeline: [repo-relative test paths]}`` for the requested
    pipelines only.  Two gate shapes are recognised, both of which really
    pass a pytest argv to a pytest process:

    * ``gate_pytest(name, priority, pipeline, [paths])`` — 4th positional
      argument is the path list;
    * ``gate_r20_phase0_loaders_complete(..., loader_tests)`` — 6th
      positional argument is the path list.
    """
    tree = ast.parse(text)
    assigns = _preflight_assignments(tree)
    enclosing = _enclosing_pipeline(tree)
    per_pipeline: dict[str, set[str]] = {p: set() for p in pipelines}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        else:
            continue
        if fname == "gate_pytest":
            paths = _resolve_path_list(node.args[3] if len(node.args) > 3 else None, assigns)
        elif fname == "gate_r20_phase0_loaders_complete":
            paths = _resolve_path_list(node.args[5] if len(node.args) > 5 else None, assigns)
        else:
            continue
        node_pipelines = enclosing.get(id(node), set())
        for raw in paths:
            rel = raw if raw.startswith("molmetal/") else f"molmetal/tests/{raw}"
            for pipeline in node_pipelines & per_pipeline.keys():
                per_pipeline[pipeline].add(rel)
    return {p: sorted(v) for p, v in per_pipeline.items()}


# ---------------------------------------------------------------------------
# (ii) the executed set — the union
# ---------------------------------------------------------------------------
def executed_test_paths(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """The executed set, per executing runner surface.

    Returns ``{surface: [repo-relative test paths]}``.  The surfaces are
    ``"ci"``, ``"preflight:<pipeline>"`` and ``"makefile_test"``.  The
    Makefile surface is reported for TRANSPARENCY only: its ``test:`` target
    names the whole ``molmetal/tests`` DIRECTORY, which would cover the
    entire population, but no CI job and no runner calls ``make test`` — so
    it is NOT an executing runner and is excluded from the coverage
    computation.
    """
    surfaces: dict[str, list[str]] = {}

    ci_file = repo_root / CI_WORKFLOW
    if ci_file.is_file():
        surfaces["ci"] = ci_test_paths(ci_file.read_text())

    pf_file = repo_root / PREFLIGHT
    if pf_file.is_file():
        per_pipeline = preflight_test_paths(pf_file.read_text())
        for pipeline, paths in per_pipeline.items():
            surfaces[f"preflight:{pipeline}"] = paths

    # Transparency only — NOT counted as executing (see the docstring).
    mk = repo_root / "Makefile"
    if mk.is_file():
        mk_paths = sorted(
            m for m in re.findall(r"molmetal/tests(?:/[A-Za-z0-9_.*/-]*)?", mk.read_text())
        )
        surfaces["makefile_test"] = mk_paths
    return surfaces


def executed_union(surfaces: dict[str, list[str]]) -> set[str]:
    """The union of the EXECUTING surfaces (the Makefile is excluded)."""
    out: set[str] = set()
    for surface, paths in surfaces.items():
        if surface == "makefile_test":
            continue
        out.update(paths)
    return out


# ---------------------------------------------------------------------------
# the baseline
# ---------------------------------------------------------------------------
def _read_sidecar(repo_root: Path = REPO_ROOT, sidecar: Path | None = None) -> dict[str, Any]:
    """Read and shape-check the baseline sidecar.  Raises on anything wrong.

    A MISSING or MALFORMED sidecar is an ERROR, not an empty baseline: an
    empty baseline would make every test file a violation, and a missing
    one would silently disable the gate (AGENTS.md §6.269.17's dead-gate
    class).  The caller decides; this function raises.
    """
    path = sidecar if sidecar is not None else repo_root / BASELINE_SIDECAR
    if not path.is_file():
        raise FileNotFoundError(f"the runner-coverage baseline sidecar is missing: {path}")
    body = json.loads(path.read_text())
    if not isinstance(body, dict):
        raise ValueError(f"the baseline sidecar is not a JSON object: {path}")
    raw = body.get("unexecuted")
    if not isinstance(raw, dict):
        raise ValueError(f"the baseline sidecar has no `unexecuted` dict: {path}")
    if not isinstance(body.get("baseline_size"), int):
        raise ValueError(f"the baseline sidecar has no integer `baseline_size`: {path}")
    return body


def load_baseline(
    repo_root: Path = REPO_ROOT,
    sidecar: Path | None = None,
) -> tuple[dict[str, str], int]:
    """Load the declared-unexecuted baseline.

    Returns ``(baseline, recorded_size)``.  ``baseline`` maps a
    repo-relative test path to the reason it is declared unexecuted.
    ``recorded_size`` is the size the sidecar itself records — the pin.
    """
    body = _read_sidecar(repo_root, sidecar)
    raw = body["unexecuted"]
    return {str(k): str(v) for k, v in raw.items()}, int(body["baseline_size"])


def _default_baseline() -> tuple[dict[str, str], int]:
    try:
        return load_baseline()
    except (FileNotFoundError, ValueError):
        return {}, -1


#: The declared-unexecuted baseline (path -> reason) and its recorded size.
#: Loaded at import time so that a consumer can read them without calling
#: the loader; a sidecar that cannot be read yields an EMPTY baseline and a
#: recorded size of ``-1``, which :func:`verify` reports as a FAIL.
BASELINE, BASELINE_SIZE = _default_baseline()

#: The declared FLOOR on the executed population.  The executed set must
#: never shrink below this without a deliberate, recorded decision — the
#: floor is what catches "a CI job quietly dropped a test file".
MIN_EXECUTED_FLOOR = 31


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def verify(repo_root: Path = REPO_ROOT, floor: int | None = None) -> dict[str, Any]:
    """Compute the verdict.  Pure: reads only, never writes, never executes.

    The returned dict always carries the measurement AND the failures, so a
    caller can render it without recomputing anything:

    ``population_n``   — the size of the test population.
    ``executed_n``     — the size of the executed set (the positive side).
    ``unexecuted_n``   — the size of the blind spot.
    ``coverage_pct``   — ``100 * executed_n / population_n``.
    ``new_unexecuted`` — on no runner and NOT declared  => FAIL.
    ``stale_baseline`` — declared but now wired or gone   => FAIL.
    ``dead_invocations`` — named by a runner but absent on disk => FAIL.
    ``floor`` / ``floor_ok`` — the declared floor.
    ``verdict``        — ``"PASS"`` or ``"FAIL"``.

    ``floor`` overrides :data:`MIN_EXECUTED_FLOOR`; ``None`` means "use the
    module constant".  :func:`main` always uses the constant — the
    override exists so that a synthetic repository in a test can be
    measured without inheriting the live floor.
    """
    population = test_population(repo_root)
    pop_set = set(population)

    surfaces = executed_test_paths(repo_root)
    executed = executed_union(surfaces)
    unexecuted = pop_set - executed

    try:
        baseline, recorded_size = load_baseline(repo_root)
        recorded_executed = sorted(_read_sidecar(repo_root).get("executed", []))
    except (FileNotFoundError, ValueError) as exc:
        baseline, recorded_size = {}, -1
        recorded_executed = []
        baseline_error = str(exc)
    else:
        baseline_error = ""

    new_unexecuted = sorted(unexecuted - set(baseline))
    stale_baseline = sorted(set(baseline) - unexecuted)
    dead_invocations = sorted(
        p for p in executed if not (repo_root / p).is_file()
    )
    executed_changed = sorted(
        set(recorded_executed) ^ set(executed)
    ) if not baseline_error else []

    failures: list[str] = []
    if baseline_error:
        failures.append(f"the baseline sidecar could not be read: {baseline_error}")
    if recorded_size != len(baseline):
        failures.append(
            f"the baseline sidecar records {recorded_size} entries but carries "
            f"{len(baseline)} — the baseline was edited without updating the pin"
        )
    if new_unexecuted:
        failures.append(
            f"{len(new_unexecuted)} test file(s) are on NO executing runner's "
            f"pytest argv and are NOT in the declared baseline — a new test "
            f"file was added and never wired: {new_unexecuted}"
        )
    if stale_baseline:
        failures.append(
            f"{len(stale_baseline)} declared baseline entr(ies) are no longer "
            f"unexecuted (the file was wired or deleted) — the baseline must be "
            f"tightened: {stale_baseline}"
        )
    if executed_changed:
        failures.append(
            f"the sidecar's recorded EXECUTED set differs from the live "
            f"measurement in {len(executed_changed)} path(s) — a runner's argv "
            f"changed and the sidecar was not refreshed: {executed_changed}"
        )
    if dead_invocations:
        failures.append(
            f"{len(dead_invocations)} path(s) named by an executing runner do "
            f"NOT exist on disk — pytest exits 4 (usage error) on such a path, "
            f"so the step can never pass: {dead_invocations}"
        )
    effective_floor = MIN_EXECUTED_FLOOR if floor is None else int(floor)
    floor_ok = len(executed) >= effective_floor
    if not floor_ok:
        failures.append(
            f"the executed population is {len(executed)} but the declared floor "
            f"is {effective_floor} — a runner stopped naming test files"
        )
    if population and not executed:
        failures.append(
            "NO test file is on any executing runner's pytest argv — the axis "
            "extraction is broken, not the repository"
        )

    return {
        "axis": AXIS,
        "population_n": len(population),
        "executed_n": len(executed),
        "unexecuted_n": len(unexecuted),
        "coverage_pct": (100.0 * len(executed) / len(population)) if population else 0.0,
        "executed": sorted(executed),
        "surfaces": {k: len(v) for k, v in surfaces.items()},
        "baseline_size": len(baseline),
        "baseline_recorded_size": recorded_size,
        "baseline_error": baseline_error,
        "new_unexecuted": new_unexecuted,
        "stale_baseline": stale_baseline,
        "executed_changed": executed_changed,
        "dead_invocations": dead_invocations,
        "floor": effective_floor,
        "floor_ok": floor_ok,
        "failures": failures,
        "verdict": "FAIL" if failures else "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    """Render the verdict and return the exit code (0 PASS / 1 FAIL)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    repo_root = REPO_ROOT
    if "--repo-root" in argv:
        repo_root = Path(argv[argv.index("--repo-root") + 1]).resolve()
    if "--json" in argv:
        print(json.dumps(verify(repo_root), indent=2, sort_keys=True))
        return EXIT_PASS if verify(repo_root)["verdict"] == "PASS" else EXIT_FAIL

    out = verify(repo_root)
    print(f"runner coverage ({out['verdict']})")
    print(f"  axis              : executing-runner pytest argv")
    print(f"  population        : {out['population_n']}")
    print(f"  executed          : {out['executed_n']}  (floor {out['floor']}, "
          f"floor_ok {out['floor_ok']})")
    print(f"  unexecuted        : {out['unexecuted_n']}  "
          f"({100.0 * out['unexecuted_n'] / max(1, out['population_n']):.2f} %)")
    print(f"  coverage          : {out['coverage_pct']:.2f} %")
    print(f"  surfaces          : {out['surfaces']}")
    print(f"  baseline          : {out['baseline_size']} entries "
          f"(recorded {out['baseline_recorded_size']})")
    if out["failures"]:
        print("FAIL:")
        for f in out["failures"]:
            print(f"  - {f}")
        return EXIT_FAIL
    print("PASS: every test file that no runner names is declared in the "
          "baseline, and every path a runner names exists on disk.")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
