#!/usr/bin/env python3
"""``cfm_todo.json`` is a plan only if every defect has a plan attached to it.

The file was restructured on 2026-10-25 from a gate table into
``diagnosis -> modules/tasks -> acceptance``.  That structure is the user's
requirement ("每一个我提出来的问题都要有对应的模型解构改进的计划"), and a
requirement stated only in prose decays the first time an entry is edited by
hand.  These tests are the decay guard:

* every diagnosis is cited by at least one module or task;
* every citation resolves to something that exists;
* every blocked_by resolves and the graph is acyclic;
* every diagnosis names an artifact on disk, unless it declares the artifact
  ABSENT — a defect with neither a number nor a planned measurement is an
  opinion;
* every non-PASS gate names its next step;
* every acceptance row names a module that exists.

Adding a diagnosis without a plan is a red test, which is the point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "cfm_todo.json"

MODULE_REF = re.compile(r"MOD\d+")
TASK_STATUSES = {"PLANNED", "IN_PROGRESS", "DONE", "BLOCKED", "DROPPED"}


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


PLAN_DOC = _plan()
DIAGNOSIS = {d["id"]: d for d in PLAN_DOC["diagnosis"]}
MODULES = {m["id"]: m for m in PLAN_DOC["modules"]}
TASKS = {t["id"]: t for t in PLAN_DOC["tasks"]}
GATES = {g["id"]: g for g in PLAN_DOC["module_gates"]}
PLAN_IDS = set(MODULES) | set(TASKS)


def test_ids_are_unique():
    for name, seq in (
        ("diagnosis", PLAN_DOC["diagnosis"]),
        ("modules", PLAN_DOC["modules"]),
        ("tasks", PLAN_DOC["tasks"]),
        ("module_gates", PLAN_DOC["module_gates"]),
    ):
        ids = [x["id"] for x in seq]
        assert len(ids) == len(set(ids)), f"duplicate {name} ids: {ids}"


def test_every_diagnosis_has_a_plan():
    """The user's requirement, as an assertion."""
    cited = set()
    for entry in list(MODULES.values()) + list(TASKS.values()):
        cited |= set(entry.get("addresses", []))
    unplanned = sorted(set(DIAGNOSIS) - cited)
    assert not unplanned, (
        f"{unplanned} name a defect and no module or task addresses it"
    )


def test_every_reference_resolves():
    dangling = []
    for d in DIAGNOSIS.values():
        dangling += [(d["id"], "addressed_by", a) for a in d["addressed_by"] if a not in PLAN_IDS]
    for entry in list(MODULES.values()) + list(TASKS.values()):
        dangling += [
            (entry["id"], "addresses", a)
            for a in entry.get("addresses", [])
            if a not in DIAGNOSIS
        ]
    for t in TASKS.values():
        dangling += [
            (t["id"], "blocked_by", b) for b in t.get("blocked_by", []) if b not in PLAN_IDS
        ]
    assert not dangling, f"references that resolve to nothing: {dangling}"


def test_blocked_by_is_acyclic():
    deps = {tid: set(t.get("blocked_by", [])) for tid, t in TASKS.items()}
    done: set[str] = set()

    def cyclic(node: str, path: tuple[str, ...]) -> tuple[str, ...] | None:
        if node in path:
            return path[path.index(node) :] + (node,)
        if node in done:
            return None
        for dep in deps.get(node, ()):
            found = cyclic(dep, path + (node,))
            if found:
                return found
        done.add(node)
        return None

    for tid in deps:
        found = cyclic(tid, ())
        assert found is None, f"blocked_by cycle: {' -> '.join(found)}"


def test_every_diagnosis_names_an_artifact():
    """A number with no path on disk is an anecdote (AGENTS.md §5.2)."""
    problems = []
    for d in DIAGNOSIS.values():
        artifact = d.get("artifact")
        assert artifact, f"{d['id']} names no artifact"
        if d.get("artifact_status") == "ABSENT":
            assert d["addressed_by"], f"{d['id']} declares its artifact ABSENT and plans nothing"
            continue
        if not (REPO_ROOT / artifact).exists():
            problems.append(f"{d['id']}: {artifact}")
    assert not problems, f"diagnosis artifacts not on disk: {problems}"


def test_every_failing_gate_names_its_next_step():
    for gate in GATES.values():
        assert gate["status"] in {"PASS", "FAIL", "BLOCKED", "SKIP"}, gate
        if gate["status"] != "PASS":
            assert gate.get("next"), f"{gate['id']} is {gate['status']} with no next step"


def test_every_task_is_actionable():
    for t in TASKS.values():
        assert t.get("title"), f"{t['id']} has no title"
        assert t.get("what"), f"{t['id']} has no what"
        assert t.get("done_when"), f"{t['id']} has no done_when"
        assert t.get("addresses"), f"{t['id']} addresses no diagnosis"
        assert t["status"] in TASK_STATUSES, (t["id"], t["status"])
        assert isinstance(t.get("priority"), int), t["id"]


def test_every_acceptance_row_names_a_real_module():
    rows = PLAN_DOC["acceptance"]["rows"]
    assert rows, "the acceptance table is empty"
    for row in rows:
        named = MODULE_REF.findall(row["module"])
        assert named, f"{row['metric']} names no module"
        unknown = [m for m in named if m not in MODULES]
        assert not unknown, f"{row['metric']} names {unknown}, which do not exist"
