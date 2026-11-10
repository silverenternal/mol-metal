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

"""Wf-A269 wave-33 — the enforcement stack's TRACKEDNESS must itself be a GATE.

==============================================================================
WHAT THIS PINS
==============================================================================
The §5.11 launch-enforcement stack (the yaml-only gate scripts, the launch
provenance recorder, the shared verdict consumer, the smoke-budget module the
pre-flight itself imports, the two path-reference guard runners) was NOT in
git's index.  MEASURED 2026-11-09 with ``git ls-files --error-unmatch``:
**8 of the 10 declared members are not in the index** and all 8 are NEVER
COMMITTED (in neither the index nor HEAD) — i.e. they exist only on this
machine.  A gate that is not tracked is not "a gate that is missing": it is
a gate that is PRESENT in the working tree (so every local run sees it
green) and ABSENT from the repository (so every clone fails closed with
"script missing" — or does not fail at all, because the gate that would
have noticed is itself untracked).

The remedy lives in ``molmetal/scripts/pre_flight_check.py``, which IS
tracked at HEAD (``git ls-files --error-unmatch`` rc 0 and
``git cat-file -e HEAD:molmetal/scripts/pre_flight_check.py`` rc 0), so the
detector survives the clean checkout it is detecting.

This module pins FOUR properties of that remedy, each independently:

  1. the HOST is tracked at HEAD (otherwise the detector is itself
     untracked and the whole exercise is circular);
  2. the MANIFEST is COMPLETE — every path the pre-flight resolves (gate
     scripts, pytest files, import-time module dependencies) is declared,
     so a NEW untracked dependency cannot slip in unnoticed;
  3. the VERDICT is fail-CLOSED — git unprobeable ⇒ FAIL, and the verdict
     agrees with an INDEPENDENT read of git's index;
  4. the pre-flight still BOOTS when its untracked import-time modules are
     absent, i.e. on a clean checkout.

==============================================================================
WHY THE VERDICT TEST IS RELATIVE, NOT A FROZEN COUNT
==============================================================================
``test_gate_verdict_matches_an_independent_read_of_gits_index`` does NOT
assert "the gate FAILs" or "8 offenders" — both would be BORN-FALSE the
moment the user commits the stack (AGENTS.md §6.269.32: "not a guard that
cannot FAIL, but a guard that cannot PASS"; the repair is a RELATIVE /
INVARIANT pin, never a freeze).  It asserts the RELATION: the gate FAILs
iff some declared member is missing from git's index, and it names every
such member in its evidence.  That stays true before AND after the commit.

CPU-only. No GPU. No network.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRE_FLIGHT_REL = "molmetal/scripts/pre_flight_check.py"
PRE_FLIGHT = REPO_ROOT / PRE_FLIGHT_REL

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_check as pfc  # noqa: E402


GATE_NAME = "p3_enforcement_stack_trackedness"


# =============================================================================
# git helpers — the test's OWN reads, independent of the module under test.
# =============================================================================
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _index_paths() -> Set[str]:
    out = _git("ls-files", "-z").stdout
    return {p for p in out.split("\0") if p}


def _head_paths() -> Set[str]:
    out = _git("ls-tree", "-r", "-z", "HEAD", "--name-only").stdout
    return {p for p in out.split("\0") if p}


# =============================================================================
# The INDEPENDENT enumeration of the pre-flight's resolved surface.
#
# This is deliberately a SEPARATE implementation from
# ``pre_flight_check._resolved_gate_surface`` (which is a runtime
# heuristic): it is an AST walk of the module source, so a mutation of the
# manifest in a COPY of the module can be detected by pointing this
# enumeration at that copy.
# =============================================================================
#: The repo-relative prefix each of the pre-flight's ROOT constants denotes.
#: The test asserts (see ``test_the_root_constant_prefixes_are_real``) that
#: the module really declares a constant with each of these names, so a
#: rename cannot silently make this enumeration resolve nothing.
_ROOT_PREFIXES: Dict[str, str] = {
    "REPO_ROOT": "",
    "MOLMETAL_ROOT": "molmetal",
    "SCRIPT_DIR": "molmetal/scripts",
    "SCRIPTS_DIR": "scripts",
    "TESTS_DIR": "molmetal/tests",
    "LAM_CHEM_DIR": "molmetal/molmetal_lam/lam_chem",
    "CONFIGS_DIR": "configs",
}

_SCRIPT_PARAM_NAMES = ("script_path", "script", "gate_script", "runner_path")


def _resolve_expr(node: ast.AST, env: Dict[str, str]) -> Optional[str]:
    """Resolve a path expression to a repo-relative POSIX path, or None."""

    def ev(n: ast.AST) -> Optional[str]:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name):
            return env.get(n.id)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            left = ev(n.left)
            right = ev(n.right)
            if left is None or right is None:
                return None
            return f"{left}/{right}" if left else right
        if isinstance(n, ast.Call):
            if (
                isinstance(n.func, ast.Name)
                and n.func.id == "Path"
                and n.args
            ):
                return ev(n.args[0])
            return None
        return None

    raw = ev(node)
    if raw is None:
        return None
    return os.path.normpath(raw).replace(os.sep, "/")


def _module_level_imports(tree: ast.Module) -> List[Tuple[str, bool]]:
    """(module_name, guarded) for every module-level molmetal import.

    ``guarded`` is True when the import sits inside a module-level
    ``try:`` whose handlers catch ImportError/Exception/BaseException — an
    import that degrades gracefully instead of killing the import.

    Function- and class-level imports are NOT module-level and are skipped.
    """

    out: List[Tuple[str, bool]] = []

    def walk(stmts: List[ast.stmt], guarded: bool) -> None:
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, ast.Try):
                handlers: List[str] = []
                for h in st.handlers:
                    exc = h.type
                    if exc is None:
                        handlers.append("BaseException")
                    elif isinstance(exc, ast.Name):
                        handlers.append(exc.id)
                    elif isinstance(exc, ast.Tuple):
                        handlers.extend(
                            e.id for e in exc.elts if isinstance(e, ast.Name)
                        )
                catches = any(
                    k in handlers
                    for k in ("ImportError", "Exception", "BaseException")
                )
                walk(st.body, guarded or catches)
                continue
            if isinstance(st, ast.ImportFrom):
                if st.level == 0 and st.module and st.module.startswith("molmetal"):
                    for alias in st.names:
                        out.append((f"{st.module}.{alias.name}", guarded))
            elif isinstance(st, ast.Import):
                for alias in st.names:
                    if alias.name.startswith("molmetal"):
                        out.append((alias.name, guarded))

    walk(tree.body, False)
    return out


def _import_to_repo_path(dotted: str) -> Optional[str]:
    """Map a dotted module name to the repo-relative file that provides it.

    ``from pkg import NAME`` puts the imported NAME (often a class, not a
    module) into the dotted string, so BOTH candidates are tried and the one
    that EXISTS on disk wins.
    """
    parts = dotted.split(".")
    candidates = [
        "/".join(parts[:-1]) + ".py" if len(parts) > 1 else "",
        "/".join(parts) + ".py",
        "/".join(parts) + "/__init__.py",
    ]
    for cand in candidates:
        if cand and (REPO_ROOT / cand).is_file():
            return cand
    return None


def _enumerate_resolved_surface(
    source_path: Path,
) -> Dict[str, Set[str]]:
    """The pre-flight's resolved surface, by AST analysis of ``source_path``.

    Returns {"gate_scripts": ..., "pytest_files": ..., "imports": ...} with
    repo-relative POSIX paths.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    env: Dict[str, str] = dict(_ROOT_PREFIXES)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            resolved = _resolve_expr(node.value, env)
            if resolved:
                env[node.targets[0].id] = resolved

    gate_fns: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("gate_"):
            gate_fns[node.name] = [a.arg for a in node.args.args]

    gate_scripts: Set[str] = set()
    pytest_files: Set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if fname is None or fname not in gate_fns:
            continue
        params = gate_fns[fname]
        if fname == "gate_pytest":
            pidx = next(
                (
                    i
                    for i, p in enumerate(params)
                    if "file" in p or "test" in p or "path" in p
                ),
                None,
            )
            if pidx is None or pidx >= len(node.args):
                continue
            arg = node.args[pidx]
            items = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
            for item in items:
                resolved = _resolve_expr(item, env)
                if resolved:
                    pytest_files.add(resolved)
            continue
        sidx = next(
            (i for i, p in enumerate(params) if p in _SCRIPT_PARAM_NAMES), None
        )
        if sidx is None or sidx >= len(node.args):
            continue
        resolved = _resolve_expr(node.args[sidx], env)
        if resolved:
            gate_scripts.add(resolved)

    imports: Set[str] = set()
    for dotted, _guarded in _module_level_imports(tree):
        mapped = _import_to_repo_path(dotted)
        if mapped:
            imports.add(mapped)

    return {
        "gate_scripts": gate_scripts,
        "pytest_files": pytest_files,
        "imports": imports,
    }


def _manifest_gaps(module: Any, source_path: Path) -> List[str]:
    """Resolved paths the module's manifest does NOT declare."""
    surface = _enumerate_resolved_surface(source_path)
    resolved = (
        surface["gate_scripts"] | surface["pytest_files"] | surface["imports"]
    )
    declared = set(module.PREFLIGHT_RESOLVED_SURFACE_PATHS)
    return sorted(resolved - declared)


# =============================================================================
# 1. The HOST is tracked at HEAD.
# =============================================================================
def test_the_host_file_is_tracked_at_head() -> None:
    """The detector must live in a file a clean checkout WILL have."""
    proc = _git("ls-files", "--error-unmatch", PRE_FLIGHT_REL)
    assert proc.returncode == 0, (
        f"{PRE_FLIGHT_REL} is not in git's index — the trackedness gate "
        f"itself would be untracked, so a clean checkout could not run it "
        f"(git ls-files --error-unmatch rc={proc.returncode})"
    )
    proc = _git("cat-file", "-e", f"HEAD:{PRE_FLIGHT_REL}")
    assert proc.returncode == 0, (
        f"{PRE_FLIGHT_REL} is not in HEAD's tree either — the detector must "
        f"be COMMITTED, not merely staged"
    )


def test_the_root_constant_prefixes_are_real() -> None:
    """The independent enumeration's ROOT map must match the module's names.

    If the pre-flight renames a ROOT constant, this test fails loudly
    instead of letting the enumeration below silently resolve nothing.
    """
    for name in _ROOT_PREFIXES:
        assert hasattr(pfc, name), (
            f"pre_flight_check.py no longer declares the ROOT constant "
            f"{name!r}; the independent enumeration in this test module "
            f"must be updated to match"
        )


# =============================================================================
# 2. The MANIFEST is COMPLETE.
# =============================================================================
def test_manifest_covers_the_exact_resolved_surface() -> None:
    """Every path the pre-flight resolves must be DECLARED in the manifest.

    This is the completeness arm: a NEW gate script that is registered but
    never declared would otherwise be unchecked, and its untracked-ness
    would be invisible again.
    """
    gaps = _manifest_gaps(pfc, PRE_FLIGHT)
    assert gaps == [], (
        "the pre-flight resolves path(s) that PREFLIGHT_RESOLVED_SURFACE_"
        "PATHS does not declare, so their trackedness is UNCHECKED: "
        f"{gaps}. Add each one to the manifest."
    )


def test_the_enforcement_stack_subset_is_declared() -> None:
    """The §5.11 stack must be a SUBSET of the declared surface."""
    declared = set(pfc.PREFLIGHT_RESOLVED_SURFACE_PATHS)
    missing = sorted(set(pfc.ENFORCEMENT_STACK_PATHS) - declared)
    assert missing == [], (
        f"§5.11 enforcement-stack member(s) not declared in the resolved "
        f"surface: {missing}"
    )


def test_resolved_surface_heuristic_is_declared() -> None:
    """The module's own RUNTIME heuristic must not find an undeclared path."""
    resolved = set(pfc._resolved_gate_surface())
    declared = set(pfc.PREFLIGHT_RESOLVED_SURFACE_PATHS)
    gaps = sorted(resolved - declared)
    assert gaps == [], (
        f"the pre-flight's runtime resolved-surface heuristic found path(s) "
        f"the manifest does not declare (a MANIFEST GAP): {gaps}"
    )


def test_the_manifest_declares_paths_that_exist() -> None:
    """A declared path that does not exist on disk is a stale declaration."""
    missing = [
        p
        for p in pfc.PREFLIGHT_RESOLVED_SURFACE_PATHS
        if not (REPO_ROOT / p).is_file()
    ]
    assert missing == [], (
        f"PREFLIGHT_RESOLVED_SURFACE_PATHS declares path(s) that do not "
        f"exist on disk: {missing}"
    )


# =============================================================================
# 3. The VERDICT is fail-CLOSED and agrees with an independent read.
# =============================================================================
def test_status_classification_branches() -> None:
    """Every branch of the classification is reachable and labelled."""
    # (a) git unprobeable -> NOT MEASURED for every path
    rows = _rows_with_git(pfc, index=None, head=None)
    assert rows, "expected one row per declared path"
    assert all(
        r["status"] == pfc.ENFORCEMENT_STACK_STATUS_NOT_MEASURED for r in rows
    ), "a tree whose trackedness cannot be probed must be NOT_MEASURED"
    assert all("could not be probed" in r["detail"] for r in rows)

    # (b) in the index -> TRACKED
    rows = _rows_with_git(pfc, index={"probe.py"}, head=None)
    assert rows[0]["status"] == pfc.ENFORCEMENT_STACK_STATUS_TRACKED

    # (c) in HEAD but not in the index -> NOT_TRACKED (removed from the index)
    rows = _rows_with_git(pfc, index=set(), head={"probe.py"})
    assert rows[0]["status"] == pfc.ENFORCEMENT_STACK_STATUS_NOT_TRACKED
    assert "NOT in git's index" in rows[0]["detail"]

    # (d) in neither -> NOT_TRACKED, NEVER COMMITTED
    rows = _rows_with_git(pfc, index=set(), head=set())
    assert rows[0]["status"] == pfc.ENFORCEMENT_STACK_STATUS_NOT_TRACKED
    assert "NEVER COMMITTED" in rows[0]["detail"]

    # (e) HEAD unreadable but the index readable -> still NOT_TRACKED
    rows = _rows_with_git(pfc, index=set(), head=None)
    assert rows[0]["status"] == pfc.ENFORCEMENT_STACK_STATUS_NOT_TRACKED


def _rows_with_git(
    module: Any, *, index: Optional[Set[str]], head: Optional[Set[str]]
) -> List[Dict[str, Any]]:
    """Classify a single synthetic path under a synthetic git state."""
    real_index = module._git_index_paths
    real_head = module._git_head_paths
    module._git_index_paths = lambda: index  # type: ignore[assignment]
    module._git_head_paths = lambda: head  # type: ignore[assignment]
    try:
        return module._enforcement_stack_trackedness(paths=["probe.py"])
    finally:
        module._git_index_paths = real_index  # type: ignore[assignment]
        module._git_head_paths = real_head  # type: ignore[assignment]


def test_gate_fails_closed_when_git_is_unprobeable(monkeypatch) -> None:
    """git cannot be probed => FAIL (never a silent PASS)."""
    monkeypatch.setattr(pfc, "_git_index_paths", lambda: None)
    result = pfc.gate_enforcement_stack_trackedness(GATE_NAME, "P0", "launch")
    assert result.status == "FAIL", (
        "a tree whose trackedness cannot be PROVEN cannot be CERTIFIED — "
        "the gate must FAIL when git cannot be probed "
        f"(got {result.status}: {result.evidence})"
    )
    assert "NOT MEASURED" in result.evidence


def test_gate_verdict_matches_an_independent_read_of_gits_index() -> None:
    """The verdict tracks the REAL index — a RELATION, not a frozen count.

    Before the stack is committed the gate FAILs; after the commit it
    PASSes.  Both are correct, and this test pins the RELATION rather than
    either number (AGENTS.md §6.269.32).
    """
    index = _index_paths()
    offenders = [
        p for p in pfc.PREFLIGHT_RESOLVED_SURFACE_PATHS if p not in index
    ]
    result = pfc.gate_enforcement_stack_trackedness(GATE_NAME, "P0", "launch")
    expected = "FAIL" if offenders else "PASS"
    assert result.status == expected, (
        f"the gate reported {result.status} but {len(offenders)} declared "
        f"member(s) are missing from git's index (git ls-files): {offenders}. "
        f"Evidence: {result.evidence}"
    )
    for path in offenders:
        assert path in result.evidence, (
            f"the gate did not NAME the untracked member {path!r}; an "
            f"unnamed offender is an unactionable verdict"
        )


def test_the_offenders_are_never_committed() -> None:
    """Every untracked member of the declared surface is UNCOMMITTED.

    A file that is in HEAD but not in the index is a DIFFERENT defect
    (a staged deletion); this test asserts the current, measured shape:
    the offenders exist only on this machine.
    """
    index = _index_paths()
    head = _head_paths()
    offenders = [
        p for p in pfc.PREFLIGHT_RESOLVED_SURFACE_PATHS if p not in index
    ]
    if not offenders:
        pytest.skip(
            "the declared surface is fully tracked — nothing to assert "
            "(this test describes the pre-commit state)"
        )
    in_head = [p for p in offenders if p in head]
    assert in_head == [], (
        f"declared member(s) are in HEAD but not in the index (a staged "
        f"deletion, not a never-committed file): {in_head}"
    )


# =============================================================================
# 4. The gate is wired where it belongs.
# =============================================================================
def test_gate_is_registered_on_the_launch_pipeline_only() -> None:
    """Launch-only: ``all`` is the commit-time pipeline and its count is
    pinned by TODO/completed/321d_preflight_gate_audit.md §2.1/§6."""
    launch_names = [n for n, _ in pfc.build_gates("launch")]
    all_names = [n for n, _ in pfc.build_gates("all")]
    assert GATE_NAME in launch_names
    assert GATE_NAME not in all_names
    assert pfc.EXPECTED_GATE_COUNTS["launch"] == len(pfc.build_gates("launch"))
    assert pfc.EXPECTED_GATE_COUNTS["all"] == len(pfc.build_gates("all"))
    # The `all` count is the doc-pinned one; this remediation deliberately
    # did NOT move it (a change would require the 4-place bump protocol of
    # catalog §6.236).  MEASURED 2026-11-09: 133.
    assert pfc.EXPECTED_GATE_COUNTS["all"] == 133


# =============================================================================
# 5. The pre-flight still BOOTS without its untracked modules.
# =============================================================================
_BLOCKED_MODULES = (
    "molmetal.scripts.wfa135_smoke_budget",
    "molmetal.scripts.wf_launch_gate_verdict",
)

_BOOT_PROBE = r'''
import json, sys
BLOCKED = {blocked!r}

class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED:
            raise ImportError("blocked by the wave-33 boot probe: " + fullname)
        return None

sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, {repo!r})
from molmetal.scripts import pre_flight_check as pfc
res = pfc.gate_enforcement_stack_trackedness("p", "P0", "launch")
print(json.dumps({{
    "smoke_budget_absent": pfc._smoke_budget is None,
    "gates_launch": len(pfc.build_gates("launch")),
    "gates_all": len(pfc.build_gates("all")),
    "verdict_status": res.status,
}}))
'''


def test_preflight_boots_without_its_untracked_import_time_modules() -> None:
    """A clean checkout has no untracked module — the pre-flight must BOOT.

    Before the repair, the pre-flight imported two untracked modules at
    MODULE level, so a partial commit of the pre-flight alone killed the
    whole pre-flight at import (every gate dead).  This test simulates the
    clean checkout by making those two modules un-importable.
    """
    script = _BOOT_PROBE.format(blocked=list(_BLOCKED_MODULES), repo=str(REPO_ROOT))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, (
        "the pre-flight could not be imported with its untracked modules "
        f"absent (rc={proc.returncode}) — on a clean checkout the whole "
        f"pre-flight would be dead at import.\nstderr:\n{proc.stderr[-4000:]}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["smoke_budget_absent"] is True, (
        "the smoke-budget import should have degraded to None when the "
        "module is absent"
    )
    assert payload["gates_launch"] == len(pfc.build_gates("launch"))
    assert payload["gates_all"] == len(pfc.build_gates("all"))
    assert payload["verdict_status"] in ("PASS", "FAIL"), (
        "the trackedness gate must still produce a verdict when the "
        "untracked modules are absent"
    )


def test_unguarded_module_level_imports_are_all_tracked() -> None:
    """An UNGUARDED module-level import of an untracked file kills the
    pre-flight on a clean checkout, so every unguarded one must be tracked.
    """
    tree = ast.parse(PRE_FLIGHT.read_text(encoding="utf-8"))
    index = _index_paths()
    offenders: List[str] = []
    for dotted, guarded in _module_level_imports(tree):
        if guarded:
            continue
        mapped = _import_to_repo_path(dotted)
        if mapped and mapped not in index:
            offenders.append(f"{dotted} -> {mapped}")
    assert offenders == [], (
        "the pre-flight has an UNGUARDED module-level import of a path that "
        "is not in git's index, so a clean checkout cannot import it: "
        f"{offenders}"
    )


def test_the_untracked_import_time_modules_are_guarded() -> None:
    """The module that is imported at module level must be imported safely.

    ``wfa135_smoke_budget`` is untracked, so its module-level import MUST be
    guarded (a clean checkout has no such module — an unguarded import
    would kill the whole pre-flight at import).

    ``wf_launch_gate_verdict`` is untracked too, but it is no longer
    imported at module level at all: the pre-flight imports it LAZILY, at
    the single use site, inside a try/except.  That is the stronger
    property, and this test pins it.
    """
    tree = ast.parse(PRE_FLIGHT.read_text(encoding="utf-8"))
    imports = dict(_module_level_imports(tree))

    smoke = "molmetal.scripts.wfa135_smoke_budget"
    assert smoke in imports, (
        f"{smoke} is not imported at module level any more — the manifest "
        f"still declares it as an import-time dependency"
    )
    assert imports[smoke] is True, (
        f"the module-level import of {smoke} is UNGUARDED, so a clean "
        f"checkout (where the module is absent) would kill the whole "
        f"pre-flight at import"
    )

    verdict = "molmetal.scripts.wf_launch_gate_verdict"
    assert verdict not in imports, (
        f"{verdict} is imported at module level again — it is not in git's "
        f"index, so an unguarded module-level import of it would kill the "
        f"whole pre-flight at import on a clean checkout"
    )
    src = PRE_FLIGHT.read_text(encoding="utf-8")
    assert verdict in src, (
        f"{verdict} is referenced nowhere in the pre-flight — the manifest "
        f"declares it and the payload hash uses it"
    )
