#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Wf-A159 — the TIMEOUT-verdict census for ``pre_flight_check.py``.

THE RULE
--------
A wall-clock TIMEOUT means **the measurement was NOT obtained**.  It is
NOT evidence that the thing under test is broken.  Reporting a timeout as
a defect verdict is a **FALSE DEFECT CLAIM** (AGENTS.md §5.2 honest
framing: a genuine defect is reported as a defect, and only a genuine
defect may be reported as one).

``pre_flight_check._timeout_gate_result(...)`` is the repo's encoding of
the correct verdict: status ``TIMEOUT`` plus evidence that says the
measurement was not obtained.  Wf-A135 introduced it for the §5.7 smoke
gates, Wf-A158 extended it to three more, and Wf-A159 closed the family.

WHAT THIS SCRIPT IS
-------------------
The **MACHINE-CHECKED GUARD** for that rule.  It parses
``molmetal/scripts/pre_flight_check.py`` with :mod:`ast`, enumerates
EVERY subprocess call site (``:func:`_run``, ``subprocess.run``,
``subprocess.Popen``), and for each one decides — from the AST of the
enclosing function — whether a wall-clock timeout at that site would be
converted into something other than a TIMEOUT verdict.

THE RULE, AS IMPLEMENTED
------------------------
A site is ``TIMEOUT`` iff the enclosing function contains a **timeout
branch** for that site, and that branch routes into the TIMEOUT verdict:

  * an ``if <rc> == 124:`` test whose body contains ``_timeout_gate_result(``
    or ``GATE_STATUS_TIMEOUT``; or
  * an ``except subprocess.TimeoutExpired`` handler (or a handler for a
    superclass of it — ``subprocess.SubprocessError``) whose body contains
    ``_timeout_gate_result(`` or ``GATE_STATUS_TIMEOUT``.

A site is ``UNHANDLED`` otherwise: a timeout there is **silently converted
into something else** — a defect verdict, a false claim, or a silent
fallback.  The census exits non-zero if any site is ``UNHANDLED``.

The one deliberate exception is the *producer*: the ``subprocess.run``
inside ``_run`` is where ``rc == 124`` is born.  Its role is recorded as
``producer`` and its verdict is ``TIMEOUT`` by construction — the CALLER
is what owes the verdict.

CPU-ONLY: this script imports nothing but the standard library.  It never
spawns a subprocess, never imports torch, never touches a GPU.

Usage
-----
::

    python molmetal/scripts/wfa159_timeout_verdict_census.py
    python molmetal/scripts/wfa159_timeout_verdict_census.py --target <path>
    python molmetal/scripts/wfa159_timeout_verdict_census.py --json-out <path>

Exit codes: ``0`` = PASS (zero UNHANDLED), ``1`` = FAIL (>= 1 UNHANDLED),
``2`` = SKIP (the target file is not on disk).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_REL = "molmetal/scripts/pre_flight_check.py"
DEFAULT_TARGET = REPO_ROOT / TARGET_REL
REPORT_DIR = REPO_ROOT / "molmetal/reports/wfa159_timeout_verdict_census"
DEFAULT_REPORT = REPORT_DIR / "timeout_verdict_census.json"

VERDICT_TIMEOUT = "TIMEOUT"
VERDICT_UNHANDLED = "UNHANDLED"

#: The subprocess entry points this census watches.
_WATCHED_CALLS = {
    "_run",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.check_output",
    "subprocess.check_call",
}

#: Names whose presence in a branch body means "this branch produces the
#: TIMEOUT verdict".
_TIMEOUT_MARKERS = ("_timeout_gate_result", "GATE_STATUS_TIMEOUT")

#: Exception names that a ``subprocess`` timeout raises (or that catch it).
_TIMEOUT_EXC_NAMES = ("TimeoutExpired", "SubprocessError")


# ---------------------------------------------------------------------------
# small AST helpers
# ---------------------------------------------------------------------------
def _dotted(node: ast.AST) -> Optional[str]:
    """``a.b.c`` for a Name/Attribute chain, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):  # bool is an int subclass — exclude
            return None
        return node.value
    return None


def _is_124_compare(test: ast.AST, rc_expr: str) -> bool:
    """True when ``test`` is ``<rc_expr> == 124`` (in either operand order)."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left = _dotted(test.left)
    right = _dotted(test.comparators[0])
    for a, b in ((left, right), (right, left)):
        if a == rc_expr and _const_int(test.comparators[0] if a == left else test.left) == 124:
            return True
    return False


def _is_zero_compare(test: ast.AST, rc_expr: str) -> bool:
    """True when ``test`` compares ``rc_expr`` against 0 (any operator)."""
    if not isinstance(test, ast.Compare):
        return False
    if _dotted(test.left) != rc_expr:
        return False
    return any(_const_int(c) == 0 for c in test.comparators)


def _branch_markers(body: List[ast.AST]) -> Tuple[bool, bool]:
    """Return ``(routes_to_timeout, contains_fail_literal)`` for a branch."""
    routes = False
    has_fail = False
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if isinstance(node, ast.Call):
            fname = _dotted(node.func)
            if fname and any(fname.endswith(m) for m in _TIMEOUT_MARKERS):
                routes = True
        elif isinstance(node, ast.Name):
            if node.id in _TIMEOUT_MARKERS:
                routes = True
        elif isinstance(node, ast.Attribute):
            if node.attr in _TIMEOUT_MARKERS:
                routes = True
        elif isinstance(node, ast.Constant) and node.value == "FAIL":
            has_fail = True
    return routes, has_fail


def _handler_catches_timeout(handler: ast.ExceptHandler) -> bool:
    """True when the handler catches ``subprocess.TimeoutExpired`` (or a superclass)."""
    if handler.type is None:
        return False
    targets: List[ast.AST] = []
    if isinstance(handler.type, ast.Tuple):
        targets = list(handler.type.elts)
    else:
        targets = [handler.type]
    for t in targets:
        name = _dotted(t)
        if name is None:
            continue
        leaf = name.split(".")[-1]
        if leaf in _TIMEOUT_EXC_NAMES:
            return True
    return False


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------
def _resolve_budget(
    call: ast.Call, func: Optional[ast.FunctionDef], module_consts: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """Return ``(budget_seconds_or_None, budget_expression)`` for a call."""
    timeout_kw = None
    for kw in call.keywords:
        if kw.arg == "timeout":
            timeout_kw = kw.value
            break
    if timeout_kw is None:
        return None, "(no timeout= declared)"
    expr = ast.unparse(timeout_kw)
    if isinstance(timeout_kw, ast.Constant) and isinstance(timeout_kw.value, (int, float)):
        return float(timeout_kw.value), expr
    if isinstance(timeout_kw, ast.Name):
        name = timeout_kw.id
        if func is not None:
            # Positional defaults align to the TAIL of ``args.args``;
            # keyword-only defaults align BY INDEX with ``kwonlyargs``.
            # (Zipping them together silently pairs a default with the
            # wrong parameter — which is how a budget of 600.0 s got
            # reported against a parameter called ``name``.)
            positional = list(func.args.args)
            pos_defaults = list(func.args.defaults)
            offset = len(positional) - len(pos_defaults)
            bound: Dict[str, ast.AST] = {}
            for i, d in enumerate(pos_defaults):
                bound[positional[offset + i].arg] = d
            for a, d in zip(func.args.kwonlyargs, func.args.kw_defaults):
                if d is not None:
                    bound[a.arg] = d
            default = bound.get(name)
            if default is not None and isinstance(default, ast.Constant):
                if isinstance(default.value, (int, float)) and not isinstance(
                    default.value, bool
                ):
                    return float(default.value), expr
        if name in module_consts:
            val = module_consts[name]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val), expr
    return None, expr


def _module_constants(tree: ast.Module) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    return out


def _rc_expression(call: ast.Call, assign: Optional[ast.AST], call_name: str) -> str:
    """The dotted expression that holds this call's return code."""
    target: Optional[ast.AST] = None
    if isinstance(assign, ast.Assign) and assign.targets:
        target = assign.targets[0]
    elif isinstance(assign, ast.AnnAssign):
        target = assign.target
    if isinstance(target, ast.Tuple) and target.elts:
        first = target.elts[0]
        if isinstance(first, ast.Name):
            return first.id
    if isinstance(target, ast.Name):
        # `_run` always returns a 3-tuple; a bare Name target therefore means
        # the caller is using the whole tuple (or the code is a bug).  For
        # `subprocess.run` the rc is an attribute of the result object.
        if call_name in ("subprocess.run", "subprocess.Popen"):
            return f"{target.id}.returncode"
        return target.id
    return "(unresolved)"


def _enclosing_function(
    node: ast.AST, funcs: List[ast.FunctionDef]
) -> Optional[ast.FunctionDef]:
    best: Optional[ast.FunctionDef] = None
    for f in funcs:
        if f.lineno <= node.lineno <= (f.end_lineno or f.lineno):
            if best is None or f.lineno > best.lineno:
                best = f
    return best


def _assign_for(node: ast.AST, funcs: List[ast.FunctionDef]) -> Optional[ast.AST]:
    """The assignment statement whose RHS is ``node`` (if any)."""
    for f in funcs:
        if not (f.lineno <= node.lineno <= (f.end_lineno or f.lineno)):
            continue
        for stmt in ast.walk(f):
            if isinstance(stmt, ast.Assign) and stmt.value is node:
                return stmt
            if isinstance(stmt, ast.AnnAssign) and stmt.value is node:
                return stmt
    return None


def run_census(target: Path = DEFAULT_TARGET) -> Dict[str, Any]:
    """Parse ``target`` and return the census as a plain dict."""
    if not target.is_file():
        return {
            "status": "SKIP",
            "reason": f"target not on disk: {target}",
            "target_file": str(target),
            "sites": [],
            "summary": {"total_sites": 0, "timeout": 0, "unhandled": 0},
        }

    text = target.read_text(encoding="utf-8")
    tree = ast.parse(text)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    consts = _module_constants(tree)

    # Map every watched call node to its enclosing function + assignment.
    sites: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _dotted(node.func)
        if fname is None or fname not in _WATCHED_CALLS:
            continue
        func = _enclosing_function(node, funcs)
        assign = _assign_for(node, funcs)
        rc_expr = _rc_expression(node, assign, fname)
        budget, budget_expr = _resolve_budget(node, func, consts)

        # --- collect the timeout branches of the enclosing function ---
        has_124_branch = False
        routes_to_timeout = False
        contains_fail = False
        branch_kinds: List[str] = []
        handler_names: List[str] = []
        if func is not None:
            for sub in ast.walk(func):
                if isinstance(sub, ast.If) and _is_124_compare(sub.test, rc_expr):
                    has_124_branch = True
                    branch_kinds.append("if_rc_eq_124")
                    r, f = _branch_markers(sub.body)
                    routes_to_timeout = routes_to_timeout or r
                    contains_fail = contains_fail or f
                elif isinstance(sub, ast.ExceptHandler) and _handler_catches_timeout(sub):
                    has_124_branch = True
                    branch_kinds.append("except_timeout_expired")
                    r, f = _branch_markers(sub.body)
                    routes_to_timeout = routes_to_timeout or r
                    contains_fail = contains_fail or f
                    handler_names.append(ast.unparse(sub.type) if sub.type else "?")

        # --- does the function map a non-zero rc onto a defect verdict? ---
        rc_nonzero_checked = False
        function_has_fail = False
        if func is not None:
            for sub in ast.walk(func):
                if isinstance(sub, ast.If) and _is_zero_compare(sub.test, rc_expr):
                    rc_nonzero_checked = True
                if isinstance(sub, ast.Constant) and sub.value == "FAIL":
                    function_has_fail = True

        func_name = func.name if func is not None else "<module>"
        role = "helper"
        if func_name == "_run":
            role = "producer"
        elif func_name.startswith("gate_") or func_name.startswith("_gate_"):
            role = "gate"

        if role == "producer":
            verdict = VERDICT_TIMEOUT
            effect = "PRODUCES_RC_124"
        elif has_124_branch and routes_to_timeout and not contains_fail:
            verdict = VERDICT_TIMEOUT
            effect = "TIMEOUT_VERDICT"
        elif has_124_branch and not contains_fail:
            verdict = VERDICT_TIMEOUT
            effect = "NOT_MEASURED_FALLBACK"
        else:
            verdict = VERDICT_UNHANDLED
            effect = "NONE" if not has_124_branch else "BRANCH_MAPS_TO_DEFECT"

        sites.append(
            {
                "file": str(target.relative_to(REPO_ROOT))
                if str(target).startswith(str(REPO_ROOT))
                else str(target),
                "line": node.lineno,
                "call": fname,
                "enclosing_function": func_name,
                "role": role,
                "rc_expression": rc_expr,
                "declared_budget_sec": budget,
                "declared_budget_expr": budget_expr,
                "has_124_branch": has_124_branch,
                "timeout_branch_kinds": branch_kinds,
                "timeout_handlers": handler_names,
                "routes_to_timeout_verdict": routes_to_timeout,
                "timeout_branch_contains_fail_literal": contains_fail,
                "rc_consumed_by_zero_check": rc_nonzero_checked,
                "enclosing_function_contains_fail_status": function_has_fail,
                "verdict_for_124": verdict,
                "timeout_branch_effect": effect,
            }
        )

    sites.sort(key=lambda s: s["line"])
    n_timeout = sum(1 for s in sites if s["verdict_for_124"] == VERDICT_TIMEOUT)
    n_unhandled = sum(1 for s in sites if s["verdict_for_124"] == VERDICT_UNHANDLED)
    return {
        "status": "PASS" if n_unhandled == 0 else "FAIL",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_file": str(target.relative_to(REPO_ROOT))
        if str(target).startswith(str(REPO_ROOT))
        else str(target),
        "target_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "target_lines": len(text.splitlines()),
        "rule": (
            "A site is TIMEOUT iff the enclosing function contains a timeout "
            "branch for it (an `if <rc> == 124:` test, or an "
            "`except subprocess.TimeoutExpired` handler) whose body routes "
            "into the TIMEOUT verdict (_timeout_gate_result( / "
            "GATE_STATUS_TIMEOUT) and contains no FAIL literal.  A site is "
            "UNHANDLED otherwise: a timeout there would be silently "
            "converted into something other than a TIMEOUT verdict."
        ),
        "sites": sites,
        "summary": {
            "total_sites": len(sites),
            "timeout": n_timeout,
            "unhandled": n_unhandled,
            "unhandled_sites": [
                f"{s['file']}:{s['line']} ({s['enclosing_function']})"
                for s in sites
                if s["verdict_for_124"] == VERDICT_UNHANDLED
            ],
            "by_role": {
                role: sum(1 for s in sites if s["role"] == role)
                for role in ("producer", "gate", "helper")
            },
            "by_effect": {
                effect: sum(1 for s in sites if s["timeout_branch_effect"] == effect)
                for effect in (
                    "PRODUCES_RC_124",
                    "TIMEOUT_VERDICT",
                    "NOT_MEASURED_FALLBACK",
                    "NONE",
                    "BRANCH_MAPS_TO_DEFECT",
                )
            },
        },
    }


def _write_report(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    ap.add_argument("--json-out", default=str(DEFAULT_REPORT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.target).resolve()
    out_path = Path(args.json_out).resolve()

    payload = run_census(target)
    if payload["status"] == "SKIP":
        print(f"SKIP: {payload['reason']}")
        return 2

    _write_report(payload, out_path)
    s = payload["summary"]
    if not args.quiet:
        print(f"census target : {payload['target_file']} (sha256 {payload['target_sha256'][:12]})")
        print(f"census sites  : {s['total_sites']} subprocess call sites")
        print(f"  TIMEOUT     : {s['timeout']}")
        print(f"  UNHANDLED   : {s['unhandled']}")
        print(f"  by role     : {s['by_role']}")
        print(f"  by effect   : {s['by_effect']}")
        for site in payload["sites"]:
            print(
                f"    {site['verdict_for_124']:9s} {site['file']}:{site['line']:<6d} "
                f"{site['enclosing_function']}({site['role']}) "
                f"budget={site['declared_budget_sec']} "
                f"effect={site['timeout_branch_effect']}"
            )
        for bad in s["unhandled_sites"]:
            print(f"    UNHANDLED: {bad}")
    print(f"wrote {out_path}")
    if payload["status"] == "PASS":
        print("PASS: every subprocess call site routes a wall-clock timeout into a TIMEOUT verdict")
        return 0
    print(
        f"FAIL: {s['unhandled']} site(s) would silently convert a wall-clock "
        f"timeout into something other than a TIMEOUT verdict"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
