#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gate-count pinner (Wf-A169).

Why this module exists
----------------------
The pre-flight gate count is pinned in many places.  On 2026-11-01 the
Wf-A169 census MEASURED six test files whose pins were **stale** — they
asserted a count that had been superseded weeks earlier, so those tests were
RED on a clean tree.  A stale pin is a defect in its own right: it fails for
a reason that has nothing to do with the code it claims to guard, and it
teaches the reader that "red" means "the thing I care about broke".

Two things are missing from the repo:

1.  A single authoritative way to ask "how many gates does pipeline *p* have
    right now, and does that agree with the constant?".
2.  A way to enumerate **every** place the count is pinned, so the next wave
    that adds a gate gets the complete re-sync list automatically instead of
    discovering the next stale pin by accident (which is how the six
    Wf-A169 stale pins happened in the first place: Wf-A154 re-synced one
    test file and missed the other five).

This module provides both.  It is:

*   **READ-ONLY** — it never writes, edits or deletes anything.
*   **CPU-only** — it imports no GPU framework and touches no CUDA path.
    ``pre_flight_check`` (the only thing it imports, and only for the live
    counts) has no top-level ``torch`` import either.
*   **torch-free** — statically asserted by
    ``molmetal/tests/test_wfa169_gate_count_pins.py``.

Public API
----------
``live_gate_counts``
    ``{pipeline: len(build_gates(pipeline))}`` for every pipeline.
``assert_gate_counts_match_expected``
    The Wf-321D invariant: live count == ``EXPECTED_GATE_COUNTS[pipeline]``.
``GateCountPin``
    One pin site: path, line, kind, pipeline, pinned value, live value,
    status.
``scan_gate_count_pins``
    Enumerate every executable gate-count pin in the repo.
``pin_census``
    The full census: executable pins + prose mentions, with counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "GateCountPin",
    "PIPELINE_NAMES",
    "assert_gate_counts_match_expected",
    "expected_gate_counts",
    "live_gate_counts",
    "pin_census",
    "scan_file_for_pins",
    "scan_gate_count_pins",
    "scan_prose_mentions",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_REL = "molmetal/scripts/pre_flight_check.py"

#: The five pipelines the pre-flight dispatcher knows about.
PIPELINE_NAMES = ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")

#: Extensions scanned for *executable* pins.  ``.md`` / ``.sh`` / ``.yaml``
#: / ``.json`` are scanned separately as PROSE (a document cannot assert).
_EXEC_SUFFIXES = (".py",)
_PROSE_SUFFIXES = (".md", ".sh", ".yaml", ".yml", ".json", ".txt")

#: The census's own output directory.  It is EXCLUDED from the scan: a
#: census that counts its own output is not a measurement — the number
#: moves every time the census is written (MEASURED: the first run's
#: denominator was inflated by the previous run's 4105-row inventory).
CENSUS_OUTPUT_REL = "molmetal/reports/wfa169_dead_gate_repair"

#: Vocabulary that marks a line as *gate-count* talk.  Tight on purpose:
#: "aggregate" (contains "gate"), "expected_total_wall_min" (a wall-clock
#: budget, not a gate count) and "the gate counts metals" (narrative) are
#: NOT gate counts.
_COUNT_VOCAB_RE = re.compile(
    r"EXPECTED_GATE_COUNTS|build_gates\s*\(|expected_total(?!_wall_min)|gate[_ ]count|GATE_COUNT",
    re.IGNORECASE,
)

#: A prose line that states a concrete gate count.  Used to flag *stale
#: prose* — a document that still names a superseded count.  These rows are
#: inventory only: AGENTS.md §5.5 rule 6 preserves the superseded rendering
#: of a document verbatim, so they are NOT edited.
_PROSE_COUNT_RE = re.compile(
    r"EXPECTED_GATE_COUNTS\s*\[\s*['\"](?P<p1>[A-Za-z_][A-Za-z_0-9]*)['\"]\s*\]"
    r"\s*(?:==|=|:)\s*(?P<n1>\d+)"
    r"|['\"](?P<p2>[A-Za-z_][A-Za-z_0-9]*)['\"]\s*:\s*(?P<n2>\d+)"
    r"|\b(?P<p3>all|launch|frontier|chemprop_v2_v3|cfm)\s*=\s*(?P<n3>\d+)"
)

#: ``'"all": 57'`` — the text-scan style of pinning a count.  The pipeline
#: name is restricted to the known pipelines: a bare ``"warmup_epochs: 5"``
#: in a YAML config is NOT a gate-count pin.
_TEXT_LITERAL_RE = re.compile(r"""^["']?(?P<pipeline>[A-Za-z_][A-Za-z_0-9]*)["']?\s*:\s*(?P<count>\d+)$""")

#: ``preflight_summary["total"] == 129`` — the summary total of the 'all'
#: pipeline, pinned from a fixture rather than from a live call.  The name
#: must contain "preflight": a bare ``summary["total"] == 2`` in
#: ``test_wfa125_preflight_warn_contract.py`` is a TWO-GATE run, not the
#: pipeline total.
_SUMMARY_TOTAL_RE = re.compile(r"(?i)^(?=.*preflight).*summary$")


# --------------------------------------------------------------------------
# live counts
# --------------------------------------------------------------------------
def _load_preflight():
    """Import ``pre_flight_check`` (CPU-only; no torch at module scope)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from molmetal.scripts import pre_flight_check  # noqa: WPS433 (runtime import)

    return pre_flight_check


def live_gate_counts() -> dict[str, int]:
    """Live ``len(build_gates(pipeline))`` for every known pipeline.

    MEASURED by live import of ``pre_flight_check`` — never a literal.
    """
    pfc = _load_preflight()
    names = list(PIPELINE_NAMES)
    for extra in getattr(pfc, "PIPELINES", ()):  # pragma: no cover - defensive
        if extra not in names:
            names.append(extra)
    return {name: len(pfc.build_gates(name)) for name in names}


def expected_gate_counts() -> dict[str, int]:
    """The ``EXPECTED_GATE_COUNTS`` constant as-is."""
    pfc = _load_preflight()
    return dict(pfc.EXPECTED_GATE_COUNTS)


def assert_gate_counts_match_expected(
    pipeline: str | None = None,
) -> dict[str, dict[str, int | str]]:
    """Assert ``len(build_gates(p)) == EXPECTED_GATE_COUNTS[p]``.

    Returns one record per pipeline checked.  Raises ``AssertionError``
    listing every drifted pipeline — not just the first — because the
    Wf-A169 census found six stale pins at once and a one-at-a-time report
    hides that.
    """
    live = live_gate_counts()
    expected = expected_gate_counts()
    names = [pipeline] if pipeline else sorted(expected)
    report: dict[str, dict[str, int | str]] = {}
    drifted: list[str] = []
    for name in names:
        actual = live.get(name)
        want = expected.get(name)
        ok = actual is not None and want is not None and actual == want
        report[name] = {"live": actual, "expected": want, "status": "ok" if ok else "drift"}
        if not ok:
            drifted.append(f"{name}: live={actual} expected={want}")
    assert not drifted, "gate-count drift — " + "; ".join(drifted)
    return report


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------
def _str_const(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _int_const(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):
            return None
        return node.value
    return None


def _is_build_gates_call(node: ast.AST) -> str | None:
    """``build_gates("all")`` (any receiver) → the pipeline literal."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name != "build_gates":
        return None
    if node.args:
        return _str_const(node.args[0])
    for kw in node.keywords:
        if kw.arg == "pipeline":
            return _str_const(kw.value)
    return None


def _is_expected_const_sub(node: ast.AST) -> str | None:
    """``EXPECTED_GATE_COUNTS["all"]`` (any receiver) → the pipeline literal."""
    if not isinstance(node, ast.Subscript):
        return None
    name = None
    if isinstance(node.value, ast.Name):
        name = node.value.id
    elif isinstance(node.value, ast.Attribute):
        name = node.value.attr
    if name != "EXPECTED_GATE_COUNTS":
        return None
    idx = node.slice
    if isinstance(idx, ast.Index):  # pragma: no cover - py<3.9 shape
        idx = idx.value  # type: ignore[attr-defined]
    return _str_const(idx)


def _resolve_path_constant(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = REPO_ROOT / "a" / "b.md"`` → repo-relative path.

    The ``/`` chain is flattened, so ``REPO_ROOT / "a" / "b.md"`` yields
    ``"a/b.md"``.  Only *string* segments are kept — the root segment
    (``REPO_ROOT`` or ``Path(__file__).resolve().parents[N]``) is dropped
    because the resolved path is always relative to the repository root.
    """
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        segments: list[str] = []
        value = node.value
        saw_root = False
        while isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
            right = value.right
            left = value.left
            if isinstance(right, ast.Constant) and isinstance(right.value, str):
                segments.append(right.value)
            else:
                segments = []
                break
            if isinstance(left, ast.Name) or isinstance(left, ast.Call):
                saw_root = True
                break
            if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                value = left
                continue
            segments = []
            break
        if not saw_root or not segments:
            continue
        out[target.id] = "/".join(reversed(segments))
    return out


def _local_bindings(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Map variable name → pipeline for values derived from ``build_gates``.

    Handles the two shapes that occur in the repo:

    *   ``runners = build_gates("all")``
    *   ``names = [r[0] for r in runners]``

    A comprehension that **filters** (``[n for n, _ in build_gates("all")
    if n.startswith("p3_")]``) is NOT a gate-count binding: its length is a
    subset size, not the pipeline size.  Binding it would make
    ``assert len(names) == 5`` look like a stale pin of 5 gates.
    """
    direct: dict[str, str] = {}
    derived: dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            pipeline = _is_build_gates_call(node.value)
            if pipeline is not None:
                direct[target.id] = pipeline
                continue
            # list/set/Generator comprehension over a build_gates-derived var
            src = node.value
            if isinstance(src, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in src.generators:
                    if gen.ifs:
                        # filtered subset — not a pipeline-size binding
                        continue
                    it = gen.iter
                    if isinstance(it, ast.Name) and it.id in direct:
                        derived[target.id] = direct[it.id]
                    elif isinstance(it, ast.Call):
                        p = _is_build_gates_call(it)
                        if p is not None:
                            derived[target.id] = p
    direct.update(derived)
    return direct


# --------------------------------------------------------------------------
# pin record
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GateCountPin:
    """One executable gate-count pin site."""

    path: str
    line: int
    kind: str
    pipeline: str | None
    pinned: int | None
    live: int | None
    status: str
    snippet: str
    detail: str = ""

    @property
    def stale(self) -> bool:
        return self.status == "stale"

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "pipeline": self.pipeline,
            "pinned": self.pinned,
            "live": self.live,
            "status": self.status,
            "snippet": self.snippet,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# the scanner
# --------------------------------------------------------------------------
class _FileScanner:
    """Extract executable gate-count pins from one Python source file."""

    def __init__(self, rel_path: str, text: str, root: Path | None = None) -> None:
        self.rel = rel_path
        self.text = text
        self.root = root or REPO_ROOT
        self.lines = text.splitlines()
        try:
            self.tree: ast.Module | None = ast.parse(text)
        except SyntaxError:
            self.tree = None
        self.path_constants = _resolve_path_constant(self.tree) if self.tree else {}

    # -- helpers ---------------------------------------------------------
    def _snippet(self, node: ast.AST) -> str:
        try:
            line = self.lines[node.lineno - 1]
        except IndexError:  # pragma: no cover - defensive
            return ""
        return line.strip()

    def _gate_count_value(
        self, node: ast.AST, scope: dict[str, str]
    ) -> tuple[str | None, int, str]:
        """Is ``node`` a gate-count expression?

        Returns ``(pipeline, adjustment, kind)``.  ``adjustment`` is the
        integer added to the live count before it is compared with the
        literal — e.g. ``EXPECTED_GATE_COUNTS["all"] - 32`` has
        ``adjustment == -32``, so the effective pin is ``literal + 32``.
        """
        # arithmetic: EXPECTED_GATE_COUNTS["all"] - 32
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            base = self._gate_count_value(node.left, scope)
            right = _int_const(node.right)
            if base[0] is not None and right is not None:
                sign = 1 if isinstance(node.op, ast.Add) else -1
                return (base[0], base[1] + sign * right, base[2])
            return (None, 0, "")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
            if not node.args:
                return (None, 0, "")
            inner = node.args[0]
            pipeline = _is_build_gates_call(inner)
            if pipeline is not None:
                return (pipeline, 0, "len-build_gates")
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "set":
                if inner.args and isinstance(inner.args[0], ast.Name):
                    var = inner.args[0].id
                    if var in scope:
                        return (scope[var], 0, "len-set-of-build_gates")
            if isinstance(inner, ast.Name) and inner.id in scope:
                return (scope[inner.id], 0, "len-of-build_gates")
            return (None, 0, "")
        pipeline = _is_expected_const_sub(node)
        if pipeline is not None:
            return (pipeline, 0, "expected-gate-counts-const")
        # preflight_summary["total"] — the 'all' pipeline total from a fixture
        if isinstance(node, ast.Subscript):
            idx = node.slice
            if isinstance(idx, ast.Index):  # pragma: no cover - py<3.9 shape
                idx = idx.value  # type: ignore[attr-defined]
            if _str_const(idx) == "total":
                base = node.value
                name = None
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute):
                    name = base.attr
                if name and _SUMMARY_TOTAL_RE.match(name):
                    return ("all", 0, "summary-total")
        return (None, 0, "")

    @staticmethod
    def _is_doc_derived(node: ast.AST) -> bool:
        """``int(match.group(1))`` — a number read out of a document."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        if node.func.id != "int" or len(node.args) != 1:
            return False
        inner = node.args[0]
        return (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "group"
        )

    # -- doc-derived resolution ------------------------------------------
    def _resolve_doc_expr(self, node: ast.AST) -> str | None:
        """Resolve a path expression to a repo-relative path."""
        if isinstance(node, ast.Name):
            return self.path_constants.get(node.id)
        # REPO_ROOT / "a" / "b.md"  (possibly nested)
        segments: list[str] = []
        value = node
        saw_root = False
        while isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
            right = value.right
            left = value.left
            if isinstance(right, ast.Constant) and isinstance(right.value, str):
                segments.append(right.value)
            else:
                return None
            if isinstance(left, ast.Name) or isinstance(left, ast.Call):
                saw_root = True
                break
            if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                value = left
                continue
            return None
        if not saw_root or not segments:
            return None
        return "/".join(reversed(segments))

    def _resolve_doc_pin(self, node: ast.Assert) -> tuple[str, int | None, str]:
        """Find the document that carries the pinned number.

        Returns ``(doc_path, pinned_number, pattern)``.  ``pinned_number`` is
        ``None`` when the document or the pattern could not be resolved.
        """
        doc_path: str | None = None
        pattern: str | None = None
        text_vars: set[str] = set()
        fn = None
        for candidate in ast.walk(self.tree):
            if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for stmt in candidate.body:
                if stmt is node or node in ast.walk(stmt):
                    fn = candidate
                    break
            if fn is not None:
                break
        if fn is None:
            return ("", None, "")
        # local path bindings:  path = REPO_ROOT / "a" / "b.md"
        local_paths: dict[str, str] = {}
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            resolved = self._resolve_doc_expr(stmt.value)
            if resolved:
                local_paths[target.id] = resolved
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = stmt.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "read_text"
            ):
                base = value.func.value
                resolved = None
                if isinstance(base, ast.Name):
                    resolved = local_paths.get(base.id) or self.path_constants.get(base.id)
                else:
                    resolved = self._resolve_doc_expr(base)
                if resolved:
                    doc_path = resolved
                    text_vars.add(target.id)
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "search"
                and len(value.args) >= 2
            ):
                pat = _str_const(value.args[0])
                var = value.args[1]
                if pat and isinstance(var, ast.Name) and var.id in text_vars:
                    pattern = pat
        if not doc_path:
            return ("", None, pattern or "")
        pinned: int | None = None
        if pattern:
            text = self._read_doc(doc_path)
            if text is not None:
                try:
                    m = re.search(pattern, text)
                except re.error:
                    m = None
                if m is not None:
                    try:
                        pinned = int(m.group(1))
                    except (IndexError, ValueError):
                        pinned = None
        return (doc_path, pinned, pattern or "")

    def _read_doc(self, rel: str) -> str | None:
        target = Path(self.root) / rel
        try:
            return target.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return None

    # -- the scan --------------------------------------------------------
    def scan(self) -> list[GateCountPin]:
        if self.tree is None:
            return []
        pins: list[GateCountPin] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assert):
                pins.extend(self._pins_from_assert(node))
        return pins

    def _enclosing_scope(self, node: ast.AST) -> dict[str, str]:
        for fn in ast.walk(self.tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for stmt in fn.body:
                    if stmt is node or node in ast.walk(stmt):
                        return _local_bindings(fn)
        return {}

    def _pins_from_assert(self, node: ast.Assert) -> list[GateCountPin]:
        out: list[GateCountPin] = []
        test = node.test
        scope = self._enclosing_scope(node)

        comparisons: list[tuple[ast.AST, ast.AST, ast.cmpop]] = []
        if isinstance(test, ast.Compare):
            left = test.left
            for op, comp in zip(test.ops, test.comparators):
                comparisons.append((left, comp, op))
                left = comp
        elif isinstance(test, ast.BoolOp):
            for value in test.values:
                if isinstance(value, ast.Compare):
                    left = value.left
                    for op, comp in zip(value.ops, value.comparators):
                        comparisons.append((left, comp, op))
                        left = comp

        for left, right, op in comparisons:
            pin = self._classify(left, right, op, node, scope)
            if pin is not None:
                out.append(pin)
        return out

    def _classify(
        self,
        left: ast.AST,
        right: ast.AST,
        op: ast.cmpop,
        node: ast.Assert,
        scope: dict[str, str],
    ) -> GateCountPin | None:
        # --- dynamic invariant: both sides are gate-count expressions ------
        l_pipe, l_adj, l_kind = self._gate_count_value(left, scope)
        r_pipe, r_adj, r_kind = self._gate_count_value(right, scope)
        if l_pipe is not None and r_pipe is not None:
            return GateCountPin(
                path=self.rel,
                line=node.lineno,
                kind="dynamic-invariant",
                pipeline=l_pipe,
                pinned=None,
                live=None,
                status="dynamic",
                snippet=self._snippet(node),
                detail=f"{l_kind} vs {r_kind}",
            )

        # --- exact equality pin ------------------------------------------
        if isinstance(op, ast.Eq):
            if l_pipe is not None:
                lit = _int_const(right)
                if lit is not None:
                    return self._mk(node, l_kind, l_pipe, lit - l_adj)
            if r_pipe is not None:
                lit = _int_const(left)
                if lit is not None:
                    return self._mk(node, r_kind, r_pipe, lit - r_adj)

        # --- floor / ceiling guard (repo convention: `>= 71`) ------------
        if isinstance(op, (ast.GtE, ast.LtE)):
            if l_pipe is not None:
                lit = _int_const(right)
                if lit is not None:
                    return self._mk(node, l_kind + "-floor", l_pipe, lit, status="floor")
            if r_pipe is not None:
                lit = _int_const(left)
                if lit is not None:
                    return self._mk(node, r_kind + "-floor", r_pipe, lit, status="floor")

        # --- doc-derived pin: the literal lives in a document ------------
        if isinstance(op, ast.Eq):
            if self._is_doc_derived(left) and r_pipe is not None:
                doc, pinned, pattern = self._resolve_doc_pin(node)
                pin = self._mk(node, "assert-doc-derived", r_pipe, pinned, status="derived")
                return GateCountPin(
                    path=pin.path,
                    line=pin.line,
                    kind=pin.kind,
                    pipeline=pin.pipeline,
                    pinned=pin.pinned,
                    live=pin.live,
                    status=pin.status,
                    snippet=pin.snippet,
                    detail=f"document={doc or 'UNRESOLVED'}; regex={pattern or 'UNRESOLVED'}",
                )
            if self._is_doc_derived(right) and l_pipe is not None:
                doc, pinned, pattern = self._resolve_doc_pin(node)
                pin = self._mk(node, "assert-doc-derived", l_pipe, pinned, status="derived")
                return GateCountPin(
                    path=pin.path,
                    line=pin.line,
                    kind=pin.kind,
                    pipeline=pin.pipeline,
                    pinned=pin.pinned,
                    live=pin.live,
                    status=pin.status,
                    snippet=pin.snippet,
                    detail=f"document={doc or 'UNRESOLVED'}; regex={pattern or 'UNRESOLVED'}",
                )

        # --- text-scan style: '"all": 57' in text -----------------------
        if isinstance(op, ast.In):
            lit = _str_const(left)
            if lit is not None and isinstance(right, ast.Name) and right.id == "text":
                m = _TEXT_LITERAL_RE.match(lit.strip())
                if m and m.group("pipeline") in PIPELINE_NAMES:
                    return self._mk(node, "assert-text-literal", m.group("pipeline"), int(m.group("count")))
        if isinstance(op, ast.Eq):
            lit = _str_const(left)
            other = _str_const(right)
            if lit is not None and other is not None:
                m = _TEXT_LITERAL_RE.match(lit.strip())
                if m and m.group("pipeline") in PIPELINE_NAMES:
                    return self._mk(node, "assert-text-literal", m.group("pipeline"), int(m.group("count")))
        return None

    def _mk(
        self,
        node: ast.Assert,
        kind: str,
        pipeline: str | None,
        pinned: int | None,
        status: str = "unknown",
    ) -> GateCountPin:
        return GateCountPin(
            path=self.rel,
            line=node.lineno,
            kind=kind,
            pipeline=pipeline,
            pinned=pinned,
            live=None,
            status=status,
            snippet=self._snippet(node),
        )


def _iter_python_files(root: Path, include: Sequence[str]) -> list[str]:
    """Python files scanned for EXECUTABLE pins.

    The census's own output directory is excluded (see
    :data:`CENSUS_OUTPUT_REL`), for the same reason the prose scan
    excludes it: a census that counts its own output is not a
    measurement, and the denominator would move every time this wave
    drops a script into its own report directory.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from wfa156_text_scan import iter_repo_files  # noqa: WPS433
    except ImportError:  # pragma: no cover - fallback
        try:
            from molmetal.scripts.wfa156_text_scan import iter_repo_files  # noqa: WPS433
        except ImportError:
            return []
    return [
        rel
        for rel in iter_repo_files(root, include=list(include) or ["*.py"], untracked=True)
        if rel.endswith(".py") and not rel.startswith(CENSUS_OUTPUT_REL)
    ]


def _iter_prose_files(root: Path, include: Sequence[str]) -> list[str]:
    """Prose files: markdown, shell, yaml, json, txt, and python.

    The extension filter matters: ``iter_repo_files`` with an empty
    ``include`` returns *every* tracked path, and this repo holds 201 GB
    of ``.pt`` checkpoints.  Prose is a document scan, so only documents
    are scanned — plus ``.py``, because a python docstring that names a
    gate count is a prose site too.

    The census's own output directory is excluded (see
    :data:`CENSUS_OUTPUT_REL`).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from wfa156_text_scan import iter_repo_files  # noqa: WPS433
    except ImportError:  # pragma: no cover - fallback
        try:
            from molmetal.scripts.wfa156_text_scan import iter_repo_files  # noqa: WPS433
        except ImportError:
            return []
    rels = iter_repo_files(root, include=list(include), untracked=True)
    keep = (*_PROSE_SUFFIXES, ".py")
    return [
        r
        for r in rels
        if r.lower().endswith(keep) and not r.startswith(CENSUS_OUTPUT_REL)
    ]


def _read(path: Path) -> tuple[str, str | None]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}"
    return data.decode("utf-8", errors="replace"), None


def scan_file_for_pins(
    rel_path: os.PathLike | str,
    *,
    root: os.PathLike | str = REPO_ROOT,
    live: dict[str, int] | None = None,
) -> list[GateCountPin]:
    """Scan ONE file for executable gate-count pins.

    Exposed separately from :func:`scan_gate_count_pins` so that a caller
    (a test, or the next wave) can point the scanner at a synthetic file
    without needing the file to be tracked by git.
    """
    root = Path(root)
    if live is None:
        live = live_gate_counts()
    path = Path(rel_path)
    rel = str(path.relative_to(root)) if path.is_absolute() else str(rel_path)
    text, err = _read(path if path.is_absolute() else root / rel)
    if err:
        return []
    scanner = _FileScanner(rel, text, root=root)
    return [
        _resolve_status(pin, live)
        for pin in scanner.scan()
    ]


def _resolve_status(pin: GateCountPin, live: dict[str, int]) -> GateCountPin:
    live_val = live.get(pin.pipeline) if pin.pipeline else None
    status = pin.status
    if status == "unknown":
        if live_val is None:
            status = "unknown-pipeline"
        elif pin.pinned == live_val:
            status = "ok"
        else:
            status = "stale"
    elif status == "floor":
        # A floor (`>= 71`) is satisfied by any count at or above the
        # bound; it is a monotonic guard, never a stale pin.
        bound = pin.pinned
        assert bound is not None
        if live_val is None:
            status = "unknown-pipeline"
        elif pin.kind.endswith("-floor"):
            status = "floor-satisfied" if live_val >= bound else "floor-violated"
    elif status == "derived":
        # A doc-derived pin carries its literal in a document.  When the
        # document could be read and parsed, the same ok/stale verdict
        # applies to it.
        if live_val is None:
            status = "unknown-pipeline"
        elif pin.pinned is None:
            status = "derived-unresolved"
        elif pin.pinned == live_val:
            status = "ok"
        else:
            status = "stale"
    return GateCountPin(
        path=pin.path,
        line=pin.line,
        kind=pin.kind,
        pipeline=pin.pipeline,
        pinned=pin.pinned,
        live=live_val,
        status=status,
        snippet=pin.snippet,
        detail=pin.detail,
    )


def scan_gate_count_pins(
    root: os.PathLike | str = REPO_ROOT,
    *,
    include: Sequence[str] = (),
    live: dict[str, int] | None = None,
    exclude: Sequence[str] = (),
) -> list[GateCountPin]:
    """Enumerate every executable gate-count pin in the repo.

    A *pin* is a site that FAILS when the live gate count differs from the
    value it names.  Prose (docstrings, comments, narrative markdown, log
    lines) is deliberately NOT a pin — per AGENTS.md §5.5 rule 6 the
    superseded rendering of prose is preserved verbatim.

    Parameters
    ----------
    root:
        Repository root.
    include:
        Extra path globs to restrict the scan to (e.g. ``["molmetal/tests"]``).
    live:
        Live counts per pipeline; computed when omitted.
    exclude:
        Path prefixes that are NOT scanned.  The census passes
        :data:`CENSUS_OUTPUT_REL` here so that the census never counts
        its own output directory.
    """
    root = Path(root)
    if live is None:
        live = live_gate_counts()

    pins: list[GateCountPin] = []
    for rel in _iter_python_files(root, include):
        if any(rel.startswith(ex) for ex in exclude):
            continue
        text, err = _read(root / rel)
        if err:
            continue
        scanner = _FileScanner(rel, text, root=root)
        for pin in scanner.scan():
            pins.append(_resolve_status(pin, live))
    return sorted(pins, key=lambda p: (p.path, p.line))


def scan_prose_mentions(
    root: os.PathLike | str = REPO_ROOT,
    *,
    include: Sequence[str] = (),
    exclude_pin_sites: Sequence[tuple[str, int]] = (),
) -> list[dict]:
    """Non-executable gate-count mentions (documents and python prose).

    Every hit is reported with ``file:line`` and a short snippet.  A prose
    mention is NEVER a pin: per AGENTS.md §5.5 rule 6 the superseded
    rendering of a document is preserved verbatim, so these rows are
    inventory, not a work list.

    ``exclude_pin_sites`` removes the statement line of an executable pin
    from the prose population, so the two populations do not overlap.
    """
    root = Path(root)
    skip = set(exclude_pin_sites)
    out: list[dict] = []
    for rel in _iter_prose_files(root, include):
        text, err = _read(root / rel)
        if err:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if (rel, idx) in skip:
                continue
            if _COUNT_VOCAB_RE.search(line):
                out.append(
                    {
                        "path": rel,
                        "line": idx,
                        "kind": "prose-" + rel.rsplit(".", 1)[-1],
                        "snippet": line.strip()[:240],
                    }
                )
    return out


def scan_stale_prose_numbers(
    root: os.PathLike | str = REPO_ROOT,
    *,
    include: Sequence[str] = (),
    live: dict[str, int] | None = None,
) -> list[dict]:
    """Prose rows that still name a SUPERSEDED gate count.

    Inventory only.  These are NOT pins: a document cannot fail, and
    AGENTS.md §5.5 rule 6 preserves the superseded rendering of a document
    verbatim (a correction is additive and dated, never a silent edit).
    A stale *runtime log line* (e.g. a shell script that prints
    ``EXPECTED_GATE_COUNTS['all'] = 66``) is the one case where the number
    is printed to a reader today; it is still not a pin.
    """
    root = Path(root)
    if live is None:
        live = live_gate_counts()
    out: list[dict] = []
    for rel in _iter_prose_files(root, include):
        text, err = _read(root / rel)
        if err:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if not _COUNT_VOCAB_RE.search(line):
                continue
            for match in _PROSE_COUNT_RE.finditer(line):
                pipeline = match.group("p1") or match.group("p2") or match.group("p3")
                number = match.group("n1") or match.group("n2") or match.group("n3")
                if pipeline not in PIPELINE_NAMES or number is None:
                    continue
                claimed = int(number)
                actual = live.get(pipeline)
                out.append(
                    {
                        "path": rel,
                        "line": idx,
                        "pipeline": pipeline,
                        "claimed": claimed,
                        "live": actual,
                        "stale": actual is not None and claimed != actual,
                        "snippet": line.strip()[:240],
                    }
                )
    return out


def pin_census(
    root: os.PathLike | str = REPO_ROOT,
    *,
    include: Sequence[str] = (),
) -> dict:
    """The full census: executable pins + prose mentions + counts.

    The returned dict keeps the two populations apart, so "no executable pin"
    can never be confused with "not looked at" (the Wf-A151 lesson).
    """
    root = Path(root)
    live = live_gate_counts()
    expected = expected_gate_counts()
    pins = scan_gate_count_pins(
        root,
        include=include,
        live=live,
        exclude=(CENSUS_OUTPUT_REL,),
    )
    prose = scan_prose_mentions(
        root,
        include=include,
        exclude_pin_sites=[(p.path, p.line) for p in pins],
    )
    prose_numbers = scan_stale_prose_numbers(root, include=include, live=live)
    py_files = _iter_python_files(root, include)
    prose_files = _iter_prose_files(root, include)
    by_file: dict[str, int] = {}
    for p in pins:
        by_file[p.path] = by_file.get(p.path, 0) + 1
    stale_files = sorted({p.path for p in pins if p.stale})
    exact = [p for p in pins if p.status in ("ok", "stale", "unknown-pipeline")]
    floors = [p for p in pins if p.status.startswith("floor")]
    dynamic = [p for p in pins if p.status == "dynamic"]
    derived = [p for p in pins if p.kind.startswith("assert-doc-derived")]
    return {
        "live_gate_counts": live,
        "expected_gate_counts": expected,
        "counts_match": live == expected,
        "files_scanned_python": len(py_files),
        "files_scanned_prose": len(prose_files),
        "executable_pin_count": len(exact),
        "executable_pin_file_count": len({p.path for p in exact}),
        "stale_pin_count": sum(1 for p in pins if p.stale),
        "stale_pin_file_count": len(stale_files),
        "stale_pin_files": stale_files,
        "ok_pin_count": sum(1 for p in pins if p.status == "ok"),
        "floor_guard_count": len(floors),
        "floor_guard_file_count": len({p.path for p in floors}),
        "floor_violated_count": sum(1 for p in pins if p.status == "floor-violated"),
        "dynamic_invariant_count": len(dynamic),
        "doc_derived_pin_count": len(derived),
        "doc_derived_unresolved_count": sum(
            1 for p in pins if p.status == "derived-unresolved"
        ),
        "prose_mention_count": len(prose),
        "prose_mention_file_count": len({m["path"] for m in prose}),
        "prose_stale_number_count": sum(1 for p in prose_numbers if p["stale"]),
        "prose_stale_number_file_count": len(
            {p["path"] for p in prose_numbers if p["stale"]}
        ),
        "pins": [p.as_dict() for p in pins],
        "prose_mentions": prose,
        "prose_number_claims": prose_numbers,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the census JSON to this path")
    ap.add_argument("--include", action="append", default=[], help="restrict the scan")
    ap.add_argument("--stale-only", action="store_true", help="print only stale pins")
    ap.add_argument("--assert-live", action="store_true", help="assert live == EXPECTED")
    args = ap.parse_args(argv)

    census = pin_census(REPO_ROOT, include=args.include)

    if args.assert_live:
        assert_gate_counts_match_expected()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pins = census["pins"]
    if args.stale_only:
        pins = [p for p in pins if p["status"] == "stale"]
    for p in pins:
        print(
            f"{p['status']:>7}  {p['path']}:{p['line']}  "
            f"pipeline={p['pipeline']} pinned={p['pinned']} live={p['live']}"
        )
    print(
        f"\nexecutable pins: {census['executable_pin_count']} "
        f"in {census['executable_pin_file_count']} files "
        f"(stale: {census['stale_pin_count']} in {census['stale_pin_file_count']} files); "
        f"floor guards: {census['floor_guard_count']}; "
        f"dynamic invariants: {census['dynamic_invariant_count']}; "
        f"doc-derived pins: {census['doc_derived_pin_count']}; "
        f"prose mentions: {census['prose_mention_count']} "
        f"in {census['prose_mention_file_count']} files"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
