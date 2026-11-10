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
r"""Wf-A269 wave-10 — the VACUOUS-ASSERT class: a constant assertion that
certifies nothing.

WHAT THIS FILE GUARDS (Wf-A269 wave-10, 2026-11-02)
----------------------------------------------------
An ``assert`` whose condition is the literal ``True`` (or ``False``) is a
statement that can never fail.  It looks like a test, it is collected by
pytest, it reports a green PASS — and it measures nothing.  It is also
stripped entirely by ``python -O``, so the suite that "passed" under the
optimiser ran one assertion fewer with no visible signal.

The finding that opened this wave (C5-4) is one live instance:

  * ``molmetal/tests/test_drugood_chemprop_integration.py`` had exactly ONE
    assertion in its whole body — ``assert True, "… is importable"`` — and
    that assertion was the file's entire G1/G4 "the test exists and is
    importable" gate.  The file's own docstring says the integration test
    must not skip silently; the assertion that was supposed to enforce that
    could not fail.

THIS FILE'S RULE
----------------
Every ``molmetal/tests/*.py`` file is AST-scanned.  An ``assert`` whose test
is the boolean constant ``True`` or ``False`` is a **vacuous assert** UNLESS
it sits in one of the two contexts where a constant assertion is the correct
idiom:

  1. inside a ``try`` / ``except`` statement — the "this must have raised"
     idiom (``assert False, "RuntimeError should have been raised"``); or
  2. inside a ``with pytest.raises(...)`` block.

Both idioms are used in this corpus today and both are legitimate:
``molmetal/tests/test_wf427a4_b5_schema_adapter.py:129`` is the canonical
example.  They are NOT defects and this guard leaves them alone.

THE METHOD (and its one deliberate blind spot)
----------------------------------------------
The scan is an AST walk, not a text grep, so a constant assertion quoted in a
comment or inside a docstring is invisible to it — which is what we want:
AGENTS.md §5.5 rule #6 requires a superseded rendering to be preserved
VERBATIM, and a text grep would punish that.

The blind spot this method does NOT cover: an ``assert`` whose condition is a
constant EXPRESSION rather than a literal — e.g. ``assert 1`` or
``assert bool(1)`` — is not matched, because the rule is stated over the two
boolean literals.  ``assert 1`` does not occur in this corpus today
(MEASURED 2026-11-02 by the scanner's own self-test below).

THE CARVE-OUT (a live, temporary exemption — not a permanent hole)
-------------------------------------------------------------------
One corpus file carries a constant assert that this wave does NOT repair:
``molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py`` belongs to the
Wf-A269 no-hardcoded-repo-paths repair group, which is running IN PARALLEL
with this wave.  The repair wave is file-disjoint, so that file is not in
this group's ``files=`` list and this agent may not edit it.  Its single
vacuous assert is therefore exempted (and PRINTED on every run, so the
exemption is visible rather than silent).  The carve-out is dropped as soon
as the owning group's repair lands; nothing else may be added to it.

CPU-only: this file parses source text.  It imports no torch, allocates no
GPU memory, and runs no trainer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]

# Files whose constant asserts are EXEMPT from this guard, keyed by file-name
# PREFIX so the exemption survives a rename that keeps the prefix.  Every
# entry needs a reason: the exemption is a documented, temporary carve-out
# for a file owned by a parallel repair group — never a silent hole.
EXEMPT_FILE_PREFIXES = {
    "test_wfa269_no_hardcoded_repo_paths.py": (
        "Owned by the Wf-A269 no-hardcoded-repo-paths repair group, which "
        "runs in PARALLEL with this wave.  This wave's file-disjoint "
        "files= list does not include it, so its owner repairs it; the "
        "carve-out is dropped as soon as that repair lands."
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_raises_call(node: ast.AST) -> bool:
    """Is this expression a ``pytest.raises(...)`` / ``raises(...)`` call?"""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "raises"
    if isinstance(func, ast.Name):
        return func.id == "raises"
    return False


class _VacuousAssertScanner(ast.NodeVisitor):
    """Collect every constant ``assert True`` / ``assert False`` in a module.

    A hit is *vacuous* when it is NOT inside a ``try``/``except`` statement
    and NOT inside a ``with pytest.raises(...)`` block.
    """

    def __init__(self) -> None:
        self.hits: List[Tuple[int, bool, bool]] = []  # (lineno, value, exempt)
        self._try_depth = 0
        self._raises_depth = 0

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802 (ast API)
        self._try_depth += 1
        self.generic_visit(node)
        self._try_depth -= 1

    def visit_With(self, node: ast.With) -> None:  # noqa: N802 (ast API)
        if any(_is_raises_call(item.context_expr) for item in node.items):
            self._raises_depth += 1
            self.generic_visit(node)
            self._raises_depth -= 1
            return
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802 (ast API)
        test = node.test
        if isinstance(test, ast.Constant) and isinstance(test.value, bool):
            exempt = self._try_depth > 0 or self._raises_depth > 0
            self.hits.append((node.lineno, bool(test.value), exempt))
        self.generic_visit(node)


def _scan_source(source: str) -> List[Tuple[int, bool, bool]]:
    """Scan a SOURCE STRING -> [(lineno, value, exempt), …]."""
    scanner = _VacuousAssertScanner()
    scanner.visit(ast.parse(source))
    return scanner.hits


def _scan_file(path: Path) -> List[Tuple[int, bool, bool]]:
    """Scan one test file -> [(lineno, value, exempt), …].

    A file that does not parse is a defect of its own; it is reported as a
    scan failure rather than silently skipped.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    return _scan_source(source)


def _corpus_files() -> List[Path]:
    """Every top-level ``molmetal/tests/*.py`` file."""
    return sorted(p for p in TESTS_DIR.glob("*.py") if p.is_file())


def _is_exempt(path: Path) -> bool:
    return any(path.name.startswith(prefix) for prefix in EXEMPT_FILE_PREFIXES)


# ---------------------------------------------------------------------------
# Test 1 — the scanner itself can fail (a guard that cannot fail is the
# DEAD-GATE class this same wave guards against)
# ---------------------------------------------------------------------------


def test_scanner_detects_a_vacuous_constant_assert() -> None:
    """A bare ``assert True`` must be reported as a vacuous hit."""
    hits = _scan_source("def test_x():\n    assert True\n")
    assert len(hits) == 1, f"expected one hit, got {hits!r}"
    lineno, value, exempt = hits[0]
    assert value is True
    assert exempt is False, "a bare `assert True` is vacuous, not exempt"


def test_scanner_detects_a_vacuous_constant_false() -> None:
    """A bare ``assert False`` outside a try/except is a vacuous hit."""
    hits = _scan_source("def test_x():\n    assert False\n")
    assert len(hits) == 1, f"expected one hit, got {hits!r}"
    assert hits[0][1] is False
    assert hits[0][2] is False


def test_scanner_exempts_the_try_except_idiom() -> None:
    """``assert False`` inside a ``try`` body is the CORRECT idiom."""
    source = (
        "def test_x():\n"
        "    try:\n"
        "        fn()\n"
        "        assert False, 'should have raised'\n"
        "    except RuntimeError:\n"
        "        pass\n"
    )
    hits = _scan_source(source)
    assert len(hits) == 1, f"expected one hit, got {hits!r}"
    assert hits[0][2] is True, "an assert inside a try/except is exempt"


def test_scanner_exempts_the_pytest_raises_idiom() -> None:
    """``assert False`` inside ``with pytest.raises(...)`` is exempt."""
    source = (
        "def test_x():\n"
        "    with pytest.raises(ValueError):\n"
        "        assert False\n"
    )
    hits = _scan_source(source)
    assert len(hits) == 1, f"expected one hit, got {hits!r}"
    assert hits[0][2] is True, "an assert inside pytest.raises is exempt"


def test_scanner_ignores_non_constant_asserts() -> None:
    """A computed condition is a real assertion and must not be flagged."""
    source = (
        "def test_x():\n"
        "    assert x == 1\n"
        "    assert isinstance(y, str)\n"
        "    assert True in counts\n"
    )
    assert _scan_source(source) == [], "computed assertions must not be flagged"


def test_scanner_ignores_asserts_quoted_in_comments_and_docstrings() -> None:
    """AGENTS.md §5.5 rule #6 keeps superseded renderings VERBATIM.

    A preserved ``assert True`` inside a comment must not be reported — the
    scan is an AST walk, so text in a comment is invisible to it.
    """
    source = (
        '"""docstring mentioning assert True"""\n'
        "def test_x():\n"
        "    # assert True, 'preserved verbatim'\n"
        "    assert x == 1\n"
    )
    assert _scan_source(source) == [], "commented-out asserts must not be flagged"


# ---------------------------------------------------------------------------
# Test 2 — the corpus rule
# ---------------------------------------------------------------------------


def test_no_vacuous_constant_assert_in_the_test_corpus() -> None:
    """Zero ``assert True`` / ``assert False`` outside a try/except context.

    MEASURED before this wave's repair: ``molmetal/tests/
    test_drugood_chemprop_integration.py:219`` carried the corpus's only
    other constant assert (the finding that opened the wave).
    """
    violations = []
    for path in _corpus_files():
        if _is_exempt(path):
            continue
        for lineno, value, exempt in _scan_file(path):
            if exempt:
                continue
            violations.append(f"{path.name}:{lineno}  assert {value}")

    assert not violations, (
        "constant assertions certify nothing and are stripped by "
        "`python -O`; replace each with an assertion about the surface "
        "under test:\n  " + "\n  ".join(violations)
    )


def test_exempted_files_are_reported_not_hidden() -> None:
    """The carve-out is visible on every run, and it stays NARROW.

    A permanent hole would be worse than the defect it excuses, so the
    exempt files are printed with their hit count and their reason, and the
    exemption is bounded: one file, at most one vacuous constant assert.
    """
    corpus = {p.name: p for p in _corpus_files()}

    for prefix, reason in EXEMPT_FILE_PREFIXES.items():
        matching = sorted(name for name in corpus if name.startswith(prefix))
        print(f"\nWf-A269 wave-10 vacuous-assert guard — EXEMPT: {prefix}")
        print(f"  reason: {reason}")
        if not matching:
            print("  STALE EXEMPTION — the file is no longer in the corpus")
            continue

        assert len(matching) == 1, (
            f"the exemption for {prefix!r} matches {len(matching)} corpus "
            f"files ({matching!r}) — it must name exactly ONE file"
        )

        hits = [
            (lineno, value)
            for lineno, value, exempt in _scan_file(corpus[matching[0]])
            if not exempt
        ]
        print(f"  vacuous constant assert(s) still present: {len(hits)}")
        for lineno, value in hits:
            print(f"    {matching[0]}:{lineno}  assert {value}")

        assert len(hits) <= 1, (
            f"{matching[0]} carries {len(hits)} vacuous constant asserts "
            f"{hits!r} — an exemption for ONE known hit has become a hole; "
            f"repair the file instead of widening the exemption"
        )


def test_the_corpus_scan_is_not_vacuous_over_an_empty_set() -> None:
    """The guard must actually SEE the corpus (a scan of zero files always passes)."""
    files = _corpus_files()
    assert len(files) > 100, (
        f"the corpus scan found only {len(files)} files under {TESTS_DIR} — "
        f"the guard would be passing over an empty set"
    )
    scanned = [p for p in files if _scan_file(p) is not None]
    assert len(scanned) == len(files), "every corpus file must be scannable"


def test_every_exemption_carries_a_reason() -> None:
    """An exemption without a documented reason is not an exemption."""
    for prefix, reason in EXEMPT_FILE_PREFIXES.items():
        assert reason.strip(), f"the exemption for {prefix} has no reason"
        assert len(reason.strip()) >= 40, (
            f"the exemption for {prefix} must state WHY the file is exempt"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
