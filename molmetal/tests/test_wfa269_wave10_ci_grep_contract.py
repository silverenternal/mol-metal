# Licensed under the Apache License, Version 2.0.
"""Wf-A269 wave-10 finding C5-5 — CI grep contract for the pitfall runner.

The ``pitfall-runner`` job in ``.github/workflows/ci.yml`` gates on the
footer emitted by :mod:`molmetal.scripts.wf_pitfall_guard.check_all`.
The pre-repair CI pattern was::

    grep -E "^\\s*PASS=7\\s+FAIL=0\\s+SKIP=0\\s*$" pitfall_check.log

but the producer's real footer (MEASURED — rendered via
``check_all._format_table`` with all seven guards PASSED) is::

    PASS=7 FAIL=0 SKIP=0 MISS=0

The pre-repair pattern omitted the trailing ``MISS=0`` counter and was
therefore anchored to a string the producer never emits: the CI step
could never pass, i.e. it was red for a reason unrelated to the code
under test (AGENTS.md §5.5 rule #6 — a gate that cannot fail is also a
gate that cannot pass).

These tests import the producer's own summary formatter, render the
footer it really emits, and assert the CI regex (read straight out of
``ci.yml``) matches it — while still rejecting every near-miss footer
so the assertion keeps its teeth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from molmetal.scripts.wf_pitfall_guard.check_all import (
    PITFALL_IDS,
    _format_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# The pre-repair CI pattern, preserved verbatim (AGENTS.md §5.5 rule #6).
OLD_CI_PATTERN = r"^\s*PASS=7\s+FAIL=0\s+SKIP=0\s*$"

# The footer the producer really emits when all seven guards pass.
EXPECTED_FOOTER = "PASS=7 FAIL=0 SKIP=0 MISS=0"


def _all_pass_results() -> Dict[str, str]:
    """Build the pytest-result mapping that makes all 7 guards PASS."""
    return {f"test_pitfall_guard_matrix[{pid}]": "PASSED" for pid in PITFALL_IDS}


def _emitted_footer(results: Dict[str, str] | None = None) -> str:
    """Render the producer's footer via its own summary formatter."""
    table = _format_table(results if results is not None else _all_pass_results())
    return table.splitlines()[-1].strip()


def _ci_grep_patterns() -> List[str]:
    """Extract every LIVE ``grep -E "<ere>" ... pitfall_check.log`` pattern.

    Commented-out lines are skipped: ci.yml preserves the superseded
    rendering as a shell comment (AGENTS.md §5.5 rule #6), and that
    preserved text is not the assertion the runner executes.
    """
    text = CI_YML.read_text(encoding="utf-8")
    patterns: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.search(r'grep\s+-E\s+"([^"]*)"\s+pitfall_check\.log', stripped)
        if match:
            patterns.append(match.group(1))
    return patterns


def test_producer_footer_shape_is_measured() -> None:
    """The producer emits the four-counter footer (MEASURED, not assumed)."""
    assert _emitted_footer() == EXPECTED_FOOTER
    assert "MISS=0" in _emitted_footer()


def test_ci_declares_exactly_one_pitfall_grep() -> None:
    """ci.yml must keep exactly one pitfall_check.log assertion."""
    patterns = _ci_grep_patterns()
    assert len(patterns) == 1, f"expected 1 pitfall_check.log grep, got {patterns}"


def test_ci_grep_matches_the_emitted_footer() -> None:
    """The CI regex must match the footer the producer actually emits."""
    (pattern,) = _ci_grep_patterns()
    footer = _emitted_footer()
    assert re.search(pattern, footer) is not None, (
        f"CI pattern {pattern!r} does not match emitted footer {footer!r}"
    )


def test_old_ci_regex_does_not_match_the_emitted_footer() -> None:
    """The pre-repair pattern could never match — that was the defect."""
    footer = _emitted_footer()
    assert re.search(OLD_CI_PATTERN, footer) is None, (
        "the superseded CI pattern unexpectedly matches the emitted footer; "
        "the producer's footer shape may have changed"
    )


def test_ci_grep_keeps_its_teeth() -> None:
    """Near-miss footers must NOT satisfy the CI regex."""
    (pattern,) = _ci_grep_patterns()
    rejected = [
        "PASS=6 FAIL=0 SKIP=0 MISS=1",  # a guard went missing
        "PASS=7 FAIL=1 SKIP=0 MISS=0",  # a guard regressed
        "PASS=7 FAIL=0 SKIP=1 MISS=0",  # a guard was skipped
        "PASS=7 FAIL=0 SKIP=0 MISS=1",  # all seven counted but one MISSed
        "PASS=7 FAIL=0 SKIP=0 MISS=0 extra",  # trailing junk
        "PASS=7",  # the loosened 'contains PASS' form
    ]
    for footer in rejected:
        assert re.search(pattern, footer) is None, (
            f"CI pattern {pattern!r} wrongly accepts {footer!r}"
        )


def test_ci_grep_does_not_degrade_to_a_substring_check() -> None:
    """The assertion must stay anchored — 'contains PASS=7' is not enough."""
    (pattern,) = _ci_grep_patterns()
    assert pattern.startswith("^")
    assert pattern.endswith("$")
    assert "MISS" in pattern, "the CI regex must assert the MISS counter"
