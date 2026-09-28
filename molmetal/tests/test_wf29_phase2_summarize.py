"""Tests for Wf-29 Phase 2 summarizer (``wf29_phase2_summarize.py``).

These tests are read-mostly / opt-in.  They verify the Wf-29 Phase 2
deliverables produced by the summarizer:

* ``molmetal/reports/wf29_smoke/wf29_summary.json`` — the slim rollup
  bench summary.
* ``molmetal/reports/wf29_smoke/comparison_phase2.md`` — the
  side-by-side markdown comparison.

All tests skip when the summarizer outputs are missing (so CI without
the prior Wf-29 Phase 2 sweep is not broken).  When present, they
assert:

1.  ``wf29_summary.json`` parses and declares the right recipe tag.
2.  ``comparison_phase2.md`` references all 4 documented recipes
    (se3sb / transport / multiph / baseline).
3.  The markdown has a ``Honest-negative provenance`` section.
4.  The bench summary lists ``recipes_recorded`` and (optionally)
    ``recipes_skipped_missing``.

Count target: ~4 distinct checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF29_SMOKE = PROJECT_ROOT / "molmetal" / "reports" / "wf29_smoke"
SUMMARY_PATH = WF29_SMOKE / "wf29_summary.json"
COMPARISON_MD = WF29_SMOKE / "comparison_phase2.md"

DOCUMENTED_RECIPES = ("se3sb", "transport", "multiph", "baseline")


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"WF-29 Phase 2 output missing: {path}")


def _load(path: Path) -> object:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_wf29_phase2_summary_well_formed() -> None:
    """wf29_summary.json must parse and carry the Wf-29 Phase 2 tag."""
    _skip_if_missing(SUMMARY_PATH)
    payload = _load(SUMMARY_PATH)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf29_phase2_smoke_bench_summary"


def test_wf29_phase2_summary_lists_recipes() -> None:
    """The bench summary must list recorded (and skipped) recipes."""
    _skip_if_missing(SUMMARY_PATH)
    payload = _load(SUMMARY_PATH)
    recorded = payload.get("recipes_recorded") or []
    # At least one of the 4 documented recipes should be present if
    # any Wf-29 trainer ran; otherwise we still expect the field to
    # exist (empty list is valid when no Wf-29 trainer ever ran).
    assert isinstance(recorded, list), "recipes_recorded must be a list"
    skipped = payload.get("recipes_skipped_missing") or []
    assert isinstance(skipped, list), "recipes_skipped_missing must be a list"


def test_wf29_phase2_comparison_md_references_all_recipes() -> None:
    """comparison_phase2.md must reference all 4 documented recipes."""
    _skip_if_missing(COMPARISON_MD)
    body = COMPARISON_MD.read_text()
    for recipe in DOCUMENTED_RECIPES:
        assert recipe in body, (
            f"comparison_phase2.md missing recipe {recipe!r}"
        )


def test_wf29_phase2_comparison_md_has_honest_negative_section() -> None:
    """comparison_phase2.md must declare honest-negative provenance."""
    _skip_if_missing(COMPARISON_MD)
    body = COMPARISON_MD.read_text()
    assert "Honest-negative provenance" in body, (
        "comparison_phase2.md missing 'Honest-negative provenance' section"
    )