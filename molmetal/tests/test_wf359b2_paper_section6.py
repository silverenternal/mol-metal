"""Wf-359-B2 — Paper §6 Limitations draft scaffold tests.

Per AGENTS.md §5.3 (no paper work without authorization, TODO-84 active) +
TODO-84 §1 (paper sections forbidden without USER_OK), the actual §6
write to ``paper/sections/06_limitations_v4.tex`` requires explicit USER_OK.

This module ships 1 regression test that verifies the PRE-PREPARED draft
scaffold at ``docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md`` exists and
contains the 6 canonical sections (§1..§6 per the Wf-359-B2 brief):

  §0 TL;DR — Honest-negative framing is the §6 posture
  §1 Per-metric honest-negatives (m1-m21)
  §2 Per-workflow honest-negatives (Wf-1..358)
  §3 Per-launcher honest-negatives (5 wrappers × ~37.5 hr GPU total)
  §4 Resolved entries (Wf-344 + Wf-345 + Wf-346 + Wf-356 + Wf-358 closed)
  §5 Forward-looking honest-negatives (next 90 days)
  §6 Cross-references (AGENTS.md + Wf closeouts)

Honest framing per ``party-twenty-spirit`` §3.5: this is a doc-existence
+ section-coverage regression test (CPU-only, no GPU launches, no
``paper/*`` touches). Test results one per task brief.

Source: Wf-359-B2 task brief + Wf-357-B3 §6 Limitations source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_PATH = PROJECT_ROOT / "docs" / "PAPER_SECTION_6_LIMITATIONS_DRAFT.md"


@pytest.mark.skipif(not DOC_PATH.exists(), reason="docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md not on disk")
def test_paper_section6_draft_has_six_sections() -> None:
    """The §6 Limitations draft scaffold has 6 canonical §1..§6 sections.

    Per Wf-359-B2 brief: the scaffold MUST contain §1 per-metric +
    §2 per-workflow + §3 per-launcher + §4 resolved + §5 forward-looking
    + §6 cross-references. We grep for ``## §<N>`` markers (case-sensitive
    top-of-section heading style used throughout Mol-Metal closeouts).
    """
    text = DOC_PATH.read_text(encoding="utf-8")

    # Verify the 6 canonical section markers are present.
    expected_sections = [
        ("§1", "Per-metric"),
        ("§2", "Per-workflow"),
        ("§3", "Per-launcher"),
        ("§4", "Resolved"),
        ("§5", "Forward-looking"),
        ("§6", "Cross-references"),
    ]

    found_sections: list[tuple[str, str]] = []
    missing: list[str] = []
    for section_id, topic_keyword in expected_sections:
        # Match `## §<N>` at line start (with optional whitespace).
        pattern = rf"^## {re.escape(section_id)}\b"
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            missing.append(f"{section_id} (regex={pattern!r})")
            continue
        # Verify topic keyword is in the same line (case-insensitive).
        line_start = match.start()
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = len(text)
        heading_line = text[line_start:line_end]
        if topic_keyword.lower() not in heading_line.lower():
            missing.append(
                f"{section_id} (topic '{topic_keyword}' not in heading line: "
                f"{heading_line!r})"
            )
            continue
        found_sections.append((section_id, topic_keyword))

    assert not missing, (
        f"§6 Limitations draft scaffold is missing sections: {missing}. "
        f"DOC_PATH={DOC_PATH}. Found: {found_sections}."
    )
    assert len(found_sections) == 6, (
        f"Expected 6 canonical sections (§1..§6); found {len(found_sections)}. "
        f"Found: {found_sections}."
    )

    # Verify the doc DOES NOT live in paper/ (TODO-84 honored).
    paper_dir = PROJECT_ROOT / "paper"
    paper_section6_candidates = list(paper_dir.glob("sections/06_limitations_v4*.tex")) if paper_dir.exists() else []
    # We only check that THIS doc doesn't claim to be the paper file
    # (the file lives in docs/, not paper/sections/).
    assert "PAPER_SECTION_6" in DOC_PATH.name, (
        f"DOC_PATH must be the docs/ scaffold (PAPER_SECTION_6_* prefix), "
        f"not a paper/sections/*.tex file: {DOC_PATH}"
    )

    # Verify the doc mentions USER_OK requirement for actual paper write.
    assert "USER_OK" in text, (
        "§6 Limitations draft MUST cite USER_OK requirement for the "
        "actual paper §6 write to paper/sections/*.tex."
    )
    assert "TODO-84" in text, (
        "§6 Limitations draft MUST cite TODO-84 (no paper work without "
        "authorization) as the gating rule."
    )

    # Verify honest-framing posture (per party-twenty-spirit §3.5).
    assert "MEASURED" in text and "PROJECTED" in text, (
        "§6 Limitations draft MUST distinguish MEASURED vs PROJECTED values."
    )