"""Test Wf-351-B3: SCI paper submission skill structural acceptance gate.

Per AGENTS.md §3 skill catalog discipline: every skill must have SKILL.md with
frontmatter + ≥ 5 sections + whenToUse trigger. This test validates structural
shape of the canonical on-disk skill at
.claude/skills/mol-metal-sci-paper-submission/SKILL.md. It does NOT execute
the workflow (Stage 1-4 are human-in-the-loop per SKILL.md §3 caveat #5 +
§7 "No auto-submit tool").
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(
    "/home/hugo/codes/mol-metal/.claude/skills/mol-metal-sci-paper-submission"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


def test_skill_md_exists() -> None:
    """SKILL.md must exist at the canonical path."""
    assert SKILL_MD.exists(), f"SKILL.md missing at {SKILL_MD}"
    assert SKILL_MD.is_file()


def test_skill_md_has_frontmatter() -> None:
    """YAML frontmatter must declare name + description fields."""
    text = SKILL_MD.read_text()
    assert text.startswith("---\n"), "SKILL.md must begin with YAML frontmatter"
    # Extract frontmatter
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m is not None, "YAML frontmatter must be closed"
    fm = m.group(1)
    assert re.search(r"^name:\s*mol-metal-sci-paper-submission\s*$", fm, re.MULTILINE), (
        "frontmatter must declare name=mol-metal-sci-paper-submission"
    )
    assert "description:" in fm, "frontmatter must include description"


def test_skill_md_has_minimum_5_sections() -> None:
    """At least 5 ## sections required (per AGENTS.md §3 discipline)."""
    text = SKILL_MD.read_text()
    sections = re.findall(r"^##\s+§?\d", text, re.MULTILINE)
    assert len(sections) >= 5, (
        f"SKILL.md must have ≥ 5 ## sections, found {len(sections)}: {sections}"
    )


def test_skill_md_has_workflow_stages() -> None:
    """The 4-stage pre-submission / authoring / submission / post-submission
    workflow must be present (canonical §4 of the on-disk SKILL.md)."""
    text = SKILL_MD.read_text()
    for stage_label in ("Pre-submission", "Authoring-time", "Submission-time", "Post-submission"):
        assert stage_label in text, (
            f"Workflow stage '{stage_label}' must be in SKILL.md §4"
        )


def test_skill_md_readonly_on_paper() -> None:
    """Per TODO-84 + AGENTS.md §5.3: skill must declare READ-only on paper/*."""
    text = SKILL_MD.read_text()
    # The on-disk SKILL.md uses phrases like "zero touches", "READ-only under",
    # "without explicit user authorization", or "no paper work without"
    readonly_markers = (
        "READ-only", "READ only", "no touches", "without explicit user authorization",
        "no paper work without", "not modify", "respects TODO-84",
    )
    assert any(marker in text for marker in readonly_markers), (
        f"SKILL.md must declare paper/* READ-only discipline per TODO-84 + §5.3; "
        f"expected one of: {readonly_markers}"
    )
    assert "TODO-84" in text, "SKILL.md must cite TODO-84 authorization gate"


def test_skill_md_cites_open_source_audits() -> None:
    """Per user directive 2026-09-29 '记得审查' (remember to audit): skill must
    cite OSI-approved licenses for every adopted tool (MIT / AGPL-3.0 / BSD /
    GPL / Apache-2.0)."""
    text = SKILL_MD.read_text()
    # OSI-approved license markers that MUST appear in the audit table
    for license_marker in ("MIT", "AGPL-3.0", "Apache-2.0"):
        assert license_marker in text, (
            f"OSI license '{license_marker}' must be cited in SKILL.md audit table"
        )


def test_skill_md_decision_discipline() -> None:
    """Per AGENTS.md §5.2 honest framing: skill must mark every tool with an
    adopt/skip decision tag (✅ / ⏸️ / ❌ / ⚠️) rather than aspirational prose."""
    text = SKILL_MD.read_text()
    for decision_tag in ("✅", "⏸️", "❌"):
        assert decision_tag in text, (
            f"Decision tag '{decision_tag}' must be present in SKILL.md audit"
        )