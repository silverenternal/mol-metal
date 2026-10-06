"""Wf-364-A4: Sci paper §6 Limitations scaffold template test.

Per AGENTS.md §5.3 (no paper work without authorization) + TODO-84:
this test verifies that the §6 Limitations LaTeX template scaffold lives
in `docs/` (NOT `paper/`) and contains the required LaTeX snippet,
MEASURED-vs-PROJECTED label discipline, and cross-references to
metrics/by_metric/*.json source paths.

Wall: PROJECTED 15-25 min CPU-only (Wf-364-A4).
"""

from pathlib import Path

import pytest


# Path constants (absolute, per project rule §5.3).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = REPO_ROOT / "docs" / "SECTION_6_LIMITATIONS_TEMPLATE.md"


def test_wf364a4_section6_template_exists():
    """Template scaffold file exists at docs/SECTION_6_LIMITATIONS_TEMPLATE.md."""
    assert TEMPLATE_PATH.exists(), (
        f"Template scaffold missing at {TEMPLATE_PATH}. "
        "Per AGENTS.md §5.3 + TODO-84, the §6 Limitations scaffold must "
        "live in docs/ (NOT paper/) until USER_OK arrives."
    )


def test_template_has_latex_snippet():
    """Template contains LaTeX snippet ready for paper/sections/06_limitations.tex."""
    content = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Required LaTeX markers
    assert (
        "\\section{Limitations}" in content
    ), "Missing \\section{Limitations} in template LaTeX snippet."
    assert (
        "\\label{sec:limitations:v4}" in content
    ), "Missing \\label{sec:limitations:v4} for v4 anchor."
    assert (
        "\\MEASURED{}" in content and "\\PROJECTED{}" in content
    ), "Missing \\MEASURED{} / \\PROJECTED{} label discipline."
    # Begin/end code fences for the LaTeX block
    assert "```latex" in content, "Missing ```latex code fence."
    assert "```" in content, "Missing closing code fence."


def test_template_cross_references_metrics():
    """Template cross-references metrics/by_metric/*.json source paths."""
    content = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert (
        "metrics/by_metric/" in content
    ), "Missing cross-reference to metrics/by_metric/*.json source paths."
    # Required source paths from docs/PAPER_SECTION_6_v4_FINAL.md
    assert "metalcytotox_ru_auc.json" in content
    assert "wf335m2_r2/result.json" in content or "wf335m1_chemprop_pll" in content
    # Honest-negative catalog cross-ref
    assert "HONEST_NEGATIVE_CATALOG.md" in content


def test_template_preserves_user_ok_gate():
    """Template explicitly preserves USER_OK gate for paper/sections/*.tex writes."""
    content = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Must mention TODO-84 + AGENTS.md §5.3 + USER_OK requirement
    assert "TODO-84" in content
    assert "AGENTS.md" in content
    assert "USER_OK" in content
    # Must NOT contain actual paper/sections/06_limitations*.tex file (no touches)
    assert "paper/sections/06_limitations_v4.tex" in content  # as string, not file
    # The path is referenced as the *target* for USER_OK write, not actual file
    # Verify the actual file is NOT created (paper/ untouched)
    paper_v4 = REPO_ROOT / "paper" / "sections" / "06_limitations_v4.tex"
    assert not paper_v4.exists(), (
        f"USER_OK-gated paper file {paper_v4} must NOT be created "
        "without explicit user authorization (per TODO-84 + AGENTS.md §5.3)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
