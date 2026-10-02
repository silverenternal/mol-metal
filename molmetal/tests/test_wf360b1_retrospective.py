"""Test for Wf-360-B1: Per-Wf-296..359 retrospective.

Verifies that the retrospective document exists and has the 6 required sections.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOC_PATH = DOCS_DIR / "WF296_TO_WF359_RETROSPECTIVE.md"

REQUIRED_SECTIONS = [
    "## §0 TL;DR",
    "## §1 Per-Wf summary",
    "## §2 What worked",
    "## §3 What didn't",
    "## §4 Lessons learned",
    "## §5 Forward-looking improvements",
    "## §6 Cross-references",
]


def test_retrospective_doc_exists_with_six_sections():
    """Verify Wf-360-B1 retrospective exists and has 6 required sections (§1..§6)."""
    assert DOC_PATH.exists(), f"Retrospective not found at {DOC_PATH}"
    assert DOCS_DIR.exists(), f"docs/ dir not found at {DOCS_DIR}"

    content = DOC_PATH.read_text(encoding="utf-8")

    # Check all 7 required section headings (§0 + §1..§6 = 7 headings total).
    missing = [s for s in REQUIRED_SECTIONS if s not in content]
    assert not missing, f"Missing required sections in {DOC_PATH.name}: {missing}"

    # Spot-check key content anchors.
    assert "Wf-296..359 Retrospective" in content
    assert "MEASURED-vs-PROJECTED" in content
    assert "honest-negative" in content.lower()
    assert "AGENTS.md" in content
    assert "Wf-335-SYN" in content
    assert "Wf-340" in content
    assert "Wf-358-A3" in content
    assert "Wf-359-A1" in content


if __name__ == "__main__":
    test_retrospective_doc_exists_with_six_sections()
    print("PASS: Wf-360-B1 retrospective has all 6 required sections (§1..§6) + §0 TL;DR.")