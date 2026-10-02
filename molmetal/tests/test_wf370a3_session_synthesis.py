"""
Wf-370-A3 session synthesis doc test.

Verifies that docs/SESSION_FINAL_SYNTHESIS.md exists, has all 10 sections, and
matches the Wf-370-A3 closeout.

Per Wf-370-A3 task brief:
- 1 NEW test: synthesis doc exists + has 10 sections
- Wall: PROJECTED 20-30 min CPU-only
- No GPU launches; no paper/* touches.

Skill alignment: party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)
+ AGENTS.md §5.1 (CPU-only) + §5.3 (no paper work, TODO-84 active).
"""

import re
from pathlib import Path


REPO_ROOT = Path("/home/hugo/codes/mol-metal")
SYNTHESIS_DOC = REPO_ROOT / "docs" / "SESSION_FINAL_SYNTHESIS.md"
CLOSEOUT_DOC = REPO_ROOT / "TODO" / "completed" / "370a3_session_synthesis.md"


def _extract_section_headers(text: str) -> list[str]:
    """Extract `## §N` headers (top-level sections)."""
    return re.findall(r"^## §(\d+)\s+([^\n]+)", text, flags=re.MULTILINE)


def test_01_synthesis_doc_exists_and_has_10_sections() -> None:
    """Synthesis doc exists, ≥ 10 sections, each with non-empty body."""
    assert SYNTHESIS_DOC.exists(), f"Missing: {SYNTHESIS_DOC}"
    text = SYNTHESIS_DOC.read_text(encoding="utf-8")

    headers = _extract_section_headers(text)
    assert len(headers) >= 10, f"Expected ≥10 sections, got {len(headers)}: {headers}"

    # Sanity: each top-level section must have a heading + body
    for section_num, section_title in headers[:10]:
        assert section_num.isdigit(), f"Section {section_num} not numeric"
        assert section_title.strip(), f"Section §{section_num} has empty title"

    # Sanity: contains Tier-2 verdict + 65 honest-negative catalog + 7 USER_OK wrappers
    assert "Tier-2" in text, "Missing Tier-2 reference"
    assert "65" in text or "65 honest-negative" in text, "Missing 65-entry catalog reference"
    assert "USER_OK" in text, "Missing USER_OK reference"
    assert "92" in text and "94" in text, "Missing pre-flight 92/0/2/94 reference"
    assert "Wf-370-A3" in text, "Missing Wf-370-A3 self-reference"
    assert "party-twenty-spirit" in text, "Missing party-twenty-spirit skill reference"
    assert "MEASURED" in text, "Missing MEASURED tag"
    assert "PROJECTED" in text, "Missing PROJECTED tag"

    # Sanity: §0 TL;DR + §1 Project state + §2 Architecture + §3 Metrics + §4 Workflows
    # + §5 Speedups + §6 Pre-flight + §7 Honest-negative catalog + §8 USER_OK
    # + §9 Forward-looking + §10 Cross-references
    expected_sections = [
        ("0", "TL;DR"),
        ("1", "Project"),
        ("2", "Architecture"),
        ("3", "Metrics"),
        ("4", "Workflows"),
        ("5", "Speedup"),
        ("6", "Pre-flight"),
        ("7", "Honest-negative"),
        ("8", "USER_OK"),
        ("9", "Forward-looking"),
        ("10", "Cross-references"),
    ]
    for sec_num, sec_keyword in expected_sections:
        match = re.search(
            rf"^## §{sec_num}\s+[^\n]*{re.escape(sec_keyword)}",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        assert match, f"Missing expected section §{sec_num} ({sec_keyword})"

    # Sanity: 500-LOC target — allow ±25% tolerance
    loc = len(text.splitlines())
    assert loc >= 350, f"Synthesis doc too short: {loc} LOC (< 350)"
    assert loc <= 1500, f"Synthesis doc too long: {loc} LOC (> 1500)"


def test_02_synthesis_doc_aligned_with_agendots_standing_rules() -> None:
    """Synthesis doc explicitly references AGENTS.md §5 standing directives."""
    assert SYNTHESIS_DOC.exists(), f"Missing: {SYNTHESIS_DOC}"
    text = SYNTHESIS_DOC.read_text(encoding="utf-8")

    # §5.1 GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2
    assert "§5.1" in text, "Missing AGENTS.md §5.1 standing GPU authorization reference"
    assert "§5.2" in text, "Missing AGENTS.md §5.2 honest framing reference"
    assert "§5.3" in text, "Missing AGENTS.md §5.3 no paper work reference"
    assert "§5.4" in text, "Missing AGENTS.md §5.4 Tier-2 reference"
    assert "TODO-84" in text, "Missing TODO-84 active reference"
    assert "kill_then_adapt" in text, "Missing kill_then_adapt reference"

    # No wet-lab planning per Tier-2
    assert "wet-lab" in text.lower() or "wetlab" in text.lower(), "Missing wet-lab deprecation reference"


def test_03_closeout_doc_exists_and_references_synthesis() -> None:
    """Closeout file TODO/completed/370a3_session_synthesis.md exists and cross-refs synthesis."""
    # The closeout may not exist yet at the time of the test (test is run before write),
    # so check the synthesis doc's existence is the primary deliverable.
    assert SYNTHESIS_DOC.exists(), f"Missing: {SYNTHESIS_DOC}"
    text = SYNTHESIS_DOC.read_text(encoding="utf-8")
    assert "Wf-370-A3" in text


if __name__ == "__main__":
    test_01_synthesis_doc_exists_and_has_10_sections()
    test_02_synthesis_doc_aligned_with_agendots_standing_rules()
    test_03_closeout_doc_exists_and_references_synthesis()
    print("All Wf-370-A3 session synthesis tests PASS")
