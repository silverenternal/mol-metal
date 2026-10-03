"""Wf-365-A3 — m6_kappa FINAL verdict documentation test.

This test verifies the FINAL m6_kappa verdict document exists at
`docs/M6_KAPPA_FINAL_VERDICT.md`, has 7 sections, and covers all 5
architectural attempts (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 +
Wf-364-A2).

Per Wf-365-A3 brief:
- 1 NEW test
- verdict doc exists + has 7 sections + covers all 3-5 attempts
- CPU-only; per AGENTS.md §5.1

Test owner: Hugo Lee + Wf-365-A3 ultracode agent.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path("/home/hugo/codes/mol-metal")
VERDICT_DOC_PATH = REPO_ROOT / "docs" / "M6_KAPPA_FINAL_VERDICT.md"


def _read_verdict_doc() -> str:
    """Read the FINAL verdict doc; raise if not present."""
    if not VERDICT_DOC_PATH.exists():
        raise FileNotFoundError(
            f"Wf-365-A3 verdict doc not found at {VERDICT_DOC_PATH}. "
            "Per Wf-365-A3 brief, this doc must exist with 7 sections "
            "covering all 5 architectural attempts (Wf-335-M1, Wf-359-A1, "
            "Wf-361-B1, Wf-361-B2, Wf-364-A2)."
        )
    return VERDICT_DOC_PATH.read_text(encoding="utf-8")


def test_wf365a3_m6_kappa_verdict() -> None:
    """Verify Wf-365-A3 FINAL m6_kappa verdict doc exists + structure OK."""
    text = _read_verdict_doc()

    # Section coverage check: must have at least 7 top-level sections (§1..§7+)
    section_pattern = re.compile(r"^##\s*§\d+", re.MULTILINE)
    sections_found = section_pattern.findall(text)
    assert len(sections_found) >= 7, (
        f"Wf-365-A3 verdict doc must have >= 7 sections; "
        f"found {len(sections_found)}: {sections_found}"
    )

    # Per-attempt coverage check: must reference all 5 architectural workflows
    expected_attempts = [
        ("Wf-335-M1", "chemprop_plus_pll structural ceiling"),
        ("Wf-359-A1", "R20-P3 W2 verification"),
        ("Wf-361-B1", "MAML cross-scaffold meta-learning"),
        ("Wf-361-B2", "Neural ODE continuous dynamics"),
        ("Wf-364-A2", "combined MAML + Neural ODE (LAST)"),
    ]

    for wf_tag, description in expected_attempts:
        assert wf_tag in text, (
            f"Wf-365-A3 verdict doc must reference {wf_tag} ({description}); "
            "not found in docs/M6_KAPPA_FINAL_VERDICT.md"
        )

    # Final verdict must be preserved: HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL
    assert "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in text, (
        "Wf-365-A3 verdict doc must preserve HONEST_NEGATIVE_PARTIAL_STRUCTURAL "
        "verdict from Wf-364-A2 §8 final state."
    )

    # Gate threshold (0.6) must be mentioned
    assert "0.6" in text, (
        "Wf-365-A3 verdict doc must reference the 0.6 gate threshold per "
        "Wf-252 / ChemAxon PMC11387870."
    )

    # Architectural lift values must trace to MEASURED JSON paths
    assert "+0.6814" in text, (
        "Wf-365-A3 verdict doc must cite Wf-335-M1 chemprop+PLL CV-R² +0.6814 "
        "MEASURED on the 60-SMILES smoke cohort."
    )

    assert "0.4737" in text, (
        "Wf-365-A3 verdict doc must cite Wf-364-A2 joint candidate aug_mean "
        "0.4737 (best MEASURED lift across all 5 attempts)."
    )

    # Forward-looking section + Tier-2 §6 entry prep must be present
    assert "Forward-looking" in text or "forward-looking" in text, (
        "Wf-365-A3 verdict doc must include forward-looking (USER_OK-gated GPU retrain) section."
    )
    assert "§6" in text or "section 6" in text.lower(), (
        "Wf-365-A3 verdict doc must include paper §6 honest-limitations entry prep."
    )

    # No paper/* touches (TODO-84 active) — allow the literal `paper/*` reference
    # since the doc explicitly says "no touches to paper/*" as standing rule text.
    paper_paths_referenced = re.findall(r"`paper/[^*\s][^`]*`", text)
    assert len(paper_paths_referenced) == 0, (
        f"Wf-365-A3 verdict doc must NOT touch paper/* per TODO-84; "
        f"found concrete paper/* file references: {paper_paths_referenced}"
    )

    print(
        "[Wf-365-A3] verdict doc verified: "
        f"sections={len(sections_found)}, attempts=5, "
        "verdict=HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL"
    )
