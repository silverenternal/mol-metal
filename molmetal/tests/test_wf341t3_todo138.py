"""Wf-341-T3 — TODO-138 Paper v2 Tier-2 re-scope plan verification tests.

Per AGENTS.md §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验")
+ Wf-337-W3 (Tier-2 marker pass) + Wf-341-T3 (Tier-2 re-scope completion PLAN).

This Wf completes the Tier-2 re-scope plan for paper §4 LEAD claims + §4.5
honest-negative disclosures + §6 limitations. Per TODO-84 (AGENTS.md §5.3),
the actual `paper/*.tex` writes are BLOCKED without user authorization —
this Wf delivers the **PLAN** as a TODO/completed/ doc, not the
paper/* edits.

These tests verify that the re-scope PLAN document exists with the 4
reorganization axes:
  1. Tier-2 venue selection (JCAMD primary)
  2. Paper §4 LEAD claims reorganization (Tier-2 tone)
  3. Paper §4.5 honest-negative disclosures reorganization (dedicated subsection)
  4. Paper §6 limitations reorganization (computational-only by design)

CPU-only; no GPU launch; no paper/* writes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/hugo/codes/mol-metal")

PLAN_PATH = REPO_ROOT / "TODO" / "completed" / "341t3_todo138_tier2_rescope_plan.md"
TODO_138_PATH = REPO_ROOT / "TODO" / "completed" / "138_paper_v2_submission_package.md"


def _read(path: Path) -> str:
    """Read a file as UTF-8 text, fail loudly if missing."""
    assert path.exists(), f"Missing required file: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Tier-2 venue selection (JCAMD primary)
# ---------------------------------------------------------------------------

def test_tier2_venue_selection():
    """Tier-2 venue must be selected with JCAMD as primary.

    Per AGENTS.md §5.4: JCAMD (J. Comput. Aided Mol. Des.) is the recommended
    primary Tier-2 venue. The plan must explicitly name JCAMD + at least
    2 backup venues (WIREs / Mol. Divers. / J. Mol. Graph. Model. /
    Comput. Biol. Chem.).
    """
    text = _read(PLAN_PATH)
    # Must name JCAMD as primary venue
    assert "JCAMD" in text, (
        "Plan must name JCAMD as the primary Tier-2 venue. "
        "Per AGENTS.md §5.4: JCAMD is the recommended Tier-2 target."
    )
    # Must name at least 2 backup venues
    backup_venues = [
        "WIREs",
        "Mol. Divers.",
        "J. Mol. Graph. Model.",
        "Comput. Biol. Chem.",
    ]
    found_backups = [v for v in backup_venues if v in text]
    assert len(found_backups) >= 2, (
        f"Plan must name at least 2 backup Tier-2 venues. "
        f"Found only: {found_backups}. Expected ≥ 2 from {backup_venues}."
    )
    # Must reject JCIM (Tier-1 downgrade)
    assert "JCIM" in text, (
        "Plan must explicitly address the JCIM (Tier-1) downgrade rationale."
    )
    # Must cite AGENTS.md §5.4
    assert "§5.4" in text or "5.4" in text, (
        "Plan must cross-reference AGENTS.md §5.4 (Tier-2 directive)."
    )


# ---------------------------------------------------------------------------
# Test 2: Paper §4 LEAD claims reorganization
# ---------------------------------------------------------------------------

def test_section4_lead_claims_reorganization():
    """Paper §4 LEAD claims reorganization must be Tier-2 tone.

    Tier-2 framing replaces 'SOTA-beating / on-par' with 'consistent with
    literature + methodological contribution'. The plan must contain:
    - 8 reworded §4.x entries (one per LEAD claim)
    - Tier-2 tone keywords ('consistent with', 'in line with')
    - Methodological contribution framing
    - Removed 'SOTA-beating' / 'beat iMoLD' language
    """
    text = _read(PLAN_PATH)
    # Must have 8 §4.x entries (one per LEAD claim)
    section4_matches = re.findall(r"§4\.\d", text)
    section4_unique = set(section4_matches)
    assert len(section4_unique) >= 8, (
        f"Plan must have ≥ 8 §4.x entries (one per LEAD claim). "
        f"Found {len(section4_unique)} unique §4.x: {section4_unique}"
    )
    # Must use Tier-2 tone keywords
    tier2_tone = ["consistent with", "in line with", "comparable to"]
    missing_tone = [kw for kw in tier2_tone if kw not in text]
    assert not missing_tone, (
        f"Plan missing Tier-2 tone keywords: {missing_tone}. "
        f"Expected at least one of: {tier2_tone}."
    )
    # Must address removal of SOTA-beating language
    removal_section = (
        "removed" in text.lower() or "reworded" in text.lower() or
        "what gets removed" in text.lower()
    )
    assert removal_section, (
        "Plan must explicitly address removal of 'SOTA-beating' language "
        "from §4 LEAD claims."
    )
    # Methodological contribution framing
    assert "methodological contribution" in text or "methodological" in text.lower(), (
        "Plan must emphasize methodological contribution framing."
    )


# ---------------------------------------------------------------------------
# Test 3: Paper §4.5 honest-negative disclosures reorganization
# ---------------------------------------------------------------------------

def test_section45_honest_negative_reorganization():
    """Paper §4.5 honest-negative disclosures must be reorganized as a
    dedicated subsection between LEADS and MAIN RESULTS.

    Per AGENTS.md §5.4 + Wf-341-T3 brief: Tier-2 framing promotes honest-
    negative from §6 (after-the-fact) to dedicated §4.5 subsection. The
    plan must:
    - Contain 8 honest-negative entries (8 MEASURED values)
    - Document lift attempts (1-3 Wf per metric)
    - Cross-link to §6 limitations
    - Use HONEST_NEGATIVE verdict labels
    """
    text = _read(PLAN_PATH)
    # Must have 8 §4.5.x entries
    section45_matches = re.findall(r"§4\.5\.\d", text)
    section45_unique = set(section45_matches)
    assert len(section45_unique) >= 8, (
        f"Plan must have ≥ 8 §4.5.x entries (one per honest-negative). "
        f"Found {len(section45_unique)} unique §4.5.x: {section45_unique}"
    )
    # Must use HONEST_NEGATIVE verdict labels
    honest_neg_count = text.count("HONEST_NEGATIVE")
    assert honest_neg_count >= 5, (
        f"Plan must use HONEST_NEGATIVE verdict labels (≥5 instances). "
        f"Found {honest_neg_count}."
    )
    # Must document lift attempts
    assert "lift attempts" in text.lower() or "rescue" in text.lower(), (
        "Plan must document lift attempts per honest-negative."
    )
    # Must cross-link to §6 limitations
    assert "§6" in text, (
        "Plan must cross-link §4.5 honest-negative to §6 limitations."
    )
    # Must include "Methodological limitations acknowledged up-front" framing
    assert "Methodological limitations" in text or "methodological due diligence" in text.lower(), (
        "Plan must use 'Methodological limitations acknowledged up-front' framing "
        "for §4.5 lead-in."
    )


# ---------------------------------------------------------------------------
# Test 4: Paper §6 limitations reorganization
# ---------------------------------------------------------------------------

def test_section6_limitations_reorganization():
    """Paper §6 limitations must be reorganized to Tier-2 acceptance bar.

    Tier-2 framing repositions §6 from 'defensive limitations' to 'scope
    declaration + computational methodology boundary'. The plan must:
    - Contain 8+ §6.x entries (computational-only scope + per-honest-neg
      cross-link)
    - Use 'computational-only by design' framing
    - Reference Tier-2 acceptance bar explicitly
    - Cite TODO-84 / AGENTS.md §5.3 (no paper/* without authorization)
    """
    text = _read(PLAN_PATH)
    # Must have 8+ §6.x entries
    section6_matches = re.findall(r"§6\.\d+", text)
    section6_unique = set(section6_matches)
    assert len(section6_unique) >= 8, (
        f"Plan must have ≥ 8 §6.x entries. "
        f"Found {len(section6_unique)} unique §6.x: {section6_unique}"
    )
    # Must use 'computational-only by design' framing
    assert "computational-only by design" in text.lower() or "computational-only" in text.lower(), (
        "Plan must use 'computational-only by design' framing for §6."
    )
    # Must reference Tier-2 acceptance bar
    tier2_acceptance = ["Tier-2 acceptance", "Tier-2 acceptance bar", "tier-2 acceptance"]
    has_tier2_acceptance = any(kw.lower() in text.lower() for kw in tier2_acceptance)
    assert has_tier2_acceptance, (
        f"Plan must reference Tier-2 acceptance bar explicitly. "
        f"Expected one of: {tier2_acceptance}."
    )
    # Must cite TODO-84 / AGENTS.md §5.3 (paper rule)
    paper_rule = ["TODO-84", "§5.3", "5.3"]
    has_paper_rule = any(kw in text for kw in paper_rule)
    assert has_paper_rule, (
        f"Plan must cite TODO-84 / AGENTS.md §5.3 (paper rule). "
        f"Expected one of: {paper_rule}."
    )
    # Must explicitly state no wet-lab gate
    no_wetlab = [
        "no wet-lab gate",
        "no wet-lab validation",
        "wet-lab validation planned",
        "wet-lab is",
    ]
    has_no_wetlab = any(kw.lower() in text.lower() for kw in no_wetlab)
    assert has_no_wetlab, (
        f"Plan must explicitly state Tier-2 acceptance has no wet-lab gate. "
        f"Expected one of: {no_wetlab}."
    )


if __name__ == "__main__":
    # Allow `python -m molmetal.tests.test_wf341t3_todo138` smoke run
    import sys
    sys.exit(pytest.main([__file__, "-v"]))