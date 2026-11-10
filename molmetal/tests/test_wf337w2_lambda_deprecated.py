"""Wf-337-W2 — LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 DEPRECATED markers.

Owner: Hugo Lee
Date: 2026-09-29
Parent: Wf-337-W1 (G6 recovery sweep) + Wf-324C + AGENTS.md §5.4
        (Tier-2 re-orientation: "我们发二区就可以了，不用规划湿实验")

Context (per AGENTS.md §5.4 standing directive 2026-09-29):
  - Tier-2 journal target (Digital Discovery / J. Cheminform., IF 5-8).
  - NO wet-lab planning required for Tier-2 acceptance.
  - LAMBDA_ROADMAP §3.2 R22-P1 m18 wet-lab portion DEPRECATED → re-scoped to
    computational-only (AlphaFold-Multimer + MetalSite + in-silico docking).
  - LAMBDA_ROADMAP §3.3 R23 (Phase I clinical trial) DEPRECATED — no clinical
    planning, no IND filing, no patient cohort, no FDA/NMPA filing.
  - Wf-337-W2 anchors these markers as a test pin so that downstream agents
    (Wf-41X, Wf-42X, Wf-50X waves) cannot silently re-introduce wet-lab /
    clinical-trial milestones into R22-P1 or R23 without an explicit
    DEPRECATED reversal + user authorization.

Tests (4 total):

  - test_r22_p1_wetlab_deprecated_marker_present
        Verifies §3.2 R22-P1 carries an explicit DEPRECATED marker for the
        wet-lab portion (AGENTS.md §5.4 contract).
  - test_r23_clinical_deprecated_marker_present
        Verifies §3.3 R23 carries an explicit DEPRECATED marker for the
        clinical-trial portion (AGENTS.md §5.4 contract).
  - test_r22_r23_rescope_computational_only_compliance
        Verifies §3.2 R22-P1 re-scope is computational-only
        (AlphaFold-Multimer / MetalSite / in-silico docking) and §3.3 R23 is
        marked as historical-record-only.
  - test_tier2_strategy_no_wetlab_no_clinical
        Verifies the Tier-2 strategy compliance: paper trajectory §4.3 =
        tier-2 journal (Digital Discovery / J. Cheminform., IF 5-8) and
        NO wet-lab / NO clinical-trial milestones remain in active scope.

All tests are CPU-only — no GPU, no torch.cuda, no real training. < 5 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED (AGENTS.md §5.4, 2026-09-29): user directive "我们发二区就可以了，
    不用规划湿实验" — Tier-2 journal + NO wet-lab planning.
  - MEASURED (LAMBDA_ROADMAP.md §3.1, 2026-09-29): scope-update marker
    "R22-P1 m18 re-scoped to computational-only (no wet-lab); R23 Phase I +
    R24 Phase II + R25 Phase III = DEPRECATED (no clinical-trial planning)".
  - MEASURED (LAMBDA_ROADMAP.md §3.3, 2026-09-29): R23 ⛔ DEPRECATED header
    present (was already there from Wf-324C).
  - Wf-337-W2 ADDS (2026-09-29): explicit ⛔ DEPRECATED marker on the R22-P1
    wet-lab portion (§3.2 R22-P1 row + §3.2 header callout).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path



# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
LAMBDA_ROADMAP = ROOT / "TODO" / "03_plans" / "LAMBDA_ROADMAP.md"
AGENTS_MD = ROOT / "AGENTS.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_lambda_roadmap() -> str:
    """Return the full text of TODO/03_plans/LAMBDA_ROADMAP.md."""
    return LAMBDA_ROADMAP.read_text(encoding="utf-8")


def _extract_section(text: str, header: str) -> str:
    """Extract a numbered section body from the LAMBDA_ROADMAP, including
    any sub-sections (### §X.Y) but stopping at the next same-len or higher
    header (### §X.0 / ### §X+1).

    `header` is the substring that uniquely identifies the section header
    line, e.g. "### §3.2". The returned slice runs from that header up to
    but NOT including the next sibling (or higher) header, OR end-of-file.

    The key heuristic: for `### §3.2`, the bound is the next `### §3.X`
    with X != 2, or `### §4.0` onwards. We approximate this by looking for
    `### §` followed by a DIFFERENT major number (`§4` vs `§3`) OR the same
    major number with a DIFFERENT minor.
    """
    start = text.find(header)
    assert start >= 0, f"Section header not found: {header!r}"

    # Parse major.minor from the header (e.g. "### §3.2" → (3, 2)).
    m = re.match(r"### §(\d+)\.(\d+)", header)
    assert m is not None, f"Header does not match §X.Y pattern: {header!r}"
    major = int(m.group(1))
    minor = int(m.group(2))

    # Walk forward, tracking all `### §` headers, and return at the first one
    # whose (major, minor) is NOT a sub-section of (major, minor).
    pos = start + len(header)
    while pos < len(text):
        # Find next `### §` header.
        next_header = text.find("\n### §", pos)
        if next_header < 0:
            # No more headers; section runs to end of file.
            return text[start:]
        # Extract the (major, minor) of the next header.
        body = text[next_header + 1 :]  # skip leading "\n"
        m2 = re.match(r"### §(\d+)\.(\d+)", body)
        if m2 is None:
            # Not a §X.Y header; keep scanning.
            pos = next_header + 1
            continue
        next_major = int(m2.group(1))
        next_minor = int(m2.group(2))
        # If different major, this is a sibling of the parent — stop.
        if next_major != major:
            return text[start:next_header]
        # Same major: only stop if minor is different (sibling, not sub-section).
        if next_minor != minor:
            return text[start:next_header]
        # Same (major, minor) — should not happen since headers are unique.
        pos = next_header + 1
    return text[start:]


# ---------------------------------------------------------------------------
# Test 1: §3.2 R22-P1 wet-lab DEPRECATED marker present
# ---------------------------------------------------------------------------
def test_r22_p1_wetlab_deprecated_marker_present():
    """Verifies §3.2 R22-P1 carries an explicit DEPRECATED marker for the
    wet-lab portion, per AGENTS.md §5.4 ("R22-P1 (m18 wet-lab) DEPRECATED").

    The marker must:
      - Appear in §3.2 R22-P1 row (the m18 Multi-modal Protein-Metal binding
        ship cell).
      - Use the word "DEPRECATED" (not just "no wet-lab" or "computational-only",
        which describe the re-scope but do NOT encode the deprecation status
        that AGENTS.md §5.4 mandates).
      - Be dated 2026-09-29 (the user directive date).
      - Be paired with the explicit reason: wet-lab portion / MTT/SRB/CellTiter-Glo
        IC50 validation.
    """
    text = _load_lambda_roadmap()
    section_3_2 = _extract_section(text, "### §3.2")

    # Hard invariant: the word DEPRECATED must appear in §3.2 (not just
    # somewhere in the doc).
    assert "DEPRECATED" in section_3_2, (
        "§3.2 missing DEPRECATED marker — per AGENTS.md §5.4, the wet-lab "
        "portion of R22-P1 (m18 Multi-modal Protein-Metal binding) must be "
        "explicitly marked DEPRECATED, not just re-scoped to computational-only."
    )

    # The R22-P1 row itself must carry the DEPRECATED marker.
    # Capture the R22-P1 row up to the next "|" at the end of that table row.
    r22_p1_match = re.search(
        r"\*\*R22-P1\*\*[^\n]*",
        section_3_2,
    )
    assert r22_p1_match is not None, (
        "R22-P1 row not found in §3.2 — cannot verify DEPRECATED marker on "
        "the actual m18 Multi-modal Protein-Metal binding ship cell."
    )
    r22_p1_row = r22_p1_match.group(0)
    assert "DEPRECATED" in r22_p1_row, (
        "R22-P1 row missing DEPRECATED marker — the wet-lab portion marker "
        "must appear in the row itself (not just the section header callout)."
    )

    # Date anchor: 2026-09-29 (user directive date).
    assert "2026-09-29" in r22_p1_row, (
        "R22-P1 DEPRECATED marker missing the 2026-09-29 date anchor — "
        "the marker must trace to the user directive date for auditability."
    )

    # Reason anchor: wet-lab portion must be called out (per §5.4 wording
    # "R22-P1 (m18 wet-lab) DEPRECATED").
    assert "wet-lab" in r22_p1_row.lower(), (
        "R22-P1 DEPRECATED marker missing the wet-lab reason anchor — "
        "AGENTS.md §5.4 mandates the marker identifies what is deprecated."
    )


# ---------------------------------------------------------------------------
# Test 2: §3.3 R23 clinical DEPRECATED marker present
# ---------------------------------------------------------------------------
def test_r23_clinical_deprecated_marker_present():
    """Verifies §3.3 R23 carries an explicit DEPRECATED marker for the
    clinical-trial portion, per AGENTS.md §5.4 ("§3.3 R23 (Phase I clinical)
    DEPRECATED").

    The marker must:
      - Appear in the §3.3 header line itself.
      - Use the canonical "⛔ DEPRECATED" glyph + word (not just "no
        clinical-trial planning").
      - Be paired with "Phase I" + "clinical trial" keywords so downstream
        agents cannot accidentally undo the marker without an explicit
        reversal + user authorization.
    """
    text = _load_lambda_roadmap()
    section_3_3 = _extract_section(text, "### §3.3")

    # Hard invariant: DEPRECATED appears in §3.3.
    assert "DEPRECATED" in section_3_3, (
        "§3.3 missing DEPRECATED marker — per AGENTS.md §5.4, R23 Phase I "
        "clinical trial must be explicitly DEPRECATED."
    )

    # Header line must carry the ⛔ DEPRECATED glyph (visual signal for any
    # reader of the doc to immediately see the section is dead).
    header_line_match = re.search(r"### §3\.3[^\n]*", section_3_3)
    assert header_line_match is not None, "§3.3 header line not found"
    header_line = header_line_match.group(0)
    assert "⛔ DEPRECATED" in header_line, (
        "§3.3 header line missing ⛔ DEPRECATED glyph — must be visible in "
        "the section header so any downstream agent sees the deprecation "
        "status without having to read the body."
    )

    # Per-phase rows must each be DEPRECATED.
    r23_p_rows = re.findall(
        r"\*\*R23-P\d\*\*[^\n]*",
        section_3_3,
    )
    assert len(r23_p_rows) >= 6, (
        f"Expected 6 R23-P1..P6 rows, found {len(r23_p_rows)} — §3.3 "
        "table structure may have been edited."
    )
    for row in r23_p_rows:
        assert "DEPRECATED" in row, (
            f"R23 phase row missing DEPRECATED marker: {row[:80]!r} — "
            "all 6 R23 phases must each carry the DEPRECATED marker."
        )

    # Phase I + clinical keyword anchors.
    assert "Phase I" in section_3_3, (
        "§3.3 missing Phase I keyword — DEPRECATED marker must identify "
        "the deprecated scope (clinical Phase I, not wet-lab only)."
    )
    assert "clinical" in section_3_3.lower(), (
        "§3.3 missing 'clinical' keyword — DEPRECATED marker must identify "
        "the deprecated scope (clinical-trial planning)."
    )


# ---------------------------------------------------------------------------
# Test 3: §3.2 + §3.3 re-scope notes compliance
# ---------------------------------------------------------------------------
def test_r22_r23_rescope_computational_only_compliance():
    """Verifies the computational-only re-scope notes that complete the
    Tier-2 strategy: §3.2 R22-P1 re-scopes to AlphaFold-Multimer + MetalSite
    + in-silico docking; §3.3 R23 marks itself as historical-record-only.

    Per AGENTS.md §5.4 the "How to apply" clause requires:
      - Every metrics/by_metric/*.json + TODO/completed/*.md is Tier-2
        shippable as is (computational-only, no wet-lab data required).
      - Plan files (WEAK_TO_STRONG / IMPROVEMENT_ENUM / LAMBDA_ROADMAP)
        should re-scope wet-lab items to in-silico alternatives.
    """
    text = _load_lambda_roadmap()
    section_3_2 = _extract_section(text, "### §3.2")
    section_3_3 = _extract_section(text, "### §3.3")

    # §3.2 R22-P1 computational-only re-scope: AlphaFold-Multimer +
    # MetalSite + in-silico docking must all appear as alternative anchors.
    assert "AlphaFold-Multimer" in section_3_2, (
        "§3.2 R22-P1 re-scope missing AlphaFold-Multimer anchor — per "
        "AGENTS.md §5.4 'How to apply', the re-scope must name an in-silico "
        "alternative to wet-lab validation."
    )
    assert "MetalSite" in section_3_2, (
        "§3.2 R22-P1 re-scope missing MetalSite anchor — per AGENTS.md §5.4 "
        "'How to apply', the re-scope must name an in-silico alternative to "
        "wet-lab binding-site characterization."
    )
    assert (
        "in-silico" in section_3_2.lower()
        or "in silico" in section_3_2.lower()
        or "computational" in section_3_2.lower()
    ), (
        "§3.2 R22-P1 re-scope missing in-silico / computational keyword — "
        "the re-scope must explicitly signal 'no wet-lab' alternatives."
    )

    # §3.3 R23 historical-record-only marker: must contain "historical
    # record" or "ARCHIVED" so downstream agents do not allocate Wf-57..62.
    assert "historical record" in section_3_3.lower() or "ARCHIVED" in section_3_3, (
        "§3.3 R23 missing 'historical record' / 'ARCHIVED' marker — "
        "downstream agents reading §3.3 must immediately see 'do not allocate "
        "Wf-57..62, do not recruit wet-lab / clinical PI'."
    )

    # §3.3 R23 must explicitly forbid Wf-57..Wf-62 allocation (per §3.3 body).
    # Look for the Wf-57 + Wf-62 pair in the §3.3 body.
    wf_57_pos = section_3_3.find("Wf-57")
    wf_62_pos = section_3_3.find("Wf-62")
    assert 0 < wf_57_pos < wf_62_pos, (
        "§3.3 R23 must explicitly reference Wf-57..Wf-62 with a "
        "'do not allocate' directive — downstream agents reading §3.3 must "
        "see the forbidden workflow range."
    )


# ---------------------------------------------------------------------------
# Test 4: Tier-2 strategy compliance (paper trajectory §4.3 + NO wet-lab)
# ---------------------------------------------------------------------------
def test_tier2_strategy_no_wetlab_no_clinical():
    """Verifies the Tier-2 strategy compliance with AGENTS.md §5.4:

    1. Paper trajectory §4.3 must point to tier-2 journal (Digital Discovery /
       J. Cheminform., IF 5-8) — NOT Nat. Comput. Sci. / Nature MI tier-1.
    2. R22 + R23 active sections must be computational-only (no wet-lab
       / no clinical-trial planning in active scope).
    3. The word "DEPRECATED" must appear in both §3.2 R22-P1 wet-lab
       portion AND §3.3 R23 clinical portion (this is a defensive
       belt-and-suspenders check covering both contracts in one assertion).
    """
    text = _load_lambda_roadmap()

    # §4.3 paper trajectory must include tier-2 journal name (one of the
    # AGENTS.md §5.4-approved tier-2 targets).
    tier2_keywords = [
        "Digital Discovery",
        "J. Cheminform",
        "J. Comput. Aided Mol. Des",
        "JCAMD",
        "J. Mol. Graph. Model",
        "Comput. Chem",
        "WIREs",
        "Mol. Divers",
        "Comput. Biol. Chem",
    ]
    section_4_3 = _extract_section(text, "### §4.3")
    has_tier2 = any(kw in section_4_3 for kw in tier2_keywords)
    assert has_tier2, (
        "§4.3 paper trajectory missing tier-2 journal name — per AGENTS.md "
        "§5.4 the paper target is Digital Discovery / J. Cheminform. (tier-2 "
        "IF 5-8), NOT Nat. Comput. Sci. / Nature MI (tier-1)."
    )

    # R22 active scope must be computational-only (no wet-lab in active
    # scope, only in DEPRECATED markers). The §3.2 R22-P2/R22-P3/R22-P4/
    # R22-P5/R22-P6 rows must NOT contain 'wet-lab' except paired with
    # DEPRECATED markers.
    section_3_2 = _extract_section(text, "### §3.2")
    # Split section 3.2 into lines for row-level analysis.
    rows_3_2 = [
        line for line in section_3_2.splitlines() if line.startswith("| **R22-P")
    ]
    assert (
        len(rows_3_2) >= 6
    ), f"Expected 6 R22-P1..P6 rows in §3.2, found {len(rows_3_2)}."
    for row in rows_3_2:
        is_r22_p1 = "**R22-P1**" in row
        row_lower = row.lower()
        # Every row must NOT claim wet-lab IC50/MTT/SRB in active scope.
        wetlab_active_violations = [
            kw
            for kw in [
                "ic50 validation",
                "mtt assay",
                "srb assay",
                "celltiter-glo",
                "wet-lab ic50",
            ]
            if kw in row_lower
            and "no " + kw not in row_lower
            and "wet-lab portion" not in row_lower
        ]
        assert not wetlab_active_violations, (
            f"R22 row claims wet-lab {wetlab_active_violations!r} in active "
            f"scope (should be DEPRECATED): {row[:100]!r}"
        )

    # §3.3 R23 must be entirely DEPRECATED — no row should claim an active
    # clinical-trial / IND-filing / patient-cohort milestone in scope.
    section_3_3 = _extract_section(text, "### §3.3")
    rows_3_3 = [
        line
        for line in section_3_3.splitlines()
        if line.startswith("| ~~**R23-P") or line.startswith("| **R23-P")
    ]
    assert (
        len(rows_3_3) >= 6
    ), f"Expected 6 R23-P1..P6 rows in §3.3, found {len(rows_3_3)}."
    for row in rows_3_3:
        assert "DEPRECATED" in row or "⛔" in row, (
            f"R23 row missing DEPRECATED marker (must be ALL DEPRECATED per "
            f"AGENTS.md §5.4): {row[:100]!r}"
        )

    # Belt-and-suspenders: both §3.2 R22-P1 wet-lab DEPRECATED marker AND
    # §3.3 R23 clinical DEPRECATED marker must be present (tests 1+2 covered
    # each separately; this is a single-pass regression net).
    assert "DEPRECATED" in section_3_2 and "DEPRECATED" in section_3_3, (
        "Cross-section regression — both §3.2 R22-P1 wet-lab AND §3.3 R23 "
        "clinical DEPRECATED markers must be present in the same LAMBDA_ROADMAP."
    )
