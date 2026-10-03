"""
Wf-375-A3 LAMBDA_ROADMAP DEPRECATED-marker verification test.

Verifies that TODO/03_plans/LAMBDA_ROADMAP.md carries the explicit DEPRECATED
markers required by AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning,
per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验"):

- §3.2 R22-P1 (m18 Multi-modal Protein-Metal binding) wet-lab portion is
  re-scoped to **computational-only** (AlphaFold-Multimer + MetalSite in-silico,
  NO wet-lab validation).
- §3.3 R23 (Phase I clinical trial registration + IND filing + GLP tox)
  is explicitly ⛔ DEPRECATED — OUT OF SCOPE per user directive.

Per Wf-375-A3 task brief:
- 1 NEW test file with multiple contract tests pinning the DEPRECATED markers
- Wall: PROJECTED 5-10 min CPU-only
- No GPU launches; no paper/* touches; no source-code edits to LAMBDA_ROADMAP.md.

Skill alignment: party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)
+ AGENTS.md §5.1 (CPU-only) + §5.3 (no paper work, TODO-84 active) + §5.4
(Tier-2 + NO wet-lab planning).
"""

import re
from pathlib import Path


REPO_ROOT = Path("/home/hugo/codes/mol-metal")
LAMBDA_ROADMAP = REPO_ROOT / "TODO" / "03_plans" / "LAMBDA_ROADMAP.md"


def _read_text() -> str:
    assert LAMBDA_ROADMAP.exists(), f"Missing: {LAMBDA_ROADMAP}"
    return LAMBDA_ROADMAP.read_text(encoding="utf-8")


def test_01_lambda_roadmap_file_exists() -> None:
    """TODO/03_plans/LAMBDA_ROADMAP.md is present (pre-flight)."""
    assert LAMBDA_ROADMAP.exists(), f"Missing: {LAMBDA_ROADMAP}"
    text = _read_text()
    # The doc is the canonical 6-year R20-R25 Lambda algorithm roadmap.
    assert "Lambda Algorithm Roadmap" in text, "Missing Lambda Algorithm Roadmap title"
    assert "R20-R25" in text, "Missing R20-R25 scope marker"


def test_02_section_3_2_r22_p1_rescoped_to_computational_only() -> None:
    """§3.2 R22-P1 (m18 Multi-modal Protein-Metal binding) is re-scoped to computational-only.

    Per AGENTS.md §5.4 (Tier-2 + NO wet-lab planning, 2026-09-29 user directive),
    the R22-P1 wet-lab validation (MTT/SRB/CellTiter-Glo IC50 + cellular uptake
    + binding affinity wet-lab assays) MUST be re-scoped to computational-only
    with an explicit DEPRECATED marker for the wet-lab portion.
    """
    text = _read_text()

    # Locate §3.2 header
    section_3_2_match = re.search(
        r"^### §3\.2 R22 详细[^\n]*",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_2_match, "Missing §3.2 R22 section header"

    # Locate next §3.3 to bound the §3.2 region
    section_3_3_match = re.search(
        r"^### §3\.3",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_3_match, "Missing §3.3 R23 section header (cannot bound §3.2)"

    section_3_2_text = text[section_3_2_match.start():section_3_3_match.start()]

    # The §3.2 header itself must carry the computational-only marker.
    assert "computational-only" in section_3_2_text, (
        "§3.2 missing 'computational-only' marker in section header"
    )
    assert "no wet-lab" in section_3_2_text, (
        "§3.2 missing 'no wet-lab' qualifier in section header"
    )

    # The R22-P1 wet-lab portion must be explicitly DEPRECATED with re-scope statement.
    # We require either the literal "⛔ DEPRECATED" or "DEPRECATED 2026-09-29" inside §3.2.
    assert (
        "⛔ DEPRECATED" in section_3_2_text
        or "DEPRECATED 2026-09-29" in section_3_2_text
    ), "§3.2 missing explicit DEPRECATED marker for R22-P1 wet-lab portion"

    # The R22-P1 table row must reference re-scope + computational-only.
    r22_p1_row_match = re.search(
        r"\|\s*\*\*R22-P1\*\*[^|]*\|[^|]*\|[^|]*\|",
        section_3_2_text,
    )
    assert r22_p1_row_match, "Missing R22-P1 table row in §3.2"

    r22_p1_row = r22_p1_row_match.group(0)
    assert "wet-lab portion" in r22_p1_row or "wet-lab" in r22_p1_row, (
        "R22-P1 row missing wet-lab qualifier"
    )
    assert (
        "DEPRECATED" in r22_p1_row or "⛔" in r22_p1_row
    ), "R22-P1 row missing DEPRECATED marker"
    assert "computational-only" in r22_p1_row, (
        "R22-P1 row missing 'computational-only' re-scope"
    )


def test_03_section_3_3_r23_deprecated_clinical_trial() -> None:
    """§3.3 R23 (Phase I clinical trial) is explicitly DEPRECATED.

    Per AGENTS.md §5.4 (Tier-2 + NO wet-lab planning, 2026-09-29 user directive),
    R23 Phase I clinical trial registration + IND filing + GLP toxicology +
    patient cohort + DSMB review + Phase I close-out are OUT OF SCOPE.
    """
    text = _read_text()

    # Locate §3.3 header
    section_3_3_match = re.search(
        r"^### §3\.3 R23 详细[^\n]*",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_3_match, "Missing §3.3 R23 section header"

    # Locate next §3.4 to bound the §3.3 region
    section_3_4_match = re.search(
        r"^### §3\.4",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_4_match, "Missing §3.4 R24 section header (cannot bound §3.3)"

    section_3_3_text = text[section_3_3_match.start():section_3_4_match.start()]

    # The §3.3 header must carry the DEPRECATED marker.
    assert "⛔ DEPRECATED" in section_3_3_text, (
        "§3.3 header missing '⛔ DEPRECATED' marker"
    )
    assert "Phase I" in section_3_3_text or "clinical trial" in section_3_3_text, (
        "§3.3 missing Phase I / clinical trial context"
    )

    # The first body blockquote must re-state the DEPRECATED status.
    assert "DEPRECATED 2026-09-29" in section_3_3_text, (
        "§3.3 missing explicit 'DEPRECATED 2026-09-29' marker in body"
    )

    # All 6 R23-P1..R23-P6 rows must carry the DEPRECATED marker (strikethrough).
    r23_phase_rows = re.findall(
        r"\|\s*~?~?\*\*R23-P\d\*\*~?~?[^\n]*",
        section_3_3_text,
    )
    assert len(r23_phase_rows) == 6, (
        f"Expected 6 R23-P1..P6 rows, got {len(r23_phase_rows)}"
    )
    for row in r23_phase_rows:
        assert "DEPRECATED" in row or "⛔" in row, (
            f"R23 row missing DEPRECATED marker: {row[:80]}..."
        )


def test_04_section_3_1_anchors_r22_r23_r24_r25_deprecation() -> None:
    """§3.1 R22-R25 strategic 4-stage table anchors DEPRECATED status for all 3 clinical rounds.

    Per AGENTS.md §5.4, R23 Phase I + R24 Phase II + R25 Phase III = DEPRECATED.
    R22-P1 m18 = computational-only re-scope. R22 m19/m20 + R22-P4/P5 remain ACTIVE.
    """
    text = _read_text()

    # Locate §3.1
    section_3_1_match = re.search(
        r"^### §3\.1",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_1_match, "Missing §3.1 R22-R25 strategic section header"

    section_3_2_match = re.search(
        r"^### §3\.2",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_2_match, "Missing §3.2 (cannot bound §3.1)"

    section_3_1_text = text[section_3_1_match.start():section_3_2_match.start()]

    # The §3.1 SCOPE UPDATE blockquote must reference the 2026-09-29 user directive.
    assert (
        "2026-09-29" in section_3_1_text
    ), "§3.1 missing 2026-09-29 timestamp marker"
    assert "我们发二区就可以了" in section_3_1_text, (
        "§3.1 missing user directive verbatim quote"
    )
    assert "不用规划湿实验" in section_3_1_text, (
        "§3.1 missing '不用规划湿实验' qualifier from user directive"
    )

    # R22 row: re-scoped to computational-only (not full DEPRECATED)
    r22_row = re.search(r"\*\*R22\*\*\s+\([^\n]*", section_3_1_text)
    assert r22_row, "Missing R22 strategic row in §3.1"

    # R23 row: DEPRECATED — clinical trial + Phase I
    assert "R23" in section_3_1_text, "§3.1 missing R23 reference"
    r23_row_match = re.search(
        r"\*\*R23\*\*[^\n]*",
        section_3_1_text,
    )
    assert r23_row_match, "Missing R23 strategic row in §3.1"
    assert (
        "DEPRECATED" in r23_row_match.group(0)
    ), "R23 row missing DEPRECATED marker"

    # R24 row: DEPRECATED — Phase II/III
    r24_row_match = re.search(
        r"\*\*R24\*\*[^\n]*",
        section_3_1_text,
    )
    assert r24_row_match, "Missing R24 strategic row in §3.1"
    assert (
        "DEPRECATED" in r24_row_match.group(0)
    ), "R24 row missing DEPRECATED marker"

    # R25 row: DEPRECATED — post-trial paper + algorithm convergence
    r25_row_match = re.search(
        r"\*\*R25\*\*[^\n]*",
        section_3_1_text,
    )
    assert r25_row_match, "Missing R25 strategic row in §3.1"
    assert (
        "DEPRECATED" in r25_row_match.group(0)
    ), "R25 row missing DEPRECATED marker"


def test_05_section_3_4_and_3_5_also_deprecated() -> None:
    """§3.4 R24 + §3.5 R25 also DEPRECATED — sanity check on full deprecation sweep.

    AGENTS.md §5.4 deprecates ALL clinical-trial rounds. The LAMBDA_ROADMAP
    must carry consistent DEPRECATED markers across §3.3 + §3.4 + §3.5.
    """
    text = _read_text()

    # §3.4 R24
    section_3_4_match = re.search(
        r"^### §3\.4[^\n]*",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_4_match, "Missing §3.4 R24 section header"

    section_3_5_match = re.search(
        r"^### §3\.5",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_5_match, "Missing §3.5 R25 section header (cannot bound §3.4)"

    section_3_4_text = text[section_3_4_match.start():section_3_5_match.start()]
    assert "⛔ DEPRECATED" in section_3_4_text, (
        "§3.4 R24 missing '⛔ DEPRECATED' marker"
    )
    assert "Phase II/III" in section_3_4_text or "Phase II" in section_3_4_text, (
        "§3.4 R24 missing Phase II/III context"
    )

    # §3.5 R25 — bound by §3.6
    section_3_6_match = re.search(
        r"^### §3\.6",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_6_match, "Missing §3.6 (cannot bound §3.5)"

    section_3_5_text = text[section_3_5_match.start():section_3_6_match.start()]
    assert "⛔ DEPRECATED" in section_3_5_text, (
        "§3.5 R25 missing '⛔ DEPRECATED' marker"
    )
    assert "Phase III" in section_3_5_text, "§3.5 R25 missing Phase III context"


def test_06_lambda_roadmap_v3_2_cross_refs_agendots_5_4() -> None:
    """§3.2 explicitly cross-references AGENTS.md §5.4 (Tier-2 + NO wet-lab planning)."""
    text = _read_text()

    # The §3.2 blockquote must cross-ref AGENTS.md §5.4 OR the 2026-09-29 user directive.
    section_3_2_match = re.search(
        r"^### §3\.2[^\n]*",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_2_match, "Missing §3.2 R22 section header"

    section_3_3_match = re.search(
        r"^### §3\.3",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_3_match, "Missing §3.3 (cannot bound §3.2)"

    section_3_2_text = text[section_3_2_match.start():section_3_3_match.start()]

    # Either explicit AGENTS.md §5.4 cross-ref OR 2026-09-29 user directive quote is acceptable.
    has_agendots_ref = "AGENTS.md §5.4" in section_3_2_text
    has_directive_ref = "2026-09-29" in section_3_2_text and (
        "我们发二区就可以了" in section_3_2_text
        or "wet-lab" in section_3_2_text.lower()
    )
    assert has_agendots_ref or has_directive_ref, (
        "§3.2 R22 missing cross-ref to AGENTS.md §5.4 or 2026-09-29 user directive"
    )


def test_07_lambda_roadmap_v3_3_cross_refs_agendots_5_4() -> None:
    """§3.3 explicitly cross-references AGENTS.md §5.4 (Tier-2 + NO wet-lab planning)."""
    text = _read_text()

    section_3_3_match = re.search(
        r"^### §3\.3[^\n]*",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_3_match, "Missing §3.3 R23 section header"

    section_3_4_match = re.search(
        r"^### §3\.4",
        text,
        flags=re.MULTILINE,
    )
    assert section_3_4_match, "Missing §3.4 (cannot bound §3.3)"

    section_3_3_text = text[section_3_3_match.start():section_3_4_match.start()]

    has_directive_ref = "2026-09-29" in section_3_3_text and (
        "我们发二区就可以了" in section_3_3_text
        or "wet-lab" in section_3_3_text.lower()
    )
    assert has_directive_ref, (
        "§3.3 R23 missing cross-ref to 2026-09-29 user directive"
    )


if __name__ == "__main__":
    test_01_lambda_roadmap_file_exists()
    test_02_section_3_2_r22_p1_rescoped_to_computational_only()
    test_03_section_3_3_r23_deprecated_clinical_trial()
    test_04_section_3_1_anchors_r22_r23_r24_r25_deprecation()
    test_05_section_3_4_and_3_5_also_deprecated()
    test_06_lambda_roadmap_v3_2_cross_refs_agendots_5_4()
    test_07_lambda_roadmap_v3_3_cross_refs_agendots_5_4()
    print("All Wf-375-A3 LAMBDA_ROADMAP DEPRECATED-marker verification tests PASS")
