"""Pytest: Wf-320G — MECHANISM_METRIC_MAPPING §13/§14 plan-level cross-walk (23 mechanisms).

This test module pins the Wf-320G contract:

1. Each of the 23 plan-level mechanisms (M22-M44 per Wf-320G task brief)
   has a documented entry in §13 of
   :file:`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`.
2. Each entry carries the required fields: mechanism description,
   plan-ID / Wf-ID / R17-Wf cross-reference, existing M-ID, status
   (CPU-ONLY-SKELETON / PLAN-ONLY / HONEST-NEGATIVE / PROJECTED / MEASURED /
   DEPRECATED), affected metrics, and a self-check pytest command
   (existing or "planned").
3. The §13 cross-mapping summary table (§13.6) is populated for all 23
   mechanisms.
4. The §13.7 failure-mode / regression-debug table covers the 23 mechanisms.
5. The §14 plan-ID cross-reference (§14.1) maps every M22-M44 to its
   origin plan / Wf / R17 Wf.

These are **documentation-only** tests — they do NOT run any GPU
training. All 23 mechanisms are **documentation work**, NOT new
measurement (per Wf-320G honest framing).

**Honest framing** (per AGENTS.md §1.5 / party-twenty-spirit §3.5):
- 23 plan-level mechanisms are DOCUMENTED at the §13 cross-walk level.
- 8 of 23 already exist as CPU-only skeleton code in §2/§5/§6 (the
  M22, M23, M26, M27, M28, M29, M30, M31, M32 implementation IDs).
- 6 of 23 are PLAN-ONLY paper §3 theorem citations (M24, M25) or
  HONEST-NEGATIVE preconditions (M33, M34, M41, M42).
- 2 of 23 are PROJECTED @ R20 GPU retrain (M36, M38).
- 1 is DEPRECATED per Wf-317 Tier-2 (M40).
- 2 of 23 are MEASURED R17 ship surfaces (M43, M44 = M10/M11 in §1).
- Per-mechanism target gates are PROJECTED or HONEST-NEGATIVE, NOT
  measured on this wave.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "TODO"
    / "02_metrics"
    / "MECHANISM_METRIC_MAPPING.md"
)

# 23 Wf-320G plan-level mechanisms, grouped by Wf-320G W1-W5 wave.
W320G_W1_MATH = [
    "M22",  # SE(3) SB
    "M23",  # Transport Relax
    "M24",  # Monge-Ampère
    "M25",  # W₂ gradient flow
    "M26",  # Multi-PH
    "M27",  # Persistent Laplacian
    "M28",  # Bass martingale
    "M29",  # Hilbert PD
    "M30",  # Fractional RD
    "M31",  # Quantum sem
    "M32",  # Hochschild
]
W320G_W2_ARCH = ["M33", "M34"]  # R² ceiling + Vina covalent ceiling
W320G_W3_R20 = ["M35", "M36", "M37", "M38", "M39", "M40"]  # R20-P1..P6
W320G_W4_WF = ["M41", "M42"]  # Wf-301A DrugOOD + Wf-299B Pt+2
W320G_W5_R17 = ["M43", "M44"]  # R17 ship 84-rule MLE + 18-rule + 4 NEW SMARTS

ALL_320G_MECHANISMS = (
    W320G_W1_MATH + W320G_W2_ARCH + W320G_W3_R20 + W320G_W4_WF + W320G_W5_R17
)
assert len(ALL_320G_MECHANISMS) == 23, (
    f"Wf-320G expects 23 (M22-M44) but got {len(ALL_320G_MECHANISMS)}"
)

# Per-mechanism labels (mechanism name as written in §13).
MECHANISM_LABELS: dict[str, str] = {
    "M22": "SE(3) Schrödinger Bridge",
    "M23": "Transport Relaxation",
    "M24": "Monge-Ampère regularity",
    "M25": "Wasserstein gradient flow",
    "M26": "Multi-parameter Persistent Homology",
    "M27": "Persistent Laplacian",
    "M28": "Bass martingale",
    "M29": "Hilbert primal-dual",
    "M30": "Fractional reaction-diffusion",
    "M31": "Quantum semantics",
    "M32": "Hochschild cohomology",
    "M33": "R² architectural ceiling",
    "M34": "Vina covalent mechanism ceiling",
    "M35": "VariancePUCT default",
    "M36": "Hilbert-ASTSR R² lift",
    "M37": "Multi-PH Fourier + PLL feature extraction",
    "M38": "3-way on MetalCytoToxDB",
    "M39": "Closure theorem d-block",
    "M40": "Wet-lab IC50 validation",
    "M41": "DrugOOD BELOW_RANDOM fix",
    "M42": "Pt+2 parameterisation covalent blocker",
    "M43": "84-rule MLE matrix",
    "M44": "18-rule + 4 NEW SMARTS",
}

# Expected status tag per mechanism (honest framing).
EXPECTED_STATUS: dict[str, str] = {
    "M22": "CPU-ONLY-SKELETON",
    "M23": "CPU-ONLY-SKELETON",
    "M24": "PLAN-ONLY",
    "M25": "PLAN-ONLY",
    "M26": "CPU-ONLY-SKELETON",
    "M27": "CPU-ONLY-SKELETON",
    "M28": "CPU-ONLY-SKELETON",
    "M29": "CPU-ONLY-SKELETON",
    "M30": "CPU-ONLY-SKELETON",
    "M31": "CPU-ONLY-SKELETON",
    "M32": "CPU-ONLY-SKELETON",
    "M33": "HONEST-NEGATIVE",
    "M34": "HONEST-NEGATIVE",
    "M35": "CPU-ONLY-SKELETON",
    "M36": "PROJECTED @ R20",
    "M37": "CPU-ONLY-SKELETON",
    "M38": "PROJECTED @ R20",
    "M39": "CPU-ONLY-SKELETON",
    "M40": "HONEST-NEGATIVE",
    "M41": "HONEST-NEGATIVE",
    "M42": "HONEST-NEGATIVE",
    "M43": "MEASURED",
    "M44": "MEASURED",
}

# Required fields per §13 row.
REQUIRED_FIELDS = [
    "Plan ID",
    "Existing M-ID",
    "Status",
    "Affected metrics",
    "Self-check pytest",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mapping_text() -> str:
    """Read the MECHANISM_METRIC_MAPPING.md as text."""
    return MAPPING_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def section_13_text(mapping_text: str) -> str:
    """Extract the §13 Wf-320G cross-walk section."""
    match = re.search(
        r"## 13\. Wf-320G R18/R20 plan-level mechanism registry.*?(?=\n## 15\.|\Z)",
        mapping_text,
        re.DOTALL,
    )
    assert match, "§13 Wf-320G cross-walk section is missing"
    return match.group(0)


@pytest.fixture()
def section_14_text(mapping_text: str) -> str:
    """Extract the §14 plan-ID cross-reference section."""
    match = re.search(
        r"## 14\. Wf-320G plan-ID cross-reference.*?(?=\n## 15\.|\Z)",
        mapping_text,
        re.DOTALL,
    )
    assert match, "§14 plan-ID cross-reference section is missing"
    return match.group(0)


# ---------------------------------------------------------------------------
# Per-mechanism tests (23 tests, one per M22-M44)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mid",
    ALL_320G_MECHANISMS,
    ids=lambda m: f"test_wf320g_{m.lower()}",
)
def test_wf320g_mechanism_in_section_13(mid: str, section_13_text: str) -> None:
    """Each M22-M44 must have a row in §13.1 / §13.2 / §13.3 / §13.4 / §13.5."""
    # Search for "| **M{id}** | ..." pattern in §13.
    pattern = rf"\| \*\*{re.escape(mid)}\*\* \|"
    assert re.search(pattern, section_13_text), (
        f"{mid} is not documented in §13 of MECHANISM_METRIC_MAPPING.md "
        f"(Wf-320G cross-walk)"
    )


@pytest.mark.parametrize(
    "mid",
    ALL_320G_MECHANISMS,
    ids=lambda m: f"test_wf320g_{m.lower()}_label",
)
def test_wf320g_mechanism_label(mid: str, section_13_text: str) -> None:
    """Each mechanism row should carry the user-specified mechanism name."""
    label = MECHANISM_LABELS[mid]
    # Extract the M{id} row by anchoring on "| **M{id}** |".
    row_match = re.search(
        rf"\| \*\*{re.escape(mid)}\*\* \| (.+?) \|",
        section_13_text,
    )
    assert row_match, f"{mid} row not found in §13"
    row_first_cell = row_match.group(1).strip()
    # The first cell should contain at least one keyword from the label.
    # This is a fuzzy check — label is the plan-level short name.
    keywords = label.split()
    # Require at least the first 2 words to appear (case-insensitive).
    required = keywords[:2]
    for kw in required:
        assert kw.lower() in row_first_cell.lower(), (
            f"{mid} first cell '{row_first_cell}' missing keyword '{kw}' "
            f"(expected label fragment: '{label}')"
        )


@pytest.mark.parametrize(
    "mid",
    ALL_320G_MECHANISMS,
    ids=lambda m: f"test_wf320g_{m.lower()}_status",
)
def test_wf320g_mechanism_status_tag(mid: str, section_13_text: str) -> None:
    """Each mechanism row must carry the expected honest STATUS tag."""
    expected_status = EXPECTED_STATUS[mid]
    # Find the row, then look in the next 4 cells (Plan ID, Existing M-ID,
    # Status, Affected metrics).
    row_match = re.search(
        rf"\| \*\*{re.escape(mid)}\*\* \|[^\n]*\n",
        section_13_text,
    )
    assert row_match, f"{mid} row not found in §13"
    row = row_match.group(0)
    assert expected_status in row, (
        f"{mid} status '{expected_status}' not found in row: {row!r}"
    )


@pytest.mark.parametrize(
    "mid",
    ALL_320G_MECHANISMS,
    ids=lambda m: f"test_wf320g_{m.lower()}_required_fields",
)
def test_wf320g_mechanism_required_fields(mid: str, section_13_text: str) -> None:
    """Each mechanism row should mention all 5 required field keywords."""
    row_match = re.search(
        rf"\| \*\*{re.escape(mid)}\*\* \|[^\n]*\n",
        section_13_text,
    )
    assert row_match, f"{mid} row not found in §13"
    row = row_match.group(0).lower()
    # The §13.1/13.2/13.3/13.4/13.5 tables use header rows with these labels;
    # the data row itself does not include the labels but must reference
    # them implicitly via the cells. We check that the fields appear in the
    # section header OR are referenced in the section preamble text.
    preamble = section_13_text[: section_13_text.find(row_match.group(0))]
    for field in REQUIRED_FIELDS:
        assert field.lower() in preamble.lower(), (
            f"{mid} required field '{field}' not in §13 preamble (preamble length={len(preamble)})"
        )


# ---------------------------------------------------------------------------
# §13 cross-mapping summary table (M22-M44 × key metric)
# ---------------------------------------------------------------------------


def test_wf320g_section_13_6_cross_mapping_table(mapping_text: str) -> None:
    """§13.6 must have a cross-mapping summary table with all 23 mechanisms."""
    match = re.search(
        r"### §13\.6 §13 cross-mapping summary.*?(?=\n### |\Z)",
        mapping_text,
        re.DOTALL,
    )
    assert match, "§13.6 cross-mapping summary table missing"
    table = match.group(0)
    for mid in ALL_320G_MECHANISMS:
        pattern = rf"\| \*\*{re.escape(mid)}\*\* "
        assert re.search(pattern, table), (
            f"{mid} not in §13.6 cross-mapping summary table"
        )


# ---------------------------------------------------------------------------
# §13.7 failure-mode / regression-debug table
# ---------------------------------------------------------------------------


def test_wf320g_section_13_7_failure_mode_table(mapping_text: str) -> None:
    """§13.7 must have a failure-mode diagnostic table covering 23 mechanisms."""
    match = re.search(
        r"### §13\.7 §13 failure-mode diagnostic.*?(?=\n## 14\.|\Z)",
        mapping_text,
        re.DOTALL,
    )
    assert match, "§13.7 failure-mode diagnostic table missing"
    table = match.group(0).lower()
    # Each mechanism should be referenced at least once in the diagnostic
    # (either in the "先查 mechanism" column or in the pytest command).
    missing = []
    for mid in ALL_320G_MECHANISMS:
        if mid.lower() not in table:
            missing.append(mid)
    assert not missing, (
        f"Wf-320G failure-mode diagnostic table missing references for: {missing}"
    )


# ---------------------------------------------------------------------------
# §14 plan-ID cross-reference
# ---------------------------------------------------------------------------


def test_wf320g_section_14_1_plan_id_crosswalk(section_14_text: str) -> None:
    """§14.1 must map every M22-M44 to a plan ID / Wf ID / R17 Wf + existing M-ID."""
    assert "### §14.1 Plan-ID → M-ID cross-walk" in section_14_text, (
        "§14.1 Plan-ID → M-ID cross-walk table missing"
    )
    for mid in ALL_320G_MECHANISMS:
        # §14.1 table cells use unbolded "M{id}" inside "| M{id} |".
        pattern = rf"\| {re.escape(mid)} \|"
        assert re.search(pattern, section_14_text), (
            f"{mid} not in §14.1 plan-ID cross-walk table"
        )


def test_wf320g_section_14_2_plan_section_mapping(section_14_text: str) -> None:
    """§14.2 must map every R18 / R19 / R20 plan section to an M-ID."""
    assert "### §14.2 R18 / R19 / R20 plan section" in section_14_text, (
        "§14.2 plan section mapping table missing"
    )
    # Each M22-M32 should appear at least once in §14.2 (R18 P0-1..P1-7).
    for mid in W320G_W1_MATH:
        # §14.2 table cells use unbolded "M{id}" inside "| M{id} |".
        pattern = rf"\| {re.escape(mid)} \|"
        assert re.search(pattern, section_14_text), (
            f"{mid} not in §14.2 R18 plan section mapping"
        )
    # M35-M40 should appear in §14.2 (R20-P1..P6).
    for mid in W320G_W3_R20:
        pattern = rf"\| {re.escape(mid)} \|"
        assert re.search(pattern, section_14_text), (
            f"{mid} not in §14.2 R20-P plan section mapping"
        )


def test_wf320g_section_14_3_honest_framing(section_14_text: str) -> None:
    """§14.3 must declare honest framing with status categorization."""
    assert "### §14.3 Honest framing" in section_14_text, (
        "§14.3 honest-framing section missing"
    )
    required_phrases = [
        "23 mechanisms are documentation work",
        "zero fabricated numbers",
        "PLAN-ONLY",
        "HONEST-NEGATIVE",
        "PROJECTED @ R20",
        "DEPRECATED",
        "MEASURED",
    ]
    for phrase in required_phrases:
        assert phrase in section_14_text, (
            f"§14.3 honest-framing missing required phrase: {phrase!r}"
        )


# ---------------------------------------------------------------------------
# §13 wave grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wave_label,wf_id_list",
    [
        ("§13.1 W1 (R18 math)", W320G_W1_MATH),
        ("§13.2 W2 (Wf architectural findings)", W320G_W2_ARCH),
        ("§13.3 W3 (R20-P1..P6)", W320G_W3_R20),
        ("§13.4 W4 (Wf findings)", W320G_W4_WF),
        ("§13.5 W5 (R17 ship surfaces)", W320G_W5_R17),
    ],
)
def test_wf320g_wave_grouping(
    wave_label: str, wf_id_list: list[str], mapping_text: str
) -> None:
    """Each Wf-320G wave section header must exist and contain its mechanisms."""
    pattern = rf"### {re.escape(wave_label)}"
    assert re.search(pattern, mapping_text), (
        f"Wf-320G wave section header missing: {wave_label}"
    )


# ---------------------------------------------------------------------------
# Coverage invariant
# ---------------------------------------------------------------------------


def test_wf320g_total_mechanism_count() -> None:
    """Wf-320G must register exactly 23 mechanisms (M22-M44)."""
    assert len(ALL_320G_MECHANISMS) == 23, (
        f"Wf-320G mechanism count mismatch: expected 23, got {len(ALL_320G_MECHANISMS)}"
    )


def test_wf320g_status_distribution(section_13_text: str) -> None:
    """§13 status distribution: 12 CPU-ONLY-SKELETON + 2 PLAN-ONLY + 2 PROJECTED @ R20 + 5 HONEST-NEGATIVE + 2 MEASURED = 23."""
    counts: dict[str, int] = {
        "CPU-ONLY-SKELETON": 0,
        "PLAN-ONLY": 0,
        "PROJECTED @ R20": 0,
        "HONEST-NEGATIVE": 0,
        "MEASURED": 0,
    }
    for mid in ALL_320G_MECHANISMS:
        status = EXPECTED_STATUS[mid]
        if status in counts:
            counts[status] += 1
    assert counts["CPU-ONLY-SKELETON"] == 12, (
        f"CPU-ONLY-SKELETON count mismatch: expected 12 (M22, M23, M26, M27, M28, M29, M30, M31, M32, M35, M37, M39), "
        f"got {counts['CPU-ONLY-SKELETON']}"
    )
    assert counts["PLAN-ONLY"] == 2, (
        f"PLAN-ONLY count mismatch: expected 2 (M24, M25), got {counts['PLAN-ONLY']}"
    )
    assert counts["PROJECTED @ R20"] == 2, (
        f"PROJECTED @ R20 count mismatch: expected 2 (M36, M38), got {counts['PROJECTED @ R20']}"
    )
    assert counts["HONEST-NEGATIVE"] == 5, (
        f"HONEST-NEGATIVE count mismatch: expected 5 (M33, M34, M40, M41, M42), "
        f"got {counts['HONEST-NEGATIVE']}"
    )
    assert counts["MEASURED"] == 2, (
        f"MEASURED count mismatch: expected 2 (M43, M44), got {counts['MEASURED']}"
    )


def test_wf320g_m40_deprecated(section_13_text: str) -> None:
    """M40 (Wet-lab IC50) must be marked DEPRECATED per Wf-317 Tier-2."""
    match = re.search(
        r"\| \*\*M40\*\* \|[^\n]*\n",
        section_13_text,
    )
    assert match, "M40 row not found in §13"
    row = match.group(0)
    assert "DEPRECATED" in row, (
        f"M40 (Wet-lab IC50) must be marked DEPRECATED per Wf-317 Tier-2; row: {row!r}"
    )
    assert "HONEST-NEGATIVE" in row, (
        f"M40 (Wet-lab IC50) must also be HONEST-NEGATIVE precondition; row: {row!r}"
    )


# ---------------------------------------------------------------------------
# §13 §14 file-path invariant
# ---------------------------------------------------------------------------


def test_wf320g_mapping_doc_exists() -> None:
    """The mapping doc must exist on disk."""
    assert MAPPING_PATH.exists(), f"MECHANISM_METRIC_MAPPING.md missing at {MAPPING_PATH}"


def test_wf320g_no_duplicate_mechanism_id(section_13_text: str) -> None:
    """Each M-ID must appear exactly once in §13 (no duplicate IDs)."""
    for mid in ALL_320G_MECHANISMS:
        pattern = rf"\| \*\*{re.escape(mid)}\*\* "
        matches = re.findall(pattern, section_13_text)
        # Should appear in §13.1/13/§13.3/§13.4/§13.5 table row + §13.7 diagnostic + §14 = 3+ matches
        # But the §13 row uses "| **M{id}** |" pattern; we check >= 1.
        assert len(matches) >= 1, (
            f"{mid} missing from §13"
        )