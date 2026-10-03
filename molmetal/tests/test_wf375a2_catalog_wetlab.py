"""Wf-375-A2: Verify the §7 WETLAB_DEPRECATED section in docs/HONEST_NEGATIVE_CATALOG.md.

Per AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning per user directive
2026-09-29 verbatim *"我们发二区就可以了，不用规划湿实验"*), the §7 WETLAB_DEPRECATED
section must catalog exactly 6 entries covering per-file + per-launcher + per-workflow
wetlab deprecation:

  §7.1  4 LOI drafts              — DO NOT SEND
  §7.2  3 wetlab SOPs             — DO NOT EXECUTE
  §7.3  1 IC50 analysis pipeline  — DO NOT RUN
  §7.4  R20-P6 wet-lab IC50       — DEPRECATED (TODO-97)
  §7.5  Wf-74 v2 wet-lab outreach — DEPRECATED
  §7.6  LAMBDA_ROADMAP §3.2/§3.3  — DEPRECATED (R22-P1 m18 + R23 Phase I)

Each entry must cite its MEASURED source path on disk + WORKFLOW attribution per
party-twenty-spirit §3.5 + AGENTS.md §5.2 honest framing mandate.
"""

from __future__ import annotations

import re
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "HONEST_NEGATIVE_CATALOG.md"


def _read_catalog() -> str:
    assert CATALOG_PATH.exists(), f"catalog missing: {CATALOG_PATH}"
    return CATALOG_PATH.read_text(encoding="utf-8")


def test_wf375a2_catalog_has_wetlab_section():
    """§7 WETLAB_DEPRECATED header must exist in the catalog."""
    text = _read_catalog()
    assert "## §7 WETLAB_DEPRECATED entries" in text, "missing §7 WETLAB_DEPRECATED section header"
    assert "per-file + per-launcher + per-workflow" in text, "§7 category description missing"


def test_wf375a2_catalog_has_six_wetlab_entries():
    """§7 must contain exactly 6 §7.N entries covering all 6 deprecation items."""
    text = _read_catalog()
    section_start = text.find("## §7 WETLAB_DEPRECATED entries")
    assert section_start > 0, "§7 section not found"
    section_end = text.find("## §8 Closing", section_start)
    assert section_end > section_start, "§8 Closing not found after §7"
    section = text[section_start:section_end]
    # Count §7.N sub-entries
    subentries = re.findall(r"### §7\.(\d+)\s", section)
    assert len(subentries) == 6, f"expected 6 §7 sub-entries, found {len(subentries)}: {subentries}"
    # Verify all 6 expected sub-entries are present (in order)
    expected = ["1", "2", "3", "4", "5", "6"]
    assert subentries == expected, f"§7 sub-entry IDs out of order: {subentries} vs {expected}"


def test_wf375a2_entry_1_loi_drafts():
    """§7.1 must cover 4 LOI drafts with DO NOT SEND status."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.1[^\n]*\n+(.*?)(?=\n### §7\.2)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.1 entry not found"
    entry = entry_match.group(1)
    # Required content markers
    assert "DO NOT SEND" in entry, "§7.1 missing DO NOT SEND status"
    assert "wetlab_loi_ornl_protein_metal" in entry, "§7.1 missing ORNL LOI file path"
    assert "wetlab_loi_pku_cisplatin_ic50" in entry, "§7.1 missing PKU LOI file path"
    assert "wetlab_loi_xiehe_nci60" in entry, "§7.1 missing 协和 LOI file path"
    assert "wetlab_loi_rusanov_2025_platinai" in entry, "§7.1 missing Rusanov LOI file path"
    assert "Tier-2" in entry, "§7.1 missing Tier-2 substitution mechanism reference"
    assert "AGENTS.md §5.4" in entry, "§7.1 missing AGENTS.md §5.4 cross-ref"


def test_wf375a2_entry_2_wetlab_sops():
    """§7.2 must cover 3 wetlab SOPs with DO NOT EXECUTE status."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.2[^\n]*\n+(.*?)(?=\n### §7\.3)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.2 entry not found"
    entry = entry_match.group(1)
    assert "DO NOT EXECUTE" in entry, "§7.2 missing DO NOT EXECUTE status"
    assert "wetlab_sop_cytotoxicity_assay" in entry, "§7.2 missing cytotoxicity SOP file"
    assert "wetlab_sop_cellular_uptake_assay" in entry, "§7.2 missing cellular uptake SOP file"
    assert "wetlab_sop_binding_affinity_assay" in entry, "§7.2 missing binding affinity SOP file"
    assert "in-silico" in entry, "§7.2 missing in-silico substitution mechanism"
    assert "AGENTS.md §5.4" in entry, "§7.2 missing AGENTS.md §5.4 cross-ref"


def test_wf375a2_entry_3_ic50_pipeline():
    """§7.3 must cover 1 IC50 analysis pipeline with DO NOT RUN status."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.3[^\n]*\n+(.*?)(?=\n### §7\.4)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.3 entry not found"
    entry = entry_match.group(1)
    assert "DO NOT RUN" in entry, "§7.3 missing DO NOT RUN status"
    assert "wetlab_ic50_analysis_pipeline" in entry, "§7.3 missing IC50 pipeline file"
    assert "m1" in entry and "m2" in entry, "§7.3 missing m1+m2 substitution mechanism"
    assert "AGENTS.md §5.4" in entry, "§7.3 missing AGENTS.md §5.4 cross-ref"


def test_wf375a2_entry_4_r20_p6_wetlab():
    """§7.4 must cover R20-P6 wet-lab IC50 validation (TODO-97) as DEPRECATED."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.4[^\n]*\n+(.*?)(?=\n### §7\.5)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.4 entry not found"
    entry = entry_match.group(1)
    assert "DEPRECATED" in entry, "§7.4 missing DEPRECATED status"
    assert "R20-P6" in entry, "§7.4 missing R20-P6 reference"
    assert "TODO-97" in entry, "§7.4 missing TODO-97 reference"
    assert "in-silico" in entry, "§7.4 missing in-silico substitution mechanism"
    assert "IMPROVEMENT_ENUM_2026_09_27" in entry, "§7.4 missing IMPROVEMENT_ENUM cross-ref"


def test_wf375a2_entry_5_wf74_v2_outreach():
    """§7.5 must cover Wf-74 v2 wet-lab outreach as DEPRECATED."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.5[^\n]*\n+(.*?)(?=\n### §7\.6)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.5 entry not found"
    entry = entry_match.group(1)
    assert "DEPRECATED" in entry, "§7.5 missing DEPRECATED status"
    assert "Wf-74" in entry, "§7.5 missing Wf-74 reference"
    assert "116_wf74_wetlab_outreach_partial" in entry or "TODO-116" in entry, (
        "§7.5 missing TODO-116 file reference"
    )
    assert "DrugOOD" in entry or "DANN" in entry, "§7.5 missing computational substitution mechanism"


def test_wf375a2_entry_6_lambda_roadmap():
    """§7.6 must cover LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 as DEPRECATED."""
    text = _read_catalog()
    entry_match = re.search(
        r"### §7\.6[^\n]*\n+(.*?)(?=\n---)",
        text,
        re.DOTALL,
    )
    assert entry_match, "§7.6 entry not found"
    entry = entry_match.group(1)
    assert "DEPRECATED" in entry, "§7.6 missing DEPRECATED status"
    assert "LAMBDA_ROADMAP" in entry, "§7.6 missing LAMBDA_ROADMAP file"
    assert "R22-P1" in entry, "§7.6 missing R22-P1 reference"
    assert "R23" in entry, "§7.6 missing R23 reference"
    assert "AlphaFold-Multimer" in entry or "computational-only" in entry, (
        "§7.6 missing computational-only substitution mechanism"
    )


def test_wf375a2_total_entries_count_is_71():
    """Catalog TL;DR must reflect 71 entries (65 + 6 new WETLAB_DEPRECATED)."""
    text = _read_catalog()
    # The TL;DR should mention 71 entries
    assert "71 honest-negative findings" in text, "TL;DR missing 71 entries count"
    # The closing should reflect 71 entries
    assert "**71 unique entries**" in text or "71 honest-negative findings" in text


def test_wf375a2_section_count_is_7():
    """Catalog now has 7 sections (§1..§7) instead of 6 — verify in TL;DR."""
    text = _read_catalog()
    # The TL;DR lists 7 sections
    assert "7 sections" in text, "TL;DR should mention 7 sections (was 6)"


def test_wf375a2_subsection_8_and_9_renumbered():
    """§7 Closing (now §8) and §8 Cross-references (now §9) must be renumbered."""
    text = _read_catalog()
    # §8 Closing (renamed from §7)
    assert "## §8 Closing" in text, "missing §8 Closing (renamed from §7)"
    # §9 Cross-references (renamed from §8)
    assert "## §9 Cross-references" in text, "missing §9 Cross-references (renamed from §8)"
    # Old §7 Closing header must NOT exist anymore
    old_section_7_closing_match = re.search(r"^## §7 Closing", text, re.MULTILINE)
    assert old_section_7_closing_match is None, "old §7 Closing header still present (should be §8)"
    # Old §8 Cross-references header must NOT exist anymore
    old_section_8_match = re.search(r"^## §8 Cross-references", text, re.MULTILINE)
    assert old_section_8_match is None, "old §8 Cross-references header still present (should be §9)"


def test_wf375a2_zero_new_preflight_gates():
    """Wf-375-A2 adds zero NEW pre-flight gates (per closing + standing directives)."""
    text = _read_catalog()
    # Closing section should mention zero NEW pre-flight gates added by Wf-375-A2
    assert "zero NEW pre-flight gates added by Wf-375-A2" in text, (
        "closing section should mention zero NEW pre-flight gates added by Wf-375-A2"
    )
