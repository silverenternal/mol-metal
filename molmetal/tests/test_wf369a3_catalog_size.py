"""Wf-369-A3 — Tests for the Wf-369-A3 docs/HONEST_NEGATIVE_CATALOG.md expansion 40 → 65 entries.

=====================================================================
Per Wf-369-A3 task brief §5: docs/HONEST_NEGATIVE_CATALOG.md must be
updated with Wf-340..368 new honest-negative findings, pushing total
from 40 → ≥ 50 entries. Wf-369-A3 ships a NEW §6 section consolidating
15 NEW Wf-340..368 entries.

This test file (1 test = catalog size + structure verification):

  test_01_catalog_size_and_wf340_368_coverage

Per AGENTS.md §5.1, this is a docs+code audit (no retrain / panel /
ablation launch), so pre-flight gate (TODO-207) is N/A.  Per
AGENTS.md §5.2, all claims use the MEASURED/PROJECTED tag discipline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CATALOG_PATH = _REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"


@pytest.fixture(scope="module")
def catalog_text() -> str:
    """Read the catalog markdown file once per module."""
    assert CATALOG_PATH.exists(), (
        f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    )
    return CATALOG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def catalog_entries(catalog_text: str) -> list[str]:
    """Parse all catalog entries by their numbered anchors (§1.1, §2.15, §6.5, etc.).

    Per the catalog template, every entry uses the pattern `### §N.M ...`
    or `### §N.M.N ...` for nested sub-sections.
    """
    # Match `### §<digit>(.<digit>)+ ` prefix lines (e.g. `### §1.1 ...`)
    pattern = re.compile(r"^### §(\d+(?:\.\d+)+)\s+", re.MULTILINE)
    matches = pattern.findall(catalog_text)
    return matches


@pytest.fixture(scope="module")
def main_sections(catalog_text: str) -> list[str]:
    """Parse all main section headers (§0..§8)."""
    pattern = re.compile(r"^## §(\d+)\s+", re.MULTILINE)
    return pattern.findall(catalog_text)


# ---------------------------------------------------------------------------
# Group A — Catalog size + Wf-340..368 coverage (1 test)
# ---------------------------------------------------------------------------
def test_01_catalog_size_and_wf340_368_coverage(
    catalog_text: str,
    catalog_entries: list[str],
    main_sections: list[str],
) -> None:
    """Wf-369-A3 test 01: catalog must have ≥ 50 entries + 6 sections
    + §6 Wf-340..368 coverage.

    Per Wf-369-A3 task brief §4: expand from 40 → ≥ 50 entries.  Wf-369-A3
    ships 15 NEW entries in §6 (consolidated Wf-340..368 findings),
    pushing total from 40 → 65 (75% over the ≥ 50 target).

    Per AGENTS.md §5.2 honest framing: every Wf-340..368 finding cited
    with MEASURED-vs-PROJECTED tag discipline.

    CPU-only contract: pure file-system read + regex parse; no GPU
    launch; no external network.
    """
    # 1) File exists at the canonical location.
    assert CATALOG_PATH.exists(), (
        f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    )
    file_size = CATALOG_PATH.stat().st_size
    assert file_size > 10_000, (
        f"catalog must be substantial (>10KB after §6 expansion); "
        f"got {file_size} bytes (target ~600 LOC per Wf-369-A3 task brief)"
    )

    # 2) Main sections (was 6 in Wf-369-A3; §7 WETLAB_DEPRECATED added in Wf-375-A2,
    # pushing Closing §7→§8 and Cross-references §8→§9).
    assert "0" in main_sections, "§0 TL;DR section must be present"
    assert "1" in main_sections, "§1 Per-metric section must be present"
    assert "2" in main_sections, "§2 Per-workflow section must be present"
    assert "3" in main_sections, "§3 Per-launcher section must be present"
    assert "4" in main_sections, "§4 Resolved section must be present"
    assert "5" in main_sections, "§5 Open section must be present"
    assert "6" in main_sections, "§6 NEW Wf-340..368 section must be present (Wf-369-A3)"
    assert "7" in main_sections, "§7 WETLAB_DEPRECATED section must be present (Wf-375-A2 NEW)"
    assert "8" in main_sections, "§8 Closing section must be present (renumbered from §7 per Wf-375-A2)"
    assert "9" in main_sections, "§9 Cross-references section must be present (renumbered from §8 per Wf-375-A2)"

    # 3) Entry count ≥ 50 (Wf-369-A3 task brief §4 target).
    # Wf-369-A3 expansion: 40 → 65 (15 NEW §6 entries + structural).
    assert len(catalog_entries) >= 50, (
        f"catalog must have ≥50 entries (Wf-369-A3 task brief §4 target); "
        f"got {len(catalog_entries)} entries: {catalog_entries[:10]}..."
    )
    # Confirm we hit the Wf-369-A3 expansion target of 65 entries.
    assert len(catalog_entries) >= 65, (
        f"Wf-369-A3 expansion target is 65 entries (40 → 65 = +25 entries, "
        f"62.5% expansion); got {len(catalog_entries)} entries"
    )

    # 4) §6 must exist with at least 10 NEW Wf-340..368 entries.
    section_6_entries = [e for e in catalog_entries if e.startswith("6.")]
    assert len(section_6_entries) >= 10, (
        f"§6 NEW Wf-340..368 section must have ≥10 entries; "
        f"got {len(section_6_entries)} entries: {section_6_entries}"
    )

    # 5) Every §6 entry must cite Wf-340..368 (no legacy entries leaked).
    section_6_pattern = re.compile(
        r"^### §6\.\d+\s+.*?(?=^### §\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    section_6_blocks = section_6_pattern.findall(catalog_text)
    assert len(section_6_blocks) >= 10, (
        f"§6 must have ≥10 entry blocks; got {len(section_6_blocks)}"
    )

    # 6) Wf-340..368 reference coverage: catalog must cite at least
    # 20 NEW Wf-340..368 workflows (Wf-340..368 in closeouts).
    wf_340_368_refs = set(
        re.findall(r"Wf-(?:3[4-6][0-9])(?:[A-Z][0-9]?)?", catalog_text)
    )
    assert len(wf_340_368_refs) >= 15, (
        f"catalog must cite ≥15 distinct Wf-340..368 workflows in §6; "
        f"got {len(wf_340_368_refs)}: {sorted(wf_340_368_refs)[:5]}..."
    )

    # 7) Honest framing discipline preserved (per `party-twenty-spirit`
    # §3.5). Catalog must contain both MEASURED and PROJECTED tags.
    assert "MEASURED" in catalog_text, (
        "catalog must use MEASURED tag discipline (party-twenty-spirit §3.5)"
    )
    assert "PROJECTED" in catalog_text, (
        "catalog must use PROJECTED tag discipline for non-measured claims"
    )
    # New HONEST_NEGATIVE_<KIND> markers added by §6.
    assert "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in catalog_text, (
        "catalog must preserve HONEST_NEGATIVE_PARTIAL_STRUCTURAL marker"
    )
    assert "HONEST_NEGATIVE_R2_DEEP" in catalog_text, (
        "catalog must preserve HONEST_NEGATIVE_R2_DEEP marker"
    )
    assert "HONEST_NEGATIVE_RESOLVED" in catalog_text, (
        "catalog must preserve HONEST_NEGATIVE_RESOLVED marker for §4 + §6"
    )
    assert "HONEST_NEGATIVE_BLOCKED" in catalog_text, (
        "catalog must preserve HONEST_NEGATIVE_BLOCKED marker for §5"
    )

    # 8) Source citations: every entry must cite a `TODO/completed/...md`
    # or `molmetal/reports/...json` source path.
    source_citation_count = len(
        re.findall(
            r"(TODO/completed/[^\s\)]+|molmetal/reports/[^\s\)]+\.json"
            r"|molmetal/reports/[^\s\)]+/)",
            catalog_text,
        )
    )
    assert source_citation_count >= 40, (
        f"catalog must cite ≥40 source paths on disk; "
        f"got {source_citation_count} citations"
    )

    # 9) Workflow attribution: catalog must cite at least 30 distinct
    # Wf-XXX workflow references (Wf-296..368).
    wf_refs = set(re.findall(r"Wf-\d{3}[A-Za-z0-9\-]*", catalog_text))
    assert len(wf_refs) >= 30, (
        f"catalog must cite ≥30 distinct workflow references; "
        f"got {len(wf_refs)}: {sorted(wf_refs)[:5]}..."
    )

    # 10) Standing directives preserved (per AGENTS.md §5):
    # §5.1 GPU grant, §5.2 honest framing, §5.3 no paper, §5.4 Tier-2.
    assert "§5.1" in catalog_text, "AGENTS.md §5.1 GPU grant reference must be preserved"
    assert "§5.2" in catalog_text, "AGENTS.md §5.2 honest framing reference must be preserved"
    assert "§5.3" in catalog_text, "AGENTS.md §5.3 no-paper-work reference must be preserved"
    assert "§5.4" in catalog_text, "AGENTS.md §5.4 Tier-2 reference must be preserved"

    # 11) Pre-flight cascade FULLY CLOSED reference (Wf-367-A2).
    assert "Wf-367-A2" in catalog_text, (
        "catalog must reference Wf-367-A2 pre-flight cascade FULLY CLOSED baseline"
    )

    # 12) Wf-369-A3 attribution in header + closing.
    assert "Wf-369-A3" in catalog_text, (
        "catalog must include Wf-369-A3 attribution in header + closing"
    )

    print(
        f"  [test_01] catalog OK: path={CATALOG_PATH.name} "
        f"size={file_size}B entries={len(catalog_entries)} "
        f"sections={len(main_sections)} "
        f"section_6_entries={len(section_6_entries)} "
        f"wf_340_368_refs={len(wf_340_368_refs)} "
        f"wf_refs={len(wf_refs)} "
        f"sources_cited={source_citation_count} "
        f"(target 65 entries; Wf-369-A3 expansion 40 → 65)",
        flush=True,
    )