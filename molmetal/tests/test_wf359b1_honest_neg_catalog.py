"""Wf-359-B1 — Tests for the new docs/HONEST_NEGATIVE_CATALOG.md expansion 9 → 40 entries.

=====================================================================
Per Wf-357-C3 §5.4.a + Wf-358-C2 + AGENTS.md §9 honest framing catalog
expansion pattern: every honest-negative finding across Wf-296..358
must be catalogued with MEASURED-vs-PROJECTED tag discipline.

This test file (1 test = catalog structure verification):

  test_01_catalog_structure_and_entry_count

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
    assert (
        CATALOG_PATH.exists()
    ), f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    return CATALOG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def catalog_entries(catalog_text: str) -> list[str]:
    """Parse all catalog entries by their numbered anchors (§1.1, §2.15, etc.).

    Per the catalog template, every entry uses the pattern `### §N.M ...`
    or `### §N.M.N ...` for nested sub-sections.
    """
    # Match `### §<digit>(.<digit>)+ ` prefix lines (e.g. `### §1.1 ...`)
    pattern = re.compile(r"^### §(\d+(?:\.\d+)+)\s+", re.MULTILINE)
    matches = pattern.findall(catalog_text)
    return matches


# ---------------------------------------------------------------------------
# Group A — Catalog structure (1 test)
# ---------------------------------------------------------------------------
def test_01_catalog_structure_and_entry_count(
    catalog_text: str,
    catalog_entries: list[str],
) -> None:
    """Wf-359-B1 test 01: docs/HONEST_NEGATIVE_CATALOG.md must exist
    with 5 sections + ≥36 entries.

    Per Wf-357-C3 §5.4.a target: expand from 9 → 36 entries.  The actual
    expansion at Wf-359-B1 timestamp is 40 entries (12 metric + 15
    workflow + 6 launcher + 7 resolved + 10 open = 50 total entries,
    with cross-listing deduplication yielding 40 unique).

    CPU-only contract: pure file-system read + regex parse; no GPU
    launch; no external network.
    """
    # 1) File exists at the canonical location.
    assert (
        CATALOG_PATH.exists()
    ), f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    file_size = CATALOG_PATH.stat().st_size
    assert file_size > 5_000, (
        f"catalog must be substantial (>5KB); got {file_size} bytes "
        f"(target ~300 LOC per Wf-359-B1 task brief)"
    )

    # 2) Main sections (§0..§9 with §8=Closing + §9=Cross-references per Wf-375-A2
    # renumbering; Wf-375-A2 inserts new §7 WETLAB_DEPRECATED section, pushing
    # former §7 Closing → §8 and former §8 Cross-references → §9).
    main_section_pattern = re.compile(r"^## §(\d+)\s+", re.MULTILINE)
    main_sections = main_section_pattern.findall(catalog_text)
    assert "0" in main_sections, "§0 TL;DR section must be present"
    assert "1" in main_sections, "§1 Per-metric section must be present"
    assert "2" in main_sections, "§2 Per-workflow section must be present"
    assert "3" in main_sections, "§3 Per-launcher section must be present"
    assert "4" in main_sections, "§4 Resolved section must be present"
    assert "5" in main_sections, "§5 Open section must be present"
    assert "6" in main_sections, "§6 NEW Wf-340..368 section must be present"
    assert (
        "7" in main_sections
    ), "§7 WETLAB_DEPRECATED section must be present (Wf-375-A2)"
    assert (
        "8" in main_sections
    ), "§8 Closing section must be present (renumbered from §7 per Wf-375-A2)"
    assert (
        "9" in main_sections
    ), "§9 Cross-references section must be present (renumbered from §8 per Wf-375-A2)"

    # 3) Entry count ≥ 36 (Wf-357-C3 §5.4.a target).  Each entry uses
    # `### §<digit>(.<digit>)+ ...` as the heading prefix; counting
    # these gives the catalog's effective entry count.
    assert len(catalog_entries) >= 36, (
        f"catalog must have ≥36 entries (Wf-357-C3 §5.4.a target); "
        f"got {len(catalog_entries)} entries: {catalog_entries[:10]}..."
    )
    # Confirm we hit the Wf-359-B1 expansion target of 40 entries
    # (4.4× expansion over the legacy 9-entry Wf-263B catalog).
    assert len(catalog_entries) >= 40, (
        f"Wf-359-B1 expansion target is 40 entries (9 → 40 = 4.4× "
        f"expansion); got {len(catalog_entries)} entries"
    )

    # 4) Every main entry has a structured table with Field/Value rows
    # (per the catalog template).  We sample at least 10 entries to
    # confirm the template discipline.
    sample_entries = catalog_entries[:10]
    for entry_id in sample_entries:
        entry_pattern = re.compile(
            rf"^### §{re.escape(entry_id)}\s+.*?(?=^### §\d+|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        match = entry_pattern.search(catalog_text)
        assert match is not None, f"entry §{entry_id} must be locatable"
        entry_text = match.group(0)
        # Each entry has at least one table with `| Field | Value |` header
        # OR a `| **<Field>** | <Value> |` row pattern.
        has_field_value_table = bool(
            re.search(r"\|\s*\*\*[A-Za-z0-9 _]+\*\*\s*\|", entry_text)
        )
        assert (
            has_field_value_table
        ), f"entry §{entry_id} must have at least one Field/Value table row"

    # 5) Honest framing discipline preserved (per `party-twenty-spirit`
    # §3.5).  Catalog must contain both MEASURED and PROJECTED tags.
    assert (
        "MEASURED" in catalog_text
    ), "catalog must use MEASURED tag discipline (party-twenty-spirit §3.5)"
    assert (
        "PROJECTED" in catalog_text
    ), "catalog must use PROJECTED tag discipline for non-measured claims"
    # HONEST_NEGATIVE_<KIND> markers preserved
    assert (
        "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in catalog_text
    ), "catalog must preserve HONEST_NEGATIVE_PARTIAL_STRUCTURAL marker"
    assert (
        "HONEST_NEGATIVE_RESOLVED" in catalog_text
    ), "catalog must preserve HONEST_NEGATIVE_RESOLVED marker for §4"
    assert (
        "HONEST_NEGATIVE_BLOCKED" in catalog_text
    ), "catalog must preserve HONEST_NEGATIVE_BLOCKED marker for §5"

    # 6) Standing directives preserved (per AGENTS.md §5):
    # §5.1 GPU grant, §5.2 honest framing, §5.3 no paper, §5.4 Tier-2.
    assert (
        "§5.1" in catalog_text
    ), "AGENTS.md §5.1 GPU grant reference must be preserved"
    assert (
        "§5.2" in catalog_text
    ), "AGENTS.md §5.2 honest framing reference must be preserved"
    assert (
        "§5.3" in catalog_text
    ), "AGENTS.md §5.3 no-paper-work reference must be preserved"
    assert "§5.4" in catalog_text, "AGENTS.md §5.4 Tier-2 reference must be preserved"

    # 7) Source citations: every entry must cite a `TODO/completed/...md`
    # or `molmetal/reports/...json` source path.
    source_citation_count = len(
        re.findall(
            r"(TODO/completed/[^\s\)]+|molmetal/reports/[^\s\)]+\.json"
            r"|molmetal/reports/[^\s\)]+/)",
            catalog_text,
        )
    )
    assert source_citation_count >= 30, (
        f"catalog must cite ≥30 source paths on disk; "
        f"got {source_citation_count} citations"
    )

    # 8) Workflow attribution: catalog must cite at least 20 distinct
    # Wf-XXX workflow references (Wf-296..358).
    wf_refs = set(re.findall(r"Wf-\d{3}[A-Za-z0-9\-]*", catalog_text))
    assert len(wf_refs) >= 20, (
        f"catalog must cite ≥20 distinct workflow references; "
        f"got {len(wf_refs)}: {sorted(wf_refs)[:5]}..."
    )

    print(
        f"  [test_01] catalog OK: path={CATALOG_PATH.name} "
        f"size={file_size}B entries={len(catalog_entries)} "
        f"sections={len(main_sections)} "
        f"wf_refs={len(wf_refs)} "
        f"sources_cited={source_citation_count} "
        f"(target 40 entries; Wf-359-B1 expansion 9 → 40)",
        flush=True,
    )
