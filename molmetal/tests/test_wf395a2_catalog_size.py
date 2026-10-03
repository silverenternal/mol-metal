"""Wf-395-A2 — Tests for the Wf-395-A2 docs/HONEST_NEGATIVE_CATALOG.md expansion
via 8 NEW §6.18-§6.25 entries from Wf-388 + Wf-390 + Wf-392 wave CPU-only docs
aggregation.

=====================================================================
Per Wf-395-A2 task brief §3: docs/HONEST_NEGATIVE_CATALOG.md must be
updated with 8 NEW entries in §6:

  - wf388_m2_real_cohort_r2_lift   (PROJECTED_PENDING_GPU_RETRAIN, m2_r2)
  - wf388_m7_diversity_wire_in     (SCAFFOLD_TO_WIRED, m7_diversity)
  - wf388_m4_sa_penalty_wire_in    (SCAFFOLD_TO_WIRED, m4_sa)
  - wf390_todo376_5metal_subset_path_a (PROJECTED_USER_OK_GATED, m2_r2)
  - wf390_todo179_scaffold_not_functional (HONEST_NEGATIVE_TRAINER_EXTENSION, m1_auc)
  - wf390_todo383_se3_sb_wire_in   (SCAFFOLD_TO_WIRED, m15_rmsd)
  - wf392_todo94_multiph_pll_wire_in (SCAFFOLD_TO_WIRED, m22_se3sb)
  - wf392_todo155_chemprop_v3_cache_wire_in (SCAFFOLD_TO_WIRED, chemprop_v3)

Per AGENTS.md §5.1, this is a docs+code audit (no retrain / panel /
ablation launch), so pre-flight gate (TODO-207) is N/A. Per AGENTS.md
§5.2, all claims use the MEASURED/PROJECTED tag discipline.
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


# Expected NEW §6.18-§6.25 entries from Wf-388/390/392 wave per
# Wf-395-A2 task brief §3.
EXPECTED_NEW_ENTRIES: list[tuple[str, str]] = [
    # (entry_id, expected_workflow_attribution)
    ("wf388_m2_real_cohort_r2_lift", "Wf-388"),
    ("wf388_m7_diversity_wire_in", "Wf-388"),
    ("wf388_m4_sa_penalty_wire_in", "Wf-388"),
    ("wf390_todo376_5metal_subset_path_a", "Wf-390"),
    ("wf390_todo179_scaffold_not_functional", "Wf-390"),
    ("wf390_todo383_se3_sb_wire_in", "Wf-390"),
    ("wf392_todo94_multiph_pll_wire_in", "Wf-392"),
    ("wf392_todo155_chemprop_v3_cache_wire_in", "Wf-392"),
]


@pytest.fixture(scope="module")
def catalog_text() -> str:
    """Read the catalog markdown file once per module."""
    assert CATALOG_PATH.exists(), (
        f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    )
    return CATALOG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def catalog_entries(catalog_text: str) -> list[str]:
    """Parse all catalog entries by their numbered anchors (§1.1, §2.15, §6.5, etc.)."""
    pattern = re.compile(r"^### §(\d+(?:\.\d+)+)\s+", re.MULTILINE)
    return pattern.findall(catalog_text)


@pytest.fixture(scope="module")
def main_sections(catalog_text: str) -> list[str]:
    """Parse all main section headers (§0..§9)."""
    pattern = re.compile(r"^## §(\d+)\s+", re.MULTILINE)
    return pattern.findall(catalog_text)


# ---------------------------------------------------------------------------
# Group A — Catalog size + Wf-395-A2 NEW entry coverage (1 test)
# ---------------------------------------------------------------------------
def test_catalog_size(catalog_text: str,
                      catalog_entries: list[str],
                      main_sections: list[str]) -> None:
    """Wf-395-A2 test: catalog must have ≥ 80 entries (post Wf-395-A2 expansion).

    Per Wf-395-A2 task brief §3: add 8 NEW §6 entries (§6.18-§6.25)
    from Wf-388 + Wf-390 + Wf-392 wave CPU-only docs aggregation.

    Catalog baseline (before Wf-395-A2): 73 entries (Wf-386-A3+B1 +
    Wf-383-A1 + Wf-375-A2 + Wf-369-A3 cumulative).

    Wf-395-A2 expansion: 73 → 81 (+8 NEW §6.18-§6.25 entries; +11.0% expansion;
    cumulative 9 → 81 = 9.0× expansion vs legacy 9-entry Wf-263B catalog).

    Per AGENTS.md §5.2 honest framing: every entry cites MEASURED source
    path on disk + WORKFLOW attribution; PROJECTED vs MEASURED tag discipline.

    CPU-only contract: pure file-system read + regex parse; no GPU
    launch; no external network.
    """
    # 1) File exists at the canonical location.
    assert CATALOG_PATH.exists(), (
        f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    )
    file_size = CATALOG_PATH.stat().st_size
    assert file_size > 10_000, (
        f"catalog must be substantial (>10KB after §6.18-§6.25 expansion); "
        f"got {file_size} bytes"
    )

    # 2) Main sections preserved (7 sections: §0..§7 + §8 Closing + §9 Cross-refs).
    for sec in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        assert sec in main_sections, f"§{sec} main section must be present"

    # 3) Entry count ≥ 80 (Wf-395-A2 expansion 73 → 81).
    assert len(catalog_entries) >= 80, (
        f"catalog must have ≥80 entries (Wf-395-A2 expansion 73 → 81); "
        f"got {len(catalog_entries)} entries"
    )

    # 4) §6 must contain ≥ 25 entries (Wf-369-A3 + Wf-383-A1 + Wf-386 +
    # Wf-395-A2 cumulative = 17 + 8 = 25).
    section_6_entries = [e for e in catalog_entries if e.startswith("6.")]
    assert len(section_6_entries) >= 25, (
        f"§6 must have ≥25 entries (Wf-369-A3 15 + Wf-383-A1 §6.16 + "
        f"Wf-386 §6.17 + Wf-395-A2 §6.18-§6.25 = 25 cumulative); "
        f"got {len(section_6_entries)} entries"
    )

    # 5) §6.18-§6.25 NEW entries must exist with the expected IDs.
    # Extract each §6.NN section header line + the body up to the next
    # `### ` heading or end of file.
    section_6_pattern = re.compile(
        r"^### §6\.\d+\s+.*?(?=^### §\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    section_6_blocks = section_6_pattern.findall(catalog_text)
    section_6_full_text = "\n".join(section_6_blocks)
    for entry_id, _ in EXPECTED_NEW_ENTRIES:
        assert entry_id in section_6_full_text, (
            f"Wf-395-A2 NEW entry ID `{entry_id}` must appear in §6.18-§6.25"
        )

    # 6) Each NEW entry must be attributed to its source workflow
    # (Wf-388, Wf-390, Wf-392). Build a per-§6.NN block dictionary by
    # splitting on `### §6.NN` headers, then verify the entry_id appears
    # in the SAME block as the expected workflow attribution.
    block_starts = list(re.finditer(
        r"^### §6\.\d+\s+",
        catalog_text,
        re.MULTILINE,
    ))
    for entry_id, expected_wf in EXPECTED_NEW_ENTRIES:
        matched_block = None
        for i, bm in enumerate(block_starts):
            block_start = bm.start()
            block_end = (
                block_starts[i + 1].start()
                if i + 1 < len(block_starts)
                else len(catalog_text)
            )
            block = catalog_text[block_start:block_end]
            if entry_id in block:
                matched_block = block
                break
        assert matched_block is not None, (
            f"could not extract block for entry `{entry_id}`"
        )
        assert expected_wf in matched_block, (
            f"NEW entry `{entry_id}` must cite workflow `{expected_wf}` "
            f"in its §6 block (found block lacks attribution)"
        )

    # 7) Honest framing discipline preserved (per `party-twenty-spirit`
    # §3.5). Catalog must contain both MEASURED and PROJECTED tags.
    assert "MEASURED" in catalog_text, (
        "catalog must use MEASURED tag discipline (party-twenty-spirit §3.5)"
    )
    assert "PROJECTED" in catalog_text, (
        "catalog must use PROJECTED tag discipline for non-measured claims"
    )

    # 8) Source citations: every NEW entry must cite at least one
    # `TODO/completed/...md` or `molmetal/reports/...json` source path.
    for entry_id, _ in EXPECTED_NEW_ENTRIES:
        matched_block = None
        for i, bm in enumerate(block_starts):
            block_start = bm.start()
            block_end = (
                block_starts[i + 1].start()
                if i + 1 < len(block_starts)
                else len(catalog_text)
            )
            block = catalog_text[block_start:block_end]
            if entry_id in block:
                matched_block = block
                break
        assert matched_block is not None, (
            f"could not extract block for entry `{entry_id}`"
        )
        source_citation = re.search(
            r"(TODO/completed/[^\s\)]+|molmetal/reports/[^\s\)]+\.json"
            r"|molmetal/reports/[^\s\)]+/)",
            matched_block,
        )
        assert source_citation, (
            f"NEW entry `{entry_id}` must cite a MEASURED source path on disk"
        )

    # 9) WETLAB_DEPRECATED section (§7) preserved (per AGENTS.md §5.4).
    section_7_entries = [e for e in catalog_entries if e.startswith("7.")]
    assert len(section_7_entries) == 6, (
        f"§7 WETLAB_DEPRECATED must have exactly 6 entries (per AGENTS.md §5.4); "
        f"got {len(section_7_entries)} entries"
    )

    # 10) Standing directives preserved (per AGENTS.md §5):
    # §5.1 GPU grant, §5.2 honest framing, §5.3 no paper, §5.4 Tier-2.
    for sd in ("§5.1", "§5.2", "§5.3", "§5.4"):
        assert sd in catalog_text, (
            f"AGENTS.md {sd} standing directive reference must be preserved"
        )

    print(
        f"  [test_catalog_size] catalog OK: path={CATALOG_PATH.name} "
        f"size={file_size}B entries={len(catalog_entries)} "
        f"sections={len(main_sections)} "
        f"section_6_entries={len(section_6_entries)} "
        f"section_7_entries={len(section_7_entries)} "
        f"new_wf_395_entries={len(EXPECTED_NEW_ENTRIES)} "
        f"(Wf-395-A2 expansion 73 → 81)",
        flush=True,
    )
