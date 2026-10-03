"""Wf-426-A4 — Tests for the Wf-426-A4 docs/HONEST_NEGATIVE_CATALOG.md expansion
via 4 NEW §6.27-§6.30 entries from Wf-425 cumulative false-claim cascade audit +
Wf-426 retro-author verification + Wf-426-A3 retrain readiness infrastructure +
Wf-425-B1 docs sync.

=====================================================================
Per Wf-426-A4 task brief §2: docs/HONEST_NEGATIVE_CATALOG.md must be
updated with 4 NEW entries in §6:

  - wf425_cumulative_false_claim_cascade_audit   (HONEST_NEGATIVE_FALSE_CLAIM_CASCADE_AUDIT)
  - wf426_retro_author_verification              (HONEST_NEGATIVE_FALSE_CLAIM_VERIFICATION)
  - wf426a3_retrain_readiness_infrastructure     (HONEST_NEGATIVE_REQUIRES_GPU_RETRAIN)
  - wf425b1_docs_sync                            (HONEST_NEGATIVE_DOCS_SYNC)

Per AGENTS.md §5.5 filesystem-truth rule strictly observed:

  - Wf-425 SHIPPED 3 of 6 closeouts on disk (425a1 + 425b2 + 425b5)
  - Wf-426 SHIPPED 0 of 5 closeouts on disk (none verified via ls)
  - Wf-426-A3 SHIPPED 0 of 1 closeouts on disk
  - Wf-425-B1 SHIPPED 0 of 1 closeouts on disk

Per AGENTS.md §5.1, this is a docs-only wave (no retrain / panel / ablation
launch), so pre-flight gate (TODO-207) is N/A. Per AGENTS.md §5.2, all
claims use the MEASURED/PROJECTED tag discipline.

Per AGENTS.md §5.6, this test file was authored via atomic_write_file from
molmetal/scripts/wf386a3_atomic_write.py.
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


# Expected NEW §6.27-§6.30 entries from Wf-425 + Wf-426 waves per Wf-426-A4
# task brief §2.
EXPECTED_NEW_ENTRIES: list[tuple[str, str, str]] = [
    # (entry_id, expected_workflow_attribution, expected_verdict_class)
    (
        "wf425_cumulative_false_claim_cascade_audit",
        "Wf-425",
        "HONEST_NEGATIVE_FALSE_CLAIM_CASCADE_AUDIT",
    ),
    (
        "wf426_retro_author_verification",
        "Wf-426",
        "HONEST_NEGATIVE_FALSE_CLAIM_VERIFICATION",
    ),
    (
        "wf426a3_retrain_readiness_infrastructure",
        "Wf-426-A3",
        "HONEST_NEGATIVE_REQUIRES_GPU_RETRAIN",
    ),
    (
        "wf425b1_docs_sync",
        "Wf-425-B1",
        "HONEST_NEGATIVE_DOCS_SYNC",
    ),
]


# 8 cascade IDs from Wf-425 cumulative false-claim cascade audit per
# Wf-426-A4 closeout §6.27 Cascade-1..Cascade-8.
EXPECTED_CASCADE_IDS: list[str] = [
    "p3_wf343c2_per_wave_runtime_check",
    "--json",
    "wf125_p1_controlled_pair.py",
    "EPOCHS=50",
    "catherine_6of6_pass",
    "kcal_sign",
    "lambda_samples_to_reward_targets",
    "use_ot_cfm",
]


@pytest.fixture(scope="module")
def catalog_text() -> str:
    """Read the catalog markdown file once per module."""
    assert (
        CATALOG_PATH.exists()
    ), f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
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
# Group A — Catalog size + Wf-426-A4 NEW entry coverage (1 test)
# ---------------------------------------------------------------------------
def test_catalog_size(
    catalog_text: str, catalog_entries: list[str], main_sections: list[str]
) -> None:
    """Wf-426-A4 test: catalog must have ≥ 86 entries (post Wf-426-A4 expansion).

    Per Wf-426-A4 task brief §2: add 4 NEW §6 entries (§6.27-§6.30)
    from Wf-425 + Wf-426 + Wf-425-B1 wave CPU-only docs aggregation.

    Catalog baseline (before Wf-426-A4): 82 entries (Wf-399-A2 expansion
    cumulative 9 → 82 = 9.1× via Wf-263B legacy 9-entry + Wf-369-A3 +
    Wf-375-A2 + Wf-383-A1 + Wf-386 + Wf-395-A2 + Wf-399-A2 cumulative).

    Wf-426-A4 expansion: 82 → 86 (+4 NEW §6.27-§6.30 entries; +4.9%
    expansion; cumulative 9 → 86 = 9.6× expansion vs legacy 9-entry
    Wf-263B catalog).

    Per AGENTS.md §5.2 honest framing: every entry cites MEASURED source
    path on disk + WORKFLOW attribution; PROJECTED vs MEASURED tag discipline.

    CPU-only contract: pure file-system read + regex parse; no GPU launch;
    no external network.
    """
    # 1) File exists at the canonical location.
    assert (
        CATALOG_PATH.exists()
    ), f"docs/HONEST_NEGATIVE_CATALOG.md must exist at {CATALOG_PATH}"
    file_size = CATALOG_PATH.stat().st_size
    assert file_size > 10_000, (
        f"catalog must be substantial (>10KB after §6.27-§6.30 expansion); "
        f"got {file_size} bytes"
    )

    # 2) Main sections preserved (10 sections: §0..§9).
    for sec in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        assert sec in main_sections, f"§{sec} main section must be present"

    # 3) Entry count ≥ 86 (Wf-426-A4 expansion 82 → 86).
    assert len(catalog_entries) >= 86, (
        f"catalog must have ≥86 entries (Wf-426-A4 expansion 82 → 86); "
        f"got {len(catalog_entries)} entries"
    )

    # 4) §6 must contain ≥ 30 entries (Wf-369-A3 + Wf-383-A1 + Wf-386 +
    # Wf-395-A2 + Wf-399-A2 + Wf-426-A4 cumulative = 17 + 1 + 1 + 8 + 1 + 4 = 32;
    # conservatively assert ≥ 30).
    section_6_entries = [e for e in catalog_entries if e.startswith("6.")]
    assert len(section_6_entries) >= 30, (
        f"§6 must have ≥30 entries (Wf-369-A3 16 + Wf-383-A1 §6.16 + "
        f"Wf-386 §6.17 + Wf-395-A2 §6.18-§6.25 + Wf-399-A2 §6.26 + "
        f"Wf-426-A4 §6.27-§6.30 = 32 cumulative); "
        f"got {len(section_6_entries)} entries"
    )

    # 5) §6.27-§6.30 NEW entries must exist with the expected IDs.
    section_6_pattern = re.compile(
        r"^### §6\.\d+\s+.*?(?=^### §\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    section_6_blocks = section_6_pattern.findall(catalog_text)
    section_6_full_text = "\n".join(section_6_blocks)
    for entry_id, _, _ in EXPECTED_NEW_ENTRIES:
        assert (
            entry_id in section_6_full_text
        ), f"Wf-426-A4 NEW entry ID `{entry_id}` must appear in §6.27-§6.30"

    # 6) Each NEW entry must be attributed to its source workflow
    # (Wf-425 / Wf-426 / Wf-426-A3 / Wf-425-B1).
    block_starts = list(
        re.finditer(
            r"^### §6\.\d+\s+",
            catalog_text,
            re.MULTILINE,
        )
    )
    for entry_id, expected_wf, _ in EXPECTED_NEW_ENTRIES:
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
        assert (
            matched_block is not None
        ), f"could not extract block for entry `{entry_id}`"
        assert expected_wf in matched_block, (
            f"NEW entry `{entry_id}` must cite workflow `{expected_wf}` "
            f"in its §6 block (found block lacks attribution)"
        )

    # 7) Honest framing discipline preserved (per `party-twenty-spirit`
    # §3.5). Catalog must contain both MEASURED and PROJECTED tags.
    assert (
        "MEASURED" in catalog_text
    ), "catalog must use MEASURED tag discipline (party-twenty-spirit §3.5)"
    assert (
        "PROJECTED" in catalog_text
    ), "catalog must use PROJECTED tag discipline for non-measured claims"

    # 8) Source citations: every NEW entry must cite at least one
    # `TODO/completed/...md` or `molmetal/reports/...json` source path.
    for entry_id, _, _ in EXPECTED_NEW_ENTRIES:
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
        assert (
            matched_block is not None
        ), f"could not extract block for entry `{entry_id}`"
        source_citation = re.search(
            r"(TODO/completed/[^\s\)]+|molmetal/reports/[^\s\)]+\.json"
            r"|molmetal/reports/[^\s\)]+/)",
            matched_block,
        )
        assert (
            source_citation
        ), f"NEW entry `{entry_id}` must cite a MEASURED source path on disk"

    # 9) WETLAB_DEPRECATED section (§7) preserved (per AGENTS.md §5.4).
    section_7_entries = [e for e in catalog_entries if e.startswith("7.")]
    assert len(section_7_entries) == 6, (
        f"§7 WETLAB_DEPRECATED must have exactly 6 entries (per AGENTS.md §5.4); "
        f"got {len(section_7_entries)} entries"
    )

    # 10) AGENTS.md §5.5 filesystem-truth rule preserved (per Wf-426-A4).
    assert (
        "§5.5" in catalog_text
    ), "AGENTS.md §5.5 filesystem-truth rule reference must be preserved"
    assert (
        "filesystem-truth" in catalog_text or "filesystem truth" in catalog_text
    ), "filesystem-truth rule must be cited by name in the catalog"

    # 11) AGENTS.md §5.6 atomic write protocol reference preserved (per
    # Wf-426-A4 closeout-author discipline).
    assert (
        "§5.6" in catalog_text
    ), "AGENTS.md §5.6 atomic write protocol reference must be preserved"

    # 12) Standing directives preserved (per AGENTS.md §5):
    # §5.1 GPU grant, §5.2 honest framing, §5.3 no paper, §5.4 Tier-2.
    for sd in ("§5.1", "§5.2", "§5.3", "§5.4"):
        assert (
            sd in catalog_text
        ), f"AGENTS.md {sd} standing directive reference must be preserved"

    print(
        f"  [test_catalog_size] catalog OK: path={CATALOG_PATH.name} "
        f"size={file_size}B entries={len(catalog_entries)} "
        f"sections={len(main_sections)} "
        f"section_6_entries={len(section_6_entries)} "
        f"section_7_entries={len(section_7_entries)} "
        f"new_wf_426a4_entries={len(EXPECTED_NEW_ENTRIES)} "
        f"(Wf-426-A4 expansion 82 → 86)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Group B — Wf-426-A4 §6.27 cascade ID coverage (1 test)
# ---------------------------------------------------------------------------
def test_section_6_27_cascade_id_coverage(catalog_text: str) -> None:
    """Wf-426-A4 test: §6.27 must reference the 8 cascade IDs from
    Wf-425 cumulative false-claim cascade audit.

    Per Wf-426-A4 closeout §6.27 Cascade-1..Cascade-8 list:
      - Cascade-1: p3_wf343c2_per_wave_runtime_check PHANTOM FAIL
      - Cascade-2: --json argparse bug in per_wave_checklist.py
      - Cascade-3: Wf-388 Wf-125 P1 CONTROLD PAIR phantom target
      - Cascade-4: Wf-388 Wf-125 wrapper flag-surface mismatch
      - Cascade-5: Wf-387-A2 cohort JSON schema mismatch
      - Cascade-6: Wf-246-C metal-coordination kcal sign fix
      - Cascade-7: Wf-394 Lambda × CFM coupling audit
      - Cascade-8: Wf-405-A2 SCAFFOLD-loss wire-in INCOMPLETE

    Per AGENTS.md §5.5 filesystem-truth rule: every cascade ID must be
    recoverable from the §6.27 block on disk.
    """
    # Extract §6.27 block only (between §6.27 header and §6.28 header).
    section_pattern = re.compile(
        r"^### §6\.27\s+.*?(?=^### §6\.28\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    section_match = section_pattern.search(catalog_text)
    assert (
        section_match is not None
    ), "§6.27 Wf-425 cumulative false-claim cascade audit block must exist"
    section_block = section_match.group(0)

    # Verify each cascade ID appears in the §6.27 block.
    missing_cascades: list[str] = []
    for cascade_id in EXPECTED_CASCADE_IDS:
        if cascade_id not in section_block:
            missing_cascades.append(cascade_id)

    assert not missing_cascades, (
        f"§6.27 must reference all 8 cascade IDs; " f"missing: {missing_cascades}"
    )

    # Verify the section header line contains the Wf-426-A4 wave attribution.
    assert "Wf-425" in section_block, "§6.27 must cite Wf-425 wave attribution"
    assert (
        "Wf-426-A4" in section_block
    ), "§6.27 must cite Wf-426-A4 wave attribution (this wave)"

    print(
        f"  [test_section_6_27_cascade_id_coverage] cascade OK: "
        f"cascades_found={len(EXPECTED_CASCADE_IDS)} "
        f"cascades_missing=0",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Group C — AGENTS.md §5.5 filesystem-truth rule enforcement (1 test)
# ---------------------------------------------------------------------------
def test_filesystem_truth_rule_citation(catalog_text: str) -> None:
    """Wf-426-A4 test: AGENTS.md §5.5 filesystem-truth rule must be cited
    in the §6.27-§6.30 NEW entries.

    Per AGENTS.md §5.5 rule #1 "report SHIPPED ≠ file exists"; per rule #6
    "filesystem is the source of truth". The Wf-426-A4 NEW entries must
    explicitly cite this rule to justify their HONEST_NEGATIVE_FALSE_CLAIM_*
    verdicts.
    """
    section_6_pattern = re.compile(
        r"^### §6\.(?:27|28|29|30)\s+.*?(?=^### §\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    section_6_new_matches = list(section_6_pattern.finditer(catalog_text))
    assert (
        len(section_6_new_matches) == 4
    ), f"expected 4 NEW §6.27-§6.30 entries; got {len(section_6_new_matches)}"
    section_6_full_text = "\n".join(m.group(0) for m in section_6_new_matches)

    # The §5.5 filesystem-truth rule must be cited at least once in §6.27-§6.30.
    assert (
        "§5.5" in section_6_full_text
    ), "§6.27-§6.30 NEW entries must cite AGENTS.md §5.5 filesystem-truth rule"

    # The rule must be cited by name at least once (filesystem-truth OR
    # filesystem truth OR filesystem is the source of truth).
    filesystem_truth_cited = (
        "filesystem-truth" in section_6_full_text
        or "filesystem truth" in section_6_full_text
        or "filesystem is the source of truth" in section_6_full_text
    )
    assert filesystem_truth_cited, (
        "§5.5 rule must be cited by name (filesystem-truth OR "
        "filesystem truth OR filesystem is the source of truth)"
    )

    print(
        "  [test_filesystem_truth_rule_citation] filesystem-truth rule OK: "
        "cited_in_§6.27-§6.30=True",
        flush=True,
    )
