"""Wf-368-B2 — INDEX.md §22 sync coverage verification for Wf-340..368.

Per Wf-368-A3 §0 user voice + Wf-368-B2 task brief: verify all
Wf-340..368 entries are present in TODO/INDEX.md §22. Per
party-twenty-spirit §3.5: every numeric claim must trace to a JSON /
report path on disk; we assert by file existence + section presence.

Background (Wf-332-S1 §0.16 + Wf-338-F2 verification pattern):
  * Wf-323..338 INDEX coverage verdict table — all 16 rows PASS.
  * Wf-339..367 — INDEX entries added via Wf-340..367 docs-only C1
    sub-flows.
  * Wf-368 — 4 closeouts (A1+A2+A3+A4); Wf-368-B2 docs-only INDEX
    sync appends §22.58.

This module ships ONE new test:

  1. ``test_index_section_22_has_wf_340_to_368_entries`` — for every
     Wf in the inclusive range [340, 368], verify that ``TODO/INDEX.md``
     has at least one ``### §22.NN`` entry referencing that Wf. Per
     party-twenty-spirit §3.5: each missing entry is an honest-negative
     and must be reported in the test's assertion message (with the
     next expected §22.NN number).

CPU-only; wall budget < 1 s.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = REPO_ROOT / "TODO" / "INDEX.md"
COMPLETED_DIR = REPO_ROOT / "TODO" / "completed"

# Inclusive range to verify per Wf-368-B2 task brief.
WF_RANGE_START = 340
WF_RANGE_END = 368

# Regex for `### §22.NN Wf-XXX ...` headings in INDEX.md.
INDEX_SECTION_PATTERN = re.compile(
    r"^### §22\.(\d+)\s+(?P<title>.+?)$",
    flags=re.MULTILINE,
)


def _load_index_text() -> str:
    """Load TODO/INDEX.md text. Skip if the file is missing."""
    assert (
        INDEX_PATH.is_file()
    ), f"INDEX.md not found at {INDEX_PATH} — repo layout drifted?"
    return INDEX_PATH.read_text(encoding="utf-8")


def _extract_wf_ids_from_index(index_text: str) -> dict[str, list[int]]:
    """Return a mapping {Wf-number (str) -> list of §22 section numbers} for every Wf-XXX referenced in a §22 heading."""
    wf_to_sections: dict[str, list[int]] = {}
    for match in INDEX_SECTION_PATTERN.finditer(index_text):
        section_no = int(match.group(1))
        title = match.group("title").strip()
        # title e.g. "Wf-368 ultracode follow-up verification wave ..."
        wf_match = re.match(r"Wf-(\d+)", title)
        if wf_match is None:
            continue
        wf_no = wf_match.group(1)
        wf_to_sections.setdefault(wf_no, []).append(section_no)
    return wf_to_sections


def _completed_files_for_wf(wf_no: int) -> list[Path]:
    """Return all closeout files for a given Wf number in TODO/completed/."""
    if not COMPLETED_DIR.is_dir():
        return []
    return sorted(COMPLETED_DIR.glob(f"{wf_no}*.md"))


def test_index_section_22_has_wf_340_to_368_entries() -> None:
    """Every Wf in [340, 368] must have at least one ``### §22.NN`` entry referencing it in TODO/INDEX.md."""
    index_text = _load_index_text()
    wf_to_sections = _extract_wf_ids_from_index(index_text)

    missing: list[int] = []
    present: list[tuple[int, list[int], int]] = []

    for wf_no in range(WF_RANGE_START, WF_RANGE_END + 1):
        wf_key = str(wf_no)
        section_nos = wf_to_sections.get(wf_key, [])
        closeout_files = _completed_files_for_wf(wf_no)
        if not section_nos:
            missing.append(wf_no)
        present.append((wf_no, sorted(set(section_nos)), len(closeout_files)))

    # Build a precise diagnostic so a failure points the reader at the
    # specific Wf number(s) that still need a §22 entry (per party-
    # twenty-spirit §3.5 honest-negative framing).
    if missing:
        max_existing_section_no = max(
            (max(s) for _, s, _ in present if s),
            default=0,
        )
        next_section_no = max_existing_section_no + 1
        details = "\n".join(
            f"  Wf-{wf_no}: NO §22 entry — expected §22.{next_section_no + i}"
            for i, wf_no in enumerate(missing)
        )
        raise AssertionError(
            f"Wf-368-B2 §22 sync gap: {len(missing)} Wf in "
            f"[{WF_RANGE_START}, {WF_RANGE_END}] have NO §22 entry in "
            f"TODO/INDEX.md:\n{details}\n"
            f"Next available §22.NN slot: §22.{next_section_no} "
            f"(see Wf-368-B2 closeout at TODO/completed/368b2_index_sync.md)"
        )

    # Sanity: when no Wf is missing, every Wf-340..368 should have at
    # least one closeout file too.  This guards against the inverse
    # bug (entry exists but file missing).
    file_missing: list[int] = []
    for wf_no, sections, file_count in present:
        if file_count == 0:
            file_missing.append(wf_no)
    assert not file_missing, (
        "Wf-368-B2 §22 entry exists but NO closeout file in "
        "TODO/completed/ for: " + ", ".join(f"Wf-{n}" for n in file_missing)
    )

    # When everything is present, also report counts so the test
    # produces a useful PASS message.
    total_sections = sum(len(s) for _, s, _ in present)
    total_files = sum(f for _, _, f in present)
    assert total_sections >= (WF_RANGE_END - WF_RANGE_START + 1), (
        f"Expected at least {WF_RANGE_END - WF_RANGE_START + 1} §22 entries "
        f"covering Wf-{WF_RANGE_START}..Wf-{WF_RANGE_END}, got {total_sections}"
    )
    assert total_files >= 1, "No closeout files found in TODO/completed/"
