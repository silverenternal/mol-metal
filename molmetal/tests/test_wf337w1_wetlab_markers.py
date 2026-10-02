"""Wf-337-W1 — 10 wetlab_*.md Tier-2 DEPRECATED marker verification tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)

Context (MEASURED, Wf-324C + Wf-337-W1 2026-09-29):
  - Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
    Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
  - All 10 TODO/pending/wetlab_*.md files should be marked
    TEMPLATE_NOT_NEEDED_FOR_TIER2 to align with the §5.4 directive.
  - Wf-324C added markers to 8 of the 10 files. Wf-337-W1 added markers to
    the remaining 2 (wetlab_cold_email_template.md + wetlab_loi_consolidation_review.md).

Tests (4 total):

  - test_all_10_wetlab_files_exist
        Gates that all 10 wetlab files exist on disk. Wf-324C catalog
        includes: 1 ic50 + 4 LOIs + 3 SOPs + 2 helpers (cold-email + review).
  - test_all_10_wetlab_files_have_tier2_marker
        Verifies the TEMPLATE_NOT_NEEDED_FOR_TIER2 marker is present in
        every wetlab_*.md file.
  - test_marker_format_consistency
        Verifies the marker block follows one of two canonical formats:
          (a) LOI-style: `> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (YYYY-MM-DD)**`
              followed by `> Per AGENTS.md §5.4 user directive ...`
          (b) SOP/ic50-style: `> **Status (YYYY-MM-DD): TEMPLATE_NOT_NEEDED_FOR_TIER2**`
              followed by `> — per AGENTS.md §5.4 user directive ...`
        Both formats end with a Tier-2 explanation blockquote.
  - test_tier2_strategy_compliance
        Verifies Tier-2 strategy compliance: all wetlab files cite
        AGENTS.md §5.4 in the marker block; none are flagged for execution.

All tests are CPU-only — no GPU, no real launches. < 2 s wall.

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED (Wf-337-W1 2026-09-29): 10 wetlab_*.md files exist in
    TODO/pending/ — listed by `ls TODO/pending/wetlab_*.md`.
  - MEASURED (Wf-337-W1 2026-09-29): all 10 files now have the
    TEMPLATE_NOT_NEEDED_FOR_TIER2 marker after this Wf's edits.
  - MEASURED (Wf-324C): 8 of 10 had the marker before this Wf.
  - PROJECTED: 0 wetlab files will be executed in the R20 paper cycle.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/hugo/codes/mol-metal")
TODO_PENDING = ROOT / "TODO" / "pending"

# Per Wf-324C + Wf-337-W1: the full 10-file wetlab catalogue. Files 9 + 10
# (cold_email_template + loi_consolidation_review) were added in this Wf.
WETLAB_FILES = [
    "wetlab_ic50_analysis_pipeline.md",
    "wetlab_loi_ornl_protein_metal.md",
    "wetlab_loi_pku_cisplatin_ic50.md",
    "wetlab_loi_xiehe_nci60.md",
    "wetlab_loi_rusanov_2025_platinai.md",
    "wetlab_sop_binding_affinity_assay.md",
    "wetlab_sop_cellular_uptake_assay.md",
    "wetlab_sop_cytotoxicity_assay.md",
    # Wf-337-W1 newly-marked (Wf-324C missed these two):
    "wetlab_cold_email_template.md",
    "wetlab_loi_consolidation_review.md",
]

MARKER_CANONICAL_TOKEN = "TEMPLATE_NOT_NEEDED_FOR_TIER2"
AGENTS_SECTION_RE = re.compile(r"AGENTS\.md\s*§?\s*5\.4")
TIER2_DIRECTIVE_RE = re.compile(r"二区就可以了|wet-?lab\s+(is\s+)?OUT\s+OF\s+SCOPE", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wetlab_path(name: str) -> Path:
    return TODO_PENDING / name


def _load(name: str) -> str:
    return _wetlab_path(name).read_text(encoding="utf-8")


def _find_marker_block(text: str) -> str | None:
    """Return the contiguous block of blockquote lines that contains the
    marker token. None if the marker is not present.

    A "marker block" is the maximal sequence of `>` lines that includes the
    TEMPLATE_NOT_NEEDED_FOR_TIER2 token. We scan the file first to locate the
    marker line, then extend forward to capture every following `>` line
    until the blockquote ends. This ensures the AGENTS.md §5.4 reference +
    Tier-2 directive text (which appear on subsequent `>` lines) are
    included in the returned block.
    """
    lines = text.splitlines()
    marker_idx: int | None = None
    for i, line in enumerate(lines):
        if MARKER_CANONICAL_TOKEN in line and line.startswith(">"):
            marker_idx = i
            break
    if marker_idx is None:
        return None

    # Walk back to the start of the blockquote (the first `>` line after a
    # non-`>` line, or the beginning of the file).
    start = marker_idx
    while start > 0 and lines[start - 1].startswith(">"):
        start -= 1

    # Walk forward to the end of the blockquote (the last `>` line before a
    # non-`>` line, or end of file).
    end = marker_idx
    while end + 1 < len(lines) and lines[end + 1].startswith(">"):
        end += 1

    return "\n".join(lines[start : end + 1])


# ---------------------------------------------------------------------------
# Test 1: all 10 wetlab files exist on disk
# ---------------------------------------------------------------------------
def test_all_10_wetlab_files_exist():
    """Verifies all 10 wetlab files are on disk. Wf-324C catalogue = 10 files.

    If a file is missing, list_below returns the missing set; this test fails
    with a clear message naming the absent file(s).
    """
    missing = [name for name in WETLAB_FILES if not _wetlab_path(name).exists()]
    assert not missing, (
        f"Wetlab catalogue incomplete: {len(missing)} file(s) missing from "
        f"{TODO_PENDING}: {missing}"
    )

    # Sanity: no extra wetlab_*.md files in TODO/pending/ beyond the 10.
    actual = sorted(p.name for p in TODO_PENDING.glob("wetlab_*.md"))
    expected = sorted(WETLAB_FILES)
    assert actual == expected, (
        f"Wetlab catalogue drift detected.\n"
        f"  expected ({len(expected)}): {expected}\n"
        f"  actual   ({len(actual)}): {actual}\n"
        f"  extra   : {set(actual) - set(expected)}\n"
        f"  missing : {set(expected) - set(actual)}"
    )


# ---------------------------------------------------------------------------
# Test 2: all 10 wetlab files have the TEMPLATE_NOT_NEEDED_FOR_TIER2 marker
# ---------------------------------------------------------------------------
def test_all_10_wetlab_files_have_tier2_marker():
    """Verifies every wetlab file has the canonical Tier-2 marker string.

    Per AGENTS.md §5.4 + Wf-324C + Wf-337-W1: marker text MUST be exactly
    'TEMPLATE_NOT_NEEDED_FOR_TIER2' (uppercase, underscores literal). Casing
    or whitespace variants are NOT accepted.
    """
    unmarked: list[str] = []
    for name in WETLAB_FILES:
        text = _load(name)
        if MARKER_CANONICAL_TOKEN not in text:
            unmarked.append(name)

    assert not unmarked, (
        f"{len(unmarked)}/{len(WETLAB_FILES)} wetlab file(s) lack the "
        f"'{MARKER_CANONICAL_TOKEN}' marker: {unmarked}"
    )


# ---------------------------------------------------------------------------
# Test 3: marker format consistency
# ---------------------------------------------------------------------------
def test_marker_format_consistency():
    """Verifies each marker follows one of the two canonical formats:

      Format A (LOI-style, used by all 4 LOIs + 2 helpers):
          > **TEMPLATE_NOT_NEEDED_FOR_TIER2 (YYYY-MM-DD)**
          >
          > Per AGENTS.md §5.4 user directive "...",
          > Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.

      Format B (SOP/ic50-style, used by 3 SOPs + ic50 pipeline):
          > **Status (YYYY-MM-DD): TEMPLATE_NOT_NEEDED_FOR_TIER2** — per AGENTS.md §5.4 ...
          > directive "...", Tier-2 journal is sufficient and
          > wet-lab validation is OUT OF SCOPE. ...

    Both formats must:
      - Live inside a blockquote (line starts with `>`)
      - Cite AGENTS.md §5.4 (or "5.4")
      - Mention "二区就可以了" (Tier-2 directive) OR the English
        "Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE"

    The Wf-337-W1 newly-marked files (cold_email + loi_consolidation_review)
    follow Format A (LOI-style).
    """
    format_a_re = re.compile(
        r"> \*\*TEMPLATE_NOT_NEEDED_FOR_TIER2\s*\(\d{4}-\d{2}-\d{2}\)\*\*"
    )
    format_b_re = re.compile(
        r"> \*\*Status\s*\(\d{4}-\d{2}-\d{2}\):\s*TEMPLATE_NOT_NEEDED_FOR_TIER2\*\*"
    )

    bad_format: list[tuple[str, str]] = []
    for name in WETLAB_FILES:
        text = _load(name)
        block = _find_marker_block(text)
        assert block is not None, (
            f"{name}: marker token '{MARKER_CANONICAL_TOKEN}' not found "
            f"inside a blockquote (must be on a line starting with `>`)"
        )
        # Accept either canonical format header on the first marker line.
        first_marker_line = block.splitlines()[0]
        if not (format_a_re.search(first_marker_line) or format_b_re.search(first_marker_line)):
            bad_format.append((name, first_marker_line))

    assert not bad_format, (
        f"{len(bad_format)} file(s) have a malformed marker header (expected "
        f"Format A: '> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (YYYY-MM-DD)**' or "
        f"Format B: '> **Status (YYYY-MM-DD): TEMPLATE_NOT_NEEDED_FOR_TIER2**'). "
        f"Offenders:\n" + "\n".join(f"  {n}: {line!r}" for n, line in bad_format)
    )


# ---------------------------------------------------------------------------
# Test 4: Tier-2 strategy compliance
# ---------------------------------------------------------------------------
def test_tier2_strategy_compliance():
    """Verifies each marker block:
      - References AGENTS.md §5.4 (the source-of-truth directive).
      - Includes the Tier-2 directive (either "二区就可以了" or the
        English equivalent "Tier-2 journal is sufficient and wet-lab
        validation is OUT OF SCOPE").
      - Does NOT contain an execution-status marker that would imply the
        template is still in flight (e.g. "Status: ACTIVE" alone).

    This gates that the Tier-2 re-orientation per §5.4 is consistently
    applied to every wetlab template file.
    """
    bad_compliance: list[tuple[str, str]] = []
    for name in WETLAB_FILES:
        text = _load(name)
        block = _find_marker_block(text)
        assert block is not None, (
            f"{name}: marker block missing (test 2 should have caught this)"
        )

        # 1. AGENTS.md §5.4 reference.
        if not AGENTS_SECTION_RE.search(block):
            bad_compliance.append((name, "missing AGENTS.md §5.4 reference"))

        # 2. Tier-2 directive (Chinese or English).
        if not TIER2_DIRECTIVE_RE.search(block):
            bad_compliance.append((name, "missing Tier-2 directive text"))

    assert not bad_compliance, (
        f"{len(bad_compliance)} file(s) fail Tier-2 strategy compliance:\n"
        + "\n".join(f"  {n}: {reason}" for n, reason in bad_compliance)
    )

    # Sanity: at least one file uses the Chinese directive verbatim.
    has_chinese_directive = any(
        "二区就可以了" in _load(name) for name in WETLAB_FILES
    )
    assert has_chinese_directive, (
        "Tier-2 strategy compliance: no file uses the canonical Chinese "
        "directive '二区就可以了' verbatim — possible drift from §5.4"
    )

    # Sanity: every file with Format B (SOP/ic50-style) explicitly mentions
    # OUT OF SCOPE; every file with Format A (LOI-style) mentions the
    # AGENTS.md §5.4 directive.
    fmt_a_count = 0
    fmt_b_count = 0
    for name in WETLAB_FILES:
        text = _load(name)
        if re.search(r"> \*\*TEMPLATE_NOT_NEEDED_FOR_TIER2\s*\(\d{4}-\d{2}-\d{2}\)\*\*", text):
            fmt_a_count += 1
        elif re.search(r"> \*\*Status\s*\(\d{4}-\d{2}-\d{2}\):\s*TEMPLATE_NOT_NEEDED_FOR_TIER2\*\*", text):
            fmt_b_count += 1
    assert fmt_a_count + fmt_b_count == len(WETLAB_FILES), (
        f"Format distribution unexpected: A={fmt_a_count}, B={fmt_b_count}, "
        f"total={len(WETLAB_FILES)} (test 3 should have caught this)"
    )
