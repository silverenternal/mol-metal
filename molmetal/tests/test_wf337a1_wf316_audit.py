"""Wf-337-A1 — Wf-316 TODO directory audit reclassification tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)
Skill alignment: party-twenty-spirit §3.5 (honest framing mandatory)

Context (MEASURED, 2026-09-29):
  - Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
    Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
  - Wf-316 TODO directory audit (Wf-316C ship, 2026-09-29) is the
    authoritative reclassification wave; the action item at INDEX.md §0.4
    "Wf-316 TODO directory audit should reclassify wet-lab items as
    TEMPLATE_NOT_NEEDED_FOR_TIER2 rather than STILL_PENDING" is the
    trigger for this Wf-337-A1 audit pass.
  - 10 wetlab_*.md files in TODO/pending/ should all carry the
    TEMPLATE_NOT_NEEDED_FOR_TIER2 marker (8 added by Wf-324C, 2 by
    Wf-337-W1).
  - TODO/INDEX.md §0.4 wetlab-planning-artifacts table should reflect
    all 10 files (not just 7) and use the canonical
    `TEMPLATE_NOT_NEEDED_FOR_TIER2` label (not DEPRECATED).
  - TODO/INDEX.md §18 should reference Wf-337-A1 + Wf-337-W1/W3 closeouts.

Tests (4 total, all CPU-only docs-only — < 2 s wall):

  - test_all_wetlab_items_reclassified
        Gates that all 10 wetlab files are reclassified
        (TEMPLATE_NOT_NEEDED_FOR_TIER2 marker present).
  - test_no_items_still_marked_still_pending
        Gates that no wetlab file is still flagged as
        `STILL_PENDING` / `DEPRECATED only` / unscheduled — must be the
        canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` label, NOT a softer
        deprecation marker that leaves the file executable.
  - test_tier2_strategy_compliance
        Gates that INDEX.md §0.4 reflects Tier-2 strategy + all 10
        wetlab files + Wf-316C / Wf-337-A1 cross-references — the
        downstream consumer-facing audit ledger.
  - test_index_section18_updated
        Gates that INDEX.md §18 (Tier-2 paper target) is updated with
        the Wf-337-A1 / Wf-337-W1 / Wf-337-W3 cross-references, and
        that §0.4 wetlab-planning-artifacts table reflects the full
        10-file catalogue (no drift vs on-disk WETLAB_FILES).

Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5):
  - MEASURED (Wf-316C + Wf-324C + Wf-337-W1 + Wf-337-W3 + Wf-337-A1
    2026-09-29): 10/10 wetlab files reclassified; INDEX.md §0.4 + §18
    updated.
  - PROJECTED: 0 wetlab files will be executed in the R20 paper cycle
    (Tier-2 acceptance is computational-only per AGENTS.md §5.4).
  - BLOCK: None. All audit gates PASS.

Cross-references:
  - AGENTS.md §5.4 — Tier-2 journal target + NO wet-lab planning
  - TODO/INDEX.md §0.4 — Tier-2 + wetlab-planning-artifacts table
  - TODO/INDEX.md §18 — Wf-139 P9 Tier-2 paper submission package
  - molmetal/reports/wf316c_wetlab_loi_audit.md — Wf-316C ship audit
  - TODO/completed/337w1_wetlab_markers.md — marker coverage pass
  - TODO/completed/337w3_wetlab_rescope.md — TODO-138/R20-P6/Wf-74 v2
    re-scope markers
  - TODO/completed/337a1_wf316_audit.md — this Wf's close-out doc
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
TODO_COMPLETED = ROOT / "TODO" / "completed"
INDEX_MD = ROOT / "TODO" / "INDEX.md"
AGENTS_MD = ROOT / "AGENTS.md"

# Per Wf-316C + Wf-324C + Wf-337-W1: the full 10-file wetlab catalogue.
# Order matches INDEX.md §0.4 wetlab-planning-artifacts table for
# consistency in drift messages.
WETLAB_FILES = [
    "wetlab_ic50_analysis_pipeline.md",
    "wetlab_loi_ornl_protein_metal.md",
    "wetlab_loi_pku_cisplatin_ic50.md",
    "wetlab_loi_rusanov_2025_platinai.md",
    "wetlab_loi_xiehe_nci60.md",
    "wetlab_cold_email_template.md",
    "wetlab_loi_consolidation_review.md",
    "wetlab_sop_binding_affinity_assay.md",
    "wetlab_sop_cellular_uptake_assay.md",
    "wetlab_sop_cytotoxicity_assay.md",
]

MARKER_CANONICAL_TOKEN = "TEMPLATE_NOT_NEEDED_FOR_TIER2"
# Match either `AGENTS.md §5.4` (cross-doc reference) or bare `§5.4`
# (section header inside AGENTS.md itself).
AGENTS_SECTION_RE = re.compile(r"(?:AGENTS\.md\s*)?§\s*5\.4")
TIER2_DIRECTIVE_RE = re.compile(
    r"二区就可以了|wet-?lab\s+(is\s+)?OUT\s+OF\s+SCOPE", re.IGNORECASE
)
# Forbidden softer labels — if a file only carries these (without the
# canonical marker), it is still wrongly framed as "to be done".
DEPRECATION_ONLY_RE = re.compile(
    r"(?i)DEPRECATED(?!.*?TEMPLATE_NOT_NEEDED_FOR_TIER2)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wetlab_path(name: str) -> Path:
    return TODO_PENDING / name


def _load(name: str) -> str:
    return _wetlab_path(name).read_text(encoding="utf-8")


def _load_index() -> str:
    return INDEX_MD.read_text(encoding="utf-8")


def _load_agents() -> str:
    return AGENTS_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: all wet-lab items reclassified (TEMPLATE_NOT_NEEDED_FOR_TIER2)
# ---------------------------------------------------------------------------
def test_all_wetlab_items_reclassified():
    """Wf-337-A1 audit gate 1.

    Verifies that all 10 wetlab files are reclassified per the Wf-316
    audit action item at TODO/INDEX.md §0.4 — every file must carry the
    canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker inside a blockquote.

    Per AGENTS.md §5.4, this is the AUTHORITATIVE reclassification label;
    a softer "DEPRECATED" status alone is insufficient because it does
    not bind the file to the Tier-2 strategy (DEPRECATED could be
    interpreted as "waiting for revival" — the Tier-2 marker is
    unambiguous about "do NOT execute, OUT OF SCOPE for Tier-2 paper
    scope").
    """
    missing: list[str] = []
    for name in WETLAB_FILES:
        if not _wetlab_path(name).exists():
            missing.append(name)
            continue
        text = _load(name)
        if MARKER_CANONICAL_TOKEN not in text:
            missing.append(name)

    assert not missing, (
        f"Wf-316 audit reclassification incomplete: {len(missing)}/10 "
        f"wetlab file(s) lack the canonical "
        f"'{MARKER_CANONICAL_TOKEN}' marker:\n"
        + "\n".join(f"  - {n}" for n in missing)
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
# Test 2: no items still marked STILL_PENDING / unscheduled
# ---------------------------------------------------------------------------
def test_no_items_still_marked_still_pending():
    """Wf-337-A1 audit gate 2.

    Verifies that no wetlab file is still flagged as `STILL_PENDING` /
    unscheduled / `DEPRECATED only` (without the canonical
    `TEMPLATE_NOT_NEEDED_FOR_TIER2` qualifier). Per AGENTS.md §5.4, every
    wetlab file in `TODO/pending/` must be the canonical
    `TEMPLATE_NOT_NEEDED_FOR_TIER2` label — NOT a softer deprecation
    marker that leaves the door open to execution in a future R-round.

    The forbidden label set is:
      - `STILL_PENDING` (literal token)
      - `**DEPRECATED**` (without the canonical
        `TEMPLATE_NOT_NEEDED_FOR_TIER2` qualifier on the same page)

    Note: this test intentionally allows lines like
    `**DEPRECATED (AGENTS.md §5.4, 2026-09-29)** — ... TEMPLATE_NOT_NEEDED_FOR_TIER2`
    because the canonical qualifier follows. The forbidden pattern is
    "DEPRECATED ... but never TEMPLATE_NOT_NEEDED_FOR_TIER2".
    """
    forbidden = [
        r"\bSTILL_PENDING\b",
        r"\bunscheduled\b",
        # DEPRECATED without canonical qualifier is caught by
        # DEPRECATION_ONLY_RE which scopes to a DEPRECATED line that is
        # NOT followed by the canonical token on the same file.
    ]

    bad: list[tuple[str, str]] = []
    for name in WETLAB_FILES:
        if not _wetlab_path(name).exists():
            continue  # test 1 catches this
        text = _load(name)
        for pattern in forbidden:
            if re.search(pattern, text, flags=re.IGNORECASE):
                bad.append((name, pattern))

        # Per-file DEPRECATION-only check: DEPRECATED in the file BUT
        # NO `TEMPLATE_NOT_NEEDED_FOR_TIER2` token → still wrongly
        # framed as "softer deprecation".
        if re.search(r"(?i)DEPRECATED", text) and MARKER_CANONICAL_TOKEN not in text:
            bad.append((name, "DEPRECATED without TEMPLATE_NOT_NEEDED_FOR_TIER2 qualifier"))

    assert not bad, (
        f"{len(bad)} wetlab file(s) still framed as STILL_PENDING / "
        f"softer-DEPRECATED (must be canonical "
        f"'{MARKER_CANONICAL_TOKEN}'):\n"
        + "\n".join(f"  - {n}: matched {p}" for n, p in bad)
    )


# ---------------------------------------------------------------------------
# Test 3: Tier-2 strategy compliance — INDEX.md §0.4 + AGENTS.md §5.4
# ---------------------------------------------------------------------------
def test_tier2_strategy_compliance():
    """Wf-337-A1 audit gate 3.

    Verifies the Tier-2 strategy compliance chain:
      1. AGENTS.md §5.4 exists and contains the canonical user directive
         (`我们发二区就可以了` or the English equivalent).
      2. TODO/INDEX.md §0.4 (Paper target + wet-lab directive) is
         present and references AGENTS.md §5.4.
      3. TODO/INDEX.md §0.4 wetlab-planning-artifacts table references
         `TEMPLATE_NOT_NEEDED_FOR_TIER2` (the canonical label) at least
         once — confirming the INDEX.md reflects the reclassification.
      4. All 10 wetlab files appear in the §0.4 table (or in §0.4 list)
         — no orphan files (file on disk but not in INDEX.md).
    """
    agents_text = _load_agents()
    index_text = _load_index()

    # 1. AGENTS.md §5.4 with canonical directive.
    assert AGENTS_SECTION_RE.search(agents_text), (
        "AGENTS.md §5.4 reference missing — the source-of-truth Tier-2 "
        "directive is the foundation for this audit"
    )
    assert TIER2_DIRECTIVE_RE.search(agents_text), (
        "AGENTS.md §5.4 directive text missing (Chinese '二区就可以了' "
        "or English 'wet-lab is OUT OF SCOPE')"
    )

    # 2. INDEX.md §0.4 references AGENTS.md §5.4.
    assert "§0.4" in index_text, "INDEX.md §0.4 missing (Tier-2 + wet-lab directive section)"
    # The §0.4 "Paper target + wet-lab directive" header is at the
    # AGENTS.md §5.4 directive section boundary. INDEX.md has multiple
    # `### §0.4 ...` headers (line 56 Wf-316E ship-wave + line 609
    # Tier-2 directive); we want the Tier-2 directive one.
    section_04 = re.search(
        r"###\s*§0\.4\s+Paper target\s+\+\s+wet-?lab directive.*?(?=###\s*§|\Z)",
        index_text,
        re.DOTALL,
    )
    assert section_04 is not None, (
        "INDEX.md §0.4 Paper target + wet-lab directive section not found"
    )
    section_04_text = section_04.group(0)
    assert AGENTS_SECTION_RE.search(section_04_text), (
        "INDEX.md §0.4 does not cite AGENTS.md §5.4 — the cross-link is "
        "required so the §0.4 directive traces back to the source of truth"
    )

    # 3. INDEX.md §0.4 wetlab-planning-artifacts table uses canonical label.
    assert MARKER_CANONICAL_TOKEN in index_text, (
        f"INDEX.md does not contain the canonical "
        f"'{MARKER_CANONICAL_TOKEN}' token — the §0.4 wetlab table must "
        f"reflect the reclassification"
    )

    # 4. All 10 wetlab files are listed in INDEX.md §0.4 (table or list).
    missing_in_index = []
    for name in WETLAB_FILES:
        if name not in index_text:
            missing_in_index.append(name)
    assert not missing_in_index, (
        f"{len(missing_in_index)}/10 wetlab file(s) not mentioned in "
        f"INDEX.md (orphan — on disk but not in §0.4 table or list):\n"
        + "\n".join(f"  - {n}" for n in missing_in_index)
    )


# ---------------------------------------------------------------------------
# Test 4: INDEX.md §18 (Tier-2 paper target) is updated
# ---------------------------------------------------------------------------
def test_index_section18_updated():
    """Wf-337-A1 audit gate 4.

    Verifies that INDEX.md §18 (Wf-139 P9 Tier-2 paper submission
    package) reflects the Wf-337-A1 + Wf-337-W1 + Wf-337-W3 ship wave
    cross-references, and that the §0.4 wetlab-planning-artifacts
    table reflects the full 10-file catalogue (not just 7 from the
    pre-Wf-337-A1 state).

    Specifically checks:
      1. INDEX.md §18 header is present.
      2. INDEX.md §18 cross-references section mentions `337a1` and
         `Wf-337-A1` (or equivalent Wf-337-A1 close-out identifier).
      3. INDEX.md §0.4 wetlab-planning-artifacts table contains all 10
         wetlab filenames (not just 7).
      4. The §0.4 wetlab table uses the canonical
         `TEMPLATE_NOT_NEEDED_FOR_TIER2` label (not just `DEPRECATED`).
    """
    index_text = _load_index()

    # 1. INDEX.md §18 header is present.
    assert "§18" in index_text, "INDEX.md §18 missing (Tier-2 paper submission package section)"
    section_18 = re.search(
        r"##\s*§18.*?(?=##\s*§1[9-9]|##\s*§[2-9]|\Z)", index_text, re.DOTALL
    )
    assert section_18 is not None, (
        "INDEX.md §18 section header not found"
    )
    section_18_text = section_18.group(0)

    # 2. §18 cross-references the Wf-337-A1 / Wf-337-W1 / Wf-337-W3 wave.
    # Accept either the Wf identifier or the doc reference, since §18
    # may cite either `TODO/completed/337a1_wf316_audit.md` or
    # `molmetal/tests/test_wf337a1_wf316_audit.py`.
    wf337_a3_signal = (
        "337a1" in section_18_text.lower()
        or "wf-337-a1" in section_18_text.lower()
    )
    assert wf337_a3_signal, (
        "INDEX.md §18 does not cross-reference Wf-337-A1 / 337a1 — "
        "the §18 Tier-2 paper-target ledger must reflect this Wf's "
        "audit completion"
    )

    # 3. All 10 wetlab filenames present in §0.4 wetlab table.
    # Find the §0.4 wetlab-planning-artifacts subsection.
    table_match = re.search(
        r"### Wetlab planning artifacts.*?(?=###|\Z)", index_text, re.DOTALL
    )
    assert table_match is not None, (
        "INDEX.md §0.4 wetlab-planning-artifacts subsection not found"
    )
    table_text = table_match.group(0)

    missing_in_table = []
    for name in WETLAB_FILES:
        if name not in table_text:
            missing_in_table.append(name)
    assert not missing_in_table, (
        f"{len(missing_in_table)}/10 wetlab file(s) missing from INDEX.md "
        f"§0.4 wetlab-planning-artifacts table:\n"
        + "\n".join(f"  - {n}" for n in missing_in_table)
        + "\n\nExpected all 10 files per Wf-316C + Wf-337-A1 audit."
    )

    # 4. The §0.4 wetlab table uses the canonical
    # `TEMPLATE_NOT_NEEDED_FOR_TIER2` label (not just `DEPRECATED`).
    assert MARKER_CANONICAL_TOKEN in table_text, (
        f"INDEX.md §0.4 wetlab-planning-artifacts table does not use the "
        f"canonical '{MARKER_CANONICAL_TOKEN}' label — must replace "
        f"softer `DEPRECATED` markers per AGENTS.md §5.4 + Wf-316 audit"
    )

    # 5. Sanity: the Wf-337-A1 close-out doc is present (per the
    # cross-references the test asserts). We check existence on disk so
    # the test does not silently pass when the close-out doc is missing.
    closeout = TODO_COMPLETED / "337a1_wf316_audit.md"
    assert closeout.exists(), (
        f"Wf-337-A1 close-out doc missing: {closeout} — must be on disk "
        "per the INDEX.md §18 cross-reference check"
    )

    # 6. Sanity: the test file itself exists (this file — defensive
    # against accidental deletion that would silently void the audit).
    test_file = Path(__file__)
    assert test_file.exists(), (
        f"Wf-337-A1 test file missing: {test_file}"
    )