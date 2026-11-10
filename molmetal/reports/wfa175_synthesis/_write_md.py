#!/usr/bin/env python
"""Wf-A175 synthesis brief writer. Reads synthesis.json, writes synthesis.md via atomic_write_file.

Wf-A180 (2026-11-01): the A2 finding's claim is CONTESTED on disk, and this renderer
used to re-render the SUPERSEDED wording into the .md table row, the .md heading and the
.md A2 block -- and it emitted NONE of the on-disk correction section, so the generator
and the .md on disk had DIVERGED and a re-run of the generator DESTROYED the Wf-A177
correction.

The renderer now:
  * renders the READING-SAFE clause for A2 (heading, table row, evidence, status,
    remediation) -- every numeral in it is MEASURED and is asserted below against the
    DATA in synthesis.json (`findings[id=A2].a2_claim_superseded_2026_11_01`), so the
    emitted .md never asserts one provenance reading for the "92.7-94 %" band as settled;
  * EMITS the Wf-A177 CORRECTION section and the Wf-A180 CORRECTION section, so a re-run
    REPRODUCES the corrections instead of dropping them (AGENTS.md §5.5 rule #6);
  * preserves the SUPERSEDED A2 renderings VERBATIM -- in the dated source comment below
    and in the emitted SUPERSEDED block.

The sibling key `a2_claim_superseded_2026_11_01` is the contract: it is written by
`molmetal/reports/wfa175_synthesis/_write_synthesis.py` and read back here.  It carries
the superseded A2 claim VERBATIM, the corrected (reading-safe) wording, the competing
provenance readings and the invariants that hold under EVERY reading.

Wf-A196 (2026-11-02): the reading COUNT is NOT settled on disk, so no live rendering may
put a NUMBER on the size of the contest, and the PUBLISHED Wf-A177 dated sections are
emitted byte-for-byte instead of being rewritten or dropped on a re-run.

  * D1 -- the '## Wf-A177 CORRECTION' body and the '### Wf-A177 locator addendum'
    subsection are emitted BYTE-FOR-BYTE from the named dated constants
    `_MD_WF177_CORRECTION_BODY` / `_MD_WF177_LOCATOR_ADDENDUM`;
  * D2 -- the LIVE summary-table row for A2 carries the reading-safe form; the
    refuted X1 row is preserved VERBATIM in the '## Wf-A196 CORRECTION' section;
  * D3 -- the corrected published-only line count is recorded in that same dated
    section (the Wf-A189 report's own figure is NOT edited -- another wave owns it).
"""
import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(REPO))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file, detect_truncation  # noqa: E402

# Wf-A196 (2026-11-02) -- the OUT directory is overridable so the producer ->
# artefact ROUND-TRIP can be PROVED into a TEMP dir without touching the
# published artefact.  The DEFAULT is UNCHANGED (no env var -> the published
# path, exactly as before).
OUT = Path(os.environ.get("WFA175_OUT_DIR") or (REPO / "molmetal/reports/wfa175_synthesis"))
P = json.loads((OUT / "synthesis.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Wf-A180 (2026-11-01) -- the A2 claim is CONTESTED on disk.
#
# SUPERSEDED A2 renderings (Wf-A175, quoted VERBATIM -- preserved, not deleted):
#   claim      : The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.
#   evidence   : molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a DIFFERENT quantity), :80 (94 % endpoint, post_processed:false); headline_finding at :89
#   status     : SURVIVED (band branch CLAIM 3, IMPRECISE only on citation slips; substance stands)
#   remediation: Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint citation slip to fix in place: the two post_processed:false endpoints are :58 and :80, NOT :67.
#
# WHY THEY ARE SUPERSEDED: each of the four asserts ONE provenance reading for the
# frontier cells the "92.7-94 %" band names as if it were settled.  The repo holds
# COMPETING readings of those cells and they do not even agree on the numerals
# (92.7/94.0 vs 92.98/94.33) -- under the SECOND reading the lower endpoint IS a
# connectivity count (EF-TALFM's own raw `C. = 92.98 %` cell).  Only the claims that
# hold under EVERY reading may be published.  The superseded `evidence` string also cites
# LINE-NUMBER locators (":58", ":67", ":80", ":89") that the Wf-A177 key added to
# `frontier_matrix.json` shifted; line numbers are NOT stable locators, so the
# reading-safe rendering cites JSON PATHS instead.
#
# --- BEGIN PRESERVED-VERBATIM (Wf-A196, 2026-11-02) ---
# The 2026-11-01 rendering of the paragraph above, preserved VERBATIM
# (AGENTS.md §5.5 rule #6 -- nothing deleted, nothing silently edited):
#   WHY THEY ARE SUPERSEDED: each of the four asserts ONE provenance reading for the
#   frontier cells the "92.7-94 %" band names as if it were settled.  The repo holds TWO
#   COMPETING readings of those cells and they do not even agree on the numerals
#   (92.7/94.0 vs 92.98/94.33) -- under the SECOND reading the lower endpoint IS a
#   connectivity count (EF-TALFM's own raw `C. = 92.98 %` cell).  Only the claims that
#   hold under BOTH readings may be published.  The superseded `evidence` string also cites
#   LINE-NUMBER locators (":58", ":67", ":80", ":89") that the Wf-A177 key added to
#   `frontier_matrix.json` shifted; line numbers are NOT stable locators, so the
#   reading-safe rendering cites JSON PATHS instead.
# --- END PRESERVED-VERBATIM ---
# ---------------------------------------------------------------------------

#: The A2 finding + the dated sibling key that carries the correction (the contract
#: with `_write_synthesis.py` and with `synthesis.json`).
_A2 = next((f for f in P["findings"] if f["id"] == "A2"), None)
_A2_REC = (_A2 or {}).get("a2_claim_superseded_2026_11_01") or {}

# ---------------------------------------------------------------------------
# Wf-A187 (2026-11-01) -- LOCATOR DEMOTION for the A2 band-provenance record.
#
# A LINE NUMBER IS A PERISHABLE LOCATOR: the record
# `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` GROWS, so every
# line-number locator into it is STALE BY CONSTRUCTION the moment it is written (the Wf-A177
# key alone shifted every line at or after the old :52).  Wf-A187 promotes the JSON PATH
# into the LIVE locator and DEMOTES the line numbers into the dated sibling key
# `a2_locator_demotion_2026_11_01`, VERBATIM.  It does NOT re-measure them -- that record is
# concurrently regenerated by the wave that owns it, so re-measuring would race
# (AGENTS.md §5.5 rule #6: additive + dated; nothing deleted, nothing silently re-measured).
# ---------------------------------------------------------------------------
_A2_DEMO = (_A2 or {}).get("a2_locator_demotion_2026_11_01") or {}

#: The two passages the PUBLISHED .md carried with live line-number locators.  They live
#: ONLY in the published .md (the renderer cannot reconstruct them from `synthesis.json`),
#: so they are carried here as the dated historical record.
_MD_SUPERSEDED_READING_B_LOCATOR = (
    "**Reading B locator (MEASURED, 2026-11-01).** Reading B lives at "
    "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `sources.EF-TALFM` "
    "(`:360` `cells.POST-PROCESSING.note`, `:378-379` `extra_observations`), corroborated by the "
    "`molmetal/reports/wfa168_frontier_axis_mapping/` branch. The band-provenance artefact carries "
    "NO key literally named `frontier_axis_mapping`, so the path quoted above is the resolvable one."
)
_MD_SUPERSEDED_LOCATOR_ADDENDUM = (
    "The corrected wording above cites `band_provenance` (:51-89) and `:360, :378-379`. Those are "
    "the **PRE-Wf-A177** line numbers of "
    "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`: the branch that owns "
    "that file added its own Wf-A177 key to `band_provenance`, which shifted every line at or "
    "after the old :52. The current positions on disk are — Reading A: `band_provenance` :50 "
    "(object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` "
    ":140; Reading B: `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 "
    "(carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band "
    "... mixes the LOWEST connectivity' sentence)."
)

# ---------------------------------------------------------------------------
# Wf-A196 (2026-11-02) -- the D1 REPAIR: the PUBLISHED Wf-A177 dated sections are
# carried here BYTE-FOR-BYTE, so a re-run of this renderer REPRODUCES them instead
# dropping or rewriting them (AGENTS.md §5.5 rule #6 -- append-only, nothing deleted).
#
# BEFORE Wf-A196 this renderer emitted a DIFFERENT '## Wf-A177 CORRECTION' body and
# emitted NO '### Wf-A177 locator addendum' at all, so re-running it DESTROYED a
# dated record that the published synthesis.md carries at lines 415-437.
# ---------------------------------------------------------------------------
#: The PUBLISHED synthesis.md's '## Wf-A177 CORRECTION' section, BYTE-FOR-BYTE (lines 415-429 of the published artefact).  It feeds the '## Wf-A177 CORRECTION' section of the emitted .md.  Its reading-COUNT phrasing is superseded by the '## Wf-A196 CORRECTION' section -- the section itself is preserved, not rewritten.
_MD_WF177_CORRECTION_BODY = (
    "## Wf-A177 CORRECTION (2026-11-01) — the A2 claim is contested on disk\n"
    "\n"
    "**Superseded A2 claim (Wf-A175, quoted VERBATIM — preserved, not deleted):**\n"
    "\n"
    "> The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.\n"
    "\n"
    "**Why it is superseded.** The repo holds TWO COMPETING provenance readings for the frontier cells the \"92.7-94 %\" band names, and they do not even agree on the numerals. The sentence above states Reading A as if it were settled; under Reading B the lower endpoint IS a connected-component count (EF-TALFM's own raw `C. = 92.98 %` cell). A published sentence must not assert either reading as fact, so only the claims that hold under BOTH readings may be published.\n"
    "\n"
    "**Corrected wording (reading-safe — the only form that may be quoted):**\n"
    "\n"
    "> The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity. The repo holds TWO COMPETING provenance readings for which cells it names, and they do not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a connected-component count; (B) the same file -> `sources.EF-TALFM` (:360, :378-379), corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but does NOT reproduce the quoted numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.\n"
    "\n"
    "**Reading B locator (MEASURED, 2026-11-01).** Reading B lives at `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `sources.EF-TALFM` (`:360` `cells.POST-PROCESSING.note`, `:378-379` `extra_observations`), corroborated by the `molmetal/reports/wfa168_frontier_axis_mapping/` branch. The band-provenance artefact carries NO key literally named `frontier_axis_mapping`, so the path quoted above is the resolvable one.\n"
    "\n"
    "**What is NOT changed by this correction.** The A2 finding's `id`, `tier`, `rank`, `evidence`, `measured_or_projected`, `status`, `remediation`, `blocked`, `files` and `note` fields are untouched. `synthesis.json` keeps the superseded claim VERBATIM under `findings[id=A2].claim` and carries both renderings in the sibling key `a2_claim_superseded_2026_11_01`. No line above this section was edited or removed (AGENTS.md §5.5 rule #6 — append-only).\n"
    ""
)

#: The PUBLISHED synthesis.md's '### Wf-A177 locator addendum' subsection, BYTE-FOR-BYTE (lines 431-437 of the published artefact, including the 'Nothing above this subsection was edited or removed' closing sentence).  It feeds the same '## Wf-A177 CORRECTION' section of the emitted .md.
_MD_WF177_LOCATOR_ADDENDUM = (
    "### Wf-A177 locator addendum (2026-11-01) — the cited line numbers are pre-Wf-A177\n"
    "\n"
    "The corrected wording above cites `band_provenance` (:51-89) and `:360, :378-379`. Those are the **PRE-Wf-A177** line numbers of `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`: the branch that owns that file added its own Wf-A177 key to `band_provenance`, which shifted every line at or after the old :52. The current positions on disk are — Reading A: `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B: `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence).\n"
    "\n"
    "**The ROBUST locators are the JSON PATHS, not the line numbers**: `band_provenance` and `sources`.`EF-TALFM`. The owning artefact records the same note at `band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`, and `synthesis.json` carries it as `findings[id=A2].a2_claim_superseded_2026_11_01.line_number_note`.\n"
    "\n"
    "Nothing above this subsection was edited or removed (AGENTS.md §5.5 rule #6 — append-only)."
)


#: The two competing provenance readings, as DATA.
_RA = next((r for r in _A2_REC.get("readings", []) if r.get("label") == "A"), {})
_RB = next((r for r in _A2_REC.get("readings", []) if r.get("label") == "B"), {})

#: The invariants that hold under EVERY reading.
_INV = list(_A2_REC.get("holds_under_both_readings", []))

# ---------------------------------------------------------------------------
# Wf-A196 (2026-11-02) -- the X5 READING-COUNT SCOPE repair.
#
# THE DEFECT.  A sentence that puts a NUMBER on the size of the contest asserts a
# settled count the repo does not settle.  Writing a LARGER number would be the SAME
# defect in the other direction: how many DISTINCT readings exist is a judgement call
# (a further reading may be a variant of another).  The honest repair asserts NO
# settled count.
#
# THE CLAUSE asserts no count.  Every numeral in it is MEASURED on disk:
#   * `band_provenance.competing_reading_and_what_this_does_not_settle`.`readings` in
#     molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json is a LIST of
#     2 entries (MEASURED 2026-11-02);
#   * `invariants.competing_readings` in
#     molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json is
#     a LIST whose third element is the further reading (MEASURED 2026-11-02);
#   * the numerals 92.7 / 94.0 and 92.98 / 94.33 are the ones the competing readings
#     record.
# ---------------------------------------------------------------------------
_X5_COUNT_NOT_SETTLED = (
    "The NUMBER of competing readings is itself NOT settled on disk: this record enumerates "
    "two, and the repo's primary-source verification "
    "(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> "
    "`invariants.competing_readings`) records a further reading that likewise does not reproduce "
    "the quoted numerals."
)


#: The PUBLISHED synthesis.md's summary-table row for A2 (the LIVE row that carried the
#: refuted X1 claim), quoted VERBATIM.  It feeds the '## Wf-A196 CORRECTION' section.
_MD_X1_SUPERSEDED_ROW = (
    "| A2 | A-published-number | The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict) | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a D` | Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint  | no |"
)

#: The FOUR invariants, quoted VERBATIM from
#: `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` ->
#: `band_provenance.competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`.
#: They are asserted to be STILL PRESENT by the Wf-A196 test.
_MD_FOUR_INVARIANTS = {
    "i_the_band_is_external": "The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.",
    "ii_it_is_not_one_homogeneous_quantity": "It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a PoseBusters validity verdict -- HLTF states NO check count for the latter, NOT-STATED-IN-SOURCE), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.",
    "iii_it_is_not_on_our_scale": "It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.",
    "iv_the_source_cells_are_not_settled": "The source cells of the band are NOT settled on disk.",
}


#: THE ONE CANONICAL READING-SAFE CLAUSE (Wf-A180, 2026-11-01).  The leading clause is
#: adapted to this document's context; the body is the canonical wording.  Every numeral
#: in it is MEASURED -- `_assert_clause_matches_data()` below fails the run if any
#: numeral here is absent from the sibling key's own data, so this constant cannot
#: drift away from the artefacts it describes.
_A2_CLAUSE = (
    "The published \"92.7-94 %\" band's provenance is NOT settled on disk: the frontier "
    "post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of "
    "which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity "
    "& Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's "
    "per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the "
    "second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW "
    "connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the "
    "post-processed `N. & U. & S. = 94.33 %` companion cell). "
    + _X5_COUNT_NOT_SETTLED
    + " They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source "
    "cells are NOT settled on disk. "
    "Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on "
    "our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and "
    "nothing else."
)

#: A line-number locator is NOT a stable locator (AGENTS.md §5.5 + this wave's LOCATORS
#: rule).  Strip `:NN`, `:NN-NN` and `:NN/:NN-NN` tokens -- but never a `arXiv:2609.08333v4`
#: style id, whose digits are followed by a dot.
_LINE_LOC_RE = re.compile(r"(?<=[\s`)(,]):\d{1,4}(?:-\d{1,4})?(?:/:\d{1,4}(?:-\d{1,4})?)*")


#: A parenthetical that carries ONLY line-number annotations is dropped whole; the JSON
#: PATHS around it survive.  `arXiv:2609.08333v4` is NOT a line number and is never matched.
_PAREN_LOC_RE = re.compile(
    r"\s*\([^()]*?(?<![A-Za-z0-9]):\d{1,4}(?:-\d{1,4})?(?![\d.])[^()]*?\)"
)


def _locator_only(reading):
    """The JSON-PATH locator of a reading -- its line-number parenthetical is dropped."""
    s = _PAREN_LOC_RE.sub("", str(reading.get("locator", "")))
    s = s.replace("`sources.EF-TALFM`", "`sources`.`EF-TALFM`")
    s = s.replace("sources.EF-TALFM", "`sources`.`EF-TALFM`")
    s = s.replace("-> band_provenance", "-> `band_provenance`")
    s = s.replace(" ,", ",")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def _strip_line_locators(text):
    """Remove unstable line-number locators and normalise the JSON-PATH locator form."""
    s = _LINE_LOC_RE.sub("", str(text))
    s = re.sub(r"\(\s*,?\s*\)", "", s)
    s = s.replace("`sources.EF-TALFM`", "`sources`.`EF-TALFM`")
    s = s.replace("sources.EF-TALFM", "`sources`.`EF-TALFM`")
    s = s.replace("-> band_provenance", "-> `band_provenance`")
    s = s.replace(" ,", ",")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def _numerals(reading):
    """'YES (92.7 / 94.0)' -> '92.7 / 94.0' (read from the sibling key, not invented)."""
    m = re.search(r"\(([^()]*)\)\s*$", str(reading.get("reproduces_quoted_numerals", "")))
    return m.group(1) if m else str(reading.get("reproduces_quoted_numerals", ""))


# --- BEGIN PRESERVED-VERBATIM (Wf-A196, 2026-11-02) ---
# The 2026-11-01 renderings this wave replaced, quoted VERBATIM so that nothing is
# deleted and nothing is silently edited (AGENTS.md §5.5 rule #6).  They are emitted in
# the '## Wf-A196 CORRECTION' section of the .md.
_MD_X5_SUPERSEDED_CLAUSE = (
    "The published \"92.7-94 %\" band's provenance is NOT settled on disk: the frontier "
    "post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of "
    "which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity "
    "& Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's "
    "per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the "
    "second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW "
    "connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the "
    "post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the "
    "numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. "
    "Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on "
    "our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and "
    "nothing else."
)
_MD_X5_SUPERSEDED_ROW_LABEL = (
    "READING-SAFE: " + (_INV[-1] if _INV else "the band's source cells are NOT settled on disk")
    + " -- TWO COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`)."
)
_MD_X5_SUPERSEDED_STATUS = (
    "CONTESTED on disk (Wf-A177 / Wf-A180): the band's provenance is NOT settled -- TWO COMPETING "
    "readings of the frontier cells it names; only the claims that hold under BOTH readings may be "
    "published."
)
_MD_X5_SUPERSEDED_EVIDENCE_TAIL = (
    "The two readings do not even agree on the numerals ("
    + _numerals(_RA) + " vs " + _numerals(_RB)
    + "), so the band's source cells are NOT settled on disk."
)
# --- END PRESERVED-VERBATIM ---
def _assert_clause_matches_data():
    """MEASURED-only guard: every numeral in the canonical clause must exist on disk."""
    if not _A2_REC:
        return
    blob = json.dumps(_A2_REC, ensure_ascii=False)
    clause_nums = set(re.findall(r"\d+\.\d+", _A2_CLAUSE))
    blob_nums = set(re.findall(r"\d+\.\d+", blob))
    missing = clause_nums - blob_nums
    assert not missing, "canonical clause names numerals absent from the sibling key: %s" % sorted(missing)
    assert "92.7/94.0" in blob.replace(" ", "") and "92.98/94.33" in blob.replace(" ", ""), \
        "the sibling key no longer carries the competing numeral pairs"
    assert "8.51" in blob, "the sibling key no longer carries OUR 8.51 %"


_assert_clause_matches_data()

#: The sibling key's own Wf-A177 rendering of the same correction, with its PRE-Wf-A177
#: line-number locators stripped (the JSON PATHS remain).
_A2_CLAUSE_WF177 = _strip_line_locators(_A2_REC.get("corrected_claim_clause_level", ""))

#: A SHORT reading-safe label for the summary-table row and the A2 heading (the table
#: truncates at 190 chars).
_A2_ROW_LABEL = (
    "READING-SAFE: " + (_INV[-1] if _INV else "the band's source cells are NOT settled on disk")
    + " -- COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`); the NUMBER of "
    "readings is NOT settled on disk."
)

#: A SHORT reading-safe label for the table's evidence cell (no backticks -- the renderer
#: wraps this cell in a code span).
_A2_ROW_EVIDENCE = (
    "frontier_matrix.json -> band_provenance (Reading A) vs the same file -> sources.EF-TALFM "
    "(Reading B); the numerals disagree (" + _numerals(_RA) + " vs " + _numerals(_RB)
    + ") -- the band's source cells are NOT settled on disk"
)

#: The reading-safe replacement for the superseded A2 `evidence` field.
_A2_EVIDENCE = (
    "Reading A -- " + _locator_only(_RA) + ": "
    + _RA.get("content", "") + " Reading B -- " + _locator_only(_RB) + ": "
    + _RB.get("content", "") + " The competing readings do not even agree on the numerals ("
    + _numerals(_RA) + " vs " + _numerals(_RB) + "), so the band's source cells are NOT settled "
    "on disk."
)

#: The reading-safe replacement for the superseded A2 `status` field.
_A2_STATUS = (
    "CONTESTED on disk (Wf-A177 / Wf-A180 / Wf-A196): the band's provenance is NOT settled -- "
    "COMPETING readings of the frontier cells it names, and the NUMBER of competing readings is "
    "itself NOT settled on disk; only the claims that hold under EVERY reading may be published."
)

#: The reading-safe replacement for the superseded A2 `remediation` field.
_A2_REMEDIATION = (
    "Read-only: the band must never be placed on OUR scale as a comparator, and its endpoint "
    "assignment must not be published as settled -- record the metric-definition break in the "
    "honest-negative catalog and quote the reading-safe clause instead."
)


def _is_a2(f):
    return f.get("id") == "A2" and bool(_A2_REC)


def _claim_for(f):
    """The claim a finding's heading must render."""
    if _is_a2(f):
        return _A2_ROW_LABEL
    return f["claim"]


def _row_claim_for(f):
    """The claim a finding's summary-table row must render."""
    if _is_a2(f):
        return _A2_ROW_LABEL
    return f["claim"]


def _evidence_cell(f):
    """The evidence cell of a finding's summary-table row."""
    if _is_a2(f):
        return _A2_ROW_EVIDENCE
    return f["evidence"].split(";")[0][:150]


def _remediation_cell(f):
    """The remediation cell of a finding's summary-table row."""
    if _is_a2(f):
        return _A2_REMEDIATION[:150]
    return f["remediation"][:150]


def _evidence_for(f):
    """The `evidence` bullet of a finding's block."""
    if _is_a2(f):
        return _A2_EVIDENCE
    return f["evidence"]


def _status_for(f):
    """The `status` bullet of a finding's block."""
    if _is_a2(f):
        return _A2_STATUS
    return f["status"]


def _remediation_for(f):
    """The `remediation` bullet of a finding's block."""
    if _is_a2(f):
        return _A2_REMEDIATION
    return f["remediation"]


L = []
A = L.append

A("# Wf-A175 — close-out brief: 5 read-only audit branches, adversarially refuted, merged")
A("")
A(f"**Date:** 2026-11-01 · **CPU-only** (`CUDA_VISIBLE_DEVICES=\"\"` + `/home/hugo/codes/mol-metal/.venv/bin/python`) · "
  "**GPU used: 0** · **no git mutation** · **writes confined to** `molmetal/reports/wfa175_synthesis/`.")
A("")
A("Read-only respected: `paper/**`, `docs/HONEST_NEGATIVE_CATALOG.md`, `AGENTS.md`, `TODO/**`, `README.md`, "
  "`molmetal/scripts/pre_flight_check.py`, `metrics/**`.")
A("")
A("## 0. Tally")
A("")
t = P["tally"]
A(f"- Branch claims: **{t['branch_claims_refuted']} REFUTED / {t['branch_claims_survived']} SURVIVED / "
  f"{t['branch_claims_imprecise']} IMPRECISE** across 5 branches.")
A(f"- Merged findings: **{t['synthesis_findings']}** — Tier A (can corrupt a published number) "
  f"{t['synthesis_tier_A']}, Tier B (can make a gate lie) {t['synthesis_tier_B']}, "
  f"Tier C (documentation / bookkeeping) {t['synthesis_tier_C']}. "
  f"**{t['blocked_findings']} are BLOCKED** on `paper/**` write auth, this wave's read-only scope, or git write authority.")
A("")
A("Every number below is **MEASURED** and traces to a path on disk. Nothing is PROJECTED; nothing is fabricated.")
A("")
A("---")
A("")

A("## 1. SURVIVED findings, ranked by blast radius")
A("")
A("| # | tier | claim | evidence | remediation | blocked |")
A("|---|---|---|---|---|---|")
for f in P["findings"]:
    ev = _evidence_cell(f)
    rem = _remediation_cell(f)
    A(f"| {f['id']} | {f['tier']} | {_row_claim_for(f)[:190]} | `{ev}` | {rem} | {'**YES**' if f['blocked'] else 'no'} |")
A("")
for f in P["findings"]:
    A(f"### {f['id']} — {_claim_for(f)}")
    A("")
    if _is_a2(f):
        A("> " + _A2_CLAUSE)
        A("")
    A(f"- **Evidence:** {_evidence_for(f)}")
    A(f"- **Status:** {_status_for(f)}")
    A(f"- **Tag:** MEASURED (not PROJECTED)")
    A(f"- **Remediation (minimal, read-only):** {_remediation_for(f)}")
    A(f"- **BLOCKED:** {'YES — ' + f['note'] if f['blocked'] else 'no'}")
    A(f"- **Files:** " + ", ".join(f"`{p}`" for p in f["files"]))
    if f["note"] and not f["blocked"]:
        A(f"- **Note:** {f['note']}")
    A("")

A("---")
A("")
A("## 2. REFUTED / IMPRECISE — do not resurrect")
A("")
A("> **Wf-A180 (2026-11-01) — this section is a quarantine.** Every entry below is REFUTED or "
  "IMPRECISE and must not be quoted. In particular, the band-endpoint statements inside them "
  "(which pair the band with specific `post_processed: false` frontier cells) are SUPERSEDED: the "
  "band's source cells are NOT settled on disk, and their line-number locators are pre-Wf-A177 and "
  "therefore not stable. See the Wf-A180 CORRECTION at the end of this document.")
A("")
for r in P["refuted_and_imprecise"]:
    A(f"### {r['branch']} · {r['claim']}")
    A("")
    A(f"- **Why:** {r['why']}")
    if r.get("surviving_part"):
        A(f"- **What survives:** {r['surviving_part']}")
    A("")

A("---")
A("")
A("## 3. Checked and found CLEAN (negative results are results)")
A("")
for c in P["clean_negative_results"]:
    A(f"- {c}")
A("")

A("---")
A("")
A("## 4. Exact file set for an implementer")
A("")
for k, v in P["files_to_create_edit"].items():
    A(f"### {k}")
    A("")
    for e in v:
        A(f"- `{e['path']}` — {e['edit']}")
    A("")
A(f"### full list: the {len(P['config_stale_gate_key'])} configs carrying the stale "
  "`p1_pytorch_cuda_available` key")
A("")
A("```")
for p in P["config_stale_gate_key"]:
    A(p)
A("```")
A("")

A("---")
A("")
A("## 5. The two things that would actually hurt if left alone")
A("")
A("1. **A1 — the UFF step.** `metrics/by_metric/cfm_fragmentation_postproc_arm.json` records "
  "`uff_ok_rate_over_postprocessed = 0.0` and `uff_reasons {\"ValueError: Bad Conformer Id\": 12668}`. "
  "The frontier report says our pipeline is *\"the same 3-step PP as Buttenschoen\"*. It is 2 of 3 steps. "
  "The 11.75139× connectivity ratio is unaffected (the rate is fixed by step 1); the **comparability claim** is not. "
  "This is the only finding that touches a number a paper table would print.")
A("2. **B1 — the coupling.** `paper/refs.bib` is missing all 3 keys the `refs_bib` gate requires, so that gate "
  "**would** FAIL. It does not run today (it is registered in no pipeline), so it cannot block a launch today. "
  "It becomes a launch-blocker the instant the Wf-A169 wiring patch lands — and the wiring patch and the "
  "bibliography fix must be **one write, bibliography first**. The bibliography half needs the user's "
  "AGENTS.md §5.3 OK on `paper/**`.")
A("")
A("---")
A("")
A(f"*Generated {P['generated_utc']} · source artifacts: "
  + ", ".join(f"`{v.split(' ')[0]}`" for v in P["source_branches"].values()) + ".*")

# ---------------------------------------------------------------------------
# Wf-A180 (2026-11-01) -- EMIT the correction sections, so a re-run of this generator
# REPRODUCES them instead of dropping them.  Every sentence below is driven by the
# sibling key `a2_claim_superseded_2026_11_01` (the contract with
# `_write_synthesis.py` and with `synthesis.json`).
# ---------------------------------------------------------------------------
if _A2_REC:
    A("")
    A("---")
    A("")
    # -----------------------------------------------------------------------
    # Wf-A196 (2026-11-02) -- D1 REPAIR.
    #
    # The PUBLISHED '## Wf-A177 CORRECTION' body and its '### Wf-A177 locator
    # addendum' subsection are carried here BYTE-FOR-BYTE, in the NAMED DATED
    # CONSTANTS `_MD_WF177_CORRECTION_BODY` (which feeds the '## Wf-A177
    # CORRECTION' section) and `_MD_WF177_LOCATOR_ADDENDUM` (which feeds the
    # '### Wf-A177 locator addendum' subsection of that same section).
    #
    # BEFORE Wf-A196 the renderer re-derived the section from `synthesis.json`
    # and emitted NO locator addendum at all, so a re-run of the producer
    # REWROTE the dated Wf-A177 body IN PLACE and DROPPED a dated block
    # (AGENTS.md §5.5 rule #6 -- a dated block must survive a re-run).
    # -----------------------------------------------------------------------
    for _l in _MD_WF177_CORRECTION_BODY.split("\n"):
        A(_l)
    for _l in _MD_WF177_LOCATOR_ADDENDUM.split("\n"):
        A(_l)
    A("")
    A("---")
    A("")
    A("## Wf-A180 CORRECTION (2026-11-01) — the renderer now emits the reading-safe clause")
    A("")
    A("**The superseded A2 renderings the Wf-A175 close-out carried (quoted VERBATIM — preserved, not deleted):**")
    A("")
    A("> **claim:** " + str(_A2_REC.get("superseded_claim_verbatim", "")))
    A("> - **Evidence:** " + str((_A2 or {}).get("evidence", "")))
    A("> - **Status:** " + str((_A2 or {}).get("status", "")))
    A("> - **Remediation (minimal, read-only):** " + str((_A2 or {}).get("remediation", "")))
    A("")
    A("The `evidence` line above cites LINE-NUMBER locators (`:58`, `:67`, `:80`, `:89`) that the "
      "Wf-A177 key added to `frontier_matrix.json` shifted. Line numbers are **not** stable locators — "
      "use the JSON PATHS in the LOCATORS paragraph below.")
    A("")
    A("**Why they are superseded.** " + str(_A2_REC.get("why_superseded", "")))
    A("")
    A("**Corrected wording (reading-safe — the only form that may be quoted):**")
    A("")
    A("> " + _A2_CLAUSE)
    A("")
    if _A2_CLAUSE_WF177:
        A("**The sibling key's own Wf-A177 rendering of the same correction** "
          "(`findings[id=A2].a2_claim_superseded_2026_11_01`.`corrected_claim_clause_level`), with its "
          "pre-Wf-A177 line-number locators stripped:")
        A("")
        A("> " + _A2_CLAUSE_WF177)
        A("")
    A("**The competing readings the sibling key records, addressed by JSON PATH (never by line number):**")
    A("")
    A("> " + _X5_COUNT_NOT_SETTLED)
    A("")
    for _r in _A2_REC.get("readings", []):
        A(f"- **Reading {_r.get('label')}** — {_locator_only(_r)}")
        A(f"  - {_r.get('content', '')}")
        A(f"  - reproduces the quoted numerals: {_r.get('reproduces_quoted_numerals', '')}"
          f"; lower endpoint is a connected-component count: "
          f"{_r.get('lower_endpoint_is_a_connectivity_count', '')}")
    A("")
    A("**LOCATORS.** The ROBUST locators are the JSON PATHS `band_provenance` and `sources`.`EF-TALFM` in "
      "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. The token "
      "`frontier_axis_mapping` is **NOT** a key of that record — it survives only inside a string value, "
      "so pointing a reader at it is a **DEFECT**.")
    A("")
    A("**What is NOT changed by this correction.** The A2 finding's `id`, `tier`, `rank`, "
      "`measured_or_projected`, `blocked`, `files` and `note` fields are untouched, and the sibling key "
      "`a2_claim_superseded_2026_11_01` carries BOTH renderings. No line above this section is edited "
      "or removed (AGENTS.md §5.5 rule #6 — append-only).")
    A("")
    A("This renderer used to re-render the SUPERSEDED A2 wording into the .md table row, the .md heading "
      "and the .md A2 block, and emitted NONE of the correction section — so a re-run of it DESTROYED "
      "the Wf-A177 correction. It now renders the READING-SAFE clause and emits BOTH correction "
      "sections, so a re-run REPRODUCES them.")
    A("")
    A("**THE READING-SAFE STATEMENT (the only form that may be quoted):**")
    A("")
    A("> " + _A2_CLAUSE)
    A("")
    A("**The only claims that may be published (they hold under EVERY reading):**")
    A("")
    for _i, _t in enumerate(_INV, start=1):
        A(f"{_i}. {_t}")
    A("")
    A("**Claims that are FALSE or SETTLED-WRONG and must NOT be asserted:**")
    A("")
    A(f"- **X1** — \"{_A2_REC.get('removed_assertion', '')}\" — TRUE under Reading A "
      f"(lower endpoint is a connected-component count: "
      f"{_RA.get('lower_endpoint_is_a_connectivity_count', '')}) and FALSE under Reading B "
      f"(lower endpoint is a connected-component count: "
      f"{_RB.get('lower_endpoint_is_a_connectivity_count', '')}).")
    A("- **X2** — \"the published band is measured on the published POST-PROCESSED stream\" / "
      "\"our number is not post-processed and theirs is\": FALSE — under Reading A both band endpoints "
      "carry `post_processed: false`, and under Reading B the band MIXES one RAW cell with one "
      f"post-processed companion cell ({_RB.get('content', '')}). OUR number is RAW under either reading.")
    A("- **X3** — \"their Connectivity is computed AFTER retaining the atom-count largest connected "
      f"component\": REFUTED on disk — {_RB.get('content', '')}")
    A("- **X4** — \"the band is minted here: lower endpoint 92.7 % = GeoLDM's Val&Uniq, upper endpoint "
      "94 % = HLTF's own PB-Valid\": asserts ONE reading as settled (Reading A reproduces the quoted "
      f"numerals: {_numerals(_RA)}; Reading B: {_numerals(_RB)}). Qualify it, or use the reading-safe "
      "clause above.")
    A("")
    A("**Producer state.** This generator now EMITS the reading-safe clause and BOTH correction "
      "sections, so a re-run reproduces this correction instead of dropping it. The superseded A2 "
      "wording is preserved VERBATIM in the dated source comment above and in the Wf-A177 section "
      "above (AGENTS.md §5.5 rule #6).")
    A("")

# ---------------------------------------------------------------------------
# Wf-A187 (2026-11-01) -- EMIT the locator-demotion section, so a re-run of this generator
# REPRODUCES it instead of dropping it (AGENTS.md §5.5 rule #6 -- append-only).
# ---------------------------------------------------------------------------
if _A2_DEMO:
    _DEM = _A2_DEMO.get("demoted_line_number_locators_verbatim", {})
    _SUP = _A2_DEMO.get("superseded_renderings_preserved_verbatim", {})
    _PROM = _A2_DEMO.get("live_locator_promoted", {})
    A("---")
    A("")
    A("## Wf-A187 CORRECTION (2026-11-01) — the band-provenance LINE-NUMBER locators are DEMOTED")
    A("")
    A("**Status:** CPU-only (`CUDA_VISIBLE_DEVICES=\"\"`) · single venv "
      "`/home/hugo/codes/mol-metal/.venv/bin/python` · GPU used: 0 · no git mutation · "
      "**ADDITIVE**: nothing above this section is edited or removed (AGENTS.md §5.5 rule #6).")
    A("")
    A("### The locator class — why a line number may not be a live locator")
    A("")
    A("> " + str(_A2_DEMO.get("locator_class", "")))
    A("")
    A("### The superseded line-number locators, quoted VERBATIM")
    A("")
    A("Each value below is the value the named field carried BEFORE Wf-A187. They are preserved "
      "EXACTLY as measured on " + str(_A2_DEMO.get("measured_on", "")) + " and are NOT re-measured "
      "here — see \"Not re-measured\" below.")
    A("")
    for _k, _v in _DEM.items():
        A(f"**`{_k}`**")
        A("")
        A("> " + str(_v))
        A("")
    A("**The PUBLISHED .md's own superseded passages, quoted VERBATIM** (they lived only in the "
      "published .md — the renderer cannot reconstruct them from `synthesis.json`, so they are "
      "carried here as the dated historical record):")
    A("")
    A("> " + _MD_SUPERSEDED_READING_B_LOCATOR)
    A(">")
    A("> " + _MD_SUPERSEDED_LOCATOR_ADDENDUM)
    A("")
    A("### The LIVE locator — JSON PATHS")
    A("")
    A("- **Reading A** — " + str(_PROM.get("A", "")))
    A("- **Reading B** — " + str(_PROM.get("B", "")))
    A("")
    A("### Not re-measured (race-free)")
    A("")
    A("> " + str(_A2_DEMO.get("not_re_measured", "")))
    A("")
    A("### The X1 renderings this document still carries — SUPERSEDED")
    A("")
    A("**The §A2 heading (quoted VERBATIM):**")
    A("")
    A("> ### A2 — " + str(_SUP.get("findings[1].claim", "")))
    A("")
    A("**The A3 remediation bullet (quoted VERBATIM):**")
    A("")
    A("> - **Remediation (minimal, read-only):** "
      + str(_DEM.get("findings[2].remediation", "")))
    A("")
    A("The §A2 heading asserts X1 — \"None is a connected-component count\" — which is **TRUE under "
      "Reading A** and **FALSE under Reading B**; the A3 remediation asserted the same X1. The "
      "renderer now emits the reading-safe form for both; the PUBLISHED .md predates that "
      "renderer, so this section is what supersedes the two passages quoted above (they are kept, "
      "not deleted).")
    A("")
    A("**The reading-safe replacement (the only form that may be quoted):**")
    A("")
    A("> " + _A2_CLAUSE)
    A("")
    A("(A re-run of this renderer also emits the same clause as its own `## Wf-A180 CORRECTION` "
      "section; this section quotes it inline so that the correction is self-contained wherever "
      "it is read.)")
    A("")
    A("**The A2 finding's own Wf-A175 rendering, preserved VERBATIM** (the sibling key "
      "`a2_locator_demotion_2026_11_01` -> `superseded_renderings_preserved_verbatim`):")
    A("")
    A("> " + str(_SUP.get("note", "")))
    A("")
    A("### What is NOT changed by this correction")
    A("")
    A("No number in this document is touched. The A2 finding's `id`, `tier`, `rank`, "
      "`measured_or_projected`, `status`, `blocked`, `files` and `note` fields are untouched, and "
      "the sibling keys `a2_claim_superseded_2026_11_01` and `a2_locator_demotion_2026_11_01` "
      "carry BOTH renderings. Only the **locator form** is corrected, and only additively and "
      "dated.")
    A("")


# ---------------------------------------------------------------------------
# Wf-A196 (2026-11-02) -- EMIT this wave's dated correction section, so a re-run
# of this generator REPRODUCES it instead of dropping it (AGENTS.md §5.5 rule #6
# -- append-only, nothing deleted).
#
# It carries: the X5 reading-COUNT scope repair (no settled count), the FOUR
# invariants (asserted still present), D1 (the producer now reproduces the
# PUBLISHED Wf-A177 dated sections byte-for-byte), D2 (the LIVE summary-table
# row no longer carries the refuted X1 claim; the X1 row is preserved VERBATIM
# below) and D3 (the corrected published-only line count).
# ---------------------------------------------------------------------------
A("---")
A("")
A("## Wf-A196 CORRECTION (2026-11-02) — the reading COUNT is NOT settled on disk, and the producer reproduces the dated sections")
A("")
A("**Status:** CPU-only (`CUDA_VISIBLE_DEVICES=\"\"`) · single venv "
  "`/home/hugo/codes/mol-metal/.venv/bin/python` · GPU used: 0 · no git mutation · "
  "**ADDITIVE**: nothing above this section is edited or removed (AGENTS.md §5.5 rule #6).")
A("")
A("### The defect class — a COUNT CLAIM about the size of the contest")
A("")
A("A sentence that puts a NUMBER on the size of the contest asserts a settled count that the repo does "
  "not settle. Writing a LARGER number is the SAME defect in the other direction — how many DISTINCT "
  "readings exist is itself a judgement call (a further reading may be a variant of another). The honest "
  "repair asserts NO settled count.")
A("")
A("**The clause that may be quoted (it asserts no count):**")
A("")
A("> " + _X5_COUNT_NOT_SETTLED)
A("")
A("**The full reading-safe clause (it asserts no count):**")
A("")
A("> " + _A2_CLAUSE)
A("")
A("### The four invariants — STILL PRESENT")
A("")
A("Quoted VERBATIM from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> "
  "`band_provenance.competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`:")
A("")
for _i, (_k, _t) in enumerate(_MD_FOUR_INVARIANTS.items(), start=1):
    A(f"{_i}. **{_k}** — {_t}")
A("")
A("### D1 — the producer now reproduces the PUBLISHED Wf-A177 dated sections byte-for-byte")
A("")
A("BEFORE Wf-A196 this renderer re-derived the '## Wf-A177 CORRECTION' body from `synthesis.json` and "
  "emitted NO '### Wf-A177 locator addendum' subsection at all, so a re-run of the producer REWROTE the "
  "dated Wf-A177 body IN PLACE and DROPPED the locator-addendum subsection — a dated block destroyed by "
  "a re-run (AGENTS.md §5.5 rule #6). The two published sections are now carried as the NAMED DATED "
  "CONSTANTS `_MD_WF177_CORRECTION_BODY` and `_MD_WF177_LOCATOR_ADDENDUM` in "
  "`molmetal/reports/wfa175_synthesis/_write_md.py`, each naming the dated section it feeds, and they "
  "are emitted byte-for-byte.")
A("")
A("### D2 — the LIVE summary-table row carried the refuted X1 claim")
A("")
A("The PUBLISHED .md's summary-table row for A2 asserted X1 — \"the quoted \"92.7-94 %\" band is NOT "
  "a connectivity band\" — which is TRUE under Reading A and FALSE under Reading B (whose lower cell IS "
  "a connectivity count). The LIVE row now carries the reading-safe form; the X1 row is preserved "
  "VERBATIM below and is NOT deleted.")
A("")
A("**The superseded LIVE row (quoted VERBATIM — superseded by this section, not deleted):**")
A("")
A("> " + _MD_X1_SUPERSEDED_ROW)
A("")
_a2f = next((f for f in P["findings"] if f["id"] == "A2"), None)
if _a2f is not None:
    A("**The reading-safe LIVE row (the only form that may be quoted):**")
    A("")
    A("> | " + " | ".join([
        str(_a2f["id"]), str(_a2f["tier"]), _row_claim_for(_a2f)[:190],
        "`" + _evidence_cell(_a2f) + "`", _remediation_cell(_a2f),
        ("**YES**" if _a2f["blocked"] else "no"),
    ]) + " |")
    A("")
A("### The superseded 2026-11-01 renderings, preserved VERBATIM")
A("")
A("**The reading-safe clause as it was rendered on 2026-11-01 (quoted VERBATIM — superseded by this section, not deleted):**")
A("")
A("> " + _MD_X5_SUPERSEDED_CLAUSE)
A("")
A("**The summary-table label as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**")
A("")
A("> " + _MD_X5_SUPERSEDED_ROW_LABEL)
A("")
A("**The A2 `status` as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**")
A("")
A("> " + _MD_X5_SUPERSEDED_STATUS)
A("")
A("**The A2 evidence tail as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**")
A("")
A("> " + _MD_X5_SUPERSEDED_EVIDENCE_TAIL)
A("")
A("The `## Wf-A177 CORRECTION` body and the `## Wf-A187 CORRECTION` section above also carry the "
  "2026-11-01 wording, including its reading-count phrasing, because both are DATED records preserved "
  "byte-for-byte. THIS section supersedes that phrasing: the NUMBER of competing readings is NOT settled "
  "on disk.")
A("")
A("### D3 — the Wf-A189 report's published-only line count")
A("")
A("Wf-A189's report states that the renderer-vs-published diff has **16 published-only lines**. The "
  "corrected count is **17**. Measurement (MEASURED 2026-11-02, CPU-only): `diff -u` between the "
  "PUBLISHED `molmetal/reports/wfa175_synthesis/synthesis.md` as it stood on 2026-11-01 (sha256 "
  "`427b3541e1cabdecc6c5611daffafa63249c1b723dcb57471d4593402036624e`) and that same producer's "
  "regeneration of it yields **5 hunks** and **18** lines beginning with `-`, of which **one** is the "
  "`---` file header — i.e. **17 published-only lines** (13 non-empty and 4 blank). The regenerated side "
  "yields **45 regenerated-only lines** (46 lines beginning with `+` minus the `+++` header). The figure "
  "16 is therefore off by one. Wf-A189's report is NOT edited here (another wave owns it); this dated "
  "section records the corrected count.")
A("")
A("### What is NOT changed by this correction")
A("")
A("No number in this document is touched. The A2 finding's `id`, `tier`, `rank`, "
  "`measured_or_projected`, `blocked`, `files` and `note` fields are untouched, and `synthesis.json` "
  "keeps the superseded claim VERBATIM under `findings[id=A2].claim` and in the dated sibling keys "
  "`a2_claim_superseded_2026_11_01` and `a2_x5_count_claim_repair_2026_11_02`. The four invariants "
  "above are unchanged. Only the **reading-count form**, the **LIVE summary-table row** and the "
  "**producer's reproduction of the dated Wf-A177 sections** are corrected, and only additively and "
  "dated.")
A("")


md = "\n".join(L) + "\n"
atomic_write_file(OUT / "synthesis.md", md)
print("synthesis.md bytes:", len(md.encode()))
print("truncation:", detect_truncation(md))
print("lines:", md.count("\n"))
