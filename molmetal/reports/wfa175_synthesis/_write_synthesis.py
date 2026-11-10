#!/usr/bin/env python
"""Wf-A175 synthesis writer. CPU-only. Writes only into molmetal/reports/wfa175_synthesis/."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(REPO))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file, detect_truncation  # noqa: E402

# Wf-A196 (2026-11-02) -- the OUT directory is overridable so the producer ->
# artefact ROUND-TRIP can be PROVED into a TEMP dir without touching the
# published artefact.  The DEFAULT is UNCHANGED (no env var -> the published
# path, exactly as before).
OUT = Path(os.environ.get("WFA175_OUT_DIR") or (REPO / "molmetal/reports/wfa175_synthesis"))
OUT.mkdir(parents=True, exist_ok=True)

F = []

# ---------------------------------------------------------------------------
# Wf-A196 (2026-11-02) -- the X5 READING-COUNT SCOPE repair.
#
# THE DEFECT.  A sentence that puts a NUMBER on the size of the contest
# asserts a settled count the repo does not settle.  It is FALSE at repo
# scope: the repo's primary-source verification records a FURTHER reading, and
# how many DISTINCT readings exist is itself a judgement call (a further
# reading may be a variant of another).  Asserting a LARGER count would be the
# SAME defect -- a settled count in the other direction.  The honest repair
# asserts NO settled count.
#
# THE CLAUSE BELOW asserts no count.  Every numeral in it is MEASURED on disk:
#   * `band_provenance.competing_reading_and_what_this_does_not_settle`.`readings`
#     in molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json
#     is a LIST of 2 entries (MEASURED 2026-11-02);
#   * `invariants.competing_readings` in
#     molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json
#     is a LIST whose third element is the further reading (MEASURED 2026-11-02);
#   * the numerals 92.7 / 94.0 and 92.98 / 94.33 are the ones the competing
#     readings record (see `a2_claim_superseded_2026_11_01`.`readings`).
# The number "3" is NOT asserted here: A186's `invariants.competing_readings_count`
# is cited BY ITS JSON PATH, never restated as a fact about the size of the contest.
#
# --- BEGIN PRESERVED-VERBATIM (Wf-A196, 2026-11-02) ---
# The defective renderings this wave replaced are quoted here VERBATIM so that
# nothing is deleted and nothing is silently edited (AGENTS.md §5.5 rule #6):
#   "the repo holds TWO competing readings"
#   "holds TWO COMPETING provenance readings"
# --- END PRESERVED-VERBATIM ---
# ---------------------------------------------------------------------------
_X5_COUNT_NOT_SETTLED = (
    "The NUMBER of competing readings is itself NOT settled on disk: the band-provenance "
    "record (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> "
    "`band_provenance.competing_reading_and_what_this_does_not_settle`.`readings`) enumerates "
    "two, and the repo's primary-source verification "
    "(`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> "
    "`invariants.competing_readings`) records a further reading that likewise does not reproduce "
    "the quoted numerals."
)


def finding(fid, tier, rank, claim, evidence, status, remediation, blocked, files, note=""):
    F.append({
        "id": fid,
        "tier": tier,
        "rank": rank,
        "claim": claim,
        "evidence": evidence,
        "measured_or_projected": "MEASURED",
        "status": status,
        "remediation": remediation,
        "blocked": blocked,
        "files": files,
        "note": note,
    })


# ---------------------------------------------------------------- TIER A
finding(
    "A1", "A-published-number", 1,
    'Our "same 3-step post-processing as Buttenschoen" is 2 of 3 steps: the UFF step ran on 0 of 12,668 molecules, so the frontier report\'s direct-comparability claim is unsound as written.',
    "metrics/by_metric/cfm_fragmentation_postproc_arm.json -> postprocessed_axis.uff_ok_rate_over_postprocessed = 0.0, uff_reasons {\"ValueError: Bad Conformer Id\": 12668}; claim site molmetal/reports/wfa175_frontier_comparability/report.md:79 (\"the same 3-step PP as Buttenschoen\") and report.md §6.2",
    "SURVIVED (frontier branch's own refutation named it THE most important correction)",
    "Read-only: amend report.md:79 to \"2 of 3 steps (LargestFragmentChooser -> Chem.AddHs); the UFF step is inapplicable to this corpus' metal complexes (0/12,668, ValueError: Bad Conformer Id)\" and strike \"directly comparable\" in §6.2. The 11.75139146567718x connectivity ratio itself is UNAFFECTED (the rate is fixed by step 1).",
    False,
    ["molmetal/reports/wfa175_frontier_comparability/report.md"],
    "This is the only finding that touches a number a paper table would print.",
)

finding(
    "A2", "A-published-number", 2,
    # ---------------------------------------------------------------------------
    # Wf-A177 (2026-11-01) -- the A2 claim is CONTESTED on disk.  APPEND-ONLY
    # (AGENTS.md §5.5 rule #6): the SUPERSEDED Wf-A175 wording is preserved
    # VERBATIM below; nothing is deleted and nothing is silently edited.
    #
    # SUPERSEDED WORDING (Wf-A175, verbatim):
    #   The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.
    #
    # WHY SUPERSEDED: the repo holds COMPETING provenance readings for the
    # cells the band names -- frontier_matrix.json -> band_provenance (:51-89)
    # vs the same file -> sources.EF-TALFM (:360, :378-379), corroborated by
    # molmetal/reports/wfa168_frontier_axis_mapping/; EF-TALFM arXiv:2609.08333v4
    # -- and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33);
    # the NUMBER of competing readings is itself NOT settled on disk.  'None is
    # a connected-component count' asserts Reading A as settled; under
    # Reading B the lower endpoint IS a connectivity count.  Only the claims that
    # hold under EVERY reading may be published: the band is EXTERNAL, it is NOT
    # one homogeneous quantity, it is NOT on our scale, and OUR 8.51 % IS a
    # connected-component count on the RAW emitted SMILES, and nothing else.
    #
    # --- BEGIN PRESERVED-VERBATIM (Wf-A196, 2026-11-02) ---
    # The 2026-11-01 rendering of the paragraph above is preserved VERBATIM
    # (AGENTS.md §5.5 rule #6 -- nothing deleted, nothing silently edited):
    #   WHY SUPERSEDED: the repo holds TWO COMPETING provenance readings for the
    #   cells the band names -- frontier_matrix.json -> band_provenance (:51-89)
    #   vs the same file -> sources.EF-TALFM (:360, :378-379), corroborated by
    #   molmetal/reports/wfa168_frontier_axis_mapping/; EF-TALFM arXiv:2609.08333v4
    #   -- and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33).
    #   'None is a connected-component count' asserts Reading A as settled; under
    #   Reading B the lower endpoint IS a connectivity count.  Only the claims that
    #   hold under BOTH readings may be published: the band is EXTERNAL, it is NOT
    #   one homogeneous quantity, it is NOT on our scale, and OUR 8.51 % IS a
    #   connected-component count on the RAW emitted SMILES, and nothing else.
    # --- END PRESERVED-VERBATIM ---
    # ---------------------------------------------------------------------------
    # Wf-A187 (2026-11-01) -- the A2 `claim` literal keeps the Wf-A175 rendering
    # BYTE-IDENTICAL, because `a2_claim_superseded_2026_11_01.superseded_in` records that
    # it does ("findings[id=A2].claim keeps the superseded wording byte-identical"), and
    # AGENTS.md §5.5 rule #6 forbids a silent edit of a published rendering.  The
    # READING-SAFE wording lives in the dated sibling key, and the dated demotion key
    # `a2_locator_demotion_2026_11_01` names this field as a superseded rendering whose
    # line-number locators are STALE BY CONSTRUCTION.  Do not navigate by them.
    "The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.",
    "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a DIFFERENT quantity), :80 (94 % endpoint, post_processed:false); headline_finding at :89",
    "SURVIVED (band branch CLAIM 3, IMPRECISE only on citation slips; substance stands)",
    "Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint citation slip to fix in place: the two post_processed:false endpoints are :58 and :80, NOT :67.",
    False,
    ["molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json", "docs/HONEST_NEGATIVE_CATALOG.md"],
    "docs/HONEST_NEGATIVE_CATALOG.md is READ-ONLY under this wave's hard rule 3.",
)

# ---------------------------------------------------------------------------
# Wf-A177 (2026-11-01) -- APPEND-ONLY record of the SUPERSEDED A2 claim, carried
# as a SIBLING of the A2 "claim" key so synthesis.json holds BOTH the
# superseded wording (verbatim) and the corrected wording (AGENTS.md §5.5 rule #6).
# ---------------------------------------------------------------------------
# --- BEGIN PRESERVED-VERBATIM (Wf-A196, 2026-11-02) ---
# The superseded renderings, preserved VERBATIM (AGENTS.md §5.5 rule #6 --
# nothing deleted, nothing silently edited).  These literals are the ONLY
# place in this producer where a settled reading COUNT is spelled out:
# every LIVE rendering asserts NO settled count.  Do not edit them -- they
# are the dated evidence.
#
_X5_SUPERSEDED_4_INDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM = (
        "The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one "
        "homogeneous quantity. The repo holds TWO COMPETING provenance readings for which cells it names, and they do "
        "not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` "
        "-> `band_provenance`: GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both "
        "`post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the "
        "same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a "
        "connected-component count; (B) the same file -> `sources`.`EF-TALFM`, corroborated by "
        "`molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN "
        "Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its "
        "post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but "
        "does NOT reproduce the quoted numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one "
        "homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW "
    "emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
)

_X5_SUPERSEDED_4_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM_CLAUSE_LEVEL = (
        "The quoted \"92.7-94 %\" band cannot be placed on our scale: the repo holds TWO COMPETING provenance readings "
        "for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` "
        "-> `band_provenance` vs the same file -> `sources`.`EF-TALFM`, corroborated by "
        "`molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree "
        "on the numerals (92.7/94.0 vs 92.98/94.33). Under BOTH readings the band is external, is NOT one homogeneous "
        "quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted "
    "SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
)

_X5_SUPERSEDED_5_FINDINGS_2_REMEDIATION = (
        "Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT "
        "on this wave's protected list). Replace the Open-Babel clause with the READING-SAFE statement: the band is "
        "EXTERNAL and is NOT one homogeneous quantity; the repo holds TWO COMPETING provenance readings for the "
        "frontier cells it names (frontier_matrix.json -> band_provenance :51-89 vs the same file -> sources.EF-TALFM "
        ":360/:378-379; EF-TALFM arXiv:2609.08333v4) that do not even agree on the numerals (92.7/94.0 vs "
        "92.98/94.33), so NO endpoint may be declared a connected-component count and the exact source cells are NOT "
    "settled on disk; OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else."
)

_X5_SUPERSEDED_4_FINDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_WHY_SUPERSEDED = (
        "The repo holds TWO COMPETING provenance readings for the frontier cells the '92.7-94 %' band names, and they "
        "do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). The superseded wording asserts Reading A as "
        "settled -- 'None is a connected-component count' -- but under Reading B the lower endpoint IS a connectivity "
    "count. A published sentence must not assert either reading as fact."
)

_X5_SUPERSEDED_5_INDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM = (
        "The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one "
        "homogeneous quantity. The repo holds TWO COMPETING provenance readings for which cells it names, and they do "
        "not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` "
        "-> `band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both "
        "`post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the "
        "same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a "
        "connected-component count; (B) the same file -> `sources.EF-TALFM` (:360, :378-379), corroborated by "
        "`molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN "
        "Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its "
        "post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but "
        "does NOT reproduce the quoted numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one "
        "homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW "
    "emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
)

_X5_SUPERSEDED_6_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM_CLAUSE_LEVEL = (
        "The quoted \"92.7-94 %\" band cannot be placed on our scale: the repo holds TWO COMPETING provenance readings "
        "for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` "
        "-> `band_provenance` :51-89 vs the same file -> `sources`.`EF-TALFM` :360/:378-379, corroborated by "
        "`molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree "
        "on the numerals (92.7/94.0 vs 92.98/94.33). Under BOTH readings the band is external, is NOT one homogeneous "
        "quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted "
    "SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
)

# The two defective PATTERNS themselves, quoted VERBATIM (the first is the
# rendering this wave removed; the second is the naive repair this wave
# refuses to install, because it is the SAME defect in the other direction).
# -- Wf-A196 (2026-11-02): the two Wf-A187 DEMOTION-SLOT values that the
# PUBLISHED synthesis.json carries at
# `findings[1].a2_locator_demotion_2026_11_01`.`demoted_line_number_locators_verbatim`.
# The producer's own copies of those two slots had DRIFTED from the published
# record; the drifted copies are preserved VERBATIM below as well (nothing is
# deleted -- AGENTS.md §5.5 rule #6), and the slots now reproduce the PUBLISHED
# values byte-for-byte so the producer -> artefact round-trip is exact.
_X5_DEMOTION_SLOT_CLAUSE_LEVEL_PUBLISHED = (
    "The quoted \"92.7-94 %\" band cannot be placed on our scale: the repo holds TWO COMPETING provenance readings "
    "for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` "
    "-> `band_provenance` :51-89 vs the same file -> `sources.EF-TALFM` :360/:378-379, corroborated by "
    "`molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree "
    "on the numerals (92.7/94.0 vs 92.98/94.33). Under BOTH readings the band is external, is NOT one homogeneous "
    "quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted "
    "SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
)

_X5_DEMOTION_SLOT_FINDINGS2_REMEDIATION_PUBLISHED = (
    "Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT "
    "on this wave's protected list). Replace the Open-Babel clause with the measured truth: the band's endpoints "
    "are frontier Validity&Uniq / PB-Valid / bond-type F1, none a connected-component count."
)

_X5_DEFECTIVE_PATTERN_TWO = "the repo holds TWO competing readings"
_X5_DEFECTIVE_PATTERN_THREE = "THREE competing readings"

# --- END PRESERVED-VERBATIM ---

_A2_SUPERSEDED_CLAIM_VERBATIM = "The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is."
_A2_CORRECTED_CLAIM_CLAUSE = "The quoted \"92.7-94 %\" band cannot be placed on our scale: the repo holds COMPETING provenance readings for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` :51-89 vs the same file -> `sources.EF-TALFM` :360/:378-379, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). " + _X5_COUNT_NOT_SETTLED + " Under EVERY reading the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk."
_A2_SUPERSEDED_RECORD = {
    "date": "2026-11-01",
    "wave": "Wf-A177",
    "finding_id": "A2",
    "rule": "AGENTS.md §5.5 rule #6 -- APPEND-ONLY: the superseded rendering is preserved VERBATIM, the correction is additive and dated.",
    "superseded_claim_verbatim": "The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.",
    "corrected_claim": "The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity. The repo holds COMPETING provenance readings for which cells it names, and they do not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance`: GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a connected-component count; (B) the same file -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but does NOT reproduce the quoted numerals (92.98 / 94.33). " + _X5_COUNT_NOT_SETTLED + " Under EVERY reading the band is external, is not one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.",
    "corrected_claim_clause_level": "The quoted \"92.7-94 %\" band cannot be placed on our scale: the repo holds COMPETING provenance readings for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` vs the same file -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). " + _X5_COUNT_NOT_SETTLED + " Under EVERY reading the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.",
    "why_superseded": "The repo holds COMPETING provenance readings for the frontier cells the '92.7-94 %' band names, and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). " + _X5_COUNT_NOT_SETTLED + " The superseded wording asserts Reading A as settled -- 'None is a connected-component count' -- but under Reading B the lower endpoint IS a connectivity count. A published sentence must not assert either reading as fact.",
    "readings": [
        {
            "label": "A",
            "locator": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance (:51-89; endpoint candidates :52-87, headline_finding :89)",
            "content": "The band's endpoints are GeoLDM's Validity & Uniq (92.7 %, post_processed: false, arXiv:2603.27113v1 Table 1(b) row 'Val&Uniq (%)' column 'GeoLDM') and HLTF's PB-Valid (94.0 %, post_processed: false, the same table's row 'PB-Valid (%)' column 'HLTF (ours)'), both on QM9; YuelBond's per-bond-type classification F1 (92.7 %, bioRxiv 10.1101/2025.05.06.652517v1) is a THIRD candidate for the same numeral.",
            "reproduces_quoted_numerals": "YES (92.7 / 94.0)",
            "lower_endpoint_is_a_connectivity_count": "NO",
            "current_line_numbers_on_disk": "`band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140"
        },
        {
            "label": "B",
            "locator": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM (:360 cells.POST-PROCESSING.note, :378-379 extra_observations), corroborated by molmetal/reports/wfa168_frontier_axis_mapping/; EF-TALFM arXiv:2609.08333v4",
            "content": "The band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell C. = 92.98 % (counted BEFORE the largest-component retention) with its post-processed N. & U. & S. = 94.33 % companion cell.",
            "reproduces_quoted_numerals": "NO (92.98 / 94.33)",
            "lower_endpoint_is_a_connectivity_count": "YES",
            "current_line_numbers_on_disk": "`sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence)"
        }
    ],
    "holds_under_both_readings": [
        "the band is EXTERNAL to Mol-Metal (it is a FRONTIER composite)",
        "it is NOT one homogeneous quantity",
        "it is NOT on our scale",
        "OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else",
        "the exact source cells of the band are NOT settled on disk"
    ],
    "removed_assertion": "None is a connected-component count, ours (8.51 %) is.",
    "superseded_in": [
        "molmetal/reports/wfa175_synthesis/_write_synthesis.py (the A2 finding's claim literal; the previous wording is preserved VERBATIM in a dated comment above it)",
        "molmetal/reports/wfa175_synthesis/synthesis.md (appended section '## Wf-A177 CORRECTION (2026-11-01) — the A2 claim is contested on disk')",
        "molmetal/reports/wfa175_synthesis/synthesis.json (this sibling key; findings[id=A2].claim keeps the superseded wording byte-identical)"
    ],
    "tag": "MEASURED",
    "line_number_note": "The ':51-89' range cited for Reading A and the ':360, :378-379' range cited for Reading B are the PRE-Wf-A177 line numbers of frontier_matrix.json; the Wf-A177 key added to that file's `band_provenance` shifted every line at or after the old :52. Current positions on disk: Reading A -> `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B -> `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence). The ROBUST locators are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line numbers. The owning artefact records the same note at `band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`."
}


# ---------------------------------------------------------------------------
# Wf-A187 (2026-11-01) -- LOCATOR DEMOTION for the A2 band-provenance record.
#
# A LINE NUMBER IS A PERISHABLE LOCATOR.  The record
# `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` GROWS (the wave
# that owns it keeps adding dated keys), so every line-number locator into it is STALE BY
# CONSTRUCTION the moment it is written -- the Wf-A177 key alone shifted every line at or
# after the old :52.  The ROBUST locator for that record is a JSON PATH: `band_provenance`
# and `sources`.`EF-TALFM`.
#
# Wf-A187 promotes the JSON PATH into the LIVE locator and DEMOTES the line numbers into
# this dated key, VERBATIM.  It does NOT re-measure them: the record is concurrently
# regenerated by the wave that owns it, so re-measuring would race (AGENTS.md §5.5 rule #6
# -- additive + dated; nothing is deleted, nothing is silently re-measured).
# ---------------------------------------------------------------------------
_A2_LOCATOR_DEMOTION = {
    "date": "2026-11-01",
    "wave": "Wf-A187",
    "rule": "AGENTS.md §5.5 rule #6 -- APPEND-ONLY: the superseded rendering is preserved VERBATIM, the correction is additive and dated.",
    "locator_class": ("A LINE NUMBER IS A PERISHABLE LOCATOR. The record "
        "`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` GROWS, so every "
        "line-number locator into it is STALE BY CONSTRUCTION the moment it is written. The ROBUST "
        "locator for that record is a JSON PATH: `band_provenance` and `sources`.`EF-TALFM`."),
    "measured_on": "2026-11-01",
    "not_re_measured": ("NOT re-measured by Wf-A187 (race-free). The record is concurrently "
        "regenerated by the wave that owns it, so re-measuring the line numbers would race against "
        "that wave. The historical values below are preserved EXACTLY as measured on 2026-11-01."),
    "demoted_line_number_locators_verbatim": {
        "findings[1].a2_claim_superseded_2026_11_01.readings[0].locator": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance (:51-89; endpoint candidates :52-87, headline_finding :89)",
        "findings[1].a2_claim_superseded_2026_11_01.readings[0].current_line_numbers_on_disk": "`band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140",
        "findings[1].a2_claim_superseded_2026_11_01.readings[1].locator": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM (:360 cells.POST-PROCESSING.note, :378-379 extra_observations), corroborated by molmetal/reports/wfa168_frontier_axis_mapping/; EF-TALFM arXiv:2609.08333v4",
        "findings[1].a2_claim_superseded_2026_11_01.readings[1].current_line_numbers_on_disk": "`sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence)",
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim": _X5_SUPERSEDED_5_INDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM,
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level": _X5_DEMOTION_SLOT_CLAUSE_LEVEL_PUBLISHED,
        "findings[1].a2_claim_superseded_2026_11_01.line_number_note": "The ':51-89' range cited for Reading A and the ':360, :378-379' range cited for Reading B are the PRE-Wf-A177 line numbers of frontier_matrix.json; the Wf-A177 key added to that file's `band_provenance` shifted every line at or after the old :52. Current positions on disk: Reading A -> `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B -> `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence). The ROBUST locators are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line numbers. The owning artefact records the same note at `band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`.",
        "findings[2].remediation": _X5_DEMOTION_SLOT_FINDINGS2_REMEDIATION_PUBLISHED,
    },
    "superseded_renderings_preserved_verbatim": {
        "note": ("These three fields are the Wf-A175 renderings of the A2 finding, preserved "
            "BYTE-IDENTICAL (AGENTS.md §5.5 rule #6) and quoted VERBATIM inside the dated "
            "'## Wf-A177 CORRECTION' section of synthesis.md. Their line-number locators are "
            "STALE BY CONSTRUCTION -- do not navigate by them; use the JSON PATHS."),
        "findings[1].claim": "The quoted \"92.7-94 %\" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.",
        "findings[1].evidence": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a DIFFERENT quantity), :80 (94 % endpoint, post_processed:false); headline_finding at :89",
        "findings[1].remediation": "Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint citation slip to fix in place: the two post_processed:false endpoints are :58 and :80, NOT :67.",
    },
    "live_locator_promoted": {
        "A": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance`",
        "B": ("molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> "
              "`sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; "
              "EF-TALFM arXiv:2609.08333v4"),
    },
}

# Wf-A187 (2026-11-01) -- promote the JSON PATH into the LIVE locator of each reading and
# DEMOTE the line numbers (the demotion record itself is spliced in below).
_A2_DEMOTED_PTR = (
    "DEMOTED (STALE BY CONSTRUCTION -- the record grows). The line numbers this field used to "
    "carry are preserved VERBATIM in `a2_locator_demotion_2026_11_01` -> "
    "`demoted_line_number_locators_verbatim` -> "
    "\"findings[1].a2_claim_superseded_2026_11_01.readings[%s].current_line_numbers_on_disk\". "
    "The ROBUST locator is the JSON PATH in `locator`."
)
for _i, _r in enumerate(_A2_SUPERSEDED_RECORD["readings"]):
    _r["locator"] = _A2_LOCATOR_DEMOTION["live_locator_promoted"][_r["label"]]
    _r["current_line_numbers_on_disk"] = _A2_DEMOTED_PTR % (_i,)
_A2_SUPERSEDED_RECORD["line_number_note"] = (
    "The line numbers this record used to carry are DEMOTED: they are STALE BY CONSTRUCTION (the "
    "record grows). The ROBUST locators are the JSON PATHS -- `band_provenance` and "
    "`sources`.`EF-TALFM` -- not the line numbers. The historical line numbers are preserved "
    "VERBATIM in `a2_locator_demotion_2026_11_01` -> `demoted_line_number_locators_verbatim`. "
    "The owning artefact records the same note at "
    "`band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`."
)

# ---------------------------------------------------------------------------
# Wf-A196 (2026-11-02) -- the X5 READING-COUNT SCOPE repair, as a DATED sibling key.
#
# Every rendering this wave replaced is preserved VERBATIM below (AGENTS.md §5.5
# rule #6 -- nothing deleted, nothing silently edited).  The repaired renderings
# live in the fields themselves; this key is the dated record of what they used
# to say.
# ---------------------------------------------------------------------------
_A2_X5_COUNT_CLAIM_REPAIR = {
    "date": "2026-11-02",
    "wave": "Wf-A196",
    "rule": "AGENTS.md §5.5 rule #6 -- APPEND-ONLY: the superseded renderings are preserved VERBATIM, the correction is additive and dated.",
    "defect_class": (
        "X5 -- READING-COUNT SCOPE.  A sentence that puts a NUMBER on the size of the contest "
        "(the defective rendering is quoted VERBATIM in "
        "`superseded_renderings_verbatim`.`defective_pattern_two_readings`) asserts a settled "
        "count the repo does not settle."
    ),
    "why_a_bare_three_would_be_the_same_defect": (
        "Writing the quoted pattern in "
        "`superseded_renderings_verbatim`.`defective_pattern_three_readings` would assert a "
        "settled count in the other direction: how many DISTINCT readings exist is a judgement "
        "call (a further reading may be a variant of another).  The repair therefore asserts NO "
        "settled count."
    ),
    "count_not_settled_clause": _X5_COUNT_NOT_SETTLED,
    "measured_on": "2026-11-02",
    "measurement_paths": [
        "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.competing_reading_and_what_this_does_not_settle.readings (a LIST of 2 entries; MEASURED 2026-11-02)",
        "molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> invariants.competing_readings (a LIST whose third element is the further reading; MEASURED 2026-11-02)",
        "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.competing_reading_and_what_this_does_not_settle.the_readings_disagree_on_the_numerals (the 92.7/94.0 vs 92.98/94.33 disagreement)",
    ],
    "superseded_renderings_verbatim": {
        "defective_pattern_two_readings": _X5_DEFECTIVE_PATTERN_TWO,
        "defective_pattern_three_readings": _X5_DEFECTIVE_PATTERN_THREE,
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim": _X5_SUPERSEDED_4_INDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM,
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level": _X5_SUPERSEDED_4_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM_CLAUSE_LEVEL,
        "findings[1].a2_claim_superseded_2026_11_01.why_superseded": _X5_SUPERSEDED_4_FINDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_WHY_SUPERSEDED,
        "findings[2].remediation": _X5_SUPERSEDED_5_FINDINGS_2_REMEDIATION,
        "findings[1].a2_locator_demotion_2026_11_01.demoted_line_number_locators_verbatim.findings[1].a2_claim_superseded_2026_11_01.corrected_claim": _X5_SUPERSEDED_5_INDINGS_1_A2_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM,
        "findings[1].a2_locator_demotion_2026_11_01.demoted_line_number_locators_verbatim.findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level": _X5_SUPERSEDED_6_CLAIM_SUPERSEDED_2026_11_01_CORRECTED_CLAIM_CLAUSE_LEVEL,
    },
    "repaired_renderings": {
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim": _A2_SUPERSEDED_RECORD["corrected_claim"],
        "findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level": _A2_SUPERSEDED_RECORD["corrected_claim_clause_level"],
        "findings[1].a2_claim_superseded_2026_11_01.why_superseded": _A2_SUPERSEDED_RECORD["why_superseded"],
    },
    "locator_note": (
        "The line-number locators inside the two repaired renderings are UNCHANGED by the Wf-A196 "
        "repair and were NOT re-measured: the Wf-A187 demotion record still carries the historical "
        "line numbers VERBATIM."
    ),
    "producer_published_drift_repaired_2026_11_02": {
        "defect_class": (
            "PRODUCER-vs-PUBLISHED DRIFT.  Two slots of the Wf-A187 demotion record "
            "(`a2_locator_demotion_2026_11_01`.`demoted_line_number_locators_verbatim`) carried "
            "copies that had DRIFTED from the PUBLISHED synthesis.json, so re-running the producer "
            "would have REWRITTEN a dated Wf-A187 record with text the published artefact does not "
            "carry."
        ),
        "drifted_slots": [
            "findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level",
            "findings[2].remediation",
        ],
        "repair": (
            "The producer now emits the PUBLISHED value for both slots, byte-for-byte, so the "
            "producer -> artefact round-trip is exact.  The DRIFTED producer copies are preserved "
            "VERBATIM in `superseded_renderings_verbatim` (nothing is deleted -- AGENTS.md §5.5 "
            "rule #6)."
        ),
        "measured_on": "2026-11-02",
        "measurement": (
            "MEASURED by diffing the published synthesis.json against a TEMP-dir regeneration of "
            "the producer (WFA175_OUT_DIR), leaf by leaf.  After the repair the only changed "
            "leaves are the four X5 renderings this wave repaired plus `generated_utc`; the "
            "numeric-leaf set is IDENTICAL (24 leaves, 0 moved) and NO leaf was removed."
        ),
        "tag": "MEASURED",
    },
    "tag": "MEASURED",
}

_a2_idx = next(i for i, f in enumerate(F) if f["id"] == "A2")
_a2_rebuilt = {}
for _k, _v in F[_a2_idx].items():
    _a2_rebuilt[_k] = _v
    if _k == "claim":
        _a2_rebuilt["a2_claim_superseded_2026_11_01"] = _A2_SUPERSEDED_RECORD
        # Wf-A187 (2026-11-01) -- the dated locator-demotion record, as a SIBLING of the
        # dated claim-supersession record (both additive + dated).
        _a2_rebuilt["a2_locator_demotion_2026_11_01"] = _A2_LOCATOR_DEMOTION
        # Wf-A196 (2026-11-02) -- the dated X5 reading-count repair record, as a SIBLING of
        # the two records above (additive + dated).
        _a2_rebuilt["a2_x5_count_claim_repair_2026_11_02"] = _A2_X5_COUNT_CLAIM_REPAIR
F[_a2_idx] = _a2_rebuilt

finding(
    "A3", "A-published-number", 3,
    'The Open-Babel mis-attribution of the band is still LIVE in the generator, not just in stale prose: molmetal/scripts/wfa161_frontier_labelling.py:934 still emits "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing", so re-running it re-emits the error.',
    "molmetal/scripts/wfa161_frontier_labelling.py:930-936 (re-measured by synthesis; string present verbatim)",
    "SURVIVED (frontier CLAIM 3, IMPRECISE -> the correction is that the mis-attribution is NOT single)",
    # Wf-A177 (2026-11-01) -- the same contested assertion was carried here as an
    # instruction to write "none a connected-component count" into another
    # artefact.  SUPERSEDED WORDING (Wf-A175, verbatim):
    #   "Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace the Open-Babel clause with the measured truth: the band's endpoints are frontier Validity&Uniq / PB-Valid / bond-type F1, none a connected-component count.",
    # Replaced with the reading-safe instruction below; the previous wording is
    # preserved VERBATIM (AGENTS.md §5.5 rule #6).
    "Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace the Open-Babel clause with the READING-SAFE statement: the band is EXTERNAL and is NOT one homogeneous quantity; the repo holds COMPETING provenance readings for the frontier cells it names (frontier_matrix.json -> band_provenance :51-89 vs the same file -> sources.EF-TALFM :360/:378-379; EF-TALFM arXiv:2609.08333v4) that do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). " + _X5_COUNT_NOT_SETTLED + " So NO endpoint may be declared a connected-component count and the exact source cells are NOT settled on disk; OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.",
    False,
    ["molmetal/scripts/wfa161_frontier_labelling.py"],
    "Only writer-side defect in the tier: a script that would regenerate the false statement.",
)

finding(
    "A4", "A-published-number", 4,
    'The band census is self-inflating and blind to the gitignored tree: 82 sites / 34 files is COMPLETE ONLY for tracked+untracked-not-ignored; 21 gitignored files outside the branch carry the band on 81 lines, including the very artifact a band-provenance audit should have read (Wf-A171 band_classification.json, 32 band lines).',
    "molmetal/reports/wfa175_band_provenance/sites.json (n_sites=82, distinct files=34, population=\"git ls-files --cached + git ls-files --others --exclude-standard\"); .gitignore:158 (molmetal/reports/**/*.json); molmetal/reports/wfa171_band_publication_audit/band_classification.json = 34,046 B / 625 lines / 32 band-bearing lines (re-measured by synthesis)",
    "SURVIVED (band CLAIM 1 SURVIVED + the refutation's scope-cap correction; Wf-A171 OPEN-5 already named it)",
    "Read-only: re-run the census with the gitignored report population included, and close Wf-A171's OPEN-5. No GPU. This is a census-scope repair, not a code repair.",
    False,
    ["molmetal/reports/wfa175_band_provenance/_census.py"],
    "A guard's own census is not a coverage measurement - the same lesson the branch teaches, applied to itself.",
)

finding(
    "A5", "A-published-number", 5,
    "The band report's own §2.2 is INVERTED against its own sites.json: report.md:106 says \"For 6 of the 7 `post` sites the statement is *correct and per-cell*\", but sites.json shows 6 of the 7 are the blanket-REFUTED ones; only molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:68 is correct.",
    "molmetal/reports/wfa175_band_provenance/report.md:106 vs molmetal/reports/wfa175_band_provenance/sites.json (raw_or_post Counter {unstated 66, raw 9, post 7}; post sites = catalog:20817, wfa155_ligand_connected_rate.json:228, wfa155_metal_coordination_sphere.json:26, wfa155_posebusters_validity.json:608, frontier_matrix.md:68, wfa161_frontier_labelling/report.md:139, wfa161_frontier_labelling.py:934)",
    "SURVIVED (band CLAIM 2 IMPRECISE -> this inversion is the correction)",
    "Read-only: correct report.md:106 to state that 6 of the 7 `post` sites are the blanket-REFUTED ones and only frontier_matrix.md:68 is correct-and-per-cell. A repairer following §2.2 verbatim would leave 6 wrong sites unrepaired.",
    False,
    ["molmetal/reports/wfa175_band_provenance/report.md"],
)

finding(
    "A6", "A-published-number", 6,
    "The frontier report's §2.1 ratio column contradicts its OWN computed artifact: 4 of the ratios printed in the report disagree with molmetal/reports/wfa175_frontier_comparability/tools/ratios.json.",
    "report.md:66 1.03175x vs ratios.json 1.0319905213270142; report.md:69 1.48071x vs 1.481042654028436; report.md:72 1.27766x vs 1.2776287210936503; report.md:76 1.15501x vs the GEOM row of ratios.json",
    "SURVIVED (frontier CLAIM 2 SURVIVED; this ratio-column defect is the IMPRECISE part)",
    "Read-only: regenerate report.md's §2.1 ratio column from tools/ratios.json rather than re-typing it. Documentation-only blast radius (rounding magnitude), but it is a self-inconsistent provenance chain in a paper-facing table.",
    False,
    ["molmetal/reports/wfa175_frontier_comparability/report.md"],
)

# ---------------------------------------------------------------- TIER B
finding(
    "B1", "B-gate-truth", 7,
    "The refs_bib gate would FAIL (paper/refs.bib has 0 of the 3 required keys) but is DEAD in the working tree: it is DEFINED at pre_flight_check.py:11118 and registered in NO pipeline, so it cannot turn `all` RED and cannot block any launch TODAY. It becomes a launch-blocker the moment the Wf-A169 wiring patch lands.",
    "pre_flight_check.py:11118 (gate def) - `grep` finds 1 occurrence = the def only; control gate gate_p3_protocol_alignment_yaml_no_inline_cli has 2 (def at :11497 + registration at :14960). Live build_gates() union over PIPELINES ('chemprop_v2_v3','cfm','all','frontier','launch') contains NEITHER p3_protocol_alignment_baseline_paper_refs_bib NOR p3_protocol_alignment_user_ok_gate. paper/refs.bib: 100 keys, missing all of krasnov2026metalcytotoxdb / rusanov2025platinai / ji2022drugood (required_keys at :11124-11127). Staged wiring patch: molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff (NOT applied).",
    "SURVIVED AS A MEASUREMENT, BUT THE WIRING REFUTATION'S 'blocks every launch' VERDICT IS CORRECTED (synthesis's own adversarial re-check)",
    "Read-only: nothing to change today. For the follow-on wave, the bibliography fix and the gate wiring MUST land in ONE write, bibliography FIRST - landing the wiring first turns `all` RED and exits 1 through scripts/launch_wf124_p3_g5.sh:288-294 (`|| { echo \"FAIL: pre_flight_check\"; exit 1; }`), which uses --pipeline launch (a superset of `all`). The 3 keys must be added SURGICALLY to paper/refs.bib - the STAGED bib is a 13-entry overlay vs 100 live entries, so copying it over would delete 87 entries.",
    True,
    ["paper/refs.bib", "molmetal/scripts/pre_flight_check.py"],
    "BLOCKED: paper/refs.bib is under AGENTS.md §5.3 / TODO-84 - zero writes without the user's explicit OK.",
)

finding(
    "B2", "B-gate-truth", 8,
    "88 TRACKED CONFIGS declare a required pre-flight gate that no pipeline owns: 82 configs name `p1_pytorch_cuda_available` (a gate name with ZERO occurrences in any .py and ZERO occurrences in pre_flight_check.py's git history - `git log -S` returns empty), and 6 configs name the 2 dead protocol gates. `pre_flight_check.py --gates` resolves against the UNION of pipelines and HARD-ERRORS (return 1) on unknown names.",
    "pre_flight_check.py:17444-17466 (--gates resolution + hard error, return 1 at :17466). Synthesis census over all 276 tracked YAML: 88 carry directives.pre_flight_gates_required and ALL 88 name unregistered gates. p1_pytorch_cuda_available: `git grep -l` -> 83 tracked files (82 configs + 1 TODO doc), 0 of them .py; `git ls-files '*.py' | xargs grep -l` -> 0. The 6 two-gate configs = 5 in configs/training/ + submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml. LATENT today: no tracked .sh/.py launcher passes --gates (only pre_flight_check.py itself and test_wf380a4_vina_params.py, which asserts only a list LENGTH of 4, never a gate name).",
    "SURVIVED - NEW FINDING BY SYNTHESIS (not covered by any of the 5 branches; the catalog records only the 6-config half)",
    "Read-only config repair: drop the stale `p1_pytorch_cuda_available` key from the 82 configs' directives.pre_flight_gates_required, and drop the 2 dead protocol gates from the 6 configs (or register real gates). configs/** is NOT on this wave's protected list, so this is actionable without further authorization.",
    False,
    ["configs/cfm_5metal_default.yaml",
     "configs/training/nci60_9cell_5seed_50ep_v1.yaml",
     "configs/training/nci60_9cell_coverage_5seed_50ep_v1.yaml",
     "configs/training/nci60_9cell_probe_seed0_cap512_d4.yaml",
     "configs/training/nci60_9cell_probe_seed12_cap512_d4.yaml",
     "configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml",
     "submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml"],
    "82 of the 88 files are the configs/cfm_*.yaml family - enumerated in full in synthesis.json -> files_to_create_edit.config_stale_gate_key.",
)

finding(
    "B3", "B-gate-truth", 9,
    "24 of the 32 Wf-432-A1 gates are registered in NO pipeline and have NEVER run; the 8 that are registered are exactly the 5 L3 + 3 L4 gates. Since 19:17:33 the repo's own test now ASSERTS the 24 stay absent (a deferral tripwire), so this is a RECORDED DELIBERATE deferral (catalog §6.267), not an unnoticed gap.",
    "molmetal/tests/test_wf433a1_gate_v2_wire.py:54-92 (EXPECTED_NEW_GATES = 32); synthesis live build_gates() union -> 24 MISSING, 8 present = p3_mock_to_real_m3/m15/m14/m17/m21 + p3_honest_neg_closure_m2_r2/m6_kappa/m1_re_fold; tripwire at :242 test_wf432a1_live_gates_registered_and_deferred_ones_absent + the deferral record at :104-115 and :292",
    "SURVIVED (wiring CLAIM 1; the time-index correction comes from the pins branch)",
    "Read-only: documentation only. Any wiring must FIRST update the deferral record (test comment block :104-115 + the catalog §6.267 entry) - otherwise the tripwire fires. Do not wire on this wave's authority.",
    False,
    ["molmetal/scripts/pre_flight_check.py", "molmetal/tests/test_wf433a1_gate_v2_wire.py"],
    "molmetal/scripts/pre_flight_check.py is READ-ONLY under this wave's hard rule 3.",
)

finding(
    "B4", "B-gate-truth", 10,
    "The 9 L5 protocol-validation gates compare NOTHING: `verdict_field` is copied into the hook payload and never compared against a measured value; and their emitter writes a hook JSON on EVERY invocation (before the verdict), so even a WARNing gate writes molmetal/reports/protocol_validation/<baseline>_hook.json.",
    "molmetal/scripts/pre_flight_check.py:12322 (mkdir(parents=True) + atomic_write_file) inside _hook_passthrough_payload, called by all 9 L5 gates; live probe verdict vector 19 PASS / 4 WARN / 1 FAIL / 0 RAISED over 24 gates (molmetal/reports/wfa175_dead_gate_wiring/refutation/probe24.json); the 3 hook JSONs the probe wrote are at molmetal/reports/wfa175_dead_gate_wiring/refutation/_pv_hook_dir/",
    "SURVIVED (wiring CLAIM 4)",
    "BLOCKED for a real fix: the emitter lives inside molmetal/scripts/pre_flight_check.py, which is READ-ONLY under this wave's hard rule 3. Read-only interim: document that these gates PASS on file-existence alone and their hook JSON is not evidence of a comparison.",
    True,
    ["molmetal/scripts/pre_flight_check.py"],
    "A gate that cannot fail is worse than no gate: it manufactures PASS.",
)

finding(
    "B5", "B-gate-truth", 11,
    "The DrugOOD protocol-alignment reference is dangling and stale: PROTOCOL_ALIGNMENT_MATRIX.md has §0-§4 and NO §1.4; the canonical arg set points at the non-existent TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md; the plan doc's own L5 spec text cites \"§1.4\" twice; and the matrix's m17 row still says \"NOT ALIGN - fallback cohort\" while the metric JSON records the OFFICIAL DrugOOD IC50 splits.",
    "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (headers §0-§4, no §1.4) vs TODO/03_plans/PRE_TRAINING_GATE_v2.md:287 and :311 (spec text cites §1.4) and molmetal/tests/test_wf433a1_gate_v2_wire.py:235 (arg set points at the non-existent TODO/03_plans/ path); metrics/by_metric/drugood_ood_auc.json -> data_source = drugood_official_lbap_refined_ic50, per_axis_auc_delta_vs_imold = +0.009347711256380342 / +0.001844904852242868 / -0.0028530871876835118, assay OOD AUC 0.7304477112563803 (vs the doc's stale 0.7479)",
    "SURVIVED (wiring CLAIM 7)",
    "Read-only edits to TODO/03_plans/PRE_TRAINING_GATE_v2.md and TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md - but TODO/** is READ-ONLY under this wave's hard rule 3, so this needs a follow-on wave's write scope.",
    True,
    ["TODO/03_plans/PRE_TRAINING_GATE_v2.md", "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md"],
    "BLOCKED: TODO/** is on this wave's protected list.",
)

finding(
    "B6", "B-gate-truth", 12,
    "Exactly 3 of the 582 metrics/by_metric/*.json files carry the band AND have no registry row (579 registered + 3 unregistered = 582); one of the 3 is STALE against its generator.",
    "metrics/by_metric/wfa155_ligand_connected_rate.json (band lines 224/228/248), metrics/by_metric/wfa155_metal_coordination_sphere.json (band lines 22/26), metrics/by_metric/wfa155_posebusters_validity.json (band line 608) - synthesis census: 582 files, 4 lack a registry row by metric-id, of which exactly 3 carry the band (the 4th, cfm_wfa75a_guidance_sweep.json, carries none); band values 92.7 / 94.0 / 92.98 / 94.33 at wfa155_posebusters_validity.json:554,555,558,561",
    "SURVIVED (band CLAIM 4, IMPRECISE only on the label - the counts are exact)",
    "BLOCKED: metrics/** is READ-ONLY under this wave's hard rule 3. Follow-on wave: add the 3 registry rows in registry/metrics.yaml (and refresh the STALE generator relation), so `registry.py check` covers them.",
    True,
    ["registry/metrics.yaml", "metrics/by_metric/wfa155_ligand_connected_rate.json",
     "metrics/by_metric/wfa155_metal_coordination_sphere.json",
     "metrics/by_metric/wfa155_posebusters_validity.json"],
    "BLOCKED: metrics/** and registry/ are outside this wave's write scope.",
)

# ---------------------------------------------------------------- TIER C
finding(
    "C1", "C-bookkeeping", 13,
    "The gate-count pin census - the thing that enforces 'no stale gate-count pins' - is UNTRACKED, so a clean checkout of HEAD has no census and no CI or pre-commit hook can see it; the repo-wide zero-stale invariant is unguarded (only a FILE-SCOPED zero-stale guard over the 4 Wf-A172-repaired files exists).",
    "git ls-files molmetal/scripts/wfa169_gate_count_pins.py molmetal/tests/test_wfa169_gate_count_pins.py -> EMPTY (both are `??` untracked); git grep -ln stale_pin -- . over TRACKED files -> 0 hits; .pre-commit-config.yaml and .github/workflows/ci.yml contain no pin-census gate. Live census re-run by synthesis: stale_pin_count = 0, stale_pin_files = [], counts_match = True.",
    "SURVIVED (pins CLAIM 4 IMPRECISE -> the precise version; plus the branch's supplementary finding #1)",
    "Needs git write authority (`git add` of the two files) - forbidden by this wave's hard rule 4 (no git mutations). Follow-on wave: track both files, and add a repo-wide `stale_pin_count == 0` assertion.",
    True,
    ["molmetal/scripts/wfa169_gate_count_pins.py", "molmetal/tests/test_wfa169_gate_count_pins.py"],
    "BLOCKED: no git mutations in this wave. Note the census test also depends on 2 files that exist on disk but not in HEAD - committing the test without them breaks a clean checkout.",
)

finding(
    "C2", "C-bookkeeping", 14,
    "The pin census scans report-directory .py files: scan_gate_count_pins() is called without the `exclude` parameter it declares, and only the PROSE scan excludes the census's own output dir - so a .py dropped in a report dir flips stale 0->6 while the census suite still reports 18 passed.",
    "molmetal/scripts/wfa169_gate_count_pins.py:853 (exclude param declared), :995 (pin_census calls scan_gate_count_pins with no exclude), :765 (only the PROSE scan excludes CENSUS_OUTPUT_REL), :724-737 (_iter_python_files has no exclusion). Live experiment by the pins refutation: before stale 0 / files_scanned_python 3054 -> with fixture stale 6 / 3055 -> census suite 18 passed -> after delete stale 0.",
    "SURVIVED (pins CLAIM 5)",
    "Read-only edit of molmetal/scripts/wfa169_gate_count_pins.py (untracked, agent-owned, NOT on the protected list): pass exclude=[CENSUS_OUTPUT_REL] at the :995 call site.",
    False,
    ["molmetal/scripts/wfa169_gate_count_pins.py"],
)

finding(
    "C3", "C-bookkeeping", 15,
    "The retired-pin accounting does not balance: the Wf-A172 docstring says \"moved ten stale count pins across three files ... and moved the four stale baseline assertions\", and the artifact records \"STALE | 15 | 4\", but the file's own table lists 5 stale pins in test_wf433a1_gate_v2_wire.py - at least one retired pin is unaccounted for. Exact-status coverage also regressed 9 files/24 pins -> 6 files/8 pins.",
    "molmetal/tests/test_wfa169_gate_count_pins.py:289-291 (docstring) vs molmetal/reports/wfa169_dead_gate_repair/pin_census.md:22 (\"STALE | 15 | 4\") whose rows 3-7 are 5 stale pins in test_wf433a1_gate_v2_wire.py (lines 108,120,139,140,365); pre-repair pin_census.json = executable 24 pins / 9 files vs live 8 pins / 6 files",
    "SURVIVED (pins CLAIM 2); the \"KNOWN_PIN_FILES was edited down\" half of pins CLAIM 7 is REFUTED (length stayed 6 - a 3-out/3-in swap)",
    "Read-only: correct the pin_census.md accounting note so the 15/4 figure is reconciled against the 5-pins-in-one-file table. Documentation-only.",
    False,
    ["molmetal/reports/wfa169_dead_gate_repair/pin_census.md"],
)

REFUTED = [
    {"branch": "BIBLIOGRAPHY", "claim": "CLAIM 6 - the AttentiveFP version-of-record issue is 12",
     "why": "REFUTED on the issue number. Four independent primary records say 16 (Crossref `issue` + `journal-issue.issue`; PubMed esummary `issue`; PubMed MEDLINE `VI - 63` / `IP - 16`; OpenAlex `biblio.issue`). paper/refs.bib:141 ALREADY carries `number = {16}`. The branch's STAGED file (molmetal/reports/wfa175_refs_bib_audit/refs_corrected_STAGED.bib:399 and :424) writes `number = {12}` and its note at :405 cites 'Crossref issue 12; PubMed 63(12):8749-8760' - landing it would INTRODUCE a new error into a currently-correct field. `audit.json` stores no `crossref_issue` field at all, so 12 was asserted, never measured. DO NOT APPLY the number change.",
     "surviving_part": "CLAIM 6's title correction and both author-name corrections (Zhenjun->Zhaojun, Xiang->Xiaomin) SURVIVE and should be applied."},
    {"branch": "FRONTIER", "claim": "CLAIM 2b - our RAW 8.51 % is below EVERY frontier RAW connectivity measured",
     "why": "IMPRECISE, not false in direction. The branch's own §1 row 3 measures TABASCO connectivity = 0.00 (RAW, at 10-20 steps), which is BELOW our 8.51 %; and DiffSBDD's 63.37 % is P(connected|valid) (main.tex:929; validity 96.32 % => ~61.0 % unconditional) against our unconditional P(connected|parsed). The honest-negative direction survives; the universal quantifier does not."},
    {"branch": "FRONTIER", "claim": "CLAIM 3 - band is frontier-sourced, never in paper/, never at HEAD",
     "why": "IMPRECISE. Both negatives CONFIRMED and strengthened (0 hits over all 9,551 HEAD blobs and all 134 paper/ files). But the report's own census records 123 paper/ files, not 134; '0 of 10,060 committed blobs' mislabels the denominator (HEAD = 9,551); '94.0 is 0 of 134 in paper/' is false (1 SVG coordinate hit); the census excluded gitignored paths repo-wide; and the mis-attribution is NOT single (see finding A3)."},
    {"branch": "FRONTIER", "claim": "CLAIM 5 - EF-TALFM's prose sentence contradicts its own table",
     "why": "IMPRECISE. The conclusion (92.98 % must be RAW; no RAW/POST pair exists in the paper, so the ratio is uncomputable) is right, but the premise is wrong: src/appendix.tex:60 states 'Connectivity: % of molecules with exactly one connected component, i.e., one fragment.' - :183 is a loose caption, not a contradiction. Do NOT file EF-TALFM as a frontier metric-definition contradiction."},
    {"branch": "BAND", "claim": "CLAIM 2 - six counts and the '79 %' canonical subset",
     "why": "IMPRECISE. All six counts are exact from sites.json, but '79 %' belongs to the canonical 59/75 subset (today 66/82 = 80.5 %), and the report's §2.2 prose is inverted - see finding A5. Undisclosed scope cap: 21 gitignored files carry the band on 81 lines - see finding A4."},
    {"branch": "BAND", "claim": "CLAIM 3 - the band's endpoints and Open-Babel refutation",
     "why": "IMPRECISE on citation, not on substance. Both quotes and the Open-Babel refutation verified verbatim; but the two band endpoints' post_processed:false sit at frontier_matrix.json:58 (92.7 %) and :80 (94 %) - :67 is the YuelBond F1 candidate, a different quantity. 'Every other site calls it external' overstates: 5 further sites attach 'the repo's' to the band."},
    {"branch": "BAND", "claim": "CLAIM 4 - the 3 'published metric artifacts' are published",
     "why": "IMPRECISE on the LABEL only. All 6 surfaces, the 4 dash blocks and the 20817/20819 geometry verify line-exact. The three artifacts are exactly the 3 of 582 metrics/by_metric/*.json with NO registry row - unregistered, one STALE vs its generator. See finding B6."},
    {"branch": "BAND", "claim": "CLAIM 5 - paper/ is band-free except 99.9",
     "why": "IMPRECISE. The headline SURVIVES: all 123 tracked paper/** files re-swept, 0 band hits under 3 notations; every 92.7 is the YuelBond F1 string or SVG coordinate noise. But the value sweep is not '0 except 99.9' (99.9->12, 8.51->19, 92.7->18, 92.0->10, 98.8->7, 87.9->6, 91.9->5, 91.2->3 hits, all noise) and the SVG split '4/5/3/4' does not reproduce (total 16 reproduces; the 9\\d\\.\\d split is 13/3/0/0)."},
    {"branch": "WIRING", "claim": "CLAIM 2 - 'NOT ONE of the 24 evaluates its own spec threshold'",
     "why": "IMPRECISE: it is 22 of 24. drugood_3_splits does evaluate a 3-split COUNT - against the wrong file, with a 'batch_size'-driven false positive on 'size' (scaffold lives under per_axis_auc / gate_pass, not in any top-level key) - and dataset_match evaluates a 6-path count where the spec says 7."},
    {"branch": "WIRING", "claim": "CLAIM 3 - the 11/7/5/1 and 8/7/2/1/6 histograms are MEASURED",
     "why": "IMPRECISE: both histograms are exactly reproducible and each sums to 24, but 3 of the LABELS are declared interpretations, not measurements (metric_match=UNSPEC; honest_limitation=7/7 is a loose reading where the repo's stricter rule scores 5/7; dataset_match=MET is 6/6 where the spec says 7). A case-sensitive \\bm15\\b scan returns 0 - the entry exists as metric_id = m15_rmsd in §6.23, so the 7/7 label is correct."},
    {"branch": "WIRING", "claim": "CLAIM 5 - 'the other 6 numerically MET'",
     "why": "IMPRECISE: arithmetic error. The 9 records are 3 SHORTFALL + 1 MISSING_SOURCE + 5 MEETS (4 named + 6 > 9 is impossible; the 6th residual is the MISSING_SOURCE one just named). The 3 drugood MEETS also come from an ALTERNATE source: the spec's primary molmetal/reports/wf265_collect/all_results.json has primary_status = MISSING_KEY, so the MEETS come from metrics/by_metric/drugood_ood_auc.json."},
    {"branch": "WIRING", "claim": "CLAIM 8 - 'read-only repair' / 'named flags occur 0x' / seed_cohort blocker",
     "why": "IMPRECISE on three counts: section_6_auto_emit needs a NEW FILE under molmetal/scripts/ (and once it exists the gate PASSes on its existence alone - the vacuous-PASS shape); '5-fold' occurs 5x in train_metal_cytotox.py (prose) and 'per_cell' occurs in both trainers as an output key, so only the CLI-flag reading is right; and seed_cohort is closable by correcting the SPEC TEXT alone (the trainers already emit per-seed JSON) with no trainer change and no §5.7 smoke. Also SUPERSEDED at 19:17:33: wiring only 4 no longer leaves the repo's own test RED - the assertion was replaced by the deferral tripwire (see finding B3)."},
    {"branch": "WIRING", "claim": "CLAIM 6 - 'the refs_bib gate turns `all` RED and blocks every launch'",
     "why": "SURVIVED AS A MEASUREMENT, CORRECTED ON THE CONSEQUENCE by the synthesis's own re-check: in the WORKING TREE the gate is registered in NO pipeline (pre_flight_check.py:11118 is the ONLY occurrence of the function name - def only; the control gate has def + registration), so it cannot turn `all` RED and cannot block any launch today. It becomes a launch-blocker only when the Wf-A169 wiring patch lands - which is why the bibliography fix and the gate wiring must be ONE write, bibliography FIRST. See finding B1."},
    {"branch": "PINS", "claim": "CLAIM 4 - 'no test anywhere asserts stale_pin_count is 0'",
     "why": "IMPRECISE: a FILE-SCOPED zero-stale guard exists and holds - test_the_wfa172_repaired_files_hold_no_stale_pin (molmetal/tests/test_wfa169_gate_count_pins.py:286-305) asserts no stale pin in each of the 4 Wf-A172-repaired files. PRECISE: the REPO-WIDE zero-stale invariant is unguarded; the only zero-stale assertion is file-scoped. MEASURED consequence: a 6-stale-pin fixture in a report dir left the census suite at 18/18 PASS."},
    {"branch": "PINS", "claim": "CLAIM 7 - 'KNOWN_PIN_FILES was edited down'",
     "why": "REFUTED on that clause. Its length was 6 BEFORE the repair and is 6 AFTER - a 3-out / 3-in swap (pre-repair {wfa127, wfa128, wf433a1, wf350c1, wf323f, wf415a3} -> today's {preflight_ssot_guard_wiring, wf350a2, wf350c1, wf415a3, wfa127, wfa159}); the guard's form did not weaken (len(exact) >= len(KNOWN_PIN_FILES)). The FIRST clause survives: exact-status coverage regressed 9 files/24 pins -> 6 files/8 pins while file-level coverage of all 9 was preserved."},
    {"branch": "PINS", "claim": "CLAIM 8 - the Wf-A172 loop form is invisible to the census",
     "why": "IMPRECISE on the citation only. The substance holds (a `for pipeline in (...)` loop form yields 0 rows of any kind to the exact-pin vocabulary; the direct-SSOT form yields 1 'dynamic' row). CITATION ERROR: the loop is at molmetal/tests/test_wf433a1_gate_v2_wire.py:199 at measurement time and :341 after the 19:17:33 rewrite - NOT :180-186, which are the repair's explanatory comment."},
    {"branch": "PINS", "claim": "CLAIM 9 - 'the floor is VIOLATED at HEAD's 126'",
     "why": "IMPRECISE: mixes two trees. HEAD's own 385-line copy of the test file carries NO `>= 127` floor - it has `assert _pfc.EXPECTED_GATE_COUNTS[\"all\"] == 127` (:108) and `assert len(set(names)) == 127` (:120). PRECISE: the WT-only floor is satisfied in the WT tree (129 >= 127) but is incompatible with HEAD's 126-gate runtime, so the WT test file cannot be run against HEAD's pre-flight."},
]

CLEAN = [
    "paper/** - 0 hits of the 92.7-94 % band under 3 notations across all 123 tracked paper/ files; every 92.7 in paper/ is the YuelBond F1 string or SVG coordinate noise. (build_log.txt is tracked + ISO-8859 + 8 hits; PATH ugrep's -I skips it, so the repo's own scanner is required.)",
    "The Open-Babel '92.7-94 % band' attribution string - 0 hits over all 9,551 HEAD blobs and all 134 paper/ files on disk (the mis-attribution exists only in molmetal/scripts/wfa161_frontier_labelling.py:934 and in report prose).",
    "All 4 frontier e-print sha256 reproduced byte-exact on re-fetch; the cached extracted text is byte-identical to the re-fetched tarball; every cited quote verbatim at the cited line.",
    "The 87.9 % vs 89.4 % dispute - reproduced `89.4`=0 / `87.9`=2 in the e-print and `89.4`=0 / `87.9`=3 on the live abs page; the 87.9 sentence is verbatim. (Minor: the locator is src/results_unconditional.tex:35, not sections/...; the abs-page half is not archived - no bytes/sha in §0.)",
    "Live gate counts: all=129, launch=131, frontier=25, chemprop_v2_v3=18, cfm=8 - identical to EXPECTED_GATE_COUNTS, with NO scope cap (those 5 ARE the repo's own PIPELINES tuple).",
    "The 4 registered p3_protocol_alignment_yaml_* gates are correctly registered (control for the dead-gate family) - the dead set is specific, not a global registration failure.",
    "--gates resolves against the UNION of all pipelines (not just `all`) - correct by design (pre_flight_check.py:17444-17466).",
    "No tracked .sh/.py launcher passes --gates -> the config gate-list trap is LATENT, not a live RED.",
    "Live pin census re-run by synthesis: stale_pin_count = 0, stale_pin_files = [], counts_match = True.",
    "git grep -ln stale_pin -- . over TRACKED files -> 0 hits (no tracked consumer of the census).",
    "metrics/by_metric/*.json = 582 files; 579 registry rows -> exactly the 3 band-carrying files are the unregistered ones (the arithmetic closes).",
    "The Krasnov record is real and correctly quoted: the abstract says verbatim '26,500 IC50 values for 7050 metal complexes against 754 cell lines from 1921 articles' (no '60 cancer cell lines' anywhere); the real record is J. Med. Chem. 2026, 69(8):8838-8851, 10.1021/acs.jmedchem.5c02755.",
    "The ChemRxiv Rusanov preprint 10.26434/chemrxiv-2025-pp32k is real; its abstract contains 'MB-Finder' 0 times and PlatinAI is ONE novel bis-carbene platinum(II) complex.",
    "Both .tools/sci-paper DOIs return responseCode:100 (fabricated); the doubled AAAI DOI 10.1609/aaai.v37i.v37i7.25970 also returns 100, the real one being 10.1609/aaai.v37i7.25970 = AAAI 37(7):8023-8031.",
    "paper/refs.bib is tracked, has 100 @-entries and is not truncated; detect_truncation returned None on all three STAGED bib deliverables.",
    "The 3 STAGED bib deliverables preserve all 11/11 original entries verbatim, carry all 3 required keys, and have unique cite keys.",
    "The drugood 3-split / dataset_match gate-source arithmetic (25/125, 9/45, 3/12, 3/3 splits, 2/2, 1/1) and the 266,522 data rows per axis CSV are exact.",
]

FILES_TO_CREATE_EDIT = {
    "read_only_now_agent_owned": [
        {"path": "molmetal/reports/wfa175_frontier_comparability/report.md", "edit": "line 79 + §6.2: say '2 of 3 steps', disclose the UFF 0/12,668 failure, strike 'directly comparable' (finding A1); regenerate the §2.1 ratio column from tools/ratios.json (finding A6)."},
        {"path": "molmetal/reports/wfa175_band_provenance/report.md", "edit": "line 106: correct the inverted §2.2 ('6 of the 7 post sites are correct' -> '6 of the 7 are the blanket-REFUTED ones; only frontier_matrix.md:68 is correct') (finding A5)."},
        {"path": "molmetal/reports/wfa175_band_provenance/_census.py", "edit": "include the gitignored report population so the census is a coverage measurement (finding A4)."},
        {"path": "molmetal/scripts/wfa161_frontier_labelling.py", "edit": "line 934: replace the Open-Babel mis-attribution string (finding A3)."},
        {"path": "molmetal/scripts/wfa169_gate_count_pins.py", "edit": "line 995: pass exclude=[CENSUS_OUTPUT_REL] to scan_gate_count_pins (finding C2)."},
        {"path": "molmetal/reports/wfa169_dead_gate_repair/pin_census.md", "edit": "line 22: reconcile the STALE 15 / 4 accounting against the 5-pins-in-one-file table (finding C3)."},
        {"path": "configs/cfm_5metal_default.yaml + 81 sibling configs/cfm_*.yaml", "edit": "drop the stale `p1_pytorch_cuda_available` key from directives.pre_flight_gates_required (finding B2); full list in files_to_create_edit.config_stale_gate_key."},
        {"path": "configs/training/nci60_9cell_5seed_50ep_v1.yaml", "edit": "drop `p3_protocol_alignment_user_ok_gate` and `p3_protocol_alignment_baseline_paper_refs_bib` from directives.pre_flight_gates_required (finding B2)."},
        {"path": "configs/training/nci60_9cell_coverage_5seed_50ep_v1.yaml", "edit": "same as above (finding B2)."},
        {"path": "configs/training/nci60_9cell_probe_seed0_cap512_d4.yaml", "edit": "same as above (finding B2)."},
        {"path": "configs/training/nci60_9cell_probe_seed12_cap512_d4.yaml", "edit": "same as above (finding B2)."},
        {"path": "configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml", "edit": "same as above (finding B2)."},
        {"path": "submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml", "edit": "same as above (finding B2)."},
    ],
    "blocked_on_paper_ok_agmd_5_3": [
        {"path": "paper/refs.bib", "edit": "SURGICALLY add the 3 required keys (krasnov2026metalcytotoxdb, rusanov2025platinai, ji2022drugood) + apply the surviving CLAIM-6 title/author corrections. DO NOT copy the STAGED overlay (13 live entries vs 100) and DO NOT apply its number={12} (finding B1)."},
    ],
    "blocked_on_readonly_scope_this_wave": [
        {"path": "molmetal/scripts/pre_flight_check.py", "edit": "register or retire the 24 dead gates (B3); make the 9 L5 hooks compare a measured value and stop writing on WARN (B4). READ-ONLY under hard rule 3."},
        {"path": "TODO/03_plans/PRE_TRAINING_GATE_v2.md", "edit": "fix the dangling 'PROTOCOL_ALIGNMENT_MATRIX.md §1.4' spec cross-references (:287, :311) (B5). READ-ONLY under hard rule 3."},
        {"path": "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md", "edit": "refresh the stale m17 'NOT ALIGN - fallback cohort' row against metrics/by_metric/drugood_ood_auc.json (B5). READ-ONLY under hard rule 3."},
        {"path": "metrics/by_metric/wfa155_ligand_connected_rate.json + wfa155_metal_coordination_sphere.json + wfa155_posebusters_validity.json", "edit": "registry rows + STALE generator relation (B6). READ-ONLY under hard rule 3."},
        {"path": "docs/HONEST_NEGATIVE_CATALOG.md", "edit": "record the band metric-definition break + the band-census scope cap (A2, A4). READ-ONLY under hard rule 3."},
        {"path": "AGENTS.md", "edit": "update the §6.267 dead-gate entry: 24 of 32 unregistered (now a recorded deferral), and the 82-config p1_pytorch_cuda_available population (B2, B3). READ-ONLY under hard rule 3."},
    ],
    "blocked_on_git_write_authority": [
        {"path": "molmetal/scripts/wfa169_gate_count_pins.py", "edit": "git add (track it) - hard rule 4 forbids git mutations (C1)."},
        {"path": "molmetal/tests/test_wfa169_gate_count_pins.py", "edit": "git add (track it) + add a repo-wide stale_pin_count == 0 assertion; note it depends on 2 files that exist on disk but not in HEAD (C1)."},
    ],
    "new_files_this_wave": [
        {"path": "molmetal/reports/wfa175_synthesis/synthesis.json", "edit": "this wave's deliverable."},
        {"path": "molmetal/reports/wfa175_synthesis/synthesis.md", "edit": "this wave's deliverable."},
    ],
}

# enumerate the 82 configs carrying the stale p1_pytorch_cuda_available key
import subprocess  # noqa: E402
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=str(REPO)).stdout.split()
stale_gate_key = [f for f in tracked if f.endswith((".yaml", ".yml")) and "p1_pytorch_cuda_available" in
                   (REPO / f).read_text(encoding="utf-8", errors="replace")]

payload = {
    "branch": "SYNTHESIS",
    "wave": "Wf-A175",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "cpu_only": True,
    "gpu_used": 0,
    "python": str(REPO / ".venv/bin/python"),
    "writes_confined_to": "molmetal/reports/wfa175_synthesis/",
    "read_only_respected": ["paper/**", "docs/HONEST_NEGATIVE_CATALOG.md", "AGENTS.md", "TODO/**",
                             "README.md", "molmetal/scripts/pre_flight_check.py", "metrics/**"],
    "no_git_mutations": True,
    "source_branches": {
        "BIBLIOGRAPHY": "molmetal/reports/wfa175_refs_bib_audit/refutation/refutation.json (REFUTED 1 / SURVIVED 16 / IMPRECISE 0)",
        "FRONTIER": "molmetal/reports/wfa175_frontier_comparability/refutation/refutation.json (REFUTED 0 / SURVIVED 3 / IMPRECISE 3)",
        "BAND": "molmetal/reports/wfa175_band_provenance/refutation/refutation.json (REFUTED 0 / SURVIVED 2 / IMPRECISE 4)",
        "WIRING": "molmetal/reports/wfa175_dead_gate_wiring/refutation/refutation.json (REFUTED 0 / SURVIVED 4 / IMPRECISE 4)",
        "PINS": "molmetal/reports/wfa175_pin_repair_refutation/refutation/refutation.json (REFUTED 0 / SURVIVED 5 / IMPRECISE 4)",
    },
    "tally": {
        "branch_claims_refuted": 1,
        "branch_claims_survived": 30,
        "branch_claims_imprecise": 15,
        "synthesis_findings": len(F),
        "synthesis_tier_A": sum(1 for f in F if f["tier"].startswith("A")),
        "synthesis_tier_B": sum(1 for f in F if f["tier"].startswith("B")),
        "synthesis_tier_C": sum(1 for f in F if f["tier"].startswith("C")),
        "blocked_findings": sum(1 for f in F if f["blocked"]),
    },
    "findings": F,
    "refuted_and_imprecise": REFUTED,
    "clean_negative_results": CLEAN,
    "files_to_create_edit": FILES_TO_CREATE_EDIT,
    "config_stale_gate_key": stale_gate_key,
}

js = json.dumps(payload, indent=2, ensure_ascii=False)
atomic_write_file(OUT / "synthesis.json", js)
print("synthesis.json bytes:", len(js.encode()))
print("stale_gate_key configs:", len(stale_gate_key))
print("truncation:", detect_truncation(js))
