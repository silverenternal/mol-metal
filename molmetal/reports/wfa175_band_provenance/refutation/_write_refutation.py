"""Write the Wf-A175 BAND-branch refutation deliverables (atomic, AGENTS.md 5.6).

Writes ONLY inside molmetal/reports/wfa175_band_provenance/refutation/.

--------------------------------------------------------------------------
Wf-A190 (2026-11-01) -- LOCATOR DEMOTION for the LINE-NUMBER LOCATOR CLASS
==========================================================================
The record `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`
was 1,136 lines on 2026-11-01 and KEEPS GROWING: it was regenerated at least
three times on that day alone (Wf-A177, Wf-A180, Wf-A182 -- Wf-A182 alone
inserted 67 lines, 1069 -> 1136, and moved 29 leaves).  A LINE NUMBER into
that record is therefore a PERISHABLE locator; a JSON PATH is ROBUST and
survives any number of regenerations.

DURABLE RULE (the repair rule this producer now emits):
  * the LIVE locator is a JSON PATH;
  * a line number may only appear inside a DATED HISTORICAL record that says,
    in words, that it was measured on a given date and that it is STALE BY
    CONSTRUCTION (the record keeps growing);
  * never delete a line number -- demote it.

THE SUPERSEDED RENDERINGS, PRESERVED VERBATIM (AGENTS.md 5.5 rule #6).  Every
line number below was MEASURED against the 1,136-line record on 2026-11-01 and
is STALE BY CONSTRUCTION -- it is kept as the historical record of what the
citation CLAIMED, not as a locator:

    "frontier_matrix.json:58,67"
    "frontier_matrix.json:58,67,80"
    "frontier_matrix.json:58/:67/:80"
    "frontier_matrix.json:52-88"
    "frontier_matrix.json:58" / ":80" / ":67"
    "frontier_matrix.md:68"
    "frontier_matrix.md:353"

WHAT THOSE LINES ACTUALLY CONTAINED on 2026-11-01 (measured on the record):
    :58   -> band_provenance.competing_reading_and_what_this_does_not_settle
             .cross_references[1]  (the string naming
             sources['HLTF'].cells['METRIC'].note)
    :67   -> band_provenance.competing_reading_and_what_this_does_not_settle
             .readings[0].current_line_numbers_on_disk
    :80   -> band_provenance.competing_reading_and_what_this_does_not_settle
             .readings[1].current_line_numbers_on_disk
    :89   -> band_provenance.competing_reading_and_what_this_does_not_settle
             .readings[1].note  ("92.98 / 94.33 would round to '93.0-94.3' ...")
             -- the LIVE band_provenance.headline_finding is at :144
    :52   -> band_provenance.competing_reading_and_what_this_does_not_settle
             (the START of the cited :52-88 range: right)
    :88   -> band_provenance.competing_reading_and_what_this_does_not_settle
             .readings[1].makes_the_lower_endpoint_a_connected_component_count
             (the END of the cited :52-88 range: wrong)
    :145  -> band_provenance.headline_finding_superseded_2026_11_01
             (the ONE claim that was right, and only by coincidence)
    :362  -> "bytes": null
    :763  -> "label": "Validity (QM9, GGFlow)"   :764 -> a GGFlow locator string
             (the LIVE sources.HLTF.cells.METRIC.note_superseded_2026_11_01 is
             at :830)
    :779  -> a GGFlow quote row (the LIVE sources.HLTF.citation is at :844)
    frontier_matrix.md:68  -> renders sources.EF-TALFM.extra_observations[0]
    frontier_matrix.md:353 -> renders band_provenance.band_as_quoted_in_the_repo

THE DURABLE REPLACEMENT PATHS (each verified to EXIST in the record on
2026-11-01; note there is NO `candidate_c` -- the record's third candidate is
`endpoint_candidates[0].candidate_b`):
    92.7 % endpoint CANDIDATE (GeoLDM Val&Uniq, post_processed:false)
        band_provenance.endpoint_candidates[0].candidate_a
    YuelBond per-bond-type classification F1 (post_processed:false) -- a
    DIFFERENT quantity
        band_provenance.endpoint_candidates[0].candidate_b
    94 % endpoint CANDIDATE (HLTF PB-Valid, post_processed:false)
        band_provenance.endpoint_candidates[1].candidate_a
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(ROOT / "molmetal/scripts"))
from wf386a3_atomic_write import atomic_write_file, detect_truncation  # noqa: E402

# Wf-A190 (2026-11-01): the output directory may be redirected into a TEMP dir
# so a re-run can be DIFFED against the published artefact (producer parity).
# Unset -> the published location, unchanged.
OUT = Path(
    os.environ.get("WFA175_REFUTATION_OUT")
    or (ROOT / "molmetal/reports/wfa175_band_provenance/refutation")
)

# Wf-A190 (2026-11-01) -- durable locators (JSON PATHS) into the record.
REC = "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json"
PATH_92 = REC + " -> band_provenance.endpoint_candidates[0].candidate_a"
PATH_YB = REC + " -> band_provenance.endpoint_candidates[0].candidate_b"
PATH_94 = REC + " -> band_provenance.endpoint_candidates[1].candidate_a"
PATH_BP = REC + " -> band_provenance"
PATH_BAQ = REC + " -> band_provenance.band_as_quoted_in_the_repo"
PATH_EO0 = REC + " -> sources.EF-TALFM.extra_observations[0]"

# ---------------------------------------------------------------------------
# Wf-A190 (2026-11-02) -- durable locators into the OTHER growing artefacts.
#
# The record is not the only growing artefact this refutation cites:
#   * `docs/HONEST_NEGATIVE_CATALOG.md` is APPEND-ONLY and grows every wave
#     (21,688 lines on 2026-11-02) -> a SECTION ANCHOR is the durable form;
#   * the three `metrics/by_metric/wfa155_*.json` artefacts are REGENERATED by
#     their producers (re-written 2026-11-01 21:13 / 21:58) -> a JSON PATH is
#     the durable form.
# Every line number demoted below is preserved VERBATIM in
# `locator_demotion_2026_11_02_stale_by_construction` (AGENTS.md 5.5 rule #6).
# ---------------------------------------------------------------------------
CATALOGUE = "docs/HONEST_NEGATIVE_CATALOG.md"
ANCHOR_62668_BAND_CENSUS = CATALOGUE + " -> section 6.266.8 -> the band-census table"
ANCHOR_6265_FOUR_BREAKS = (
    CATALOGUE
    + " -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR "
    "BAND -- FOUR INDEPENDENT BREAKS'"
)
ANCHOR_DENOMINATOR_ROW = ANCHOR_6265_FOUR_BREAKS + " -> the DENOMINATOR row"
ANCHOR_POST_PROCESSING_ROW = ANCHOR_6265_FOUR_BREAKS + " -> the POST-PROCESSING row"
ANCHOR_6252_BAND = (
    CATALOGUE + " -> section 6.252 -> subsection 'The published band, and the denominator trap'"
)
ANCHOR_6252_MAY_MAY_NOT = (
    CATALOGUE
    + " -> section 6.252 -> subsection 'The sentence we may say, and the sentences we may not'"
)
LIGAND = "metrics/by_metric/wfa155_ligand_connected_rate.json"
METAL = "metrics/by_metric/wfa155_metal_coordination_sphere.json"
POSEB = "metrics/by_metric/wfa155_posebusters_validity.json"
P_LIGAND_DENOM = LIGAND + " -> comparability.breaks[0].frontier"
P_LIGAND_POST = LIGAND + " -> comparability.breaks[2].frontier"
P_METAL_SUMMARY = METAL + " -> comparability_statement.summary"
P_METAL_POST = METAL + " -> comparability_statement.break_3_post_processing"
P_POSEB_POST = POSEB + " -> comparability_vs_frontier_band.breaks[2].frontier"
P_POSEB_STMT = POSEB + " -> comparability_vs_frontier_band.statement"
P_POSEB_BAND = POSEB + " -> comparability_vs_frontier_band.frontier_band.band_connected_pct"
P_POSEB_EF_TALFM = (
    POSEB + " -> comparability_vs_frontier_band.frontier_band.ef_talfm_arxiv_2609_08333"
)
# Wf-A190 (2026-11-02) -- SOURCE-FILE citations whose cited line MEASURED-DRIFTED.
# A source file is not regenerated, so a `file:line` into it is not the LOCATOR
# CLASS -- but a live claim pointing at a line that no longer carries the quoted
# content IS a live defect.  Each was re-measured on 2026-11-02 and repaired in
# place; the superseded line numbers survive VERBATIM in the dated block below.
A161_PY = "molmetal/scripts/wfa161_frontier_labelling.py"
A161_MD = "molmetal/reports/wfa161_frontier_labelling/report.md"
A161_SENTENCE = (
    "the sentence 'Our 92.7-94 % band comes from a DIFFERENT perception branch "
    "(Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.'"
)
GEN_LIGAND = (
    "molmetal/scripts/wf_a155_ligand_connected_rate.py -> the emitter of "
    + LIGAND + " -> comparability.breaks[2].frontier"
)
GEN_POSEB = (
    "molmetal/scripts/wf_a155_posebusters_validity.py -> the emitter of "
    + POSEB + " -> comparability_vs_frontier_band.breaks[2].frontier"
)
GEN_METAL = (
    "molmetal/scripts/wf_a155_metal_coordination_sphere.py -> the emitter of "
    + METAL + " -> comparability_statement.break_3_post_processing"
)

REFUTATION = {
    "branch": "BAND",
    "wave": "Wf-A175",
    "role": "ADVERSARIAL REFUTATION of the BAND branch's audit",
    "generated": "2026-11-01",
    "cpu_only": True,
    "writes_confined_to": "molmetal/reports/wfa175_band_provenance/refutation/",
    "method": {
        "population": "git ls-files --cached + git ls-files --others --exclude-standard (same as the BAND branch's _census.py); plus a SEPARATE pass over git ls-files --others --ignored --exclude-standard to test the census's blind spot",
        "census_regexes": {
            "canonical": "92\\.7\\s*[--]\\s*94 (catalog 6.266.8 method)",
            "band_v1": "92\\.7\\s*[---------~]\\s*94",
        },
        "helper": "molmetal/reports/wfa175_band_provenance/refutation/_recheck.py -> _recheck.json",
        "no_gpu": "every python run prefixed with CUDA_VISIBLE_DEVICES=\"\" and /home/hugo/codes/mol-metal/.venv/bin/python",
    },
    "tally": {"REFUTED": 0, "SURVIVED": 2, "IMPRECISE": 4},
    "claims": [
        {
            "id": 1,
            "verdict": "SURVIVED",
            "claim": "Census reproduces canonical 32 files / 74 matching lines; today's tree 34 files / 81 lines / 82 occurrences; the whole delta is the 2 post-census Wf-A171 reports (4+3 lines, mtime 12:37/12:40); census not stale, tree grew.",
            "what_i_checked": [
                "Re-ran the canonical regex 92\\.7\\s*[--]\\s*94 over today's tree with my own scanner (_recheck.py).",
                "Compared the per-file matching-LINE count of EVERY one of the 32 files in the catalog's 6.266.8 table against today: 0 differences (canonical_vs_today_diffs == []).",
                "Today's tree minus the 2 Wf-A171 files: 32 files / 74 matching lines -- exact reproduction.",
                "wfa171_band_publication_audit/report.md mtime 2026-11-01 12:37:07 (4 lines), wfa171_frontier_postproc/report.md mtime 12:40:33 (3 lines) -- matches the claim's 12:37 / 12:40.",
                "Total today (excluding the branch's own deliverables): 34 files / 81 lines / 82 occurrences -- matches.",
            ],
            "counter_evidence": [
                "UNIT SLIP in the claim as summarised: the catalog's '74 matches' are matching LINES; occurrences today are 82, i.e. the delta is 7 lines but 8 OCCURRENCES. The extra occurrence is molmetal/reports/wfa161_paper_unqualified_numbers/report.md:35, which carries the band twice on one line. The branch's own report.md L47-48 and L63-66 states this correctly, so the branch's report is precise and only the one-line summary is loose.",
                "SELF-REFERENCE: the census population is 'tracked + untracked-not-ignored', so the branch's OWN deliverables enter it after the fact. Re-running the branch's own _census.py now returns 36 files / 96 lines / 97 occurrences (its _build_sites.py = 7 lines, report.md = 8 lines). The 34/81/82 figure is true at 18:57 (census time) and false for the tree as it stands at 19:04+.",
            ],
            "evidence_paths": [
                ANCHOR_62668_BAND_CENSUS,
                "molmetal/reports/wfa175_band_provenance/_census_raw.json",
                "molmetal/reports/wfa175_band_provenance/refutation/_recheck.json",
            ],
        },
        {
            "id": 2,
            "verdict": "IMPRECISE",
            "claim": "Per-site classification of 82 sites: assertion quoted 79 / ours 2 / unattributed 1; raw_or_post unstated 66 / raw 9 / post 7; denominator_stated True 17 / False 65; severity none 58 / low 18 / high 6; 79% of sites never state RAW/POST in their block.",
            "what_i_checked": [
                "Recomputed all six counts from sites.json: assertion {quoted:79, ours:2, unattributed:1}; raw_or_post {unstated:66, post:7, raw:9}; denominator_stated {False:65, True:17}; severity {none:58, low:18, high:6}; n=82. ALL EXACT.",
                "The 2 'ours' sites are wfa161_frontier_labelling.py:934 and wfa161_frontier_labelling/report.md:139 (both contain 'Our 92.7-94 % band ...'). The 1 'unattributed' site is wfa163_narrative_truth_scan.py:534 (docstring 'True if the token at pos is one end of a range (\"92.7-94 %\")'). The 6 'high' sites are catalog:20817, ligand:228, metal_coord:26, posebusters:608 + the wfa161 pair.",
            ],
            "counter_evidence": [
                "THE 79% IS ATTACHED TO THE WRONG DENOMINATOR in the one-line summary: 79% = 59/75, the CANONICAL (pre-Wf-A171) subset (report.md L105). Today's 82-site table gives 66/82 = 80.5%. The summary reads as if 79% of the 82 sites are unstated.",
                "THE BRANCH'S OWN 2.2 PROSE IS INVERTED: report.md L105-108 says 'For 6 of the 7 `post` sites the statement is correct and per-cell (they name 94.33 % as post-processed)'. sites.json says the opposite: of the 7 post sites, 6 are the blanket-REFUTED ones (catalog:20817, ligand:228, metal_coord:26, posebusters:608, wfa161 py:934, wfa161 report.md:139) and only ONE (the record's `sources`.`EF-TALFM`.`extra_observations`[0]) is correct-and-per-cell. 6 + 6 > 7. A repairer reading 2.2 would leave 6 wrong sites in place.",
                "UNDISCLOSED SCOPE CAP: the census population excludes gitignored files. Measured: 411,240 ignored-untracked files exist; 21 of them OUTSIDE the branch's own directory carry the band on 81 raw lines, e.g. molmetal/reports/wfa171_band_publication_audit/band_classification.json (32 lines), wfa158_frontier_postproc_matrix/frontier_matrix.json (4), wfa164_coordination_sphere_narrative/default_pool_artifact.json (4), wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json (3). The report never mentions .gitignore or Wf-A171's OPEN-5, which names exactly this blind spot. 'Full table: sites.json (82 records)' is complete only for the non-ignored tree.",
            ],
            "evidence_paths": [
                "molmetal/reports/wfa175_band_provenance/sites.json",
                "molmetal/reports/wfa175_band_provenance/report.md -> section 2 -> subsections 2.1 to 2.4 (the six per-site counts)",
                "molmetal/reports/wfa171_band_publication_audit/report.md:185-186",
                ".gitignore:158",
            ],
        },
        {
            "id": 3,
            "verdict": "IMPRECISE",
            "claim": "The band is asserted as OURS with refuted provenance at " + A161_PY + " and " + A161_MD + " (" + A161_SENTENCE + " -- the two line numbers the branch used, :934 and :139, are DEMOTED into the dated block, see locator_demotion_2026_11_02_stale_by_construction); Open Babel refuted by Wf-A168; 'DIFFERENT post-processing' refuted by the record's `band_provenance`.`endpoint_candidates`[0].`candidate_a` and [1].`candidate_a` (both `post_processed: false`); not a new finding (Wf-A171 OPEN-1 medium), raised to high.",
            "what_i_checked": [
                "Both cited lines exist and carry the sentence verbatim ('Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.').",
                "Open Babel refutation: molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md 1.3 (L115-127) does pin 92.7 % = HLTF arXiv:2603.27113v1 Table 1(b) 'Val&Uniq (%)' GeoLDM cell and 94 % = same table 'PB-Valid (%)' HLTF cell; L110 records Open Babel as HLTF's all-atom branch.",
                "Prior art: wfa171_band_publication_audit/report.md:170-172 = 'OPEN-1 (medium)' for exactly this file:line and exactly the Open-Babel contradiction. 'Not a new finding' is correct.",
                "Severity: 'high' is a judgement; the site is a generator, and the defect does re-emit.",
            ],
            "counter_evidence": [
                "CITATION SLIP (restated in JSON PATH terms by Wf-A190 on 2026-11-01; the line numbers the branch used are DEMOTED into the dated historical record 'locator_demotion_2026_11_01_stale_by_construction', because the record keeps growing and every line number in it is stale by construction): the two BAND endpoints' post_processed:false are at `band_provenance`.`endpoint_candidates`[0].`candidate_a` (92.7 % = GeoLDM Val&Uniq, 'endpoint': '92.7 %') and `band_provenance`.`endpoint_candidates`[1].`candidate_a` (94 % = HLTF PB-Valid, 'endpoint': '94 %'). `band_provenance`.`endpoint_candidates`[0].`candidate_b` is the YUELBOND per-bond-type F1 -- a DIFFERENT quantity. The branch's 3.2 (which cited all three candidates) is right; its 3.1 -- and the one-line claim -- cited the 92.7 % endpoint's `candidate_a` AND its `candidate_b` for 'both band endpoints', which is wrong, because `candidate_b` is a different quantity. The conclusion is unaffected (all three are false).",
                "OWNERSHIP OVERSTATEMENT: 'opposite to the external stance of the 79 sites' is right for the 79 'quoted' sites, but 5 further sites attach 'the repo's' to the band (sites.json ownership_ambiguity=True: the record's `band_provenance`.`band_as_quoted_in_the_repo`, wfa168 synthesis.md:20 and :115, wfa171_frontier_postproc/report.md:314, wfa158_frontier_postproc_matrix.py:456). The branch's own 3.1 sentence 'Every other site in the repo calls the band external' is contradicted by its own 3.5.",
                "NO PRIMARY RE-FETCH (the branch says so, report.md 5.1): the 'both endpoints are RAW' claim is MEASURED-BY-CITATION from an artifact that records a live fetch, not re-measured in this wave. The one-line claim carries it as a fresh refutation.",
            ],
            "evidence_paths": [
                A161_PY + " (" + A161_SENTENCE + ")",
                A161_MD + " (" + A161_SENTENCE + ")",
                "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance",
                "molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:108-130,305-320",
                "molmetal/reports/wfa171_band_publication_audit/report.md:170-172",
            ],
        },
        {
            "id": 4,
            "verdict": "IMPRECISE",
            "claim": "New finding: the blanket 'the published band is measured on the published POST-PROCESSED stream' assertion survives on 4 dash sites' blocks, 3 published metric artifacts, the catalogue's 92.7-94 % four-breaks POST-PROCESSING row (" + ANCHOR_POST_PROCESSING_ROW + ") and 2 generators; refuted by the record's `band_provenance`.`endpoint_candidates`[0].`candidate_a`, [0].`candidate_b` and [1].`candidate_a` (all `post_processed: false`) and synthesis.md:310; invisible to the census because the characterisation and the band string are on different lines.",
            "what_i_checked": [
                "All 6 claim surfaces verified line-exact: catalog:20819 ('| **POST-PROCESSING** | the band is measured on the published **post-processed** stream | ...'), metrics/by_metric/wfa155_ligand_connected_rate.json:238, wfa155_metal_coordination_sphere.json:29, wfa155_posebusters_validity.json:596, generators molmetal/scripts/wf_a155_ligand_connected_rate.py:223 and molmetal/scripts/wf_a155_posebusters_validity.py:489.",
                "The 4 dash sites verified: catalog:20817 (band string in the DENOMINATOR row), ligand:228, metal_coord:26, posebusters:608 -- all classified raw_or_post='post', severity='high' in sites.json.",
                "The 'two lines apart' claim verified: band string at catalog L20817, characterisation at L20819 (L20818 between them); same pattern in the three metric artifacts (228 vs 238; 26 vs 29; 608 vs 596).",
                "Refuting evidence verified: the record's `band_provenance`.`endpoint_candidates`[0].`candidate_a` (92.7 % = GeoLDM Val&Uniq), [0].`candidate_b` (YuelBond per-bond-type F1) and [1].`candidate_a` (94 % = HLTF PB-Valid) all carry 'post_processed': false; wfa168 synthesis.md:310 'Never assert \"the band is POST-PROCESSED\" as a blanket claim' (and :317 'Never assert \"the band is RAW\" either').",
                "The third framing verified live in a generator: molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464 ('the same paper reports every model twice, once RAW and once POST-PROCESSED'), and the artifact on disk carries the OLD text -- i.e. artifact STALE vs generator.",
            ],
            "counter_evidence": [
                "'3 PUBLISHED metric artifacts' is the wrong label: measured against registry/metrics.yaml, those three .json files are EXACTLY the 3 of 582 files in metrics/by_metric/ that have NO registry row (579 registered, 582 on disk, 3 unregistered = these three). Under the repo's own SSOT (registry.py check / AGENTS.md 2) they are UNREGISTERED artefacts, not published metrics -- which lowers their live severity relative to the label.",
                "The branch's report 3.2 heading says '6 dash-census sites carry a `post` characterisation that is wrong' while 2.2 says 'For 6 of the 7 post sites the statement is correct and per-cell' -- the two sections disagree (see claim 2).",
            ],
            "evidence_paths": [
                ANCHOR_DENOMINATOR_ROW,
                ANCHOR_POST_PROCESSING_ROW,
                P_LIGAND_DENOM,
                P_LIGAND_POST,
                P_METAL_SUMMARY,
                P_METAL_POST,
                P_POSEB_POST,
                P_POSEB_STMT,
                GEN_LIGAND,
                GEN_POSEB,
                GEN_METAL,
                "registry/metrics.yaml (0 rows for wfa155_*)",
            ],
        },
        {
            "id": 5,
            "verdict": "IMPRECISE",
            "claim": "paper/'s zero is a real zero: 4 notations all 0 hits; the only 92.7 in paper/ is a different quantity (YuelBond per-bond-type F1, 1,000 molecules, named attribution) at 04_evaluation.tex:3179 and 06_limitations.tex:387; paper/build_log.txt (tracked, ISO-8859, 8 hits) is skipped entirely by PATH grep -I; paper/figures/*.svg 16 hits are all translate(x y) coordinate noise; therefore the paper-side repair proposed at wfa168 synthesis.md:407 is a no-op.",
            "what_i_checked": [
                "Independent sweep of all 123 tracked paper/ files (115 utf-8 + 1 non-utf-8 + 7 binary): dash-band regex 92.7 <dash> 94 = 0 hits; wide notation 9[234][.,]?[0-9]? <-> 9[234][.,]?[0-9]? = 0 hits; 92.7.{0,40}?94 = 0 hits.",
                "Every occurrence of '92.7' in tracked paper/ is either the YuelBond F1 string (04_evaluation.tex:3179 'F1=92.7\\\\%', 06_limitations.tex:387 'F1=92.7\\\\%)', plus 8 in build_log.txt) or SVG coordinate noise (8 transform= lines). 18 hits total, zero of them a band.",
                "paper/build_log.txt: tracked (git ls-files), file(1) = 'pdfTeX transcript, ISO-8859 text', 12,767,523 B, 8 occurrences of 'F1=92.7%'; PATH grep (ugrep 7.8.4) -c '92\\.7' returns 0 with rc=1 (skips it), while the UTF-8-safe scan finds 8 -- the claim about the grep blind spot is exact.",
                "The no-op inference holds: synthesis.md:407 (item 6) proposes 'replacing a bare 92.7-94 % with the four-break sentence'; there is no bare band in paper/, so there is nothing to replace.",
            ],
            "counter_evidence": [
                "THE VALUE SWEEP IS NOT '0 HITS EXCEPT 99.9'. Measured over the same tracked paper/ files: 99.9 -> 12 hits, 8.51 -> 19, 92.7 -> 18, 92.0 -> 10, 98.8 -> 7, 87.9 -> 6, 91.9 -> 5, 91.2 -> 3 (build_log.txt 'Overfull \\\\hbox (38.5177pt ...)' noise and SVG coordinates). The report's own 4.3 says '0 matches in any tracked paper/** file except 99.9 inside paper/build_log.txt (4 hits)' -- it names only 99.9, and its '4 hits' counts the 'try distortion; 99.9%' quotes only, not the 8 '99.99094pt' hbox hits.",
                "THE SVG PER-FILE SPLIT DOES NOT REPRODUCE. The branch reports '16 further 9x.x hits (4 in fig1, 5 in fig2, 3 in fig3, 4 in fig4)'. The total 16 reproduces only under the regex 9\\d\\.\\d(?!\\d), whose per-file split is 13 / 3 / 0 / 0 (fig1, fig2, fig3, fig4). Generic 9\\d\\.\\d+ gives 619 (192/173/116/138); the value-list sweep gives 43 (14/10/8/11). No pattern I tried yields 4/5/3/4. The 'all 16 are transform=\"translate(x y)\"' characterisation IS correct under the reproducing regex (13/3/0/0, every hit inside a transform attribute).",
                "SCOPE NOTE: 'paper/ is clean' must not be read as 'paper/ quotes no 94'. The paper does quote TargetDiff's PB rate as a bare 94 % / 0.94 (paper/sections/04_evaluation.tex:2159, 2246-2247, 2308; paper/sections/04_results_table2.tex:54) -- a different quantity from the band, so the band-clean conclusion is unaffected.",
            ],
            "evidence_paths": [
                "paper/sections/04_evaluation.tex:3179",
                "paper/sections/06_limitations.tex:387",
                "paper/build_log.txt",
                "paper/figures/fig1_mlc_architecture.svg",
                "molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:406-407",
                "molmetal/reports/wfa175_band_provenance/refutation/_recheck.json",
            ],
        },
        {
            "id": 6,
            "verdict": "SURVIVED",
            "claim": "4 bare numeric band values the dash regex misses, inside a published metric artifact: " + P_POSEB_BAND + " (92.7 / 94.0) and " + P_POSEB_EF_TALFM + " (connected_pct 92.98 / novel_unique_sanitized_pct 94.33); independently re-confirms Wf-A171 OPEN-3 / OPEN-4 (a band census must also sweep by VALUE).",
            "what_i_checked": [
                "Read the file at the cited lines: 553 '\"band_connected_pct\": [', 554 '92.7,', 555 '94.0', 557 '\"ef_talfm_arxiv_2609_08333\": {', 558 '\"connected_pct\": 92.98,', 561 '\"novel_unique_sanitized_pct\": 94.33,'. All four values and both field names are exactly as claimed.",
                "wfa171_band_publication_audit/report.md:179-184 confirms OPEN-3 (band_connected_pct = [92.7, 94.0] as bare numbers) and OPEN-4 ('any band census must sweep by VALUE (92.7, 92.98, 94.0, 94.33)').",
            ],
            "counter_evidence": [
                "Only qualifier, and it does not touch the claim: the file is one of the 3 metrics/by_metric/*.json with NO registry row, so 'published metric artifact' is loose (see claim 4). The four sites and the values are exact.",
            ],
            "evidence_paths": [
                P_POSEB_BAND,
                P_POSEB_EF_TALFM,
                "molmetal/reports/wfa171_band_publication_audit/report.md:179-184",
            ],
        },
    ],
    "additional_findings": [
        {
            "id": "F1",
            "finding": "WRITE SCOPE CLEAN: no tracked file in the repo has an mtime after 2026-11-01 12:55; every untracked file written after 18:40 sits under another wave's own report directory (wfa175_refs_bib_audit, wfa175_frontier_comparability, wfa175_pin_repair_refutation, wfa175_dead_gate_wiring). The BAND branch's writes are confined to molmetal/reports/wfa175_band_provenance/. The 'nothing else was written' claim holds.",
        },
        {
            "id": "F2",
            "finding": "THE BRANCH'S OWN CENSUS HAS THE DEFECT IT NAMES. Its headline item 4 is 'a guard's census is not a coverage measurement', but the census excludes gitignored files, and Wf-A171's OPEN-5 had already logged exactly that blind spot ('.gitignore:158 hides every molmetal/reports/**/*.json from the census'). 21 ignored files (81 lines) carry the band. The branch's own report never mentions .gitignore or OPEN-5, and its 6-item repair list does not include closing it.",
        },
        {
            "id": "F3",
            "finding": "POPULATION NUMBERS IN REPORT 1 ARE ALREADY STALE: the report says 10,031 candidates = 9,551 tracked + 480 untracked-not-ignored; measured now: 10,056 = 9,550 tracked + 506 untracked (sibling waves). The band census itself is unaffected (none of the new files carries the band).",
        },
    ],
    # ------------------------------------------------------------------
    # Wf-A190 (2026-11-01) -- LOCATOR DEMOTION for the LINE-NUMBER LOCATOR
    # CLASS.  Append-only + dated (AGENTS.md 5.5 rule #6): the superseded
    # renderings survive VERBATIM below; the LIVE locators are JSON PATHS.
    # ------------------------------------------------------------------
    "locator_demotion_2026_11_01_stale_by_construction": {
        "measured_on": "2026-11-01",
        "wave": "Wf-A190",
        "record": REC,
        "record_line_count_at_measurement": 1136,
        "record_growth_note": (
            "the record was regenerated at least three times on 2026-11-01 "
            "(Wf-A177, Wf-A180, Wf-A182; Wf-A182 alone inserted 67 lines, "
            "1069 -> 1136, and moved 29 leaves). A LINE NUMBER is therefore a "
            "PERISHABLE locator; a JSON PATH is ROBUST and survives any number "
            "of regenerations."
        ),
        "rule": (
            "the LIVE locator is a JSON PATH. A line number may only appear "
            "inside a DATED historical record that says, in words, that it was "
            "measured on a given date and that it is STALE BY CONSTRUCTION. "
            "Never delete a line number: demote it."
        ),
        "superseded_strings_verbatim": [
            {
                "where": "claims[1].counter_evidence[1]",
                "text": "only ONE (frontier_matrix.md:68) is correct-and-per-cell",
            },
            {
                "where": "claims[2].claim",
                "text": "'DIFFERENT post-processing' refuted by frontier_matrix.json:58,67 post_processed:false",
            },
            {
                "where": "claims[2].counter_evidence[0]",
                "text": (
                    "CITATION SLIP: the two BAND endpoints' post_processed:false "
                    "are at frontier_matrix.json:58 (92.7 % = GeoLDM Val&Uniq, "
                    "'endpoint': '92.7 %') and :80 (94 % = HLTF PB-Valid, "
                    "'endpoint': '94 %'). Line :67 is candidate_b of the 92.7 % "
                    "endpoint = the YUELBOND per-bond-type F1 -- a DIFFERENT "
                    "quantity. The branch's 3.2 (58,67,80) is right; its 3.1 -- "
                    "and the one-line claim -- cite '58,67' for 'both band "
                    "endpoints', which is wrong. The conclusion is unaffected "
                    "(all three are false)."
                ),
            },
            {
                "where": "claims[2].counter_evidence[1]",
                "text": "frontier_matrix.md:353",
            },
            {
                "where": "claims[2].evidence_paths[2]",
                "text": "molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:52-88",
            },
            {
                "where": "claims[3].claim",
                "text": "refuted by frontier_matrix.json:58,67,80 and synthesis.md:310",
            },
            {
                "where": "claims[3].what_i_checked[3]",
                "text": "Refuting evidence verified: frontier_matrix.json:58/:67/:80 all 'post_processed': false",
            },
            {
                "where": "refutation.md, per-claim table row 2",
                "text": "only `frontier_matrix.md:68` is correct",
            },
            {
                "where": "refutation.md, per-claim table row 3",
                "text": (
                    "Citation slip: the two band endpoints' `post_processed:false` "
                    "are at `frontier_matrix.json:58` (92.7 %) and **:80** (94 %); "
                    "**:67 is the YuelBond F1 candidate**, a different quantity."
                ),
            },
        ],
        "what_the_superseded_lines_actually_contained_on_2026_11_01": {
            "58": (
                "band_provenance.competing_reading_and_what_this_does_not_settle"
                ".cross_references[1] -- the string naming "
                "sources['HLTF'].cells['METRIC'].note"
            ),
            "67": (
                "band_provenance.competing_reading_and_what_this_does_not_settle"
                ".readings[0].current_line_numbers_on_disk"
            ),
            "80": (
                "band_provenance.competing_reading_and_what_this_does_not_settle"
                ".readings[1].current_line_numbers_on_disk"
            ),
            "89": (
                "band_provenance.competing_reading_and_what_this_does_not_settle"
                ".readings[1].note -- \"92.98 / 94.33 would round to '93.0-94.3' "
                "...\"; the LIVE band_provenance.headline_finding is at :144"
            ),
            "52-88": (
                ":52 IS band_provenance.competing_reading_and_what_this_does_not_"
                "settle (start of the range: right); :88 is .readings[1]."
                "makes_the_lower_endpoint_a_connected_component_count (end of the "
                "range: wrong)"
            ),
            "145": (
                "band_provenance.headline_finding_superseded_2026_11_01 -- the ONE "
                "claim that was right, and only by coincidence"
            ),
            "362": "\"bytes\": null",
            "763-764": (
                "\"label\": \"Validity (QM9, GGFlow)\" and a GGFlow locator string; "
                "the LIVE sources.HLTF.cells.METRIC.note_superseded_2026_11_01 "
                "is at :830"
            ),
            "779": (
                "a GGFlow quote row; the LIVE sources.HLTF.citation is at :844"
            ),
            "frontier_matrix.md:68": "renders sources.EF-TALFM.extra_observations[0]",
            "frontier_matrix.md:353": "renders band_provenance.band_as_quoted_in_the_repo",
        },
        "durable_replacements": {
            "92.7 % endpoint CANDIDATE (GeoLDM Val&Uniq, post_processed:false)": PATH_92,
            "YuelBond per-bond-type classification F1 (post_processed:false) -- a DIFFERENT quantity": PATH_YB,
            "94 % endpoint CANDIDATE (HLTF PB-Valid, post_processed:false)": PATH_94,
            "the band_provenance block": PATH_BP,
            "the band as quoted in the repo": PATH_BAQ,
            "the only `post` site that is correct and per-cell": PATH_EO0,
        },
        "no_candidate_c_note": (
            "the record has NO `candidate_c`: the third candidate for the same "
            "numeral is `endpoint_candidates[0].candidate_b` (YuelBond's "
            "per-bond-type classification F1), and the 94 % endpoint is "
            "`endpoint_candidates[1].candidate_a`."
        ),
        "the_citation_slip_finding_survives": (
            "the SUBSTANTIVE finding is unchanged and now carried by PATHS: a "
            "claim that cited the 92.7 % endpoint's `candidate_a` AND its "
            "`candidate_b` for 'both band endpoints' was wrong, because "
            "`band_provenance.endpoint_candidates[0].candidate_b` is the YUELBOND "
            "per-bond-type classification F1 -- a DIFFERENT quantity. The "
            "conclusion is unaffected (all three are false)."
        ),
    },
    # ------------------------------------------------------------------
    # Wf-A190 (2026-11-02) -- LOCATOR DEMOTION for the OTHER growing
    # artefacts this refutation cites: the APPEND-ONLY catalogue and the
    # three REGENERATED metric artefacts.  Same rule, same protocol
    # (AGENTS.md 5.5 rule #6): the LIVE locator is a SECTION ANCHOR (markdown)
    # or a JSON PATH (JSON); every demoted line number survives VERBATIM.
    # ------------------------------------------------------------------
    "locator_demotion_2026_11_02_stale_by_construction": {
        "measured_on": "2026-11-02",
        "wave": "Wf-A190",
        "growing_artefacts": {
            "docs/HONEST_NEGATIVE_CATALOG.md": (
                "APPEND-ONLY (AGENTS.md 5.5 rule #6) and grows every wave: "
                "21,688 lines on 2026-11-02. A LINE NUMBER is perishable; a "
                "SECTION ANCHOR is robust."
            ),
            "metrics/by_metric/wfa155_ligand_connected_rate.json": (
                "REGENERATED by molmetal/scripts/wf_a155_ligand_connected_rate.py "
                "-- 306 lines on 2026-11-02. A JSON PATH is robust."
            ),
            "metrics/by_metric/wfa155_metal_coordination_sphere.json": (
                "REGENERATED by molmetal/scripts/wf_a155_metal_coordination_sphere.py "
                "-- 73,150 lines on 2026-11-02. A JSON PATH is robust."
            ),
            "metrics/by_metric/wfa155_posebusters_validity.json": (
                "REGENERATED by molmetal/scripts/wf_a155_posebusters_validity.py "
                "-- 1,375 lines on 2026-11-02. A JSON PATH is robust."
            ),
            "molmetal/reports/wfa175_band_provenance/report.md": (
                "this branch's OWN report; it grew when Wf-A190 appended sections "
                "7 and 8 and grows again with section 9. A SECTION ANCHOR is robust."
            ),
        },
        "rule": (
            "the LIVE locator is a SECTION ANCHOR (markdown) or a JSON PATH (JSON). "
            "A line number may only appear inside a DATED historical record that "
            "says, in words, that it was measured on a given date and that it is "
            "STALE BY CONSTRUCTION. Never delete a line number: demote it."
        ),
        "superseded_strings_verbatim": [
            {
                "where": "claims[0].evidence_paths[0]",
                "text": "docs/HONEST_NEGATIVE_CATALOG.md:21287-21295",
            },
            {
                "where": "claims[1].evidence_paths[1]",
                "text": "molmetal/reports/wfa175_band_provenance/report.md:86-123",
            },
            {
                "where": "claims[3].claim",
                "text": "catalog:20819",
            },
            {
                "where": "claims[3].evidence_paths[0]",
                "text": "docs/HONEST_NEGATIVE_CATALOG.md:20817,20819",
            },
            {
                "where": "claims[3].evidence_paths[1]",
                "text": "metrics/by_metric/wfa155_ligand_connected_rate.json:228,238",
            },
            {
                "where": "claims[3].evidence_paths[2]",
                "text": "metrics/by_metric/wfa155_metal_coordination_sphere.json:26,29",
            },
            {
                "where": "claims[3].evidence_paths[3]",
                "text": "metrics/by_metric/wfa155_posebusters_validity.json:596,608",
            },
            {
                "where": "claims[5].claim",
                "text": (
                    "metrics/by_metric/wfa155_posebusters_validity.json:554,555,558,561 "
                    "(band_connected_pct 92.7 / 94.0 and EF-TALFM 92.98 / 94.33)"
                ),
            },
            {
                "where": "claims[5].evidence_paths[0]",
                "text": "metrics/by_metric/wfa155_posebusters_validity.json:552-561",
            },
            {
                "where": "refutation.md, per-claim table row 6",
                "text": (
                    "`wfa155_posebusters_validity.json:554,555,558,561` = 92.7 / 94.0 "
                    "/ 92.98 / 94.33, field names exact"
                ),
            },
            {
                "where": "claims[2].claim",
                "text": (
                    "The band is asserted as OURS with refuted provenance at "
                    "wfa161_frontier_labelling.py:934 and "
                    "wfa161_frontier_labelling/report.md:139;"
                ),
            },
            {
                "where": "claims[2].evidence_paths[0]",
                "text": "molmetal/scripts/wfa161_frontier_labelling.py:934",
            },
            {
                "where": "claims[2].evidence_paths[1]",
                "text": "molmetal/reports/wfa161_frontier_labelling/report.md:139",
            },
            {
                "where": "claims[3].evidence_paths[8]",
                "text": "molmetal/scripts/wf_a155_ligand_connected_rate.py:223",
            },
            {
                "where": "claims[3].evidence_paths[9]",
                "text": "molmetal/scripts/wf_a155_posebusters_validity.py:489",
            },
            {
                "where": "claims[3].evidence_paths[10]",
                "text": "molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464",
            },
        ],
        "what_the_superseded_line_numbers_resolve_to_on_2026_11_02": {
            "catalog 20817": (
                "STILL EXACT: the DENOMINATOR row of the four-breaks table -- "
                "'| **DENOMINATOR** | 92.7-94 % measured over **10,000-15,000** "
                "generated molecules from a training-set-distribution sampling | ...' "
                "(the catalogue is append-only, so earlier lines did not move)"
            ),
            "catalog 20819": (
                "STILL EXACT: the POST-PROCESSING row -- '| **POST-PROCESSING** | the "
                "band is measured on the published **post-processed** stream | both arms "
                "are **RAW** ... |'"
            ),
            "catalog 21287-21295": (
                "STILL EXACT: the band-census table under the '### 6.266.8 -- The "
                "frontier papers, pinned from primary sources (Wf-A168) ...' heading; "
                "L21287 = '**CORRECTED (MEASURED 2026-11-01):** the band occurs in "
                "**32 tracked text files / 74 matches** -- not 5 files / 13 lines.'"
            ),
            "catalog 20288": (
                "STILL EXACT: the '## 6.264 -- THE CENSUS WAS **REPUBLISHED A THIRD "
                "TIME** ...' heading"
            ),
            "catalog 18670-18674": (
                "STILL EXACT: the section 6.252 band block; L18674 = '=> **The "
                "published band is POST-PROCESSED. Our 8.51 % is RAW.** Quoting them'"
            ),
            "ligand 228": (
                "STILL EXACT: '\"frontier\": \"92.7-94 % is measured over 10,000-15,000 "
                "generated molecules from a training-set-distribution sampling.\"' == "
                "comparability.breaks[0].frontier"
            ),
            "ligand 238": (
                "STALE in TEXT, exact in FIELD: L238 is still `comparability.breaks[2]"
                ".frontier`, but it now reads the two-readings rendering ('under the "
                "`band_provenance` reading the band's two endpoints are RAW ...'). The "
                "blanket sentence 'the published band is measured on the published "
                "POST-PROCESSED stream.' is GONE from this artefact entirely -- a "
                "TEXT-ONLY replacement by Wf-A183/Wf-A189 on 2026-11-01."
            ),
            "metal_coord 26": (
                "STALE: L26 is now '\"n_rows_analyzed\": 2112,'; the cited field "
                "(comparability_statement.summary) is now at L38"
            ),
            "metal_coord 29": (
                "STALE: L29 is now '\"single_fragment_rate_percent\": 25.05'; the cited "
                "field (comparability_statement.break_3_post_processing) is now at L68; "
                "the blanket rendering survives ONLY inside "
                "comparability_statement.break_3_post_processing_superseded_2026_11_01_"
                "wfa176.rendering (L86)"
            ),
            "posebusters 554": (
                "STALE: L554 is now '\"n_pb_valid\": 29,'; the value 92.7 is now at L576 "
                "(comparability_vs_frontier_band.frontier_band.band_connected_pct[0]; "
                "the key is at L575)"
            ),
            "posebusters 555": (
                "STALE: L555 is now '\"n_scored\": 1472,'; the value 94.0 is now at L577 "
                "(band_connected_pct[1])"
            ),
            "posebusters 558": (
                "STALE: L558 is now '\"n_unparseable\": 0,'; the value 92.98 is now at "
                "L580 (frontier_band.ef_talfm_arxiv_2609_08333.connected_pct)"
            ),
            "posebusters 561": (
                "STALE: L561 is now '\"strict_denominator_definition\": \"every row in "
                "scope, including rows that could not be parsed or embedded\"'; the "
                "value 94.33 is now at L583 "
                "(frontier_band.ef_talfm_arxiv_2609_08333.novel_unique_sanitized_pct)"
            ),
            "posebusters 596": (
                "STALE: L596 is now '\"dataset\": \"GEOM-Drugs with explicit hydrogens\"'; "
                "the live POST-PROCESSING break is "
                "comparability_vs_frontier_band.breaks[2].frontier at L621; the blanket "
                "rendering survives ONLY inside "
                "comparability_vs_frontier_band.superseded_reading_2026_11_01.superseded"
                "[0].text (L642)"
            ),
            "posebusters 608": (
                "STALE: L608 is now '\"ours\": \"ONE pool of 2112 rows spanning 3 epoch "
                "labels ...\"'; the cited field "
                "(comparability_vs_frontier_band.statement) is now at L633"
            ),
            "report.md 86-123": (
                "STALE: the branch's own report.md grew when Wf-A190 appended sections "
                "7 and 8; the cited range is now section 2, subsections 2.1 to 2.4"
            ),
        },
        "durable_replacements": {
            "the catalogue's band-census table (32 tracked text files / 74 matches)": ANCHOR_62668_BAND_CENSUS,
            "the catalogue's four-breaks table, DENOMINATOR row": ANCHOR_DENOMINATOR_ROW,
            "the catalogue's four-breaks table, POST-PROCESSING row": ANCHOR_POST_PROCESSING_ROW,
            "the catalogue's 37.4-96.0 % band section": ANCHOR_6252_BAND,
            "the catalogue's may-say / may-not-say block": ANCHOR_6252_MAY_MAY_NOT,
            "the ligand artefact's DENOMINATOR break (the dash site)": P_LIGAND_DENOM,
            "the ligand artefact's POST-PROCESSING break": P_LIGAND_POST,
            "the metal-coordination artefact's summary (the dash site)": P_METAL_SUMMARY,
            "the metal-coordination artefact's POST-PROCESSING leaf": P_METAL_POST,
            "the posebusters artefact's POST-PROCESSING break": P_POSEB_POST,
            "the posebusters artefact's statement (the dash site)": P_POSEB_STMT,
            "the posebusters artefact's band_connected_pct pair (92.7 / 94.0)": P_POSEB_BAND,
            "the posebusters artefact's EF-TALFM pair (92.98 / 94.33)": P_POSEB_EF_TALFM,
        },
        "corrections": [
            {
                "what": (
                    "the section number of the catalogue's four-breaks table: the "
                    "branch said 6.264; the table is under 6.265."
                ),
                "superseded_rendering_verbatim": (
                    "docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264 -> subsection '4. "
                    "THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT "
                    "BREAKS'"
                ),
                "corrected_to": ANCHOR_6265_FOUR_BREAKS,
                "measured": (
                    "'## 6.264 --' is at L20288 (the census-republish wave) and the "
                    "four-breaks subsection heading '### 4. THE PUBLISHED 92.7-94 % BAND "
                    "IS **NOT OUR BAND** -- **FOUR** INDEPENDENT BREAKS' is at L20808, "
                    "inside '## 6.265 --' at L20625. The same correction is recorded in "
                    "the sibling Wf-A186 artefact "
                    "(molmetal/reports/wfa186_frontier_primary_source/"
                    "primary_source_verification.json -> locator_demotion_2026_11_02_"
                    "stale_by_construction -> corrections[0])."
                ),
            }
        ],
        "verification_records_left_verbatim": [
            {
                "where": "claims[1].what_i_checked[2], claims[2].what_i_checked[2], claims[3].what_i_checked[0..2], claims[5].what_i_checked[0]",
                "why": (
                    "these are VERIFICATION RECORDS -- they certify what was read on "
                    "2026-11-01 and they quote the line numbers as they stood then. "
                    "They are NOT live claims and are preserved VERBATIM; repairing "
                    "them would destroy the evidence that the check was made. Their "
                    "line numbers have MOVED -- see "
                    "what_the_superseded_line_numbers_resolve_to_on_2026_11_02."
                ),
            },
            {
                "where": "claims[*].counter_evidence[*]",
                "why": (
                    "these quote the BRANCH's own prose and the record's superseded "
                    "renderings (a REFUTATION QUOTE); they are preserved VERBATIM."
                ),
            },
            {
                "where": "claims[3].what_i_checked[4]",
                "why": (
                    "a VERIFICATION RECORD that has since become FALSE: it says "
                    "'the artifact on disk carries the OLD text -- i.e. artifact "
                    "STALE vs generator'. MEASURED on 2026-11-02: the artefact and "
                    "its generator now carry the SAME text (the 'ONE table / Table "
                    "1(b) ... side by side' rendering), so the artefact is no longer "
                    "stale vs its generator. It is preserved VERBATIM because it is "
                    "the evidence that the check was made; the correction lives in "
                    "source_file_citations_drifted_2026_11_02 and in "
                    "molmetal/reports/wfa175_band_provenance/report.md -> section 9.2b."
                ),
            },
        ],
        "source_file_citations_drifted_2026_11_02": {
            "rule": (
                "a source file is not regenerated, so a `file:line` into it is not "
                "the LOCATOR CLASS -- but a LIVE claim pointing at a line that no "
                "longer carries the quoted content IS a live defect. Each was "
                "re-measured on 2026-11-02, repaired in place, and the superseded "
                "line number preserved VERBATIM in superseded_strings_verbatim."
            ),
            "drift": {
                "molmetal/scripts/wfa161_frontier_labelling.py:934": (
                    "STALE -- L934 is now 'validity on QM9 -- a SIX-CHECK pose verdict "
                    "(bond lengths/angles,'; the quoted sentence is at L913"
                ),
                "molmetal/reports/wfa161_frontier_labelling/report.md:139": (
                    "STALE -- L139 is now the X-class entry \"2. **Post-processing.** The "
                    "frontier's connectivity rates are NOT computed on the atom-count "
                    "largest connected component after RDKit sanitization: EF-\"; the "
                    "quoted sentence is at L158"
                ),
                "molmetal/scripts/wf_a155_ligand_connected_rate.py:223": (
                    "STALE -- L223 is now 'why_superseded': \"the superseded rendering "
                    "asserted ONE provenance reading as settled: ...'; the quoted string "
                    "survives ONLY at L293, inside a commented-out superseded block"
                ),
                "molmetal/scripts/wf_a155_posebusters_validity.py:489": (
                    "STALE -- L489 is now 'for label, count in sorted(epoch_counts.items(), "
                    "key=lambda kv: str(kv[0]))'; the quoted string survives ONLY at L611 "
                    "(commented) and at L1545 (the emitter's own `superseded` list)"
                ),
                "molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464": (
                    "STALE -- L452 and L456 are Wf-A189 comments and L464 is the DATED "
                    "superseded `rendering`; the quoted string survives ONLY at L592, "
                    "inside a comment; the LIVE `break_3_post_processing` is emitted at "
                    "L634 / L789"
                ),
            },
        },
    },
}

MD = """# Wf-A175 -- ADVERSARIAL REFUTATION of branch BAND

**Date:** 2026-11-01 · CPU-only (`CUDA_VISIBLE_DEVICES=""` + `/home/hugo/codes/mol-metal/.venv/bin/python`) ·
writes confined to `molmetal/reports/wfa175_band_provenance/refutation/` · no `os.walk` (population via `git ls-files`).

**Tally: REFUTED 0 | SURVIVED 2 | IMPRECISE 4**

Every number below is MEASURED by me against the working tree at `/home/hugo/codes/mol-metal`
(this directory: `_recheck.py`, `_recheck.json`).

## Per-claim

| # | Verdict | One-line reason |
|---|---|---|
| 1 | SURVIVED | Canonical regex reproduces the catalog's 32 files / 74 lines cell-for-cell (per-file diff empty); the +2 files are the 2 Wf-A171 reports (4+3 lines, mtime 12:37:07 / 12:40:33). Caveat: the catalog counts LINES; occurrences are 75/82 (one line holds the band twice). Caveat: the census is self-inflating -- re-running the branch's own `_census.py` now returns 36 files / 96 lines / 97 occurrences because the branch's own `report.md` (8 lines) and `_build_sites.py` (7 lines) joined the population. |
| 2 | IMPRECISE | All six counts are EXACT from `sites.json`. But the "79 %" belongs to the canonical 59/75 subset (today 66/82 = 80.5 %), and the branch's own §2.2 prose is inverted ("6 of the 7 `post` sites are correct and per-cell" -- `sites.json` says 6 of 7 are the blanket-REFUTED ones, only the record's `sources`.`EF-TALFM`.`extra_observations`[0] is correct). Undisclosed scope cap: 21 gitignored files outside the branch carry the band on 81 lines, incl. `wfa171_band_publication_audit/band_classification.json` (32 lines) -- Wf-A171's OPEN-5 already named this blind spot; the report never mentions `.gitignore`. |
| 3 | IMPRECISE | Both quotes verified verbatim at the exact lines; Open Babel refutation and the OPEN-1 prior art verified. Citation slip (restated in JSON PATH terms by Wf-A190 on 2026-11-01): the two band endpoints' `post_processed:false` are at `band_provenance.endpoint_candidates[0].candidate_a` (92.7 %) and `band_provenance.endpoint_candidates[1].candidate_a` (94 %); **`band_provenance.endpoint_candidates[0].candidate_b` is the YuelBond F1 candidate**, a different quantity. "Every other site calls it external" overstates: 5 further sites attach "the repo's" to the band (the branch's own §3.5). No primary re-fetch (branch states it). |
| 4 | IMPRECISE | All 6 surfaces + the 4 dash blocks + the "two lines apart" geometry + the refuting evidence verified line-exact. Only the label is wrong: those three "published metric artifacts" are EXACTLY the 3 of 582 `metrics/by_metric/*.json` with **no registry row** (579 registered) -- unregistered, and one is STALE vs its generator. |
| 5 | IMPRECISE | The headline SURVIVES (I re-swept all 123 tracked `paper/**` files: 0 band hits under 3 notations; every `92.7` is the YuelBond F1 string or SVG noise; `build_log.txt` is tracked + ISO-8859 + 8 hits and PATH ugrep does skip it -- `grep -c` rc=1 vs 8 found). But the value sweep is not "0 except 99.9" (99.9→12, 8.51→19, 92.7→18, 92.0→10, 98.8→7, 87.9→6, 91.9→5, 91.2→3 hits, all noise), and the SVG split "4/5/3/4" does not reproduce (the total 16 reproduces under `9\\d\\.\\d(?!\\d)`, whose split is 13/3/0/0). |
| 6 | SURVIVED | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.band_connected_pct` = 92.7 / 94.0 and `...frontier_band.ef_talfm_arxiv_2609_08333` = 92.98 / 94.33, field names exact; OPEN-3 / OPEN-4 confirmed at `wfa171_band_publication_audit/report.md:179-184`. |

## The one correction that matters

The branch teaches "a guard's own census is not a coverage measurement" and then ships a census with
the same class of blind spot it names: its population is `tracked + untracked-not-ignored`, so
**411,240 gitignored files are invisible**, and **21 of them carry the band on 81 lines** -- including
Wf-A171's own `band_classification.json` (32 lines), which is exactly the artifact a band-provenance
audit should have read. Wf-A171 logged this as **OPEN-5** (`.gitignore:158` hides every
`molmetal/reports/**/*.json`). "82 sites / 34 files" is complete only for the non-ignored tree, and
the branch's 6-item repair list does not close it.

Second, the report's §2.2 ("6 of the 7 `post` sites are correct and per-cell") contradicts its own
`sites.json`: 6 of the 7 are the blanket-refuted ones. A repairer following §2.2 would leave 6 wrong
sites unrepaired.

## Wf-A190 (2026-11-01) -- LOCATOR DEMOTION: the record is cited by JSON PATH now

**MEASURED 2026-11-01. Every line number below is STALE BY CONSTRUCTION.**

The record `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` was
**1,136 lines** on 2026-11-01 and KEEPS GROWING: it was regenerated at least three times on
that day alone (Wf-A177, Wf-A180, Wf-A182 -- Wf-A182 alone inserted 67 lines, 1069 -> 1136,
and moved 29 leaves). A LINE NUMBER into it is a PERISHABLE locator; a JSON PATH is ROBUST
and survives any number of regenerations. Nothing is deleted (AGENTS.md 5.5 rule #6): the
superseded renderings survive VERBATIM below.

**Superseded renderings, PRESERVED VERBATIM (measured 2026-11-01 against the 1,136-line
record; STALE BY CONSTRUCTION):**

| where | superseded rendering (verbatim) |
|---|---|
| claims[1].counter_evidence[1] | `only ONE (frontier_matrix.md:68) is correct-and-per-cell` |
| claims[2].claim | `'DIFFERENT post-processing' refuted by frontier_matrix.json:58,67 post_processed:false` |
| claims[2].counter_evidence[0] | `CITATION SLIP: the two BAND endpoints' post_processed:false are at frontier_matrix.json:58 (92.7 % = GeoLDM Val&Uniq, 'endpoint': '92.7 %') and :80 (94 % = HLTF PB-Valid, 'endpoint': '94 %'). Line :67 is candidate_b of the 92.7 % endpoint = the YUELBOND per-bond-type F1 -- a DIFFERENT quantity. The branch's 3.2 (58,67,80) is right; its 3.1 -- and the one-line claim -- cite '58,67' for 'both band endpoints', which is wrong. The conclusion is unaffected (all three are false).` |
| claims[2].counter_evidence[1] | `frontier_matrix.md:353` |
| claims[2].evidence_paths[2] | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:52-88` |
| claims[3].claim | `refuted by frontier_matrix.json:58,67,80 and synthesis.md:310` |
| claims[3].what_i_checked[3] | `Refuting evidence verified: frontier_matrix.json:58/:67/:80 all 'post_processed': false` |
| this file, per-claim table row 2 | `only `frontier_matrix.md:68` is correct` |
| this file, per-claim table row 3 | `Citation slip: the two band endpoints' `post_processed:false` are at `frontier_matrix.json:58` (92.7 %) and **:80** (94 %); **:67 is the YuelBond F1 candidate**, a different quantity.` |

**What those lines ACTUALLY contained on 2026-11-01 (measured on the record):**

| claimed line | what it actually held |
|---|---|
| `:58` | `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`cross_references`[1] -- the string naming `sources`['HLTF'].`cells`['METRIC'].`note` |
| `:67` | `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings`[0].`current_line_numbers_on_disk` |
| `:80` | `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings`[1].`current_line_numbers_on_disk` |
| `:89` | `band_provenance`.`competing_reading_and_what_this_does_not_settle`.`readings`[1].`note`; the LIVE `band_provenance`.`headline_finding` is at `:144` |
| `:52-88` | `:52` IS `band_provenance`.`competing_reading_and_what_this_does_not_settle` (start right); `:88` is `.readings`[1].`makes_the_lower_endpoint_a_connected_component_count` (end wrong) |
| `:145` | `band_provenance`.`headline_finding_superseded_2026_11_01` -- the ONE claim that was right, and only by coincidence |

**The durable replacements (JSON PATHS; each verified to EXIST in the record):**

| what | durable locator |
|---|---|
| the 92.7 % endpoint CANDIDATE (GeoLDM Val&Uniq, `post_processed: false`) | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance`.`endpoint_candidates`[0].`candidate_a` |
| the YuelBond per-bond-type classification F1 (`post_processed: false`) -- a DIFFERENT quantity | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance`.`endpoint_candidates`[0].`candidate_b` |
| the 94 % endpoint CANDIDATE (HLTF PB-Valid, `post_processed: false`) | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance`.`endpoint_candidates`[1].`candidate_a` |
| the `band_provenance` block | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` |
| the band as quoted in the repo | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance`.`band_as_quoted_in_the_repo` |
| the only `post` site that is correct and per-cell | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `sources`.`EF-TALFM`.`extra_observations`[0] |

**NOTE -- the record has NO `candidate_c`.** The third candidate for the same numeral is
`band_provenance`.`endpoint_candidates`[0].`candidate_b` (YuelBond's per-bond-type
classification F1); the 94 % endpoint is `band_provenance`.`endpoint_candidates`[1].`candidate_a`.

**The CITATION SLIP finding SURVIVES, in PATH terms.** A claim that cited the 92.7 %
endpoint's `candidate_a` AND its `candidate_b` for "both band endpoints" was wrong, because
`band_provenance`.`endpoint_candidates`[0].`candidate_b` is the YUELBOND per-bond-type
classification F1 -- a DIFFERENT quantity. The conclusion is unaffected (all three are false).

## Wf-A190 (2026-11-02) -- LOCATOR DEMOTION II: the catalogue and the three regenerated metric artefacts

**MEASURED 2026-11-02. Every line number below is STALE BY CONSTRUCTION.**

The record was not the only growing artefact this refutation cites. `docs/HONEST_NEGATIVE_CATALOG.md`
is **APPEND-ONLY** and grows every wave (21,688 lines on 2026-11-02); the three
`metrics/by_metric/wfa155_*.json` artefacts are **REGENERATED** by their producers (re-written
2026-11-01 21:13 / 21:58). The LIVE locator is therefore a **SECTION ANCHOR** (markdown) or a
**JSON PATH** (JSON). Nothing is deleted (AGENTS.md 5.5 rule #6): every demoted line number
survives VERBATIM in the JSON block `locator_demotion_2026_11_02_stale_by_construction`.

**Superseded renderings, PRESERVED VERBATIM (measured 2026-11-01; STALE BY CONSTRUCTION):**

| where | superseded rendering (verbatim) |
|---|---|
| claims[0].evidence_paths[0] | `docs/HONEST_NEGATIVE_CATALOG.md:21287-21295` |
| claims[1].evidence_paths[1] | `molmetal/reports/wfa175_band_provenance/report.md:86-123` |
| claims[3].claim | `catalog:20819` |
| claims[3].evidence_paths[0] | `docs/HONEST_NEGATIVE_CATALOG.md:20817,20819` |
| claims[3].evidence_paths[1] | `metrics/by_metric/wfa155_ligand_connected_rate.json:228,238` |
| claims[3].evidence_paths[2] | `metrics/by_metric/wfa155_metal_coordination_sphere.json:26,29` |
| claims[3].evidence_paths[3] | `metrics/by_metric/wfa155_posebusters_validity.json:596,608` |
| claims[5].claim | `metrics/by_metric/wfa155_posebusters_validity.json:554,555,558,561 (band_connected_pct 92.7 / 94.0 and EF-TALFM 92.98 / 94.33)` |
| claims[5].evidence_paths[0] | `metrics/by_metric/wfa155_posebusters_validity.json:552-561` |
| this file, per-claim table row 6 | `` `wfa155_posebusters_validity.json:554,555,558,561` = 92.7 / 94.0 / 92.98 / 94.33, field names exact `` |

**What those line numbers resolve to on 2026-11-02 (MEASURED):**

| cited line | what it holds on 2026-11-02 |
|---|---|
| catalogue `20817` | STILL EXACT -- the DENOMINATOR row of the four-breaks table (append-only, so earlier lines did not move) |
| catalogue `20819` | STILL EXACT -- the POST-PROCESSING row |
| catalogue `21287-21295` | STILL EXACT -- the band-census table under the `### 6.266.8` heading |
| catalogue `18670-18674` | STILL EXACT -- the `6.252` band block |
| ligand `:228` | STILL EXACT -- `comparability.breaks[0].frontier` (the DENOMINATOR break) |
| ligand `:238` | same FIELD, TEXT CHANGED -- `comparability.breaks[2].frontier` now reads the two-readings rendering; the blanket sentence is GONE from this artefact entirely |
| metal_coord `:26` | STALE -- `"n_rows_analyzed": 2112,`; the cited `comparability_statement.summary` is now at L38 |
| metal_coord `:29` | STALE -- `"single_fragment_rate_percent": 25.05`; the cited leaf is now at L68; the blanket rendering survives ONLY inside `comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa176.rendering` (L86) |
| posebusters `:554` | STALE -- `"n_pb_valid": 29,`; the value 92.7 is now at L576 (`band_connected_pct[0]`, key at L575) |
| posebusters `:555` | STALE -- `"n_scored": 1472,`; the value 94.0 is now at L577 |
| posebusters `:558` | STALE -- `"n_unparseable": 0,`; the value 92.98 is now at L580 (`connected_pct`) |
| posebusters `:561` | STALE -- `"strict_denominator_definition"`; the value 94.33 is now at L583 (`novel_unique_sanitized_pct`) |
| posebusters `:596` | STALE -- `"dataset": "GEOM-Drugs with explicit hydrogens"`; the live break is `comparability_vs_frontier_band.breaks[2].frontier` (L621); the blanket rendering survives ONLY inside `superseded_reading_2026_11_01.superseded[0].text` (L642) |
| posebusters `:608` | STALE -- the `ours` cell of `breaks[0]`; the cited `comparability_vs_frontier_band.statement` is now at L633 |

**The durable replacements (SECTION ANCHORS and JSON PATHS; each verified to RESOLVE on 2026-11-02):**

| what | durable locator |
|---|---|
| the catalogue's band-census table (32 tracked text files / 74 matches) | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.266.8 -> the band-census table` |
| the catalogue's four-breaks table, DENOMINATOR row | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the DENOMINATOR row` |
| the catalogue's four-breaks table, POST-PROCESSING row | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the POST-PROCESSING row` |
| the catalogue's 37.4-96.0 % band section | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The published band, and the denominator trap'` |
| the catalogue's may-say / may-not-say block | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The sentence we may say, and the sentences we may not'` |
| the ligand artefact's DENOMINATOR break (the dash site) | `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[0].frontier` |
| the ligand artefact's POST-PROCESSING break | `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` |
| the metal-coordination artefact's summary (the dash site) | `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.summary` |
| the metal-coordination artefact's POST-PROCESSING leaf | `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing` |
| the posebusters artefact's POST-PROCESSING break | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.breaks[2].frontier` |
| the posebusters artefact's statement (the dash site) | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.statement` |
| the posebusters artefact's band_connected_pct pair (92.7 / 94.0) | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.band_connected_pct` |
| the posebusters artefact's EF-TALFM pair (92.98 / 94.33) | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.ef_talfm_arxiv_2609_08333` |

**ONE CORRECTION -- the section number of the four-breaks table.** The branch said the table is under
`6.264`; it is under **`6.265`**. MEASURED on 2026-11-02: `## 6.264 --` is at L20288 (the
census-republish wave) and the four-breaks subsection heading `### 4. THE PUBLISHED 92.7-94 % BAND IS
**NOT OUR BAND** -- **FOUR** INDEPENDENT BREAKS` is at L20808, inside `## 6.265 --` at L20625. The
superseded rendering `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264 -> subsection '4. THE
PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS'` is preserved VERBATIM in the
JSON block's `corrections[0]`. The sibling Wf-A186 artefact records the same correction.

**WHAT WAS LEFT ALONE, AND WHY.** The `what_i_checked` entries are VERIFICATION RECORDS -- they
certify what was read on 2026-11-01 and quote the line numbers as they stood then; they are NOT live
claims and are preserved VERBATIM (their line numbers have MOVED -- see the table above). The
`counter_evidence` entries quote the BRANCH's own prose and the record's superseded renderings (a
REFUTATION QUOTE) and are preserved VERBATIM. `molmetal/reports/wfa175_band_provenance/sites.json`
is a CENSUS DUMP whose `line` values are the census's own datum, not a citation, and is left alone.

**SOURCE-FILE CITATIONS THAT MEASURED DRIFTED (2026-11-02).** A source file is not regenerated, so
a `file:line` into it is not the LOCATOR CLASS -- but a LIVE claim pointing at a line that no longer
carries the quoted content IS a live defect. Six such citations were re-measured and repaired in
place; the superseded line numbers survive VERBATIM in the JSON block
`locator_demotion_2026_11_02_stale_by_construction` -> `superseded_strings_verbatim`.

| superseded rendering (VERBATIM) | where it was | what the cited line holds on 2026-11-02 | durable replacement |
|---|---|---|---|
| `molmetal/scripts/wfa161_frontier_labelling.py:934` | `claims[2].claim`, `claims[2].evidence_paths[0]` | `validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,` -- the quoted sentence is at L913 | the file + the quoted sentence ("Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale.") |
| `molmetal/reports/wfa161_frontier_labelling/report.md:139` | `claims[2].claim`, `claims[2].evidence_paths[1]` | the X-class entry "**Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-" -- the quoted sentence is at L158 | the file + the quoted sentence |
| `molmetal/scripts/wf_a155_ligand_connected_rate.py:223` | `claims[3].evidence_paths[8]` | `why_superseded`: "the superseded rendering asserted ONE provenance reading as settled: ..." | the file -> the emitter of `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` |
| `molmetal/scripts/wf_a155_posebusters_validity.py:489` | `claims[3].evidence_paths[9]` | `for label, count in sorted(epoch_counts.items(), key=lambda kv: str(kv[0]))` | the file -> the emitter of `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.breaks[2].frontier` |
| `molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464` | `claims[3].evidence_paths[10]` | L452 / L456 are Wf-A189 comments, L464 is the DATED superseded `rendering`; the quoted string survives only at L592, inside a comment | the file -> the emitter of `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing` |

**ONE VERIFICATION RECORD HAS SINCE BECOME FALSE, AND IS PRESERVED VERBATIM.** `claims[3].what_i_checked[4]`
says the artefact on disk "carries the OLD text -- i.e. artifact STALE vs generator". MEASURED on
2026-11-02: the artefact and its generator now carry the SAME text, so the artefact is no longer
stale vs its generator. It is preserved VERBATIM because it is the evidence that the check was made;
the correction lives in the JSON block -> `verification_records_left_verbatim` and in
`molmetal/reports/wfa175_band_provenance/report.md -> section 9.2b`.
"""

if __name__ == "__main__":
    atomic_write_file(OUT / "refutation.json", json.dumps(REFUTATION, indent=2, ensure_ascii=False))
    atomic_write_file(OUT / "refutation.md", MD)
    for p in (OUT / "refutation.json", OUT / "refutation.md"):
        txt = p.read_text(encoding="utf-8")
        print(p, len(txt), "chars; truncation:", detect_truncation(txt))
