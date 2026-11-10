# Wf-A175 -- ADVERSARIAL REFUTATION of branch BAND

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
| 5 | IMPRECISE | The headline SURVIVES (I re-swept all 123 tracked `paper/**` files: 0 band hits under 3 notations; every `92.7` is the YuelBond F1 string or SVG noise; `build_log.txt` is tracked + ISO-8859 + 8 hits and PATH ugrep does skip it -- `grep -c` rc=1 vs 8 found). But the value sweep is not "0 except 99.9" (99.9→12, 8.51→19, 92.7→18, 92.0→10, 98.8→7, 87.9→6, 91.9→5, 91.2→3 hits, all noise), and the SVG split "4/5/3/4" does not reproduce (the total 16 reproduces under `9\d\.\d(?!\d)`, whose split is 13/3/0/0). |
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
