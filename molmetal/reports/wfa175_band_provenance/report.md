# Wf-A175 — BRANCH BAND: band-provenance audit of the 92.7–94 % connectivity band

**Date:** 2026-11-01 · **Branch:** BAND · **CPU-only** (every command run with
`CUDA_VISIBLE_DEVICES=""` and `/home/hugo/codes/mol-metal/.venv/bin/python`; no training, no sampling,
no CUDA call, no decode).
**Method note:** no `os.walk` anywhere; the file population comes from `git ls-files --cached` +
`git ls-files --others --exclude-standard`, and every file is decoded UTF-8-safely (lossy for
non-UTF-8 *text*, skipped only for binary). PATH `grep` (ugrep `-I`) was **not** used for any count,
because it silently skips the repo's non-UTF-8 tracked text files — the defect catalog §6.263 /
§6.266.8 already records that trap twice.

**Deliverables:** `sites.json` (one record per band match, 82 records), `coverage_gap_sites.json`
(8 sites that characterise the band but do not carry the band string), `report.md` (this file),
plus the reproducible helpers `_census.py`, `_build_sites.py`, `_census_raw.json`,
`_occurrences_v1_dash_any.json`.

---

## 0. Headline

1. The canonical census **reproduces EXACTLY**: **32 files / 74 matching lines** — the number
   recorded in `docs/HONEST_NEGATIVE_CATALOG.md` §6.266.8 (Wf-A170). Today's tree carries
   **34 files / 81 lines / 82 occurrences**; the entire delta is **2 files / 7 lines / 7
   occurrences** created *after* that census by the Wf-A171 wave itself. **MEASURED.**
2. **`paper/` is genuinely clean** — not just under the census regex but under four notations and
   a value sweep of every frontier constituent number. **MEASURED.**
3. **Two sites assert the band as OURS, and both carry a provenance that is REFUTED.** One of them
   is a *generator* script, so the defect is live, not historical. **MEASURED.**
4. **A RAW-vs-POST defect survives on 4 on-disk surfaces + 2 generators** and is **invisible to the
   dash-joined census**, because the band's characterisation lives on a line that does not carry the
   band string. **MEASURED.**

---

## 1. Census — count reconciliation

**Regex:** `92\.7\s*[-‐‑‒–—―~]\s*94` (Python `re`), i.e. 92.7, a dash of any common Unicode
hyphen/minus/en-dash/em-dash/tilde, optional whitespace, then 94.

**Population (MEASURED 2026-11-01):** 10,031 candidate files = 9,551 tracked
(`git ls-files --cached`) + 480 untracked-and-not-ignored (`git ls-files --others
--exclude-standard`). Classification of that population: **9,733 UTF-8**, **2 non-UTF-8 text**
(lossily decoded and therefore searched), **252 binary** (skipped), **44 missing** (skipped).

| population | files | matching lines | occurrences |
|---|---|---|---|
| **canonical (Wf-A170 census population)** | **32** | **74** | **75** |
| today (canonical + 2 post-census Wf-A171 report files) | **34** | **81** | **82** |

The canonical row reproduces the recorded `32 files / 74 matches` **exactly**, and the per-directory
breakdown matches the catalog's table cell-for-cell:

| group | files | lines (catalog "matches") | occurrences |
|---|---|---|---|
| `molmetal/reports/` | 12 | 25 | 26 |
| `molmetal/scripts/` | 8 | 22 | 22 |
| `molmetal/tests/` | 4 | 6 | 6 |
| `metrics/by_metric/` | 3 | 6 | 6 |
| `docs/HONEST_NEGATIVE_CATALOG.md` + `AGENTS.md` + `TODO/INDEX.md` + `TODO/README.md` + `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` | 5 | 15 | 15 |
| **`paper/**`** | **0** | **0** | **0** |
| **`registry/`** | **0** | **0** | **0** |

⚠️ **Why 75 occurrences but 74 lines.** `molmetal/reports/wfa161_paper_unqualified_numbers/report.md:35`
carries the band **twice on one line** (in one Markdown table row). The catalog counts *lines*; this
audit counts *occurrences* and reports both. This is the whole of the 75 vs 74 difference — no
other file has two band occurrences on one line.

**The +2-file delta is fully explained.** The two extra files are
`molmetal/reports/wfa171_band_publication_audit/report.md` (4 occurrences / 4 lines) and
`molmetal/reports/wfa171_frontier_postproc/report.md` (3 occurrences / 3 lines), both mtime
2026-11-01 12:37 / 12:40 — i.e. written **after** the Wf-A170 census that produced the recorded
32 / 74. 74 + 4 + 3 = 81 lines; 32 + 2 = 34 files. **The census is not stale; the tree grew.**

---

## 2. Per-site classification

Full table: `sites.json` (82 records). Each record carries `path`, `line`, `matched_text`,
`full_sentence`, `assertion`, `raw_or_post`, `denominator_stated`, `severity`, plus
`ownership_ambiguity` and a free-text `note`. Classification is hand-curated from the sentence
context read on disk (see `_build_sites.py::CLASS`); the census half is machine-derived. The script
**fails loudly** if a census match has no class or a class has no census match.

### 2.1 `assertion` — is the band claimed as ours, quoted, or bare?

| class | canonical (75 occ.) | today (82 occ.) | meaning |
|---|---|---|---|
| `quoted` | **72** | **79** | attributed to external literature (frontier) |
| `ours` | **2** | **2** | asserted as a Mol-Metal result — **DEFECT** |
| `unattributed` | **1** | **1** | a bare number with no attribution either way |

The single `unattributed` site is
`molmetal/scripts/wfa163_narrative_truth_scan.py:534` — a docstring example
(`"""True if the token at `pos` is one end of a range ("92.7-94 %")."""`). It asserts nothing, so
severity `none`.

### 2.2 `raw_or_post` — does the surrounding text state RAW or POST-PROCESSED?

| class | canonical | today |
|---|---|---|
| `unstated` | **59** | **66** |
| `raw` | **9** | **9** |
| `post` | **7** | **7** |

**79 % of the band sites (59/75) do not state, anywhere in their enclosing block, whether the band
is RAW or POST-PROCESSED.** For 6 of the 7 `post` sites the statement is *correct and per-cell*
(they name 94.33 % as post-processed). For **6 sites the `post` statement is a blanket claim that is
REFUTED** — §3.2 below.

> ⚠️ **CORRECTION 2026-11-01 — see §7.1.** The FIRST clause above ("For 6 of the 7
> `post` sites the statement is *correct and per-cell*") is **INVERTED** relative to the
> `sites.json` it cites: **6 of the 7 `post` sites are the blanket-REFUTED ones**, and
> `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.extra_observations[0]` is the ONLY `post`
> site that is correct and per-cell. The SECOND clause (6 blanket-REFUTED) and §3.2 are correct.
> The superseded rendering is preserved VERBATIM above per AGENTS.md §5.5 rule #6; the dated
> correction is §7.1.

### 2.3 `denominator_stated` — is the denominator named?

`True` at **16** canonical / **17** today sites; `False` at 59 / 65. The denominators actually seen
in the repo: **10,000–15,000** (the band), **10,000** (HLTF QM9 Table 1(b) / EF-TALFM App. A.2),
**1,000** (YuelBond CDG), **9,433** (EF-TALFM N.&U.&S. subset), and **three different denominators
for one band** (§6.266.8).

### 2.4 `severity`

| class | canonical | today |
|---|---|---|
| `none` | 52 | 58 |
| `low` | 17 | 18 |
| `high` | **6** | **6** |

---

## 3. Load-bearing sites

### 3.1 The band asserted as OURS — 2 sites, 1 generator, 1 generated artifact

> `molmetal/scripts/wfa161_frontier_labelling.py:934`
> "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT
> post-processing, so it is not on the frontier's scale."

> `molmetal/reports/wfa161_frontier_labelling/report.md:139` (identical sentence; the report is
> generated from the script, so the defect has **two** on-disk surfaces)

**Severity: high.** Three independent problems in one sentence:

1. **Ownership.** Every other site in the repo calls the band external — "the published frontier
   band" (`molmetal/scripts/wf_a155_ligand_connected_rate.py:91`), "EXTERNAL number; this repo
   publishes no measurement of it" (`molmetal/scripts/wfa158_population_reconciliation.py:1798`),
   "PUBLISHED FRONTIER BAND (external literature -- not our metric)"
   (`molmetal/reports/wfa158_population_reconciliation/census.md:34`),
   "the published 92.7–94 % validity band is **NOT our band**"
   (`docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264`, and `AGENTS.md:63,172,372`, `TODO/INDEX.md:3`,
   `TODO/README.md:3`). This one sentence inverts that.
2. **Provenance.** "Open Babel" is **REFUTED** by
   `molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:115-127`: the 92.7 % endpoint is
   HLTF `arXiv:2603.27113v1` Table 1(b) row "Val&Uniq (%)", **GeoLDM** column, and 94 % is the same
   table's "PB-Valid (%)" **HLTF** column. Wf-A168 records that Open Babel is HLTF's *all-atom*
   branch (`synthesis.md:110`) — a different set of cells (the +PP column 92.2 / 92.0 / 91.2).
3. **Post-processing.** "a DIFFERENT post-processing" is **REFUTED**: both band endpoints carry
   `"post_processed": false` in
   `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.endpoint_candidates[0].candidate_a` (92.7 % CANDIDATE: GeoLDM Val&Uniq) and `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.endpoint_candidates[1].candidate_a` (94 % CANDIDATE: HLTF PB-Valid) — each carries `"post_processed": false`.

**Prior art (not new):** Wf-A171 already logged this exact file:line as **OPEN-1 (medium)** —
`molmetal/reports/wfa171_band_publication_audit/report.md:170-172` — and left it unrepaired because
the script was outside that branch's file ownership. This branch **agrees with the finding and
raises it to high**, on the additional ground that the "Open Babel" provenance is separately
refutable and the sentence is a *generator* (so it re-emits on every run). It is **not** a new
finding.

### 3.2 The blanket "the band is POST-PROCESSED" claim — REFUTED, and invisible to the census

**6 dash-census sites carry a `post` characterisation that is wrong.** Four of them
(`docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the DENOMINATOR row`,
`metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[0].frontier`,
`metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.summary`,
`metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.statement`) sit inside a block whose POST-PROCESSING
break asserts the blanket claim — the claim text itself lives on a *different* line, so the table
below lists **where the wrong sentence physically is**, not where the band string is:
**3 published metric artifacts + 1 catalog row + 2 generator lines.** The other two are the
`wfa161_frontier_labelling` pair (§3.1), whose sentence asserts both ownership and a post-processing
difference.

| durable locator (the superseded `file:line` is preserved VERBATIM in §9.1) | text | verdict |
|---|---|---|
| `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the POST-PROCESSING row` | "the band is measured on the published **post-processed** stream" (§6.265 §4 POST-PROCESSING row) | REFUTED |
| `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` | "the published band is measured on the published POST-PROCESSED stream." — the blanket sentence is GONE from this artefact (regenerated 2026-11-01; see §9.2) | REFUTED |
| `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa176.rendering` | "POST-PROCESSING — the published band is measured on the published POST-PROCESSED stream; ..." — the blanket rendering survives here ONLY inside the DATED superseded block (see §9.2) | REFUTED (artefact also regenerated — Wf-A171 §6) |
| `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.superseded_reading_2026_11_01.superseded[0].text` | "measured on the published POST-PROCESSED stream" — the blanket rendering survives here ONLY inside the DATED superseded block (see §9.2) | REFUTED |
| `molmetal/scripts/wf_a155_ligand_connected_rate.py` (the emitter of `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier`) | `"POST-PROCESSED stream."` (emitter of the artifact above) — the generator is REPAIRED: the string survives only inside a commented-out superseded block | REFUTED |
| `molmetal/scripts/wf_a155_posebusters_validity.py` (the emitter of `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.breaks[2].frontier`) | `"frontier": "measured on the published POST-PROCESSED stream",` (emitter) — the generator is REPAIRED | REFUTED |

**Refuting evidence, all on disk:**
- `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.endpoint_candidates[0].candidate_a` (92.7 % CANDIDATE: GeoLDM Val&Uniq), `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.endpoint_candidates[1].candidate_a` (94 % CANDIDATE: HLTF PB-Valid) and `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.endpoint_candidates[0].candidate_b` (the third candidate, YuelBond's per-bond-type F1) —
  `/band_provenance/endpoint_candidates` → **both** band endpoints carry `"post_processed": false`
  (and so does the third candidate, YuelBond's F1).
- `molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:310` — "**Never assert 'the band is
  POST-PROCESSED' as a blanket claim**"; and `:317` — "**Never assert 'the band is RAW' either**".
  The honest statement is per-cell.
- `molmetal/reports/wfa171_frontier_postproc/report.md:164-179` — all six band endpoints are HLTF
  QM9 Table 1(b) cells / EF-TALFM appendix cells, with their denominators.

**A THIRD framing, also wrong, in a live generator.**
`molmetal/scripts/wf_a155_metal_coordination_sphere.py` -> the emitter of
`metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing`
now says "the frontier publishes the post-processing axis in ONE table -- HLTF arXiv:2603.27113v1
Table 1(b) reports six models RAW and POST-PROCESSED side by side -- so 'the frontier number' is one
of two numbers depending on the arm." That was already refuted as **Wf-A164 refutation D3**
(`molmetal/reports/wfa164_coordination_sphere_refutation/refutation.md:59-63`):
the raw/PP pair lives on **GEOM-DRUGS Table 2 (N = 100,000)**; **QM9 Table 1(b) (N = 10,000) is
raw-only**, so the band has **one** arm. **MEASURED 2026-11-02:** the generator and the artefact on
disk now carry the SAME text (both hold the "ONE table / Table 1(b) ... side by side" rendering), so
this is a live, still-unrepaired defect in a `metrics/**` artefact -- not an artefact-stale-vs-generator
drift. **Wf-A186 refuted the same Table-1-vs-Table-2 confusion from the PRIMARY SOURCE**
(`molmetal/reports/wfa186_frontier_primary_source/report.md -> section 7`); `metrics/**` is outside
this wave's write scope, so it is REPORTED here, not repaired. (Correction to the previous rendering
of this paragraph -- see §9.2b.)

**Why the census cannot see this — the coverage-measurement failure.** The band's RAW/POST
characterisation never sits on the same line as the band string. `docs/HONEST_NEGATIVE_CATALOG.md`
L20817 (band string, DENOMINATOR row) and L20819 (POST-PROCESSING row) are **two lines apart**.
The same holds inside all three metric artifacts. **A line-level regex census cannot measure whether
the band was qualified.** The 8 records in `coverage_gap_sites.json` are the sites a dash-regex
census structurally cannot reach.

**Likely root cause (MEASURED for the two texts; the causal link is inference):** §6.252
(`docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The published band, and the denominator trap'`)
asserts the identical blanket sentence — "⇒ **The
published band is POST-PROCESSED. Our 8.51 % is RAW.**" — for a **different** band: "The published
connected-rate band is **37.4 %** (GeoLDM, via arXiv:2505.00518) to **96.0 %** (MegalodonFlow, MOSAIC
arXiv:2610.04161)". §6.265's four-break table appears to have inherited that row verbatim. **Two
different bands are in circulation under overlapping prose** — the 37.4–96.0 % band (§6.252) and
the 92.7–94 % band (§6.265).

### 3.3 Bare-value sites the dash regex cannot reach

A value sweep over `metrics/`, `paper/` and `registry/` for the band endpoints finds **4 bare numeric
values in a published metric artifact** that the dash-joined census does not see:

| durable JSON path (the superseded `file:line` is preserved VERBATIM in §9.1) | value | field |
|---|---|---|
| `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.band_connected_pct[0]` | `92.7` | `/comparability_vs_frontier_band/frontier_band/band_connected_pct[0]` |
| `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.band_connected_pct[1]` | `94.0` | `…/band_connected_pct[1]` |
| `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.ef_talfm_arxiv_2609_08333.connected_pct` | `92.98` | `…/ef_talfm_arxiv_2609_08333/connected_pct` |
| `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.frontier_band.ef_talfm_arxiv_2609_08333.novel_unique_sanitized_pct` | `94.33` | `…/ef_talfm_arxiv_2609_08333/novel_unique_sanitized_pct` |
This independently re-confirms Wf-A171's **OPEN-3** (low) and **OPEN-4** (low): any band census must
sweep by **VALUE** (92.7 / 92.98 / 94.0 / 94.33) as well as by the dash-joined regex. The four sites
are protected today only by the enclosing `comparability_vs_frontier_band` block name and by
`molmetal/tests/test_wfa164_coordination_sphere_narrative.py:354-370`.

### 3.4 Low-severity "unlabelled restatement" sites (already logged)

`molmetal/scripts/wfa156_bond_threshold_frontier.py:402` and the report copies it generates
(`molmetal/reports/wfa156_bond_threshold_frontier/report.md:44`,
`molmetal/reports/wfa164_bond_threshold_narrative/report.md:44`,
`molmetal/reports/wfa164_bond_threshold_narrative/c7_pool_check/report.md:44`) call it "the
92.7-94 % frontier band" **without labelling it a restatement**. Already logged as item 5 (Low) in
`molmetal/reports/wfa164_c_refutation/refutation.md:69-70`. The generator's own constant
(`molmetal/scripts/wfa156_bond_threshold_frontier.py:317-325`) *does* document the restatement
("the string '92.7' occurs ZERO times in the fetched arXiv 2609.08333v1 HTML … It is the branch's
restatement of the band, NOT the paper's own wording"), so the wording is inconsistent between the
constant and the prose that quotes it.

### 3.5 Ownership ambiguity — 5 sites

Five sites attach "the repo's" to the band itself
(`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.band_as_quoted_in_the_repo`,
`molmetal/scripts/wfa158_frontier_postproc_matrix.py` -> the emitter of `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.band_as_quoted_in_the_repo`,
`molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:20,115`,
`molmetal/reports/wfa171_frontier_postproc/report.md:314`), while the catalog, `AGENTS.md` and
`TODO/` all say "**NOT our band**". The two are reconcilable only by distinguishing *the repo
published a restatement* from *the repo measured it* — a distinction no document states outright,
and the one that §3.1's sentence collapses. `molmetal/reports/wfa171_frontier_postproc/report.md:182-183`
says it most sharply: "This is the fourth independent comparability break, and it is the repo's own
band, not the frontier's."

---

## 4. The `paper/` question — is the zero real?

**Yes. MEASURED, four independent ways, all on the tracked tree.**

1. **Census regex over `paper/**`:** 123 tracked files scanned, **0 matches, 0 files.**
2. **Wide-notation probe** (`9[234][.,]?[0-9]?\s*(?:-|–|—|to|through)\s*9[234][.,]?[0-9]?` and
   `9[23][.,][0-9]+\s*%`): **0 band matches.**
3. **Value sweep** for every frontier constituent — `92.98`, `94.33`, `93.13`, `91.9`, `99.9`,
   `98.25`, `98.21`, `98.8`, `92.2`, `92.0`, `91.2`, `87.9` — and for ours, `8.51`, `25.05`: **0
   matches in any tracked `paper/**` file** except `99.9` inside `paper/build_log.txt` (4 hits, a
   YuelBond "try distortion; 99.9%" quote inside a LaTeX transcript).
4. **`92.7` itself in `paper/`** — the only scalar hits, all a **different quantity**:

   > `paper/sections/04_evaluation.tex:3179` — "(Wang \& Dokholyan 2025 bioRxiv
   > 10.1101/2025.05.06.652517, F1=92.7\% on distorted molecules) is shipped as a drop-in
   > alternative"

   > `paper/sections/06_limitations.tex:387` — "F1=92.7\%) ships as a parallel drop-in alternative
   > but is"

   Both are the **YuelBond per-bond-type classification F1 over 1,000 coordinate-noised molecules**
   (denominator per `frontier_matrix.json` `/band_provenance/endpoint_candidates[0]/candidate_b`),
   **explicitly attributed to a named primary record**, and **not** the validity band. They are
   also the paper's *own* framing of an external number — the correct shape.

5. **`paper/build_log.txt` (tracked, ISO-8859, 12,767,523 B)** — a LaTeX build transcript carrying
   **8 copies** of the same `F1=92.7%` string. **PATH `grep -I` skips this file entirely** (it is
   the repo's non-UTF-8 tracked text file); the UTF-8-safe scanner finds it. It is a build artefact
   of the same paper source, not a claim site, and it changes no verdict.

6. **`paper/figures/*.svg`** — 16 further `9x.x` hits (4 in `fig1_mlc_architecture.svg`, 5 in
   `fig2_pipeline.svg`, 3 in `fig3_click_reactions.svg`, 4 in `fig4_homotype_vs_tanimoto.svg`).
   **All 16 are SVG `transform="translate(x y)"` coordinate noise**, not text.

**Conclusion:** `paper/` does not quote the band at all, in any notation, and does not quote our
8.51 % / 25.05 % next to it. The paper-side repair proposed at
`molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md:407` ("replacing a bare '92.7–94 %'
with the four-break sentence") has **nothing to repair** — there is no bare band in `paper/`.
That proposal should be closed as a no-op rather than queued for a `paper/**` edit under TODO-84.

---

## 5. What this branch did NOT establish

1. **No primary-record re-fetch.** Every refutation above cites an **on-disk** primary-source
   artifact (`frontier_matrix.json` `retrieval_sha256` / `sources.json`, the Wf-A168 HLTF pin set,
   the Wf-A164 live re-verification). This branch did **not** re-fetch arXiv 2603.27113v1 or
   2609.08333 itself, so the "both endpoints are RAW" claim is **MEASURED-BY-CITATION** (a value
   read out of an artifact that records a live fetch with a sha256), not re-measured here.
2. **No repair.** Every file named as load-bearing is outside this branch's write scope
   (`molmetal/scripts/**`, `metrics/**`, `docs/HONEST_NEGATIVE_CATALOG.md` are all read-only here).
   The two `wfa161_frontier_labelling` sites are inside `molmetal/reports/**` but their **emitter**
   is not, and repairing the artifact without repairing the generator would re-introduce the defect
   on the next run. **OPEN.**
3. **No metric was re-measured.** The classification is a **textual** property of the 82 sites; it
   does not re-derive any rate, and it does not touch the registry or `metrics/**`.
4. **The 44 missing files** in the population (paths that no longer exist) were skipped; they are
   the scanner's own "file disappeared between `git ls-files` and `open`" class, not unreadable
   content.

---

## 6. Recommended repair order (for the main thread; nothing applied here)

1. `molmetal/scripts/wfa161_frontier_labelling.py:934` — drop "Our", drop "Open Babel", and state
   the two endpoints' real provenance (HLTF Table 1(b), RAW). Regenerate
   `molmetal/reports/wfa161_frontier_labelling/report.md`. **Highest severity.**
2. `molmetal/scripts/wf_a155_ligand_connected_rate.py:220-223` and
   `molmetal/scripts/wf_a155_posebusters_validity.py:488-490` — replace the blanket
   "POST-PROCESSED stream" with the per-cell statement, then regenerate the two metric artifacts.
3. `molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464` — replace the "reports every model
   twice, once RAW and once POST-PROCESSED" claim (Wf-A164 D3) with the one-arm statement; the
   artifact on disk is stale and will be regenerated.
4. `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the POST-PROCESSING row` (and, if the rule is to leave no
   supersession behind, `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The published band, and the denominator trap'`) — append a dated, additive correction (§5.5 rule #6) noting the blanket POST-PROCESSED
   row is refuted for the 92.7–94 % band and that it was inherited from §6.252's **different**
   band (37.4–96.0 %). **Read-only here.**
5. Add a **VALUE-sweep** (92.7 / 92.98 / 94.0 / 94.33) to the band census, and add a **band-qualifier**
   census that looks for the RAW/POST/denominator characterisation **independently of** the band
   string — otherwise the census keeps reporting 0 on a defect that is present.

---

*All numbers in this report are MEASURED on 2026-11-01 against the working tree at
`/home/hugo/codes/mol-metal`, CPU-only, and every one traces to a path listed above. No number in
this report is PROJECTED. Nothing outside `molmetal/reports/wfa175_band_provenance/` was written.*

---

## 7. CORRECTION — 2026-11-01 (Wf-A175 A4 + A5 repair wave)

**Additive and dated, per AGENTS.md §5.5 rule #6.** Nothing above this line was edited or deleted;
the superseded renderings are quoted **VERBATIM** below and left in place. Every number in this
section is **MEASURED** on 2026-11-01 against the working tree at `/home/hugo/codes/mol-metal`,
CPU-only (`CUDA_VISIBLE_DEVICES=""` + `/home/hugo/codes/mol-metal/.venv/bin/python`), and every one
names the file it was read from.

### 7.1 A5 — §2.2's `post` reading was INVERTED relative to the `sites.json` it cites

**Superseded rendering (report.md:105-108, quoted VERBATIM, unchanged):**

> **79 % of the band sites (59/75) do not state, anywhere in their enclosing block, whether the band
> is RAW or POST-PROCESSED.** For 6 of the 7 `post` sites the statement is *correct and per-cell*
> (they name 94.33 % as post-processed). For **6 sites the `post` statement is a blanket claim that is
> REFUTED** — §3.2 below.

**The first clause of that paragraph is wrong; the second clause (§3.2's "6 … are wrong") is
right.** The report therefore contradicted itself: 6 correct + 6 refuted cannot both be true of
7 sites.

**Corrected reading — MEASURED against `molmetal/reports/wfa175_band_provenance/sites.json`
(the very file §2.2's table is derived from):**

| MEASURED fact | value | where |
|---|---|---|
| `sites.json` records | **82** | `sites.json` → `sites` |
| `raw_or_post` = `post` records | **7** | `sites.json` |
| … of which `severity` = `high` (the blanket-REFUTED class) | **6** | `sites.json` |
| … of which `severity` = `none` | **1** | `sites.json` |
| `post` records whose `note` says "correct per-cell" | **1** — `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.extra_observations[0]` | `sites.json` |
| `post` records whose `note` contains the word "REFUTED" | **5** — the 6th blanket-REFUTED site (`wfa161_frontier_labelling/report.md:139`) carries only a pointer note ("GENERATED TWICE from wfa161_frontier_labelling.py:934 — same defect, second surface"), so the count of *notes* saying "REFUTED" (5) is one lower than the count of blanket-REFUTED *sites* (6) | `sites.json` |

**In one sentence: 6 of the 7 `post` sites are the blanket-REFUTED ones, and
`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.extra_observations[0]` is the ONLY `post` site
that is correct and per-cell.** The 7 `post` records, verbatim from `sites.json`:

| `sites.json` record | `severity` | `sites.json` note (abridged) | still live on disk 2026-11-01? |
|---|---|---|---|
| `docs/HONEST_NEGATIVE_CATALOG.md:20817` | high | "the adjacent POST-PROCESSING row (L20819) asserts the band is POST-PROCESSED -- REFUTED" | **yes** — `:20819` reads "the band is measured on the published **post-processed** stream" |
| `metrics/by_metric/wfa155_ligand_connected_rate.json:228` | high | "the block's POST-PROCESSING break (L237-238) asserts POST-PROCESSED -- REFUTED" | **yes** — `:238` reads "the published band is measured on the published POST-PROCESSED stream." |
| `metrics/by_metric/wfa155_metal_coordination_sphere.json:26` | high | "break_3_post_processing (L29) asserts POST-PROCESSED -- REFUTED" | **yes** — `:29` reads "POST-PROCESSING — the published band is measured on the published POST-PROCESSED stream; …" |
| `metrics/by_metric/wfa155_posebusters_validity.json:608` | high | "break 3 (L595-596) asserts POST-PROCESSED -- REFUTED" | **yes** — `:596` reads "measured on the published POST-PROCESSED stream" |
| `molmetal/reports/wfa161_frontier_labelling/report.md:139` | high | "GENERATED TWICE from wfa161_frontier_labelling.py:934 -- same defect, second surface" | **no — REPAIRED by Wf-A176 on 2026-11-01** (see below) |
| `molmetal/scripts/wfa161_frontier_labelling.py:934` | high | "LOAD-BEARING DEFECT: 'Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) …'" | **no — REPAIRED by Wf-A176 on 2026-11-01** (see below) |
| `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.extra_observations[0]` | **none** | "names 94.33 as post-processed; correct per-cell" | yes — the sibling rendering `frontier_matrix.md` still reads "… a post-processed companion cell (N. & U. & S. = 94.33)." |

**Two of the six have since been repaired on disk, so the two `sites.json` rows that are now stale
are the `wfa161_frontier_labelling` pair.** MEASURED on 2026-11-01:

- `molmetal/scripts/wfa161_frontier_labelling.py:951-965` now carries the superseded emission
  **PRESERVED VERBATIM inside a comment** (AGENTS.md §5.5 rule #6 was honoured by the repairing
  wave), with the header `# Wf-A176 (2026-11-01) CORRECTION — the superseded emission, PRESERVED
  VERBATIM per AGENTS.md §5.5 rule #6:`;
- the live emitter at `molmetal/scripts/wfa161_frontier_labelling.py:966-975` now reads "The
  published 92.7-94 % band is **NOT ours** and is **not a connectivity band**";
- the regenerated artefact `molmetal/reports/wfa161_frontier_labelling/report.md:162` now reads
  "1. **OWNERSHIP.** The 92.7-94 % band is **NOT ours**."

Consequence: `sites.json`'s `assertion = ours` (2 records) and the report's §3.1 "generator" framing
describe a defect that **no longer exists on disk**. They are stale classification records, not live
defects. The remaining **4** blanket-REFUTED sites (catalog row + 3 metric artefacts) are still live.

**Also corrected (a denominator, not an inversion):** the "79 %" belongs to the **canonical** subset
(59/75 = 78.7 %). The same ratio on today's 82 records is 66/82 = **80.5 %**. The "canonical / today"
columns of the §2.2 table are correct as written.

### 7.2 A4 — the census population excluded the whole gitignored tree

**Superseded rendering (report.md:40-48, quoted VERBATIM, unchanged):**

> **Population (MEASURED 2026-11-01):** 10,031 candidate files = 9,551 tracked
> (`git ls-files --cached`) + 480 untracked-and-not-ignored (`git ls-files --others
> --exclude-standard`). Classification of that population: **9,733 UTF-8**, **2 non-UTF-8 text**
> (lossily decoded and therefore searched), **252 binary** (skipped), **44 missing** (skipped).
>
> | population | files | matching lines | occurrences |
> |---|---|---|---|
> | **canonical (Wf-A170 census population)** | **32** | **74** | **75** |
> | today (canonical + 2 post-census Wf-A171 report files) | **34** | **81** | **82** |

**The defect.** The population was "tracked + untracked-and-not-ignored", so **everything gitignored
was invisible** — and `.gitignore:158` (`molmetal/reports/**/*.json`) hides exactly the JSON
artefacts a band-provenance audit must read. The band-provenance audit's own sibling artefact,
`molmetal/reports/wfa171_band_publication_audit/band_classification.json` (**32** band lines), is
one of them.

**The fix.** `molmetal/reports/wfa175_band_provenance/_census.py` now enumerates **three** git
groups, with **three separate `git ls-files` calls** (a single combined call is wrong: `--exclude-standard`
also filters the `--cached` listing, so `git ls-files --cached --others --ignored --exclude-standard`
returns only **17** of the **9,550** tracked files — MEASURED). Files are streamed in 4 MiB chunks
(never fully materialised), the ignored tree is **153.96 GB** in one `.pt` file, and a file is
skipped as binary when its first 8 KiB contain a NUL (the same rule as before).

**Corrected population — MEASURED** (`_census_raw_v2_incl_ignored.json`, `generated_utc`
`2026-11-01T11:45:37Z`; 83.3 s wall, 153,958,133,020 bytes read, CPU-only):

| git group | command | files |
|---|---|---|
| tracked | `git ls-files --cached` | **9,550** |
| untracked, not ignored | `git ls-files --others --exclude-standard` | **523** |
| untracked, **IGNORED** | `git ls-files --others --ignored --exclude-standard` | **411,252** |
| **union (corrected population)** | | **421,325** |

Classification of the 421,325: **313,176 binary** (skipped), **105,585 UTF-8**,
**22 non-UTF-8 text** (lossily decoded and searched), **2,542 missing** (git-listed, absent on disk).

**Corrected band counts — MEASURED** (regex `92\.7\s*[-‐‑‒–—―~]\s*94`, `v1_dash_any`):

| population | files | matching lines | occurrences |
|---|---|---|---|
| legacy population, re-measured today (`tracked + untracked-not-ignored`) | **42** | **129** | **135** |
| **corrected population, this branch's own directory EXCLUDED** | **63** | **203** | **211** |
| corrected population, this branch's own directory INCLUDED | 70 | 658 | 676 |

Split of the 63 / 203 / 211 by git group: **tracked 5 files / 19 lines**, **untracked-not-ignored
34 files / 92 lines**, **IGNORED 24 files / 92 lines**. The branch's own directory contributes
**7 files / 455 lines / 465 occurrences** on its own (its `report.md`, `sites.json`,
`_build_sites.py`, the occurrence dumps and the refutation artefacts all carry the band string), which
is why the reproducible headline excludes it — a census that counts its own output is not a
measurement.

**The gitignored tree contributes 24 band-carrying files / 92 lines / 95 occurrences.** The largest
is `molmetal/reports/wfa171_band_publication_audit/band_classification.json` (**32** lines); the
others include `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` (4),
`molmetal/reports/wfa164_coordination_sphere_narrative/default_pool_artifact.json` (4),
`…/default_pool_diff.json` (4), `…/synthetic_pool_artifact.json` (4),
`molmetal/reports/wfa164_coordination_sphere_refutation/default_pool_recheck.json` (4),
`molmetal/reports/wfa175_frontier_comparability/sources.json` (5) and 17 more. Full list:
`_census_raw_v2_incl_ignored.json` → `per_file`.

**Coverage gap this opens.** `sites.json` holds **82** records over **34** distinct paths. The
corrected census finds **63** band-carrying files / **203** lines outside this branch's own
directory, so **29 files carrying 111 lines / 118 occurrences are NOT in `sites.json` at all**
(**24** of them in the gitignored tree, **5** untracked-but-not-ignored). `sites.json` was not
regenerated: the per-site classification is hand-curated in `_build_sites.py`, which is outside this
branch's write scope. **The 29 newly-visible sites are counted but NOT classified** — the honest
statement is "census sees them, the classifier has not ruled on them".

**A/B regression — the fix does not move the legacy numbers.** The pre-fix `_census.py` logic was
re-run unchanged on the legacy population and the corrected script's `nonignored` population was
compared with it: **identical** on the population size (10,073), on the classification
(binary 252 / utf8 9,775 / missing 44 / non-utf8-text 2) and on all four regex variants
(v1 42/129/135, v2 36/97/102, v3 41/126/131, v4 50/194/208). The corrected census is a **strict
superset**, not a different measurement.

### 7.3 What the corrected census does NOT change

1. **`paper/` stays clean — and the verdict is now measured over MORE files.** **134** `paper/**`
   files are in the corrected population (**123** tracked + **11** gitignored, which the legacy
   census never saw) and **none** of them appears in the census's band-file list (no `paper/` path
   carries a band line under `v1`). The population fix therefore *strengthens* §4 rather than
   touching it.
2. **The canonical Wf-A170 reconciliation is untouched.** The canonical row (32 files / 74 lines)
   was a legacy-population measurement; the legacy population's numbers are unchanged by this fix
   (A/B regression above).
3. **No per-site classification changed.** §2.1 (`quoted` 79 / `ours` 2 / `unattributed` 1), §2.3
   (`denominator_stated` True 17 / False 65) and §2.4 (`none` 58 / `low` 18 / `high` 6) are
   properties of `sites.json`, which was **not** regenerated. The 29 newly-visible band files are
   counted, not classified.
4. **No defect was repaired by this wave.** The four still-live blanket-REFUTED sites
   (`docs/HONEST_NEGATIVE_CATALOG.md:20819`, `metrics/by_metric/wfa155_ligand_connected_rate.json:238`,
   `metrics/by_metric/wfa155_metal_coordination_sphere.json:29`,
   `metrics/by_metric/wfa155_posebusters_validity.json:596`) and the §6 repair list are **OUT OF
   SCOPE** for this branch: the catalog and `metrics/**` are owned by the main thread, and
   `molmetal/scripts/**` was not in this wave's write scope.
5. **§3.1's "generator" framing is stale** (see §7.1): the `wfa161_frontier_labelling` generator was
   repaired by Wf-A176 on 2026-11-01, so the defect is no longer live — but `sites.json` and the
   §3.1 prose still describe it as live. **This correction does not delete §3.1**; it flags it.

### 7.4 Reproduction

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/reports/wfa175_band_provenance/_census.py
```
writes `molmetal/reports/wfa175_band_provenance/_census_raw_v2_incl_ignored.json` and
`molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json`.
The legacy files `_census_raw.json` and `_occurrences_v1_dash_any.json` are **deliberately left
untouched** by the fixed script, so the 18:57 measurement and `_build_sites.py`'s input remain
intact. `safe_overwrite_check` on the rewritten `_census.py` returned a diff ratio of **0.846**
(> the 0.5 threshold) — the rewrite was deliberate and is recorded here; `detect_truncation`
returned `None`.

*Correction appended 2026-11-01 by the Wf-A175 A4/A5 repair wave. All numbers MEASURED; no number
is PROJECTED; no file outside `molmetal/reports/wfa175_band_provenance/` was written.*

### 7.5 Addendum — the tree moved while this wave was open (2026-11-01, 19:52 local)

Two things changed between the census run quoted in §7.2 and this addendum. Both are
**concurrent-writer effects**, not a change in the measurement method.

**(a) The §7.2 numbers are a timestamped snapshot.** Superseded rendering, quoted VERBATIM:

> | legacy population, re-measured today (`tracked + untracked-not-ignored`) | **42** | **129** | **135** |
> | **corrected population, this branch's own directory EXCLUDED** | **63** | **203** | **211** |
> | corrected population, this branch's own directory INCLUDED | 70 | 658 | 676 |

Current run — `molmetal/reports/wfa175_band_provenance/_census_raw_v2_incl_ignored.json`,
`generated_utc` **2026-11-01T11:51:52Z**, 50.5 s wall, 154.05 GB read, CPU-only:

| population | files | matching lines | occurrences |
|---|---|---|---|
| legacy population, branch's own dir **EXCLUDED** | **39** | **113** | **118** |
| legacy population, branch's own dir INCLUDED | 42 | 134 | 140 |
| corrected population, branch's own dir **EXCLUDED** | **63** | **205** | **213** |
| corrected population, branch's own dir INCLUDED | 71 | 9012 | 9490 |

The two **INCLUDED** rows are **self-referential and are not measurements**: this report's own text
carries the band string, so every word written here moves them (the branch's own directory now
contributes 3 files / 21 lines to the legacy population and 8 files / 8807 lines to the full
population, most of the latter being the census's own occurrence dump). The two **EXCLUDED** rows are
the reproducible ones. They moved by 2 lines because **concurrent sibling waves** rewrote files
outside this branch while the census was running — MEASURED:
`molmetal/scripts/wf_a155_ligand_connected_rate.py` and
`molmetal/scripts/wf_a155_metal_coordination_sphere.py` carry mtime **19:51:14 local (11:51:14Z)**,
38 s before this run's `generated_utc` and 6 min after the §7.2 run's (`11:45:37Z`).
The EXCLUDED rows are invariant under **this** branch's own writes; sibling waves can still move them.

**(b) Wf-A176 repaired the three §3.2 GENERATORS while this wave was open.** MEASURED 2026-11-01
19:52 local:

- `molmetal/scripts/wf_a155_ligand_connected_rate.py:220-225` — the "POST-PROCESSED stream." text now
  survives only as an AGENTS.md §5.5 rule #6 PRESERVED-VERBATIM comment under the header
  `# Wf-A176 (2026-11-01) CORRECTION — the superseded text, PRESERVED VERBATIM …`;
- `molmetal/scripts/wf_a155_posebusters_validity.py:487-490` — same treatment;
- `molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-459` — the Wf-A164 **D3** "reports every
  model twice, once RAW and once POST-PROCESSED" claim now survives only as a preserved-verbatim
  comment.

The three **metric artefacts** still carry the old blanket text on disk —
`metrics/by_metric/wfa155_ligand_connected_rate.json:238` ("the published band is measured on the
published POST-PROCESSED stream."), `metrics/by_metric/wfa155_metal_coordination_sphere.json:29`
("POST-PROCESSING — the published band is measured on the published POST-PROCESSED stream; …"),
`metrics/by_metric/wfa155_posebusters_validity.json:596` ("measured on the published POST-PROCESSED
stream") — and `docs/HONEST_NEGATIVE_CATALOG.md:20819` still carries its POST-PROCESSING row ("the band
is measured on the published **post-processed** stream"). **Those four surfaces remain LIVE**; the
generator half of §3.2's table is repaired. `metrics/**` and
`docs/HONEST_NEGATIVE_CATALOG.md` are outside this branch's write scope, so nothing here was changed
to fix them — the repair belongs to the wave that owns them (and to the main thread for the catalog).

**(c) The EXCLUDED rows are INVARIANT — MEASURED on two consecutive runs.** The census was re-run
after this section was written (`generated_utc` **2026-11-01T11:54:36Z**, 61.0 s wall, 154.05 GB
read, CPU-only) and the two EXCLUDED rows are **byte-identical** to the 11:51:52Z run:
`all_excl_self` = **63 files / 205 lines / 213 occurrences** and `nonignored_excl_self` =
**39 files / 113 lines / 118 occurrences**. The INCLUDED rows moved again in the same interval —
`all` went **9,012 lines (11:51:52Z) -> 29,133 lines (11:54:36Z)** — because the census's own
occurrence dump is itself a band-carrying file inside the branch directory, so each run inflates
the next run's INCLUDED count (a 3.2x feedback in one run). **That is the concrete reason the
EXCLUDED rows are the headline and the INCLUDED rows are not a measurement.**

*Addendum appended 2026-11-01. All numbers MEASURED; no number is PROJECTED.*

### 7.6 Final census state (2026-11-01, 19:58 local) — what is on disk right now

Superseded run citation (§7.5(c)), quoted VERBATIM:

> **(c) The EXCLUDED rows are INVARIANT — MEASURED on two consecutive runs.** The census was re-run
> after this section was written (`generated_utc` **2026-11-01T11:54:36Z**, 61.0 s wall, 154.05 GB
> read, CPU-only) and the two EXCLUDED rows are **byte-identical** to the 11:51:52Z run:
> `all_excl_self` = **63 files / 205 lines / 213 occurrences** and `nonignored_excl_self` =
> **39 files / 113 lines / 118 occurrences**. The INCLUDED rows moved again in the same interval —
> `all` went **9,012 lines (11:51:52Z) -> 29,133 lines (11:54:36Z)** — because the census's own
> occurrence dump is itself a band-carrying file inside the branch directory, so each run inflates
> the next run's INCLUDED count (a 3.2x feedback in one run). **That is the concrete reason the
> EXCLUDED rows are the headline and the INCLUDED rows are not a measurement.**

**Two changes were made to `_census.py` to remove that feedback, and the census was re-run twice:**

1. the **occurrence dump is de-self-referenced** — `_occurrences_v1_dash_any_v2_incl_ignored.json`
   no longer contains records from `molmetal/reports/wfa175_band_provenance/` itself (the *counts*
   in `_census_raw_v2_incl_ignored.json` still include the branch's own files; only the evidence
   dump is de-self-referenced);
2. **`per_file` now lists only the band-carrying files** (146 entries) instead of all 421,327
   candidates — the population composition lives in `populations[].kinds`, which is what the report
   cites.

**The artefact-size chain, MEASURED run by run** (each run overwrites the artefact at the END of
its scan, so a run measures the PREVIOUS run's artefact size on disk):

| run (`generated_utc`) | `_census_raw_v2_incl_ignored.json` | `_occurrences_..._v2_incl_ignored.json` | `all` row (lines / occ) |
|---|---|---|---|
| 11:54:36Z (pre-fix) | 88,671,677 B | 23,989,813 B | 29,133 / 31,883 |
| 11:56:51Z (post-fix, first run) | 65,148 B | 375,620 B | **96,312 / 114,230** |
| **11:58:14Z (post-fix, on disk now)** | **65,116 B** | **375,971 B** | **1,303 / 1,361** |

The 11:56:51Z run still reported 96,312 lines because the **88,671,677 B** artefact from the
pre-fix run was still on disk **during** its scan and was itself band-carrying; the next run
(11:58:14Z) scanned the already-compacted tree and read **1,303**. So the "all" row converged
**96,312 -> 1,303 lines** once, and the census is now **re-runnable** (a further run does not
re-inflate). The de-self-referenced dump holds **214 records, 0 of them from this branch's own
directory** (MEASURED on disk).

**On-disk state (`_census_raw_v2_incl_ignored.json`, `generated_utc` **2026-11-01T11:58:14Z**,
70.4 s wall, 153.96 GB read, CPU-only):**

| population | candidates | v1 files | v1 lines | v1 occurrences |
|---|---|---|---|---|
| `all` (tracked + untracked + **IGNORED**) | **421,327** | 71 | 1,303 | 1,361 |
| `nonignored` (the legacy population) | **10,073** | 42 | 135 | 141 |
| **`all_excl_self`** (the reproducible headline) | **421,313** | **63** | **206** | **214** |
| **`nonignored_excl_self`** (the reproducible legacy baseline) | **10,067** | **39** | **114** | **119** |

Population groups: tracked **9,550** + untracked **523** + IGNORED **411,254** = union **421,327**.
Classification of the 421,327: binary **313,176** / utf8 **105,587** / non-utf8-text **22** /
missing **2,542**. Classification of the 10,073 legacy candidates: binary **252** / utf8 **9,775** /
non-utf8-text **2** / missing **44** — **identical to the A/B run in §7.2**, as it must be.

**Drift, stated plainly (this SUPERSEDES "INVARIANT" as an unqualified claim).** Between the
11:56:51Z run and the 11:58:14Z run the two EXCLUDED rows moved by **+1 line / +1 occurrence**
(205->206, 113->114). Attribution, MEASURED by `find -newermt`: the only text files modified
outside this branch's directory inside that window were
**`molmetal/scripts/wfa161_frontier_labelling.py`** and
**`molmetal/reports/wfa175_frontier_comparability/report.md`** (a sibling Wf-A175 wave), and both
are band-carrying — `grep -c` returns **6** matching lines each. So "invariant" holds **under THIS
branch's own writes only**; a concurrent sibling wave moves the count by ~1 line per 90 s.
**Read the counts as a timestamped snapshot, not as a constant.**

**Filesystem truth (AGENTS.md §5.5).** `ls -la --time-style=+%H:%M:%S` on the branch directory at
19:58 local:

```
_census_raw.json                                 578    18:57:48   <- LEGACY, UNTOUCHED
_occurrences_v1_dash_any.json                  57,677    18:57:48   <- LEGACY, UNTOUCHED
sites.json                                      78,999    19:01:46   <- UNTOUCHED (not regenerated)
coverage_gap_sites.json                          6,318    19:02:11   <- UNTOUCHED
_census.py                                      13,860    19:55:42   <- FIXED (A4)
_census_raw_v2_incl_ignored.json                65,116    19:58:14   <- CORRECTED CENSUS
_occurrences_v1_dash_any_v2_incl_ignored.json  375,971    19:58:14   <- CORRECTED OCCURRENCES
report.md                                       40,316    19:54:47   <- + this section appended
```

The two legacy files keep their 18:57:48 mtime, i.e. the first measurement is preserved verbatim
and `_build_sites.py`'s input is untouched. `sites.json` also keeps its 19:01:46 mtime: **this wave
did not regenerate the classification**, so the 29 band-carrying files the corrected census finds
outside `sites.json` remain counted-but-unclassified (see §7.2).

*§7.6 appended 2026-11-01, 19:58 local. All numbers MEASURED; no number is PROJECTED.*

### 7.7 Final per-git-group split (2026-11-01, 19:58 local) — the A4 headline, re-measured

Computed from `_census_raw_v2_incl_ignored.json` → `per_file` (`generated_utc` 11:58:14Z), split by
each record's `group` key, with this branch's own directory separated out:

| git group | files | v1 lines | v1 occurrences |
|---|---|---|---|
| tracked | 5 | 19 | 19 |
| untracked, not ignored | 34 | 95 | 100 |
| **IGNORED (the invisible tree)** | **24** | **92** | **95** |
| **total, this branch's own directory EXCLUDED** | **63** | **206** | **214** |

**The A4 headline is stable.** The gitignored tree contributes **24 band-carrying files / 92 lines
/ 95 occurrences** — the **same counts** as the §7.2 measurement taken at 11:45:37Z (24 / 92 / 95),
12.5 minutes earlier. (File *identity* was not re-diffed between the two runs; the counts are
equal, the list was not re-compared.) The hidden population did not change while the wave ran.

**With this branch's own directory INCLUDED** the split becomes tracked **5 / 19 / 19**,
untracked **37 / 116 / 122**, **ignored 29 / 1,168 / 1,220** = **71 / 1,303 / 1,361** (the `all`
row). The entire gap between 1,303 and 206 is this branch's own **8** band-carrying artefacts
(**8 files / 1,097 lines / 1,147 occurrences**): `_occurrences_v1_dash_any_v2_incl_ignored.json`
(639/679), `sites.json` (253/257), `_occurrences_v1_dash_any.json` (177/181, the LEGACY dump),
`report.md` (11/11), `_build_sites.py` (7/7), `coverage_gap_sites.json` (4/4),
`refutation/refutation.json` (3/4), `refutation/_write_refutation.py` (3/4) — they carry the band
because they *quote* it. **This is why the `all` row must never be quoted as a measurement.**
(`_census.py` and `refutation/refutation.md` are also listed in `per_file`, but only under the
looser variants — they carry no `v1_dash_any` match.)

**`paper/` re-confirmed on the final run:** of the 146 band-carrying files in `per_file`, **0** are
under `paper/`. The §7.3 verdict (no band string anywhere in `paper/**`) holds on the corrected
population, which now includes the 11 gitignored `paper/` files the legacy census could not see.

*§7.7 appended 2026-11-01, 19:58 local. All numbers MEASURED; no number is PROJECTED.*

---

## 8. Wf-A190 (2026-11-01) — LOCATOR DEMOTION: the record is cited by JSON PATH, not by line number

**The rule.** A locator into `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`
(or into its sibling rendering `frontier_matrix.md`) is durable only as a **JSON PATH**. The record is
regenerated and keeps growing, so **every line number into it is STALE BY CONSTRUCTION** the moment
the record is re-emitted. Every LIVE locator in this report has been converted to a JSON PATH; every
superseded line-number rendering is preserved **VERBATIM** in §8.1 below (AGENTS.md §5.5 rule #6:
append-only + dated, superseded rendering preserved verbatim, never a silent edit, never a deletion).

**MEASURED on 2026-11-01:** the record was **1,136 lines / 108,408 bytes** at the time of the
measurement. **Every line number below was measured on 2026-11-01 and is STALE BY CONSTRUCTION.**

### 8.1 Superseded renderings — VERBATIM (superseded 2026-11-01)

| # | superseded rendering (VERBATIM) | where it was | durable replacement |
|---|---|---|---|
| 1 | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:68` | report.md L113, L391, L395, L406 | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.extra_observations[0]` |
| 2 | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58,67` | report.md L163 | `... -> band_provenance.endpoint_candidates[0].candidate_a` (92.7 % CANDIDATE: GeoLDM Val&Uniq) and `... -> band_provenance.endpoint_candidates[1].candidate_a` (94 % CANDIDATE: HLTF PB-Valid) |
| 3 | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58,67,80` | report.md L194 | the three candidates: `...endpoint_candidates[0].candidate_a`, `...endpoint_candidates[1].candidate_a`, `...endpoint_candidates[0].candidate_b` (YuelBond per-bond-type F1) |
| 4 | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:353` | report.md L260 | `... -> band_provenance.band_as_quoted_in_the_repo` |

### 8.2 What each cited line ACTUALLY contained on 2026-11-01 — MEASURED

CLAIMED-vs-ACTUAL, read off the record on disk on 2026-11-01 (line numbers = the record's own lines):

| cited line | what the citing sentence claimed it held | what it ACTUALLY held | verdict |
|---|---|---|---|
| `frontier_matrix.json:58` | an endpoint candidate carrying `"post_processed": false` | `band_provenance.cross_references[1]` — a string naming `sources['HLTF'].cells['METRIC'].note` | **WRONG** |
| `frontier_matrix.json:67` | an endpoint candidate carrying `"post_processed": false` | `band_provenance.readings[0].current_line_numbers_on_disk` | **WRONG** |
| `frontier_matrix.json:80` | the third candidate (YuelBond F1) | `band_provenance.readings[1].current_line_numbers_on_disk` | **WRONG** |
| `frontier_matrix.md:68` | the only correct-and-per-cell `post` site | renders `sources.EF-TALFM.extra_observations[0]` | RIGHT, but perishable |
| `frontier_matrix.md:353` | "the repo's" band | renders `band_provenance.band_as_quoted_in_the_repo` | RIGHT, but perishable |

The record's own `band_provenance.line_number_note` says the same thing in the record's voice: the
`':51-89'` and `':360, :378-379'` ranges it carries are "the PRE-Wf-A177 line numbers" — the record
itself carries a demotion note.

### 8.3 The FINDING survives the demotion (this is what the line numbers were carrying)

The substantive claim is unchanged and now sits on durable paths: **both band endpoints carry
`"post_processed": false`** — the 92.7 % CANDIDATE (`band_provenance.endpoint_candidates[0].candidate_a`,
GeoLDM Val&Uniq) and the 94 % CANDIDATE (`band_provenance.endpoint_candidates[1].candidate_a`, HLTF
PB-Valid) — **and so does the third candidate**, YuelBond's per-bond-type F1
(`band_provenance.endpoint_candidates[0].candidate_b`). The three candidates are a 2-element list
(`endpoint_candidates[0]`, `endpoint_candidates[1]`); **there is no `candidate_c`** — the third
candidate for the same numeral is `endpoint_candidates[0].candidate_b`, a *different quantity*
(F1, not a validity rate).

**Shape of the record, for anyone who wants to cite it without a line number:**
`band_provenance` (the object) → `band_provenance.band_as_quoted_in_the_repo` →
`band_provenance.endpoint_candidates[0].candidate_a` / `[0].candidate_b` /
`band_provenance.endpoint_candidates[1].candidate_a` → `band_provenance.headline_finding` →
`band_provenance.headline_finding_superseded_2026_11_01` →
`sources.EF-TALFM.extra_observations[0]` → `sources.EF-TALFM.cells.POST-PROCESSING.note` →
`sources.HLTF.cells.METRIC.note_superseded_2026_11_01` → `sources.HLTF.citation`.

*§8 appended 2026-11-01 (Wf-A190). All numbers MEASURED on 2026-11-01; the line numbers above are
STALE BY CONSTRUCTION by design.*


## 9. Wf-A190 (2026-11-02) — LOCATOR DEMOTION II: the catalogue and the three regenerated metric artefacts

**MEASURED 2026-11-02. Every line number below is STALE BY CONSTRUCTION.**

The record was not the only growing artefact this branch cites. `docs/HONEST_NEGATIVE_CATALOG.md` is
**APPEND-ONLY** and grows every wave (21,688 lines on 2026-11-02); the three
`metrics/by_metric/wfa155_*.json` artefacts are **REGENERATED** by their producers (re-written
2026-11-01 21:13 / 21:58); and this report grew when §7 and §8 were appended. The LIVE locator is
therefore a **SECTION ANCHOR** (markdown) or a **JSON PATH** (JSON). Nothing is deleted
(AGENTS.md §5.5 rule #6): every demoted line number survives VERBATIM below.

### 9.1 Superseded renderings — VERBATIM (superseded 2026-11-02)

| # | superseded rendering (VERBATIM) | where it was | durable replacement |
|---|---|---|---|
| 1 | `docs/HONEST_NEGATIVE_CATALOG.md:20288` | §3.1 (the "NOT our band" quote) | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264` |
| 2 | `docs/HONEST_NEGATIVE_CATALOG.md:20817` | §3.2 lead-in (the 4 dash sites) | the DENOMINATOR row anchor |
| 3 | `metrics/by_metric/wfa155_ligand_connected_rate.json:228` | §3.2 lead-in | `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[0].frontier` |
| 4 | `metrics/by_metric/wfa155_metal_coordination_sphere.json:26` | §3.2 lead-in | `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.summary` |
| 5 | `metrics/by_metric/wfa155_posebusters_validity.json:608` | §3.2 lead-in | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.statement` |
| 6 | `docs/HONEST_NEGATIVE_CATALOG.md:20819` | §3.2 table, row 1 | the POST-PROCESSING row anchor |
| 7 | `metrics/by_metric/wfa155_ligand_connected_rate.json:238` | §3.2 table, row 2 | `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` |
| 8 | `metrics/by_metric/wfa155_metal_coordination_sphere.json:29` | §3.2 table, row 3 | `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing_superseded_2026_11_01_wfa176.rendering` |
| 9 | `metrics/by_metric/wfa155_posebusters_validity.json:596` | §3.2 table, row 4 | `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.superseded_reading_2026_11_01.superseded[0].text` |
| 10 | `molmetal/scripts/wf_a155_ligand_connected_rate.py:223` | §3.2 table, row 5 | the source file + the artefact field it emits (the generator is REPAIRED) |
| 11 | `molmetal/scripts/wf_a155_posebusters_validity.py:489` | §3.2 table, row 6 | the source file + the artefact field it emits (the generator is REPAIRED) |
| 12 | `docs/HONEST_NEGATIVE_CATALOG.md:18670-18674` | §3.2 root-cause paragraph | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252 -> subsection 'The published band, and the denominator trap'` |
| 13 | `metrics/by_metric/wfa155_posebusters_validity.json:554`, `:555`, `:558`, `:561` | §3.3 table | the four `comparability_vs_frontier_band.frontier_band` JSON paths |
| 15 | `molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464` | §3.2's third framing; `refutation.json` -> `claims[3].evidence_paths[10]` | the file -> the emitter of `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing` |
| 16 | `molmetal/scripts/wf_a155_ligand_connected_rate.py:223` | `refutation.json` -> `claims[3].evidence_paths[8]` | the file -> the emitter of `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` |
| 17 | `molmetal/scripts/wf_a155_posebusters_validity.py:489` | `refutation.json` -> `claims[3].evidence_paths[9]` | the file -> the emitter of `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.breaks[2].frontier` |
| 18 | `molmetal/scripts/wfa161_frontier_labelling.py:934` and `molmetal/reports/wfa161_frontier_labelling/report.md:139`; `molmetal/scripts/wfa158_frontier_postproc_matrix.py:456` | `refutation.json` -> `claims[2].claim`, `claims[2].evidence_paths[0..1]`; this report §3.5 | the file + the quoted sentence / the emitter of `band_provenance.band_as_quoted_in_the_repo` |

### 9.2 What the demoted line numbers resolve to on 2026-11-02 — MEASURED

| demoted line | what it holds on 2026-11-02 |
|---|---|
| catalogue `20288` | STILL EXACT — the `## §6.264 — THE CENSUS WAS **REPUBLISHED A THIRD TIME** …` heading (append-only, so earlier lines did not move) |
| catalogue `20817` / `20819` | STILL EXACT — the DENOMINATOR and POST-PROCESSING rows of the four-breaks table |
| catalogue `18670-18674` | STILL EXACT — the §6.252 band block |
| ligand `:228` | STILL EXACT — `comparability.breaks[0].frontier` (the DENOMINATOR break) |
| ligand `:238` | same FIELD, TEXT CHANGED — `comparability.breaks[2].frontier` now reads the two-readings rendering; the blanket sentence is GONE from that artefact entirely |
| metal_coord `:26` | STALE — `"n_rows_analyzed": 2112,`; the cited `comparability_statement.summary` is at L38 |
| metal_coord `:29` | STALE — `"single_fragment_rate_percent": 25.05`; the cited leaf is at L68; the blanket rendering survives ONLY inside `break_3_post_processing_superseded_2026_11_01_wfa176.rendering` (L86) |
| posebusters `:554` / `:555` | STALE — `"n_pb_valid": 29,` / `"n_scored": 1472,`; the values 92.7 / 94.0 are at L576 / L577 |
| posebusters `:558` / `:561` | STALE — `"n_unparseable": 0,` / `"strict_denominator_definition"`; the values 92.98 / 94.33 are at L580 / L583 |
| posebusters `:596` | STALE — `"dataset": "GEOM-Drugs with explicit hydrogens"`; the live break is `comparability_vs_frontier_band.breaks[2].frontier` (L621); the blanket rendering survives ONLY inside `superseded_reading_2026_11_01.superseded[0].text` (L642) |
| posebusters `:608` | STALE — the `ours` cell of `breaks[0]`; the cited `comparability_vs_frontier_band.statement` is at L633 |
| ligand generator `:223` | STALE — L223 is now `'why_superseded': "the superseded rendering asserted ONE provenance reading as settled: ...`; the quoted string survives ONLY at L293, inside a commented-out superseded block |
| posebusters generator `:489` | STALE — L489 is now `for label, count in sorted(epoch_counts.items(), key=lambda kv: str(kv[0]))`; the quoted string survives ONLY at L611 (commented) and at L1545 (the emitter's own `superseded` list) |
| metal_coord generator `:456` | STALE — L456 is now `#: artefact instead of silently DROPPING the dated history.`; the quoted string survives ONLY at L592, inside a comment |

### 9.2b SOURCE-FILE CITATIONS THAT MEASURED DRIFTED — the cited line no longer holds the quoted content

These are NOT the locator class (a source file is not regenerated), but a live claim that points at a
line that no longer carries the quoted content is a live defect. Each was re-measured on 2026-11-02
and each was repaired in place, with the superseded rendering preserved in §9.1.

| superseded rendering (VERBATIM) | where it was | what the cited line holds on 2026-11-02 | durable replacement |
|---|---|---|---|
| `molmetal/scripts/wfa161_frontier_labelling.py:934` | `refutation.json` -> `claims[2].claim` and `claims[2].evidence_paths[0]` | `validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,` — the quoted sentence is at L913 | the file + the quoted sentence: "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing, so it is not on the frontier's scale." |
| `molmetal/reports/wfa161_frontier_labelling/report.md:139` | `refutation.json` -> `claims[2].claim` and `claims[2].evidence_paths[1]` | the X-class entry "**Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-" — the quoted sentence is at L158 | the file + the quoted sentence (same string as above) |
| `molmetal/scripts/wfa158_frontier_postproc_matrix.py:456` | this report, §3.5 | `"bytes": 468625,` | `molmetal/scripts/wfa158_frontier_postproc_matrix.py` -> the emitter of `band_provenance.band_as_quoted_in_the_repo` (emitted at L1088) |
| `molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464` | this report, §3.2; `refutation.json` -> `claims[3].evidence_paths[10]` | L452 and L456 are Wf-A189 comments; L464 is the DATED superseded `rendering`; the quoted string survives only at L592, inside a comment; the LIVE `break_3_post_processing` is emitted at L634 / L789 | the file -> the emitter of `metrics/by_metric/wfa155_metal_coordination_sphere.json -> comparability_statement.break_3_post_processing` |
| `molmetal/scripts/wf_a155_ligand_connected_rate.py:223` | `refutation.json` -> `claims[3].evidence_paths[8]` | `'why_superseded': "the superseded rendering asserted ONE provenance reading as settled: ...` | the file -> the emitter of `metrics/by_metric/wfa155_ligand_connected_rate.json -> comparability.breaks[2].frontier` |
| `molmetal/scripts/wf_a155_posebusters_validity.py:489` | `refutation.json` -> `claims[3].evidence_paths[9]` | `for label, count in sorted(epoch_counts.items(), key=lambda kv: str(kv[0]))` | the file -> the emitter of `metrics/by_metric/wfa155_posebusters_validity.json -> comparability_vs_frontier_band.breaks[2].frontier` |

**ONE CLAIM IN THIS REPORT WAS FALSE ON 2026-11-02 AND IS CORRECTED IN PLACE.** §3.2 said "The
generator still carries the wrong text; the artifact on disk carries an older wrong text; neither is
realised as correct." MEASURED: the artefact and the generator now carry the SAME text
(`comparability_statement.break_3_post_processing` in the artefact = the generator's emitted
`break_3_post_processing`), so the "older wrong text" half is false. The generator text is STILL
wrong — but for a different reason than §3.2 states: it now says "HLTF arXiv:2603.27113v1 Table 1(b)
reports six models RAW and POST-PROCESSED side by side", and Table 1(b) is QM9 with NO `+PP` rows
(the `+PP` rows are Table 2, GEOM-DRUGS). That is the same Table-1-vs-Table-2 confusion **Wf-A186
already refuted from the PRIMARY SOURCE**
(`molmetal/reports/wfa186_frontier_primary_source/report.md -> section 7`). `metrics/**` is outside
this wave's write scope, so it is REPORTED here, not repaired.

### 9.3 ONE CORRECTION — the section number of the four-breaks table

This branch says the table is under `§6.264`; it is under **`§6.265`**. MEASURED on 2026-11-02:
`## §6.264 —` is at L20288 (the census-republish wave) and the four-breaks subsection heading
`### 4. THE PUBLISHED 92.7–94 % BAND IS **NOT OUR BAND** — **FOUR** INDEPENDENT BREAKS` is at
L20808, inside `## §6.265 —` at L20625. The superseded rendering
`docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS'`
is preserved VERBATIM in `coverage_gap_sites.json` ->
`locator_demotion_2026_11_02_stale_by_construction` -> `corrections[0]`, and in the sibling Wf-A186
artefact `molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` ->
`locator_demotion_2026_11_02_stale_by_construction` -> `corrections[0]`.

### 9.4 WHAT WAS LEFT ALONE, AND WHY

- **§7 (CORRECTION — 2026-11-01) and §8 (LOCATOR DEMOTION — 2026-11-01)** are DATED historical
  records; their line numbers are preserved VERBATIM and are not live claims.
- **`molmetal/reports/wfa175_band_provenance/sites.json`** is a CENSUS DUMP produced by
  `_build_sites.py`; every `line` in it is the census's own datum, not a citation. It and its
  producer are left unchanged so the census stays reproducible.
- **`coverage_gap_sites.json`'s `path` / `line_measured_2026_11_01` / `text`** are the census datum
  — the evidence that the defect was found — and are preserved VERBATIM; its live claims (`block`,
  `claim`, `note`) were repaired and each site gained a `durable_locator`.
- **Source-file citations** (`molmetal/scripts/**`, other waves' reports, `paper/**`) keep the
  repo's `file:line` form; they are not regenerated artefacts. Where the cited line was MEASURED
  to no longer hold the quoted content, the row now says so and §9.2 records the drift.
- **`AGENTS.md` / `TODO/` citations** are left as they are (outside this wave's write scope).

## 10. Wf-A190 (2026-11-02) — THE CORPUS-WIDE VERBATIM PRESERVATION LEDGER

**MEASURED 2026-11-02. APPEND-ONLY (AGENTS.md §5.5 rule #6) — nothing above this line changed.**

This section is the SECOND, independent copy of every superseded rendering recorded in the
dated demotion blocks of the JSON artefacts this branch owns (plus the sibling Wf-A186 artefact).
A dated demotion block that holds the ONLY copy of a superseded rendering is not verifiable: a
paraphrase, a re-typing, or a deletion inside that block would pass unnoticed. The ledger below
is byte-copied from the `text` field of each dated entry, so any edit on either side breaks the
agreement, and
`molmetal/tests/test_wfa190_locator_class.py::test_t2_dated_demotions_preserve_their_superseded_strings_verbatim`
fails.

Every entry below was copied programmatically from the named JSON path; none was re-typed:

- `molmetal/reports/wfa175_band_provenance/coverage_gap_sites.json` -> `locator_demotion_2026_11_01_stale_by_construction` -> `superseded_strings_verbatim`
- `molmetal/reports/wfa175_band_provenance/coverage_gap_sites.json` -> `locator_demotion_2026_11_02_stale_by_construction` -> `superseded_strings_verbatim`
- `molmetal/reports/wfa175_band_provenance/refutation/refutation.json` -> `locator_demotion_2026_11_01_stale_by_construction` -> `superseded_strings_verbatim`
- `molmetal/reports/wfa175_band_provenance/refutation/refutation.json` -> `locator_demotion_2026_11_02_stale_by_construction` -> `superseded_strings_verbatim`
- `molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `locator_demotion_2026_11_02_stale_by_construction` -> `superseded_strings_verbatim`

```text

=== molmetal/reports/wfa175_band_provenance/coverage_gap_sites.json -> locator_demotion_2026_11_01_stale_by_construction
--- where: sites[0].refuting_evidence[0]
molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58,67 -- /band_provenance/endpoint_candidates -> BOTH band endpoints carry "post_processed": false
--- where: sites[1..5].refuting_evidence[0] (5 sites: metrics/by_metric/wfa155_ligand_connected_rate.json:238, metrics/by_metric/wfa155_metal_coordination_sphere.json:29, metrics/by_metric/wfa155_posebusters_validity.json:596, molmetal/scripts/wf_a155_ligand_connected_rate.py:223, molmetal/scripts/wf_a155_posebusters_validity.py:489)
frontier_matrix.json:58,67

=== molmetal/reports/wfa175_band_provenance/coverage_gap_sites.json -> locator_demotion_2026_11_02_stale_by_construction
--- where: sites[0].block
S6.264 S4 four-break table (the same table whose DENOMINATOR row carries the band string at L20817)
--- where: sites[0].note
Likely inherited from S6.252 (L18670-18674), which asserts the SAME blanket sentence for a DIFFERENT band: 'The published connected-rate band is 37.4 % ... to 96.0 %'. Two different bands are in circulation.
--- where: sites[1].claim
blanket POST-PROCESSED, for the 37.4-96.0 % band of L18670-18671
--- where: sites[5].note
This is the GENERATOR of metrics/by_metric/wfa155_ligand_connected_rate.json:238 -- repairing the artefact without repairing this line would re-introduce the defect.
--- where: sites[6].note
GENERATOR of metrics/by_metric/wfa155_posebusters_validity.json:596.
--- where: sites[7].note
A THIRD framing of the RAW/POST axis: not 'always post-processed', not 'always raw', but 'arm-dependent' -- and for THIS band it is wrong for the same reason. NOT yet realised on disk (the artefact is stale), so this is a live generator defect.

=== molmetal/reports/wfa175_band_provenance/refutation/refutation.json -> locator_demotion_2026_11_01_stale_by_construction
--- where: claims[1].counter_evidence[1]
only ONE (frontier_matrix.md:68) is correct-and-per-cell
--- where: claims[2].claim
'DIFFERENT post-processing' refuted by frontier_matrix.json:58,67 post_processed:false
--- where: claims[2].counter_evidence[0]
CITATION SLIP: the two BAND endpoints' post_processed:false are at frontier_matrix.json:58 (92.7 % = GeoLDM Val&Uniq, 'endpoint': '92.7 %') and :80 (94 % = HLTF PB-Valid, 'endpoint': '94 %'). Line :67 is candidate_b of the 92.7 % endpoint = the YUELBOND per-bond-type F1 -- a DIFFERENT quantity. The branch's 3.2 (58,67,80) is right; its 3.1 -- and the one-line claim -- cite '58,67' for 'both band endpoints', which is wrong. The conclusion is unaffected (all three are false).
--- where: claims[2].counter_evidence[1]
frontier_matrix.md:353
--- where: claims[2].evidence_paths[2]
molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:52-88
--- where: claims[3].claim
refuted by frontier_matrix.json:58,67,80 and synthesis.md:310
--- where: claims[3].what_i_checked[3]
Refuting evidence verified: frontier_matrix.json:58/:67/:80 all 'post_processed': false
--- where: refutation.md, per-claim table row 2
only `frontier_matrix.md:68` is correct
--- where: refutation.md, per-claim table row 3
Citation slip: the two band endpoints' `post_processed:false` are at `frontier_matrix.json:58` (92.7 %) and **:80** (94 %); **:67 is the YuelBond F1 candidate**, a different quantity.

=== molmetal/reports/wfa175_band_provenance/refutation/refutation.json -> locator_demotion_2026_11_02_stale_by_construction
--- where: claims[0].evidence_paths[0]
docs/HONEST_NEGATIVE_CATALOG.md:21287-21295
--- where: claims[1].evidence_paths[1]
molmetal/reports/wfa175_band_provenance/report.md:86-123
--- where: claims[3].claim
catalog:20819
--- where: claims[3].evidence_paths[0]
docs/HONEST_NEGATIVE_CATALOG.md:20817,20819
--- where: claims[3].evidence_paths[1]
metrics/by_metric/wfa155_ligand_connected_rate.json:228,238
--- where: claims[3].evidence_paths[2]
metrics/by_metric/wfa155_metal_coordination_sphere.json:26,29
--- where: claims[3].evidence_paths[3]
metrics/by_metric/wfa155_posebusters_validity.json:596,608
--- where: claims[5].claim
metrics/by_metric/wfa155_posebusters_validity.json:554,555,558,561 (band_connected_pct 92.7 / 94.0 and EF-TALFM 92.98 / 94.33)
--- where: claims[5].evidence_paths[0]
metrics/by_metric/wfa155_posebusters_validity.json:552-561
--- where: refutation.md, per-claim table row 6
`wfa155_posebusters_validity.json:554,555,558,561` = 92.7 / 94.0 / 92.98 / 94.33, field names exact
--- where: claims[2].claim
The band is asserted as OURS with refuted provenance at wfa161_frontier_labelling.py:934 and wfa161_frontier_labelling/report.md:139;
--- where: claims[2].evidence_paths[0]
molmetal/scripts/wfa161_frontier_labelling.py:934
--- where: claims[2].evidence_paths[1]
molmetal/reports/wfa161_frontier_labelling/report.md:139
--- where: claims[3].evidence_paths[8]
molmetal/scripts/wf_a155_ligand_connected_rate.py:223
--- where: claims[3].evidence_paths[9]
molmetal/scripts/wf_a155_posebusters_validity.py:489
--- where: claims[3].evidence_paths[10]
molmetal/scripts/wf_a155_metal_coordination_sphere.py:452-464

=== molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> locator_demotion_2026_11_02_stale_by_construction
--- where: retrieval[1].authors_note
the repo cites the second author as 'Alexander Arjun Lobo' (molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:779); the paper's own \author line reads 'Alexander Lobo'.
--- where: x_class_status.X1.live_status
SURVIVES ONLY IN DATED SUPERSEDED KEYS (correct per AGENTS.md 5.5 rule #6): molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:145 (band_provenance.headline_finding_superseded_2026_11_01) and :763-764 (sources.HLTF.cells.METRIC.note_superseded_2026_11_01); the LIVE band_provenance.headline_finding is READING-QUALIFIED and no longer carries it. The repo's own X-class list already marks it FORBIDDEN (molmetal/reports/wfa161_frontier_labelling/report.md:314).
--- where: x_class_status.X2.live_status
ONE LIVE SURFACE REMAINS: docs/HONEST_NEGATIVE_CATALOG.md:20819 (the class-(a) POST-PROCESSING row of the 92.7-94 % band's four-breaks table). The metric artefact metrics/by_metric/wfa155_metal_coordination_sphere.json carries the axis only in reading-conditional form ('whether the band as a whole is RAW is NOT settled on disk'), and the two generator scripts carry it only in comments. The repo already refuted it from its OWN record (frontier_matrix.json -> band_provenance -> endpoint_candidates, post_processed: false); this wave REFUTES IT FROM THE PRIMARY SOURCE (F2).
--- where: findings.F2.consequence
HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS ONLY: the QM9 sentence in Section 4.1 (sections/experiments.tex:8) carries NO post-processing clause, while the immediately following GEOM-DRUGS sentence reads 'On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)'. The QM9 Table 1 carries no '+PP' rows at all; the '+PP' rows live in Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main), whose caption declares that 'PP' denotes standardized post-processing. Both of the band's Reading-A endpoints -- GeoLDM Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 cells, so they carry NO post-processing. CONSEQUENCE: X2 is REFUTED BY THE PRIMARY SOURCE for the 92.7-94 % band. The LIVE catalogue row docs/HONEST_NEGATIVE_CATALOG.md:20819 -- the class-(a) POST-PROCESSING row of the four-breaks table -- asserts 'the band is measured on the published post-processed stream' and is WRONG for THIS band. It was inherited from a DIFFERENT band: catalogue section 6.252 (line 18603), whose band is 37.4 % to 96.0 % and whose frontier numbers ARE post-processed (Buttenschoen et al. arXiv:2505.00518 raw -> post-processed pairs). CAVEAT (not a refutation): HLTF also applies an OPTIONAL valence repair that DELETES low-confidence incident bonds; the paper states 'We report metrics both with and without this repair' and does not mark which arm Table 1 carries. That repair deletes BONDS, not components, so it is not the largest-fragment retention the band question is about.
--- where: invariants.competing_readings[2]
Reading C (docs/HONEST_NEGATIVE_CATALOG.md:21273): the band is a composite of the EF-TALFM cells 92.98 % / 94.33 % / 93.13 %, which have three different denominators. Also does NOT reproduce the quoted numerals. This THIRD reading is NOT among the two recorded in frontier_matrix.json -- the contest is three-way, not two-way.
--- where: our_side.catalogue_quote.path
docs/HONEST_NEGATIVE_CATALOG.md:18680-18684
--- where: catalogue_row.inherited_from.band_locator
docs/HONEST_NEGATIVE_CATALOG.md:18671
--- where: catalogue_row.inherited_from.may_may_not_block
docs/HONEST_NEGATIVE_CATALOG.md:18678-18688
```

**41 dated entries, each preserved VERBATIM in this ledger and in its own dated block.**

**THIS LEDGER IS A FROZEN, DATED SNAPSHOT — DO NOT REGENERATE IT.** It was produced once, on
2026-11-02, by a one-off script that read the three JSON artefacts named above and copied each
`text` field byte-for-byte; no generator is checked in, deliberately: a ledger that could be
regenerated would bless a paraphrase of a superseded record, which is precisely what the test
below it exists to catch. A later wave that adds a dated entry must add its OWN dated ledger
section rather than re-running this one.
