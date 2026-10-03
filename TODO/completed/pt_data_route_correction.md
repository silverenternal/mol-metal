# Pt data route correction — killing the false "no Pt + real labels" honest-negative

**Date:** 2026-10-03 · **Task:** workflow task "repair the loader / kill the false honest-negative"
**Scope:** CPU + network only. No GPU, no training, no paper/* (§5.3), no wet-lab (§5.4/§10). Nothing committed.

## §1 What the project believed, and why it was wrong

The repo carried a false honest-negative: "per-cell-line IC50 values are NOT publicly
exposed by the MB-Finder API". Old wording, verbatim:

- `molmetal/scripts/platinai_real_label_loader.py` docstring (pre-fix):
  `(per-cell-line IC50 endpoint NOT publicly exposed - honest-negative below)`
- same file honest-framing block (pre-fix):
  `Per-cell-line IC50 NUMERIC VALUES are NOT exposed by the public API (only the per-cell-line coverage MATRIX). We therefore emit has_<line> boolean flags (real coverage) instead of synthetic pIC50.`
- the same sentence was written into every coverage-mode `*.meta.json` `framing` field.
- downstream reports repeated it (full list in §5).

**Why it was believed:** Wf-292D's ChemRxiv scrape hit Cloudflare 403 (chemrxiv.org is
CF-protected), and the 2026-09-29 endpoint probe list covered only get-substances /
get-count / get-tissues / get-cell-lines / get-references. `get-cell-lines` returns a
coverage COUNT matrix, so that wave concluded numeric values were not public. It never
probed `get-ceil-line-bio-data`. The conclusion was wrong from the day it was written
(2026-09-29); it was first corrected 2026-10-03.

## §2 What is actually true (all MEASURED live, 2026-10-03)

Endpoint: `POST https://api.mb-finder.org/api/v2/get-ceil-line-bio-data` (server
spelling is "ceil"), body `{"cell_line": "A2780cis", "per_page": 5, "page": N}`.
Public, no auth; api.mb-finder.org answers plain JSON 200 (not CF-protected).
per_page is server-capped at 5; `meta.last_page = 256` for A2780cis (MEASURED).

MEASURED value examples (values are strings; BOTH forms occur):

| cell line | code | value | doi |
|---|---|---|---|
| A2780cis | MB-Pt-000017 | `1.2` | 10.1021/ic8012359 |
| A2780cis | MB-Pt-000132 | `0.6 ± 0.1` | 10.1021/tx200525n |
| A2780cis | MB-Pt-000169 | `70 ± 10` | 10.1021/jm061435l |
| A2780 | MB-Pt-000092 | `0.0046 ± 0.0023` | 10.1039/d0dt04135a |

**Join verified live (not assumed):** code = basename(`compounds.imageUrl`) minus
`.svg`/query; it equals get-substances `imageId` for real rows. MB-Pt-000017 joins to
substance id 13, `smiles = CC(C)=[NH][Pt]1([P](c2ccccc2)...` (contains `[Pt]`), mw
648.67, substances page 1545.

**End-to-end smoke (new CLI mode, real API, 5 requests):** A2780cis pages 1-2 +
substances pages 1545-1547 →
`rows=10 reqs=2 parsed_numeric=10 parse_failures=0; index_size=15; matched=3 unmatched=7`.
CSV `.data/molmetal/pt_biodata_smoke_a2780cis.csv`: 10 real measured rows; 3 joined to
SMILES, 7 honestly unmatched (codes outside the 3-page index window).

## §3 Files changed (working tree; not committed)

1. `molmetal/scripts/platinai_real_label_loader.py` (831 lines; +375/-14 per git diff):
   - docstring: new endpoint documented + CORRECTION block (old line kept as a quote;
     states what was wrong and that it was wrong since 2026-09-29);
   - honest-framing section corrected, old sentence quoted, two modes documented;
   - coverage-mode meta `framing` corrected;
   - NEW: `compound_code_from_image_url`, `parse_measurement_value` (both forms),
     `fetch_cell_line_biodata`, `flatten_biodata_row`, `fetch_cell_line_biodata_pages`
     (paginator reusing `_post` retry/429 backoff), `build_structure_index` (reuses the
     existing `fetch_pt_compound_page` paging helper), `join_structures`,
     `write_biodata_outputs`;
   - NEW CLI `--biodata-cell-line` (+ `--biodata-max-pages`, `--substances-start-page`,
     `--substances-max-pages`) → `_run_biodata_mode`;
   - usage docstring venv path fixed to `.venv/bin/python` (§5.9).
2. `molmetal/tests/test_platinai_real_label_loader_biodata.py` (147 lines, NEW): 5 tests.
3. `molmetal/tests/fixtures/mbfinder_a2780cis_biodata.json` (15,584 bytes, NEW): recorded
   live pages (10 bio-data rows + 15 substance rows).
4. This closeout.

## §4 Test + regression proof

`./.venv/bin/python -m pytest molmetal/tests/test_platinai_real_label_loader_biodata.py -q`
→ `5 passed` (0.3 s; fully offline — monkeypatched `_post`, fixture only; no network).

**Proof it fails when the endpoint breaks:** temporarily changed `BIODATA_ENDPOINT` to
the common typo `"get-cell-line-bio-data"` →

```
FAILED ...::test_biodata_pagination_offline
FAILED ...::test_structure_join_offline
FAILED ...::test_biodata_mode_writes_outputs_offline
3 failed, 2 passed   (pagination: >assert requests == 2 / E assert 1 == 2)
```

Restored → `5 passed`. The 2 tests that still pass are pure parser / join-key unit
tests (no endpoint); the 3 wire-level tests all fail.

## §5 Repo-wide grep for the false phrase (MEASURED: grep -rniE, 2026-10-03)

Patterns: not publicly exposed | not exposed by the public | not exposed by any of
these 5 | only exposes coverage matrix | only expose the coverage | never the IC50.
Total: 10 hits in 6 files.

**Live code — FIXED (this task):** loader; its 2 remaining matches (lines 32, 66) are
inside the CORRECTION blocks quoting the old text on purpose (audit trail).

**Live code — NOT fixed (not owned by this task; follow-up needed):**
- `molmetal/scripts/platinai_selenium_loader.py:12-13` ("the API does NOT expose
  per-cell-line IC50 NUMERIC values — only a presence/absence matrix") and `:477`
  ("Wf-295B Strategy S7 (MB-Finder portal) only exposes coverage matrix, not IC50 NUMERIC").

**Historical reports — REPORTED, not edited (§5.5):**
- `molmetal/reports/wf295b_platinai_loader/loader_report.md:51,83,87`
- `molmetal/reports/wf300b_selenium_prep/plan.md:15`
- `molmetal/reports/wf301c_platinai_stratified/stratified_report.md:84,90`
- `molmetal/reports/wf304b_selenium/selenium_report.md:57`

**Adjacent / superseded (report only):**
- `docs/HONEST_NEGATIVE_CATALOG.md` §2.10 + `TODO/completed/207_unified_pre_flight_checklist_v2.md`
  §9.AN (Wf-304B Selenium IC50 NUMERIC = 0/0/0/0): the Selenium block was a real event,
  but both entries imply the numeric route is blocked — superseded by this correction.
- `molmetal/reports/wf299d_cross_cohort/plan.md:50` and
  `molmetal/reports/wf299_synthesis/cfm_followup_summary.md:41` (numeric IC50 listed as
  still-to-be-scraped / blocked).
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md:198` (per-compound query
  estimate ~14,900 queries) — superseded by pagination: 256 pages per cell line.

**docs/:** no exact-phrase hits (only the adjacent catalog entry above).

## §6 Second suspected over-claim (OPEN — needs re-verification)

Wf-301C concluded the Pt cohort is effectively A549-only ("ZERO compounds have MCF7,
A2780, or A2780cis bio-data" across 9 sampled pages). MEASURED today: the A2780cis
numeric endpoint alone has 256 pages (~1,280 rows). These cannot both be final. A
cross-check via get-cell-lines (substance id 13) returned HTTP 429 (one-off probe did
not use the loader's backoff) and was STOPPED rather than hammered. Status: OPEN —
resolve during the full crawl. Do not quote either claim as final.

## §7 Disk evidence (§5.5)

Commands actually run (post-fix):

```
$ ls -la molmetal/scripts/platinai_real_label_loader.py \
         molmetal/tests/test_platinai_real_label_loader_biodata.py \
         molmetal/tests/fixtures/mbfinder_a2780cis_biodata.json \
         .data/molmetal/pt_biodata_smoke_a2780cis.csv \
         .data/molmetal/pt_biodata_smoke_a2780cis.meta.json
-rw-r--r-- 1 hugo hugo  2420 Oct  3 09:30 .data/molmetal/pt_biodata_smoke_a2780cis.csv
-rw-r--r-- 1 hugo hugo  1281 Oct  3 09:30 .data/molmetal/pt_biodata_smoke_a2780cis.meta.json
-rw-r--r-- 1 hugo hugo 36148 Oct  3 09:30 molmetal/scripts/platinai_real_label_loader.py
-rw-r--r-- 1 hugo hugo 15584 Oct  3 09:28 molmetal/tests/fixtures/mbfinder_a2780cis_biodata.json
-rw-r--r-- 1 hugo hugo  6252 Oct  3 09:30 molmetal/tests/test_platinai_real_label_loader_biodata.py
$ wc -l molmetal/scripts/platinai_real_label_loader.py molmetal/tests/test_platinai_real_label_loader_biodata.py
  831 molmetal/scripts/platinai_real_label_loader.py
  147 molmetal/tests/test_platinai_real_label_loader_biodata.py
$ git diff --stat molmetal/scripts/platinai_real_label_loader.py
 1 file changed, 375 insertions(+), 14 deletions(-)
```

Closeout self-check (`ls -la TODO/completed/pt_data_route_correction.md`) was run
post-write; result in the task report. Written atomically via
`wf386a3_atomic_write.write_closeout` (§5.6).

## §8 Remaining work (full crawl) — PROJECTED, not measured

- Full per-cell-line crawl: A2780cis `last_page=256` (MEASURED); the other 3 lines are
  unmeasured → PROJECTED ~256 pages each. 4 × 256 = 1,024 requests ≈ 43 min at 2.5 s/req.
- Full Pt structure index: Pt codes sit on ~pages 828..1547 (PROJECTED from sampled
  pages 850 / 1545-1547) → ~720 requests ≈ 30 min.
- Total PROJECTED ~1.5 hr + retry slack, CPU/network only. Then: build the Pt
  multi-cell numeric dataset (value_um + DOI + SMILES) + Bemis-Murcko split.
- Follow-ups: fix the selenium-loader claim; amend historical reports; update the
  HONEST_NEGATIVE_CATALOG §2.10 / §9.AN entries; re-verify the wf301c A549-only claim.

## §9 Honest framing (§5.2)

- MEASURED: endpoint liveness + value examples + join match + smoke counts + grep hit
  counts + test results + file sizes — every number above comes from a command run today.
- PROJECTED: crawl walltimes and page totals beyond A2780cis (labeled).
- OPEN: the wf301c "A549-only" tension (unresolved; 429 stopped the cross-check).
- NOT FIXED here: selenium loader (not owned), historical files (report-only per §5.5).
