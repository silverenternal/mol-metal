# Wf-A171 - Adversarial audit: does the 92.7-94 % band ever leak into an emitted metric?

**VERDICT: DOES NOT LEAK.**

Plain statement: no occurrence of the 92.7-94 % band reaches a published metric's VALUE or HEADLINE
slot, and no occurrence reaches a stdout line that a human or a log-scraper could mistake for one of
our measured results. Every occurrence inside `metrics/by_metric/` sits inside a sub-structure whose
own text says the numbers are NOT comparable - `comparability`, `comparability_statement`,
`comparability_vs_frontier_band`, `what_this_is_not`, `not_comparable_to`. The three metrics' own
measured values are 0.2505 / 0.0284 (ligand), 0.2609 (coordination), 0.0137 / 0.0197 (posebusters).
The band is a cited external reference throughout.

No `patch_if_leak.diff` is staged: step 6 of the brief is conditional on a PROVEN leak, and none was
found.

---

## 1. Census re-measured (UTF-8-safe scanner)

    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan \
      --pattern "92\.7\s*[\u2013\u2014-]\s*94" --limit 20000 --json

MEASURED: `files_with_matches=32 total_matches=74 files_scanned=9967 files_skipped=44
files_non_utf8_text=2 files_binary=252 files_ugrep_I_skipped=254`.

| group (same tool, `--root`) | files | matches |
|---|---|---|
| `molmetal/scripts/` | 8 | 22 |
| `metrics/by_metric/` | 3 | 6 |
| `paper/` | 0 | 0 |
| `registry/` | 0 | 0 |
| `metrics/paper_tables/` | 0 | 0 |

These MATCH the brief exactly - no discrepancy in the brief's numbers.

These counts were MEASURED **before** this branch wrote its own two artifacts. Re-running the same
command afterwards gives **33 files / 78 matches**: the delta is this branch's own `report.md` (4
matches). `band_classification.json` does not appear in that count - see section 2b.

## 2. TWO census blind spots found - the brief UNDERCOUNTS

### 2a. Regex blind spot (double hyphen + numeric list)

    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan \
      --root molmetal/scripts --pattern "92\.7.{0,12}94(\.0)?\s*%|92\.7\s*[-\u2013\u2014,]\s*94" \
      --limit 20000 --json

MEASURED: `files_with_matches=9 total_matches=24`. The extra file is
`molmetal/scripts/wf_a155_posebusters_validity.py` (2 matches), and it is the most important file in
the audit, because the brief's regex cannot match either of its forms:

* `molmetal/scripts/wf_a155_posebusters_validity.py:22` - `92.7--94 %` (DOUBLE hyphen).
* `molmetal/scripts/wf_a155_posebusters_validity.py:150` - `"band_connected_pct": [92.7, 94.0],`
  (a numeric list).

So the brief's scripts census misses 1 file / 2 matches, and the file it misses is the ONE place in
the repo where the band enters a published metric as bare NUMBERS rather than as a string. Over
`metrics/` the broad pattern adds nothing (same 3 files, same 6 matches).

### 2b. .gitignore blind spot (report-tree JSONs)

`.gitignore:158` (`molmetal/reports/**/*.json`) makes the scanner's repo-wide walk SKIP every JSON
under `molmetal/reports/`. Three of the class-(c) sinks traced in this audit are hidden that way;
only their `.md` siblings are visible. Measured with the per-file probe
`wfa156_text_scan.py scan --file <path> --pattern <brief pattern> --json`:

| hidden file | matches |
|---|---|
| `molmetal/reports/wfa156_bond_threshold_frontier/report.json` | 1 |
| `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` | 4 |
| `molmetal/reports/wfa158_population_reconciliation/reconciliation.json` | 3 |

Tool caveat, measured: `wfa156_text_scan.py scan --root molmetal/reports --include-ignored` returns
`files_with_matches=0 total_matches=0` over 19957 scanned files. `--include-ignored` silently
reports ZERO and is NOT a usable workaround; the `--file` probe is.

Impact on the verdict: none - all three are report-tree artifacts, not the published metric tree.

## 3. Classification summary

`molmetal/scripts/` (24 rows: 22 brief-pattern + 2 blind-spot):

| class | count | what it is |
|---|---|---|
| a | 9 | module docstrings / comments describing the citation |
| b | 4 | constants: `FRONTIER_BAND` x3, `FRONTIER_BAND["band_connected_pct"]` |
| c | 11 | strings/numbers written into an emitted JSON or report .md |
| d | **0** | **no occurrence reaches stdout** |

`metrics/by_metric/` (7 rows: 6 brief-pattern + 1 numeric blind spot): a=0, b=1, c=6.

Across all 31 rows: `reaches_published_metric` = 10, `reaches_metric_value_slot` = **0**, class (d) = **0**.

Sinks of the class (c) rows:

* `metrics/by_metric/wfa155_ligand_connected_rate.json` - `comparability/frontier_band` (L248),
  `comparability/breaks[0]/frontier` (L228), `comparability/statement` (L224).
* `metrics/by_metric/wfa155_metal_coordination_sphere.json` - `what_this_is_not/not_a_frontier_comparison`
  (L22), `comparability_statement/summary` (L26).
* `metrics/by_metric/wfa155_posebusters_validity.json` - `comparability_vs_frontier_band/statement`
  (L608) and `comparability_vs_frontier_band/frontier_band/band_connected_pct` = `[92.7, 94.0]` (L553-556).
* `molmetal/reports/**` only: `wfa156_bond_threshold_frontier/report.json`,
  `wfa158_frontier_postproc_matrix/frontier_matrix.json`,
  `wfa158_population_reconciliation/reconciliation.json` + `census.md`,
  `wfa161_frontier_labelling/report.md`, `wfa161_paper_unqualified_numbers/report.md`.

## 4. Classification of the metrics/by_metric/ matches (metric ids)

| metric id | band occurrences | measured value of the metric, or cited reference? |
|---|---|---|
| `wfa155_ligand_connected_rate` | 3 (L224, L228, L248), all under `comparability` | **CITED REFERENCE.** Our measured values: `metal_inclusive.single_fragment_rate` = 0.25047348484848486, `ligand_only.single_fragment_rate` = 0.028409090909090908. |
| `wfa155_metal_coordination_sphere` | 2 (L22, L26), under `what_this_is_not` and `comparability_statement` | **CITED REFERENCE.** Our measured rate: 0.2608695652173913. |
| `wfa155_posebusters_validity` | 1 string (L608) + 1 numeric (L554), under `comparability_vs_frontier_band` | **CITED REFERENCE.** Our measured values: `pb_valid_rate_strict_denominator` = 0.013731060606060606, `pb_valid_rate_scored_denominator` = 0.01970108695652174. |

All three carry `tag: MEASURED`; none has a top-level `value` key.

**Numeric sweep of the WHOLE published metric tree** (every numeric leaf equal to 92.7 / 92.98 /
94.33 / 94.0): 57 numeric hits, of which 53 are unrelated integers (row indices, counts, epochs, atom
counts) and **4 are the band - all four in ONE file**, `metrics/by_metric/wfa155_posebusters_validity.json`,
at `/comparability_vs_frontier_band/frontier_band/band_connected_pct/0` = 92.7,
`.../band_connected_pct/1` = 94.0, `.../ef_talfm_arxiv_2609_08333/connected_pct` = 92.98,
`.../ef_talfm_arxiv_2609_08333/novel_unique_sanitized_pct` = 94.33. Nothing resembling the band
exists anywhere else in `metrics/by_metric/`. `registry/metrics.yaml` has 0 hits for
92.7 / 92.98 / 94.33; `metrics/paper_tables/` (1 JSON) has 0 hits.

## 5. Q3 - the load-bearing trace (can any match reach a published metric or a printed headline?)

**No. PROVEN for every row, by these five independent checks:**

1. **Registry harvest** - `molmetal/scripts/registry.py::_extract_value_basis` returns `(None, None)`
   for all three files (none has a top-level `value`), and `assess_citability` returns
   `quotable=False`, `not_quotable=['no scalar headline']` in BOTH the current untracked state AND a
   simulated-tracked state (`tracked={rel}`) - so even after `git add` the band is not quotable.
   `registry/metrics.yaml` has 0 band matches and 0 rows for these metrics (all three files are
   UNTRACKED, `??`).
2. **Fragmentation census harvest** - `wfa158_population_reconciliation.py::census_q2` yields
   `n_census_rows=1029` over `n_metrics_scanned=582`; 7 rows come from the wfa155 files; **0 rows have
   a value within 1.0 of 92.7**. `classify_key()` returns `None` for `band_connected_pct` and
   `connected_pct`, so 92.7 / 94.0 / 92.98 / 94.33 are dropped before rendering.
3. **Paper and section generators** - `generate_section6.load_failed_metrics` selects 11 metrics, none
   wfa155; `generate_section4`'s `MetricSpec.json_key` list has no wfa155 key;
   `generate_abstract.load_snapshot` uses an explicit 5-key tuple with no wfa155 key;
   `paper_metric_tables.py` uses an explicit 15-file allowlist with no wfa155 file;
   `metrics/paper_tables/` has 0 band matches; `paper/` has 0 band matches.
4. **stdout audit** - an AST walk of `molmetal/scripts/*.py` for `print(...)` and `lines.append(...)`
   whose argument contains the band yields 2 hits, both `lines.append` (aliased `A`) into a report
   `.md` under `molmetal/reports/`. **Zero hits on `print()`; zero hits on any path under `metrics/`.**
   Hence class (d) is empty - no printed headline can be mistaken for our measured result.
5. **Report-tree-only scripts** - `wfa156_bond_threshold_frontier.py` (docstring L91-92, `--out-dir`
   default L1814), `wfa158_frontier_postproc_matrix.py` (L100), `wfa158_population_reconciliation.py`
   (L80), `wfa161_frontier_labelling.py` (L95), `wfa161_paper_unqualified_numbers.py` (L1119-1132) all
   write only under `molmetal/reports/**`. The last one is READ-ONLY on `paper/**` by design.

## 6. Staleness finding (measured; does not change the verdict)

`metrics/by_metric/wfa155_metal_coordination_sphere.json` is **STALE** relative to
`molmetal/scripts/wf_a155_metal_coordination_sphere.py`:

* the script emits `"frontier_reference": FRONTIER_BAND_CITATION` (L2169);
* the on-disk artifact's top-level keys do NOT include `frontier_reference`, `end_to_end_headline` or
  `published_cross_check`;
* the artifact contains no substring `PCQM4Mv2` and no substring `not 92.7-94`.

Consequence: the script-side occurrence at L380 has a PROVEN code sink but the sink is **not
realized on disk yet**. The other two band-bearing artifacts are CURRENT. Impact on the verdict:
none - the stale keys are citation fields, not value slots.

## 7. OPEN ITEMS (not smoothed over)

* **OPEN-1 (medium)** - `molmetal/scripts/wfa161_frontier_labelling.py:934` writes the sentence
  "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel)" into
  `molmetal/reports/wfa161_frontier_labelling/report.md`. "Our" is the AGENTS.md 1.1 ownership shape,
  and "Open Babel" is contradicted by `molmetal/reports/wfa168_frontier_axis_mapping/synthesis.md`
  L115-127 (92.7 % = HLTF `arXiv:2603.27113v1` Table 1(b) "Val&Uniq (%)" GeoLDM cell; 94 % = the same
  table's "PB-Valid (%)" HLTF cell). NOT a metric leak (report tree). Not repaired: the script is
  outside this branch's FILE OWNERSHIP.
* **OPEN-2 (medium)** - the coordination-sphere artifact is stale (section 6). `metrics/**` is
  outside this branch's FILE OWNERSHIP.
* **OPEN-3 (low)** - `band_connected_pct = [92.7, 94.0]` is the only place the band sits in a
  published metric as bare NUMBERS. Protected today by the `frontier_band` key name, the enclosing
  `comparability_vs_frontier_band` block, and `molmetal/tests/test_wfa164_coordination_sphere_narrative.py`
  L354-370 (which asserts no machine number in the coordination artifact equals 92.7 or 94).
* **OPEN-4 (low)** - any band census must sweep by VALUE (92.7, 92.98, 94.0, 94.33) in addition to
  the dash-joined regex; otherwise the numeric form is invisible, as it was here.
* **OPEN-5 (medium)** - `.gitignore:158` hides every `molmetal/reports/**/*.json` from the census,
  and `wfa156_text_scan.py --root <dir> --include-ignored` silently reports 0 matches (section 2b).

## 8. Commands used (all CPU-only, `CUDA_VISIBLE_DEVICES=""`, `.venv/bin/python`)

    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan --pattern "92\.7\s*[\u2013\u2014-]\s*94" --limit 20000 --json
    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan --root molmetal/scripts --pattern "92\.7.{0,12}94(\.0)?\s*%|92\.7\s*[-\u2013\u2014,]\s*94" --limit 20000 --json
    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan --root molmetal/reports --pattern "92\.7\s*[\u2013\u2014-]\s*94" --limit 20000 --json
    CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa156_text_scan.py scan --file <report.json> --pattern "92\.7\s*[\u2013\u2014-]\s*94" --limit 20000 --json
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I   # registry._extract_value_basis / assess_citability
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I   # wfa158_population_reconciliation.census_q2
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I   # generate_section6.load_failed_metrics
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I   # AST stdout audit over molmetal/scripts/*.py
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -I   # numeric sweep of metrics/by_metric/*.json

Machine-readable row-level detail: `band_classification.json` (31 rows).
