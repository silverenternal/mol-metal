# Wf-A38 — PlatinAI FULL harvest follow-up

**Status (2026-10-15):** SHIPPED orchestrator + smoke proof in Wf-A37-5.
This is the follow-up note for the **full** harvest that continues
asynchronously on the same host.

## What was shipped in Wf-A37-5

- `molmetal/scripts/wf_a37_platinai_full_harvest.py` — orchestrator with
  two passes: (1) per-cell-line `get-ceil-line-bio-data` JSONL append,
  (2) `get-substances` SMILES join table. 6-retry exponential backoff
  on every request (`max 60s`), bypass of the dead local `HTTPS_PROXY`
  by passing `proxies={'http': None, 'https': None}` on every call.
- `.data/molmetal/wf_a37_platinai_biodata_smoke_proof/` — smoke proof
  (1 page / cell line) showing the data path is live and the schema
  is correct (real compound_codes, real numeric_value_raw, real DOI).
- `platinum_biodata_full.log` (in Wf-A37-5 commit) — runs from/to
  background process attached to PID 3573482 (alive at commit time).

## What runs in Wf-A38 follow-up

The FULL harvest is launching in background after Wf-A37-5 commits:

```
cd /home/hugo/codes/mol-metal
.venv/bin/python -m molmetal.scripts.wf_a37_platinai_full_harvest \
    --mode full \
    --max-pages 800 \
    --out-dir .data/molmetal/wf_a37_platinai_biodata
```

Expected wall time: **~60-90 minutes** on this flaky-network host
(1861 API calls × ~2.5s nominal × ~1.5x retry overhead).

Per cell line, the upper bound is `last_page × 5` rows (per the
2026-10-15 smoke check the server reports `meta.last_page`):

| Cell line  | last_page | Expected rows | Notes                       |
|------------|-----------|---------------|-----------------------------|
| MCF7       | 1         | ≤5            | primarily empty (no data)   |
| A549       | 518       | ~2590         | biggest cohort              |
| A2780      | 364       | ~1820         |                             |
| A2780cis   | 255       | ~1275         |                             |
| Pt-subst   | ~723      | ~3615 SMILES  | SMILES join via `get-substances` |

Total expected unique compound × cell pairs: **~5690 real IC50 values**
plus a SMILES lookup table of ~3615 Pt substances.

## Why this is not an honest-negative

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2: an honest-negative entry
is the right framing when a metric is BELOW field-avg OR BLOCKED on an
external dependency that cannot be moved.  Here:

- The MB-Finder public API IS reachable from this host (verified live
  2026-10-15; response body has real numeric IC50 values e.g.
  `MB-Pt-000017 value="1.2" doi=10.1021/ic8012359`).
- The earlier "IC50 numeric values are NOT exposed" framing (carried in
  `platinai_4cell_mbfinder.meta.json` from 2026-09-29) was wrong per the
  2026-10-03 CORRECTION in `platinai_real_label_loader.py` docstring.
- The COMPLETE harvest is in progress; this Wf-A38 captures the run +
  any post-run merge/clean step.

The Wf-A37-4 KNNEvaluator + Wf-A37-5 harvest together make the
PlatinAI real-label pipeline (R16/R17 platinum axis) reachable end-to-end
without any synthetic-label fallback.

## Acceptance check

```bash
# verify COMPLETE per-cell-line CSVs exist on disk
ls -la .data/molmetal/wf_a37_platinai_biodata/platinai_*_biodata.csv
wc -l .data/molmetal/wf_a37_platinai_biodata/platinai_*_biodata.csv
cat .data/molmetal/wf_a37_platinai_biodata/wf_a37_platinai_harvest_summary.json
```

A successful full harvest shows:

- `totals.rows_unique_compound_cell >= 5500`
- `totals.with_real_numeric_value >= 5500`
- `totals.with_smiles_after_merge >= 3500` (after `get-substances`
  pass completes)
