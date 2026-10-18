# DeepSeek Brief: Re-harvest MB-Finder IC50 Data to Match Rusanov 2025 Baseline

**Date**: 2026-10-16
**Owner**: mol-metal team (precious-metal anticancer drug generation)
**Background context**: We're writing a paper on conditional flow matching (CFM) for de novo
generation of platinum anticancer complexes. The protocol must match Rusanov 2025 (ChemRxiv
DOI 10.26434/chemrxiv-2025-pp32k) which uses MB-Finder as the 17,732-IC50 baseline dataset.

---

> ## ⚠️ CORRECTION (2026-10-17) — read this before acting on anything below
>
> This brief was written from the paper's headline number and several of its premises
> did not survive contact with the API. The work was re-scoped and executed; the
> corrections are recorded here so the original text below is not read as fact.
>
> **1. "4 cells, ≥17,000 rows" was the wrong model.** The V2 SI ("Data curation",
> p. S27) states verbatim: *"a labelled dataset of 4,078 structurally unique platinum
> complexes with cytotoxicity data ... IC50 values from 592 cell lines were
> standardized to µM."* The 17,732 is a **cross-cell-line Pt total**; the four modelled
> cells (A549/MCF-7/A2780/A2780cis = 8,428 rows) are only 26% of it. `GET
> /api/v2/get-count` returns `{"compoundsCount":7620,"cellsCount":832,
> "ic50Count":32022,"metalCounts":{"Cu":2676,"Pt":3614,"Au":1218,"Re":144}}`.
>
> **2. "MCF7 entirely missing" was a spelling bug, not missing data.** The endpoint
> requires `cell_line`; `"MCF7"` matches nothing (`total=0`) while `"MCF-7"` returns
> 2,747. Also: the endpoint returns `meta.total`/`meta.last_page`, so the "scan
> unfiltered pages 1..1500 and filter post-hoc" plan was never needed.
>
> **3. The deliverable's `smiles` field cannot be filled from this endpoint.**
> `get-ceil-line-bio-data` does not return a structure — its `compounds` object carries
> only `incubationTime/numeric_value/title/doi/method/stability/in_vivo/imageUrl`.
> SMILES lives behind `get-substances`, joined on the compound code (`MB-Pt-000010`).
> Any schema expecting `smiles` from the biodata endpoint will get null.
>
> **4. The harvested files are NOT Pt-only.** `_flatten` wrote
> `comp.get("metal") or "Pt"`, but the API never returns `metal`, so every row was
> labelled Pt. Measured on the real codes: 60.7% `MB-Pt-`, 25.8% `MB-Cu-`, 11.8%
> `MB-Au-`. A Pt filter keyed on the `metal` field keeps everything. Filter on the
> `MB-Pt-` prefix instead (as `platinai_real_label_loader.py` already does).
>
> ### What was actually done
>
> - **FINAL (2026-10-19 01:08): 485 of 832 cell lines → 30,185 rows**, of which
>   **17,735 are platinum** = **100.02% of the 17,732 target**.
> - 3,630 unique Pt compounds, **all** carrying SMILES; **15,953 rows** are
>   training-ready (`value_mean` + SMILES). 0 duplicate `biodata_id`, 0 non-Pt
>   rows in the Pt view, 0 empty SMILES.
> - Pt-only view: `pt/platinai_pt_all_cells_with_smiles.{jsonl,csv}` + per-cell
>   `pt/platinai_<stem>_pt_with_smiles.*` (scripts `wf_a48_pt_only_view.py`,
>   `wf_a49_smiles_join.py`). Summary: `pt/pt_only_summary.json` +
>   `pt/smiles_join_summary.json`.
> - `get-substances` harvest **completed**: 1,543/1,543 pages → 7,703 substances,
>   all with SMILES, 0 pages unrecovered (`wf_a47_substances_harvest.py`).
> - Gap-fill was `wf_a46_mcf7_harvest.py --all-cells --pt-target 17732`; the last
>   307 Pt rows took ~5.5 h because of six IP-level ban windows of 25-60 min.
>
> ### The remaining honest-negative
>
> The 17,732 **row** target is met; what is *not* available from this source is
> 17,732 **distinct platinum complexes**. 485 cell lines yield 3,630 Pt compounds
> (vs `get-count`'s `metalCounts.Pt` = 3,614), while Rusanov's own V2 SI describes
> 4,078 unique Pt complexes. So the baseline is matched on measurements, not on
> chemical space — that distinction should be stated in §3.

---

## TL;DR

We have **5,681 measured IC50 rows** from MB-Finder (3 of 4 cells). We need
**17,732 measured IC50 rows** (4 cells) to match Rusanov's published baseline. The
gap is **MCF7 (entirely missing, 0 rows)** + **deeper harvest on A549/A2780/A2780cis**.

Target deliverable: 4 JSONL files (one per cell line), each with
`compound_code, cell_line, value_mean, value_sd, method, doi, imageUrl, smiles, metal, _page`
schema, totaling **≥17,000 rows** across 4 cells. Plus a `summary.json` with
per-cell row counts and overlap stats.

---

## Existing Assets (read-only, do not re-fetch)

### 1. Rusanov 2025 paper data (already on disk, on server)

| Path | Format | n_rows | Notes |
|---|---|---|---|
| `.data/molmetal/rusanov_2025/parsed/s5_ece.json` | JSON | 24 ECE | KANO vs MKANO × 4 cells × 3 splits |
| `.data/molmetal/rusanov_2025/parsed/s7_pt_wet_lab.json` | JSON | 20×4=80 | Pt-1-19 + Cisplatin wet-lab validation |
| `.data/molmetal/rusanov_2025/chemrxiv-2025-pp32k/table s2-10.xlsx` | xlsx | various | V2 SI xlsx (S6 has 211,859 Pt screening library + MKANO predicted) |
| `.data/molmetal/rusanov_2025/INVENTORY.md` | markdown | — | complete file inventory with hashes |

### 2. MB-Finder partial harvest (already on disk, can be merged with new fetch)

| Path | n_rows | Cell | Status |
|---|---|---|---|
| `.data/molmetal/wf_a37_platinai_biodata/platinai_a549_biodata.jsonl` | 2,590 | A549 | done |
| `.data/molmetal/wf_a37_platinai_biodata/platinai_a2780_biodata.jsonl` | 1,818 | A2780 | done |
| `.data/molmetal/wf_a37_platinai_biodata/platinai_a2780cis_biodata.jsonl` | 1,273 | A2780cis | done |
| `.data/molmetal/wf_a37_platinai_biodata/platinai_mcf7_biodata.jsonl` | **0 (missing)** | MCF7 | **need to fetch** |
| `.data/molmetal/platinai_biodata_full/*.jsonl` | 1,500-1,600 each × 3 cells | same 3 cells | alternate snapshot, dedup |

**Total currently**: 5,681 rows. **Gap to 17,732**: 12,051 rows.
- MCF7 entirely missing: 0 → 4,000+ (estimate, MB-Finder has ~5K Pt per cell per V2 paper 17,732/4 ≈ 4,433)
- A549: 2,590 → ~5,000 (need ~2,400 more)
- A2780: 1,818 → ~4,000 (need ~2,200 more)
- A2780cis: 1,273 → ~4,000 (need ~2,800 more)

### 3. Existing fetch script (reference, can be re-used or rewritten)

- `molmetal/scripts/wf_a40_smiles_fetcher.py` — page-by-page scan of MB-Finder `get-substances` endpoint
- `molmetal/scripts/wf_a37_platinai_full_harvest.py` — full PlatinAI harvest driver (was killed at page 706-722 stuck)
- `molmetal/scripts/platinai_real_label_loader.py` — has `_post(endpoint, body, ...)` helper for MB-Finder API

The `_post` function signature:
```python
def _post(endpoint: str, body: dict, *, max_retries: int = 4, timeout: int = 20) -> Optional[list]:
    """POST one page. Returns the data list (or None on transport/server failure)."""
    url = f"https://api.mb-finder.org/api/v2/{endpoint}"
    ...
    r = requests.post(url, json=body, timeout=timeout, proxies={"http": None, "https": None})
    ...
    return payload.get("data") or []
```

---

## What Needs to be Filled

### Gap analysis

| Cell | Current rows | Need (estimate) | Gap |
|---|---|---|---|
| MCF7 | **0** | ~4,000 | **-4,000 (entire cell missing)** |
| A549 | 2,590 | ~5,000 | -2,400 |
| A2780 | 1,818 | ~4,000 | -2,200 |
| A2780cis | 1,273 | ~4,000 | -2,800 |
| **Total** | **5,681** | **~17,000** | **-12,051** |

### MB-Finder API endpoint (already tested, working)

```
POST https://api.mb-finder.org/api/v2/get-ceil-line-bio-data
Content-Type: application/json
Body: {"per_page": 5, "page": N}
```

- Returns list of biodata records (per_page capped at 5)
- Each record: `compound_code, cell_line, biodata_id, value_mean, value_sd, method, doi, imageUrl, smiles, metal, ...`
- **Filter to Pt compounds**: keep only records where `compound_code` starts with `MB-Pt-`
  OR where SMILES contains `[Pt`, `[Pt+`, `->[Pt`, etc.
- No auth, no rate limit headers (but server throttles — see below)

### Rate limit profile (empirically measured)

- 4-concurrent fetcher loop → 100% ConnectTimeout after 30+ pages
- 1 fetcher with `--delay 12.0` (seconds between calls) → ~80% success rate
- 1 fetcher with `--delay 20.0` → ~90% success rate
- Recommended: **single fetcher, 20-30s sleep between calls, 4 retries with exponential backoff (2/4/8/16s)**
- Resume support: load existing JSONL, skip already-fetched `(compound_code, cell_line)` pairs

### Page range to scan

- The 4 cell lines are spread across pages ~1-1000+ (per the V2 paper, MB-Finder
  has ~4,400 Pt compounds total in 17,732 IC50 across 4 cells, so ~1,100 compounds
  per cell, ~5 substances per page, ~220 pages per cell, but with mixed cells
  per page, may need to scan ~500-1000 pages)
- Conservative: scan pages 1-1500, abort on 30 consecutive empty pages (Pt cohort ended)

---

## Output Specification

### Files to create (4 + 1 summary)

```
.data/molmetal/wf_a37_platinai_biodata/platinai_mcf7_biodata.jsonl      (NEW)
.data/molmetal/wf_a37_platinai_biodata/platinai_a549_biodata.jsonl     (extend)
.data/molmetal/wf_a37_platinai_biodata/platinai_a2780_biodata.jsonl    (extend)
.data/molmetal/wf_a37_platinai_biodata/platinai_a2780cis_biodata.jsonl (extend)
.data/molmetal/wf_a37_platinai_biodata/harvest_summary_2026-10-16.json  (NEW)
```

### JSONL row schema (per line, JSON object)

```json
{
  "compound_code": "MB-Pt-000005",
  "cell_line": "A2780",
  "biodata_id": 504,
  "numeric_value_raw": "37.7",
  "value_mean": 37.7,
  "value_sd": null,
  "method": "SRB",
  "incubationTime": "48.0",
  "stability": null,
  "in_vivo": null,
  "doi": "10.1039/c5tb02429c",
  "title": "...",
  "imageUrl": "https://api.mb-finder.org/storage/images/MB-Pt-000005.svg",
  "smiles": null,
  "metal": "Pt",
  "_page": 1
}
```

### Summary JSON

```json
{
  "fetched_at": "2026-10-16T...",
  "endpoint": "POST https://api.mb-finder.org/api/v2/get-ceil-line-bio-data",
  "per_page": 5,
  "pages_scanned_per_cell": {"MCF7": ..., "A549": ..., ...},
  "rows_per_cell": {"MCF7": ..., "A549": ..., ...},
  "unique_compounds_per_cell": {"MCF7": ..., ...},
  "rows_with_smiles": ...,
  "rows_with_doi": ...,
  "empty_pages_aborted": ...,
  "total_walltime_seconds": ...
}
```

### Validation on completion

Run after fetch:

```bash
.venv/bin/python -c "
import json
import collections
total = collections.Counter()
for cell in ['MCF7', 'A549', 'A2780', 'A2780cis']:
    p = f'.data/molmetal/wf_a37_platinai_biodata/platinai_{cell.lower()}_biodata.jsonl'
    n = sum(1 for _ in open(p))
    total[cell] = n
    print(f'  {cell}: {n}')
print(f'  TOTAL: {sum(total.values())} (target 17000+)
"
.venv/bin/python molmetal/scripts/registry.py check
```

Both should pass: total ≥ 17,000, registry check still PASS.

---

## Out-of-scope (do NOT do these)

1. **Do not modify existing jsonl files in destructive way** — append, don't replace.
   If dedup needed, dedup by (compound_code, cell_line) keeping first occurrence.
2. **Do not re-fetch non-Pt compounds** — filter to Pt only (MB-Pt- prefix or Pt in SMILES).
3. **Do not touch `cfm_todo.json`, `TODO/02_data/datasets.md`, `registry/datasets.yaml`**
   — those are updated by the team after this task completes.
4. **Do not run any GPU training** — CPU only, this is just a data harvest task.
5. **Do not modify S5/S7/S9 JSON files** (parsed/) — they are already correct.
6. **Do not chase S1 (V1 supplementary_dataset_1.xlsx, 4,078 Pt / 18,995 IC50 main
   training set)** — V1 link is 404, no figshare/Zenodo/ResearchGate mirror exists.
   This is acceptable: paper §3 will cite the MB-Finder 17,732 IC50 directly as the
   Rusanov 2025 baseline (per V2 abstract: "we integrated our manually curated
   comprehensive platinum database of 17,732 IC50 values").

---

## Reference: existing fetch pattern

The current `wf_a40_smiles_fetcher.py` was killed at page 706-722 stuck (server
rate limit). Don't repeat the same 4-concurrent pattern. Use 1 fetcher with
20-30s delay.

Reference pattern (from `platinai_real_label_loader.py`):

```python
def _post(endpoint: str, body: dict, *, max_retries: int = 4, timeout: int = 20):
    url = f"https://api.mb-finder.org/api/v2/{endpoint}"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=body, timeout=timeout, 
                            proxies={"http": None, "https": None})
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict) and payload.get("success") is False:
                    return []  # soft rate-limit
                return payload.get("data") or []
            if r.status_code == 429:
                time.sleep(min(60.0, 3.0 * (2**attempt)))
                continue
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(60.0, 2.0 * (2**attempt)))
    return None
```

Existing schema for `get-ceil-line-bio-data` page-by-page scan is similar to
`get-substances` scan in `wf_a40_smiles_fetcher.py`. Read that file for
reference, but adapt the body schema: `{"per_page": 5, "page": N}` (no
compound_code filter, the server returns all cells in the page and we filter
post-hoc).

---

## Success Criteria

1. `platinai_mcf7_biodata.jsonl` exists and has ≥ 1,000 rows
2. All 3 other cell JSONLs have ≥ 4,000 rows (extending the existing 5,681)
3. Total across 4 cells ≥ 17,000 rows
4. `harvest_summary_2026-10-16.json` is created with stats
5. `python molmetal/scripts/registry.py check` still reports PASS (no schema breakage)
6. All 4 JSONLs are atomic-written (use `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`)
7. Resume support: if interrupted, re-running should skip already-fetched rows

---

## Estimated effort

- MCF7: ~30-60 min (full new cell)
- A549: ~20-30 min (extend from 2,590 to 5,000)
- A2780: ~20-30 min (extend from 1,818 to 4,000)
- A2780cis: ~20-30 min (extend from 1,273 to 4,000)
- Total wall-time: **2-3 hours** (single-threaded, 20s delay per call, ~5 calls per page)

**CPU-only, no GPU, no auth, public endpoint, resumable.**

---

## Reference for paper

When the paper §3 cites this data, the citation should be:

> "We harvested 30,185 IC50 measurements from MB-Finder (the curated metal-complex
> database of Rusanov et al. 2025, https://www.mb-finder.org) across 485 cell lines
> via the public `get-ceil-line-bio-data` API endpoint; 17,735 of these are platinum
> (3,630 unique complexes). Structures were joined from `get-substances` on the
> compound code. DOI: 10.26434/chemrxiv-2025-pp32k"

⚠️ The original citation text in this section claimed "17,732 IC50 values ... across
4 cell lines (MCF-7, A549, A2780, A2780cis)". That was never true: 17,732 is the
paper's *cross-cell-line* Pt total and those 4 cells supply 8,428 rows. See the
CORRECTION block at the top.

The task is to produce the harvested JSONL files that back this citation.