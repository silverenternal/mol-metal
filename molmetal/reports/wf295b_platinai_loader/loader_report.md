# Wf-295B — PlatinAI Real-Label Loader (MB-Finder portal partial harvest)

**Date:** 2026-09-29
**Workflow:** Wf-295B (TODO-119 P1.3 PlatinAI real-label loader implementation)
**Strategy:** S7 (MB-Finder portal partial harvest) — top-1 pick from Wf-292D
**Owner:** Hugo Lee + ultracode
**Status:** SMOKE PASS — 100/100 Pt compounds loaded with real bio-data, 0 NaN SMILES

---

## §0 TL;DR

- **Strategy chosen:** S7 (MB-Finder portal partial harvest) — top-1 recommendation from `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` §3 R1
- **Implementation status:** WORKING — `molmetal/scripts/platinai_real_label_loader.py` shipped + smoke-tested
- **n_smoke_mols:** 100 Pt compounds (pages 850-869, MB-Pt-002305 .. MB-Pt-003616)
- **n_real_labels:** 100/100 compounds have ≥1 cell-line bio-data record (REAL, not synthetic)
- **Per-cell-line coverage:** A549=100, MCF7/A2780/A2780cis=0 (single-paper slice — these compounds come from one Rusanov 2025 paper that focused on A549)
- **Output:** `/home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell_mbfinder.csv` + `.meta.json`

---

## §1 Strategy Selection Rationale

Per Wf-292D §3, the top-2 strategies were:
- **R1 (S7) MB-Finder portal partial harvest** — RUN NOW (~1 hr walltime, 60% reliability)
- **R2 (S1) Selenium + undetected-chromedriver** — RUN AFTER R1 + USER OK (~2 hr, 60-75% reliability)

We picked **S7 (R1)** because:
1. Lower walltime (no headless Chrome install footprint)
2. No ban-risk from Cloudflare (portal API is rate-limited but not gated)
3. Direct paper-gate compliant (no institutional-login wall to bypass)
4. Wf-292D §5.1 sign-off: walltime ≤ 1 hr ✅
5. Wf-292D §5.2 sign-off: portal endpoint verified pre-launch ✅

---

## §2 MB-Finder API Endpoints Discovered

By static-analysis of the React/Vite bundle at `https://www.mb-finder.org/assets/index-CmBf1tCC.js` (18.9 MB) + dynamic probing, we discovered 5 working endpoints on `https://api.mb-finder.org/api/v2/`:

| Endpoint | Method | Returns |
|---|---|---|
| `/get-count` | GET | `{compoundsCount: 7622, cellsCount: 827, referenceCount: 2196, ic50Count: 31996, metalCounts: {Cu: 2676, Pt: 3616, Au: 1218, Re: 144}}` |
| `/get-tissues` | POST | 30 tissues (Adrenal, Bladder, ..., Uterus) |
| `/get-substances` | POST | Paginated substance list (`per_page` capped at 5 server-side; total=7735; 1547 pages) |
| `/get-cell-lines` | POST | Per-substance cell-line coverage + bioDataCount (real per-cell counts) |
| `/get-references` | POST | Per-substance literature references (DOIs + abstracts) |

**Rate limit:** ~1 request per 2-3 s. We use `--delay 2.5` as default. 429 throttle triggers exponential backoff.

**Honest caveat:** Per-cell-line IC50 **NUMERIC VALUES** are NOT exposed by any of these 5 endpoints. The API exposes only the **coverage matrix** (compound × cell-line → boolean + bioDataCount). This is the binding constraint for full real-label parity with Rusanov 2025 Table S4-S5.

---

## §3 Implementation: `molmetal/scripts/platinai_real_label_loader.py`

### §3.1 What it does

1. **Probe `/get-count`** for aggregate scope (3,616 Pt compounds, 31,996 IC50 measurements)
2. **Paginate `/get-substances`** starting from `--start-page 850` (deep in Pt range)
3. **Filter for `MB-Pt-*` imageId prefix** (skip Cu/Au/Re compounds)
4. **For each Pt compound**, fetch `/get-cell-lines` to get per-cell-line bio-data coverage
5. **Write CSV** with: `mb_id, imageId, smiles, mw, ceilLineCount, bioDataCount, has_mcf7, has_a549, has_a2780, has_a2780cis, n_cell_lines_with_data, n_biodata_total, total_cell_lines_listed`
6. **Write .meta.json** with: framing, honest caveat, coverage stats, endpoint attribution

### §3.2 Usage

```bash
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/platinai_real_label_loader.py \
    --out .data/molmetal --n-target 100 --start-page 850 --delay 2.5
```

### §3.3 Output schema (`.data/molmetal/platinai_4cell_mbfinder.csv`)

```
mb_id,imageId,smiles,mw,ceilLineCount,bioDataCount,has_mcf7,has_a549,has_a2780,has_a2780cis,n_cell_lines_with_data,n_biodata_total,total_cell_lines_listed
3662,MB-Pt-002305,[OH2][Pt]([Cl])([Cl])N12CCN(CC1)CC2,396.18,6,6,0,1,0,0,5,22,5
3661,MB-Pt-002304,[OH2][Pt]([OH2])([OH2])([Cl])N12CN3CN(CN(C3)C1)C2.[Cl],460.22,5,5,0,1,0,0,5,22,5
...
```

The `has_<line>` flags are REAL (compound has bio-data records for that cell line), but the IC50 NUMERIC values are NOT exposed by the public API.

### §3.4 Honest framing (per Wf-292D §3 R1)

> "REAL MB-Finder-hosted Rusanov 2025 Pt compounds + REAL per-cell-line bio-data COVERAGE MATRIX. Per-cell-line IC50 NUMERIC VALUES are NOT exposed by the public MB-Finder API (only presence/absence + count). We emit has_<line> boolean flags (real coverage) instead of synthetic pIC50. This replaces the SYNTHETIC pIC50 loader (Wf-278B) as the closest-to-real PlatinAI dataset without academic outreach."

---

## §4 Smoke Test Results (2026-09-29)

```
=== Wf-295B: MB-Finder real-label loader (Strategy S7, Wf-292D top-1) ===
Source: https://api.mb-finder.org/api/v2
n_target: 100   start_page: 850   delay: 2.5s

[1/3] Probing MB-Finder aggregate counts...
      compounds=7622 Pt=3616 IC50s=31996 cells=827
[2/3] Paginating Pt compounds starting page 850...
      Got 100 Pt compounds in 120 API requests (20 pages, 0 non-Pt skipped)
[3/3] Writing outputs to /home/hugo/codes/mol-metal/.data/molmetal...
      CSV:  /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell_mbfinder.csv
      Meta: /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell_mbfinder.meta.json

=== Smoke verification ===
n_smoke_mols:        100
n_with_real_label:   100 (compounds with >=1 cell-line bio-data)
n_nan_smiles:        0
coverage mcf7/a549/a2780/a2780cis: 0/100/0/0
```

**Verdict:**
- 100/100 mols have non-NaN SMILES ✅
- 100/100 mols have ≥1 real bio-data record ✅
- 0 NaN labels ✅
- Coverage: A549=100, MCF7=0, A2780=0, A2780cis=0 — single-paper slice
- API requests: 120 (1 get-count + 20 get-substances + 100 get-cell-lines + retries)
- Walltime: ~5 min (100 compounds × 2.5s/request average)

---

## §5 Honest Caveats & Limitations

### §5.1 Per-cell-line IC50 NUMERIC values NOT exposed
- All 5 discoverable MB-Finder API endpoints return coverage + counts but not the actual IC50 measurement values
- Per-compound IC50 requires either:
  - **S1 (Selenium + undetected-chromedriver)** against ChemRxiv supplementary xlsx (Wf-292D §3 R2, ~2 hr, requires user OK)
  - **S6 (Rusanov direct outreach)** for the extended data (Wf-292D §3, 2-6 weeks latency, blocked by `TODO/pending/84_no_paper_task_without_authorization.md`)
- Until then, `has_<line>` flags are the best real signal we can ship without auth wall

### §5.2 Single-paper slice (coverage concentrated on A549)
- Pages 850-869 yield compounds from one Rusanov 2025 sub-paper focused on A549 only
- For 4-cell balanced coverage, would need to **stratify sampling across page ranges** (e.g. pages 850-870 for A549-heavy, 880-900 for A2780, 900-950 for MCF7)
- Honest framing: this 100-mol smoke sample is **biased** toward A549 cell-line bio-activity

### §5.3 Rate limit + 429 throttle
- ~1 request per 2-3 s sustainable; burst >5 reqs in 10 s triggers HTTP 429
- Loader uses `--delay 2.5` default + exponential backoff (3s, 6s, 12s, 24s) on 429
- For 3,616 full Pt cohort harvest: ~3-4 hr walltime at 2.5s/req (manageable but non-trivial)

### §5.4 API contract stability
- API endpoints discovered 2026-09-29 may rotate; re-probe required if `get-substances` returns 404
- Server caps `per_page` at 5 (no way to increase)
- Server returns substances sorted by descending ID; Pt range is approximately pages 850-1547

---

## §6 Future Work

1. **Stratified sampling** — Sample across pages 850-1547 to get 4-cell balanced coverage (next session, ~30 min)
2. **S1 (Selenium UC) for per-cell IC50 NUMBERS** — Defer to user OK per Wf-292D §5.4 + standing party-twenty-spirit §3.5
3. **S6 (Rusanov outreach)** — Defer until `TODO/pending/84_no_paper_task_without_authorization.md` gate lifts
4. **Cross-validate against Wf-278B synthetic CSV** — Run `molmetal/scripts/cross_validate_platinai_real_vs_synth.py` to compare real coverage matrix vs synthetic pIC50 distribution

---

## §7 Cross-references

- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` — §3 R1 (S7), §5 sign-off
- `molmetal/scripts/platinai_real_label_loader.py` — this Wf's implementation
- `molmetal/scripts/build_platinai_4cell_csv.py` — Wf-278B SYNTHETIC loader (now superseded for paper §3.7 framing)
- `molmetal/data/platinai_4cell.csv` — pre-Wf-278B CSV (CSD-refcode-keyed, provenance unclear)
- `molmetal/.data/molmetal/platinai_4cell.csv` — Wf-278B synthetic output
- `molmetal/.data/molmetal/platinai_4cell_mbfinder.csv` — this Wf's REAL output (NEW)
- `molmetal/.data/molmetal/platinai_4cell_mbfinder.meta.json` — this Wf's REAL meta (NEW)
- `TODO/completed/175_platinai_226k_gap.md` — §1.1-§1.5 (URL fetch attempts)
- `TODO/pending/84_no_paper_task_without_authorization.md` — blocks S6 outreach
- `TODO/03_plans/LAMBDA_ROADMAP.md` §2.4 — R21-P3 (Hilbert-ASTSR R² lift on real PlatinAI pIC50)

---

*Wf-295B shipped: real-label loader sourcing Rusanov 2025 Pt compounds from MB-Finder public API (Strategy S7, Wf-292D top-1). 100/100 smoke mols with real bio-data, 0 NaN, honest framing on per-cell-line IC50 limitation (deferred to S1/S6 per Wf-292D).*