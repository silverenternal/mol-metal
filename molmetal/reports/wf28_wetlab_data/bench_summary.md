# Wf-28 Phase 5 — wet-lab data end-to-end benchmark

_Generated_: 2026-09-25 21:15:29 UTC

End-to-end verification of the Wf-28 wet-lab data preparation pipeline (Phase 1 tmQM subset + Phase 2 NCI-60 9-cell precious-metal long-form + Phase 2 PlatinAI 4-cell cross-link), each compared against the PlatinAI 5K real-data benchmark on the same 2-D descriptor axis (Lipinski MW / logP + QED + Ertl SA unit).

## 1. Inputs

| Corpus | Path |
|---|---|
| tmqm_dblock_csv | `/home/hugo/codes/mol-metal/molmetal/data/tmqm_dblock_subset.csv` |
| nci60_precious_9cell_csv | `/home/hugo/codes/mol-metal/molmetal/data/nci60_precious_9cell.csv` |
| platinai_4cell_csv | `/home/hugo/codes/mol-metal/molmetal/data/platinai_4cell.csv` |
| platinai_5k_csv | `/home/hugo/codes/mol-metal/molmetal/data/platinai_5k.csv` |

## 2. Per-corpus benchmark metrics

| Corpus | n_candidates | n_rows | mean_mw | mean_logp | mean_qed | mean_sa | parse_fail_rate | secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tmqm_dblock | 4,982 | 5,000 | 592.0 | 3.76 | 0.257 | 0.552 | 0.00% | 9.27 |
| nci60_precious_9cell | 5,207 | 5,207 | 578.7 | 5.24 | 0.282 | 0.742 | 0.00% | 10.26 |
| platinai_4cell | 4,078 | 4,078 | 674.8 | 3.13 | 0.263 | 0.583 | 0.00% | 7.27 |
| platinai_5k | 5,000 | 5,000 | 657.9 | 4.19 | 0.233 | 0.579 | 0.04% | 9.43 |

## 3. Descriptor stds (reference only)

| Corpus | σ_mw | σ_logp | σ_qed | σ_sa |
|---|---:|---:|---:|---:|
| tmqm_dblock | 162.1 | 4.60 | 0.144 | 0.112 |
| nci60_precious_9cell | 204.2 | 4.82 | 0.188 | 0.096 |
| platinai_4cell | 182.3 | 5.38 | 0.141 | 0.105 |
| platinai_5k | 184.4 | 4.76 | 0.138 | 0.100 |

## 4. Comparability vs PlatinAI 5K (the r18 reference)

Verdict = OK iff each candidate mean is within ±2 σ_ref of the PlatinAI 5K reference.  This is a cheap distribution-agnostic guardrail — the wet-lab tier should sit in the same descriptor neighbourhood as the published r18 reference corpus.

| Candidate → Reference | Δ_mw | Δ_logp | Δ_qed | Δ_sa | Verdict | Notes |
|---|---:|---:|---:|---:|---|---|
| tmqm_dblock → platinai_5k | -65.96 | -0.44 | +0.024 | -0.027 | OK |  |
| nci60_precious_9cell → platinai_5k | -79.19 | +1.05 | +0.048 | +0.163 | OK |  |
| platinai_4cell → platinai_5k | +16.92 | -1.06 | +0.030 | +0.004 | OK |  |

## 5. Per-axis reading guide

* **mean_mw** (Lipinski rule-of-5 ceiling = 500 Da): the four corpora cluster between 570–680 Da.  Slightly above Lipinski, which is expected — these are coordination complexes with inorganic cores that intentionally exceed the small-molecule Rule-of-5 envelope.
* **mean_logp** (Crippen): all four in the [3.1, 5.2] window. PlatinAI 4-cell is the most hydrophilic (mean_logp ≈ 3.1); the NCI-60 precious-metal tier is the most lipophilic (mean_logp ≈ 5.2).  Differences are within the ±2σ envelope.
* **mean_qed** (Bickerton, drug-likeness, [0, 1]): all four in [0.23, 0.28] — the low QED reflects the metallodug-shaped structures (large MW, complex coordination) which is consistent across all four corpora.
* **mean_sa** (Ertl SA unit, [0, 1] higher = easier): NCI-60 (0.74) sits noticeably above the rest (0.55–0.58). This is expected — the precious-metal long-form draws from a *measured cytotoxicity* substrate which already passed wet-lab synthesis, so it is enriched for easy-to-synthesise scaffolds. PlatinAI 5K (0.58) and tmQM (0.55) are broadly similar; 4-cell (0.58) matches the r18 reference.

## 6. Comparability verdict

Of 3 candidate corpora, **3** are within ±2 σ_ref of the PlatinAI 5K reference.  **All corpora pass** — the wet-lab tier is in the same descriptor neighbourhood as the r18 SBDD reference.

## 7. Honest framing

* This benchmark is **CPU-only** — RDKit descriptors + pandas.  It does not invoke the SBDD model stack.
* The ±2 σ window is intentionally wide.  For a stricter KS-test / MMD verdict, see the r18 full PlatinAI bench (`molmetal/scripts/benchmark_r18_platinai.py`).
* `mean_sa` is the Ertl SA-score **unit-transformed** to [0, 1] (higher = easier to synthesise).  This matches the r18 metric direction (Vina, pIC50 also 'higher = better').
* The benchmark exercises the actual CSV files on disk (not mocks).  If a CSV is missing, that corpus's row is missing from the summary but no exception is raised.
* `--max-rows N` caps each corpus at N SMILES for smoke tests and is the key knob for fitting the bench under a fixed wall-clock budget.

## 8. Reproducibility

* Seed: `20260926` (deterministic — no RNG in the script today)
* CPU-only (RDKit + pandas + stdlib)
* Wall-clock (full sweep on the four real CSVs): ≈ 35 s on a single CPU thread (see the `secs` column in §2).
* CLI: `python -m molmetal.scripts.bench_wetlab_data`
