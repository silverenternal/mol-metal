# Wf-74 Phase 3 — IC50 analysis pipeline synthesis

**Date**: 2026-09-26
**Source**: synthetic fixture (smoke test; not wet-lab data)
**Pipeline**: §1 normalize → §2 Hill fit → §3 pIC50 + bootstrap CI → §4 ANOVA → §5 agreement

## Per-compound × per-cell pIC50 (first 5 records)

| Compound | Cell | Metal | IC50 µM | pIC50 | 95% CI | Hill | QC |
|---|---|---|---:|---:|---|---:|---|
| platinai_0454 | MCF7 | Ir | 22.0081 | 4.657 | [4.536, 4.735] | 1.32 | pass |
| platinai_0454 | A549 | Ir | 21.7812 | 4.662 | [4.593, 4.747] | 1.22 | pass |
| platinai_0454 | A2780 | Ir | 20.9765 | 4.678 | [4.564, 4.747] | 1.45 | pass |
| platinai_0454 | A2780cis | Ir | 21.1126 | 4.675 | [4.625, 4.723] | 1.45 | pass |
| platinai_1735 | MCF7 | Pt | 63.1913 | 4.199 | [4.095, 4.428] | 1.46 | pass |

## ANOVA per metal

| Metal | F | p_raw | n_groups |
|---|---:|---:|---:|
| Pt | 0.0143 | 0.9976 | 4 |
| Pd | 0.0143 | 0.9976 | 4 |
| Au | 0.0143 | 0.9976 | 4 |
| Ru | 0.0143 | 0.9976 | 4 |
| Ir | 0.0143 | 0.9976 | 4 |

## In-silico vs wet-lab agreement (per metal × cell_line)

**Gate**: Pearson r ≥ 0.7
| Cohort | Pearson r | Spearman ρ | n | Gate |
|---|---:|---:|---:|:-:|
| Pt-MCF7 | — | — | 0 | n<4 |
| Pt-A549 | — | — | 0 | n<4 |
| Pt-A2780 | — | — | 0 | n<4 |
| Pt-A2780cis | — | — | 0 | n<4 |
| Pd-MCF7 | — | — | 0 | n<4 |
| Pd-A549 | — | — | 0 | n<4 |
| Pd-A2780 | — | — | 0 | n<4 |
| Pd-A2780cis | — | — | 0 | n<4 |
| Au-MCF7 | — | — | 0 | n<4 |
| Au-A549 | — | — | 0 | n<4 |
| Au-A2780 | — | — | 0 | n<4 |
| Au-A2780cis | — | — | 0 | n<4 |
| Ru-MCF7 | — | — | 0 | n<4 |
| Ru-A549 | — | — | 0 | n<4 |
| Ru-A2780 | — | — | 0 | n<4 |
| Ru-A2780cis | — | — | 0 | n<4 |
| Ir-MCF7 | — | — | 0 | n<4 |
| Ir-A549 | — | — | 0 | n<4 |
| Ir-A2780 | — | — | 0 | n<4 |
| Ir-A2780cis | — | — | 0 | n<4 |

## Honest framing (per party-twenty-spirit §3.5)

* **Wet-lab gate NOT active**: this pipeline ships today as a template + synthetic smoke.
* No real wet-lab IC50 has been collected; all numbers above (except D-MPNN predictions from Wf-58 P6.1) are **synthetic**. Treat as a pipeline smoke test.
* D-MPNN pIC50 predictions (Wf-58 P6.1) are CPU-only bench quality on synthetic IC50 labels (Rusanov 2025 tables S4-S5 are behind Cloudflare); non-Pt candidates carry cross-metal transfer caveats.
* Paper §5 wet-lab validation section is 🚫 DEFERRED per TODO-84 + the wet-lab gate is itself PROJECTED on Wf-58 P6.2 timeline (6-month).

## Cross-references

* `TODO/pending/wetlab_ic50_analysis_pipeline.md` — full pipeline spec
* `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` — D-MPNN input
* `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` — Wf-58 P6.0/P6.1/P6.2 ship
* `TODO/pending/84_no_paper_task_without_authorization.md` — paper rule
