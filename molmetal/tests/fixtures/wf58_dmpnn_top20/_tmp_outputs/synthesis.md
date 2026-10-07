# Wf-58 Phase 2 — D-MPNN pIC50 prediction on top-20 wet-lab candidates

**Date**: 2026-10-07
**Model**: 4-cell-line multi-task D-MPNN (hidden=32, depth=2, dropout=0.1)
**Split**: scaffold (Bemis-Murcko) 70/10/20 (per-seed)
**Seeds**: [0]  
**Epochs**: 2, lr=0.001, batch=8, alpha=0.5
**Device**: cpu

## Headline numbers

| Metric | Value | Gate |
|---|---|---|
| Mean AUC (platinai_4cell test fold) | nan ± nan (n=1) | ≥ 0.73 |
| A2780cis AUC (cisplatin-resistant anchor) | nan ± nan | — |

## Per-cell-line mean pIC50 across top-20 (3-seed pooled)

| Cell line | Mean | Median | Std | Min | Max | n |
|---|---|---|---|---|---|---|
| MCF7 | 0.615 | 0.643 | 0.103 | 0.482 | 0.774 | 5 |
| A549 | 0.412 | 0.430 | 0.044 | 0.361 | 0.476 | 5 |
| A2780 | 0.356 | 0.360 | 0.031 | 0.308 | 0.402 | 5 |
| A2780cis | -0.160 | -0.130 | 0.040 | -0.227 | -0.128 | 5 |

## 3-seed reproducibility gate (per-compound pIC50 std ≤ 0.15 log units)

* **Pooled std across cells**: min = 0.0000, mean = 0.0000, max = 0.0000
* **Gate pass rate**: 5/5 compounds
* **Gate**: pooled std ≤ 0.15 log units

## Per-compound breakdown

| Rank | ID | Metal | Pooled std (log) | Gate | Mean pIC50 (across cells) |
|---:|---|---|---:|:-:|---|
| 1 | test_001 | Pt | 0.0000 | PASS | 0.256 |
| 2 | test_002 | Pd | 0.0000 | PASS | 0.262 |
| 3 | test_003 | Ir | 0.0000 | PASS | 0.356 |
| 4 | test_004 | Ru | 0.0000 | PASS | 0.326 |
| 5 | test_005 | Au | 0.0000 | PASS | 0.330 |

## Honest framing

* **Cross-metal transfer caveat**: top-20 spans Pt (n=2) / Pd (n=3) / Au (n=1) / Ru (n=3) / Ir (n=11). The D-MPNN is trained on PlatinAI 4-cell (Pt-only), so non-Pt predictions are **distribution-shifted**. Treat per-cell pIC50 numbers for Pd / Au / Ru / Ir as **rank-only**.
* **Synthetic IC50 labels**: ``platinai_4cell.csv`` carries synthetic pIC50 labels (Rusanov 2025 ChemRxiv tables S4-S5 are behind a Cloudflare JS challenge; see ``build_platinai_4cell.py`` honest framing). AUC numbers are bench-quality, not real-data quality.
* **CPU-only from-scratch D-MPNN**: this is the from-scratch edge-aware D-MPNN baseline, not the production chemprop D-MPNN. Wf-53 Phase 2 chemprop bench is the binding truth.
* **No wet-lab synthesis** — this is the in-silico prediction layer for the Wf-58 wet-lab outreach plan (TODO-97 R20-P6).
* **Paper gate**: per TODO-84, no paper §5 claim derives from this synthesis; outputs feed the wet-lab cooperation pipeline, not paper text.

## Cross-references

* ``TODO/pending/97_r20_p6_wetlab_ic50.md`` — R20-P6 wet-lab plan
* ``molmetal/reports/wf58_wetlab_dryrun/ranking_summary.md`` — Wf-58 P1 ranking
* ``molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`` — input top-20
* ``TODO/completed/82d_m16m19m8_retrain.md`` — Wf-32 P2 ship (m18 anchor)
* ``molmetal/scripts/train_platinai_4cell.py`` — 4-cell D-MPNN model defs
