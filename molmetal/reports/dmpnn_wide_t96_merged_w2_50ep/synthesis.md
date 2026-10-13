# LOMO 5-fold × 3-seed × 50-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-07  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2  
**Epochs/seed**: 50  
**Cohort CSV**: .data/molmetal/merged/metal_lomo_merged_t96.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: False  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

Leave-one-metal-out (LOMO) cross-metal transfer benchmark: hold out
one metal's rows entirely, train a multi-task D-MPNN on the rest, and
measure ROC-AUC on the held-out metal (the m19 protocol). The active
components are echoed from the config below — not asserted.

1. **Wasserstein transfer regularisation — ON** (weight 0.05).
   Sliced-W₂ over per-metal embeddings (canonical M1 primitive from
   `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`),
   added to the BCE+MSE loss and injected every 5 epochs.
   **Wf-E1 (2026-10-05)**: this term is a NO-OP unless
   `--wasserstein-real` is passed — the pre-Wf-E1 code detached the
   per-metal embeddings, so W₂ contributed ZERO gradient while
   costing 2.4x wall-time (Rh LOMO seeds 0-2, mean Δ test_AUC
   = −0.0023, within noise). `w2_effective_weight` in the per-seed
   JSON is the authoritative record of whether W₂ reached the graph.
2. **Rh closure-theorem oversampling — ON** (500 synthetic
   Rh rows appended to the Rh train pool; Wf-125 P1 Catherine 6/6
   PASS cohort). It is *inactive* by design (MetalCytoToxDB Rh is
   ~85 % inactive at the IC50<10 µM gate) — a *chemistry*
   augmentation, not a *label* augmentation.
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — OFF** (threshold 5.0; engage with `--use-l2-guard`).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.6646 ± 0.0884**  
(n=15 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6321 ± 0.0390** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6452 | 0.0280 | 3 |
| Ir | 0.6001 | 0.0094 | 3 |
| Os | 0.8277 | 0.0200 | 3 |
| Rh | 0.6321 | 0.0390 | 3 |
| Re | 0.6181 | 0.0443 | 3 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 7196 | 17815 | 0.289 | 0.8385 | 0.6093 | 1272.1 |
| Ru | 1 | 7196 | 17815 | 0.289 | 0.8581 | 0.6484 | 1192.8 |
| Ru | 2 | 7196 | 17815 | 0.289 | 0.8373 | 0.6777 | 1026.3 |
| Ir | 0 | 19218 | 4458 | 0.493 | 0.9109 | 0.6088 | 2268.4 |
| Ir | 1 | 19218 | 4458 | 0.493 | 0.8951 | 0.5871 | 2292.9 |
| Ir | 2 | 19218 | 4458 | 0.493 | 0.9130 | 0.6045 | 1991.8 |
| Os | 0 | 22308 | 1024 | 0.486 | 0.9035 | 0.8095 | 3255.5 |
| Os | 1 | 22308 | 1024 | 0.486 | 0.9165 | 0.8180 | 2956.8 |
| Os | 2 | 22308 | 1024 | 0.486 | 0.8993 | 0.8555 | 2689.6 |
| Rh | 0 | 24662 | 572 | 0.420 | 0.6611 | 0.6858 | 3823.8 |
| Rh | 1 | 24662 | 552 | 0.534 | 0.7111 | 0.6157 | 4668.6 |
| Rh | 2 | 24662 | 535 | 0.445 | 0.7265 | 0.5947 | 3553.9 |
| Re | 0 | 22451 | 865 | 0.279 | 0.9055 | 0.5922 | 2286.6 |
| Re | 1 | 22451 | 865 | 0.279 | 0.9093 | 0.6806 | 2267.4 |
| Re | 2 | 22451 | 865 | 0.279 | 0.9069 | 0.5817 | 2840.4 |

## §3 Honest framing (party-twenty-spirit §3.5)

**MEASURED here**: the numbers above are computed end-to-end on this
host with the chosen (epochs, seeds, wasserstein_weight) configuration.
No silent PROJECTED-to-MEASURED promotion.

**PROJECTED (TODO-121 §4)**: Rh LOMO AUC ≥ 0.50; overall LOMO mean
≥ 0.60.

**Baseline reference (MEASURED, pre-repair ≤24 h cohort)**: Rh 0.3812,
overall 0.5575 — see
`molmetal/reports/wf_a2_m19_rh_diagnosis/diagnosis.json`. That baseline
used a DIFFERENT cohort (`molmetal/data/metal_lomo.csv`), so any delta
must be computed against the same cohort; the cohort CSV for THIS run
is echoed in the header above.

**Hit / miss summary**: Rh target HIT,
overall target HIT,
original gate BELOW GATE.

**Caveats**:

* The synthetic Rh cohort is *inactive* by default (MetalCytoToxDB
  Rh is ~85 % inactive at the IC50<10 µM gate). The oversample is
  a *chemistry* augmentation, not a *label* augmentation.
* The Wasserstein regulariser is injected every 5 epochs (not every
  batch) to keep the per-epoch wall-time reasonable. Wf-E1: it is
  only computed at all when `--wasserstein-real` is passed;
  `w2_effective_weight` in the per-seed JSON is the authoritative
  record of whether W₂ reached the gradient.
* The W₂ snapshot at the end of training is recorded in
  `final_per_metal_w2` for diagnostic purposes (see per-seed JSON);
  it is empty when `--wasserstein-real` is off.

## §4 Cross-references

* TODO-121 m19 LOMO strengthen (parent plan)
* Wf-125 Phase 1 close-out (synthetic Rh cohort + W₂ building block)
* `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`
* `molmetal_lam.reward.closure_oversample.ClosureOversampler`
