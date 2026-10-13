# LOMO 5-fold × 3-seed × 50-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-06  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2  
**Epochs/seed**: 50  
**Cohort CSV**: .data/molmetal/merged/metal_lomo_merged_t96.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.0  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: False  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

Leave-one-metal-out (LOMO) cross-metal transfer benchmark: hold out
one metal's rows entirely, train a multi-task D-MPNN on the rest, and
measure ROC-AUC on the held-out metal (the m19 protocol). The active
components are echoed from the config below — not asserted.

1. **Wasserstein transfer regularisation — OFF** (weight 0.0).
   The objective is BCE + L2 guard only. Do NOT read this run as a
   Wasserstein result: per Wf-E1 the term requires BOTH a non-zero
   `wasserstein_weight` AND `--wasserstein-real`.
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

**Overall LOMO mean test AUC = 0.6661 ± 0.0827**  
(n=15 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6196 ± 0.0248** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6183 | 0.0327 | 3 |
| Ir | 0.6484 | 0.0279 | 3 |
| Os | 0.8237 | 0.0064 | 3 |
| Rh | 0.6196 | 0.0248 | 3 |
| Re | 0.6205 | 0.0048 | 3 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 7196 | 17815 | 0.289 | 0.8307 | 0.6113 | 172.0 |
| Ru | 1 | 7196 | 17815 | 0.289 | 0.8607 | 0.5823 | 173.7 |
| Ru | 2 | 7196 | 17815 | 0.289 | 0.8335 | 0.6613 | 168.5 |
| Ir | 0 | 19218 | 4458 | 0.493 | 0.9120 | 0.6261 | 395.6 |
| Ir | 1 | 19218 | 4458 | 0.493 | 0.9015 | 0.6878 | 455.9 |
| Ir | 2 | 19218 | 4458 | 0.493 | 0.9228 | 0.6315 | 514.9 |
| Os | 0 | 22308 | 1024 | 0.486 | 0.9072 | 0.8175 | 553.0 |
| Os | 1 | 22308 | 1024 | 0.486 | 0.9138 | 0.8212 | 509.5 |
| Os | 2 | 22308 | 1024 | 0.486 | 0.9059 | 0.8325 | 542.6 |
| Rh | 0 | 24662 | 572 | 0.420 | 0.6474 | 0.6489 | 542.3 |
| Rh | 1 | 24662 | 552 | 0.534 | 0.7013 | 0.6216 | 585.8 |
| Rh | 2 | 24662 | 535 | 0.445 | 0.7355 | 0.5883 | 539.6 |
| Re | 0 | 22451 | 865 | 0.279 | 0.9074 | 0.6255 | 250.5 |
| Re | 1 | 22451 | 865 | 0.279 | 0.9144 | 0.6140 | 238.8 |
| Re | 2 | 22451 | 865 | 0.279 | 0.9096 | 0.6220 | 217.7 |

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
