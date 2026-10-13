# LOMO 5-fold × 5-seed × 50-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-06  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 50  
**Cohort CSV**: .data/molmetal/MetalCytoToxDB.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.0  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 0 (0 actually appended)  
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
2. **Rh closure-theorem oversampling — OFF** (requested 0; 0 rows appended).
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — OFF** (threshold 5.0; engage with `--use-l2-guard`).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.6747 ± 0.0803**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6356 ± 0.0353** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6429 | 0.0217 | 5 |
| Ir | 0.6055 | 0.0166 | 5 |
| Os | 0.8174 | 0.0189 | 5 |
| Rh | 0.6356 | 0.0353 | 5 |
| Re | 0.6722 | 0.0464 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 6851 | 18189 | 0.286 | 0.8252 | 0.6355 | 67.1 |
| Ru | 1 | 6851 | 18189 | 0.286 | 0.8561 | 0.6667 | 62.6 |
| Ru | 2 | 6851 | 18189 | 0.286 | 0.8559 | 0.6096 | 56.3 |
| Ru | 3 | 6851 | 18189 | 0.286 | 0.8665 | 0.6357 | 61.2 |
| Ru | 4 | 6851 | 18189 | 0.286 | 0.8438 | 0.6670 | 67.9 |
| Ir | 0 | 19159 | 4514 | 0.492 | 0.9138 | 0.6317 | 219.5 |
| Ir | 1 | 19159 | 4514 | 0.492 | 0.9207 | 0.6109 | 283.9 |
| Ir | 2 | 19159 | 4514 | 0.492 | 0.9108 | 0.5903 | 269.1 |
| Ir | 3 | 19159 | 4514 | 0.492 | 0.9238 | 0.6096 | 311.4 |
| Ir | 4 | 19159 | 4514 | 0.492 | 0.9150 | 0.5850 | 354.6 |
| Os | 0 | 22231 | 1100 | 0.481 | 0.9116 | 0.8273 | 521.2 |
| Os | 1 | 22231 | 1100 | 0.481 | 0.9176 | 0.8141 | 453.2 |
| Os | 2 | 22231 | 1100 | 0.481 | 0.9173 | 0.8445 | 471.6 |
| Os | 3 | 22231 | 1100 | 0.481 | 0.9247 | 0.8144 | 542.0 |
| Os | 4 | 22231 | 1100 | 0.481 | 0.9216 | 0.7869 | 569.9 |
| Rh | 0 | 24671 | 530 | 0.419 | 0.5949 | 0.6469 | 559.1 |
| Rh | 1 | 24671 | 537 | 0.417 | 0.6353 | 0.6549 | 586.3 |
| Rh | 2 | 24671 | 590 | 0.478 | 0.6345 | 0.5996 | 583.8 |
| Rh | 3 | 24671 | 573 | 0.482 | 0.6236 | 0.6855 | 558.4 |
| Rh | 4 | 24671 | 556 | 0.451 | 0.6329 | 0.5913 | 537.8 |
| Re | 0 | 22440 | 868 | 0.280 | 0.9118 | 0.6730 | 343.1 |
| Re | 1 | 22440 | 868 | 0.280 | 0.9231 | 0.7537 | 231.7 |
| Re | 2 | 22440 | 868 | 0.280 | 0.9214 | 0.6146 | 228.0 |
| Re | 3 | 22440 | 868 | 0.280 | 0.9256 | 0.6756 | 205.8 |
| Re | 4 | 22440 | 868 | 0.280 | 0.9184 | 0.6441 | 306.9 |

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
