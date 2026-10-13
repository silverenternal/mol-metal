# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-11  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: molmetal/data/metal_lomo.csv  
**Time(h) filter threshold**: 96  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 0 (0 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: True  
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
4. **Wf-127 P2 L2 grad-norm guard — ON** (clips grad-norm to ≤ 5.0 each step).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.5186 ± 0.0910**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3845 ± 0.0077** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5966 | 0.0222 | 5 |
| Ir | 0.5798 | 0.0251 | 5 |
| Os | 0.5376 | 0.0895 | 5 |
| Rh | 0.3845 | 0.0077 | 5 |
| Re | 0.4944 | 0.0588 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7405 | 0.6200 | 26.4 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.8076 | 0.5837 | 10.3 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7604 | 0.6260 | 7.9 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7428 | 0.5832 | 7.0 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7359 | 0.5702 | 7.2 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7987 | 0.5802 | 16.0 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7896 | 0.5972 | 17.7 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7812 | 0.5961 | 15.1 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7667 | 0.5311 | 13.1 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7846 | 0.5944 | 17.1 |
| Os | 0 | 4665 | 87 | 0.437 | 0.8014 | 0.5736 | 14.5 |
| Os | 1 | 4665 | 87 | 0.437 | 0.7421 | 0.3617 | 12.8 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7639 | 0.5854 | 15.4 |
| Os | 3 | 4665 | 87 | 0.437 | 0.7940 | 0.5585 | 22.6 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7753 | 0.6090 | 19.4 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8119 | 0.3947 | 19.5 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.8324 | 0.3894 | 20.6 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7353 | 0.3847 | 16.3 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.7926 | 0.3814 | 19.6 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.7852 | 0.3720 | 21.1 |
| Re | 0 | 4594 | 166 | 0.163 | 0.7980 | 0.4259 | 15.2 |
| Re | 1 | 4594 | 166 | 0.163 | 0.8048 | 0.4964 | 15.1 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7548 | 0.5045 | 10.9 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7697 | 0.5965 | 14.6 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7346 | 0.4486 | 10.8 |

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

**Hit / miss summary**: Rh target MISS,
overall target MISS,
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
