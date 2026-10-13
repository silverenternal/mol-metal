# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-08  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: molmetal/data/metal_lomo.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
**Rh oversample**: 0 (0 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: True  
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
2. **Rh closure-theorem oversampling — OFF** (requested 0; 0 rows appended).
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — ON** (clips grad-norm to ≤ 5.0 each step).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.5349 ± 0.0895**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3794 ± 0.0255** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5888 | 0.0123 | 5 |
| Ir | 0.5901 | 0.0247 | 5 |
| Os | 0.5560 | 0.0186 | 5 |
| Rh | 0.3794 | 0.0255 | 5 |
| Re | 0.5603 | 0.0841 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7250 | 0.5885 | 162.1 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.8050 | 0.5718 | 154.6 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7530 | 0.6056 | 136.5 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7486 | 0.5989 | 134.4 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7362 | 0.5794 | 137.5 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7832 | 0.5559 | 283.0 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7843 | 0.5650 | 236.4 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7835 | 0.6134 | 223.6 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7700 | 0.6033 | 216.0 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7942 | 0.6130 | 224.8 |
| Os | 0 | 4665 | 87 | 0.437 | 0.8074 | 0.5650 | 355.9 |
| Os | 1 | 4665 | 87 | 0.437 | 0.8106 | 0.5419 | 298.4 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7593 | 0.5271 | 331.0 |
| Os | 3 | 4665 | 87 | 0.437 | 0.8018 | 0.5768 | 312.3 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7879 | 0.5693 | 351.5 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8102 | 0.3959 | 365.9 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.8234 | 0.3885 | 329.4 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7366 | 0.3765 | 346.1 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.7857 | 0.4043 | 348.7 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.8051 | 0.3319 | 357.0 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8105 | 0.5099 | 301.9 |
| Re | 1 | 4594 | 166 | 0.163 | 0.8025 | 0.6960 | 294.1 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7533 | 0.6018 | 240.7 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7737 | 0.4487 | 222.8 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7504 | 0.5453 | 299.2 |

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
