# Wf-125 Phase 2 — LOMO 5-fold re-bench with Wasserstein transfer + Rh oversample

**Date**: 2026-09-27  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1  
**Epochs/seed**: 6  
**Wasserstein weight**: 0.05  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  

## §1 Goal

TODO-121 + Wf-125 Phase 2 stacks three innovations on top of the
existing LOMO trainer:

1. **Wasserstein transfer regularisation** — sliced-W₂ over per-metal
   embeddings (canonical M1 primitive from
   `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`),
   added to the BCE+MSE loss with weight
   `0.05`.
2. **Rh closure-theorem oversampling** — the 500-compound Catherine 6/6
   PASS synthetic Rh cohort from Wf-125 P1 is appended to the train
   pool when the held-out metal is Rh.
3. **Multi-task D-MPNN backbone with per-metal embedding extraction** —
   the existing D-MPNN-MT backbone is reused; per-metal embeddings
   (pooled `2H`-dim vector before the readout head) are extracted
   every epoch end so the W₂ penalty can be computed per-metal.

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.4568 ± 0.1726**  
(n=10 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.1362 ± 0.0649** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5592 | 0.0062 | 2 |
| Ir | 0.5948 | 0.0116 | 2 |
| Os | 0.4300 | 0.0012 | 2 |
| Rh | 0.1362 | 0.0649 | 2 |
| Re | 0.5640 | 0.0106 | 2 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1892 | 3668 | 0.263 | 0.7547 | 0.5654 | 41.2 |
| Ru | 1 | 1892 | 3668 | 0.263 | 0.7476 | 0.5529 | 43.0 |
| Ir | 0 | 4076 | 1242 | 0.448 | 0.6440 | 0.6064 | 71.0 |
| Ir | 1 | 4076 | 1242 | 0.448 | 0.6805 | 0.5831 | 69.4 |
| Os | 0 | 5115 | 87 | 0.437 | 0.6989 | 0.4288 | 75.7 |
| Os | 1 | 5115 | 87 | 0.437 | 0.7098 | 0.4313 | 119.9 |
| Rh | 0 | 4646 | 608 | 0.058 | 0.5957 | 0.0713 | 123.2 |
| Rh | 1 | 4646 | 608 | 0.058 | 0.6691 | 0.2011 | 123.8 |
| Re | 0 | 5044 | 166 | 0.163 | 0.6678 | 0.5534 | 125.9 |
| Re | 1 | 5044 | 166 | 0.163 | 0.7065 | 0.5746 | 124.4 |

## §3 Honest framing (party-twenty-spirit §3.5)

**MEASURED here**: the numbers above are computed end-to-end on this
host with the chosen (epochs, seeds, wasserstein_weight) configuration.
No silent PROJECTED-to-MEASURED promotion.

**PROJECTED (TODO-121 §4)**: Rh LOMO AUC ≥ 0.50 (vs current 0.368);
overall LOMO mean ≥ 0.60 (vs current 0.535).

**Hit / miss summary**: Rh target MISS,
overall target MISS,
original gate BELOW GATE.

**Caveats**:

* The synthetic Rh cohort is *inactive* by default (MetalCytoToxDB
  Rh is ~85 % inactive at the IC50<10 µM gate). The oversample is
  a *chemistry* augmentation, not a *label* augmentation.
* The Wasserstein regulariser is injected every 5 epochs (not every
  batch) to keep the per-epoch wall-time reasonable.
* The W₂ snapshot at the end of training is recorded in
  `final_per_metal_w2` for diagnostic purposes (see per-seed JSON).

## §4 Cross-references

* TODO-121 m19 LOMO strengthen (parent plan)
* Wf-125 Phase 1 close-out (synthetic Rh cohort + W₂ building block)
* `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`
* `molmetal_lam.reward.closure_oversample.ClosureOversampler`
