# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-07  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 50  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: False  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

TODO-121 + Wf-125 Phase 2 stacks three innovations on top of the
existing LOMO trainer:

1. **Wasserstein transfer regularisation** — sliced-W₂ over per-metal
   embeddings (canonical M1 primitive from
   `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`),
   added to the BCE+MSE loss with weight
   `0.05`.
   **Wf-E1 (2026-10-05)**: this term is OFF unless
   `--wasserstein-real` is passed. Wf-C3 measured that the
   pre-Wf-E1 code detached the per-metal embeddings, so the W₂
   contributed ZERO gradient while costing 2.4x wall-time
   (Rh LOMO seeds 0-2, mean Δ test_AUC = −0.0023, within noise).
   `--wasserstein-real` keeps the graph so the W₂ really backprops;
   the per-epoch extraction is then evaluated at the epoch-start
   parameters (stale-direction pull within the injection epoch).
2. **Rh closure-theorem oversampling** — the 500-compound Catherine 6/6
   PASS synthetic Rh cohort from Wf-125 P1 is appended to the train
   pool when the held-out metal is Rh.
3. **Multi-task D-MPNN backbone with per-metal embedding extraction** —
   the existing D-MPNN-MT backbone is reused; per-metal embeddings
   (pooled `2H`-dim vector before the readout head) are extracted
   every epoch end so the W₂ penalty can be computed per-metal.
4. **Wf-127 P2 L2 grad-norm guard** (engaged by ``--use-l2-guard``) —
   clips grad-norm to ``≤ {l2_max:.1f}`` each step. Belt-and-braces
   against the W₂ penalty blowing up the optimisation; off by
   default to preserve Wf-125 P2 ship behaviour.
5. **Wf-127 P3 ASTSR scale clip** (engaged by ``--use-astr-scale-guard``)
   — value-space clip on predicted pIC50 at test time to
   ``[-10.0, 10.0]``.
   Catches explosive predictions the W₂ regulariser can produce
   early in training; off by default.

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.6824 ± 0.0858**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6437 ± 0.0310** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6178 | 0.0180 | 5 |
| Ir | 0.6351 | 0.0176 | 5 |
| Os | 0.8388 | 0.0192 | 5 |
| Rh | 0.6437 | 0.0310 | 5 |
| Re | 0.6765 | 0.0491 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 7351 | 18189 | 0.286 | 0.8250 | 0.6520 | 1283.2 |
| Ru | 1 | 7351 | 18189 | 0.286 | 0.8518 | 0.6077 | 1236.1 |
| Ru | 2 | 7351 | 18189 | 0.286 | 0.8479 | 0.6013 | 1031.8 |
| Ru | 3 | 7351 | 18189 | 0.286 | 0.8646 | 0.6194 | 979.5 |
| Ru | 4 | 7351 | 18189 | 0.286 | 0.8563 | 0.6089 | 903.3 |
| Ir | 0 | 19659 | 4514 | 0.492 | 0.9036 | 0.6392 | 2372.4 |
| Ir | 1 | 19659 | 4514 | 0.492 | 0.9064 | 0.6193 | 1986.4 |
| Ir | 2 | 19659 | 4514 | 0.492 | 0.9068 | 0.6524 | 2780.7 |
| Ir | 3 | 19659 | 4514 | 0.492 | 0.9183 | 0.6103 | 2423.5 |
| Ir | 4 | 19659 | 4514 | 0.492 | 0.9025 | 0.6545 | 2456.1 |
| Os | 0 | 22731 | 1100 | 0.481 | 0.8974 | 0.8085 | 2669.7 |
| Os | 1 | 22731 | 1100 | 0.481 | 0.9108 | 0.8421 | 3054.2 |
| Os | 2 | 22731 | 1100 | 0.481 | 0.9083 | 0.8588 | 3576.3 |
| Os | 3 | 22731 | 1100 | 0.481 | 0.9228 | 0.8582 | 3195.5 |
| Os | 4 | 22731 | 1100 | 0.481 | 0.9120 | 0.8266 | 2703.9 |
| Rh | 0 | 25171 | 530 | 0.419 | 0.6732 | 0.6459 | 3368.4 |
| Rh | 1 | 25171 | 537 | 0.417 | 0.6441 | 0.6631 | 3436.3 |
| Rh | 2 | 25171 | 590 | 0.478 | 0.6444 | 0.5880 | 4206.1 |
| Rh | 3 | 25171 | 573 | 0.482 | 0.6300 | 0.6799 | 3170.9 |
| Rh | 4 | 25171 | 556 | 0.451 | 0.6413 | 0.6414 | 3403.1 |
| Re | 0 | 22940 | 868 | 0.280 | 0.9099 | 0.6035 | 2781.8 |
| Re | 1 | 22940 | 868 | 0.280 | 0.9170 | 0.6702 | 2066.8 |
| Re | 2 | 22940 | 868 | 0.280 | 0.9086 | 0.6481 | 2113.6 |
| Re | 3 | 22940 | 868 | 0.280 | 0.9255 | 0.7370 | 2209.9 |
| Re | 4 | 22940 | 868 | 0.280 | 0.9087 | 0.7238 | 2679.7 |

## §3 Honest framing (party-twenty-spirit §3.5)

**MEASURED here**: the numbers above are computed end-to-end on this
host with the chosen (epochs, seeds, wasserstein_weight) configuration.
No silent PROJECTED-to-MEASURED promotion.

**PROJECTED (TODO-121 §4)**: Rh LOMO AUC ≥ 0.50 (vs current 0.368);
overall LOMO mean ≥ 0.60 (vs current 0.535).

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
