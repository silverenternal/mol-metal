# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-08  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Wasserstein weight (requested)**: 0.0  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 0 (0 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: True  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

TODO-121 + Wf-125 Phase 2 stacks three innovations on top of the
existing LOMO trainer:

1. **Wasserstein transfer regularisation** — sliced-W₂ over per-metal
   embeddings (canonical M1 primitive from
   `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`),
   added to the BCE+MSE loss with weight
   `0.0`.
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

**Overall LOMO mean test AUC = 0.5845 ± 0.0764**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.5022 ± 0.0613** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5618 | 0.0164 | 5 |
| Ir | 0.5514 | 0.0442 | 5 |
| Os | 0.6462 | 0.0428 | 5 |
| Rh | 0.5022 | 0.0613 | 5 |
| Re | 0.6609 | 0.0580 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7799 | 0.5429 | 48.2 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.7972 | 0.5485 | 38.0 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7796 | 0.5821 | 41.1 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7632 | 0.5805 | 35.3 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.8115 | 0.5548 | 25.1 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.8276 | 0.5070 | 50.5 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.8232 | 0.4950 | 48.0 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.8165 | 0.6064 | 50.1 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7922 | 0.5578 | 47.5 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.8028 | 0.5906 | 47.5 |
| Os | 0 | 4665 | 87 | 0.437 | 0.8426 | 0.6380 | 61.3 |
| Os | 1 | 4665 | 87 | 0.437 | 0.8384 | 0.7078 | 52.1 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7983 | 0.6633 | 52.3 |
| Os | 3 | 4665 | 87 | 0.437 | 0.8092 | 0.6463 | 53.6 |
| Os | 4 | 4665 | 87 | 0.437 | 0.8203 | 0.5755 | 50.1 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8492 | 0.4591 | 53.3 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.8420 | 0.5184 | 57.4 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7984 | 0.4663 | 52.1 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.8283 | 0.6155 | 52.1 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.8393 | 0.4515 | 50.4 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8641 | 0.5922 | 52.0 |
| Re | 1 | 4594 | 166 | 0.163 | 0.8379 | 0.6809 | 50.1 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7894 | 0.6259 | 56.7 |
| Re | 3 | 4594 | 166 | 0.163 | 0.8105 | 0.6438 | 67.5 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7836 | 0.7618 | 83.0 |

## §3 Honest framing (party-twenty-spirit §3.5)

**MEASURED here**: the numbers above are computed end-to-end on this
host with the chosen (epochs, seeds, wasserstein_weight) configuration.
No silent PROJECTED-to-MEASURED promotion.

**PROJECTED (TODO-121 §4)**: Rh LOMO AUC ≥ 0.50 (vs current 0.368);
overall LOMO mean ≥ 0.60 (vs current 0.535).

**Hit / miss summary**: Rh target HIT,
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
