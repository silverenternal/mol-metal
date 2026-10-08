# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-09  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
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

**Overall LOMO mean test AUC = 0.5496 ± 0.0967**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3780 ± 0.0160** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6031 | 0.0112 | 5 |
| Ir | 0.5896 | 0.0236 | 5 |
| Os | 0.5849 | 0.0231 | 5 |
| Rh | 0.3780 | 0.0160 | 5 |
| Re | 0.5924 | 0.0909 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7351 | 0.6210 | 40.8 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.7999 | 0.5926 | 28.6 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7609 | 0.6114 | 27.3 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7427 | 0.5956 | 27.6 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7415 | 0.5949 | 29.8 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7799 | 0.5548 | 76.9 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7741 | 0.6288 | 82.3 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7777 | 0.5833 | 48.5 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7672 | 0.5911 | 46.0 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7847 | 0.5901 | 67.1 |
| Os | 0 | 4665 | 87 | 0.437 | 0.8058 | 0.5851 | 78.6 |
| Os | 1 | 4665 | 87 | 0.437 | 0.8136 | 0.5666 | 57.5 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7530 | 0.5540 | 58.6 |
| Os | 3 | 4665 | 87 | 0.437 | 0.7747 | 0.6187 | 57.9 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7807 | 0.5999 | 91.1 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8070 | 0.3843 | 100.8 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.8090 | 0.3802 | 72.4 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7223 | 0.3671 | 68.1 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.7860 | 0.4027 | 86.5 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.7898 | 0.3554 | 84.0 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8024 | 0.5518 | 78.9 |
| Re | 1 | 4594 | 166 | 0.163 | 0.7991 | 0.7518 | 96.8 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7471 | 0.6188 | 99.7 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7744 | 0.5584 | 98.7 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7554 | 0.4809 | 99.8 |

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
