# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-06  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Wasserstein weight (requested)**: 0.0  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
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

**Overall LOMO mean test AUC = 0.5594 ± 0.0958**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.4046 ± 0.0704** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5961 | 0.0140 | 5 |
| Ir | 0.6081 | 0.0221 | 5 |
| Os | 0.5786 | 0.0191 | 5 |
| Rh | 0.4046 | 0.0704 | 5 |
| Re | 0.6098 | 0.0962 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1942 | 3668 | 0.263 | 0.6841 | 0.6004 | 12.9 |
| Ru | 1 | 1942 | 3668 | 0.263 | 0.7387 | 0.5801 | 11.7 |
| Ru | 2 | 1942 | 3668 | 0.263 | 0.7309 | 0.6208 | 11.1 |
| Ru | 3 | 1942 | 3668 | 0.263 | 0.7369 | 0.5905 | 10.3 |
| Ru | 4 | 1942 | 3668 | 0.263 | 0.7177 | 0.5885 | 10.9 |
| Ir | 0 | 4126 | 1242 | 0.448 | 0.7653 | 0.5804 | 18.7 |
| Ir | 1 | 4126 | 1242 | 0.448 | 0.7460 | 0.6272 | 17.7 |
| Ir | 2 | 4126 | 1242 | 0.448 | 0.7384 | 0.5817 | 21.5 |
| Ir | 3 | 4126 | 1242 | 0.448 | 0.7423 | 0.6273 | 20.7 |
| Ir | 4 | 4126 | 1242 | 0.448 | 0.7386 | 0.6240 | 18.2 |
| Os | 0 | 5165 | 87 | 0.437 | 0.7853 | 0.5776 | 22.1 |
| Os | 1 | 5165 | 87 | 0.437 | 0.7501 | 0.5870 | 23.9 |
| Os | 2 | 5165 | 87 | 0.437 | 0.7442 | 0.5720 | 24.9 |
| Os | 3 | 5165 | 87 | 0.437 | 0.7706 | 0.5489 | 24.3 |
| Os | 4 | 5165 | 87 | 0.437 | 0.7508 | 0.6074 | 25.7 |
| Rh | 0 | 5663 | 48 | 0.250 | 0.4036 | 0.4514 | 23.0 |
| Rh | 1 | 5663 | 58 | 0.310 | 0.3538 | 0.4146 | 26.5 |
| Rh | 2 | 5663 | 49 | 0.265 | 0.4005 | 0.3419 | 24.2 |
| Rh | 3 | 5663 | 51 | 0.275 | 0.3823 | 0.5039 | 26.6 |
| Rh | 4 | 5663 | 60 | 0.317 | 0.3916 | 0.3113 | 23.3 |
| Re | 0 | 5094 | 166 | 0.163 | 0.7870 | 0.6458 | 23.1 |
| Re | 1 | 5094 | 166 | 0.163 | 0.7644 | 0.4760 | 20.9 |
| Re | 2 | 5094 | 166 | 0.163 | 0.7320 | 0.6932 | 19.2 |
| Re | 3 | 5094 | 166 | 0.163 | 0.7389 | 0.5171 | 21.7 |
| Re | 4 | 5094 | 166 | 0.163 | 0.7231 | 0.7170 | 24.6 |

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
