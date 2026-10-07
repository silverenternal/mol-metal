# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-07  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2  
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
