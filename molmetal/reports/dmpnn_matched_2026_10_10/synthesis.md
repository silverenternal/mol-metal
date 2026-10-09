# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-10  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Wasserstein weight (requested)**: 0.05  
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

**Overall LOMO mean test AUC = 0.5063 ± 0.0999**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3965 ± 0.0216** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5662 | 0.0343 | 5 |
| Ir | 0.5734 | 0.0139 | 5 |
| Os | 0.5806 | 0.0253 | 5 |
| Rh | 0.3965 | 0.0216 | 5 |
| Re | 0.4148 | 0.1156 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 610 | 1255 | 0.257 | 0.8089 | 0.5863 | 24.7 |
| Ru | 1 | 610 | 1255 | 0.257 | 0.5652 | 0.5307 | 9.3 |
| Ru | 2 | 610 | 1255 | 0.257 | 0.6946 | 0.5225 | 10.7 |
| Ru | 3 | 610 | 1255 | 0.257 | 0.7054 | 0.5789 | 9.4 |
| Ru | 4 | 610 | 1255 | 0.257 | 0.7357 | 0.6124 | 9.9 |
| Ir | 0 | 1251 | 534 | 0.453 | 0.6846 | 0.5916 | 13.4 |
| Ir | 1 | 1251 | 534 | 0.453 | 0.5498 | 0.5769 | 14.9 |
| Ir | 2 | 1251 | 534 | 0.453 | 0.7519 | 0.5542 | 17.9 |
| Ir | 3 | 1251 | 534 | 0.453 | 0.6808 | 0.5835 | 16.0 |
| Ir | 4 | 1251 | 534 | 0.453 | 0.7163 | 0.5608 | 11.7 |
| Os | 0 | 1707 | 36 | 0.361 | 0.7753 | 0.5585 | 17.1 |
| Os | 1 | 1707 | 36 | 0.361 | 0.6333 | 0.5619 | 19.8 |
| Os | 2 | 1707 | 36 | 0.361 | 0.6940 | 0.5920 | 13.8 |
| Os | 3 | 1707 | 36 | 0.361 | 0.7471 | 0.6254 | 17.2 |
| Os | 4 | 1707 | 36 | 0.361 | 0.7323 | 0.5652 | 17.2 |
| Rh | 0 | 1686 | 51 | 0.333 | 0.7377 | 0.4308 | 14.4 |
| Rh | 1 | 1686 | 51 | 0.333 | 0.6704 | 0.3720 | 13.1 |
| Rh | 2 | 1686 | 51 | 0.333 | 0.7703 | 0.3754 | 19.3 |
| Rh | 3 | 1686 | 51 | 0.333 | 0.7585 | 0.4066 | 22.6 |
| Rh | 4 | 1686 | 51 | 0.333 | 0.7004 | 0.3979 | 25.6 |
| Re | 0 | 1688 | 74 | 0.189 | 0.7460 | 0.2107 | 14.9 |
| Re | 1 | 1688 | 74 | 0.189 | 0.6474 | 0.4024 | 18.0 |
| Re | 2 | 1688 | 74 | 0.189 | 0.7525 | 0.5631 | 17.5 |
| Re | 3 | 1688 | 74 | 0.189 | 0.7365 | 0.4298 | 19.9 |
| Re | 4 | 1688 | 74 | 0.189 | 0.7220 | 0.4679 | 13.8 |

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
