# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-06  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 50  
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

**Overall LOMO mean test AUC = 0.6791 ± 0.0869**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6480 ± 0.0575** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6420 | 0.0313 | 5 |
| Ir | 0.6148 | 0.0282 | 5 |
| Os | 0.8247 | 0.0178 | 5 |
| Rh | 0.6480 | 0.0575 | 5 |
| Re | 0.6658 | 0.0673 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 7351 | 18189 | 0.286 | 0.8245 | 0.6820 | 65.6 |
| Ru | 1 | 7351 | 18189 | 0.286 | 0.8595 | 0.6734 | 58.6 |
| Ru | 2 | 7351 | 18189 | 0.286 | 0.8467 | 0.6011 | 65.7 |
| Ru | 3 | 7351 | 18189 | 0.286 | 0.8832 | 0.6175 | 68.0 |
| Ru | 4 | 7351 | 18189 | 0.286 | 0.8642 | 0.6362 | 61.9 |
| Ir | 0 | 19659 | 4514 | 0.492 | 0.9102 | 0.6200 | 120.6 |
| Ir | 1 | 19659 | 4514 | 0.492 | 0.9130 | 0.5760 | 140.7 |
| Ir | 2 | 19659 | 4514 | 0.492 | 0.9127 | 0.6597 | 141.0 |
| Ir | 3 | 19659 | 4514 | 0.492 | 0.9175 | 0.5958 | 130.3 |
| Ir | 4 | 19659 | 4514 | 0.492 | 0.9123 | 0.6227 | 144.5 |
| Os | 0 | 22731 | 1100 | 0.481 | 0.9032 | 0.8178 | 157.5 |
| Os | 1 | 22731 | 1100 | 0.481 | 0.9169 | 0.8214 | 155.5 |
| Os | 2 | 22731 | 1100 | 0.481 | 0.9136 | 0.8053 | 164.7 |
| Os | 3 | 22731 | 1100 | 0.481 | 0.9315 | 0.8583 | 147.2 |
| Os | 4 | 22731 | 1100 | 0.481 | 0.9135 | 0.8205 | 154.2 |
| Rh | 0 | 25171 | 530 | 0.419 | 0.6239 | 0.6800 | 164.5 |
| Rh | 1 | 25171 | 537 | 0.417 | 0.5780 | 0.6881 | 172.1 |
| Rh | 2 | 25171 | 590 | 0.478 | 0.6579 | 0.5935 | 141.1 |
| Rh | 3 | 25171 | 573 | 0.482 | 0.6672 | 0.7124 | 123.8 |
| Rh | 4 | 25171 | 556 | 0.451 | 0.6567 | 0.5657 | 159.9 |
| Re | 0 | 22940 | 868 | 0.280 | 0.9124 | 0.6351 | 160.9 |
| Re | 1 | 22940 | 868 | 0.280 | 0.9165 | 0.7117 | 195.6 |
| Re | 2 | 22940 | 868 | 0.280 | 0.9064 | 0.6921 | 351.3 |
| Re | 3 | 22940 | 868 | 0.280 | 0.9216 | 0.5502 | 316.7 |
| Re | 4 | 22940 | 868 | 0.280 | 0.9133 | 0.7402 | 400.3 |

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
