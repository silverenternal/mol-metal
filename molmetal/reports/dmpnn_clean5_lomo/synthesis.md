# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-08  
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

**Overall LOMO mean test AUC = 0.5575 ± 0.0949**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3812 ± 0.0270** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5972 | 0.0121 | 5 |
| Ir | 0.5975 | 0.0186 | 5 |
| Os | 0.5831 | 0.0285 | 5 |
| Rh | 0.3812 | 0.0270 | 5 |
| Re | 0.6288 | 0.0551 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7282 | 0.6142 | 54.3 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.8049 | 0.5906 | 45.1 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7467 | 0.6075 | 30.1 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7476 | 0.5806 | 34.4 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7388 | 0.5928 | 31.3 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7860 | 0.5797 | 56.9 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7999 | 0.6277 | 74.8 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7659 | 0.6041 | 80.8 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7437 | 0.5763 | 71.4 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7852 | 0.5999 | 60.8 |
| Os | 0 | 4665 | 87 | 0.437 | 0.7871 | 0.5698 | 77.2 |
| Os | 1 | 4665 | 87 | 0.437 | 0.7824 | 0.5738 | 45.5 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7633 | 0.5701 | 23.1 |
| Os | 3 | 4665 | 87 | 0.437 | 0.7957 | 0.6396 | 26.3 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7773 | 0.5623 | 25.3 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8125 | 0.3781 | 24.5 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.7679 | 0.4309 | 27.9 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7207 | 0.3526 | 21.7 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.7887 | 0.3814 | 21.2 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.7923 | 0.3628 | 19.2 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8103 | 0.5697 | 18.4 |
| Re | 1 | 4594 | 166 | 0.163 | 0.8042 | 0.6723 | 22.9 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7439 | 0.6577 | 19.4 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7717 | 0.6886 | 18.6 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7472 | 0.5554 | 16.6 |

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
