# Wf-131 Phase 3 — LOMO 5-fold × 5-seed × 50-epoch full re-bench on MetalCytoToxDB 5-metal

**Date**: 2026-10-05  
**Metals**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Wasserstein weight**: 0.05  
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

**Overall LOMO mean test AUC = 0.5574 ± 0.0922**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3864 ± 0.0197** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5862 | 0.0158 | 5 |
| Ir | 0.6078 | 0.0145 | 5 |
| Os | 0.5827 | 0.0286 | 5 |
| Rh | 0.3864 | 0.0197 | 5 |
| Re | 0.6237 | 0.0569 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1942 | 3668 | 0.263 | 0.6874 | 0.5773 | 41.3 |
| Ru | 1 | 1942 | 3668 | 0.263 | 0.7628 | 0.5717 | 38.6 |
| Ru | 2 | 1942 | 3668 | 0.263 | 0.7424 | 0.5810 | 35.0 |
| Ru | 3 | 1942 | 3668 | 0.263 | 0.7172 | 0.5841 | 35.3 |
| Ru | 4 | 1942 | 3668 | 0.263 | 0.7204 | 0.6167 | 36.2 |
| Ir | 0 | 4126 | 1242 | 0.448 | 0.7873 | 0.6101 | 64.6 |
| Ir | 1 | 4126 | 1242 | 0.448 | 0.7499 | 0.6137 | 69.4 |
| Ir | 2 | 4126 | 1242 | 0.448 | 0.7423 | 0.5833 | 66.0 |
| Ir | 3 | 4126 | 1242 | 0.448 | 0.7450 | 0.6042 | 63.4 |
| Ir | 4 | 4126 | 1242 | 0.448 | 0.7574 | 0.6278 | 66.3 |
| Os | 0 | 5165 | 87 | 0.437 | 0.7858 | 0.5585 | 82.6 |
| Os | 1 | 5165 | 87 | 0.437 | 0.7812 | 0.6063 | 83.5 |
| Os | 2 | 5165 | 87 | 0.437 | 0.7493 | 0.6071 | 93.4 |
| Os | 3 | 5165 | 87 | 0.437 | 0.7637 | 0.6028 | 114.9 |
| Os | 4 | 5165 | 87 | 0.437 | 0.7600 | 0.5387 | 113.6 |
| Rh | 0 | 5146 | 108 | 0.324 | 0.7848 | 0.3838 | 116.5 |
| Rh | 1 | 5146 | 108 | 0.324 | 0.7661 | 0.4237 | 105.8 |
| Rh | 2 | 5146 | 108 | 0.324 | 0.7279 | 0.3693 | 101.3 |
| Rh | 3 | 5146 | 108 | 0.324 | 0.7734 | 0.3845 | 110.2 |
| Rh | 4 | 5146 | 108 | 0.324 | 0.7718 | 0.3708 | 111.6 |
| Re | 0 | 5094 | 166 | 0.163 | 0.7769 | 0.5883 | 107.1 |
| Re | 1 | 5094 | 166 | 0.163 | 0.7750 | 0.5661 | 108.3 |
| Re | 2 | 5094 | 166 | 0.163 | 0.7470 | 0.6182 | 96.9 |
| Re | 3 | 5094 | 166 | 0.163 | 0.7596 | 0.6148 | 106.3 |
| Re | 4 | 5094 | 166 | 0.163 | 0.7217 | 0.7310 | 107.3 |

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
