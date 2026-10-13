# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-12  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: .data/molmetal/MetalCytoToxDB.csv  
**Time(h) filter threshold**: 96  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 0 (0 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: True  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

Leave-one-metal-out (LOMO) cross-metal transfer benchmark: hold out
one metal's rows entirely, train a multi-task D-MPNN on the rest, and
measure ROC-AUC on the held-out metal (the m19 protocol). The active
components are echoed from the config below — not asserted.

1. **Wasserstein transfer regularisation — OFF** (weight 0.0).
   The objective is BCE + L2 guard only. Do NOT read this run as a
   Wasserstein result: per Wf-E1 the term requires BOTH a non-zero
   `wasserstein_weight` AND `--wasserstein-real`.
2. **Rh closure-theorem oversampling — OFF** (requested 0; 0 rows appended).
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — ON** (clips grad-norm to ≤ 5.0 each step).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.6855 ± 0.0720**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.6261 ± 0.0103** (target ≥ 0.5)  
**Verdict (Rh target)**: HIT  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6422 | 0.0175 | 5 |
| Ir | 0.6414 | 0.0207 | 5 |
| Os | 0.8161 | 0.0203 | 5 |
| Rh | 0.6261 | 0.0103 | 5 |
| Re | 0.7018 | 0.0056 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 6851 | 18189 | 0.286 | 0.7786 | 0.6100 | 53.6 |
| Ru | 1 | 6851 | 18189 | 0.286 | 0.8185 | 0.6585 | 47.5 |
| Ru | 2 | 6851 | 18189 | 0.286 | 0.8383 | 0.6452 | 49.8 |
| Ru | 3 | 6851 | 18189 | 0.286 | 0.8124 | 0.6405 | 50.1 |
| Ru | 4 | 6851 | 18189 | 0.286 | 0.8196 | 0.6568 | 49.5 |
| Ir | 0 | 19159 | 4514 | 0.492 | 0.9024 | 0.6064 | 90.7 |
| Ir | 1 | 19159 | 4514 | 0.492 | 0.8966 | 0.6594 | 90.1 |
| Ir | 2 | 19159 | 4514 | 0.492 | 0.9009 | 0.6331 | 89.8 |
| Ir | 3 | 19159 | 4514 | 0.492 | 0.9077 | 0.6437 | 89.9 |
| Ir | 4 | 19159 | 4514 | 0.492 | 0.9075 | 0.6643 | 90.4 |
| Os | 0 | 22231 | 1100 | 0.481 | 0.8905 | 0.7836 | 99.4 |
| Os | 1 | 22231 | 1100 | 0.481 | 0.8957 | 0.8350 | 100.4 |
| Os | 2 | 22231 | 1100 | 0.481 | 0.8941 | 0.8174 | 100.7 |
| Os | 3 | 22231 | 1100 | 0.481 | 0.9156 | 0.8052 | 100.2 |
| Os | 4 | 22231 | 1100 | 0.481 | 0.9073 | 0.8392 | 101.1 |
| Rh | 0 | 22203 | 1131 | 0.468 | 0.8970 | 0.6429 | 100.2 |
| Rh | 1 | 22203 | 1131 | 0.468 | 0.8997 | 0.6264 | 101.0 |
| Rh | 2 | 22203 | 1131 | 0.468 | 0.9081 | 0.6140 | 98.4 |
| Rh | 3 | 22203 | 1131 | 0.468 | 0.9203 | 0.6301 | 100.5 |
| Rh | 4 | 22203 | 1131 | 0.468 | 0.9124 | 0.6170 | 99.6 |
| Re | 0 | 22440 | 868 | 0.280 | 0.9015 | 0.6984 | 101.1 |
| Re | 1 | 22440 | 868 | 0.280 | 0.9118 | 0.6984 | 101.1 |
| Re | 2 | 22440 | 868 | 0.280 | 0.8975 | 0.6966 | 100.9 |
| Re | 3 | 22440 | 868 | 0.280 | 0.9145 | 0.7036 | 100.9 |
| Re | 4 | 22440 | 868 | 0.280 | 0.9054 | 0.7119 | 100.9 |

## §3 Honest framing (party-twenty-spirit §3.5)

**MEASURED here**: the numbers above are computed end-to-end on this
host with the chosen (epochs, seeds, wasserstein_weight) configuration.
No silent PROJECTED-to-MEASURED promotion.

**PROJECTED (TODO-121 §4)**: Rh LOMO AUC ≥ 0.50; overall LOMO mean
≥ 0.60.

**Baseline reference (MEASURED, pre-repair ≤24 h cohort)**: Rh 0.3812,
overall 0.5575 — see
`molmetal/reports/wf_a2_m19_rh_diagnosis/diagnosis.json`. That baseline
used a DIFFERENT cohort (`molmetal/data/metal_lomo.csv`), so any delta
must be computed against the same cohort; the cohort CSV for THIS run
is echoed in the header above.

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
