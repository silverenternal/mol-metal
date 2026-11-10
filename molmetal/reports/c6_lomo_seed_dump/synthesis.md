# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-20  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: /home/hugo/codes/mol-metal/molmetal/data/metal_lomo.csv  
**Time(h) filter threshold**: 24.0  
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

**Overall LOMO mean test AUC = 0.5406 ± 0.0936**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3705 ± 0.0191** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5885 | 0.0179 | 5 |
| Ir | 0.5908 | 0.0190 | 5 |
| Os | 0.5808 | 0.0304 | 5 |
| Rh | 0.3705 | 0.0191 | 5 |
| Re | 0.5721 | 0.0742 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7343 | 0.6157 | 18.7 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.8152 | 0.5664 | 12.1 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7537 | 0.5985 | 12.1 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7485 | 0.5722 | 10.8 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7427 | 0.5897 | 10.2 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7901 | 0.5760 | 19.4 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7955 | 0.6121 | 18.6 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7539 | 0.5614 | 17.8 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7591 | 0.6030 | 21.1 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7963 | 0.6014 | 21.4 |
| Os | 0 | 4665 | 87 | 0.437 | 0.7915 | 0.5674 | 26.5 |
| Os | 1 | 4665 | 87 | 0.437 | 0.8043 | 0.5580 | 21.3 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7518 | 0.5967 | 25.8 |
| Os | 3 | 4665 | 87 | 0.437 | 0.7836 | 0.6327 | 19.7 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7876 | 0.5494 | 24.2 |
| Rh | 0 | 4646 | 108 | 0.324 | 0.8122 | 0.3841 | 22.9 |
| Rh | 1 | 4646 | 108 | 0.324 | 0.8228 | 0.3769 | 22.6 |
| Rh | 2 | 4646 | 108 | 0.324 | 0.7322 | 0.3755 | 23.3 |
| Rh | 3 | 4646 | 108 | 0.324 | 0.7962 | 0.3832 | 24.4 |
| Rh | 4 | 4646 | 108 | 0.324 | 0.7817 | 0.3329 | 23.9 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8082 | 0.4592 | 18.6 |
| Re | 1 | 4594 | 166 | 0.163 | 0.7943 | 0.6864 | 25.2 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7583 | 0.5905 | 22.7 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7728 | 0.5374 | 22.3 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7430 | 0.5871 | 19.8 |

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
