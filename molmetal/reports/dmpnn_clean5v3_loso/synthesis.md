# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-06  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: molmetal/data/metal_lomo.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.0  
**W₂ actually applied to the gradient (Wf-E1)**: 0.0  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: False  
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
2. **Rh closure-theorem oversampling — ON** (500 synthetic
   Rh rows appended to the Rh train pool; Wf-125 P1 Catherine 6/6
   PASS cohort). It is *inactive* by design (MetalCytoToxDB Rh is
   ~85 % inactive at the IC50<10 µM gate) — a *chemistry*
   augmentation, not a *label* augmentation.
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — OFF** (threshold 5.0; engage with `--use-l2-guard`).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.5435 ± 0.1131**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.3380 ± 0.0313** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5931 | 0.0280 | 5 |
| Ir | 0.5845 | 0.0148 | 5 |
| Os | 0.6140 | 0.0749 | 5 |
| Rh | 0.3380 | 0.0313 | 5 |
| Re | 0.5881 | 0.0555 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1942 | 3668 | 0.263 | 0.6919 | 0.6077 | 20.5 |
| Ru | 1 | 1942 | 3668 | 0.263 | 0.7457 | 0.5918 | 11.4 |
| Ru | 2 | 1942 | 3668 | 0.263 | 0.6871 | 0.5672 | 12.9 |
| Ru | 3 | 1942 | 3668 | 0.263 | 0.6969 | 0.5610 | 12.8 |
| Ru | 4 | 1942 | 3668 | 0.263 | 0.6774 | 0.6377 | 11.4 |
| Ir | 0 | 4126 | 1242 | 0.448 | 0.8081 | 0.5859 | 20.4 |
| Ir | 1 | 4126 | 1242 | 0.448 | 0.7520 | 0.5663 | 19.3 |
| Ir | 2 | 4126 | 1242 | 0.448 | 0.7278 | 0.5770 | 20.1 |
| Ir | 3 | 4126 | 1242 | 0.448 | 0.7344 | 0.5824 | 19.9 |
| Ir | 4 | 4126 | 1242 | 0.448 | 0.7735 | 0.6111 | 19.5 |
| Os | 0 | 5165 | 87 | 0.437 | 0.7796 | 0.5344 | 25.3 |
| Os | 1 | 5165 | 87 | 0.437 | 0.7719 | 0.7470 | 25.1 |
| Os | 2 | 5165 | 87 | 0.437 | 0.7387 | 0.6257 | 24.4 |
| Os | 3 | 5165 | 87 | 0.437 | 0.7851 | 0.6106 | 25.1 |
| Os | 4 | 5165 | 87 | 0.437 | 0.7565 | 0.5521 | 25.6 |
| Rh | 0 | 5663 | 54 | 0.315 | 0.5077 | 0.2774 | 24.8 |
| Rh | 1 | 5663 | 54 | 0.333 | 0.4372 | 0.3642 | 27.3 |
| Rh | 2 | 5663 | 54 | 0.315 | 0.3897 | 0.3585 | 25.8 |
| Rh | 3 | 5663 | 54 | 0.315 | 0.4552 | 0.3474 | 27.0 |
| Rh | 4 | 5663 | 54 | 0.315 | 0.3742 | 0.3426 | 24.6 |
| Re | 0 | 5094 | 166 | 0.163 | 0.7874 | 0.4912 | 23.9 |
| Re | 1 | 5094 | 166 | 0.163 | 0.7743 | 0.5958 | 23.6 |
| Re | 2 | 5094 | 166 | 0.163 | 0.7406 | 0.5902 | 22.0 |
| Re | 3 | 5094 | 166 | 0.163 | 0.7384 | 0.5989 | 24.1 |
| Re | 4 | 5094 | 166 | 0.163 | 0.7394 | 0.6645 | 22.2 |

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
