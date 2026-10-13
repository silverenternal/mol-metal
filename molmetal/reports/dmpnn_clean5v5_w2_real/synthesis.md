# LOMO 5-fold × 5-seed × 50-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-06  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 50  
**Cohort CSV**: molmetal/data/metal_lomo.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
**Rh oversample**: 500 (500 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: False  
**Wf-127 P3 ASTSR scale clip [-10.0, 10.0]**: False  

## §1 Goal

Leave-one-metal-out (LOMO) cross-metal transfer benchmark: hold out
one metal's rows entirely, train a multi-task D-MPNN on the rest, and
measure ROC-AUC on the held-out metal (the m19 protocol). The active
components are echoed from the config below — not asserted.

1. **Wasserstein transfer regularisation — ON** (weight 0.05).
   Sliced-W₂ over per-metal embeddings (canonical M1 primitive from
   `molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`),
   added to the BCE+MSE loss and injected every 5 epochs.
   **Wf-E1 (2026-10-05)**: this term is a NO-OP unless
   `--wasserstein-real` is passed — the pre-Wf-E1 code detached the
   per-metal embeddings, so W₂ contributed ZERO gradient while
   costing 2.4x wall-time (Rh LOMO seeds 0-2, mean Δ test_AUC
   = −0.0023, within noise). `w2_effective_weight` in the per-seed
   JSON is the authoritative record of whether W₂ reached the graph.
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

**Overall LOMO mean test AUC = 0.5833 ± 0.1103**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.4106 ± 0.0821** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5913 | 0.0134 | 5 |
| Ir | 0.6324 | 0.0359 | 5 |
| Os | 0.5743 | 0.0274 | 5 |
| Rh | 0.4106 | 0.0821 | 5 |
| Re | 0.7078 | 0.0629 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1942 | 3668 | 0.263 | 0.7187 | 0.5883 | 81.0 |
| Ru | 1 | 1942 | 3668 | 0.263 | 0.7779 | 0.6035 | 64.5 |
| Ru | 2 | 1942 | 3668 | 0.263 | 0.7657 | 0.5933 | 78.2 |
| Ru | 3 | 1942 | 3668 | 0.263 | 0.7969 | 0.6040 | 78.0 |
| Ru | 4 | 1942 | 3668 | 0.263 | 0.7633 | 0.5673 | 79.3 |
| Ir | 0 | 4126 | 1242 | 0.448 | 0.8122 | 0.5905 | 138.5 |
| Ir | 1 | 4126 | 1242 | 0.448 | 0.7914 | 0.5943 | 134.5 |
| Ir | 2 | 4126 | 1242 | 0.448 | 0.7771 | 0.6328 | 134.2 |
| Ir | 3 | 4126 | 1242 | 0.448 | 0.8013 | 0.6790 | 141.0 |
| Ir | 4 | 4126 | 1242 | 0.448 | 0.7956 | 0.6654 | 137.3 |
| Os | 0 | 5165 | 87 | 0.437 | 0.8450 | 0.5320 | 137.1 |
| Os | 1 | 5165 | 87 | 0.437 | 0.8427 | 0.5602 | 157.4 |
| Os | 2 | 5165 | 87 | 0.437 | 0.8112 | 0.5975 | 128.8 |
| Os | 3 | 5165 | 87 | 0.437 | 0.8242 | 0.6093 | 105.0 |
| Os | 4 | 5165 | 87 | 0.437 | 0.8122 | 0.5725 | 108.3 |
| Rh | 0 | 5663 | 48 | 0.250 | 0.4083 | 0.5104 | 136.7 |
| Rh | 1 | 5663 | 58 | 0.310 | 0.4046 | 0.3792 | 176.2 |
| Rh | 2 | 5663 | 49 | 0.265 | 0.4036 | 0.3280 | 181.5 |
| Rh | 3 | 5663 | 51 | 0.275 | 0.3148 | 0.5068 | 141.7 |
| Rh | 4 | 5663 | 60 | 0.317 | 0.4600 | 0.3286 | 171.8 |
| Re | 0 | 5094 | 166 | 0.163 | 0.8295 | 0.6033 | 132.6 |
| Re | 1 | 5094 | 166 | 0.163 | 0.8323 | 0.7787 | 125.3 |
| Re | 2 | 5094 | 166 | 0.163 | 0.7817 | 0.7555 | 123.6 |
| Re | 3 | 5094 | 166 | 0.163 | 0.7902 | 0.6740 | 129.8 |
| Re | 4 | 5094 | 166 | 0.163 | 0.7766 | 0.7274 | 128.3 |

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
