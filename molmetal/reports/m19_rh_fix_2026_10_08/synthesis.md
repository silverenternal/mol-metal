# LOMO 5-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-08  
**Metals (held-out folds)**: Ru Ir Os Rh Re  
**Seeds**: 0 1 2 3 4  
**Epochs/seed**: 30  
**Cohort CSV**: molmetal/data/metal_lomo.csv  
**Leak-controlled (--drop-leaked-ligands)**: False  
**Wasserstein weight (requested)**: 0.05  
**W₂ actually applied to the gradient (Wf-E1)**: 0.05  
**Rh oversample**: 0 (0 actually appended)  
**Quick mode**: False  
**Wf-127 P2 L2 guard (grad-norm clip ≤ 5.0)**: True  
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
2. **Rh closure-theorem oversampling — OFF** (requested 0; 0 rows appended).
3. **Multi-task D-MPNN backbone** with per-metal embedding extraction
   (pooled `2H`-dim vector before the readout head); computed every
   epoch end only when W₂ is active.
4. **Wf-127 P2 L2 grad-norm guard — ON** (clips grad-norm to ≤ 5.0 each step).
5. **Wf-127 P3 ASTSR scale clip — OFF** (range [-10.0, 10.0]; engage with `--use-astr-scale-guard`).

## §2 MEASURED (this run)

**Overall LOMO mean test AUC = 0.5862 ± 0.1028**  
(n=25 runs, 5 folds, gate ≥ 0.7)

**Rh LOMO AUC = 0.4056 ± 0.0597** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: MISS  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.5838 | 0.0036 | 5 |
| Ir | 0.5962 | 0.0114 | 5 |
| Os | 0.6521 | 0.0179 | 5 |
| Rh | 0.4056 | 0.0597 | 5 |
| Re | 0.6933 | 0.0155 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1442 | 3668 | 0.263 | 0.7279 | 0.5830 | 33.3 |
| Ru | 1 | 1442 | 3668 | 0.263 | 0.7866 | 0.5786 | 42.6 |
| Ru | 2 | 1442 | 3668 | 0.263 | 0.7427 | 0.5847 | 36.2 |
| Ru | 3 | 1442 | 3668 | 0.263 | 0.7255 | 0.5829 | 25.9 |
| Ru | 4 | 1442 | 3668 | 0.263 | 0.7552 | 0.5897 | 27.9 |
| Ir | 0 | 3626 | 1242 | 0.448 | 0.7855 | 0.5854 | 47.8 |
| Ir | 1 | 3626 | 1242 | 0.448 | 0.7834 | 0.5892 | 46.6 |
| Ir | 2 | 3626 | 1242 | 0.448 | 0.7493 | 0.5953 | 47.5 |
| Ir | 3 | 3626 | 1242 | 0.448 | 0.7661 | 0.5932 | 50.2 |
| Ir | 4 | 3626 | 1242 | 0.448 | 0.7642 | 0.6180 | 65.4 |
| Os | 0 | 4665 | 87 | 0.437 | 0.8056 | 0.6262 | 74.9 |
| Os | 1 | 4665 | 87 | 0.437 | 0.7901 | 0.6625 | 61.3 |
| Os | 2 | 4665 | 87 | 0.437 | 0.7659 | 0.6711 | 59.9 |
| Os | 3 | 4665 | 87 | 0.437 | 0.7934 | 0.6654 | 60.9 |
| Os | 4 | 4665 | 87 | 0.437 | 0.7669 | 0.6353 | 72.3 |
| Rh | 0 | 5163 | 48 | 0.250 | 0.4183 | 0.4977 | 103.1 |
| Rh | 1 | 5163 | 58 | 0.310 | 0.4305 | 0.3729 | 80.0 |
| Rh | 2 | 5163 | 49 | 0.265 | 0.4128 | 0.3205 | 68.1 |
| Rh | 3 | 5163 | 51 | 0.275 | 0.3433 | 0.4373 | 70.5 |
| Rh | 4 | 5163 | 60 | 0.317 | 0.3320 | 0.3999 | 66.9 |
| Re | 0 | 4594 | 166 | 0.163 | 0.8167 | 0.6964 | 60.7 |
| Re | 1 | 4594 | 166 | 0.163 | 0.7897 | 0.7093 | 60.1 |
| Re | 2 | 4594 | 166 | 0.163 | 0.7481 | 0.6640 | 77.0 |
| Re | 3 | 4594 | 166 | 0.163 | 0.7441 | 0.6945 | 99.1 |
| Re | 4 | 4594 | 166 | 0.163 | 0.7321 | 0.7024 | 71.0 |

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
