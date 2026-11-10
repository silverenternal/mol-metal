# LOMO 2-fold × 5-seed × 30-epoch cross-metal re-bench on MetalCytoToxDB

**Date**: 2026-10-23  
**Metals (held-out folds)**: Ru Ir  
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

**Overall LOMO mean test AUC = 0.6028 ± 0.0203**  
(n=10 runs, 2 folds, gate ≥ 0.7)

**Rh LOMO AUC = nan ± nan** (target ≥ 0.5)  
**Verdict (Rh target)**: MISS  
**Verdict (overall target ≥ 0.60)**: HIT  
**Verdict (gate ≥ 0.70)**: BELOW GATE  

### §2.1 Per-fold breakdown

| Held-out metal | Mean ROC-AUC | Std | n_runs |
|---|---|---|---|
| Ru | 0.6057 | 0.0234 | 5 |
| Ir | 0.5999 | 0.0161 | 5 |

### §2.2 Per-run detail

| Fold | Seed | n_train | n_test | pos_rate_test | val_AUC | test_AUC | elapsed_s |
|---|---|---|---|---|---|---|---|
| Ru | 0 | 1492 | 3618 | 0.263 | 0.7348 | 0.6275 | 15.2 |
| Ru | 1 | 1492 | 3618 | 0.262 | 0.7824 | 0.6178 | 10.4 |
| Ru | 2 | 1492 | 3618 | 0.261 | 0.7496 | 0.6085 | 12.4 |
| Ru | 3 | 1492 | 3618 | 0.264 | 0.7551 | 0.5607 | 10.6 |
| Ru | 4 | 1492 | 3618 | 0.262 | 0.7250 | 0.6142 | 12.4 |
| Ir | 0 | 3676 | 1192 | 0.450 | 0.7914 | 0.5950 | 19.8 |
| Ir | 1 | 3676 | 1192 | 0.449 | 0.7960 | 0.6217 | 20.3 |
| Ir | 2 | 3676 | 1192 | 0.441 | 0.7754 | 0.6073 | 21.6 |
| Ir | 3 | 3676 | 1192 | 0.451 | 0.7695 | 0.5728 | 20.5 |
| Ir | 4 | 3676 | 1192 | 0.450 | 0.7876 | 0.6026 | 22.3 |

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
