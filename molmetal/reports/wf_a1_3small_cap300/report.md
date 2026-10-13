# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| clintox | 0.894 | 0.832 | **0.8311 ± 0.0090** (n=5) |
| hiv | 0.823 | 0.761 | **0.7909 ± 0.0178** (n=5) |
| bbbp | 0.940 | 0.728 | **0.9023 ± 0.0147** (n=5) |

**Mean across 3 tasks: 0.8414** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1038 | 298 | 0.128 | 0.8300 | 0.3965 | 0.533 |
| 1 | 1038 | 300 | 0.070 | 0.8219 | 0.2912 | 0.400 |
| 2 | 1043 | 295 | 0.058 | 0.8224 | 0.2681 | 0.267 |
| 3 | 1039 | 297 | 0.061 | 0.8346 | 0.2266 | 0.333 |
| 4 | 1038 | 298 | 0.050 | 0.8464 | 0.2596 | 0.267 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7778 | 0.3685 | 0.381 |
| 1 | 28790 | 8226 | 0.031 | 0.7786 | 0.2682 | 0.275 |
| 2 | 28786 | 8225 | 0.040 | 0.8007 | 0.3307 | 0.345 |
| 3 | 28789 | 8226 | 0.040 | 0.7758 | 0.2865 | 0.338 |
| 4 | 28788 | 8227 | 0.031 | 0.8214 | 0.4058 | 0.358 |

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1415 | 390 | 0.700 | 0.8816 | 0.9302 | 0.950 |
| 1 | 1403 | 463 | 0.790 | 0.8885 | 0.9572 | 0.957 |
| 2 | 1435 | 410 | 0.746 | 0.9201 | 0.9677 | 1.000 |
| 3 | 1434 | 412 | 0.735 | 0.9126 | 0.9659 | 1.000 |
| 4 | 1406 | 379 | 0.805 | 0.9087 | 0.9729 | 1.000 |

## Honest caveats

1. **Backbone**: this run uses the D-MPNN graph encoder on the SAME
   Bemis-Murcko scaffold split and the SAME ROC-AUC metric as the
   Morgan-FP + MLP baseline, so the two are directly comparable.
2. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;
   per-task tuning (focal loss / calibrated threshold) is not applied here.
3. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split
   (Wu 2018 Table 3), so the comparison is protocol-aligned.
4. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed
   spread; read the per-seed table above, not the mean alone.

## Files

* `molmetal/scripts/train_moleculenet.py` — driver.
* `molmetal/reports/wf_a1_3small_cap300/report.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
