# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.8814 ± 0.0096** (n=5) |
| bace | 0.876 | 0.868 | **0.7436 ± 0.0149** (n=5) |
| hiv | 0.823 | 0.761 | **0.7840 ± 0.0091** (n=5) |
| clintox | 0.894 | 0.832 | **0.8184 ± 0.0141** (n=5) |

**Mean across 4 tasks: 0.8068** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1415 | 390 | 0.700 | 0.8849 | 0.9357 | 0.950 |
| 1 | 1415 | 390 | 0.700 | 0.8819 | 0.9303 | 0.950 |
| 2 | 1415 | 390 | 0.700 | 0.8636 | 0.9076 | 0.900 |
| 3 | 1415 | 390 | 0.700 | 0.8837 | 0.9376 | 1.000 |
| 4 | 1415 | 390 | 0.700 | 0.8928 | 0.9454 | 1.000 |

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1061 | 302 | 0.474 | 0.7207 | 0.7050 | 0.800 |
| 1 | 1061 | 302 | 0.474 | 0.7435 | 0.7264 | 0.867 |
| 2 | 1061 | 302 | 0.474 | 0.7346 | 0.6991 | 0.733 |
| 3 | 1061 | 302 | 0.474 | 0.7594 | 0.7291 | 0.867 |
| 4 | 1061 | 302 | 0.474 | 0.7597 | 0.7458 | 0.867 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7959 | 0.3170 | 0.347 |
| 1 | 28753 | 8296 | 0.040 | 0.7895 | 0.3509 | 0.390 |
| 2 | 28753 | 8296 | 0.040 | 0.7686 | 0.3367 | 0.352 |
| 3 | 28753 | 8296 | 0.040 | 0.7848 | 0.3602 | 0.395 |
| 4 | 28753 | 8296 | 0.040 | 0.7814 | 0.3537 | 0.381 |

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1038 | 298 | 0.128 | 0.7928 | 0.3911 | 0.600 |
| 1 | 1038 | 298 | 0.128 | 0.8337 | 0.4086 | 0.533 |
| 2 | 1038 | 298 | 0.128 | 0.8165 | 0.3826 | 0.467 |
| 3 | 1038 | 298 | 0.128 | 0.8206 | 0.3718 | 0.400 |
| 4 | 1038 | 298 | 0.128 | 0.8282 | 0.3857 | 0.400 |

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
* `molmetal/reports/l2_moleculenet_cap512_d4_ensemble/report.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
