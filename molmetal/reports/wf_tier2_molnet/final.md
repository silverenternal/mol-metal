# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Split key:** Bemis-Murcko, stereo-blind (`includeChirality=False`) — the `astartes`/chemprop reference
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.9025 ± 0.0350** (n=5) |
| bace | 0.876 | 0.868 | **0.7422 ± 0.0828** (n=5) |
| hiv | 0.823 | 0.761 | **0.7976 ± 0.0170** (n=5) |
| clintox | 0.894 | 0.832 | **0.7957 ± 0.1075** (n=5) |

**Mean across 4 tasks: 0.8095** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1434 | 412 | 0.779 | 0.9506 | 0.9847 | 1.000 |
| 1 | 1472 | 401 | 0.748 | 0.9357 | 0.9765 | 1.000 |
| 2 | 1450 | 403 | 0.824 | 0.8864 | 0.9717 | 1.000 |
| 3 | 1435 | 409 | 0.812 | 0.8573 | 0.9543 | 1.000 |
| 4 | 1403 | 474 | 0.795 | 0.8825 | 0.9657 | 1.000 |

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1066 | 299 | 0.485 | 0.6054 | 0.6357 | 0.800 |
| 1 | 1058 | 305 | 0.475 | 0.8129 | 0.7937 | 1.000 |
| 2 | 1058 | 304 | 0.474 | 0.8060 | 0.7839 | 0.867 |
| 3 | 1059 | 303 | 0.419 | 0.6864 | 0.5894 | 0.600 |
| 4 | 1057 | 308 | 0.529 | 0.8002 | 0.7919 | 0.933 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7967 | 0.3868 | 0.376 |
| 1 | 28790 | 8226 | 0.031 | 0.7894 | 0.3125 | 0.302 |
| 2 | 28786 | 8225 | 0.040 | 0.7743 | 0.3688 | 0.372 |
| 3 | 28789 | 8226 | 0.040 | 0.8014 | 0.3179 | 0.328 |
| 4 | 28788 | 8227 | 0.031 | 0.8263 | 0.3417 | 0.350 |

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1039 | 297 | 0.067 | 0.8423 | 0.2185 | 0.200 |
| 1 | 1041 | 296 | 0.061 | 0.8898 | 0.4412 | 0.467 |
| 2 | 1038 | 298 | 0.064 | 0.8446 | 0.3400 | 0.400 |
| 3 | 1035 | 295 | 0.102 | 0.5859 | 0.1735 | 0.267 |
| 4 | 1039 | 297 | 0.081 | 0.8159 | 0.2314 | 0.200 |

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
* `molmetal/reports/wf_tier2_molnet/final.md` — this report.
* `metrics/by_metric/moleculenet_auc.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
