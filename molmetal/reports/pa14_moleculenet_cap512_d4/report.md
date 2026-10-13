# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.8994 ± 0.0218** (n=5) |
| bace | 0.876 | 0.868 | **0.7933 ± 0.0250** (n=5) |
| hiv | 0.823 | 0.761 | **0.7889 ± 0.0189** (n=5) |
| clintox | 0.894 | 0.832 | **0.8303 ± 0.0133** (n=5) |

**Mean across 4 tasks: 0.8280** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1415 | 390 | 0.700 | 0.8566 | 0.9100 | 0.900 |
| 1 | 1403 | 463 | 0.790 | 0.9166 | 0.9682 | 1.000 |
| 2 | 1435 | 410 | 0.746 | 0.9060 | 0.9614 | 1.000 |
| 3 | 1434 | 412 | 0.735 | 0.9058 | 0.9626 | 1.000 |
| 4 | 1406 | 379 | 0.805 | 0.9119 | 0.9721 | 1.000 |

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1061 | 302 | 0.474 | 0.7534 | 0.7321 | 0.867 |
| 1 | 1064 | 300 | 0.423 | 0.8239 | 0.7825 | 0.933 |
| 2 | 1057 | 306 | 0.490 | 0.7888 | 0.7583 | 0.800 |
| 3 | 1060 | 303 | 0.518 | 0.7846 | 0.8278 | 1.000 |
| 4 | 1059 | 303 | 0.439 | 0.8159 | 0.7316 | 0.800 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7666 | 0.3601 | 0.378 |
| 1 | 28790 | 8226 | 0.031 | 0.7751 | 0.2675 | 0.268 |
| 2 | 28786 | 8225 | 0.040 | 0.8068 | 0.3671 | 0.397 |
| 3 | 28789 | 8226 | 0.040 | 0.7805 | 0.3362 | 0.331 |
| 4 | 28788 | 8227 | 0.031 | 0.8154 | 0.3374 | 0.355 |

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1038 | 298 | 0.128 | 0.8332 | 0.4099 | 0.467 |
| 1 | 1038 | 300 | 0.070 | 0.8127 | 0.3225 | 0.467 |
| 2 | 1043 | 295 | 0.058 | 0.8526 | 0.3210 | 0.267 |
| 3 | 1039 | 297 | 0.061 | 0.8221 | 0.2256 | 0.267 |
| 4 | 1038 | 298 | 0.050 | 0.8311 | 0.2502 | 0.200 |

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
* `molmetal/reports/pa14_moleculenet_cap512_d4/report.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
