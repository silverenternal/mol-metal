# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.9065 ± 0.0227** (n=5) |
| bace | 0.876 | 0.868 | **0.7421 ± 0.0446** (n=5) |
| hiv | 0.823 | 0.761 | **0.7900 ± 0.0109** (n=5) |
| clintox | 0.894 | 0.832 | **0.8209 ± 0.0122** (n=5) |

**Mean across 4 tasks: 0.8149** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1415 | 390 | 0.700 | 0.8655 | 0.9160 | 0.950 |
| 1 | 1403 | 463 | 0.790 | 0.9013 | 0.9605 | 0.957 |
| 2 | 1435 | 410 | 0.746 | 0.9313 | 0.9728 | 1.000 |
| 3 | 1434 | 412 | 0.735 | 0.9132 | 0.9656 | 1.000 |
| 4 | 1406 | 379 | 0.805 | 0.9212 | 0.9761 | 1.000 |

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1061 | 302 | 0.474 | 0.6770 | 0.6623 | 0.800 |
| 1 | 1064 | 300 | 0.423 | 0.8169 | 0.7761 | 0.933 |
| 2 | 1057 | 306 | 0.490 | 0.7450 | 0.7625 | 0.933 |
| 3 | 1060 | 303 | 0.518 | 0.7381 | 0.7578 | 0.867 |
| 4 | 1059 | 303 | 0.439 | 0.7337 | 0.6705 | 0.667 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7793 | 0.3645 | 0.381 |
| 1 | 28790 | 8226 | 0.031 | 0.7873 | 0.3194 | 0.292 |
| 2 | 28786 | 8225 | 0.040 | 0.7971 | 0.3023 | 0.365 |
| 3 | 28789 | 8226 | 0.040 | 0.7792 | 0.3215 | 0.331 |
| 4 | 28788 | 8227 | 0.031 | 0.8073 | 0.3757 | 0.350 |

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1038 | 298 | 0.128 | 0.8281 | 0.3802 | 0.467 |
| 1 | 1038 | 300 | 0.070 | 0.8294 | 0.2866 | 0.400 |
| 2 | 1043 | 295 | 0.058 | 0.8305 | 0.3477 | 0.467 |
| 3 | 1039 | 297 | 0.061 | 0.7981 | 0.1577 | 0.067 |
| 4 | 1038 | 298 | 0.050 | 0.8184 | 0.2324 | 0.200 |

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
* `/home/hugo/codes/mol-metal/molmetal/reports/wf_tier2_molnet/final.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).
