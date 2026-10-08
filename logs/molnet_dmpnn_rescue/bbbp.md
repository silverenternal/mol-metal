# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.8939 ± 0.0180** (n=5) |

**Mean across 1 tasks: 0.8939** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ✅ PASS — mean AUC ≥ 0.80

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1415 | 390 | 0.700 | 0.8630 | 0.9145 | 0.950 |
| 1 | 1403 | 463 | 0.790 | 0.8972 | 0.9628 | 1.000 |
| 2 | 1435 | 410 | 0.746 | 0.9170 | 0.9639 | 1.000 |
| 3 | 1434 | 412 | 0.735 | 0.9037 | 0.9618 | 1.000 |
| 4 | 1406 | 379 | 0.805 | 0.8887 | 0.9645 | 1.000 |

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
* `logs/molnet_dmpnn_rescue/bbbp.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
