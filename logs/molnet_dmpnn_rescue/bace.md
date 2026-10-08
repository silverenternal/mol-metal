# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** D-MPNN atom+bond features (atom+bond message passing)
**Model:** D-MPNN graph encoder (shared trunk) + per-task classification head
**Backbone:** dmpnn
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bace | 0.876 | 0.868 | **0.6805 ± 0.0560** (n=5) |

**Mean across 1 tasks: 0.6805** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ⚠️ BELOW GATE — measured honestly; not paper-promotable as headline Tier-2 metric

## Per-seed detail

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1061 | 302 | 0.474 | 0.5864 | 0.5476 | 0.467 |
| 1 | 1064 | 300 | 0.423 | 0.7354 | 0.6639 | 0.867 |
| 2 | 1057 | 306 | 0.490 | 0.7401 | 0.7470 | 0.933 |
| 3 | 1060 | 303 | 0.518 | 0.6655 | 0.6689 | 0.600 |
| 4 | 1059 | 303 | 0.439 | 0.6751 | 0.5617 | 0.400 |

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
* `logs/molnet_dmpnn_rescue/bace.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
