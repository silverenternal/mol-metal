# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Featurization:** AttentiveFP atom+bond features (D-MPNN message passing + per-atom attentive readout)
**Model:** AttentiveFP graph encoder (single-task) + BCEWithLogits head with class-weighted pos_weight
**Backbone:** attentivefp
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bace | 0.876 | 0.868 | **0.7636 ± 0.0398** (n=5) |

**Mean across 1 tasks: 0.7636** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ⚠️ BELOW GATE — measured honestly; not paper-promotable as headline Tier-2 metric

## Per-seed detail

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1061 | 302 | 0.474 | 0.6986 | 0.6648 | 0.733 |
| 1 | 1064 | 300 | 0.423 | 0.8020 | 0.7393 | 0.933 |
| 2 | 1057 | 306 | 0.490 | 0.7396 | 0.7272 | 0.933 |
| 3 | 1060 | 303 | 0.518 | 0.8024 | 0.8072 | 1.000 |
| 4 | 1059 | 303 | 0.439 | 0.7754 | 0.6713 | 0.733 |

## Honest caveats

1. **Backbone**: this run uses the AttentiveFP graph encoder
   (D-MPNN message passing + per-atom attentive readout; Yang et al. 2019)
   on the SAME Bemis-Murcko scaffold split and the SAME ROC-AUC metric as
   the D-MPNN / Morgan-MLP runs, so the three are directly comparable.
2. **Head loss**: AttentiveFP is single-task graph encoder; we use
   BCEWithLogitsLoss with the same `_resolve_pos_weight` dispatch as the
   D-MPNN path (so `--class-weight {on,off,balanced}` is protocol-aligned).
3. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;
   per-task tuning (focal loss / calibrated threshold) is not applied here.
4. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split
   (Wu 2018 Table 3), so the comparison is protocol-aligned.
5. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed
   spread; read the per-seed table above, not the mean alone.

## Files

* `molmetal/scripts/train_moleculenet.py` — driver.
* `molmetal/reports/blk_d1_attentivefp_5seed_2026_10_09/report.md` — this report.
* `metrics/by_metric/moleculenet_auc_dmpnn.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
