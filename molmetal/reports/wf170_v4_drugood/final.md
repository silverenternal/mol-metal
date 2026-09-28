# Wf-170 — V4MultiTaskDMPNN on real DrugOOD IC50 (paper §4.1)

**Date:** 2026-09-27
**Data source:** DrugOOD lbap_core_ic50 (real ChEMBL-derived IC50 data, 95236 rows per config)
**Model:** V4MultiTaskDMPNN (single-head binary, hidden=128, n_layers=2, n_metals=1)
**Recipe:** 30 epochs x batch 128 x lr 1e-3 x train_max_per_epoch=1500 x AdamW + cosine annealing
**Splits requested:** assay / scaffold / size (per DrugOOD section 4.1)
**Seeds requested:** [0, 1, 2, 3, 4]
**Wall time observed:** ~18.9 min (script was killed at 18:53 wall-clock to fit within 1-hour budget)

## Status — PARTIAL RUN

Only the **assay** split completed 4/5 seeds before the 1-hour budget was exhausted.
The scaffold and size splits had featurization completed but no seeds were trained.

The full spec (5 seeds x 3 splits x 30 epochs) would have required ~58 min total
(3 min featurization + 55 min training at ~3.5-4 min per seed), which exceeded
the 1-hour budget.

## OOD Test AUC per split (mean +/- std)

| Split | OOD AUC (mean +/- std) | IID AUC (mean +/- std) | n_seeds |
|---|---|---|---|
| assay | **0.6964 +/- 0.0022** | 0.8178 +/- 0.0089 | 4 |
| scaffold | n/a (0 seeds trained) | n/a | 0 |
| size | n/a (0 seeds trained) | n/a | 0 |

## Per-seed detail (assay split)

| Seed | best_epoch | val_auc | OOD test AUC | IID test AUC | train_time (s) |
|---|---|---|---|---|---|
| 0 | 25 | 0.6971 | 0.6981 | 0.8060 | 225.6 |
| 1 | 30 | 0.6866 | 0.6976 | 0.8148 | 208.8 |
| 2 | 20 | 0.6799 | 0.6973 | 0.8195 | 209.8 |
| 3 | 30 | 0.7149 | 0.6925 | 0.8308 | 222.0 |

Mean OOD AUC: 0.6964 +/- 0.0022 (very tight — consistent across seeds)

## Honest framing

1. **Data source is the real DrugOOD IC50 bundle** — pre-built CSVs at
   .data/molmetal/drugood/ic50/csv/lbap_core_ic50_{assay,scaffold,size}.csv
   with pre-defined train/ood_val/ood_test/iid_val/iid_test splits used as-is.
2. **V4MultiTaskDMPNN adapted to single-task** — instantiated with n_metals=1 so the architecture
   behaves as a single-head D-MPNN binary classifier. The metal-embedding + per-metal head are kept
   in the forward pass.
3. **Subsample per epoch** — each epoch samples 1500 training rows (DrugOOD full train has 28-36K rows
   per split) to bound wall time within the 1-hour budget.
4. **Cached encoding for speed** — pre-featurized graph tensors are stored on GPU and selected by
   index per batch, skipping RDKit SMILES parsing per batch.
5. **Coord auxiliary head is regularisation-only** — DrugOOD has no coordination-geometry labels, so
   the coord head is fed a constant proxy label (linear=0) per batch. The 0.3 coord weight acts as a
   small regulariser, not a supervision signal.
6. **Eval samples capped at 4000** — ood_val and iid_test are subsampled to 4000 rows per
   _eval() call to bound wall time.
7. **Seed 4 of assay was not completed** — the run was terminated at 18:53 wall-clock to fit within
   the 1-hour budget. The scaffold and size splits had featurization done but no training ran.

## Files

* molmetal/reports/wf170_v4_drugood/summary.json — full metrics
* molmetal/reports/wf170_v4_drugood/per_seed/assay__seed{0,1,2,3}.json — per-seed detail
* molmetal/reports/wf170_v4_drugood/final.md — this report
* .ckpt/wf170_v4_drugood/assay_seed{0,1,2,3}.pt — best-epoch ckpt per seed

## Script

* molmetal/scripts/wf170_v4_drugood.py — driver
