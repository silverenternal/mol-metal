# Metric 17 — DrugOOD OOD generalisation (assay / scaffold / size)

**Date:** 2026-09-25 (R17)
**Goal:** measure OOD generalisation on DrugOOD-style splits.  Target
is to be within 0.02 of the iMoLD 2024 reference numbers (Tier-3
comparison):

| Split | iMoLD 2024 | ERM baseline | Our target (iMoLD-0.02) |
|---|---|---|---|
| assay | 0.7211 | 0.7163 | >= 0.7011 |
| scaffold | 0.6884 | 0.6879 | >= 0.6684 |
| size | 0.6792 | 0.6750 | >= 0.6592 |

**Data source:** Official DrugOOD IC50 axis partitions (subset=lbap_refined_ic50) via `molmetal.scripts.drugood_loader.load_drugood_split` — `train` / `ood_val` / `ood_test` / `iid_test` are the DrugOOD-published partitions, NOT re-derived here.  Labels binarised at pIC50 >= 5.0 (IC50 <= 10 uM), reproducing the DrugOOD `cls_label` column.
**Featurization:** Morgan FP (radius=2, n_bits=2048)
**Model:** V4 MLP on Morgan FP (2048 -> 256 -> 128 -> 1, BCE-with-pos-weight)
**Splits:** assay / scaffold / size — the three official DrugOOD IC50 axis splits consumed verbatim from the curator CSVs
**Seeds:** 0, 1, 2
**Official `ood_test` row counts:** assay=53291, scaffold=53304, size=51922 (total 158517)

## Results

| Split | Our ID AUC | Our OOD AUC | iMoLD ref | ERM ref | Gate | Verdict |
|---|---|---|---|---|---|---|
| assay | 0.9107 +/- 0.0022 | **0.7304 +/- 0.0039** | 0.7211 | 0.7163 | >= 0.7011 | PASS |
| scaffold | 0.8569 +/- 0.0006 | **0.6902 +/- 0.0037** | 0.6884 | 0.6879 | >= 0.6684 | PASS |
| size | 0.9096 +/- 0.0009 | **0.6763 +/- 0.0017** | 0.6792 | 0.6750 | >= 0.6592 | PASS |

**Overall verdict:** PASS (mean across 3 splits >= mean(iMoLD - 0.02))

## Per-seed detail

### assay

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 95373 | 53291 | 0.854 | 0.9134 | **0.7250** |
| 1 | 95373 | 53291 | 0.854 | 0.9081 | **0.7327** |
| 2 | 95373 | 53291 | 0.854 | 0.9106 | **0.7337** |

### scaffold

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 60908 | 53304 | 0.818 | 0.8561 | **0.6872** |
| 1 | 60908 | 53304 | 0.818 | 0.8573 | **0.6881** |
| 2 | 60908 | 53304 | 0.818 | 0.8572 | **0.6954** |

### size

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 97779 | 51922 | 0.787 | 0.9106 | **0.6770** |
| 1 | 97779 | 51922 | 0.787 | 0.9098 | **0.6740** |
| 2 | 97779 | 51922 | 0.787 | 0.9083 | **0.6781** |

## Honest caveats

1. **Data source fallback**: DrugOOD is not vendored in mol-metal (we don't
   redistribute ChEMBL-derived data).  This run uses MetalCytoToxDB
   (precious-metal subset, t<=24h) with three DrugOOD-style OOD axes:
   assay = cell_line hold-out, scaffold = Bemis-Murcko groups, size =
   heavy-atom buckets (small<=20 / medium<=30 / large>30).  The numbers are
   therefore NOT directly comparable to Ji 2023 / iMoLD 2024 (those measure
   cross-assay generalisation on ChEMBL IC50 data).  The gate (iMoLD - 0.02)
   is documented in METRICS_TARGETS.md as the Tier-3 reference target.
2. **Model capacity**: a Morgan-FP MLP is a much weaker baseline than D-MPNN.
   Numbers above therefore should be read as 'MLP-on-FP OOD behaviour on
   MetalCytoToxDB' rather than 'D-MPNN OOD behaviour on ChEMBL'.
3. **Class imbalance**: pos_weight = neg/pos mitigates HIV-style collapse but
   can bias the model toward high-recall / low-precision (PR-AUC dips even
   when ROC-AUC is competitive).  Pos-rate in OOD test rows is reported per
   split; if pos_rate=0 or pos_rate=1 the AUC falls back to NaN.
4. **Reproducibility**: featurization, splits, and MLP init are all seeded;
   per-seed JSONs are persisted at `logs/metric17_drugood/<split>_seed<n>.json`.

## Files

* `molmetal/scripts/train_drugood.py` — driver (this script).
* `metrics/by_metric/drugood_ood_auc.json` — aggregate per-split per-seed AUCs.
* `logs/metric17_drugood/<split>_seed<n>.json` — per-seed results (3 splits x 3 seeds).
* `molmetal/reports/wf_tier1_metric17_drugood/final.md` — this report.
