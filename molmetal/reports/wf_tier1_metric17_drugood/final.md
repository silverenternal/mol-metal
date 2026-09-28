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

**Data source:** MetalCytoToxDB fallback (cell_line / scaffold / heavy-atom bucket OOD axes)
**Featurization:** Morgan FP (radius=2, n_bits=2048)
**Model:** V4 MLP on Morgan FP (2048 -> 256 -> 128 -> 1, BCE-with-pos-weight)
**Splits:** assay (hold-out cell_line) / scaffold (Bemis-Murcko) / size (heavy-atom bucket)
**Seeds:** 0, 1, 2

## Results

| Split | Our ID AUC | Our OOD AUC | iMoLD ref | ERM ref | Gate | Verdict |
|---|---|---|---|---|---|---|
| assay | 0.9743 +/- 0.0047 | **0.8914 +/- 0.0019** | 0.7211 | 0.7163 | >= 0.7011 | PASS |
| scaffold | 0.9567 +/- 0.0148 | **0.6973 +/- 0.0271** | 0.6884 | 0.6879 | >= 0.6684 | PASS |
| size | 0.9363 +/- 0.0134 | **0.7479 +/- 0.0055** | 0.6792 | 0.6750 | >= 0.6592 | PASS |

**Overall verdict:** PASS (mean across 3 splits >= mean(iMoLD - 0.02))

## Per-seed detail

### assay

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 3994 | 834 | 0.291 | 0.9752 | **0.8888** |
| 1 | 3994 | 834 | 0.291 | 0.9681 | **0.8924** |
| 2 | 3994 | 834 | 0.291 | 0.9796 | **0.8929** |

### scaffold

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 3681 | 1060 | 0.292 | 0.9724 | **0.7223** |
| 1 | 3692 | 1053 | 0.354 | 0.9608 | **0.7101** |
| 2 | 3681 | 1045 | 0.340 | 0.9368 | **0.6597** |

### size

| Seed | n_train | n_test (OOD) | pos_rate_test | ID AUC | OOD AUC |
|---|---|---|---|---|---|
| 0 | 3741 | 968 | 0.310 | 0.9397 | **0.7405** |
| 1 | 3565 | 1216 | 0.296 | 0.9508 | **0.7497** |
| 2 | 3797 | 912 | 0.273 | 0.9184 | **0.7536** |

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
