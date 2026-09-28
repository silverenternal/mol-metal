# Wf-185 — chemprop v2 retrain on DrugOOD lbap_refined_ic50_assay

**Date:** 2026-09-27  
**Data:** DrugOOD lbap_refined_ic50_assay  
**Train rows:** 95373  
**Model:** chemprop v2 AtomMessagePassing + SumAggregation + BinaryClassificationFFN (via ChempropV2MTModel Ru head)  
**Backbone:** hidden=128, depth=2, head_hidden=128  
**Recipe:** 20 epochs × batch 128 × lr 0.001 × 5 seeds  
**Wall time:** 75.54 min  

## Per-split AUC (mean ± std across seeds)

| Split | AUC mean | AUC std | n seeds |
|---|---|---|---|
| train | 0.9274 | 0.0232 | 5 |
| iid_val | 0.9017 | 0.0203 | 5 |
| iid_test | 0.8968 | 0.0084 | 5 |
| ood_val | 0.7446 | 0.0153 | 5 |
| ood_test | 0.7137 | 0.0079 | 5 |

## Per-seed OOD test AUC

| seed | train | iid_val | iid_test | ood_val | ood_test | epochs | elapsed_s |
|---|---|---|---|---|---|---|---|
| 0 | 0.8819 | 0.8622 | 0.8821 | 0.7671 | 0.7221 | 6 | 1515.5 |
| 1 | 0.9361 | 0.9048 | 0.9052 | 0.7342 | 0.7156 | 18 | 1506.7 |
| 2 | 0.9400 | 0.9102 | 0.8958 | 0.7576 | 0.7019 | 20 | 680.7 |
| 3 | 0.9326 | 0.9115 | 0.9047 | 0.7264 | 0.7076 | 20 | 415.7 |
| 4 | 0.9465 | 0.9198 | 0.8963 | 0.7376 | 0.7215 | 20 | 413.7 |

## Files

* `summary.json` — full metrics + per-seed
* `per_seed/seed<n>.json` — per-seed detail
* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization
