# Wf-185 — chemprop v2 retrain on DrugOOD lbap_refined_ic50_size

**Date:** 2026-09-28  
**Data:** DrugOOD lbap_refined_ic50_size  
**Train rows:** 97779  
**Model:** chemprop v2 AtomMessagePassing + SumAggregation + BinaryClassificationFFN (via ChempropV2MTModel Ru head)  
**Backbone:** hidden=128, depth=2, head_hidden=128  
**Recipe:** 20 epochs × batch 128 × lr 0.001 × 5 seeds  
**Wall time:** 38.14 min  

## Per-split AUC (mean ± std across seeds)

| Split | AUC mean | AUC std | n seeds |
|---|---|---|---|
| train | 0.9392 | 0.0031 | 5 |
| iid_val | 0.8805 | 0.0164 | 5 |
| iid_test | 0.8992 | 0.0082 | 5 |
| ood_val | 0.7939 | 0.0157 | 5 |
| ood_test | 0.6757 | 0.0142 | 5 |

## Per-seed OOD test AUC

| seed | train | iid_val | iid_test | ood_val | ood_test | epochs | elapsed_s |
|---|---|---|---|---|---|---|---|
| 0 | 0.9455 | 0.8792 | 0.8913 | 0.8078 | 0.6674 | 20 | 468.1 |
| 1 | 0.9374 | 0.8666 | 0.8985 | 0.8072 | 0.6724 | 20 | 445.7 |
| 2 | 0.9380 | 0.8866 | 0.9116 | 0.7958 | 0.6565 | 20 | 461.5 |
| 3 | 0.9372 | 0.9083 | 0.8898 | 0.7940 | 0.6843 | 20 | 468.7 |
| 4 | 0.9380 | 0.8620 | 0.9047 | 0.7647 | 0.6978 | 20 | 444.4 |

## Files

* `summary.json` — full metrics + per-seed
* `per_seed/seed<n>.json` — per-seed detail
* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization
