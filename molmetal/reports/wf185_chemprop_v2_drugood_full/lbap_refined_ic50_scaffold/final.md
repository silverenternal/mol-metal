# Wf-185 — chemprop v2 retrain on DrugOOD lbap_refined_ic50_scaffold

**Date:** 2026-09-27  
**Data:** DrugOOD lbap_refined_ic50_scaffold  
**Train rows:** 60908  
**Model:** chemprop v2 AtomMessagePassing + SumAggregation + BinaryClassificationFFN (via ChempropV2MTModel Ru head)  
**Backbone:** hidden=128, depth=2, head_hidden=128  
**Recipe:** 20 epochs × batch 128 × lr 0.001 × 5 seeds  
**Wall time:** 53.57 min  

## Per-split AUC (mean ± std across seeds)

| Split | AUC mean | AUC std | n seeds |
|---|---|---|---|
| train | 0.9421 | 0.0031 | 5 |
| iid_val | 0.9142 | 0.0126 | 5 |
| iid_test | 0.8302 | 0.0165 | 5 |
| ood_val | 0.7518 | 0.0193 | 5 |
| ood_test | 0.6707 | 0.0042 | 5 |

## Per-seed OOD test AUC

| seed | train | iid_val | iid_test | ood_val | ood_test | epochs | elapsed_s |
|---|---|---|---|---|---|---|---|
| 0 | 0.9412 | 0.9339 | 0.8382 | 0.7799 | 0.6667 | 20 | 272.8 |
| 1 | 0.9415 | 0.8999 | 0.8040 | 0.7252 | 0.6782 | 20 | 271.0 |
| 2 | 0.9478 | 0.9224 | 0.8217 | 0.7670 | 0.6711 | 20 | 275.6 |
| 3 | 0.9384 | 0.9029 | 0.8530 | 0.7427 | 0.6669 | 20 | 950.6 |
| 4 | 0.9416 | 0.9118 | 0.8343 | 0.7443 | 0.6703 | 20 | 1444.5 |

## Files

* `summary.json` — full metrics + per-seed
* `per_seed/seed<n>.json` — per-seed detail
* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization
