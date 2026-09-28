# Wf-174 BD-11 — chemprop v2 retrain on DrugOOD lbap_refined_ic50_assay

**Date:** 2026-09-27  
**Data:** DrugOOD lbap_refined_ic50_assay (266,522 rows × 5 splits)  
**Train rows:** 300  
**Model:** chemprop v2 AtomMessagePassing + SumAggregation + BinaryClassificationFFN (via ChempropV2MTModel Ru head)  
**Backbone:** hidden=64, depth=2, head_hidden=64  
**Recipe:** 1 epochs × batch 128 × lr 0.001 × 1 seeds  
**Wall time:** 0.00 min  
**Time budget:** 300.0s  

## Per-split AUC (mean ± std across seeds)

| Split | AUC mean | AUC std | n seeds |
|---|---|---|---|
| train | 0.5214 | 0.0000 | 1 |
| iid_val | 0.6758 | 0.0000 | 1 |
| iid_test | 0.5781 | 0.0000 | 1 |
| ood_val | nan | 0.0000 | 1 |
| ood_test | 0.5312 | 0.0000 | 1 |

## Per-seed OOD test AUC

| seed | train | iid_val | iid_test | ood_val | ood_test | epochs | elapsed_s |
|---|---|---|---|---|---|---|---|
| 0 | 0.5214 | 0.6758 | 0.5781 | nan | 0.5312 | 1 | 0.2 |

## Honest framing

Wf-174 BD-11 chemprop v2 retrain on real DrugOOD lbap_refined_ic50_assay (lbap_refined_ic50_assay). Train fold size = 300 rows; eval splits = iid_val(100), iid_test(100), ood_val(100), ood_test(100). Backbone is hidden_dim=64, depth=2, head_hidden_dim=64 (smaller than chemprop v2 defaults 300/3 — per-step cost ~3-4x lower but capacity also reduced). 1 epochs x batch 128 x lr 0.001 x 1 seeds (CPU-friendly, wall 0.00 min). 1/1 seeds ran cleanly. AUC is sigmoid(chemprop BinaryClassificationFFN.train_step(z)[:,0]) on the metal_heads['Ru'] head; metal_weights zeroed for Ir/Os/Rh/Re, coord_weight=0 — ChempropV2MTModel behaves as single-task binary classifier through the Ru head. Eval subsampled to eval_max_samples=100 on non-train splits (100 on train) to bound eval wall-time. BD-4 MolGraphCache hit rate = 100.0% on this run (n_skip=700, n_new=0, featurize_wall=0.0s). NO Wasserstein transfer, NO L2 guard, NO LR scheduler — pure chemprop v2 BCE baseline. Honest framing: ood_test AUC = 0.5312 (gate ≥0.55 → FAIL); if PASS, this replaces the LOMO v1 'broken like LOMO v1' honest-negative baseline for the BD-11 chemprop v2 line on lbap_refined_ic50_assay.

## Files

* `summary.json` — full metrics + per-seed
* `per_seed/seed<n>.json` — per-seed detail
* `cache/<hash>.npz` — BD-4 MolGraphCache persistent featurization
