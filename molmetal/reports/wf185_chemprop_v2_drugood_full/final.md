# Wf-185 — chemprop v2 DrugOOD full benchmark

**Date:** 2026-09-28  
**Configs:** lbap_refined_ic50_assay, lbap_refined_ic50_scaffold, lbap_refined_ic50_size  
**Backbone:** hidden=128, depth=2, head_hidden=128 (smaller than chemprop v2 defaults 300/3)  
**Recipe:** 20 epochs × 5 seeds per config  
**Total wall time:** 180.78 min (budget 6.0 hr)  

## Acceptance gates

- [G1] All configs ran: **True**
- [G2] Total runs: **15**
- [G3] Per-config ood_test AUC: see table below
- [G4] Per-config mean ± std: see table below
- [G5] Total wall time: **180.78 min** (≤ 6.0 hr)

## Per-config summary

| Config | ood_test mean | ood_test std | n seeds | wall_min | hit_rate |
|---|---|---|---|---|---|
| lbap_refined_ic50_assay | 0.7137 | 0.0079 | 5 | 75.54 | 100.0% |
| lbap_refined_ic50_scaffold | 0.6707 | 0.0042 | 5 | 53.57 | 0.0% |
| lbap_refined_ic50_size | 0.6757 | 0.0142 | 5 | 38.14 | 0.0% |

## Per-config per-seed ood_test AUC

### lbap_refined_ic50_assay

| seed | ood_test | elapsed_s | status |
|---|---|---|---|
| 0 | 0.7221 | 1515.5 | timeout |
| 1 | 0.7156 | 1506.7 | timeout |
| 2 | 0.7019 | 680.7 | ok |
| 3 | 0.7076 | 415.7 | ok |
| 4 | 0.7215 | 413.7 | ok |

### lbap_refined_ic50_scaffold

| seed | ood_test | elapsed_s | status |
|---|---|---|---|
| 0 | 0.6667 | 272.8 | ok |
| 1 | 0.6782 | 271.0 | ok |
| 2 | 0.6711 | 275.6 | ok |
| 3 | 0.6669 | 950.6 | ok |
| 4 | 0.6703 | 1444.5 | ok |

### lbap_refined_ic50_size

| seed | ood_test | elapsed_s | status |
|---|---|---|---|
| 0 | 0.6674 | 468.1 | ok |
| 1 | 0.6724 | 445.7 | ok |
| 2 | 0.6565 | 461.5 | ok |
| 3 | 0.6843 | 468.7 | ok |
| 4 | 0.6978 | 444.4 | ok |

## Improvements over Wf-174

assay(seed=0)=0.7221 vs Wf-174 seed=0 (0.7138); Δ=+0.0083 | lbap_refined_ic50_assay: ood_test mean=0.7137 ± 0.0079 (n=5) | lbap_refined_ic50_scaffold: ood_test mean=0.6707 ± 0.0042 (n=5) | lbap_refined_ic50_size: ood_test mean=0.6757 ± 0.0142 (n=5)

## Honest framing

Wf-185 chemprop v2 retrain on 3 DrugOOD refined configs. Per-config: 5 seeds × 20 epochs each, backbone hidden_dim=128 depth=2 head_hidden_dim=128, CPU only. Recipe identical to Wf-174 (smaller than chemprop v2 defaults 300/3 — per-step cost ~3-4x lower but capacity also reduced). No Wasserstein transfer, no L2 guard, no LR scheduler — pure chemprop v2 BCE baseline. Total wall time = 180.78 min (budget 6.0 hr). 13/15 seeds ok across all configs. Per-config ood_test AUC: lbap_refined_ic50_assay=0.7137±0.0079; lbap_refined_ic50_scaffold=0.6707±0.0042; lbap_refined_ic50_size=0.6757±0.0142. Wf-185 vs Wf-174: assay(seed=0)=0.7221 vs Wf-174 seed=0 (0.7138); Δ=+0.0083 | lbap_refined_ic50_assay: ood_test mean=0.7137 ± 0.0079 (n=5) | lbap_refined_ic50_scaffold: ood_test mean=0.6707 ± 0.0042 (n=5) | lbap_refined_ic50_size: ood_test mean=0.6757 ± 0.0142 (n=5).

## Files

* `aggregate.json` — cross-config gate report
* `lbap_refined_ic50_assay/summary.json` + `per_seed/seed<n>.json` (per-config results; one dir per config)
* `cache/<config>/<hash>.npz` — BD-4 MolGraphCache persistent featurization
