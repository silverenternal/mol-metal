# Wf-29 Phase 2 — 4-Recipe Smoke Comparison

_Generated_: 2026-09-26  
_Smoke dir_: `/home/hugo/codes/mol-metal/molmetal/reports/wf29_smoke`  
_Recipes recorded_: se3sb, transport, multiph, baseline  

## Per-recipe headline (Ru test AUC)

| Recipe | Init | n_seeds | Ru AUC mean | Ru AUC per seed | Wall (s) | Epochs | Provenance |
|---|---|---|---|---|---|---|---|
| se3sb | se3sb | 3 | 0.5275 | 0.503, 0.589, 0.490 | 269.8 | 10 | MEASURED |
| transport | transportrelax | 3 | 0.5358 | 0.573, 0.560, 0.474 | 275.4 | 10 | MEASURED |
| multiph | multiph | 3 | 0.5330 | 0.564, 0.592, 0.443 | 245.2 | 10 | MEASURED |
| baseline | baseline | 3 | 0.6576 | 0.737, 0.552, 0.684 | 363.8 | 10 | MEASURED |

## Per-metal mean test AUC

| Recipe | Ru | Ir | Rh | Re | Os |
|---|---|---|---|---|---|
| se3sb | 0.528 | 0.578 | 0.706 | 0.534 | 0.634 |
| transport | 0.536 | 0.584 | 0.600 | 0.501 | 0.619 |
| multiph | 0.533 | 0.583 | 0.663 | 0.505 | 0.652 |
| baseline | — | — | — | — | — |

## Honest-negative provenance

All ru_mean_test_auc values come from the Wf-29 Phase 2 smoke benchmark (1-3 seeds × 10 epochs × small arch on GPU 1) and are MEASURED — they are NOT directly comparable to the Wf-31 production runs which use 5 seeds × 50 epochs on the full data.
