# Wf-310D DrugOOD DANN full aggregation -- claim cards

Date: 2026-09-29
n_seeds: 3  test_auc_mean: 0.8024  test_auc_std: 0.0224
wall_min: 5.86  delta_vs_wf185: +0.1317
delta_vs_wf301a_smoke: -0.0000

## Card 1 -- per-seed ood_test_auc (MEASURED)

| seed | test_auc | train_auc | val_auc | elapsed_s | n_test |
|------|----------|-----------|---------|-----------|--------|
| 0 | 0.8311 | 0.8934 | 0.7951 | 119.8 | 19047 |
| 1 | 0.7995 | 0.8666 | 0.7625 | 164.8 | 19047 |
| 2 | 0.7765 | 0.8504 | 0.7301 | 66.7 | 19047 |

## Card 2 -- pass-gate verdict (MEASURED)

- Wf-185 v2 baseline: 0.6707 ± 0.0042
- Wf-301A smoke (CPU, 1-epoch, 3-seed): 0.8024
- Field-avg target band: 0.55 - 0.65
- Full DANN 3-seed: 0.8024 ± 0.0224
- delta_vs_wf185: +0.1317
- delta_vs_wf301a_smoke: -0.0000  (full vs smoke -- both ECFP4 + DANN, different epoch count + different seed count)

## Card 3 -- honest-negative flags (per Wf-301A honest-negative)

- Wf-185 Δ is across DIFFERENT subsets (DrugOOD scaffold vs   lbap_refined_ic50 scaffold); cross-substrate comparison is   directional only.
- Full run vs smoke Δ mixes epoch-budget effect with seed-noise.
- DANN scaffold-binning may not generalise beyond top-15 + OOV   scaffold frequency (per Wf-301A honest-negative §HONEST_NEGATIVE).

## Card 4 -- wall + GPU cost (PROJECTED)

- Wall: 5.9 min (0.10 hr) -- MEASURED.
- GPU: cuda:1 (RTX 5090) per Wf-306A launch_wf306a_drugood_vina_gpu.sh.
- ~3-4 hr GPU-1 PROJECTED per Wf-306A Phase-A budget.
