# Wf-330-T1 triangle closure on PlatinAI 4-cell (F5_RDKIT_2D 12-d, real pIC50)

n_seeds = 5, n_cell_lines = 4, n_trajectories = 100, n_rounds = 5, hilbert_iters = 100

## Global metrics (mean of per-cell means)

| metric | value |
|--------|------:|
| mean_hilbert_r2 | -194.4127 |
| mean_evolutionary_r2 | -111640.7007 |
| mean_variance_reduction | 0.0013 |
| delta_hilbert_r2_vs_wf329w2 | -96.3327 |
| delta_varred_vs_wf329w2 | -0.0084 |
| recovery_status | HONEST_NEGATIVE_PERSISTENT |

## Per-cell breakdown

| cell | n_seeds | mean_hilbert_r2 | std_hilbert_r2 | mean_evolutionary_r2 | std_evolutionary_r2 | mean_variance_reduction | std_variance_reduction |
|------|--------:|----------------:|---------------:|----------------------:|---------------------:|----------------------:|---------------------:|
| mcf7 | 5 | -201.8890 | 23.8021 | -137520.3855 | 271344.4989 | 0.0013 | 0.0025 |
| a549 | 5 | -194.2751 | 22.9881 | -128842.9076 | 254008.3700 | 0.0013 | 0.0026 |
| a2780 | 5 | -215.1872 | 28.1540 | -75651.2511 | 147000.8401 | 0.0013 | 0.0025 |
| a2780cis | 5 | -166.2995 | 31.7800 | -104548.2587 | 206322.2968 | 0.0013 | 0.0025 |

## Honest-negative recovery verdict

- Wf-329-W2 MEASURED baseline: hilbert_r2=-98.0800, varred=+0.0097
- PROJECTED: hilbert_r2 ≥ 0.6000, varred ≥ 0.0500
- MEASURED: hilbert_r2 = -194.4127, varred = +0.0013
- RECOVERY_STATUS = **HONEST_NEGATIVE_PERSISTENT**
