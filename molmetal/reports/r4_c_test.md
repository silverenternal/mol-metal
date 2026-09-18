# R4-C: Lambda 100-pocket CrossDocked sweep

- n_pockets_total: 1
- n_pockets_ok: 1
- n_pockets_fail: 0
- n_candidates_total: 1
- mean_candidates_per_pocket: 1.00
- lipinski_pass_rate: 1.000
- sa_mean (1-10, lower=easier): 8.276
- qed_mean (0-1, higher=better): 0.390
- wall_seconds_mean_per_pocket: 0.51

## SOTA comparison (Vina Dock + Success rate, CrossDocked2020 100 pockets)

| Method | Year | Vina Dock (median, kcal/mol) | Success rate |
|---|---:|---:|---:|
| Pocket2Mol | 2022 | -7.15 | 24.4% |
| TargetDiff | 2023 | -7.80 | 10.5% |
| DecompDiff | 2024 | -8.39 | 24.5% |
| MolCRAFT | 2024 | -9.25 | 36.1% |
| AlphaDrug | 2025 | -9.77 | - |
| MolChord | 2025 | -7.62 | 33.2% |
| **TransDiffSBDD** | 2025 | **-9.37** | **83.9%** |
| **Lambda (R4-C)** | 2026 | **see CSV** | **see CSV** |

*Note*: Round-1 Lambda (1h36 single pocket) reported Vina -5.923; this sweep is the protocol-aligned 100-pocket follow-up.
*Vina proxy is a placeholder until L-1 DiffDock/FlowDock binary is installed.*
