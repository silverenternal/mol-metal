# Wf-40 Phase 2 — Lambda × CFM Bidirectional coupling e2e

_Generated 2026-09-26T05:36:32Z — n_samples=50, seeds=[0, 1, 2, 3, 4]_

## Per-dataset aggregate (cross-seed mean ± std)

| dataset | mean_wall_s | std_wall_s | mean_bridge_rmse | std_bridge_rmse | mean_rt_loss | std_rt_loss | tile_hit_rate | success_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| platinai_5k | 0.0126 | 0.0009 | 1.2401 | 0.0001 | 38.4617 | 0.0000 | 1.000 | 1.000 |
| metalcytotox_5metal_multitask | 0.0080 | 0.0000 | 0.8360 | 0.0001 | 10.7820 | 0.0000 | 1.000 | 1.000 |

## Run configuration

  - bridge_source: /home/hugo/codes/mol-metal/models/coupling_adapter_platinai5k.npz
  - alpha: 0.5
  - n_steps: 50
  - pocket_size: 64
  - peak_memory_mb: 3.474
