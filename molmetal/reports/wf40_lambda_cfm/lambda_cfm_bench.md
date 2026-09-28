# Wf-40 Phase 1 — Lambda × CFM end-to-end bench

_Generated 2026-09-26T05:34:54Z — n_samples=200, seeds=[0, 1, 2, 3, 4]_

## Per-dataset aggregate

| dataset | method | mean_wall_s | mean_bridge_rmse | decode_ratio | mean_rt_loss | success_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| platinai_5k | lambda_to_cfm | 0.0070 | 1.1830 | 1.000 | nan | 1.000 |
| platinai_5k | cfm_to_lambda | 0.0081 | 1.1830 | 0.000 | nan | 1.000 |
| platinai_5k | bidirectional | 0.0123 | 1.1876 | 1.000 | 36.4104 | 1.000 |
| metalcytotox_5metal_multitask | lambda_to_cfm | 0.0057 | 0.7868 | 1.000 | nan | 1.000 |
| metalcytotox_5metal_multitask | cfm_to_lambda | 0.0065 | 0.7868 | 0.000 | nan | 1.000 |
| metalcytotox_5metal_multitask | bidirectional | 0.0080 | 0.8158 | 1.000 | 10.4517 | 1.000 |

## Bridge source

  - /home/hugo/codes/mol-metal/models/coupling_adapter_platinai5k.npz
  - alpha = 0.5
  - n_steps = 50
  - pocket_size = 64
