# Wf-39 Phase 1 — MCTS Real-Trajectory Bench Summary

Tool version: `0.1.0-wf39-phase1`

## Per-strategy aggregate

| strategy | n_traj | n_ok | n_err | success_rate | mean_wall_s | mean_n_states | mean_best | mean_leaf_var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `proof_search` | 5 | 5 | 0 | 1.000 | 4.6111 | 477.40 | 1.500 | 0.0000 |
| `variance_mcts` | 5 | 5 | 0 | 1.000 | 4.1633 | 477.20 | 1.500 | 0.0000 |

## Vulnerability scan

### vuln_a — verdict `OK`

- `proof_search_error_rate` = `0.0`
- `variance_mcts_error_rate` = `0.0`
- `proof_search_top_error_mode` = `none`
- `variance_mcts_top_error_mode` = `none`

### vuln_b — verdict `OK`

- `mean_leaf_value_var_proof_search` = `0.0`
- `mean_leaf_value_var_variance_mcts` = `0.0`
- `variance_ratio` = `0.0`
- `target_ratio_max` = `0.8`

### vuln_c — verdict `CRITICAL`

- `saturation_rate_proof_search` = `1.0`
- `saturation_rate_variance_mcts` = `1.0`

## Datasets

- `platinai_5k`: 5 rows sampled
