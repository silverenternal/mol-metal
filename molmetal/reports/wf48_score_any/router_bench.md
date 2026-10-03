# Wf-48 Phase 1 — score_any() Router Benchmark

Recipe: `wf48_score_any_router`
Tool version: `0.1.0-wf48-phase1`

## Aggregate

- n_scripts: `3`
- n_ok: `0`
- n_fail: `3`
- n_expected_fail: `0`
- n_routing_pass: `1`
- n_routing_fail: `1`
- mean_wall_time_s: `0.0131`
- mean_router_overhead_s: `0.0006`
- tools_used_aggregate: `QED (RDKit Bickerton), SA score (RDKit Ertl)`

## Per-script

| script | status | wall_s | overhead_s | tools_used | accuracy |
|---|---|---:|---:|---|---|
| `bench_mcts_real_trajectory.py` | FAIL | 0.014 | 0.001 | QED (RDKit Bickerton), SA score (RDKit Ertl) | PASS |
| `bench_deflexpressor_end_to_end.py` | FAIL | 0.001 | 0.001 | QED (RDKit Bickerton), SA score (RDKit Ertl) | N/A |
| `benchmark_astsr_extended.py` | FAIL | 0.024 | 0.001 | QED (RDKit Bickerton), SA score (RDKit Ertl) | FAIL |
