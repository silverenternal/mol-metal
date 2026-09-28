# Wf-48 Phase 1 — score_any() Router Benchmark

Recipe: `wf48_score_any_router`
Tool version: `0.1.0-wf48-phase1`

## Aggregate

- n_scripts: `1`
- n_ok: `1`
- n_fail: `0`
- n_expected_fail: `0`
- n_routing_pass: `1`
- n_routing_fail: `0`
- mean_wall_time_s: `7.3075`
- mean_router_overhead_s: `0.0005`
- tools_used_aggregate: `QED (RDKit Bickerton), SA score (RDKit Ertl)`

## Per-script

| script | status | wall_s | overhead_s | tools_used | accuracy |
|---|---|---:|---:|---|---|
| `bench_mcts_real_trajectory.py` | OK | 7.308 | 0.001 | QED (RDKit Bickerton), SA score (RDKit Ertl) | PASS |
