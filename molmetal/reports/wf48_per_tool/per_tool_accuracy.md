# Wf-48 Phase 2 — Per-Tool Accuracy Benchmark

Recipe: `wf48_per_tool_accuracy`
Tool version: `0.1.0-wf48-phase2`

## Aggregate

- n_tools: `3`
- n_measured: `0`
- n_not_shipped: `0`
- n_missing_dep: `2`
- n_partial: `1`
- n_error: `0`
- mean_accuracy: `0.0000`
- mean_speed_s: `0.0007`
- mean_failure_rate: `0.6667`

## Per-tool

| tool | metric | backend | status | n_calls | accuracy | mean_speed_s | failure_rate |
|---|---|---|---|---:|---:|---:|---:|
| `AutoDock Vina 1.2.7` | `vina_score` | `python+cli` | MISSING_DEP | 1 | 0.000 | 0.001 | 1.000 |
| `DiffDock (R17 ship)` | `diffdock_pose` | `python` | MISSING_DEP | 1 | 0.000 | 0.001 | 1.000 |
| `Boltz-style pIC50 (R17 ship, D-MPNN)` | `boltz_affinity` | `python` | PARTIAL | 1 | 0.000 | 0.000 | 0.000 |
