# Wf-48 Phase 2 — Per-Tool Accuracy Benchmark

Recipe: `wf48_per_tool_accuracy`
Tool version: `0.1.0-wf48-phase2`

## Aggregate

- n_tools: `6`
- n_measured: `2`
- n_not_shipped: `0`
- n_missing_dep: `3`
- n_partial: `1`
- n_error: `0`
- mean_accuracy: `0.1875`
- mean_speed_s: `0.6500`
- mean_failure_rate: `0.5000`

## Per-tool

| tool | metric | backend | status | n_calls | accuracy | mean_speed_s | failure_rate |
|---|---|---|---|---:|---:|---:|---:|
| `AutoDock Vina 1.2.7` | `vina_score` | `python+cli` | MISSING_DEP | 1 | 0.000 | 1.009 | 1.000 |
| `DiffDock (R17 ship)` | `diffdock_pose` | `python` | MISSING_DEP | 1 | 0.000 | 0.001 | 1.000 |
| `Boltz-style pIC50 (R17 ship, D-MPNN)` | `boltz_affinity` | `python` | PARTIAL | 2 | 0.000 | 0.099 | 0.000 |
| `Voxelization complexity (AF2 surrogate)` | `af2_confidence` | `python` | MISSING_DEP | 1 | 0.000 | 0.001 | 1.000 |
| `PoseBusters (R17 ship)` | `pb_pass_rate` | `python` | OK | 2 | 0.125 | 2.788 | 0.000 |
| `AiZynthFinder (R17 ship)` | `aizynth_synthesis` | `python` | OK | 2 | 1.000 | 0.002 | 0.000 |
