# `tests/` — Pytest test suite

The pytest suite for Mol-Metal. Tests are organized **by mechanism ID** (M1-M21)
and **by round** (R4, R10, R11, R15, R17) so failures point immediately at the
mechanism regressed and the round that introduced the regression. The
`test_pitfall_guards.py` / `test_pitfall_guard_json.py` files encode the
canonical pitfall checks that gate every workflow.

## Layout

| Group | Files | Purpose |
|-------|-------|---------|
| **Mechanism tests** | `test_lambda_combinators_expansion.py`, `test_lambda_astsr.py`, `test_proof_search_strengthened.py`, `test_pocket_conditioned_lipman.py`, `test_lipman_spatial_contract.py`, `test_metal_ligand_exchange_matrix.py`, `test_reaction_confidence.py`, `test_cfm_fix*.py` | One file per M-ID (M1-M17). |
| **Round tests** | `test_r4c_*`, `test_r10_*`, `test_round10_*`, `test_round11_*`, `test_r15_*`, `test_r17_*` | Per-round regression. |
| **Adapter tests** | `test_aizynth_*`, `test_reinvent_*`, `test_docking_*`, `test_posebusters_*`, `test_synflownet_env.py` | Per-evaluator smoke. |
| **Data tests** | `test_data.py`, `test_cross_attention_fusion.py`, `test_egnn_coord_scatter_wired.py`, `test_reproducer_split.py` | Leakage + correctness. |
| **Pitfall guards** | `test_pitfall_guards.py`, `test_pitfall_guard_json.py` | Workflow gate checks. |
| **Layer metrics** | `test_layer_metrics_l1_l3.py`, `test_layer_metrics_l4_l6.py`, `test_layer_metrics_l7_l8.py`, `test_layer_metrics_l9_cross.py` | Per-MLC-layer metrics. |
| **Hardware** | `test_rocm_*`, `test_gpu_*` | ROCm / GPU smoke. |
| **Sweep presets** | `test_sweep_config.py`, `test_r17_paper_sweep_presets.py`, `test_round5_kernels_wired.py` | Cfg / OT / kernel ablations. |

## Usage

```bash
# Run the full pitfall-guard gate (fast, required before commit)
cd /home/hugo/codes/mol-metal
pytest molmetal/tests/test_pitfall_guards.py molmetal/tests/test_pitfall_guard_json.py -q

# Run mechanism-M1 (Lambda calculus AST) tests
pytest molmetal/tests/test_lambda_combinators_expansion.py -q

# Run R17 round regression
pytest molmetal/tests/test_r17_*.py molmetal/tests/test_round11_*.py -q
```

```bash
# Run all CFM-fix tests (M13, M15, M16, M17)
pytest molmetal/tests/test_cfm_fix*.py molmetal/tests/test_cfm_rescue*.py -q
```

## Common pitfalls

* `test_lipman_spatial_contract.py` requires the **flat-clone** checkpoint
  format — failing here almost always means M17 (loader) regressed.
* `test_reproducer_split.py` enforces **scaffold split** — a flake here is
  usually a data-side leak, not a model bug.
* `test_pitfall_guards.py` is the **workflow gate** — never `pytest.skip`
  these in CI.
* ROCm tests are tagged `@pytest.mark.requires_rocm` and skipped on CPU.
* Mechanism ↔ metric cross-refs: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  lists each mechanism's canonical test file.
* Fixtures: `conftest.py` (project root) sets `PROJECT_ROOT` + seeds.

## Cross-reference index

* Mechanisms ↔ tests: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §1.
* Workflow gates: `wf_*` scripts in `molmetal/scripts/` invoke the pitfall
  guards before any non-trivial work.
