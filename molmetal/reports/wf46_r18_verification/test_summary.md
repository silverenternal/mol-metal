# Wf-46 Phase 3 — R18 ship test run summary

Generated at `2026-09-26T07:22:20Z` (UTC).

## Headline numbers

| Metric | Value |
|---|---:|
| pytest return code | `1` |
| test files discovered | 81 |
| test files with tests | 71 |
| tests collected (collect-only) | 2562 |
| tests collected (full run) | 2562 |
| passed | 2500 |
| failed | 23 |
| errors | 0 |
| skipped (incl. xfail) | 39 |
| xfailed | 0 |
| xpassed | 0 |
| collect wall-time (s) | 7.58 |
| full-run wall-time (s) | 346.31 |
| total wall-time (s) | 354.03 |
| pass rate (passed / run) | 97.58% |

## By category

| Category | Files | Tests | Passed | Failed | Errors | Skipped | Wall-time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ml_methods` | 23 | 1103 | 1061 | 16 | 0 | 26 | 43.92 |
| `math_skills` | 14 | 484 | 484 | 0 | 0 | 0 | 1.47 |
| `r18_orchestrator` | 6 | 363 | 362 | 1 | 0 | 0 | 161.17 |
| `integration` | 16 | 291 | 290 | 1 | 0 | 0 | 89.95 |
| `adapter` | 20 | 91 | 83 | 2 | 0 | 6 | 17.67 |
| `pitfall` | 2 | 24 | 23 | 1 | 0 | 0 | 12.88 |

## Per-file (top 30 by test count)

| File | Tests | Passed | Failed | Skipped | Wall-time (s) |
|---|---:|---:|---:|---:|---:|
| `molmetal/tests/test_metal_ligand_exchange_matrix.py` | 659 | 659 | 0 | 0 | 0.20 |
| `molmetal/tests/test_f2a_metal_ligand_exchange.py` | 316 | 306 | 0 | 10 | 0.12 |
| `molmetal/tests/test_r18_modules_import.py` | 189 | 188 | 1 | 0 | 134.35 |
| `molmetal/tests/test_r18_imports.py` | 53 | 53 | 0 | 0 | 0.00 |
| `molmetal/tests/test_bergman_kernel.py` | 49 | 49 | 0 | 0 | 0.02 |
| `molmetal/tests/test_r18_train_scripts.py` | 47 | 47 | 0 | 0 | 20.15 |
| `molmetal/tests/test_multi_parameter_ph.py` | 45 | 45 | 0 | 0 | 0.04 |
| `molmetal/tests/test_wasserstein_gradient_flow.py` | 42 | 42 | 0 | 0 | 0.01 |
| `molmetal/tests/test_se3_schrodinger_bridge.py` | 40 | 40 | 0 | 0 | 0.14 |
| `molmetal/tests/test_r18_pipeline_orchestrator.py` | 39 | 39 | 0 | 0 | 3.38 |
| `molmetal/tests/test_sb_bass_martingale.py` | 38 | 38 | 0 | 0 | 0.03 |
| `molmetal/tests/test_monge_ampere_regularity.py` | 34 | 34 | 0 | 0 | 0.12 |
| `molmetal/tests/test_multi_ph_pll_integration.py` | 34 | 34 | 0 | 0 | 0.24 |
| `molmetal/tests/test_bass_existence.py` | 32 | 32 | 0 | 0 | 0.00 |
| `molmetal/tests/test_quantum_semantics.py` | 32 | 32 | 0 | 0 | 0.03 |
| `molmetal/tests/test_hilbert_primal_dual.py` | 31 | 31 | 0 | 0 | 0.54 |
| `molmetal/tests/test_hochschild_cohomology.py` | 31 | 31 | 0 | 0 | 0.01 |
| `molmetal/tests/test_medchem_gen_integration.py` | 31 | 31 | 0 | 0 | 0.05 |
| `molmetal/tests/test_ph_hardness_guardrail.py` | 31 | 31 | 0 | 0 | 0.01 |
| `molmetal/tests/test_lambda_cfm_joint_inference.py` | 30 | 30 | 0 | 0 | 0.04 |
| `molmetal/tests/test_transport_relaxation.py` | 29 | 29 | 0 | 0 | 0.34 |
| `molmetal/tests/test_hilbert_astsr_integration.py` | 26 | 26 | 0 | 0 | 4.10 |
| `molmetal/tests/test_fractional_ode.py` | 25 | 25 | 0 | 0 | 0.16 |
| `molmetal/tests/test_multi_ph_integration.py` | 25 | 25 | 0 | 0 | 0.12 |
| `molmetal/tests/test_se3_lipman_integration.py` | 25 | 25 | 0 | 0 | 0.43 |
| `molmetal/tests/test_se3_sb_integration.py` | 25 | 25 | 0 | 0 | 0.30 |
| `molmetal/tests/test_transport_relaxation_integration.py` | 25 | 25 | 0 | 0 | 0.02 |
| `molmetal/tests/test_w_flow_cfm_integration.py` | 25 | 25 | 0 | 0 | 0.07 |
| `molmetal/tests/test_integration_scoring.py` | 23 | 23 | 0 | 0 | 23.81 |
| `molmetal/tests/test_r18_test_inventory.py` | 20 | 20 | 0 | 0 | 0.00 |

## Per-file (top 10 slowest)

| File | Tests | Wall-time (s) |
|---|---:|---:|
| `molmetal/tests/test_r18_modules_import.py` | 189 | 134.35 |
| `molmetal/tests/test_integration.py` | 5 | 59.98 |
| `molmetal/tests/test_integration_scoring.py` | 23 | 23.81 |
| `molmetal/tests/test_r18_train_scripts.py` | 47 | 20.15 |
| `molmetal/tests/test_coupling_adapter_platinai5k.py` | 11 | 16.21 |
| `molmetal/tests/test_dmpnn_multitask.py` | 3 | 14.65 |
| `molmetal/tests/test_pitfall_guard_json.py` | 6 | 12.86 |
| `molmetal/tests/test_baselines_lightgbm_lomo_temporal.py` | 12 | 9.91 |
| `molmetal/tests/test_explicit_metal_prior_sampling.py` | 5 | 7.46 |
| `molmetal/tests/test_metal_hybrid.py` | 3 | 4.45 |

## Failed files

| File | Failed | First failure (truncated) |
|---|---:|---|
| `molmetal/tests/test_baselines_lightgbm_lomo_temporal.py` | 5 | `test_lomo_runs_on_three_metals` |
| `molmetal/tests/test_baselines.py` | 3 | `test_xgb_smoke_ru` |
| `molmetal/tests/test_metal_hybrid_v4.py` | 2 | `test_v4_forward_smoke` |
| `molmetal/tests/test_dmpnn.py` | 2 | `test_dmpnn_train_step_loss_decreases` |
| `molmetal/tests/test_vina_3engine_parity.py` | 2 | `test_vina_3engine_parity` |
| `molmetal/tests/test_r18_modules_import.py` | 1 | `test_help_ship_script[verify_train_affinity_regression]` |
| `molmetal/tests/test_pitfall_guards.py` | 1 | `test_p3_metal_seed_cisplatin_is_a_valid_choice` |
| `molmetal/tests/test_dmpnn_baseline.py` | 1 | `test_end_to_end_baseline_run_small_subset` |
| `molmetal/tests/test_explicit_metal_prior_sampling.py` | 1 | `test_actual_rocm_sampler_prior_active_on_every_graph_and_reduces_energy` |
| `molmetal/tests/test_integration.py` | 1 | `test_001_lipman_to_vina_e2e` |
| `molmetal/tests/test_metal_hybrid_v3.py` | 1 | `test_encoder_load` |
| `molmetal/tests/test_dmpnn_attn.py` | 1 | `test_dmpnn_attn_train_step_loss_decreases` |

## Methodology

- **Phase A (collect-only)** runs `pytest --collect-only -q`
  over the R18-relevant test files to enumerate all tests.
- **Phase B (full run)** runs `pytest -q --tb=line --junit-xml`
  over the same file set and parses the JUnit XML for
  per-testcase status + per-test wall-time.
- **Per-module roll-up** aggregates per-testcase entries back
  into the file's `tests_count / passed / failed / skipped /
  wall_time` schema.

JSON written: `/home/hugo/codes/mol-metal/molmetal/reports/wf46_r18_verification/test_run_results.json`