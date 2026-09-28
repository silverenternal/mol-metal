# Test inventory — R17 + R18 ship waves

Generated: `2026-09-25T17:05:13+00:00`  
Tool: `collect_all_tests.py`  
Mode: **static-collect (no pytest execution)** (no pytest execution; counts are AST-estimated).

## Summary

| Metric | Count |
|---|---:|
| test files | 198 |
| test functions | 2637 |
| `@pytest.mark.skip` | 1 |
| `@pytest.mark.skipif` | 50 |
| `@pytest.mark.xfail` | 0 |
| parse errors | 0 |
| R17 ship files | 5 |
| R18 ship files | 53 |

## By ship wave

| Wave | Files | Tests | @skip | @skipif | @xfail |
|---|---:|---:|---:|---:|---:|
| R18 ship | 53 | 1147 | 0 | 3 | 0 |
| misc | 87 | 881 | 1 | 36 | 0 |
| P2 ML | 27 | 317 | 0 | 5 | 0 |
| Docking | 14 | 155 | 0 | 4 | 0 |
| Layer metrics | 4 | 61 | 0 | 0 | 0 |
| Integration | 6 | 38 | 0 | 0 | 0 |
| R17 ship | 5 | 20 | 0 | 2 | 0 |
| Pitfall | 2 | 18 | 0 | 0 | 0 |

## By category

| Category | Files | Tests | @skip | @skipif | @xfail |
|---|---:|---:|---:|---:|---:|
| misc | 87 | 881 | 1 | 36 | 0 |
| r18_lam_chem | 23 | 457 | 0 | 3 | 0 |
| r18_math_skills | 9 | 340 | 0 | 0 | 0 |
| p2_ml_metal_hybrid | 9 | 190 | 0 | 0 | 0 |
| r18_math_adapters | 6 | 158 | 0 | 0 | 0 |
| docking_vina | 14 | 155 | 0 | 4 | 0 |
| p2_ml_pretrain | 9 | 67 | 0 | 4 | 0 |
| r18_search_alg | 2 | 63 | 0 | 0 | 0 |
| layer_metrics | 4 | 61 | 0 | 0 | 0 |
| p2_ml_baselines | 9 | 60 | 0 | 1 | 0 |
| r18_cfm_rescue | 10 | 52 | 0 | 0 | 0 |
| r18_flow_lipman | 2 | 46 | 0 | 0 | 0 |
| integration_e2e | 6 | 38 | 0 | 0 | 0 |
| r18_reward | 1 | 31 | 0 | 0 | 0 |
| r17_legacy | 5 | 20 | 0 | 2 | 0 |
| pitfall_guards | 2 | 18 | 0 | 0 | 0 |

## Categories rollup (spec-aligned)

A shorter, dashboard-friendly view that uses the category names
from the wf_final_verification spec (math_skills, adapters, etc.).
When a category maps onto multiple internal buckets, the counts
are summed (e.g. `p2_ml` = `p2_ml_baselines` + `p2_ml_metal_hybrid`
+ `p2_ml_pretrain`).


| Category | Tests |
|---|---:|
| math_skills | 340 |
| adapters | 158 |
| search_alg | 63 |
| reward | 31 |
| flow_lipman | 46 |
| lam_chem | 457 |
| cfm_pipeline | 52 |
| lambda_cfm | 457 |
| pitfall | 18 |
| integration | 38 |
| p2_ml | 317 |
| docking | 155 |
| layer_metrics | 61 |
| r17_legacy | 20 |
| misc | 881 |

## Per-file details

Each row is one `tests/test_*.py` file. The ``Wave`` column tags the
ship wave the file belongs to (R17 / R18 / P2 ML / Docking / etc.);
``Category`` is the finer-grained family within that wave.

| File | Wave | Category | Tests | Classes | @skip | @skipif | @xfail |
|---|---|---|---:|---:|---:|---:|---:|
| `molmetal/tests/test_3d_embed.py` | misc | misc | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_adapters.py` | misc | misc | 44 | 6 | 0 | 0 | 0 |
| `molmetal/tests/test_aizynth_adapter.py` | P2 ML | p2_ml_pretrain | 28 | 4 | 0 | 1 | 0 |
| `molmetal/tests/test_aizynth_isolated.py` | P2 ML | p2_ml_pretrain | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_aizynth_real_backend.py` | P2 ML | p2_ml_pretrain | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_aizynth_v4_api.py` | P2 ML | p2_ml_pretrain | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_atom_training_contract.py` | R17 ship | r17_legacy | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_attentive_dmpnn_padding.py` | P2 ML | p2_ml_baselines | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_baselines.py` | P2 ML | p2_ml_baselines | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_baselines_lightgbm_lomo_temporal.py` | P2 ML | p2_ml_baselines | 12 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_bass_existence.py` | R18 ship | r18_math_skills | 32 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_bergman_kernel.py` | R18 ship | r18_math_skills | 49 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_biolm_score_adapter.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_buyable_gate.py` | R18 ship | r18_lam_chem | 56 | 6 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_fix1_bond_head_default.py` | R18 ship | r18_cfm_rescue | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_fix2_midpoint_solver.py` | R18 ship | r18_cfm_rescue | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_fix3_decode_smoke.py` | R18 ship | r18_cfm_rescue | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_p1_fixes.py` | R18 ship | r18_cfm_rescue | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_path_import.py` | R18 ship | r18_cfm_rescue | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_rescue_bondhead_default.py` | R18 ship | r18_cfm_rescue | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_rescue_data_scale.py` | R18 ship | r18_cfm_rescue | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_rescue_decode_smoke.py` | R18 ship | r18_cfm_rescue | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_rescue_ode_solver.py` | R18 ship | r18_cfm_rescue | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cfm_rescue_yuelbond.py` | R18 ship | r18_cfm_rescue | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cisplatin_resistance.py` | misc | misc | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cisplatin_resistance_dmpnn.py` | misc | misc | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_click_rule_effect_size_study.py` | R18 ship | r18_lam_chem | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_clone_integration_adapters.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_compare_to_published.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_conformer_embed.py` | misc | misc | 9 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_coordination_gnn.py` | R18 ship | r18_math_adapters | 32 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_counterion_utils.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_coupling_adapter.py` | R18 ship | r18_lam_chem | 18 | 0 | 0 | 3 | 0 |
| `molmetal/tests/test_coupling_adapter_platinai5k.py` | R18 ship | r18_lam_chem | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cross_attention_fusion.py` | R18 ship | r18_lam_chem | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cross_attention_m4.py` | R18 ship | r18_lam_chem | 20 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_cross_attention_v3.py` | R18 ship | r18_lam_chem | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_data.py` | misc | misc | 54 | 5 | 0 | 0 | 0 |
| `molmetal/tests/test_debug_mcts_flag.py` | misc | misc | 22 | 2 | 0 | 0 | 0 |
| `molmetal/tests/test_decoder_m3_sequence.py` | R18 ship | r18_lam_chem | 20 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_deflex_wireup_phase1_f5.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_deflex_wireup_phase2_pocket_macro.py` | misc | misc | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_deflex_wireup_phase3_learned_prior.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_deflex_wireup_phase4_v2_switch.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_deflex_wireup_phase5_integration.py` | misc | misc | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_design_loop.py` | Integration | integration_e2e | 12 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_diagnose_metric.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dmpnn.py` | P2 ML | p2_ml_baselines | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dmpnn_attn.py` | P2 ML | p2_ml_baselines | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dmpnn_baseline.py` | P2 ML | p2_ml_baselines | 15 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dmpnn_multitask.py` | P2 ML | p2_ml_baselines | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dmpnn_scatter_wiring.py` | P2 ML | p2_ml_baselines | 3 | 0 | 0 | 1 | 0 |
| `molmetal/tests/test_docking_adapters.py` | Docking | docking_vina | 40 | 3 | 0 | 0 | 0 |
| `molmetal/tests/test_docking_pose_chemistry.py` | Docking | docking_vina | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_dry_run_flag.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_duplicate_tile_fix.py` | R17 ship | r17_legacy | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_e2e.py` | Integration | integration_e2e | 4 | 1 | 0 | 0 | 0 |
| `molmetal/tests/test_egnn_coord_scatter_wired.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_egnn_predictor_device.py` | misc | misc | 1 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_egnn_velocity_cfg.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_explicit_metal_prior_sampling.py` | P2 ML | p2_ml_metal_hybrid | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_f2a_metal_ligand_exchange.py` | P2 ML | p2_ml_metal_hybrid | 61 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_fractional_ode.py` | misc | misc | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_full_pipeline_demo.py` | Integration | integration_e2e | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_fused_silu_mlp_wired.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_generate_abstract.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_generate_atom_types.py` | misc | misc | 6 | 1 | 0 | 0 | 0 |
| `molmetal/tests/test_generate_section4.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_generate_section6.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_gpu_molecular_metrics.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_gpu_vina_smoke.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_graph_batch_independence.py` | misc | misc | 1 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_hilbert_primal_dual.py` | R18 ship | r18_reward | 31 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_hilbert_score_repellent_integration.py` | R18 ship | r18_math_adapters | 15 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_hybrid.py` | Integration | integration_e2e | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_integration.py` | Integration | integration_e2e | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_astsr.py` | R18 ship | r18_lam_chem | 20 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_cfm_integration.py` | R18 ship | r18_lam_chem | 18 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_cfm_joint_inference.py` | R18 ship | r18_lam_chem | 30 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_combinators_expansion.py` | R18 ship | r18_lam_chem | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_measured_reward_dispatch.py` | R18 ship | r18_lam_chem | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lambda_reward_coupling.py` | R18 ship | r18_lam_chem | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_layer_metrics_l1_l3.py` | Layer metrics | layer_metrics | 17 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_layer_metrics_l4_l6.py` | Layer metrics | layer_metrics | 18 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_layer_metrics_l7_l8.py` | Layer metrics | layer_metrics | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_layer_metrics_l9_cross.py` | Layer metrics | layer_metrics | 15 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_layered_generation.py` | R18 ship | r18_lam_chem | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_learned_prior.py` | misc | misc | 10 | 0 | 0 | 1 | 0 |
| `molmetal/tests/test_lipman_import_isolation.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lipman_sampling_contract.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_lipman_spatial_contract.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_m5_schedulers.py` | R18 ship | r18_flow_lipman | 26 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_macroguide_ph.py` | R18 ship | r18_lam_chem | 29 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_md_relaxation_stub.py` | R18 ship | r18_lam_chem | 24 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_generator_adapter.py` | P2 ML | p2_ml_metal_hybrid | 28 | 3 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_geometry_known_pt.py` | P2 ML | p2_ml_metal_hybrid | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_hybrid.py` | P2 ML | p2_ml_metal_hybrid | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_hybrid_v3.py` | P2 ML | p2_ml_metal_hybrid | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_hybrid_v4.py` | P2 ML | p2_ml_metal_hybrid | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_ligand_exchange_matrix.py` | P2 ML | p2_ml_metal_hybrid | 27 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metal_smiles.py` | P2 ML | p2_ml_metal_hybrid | 50 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_metalloprotein_targets.py` | misc | misc | 14 | 2 | 0 | 0 | 0 |
| `molmetal/tests/test_metric_credibility_audit.py` | misc | misc | 15 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_metric_hooks.py` | misc | misc | 22 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_metrics_v2.py` | misc | misc | 28 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_minibatch_ot.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_mmp_targets.py` | misc | misc | 18 | 2 | 0 | 0 | 0 |
| `molmetal/tests/test_molsimplify_check.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_monge_ampere_regularity.py` | R18 ship | r18_math_skills | 34 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_multi_parameter_ph.py` | R18 ship | r18_math_skills | 45 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_multi_ph_integration.py` | R18 ship | r18_math_adapters | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_non_click_medchem_rules.py` | R18 ship | r18_lam_chem | 33 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_ood_protocol.py` | R18 ship | r18_lam_chem | 48 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_oov_error_message.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_ot_effective_backend.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_persistent_mcts_tree.py` | misc | misc | 16 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_ph_hardness_guardrail.py` | R18 ship | r18_math_skills | 31 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pharmacophore_filter.py` | misc | misc | 16 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_physical_sweep_reporting.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pic50_predictor.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pic50_uncertainty.py` | misc | misc | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pitfall_guard_json.py` | Pitfall | pitfall_guards | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pitfall_guards.py` | Pitfall | pitfall_guards | 12 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pll_signature.py` | misc | misc | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pocket_conditioned_lipman.py` | Docking | docking_vina | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pocket_docking_reward.py` | Docking | docking_vina | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pocket_invariance_combined.py` | Docking | docking_vina | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pocket_macro_inference_path.py` | Docking | docking_vina | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_ports.py` | misc | misc | 28 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_posebusters_adapter.py` | Docking | docking_vina | 28 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_posebusters_doc.py` | Docking | docking_vina | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_posebusters_mmff94s_relax.py` | Docking | docking_vina | 18 | 4 | 0 | 1 | 0 |
| `molmetal/tests/test_prepare_crossdocked_receptor.py` | Docking | docking_vina | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_proof_search_strengthened.py` | misc | misc | 38 | 4 | 0 | 0 | 0 |
| `molmetal/tests/test_protocol_alignment.py` | misc | misc | 19 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pt_click_compat_smoke.py` | R18 ship | r18_lam_chem | 15 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_pt_metal_ligand_exchange_known.py` | R18 ship | r18_lam_chem | 10 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_quotient_space_diffusion.py` | R18 ship | r18_math_adapters | 36 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r10_wrap_ordering.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r15_yuelbond_wire.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r17_legacy_compat.py` | R17 ship | r17_legacy | 1 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r17_paper_sweep_presets.py` | R17 ship | r17_legacy | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r18_imports.py` | R18 ship | r18_lam_chem | 22 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_r4c_full_sweep.py` | misc | misc | 6 | 0 | 0 | 5 | 0 |
| `molmetal/tests/test_r4c_pair_execution.py` | misc | misc | 12 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_reaction_confidence.py` | misc | misc | 13 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_real_qm9_ot_protocol.py` | misc | misc | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_receptor_preparation_for_evaluation.py` | Docking | docking_vina | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_reducer.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_reinvent_prior_adapter.py` | misc | misc | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_reproduce_R17.py` | R17 ship | r17_legacy | 5 | 0 | 0 | 2 | 0 |
| `molmetal/tests/test_reproducibility_split.py` | misc | misc | 14 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_retrosynthesis.py` | P2 ML | p2_ml_pretrain | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_rocm_egnn.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_rocm_env.py` | misc | misc | 8 | 2 | 0 | 0 | 0 |
| `molmetal/tests/test_rocm_lipman.py` | misc | misc | 8 | 3 | 1 | 0 | 0 |
| `molmetal/tests/test_rocm_throughput.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_round11_engine_parity.py` | misc | misc | 7 | 0 | 0 | 1 | 0 |
| `molmetal/tests/test_round5_kernels_wired.py` | misc | misc | 8 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_run_pb_production.py` | misc | misc | 7 | 0 | 0 | 1 | 0 |
| `molmetal/tests/test_sa_score.py` | misc | misc | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_sb_bass_martingale.py` | R18 ship | r18_math_skills | 38 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_scatter_backend_selector.py` | misc | misc | 10 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_score_repellent_mc.py` | R18 ship | r18_search_alg | 23 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_se3_sb_integration.py` | R18 ship | r18_math_adapters | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_se3_schrodinger_bridge.py` | R18 ship | r18_math_skills | 40 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_skeleton_jump.py` | R18 ship | r18_lam_chem | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_smoke_flag.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_smoke_scripts.py` | Integration | integration_e2e | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_sota_aligned.py` | misc | misc | 10 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_stereo_aware_reduction.py` | misc | misc | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_sweep_config.py` | misc | misc | 12 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_symbolic_regression.py` | misc | misc | 18 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_synflownet_env.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_syntemol_reactions.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_synthemol_reactions.py` | misc | misc | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_temporal_grids.py` | misc | misc | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_tmqm_loading.py` | P2 ML | p2_ml_pretrain | 3 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_tmqm_pretrain.py` | P2 ML | p2_ml_pretrain | 9 | 0 | 0 | 3 | 0 |
| `molmetal/tests/test_tmqm_shape_bridge.py` | P2 ML | p2_ml_pretrain | 5 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_tmqm_wireup.py` | P2 ML | p2_ml_pretrain | 6 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_training_novelty.py` | misc | misc | 4 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_transport_relaxation.py` | R18 ship | r18_math_skills | 29 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_transport_relaxation_integration.py` | R18 ship | r18_math_adapters | 25 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_typed_dispatch_triton.py` | misc | misc | 11 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_v4_multitask_dmpnn.py` | P2 ML | p2_ml_baselines | 12 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_validate_generate_metrics.py` | misc | misc | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_variance_puct_integration.py` | R18 ship | r18_search_alg | 40 | 5 | 0 | 0 | 0 |
| `molmetal/tests/test_velocity_singleton_batch.py` | misc | misc | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_vina_3engine_parity.py` | Docking | docking_vina | 2 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_vina_adapter.py` | Docking | docking_vina | 22 | 3 | 0 | 3 | 0 |
| `molmetal/tests/test_vina_gpu_adapter.py` | Docking | docking_vina | 7 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_w_flow.py` | R18 ship | r18_flow_lipman | 20 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_wasserstein_gradient_flow.py` | R18 ship | r18_math_skills | 42 | 0 | 0 | 0 | 0 |
| `molmetal/tests/test_wetlab_channel.py` | misc | misc | 29 | 0 | 0 | 25 | 0 |
| `molmetal/tests/test_wf_coupling_ablation.py` | misc | misc | 7 | 0 | 0 | 2 | 0 |
| `molmetal/tests/test_wf_lambda_boost.py` | misc | misc | 42 | 5 | 0 | 0 | 0 |
| `molmetal/tests/test_zenodo_fetch.py` | misc | misc | 35 | 5 | 0 | 1 | 0 |

## Ship-wave contributions

Each R17 / R18 ship module ships with one or more pytest tests.
The mapping below links ship waves to the tests that guard them.
Use it to verify that no shipped module is test-less before
closing the ship wave.


| Wave | Category | Files contributing | Total tests |
|---|---|---:|---:|
| Docking | docking_vina | 14 | 155 |
| Integration | integration_e2e | 6 | 38 |
| Layer metrics | layer_metrics | 4 | 61 |
| P2 ML | p2_ml_baselines | 9 | 60 |
| P2 ML | p2_ml_metal_hybrid | 9 | 190 |
| P2 ML | p2_ml_pretrain | 9 | 67 |
| Pitfall | pitfall_guards | 2 | 18 |
| R17 ship | r17_legacy | 5 | 20 |
| R18 ship | r18_cfm_rescue | 10 | 52 |
| R18 ship | r18_flow_lipman | 2 | 46 |
| R18 ship | r18_lam_chem | 23 | 457 |
| R18 ship | r18_math_adapters | 6 | 158 |
| R18 ship | r18_math_skills | 9 | 340 |
| R18 ship | r18_reward | 1 | 31 |
| R18 ship | r18_search_alg | 2 | 63 |
| misc | misc | 87 | 881 |

## Notes

* This inventory is **static / AST-only** — no `pytest` was
  invoked. Counts above are *expected* counts based on parsing
  `test_*.py` files; actual pass/fail/skip numbers must come
  from running `pytest --collect-only` on a populated env.
* `@pytest.mark.skip` / `skipif` / `xfail` decorators are
  detected via regex; a decorator split across multiple lines
  is still counted (the regex is anchored to the marker, not
  to line boundaries).
* The classification table is curated in
  `molmetal/scripts/collect_all_tests.py`; extend it in
  lock-step with `verify_r18_all_imports.SHIP_MODULES` when
  adding new ship modules.

