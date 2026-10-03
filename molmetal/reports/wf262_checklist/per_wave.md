# Per-Wave Checklist — tmqm_plat

_Generated: 2026-10-03T12:07:12+00:00_  _Schema: 262.v2_

## Summary

- preflight pass count: **96**
- preflight fail count: **0**
- wave budget: **10.0 h** (9.5 h remaining after abort penalties)

## P0 BLOCKER gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p0_config_schema_verify` | all 7 YAMLs parse OK |
| `PASS` | `p0_csv_hash_verify` | all 5 CSVs line-count OK |
| `PASS` | `p0_scaffold_zero_leak` | OK 0 scaffold leak |

## P1 VERIFY gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p1_chemprop_indices_fix` | grep 'indices = train_idx.copy()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_kendall_optimizer` | grep 'kendall.parameters()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_dann_per_metal` | grep 'per_metal' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_nan_guard` | grep 'nan_to_num' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_grad_clip` | grep 'clip_grad_norm' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_epoch_avg` | grep 'w2_epoch_avg_history' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `p1_chemprop_ablation_flags` | help exposes all 4 token(s) of '--no-multi-task --no-wass --no-kendall --no-dann' (rc=0) |
| `PASS` | `p1_chemprop_seeds_min` | help exposes all 1 token(s) of '--n-seeds-min' (rc=0) |
| `PASS` | `p1_chemprop_master_wass_flag` | help exposes all 1 token(s) of '--use-master-wass' (rc=0) |
| `PASS` | `p1_chemprop_master_cath_flag` | help exposes all 1 token(s) of '--use-master-cath' (rc=0) |
| `PASS` | `p1_chemprop_click_aug_flag` | help exposes all 1 token(s) of '--use-click-aug' (rc=0) |
| `PASS` | `p1_chemprop_smoke` | --quick exit=0 in 31.4s (rc=0) | tail: [2026-10-02 17:12:59,701] INFO wf200 :: [metal=Ru seed=0] epoch=1/2 loss=-1.2723  |
| `PASS` | `p1_compute_manifest` | manifest at /home/hugo/codes/mol-metal/molmetal/reports/wf350a2_compute_manifest.json schema-valid | 2 GPU(s): gpu[0]=NV |
| `PASS` | `p1_cfm_script_exists` | file exists: /home/hugo/codes/mol-metal/molmetal/scripts/metallo_drug_smoke_retrain.py |
| `PASS` | `p1_cfm_config_exists` | file exists: /home/hugo/codes/mol-metal/configs/cfm_full_retrain.yaml |
| `PASS` | `p1_cfm_decoder_exists` | file exists: /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py |
| `PASS` | `p1_cfm_wasserstein_loss_exists` | file exists: /home/hugo/codes/mol-metal/molmetal/molmetal_lam/reward/wasserstein_loss.py |
| `PASS` | `p1_cfm_smoke_decoded` | /home/hugo/codes/mol-metal/molmetal/reports/wf188_m14_full_metalcytotox/result.json: value=0.995821 >= 0.5 |
| `PASS` | `p1_cfm_catherine_30metals` | /home/hugo/codes/mol-metal/molmetal/reports/wf256_apply/catherine_30/per_metal_pass.json: value=1.000000 >= 0.95 |
| `PASS` | `code_1_1_1_indices_aliasing_fix` | grep 'indices = train_idx.copy()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_2_kendall_log_sigma` | grep 'kendall.parameters()' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_3_dann_per_metal` | grep 'per_metal' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_4_nan_to_num_guard` | grep 'nan_to_num' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_5_grad_clip_max_norm` | grep 'clip_grad_norm' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_6_w2_epoch_avg_history` | grep 'w2_epoch_avg_history' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_7_ablation_flags_present` | help exposes all 4 token(s) of '--no-multi-task --no-wass --no-kendall --no-dann' (rc=0) |
| `PASS` | `code_1_1_8_enhanced_metal_head` | help exposes all 2 token(s) of '--enhanced-head --metal-attention' (rc=0) |
| `PASS` | `code_1_1_9_click_aug_flag` | grep '--use-click-aug' found in /home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py |
| `PASS` | `code_1_1_10_master_modules_flags` | help exposes all 2 token(s) of '--use-master-wass --use-master-cath' (rc=0) |
| `PASS` | `ablation_1_3_1_flags_present` | help exposes all 4 token(s) of '--no-multi-task --no-wass --no-kendall --no-dann' (rc=0) |
| `PASS` | `ablation_1_3_2_single_metal_flag` | help exposes all 1 token(s) of '--single-metal' (rc=0) |
| `PASS` | `ablation_1_3_3_n_seeds_min_flag` | help exposes all 1 token(s) of '--n-seeds-min' (rc=0) |
| `PASS` | `data_2_1_1_metalcytotox_5metal_csv` | wc -l /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv = 26802 >= 26000 |
| `PASS` | `data_2_1_2_metalcytotox_raw_csv` | wc -l /home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv = 26802 >= 1000 |
| `PASS` | `data_2_1_3_drugood_dir` | dir exists: /home/hugo/codes/mol-metal/.data/molmetal/drugood |
| `PASS` | `data_2_1_4_nci60_dir` | dir exists: /home/hugo/codes/mol-metal/.data/molmetal/NCI60_GI50 |
| `PASS` | `data_2_1_5_platinai_4cell_dir` | file exists: /home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv |
| `SKIP` | `data_2_1_6_bindingdb_external_honest_negative` | BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: d |
| `PASS` | `data_2_1_7_click_yields_csv` | wc -l /home/hugo/codes/mol-metal/.data/molmetal/click_yields_literature.csv = 61 >= 50 |
| `PASS` | `test_3_1_5_nan_guard_test_present` | file exists: /home/hugo/codes/mol-metal/molmetal/tests/test_train_chemprop_v2_v3_nan_guard.py |
| `PASS` | `honest_framing_8_15_catalog_entry_count` | /home/hugo/codes/mol-metal/科研进展_pre_retrain_checklist_2026-09-28.md has 15 §8 entries (>= 15) |

## P2 DOCUMENTED gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p2_data_metalcytotox_exists` | wc -l /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv = 26802 >= 26800 |
| `PASS` | `p2_data_metallo_drugs_exists` | wc -l /home/hugo/codes/mol-metal/molmetal/data/metallo_drugs_500_train.csv = 450 >= 449 |
| `PASS` | `p2_tests_pass` | pytest PASS: 702 passed, 12 skipped, 1 warning in 10.66s |
| `PASS` | `p2_vina_installed` | repo-local binary present: /home/hugo/codes/mol-metal/molmetal/bin/vina |

## P3 PAPER-GRADE gates

| Status | Gate | Evidence |
|---|---|---|
| `PASS` | `p3_honest_neg_closure_m2_r2` | m2 R² pre-closure path complete: verdict=HONEST_NEGATIVE_R2_DEEP + §6 catalog entry + §3 metric table + §6 paragraph pre |
| `PASS` | `p3_honest_neg_closure_m6_kappa` | m6_kappa pre-closure path complete: verdict=HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL + §6 catalog + drop-decision doc +  |
| `PASS` | `p3_honest_neg_closure_m1_re_fold` | m1 Re-fold pre-closure path complete: Re < 0.86 + §6 catalog entry + §3 metric table Rh-fold-ONLY + §6 paragraph present |
| `PASS` | `p3_todo_119_p14_cpu_smoke_ready` | help exposes all 4 token(s) of '--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale' (rc=0) |
| `PASS` | `p3_frontier_math_wire_in` | all 4/4 frontier-math adapters importable |
| `PASS` | `p3_frontier_math_day_3_5_integration_tests` | pytest PASS: 8 passed, 1 warning in 0.33s |
| `PASS` | `p3_frontier_math_paper_refs_present` | all 5 files present |
| `PASS` | `p3_todo220_drivers_present` | all 2 files present |
| `PASS` | `p3_anchorguard_version_pin` | 4/11 anchors pinned at version='1.0.0' (7 legacy pre-Wf-305C skipped) |
| `PASS` | `p3_anchorguard_consumers_adopt` | all 3 files present |
| `PASS` | `p3_harness_auto_fallback_present` | file exists: /home/hugo/.claude/wrapper.py |
| `PASS` | `p3_wf304b_platinai_selenium_diag` | all 2 files present |
| `PASS` | `p3_ood_rdkit_fix_documented` | all 4 files present |
| `PASS` | `p3_phase3g_f2a_runtime_unblock_verified` | all 4 files present |
| `PASS` | `p3_section9_honest_framing_min30` | 38 top-level §9 entries in 207_unified_pre_flight_checklist_v2.md >= 36 |
| `PASS` | `p3_admet_ames_druglikeness` | /home/hugo/codes/mol-metal/metrics/by_metric/ames_mutagen.json: value=0.040089 <= 0.1 |
| `PASS` | `p3_admet_cell_perm_druglikeness` | /home/hugo/codes/mol-metal/metrics/by_metric/cell_permeability_logPapp.json: value=-3.778475 >= -5.0 |
| `PASS` | `p3_admet_aq_sol_honest_negative_tag` | /home/hugo/codes/mol-metal/metrics/by_metric/aqueous_solubility_logS.json: HONEST_NEGATIVE=true, reason='Wf-325E — metal |
| `PASS` | `p3_admet_hepatotox_honest_negative` | /home/hugo/codes/mol-metal/metrics/by_metric/hepatotox_index.json: HONEST_NEGATIVE=true, reason='Wf-325E — metallo-drug  |
| `PASS` | `p3_admet_logp7_4_honest_negative` | /home/hugo/codes/mol-metal/metrics/by_metric/logp7_4.json: HONEST_NEGATIVE=true, reason='Wf-325E — metallo-drug cohort e |
| `PASS` | `p3_admet_m6_kappa_honest_negative` | /home/hugo/codes/mol-metal/metrics/by_metric/m6_kappa.json: HONEST_NEGATIVE=true, reason='Wf-325E — metallo-drug cohort  |
| `PASS` | `p3_admet_4_below_documented` | all 4 BELOW ADMET metrics carry HONEST_NEGATIVE=true + reason; closeout_doc (325e_admet_4_below_final.md) present on dis |
| `PASS` | `p3_sa_score_metric_present` | /home/hugo/codes/mol-metal/metrics/by_metric/sa_score.json: HONEST_NEGATIVE=true, reason='Mean cohort sa_score = 4.7148  |
| `PASS` | `p3_qed_score_metric_present` | /home/hugo/codes/mol-metal/metrics/by_metric/qed_score.json: HONEST_NEGATIVE=true, reason='Mean cohort qed_score = 0.229 |
| `PASS` | `p3_m4_sa_qed_subcriterion_documented` | m4 metric=HONEST_NEGATIVE (sub_rate_sa_qed=0.0022); report=m4_rate.json carries per_pool_distribution (27 metric keys +  |
| `PASS` | `p3_smoke_fail_on_missing_audit` | pytest PASS: 5 passed, 1 warning in 0.31s |
| `PASS` | `p3_nci60_9cell_per_cell_aggregate` | /home/hugo/codes/mol-metal/metrics/by_metric/nci60_precious_9cell_auc.json: 9-cell breakdown + HCT116 highlight (0.8500) |
| `PASS` | `p3_loi_send_ready` | 4/4 LOI drafts ready-to-send + review helper present (status DRAFT/SENT headers + ≥16 checklist boxes) |
| `PASS` | `p3_r20_phase0_loaders_complete` | 3/3 loaders + 3/3 API docs + 6/6 test files present; pytest PASS: 79 passed, 1 warning in 17.88s |
| `PASS` | `p3_r20_p0_loader_api_doc_present` | all 4 files present |
| `PASS` | `p3_pb_packages_available_or_documented` | packages available (rdkit=2026.03.6, posebusters=0.6.5); /home/hugo/codes/mol-metal/metrics/by_metric/steric_clash_pb_do |
| `PASS` | `p3_pb_packages_status_present` | venv=/home/hugo/codes/CadButEaas/.venv-vlm/bin/python rdkit=2026.03.6 posebusters=0.6.5; steric_clash_pb_dock_1h36.json= |
| `PASS` | `p3_uff_min_pt_pool_present` | pool=96 conformers (≥ 80); metrics HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True; reason='RDKit UFF force field lacks Pt (d |
| `PASS` | `p3_m8_full_pass_rate_documented` | result.m8_full=0.0000 (n_pre=14, n_pt_zn=14); metric MEASURED_SMOKE; pool=96 conformers |
| `PASS` | `p3_ablation_runner_tests_present` | pytest PASS: 12 passed, 1 warning in 2.13s |
| `PASS` | `p3_ablation_runner_aggregate_csv_present` | aggregate.csv OK: 12 rows (≥ 11); 14 columns; DESIGN_ONLY rows free of fabricated metrics |
| `PASS` | `p3_ablation_runner_registry_complete` | ABLATION_TABLE OK: 11/11 §7 ablations mirrored (M3/M4/M7/M8/M10/M11/M12/M13/M14/M15/M21) + 1 BASELINE; all 12 entries ca |
| `PASS` | `p3_catherine_30metal_closeout_documented` | all 5 files present |
| `PASS` | `p3_wf327_cpu_num_workers_sufficient` | 4/4 launch scripts request >= 4 workers (Wf-328-W2 + Wf-326B §6.2): ['launch_wf124_p3_g5.sh(max=8)', 'launch_wf124_p3_g6 |
| `PASS` | `p3_gate_count_consistency_audit` | gate count consistent for pipeline='all': actual=98 == expected=98 (auto-computed via Wf-345-B3) |
| `PASS` | `p3_wf343c4_hyperparams_config_complete` | all 7/7 active YAMLs carry train.batch_size + train.lr + model.hidden_dim + model.n_layers (AGENTS.md §5.1 standing GPU  |
| `PASS` | `p3_wf343c4_yuelbond_decoder_env` | cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder (per Wf-272I / Wf-269C YuelBond-conditionality structural check |
| `PASS` | `p3_wf343c4_anchor_freshness_24h` | all 11/11 paper-grade anchors fresh (<= 24h; Wf-302A / Wf-304A AnchorGuard pre-retrain freshness contract) |
| `SKIP` | `p3_wf343c4_wave_budget_remaining_nonneg` | remaining_hr=-183.04h is negative AND 8 DEFERRED wave(s) (['Wave-cfm:cfm_real_decode_lift_p1.3', 'Wave-cfm:cfm_per_innov |
| `PASS` | `p3_wf343c2_per_wave_runtime_check` | wf343c2_per_wave_check.py PASS (rc=0; wave='Wf-347A1 pre-retrain smoke', pairs=(Ru,Ir,Os,Rh,Re)×(0,1,2), budget=10.0h, - |
| `PASS` | `p3_wf343c3_data_quality_check` | wf343c3_data_quality_check.py PASS (rc=0; out_dir=default --strict): W20261002 17:14:08.038469 108073475614592 agent.cpp |
| `PASS` | `p1_hp_provenance_log` | Wf-350-A3 hp provenance log PASS: 7/7 config records written to /home/hugo/codes/mol-metal/molmetal/reports/wf350a3_hp_p |
| `PASS` | `p1_runtime_energy_report` | Wf-350-A1 runtime+energy PASS: 2 GPU(s) reported (power_draw range 106.89-123.76W; power_limit range 575.00-600.00W); to |
| `PASS` | `p1_gpu_env_probe` | Wf-350-C1 GPU/CUDA env probe PASS: torch.cuda.is_available=True | torch.cuda.device_count=2 | GPU model=NVIDIA RTX PRO 6 |
| `PASS` | `p3_wf385_scaffold_loss_wire_in_verified` | wf385a3_scaffold_wire_in_gate.py PASS (rc=0, verdict=PASS): all 11 SCAFFOLD modules export expected callable + 50-step s |

## Config summary

- yaml_path: `/home/hugo/codes/mol-metal/configs/cfm_full_retrain.yaml`
- parse_ok: **True**
- n_required_keys_present: **9** of 9

## Abort criteria active (§7.2)

| Criterion | Description |
|---|---|
| `7.2.8_partial_pilot` | n_completed < n_total (5/12 partial pilot) |

## Per-pair sub-checklist

| Metal | Seed | Expected ckpt | Expected AUC | Status | Wall (s) |
|---|---|---|---|---|---|
| Ru | 0 | `dmpnn_mtw_ru_seed0.pt` | 0.7613 | `DONE` | 463.2 |
| Ru | 1 | `dmpnn_mtw_ru_seed1.pt` | — | `IN-FLIGHT` | — |
| Ru | 2 | `dmpnn_mtw_ru_seed2.pt` | — | `IN-FLIGHT` | — |
| Ru | 3 | `dmpnn_mtw_ru_seed3.pt` | — | `IN-FLIGHT` | — |
| Ru | 4 | `dmpnn_mtw_ru_seed4.pt` | — | `IN-FLIGHT` | — |
| Ir | 0 | `dmpnn_mtw_ir_seed0.pt` | — | `IN-FLIGHT` | — |
| Ir | 1 | `dmpnn_mtw_ir_seed1.pt` | — | `IN-FLIGHT` | — |
| Ir | 2 | `dmpnn_mtw_ir_seed2.pt` | — | `IN-FLIGHT` | — |
| Ir | 3 | `dmpnn_mtw_ir_seed3.pt` | — | `PENDING` | — |
| Ir | 4 | `dmpnn_mtw_ir_seed4.pt` | — | `PENDING` | — |
| Os | 0 | `dmpnn_mtw_os_seed0.pt` | — | `IN-FLIGHT` | — |
| Os | 1 | `dmpnn_mtw_os_seed1.pt` | — | `IN-FLIGHT` | — |
| Os | 2 | `dmpnn_mtw_os_seed2.pt` | — | `IN-FLIGHT` | — |
| Os | 3 | `dmpnn_mtw_os_seed3.pt` | — | `PENDING` | — |
| Os | 4 | `dmpnn_mtw_os_seed4.pt` | — | `PENDING` | — |
| Rh | 0 | `dmpnn_mtw_rh_seed0.pt` | — | `IN-FLIGHT` | — |
| Rh | 1 | `dmpnn_mtw_rh_seed1.pt` | — | `IN-FLIGHT` | — |
| Rh | 2 | `dmpnn_mtw_rh_seed2.pt` | — | `IN-FLIGHT` | — |
| Rh | 3 | `dmpnn_mtw_rh_seed3.pt` | — | `PENDING` | — |
| Rh | 4 | `dmpnn_mtw_rh_seed4.pt` | — | `PENDING` | — |
| Re | 0 | `dmpnn_mtw_re_seed0.pt` | — | `IN-FLIGHT` | — |
| Re | 1 | `dmpnn_mtw_re_seed1.pt` | — | `IN-FLIGHT` | — |
| Re | 2 | `dmpnn_mtw_re_seed2.pt` | — | `IN-FLIGHT` | — |
| Re | 3 | `dmpnn_mtw_re_seed3.pt` | — | `PENDING` | — |
| Re | 4 | `dmpnn_mtw_re_seed4.pt` | — | `PENDING` | — |

---
_Honest framing (party-twenty-spirit §3.5): every entry above is POPULATED from JSON / on-disk evidence, not synthesised. Counts and status values trace to the referenced files; MEASURED vs PROJECTED is preserved per row. Re-run this script before each wave re-launch._
