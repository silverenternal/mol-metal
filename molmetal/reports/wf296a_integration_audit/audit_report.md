# Wf-296A — Pipeline Integration Audit

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Goal**: Find artifacts produced by Wf-256..295 that are NOT integrated into the main pipeline.

## 1. NEW scripts not in any wrapper

| Script | Path | Invoked from launchers? | Status |
|---|---|---|---|
| `wf278a_g4_aggregate.py` | `molmetal/scripts/wf278a_g4_aggregate.py` | NO (only self-refs in `molmetal/reports/wf278a_g4_aggregation_fix/fix_report.md` and synthesis reports) | **ORPHAN** |
| `wf286b_ecfp4_reeval.py` | `molmetal/scripts/wf286b_ecfp4_reeval.py` | NO (only self-refs in `molmetal/reports/wf286b_ecfp4_reeval/reeval_report.md`) | **ORPHAN** |
| `build_platinai_4cell_csv.py` | `molmetal/scripts/build_platinai_4cell_csv.py` | NO (referenced only in `molmetal/reports/wf278b_g6_platinai/loader_report.md` and `wf292d_platinai_bypass/strategy_report.md`) | **ORPHAN** |
| `drugood_ecfp4_scaffold_adapter.py` | `molmetal/scripts/drugood_ecfp4_scaffold_adapter.py` | NO (only via `wf286b_ecfp4_reeval.py` import + `molmetal/reports/wf275b_drugood_adapter/adapter_report.md`) | **ORPHAN** |
| `cfm_property_panel.py` | `molmetal/scripts/cfm_property_panel.py` | NO (only self-refs) | **ORPHAN** |
| `cfm_vina_dock_panel.py` | `molmetal/scripts/cfm_vina_dock_panel.py` | NO (only self-refs) | **ORPHAN** |
| `abort_recovery.py` | `molmetal/scripts/abort_recovery.py` | NO (only TODO/INDEX.md + TODO/completed/207_unified_pre_flight_checklist_v2.md) | **ORPHAN** |
| `paper_auto_update.py` | `molmetal/scripts/paper_auto_update.py` | NO (only mentioned as downstream consumer in `train_cfm_yuelbond_wasserstein.py` comments) | **ORPHAN** |
| `paper_dashboard.py` | `molmetal/scripts/paper_dashboard.py` | NO (CLI only — must be run manually) | **ORPHAN** (manual CLI) |
| `pre_flight_check.py` | `molmetal/scripts/pre_flight_check.py` | YES (Wf-124 P3 G4/G5/G6 launchers) | WIRED |
| `per_wave_checklist.py` | `molmetal/scripts/per_wave_checklist.py` | YES (Wf-124 P3 G4/G5/G6 launchers) | WIRED |
| `wave_budget_tracker.py` | `molmetal/scripts/wave_budget_tracker.py` | YES (Wf-124 P3 G4/G5/G6 launchers) | WIRED |
| `build_platinai_4cell.py` | `molmetal/scripts/build_platinai_4cell.py` | referenced by `train_platinai_4cell.py` | partial |
| `build_platinai_rusanov_5metal.py` | `molmetal/scripts/build_platinai_rusanov_5metal.py` | NO | **ORPHAN** |

**n_orphan_scripts = 10** (counting wf278a_g4_aggregate, wf286b_ecfp4_reeval, build_platinai_4cell_csv, drugood_ecfp4_scaffold_adapter, cfm_property_panel, cfm_vina_dock_panel, abort_recovery, paper_auto_update, paper_dashboard (manual CLI), build_platinai_rusanov_5metal)

## 2. NEW CLI flags not in default launch args

| Flag | Defined in | Invoked from launchers? |
|---|---|---|
| `--no-kendall` | `train_dmpnn_multitask_wasserstein.py` | **NO** (zero hits in any `*.sh` / `*.yaml` / `Makefile`) |
| `--no-dann` | `train_dmpnn_multitask_wasserstein.py` | **NO** |
| `--domain-id` | `train_dmpnn_multitask_wasserstein.py` | **NO** |
| `--lambda-grl` | `train_dmpnn_multitask_wasserstein.py` | **NO** |
| `--grl-alpha` | `train_dmpnn_multitask_wasserstein.py` | **NO** |
| `--num-domains` | `train_dmpnn_multitask_wasserstein.py` | **NO** |
| `--chemprop-min-epochs` | `train_chemprop_v2_v3.py` | **NO** |
| `--chemprop-lomo-init-scale` | `train_chemprop_v2_v3.py` | **NO** |
| `--chemprop-mode v3-fix` | `train_chemprop_v2_v3.py` | **NO** (only paper_dashboard.py comments + wf263a_verify_drugood.py mention) |
| `--chemprop-mode shared-encoder` | `train_chemprop_v2_v3.py` | **NO** (only self-refs in train script) |
| `--no-yuelbond` | `train_cfm_yuelbond_wasserstein.py` | **NO** |
| `--no-wasserstein` | `train_cfm_yuelbond_wasserstein.py` | **NO** |
| `--no-closure-bounded` | `train_cfm_yuelbond_wasserstein.py` | **NO** |
| `--use-tanimoto-ood-filter` | `train_cfm_yuelbond_wasserstein.py` | **NO** |
| `--ood-filter-threshold` | `train_cfm_yuelbond_wasserstein.py` | **NO** |
| `--ood-filter-train-pool` | `train_cfm_yuelbond_wasserstein.py` | **NO** |

**n_orphan_flags = 16** (Wf-283B DANN/Kendall/DANN/GRL wiring is COMPLETELY dead; Wf-269A shared-encoder IMPLEMENT only with no launcher; Wf-266 v3-fix IMPLEMENT only; Wf-269C Tanimoto OOD filter IMPLEMENT only)

## 3. NEW tests not in pre-flight

**Pre-flight gates only these 3 tests** (per `molmetal/scripts/pre_flight_check.py:744-746`):
- `test_wf245_ablation_flags.py`
- `test_wf146_resume.py`
- `test_train_chemprop_v2_v3_nan_guard.py`

**Orphan tests** (new tests NOT gated by pre-flight):

| Test file | Workflow origin |
|---|---|
| `test_abort_recovery.py` | Wf-262 |
| `test_cfm_decode_real.py` | Wf-188 / m14 |
| `test_cfm_panels.py` | Wf-264 |
| `test_drugood_ecfp4_scaffold_adapter.py` | Wf-275B |
| `test_wave_budget_tracker.py` | Wf-262 |
| `test_wf260_dataloader_fix.py` | Wf-260 |
| `test_wf261_scaffold_split.py` | Wf-261 |
| `test_wf263a_chemprop_v3_fix.py` | Wf-263A |
| `test_wf264a_drugood_post_fix_updater.py` | Wf-264A |
| `test_wf264_cfm_property_panel.py` | Wf-264 |
| `test_wf264_cfm_vina_dock_panel.py` | Wf-264 |
| `test_wf266_v3_lomo_fix.py` | Wf-266 |
| `test_wf268a_cfm_ablation_flags.py` | Wf-268A |
| `test_wf269a_chemprop_path_c.py` | Wf-269A |

**n_orphan_tests = 14** (gating coverage gap is HIGH — pre-flight misses Wf-260..269A test additions)

Total test files: 483 — pre-flight only gates 3 (0.62% coverage).

## 4. NEW catalog/dashboard not propagated

| Report | File | Referenced in `paper_dashboard.py`? |
|---|---|---|
| Wf-287A m14 anchor | `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` | **NO** (paper_dashboard.py only references `result.json`, NOT the superseding `m14_anchor.json`) — **ORPHAN** |
| Wf-253D m17 DrugOOD reeval | `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json` | **NO** (no `aggregate.json` exists, only `.json`; paper_dashboard has `claim_wf271b_m17_drugood_v3_reeval` but NOT wf253d) — **ORPHAN** |
| Wf-269C CFM real YuelBond | `molmetal/reports/wf269c_cfm_real/wf269c_report.json` | YES (line `wf269c_report.json::configs` + TODO/m14_lead_deployment_precondition.md) — WIRED |
| Wf-286B ECFP4 reeval | `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json` | YES (`aggregate.json::test_auc_mean` + `::per_seed[0]`) — WIRED |

**n_orphan_catalog = 2** (Wf-287A m14_anchor.json + Wf-253D m17_drugood_reeval.json)

## Summary

| Category | Count |
|---|---|
| Orphan scripts | 10 |
| Orphan flags | 16 |
| Orphan tests | 14 |
| Orphan catalog | 2 |
| **Total orphans** | **42** |

## Recommended actions (Wf-296B/C/D follow-up candidates)

1. **Wf-296B (scripts)**: Wire 10 orphan scripts into a unified `scripts/launch_all_reevals.sh` wrapper or document them as standalone CLI tools.
2. **Wf-296C (flags)**: Either delete unused flags (--no-kendall, --no-dann, --chemprop-mode v3-fix, --chemprop-mode shared-encoder, all CFM flags) OR add a `configs/wf296c_*.yaml` to drive them.
3. **Wf-296D (tests)**: Add 14 missing tests to pre_flight_check.py §5 P2 tests gate.
4. **Wf-296E (catalog)**: Wire `m14_anchor.json` and `wf253d_m17_drugood_reeval.json` into paper_dashboard.py catalog.
