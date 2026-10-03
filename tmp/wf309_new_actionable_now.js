export const meta = {
  name: 'wf-309-new-actionable-now',
  description: 'Wf-309: New actionable-now code tasks (NOT covered by Wf-308). (A) Wf-139 P7 / TODO-135 Phase 1 — m19 LOMO Rh cohort 500 generation (250 active + 250 inactive, RDKit CPU-only, 1-2 weeks, P0 per IMPROVEMENT_ENUM §1.2). (B) TODO-234 frontier math wire-in RE-DO — use correct paths per Wf-305B (4 adapters at molmetal/adapters/, NOT _reference/), 4 integration smoke tests. (C) R20-P1 VariancePUCT default + R20-P2 R² lift CPU smoke (TODO-92). (D) TODO-135 file restore + Phase 1 plan doc. (E) Synthesis.',
  phases: [
    { title: 'New actionable-now code tasks (parallel)', detail: '4 parallel: Rh cohort, frontier wire-in re-do, R20-P1/P2, TODO-135 restore' },
    { title: 'Synthesis', detail: 'Aggregate + remaining actionable + USER_OK-gated GPU queue' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('New actionable-now code tasks (parallel)')

const [wf309A, wf309B, wf309C, wf309D] = await parallel([

  () => agent(`Wf-309A: Wf-139 P7 / TODO-135 Phase 1 — m19 LOMO Rh cohort 500 generation.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §0 + §1.2: P0 task, CPU-only RDKit. 500 Rh cohort (250 active + 250 inactive with balanced labels) to recover Rh fold AUC from 0.136 regression (Wf-121 §7.2.6). File TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md is MISSING — needs restore.

TASK PART 1 — TODO-135 file restore:
1. Read TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.2 — Wf-139 P7 full spec
2. Read TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md §0 row 4 — TODO-135 context
3. Read molmetal/reports/wf121_m19_lomo/ — Rh fold 0.136 regression evidence
4. Restore TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (NEW file with §1.2 spec verbatim)

TASK PART 2 — Rh cohort 500 generation:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — 26,633 rows (Wf-261)
2. Identify Rh cohort baseline:
   grep -E ",Rh," metalcytotox_5metal_multitask_v4.csv | wc -l
3. Compute imbalance: how many active vs inactive Rh compounds currently
4. Write NEW script: molmetal/scripts/wf139_p7_rh_cohort_500.py
   - Generate 500 Rh cohort: 250 active + 250 inactive (balanced)
   - Label assignment: use existing labels if balanced, OR similarity-based label (per IMPROVEMENT_ENUM §1.2 soft caveat)
   - Scaffold-split verification: no scaffold leak (per Wf-139 P7 §0)
5. Save to .data/molmetal/wf139_p7_rh_cohort_500.csv
   Columns: smiles, metal=Rh, active_inactive, fold_id, scaffold_id, year, source
6. Write 12 tests: molmetal/tests/test_wf139_p7_rh_cohort_500.py
   - 5 tests: cohort structure (500 rows, balanced 250/250)
   - 3 tests: scaffold split (scaffold_id ≥ 50 unique, no train/test overlap)
   - 2 tests: label distribution sanity (no constant-inactive)
   - 2 tests: integration with downstream LOMO consumer
7. Run tests: ${PY} -m pytest molmetal/tests/test_wf139_p7_rh_cohort_500.py -v 2>&1 | tail -20
8. Document: TODO/completed/139_wf139_p7_rh_cohort_500_closeout.md

Honest framing per IMPROVEMENT_ENUM §1.2:
- 500 cohort + balanced labels + ≥12 tests PASS + no leak = HALT hard
- PARTIAL soft: similarity-based label caveat uncertain

OUTPUT (PLAIN TEXT): file_restore_status + cohort_path + n_active + n_inactive + n_balanced + n_tests_pass + scaffold_no_leak. Do NOT call StructuredOutput.`, {
    phase: 'New actionable-now code tasks (parallel)',
    label: 'wf309A',
  }),

  () => agent(`Wf-309B: TODO-234 frontier math wire-in RE-DO using correct paths per Wf-305B.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-305B honest-negative + Wf-306B verification: the 4 frontier math adapters actually live at TOP LEVEL of molmetal/adapters/ as standalone drop-in modules (NOT inside _reference/):
  - molmetal/adapters/multi_ph_integration.py (W2.3 ship)
  - molmetal/adapters/se3_lipman_integration.py (Integration 1)
  - molmetal/adapters/se3_sb_integration.py (W2.1)
  - molmetal/adapters/transport_relaxation_integration.py (W2.2)

Per TODO-234 §4 + Wf-305B: wire-in INTO flow_matching_lipman/__init__.py is FORBIDDEN by R17 invariant. The right approach: write INTEGRATION WRAPPERS that call the standalone adapters from a NEW pipeline module (NOT modifying flow_matching_lipman/).

TASK:
1. Read molmetal/adapters/multi_ph_integration.py — public API
2. Read molmetal/adapters/se3_lipman_integration.py — public API
3. Read molmetal/adapters/se3_sb_integration.py — public API
4. Read molmetal/adapters/transport_relaxation_integration.py — public API
5. Write NEW integration wrappers: molmetal/adapters/frontier_math_integration.py
   - Class FrontierMathIntegration
   - __init__(self, config) → stores config
   - get_multi_ph_features(pocket_data) → calls multi_ph_integration.MultiPHPocketFeatures
   - get_se3_lipman_solver() → calls se3_lipman_integration.SE3LipmanSolver
   - get_se3_sb_decoder() → calls se3_sb_integration.SE3SBDecoder
   - get_transport_relaxation() → calls transport_relaxation_integration.RelaxedEndpointTerm
6. Write 4 integration smoke tests: molmetal/tests/test_wf309b_frontier_math_integration.py
   - test_multi_ph_forward (1 dummy forward pass)
   - test_se3_lipman_solver_init
   - test_se3_sb_decoder_init
   - test_transport_relaxation_init
7. Run: ${PY} -m pytest molmetal/tests/test_wf309b_frontier_math_integration.py -v 2>&1 | tail -20
8. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- This is INTEGRATION wrapper work (NEW file), NOT modification of flow_matching_lipman/__init__.py
- 4 adapters are STANDALONE; wrappers just compose them
- 3 research-only modules (bass_existence / sb_bass_martingale / monge_ampere_regularity) honestly research-only by design — DO NOT wire-in
- Per Wf-234 §1 G1..G3 + TODO-234 §1: integration plan audit DONE; this is execution phase

OUTPUT (PLAIN TEXT): n_adapters_composed + n_tests_pass + wrapper_path + preflight_status + n_research_only_excluded. Do NOT call StructuredOutput.`, {
    phase: 'New actionable-now code tasks (parallel)',
    label: 'wf309B',
  }),

  () => agent(`Wf-309C: R20-P1 VariancePUCT default + R20-P2 R² lift CPU smoke (TODO-92 + TODO-93).

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md: R20-P1 = VariancePUCT default (replace Dirichlet PUCT as default PUCT selector). R20-P2 = Hilbert ASTSR R² lift ship surface. Both have CPU-only first phases per mol-metal-variance-puct-default + mol-metal-hilbert-astsr-r2 skills.

TASK PART 1 — R20-P1 VariancePUCT default:
1. Read molmetal/molmetal_lam/search_alg/proof_search.py — current default PUCT selector (Dirichlet)
2. Read .claude/skills/mol-metal-variance-puct-default/SKILL.md — full R20-P1 spec
3. Implement: replace Dirichlet default with VariancePUCT
   - Add VariancePUCT class in molmetal/molmetal_lam/search_alg/proof_search.py
   - Wire as default in main proof_search flow
   - Keep Dirichlet PUCT as opt-in fallback (--puct-impl dirichlet)
4. Write 8 tests: molmetal/tests/test_wf309c_variance_puct.py
   - test_variance_puct_default_selected
   - test_variance_puct_forward
   - test_variance_puct_vs_dirichlet_consistency
   - test_puct_impl_cli_flag
   - 4 more boundary tests
5. Run: ${PY} -m pytest molmetal/tests/test_wf309c_variance_puct.py -v 2>&1 | tail -20
6. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 2 — R20-P2 R² lift CPU smoke:
1. Read .claude/skills/mol-metal-hilbert-astsr-r2/SKILL.md — full R20-P2 spec
2. Read molmetal/scripts/hilbert_astsr_e2e.py (if exists) or related file
4. Implement CPU smoke: F5_RDKIT_2D feature extractor + Hilbert primal-dual head
5. Write 4 tests: molmetal/tests/test_wf309c_r2_lift_cpu_smoke.py
   - test_f5_rdkit_2d_extract
   - test_hilbert_primal_dual_init
   - test_r2_lift_forward
   - test_r2_score_shape
6. Run: ${PY} -m pytest molmetal/tests/test_wf309c_r2_lift_cpu_smoke.py -v 2>&1 | tail -20

Honest framing:
- Both are CPU-only first phases
- R20-P1 full production = GPU portion (5-7 day ultracode window per Wf-304C), NOT in scope
- R20-P2 full R² lift on PlatinAI 4-cell = GPU + user OK, NOT in scope
- VariancePUCT CPU portion = proof_search entry point default change
- R² CPU smoke = feature extractor forward pass shape verification

OUTPUT (PLAIN TEXT): n_puct_tests_pass + puct_default_replaced + n_r2_tests_pass + r2_smoke_status + gpu_production_deferred. Do NOT call StructuredOutput.`, {
    phase: 'New actionable-now code tasks (parallel)',
    label: 'wf309C',
  }),

  () => agent(`Wf-309D: TODO-135 file restore + Phase 1 plan doc (supplementary to Wf-309A).

USE Bash + Read + Write + Edit.

CONTEXT: TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md is MISSING (per Wf-309A discovery). Wf-309A will restore it. This Wf-309D does additional restore work + Wf-135 Phase 1 detailed plan.

TASK PART 1 — TODO-135 file restore (parallel work to Wf-309A):
1. Read TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.2 — full Wf-139 P7 + TODO-135 spec
2. Read TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md §0 row 4 — context
3. Write TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (parallel file restore if Wf-309A hasn't done it; verify it doesn't exist first)
4. Sections:
   - §0: TL;DR (1 paragraph)
   - §1: Problem statement (Rh fold AUC 0.136 regression)
   - §2: Phase 1 plan (Rh cohort 500 = 250 active + 250 inactive; RDKit CPU; scaffold no leak)
   - §3: Phase 2 plan (5-metal × 5-seed × 50-epoch re-bench; GPU-1; USER_OK gate; ~2 weeks)
   - §4: Acceptance gates (per IMPROVEMENT_ENUM §1.2)
   - §5: Honest framing
   - §6: Cross-references

TASK PART 2 — Wf-135 Phase 1 detailed plan doc:
1. Read IMPROVEMENT_ENUM §1.4 — Wf-135 Phase 2 spec (5-metal × 5-seed × 50-epoch)
2. Read molmetal/reports/wf121_m19_lomo/ — Rh fold 0.136 baseline
3. Read molmetal/reports/wf39_p1_p4_dock_redesign/ (if exists) — Phase 2 design context
4. Write NEW plan doc: TODO/completed/135_wf135_phase1_plan.md
   - Phase 1 = cohort generation (already addressed by Wf-309A)
   - Phase 2 detailed plan (GPU portion)
   - USER_OK gate per mol-metal-gpu-execution §0
   - Wall budget 2 weeks GPU
   - Acceptance: Rh fold AUC ≥ 0.30 (recover from 0.136), Overall LOMO mean ≥ 0.60
5. Move pending/135_*.md → completed/ if appropriate

Honest framing:
- This is documentation/restore work, NO code execution
- Wf-135 Phase 2 needs USER_OK + 2 weeks GPU; deferred
- Wf-135 Phase 1 will be done by Wf-309A in parallel

OUTPUT (PLAIN TEXT): file_restore_status + plan_doc_path + phase_2_user_ok_status + phase_2_wall. Do NOT call StructuredOutput.`, {
    phase: 'New actionable-now code tasks (parallel)',
    label: 'wf309D',
  }),
])

phase('Synthesis')

const wf309E = await agent(`Wf-309E: Aggregate Wf-309 new actionable-now code tasks outcomes.

USE Read only.

UPSTREAM RESULTS:
- Wf-309A: Wf-139 P7 Rh cohort 500
- Wf-309B: TODO-234 frontier math wire-in re-do
- Wf-309C: R20-P1 VariancePUCT + R20-P2 R² CPU smoke
- Wf-309D: TODO-135 file restore + Phase 1 plan

TASK:
1. Read all 4 outputs
2. Aggregate: completed + failed + partial
3. Identify remaining actionable + USER_OK-gated GPU queue
4. Output to molmetal/reports/wf309_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + actionable_now_closed + user_ok_gpu_queue + remaining_blockers (8-12 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf309E',
})

return {
  ultracode: 'wf-309-new-actionable-now',
  wf309a: wf309A,
  wf309b: wf309B,
  wf309c: wf309C,
  wf309d: wf309D,
  wf309e: wf309E,
}
