export const meta = {
  name: 'wf-310-ultracode-queue-comprehensive',
  description: 'Wf-310: COMPREHENSIVE ultracode queue prep — ALL previously USER-OK-gated tasks wrapped + actionable-now CPU fixes. (B) TODO-182 G6 final wrapper (7.5 hr). (C) TODO-179 + TODO-220 Driver 1+2 + Wf-219 §4.1 Ru short wrappers. (D) Wf-301A drugood DANN full + Wf-301B Vina 100mol full wrappers. (E) Wf-135 Phase 2 + Wf-219 §4.2 production backbone wrappers. (F) TODO-234 frontier math full wire-in 5-7 day plan + CPU portion. (G) 4 LOI send-ready package. (H) Actionable-now: G5 aggregation fix + missing 2 pairs + TODO-178b smoke audit + ADMET integration + TODO-220 --execute CPU + Wf-219 temporal CPU + Wf-139 P7 Rh cohort 500 + TODO-234 wire-in RE-DO + R20-P1 VariancePUCT + R20-P2 R² smoke + TODO-135 restore. (I) Synthesis + ultracode queue ready-to-launch dashboard.',
  phases: [
    { title: 'Comprehensive ultracode queue (parallel)', detail: '8 parallel: G5 fix, G6 wrapper, TODO-179/220/219 wrappers, drugood+vina wrappers, LOMO+production wrappers, frontier plan, LOI pkg, actionable-now bundle' },
    { title: 'Synthesis', detail: 'Final ready-to-launch dashboard + remaining blockers' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Comprehensive ultracode queue (parallel)')

const [wf310A, wf310B, wf310C, wf310D, wf310E, wf310F, wf310G, wf310H] = await parallel([

  () => agent(`Wf-310A: G5 aggregation fix + 2 missing pairs补跑 + Wf-310 integration template.

USE Bash + Read + Edit.

CONTEXT: G5 (TODO-181) 50-epoch full sweep COMPLETED 2026-09-29 03:11:13 CST — 13/15 pairs MEASURED. 2 missing (Ir seed=0 + Ru seed=2). training_summary.json is stub (total_pairs=0, runs=[]). Master log "aggregated 0 pairs (expected 15)". Per Wf-304A Wf-296D infra ready.

TASK:
1. Read molmetal/scripts/wf278a_g4_aggregate.py — current G4 aggregator (Wf-298B multi-root patch)
2. Read results/wf124_p3_50epoch_full_sweep_g5/ — 13 per-seed JSONs
3. Identify why aggregation failed (likely JSON schema mismatch with v2 G4 schema)
4. Write FIX: aggregate_g5_sweep.py at molmetal/scripts/wf278a_g5_aggregate.py
   - Read 13 per-seed JSONs
   - Aggregate per-metal mean ± std
   - Emit training_summary.json with real total_pairs=15 (or 13) + per-metal breakdown
   - Cross-check with G4 cascade (Ru seed=0 from G4)
5. Retry 2 missing pairs (Ir seed=0 + Ru seed=2):
   ${PY} molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal Ir --seed 0 --epochs 50 --hidden 300 --depth 3 --device cuda:1 --out-dir results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/ir --no-enforce-budget 2>&1 | tail -30
   ${PY} molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal Ru --seed 2 --epochs 50 --hidden 300 --depth 3 --device cuda:1 --out-dir results/wf124_p3_50epoch_full_sweep_g5/Ru/seed2/ru --no-enforce-budget 2>&1 | tail -30
   Note: these need GPU; if cuda:1 busy, document and skip (USER_OK gate)
6. Run G5 aggregator: ${PY} molmetal/scripts/wf278a_g5_aggregate.py --input-dir results/wf124_p3_50epoch_full_sweep_g5 --output results/wf124_p3_50epoch_full_sweep_g5/training_summary.json
7. Update metrics/by_metric/wf181_g5_50epoch_full_sweep.json
8. Update paper_dashboard.py with G5 50ep claim cards (vs G4 30ep comparison)
9. Document: TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md

Honest framing per party-twenty-spirit §3.5:
- G5 13/15 MEASURED today; 2 missing if not retried
- 30ep (G4) vs 50ep (G5) comparison is honest if both are real
- Ru 50ep 0.7860 vs 30ep 0.7527 = Δ +0.033 lift (real MEASURED)

OUTPUT (PLAIN TEXT): g5_aggregator_path + n_pairs_aggregated + per_metal_mean_std + Ru_target_short_retried + g5_repair_status. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310A',
  }),

  () => agent(`Wf-310B: TODO-182 G6 final USER_OK-gated launch wrapper + 60-tuple ready manifest + per-wave checklist.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-296C: G6 launcher wired with 60/60 CONFIRMED tuples (4 configs × 5 metals × 3 seeds). Per Wf-306D: USER_OK gate added. Per Wf-295D §1.3: 8 sub-waves a..h. Per Wf-300C: pattern for USER_OK gate (kill_then_adapt standing directive).

TASK:
1. Read scripts/launch_wf124_p3_g6.sh — current G6 launcher (Wf-296C + Wf-306D)
2. Read molmetal/configs/wf124_p3_per_innovation_ablation_g6.yaml — 60-tuple config
3. Verify USER_OK gate enforced (per Wf-306D):
   bash -n scripts/launch_wf124_p3_g6.sh && echo SYNTAX_OK
4. Write final wrapper: scripts/launch_wf310b_g6_final.sh
   - USER_OK gate enforced
   - pre_flight_check.py first (32/32 v3 PASS required)
   - wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required (per Wf-296C)
   - per_wave_checklist.py emission per sub-wave (per Wf-262 §10.1)
   - 8 sub-waves a..h per Wf-295D §1.3
   - abort_recovery.py on unexpected abort (per TODO-207 §7.3)
   - Total wall: ~7.5 hr GPU-1 (cuda:1)
5. Generate 60-tuple ready manifest: molmetal/reports/wf310b_g6_manifest/60_tuples.json
   - For each tuple: config, metal, seed, expected wall, expected test_auc (PROJECTED), CLI args
   - Use Wf-296C CONFIRMED list verbatim
6. Write 8 tests: molmolmetal/tests/test_wf310b_g6_launcher.py
   - test_user_ok_gate_blocks
   - test_user_ok_unblocks
   - test_pre_flight_called_first
   - test_per_wave_checklist_emitted
   - test_abort_recovery_wired
   - test_60_tuples_manifest_complete
   - test_yaml_config_parses
   - test_dry_run_idempotent
7. Verify dry-run: bash scripts/launch_wf310b_g6_final.sh --dry-run 2>&1 | tail -20
8. Document: molmetal/reports/wf310b_g6_launch_wrapper/launch_plan.md

Honest framing:
- 7.5 hr GPU wall; USER_OK gate required per mol-metal-gpu-execution §0
- 60-tuple manifest is ready (Wf-296C)
- pre_flight 32/32 v3 required (Wf-306C)

OUTPUT (PLAIN TEXT): wrapper_path + n_tuples_in_manifest + n_tests_pass + syntax_check + dry_run_verdict + user_ok_gate + total_wall. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310B',
  }),

  () => agent(`Wf-310C: TODO-179 + TODO-220 Driver 1+2 + Wf-219 §4.1 Ru short USER_OK-gated wrappers + --execute CPU.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-303C Task 3 + Wf-306B + Wf-308C: TODO-220 stub drivers (wf219_temporal_split_driver.py + nci60_9cell_chemprop_driver.py) shipped with --dry-run mode; --execute CPU code ready. Per Wf-308B: TODO-179 GPU wrapper. Per Wf-306B: Wf-219 Ru wrapper (10 min). Need final USER_OK-gated wrappers + actual --execute CPU smoke.

TASK PART 1 — TODO-220 Driver 1 (temporal-split) actual --execute CPU smoke:
1. Read molmetal/scripts/wf219_temporal_split_driver.py — current --execute path (Wf-308C)
2. Verify CSV: ls -la molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
3. If CSV present, run actual --execute smoke:
   ${PY} molmetal/scripts/wf219_temporal_split_driver.py --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --train-cutoff 2023-01-01 --test-cutoff 2024-01-01 --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf310c_temporal --execute 2>&1 | tail -20
4. Verify: JSON emitted with n_seeds_completed + per_seed_test_auc

TASK PART 2 — TODO-220 Driver 2 (NCI-60 9-cell) actual --execute CPU smoke:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — current --execute path (Wf-308C)
2. Verify CSV: ls -la molmetal/.data/molmetal/nci60_9cell.csv
3. If CSV present, run actual --execute smoke
4. If MISSING: document, don't fail

TASK PART 3 — TODO-179 W₂ PlatinAI 4-cell alignment GPU wrapper:
1. Read TODO/pending/179_wf123_p3_w2_pic50_alignment.md — full spec
2. Write NEW GPU wrapper: scripts/launch_wf310c_t179_w2_gpu.sh
   - USER_OK gate enforced
   - Wall: ~4-5 hr GPU-1 (cuda:1)
3. Document: molmetal/reports/wf310c_t179_w2_wrapper/launch_plan.md

TASK PART 4 — TODO-220 GPU wrapper final:
1. Write NEW GPU wrapper: scripts/launch_wf310c_todo220_gpu.sh
   - USER_OK gate enforced
   - 2 phases: Driver 1 (2.9 hr) + Driver 2 (1.0 hr) = 3.9 hr sequential
   - pre_flight_check.py first
2. Document: molmetal/reports/wf310c_todo220_launch/launch_plan.md

TASK PART 5 — Wf-219 §4.1 Ru short-epoch wrapper final:
1. Read molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh — existing wrapper (Wf-306B)
2. Verify USER_OK gate enforced
3. Verify dry-run: USER_OK=1 bash molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh --dry-run
4. Final ready for user OK; 13.5 min GPU-1 (Ru × 3 seeds × 30 epochs)

Honest framing:
- All GPU wrappers require USER_OK per kill_then_adapt
- TODO-220 --execute CPU smoke is actionable-now if CSVs present
- TODO-179 GPU wall 4-5 hr; TODO-220 GPU wall 3.9 hr; Wf-219 §4.1 GPU wall 13.5 min
- Total GPU queue: ~16 hr

OUTPUT (PLAIN TEXT): todo220_temporal_execute_status + todo220_nci60_execute_status + t179_wrapper_path + todo220_gpu_wrapper_path + wf219_ru_wrapper_path + per_task_user_ok + total_gpu_queue_wall. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310C',
  }),

  () => agent(`Wf-310D: Wf-301A drugood DANN full + Wf-301B Vina 100mol full USER_OK wrappers + aggregation template.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-301A: drugood DANN smoke 0.8024 on 200-row tiny CSV (1 epoch). Per Wf-301B: Vina pt-strip 5/5 docked on 1h36 pocket. Full runs:
- Wf-301A drugood: 95k-row × 5-seed × 50-epoch chemprop v3 + DANN scaffold-bin GRL = ~3-4 hr GPU-1
- Wf-301B Vina: 100 mol × 4 pocket panel = ~1-2 hr GPU-1
Per Wf-306A: launch_wf306a_drugood_vina_gpu.sh exists with USER_OK gate. Need final aggregation template.

TASK PART 1 — drugood DANN full GPU wrapper final:
1. Read molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh — existing wrapper
2. Verify USER_OK gate, pre_flight_check.py, wave_budget_tracker.py integration
3. Write NEW aggregation template: molmetal/scripts/wf310d_drugood_dann_full_aggregate.py
   - Aggregate per-seed ood_test_auc across 5 seeds
   - Compare to Wf-185 baseline 0.6707 + Wf-301A smoke 0.8024 + field-avg 0.55-0.65
   - Emit claim cards
4. Document: molmetal/reports/wf310d_drugood_full/aggregation_plan.md

TASK PART 2 — Vina full 100-mol × 4-pocket GPU wrapper final:
1. Read launch_wf306a_drugood_vina_gpu.sh --phase B section
2. Write NEW aggregation template: molmetal/scripts/wf310d_vina_full_aggregate.py
   - Aggregate per-pocket mean ± std kcal/mol
   - Compare to field-avg -7.5 to -9.5 kcal/mol
3. Document: molmetal/reports/wf310d_vina_full/aggregation_plan.md

TASK PART 3 — combined launch wrapper ready:
- Verify: bash -n scripts/launch_wf306a_drugood_vina_gpu.sh && echo OK
- Verify dry-run: WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase all --dry-run

Honest framing:
- DrugOOD full = 3-4 hr GPU; Vina full = 1-2 hr GPU; combined ~5 hr
- Per Wf-301A honest-negative: Wf-185 Δ is across subsets; Wf-301A result is on different CSV
- Wf-301B honest-negative: top-1 -10.299 exceeds -9.5; mean -5.564 below field-avg band

OUTPUT (PLAIN TEXT): drugood_dann_wrapper_verified + vina_wrapper_verified + drugood_aggregation_path + vina_aggregation_path + total_gpu_wall + n_aggregation_templates. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310D',
  }),

  () => agent(`Wf-310E: Wf-135 Phase 2 + Wf-219 §4.2 production backbone USER_OK wrappers + LOMO test summary.

USE Bash + Read + Write + Edit.

CONTEXT: Per IMPROVEMENT_ENUM §1.4: Wf-135 Phase 2 = m19 LOMO 5-metal × 5-seed × 50-epoch re-bench (2 weeks GPU, Rh fold AUC ≥ 0.30 target). Per Wf-307A: Wf-309A Rh cohort 500 (Phase 1 prerequisite). Per Wf-219 §4.2: production backbone (hidden=300/depth=3, ~80 min GPU-1) — paper BLOCKED per TODO-84.

TASK PART 1 — Wf-135 Phase 2 GPU wrapper:
1. Read TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (or Wf-309D output if just restored)
2. Write NEW GPU wrapper: scripts/launch_wf310e_wf135_phase2_gpu.sh
   - USER_OK gate
   - Phase 1 prerequisite: Wf-309A Rh cohort 500 complete
   - Phase 2: 5-metal × 5-seed × 50-epoch = 1250 runs
   - Wall: ~2 weeks GPU-1
   - Acceptance: Rh fold AUC ≥ 0.30, Overall LOMO mean ≥ 0.60
3. Document: molmetal/reports/wf310e_wf135_phase2/launch_plan.md

TASK PART 2 — Wf-219 §4.2 production backbone GPU wrapper:
1. Read TODO/pending/180_wf124_p3_remaining_metals_seeds.md — context
2. Write NEW GPU wrapper: scripts/launch_wf310e_wf219_production_gpu.sh
   - USER_OK gate
   - **Paper BLOCKED per TODO-84**: only launch if user explicitly OKs paper work
   - Wall: ~80 min GPU-1 (Ru hidden=300/depth=3)
3. Document: molmetal/reports/wf310e_wf219_production/launch_plan.md (with paper BLOCKED notice)

TASK PART 3 — LOMO test summary:
1. Read molmetal/reports/wf121_m19_lomo/ — current Rh fold 0.136 regression
2. Summarize what's needed to recover
3. Verify Phase 1 (Wf-309A Rh cohort 500) is prerequisite
4. Verify Phase 2 GPU wrapper ready

Honest framing:
- Wf-135 Phase 2: 2 weeks GPU; Wf-219 §4.2: 80 min GPU; combined ~2 weeks
- Wf-219 §4.2 paper BLOCKED — requires TODO-84 lift
- Rh fold 0.136 regression is the binding issue (recover to ≥ 0.30)
- All GPU wrappers require USER_OK per kill_then_adapt

OUTPUT (PLAIN TEXT): wf135_phase2_wrapper_path + wf219_production_wrapper_path + paper_block_status + lomo_recovery_target + total_gpu_wall. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310E',
  }),

  () => agent(`Wf-310F: TODO-234 frontier math full wire-in 5-7 day plan + CPU portion execution.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-305B + Wf-306B + Wf-309B: TODO-234 frontier math wire-in needs to use CORRECT paths (molmetal/adapters/, NOT _reference/). 4 standalone adapters exist + 3 research-only (excluded). Per Wf-234 §4 + §5: 5-7 day CPU+GPU ultracode window. CPU portion = actionable-now.

TASK PART 1 — TODO-234 plan breakdown:
1. Read TODO/pending/234_wf234_frontier_math_audit_integration.md — full audit
2. Read molmetal/adapters/{multi_ph,se3_lipman,se3_sb,transport_relaxation}_integration.py — APIs
3. Write NEW day-by-day plan: TODO/completed/234_wf234_frontier_math_cpu_plan.md
   - Day 1-2: FrontierMathIntegration class (NEW molmetal/adapters/frontier_math_integration.py)
   - Day 3-4: CFM training loop integration test (CPU)
   - Day 5-6: Multi-task backbone integration test (CPU)
   - Day 7: GPU production wire-in (USER_OK gate)

TASK PART 2 — FrontierMathIntegration wrapper (Day 1-2 deliverable, actionable-now):
1. Write NEW file: molmetal/adapters/frontier_math_integration.py
   - Class FrontierMathIntegration (per Wf-309B)
   - 4 methods: get_multi_ph_features, get_se3_lipman_solver, get_se3_sb_decoder, get_transport_relaxation
2. Write 4 smoke tests: molmetal/tests/test_wf310f_frontier_math_integration.py
   - test_multi_ph_forward
   - test_se3_lipman_solver_init
   - test_se3_sb_decoder_init
   - test_transport_relaxation_init
3. Run: ${PY} -m pytest molmetal/tests/test_wf310f_frontier_math_integration.py -v 2>&1 | tail -20
4. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 3 — GPU production wire-in wrapper (Day 7, USER_OK gate):
1. Write NEW GPU wrapper: scripts/launch_wf310f_frontier_math_gpu.sh
   - USER_OK gate
   - Wall: ~3-5 hr GPU-1
2. Document: molmetal/reports/wf310f_frontier_math_gpu/launch_plan.md

Honest framing:
- 5-7 day ultracode window; Day 1-2 CPU actionable-now; Day 3-7 deferred
- 3 research-only modules (bass_existence / sb_bass_martingale / monge_ampere_regularity) honestly excluded
- FrontierMathIntegration composes standalone adapters (does NOT modify flow_matching_lipman/__init__.py per R17)

OUTPUT (PLAIN TEXT): plan_doc_path + wrapper_path + n_smoke_tests_pass + preflight_status + cpu_portion_status + gpu_portion_deferred + n_research_only_excluded. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310F',
  }),

  () => agent(`Wf-310G: 4 LOI send-ready package + cold-email review helper.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-305A + Wf-308D: 4 LOI drafts (ORNL / PKU / Xiehe / RUSANOV) all DRAFT — not sent. Hugo coordinates sending per TODO-84. Need final review package + cold-email template.

TASK PART 1 — 4 LOI consolidation:
1. Read all 4 LOI drafts:
   - TODO/pending/wetlab_loi_ornl_protein_metal.md
   - TODO/pending/wetlab_loi_pku_cisplatin_ic50.md
   - TODO/pending/wetlab_loi_xiehe_nci60.md
   - TODO/pending/wetlab_loi_rusanov_2025_platinai.md
2. Verify each LOI has:
   - Subject line
   - PI contact info (or TODO)
   - 16 sections parallel to ORNL template
   - Asks + Offers + Timeline + Risk
3. Write NEW consolidation: TODO/pending/wetlab_loi_send_ready_package.md
   - 4-row summary table (status, target PI, email, ask, offer)
   - Per-LOI next-action checklist for Hugo
   - Cold-email sending protocol (subject + body + CC + attach)
4. Write NEW cold-email template: TODO/pending/wetlab_cold_email_template.md
   - Subject line variants
   - Body template (3 paragraphs: purpose / what we need / what we offer)
   - Attachments list (project summary PDF, paper draft, 1-pager)

TASK PART 2 — Send-ready checklist:
1. For each LOI, verify:
   - Subject: clear + specific
   - Body: 200-300 words (concise)
   - Attachments: 1-pager + references
   - Signature: Hugo Lee + title + ORCID
2. Mark each LOI ready-to-send if checklist passes
3. Document blockers honestly (e.g. "PI email missing" → Hugo needs to find)

Honest framing per TODO-84:
- LOI send is USER's responsibility
- 0 sent today
- 4 LOI ready-to-send subject to Hugo review + actual send
- 4 wet-lab cooperation pathways blocked on LOI send (Wf-58 P6.2 wetlab IC50 etc.)

OUTPUT (PLAIN TEXT): n_lois_audited + n_lois_ready_to_send + n_lois_blocked_on_pi_contact + send_protocol_path + per_loi_status. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310G',
  }),

  () => agent(`Wf-310H: Actionable-now CPU bundle — TODO-178b + ADMET + TODO-220 --execute CPU + Wf-219 temporal CPU + Wf-139 P7 Rh cohort 500 + TODO-234 wire-in RE-DO + R20-P1 + R20-P2 + TODO-135 restore.

USE Bash + Read + Write + Edit.

CONTEXT: Multiple actionable-now CPU tasks identified by Wf-303A/B + Wf-307A/B/C/D + Wf-309A/B/C/D + Wf-308A/B/C/D. Need ALL done in single batch (no user OK, no paper, no wet-lab blocking).

TASK PART 1 — TODO-178b smoke-test FAIL-on-missing global audit:
1. Write NEW meta-test: molmetal/tests/test_smoke_fail_on_missing_audit.py
2. Walk all smoke tests in molmetal/tests/, distinguish legit vs illegitimate skips
3. Fix illegitimate skips (replace pytest.skip with pytest.fail + AssertionError)
4. Add pre_flight v3 gate
5. Run: ${PY} -m pytest molmetal/tests/test_smoke_fail_on_missing_audit.py -v 2>&1 | tail -30
6. Document: TODO/completed/178b_smoke_fail_on_missing_hardening.md

TASK PART 2 — ADMET pass-through integration (Wf-303C ships admet_pass_through.py):
1. Read molmetal/scripts/admet_pass_through.py
2. Add 3 pre_flight gates: ames_mutagen ≤ 0.10 / logPapp ≥ -5.0 / logS HONEST_NEGATIVE
3. Add 3 paper_dashboard claim cards
4. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Verify dashboard: ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v310h.html 2>&1 | tail -10

TASK PART 3 — TODO-220 --execute CPU implementation:
1. Implement --execute path in wf219_temporal_split_driver.py + nci60_9cell_chemprop_driver.py
2. Run CPU smoke if CSVs present
3. Document

TASK PART 4 — Wf-219 §4.3 temporal-split actual CPU smoke:
1. Run temporal-split with --split temporal flag (per Wf-303C Task 3)
2. Verify JSON emitted

TASK PART 5 — Wf-139 P7 Rh cohort 500 generation:
1. Restore TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md
2. Write molmetal/scripts/wf139_p7_rh_cohort_500.py
3. Generate 500 cohort (250 active + 250 inactive)
4. Verify scaffold no leak
5. Write 12 tests
6. Document: TODO/completed/139_wf139_p7_rh_cohort_500_closeout.md

TASK PART 6 — TODO-234 frontier math wire-in RE-DO (per Wf-309B):
1. NEW molmetal/adapters/frontier_math_integration.py (4 methods)
2. 4 integration smoke tests
3. Verify pre_flight preserved

TASK PART 7 — R20-P1 VariancePUCT default + R20-P2 R² CPU smoke:
1. Implement VariancePUCT class in proof_search.py
2. Wire as default (with --puct-impl dirichlet opt-in)
3. 8 VariancePUCT tests
4. R² lift 4 smoke tests (F5_RDKIT_2D + Hilbert primal-dual)

TASK PART 8 — TODO-135 file restore:
1. Write TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (if not done by Task 5)
2. Sections §0-§6

Honest framing:
- All 8 tasks actionable-now (CPU-only)
- 1-2 weeks wall total for full bundle
- Each task has 12 tests acceptance per IMPROVEMENT_ENUM §1.1
- No GPU, no user OK, no paper blocking

OUTPUT (PLAIN TEXT): n_tasks_completed + n_tests_total + preflight_status + per_task_summary + per_task_status. Do NOT call StructuredOutput.`, {
    phase: 'Comprehensive ultracode queue (parallel)',
    label: 'wf310H',
  }),
])

phase('Synthesis')

const wf310I = await agent(`Wf-310I: Aggregate Wf-310 comprehensive ultracode queue outcomes — final ready-to-launch dashboard.

USE Read only.

UPSTREAM RESULTS:
- Wf-310A: G5 aggregation fix + 2 missing pairs
- Wf-310B: TODO-182 G6 final wrapper
- Wf-310C: TODO-179 + TODO-220 + Wf-219 §4.1 wrappers
- Wf-310D: Wf-301A drugood DANN + Wf-301B Vina wrappers
- Wf-310E: Wf-135 Phase 2 + Wf-219 §4.2 wrappers
- Wf-310F: TODO-234 frontier math 5-7 day plan
- Wf-310G: 4 LOI send-ready package
- Wf-310H: 8 actionable-now CPU tasks bundle

TASK:
1. Read all 8 outputs
2. Aggregate: completed + failed + partial
3. Build final ready-to-launch dashboard:
   - USER_OK-gated GPU queue (sorted by wall)
   - ACTIONABLE-NOW CPU queue
   - LOI send-ready queue
   - Remaining blockers
4. Output to molmetal/reports/wf310_synthesis/ultracode_queue_dashboard.md

OUTPUT (PLAIN TEXT): per_agent_summary + user_ok_gpu_queue + actionable_now_cpu_queue + loi_send_queue + total_remaining_blockers (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf310I',
})

return {
  ultracode: 'wf-310-ultracode-queue-comprehensive',
  wf310a: wf310A,
  wf310b: wf310B,
  wf310c: wf310C,
  wf310d: wf310D,
  wf310e: wf310E,
  wf310f: wf310F,
  wf310g: wf310G,
  wf310h: wf310H,
  wf310i: wf310I,
}