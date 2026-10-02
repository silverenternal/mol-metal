export const meta = {
  name: 'wf-308-remaining-gaps-round2',
  description: 'Wf-308: Remaining gaps round 2 — actionable-now CPU-only follow-up from Wf-307 honest-negative + Wf-303C. (A) TODO-178b smoke-test FAIL-on-missing global audit (0.5 day, party-twenty-spirit §3.5 hygiene). (B) ADMET pass-through integration — wire Wf-303C admet_pass_through.py into pre_flight_check.py + paper_dashboard.py. (C) TODO-220 --execute CPU implementation — temporal-split + NCI-60 9-cell driver execute path (CPU code only; GPU run needs USER_OK). (D) Wf-219 §4.3 temporal-split actual CPU smoke run + 4 LOI consolidation review helper. (E) Synthesis.',
  phases: [
    { title: 'Remaining gaps round 2 (parallel)', detail: '4 parallel: smoke audit, ADMET integration, TODO-220 --execute CPU, Wf-219 CPU smoke + LOI review' },
    { title: 'Synthesis', detail: 'Aggregate + remaining actionable + USER_OK-gated GPU queue' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Remaining gaps round 2 (parallel)')

const [wf308A, wf308B, wf308C, wf308D] = await parallel([

  () => agent(`Wf-308A: TODO-178b smoke-test FAIL-on-missing global audit.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-307B: TODO-178b smoke-test FAIL-on-missing hardening (0.5 day, party-twenty-spirit §3.5 hygiene). Per Wf-303C verified _metalcytotox_preflight raises AssertionError (line 562-625) + meta-test (line 763+), but the audit only covers ONE module. Global audit across molmetal/tests/ is missing.

TASK:
1. Read molmetal/tests/test_wasserstein_closure_integration.py — Wf-303C reference (line 763+ meta-test)
2. Read molmetal/scripts/pre_flight_check.py — current 32-gate v3 structure (Wf-306C)
3. Audit all smoke tests in molmetal/tests/:
   grep -rn "pytest.skip\|skipif\|@pytest.mark.smoke" molmetal/tests/ 2>&1 | head -50
4. Find tests marked as smoke or named *smoke* that should FAIL-on-missing but currently skip
5. Write NEW meta-test: molmetal/tests/test_smoke_fail_on_missing_audit.py
   - Walk all tests in molmetal/tests/
   - Find tests marked with @pytest.mark.smoke or named *smoke*
   - Audit each: does it raise (not skip) on missing data?
   - Distinguish legitimate skips (GPU-only on CPU env) vs illegitimate skips
6. Run audit: ${PY} -m pytest molmetal/tests/test_smoke_fail_on_missing_audit.py -v 2>&1 | tail -30
7. For any illegitimate skip-on-missing, fix them:
   - Replace pytest.skip with pytest.fail + AssertionError
   - Per party-twenty-spirit §3.5: NEVER silently skip; ALWAYS fail loudly
8. Add NEW pre-flight v3 gate (per Wf-306C expansion):
   - Gate: smoke_fail_on_missing_audit_test_present + passes
9. Re-run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
10. Document: TODO/completed/178b_smoke_fail_on_missing_hardening.md

Honest framing:
- 0.5 day wall; CPU-only
- Hygiene work per party-twenty-spirit §3.5
- Legitimate skips (GPU on CPU env) should stay; illegitimate skips → fix to fail
- Pre-flight v3 should preserve 32/32 gates (or honest-count disclosure if any gate fails)

OUTPUT (PLAIN TEXT): n_smoke_tests_audited + n_illegitimate_skips_found + n_skips_fixed + meta_test_status + preflight_status + per_party_twenty_spirit_compliance. Do NOT call StructuredOutput.`, {
    phase: 'Remaining gaps round 2 (parallel)',
    label: 'wf308A',
  }),

  () => agent(`Wf-308B: ADMET pass-through integration — wire Wf-303C admet_pass_through.py into pre_flight_check.py + paper_dashboard.py.

USE Bash + Read + Edit.

CONTEXT: Per Wf-303C Task 2: NEW molmetal/scripts/admet_pass_through.py (~340 LOC) emits 3 metric JSONs from 449 metallo-drug SMILES via existing metrics_v2. Results: ames_mutagen 0.0401 PASS / cell_permeability_logPapp -3.637 PASS / aqueous_solubility_logS -7.612 FAIL honest-negative. JSON files at metrics/by_metric/{ames_mutagen,cell_permeability_logPapp,aqueous_solubility_logS}.json. Per Wf-307C: integration work NOT yet done.

TASK:
1. Read molmetal/scripts/admet_pass_through.py — Wf-303C output
2. Read molmetal/scripts/pre_flight_check.py — current 32-gate v3 (Wf-306C)
3. Add NEW gates (per Wf-307C plan, verify not already added):
   - Gate: ames_mutagen.json exists + value ≤ 0.10 (druglikeness)
   - Gate: cell_permeability_logPapp.json exists + value ≥ -5.0 (druglikeness)
   - Gate: aqueous_solubility_logS.json exists + HONEST_NEGATIVE tag present
4. Update pre_flight_check.py gate counter if needed
5. Read molmetal/scripts/paper_dashboard.py — current 34+ claims (Wf-304A)
6. Add NEW claim cards (per Wf-307C plan):
   - Claim: ames_mutagen 0.0401 (PASS, druglikeness)
   - Claim: cell_permeability_logPapp -3.637 (PASS, druglikeness)
   - Claim: aqueous_solubility_logS -7.612 (FAIL honest-negative, druglikeness)
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Verify paper-dashboard: ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v308b.html 2>&1 | tail -10
9. Document: TODO/completed/admet_pass_through_integration.md

Honest framing:
- This is INTEGRATION work, not new MEASUREMENT
- 3 metrics already MEASURED per Wf-303C; this surfaces them in pre-flight + paper-dashboard
- aqueous_solubility_logS -7.612 IS a real honest-negative — keep it visible
- Pre-flight v3 should preserve 32/32 (now +3 = 35/35 if gates added) or honest-count disclosure

OUTPUT (PLAIN TEXT): n_admet_gates_added + preflight_status + n_dashboard_claims_added + dashboard_status + per_metric_verdict + per_metric_visible_in_pre_flight. Do NOT call StructuredOutput.`, {
    phase: 'Remaining gaps round 2 (parallel)',
    label: 'wf308B',
  }),

  () => agent(`Wf-308C: TODO-220 --execute CPU implementation — temporal-split + NCI-60 9-cell driver execute path.

USE Bash + Read + Edit.

CONTEXT: Per Wf-305C + Wf-306B: TODO-220 stub drivers (wf219_temporal_split_driver.py + nci60_9cell_chemprop_driver.py) shipped with --dry-run mode. --execute path not implemented. Per Wf-306B: --execute CPU code is the actionable-now piece; full GPU run needs USER_OK.

TASK PART 1 — wf219_temporal_split_driver --execute CPU implementation:
1. Read molmetal/scripts/wf219_temporal_split_driver.py — current --dry-run path
2. Read molmetal/scripts/wf219_krasnov_aligned.py — reference impl with --split {scaffold,temporal} flag (Wf-303C Task 3)
3. Implement --execute path:
   - Load metalcytotox_5metal_multitask_v4.csv
   - Apply temporal split: train ≤ 2023, test 2024-2025 (per Krasnov 2026 exact protocol)
   - Per-seed chemprop v2 training loop (CPU, 1 epoch smoke)
   - Emit JSON with per-seed test_auc + per-metal breakdown
4. Verify CSV exists: ls -la molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
5. If CSV MISSING: document, do not run --execute (per Wf-306B honest-negative)
6. If CSV PRESENT: run --execute smoke (1 seed, 1 epoch, hidden=64, depth=2):
   ${PY} molmetal/scripts/wf219_temporal_split_driver.py --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --train-cutoff 2023-01-01 --test-cutoff 2024-01-01 --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf308c_temporal_smoke --execute 2>&1 | tail -20
7. Verify: JSON emitted with n_seeds_completed + per_seed_test_auc + per_metal_auc

TASK PART 2 — nci60_9cell_chemprop_driver --execute CPU implementation:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — current --dry-run path
2. Implement --execute path:
   - Load nci60_9cell.csv (verify exists; if missing, document + skip)
   - Per-cell-line multi-task chemprop v2 training loop (CPU, 1 epoch smoke)
   - Emit JSON with per-cell-line test_auc
3. Verify CSV exists: ls -la molmetal/.data/molmetal/nci60_9cell.csv
4. If CSV MISSING: document, do not run --execute
5. If CSV PRESENT: run --execute smoke (1 cell, 1 epoch):
   ${PY} molmetal/scripts/nci60_9cell_chemprop_driver.py --csv molmetal/.data/molmetal/nci60_9cell.csv --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf308c_nci60_smoke --execute 2>&1 | tail -20

TASK PART 3 — GPU wrapper for full --execute (USER_OK gated):
1. Write NEW launch wrapper: scripts/launch_wf308c_todo220_gpu.sh
   - USER_OK gate enforced (per Wf-300C pattern)
   - Total wall: 2.9 hr (Driver 1: 5 metals × 5 seeds × 50 epochs) + 1.0 hr (Driver 2: 9-cell) = 3.9 hr GPU
2. bash -n syntax check
3. Document: molmetal/reports/wf308c_todo220_launch/launch_plan.md

Honest framing:
- --execute CPU implementation = actionable-now (no user OK)
- Full --execute GPU = USER_OK required per kill_then_adapt
- CSVs may be MISSING (per Wf-306B honest-negative); handle gracefully

OUTPUT (PLAIN TEXT): todo220_temporal_csv_status + todo220_temporal_execute_smoke_status + n_seeds_completed + todo220_nci60_csv_status + todo220_nci60_execute_smoke_status + gpu_wrapper_path + user_ok_status. Do NOT call StructuredOutput.`, {
    phase: 'Remaining gaps round 2 (parallel)',
    label: 'wf308C',
  }),

  () => agent(`Wf-308D: Wf-219 §4.3 temporal-split actual CPU smoke run + 4 LOI consolidation review helper.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-303C Task 3: Wf-219 §4.3 added --split {scaffold,temporal} flag; Wf-306B verified dry-run OK but actual CPU smoke not yet run. Per Wf-305A: Rusanov LOI drafted; 3 older LOIs (ORNL/PKU/Xiehe) also DRAFT — all need Hugo review + send.

TASK PART 1 — Wf-219 §4.3 temporal-split actual CPU smoke:
1. Read molmetal/scripts/wf219_krasnov_aligned.py — current --split {scaffold,temporal} flag
2. Verify CSV exists: ls -la molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
3. If MISSING: document + skip
4. If PRESENT: run temporal-split actual smoke:
   ${PY} molmetal/scripts/wf219_krasnov_aligned.py --split temporal --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf308d_temporal_smoke 2>&1 | tail -20
5. Verify: JSON emitted with n_seeds_completed + per_seed_test_auc
6. Document per-metal breakdown (Ru/Ir/Os/Rh/Re)

TASK PART 2 — 4 LOI consolidation review helper:
1. Read all 4 LOI drafts:
   - TODO/pending/wetlab_loi_ornl_protein_metal.md
   - TODO/pending/wetlab_loi_pku_cisplatin_ic50.md
   - TODO/pending/wetlab_loi_xiehe_nci60.md
   - TODO/pending/wetlab_loi_rusanov_2025_platinai.md (Wf-305A NEW)
2. For each LOI, audit:
   - Status (DRAFT / SENT / REPLIED / DECLINED)
   - Target PI + email (if known)
   - What's asked
   - What's offered
   - Action required from Hugo
3. Write NEW review helper: TODO/pending/wetlab_loi_consolidation_review.md
   - 4-row summary table
   - Recommended send order
   - Per-LOI next-action checklist for Hugo
4. NO send (cold-email is user's responsibility per TODO-84)

TASK PART 3 — G5 retrain integration template:
1. Read results/wf124_p3_50epoch_full_sweep_g5/ — current G5 retrain output (PID 1015907)
2. Write NEW template script: molmetal/scripts/wf308d_g5_results_integration.py
   - Detect when G5 retrain completes (per-seed JSON files emitted)
   - Aggregate per-metal mean ± std
   - Update molmetal/reports/wf219_krasnov_aligned/sweep_summary.json with G5 rows
   - Update metrics/by_metric/wf219_krasnov_aligned_*.json
   - Update paper_dashboard.py with new claim cards
   - Trigger pre_flight_check.py (verify 32+/32+ PASS preserved)
3. Document: molmetal/reports/wf308d_g5_integration_template/integration_plan.md
4. NO actual integration (waits for G5 retrain to finish — USER OK already in flight via PID 1015907)

Honest framing:
- TASK PART 1: actionable-now if CSV present
- TASK PART 2: documentation consolidation, zero code execution
- TASK PART 3: template only, no actual integration; waits for G5 retrain
- Per TODO-84: no paper work without authorization; LOI send is user's responsibility

OUTPUT (PLAIN TEXT): wf219_temporal_smoke_status + n_lois_audited + n_lois_recommended_send + g5_integration_template_path + per_loi_next_action. Do NOT call StructuredOutput.`, {
    phase: 'Remaining gaps round 2 (parallel)',
    label: 'wf308D',
  }),
])

phase('Synthesis')

const wf308E = await agent(`Wf-308E: Aggregate Wf-308 remaining gaps round 2 outcomes.

USE Read only.

UPSTREAM RESULTS:
- Wf-308A: TODO-178b smoke-test audit
- Wf-308B: ADMET pass-through integration
- Wf-308C: TODO-220 --execute CPU implementation
- Wf-308D: Wf-219 temporal-split CPU smoke + 4 LOI review + G5 integration template

TASK:
1. Read all 4 outputs
2. Aggregate: completed + failed + partial
3. Identify remaining actionable + USER_OK-gated GPU queue
4. Output to molmetal/reports/wf308_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + actionable_closed + user_ok_gpu_queue + remaining_blockers (8-12 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf308E',
})

return {
  ultracode: 'wf-308-remaining-gaps-round2',
  wf308a: wf308A,
  wf308b: wf308B,
  wf308c: wf308C,
  wf308d: wf308D,
  wf308e: wf308E,
}