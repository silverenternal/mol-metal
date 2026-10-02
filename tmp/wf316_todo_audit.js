export const meta = {
  name: 'wf-316-todo-directory-audit-cleanup',
  description: 'Wf-316: TODO directory audit + cleanup. (A) Inventory pending/ files (20 entries) + classify each as DONE (closeout doc exists in completed/) vs STILL_PENDING (work remains) vs KEEP_RULE (rule memo that must stay in pending/, e.g. TODO-84). (B) For DONE files: git mv to completed/ + delete original pending/ file. (D) For STILL_PENDING files: emit TODO/completed/<id>_status_hold.md honest-negative marker + keep in pending/. (E) For KEEP_RULE files: keep in pending/ + add cross-reference index. (F) For wetlab SOPs: keep templates in pending/ (per Wf-310G Part 2 they are TEMPLATE — not executed). (G) Update TODO/INDEX.md §0 + §22 with audit summary. (H) Synthesis + TODO/README.md sync.',
  phases: [
    { title: 'TODO directory audit (parallel)', detail: '8 parallel agents covering pending file classification + move + index' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md update + completion summary' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('TODO directory audit (parallel)')

const [wf316A, wf316B, wf316C, wf316D, wf316E, wf316F] = await parallel([

  () => agent(`Wf-316A: Audit + move pending Wf-123/Wf-124 P3 gap TODOs (177/179/180/181/182/183).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-313 / Wf-310H / Wf-310F / Wf-310E / Wf-312A / Wf-314 outputs, several pending files are actually DONE (closeout docs in TODO/completed/). Need to inventory + classify + move DONE files to completed/ + delete pending versions + emit status-hold docs for STILL_PENDING.

TASK PART 1 — Inventory 6 Wf-123/Wf-124 P3 gap TODOs:
1. List TODO/pending/{177,179,180,181,182,183}_*.md
2. List TODO/completed/{177,179,180,181,182,183}_*.md (look for closeout docs)
3. For each pending file, classify:
   - DONE: git mv completed/, delete from pending
   - STILL_PENDING: emit status-hold markdown + keep in pending
   - DUPLICATE: delete pending (closeout already exists)

TASK PART 2 — Specific pending files:
1. 177_wf123_p3_m1m2_downstream_metric_lift.md:
   - Check TODO/completed/177_wf313g_downstream_lift_dashboard.md (Wf-313G)
   - If exists: git mv pending/177 → completed/, delete from pending
   - If not: STILL_PENDING
2. 179_wf123_p3_w2_pic50_alignment.md:
   - Wf-310C PART 3 + Wf-310F frontier wire-in covered this
   - Check TODO/completed/179_*.md or 234_*.md (related)
   - If wrapper exists: git mv pending/179 → completed/
   - Otherwise: STILL_PENDING (wrapper is in pipeline Phase 3)
3. 180_wf124_p3_remaining_metals_seeds.md:
   - Wf-310A G5 retrain covered remaining pairs
   - Check TODO/completed/180_*.md (look for daily_summary_2026_09_27)
   - If closeout exists: git mv pending/180 → completed/
   - Otherwise: STILL_PENDING
4. 181_wf124_p3_50_epoch_full_sweep.md:
   - Wf-310A closeout at TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md
   - git mv pending/181 → completed/
5. 182_wf124_p3_per_innovation_ablation.md:
   - Wf-310B + G6 in flight
   - TODO/completed/182_*.md check
   - If wrapper closeout exists: git mv pending/182 → completed/
   - Otherwise: STILL_PENDING
6. 183_wf123_wf124_p3_gap_routing_plan.md:
   - This is a planning doc, not a ship surface
   - git mv pending/183 → completed/ (planning docs ship after use)
   - Or keep in pending if useful

TASK PART 3 — Verify with pre-flight:
1. ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
2. Verify file moves correct

TASK PART 4 — Document audit:
1. molmetal/reports/wf316a_pending_audit_wf123_124.md
2. List per-file: classification (DONE/STILL_PENDING/KEEP) + action taken

Honest framing per party-twenty-spirit §3.5:
- DONE files: ship closeout already exists → move to completed/
- STILL_PENDING files: status-hold marker + keep in pending/
- DO NOT fabricate completion

OUTPUT (PLAIN TEXT): n_files_audited + n_done_moved + n_pending_kept + n_duplicates_deleted + preflight_status + per_file_classification. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316A',
  }),

  () => agent(`Wf-316B: Audit + move pending TODO-220 (Wf-219 temporal + NCI-60 driver) + TODO-234 (frontier math).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-308C + Wf-310C + Wf-313D + Wf-313C: TODO-220 and TODO-234 have closeout docs at TODO/completed/220_wf219_nci60_9cell_acceptance.md + 220_wf220_temporal_split_acceptance.md + 234_wf234_frontier_math_cpu_plan.md + 234_wf234_frontier_math_cpu_portion.md.

TASK:
1. List pending files:
   - TODO/pending/220_wf219_temporal_split_nci60_driver.md
   - TODO/pending/234_wf234_frontier_math_audit_integration.md
2. List completed closeouts:
   - TODO/completed/220_*.md
   - TODO/completed/234_*.md
3. For TODO-220:
   - Wf-308C: TODO-220 drivers --execute path SHIPPED
   - Wf-310C: TODO-220 + TODO-179 + Wf-219 §4.1 wrappers READY
   - Wf-313D: §6.3 + §6.4 acceptance gates SHIPPED
   - git mv pending/220 → completed/ (delete from pending)
4. For TODO-234:
   - Wf-309B: 4 frontier adapters composed + 4 smoke tests PASS
   - Wf-310F: Day 1-2 CPU portion SHIPPED
   - Wf-313C: Day 3-5 CPU portion SHIPPED
   - git mv pending/234 → completed/ (delete from pending)
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: molmetal/reports/wf316b_todo220_234_audit.md

Honest framing:
- TODO-220 has CPU + GPU components — CPU done (Wf-308C + Wf-310C + Wf-313D), GPU in pipeline Phase 3
- TODO-234 has CPU + GPU components — CPU done (Wf-309B + Wf-310F + Wf-313C), GPU in pipeline Phase 3
- Closeout doc scope is CPU portion; GPU portion is in pipeline (separate)

OUTPUT (PLAIN TEXT): per_file_status + n_moved + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316B',
  }),

  () => agent(`Wf-316C: Audit + classify wetlab LOI 4 drafts (Rusanov / PKU / Xiehe / ORNL) + consolidation review.

USE Bash + Read + Edit.

CONTEXT: Per Wf-308D + Wf-310G + Wf-312E + Wf-313B: 4 LOI drafts exist + send-ready package + cold-email template + consolidation review. All 4 LOIs are DRAFT (0 sent — user responsibility per TODO-84).

TASK:
1. List pending files:
   - TODO/pending/wetlab_loi_rusanov_2025_platinai.md
   - TODO/pending/wetlab_loi_pku_cisplatin_ic50.md
   - TODO/pending/wetlab_loi_xiehe_nci60.md
   - TODO/pending/wetlab_loi_ornl_protein_metal.md
   - TODO/pending/wetlab_loi_consolidation_review.md
   - TODO/pending/wetlab_cold_email_template.md
   - TODO/pending/wetlab_loi_send_ready_package.md
2. List completed closeouts:
   - TODO/completed/wetlab_loi_send_ready_package_final.md (Wf-313B)
   - TODO/completed/308d_wf312e_loi_send_ready_and_g5_integration_final.md (Wf-312E)
3. For each LOI draft:
   - Status: DRAFT (user responsibility per TODO-84 to send)
   - Recommendation: KEEP in pending/ until actually sent
   - Or: git mv to completed/ if there's a draft closeout
4. For wetlab_loi_send_ready_package.md: check if Wf-313B supersedes
   - If wetlab_loi_send_ready_package_final.md exists in completed/, the original send_ready_package.md is duplicate
   - git mv pending/wetlab_loi_send_ready_package.md → completed/wetlab_loi_send_ready_package_v1.md (keep versioned)
5. For wetlab_cold_email_template.md: KEEP in pending/ (template)
6. For wetlab_loi_consolidation_review.md: KEEP in pending/ (active review helper)
7. Document: molmetal/reports/wf316c_wetlab_loi_audit.md

Honest framing:
- 4 LOIs are DRAFT, not sent — user responsibility
- Wetlab work is NOT auto-shippable (requires actual wet-lab data collection)
- Keep templates in pending/ unless superseded by closeout

OUTPUT (PLAIN TEXT): n_lois_audited + per_loi_classification + n_moved + n_kept + n_duplicates. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316C',
  }),

  () => agent(`Wf-316D: Audit + classify wetlab SOPs (3) + wetlab_ic50_analysis_pipeline + TODO-84 rule file.

USE Bash + Read + Edit.

CONTEXT: Per TODO-84 rule memo + Wf-310G: wetlab SOPs are TEMPLATE (no wet-lab data collected yet). TODO-84 is the rule that stays in pending/ forever (no paper work without authorization).

TASK:
1. List pending files:
   - TODO/pending/wetlab_sop_cytotoxicity_assay.md
   - TODO/pending/wetlab_sop_cellular_uptake_assay.md
   - TODO/pending/wetlab_sop_binding_affinity_assay.md
   - TODO/pending/wetlab_ic50_analysis_pipeline.md
   - TODO/pending/84_no_paper_task_without_authorization.md
2. For 3 wetlab SOPs:
   - Per Wf-313B: SOPs enhanced + 24 tests (8/SOP)
   - Per Wf-310G: SOPs are institutional-tailored templates, no execution
   - Recommendation: KEEP in pending/ (templates, not ship surface)
   - Or: git mv to completed/ if Wf-313B closeout exists
3. For wetlab_ic50_analysis_pipeline.md:
   - Per Wf-313B: IC50 pipeline enhanced with 4 NEW matrix + 6 tests/matrix
   - Check if Wf-313B closeout exists
   - If yes: git mv pending → completed
   - If no: KEEP in pending
4. For TODO-84:
   - This is the RULE memo (active rule)
   - Per Wf-310G §2: NEVER move TODO-84 — it's the cross-cutting rule
   - KEEP in pending/ (with explicit comment)
7. Document: molmetal/reports/wf316d_wetlab_sop_audit.md

Honest framing:
- Wetlab SOPs are TEMPLATES (Wf-310G + Wf-313B enhanced them but no execution)
- TODO-84 is rule memo, must stay in pending/
- IC50 pipeline may have closeout pending Wf-313B

OUTPUT (PLAIN TEXT): per_file_classification + n_sop_moved + n_pipeline_moved + rule_keep_reason + n_kept. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316D',
  }),

  () => agent(`Wf-316E: Check pending 180_daily_summary + other miscellaneous pending files + INDEX.md §0 update.

USE Bash + Read + Edit.

CONTEXT: After Wf-316A-D, pending/ should be cleaner. Need to update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table) to reflect new state.

TASK:
1. List remaining pending files after Wf-316A-D (defer to actual file system state)
2. Check if any pending file is a daily_summary or per-day file that should be moved to completed/
3. For each remaining pending file, classify:
   - DONE: git mv
   - STILL_PENDING: keep
   - RULE: keep
5. Update TODO/INDEX.md §0 with Wf-309/310/311/312/313/314/315 ship entries
6. Update §22 (detailed table) with closeout docs that landed
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/316_index_md_update.md

Honest framing:
- Daily summaries are typically moved to completed/ after the day ends
- INDEX.md should reflect current state, not stale state

OUTPUT (PLAIN TEXT): n_remaining_pending + n_moved + preflight_status + index_md_updated. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316E',
  }),

  () => agent(`Wf-316F: Synthesize TODO directory final state — pending vs completed count + per-file ledger.

USE Bash + Read + Write + Edit.

CONTEXT: After Wf-316A-E audit, need to synthesize final state and emit comprehensive overview.

TASK:
1. List all files in TODO/pending/ (after Wf-316A-E moves)
2. List all files in TODO/completed/ (count + recent additions)
3. Categorize pending:
   - STILL_PENDING: work remains (count + per-file)
   - RULE: TODO-84 (1)
   - TEMPLATE: wetlab SOPs/IC50/LOI drafts (count)
4. Emit:
   - TODO/completed/316_todo_directory_final_state.md
   - Section 1: Before-state (20 files in pending)
   - Section 2: Wf-316A-E moves (per-file audit)
   - Section 3: After-state (X files in pending)
   - Section 4: Per-file ledger (20 rows × {file, classification, action, destination})
5. Update TODO/README.md with current state + Wf-309/310/311/312/313/314/315/316 entries
6. Verify: ls TODO/pending/ | wc -l (should be < 20)
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- Per party-twenty-spirit §3.5: every move documented
- DO NOT fabricate completion — only audit what's actually done

OUTPUT (PLAIN TEXT): n_before + n_after + n_moved + n_kept + per_category_count + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'TODO directory audit (parallel)',
    label: 'wf316F',
  }),
])

phase('Synthesis')

const wf316G = await agent(`Wf-316G: Aggregate Wf-316 TODO directory audit + cleanup outcomes — final INDEX.md sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-316A: Wf-123/124 P3 gap TODOs audit + move
- Wf-316B: TODO-220 + TODO-234 audit + move
- Wf-316C: Wetlab LOI 4 drafts audit
- Wf-316D: Wetlab SOPs + IC50 pipeline + TODO-84 audit
- Wf-316E: INDEX.md update
- Wf-316F: Final state overview + ledger

TASK:
1. Read all 6 outputs
2. Aggregate: completed moves + still-pending keeps + duplicates deleted
3. Final INDEX.md update with Wf-309/310/311/312/313/314/315/316 ship wave
4. Output: TODO/completed/316_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_moves + n_keeps + n_duplicates + final_count + INDEX_md_status (10-15 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf316G',
})

return {
  ultracode: 'wf-316-todo-directory-audit-cleanup',
  wf316a: wf316A,
  wf316b: wf316B,
  wf316c: wf316C,
  wf316d: wf316D,
  wf316e: wf316E,
  wf316f: wf316F,
  wf316g: wf316G,
}