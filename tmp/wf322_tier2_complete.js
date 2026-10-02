export const meta = {
  name: 'wf-322-tier2-complete-plus-new',
  description: 'Wf-322: Wf-317 Tier-2 strategy complete + new concrete actionable-now code tasks. (A) Wf-317F partial — 4 LOI pending files final Tier-2 TEMPLATE_NOT_NEEDED markers. (B) Wf-317G partial — 3 wetlab SOP + IC50 pipeline final Tier-2 markers. (C) Wf-317H partial — IMPROVEMENT_QUEUE.md TODO-138 Tier-2 + INDEX.md sync. (D) Wf-317I synthesis final — INDEX.md §0 ship wave sync + README.md Wf-309..322 entries. (E) TODO-148 R20 §1.1 3 loader modules API doc writing (NEW from Wf-317E). (F) Wf-176 m19 LOMO Rh oversample final closeout — synthesize Wf-309A Phase 1 + Wf-321A Phase 2 scaffold + Wf-135 pipeline. (G) New concrete code-class: TODO-130 G3 closure theorem Catherine 30-metal bench (R20-P5 alignment per LAMBDA_ROADMAP §3.3 R22-P5). (H) Synthesis + Wf-322 final INDEX.md sync.',
  phases: [
    { title: 'Wf-317 Tier-2 complete + new tasks (parallel)', detail: '7 parallel agents completing Wf-317 partial + new concrete tasks' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('Wf-317 Tier-2 complete + new tasks (parallel)')

const [wf322A, wf322B, wf322C, wf322D, wf322E, wf322F, wf322G] = await parallel([

  () => agent(`Wf-322A: Wf-317F partial completion — 4 LOI pending files final Tier-2 markers.

USE Bash + Read + Edit.

CONTEXT: Per Wf-317F partial: 4 LOI pending files (Rusanov / PKU / Xiehe / ORNL) need final TEMPLATE_NOT_NEEDED_FOR_TIER2 markers. Per AGENTS.md §5.4 user directive.

TASK:
1. Read 4 LOI files:
   - TODO/pending/wetlab_loi_rusanov_2025_platinai.md
   - TODO/pending/wetlab_loi_pku_cisplatin_ic50.md
   - TODO/pending/wetlab_loi_xiehe_nci60.md
   - TODO/pending/wetlab_loi_ornl_protein_metal.md
2. For each LOI, verify Tier-2 marker is present (add if missing):
   **Status (2026-09-29)**: TEMPLATE_NOT_NEEDED_FOR_TIER2 — per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验"
3. Verify pre-flight: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
4. Document: TODO/completed/322a_4_loi_tier2_final.md

Honest framing:
- LOIs are deprecated, not deleted (kept as historical reference)
- TODO-84 still applies for paper/* but wet-lab send is no longer project scope
- Tier-2 = pure computational

OUTPUT (PLAIN TEXT): n_lois_verified + per_loi_marker_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322A',
  }),

  () => agent(`Wf-322B: Wf-317G partial completion — 3 wetlab SOP + IC50 pipeline final Tier-2 markers.

USE Bash + Read + Edit.

CONTEXT: Per Wf-317G partial: 3 wetlab SOPs + IC50 pipeline need final TEMPLATE_NOT_NEEDED_FOR_TIER2 markers.

TASK:
1. Read 4 pending files:
   - TODO/pending/wetlab_sop_cytotoxicity_assay.md
   - TODO/pending/wetlab_sop_cellular_uptake_assay.md
   - TODO/pending/wetlab_sop_binding_affinity_assay.md
   - TODO/pending/wetlab_ic50_analysis_pipeline.md
2. For each, verify Tier-2 marker is present (add if missing):
   **Status (2026-09-29)**: TEMPLATE_NOT_NEEDED_FOR_TIER2 — per AGENTS.md §5.4 user directive
3. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
4. Document: TODO/completed/322b_wetlab_sop_tier2_final.md

OUTPUT (PLAIN TEXT): n_files_verified + per_file_marker_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322B',
  }),

  () => agent(`Wf-322C: Wf-317H partial completion — IMPROVEMENT_QUEUE.md TODO-138 Tier-2 + INDEX.md sync.

USE Bash + Read + Edit.

CONTEXT: Per Wf-317H partial: IMPROVEMENT_QUEUE.md TODO-138 row needs Tier-2 update + INDEX.md sync.

TASK:
1. Read TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md — find TODO-138 row
2. Update paper target: "JCIM" → "Tier-2 journal"
3. Verify INDEX.md §0 includes Wf-309/310/311/312/313/314/315/316/317/318/319/320/321 entries (per Wf-317I synthesis final sync)
4. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/322c_improvement_queue_tier2_final.md

OUTPUT (PLAIN TEXT): n_rows_updated + preflight_status + index_md_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322C',
  }),

  () => agent(`Wf-322D: Wf-317I synthesis final — INDEX.md §0 ship wave sync + README.md Wf-309..322 entries.

USE Bash + Read + Edit.

CONTEXT: Per Wf-317I synthesis: INDEX.md §0 + §22 needs to reflect all completed workflows (Wf-309 through Wf-322).

TASK:
1. Read TODO/INDEX.md §0 + §22 (current ship wave entries)
2. Add ship wave entries for Wf-309..322:
   - Wf-309: actionable-now new (DONE)
   - Wf-310: ultracode queue comprehensive (DONE)
   - Wf-311: hard metric ship (DONE)
   - Wf-312: Wf-308 follow-up (DONE)
   - Wf-313: actionable-now cleanup (DONE)
   - Wf-314: follow-ups (DONE)
   - Wf-315: more code-class (DONE)
   - Wf-316: TODO audit (in flight)
   - Wf-317: Tier-2 strategy update (DONE)
   - Wf-318: actionable-now closeout (in flight)
   - Wf-319: Wf-315 closeout actual (in flight)
   - Wf-320: Wf-313 closeout actual (in flight)
   - Wf-321: more code-class (in flight)
   - Wf-322: Tier-2 complete (this Wf)
3. Update TODO/README.md with Wf-309..322 entries
4. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/322d_index_readme_wf309_322_sync.md

Honest framing:
- Pure docs update, no new measurements
- MEASURED-vs-PROJECTED tags preserved

OUTPUT (PLAIN TEXT): n_wf_entries_added + index_md_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322D',
  }),

  () => agent(`Wf-322E: TODO-148 R20 §1.1 3 loader modules API doc writing.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §1.1: 3 loader modules (binding_db.py + drugood_loader.py + nci60.py) need ≥12 tests + API doc. Per Wf-320F + Wf-321B: tests exist; API docs may not.

TASK:
1. Check if molmetal/data/binding_db.md + drugood_loader.md + nci60.md exist
2. For each loader without API doc: write molmetal/data/<loader>.md
   - Classes + methods + return schema + sample usage
3. Write NEW API doc: molmetal/data/R20_P0_loaders_api.md (index of 3 loader docs)
4. Add pre-flight gate: p3_r20_p0_loader_api_doc_present (verify 3 loader .md files + index)
5. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/322e_todo_148_loader_api_doc.md

Honest framing:
- API docs are documentation work
- Loaders already have code + tests
- Tier-2 strategy preserved

OUTPUT (PLAIN TEXT): n_api_docs_written + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322E',
  }),

  () => agent(`Wf-322F: Wf-176 m19 LOMO Rh oversample final closeout.

USE Bash + Read + Edit.

CONTEXT: Per Wf-309A Phase 1 (Rh cohort SHIPPED) + Wf-321A Phase 2 CPU scaffold + Wf-135 in pipeline. Need final closeout synthesis.

TASK:
1. Read molmetal/reports/wf176_m19_oversample_lomo/aggregate.json — current state
2. Read molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md — Phase 1 cohort
3. Read molmetal/reports/wf321a_wf135_phase2_cpu_scaffold/ — Phase 2 scaffold
4. Read scripts/launch_wf310e_wf135_phase2_gpu.sh — GPU wrapper
5. Emit final closeout: TODO/completed/322f_wf176_m19_lomo_final_closeout.md
   - Phase 1 (Wf-309A): 500 cohort MEASURED
   - Phase 2 scaffold (Wf-321A): CPU scaffold verified
   - Phase 2 GPU: in pipeline (Wf-135)
   - Recovery path: USER_OK required per mol-metal-gpu-execution §0
   - Acceptance: per IMPROVEMENT_ENUM §1.4 (Rh ≥ 0.30, LOMO ≥ 0.60, ≥16 tests)
6. Add pre-flight gate: p3_wf176_m19_lomo_closeout_documented
7. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

OUTPUT (PLAIN TEXT): phase_1_status + phase_2_scaffold_status + phase_2_gpu_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322F',
  }),

  () => agent(`Wf-322G: TODO-130 G3 closure theorem Catherine 30-metal bench — R20-P5 alignment (NEW from Wf-317E LAMBDA_ROADMAP).

USE Bash + Read + Edit.

CONTEXT: Per LAMBDA_ROADMAP §3.3 R22-P5: closure theorem d-block 30 metal benchmark. Per Wf-321B: skill stubs are in. Need concrete Catherine 30-metal CPU smoke benchmark.

TASK:
1. Read molmetal/scripts/bench_catherine_30metals.py — existing benchmark
2. Read molmetal/scripts/audit_catherine_30metals.py — existing audit
3. Verify Catherine 30-metal benchmark integrity:
   - 30 metal coverage (5 + 25 NEW)
   - Closure property coverage
   - Audit gate output
4. Add NEW closeout: TODO/completed/322g_todo130_g3_closure_30metal_closeout.md
   - Per LAMBDA_ROADMAP §3.3 R22-P5 (computational-only after Tier-2 strategy)
   - Per Wf-317 Tier-2 strategy: closure theorem ship surface is computational
   - Recovery path: 30-metal full sweep GPU (in pipeline Phase 3 R20-P5)
5. Add pre-flight gate: p3_catherine_30metal_closeout_documented
6. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- Catherine 30-metal CPU smoke already exists (per Wf-321B)
- This is documentation + closeout work
- Full 30-metal GPU sweep in pipeline (R20-P5)

OUTPUT (PLAIN TEXT): catherine_30metal_status + n_metals_covered + preflight_status + recovery_path. Do NOT call StructuredOutput.`, {
    phase: 'Wf-317 Tier-2 complete + new tasks (parallel)',
    label: 'wf322G',
  }),
])

phase('Synthesis')

const wf322H = await agent(`Wf-322H: Aggregate Wf-322 Tier-2 complete + new tasks outcomes — final INDEX.md sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-322A: 4 LOI pending Tier-2 markers (partial from Wf-317F)
- Wf-322B: 3 wetlab SOP + IC50 pipeline Tier-2 markers (partial from Wf-317G)
- Wf-322C: IMPROVEMENT_QUEUE.md TODO-138 Tier-2 + INDEX.md sync (partial from Wf-317H)
- Wf-322D: INDEX.md §0 + README.md Wf-309..322 ship wave entries
- Wf-322E: TODO-148 3 loader modules API docs
- Wf-322F: Wf-176 m19 LOMO Rh oversample final closeout
- Wf-322G: TODO-130 G3 Catherine 30-metal closeout

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
4. Output: TODO/completed/322_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf322H',
})

return {
  ultracode: 'wf-322-tier2-complete-plus-new',
  wf322a: wf322A,
  wf322b: wf322B,
  wf322c: wf322C,
  wf322d: wf322D,
  wf322e: wf322E,
  wf322f: wf322F,
  wf322g: wf322G,
  wf322h: wf322H,
}