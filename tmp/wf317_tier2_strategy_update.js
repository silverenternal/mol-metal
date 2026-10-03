export const meta = {
  name: 'wf-317-tier2-strategy-update',
  description: 'Wf-317: Tier-2 journal + NO wet-lab strategy update (per user directive 2026-09-29 "我们发二区就可以了，不用规划湿实验"). (A) Update TODO/INDEX.md: replace JCIM Tier-1 with Tier-2 journal in BD-10 + paper section. (B) Update TODO/README.md: Tier-2 target. (C) Update WEAK_TO_STRONG_PLAN.md: deprecate wet-lab focus (m8-prod production sweep → in-silico; m18/m19/m20 wet-lab → in-silico alternatives). (D) Update IMPROVEMENT_ENUM_2026_09_27.md: R20-P6 wet-lab IC50 validation → in-silico metric lift. (E) Update LAMBDA_ROADMAP.md: R22-P1 wet-lab → computational-only; R23 Phase I clinical → DEPRECATED. (F) Update 4 LOI pending files: add Tier-2 strategy header (TEMPLATE_NOT_NEEDED_FOR_TIER2). (G) Update 3 wetlab SOP pending files: add Tier-2 strategy header. (H) Update IMPROVEMENT_QUEUE_2026_09_27.md: TODO-138 paper v2 → Tier-2 journal target. (I) Synthesis + verify pre-flight.',
  phases: [
    { title: 'Tier-2 strategy update (parallel)', detail: '8 parallel agents updating plan files + wetlab templates' },
    { title: 'Synthesis', detail: 'Verify + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Tier-2 strategy update (parallel)')

const [wf317A, wf317B, wf317C, wf317D, wf317E, wf317F, wf317G, wf317H] = await parallel([

  () => agent(`Wf-317A: Update TODO/INDEX.md — replace JCIM Tier-1 with Tier-2 journal target.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4 (NEW 2026-09-29): paper target downgraded from JCIM Tier-1 to Tier-2. Need to update all JCIM references in TODO/INDEX.md.

TASK:
1. Read TODO/INDEX.md (1449 lines)
2. grep for all JCIM references + Q1/Q2/Tier-1/Tier-2 references
3. For each JCIM reference: replace with "Tier-2 journal" or "Tier-2 computational"
4. Add new section after BD-10 update:
   **Note about paper target**: Tier-2 journal (downgraded 2026-09-29); pure computational submission is in scope; no wet-lab data required.
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/317a_index_md_tier2_update.md

Honest framing:
- Pure text update, no measurement
- Tier-2 = computational-only per user directive
- TODO-84 still applies for paper/* touches

OUTPUT (PLAIN TEXT): n_jcim_replaced + n_new_sections + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317A',
  }),

  () => agent(`Wf-317B: Update TODO/README.md — Tier-2 target + 24 frontier papers (no change).

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: Tier-2 target.

TASK:
1. Read TODO/README.md
2. Update paper target section: JCIM → Tier-2 journal
3. Update ship wave summary to include Wf-309/310/311/312/313/314/315/316/317
4. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/317b_readme_tier2_update.md

OUTPUT (PLAIN TEXT): n_sections_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317B',
  }),

  () => agent(`Wf-317C: Update WEAK_TO_STRONG_PLAN.md — deprecate wet-lab focus, re-scope to in-silico.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: no wet-lab planning. WEAK_TO_STRONG_PLAN.md has wet-lab items in §2.6 m8-prod production sweep (PoseBusters) + §2.7 m10 Vina Min Pt-pool + §2.9 m19 LOMO + §3.5 F2(a) MetalLigandExchange + §3.7 Multi-PH.

TASK:
1. Read TODO/03_plans/WEAK_TO_STRONG_PLAN.md
2. Update wet-lab focus items:
   - §2.6 m8-prod: remove "wet-lab R22-P4 future-work" reference
   - §3.5 F2(a): keep (CPU-only math SMARTS rule, no wet-lab)
   - §2.9 m19: keep (CPU + GPU in-silico)
3. Add §5 NEW: "Tier-2 strategy + deprecation" section:
   - Tier-2 target = computational-only
   - Wet-lab items (LOI / SOPs / R22-P4 wet-lab) DEPRECATED
   - In-silico alternatives: GPU retrain + ablations + closeout docs
4. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/317c_weak_to_strong_tier2_update.md

OUTPUT (PLAIN TEXT): n_sections_modified + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317C',
  }),

  () => agent(`Wf-317D: Update IMPROVEMENT_ENUM_2026_09_27.md — R20-P6 wet-lab IC50 → in-silico metric lift.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: R20-P6 wet-lab IC50 validation deprecated. IMPROVEMENT_ENUM §0 has 4 P0 tasks; Task #4 = TODO-138 paper v2.

TASK:
1. Read TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md
2. Update:
   - Header "3 个 P0 任务" → adjust count if needed
   - Task #2 Wf-134 / TODO-134: keep (CPU + GPU in-silico)
   - Add Wf-316 + Wf-317 references to ship wave
3. Add §2.5 NEW: "Tier-2 strategy + wet-lab deprecation":
   - Tier-2 = computational-only
   - 4 LOIs + 3 SOPs + R20-P6 wet-lab DEPRECATED for project scope
4. Update §0 enumeration table: add Wf-316 + Wf-317 entries if relevant
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/317d_improvement_enum_tier2_update.md

OUTPUT (PLAIN TEXT): n_sections_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317D',
  }),

  () => agent(`Wf-317E: Update LAMBDA_ROADMAP.md — R22-P1 wet-lab → computational-only; R23 Phase I → DEPRECATED.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: no wet-lab planning. LAMBDA_ROADMAP §3.2 has R22-P1 (m18 wet-lab) + §3.3 R23 (Phase I clinical).

TASK:
1. Read TODO/03_plans/LAMBDA_ROADMAP.md
2. Update:
   - §3.2 R22-P1 m18 multi-modal protein-metal: re-scope as computational-only
   - §3.3 R23 Phase I clinical: add DEPRECATED marker
   - §3.4 R24 Phase II: add DEPRECATED marker
   - §3.5 R25 Phase III: add DEPRECATED marker
3. Update §1.6 R20 honesty framing: Tier-2 = computational-only (no wet-lab)
4. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/317e_lambda_roadmap_tier2_update.md

OUTPUT (PLAIN TEXT): n_phases_deprecated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317E',
  }),

  () => agent(`Wf-317F: Update 4 LOI pending files — add Tier-2 strategy header (TEMPLATE_NOT_NEEDED_FOR_TIER2).

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: 4 LOI drafts deprecated for Tier-2 scope.

TASK:
1. Read 4 LOI files:
   - TODO/pending/wetlab_loi_rusanov_2025_platinai.md
   - TODO/pending/wetlab_loi_pku_cisplatin_ic50.md
   - TODO/pending/wetlab_loi_xiehe_nci60.md
   - TODO/pending/wetlab_loi_ornl_protein_metal.md
2. For each LOI, add header banner (after §0 / at top):
   **Status (2026-09-29)**: TEMPLATE_NOT_NEEDED_FOR_TIER2 — per AGENTS.md §5.4 user
   directive "我们发二区就可以了，不用规划湿实验", Tier-2 journal is sufficient and
   wet-lab validation is OUT OF SCOPE. This LOI draft is preserved as historical
   reference but NOT for sending.
3. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
4. Document: TODO/completed/317f_4_loi_tier2_template_marker.md

Honest framing:
- LOIs are deprecated, not deleted (kept as historical reference)
- TODO-84 still applies for paper/* but wet-lab send is no longer project scope
- Tier-2 = pure computational

OUTPUT (PLAIN TEXT): n_lois_updated + per_loi_marker + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317F',
  }),

  () => agent(`Wf-317G: Update 3 wetlab SOP + IC50 pipeline pending files — add Tier-2 strategy header.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: 3 wetlab SOPs + IC50 pipeline deprecated for Tier-2.

TASK:
1. Read 4 pending files:
   - TODO/pending/wetlab_sop_cytotoxicity_assay.md
   - TODO/pending/wetlab_sop_cellular_uptake_assay.md
   - TODO/pending/wetlab_sop_binding_affinity_assay.md
   - TODO/pending/wetlab_ic50_analysis_pipeline.md
2. For each, add header banner:
   **Status (2026-09-29)**: TEMPLATE_NOT_NEEDED_FOR_TIER2 — per AGENTS.md §5.4 user
   directive "我们发二区就可以了，不用规划湿实验", Tier-2 journal is sufficient and
   wet-lab validation is OUT OF SCOPE. This template is preserved as historical
   reference but NOT for execution.
3. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
4. Document: TODO/completed/317g_wetlab_sop_tier2_marker.md

OUTPUT (PLAIN TEXT): n_files_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317G',
  }),

  () => agent(`Wf-317H: Update IMPROVEMENT_QUEUE_2026_09_27.md — TODO-138 paper v2 → Tier-2 journal target.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §5.4: TODO-138 paper v2 re-scoped to Tier-2 journal.

TASK:
1. Read TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md
2. Find TODO-138 row in the queue table (row 5)
3. Update paper target: "JCIM" → "Tier-2 journal"
4. Add Wf-316 + Wf-317 entries to ship wave summary
5. Update §0 done-this-session to include Wf-309/310/311/312/313/314/315/316/317
6. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
7. Document: TODO/completed/317h_improvement_queue_tier2_update.md

OUTPUT (PLAIN TEXT): n_rows_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Tier-2 strategy update (parallel)',
    label: 'wf317H',
  }),
])

phase('Synthesis')

const wf317I = await agent(`Wf-317I: Aggregate Wf-317 Tier-2 strategy update outcomes — final INDEX.md sync + verify.

USE Read only.

UPSTREAM RESULTS:
- Wf-317A: TODO/INDEX.md Tier-2 update
- Wf-317B: TODO/README.md Tier-2 update
- Wf-317C: WEAK_TO_STRONG_PLAN.md deprecate wet-lab
- Wf-317D: IMPROVEMENT_ENUM.md R20-P6 → in-silico
- Wf-317E: LAMBDA_ROADMAP.md R22-P1 → computational-only
- Wf-317F: 4 LOI pending files Tier-2 marker
- Wf-317G: 3 wetlab SOP + IC50 pipeline pending files Tier-2 marker
- Wf-317H: IMPROVEMENT_QUEUE.md TODO-138 → Tier-2

TASK:
1. Read all 8 outputs
2. Aggregate: completed + failed + partial
3. Final INDEX.md sync (if Wf-316 not done)
4. Output: TODO/completed/317_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_files_updated + n_deprecations + final_state_verification. Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf317I',
})

return {
  ultracode: 'wf-317-tier2-strategy-update',
  wf317a: wf317A,
  wf317b: wf317B,
  wf317c: wf317C,
  wf317d: wf317D,
  wf317e: wf317E,
  wf317f: wf317F,
  wf317g: wf317G,
  wf317h: wf317H,
  wf317i: wf317I,
}
