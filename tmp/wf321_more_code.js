export const meta = {
  name: 'wf-321-more-code-class-tasks',
  description: 'Wf-321: More code-class actionable-now tasks (NOT covered by Wf-313/315/318/319/320 + GPU pipeline). (A) TODO-176 m19 LOMO Rh oversample Phase 2 CPU scaffold — post-Wf-309A Phase 1 (cohort ship), CPU scaffold for GPU Phase 2 v2 driver verification. (B) TODO-99 R20 skill ecosystem audit — verify all 24 existing skills + identify 8 NEW skills per LAMBDA_ROADMAP §1.5 needed for R20-P1..P6 + create skill stubs. (C) Integration test suite — test_e2e_cross_module_smoke.py covering 24-skill × 5-pipeline × cross-domain. (D) Pre-flight gate consolidation audit — dedupe overlapping gates + verify all 45+ gates are MEASURED-only. (E) MECHANISM_METRIC_MAPPING §6 ablation_runner integration test — verify ablation_runner.py mirrors §5 ablation table. (F) Chemprop v2 status 2026-09-27 reconciliation — chemprop_v2_5metal_5seed_50epoch.json + metal_lomo_chemprop_v2.json gap analysis. (G) Skill doc freshness audit — 24 skills × date check + content staleness check. (H) TODO-119 P1.4 chemprop v3 LOMO CPU scaffold (pre-GPU verification). (I) Synthesis + INDEX.md update.',
  phases: [
    { title: 'More code-class tasks (parallel)', detail: '8 parallel agents covering CPU actionable work NOT in prior Wf-313/315/318/319/320' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('More code-class tasks (parallel)')

const [wf321A, wf321B, wf321C, wf321D, wf321E, wf321F, wf321G, wf321H] = await parallel([

  () => agent(`Wf-321A: TODO-176 m19 LOMO Rh oversample Phase 2 CPU scaffold — post-Wf-309A Phase 1.

USE Bash + Read + Write + Edit.

CONTEXT: Per IMPROVEMENT_ENUM §1.4: Wf-135 Phase 2 = 5-metal × 5-seed × 50-epoch re-bench with balanced Rh cohort. Per Wf-309A: Phase 1 cohort SHIPPED (250 active + 250 inactive, 15/15 tests PASS). Phase 2 is GPU-1 (~2 weeks wall) — IN GPU PIPELINE.

Per user feedback "don't poll GPU", this agent focuses on CPU scaffold that VERIFIES the Phase 2 setup + pre-GPU smoke:

TASK:
1. Read molmetal/scripts/train_metal_lomo_v2.py — current LOMO trainer
2. Verify --balanced-cohort-json flag exists + reads Phase 1 cohort
3. Write CPU smoke test: 1 epoch × 5 metals × 1 seed (vs GPU 5 × 5 × 50)
4. Add pre-flight gate: p3_wf135_phase2_cpu_smoke_ready (verifies cohort + trainer + balanced_cohort_json wiring)
5. Write 12 NEW tests: molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py
   - 3 tests: cohort JSON load + scaffold-split integrity
   - 3 tests: trainer CLI flag wiring
   - 3 tests: pre-flight gate validation
   - 3 tests: GPU portion integration readiness (verify launch wrapper exists)
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py -v 2>&1 | tail -20
7. Run pre-flight: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/321a_wf135_phase2_cpu_scaffold.md

Honest framing:
- Phase 1 (cohort) SHIPPED via Wf-309A
- Phase 2 (5×5×50) is in pipeline (scripts/launch_wf310e_wf135_phase2_gpu.sh)
- This is CPU scaffold VERIFICATION only (no GPU work)

OUTPUT (PLAIN TEXT): n_tests_pass + scaffold_status + preflight_status + gpu_portion_phase2_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321A',
  }),

  () => agent(`Wf-321B: TODO-99 R20 skill ecosystem audit — 8 NEW skills per LAMBDA_ROADMAP §1.5.

USE Bash + Read + Write + Edit.

CONTEXT: Per LAMBDA_ROADMAP §1.5: R20-P1..P6 need 4 NEW skills + R21-P1..P6 need 4 NEW skills. Currently 24 skills in .claude/skills/. Need to audit + identify gaps.

TASK:
1. List existing skills: ls /home/hugo/codes/mol-metal/.claude/skills/
2. Read LAMBDA_ROADMAP.md §1.5 — R20 + R21 skill candidates
3. For each candidate skill:
   - Check if already exists
   - If not: write skill STUB at .claude/skills/<skill_name>/SKILL.md
4. Skill stub template:
   - frontmatter (name, description, tools, activate-on keywords)
   - When to use (§1)
   - Core contract (§2)
   - Reference impl pointer (§3)
5. Write 8 NEW skill stubs (R20 + R21):
   - mol-metal-variance-puct-default (already exists - skip)
   - mol-metal-hilbert-astsr-r2 (already exists - skip)
   - mol-metal-multi-ph-pll-feature (already exists - skip)
   - mol-metal-3way-multimetal (NEW stub for R20-P4)
   - mol-metal-closure-dblock-bench (already exists - skip)
   - mol-metal-r18-math (already exists - skip)
   - mol-metal-r20-skill-pack (NEW summary skill)
   - mol-metal-r21-skill-pack (NEW summary skill)
   - mol-metal-wetlab-protocol (already exists - skip)
   - mol-metal-wetlab-bayesian (already exists - skip)
   - mol-metal-mechanism-mapping (already exists - skip)
6. For each NEW skill: write 4 stub tests
7. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/321b_r20_r21_skill_ecosystem_audit.md

OUTPUT (PLAIN TEXT): n_skills_audited + n_new_skill_stubs + n_existing_verified + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321B',
  }),

  () => agent(`Wf-321C: Integration test suite — test_e2e_cross_module_smoke.py covering 24-skill × 5-pipeline × cross-domain.

USE Bash + Read + Write + Edit.

CONTEXT: Per MERGE_GATE.md Gate 4 — end-to-end smoke covering all 4 tracks. Per WEAK_TO_STRONG_PLAN §3: mechanisms M1-M44. Need comprehensive integration test suite.

TASK:
1. Read molmetal/tests/test_cross_module_smoke.py — existing integration test
2. Read molmetal/molmetal_lam/, molmetal/adapters/, molmetal/scripts/ — module inventory
3. Write NEW integration test: molmetal/tests/test_wf321c_e2e_cross_pipeline.py
   - 4 pipeline × 5 test per pipeline = 20 integration tests
   - Pipeline 1: cbr+adapters+metrics (chemprop + cfm + adapters)
   - Pipeline 2: lambda proof_search + catherine closure
   - Pipeline 3: lambda + CFM coupling (TODO-21 + Wf-269C N1 transition)
   - Pipeline 4: click + metal_ligand_exchange + aqua (84-rule matrix)
4. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf321c_e2e_cross_pipeline.py -v 2>&1 | tail -30
5. Add pre-flight gate: p3_e2e_cross_pipeline_tests_present (verifies 20 tests PASS)
6. Update pre_flight_check.py gate counter
7. Document: TODO/completed/321c_e2e_cross_pipeline.md

OUTPUT (PLAIN TEXT): n_tests_pass + per_pipeline_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321C',
  }),

  () => agent(`Wf-321D: Pre-flight gate consolidation audit — dedupe overlapping gates + verify all 45+ gates are MEASURED-only.

USE Bash + Read + Edit.

CONTEXT: Per Wf-314A + Wf-315A + Wf-312C + Wf-313 + Wf-318 etc: pre-flight gate count keeps growing. Need audit for dedupe + MEASURED-only.

TASK:
1. Read molmetal/scripts/pre_flight_check.py — list all gates
2. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py --list-gates 2>&1 | tail -50
3. Identify gates with overlapping coverage (e.g., p3_section9_honest_framing + p3_admet_honest_negative both check similar things)
4. For each overlap: either DEDUPE or DOCUMENT why both are needed
5. Verify all gates are MEASURED-only (no PROJECTION or ASPIRATIONAL gates)
6. Add new gate: p3_gate_count_consistency_audit (verifies gate list size matches documentation)
7. Update pre_flight_check.py gate counter if needed
8. Document: TODO/completed/321d_preflight_gate_audit.md

OUTPUT (PLAIN TEXT): n_gates_total + n_overlaps_found + n_deduped + n_documented + n_measured_only + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321D',
  }),

  () => agent(`Wf-321E: MECHANISM_METRIC_MAPPING §6 ablation_runner integration test — verify ablation_runner.py mirrors §5 ablation table.

USE Bash + Read + Edit.

CONTEXT: Per MECHANISM_METRIC_MAPPING §5: 11 R17 ablations declared. Per Wf-321B: ablation_runner.py is the script that executes these. Need to verify the script mirrors the table.

TASK:
1. Read TODO/02_metrics/MECHANISM_METRIC_MAPPING.md §5 — ablation table
2. Read molmetal/scripts/ablation_runner.py — current ablation execution
3. Cross-check: are all 11 ablations in §5 also in ablation_runner.py?
4. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "from molmetal.scripts.ablation_runner import ABLATION_REGISTRY; print(len(ABLATION_REGISTRY))" 2>&1
5. Document: TODO/completed/321e_ablation_runner_integration_test.md
6. Add pre-flight gate: p3_ablation_runner_registry_complete (verifies 11 ablations)

OUTPUT (PLAIN TEXT): n_ablations_in_table + n_ablations_in_script + consistency_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321E',
  }),

  () => agent(`Wf-321F: Chemprop v2 status 2026-09-27 reconciliation — chemprop_v2_5metal_5seed_50epoch.json + metal_lomo_chemprop_v2.json gap analysis.

USE Bash + Read + Edit.

CONTEXT: Per chemprop_v2_status_2026_09_27.json: chemprop_v2_5metal_5seed_50epoch.json + metal_lomo_chemprop_v2.json do NOT exist (per the honest_framing_note: refused to fabricate). Per Wf-320D: TODO-155 §3 chemprop v2 featurization fix in progress.

TASK:
1. Read metrics/by_metric/chemprop_v2_status_2026_09_27.json — current status
2. Document the gap: chemprop v2 full 5x5x50 + chemprop v2 LOMO 5x5x50 + chemprop v2 status reconciliation
3. Update metrics/by_metric/chemprop_v2_status_2026_09_27.json with Wf-320D progress note
4. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/321f_chemprop_v2_status_reconciliation.md

Honest framing:
- Chemprop v2 numbers CANNOT be fabricated
- Wf-320D featurization fix is prerequisite
- After Wf-320D ships, chemprop v2 numbers can be MEASURED (not fabricated)

OUTPUT (PLAIN TEXT): n_gaps_documented + chemprop_v2_status + preflight_status + recovery_path. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321F',
  }),

  () => agent(`Wf-321G: Skill doc freshness audit — 24 skills × date check + content staleness check.

USE Bash + Read + Edit.

CONTEXT: Per AGENTS.md §3.3: 24 skills exist. Skills may have stale content (referring to old workflow IDs, old TODO numbers, etc.). Need audit + refresh.

TASK:
1. List all 24 skills: ls /home/hugo/codes/mol-metal/.claude/skills/
2. For each skill:
   - Check mtime (last modified date)
   - Check content for stale references (old TODO-XXX, old Wf-XXX)
3. Categorize: FRESH (< 7 days) / RECENT (7-30 days) / STALE (> 30 days)
4. For STALE skills: write 2-line refresh note in skill description
5. Document: TODO/completed/321g_skill_doc_freshness_audit.md

OUTPUT (PLAIN TEXT): n_skills_total + per_category_count + per_skill_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321G',
  }),

  () => agent(`Wf-321H: TODO-119 P1.4 chemprop v3 LOMO CPU scaffold (pre-GPU verification).

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG §3.4: TODO-119 P1.4 chemprop v3 LOMO is 4-wave sweep (4 ablations × 1 seed × 3 ep smoke then 5×5×50 full). Currently in GPU pipeline (Wf-300C Wave-A). Need CPU scaffold verification.

TASK:
1. Read molmetal/scripts/train_chemprop_v2_v3.py — current v3 trainer
2. Verify --chemprop-version {v2,v3} flag exists
3. Write CPU smoke: 1-epoch × 5-metal × 1-seed smoke
4. Add pre-flight gate: p3_todo_119_p14_cpu_smoke_ready
5. Write 12 NEW tests: molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py
   - 3 tests: chemprop v2 vs v3 flag correctness
   - 3 tests: LOMO scaffold-split integrity
   - 3 tests: pre-flight gate validation
   - 3 tests: GPU portion integration readiness (launch wrapper exists)
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py -v 2>&1 | tail -20
7. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/321h_todo_119_p14_cpu_scaffold.md

Honest framing:
- P1.4 full sweep is in GPU pipeline (Wf-300C Wave-A)
- This is CPU scaffold VERIFICATION (no GPU work)
- Chemprop v3 LOMO needs chemprop v2 featurization fix (Wf-320D) first

OUTPUT (PLAIN TEXT): n_tests_pass + scaffold_status + preflight_status + gpu_portion_status. Do NOT call StructuredOutput.`, {
    phase: 'More code-class tasks (parallel)',
    label: 'wf321H',
  }),
])

phase('Synthesis')

const wf321I = await agent(`Wf-321I: Aggregate Wf-321 more code-class tasks outcomes — INDEX.md final sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-321A: TODO-176 m19 LOMO Phase 2 CPU scaffold
- Wf-321B: TODO-99 R20 skill ecosystem audit
- Wf-321C: Integration test suite
- Wf-321D: Pre-flight gate consolidation audit
- Wf-321E: MECHANISM_METRIC_MAPPING §6 ablation_runner integration
- Wf-321F: Chemprop v2 status reconciliation
- Wf-321G: Skill doc freshness audit
- Wf-321H: TODO-119 P1.4 chemprop v3 LOMO CPU scaffold

TASK:
1. Read all 8 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
4. Output: TODO/completed/321_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf321I',
})

return {
  ultracode: 'wf-321-more-code-class-tasks',
  wf321a: wf321A,
  wf321b: wf321B,
  wf321c: wf321C,
  wf321d: wf321D,
  wf321e: wf321E,
  wf321f: wf321F,
  wf321g: wf321G,
  wf321h: wf321H,
  wf321i: wf321I,
}
