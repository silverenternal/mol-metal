export const meta = {
  name: 'wf-324-project-status-refresh',
  description: 'Wf-324: Project status refresh in TODO/ + README.md. (A) README.md comprehensive refresh — project status + Wf-309..323 ship waves + Tier-2 strategy + GPU pipeline + 15 FAIL pre-flight gates honest disclosure + key MEASURED metrics. (B) TODO/INDEX.md comprehensive refresh — §0 ship wave timeline (Wf-256..323) + §22 detailed table + §22.15..22.16 honest-framing disclosures + R20 plan status + Tier-2 deprecations. (C) TODO/README.md refresh — 407 completed/ + 13 pending/ + Tier-2 markers + Wf-309..323 entries. (D) AGENTS.md refresh — append Wf-322D honest disclosure + Wf-323 fixes + R20/R21 skill ecosystem updates. (E) TODO/decisions.md + TODO/risks.md refresh — current state. (F) TODO/02_metrics/METRICS_TARGETS.md — update with Wf-310A G5 results + Wf-318B cascade delta + Wf-323B v3 schema fix. (G) per_paper_section/section_4_2 + section_4_6 + section_5 ablation refresh (post Wf-323). (H) Synthesis + verify pre-flight + final INDEX.md check.',
  phases: [
    { title: 'Project status refresh (parallel)', detail: '7 parallel agents refreshing all status docs' },
    { title: 'Synthesis', detail: 'Final INDEX.md check + pre-flight verification' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('Project status refresh (parallel)')

const [wf324A, wf324B, wf324C, wf324D, wf324E, wf324F, wf324G] = await parallel([

  () => agent(`Wf-324A: README.md comprehensive refresh.

USE Bash + Read + Edit.

CONTEXT: Per Wf-322D: README.md updated to 342 lines. Need final comprehensive refresh with all Wf-309..323 ship waves + Tier-2 strategy + GPU pipeline + 15 FAIL pre-flight gates disclosure.

TASK:
1. Read README.md (current 342 lines)
2. Read molmetal/AGENTS.md §5.4 (Tier-2 strategy)
3. Read TODO/completed/ directory — count files + recent closeouts
4. Update README.md sections:
   - Project status (Tier-2 + computational-only)
   - Recent ship wave (Wf-256..323 timeline)
   - GPU pipeline status (G6 + frontier Wave A + 6-queue)
   - Pre-flight v3 status (HONEST-NEGATIVE REGRESSION disclosure)
   - Key MEASURED metrics (G5 0.7227, NCI-60 0.5282, Chemprop v2 LOMO 0.9015, etc.)
   - TODO directory status (407 completed/ + 13 pending/)
   - Skill ecosystem (28 skills)
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/324a_readme_refresh.md

Honest framing:
- Pure docs update
- 15 FAIL pre-flight gates disclosed honestly
- MEASURED numbers only

OUTPUT (PLAIN TEXT): n_sections_updated + n_lines_added + preflight_status + per_section_summary. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324A',
  }),

  () => agent(`Wf-324B: TODO/INDEX.md comprehensive refresh.

USE Bash + Read + Edit.

CONTEXT: Per Wf-322D: INDEX.md updated to 1955 lines. Need final comprehensive refresh with Wf-309..323 ship waves + Tier-2 strategy + R20 plan status.

TASK:
1. Read TODO/INDEX.md (current 1955 lines)
2. Update §0 (ship wave timeline) with Wf-309..323 entries
3. Update §22 (detailed table) with per-Wf status
4. Update §22.16 (honest-framing disclaimer) with Wf-323 fixes
5. Add new §22.17: Wf-323 fixes (M14 toggle re-integrated, v3 schema drift fixed, p1_chemprop_smoke flipped FAIL to PASS, Vina Pt-reject discovered)
6. Update §18 (paper target) — Tier-2 journal (re-scoped from JCIM per AGENTS.md §5.4)
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/324b_index_refresh.md

OUTPUT (PLAIN TEXT): n_sections_updated + n_lines_added + preflight_status + per_section_summary. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324B',
  }),

  () => agent(`Wf-324C: TODO/README.md refresh — directory overview.

USE Bash + Read + Edit.

TASK:
1. Read TODO/README.md (current 342 lines)
2. Update directory status: 407 completed/ + 13 pending/ + 8 archive/
3. Add Tier-2 markers section
4. Add Wf-309..323 ship wave entries
5. Add skill ecosystem status (28 skills)
6. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
7. Document: TODO/completed/324c_readme_todo_refresh.md

OUTPUT (PLAIN TEXT): n_lines_added + per_section_summary + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324C',
  }),

  () => agent(`Wf-324D: AGENTS.md refresh — append Wf-322D honest disclosure + Wf-323 fixes + R20/R21 skill ecosystem updates.

USE Bash + Read + Edit.

TASK:
1. Read AGENTS.md
2. Append Wf-322D pre-flight REGRESSION disclosure to §5
3. Append Wf-323 fixes:
   - M14 --lr-warmup-steps re-integrated
   - v3 schema drift fix
   - p1_chemprop_smoke gate flipped FAIL to PASS
   - Vina Pt-reject HONEST_NEGATIVE
4. Update §3.3 skill catalog with 28 skills (24 existing + 4 NEW STUBs per Wf-321B)
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/324d_agents_refresh.md

OUTPUT (PLAIN TEXT): n_sections_appended + n_skills_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324D',
  }),

  () => agent(`Wf-324E: TODO/decisions.md + TODO/risks.md refresh.

USE Bash + Read + Edit.

TASK:
1. Read TODO/decisions.md (current state)
2. Read TODO/risks.md (current state)
3. Update with Wf-309..323 decisions:
   - Decision: Tier-2 journal target (downgraded from JCIM)
   - Decision: NO wet-lab planning
   - Decision: GPU standing grant (no per-launch OK)
   - Decision: Write pipelines > poll GPU
4. Update risks.md with Wf-322D honest disclosure:
   - Risk: 15 FAIL pre-flight gates
   - Risk: Wf-323 fixes (M14 + v3 + Vina Pt)
   - Risk: Vina 1.2.7 Pt-reject blocker
5. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/324e_decisions_risks_refresh.md

OUTPUT (PLAIN TEXT): n_decisions_added + n_risks_updated + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324E',
  }),

  () => agent(`Wf-324F: TODO/02_metrics/METRICS_TARGETS.md — update with Wf-310A G5 + Wf-318B + Wf-323B.

USE Bash + Read + Edit.

TASK:
1. Read TODO/02_metrics/METRICS_TARGETS.md
2. Update Table 2 with Wf-310A G5 results (5-metal grand aggregate 0.7227 ± 0.0417)
3. Update cascade delta section with Wf-318B 5-metal table (Ru +0.030, Ir +0.027, Os +0.025 MEASURED lifts; Rh -0.043, Re -0.037 HONEST_NEGATIVE)
4. Update v3 schema fix note (per Wf-323B)
5. Update honest framing with Tier-2 strategy
6. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
7. Document: TODO/completed/324f_metrics_targets_refresh.md

OUTPUT (PLAIN TEXT): n_metrics_updated + per_metric_value + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324F',
  }),

  () => agent(`Wf-324G: per_paper_section/section_4_2 + section_4_6 + section_5 ablation refresh.

USE Bash + Read + Edit.

TASK:
1. Read metrics/by_paper_section/section_4_2_main_results.json
2. Read metrics/by_paper_section/section_4_6_pb_panel.json
3. Read metrics/by_paper_section/section_5_ablation.json
4. Update section_4_2 with Wf-310A G5 5-metal aggregate + Wf-318B cascade delta
5. Update section_4_6 with Wf-319B UFF-min Pt-pool + Wf-323C m8_full HONEST_NEGATIVE
6. Update section_5 with Wf-319A + Wf-323D ablation table integration
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/324g_per_paper_section_refresh.md

OUTPUT (PLAIN TEXT): n_metrics_updated + per_section_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Project status refresh (parallel)',
    label: 'wf324G',
  }),
])

phase('Synthesis')

const wf324H = await agent(`Wf-324H: Aggregate Wf-324 project status refresh — final INDEX.md check + pre-flight verification.

USE Read only.

UPSTREAM RESULTS:
- Wf-324A: README.md refresh
- Wf-324B: TODO/INDEX.md comprehensive refresh
- Wf-324C: TODO/README.md refresh
- Wf-324D: AGENTS.md refresh
- Wf-324E: decisions.md + risks.md refresh
- Wf-324F: METRICS_TARGETS.md refresh
- Wf-324G: per_paper_section refresh

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial
3. Final INDEX.md sync verification
4. Output: TODO/completed/324_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + per_section_status + final_project_status_summary (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf324H',
})

return {
  ultracode: 'wf-324-project-status-refresh',
  wf324a: wf324A,
  wf324b: wf324B,
  wf324c: wf324C,
  wf324d: wf324D,
  wf324e: wf324E,
  wf324f: wf324F,
  wf324g: wf324G,
  wf324h: wf324H,
}