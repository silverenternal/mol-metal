export const meta = {
  name: 'wf-312-wf308-followup',
  description: 'Wf-312: Wf-308 real measurement follow-up + honest-negative surfaces. (A) Re degenerate n_test=1 fix + NCI-60 9-cell paper-grade multi-seed multi-epoch CPU smoke (HCT116 0.6829 MEASURED; PC3 0.2153 honest-negative). (B) NCI-60 9-cell ship to paper §4 + paper_dashboard + pre_flight v3 + by_paper_section. (C) ADMET extension — aqueous_solubility_logS -7.612 honest-negative + extend ADMET to 5 additional fields (Caco-2 / BBB / clearance / half-life / VDss) via metrics_v2. (D) §9 honest framing catalog 30 → 35+ entries — add Re degenerate / PC3 / aq_sol / hepatotox / logp7_4 / m6_kappa NEW honest-negatives + Vina covalent architectural ceiling. (E) Wf-308D verify + 4 LOI send-ready final integration + G5 retrain integration. (F) Synthesis.',
  phases: [
    { title: 'Wf-308 real follow-up (parallel)', detail: '5 parallel: Re fix + NCI-60 grade, NCI-60 ship, ADMET extension, §9 expansion, Wf-308D verify' },
    { title: 'Synthesis', detail: 'Aggregate + paper §4/§6 final state + honest-negative catalog' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Wf-308 real follow-up (parallel)')

const [wf312A, wf312B, wf312C, wf312D, wf312E] = await parallel([

  () => agent(`Wf-312A: Re degenerate n_test=1 fix + NCI-60 9-cell paper-grade multi-seed multi-epoch CPU smoke.

USE Bash + Read + Edit.

CONTEXT: Per Wf-308C temporal_split execute: Re seed=0 test_auc=0.5175 but n_test=1 (degenerate). Per Wf-308C NCI-60 9-cell 27 runs at 1 epoch CPU; HCT116 0.6829 (highlight) + PC3 0.2153 (honest-negative). Need to:
- Investigate Re cohort size and fix degenerate issue
- Re-run NCI-60 with 3 seeds × 5 epochs (paper-grade CPU smoke)

TASK PART 1 — Re degenerate fix:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — Re cohort size
   grep -c ",Re," /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
2. Read molmetal/scripts/wf219_temporal_split_driver.py — temporal split logic
3. Investigate: Re cohort size, year distribution, scaffold count
4. If Re cohort < 50 rows: document + use scaffold split (not temporal) for Re
5. Update driver with fallback: --split {temporal,scaffold} + Re scaffold fallback
6. Add test: test_wf312a_re_non_degenerate

TASK PART 2 — NCI-60 9-cell paper-grade multi-seed multi-epoch CPU smoke:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — current --execute path
2. Re-run with 3 seeds × 5 epochs (vs 3 seeds × 1 epoch):
   ${PY} molmetal/scripts/nci60_9cell_chemprop_driver.py --csv /home/hugo/codes/mol-metal/molmetal/data/nci60_precious_9cell.csv --seeds 0 1 2 --epochs 5 --hidden 128 --depth 3 --device cpu --out /tmp/wf312a_nci60_grade --execute 2>&1 | tail -30
3. Verify: per-cell mean ± std AUC across 3 seeds
4. Document: per-cell median + std + verdict (PASS / BELOW_GATE / HONEST_NEGATIVE)
5. Save aggregate: molmetal/metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json (NEW)
6. Update paper_dashboard with NCI-60 9-cell claim cards (per-cell breakdown)

Honest framing per party-twenty-spirit §3.5:
- Re degenerate is real cohort-size limitation
- NCI-60 5 epoch still CPU; full paper-grade needs GPU per kill_then_adapt
- PC3 0.2153 honest-negative kept visible

OUTPUT (PLAIN TEXT): re_cohort_size + re_split_strategy + nci60_per_cell_mean_std + nci60_3seed_mean + paper_dashboard_claims + per_cell_verdict. Do NOT call StructuredOutput.`, {
    phase: 'Wf-308 real follow-up (parallel)',
    label: 'wf312A',
  }),

  () => agent(`Wf-312B: NCI-60 9-cell ship to paper §4 + paper_dashboard + pre_flight v3 + by_paper_section.

USE Bash + Read + Edit.

CONTEXT: Per Wf-308C: NCI-60 9-cell MEASURED 27 runs CPU 1 epoch. HCT116 0.6829 (highlight) + PC3 0.2153 (honest-negative). Not yet integrated into paper §4 main results or by_paper_section/. Need ship surface.

TASK:
1. Read molmetal/reports/wf308c_nci60_smoke/ — per-cell results
2. Read metrics/by_metric/nci60_precious_9cell_auc.json — current 0.81 n=3
3. Aggregate per-cell mean ± std + overall mean
4. Update metrics/by_metric/nci60_precious_9cell_auc.json with:
   - per_cell_breakdown (9 cells × mean/std)
   - highlight: HCT116 0.6829
   - honest_negative: PC3 0.2153
   - overall_mean: 0.5282
5. Update molmetal/scripts/paper_dashboard.py with claim cards:
   - claim_nci60_9cell_hct116_highlight (BADGE_MEASURED, 0.6829)
   - claim_nci60_9cell_pc3_honest_negative (BADGE_HONEST_NEGATIVE, 0.2153)
   - claim_nci60_9cell_overall_mean (BADGE_MEASURED, 0.5282)
6. Update molmetal/scripts/pre_flight_check.py with new gate:
   - p3_nci60_9cell_per_cell_aggregate (verify 9 entries + overall mean)
7. Update metrics/by_paper_section/section_4_2_main_results.json:
   - Add NCI-60 9-cell cohort with per-cell breakdown
   - Highlight HCT116 + honest-negative PC3
8. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Verify dashboard: ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v312b.html 2>&1 | tail -10
10. Document: TODO/completed/nci60_9cell_paper4_ship.md

Honest framing:
- 27 runs CPU smoke is real but limited
- HCT116 highlight real; PC3 honest-negative real
- Full paper-grade NCI-60 needs GPU per kill_then_adapt

OUTPUT (PLAIN TEXT): nci60_aggregate_status + nci60_paper_dashboard_claims + nci60_pre_flight_gate + nci60_section_4_2_status + per_cell_highlight + per_cell_honest_negative. Do NOT call StructuredOutput.`, {
    phase: 'Wf-308 real follow-up (parallel)',
    label: 'wf312B',
  }),

  () => agent(`Wf-312C: ADMET extension — ship honest-negative + extend to 5 additional ADMET fields via metrics_v2.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-308B + Wf-303C: aqueous_solubility_logS -7.612 FAIL honest-negative already shipped (gate #36). Per admet_pass_through.py: 449 mols computed. Existing ADMET coverage: ames + cell_perm + aq_sol + hepatotox + herg + plasma_protein_binding + logp7_4 (7 fields). Need:
- Extend to 5 additional ADMET fields via metrics_v2 (Caco-2 / BBB / clearance / half-life / VDss)
- Ship hepatotox 0.548 + logp7_4 6.413 + m6_kappa 0.472 as honest-negative (currently NOT in pre_flight)
- Expand to comprehensive ADMET coverage

TASK PART 1 — extend ADMET to 5 additional fields:
1. Read molmetal/scripts/admet_pass_through.py — current ADMET computation
2. Read molmetal/molmetal_lam/sbdd_env/metrics_v2.py — available ADMET functions
3. Add NEW ADMET metrics: Caco-2_perm / BBB / clearance / half_life / VDss
4. Re-run on 449 metallo-drug SMILES
5. Save 5 NEW JSON: metrics/by_metric/{caco2_perm,bbb,clearance,half_life,vdss}.json

TASK PART 2 — ship honest-negative ADMET gates:
1. Read metrics/by_metric/{hepatotox_index,logp7_4,m6_kappa}.json — current values
2. Add NEW pre_flight v3 gates:
   - p3_admet_hepatotox_honest_negative (hepatotox 0.548 > 0.5 = HONEST_NEGATIVE)
   - p3_admet_logp7_4_honest_negative (logp7_4 6.413 out of 0-5 range = HONEST_NEGATIVE)
   - p3_admet_m6_kappa_honest_negative (m6_kappa 0.472 < 0.6 = HONEST_NEGATIVE)
3. Update pre_flight check gate count: 36 → 39 (if all added)
4. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 3 — ADMET aggregate ship:
1. Aggregate all ADMET metrics: metrics/by_metric/admet_full_coverage.json
2. Update paper_dashboard with NEW ADMET claim cards:
   - 7 existing + 5 NEW = 12 ADMET claims
3. Update by_paper_section/section_5_ablation.json with ADMET panel

Honest framing:
- hepatotox 0.548 / logp7_4 6.413 / m6_kappa 0.472 — all real, all BELOW threshold
- 5 NEW ADMET fields are projections from RDKit descriptors (not wet-lab)
- Druglikeness honest-negative cluster: 4/12 ADMET metrics are BELOW threshold

OUTPUT (PLAIN TEXT): n_admet_metrics_extended + n_admet_metrics_total + n_admet_honest_negative_gates + preflight_status + n_dashboard_admet_claims. Do NOT call StructuredOutput.`, {
    phase: 'Wf-308 real follow-up (parallel)',
    label: 'wf312C',
  }),

  () => agent(`Wf-312D: §9 honest framing catalog 30 → 35+ entries — add Wf-308 NEW honest-negatives.

USE Bash + Read + Edit.

CONTEXT: Per Wf-306C: §9 honest framing catalog 25 entries. Per Wf-311A in-flight: 25 → 30+ entries (add G5 50ep / Vina covalent / R² ceiling / m19 LOMO / DrugOOD BELOW_RANDOM). Need additional entries from Wf-308:
- Re degenerate n_test=1 (Wf-312A)
- NCI-60 PC3 0.2153 honest-negative (Wf-308C)
- aqueous_solubility_logS -7.612 (Wf-308B)
- hepatotox 0.548 slightly above 0.5 (Wf-312C)
- logp7_4 6.413 out of 0-5 range (Wf-312C)
- m6_kappa 0.472 BELOW 0.6 (Wf-312C)
- Vina covalent mechanism architectural ceiling 0/10 ≥-7 (Wf-143 + Wf-301B)
- Wf-308 §3.5 violations fix (per Wf-311A in flight)

TASK:
1. Read TODO/completed/207_unified_pre_flight_checklist_v2.md §9 — current 25 entries (Wf-306C) → 30+ (Wf-311A in-flight)
2. Add NEW entries:
   - §9.AV Wf-308 Re degenerate n_test=1 (per Wf-312A)
   - §9.AW NCI-60 PC3 0.2153 honest-negative (per Wf-308C)
   - §9.AX Wf-308B ADMET aqueous_solubility_logS -7.612 FAIL
   - §9.AY Wf-312C hepatotox 0.548 slightly above 0.5
   - §9.AZ Wf-312C logp7_4 6.413 out of 0-5
   - §9.BA Wf-312C m6_kappa 0.472 BELOW 0.6
   - §9.BB Wf-143 + Wf-301B Vina covalent ceiling 0/10 ≥-7 (per architectural ceiling discovery)
3. Update entry count: 30+ → 35+ entries
4. Verify pre_flight §9 entry count gate: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
5. Document: TODO/completed/312_honest_framing_35plus.md

Honest framing:
- Each new entry is a real honest-negative, not a fabrication
- 4 ADMET honest-negatives cluster around 449-mol metallo-drug subset
- Vina covalent ceiling is a real architectural finding

OUTPUT (PLAIN TEXT): n_new_section9_entries + n_total_section9_entries + per_entry_summary + preflight_status + per_entry_source_citation. Do NOT call StructuredOutput.`, {
    phase: 'Wf-308 real follow-up (parallel)',
    label: 'wf312D',
  }),

  () => agent(`Wf-312E: Wf-308D verify + 4 LOI send-ready final integration + G5 retrain integration (wait for Wf-308D outputs).

USE Bash + Read + Edit.

CONTEXT: Wf-308D truncated in summary — 4 LOI consolidation + Wf-219 §4.3 temporal-split actual CPU smoke + G5 retrain integration template. Need to verify Wf-308D outputs are complete + ship.

TASK PART 1 — Wf-308D verify:
1. Check Wf-308D outputs:
   - molmetal/reports/wf308d_wf219_temporal/ — actual CPU smoke result
   - molmetal/reports/wf308d_loi_review/wetlab_loi_consolidation_review.md — 4 LOI summary
   - molmetal/reports/wf308d_g5_integration_template/integration_plan.md — G5 template
2. If any output MISSING: re-run Wf-308D parts
3. Document any blockers

TASK PART 2 — 4 LOI send-ready final integration:
1. Read all 4 LOIs + wetlab_loi_consolidation_review.md
2. Verify each LOI is ready-to-send per checklist
3. Update TODO/pending/wetlab_loi_consolidation_review.md with current status:
   - LOI-ORNL: DRAFT + checklist items per Wf-308D
   - LOI-PKU: DRAFT + checklist
   - LOI-Xiehe: DRAFT + checklist
   - LOI-RUSANOV: DRAFT + checklist (per Wf-305A)
4. Add pre_flight gate:
   - p3_loi_send_ready (verify 4 LOI + checklist complete)
5. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 3 — G5 retrain integration template:
1. Read molmetal/reports/wf308d_g5_integration_template/integration_plan.md (or re-write if missing)
2. Verify G5 retrain integration script: molmetal/scripts/wf308d_g5_results_integration.py
3. Update integration template to handle:
   - G5 13/15 pairs MEASURED (per Wf-310A aggregation)
   - G5 2 missing pairs retry (per Wf-310A)
   - G5 cascade with G4 Ru seed=0 baseline
4. Document: TODO/completed/308d_g5_integration_final.md

Honest framing:
- 4 LOI ready-to-send subject to Hugo review + actual send
- G5 retrain integration waits for G5 results final aggregation
- This is INTEGRATION work, not new measurement

OUTPUT (PLAIN TEXT): wf308d_verify_status + n_lois_ready_to_send + g5_integration_status + preflight_status + per_loi_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-308 real follow-up (parallel)',
    label: 'wf312E',
  }),
])

phase('Synthesis')

const wf312F = await agent(`Wf-312F: Aggregate Wf-312 Wf-308 follow-up outcomes — paper §4/§6 final state + honest-negative catalog.

USE Read only.

UPSTREAM RESULTS:
- Wf-312A: Re degenerate fix + NCI-60 9-cell paper-grade multi-seed multi-epoch
- Wf-312B: NCI-60 9-cell ship to paper §4 + dashboard + pre_flight + by_paper_section
- Wf-312C: ADMET extension to 12 fields + 3 NEW honest-negative gates
- Wf-312D: §9 honest framing 30 → 35+ entries
- Wf-312E: Wf-308D verify + 4 LOI + G5 integration final

TASK:
1. Read all 5 outputs
2. Aggregate: completed + failed + partial
3. Build paper §4/§6 final state + honest-negative catalog summary
4. Output to molmetal/reports/wf312_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + paper_4_6_final_state + honest_negative_catalog + n_metrics_shipped + remaining_blockers (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf312F',
})

return {
  ultracode: 'wf-312-wf308-followup',
  wf312a: wf312A,
  wf312b: wf312B,
  wf312c: wf312C,
  wf312d: wf312D,
  wf312e: wf312E,
  wf312f: wf312F,
}