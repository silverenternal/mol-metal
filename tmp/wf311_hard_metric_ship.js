export const meta = {
  name: 'wf-311-hard-metric-ship',
  description: 'Wf-311: Hard metric ship surface + paper §4/§6 integrity (NOT covered by Wf-308/309/310). (A) Paper §4 5 Wf-264 §3.5 violations fix + §9 honest framing catalog 25 → 30+ entries (add G5 50ep real results + Vina pt-strip + drugood smoke + Vina covalent ceiling + R² architectural ceiling + 4 m6/rgba μ fields). (B) Hard metric paper-grade aggregation — vina_triple_success × 5 pockets + docked_pose_rmsd 0.6578 + chemprop_v2_lomo_v2 0.9015 + nci60_9cell 0.81 integrate into paper §4 + by_paper_section refresh. (C) Paper §4 + §6 rewrite per TODO-144 §1 — 7/9 lead claims MEASURED + 1/9 DEFERRED + 1/9 DEFERRED_GPU status emit. (D) OOD reachability + m18 protein_metal + gi50_proxy + pharmacophore + metal_coord_compliance Phase-3G completion ship surface. (E) Diversity_subpocket + steric_clash + com_shift + m6_kappa 0.472 + hepatotox 0.548 + logp7_4 6.41 ship honest-negative. (F) Synthesis.',
  phases: [
    { title: 'Hard metric ship surface (parallel)', detail: '5 parallel: paper violations fix, paper aggregation, paper rewrite, OOD/m18/gi50 ship, diversity/steric ship' },
    { title: 'Synthesis', detail: 'Aggregate + remaining blockers + paper §4 final state' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Hard metric ship surface (parallel)')

const [wf311A, wf311B, wf311C, wf311D, wf311E] = await parallel([

  () => agent(`Wf-311A: Paper §4 5 Wf-264 §3.5 violations fix + §9 honest framing catalog 25 → 30+ entries.

USE Bash + Read + Edit.

CONTEXT: Per Wf-264 audit: paper §4 has 5 §3.5 honest-framing violations (broken source-pointer citations in §9 + 1 seed-count inconsistency + 1 numeric-transcription error + 1 partial-disclosure). Per TODO-207 §9 catalog: 25 entries (Wf-263B expansion to 9 + Wf-306C to 25). Need to:
- Fix 5 violations in paper §4
- Expand §9 catalog 25 → 30+ with NEW MEASURED findings

TASK PART 1 — paper §4 5 violations fix:
1. Read TODO/completed/188_paper_4_krasnov_protocol_aligned.md — paper §4 main doc
2. Read molmetal/reports/wf264_audit/paper_4_claim_audit.md — 5 violations list
3. For each violation, fix:
   - Violation 1: broken source-pointer citation (locate + correct path)
   - Violation 2: seed-count inconsistency (locate + correct)
   - Violation 3: numeric-transcription error (locate + correct)
   - Violation 4: partial-disclosure (locate + complete)
   - Violation 5: source-pointer (locate + correct)
4. Document each fix in TODO/completed/264_wf264_paper_violations_fix.md

TASK PART 2 — §9 honest framing catalog 25 → 30+ entries:
1. Read TODO/completed/207_unified_pre_flight_checklist_v2.md §9 — current 25 entries (per Wf-306C)
2. Add NEW entries:
   - §9.AR G5 50-epoch full sweep MEASURED (13/15 pairs) per Wf-310A
   - §9.AS Vina covalent mechanism architectural ceiling — 0/10 pockets reach industry ≥−7 (Wf-143 + Wf-301B honest-negative)
   - §9.AT R² architectural ceiling — 3 of 4 (platinai / hilbert / metalcytotox_combined) per Wf-297B
   - §9.AU m19 LOMO Rh fold 0.136 regression (per Wf-121)
   - §9.AV DrugOOD v3 0.4574 BELOW_RANDOM (per Wf-253D + Wf-301A smoke 0.8024)
3. Update pre_flight v3 gate: §9 entry count ≥ 30 (was 25)
4. Update §9 entry count: 25 → 30+ entries
5. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- 5 violations are minor (not in main results)
- NEW §9 entries reflect post-Wf-301/304/306 discoveries

OUTPUT (PLAIN TEXT): n_violations_fixed + per_violation_summary + n_honest_framing_entries + preflight_status + n_new_section9_entries. Do NOT call StructuredOutput.`, {
    phase: 'Hard metric ship surface (parallel)',
    label: 'wf311A',
  }),

  () => agent(`Wf-311B: Hard metric paper-grade aggregation — vina_triple_success × 5 pockets + docked_pose_rmsd 0.6578 + chemprop_v2_lomo_v2 0.9015 + nci60_9cell 0.81 integrate into paper §4.

USE Bash + Read + Write + Edit.

CONTEXT: Many MEASURED hard metrics on disk (per metrics/by_metric/) but NOT yet integrated into paper §4 main results or by_paper_section/ aggregation. Specifically:
- vina_triple_success × 5 pockets (1h36/3eml/3ny8/4aaw/4rlu): MEASURED 5/5
- vina_high_affinity: MEASURED × 5 pockets
- vina_relaxed_success: MEASURED × 5 pockets
- vina_reference: MEASURED × 5 pockets
- docked_pose_rmsd 1h36: 0.6578 MEASURED
- ligand_vs_pocket_rmsd: 3.8333 MEASURED
- rigid_fragment_rmsd_1h36: n=20 MEASURED
- chemprop_v2_lomo_v2: 0.9015 per-cell median
- nci60_precious_9cell_auc: 0.81 n=3
- per_metal_pic50_wasserstein: 4 entries
- multi_task_4domain_v2: paper §4 7/9 lead claims MEASURED

TASK:
1. Read metrics/by_metric/vina_triple_success*.json × 5 pockets — per-pocket data
2. Read metrics/by_metric/vina_high_affinity*.json × 5 pockets — high-affinity
3. Read metrics/by_metric/vina_relaxed_success*.json × 5 pockets — relaxed success
4. Read metrics/by_metric/vina_reference*.json × 5 pockets — reference
5. Read metrics/by_metric/docked_pose_rmsd.json + ligand_vs_pocket_rmsd.json + rigid_fragment_rmsd*.json — RMSD panel
6. Read metrics/by_metric/chemprop_v2_lomo_v2.json — 0.9015 LOMO
7. Read metrics/by_metric/nci60_precious_9cell_auc.json — 0.81
9. Read metrics/by_paper_section/section_4_2_main_results.json — current §4.2 aggregation
10. Write NEW aggregation: molmetal/scripts/wf311b_paper4_hard_metric_aggregate.py
    - Aggregate all Vina × 5 pockets → molmetal/reports/wf311b_vina_5pocket_aggregate.json
    - Aggregate RMSD panel → molmetal/reports/wf311b_rmsd_aggregate.json
    - Aggregate chemprop_v2_lomo_v2 + nci60_precious_9cell_auc → molmetal/reports/wf311b_main_results_aggregate.json
    - Update metrics/by_paper_section/section_4_2_main_results.json with all aggregations
    - Update metrics/by_paper_section/section_4_6_pb_panel.json with PB pass rate
    - Update metrics/by_paper_section/section_5_ablation.json with ablation data
11. Update paper_dashboard.py with claim cards for each aggregated metric
12. Run aggregator: ${PY} molmetal/scripts/wf311b_paper4_hard_metric_aggregate.py 2>&1 | tail -20
13. Verify paper_dashboard: ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v311b.html 2>&1 | tail -10
14. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- These are MEASURED aggregations, not new measurements
- per-pocket Vina stats are smoke partial on test_001 / pKd scores, NOT full real-cohort
- chemprop_v2_lomo_v2 0.9015 is per-cell median — not aggregate mean

OUTPUT (PLAIN TEXT): n_vina_pockets_aggregated + vina_per_pocket_summary + rmsd_aggregate + main_results_aggregate + per_paper_section_refreshed + n_dashboard_claims_added. Do NOT call StructuredOutput.`, {
    phase: 'Hard metric ship surface (parallel)',
    label: 'wf311B',
  }),

  () => agent(`Wf-311C: Paper §4 + §6 rewrite per TODO-144 §1 — 7/9 lead claims MEASURED + 1/9 DEFERRED + 1/9 DEFERRED_GPU status emit.

USE Bash + Read + Edit.

CONTEXT: Per metric multi_task_4domain_v2.json status: "SPRINT_A_FLAGS_AND_COUNTS_FRESH — 6 mechanisms enumerated with status/schedule/verify_gate; 7/9 paper §4 lead claims have MEASURED data; 1/9 DEFERRED (m17 pexivory); 1/9 DEFERRED_GPU (m3 LOMO 5/5 sweep). Paper §4 + §6 rewrite needed per TODO-144 §1."

TASK:
1. Read metrics/by_metric/multi_task_4domain_v2.json — current 6 mechanism status
2. Read TODO/03_plans or relevant paper §4 TODO — TODO-144 §1 spec
3. Read TODO/completed/188_paper_4_krasnov_protocol_aligned.md — current paper §4
4. For each of 9 lead claims, document current state:
   - Claim 1-7 (MEASURED): status + source JSON + field-avg + verdict
   - Claim 8 (DEFERRED m17 pexivory): why deferred + when unblocked
   - Claim 9 (DEFERRED_GPU m3 LOMO 5/5 sweep): why GPU-blocked + wall budget + user OK requirement
5. Update paper §4 main results table with current state of each claim
6. Update paper §6 limitations with the deferred claims as honest-negative
7. Verify by_paper_section/section_4_2_main_results.json + section_5_ablation.json are consistent
8. Document: TODO/completed/144_paper_4_6_rewrite_todo144.md

Honest framing per party-twenty-spirit §3.5:
- 7/9 MEASURED is real; 1/9 DEFERRED + 1/9 DEFERRED_GPU are honest
- §6 limitations must include both deferred claims
- §4 main table can only include 7/9 + honest-negative for 2/9

OUTPUT (PLAIN TEXT): n_lead_claims_measured + n_lead_claims_deferred + n_lead_claims_deferred_gpu + paper_4_table_status + paper_6_limitations_status + n_section_4_2_refreshed. Do NOT call StructuredOutput.`, {
    phase: 'Hard metric ship surface (parallel)',
    label: 'wf311C',
  }),

  () => agent(`Wf-311D: OOD reachability + m18 protein_metal + gi50_proxy + pharmacophore + metal_coord_compliance Phase-3G completion ship surface.

USE Bash + Read + Write + Edit.

CONTEXT: Multiple MEASURED ship surfaces on disk but not fully wired into paper §4 or pre_flight_check.py:
- ood_per_element_reachability 0.4 (MEASURED)
- ood_per_metal_reachability (n=30)
- metric18_protein_metal + _per_residue + _v2 (4 files)
- gi50_proxy n=5000
- pharmacophore_pass_rate 0.7 (n=10 curated smoke, needs production pool)
- metal_coord_compliance (DESIGN+1-PILOT, needs Phase-3G completion + 10x3 rotation)

TASK PART 1 — OOD reachability ship:
1. Read metrics/by_metric/ood_per_element_reachability.json + ood_per_metal_reachability.json
2. Aggregate to single metric: ood_reachability_summary.json
3. Update paper_dashboard with claim cards

TASK PART 2 — m18 protein_metal ship:
1. Read metrics/by_metric/metric18_protein_metal*.json × 4 files
2. Aggregate to: protein_metal_per_residue_summary.json
3. Update paper_dashboard

TASK PART 3 — gi50_proxy ship:
1. Read metrics/by_metric/gi50_proxy.json (n=5000)
2. Verify aggregation already exists; if not, aggregate
3. Update paper_dashboard

TASK PART 4 — pharmacophore_pass_rate expand to production:
1. Read metrics/by_metric/pharmacophore_pass_rate.json (n=10 curated smoke)
2. Expand to production pool: 449 mols from admet_pass_through CSV
3. Re-measure pharmacophore_pass_rate on full pool
4. Save: metrics/by_metric/pharmacophore_pass_rate_prod.json
5. Update paper_dashboard

TASK PART 5 — metal_coord_compliance Phase-3G completion:
1. Read metrics/by_metric/metal_coord_compliance.json — current 1.0 on singleton
2. Read TODO/completed/290 metal_ligand_exchange_matrix (if relevant)
3. If metal_ligand_exchange_matrix SHIPPED: re-run on 10x3 rotation (CPU)
4. Document: Phase-3G completion honest-positive (or honest-negative if singleton)

Honest framing:
- OOD reachability 0.4 — partial (per Wf-297B)
- m18 protein_metal — DESIGN+1-PILOT singleton collapse (per metric18 status)
- gi50_proxy — model derived, n=5000 sample
- pharmacophore_pass_rate 0.7 — biased upward on curated smoke; production 0.X pending

OUTPUT (PLAIN TEXT): ood_aggregation_status + m18_aggregation_status + gi50_status + pharmacophore_prod_status + metal_coord_phase3g_status + per_metric_value. Do NOT call StructuredOutput.`, {
    phase: 'Hard metric ship surface (parallel)',
    label: 'wf311D',
  }),

  () => agent(`Wf-311E: Diversity_subpocket + steric_clash + com_shift + m6_kappa 0.472 + hepatotox 0.548 + logp7_4 6.41 ship honest-negative + Wf-219 Ru agg fix + r13_100x3_partial补.

USE Bash + Read + Write + Edit.

CONTEXT: Multiple ship surfaces + honest-negatives need paper-grade aggregation:
- diversity_subpocket (DESIGN_ONLY, full pilot smoke)
- steric_clash_pb_dock (DEFERRED)
- com_shift 1h36/test_001 NOT staged (per Wf-121)
- m6_kappa 0.472 (BELOW ≥0.6)
- hepatotox_index 0.548 (slightly above ≤0.5)
- logp7_4 6.413 (out of 0-5 range)
- Wf-219 Ru agg missing Ru
- r13_100x3_partial Round-13 paper-grade 300 cells NOT delivered

TASK PART 1 — diversity_subpocket ship:
1. Read metrics/by_metric/diversity_subpocket.json — current DESIGN_ONLY
2. Run smoke on real generated pool (449 mols or CFM generation n=100)
3. Update status from DESIGN_ONLY to MEASURED if feasible

TASK PART 2 — steric_clash_pb_dock ship:
1. Read metrics/by_metric/steric_clash_pb_dock.json — current DEFERRED
2. Run smoke on CFM generated pool
3. Update status from DEFERRED to MEASURED if feasible

TASK PART 3 — com_shift 1h36/test_001 NOT staged补:
1. Read metrics/by_metric/com_shift*.json × 5 pockets — current states
2. Identify which pocket is NOT staged
3. Stage it (or document why not)

TASK PART 4 — m6_kappa + hepatotox + logp7_4 ship honest-negative:
1. Read metrics/by_metric/{m6_kappa, hepatotox_index, logp7_4}.json
2. Verify honest-negative status (BELOW or slightly ABOVE threshold)
3. Update §9 honest framing catalog (per Wf-311A) with these
4. Update paper_dashboard with honest-negative claim cards

TASK PART 5 — Wf-219 Ru agg fix + r13_100x3_partial:
1. Read molmetal/reports/wf219_krasnov_aligned/sweep_summary.json — Ru missing
2. Verify Ru entries (per Wf-310A aggregation)
3. Read metrics/by_round/r13_100x3_partial.json — partial completion
4. Document round-13 deferred workstream

Honest framing:
- m6_kappa 0.472 BELOW ≥0.6 is honest-negative
- hepatotox 0.548 slightly above ≤0.5 is honest-negative
- logp7_4 6.413 out of 0-5 range is honest-negative
- These are real metrics; honest-negative framing is required

OUTPUT (PLAIN TEXT): diversity_subpocket_status + steric_clash_status + com_shift_staging_status + n_honest_negatives_added + wf219_ru_agg_status + r13_100x3_partial_status. Do NOT call StructuredOutput.`, {
    phase: 'Hard metric ship surface (parallel)',
    label: 'wf311E',
  }),
])

phase('Synthesis')

const wf311F = await agent(`Wf-311F: Aggregate Wf-311 hard metric ship surface outcomes — paper §4/§6 final state.

USE Read only.

UPSTREAM RESULTS:
- Wf-311A: paper §4 violations fix + §9 expansion
- Wf-311B: paper §4 hard metric aggregation
- Wf-311C: paper §4/§6 rewrite per TODO-144
- Wf-311D: OOD + m18 + gi50 + pharmacophore + metal_coord ship
- Wf-311E: diversity + steric + com_shift + m6/hepatotox/logp7 honest-negative ship

TASK:
1. Read all 5 outputs
2. Aggregate: completed + failed + partial
3. Build paper §4/§6 final state summary
4. Output to molmetal/reports/wf311_synthesis/paper_4_final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + paper_4_final_state + paper_6_limitations + remaining_blockers + n_hard_metrics_shipped (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf311F',
})

return {
  ultracode: 'wf-311-hard-metric-ship',
  wf311a: wf311A,
  wf311b: wf311B,
  wf311c: wf311C,
  wf311d: wf311D,
  wf311e: wf311E,
  wf311f: wf311F,
}