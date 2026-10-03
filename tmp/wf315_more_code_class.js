export const meta = {
  name: 'wf-315-additional-code-class-tasks',
  description: 'Wf-315: Additional code-class actionable-now tasks (NOT covered by Wf-309/310/311/312/314). (A) SA + QED evaluator wiring into metrics_v2.py + admet_pass_through.py + pre-flight gate (WEAK_TO_STRONG §3.6 Phase 1 Week 2 — pure CPU). (B) ablation_runner.py unit tests + 11 R17 ablation smoke execution + aggregate.csv refresh (1037 LOC module with NO unit tests!). (C) UFF-min on Pt-pool — RDKit UFF relaxation for Vina pre-drug 100×3 (WEAK_TO_STRONG §2.7 Phase 1 Week 2). (D) m8-prod PoseBusters smoke retry — verify packages installed + 1h36 PB check + production 100×3 preparation (WEAK_TO_STRONG §2.6 Phase 2). (E) TODO-178 G2 metalcytotox CSV final audit — schema validation + FAIL-on-missing + per-metal row count (Wf-310H says done but verify). (F) TODO-220 NCI-60 9-cell per-cell scaffold-split fallback (Wf-312A only did Re degenerate fix; extend to other 8 cells). (G) m6_kappa MetalCytoToxDB validation refresh — re-run on 5,460-row cohort with new Wf-309A W2 placeholder (already done Wf-311D PART 2; verify integration). (H) Synthesis + TODO/INDEX.md update.',
  phases: [
    { title: 'Additional code-class tasks (parallel)', detail: '7 parallel agents covering more actionable code work' },
    { title: 'Synthesis', detail: 'Aggregate + TODO/INDEX.md sync + paper §6 delta' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Additional code-class tasks (parallel)')

const [wf315A, wf315B, wf315C, wf315D, wf315E, wf315F, wf315G] = await parallel([

  () => agent(`Wf-315A: SA + QED evaluator wiring into metrics_v2.py + admet_pass_through.py + pre-flight gate (WEAK_TO_STRONG §3.6).

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG_PLAN.md §3.6: m5 SA + m6 QED evaluators not wired into metrics_v2.py + admet_pass_through.py. SA score exists at molmetal/molmetal_lam/sbdd_env/sa_score.py (sa_score_ertl function); QED exists at molmetal/molmetal_lam/sbdd_env/qed_scorer.py (qed_from_smiles). Need to wire both into the pass-through pipeline + emit per-cohort aggregate metrics + add to paper_dashboard.

TASK:
1. Read molmetal/molmetal_lam/sbdd_env/sa_score.py — sa_score_ertl signature
2. Read molmetal/molmetal_lam/sbdd_env/qed_scorer.py — qed_from_smiles signature
3. Read molmetal/molmetal_lam/sbdd_env/metrics_v2.py — existing _mean helpers pattern
4. Add to metrics_v2.py:
   - def sa_score(smiles: str) -> float
   - def qed_score(smiles: str) -> float
   - def sa_score_mean(candidates: Sequence[str]) -> float
   - def qed_score_mean(candidates: Sequence[str]) -> float
   - Update __all__ exports
5. Add to molmetal/scripts/admet_pass_through.py:
   - 2 NEW METRIC_SPECS entries (sa_score, qed_score)
   - Update verdict logic (sa < 2.8 PASS, qed > 0.5 PASS per Wf-139 P6 specs)
6. Re-run pass-through on 449 metallo-drug SMILES:
   ${PY} molmetal/scripts/admet_pass_through.py 2>&1 | tail -10
7. Save 2 NEW JSONs:
   - metrics/by_metric/sa_score.json
   - metrics/by_metric/qed_score.json
8. Add 2 NEW pre_flight gates:
   - p3_sa_score_metric_present
   - p3_qed_score_metric_present
9. Add 2 NEW paper_dashboard claim cards
10. Write 8 NEW tests: molmetal/tests/test_wf315a_sa_qed_wiring.py
    - 2 tests: SA/QED function correctness
    - 2 tests: pass-through emission
    - 2 tests: pre-flight gate validation
    - 2 tests: integration with downstream consumer
11. Run: ${PY} -m pytest molmetal/tests/test_wf315a_sa_qed_wiring.py -v 2>&1 | tail -20
12. Document: TODO/completed/315a_sa_qed_evaluator_wiring.md

Honest framing:
- Per WEAK_TO_STRONG §3.6: m5 SA + m6 QED wire is pure CPU (1-2 days)
- SA/QED evaluators exist; this is INTEGRATION work
- 449 SMILES already on disk for admet_pass_through

OUTPUT (PLAIN TEXT): n_metrics_wired + n_tests_pass + per_metric_value + preflight_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315A',
  }),

  () => agent(`Wf-315B: ablation_runner.py unit tests + 11 R17 ablation smoke execution + aggregate.csv refresh.

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG §3.1-§3.7 + TODO/MECHANISM_METRIC_MAPPING.md §5: 11 R17 ablations declared (M3/M4/M7/M8/M10/M11/M12/M13/M14/M15/M21). molmetal/scripts/ablation_runner.py is 1037 LOC with NO unit tests (test_ablation* missing). Need to write tests + execute ablations + emit aggregate.csv for paper §6 ablation table.

TASK PART 1 — unit tests for ablation_runner:
1. Read molmetal/scripts/ablation_runner.py — 1037 LOC structure
2. Identify key functions: ablate_<mechanism>() × 5 toggleable (M7/M12/M13/M14/M15), DESIGN_ONLY × 6 (M3/M4/M8/M10/M11/M21)
3. Write 12 NEW tests: molmetal/tests/test_wf315b_ablation_runner.py
   - 3 tests: 5 toggleable ablations each (M7, M12, M13, M14, M15)
   - 3 tests: 6 DESIGN_ONLY ablations (each registers DESIGN_ONLY status)
   - 2 tests: aggregate.csv generation
   - 2 tests: final.md generation
   - 2 tests: integration with metallo_drug_smoke_retrain.py
4. Run: ${PY} -m pytest molmetal/tests/test_wf315b_ablation_runner.py -v 2>&1 | tail -20

TASK PART 2 — 11 R17 ablations smoke execution:
1. Run ablation_runner.py end-to-end:
   ${PY} molmetal/scripts/ablation_runner.py --output-dir molmetal/reports/wf_ablation_runner_smoke 2>&1 | tail -30
2. Verify aggregate.csv has 11 rows × 5 columns
3. Verify final.md has human-readable summary table

TASK PART 3 — pre-flight gate addition:
1. p3_ablation_runner_tests_present + passes
2. p3_ablation_runner_aggregate_csv_present (verify 11 rows)
3. Update pre_flight_check.py gate counter

TASK PART 4 — paper_dashboard integration:
1. Add 1 NEW claim card: claim_wf315b_ablation_table_aggregate
2. Update section_5_ablation.json with new ablation table
3. Document: TODO/completed/315b_ablation_runner_smoke.md

Honest framing per party-twenty-spirit §3.5:
- 5 toggleable ablations (M7/M12/M13/M14/M15) — measured smoke
- 6 DESIGN_ONLY (M3/M4/M8/M10/M11/M21) — design-only status, NOT fabricated
- 200-step CFM retrain + 16-sample decode = SMOKE, not full 10k training

OUTPUT (PLAIN TEXT): n_tests_pass + n_ablations_executed + n_toggleable_measured + n_design_only + aggregate_csv_rows + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315B',
  }),

  () => agent(`Wf-315C: UFF-min on Pt-pool — RDKit UFF relaxation for Vina pre-drug 100×3 (WEAK_TO_STRONG §2.7).

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG §2.7 + §2.6: m10 Vina Min Score needs UFF-relaxed Pt-pool as input. Per §2.6 Phase 2: MMFF94 relax on 100×3 production sweep. UFF-min is pure CPU (RDKit UFF). Need to:
- Generate UFF-relaxed pool of 100 Pt-drug conformers from MetalCytoToxDB Pt subset
- Per-conformer UFF energy + per-pool distribution
- Verify Vina Min Score ceiling improvement (WEAK_TO_STRONG §2.7)

TASK:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — Pt rows
3. Read molmetal/molmetal_lam/sbdd_env/ — RDKit-based UFF minimization code (if exists)
4. Sample 100 Pt rows from MetalCytoToxDB (balanced active/inactive if possible)
6. For each, compute UFF-min conformer via RDKit:
   ${PY} -c "from rdkit.Chem import AllChem; print('OK')"
7. Compute per-conformer UFF energy + per-pool distribution
8. Save:
   - molmetal/reports/wf315c_uff_min_pt_pool/pool.json (100 conformers + energies)
   - metrics/by_metric/uff_energy_pt_pool.json (mean ± std + distribution)
9. Add pre_flight gate: p3_uff_min_pt_pool_present (verify 100 conformers + energies)
11. Write 8 NEW tests: molmetal/tests/test_wf315c_uff_min_pt_pool.py
    - 3 tests: per-conformer UFF energy sanity
    - 2 tests: pool distribution
    - 2 tests: pre-flight gate validation
    - 1 test: integration with Vina
12. Run: ${PY} -m pytest molmetal/tests/test_wf315c_uff_min_pt_pool.py -v 2>&1 | tail -20
13. Document: TODO/completed/315c_uff_min_pt_pool.md

Honest framing:
- UFF-min is pure CPU (RDKit UFF)
- 100 conformers is smoke scale (not full 100×3 production sweep)
- m10 Vina Min Score ceiling improvement is PROJECTED (WEAK_TO_STRONG §2.7 +0.5 kcal/mol conservative)
- Full UFF-min × Vina pipeline requires GPU portion (out of scope)

OUTPUT (PLAIN TEXT): n_conformers_generated + per_pool_energy_distribution + preflight_status + n_tests_pass. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315C',
  }),

  () => agent(`Wf-315D: m8-prod PoseBusters smoke retry — verify packages + 1h36 PB check + production 100×3 preparation (WEAK_TO_STRONG §2.6).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-311E Part 2: steric_clash_pb_dock.json is DEFERRED because "RDKit + posebusters Python packages not installed in this snapshot". Need to verify current state + attempt installation + if possible, run 1h36 PB check.

TASK PART 1 — package availability check:
1. ${PY} -c "import rdkit; print('rdkit:', rdkit.__version__)" 2>&1
2. ${PY} -c "import posebusters; print('posebusters:', posebusters.__version__)" 2>&1
3. ${PY} -c "from rdkit.Chem import AllChem; m=AllChem.MolFromSmiles('CCO'); AllChem.EmbedMolecule(m); print('RDKit OK')" 2>&1

TASK PART 2 — if packages available:
1. Load 1h36 docked poses (need to determine availability)
2. Run PB check on 1h36 poses
3. Compute m8_prod (1h36 only) = mean PB pass rate
4. Save: molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json
5. Update metrics/by_metric/steric_clash_pb_dock.json status from DEFERRED → MEASURED_SMOKE

TASK PART 3 — if packages NOT available:
1. Document the missing packages as recovery path
2. Update steric_clash_pb_dock.json with HONEST_NEGATIVE_BLOCKED tag + clear blocker description
3. NO measurement (don't fabricate)

TASK PART 4 — pre-flight gate:
1. p3_pb_packages_available or p3_pb_packages_missing_documented
2. Update pre_flight_check.py

TASK PART 5 — write 6 NEW tests: molmetal/tests/test_wf315d_pb_smoke.py
    - 3 tests: package availability
    - 2 tests: 1h36 PB pass rate computation
    - 1 test: pre-flight gate

Honest framing:
- Either MEASURED or HONEST_NEGATIVE_BLOCKED — never fabricate
- Wf-315D is just smoke (1h36 only); full production 100×3 is BLOCKED on F2(a) MetalLigandExchange SMARTS rule (TODO-29) + GPU portion

OUTPUT (PLAIN TEXT): rdkit_version + posebusters_version + package_status + pb_1h36_result + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315D',
  }),

  () => agent(`Wf-315E: TODO-178 G2 metalcytotox CSV final audit — schema validation + FAIL-on-missing + per-metal row count (Wf-310H verification).

USE Bash + Read + Edit.

CONTEXT: Per TODO/pending/183_wf123_wf124_p3_gap_routing_plan.md: G2 (TODO-178) status was PARTIAL — CSV exists on disk at .data/molmetal/metalcytotox_5metal_multitask.csv (26801 rows, 5 metals) but schema is smiles,metal,scaffold_id,split,activity (5 cols — MISSING pic50,cell_line,source); smoke test SKIPs gracefully. Per Wf-310H + Wf-308A: should be FAIL-on-missing now. Need final audit verification.

TASK:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv (latest version)
2. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask.csv (original)
3. Verify schema: columns + types + missing values + per-metal row count
4. Verify smoke test FAIL-on-missing:
   ${PY} -m pytest molmetal/tests/test_wasserstein_closure_integration.py -v 2>&1 | tail -10
5. Verify pre-flight: gate 'p3_metalcytotox_csv_schema_valid' PASS
6. Per-metal row count + pos_rate:
   - Ru: n + pos_rate
   - Ir: n + pos_rate
   - Os: n + pos_rate
   - Rh: n + pos_rate
   - Re: n + pos_rate
7. Save audit: molmetal/reports/wf315e_g2_audit/audit.json
8. Write 6 NEW tests: molmetal/tests/test_wf315e_metalcytotox_csv_audit.py
    - 2 tests: schema validation
    - 2 tests: FAIL-on-missing
    - 2 tests: per-metal row count
9. Document: TODO/completed/315e_g2_metalcytotox_csv_audit.md

Honest framing:
- G2 was "PARTIAL" pre-Wf-310H; now should be PASS
- Verify, don't re-do work
- Pre-flight should PASS

OUTPUT (PLAIN TEXT): n_rows_total + per_metal_count + pos_rate + preflight_status + n_tests_pass + fail_on_missing_status. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315E',
  }),

  () => agent(`Wf-315F: TODO-220 NCI-60 9-cell per-cell scaffold-split fallback (Wf-312A only did Re degenerate fix; extend to other 8 cells).

USE Bash + Read + Edit.

CONTEXT: Per Wf-312A: Re temporal split is degenerate (n_test=1). Wf-312A implemented --split {temporal,scaffold,auto} fallback in wf219_temporal_split_driver.py for Re specifically. NCI-60 9-cell driver (nci60_9cell_chemprop_driver.py) does NOT yet have this fallback — only Re degenerate case was handled. Need to extend scaffold fallback to all 9 NCI-60 cells.

TASK:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — current per-cell split logic
2. Read molmetal/scripts/wf219_temporal_split_driver.py — Wf-312A's scaffold fallback pattern
3. For each of 9 cells (MCF7/A2780/A549/HCT116/PC3/HL60/A2780cis/DU145/SKOV3):
   - Investigate row count + year distribution
   - Identify which cells have temporal-degenerate (n_test < 3) or insufficient temporal split
4. Extend nci60_9cell_chemprop_driver.py with --split {temporal,scaffold,auto}:
   - auto mode: scaffold fallback when temporal n_test < 50 (per Wf-312A pattern)
   - per-cell split_choice recorded in JSON
5. Write 10 NEW tests: molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py
    - 5 tests: 5-cell × split decision
    - 3 tests: scaffold fallback correctness
    - 2 tests: integration with downstream per-cell metrics
6. Run: ${PY} -m pytest molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py -v 2>&1 | tail -20
7. Document: TODO/completed/315f_nci60_per_cell_scaffold_fallback.md

Honest framing:
- Some cells may be fine on temporal (don't force scaffold)
- Re is most-degenerate (n_test=1 per Wf-312A)
- Other cells may need scaffold fallback (need to verify per-cell)
- This is INTEGRATION work, NOT new measurement

OUTPUT (PLAIN TEXT): n_cells_with_scaffold_fallback + per_cell_split_decision + n_tests_pass + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315F',
  }),

  () => agent(`Wf-315G: m6_kappa MetalCytoToxDB validation refresh + ADMET pass-through consistency check (post-Wf-309A W2 placeholder + Wf-312C R02 honest-negative).

USE Bash + Read + Edit.

CONTEXT: Per Wf-311D PART 2: m6_kappa validation done (global Pearson r 0.081 / Spearman ρ 0.111 per Wf-253D on 5,460-row cohort). Per Wf-312C: hepatotox + logp7_4 + m6_kappa got HONEST_NEGATIVE=true tags retroactively. Need to verify these are properly integrated across:
- metrics/by_metric/m6_kappa.json (HONEST_NEGATIVE tag + new validation)
- paper_dashboard.py (claim cards)
- by_paper_section/section_5_ablation.json (ablation table)
- pre_flight_check.py (gates)

TASK:
1. Read metrics/by_metric/m6_kappa.json — verify HONEST_NEGATIVE=true + new validation data
2. Read metrics/by_metric/hepatotox_index.json — verify tag
3. Read metrics/by_metric/logp7_4.json — verify tag
4. Cross-check paper_dashboard.py for all 3 claim cards + 3 HN catalog entries
5. Cross-check by_paper_section/section_5_ablation.json — verify ADMET extension + R02 honest-negative cluster
7. Pre-flight: ${PY} -m pytest molmetal/tests/ -k "admet or kappa or honest" -v 2>&1 | tail -20
8. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- These tags were added by Wf-312C; this is verification work
- If inconsistencies found: fix them
- If everything correct: emit audit doc confirming consistency

OUTPUT (PLAIN TEXT): tag_consistency_status + paper_dashboard_status + section_5_status + preflight_status + n_consistency_issues_found. Do NOT call StructuredOutput.`, {
    phase: 'Additional code-class tasks (parallel)',
    label: 'wf315G',
  }),
])

phase('Synthesis')

const wf315H = await agent(`Wf-315H: Aggregate Wf-315 additional code-class tasks outcomes — TODO/INDEX.md update + paper §6 limitations delta.

USE Read only.

UPSTREAM RESULTS:
- Wf-315A: SA + QED evaluator wiring (WEAK_TO_STRONG §3.6)
- Wf-315B: ablation_runner.py unit tests + 11 R17 ablations smoke
- Wf-315C: UFF-min Pt-pool (WEAK_TO_STRONG §2.7)
- Wf-315D: m8-prod PoseBusters smoke retry (WEAK_TO_STRONG §2.6)
- Wf-315E: TODO-178 G2 metalcytotox CSV audit
- Wf-315F: NCI-60 9-cell per-cell scaffold-split fallback
- Wf-315G: m6_kappa + ADMET consistency audit

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table) with Wf-315 entries
4. Output to molmetal/reports/wf315_synthesis/code_class_final.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_code_class_tasks_completed + n_honest_negatives_documented + remaining_blockers + n_metrics_updated (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf315H',
})

return {
  ultracode: 'wf-315-additional-code-class-tasks',
  wf315a: wf315A,
  wf315b: wf315B,
  wf315c: wf315C,
  wf315d: wf315D,
  wf315e: wf315E,
  wf315f: wf315F,
  wf315g: wf315G,
  wf315h: wf315H,
}
