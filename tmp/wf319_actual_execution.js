export const meta = {
  name: 'wf-319-wf315-closeout-actual-execution',
  description: 'Wf-319: Wf-315 actionable-now closeout ACTUAL EXECUTION (Wf-315 B/C/D/E/F/G only analyzed, did not execute per user feedback). (A) ablation_runner.py unit tests + 11 R17 ablation smoke + aggregate.csv (Wf-315B actual; 1037 LOC, 0 tests!). (B) UFF-min Pt-pool actual — RDKit UFF relaxation for 100 Pt-drug conformers (Wf-315C actual). (C) PoseBusters packages verify install + 1h36 PB check actual (Wf-315D actual). (D) TODO-178 G2 metalcytotox CSV final audit + FAIL-on-missing actual (Wf-315E actual). (E) NCI-60 9-cell per-cell scaffold-split fallback actual — apply Wf-312A pattern to other 8 cells (Wf-315F actual). (F) m6_kappa + ADMET consistency audit actual — verify all tags + claim cards wired (Wf-315G actual). (G) Wf-296D deep wire — 84-pattern matrix + YuelBond precondition runtime wire (NEW from Wf-315B). (H) Synthesis + INDEX.md update.',
  phases: [
    { title: 'Wf-315 closeout actual execution (parallel)', detail: '7 parallel agents actually EXECUTING Wf-315 deferred work' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Wf-315 closeout actual execution (parallel)')

const [wf319A, wf319B, wf319C, wf319D, wf319E, wf319F, wf319G] = await parallel([

  () => agent(`Wf-319A: ablation_runner.py unit tests + 11 R17 ablation smoke + aggregate.csv — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315B analysis: ablation_runner.py is 1037 LOC with NO unit tests. Need to write tests + execute 11 R17 ablations smoke + emit aggregate.csv for paper §6.

TASK PART 1 — unit tests for ablation_runner:
1. Read molmetal/scripts/ablation_runner.py — identify key functions
2. Identify 5 toggleable (M7/M12/M13/M14/M15) + 6 DESIGN_ONLY (M3/M4/M8/M10/M11/M21)
3. Write 12 NEW tests: molmetal/tests/test_wf319a_ablation_runner.py
   - 3 tests: 5 toggleable ablations each (M7, M12, M13, M14, M15)
   - 3 tests: 6 DESIGN_ONLY ablations (each registers DESIGN_ONLY status)
   - 2 tests: aggregate.csv generation
   - 2 tests: final.md generation
   - 2 tests: integration with metallo_drug_smoke_retrain.py
4. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf319a_ablation_runner.py -v 2>&1 | tail -20

TASK PART 2 — 11 R17 ablations smoke execution:
1. Run ablation_runner.py end-to-end:
   /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/ablation_runner.py --output-dir molmetal/reports/wf319a_ablation_runner 2>&1 | tail -30
2. Verify aggregate.csv has 11 rows × 5 columns
3. Verify final.md has human-readable summary table

TASK PART 3 — pre-flight gate addition:
1. Add p3_ablation_runner_tests_present + passes (verifies 12 tests PASS)
2. Add p3_ablation_runner_aggregate_csv_present (verifies 11 rows)
3. Update pre_flight_check.py gate counter

TASK PART 4 — paper_dashboard integration:
1. Add 1 NEW claim card: claim_wf319a_ablation_table_aggregate
2. Update section_5_ablation.json with new ablation table
3. Document: TODO/completed/319a_ablation_runner_smoke.md

Honest framing:
- 5 toggleable ablations — measured smoke
- 6 DESIGN_ONLY — design-only status, NOT fabricated
- 200-step CFM retrain + 16-sample decode = SMOKE, not full 10k training

OUTPUT (PLAIN TEXT): n_tests_pass + n_ablations_executed + n_toggleable_measured + n_design_only + aggregate_csv_rows + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319A',
  }),

  () => agent(`Wf-319B: UFF-min Pt-pool ACTUAL EXECUTION — RDKit UFF relaxation for 100 Pt-drug conformers.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315C analysis: UFF-min Pt-pool needs actual execution. 100 Pt-drug conformers from MetalCytoToxDB Pt subset. RDKit UFF minimization.

TASK:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — Pt rows count
3. Sample 100 Pt rows from MetalCytoToxDB (balanced active/inactive if possible)
4. Check RDKit UFF minimization API:
   /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "from rdkit.Chem import AllChem; print('OK')"
6. For each Pt SMILES:
   - Parse with RDKit
   - Add hydrogens
   - Embed conformer (ETKDG)
   - UFF minimize (maxIters=200)
   - Compute per-conformer UFF energy
7. Save:
   - molmetal/reports/wf319b_uff_min_pt_pool/pool.json (100 conformers + energies)
   - metrics/by_metric/uff_energy_pt_pool.json (mean ± std + distribution)
8. Add pre_flight gate: p3_uff_min_pt_pool_present (verify 100 conformers + energies)
9. Write 8 NEW tests: molmetal/tests/test_wf319b_uff_min_pt_pool.py
   - 3 tests: per-conformer UFF energy sanity
   - 2 tests: pool distribution
   - 2 tests: pre-flight gate validation
   - 1 test: integration with Vina
10. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf319b_uff_min_pt_pool.py -v 2>&1 | tail -20
11. Document: TODO/completed/319b_uff_min_pt_pool.md

Honest framing:
- UFF-min is pure CPU (RDKit UFF)
- 100 conformers is smoke scale
- m10 Vina Min Score ceiling improvement is PROJECTED (per WEAK_TO_STRONG §2.7)
- Full UFF-min × Vina pipeline requires GPU portion (out of scope)

OUTPUT (PLAIN TEXT): n_conformers_generated + per_pool_energy_distribution + preflight_status + n_tests_pass. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319B',
  }),

  () => agent(`Wf-319C: PoseBusters packages verify install + 1h36 PB check — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315D analysis: PB packages install status + 1h36 PB check needs actual execution.

TASK PART 1 — package availability check:
1. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "import rdkit; print('rdkit:', rdkit.__version__)" 2>&1
2. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "import posebusters; print('posebusters:', posebusters.__version__)" 2>&1
3. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "from rdkit.Chem import AllChem; m=AllChem.MolFromSmiles('CCO'); AllChem.EmbedMolecule(m); print('RDKit OK')" 2>&1

TASK PART 2 — if packages available:
1. Load 1h36 docked poses (need to determine availability)
   ls -la molmetal/.data/molmetal/docked_poses_1h36/ 2>&1
2. If poses available: run PB check
3. Compute m8_prod (1h36 only) = mean PB pass rate
4. Save: molmetal/reports/wf319c_pb_smoke/m8_prod_1h36.json
5. Update metrics/by_metric/steric_clash_pb_dock.json status from DEFERRED → MEASURED_SMOKE

TASK PART 3 — if packages NOT available:
1. Try alt env: /home/hugo/codes/mol-metal/.venv/bin/python -c "import rdkit; print(rdkit.__version__)"
2. If alt env has packages: use alt env for PB check
3. If still NOT available: HONEST_NEGATIVE_BLOCKED + recovery path

TASK PART 4 — pre-flight gate:
1. p3_pb_packages_status_present (verifies package availability documented)
2. Update pre_flight_check.py

TASK PART 5 — write 8 NEW tests: molmetal/tests/test_wf319c_pb_smoke.py
   - 3 tests: package availability (3 envs)
   - 3 tests: 1h36 PB pass rate computation
   - 2 tests: pre-flight gate validation

Honest framing:
- Either MEASURED or HONEST_NEGATIVE_BLOCKED — never fabricate
- Wf-319C is just smoke (1h36 only); full production 100×3 is BLOCKED on F2(a) MetalLigandExchange SMARTS rule (TODO-29) + GPU

OUTPUT (PLAIN TEXT): rdkit_version + posebusters_version + package_status + pb_1h36_result + preflight_status + n_tests_pass. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319C',
  }),

  () => agent(`Wf-319D: TODO-178 G2 metalcytotox CSV final audit + FAIL-on-missing — ACTUAL EXECUTION.

USE Bash + Read + Edit.

CONTEXT: Per Wf-315E analysis: G2 metalcytotox CSV needs final audit + verify FAIL-on-missing.

TASK:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv (latest version)
2. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask.csv (original)
3. Verify schema: columns + types + missing values + per-metal row count
4. Verify smoke test FAIL-on-missing:
   PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wasserstein_closure_integration.py -v 2>&1 | tail -10
5. Per-metal row count + pos_rate:
   - Ru: n + pos_rate
   - Ir: n + pos_rate
   - Os: n + pos_rate
   - Rh: n + pos_rate
   - Re: n + pos_rate
6. Save audit: molmetal/reports/wf319d_g2_audit/audit.json
7. Write 8 NEW tests: molmetal/tests/test_wf319d_metalcytotox_csv_audit.py
   - 2 tests: schema validation (v3 + v4 + original)
   - 2 tests: FAIL-on-missing
   - 2 tests: per-metal row count
   - 2 tests: integration with downstream consumer
8. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf319d_metalcytotox_csv_audit.py -v 2>&1 | tail -20
9. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
10. Document: TODO/completed/319d_g2_metalcytotox_csv_audit.md

Honest framing:
- G2 was "PARTIAL" pre-Wf-310H; now should be PASS
- Verify, don't re-do work
- Pre-flight should PASS

OUTPUT (PLAIN TEXT): n_rows_total + per_metal_count + pos_rate + preflight_status + n_tests_pass + fail_on_missing_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319D',
  }),

  () => agent(`Wf-319E: NCI-60 9-cell per-cell scaffold-split fallback ACTUAL — extend Wf-312A Re fix to other 8 cells.

USE Bash + Read + Edit.

CONTEXT: Per Wf-315F analysis: Wf-312A implemented --split {temporal,scaffold,auto} fallback only for Re. Need to extend to other 8 NCI-60 cells.

TASK:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — current per-cell split logic
2. Read molmetal/scripts/wf219_temporal_split_driver.py — Wf-312A scaffold fallback pattern
3. For each of 9 cells (MCF7/A2780/A549/HCT116/PC3/HL60/A2780cis/DU145/SKOV3):
   - Check current temporal split n_test (may need to run temporal first to get n_test)
   - Identify which cells have temporal-degenerate (n_test < 50)
4. Extend nci60_9cell_chemprop_driver.py with --split {temporal,scaffold,auto}:
   - auto mode: scaffold fallback when temporal n_test < 50
   - per-cell split_choice recorded in JSON
5. Write 10 NEW tests: molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py
   - 5 tests: 5-cell × split decision
   - 3 tests: scaffold fallback correctness
   - 2 tests: integration with downstream per-cell metrics
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py -v 2>&1 | tail -20
7. Document: TODO/completed/319e_nci60_per_cell_scaffold_fallback.md

Honest framing:
- Some cells may be fine on temporal (don't force scaffold)
- Re is most-degenerate (n_test=1 per Wf-312A)
- Other cells may need scaffold fallback
- This is INTEGRATION work, NOT new measurement

OUTPUT (PLAIN TEXT): n_cells_with_scaffold_fallback + per_cell_split_decision + n_tests_pass + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319E',
  }),

  () => agent(`Wf-319F: m6_kappa + ADMET consistency audit ACTUAL — verify all tags + claim cards wired.

USE Bash + Read + Edit.

CONTEXT: Per Wf-315G analysis: m6_kappa + hepatotox + logp7_4 need consistency audit. Tags added by Wf-312C + Wf-314A.

TASK:
1. Read metrics/by_metric/m6_kappa.json — verify HONEST_NEGATIVE=true + new validation data
2. Read metrics/by_metric/hepatotox_index.json — verify tag
3. Read metrics/by_metric/logp7_4.json — verify tag
4. Cross-check paper_dashboard.py for all 3 claim cards + 3 HN catalog entries
5. Cross-check by_paper_section/section_5_ablation.json — verify ADMET extension + R02 honest-negative cluster
6. Read molmetal/scripts/pre_flight_check.py — verify pre-flight gates p3_admet_hepatotox_honest_negative + p3_admet_logp7_4_honest_negative + p3_admet_m6_kappa_honest_negative exist
7. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Emit audit doc: TODO/completed/319f_m6_kappa_admet_consistency.md

Honest framing:
- These tags were added by Wf-312C + Wf-314A; this is verification work
- If inconsistencies found: fix them
- If everything correct: emit audit doc confirming consistency

OUTPUT (PLAIN TEXT): tag_consistency_status + paper_dashboard_status + section_5_status + preflight_status + n_consistency_issues_found + n_gates_verified. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319F',
  }),

  () => agent(`Wf-319G: Wf-296D deep wire — 84-pattern matrix + YuelBond precondition runtime wire (NEW from Wf-315B).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315B analysis: Wf-296D was audit-only; runtime wire not done. 84-pattern matrix (F2(a)) + YuelBond precondition should be wired into runtime for MCTS dispatch.

TASK:
1. Read molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py — current runtime
2. Read molmetal/molmetal_lam/lam_chem/pt_click_compat.py — 84-pattern matrix location
3. Read molmetal/molmetal_lam/search_alg/proof_search.py — current MCTS dispatch
4. Identify: 84-pattern matrix + YuelBond precondition not wired into runtime
5. Write NEW module: molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py
   - Wire 84-pattern matrix dispatch (per-metal ligand exchange)
   - Wire YuelBond precondition check (per Wf-269C YuelBond load-bearing finding)
6. Add 12 NEW tests: molmetal/tests/test_wf319g_wf296d_deep_wire.py
   - 4 tests: 84-pattern runtime dispatch
   - 4 tests: YuelBond precondition check
   - 4 tests: integration with proof_search.py
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf319g_wf296d_deep_wire.py -v 2>&1 | tail -20
8. Update paper_dashboard.py with claim_wf319g_wf296d_deep_wire_shipped
9. Update MECHANISM_METRIC_MAPPING.md: add F2(a) runtime as M45
10. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
11. Document: TODO/completed/319g_wf296d_deep_wire.md

Honest framing:
- This is INTEGRATION wire-up work (CPU-only)
- 84-pattern matrix already shipped in pt_click_compat.py; wiring runtime is new
- YuelBond precondition is per Wf-269C finding
- This work complements Wf-318F (F2(a) runtime wiring) — both touch same area but different scope

OUTPUT (PLAIN TEXT): n_patterns_wired + n_yuelbond_checks + preflight_status + n_tests_pass. Do NOT call StructuredOutput.`, {
    phase: 'Wf-315 closeout actual execution (parallel)',
    label: 'wf319G',
  }),
])

phase('Synthesis')

const wf319H = await agent(`Wf-319H: Aggregate Wf-319 Wf-315 closeout outcomes — INDEX.md final sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-319A: ablation_runner.py unit tests + 11 R17 ablation smoke
- Wf-319B: UFF-min Pt-pool actual execution
- Wf-319C: PB packages install + 1h36 PB check
- Wf-319D: TODO-178 G2 metalcytotox CSV audit
- Wf-319E: NCI-60 per-cell scaffold-split fallback
- Wf-319F: m6_kappa + ADMET consistency audit
- Wf-319G: Wf-296D deep wire

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
4. Output: TODO/completed/319_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf319H',
})

return {
  ultracode: 'wf-319-wf315-closeout-actual-execution',
  wf319a: wf319A,
  wf319b: wf319B,
  wf319c: wf319C,
  wf319d: wf319D,
  wf319e: wf319E,
  wf319f: wf319F,
  wf319g: wf319G,
  wf319h: wf319H,
}