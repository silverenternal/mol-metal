export const meta = {
  name: 'wf-307-actionable-now-batch2',
  description: 'Wf-307: Actionable-now batch 2 — TODO audit follow-up from Wf-303A/B. (A) Wf-134 Phase 1 — ref SMILES extraction from 5/5 pockets (CPU, 280 LOC + 12 tests, 2-3 days, NO auth gate). (B) TODO-178b smoke-test FAIL-on-missing hardening (0.5 day, party-twenty-spirit §3.5 hygiene). (C) ADMET pass-through integration — wire Wf-303C admet_pass_through.py into pre_flight_check.py + paper_dashboard.py. (D) TODO-234 frontier math wire-in CPU portion — 4 adapters audit verification + integration tests. (E) Synthesis.',
  phases: [
    { title: 'Actionable-now batch 2 (parallel)', detail: '4 parallel: Wf-134 P1, TODO-178b, ADMET integration, TODO-234 CPU' },
    { title: 'Synthesis', detail: 'Aggregate + remaining actionable' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Actionable-now batch 2 (parallel)')

const [wf307A, wf307B, wf307C, wf307D] = await parallel([

  () => agent(`Wf-307A: Wf-134 Phase 1 — ref SMILES extraction from 5/5 pockets.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-303A/B audit: Wf-134 / TODO-134 = m15 Docking RMSD redesign. Phase 1 = ref SMILES extract from 5/5 pockets (CPU, 1-2 days, NO auth gate). Scope: ~280 LOC + ≥12 tests. Source: TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.3.

TASK:
1. Read TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.3 — full Wf-134 scope (3 phases)
2. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — 26,633 rows (Wf-261)
3. Identify 5 reference pockets per Wf-134 §1.3:
   - Likely: 1h36 (MMP13), 1hov (CA2), 2oxw (MMP2), 3ehy, 4f5s (or per IMPROVEMENT_ENUM §1.3)
4. Write NEW extraction script: molmetal/scripts/wf134_p1_extract_ref_smiles.py
   - For each pocket, extract reference ligand SMILES from PDB via RDKit
   - Save to molmetal/.data/molmetal/wf134_p1_ref_smiles.csv
   - Columns: pocket_id, ref_ligand_smiles, ref_ligand_name, source_pdb, extraction_method, n_atoms
5. Write 12 tests: molmetal/tests/test_wf134_p1_extract_ref_smiles.py
   - 5 tests: each pocket extraction succeeds
   - 3 tests: SMILES sanity (RDKit parsable, no SMILES bomb)
   - 2 tests: CSV structure (5 rows, 6 cols)
   - 2 tests: integration with downstream m15 RMSD placeholder
6. Run: ${PY} molmetal/scripts/wf134_p1_extract_ref_smiles.py --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --out .data/molmetal/wf134_p1_ref_smiles.csv 2>&1 | tail -20
7. Run tests: ${PY} -m pytest molmetal/tests/test_wf134_p1_extract_ref_smiles.py -v 2>&1 | tail -20
8. Document Phase 1 close-out: TODO/completed/134_wf134_p1_ref_smiles_extract.md
10. Move pending/134_*.md → completed/ (if standalone file exists; otherwise create new completed file)

Honest framing:
- 2-3 day wall; CPU-only; no auth gate
- Phase 2 (Vina redocking 5pocket×3seed×5poses=75) and Phase 3 (RMSD metric JSON) DEFERRED — Phase 2 needs GPU-0 user OK
- 280 LOC target; 12 tests target

OUTPUT (PLAIN TEXT): n_pockets_extracted + ref_smiles_csv_path + n_tests_pass + phase_1_status + phase_2_3_deferred. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now batch 2 (parallel)',
    label: 'wf307A',
  }),

  () => agent(`Wf-307B: TODO-178b smoke-test FAIL-on-missing hardening (party-twenty-spirit §3.5 hygiene).

USE Bash + Read + Edit.

CONTEXT: Per Wf-303A/B audit: TODO-178b = smoke-test FAIL-on-missing hardening (0.5 day CPU, party-twenty-spirit §3.5 hygiene). Wf-303C verified _metalcytotox_preflight raises AssertionError on 4 conditions, but is there a corresponding meta-test that audits ALL critical smoke tests have FAIL-on-missing semantics (not pytest.skip)?

TASK:
1. Read molmetal/tests/test_wasserstein_closure_integration.py — Wf-303C reference (line 763+ meta-test)
2. Read molmetal/scripts/pre_flight_check.py — current pre-flight uses pytest.skip on missing data
3. Audit all smoke tests in molmetal/tests/:
   grep -rn "pytest.skip\|skipif" molmetal/tests/ 2>&1 | head -30
4. Find tests that should FAIL-on-missing but currently skip:
   - Smoke tests that gate critical paths
   - Tests that depend on data files
5. Add NEW meta-test: molmetal/tests/test_smoke_fail_on_missing_audit.py
   - Walk all tests in molmetal/tests/
   - Find tests marked with @pytest.mark.smoke or named *smoke*
   - Audit each: does it raise (not skip) on missing data?
6. Run: ${PY} -m pytest molmetal/tests/test_smoke_fail_on_missing_audit.py -v 2>&1 | tail -20
7. For any smoke tests that skip-on-missing (instead of fail), fix them:
   - Replace pytest.skip with pytest.fail + AssertionError
   - Per party-twenty-spirit §3.5: NEVER silently skip; ALWAYS fail loudly
8. Document in: TODO/completed/178b_smoke_fail_on_missing_hardening.md

Honest framing:
- 0.5 day wall; CPU-only
- This is hygiene, not feature work
- Some skips are LEGITIMATE (e.g. GPU-only tests on CPU env); the audit distinguishes legit vs illegitimate skips

OUTPUT (PLAIN TEXT): n_smoke_tests_audited + n_illegitimate_skips_found + n_skips_fixed + meta_test_status + per_party_twenty_spirit_compliance. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now batch 2 (parallel)',
    label: 'wf307B',
  }),

  () => agent(`Wf-307C: ADMET pass-through integration — wire Wf-303C admet_pass_through.py into pre_flight_check.py + paper_dashboard.py.

USE Bash + Read + Edit.

CONTEXT: Per Wf-303C Task 2: NEW molmetal/scripts/admet_pass_through.py (~340 LOC) emits 3 metric JSONs from 449 metallo-drug SMILES via existing metrics_v2. Results: ames_mutagen 0.0401 PASS / cell_permeability_logPapp -3.637 PASS / aqueous_solubility_logS -7.612 FAIL honest-negative. JSON files at metrics/by_metric/{ames_mutagen,cell_permeability_logPapp,aqueous_solubility_logS}.json. Not yet integrated into pre_flight_check.py or paper_dashboard.py.

TASK:
1. Read molmetal/scripts/admet_pass_through.py — Wf-303C output
2. Read molmetal/scripts/pre_flight_check.py — current 25-gate structure
3. Add NEW gates:
   - Gate: ames_mutagen.json exists + ≤ 0.10 (druglikeness)
   - Gate: cell_permeability_logPapp.json exists + ≥ -5.0 (druglikeness)
   - Gate: aqueous_solubility_logS.json exists + HONEST_NEGATIVE tag present
4. Update pre_flight_check.py to count these 3 gates
5. Read molmetal/scripts/paper_dashboard.py — current 34 claims / 18 honest-negatives
6. Add NEW claim cards:
   - Claim: ames_mutagen 0.0401 (PASS)
   - Claim: cell_permeability_logPapp -3.637 (PASS)
   - Claim: aqueous_solubility_logS -7.612 (FAIL honest-negative)
7. Verify pre_flight_check.py runs:
   ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Verify paper_dashboard.py runs:
   ${PY} molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v307c.html 2>&1 | tail -10
9. Document in: TODO/completed/admet_pass_through_integration.md

Honest framing:
- This is INTEGRATION work, not new MEASUREMENT
- 3 metrics already MEASURED per Wf-303C; this just surfaces them in pre-flight + paper-dashboard
- aqueous_solubility_logS -7.612 IS a real honest-negative — keep it visible

OUTPUT (PLAIN TEXT): n_admet_gates_added + preflight_status + n_dashboard_claims_added + dashboard_status + per_metric_verdict. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now batch 2 (parallel)',
    label: 'wf307C',
  }),

  () => agent(`Wf-307D: TODO-234 frontier math wire-in CPU portion + audit verification.

USE Bash + Read + Edit.

CONTEXT: Per Wf-303A/B audit: TODO-234 = Wf-234 frontier math wire-in. Per Wf-304C: 0/4 outer adapters wired (now 4/4 per Wf-305B lazy import). Per Wf-305B: 4 adapters wired via lazy import — but actual smoke (forward-pass through each adapter) NOT yet run. CPU portions = 3-5 days; full task = 1-2 weeks (R17 invariant gates GPU portion).

TASK:
1. Read molmetal/adapters/flow_matching_lipman/__init__.py — Wf-305B wire-in
2. Read molmetal/adapters/flow_matching_lipman/_reference/multi_ph/__init__.py
3. Read molmetal/adapters/flow_matching_lipman/_reference/se3_lipman/__init__.py
4. Read molmetal/adapters/flow_matching_lipman/_reference/se3_sb/__init__.py
5. Read molmetal/adapters/flow_matching_lipman/_reference/transport_relaxation/__init__.py
6. Run smoke test per adapter (CPU-only, 1 forward-pass):
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import MultiPHScheduler; m = MultiPHScheduler(); print('multi_ph OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import SE3LipmanScheduler; m = SE3LipmanScheduler(); print('se3_lipman OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import SE3SBScheduler; m = SE3SBScheduler(); print('se3_sb OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import TransportRelaxationScheduler; m = TransportRelaxationScheduler(); print('transport_relaxation OK:', m)" 2>&1 | tail -5
7. For each adapter that imports OK, run a 1-iteration forward-pass smoke:
   - Use dummy tensor (e.g., torch.zeros(2, 3, 64))
   - Verify output shape matches expectation
   - Time the forward-pass
8. Write 4 forward-pass tests: molmetal/tests/test_wf307d_frontier_adapters_smoke.py
   - test_multi_ph_forward
   - test_se3_lipman_forward
   - test_se3_sb_forward
   - test_transport_relaxation_forward
9. Run: ${PY} -m pytest molmetal/tests/test_wf307d_frontier_adapters_smoke.py -v 2>&1 | tail -20
10. Document CPU portion close-out: TODO/completed/234_wf234_frontier_math_cpu_portion.md

Honest framing:
- This is CPU-only smoke; full wire-in (1-2 weeks) needs user OK + R17 invariant satisfied
- GPU portions DEFERRED per Wf-304C §R17 invariant
- 3 research-only modules (bass_existence / sb_bass_martingale / monge_ampere_regularity) honestly research-only by design — DO NOT wire-in

OUTPUT (PLAIN TEXT): n_adapters_actually_loaded + n_forward_pass_smoked + n_tests_pass + cpu_portion_status + gpu_portion_deferred. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now batch 2 (parallel)',
    label: 'wf307D',
  }),
])

phase('Synthesis')

const wf307E = await agent(`Wf-307E: Aggregate Wf-307 actionable-now batch 2 outcomes.

USE Read only.

UPSTREAM RESULTS:
- Wf-307A: Wf-134 Phase 1 ref SMILES extraction
- Wf-307B: TODO-178b smoke-test FAIL-on-missing hardening
- Wf-307C: ADMET pass-through integration
- Wf-307D: TODO-234 frontier math CPU portion

TASK:
1. Read all 4 outputs
2. Aggregate: completed + failed + partial
3. Identify remaining actionable-now tasks from Wf-303A/B
4. Output to molmetal/reports/wf307_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + actionable_now_closed + remaining_actionable (8-12 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf307E',
})

return {
  ultracode: 'wf-307-actionable-now-batch2',
  wf307a: wf307A,
  wf307b: wf307B,
  wf307c: wf307C,
  wf307d: wf307D,
  wf307e: wf307E,
}
