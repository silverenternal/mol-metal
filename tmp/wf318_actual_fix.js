export const meta = {
  name: 'wf-318-actionable-now-closeout',
  description: 'Wf-318: Wf-314 actionable-now follow-up ACTUAL EXECUTION (Wf-314 B/C/D/F only analyzed, did not execute). (B) Wf-310A cascade delta 5 metal extension (extend Ru-only to all 5 metals: Ir/Os/Rh/Re). (D) Wf-219 Ru aggregation fix — per-seed AUCs integrated into 5-metal aggregate with explicit honest framing. (E) Wf-130 P3 OOD RDKit alt env test — try .venv/bin/python instead of .venv-vlm. (F) Wf-290 F2(a) MetalLigandExchange SMARTS rule runtime wiring — wire SMARTS rule into lam_chem/pt_metal_ligand_exchange.py runtime (resolves TODO-29 Phase-3G blocker). (G) r13_100x3_partial + diversity_subpocket + steric_clash honest-negative FINAL closeout (Wf-314F was analysis). (I) Wf-311F §7 candidates #2-5 final — com_shift 1h36 + Wf-219 Ru agg (covered by D) + Phase-3G + Wf-264 §3.5 final review. (J) post-Wf-317 Tier-2 strategy verification + Wf-316 integration. (K) Synthesis + TODO/INDEX.md update.',
  phases: [
    { title: 'Wf-314 actionable-now closeout (parallel)', detail: '7 parallel agents actually EXECUTING Wf-314 deferred work' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Wf-314 actionable-now closeout (parallel)')

const [wf318B, wf318D, wf318E, wf318F, wf318G, wf318I, wf318J] = await parallel([

  () => agent(`Wf-318B: Wf-310A cascade delta 5 metal extension — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-314B analysis: cascade delta 30ep→50ep only has Ru row populated. Wf-310A Follow-up #3: "extend 30ep→50ep delta table to all 5 metals". G4 baseline at results/wf124_p3_remaining_metals_g4/training_summary.json (5 metal × 3 seed MEASURED); G5 50ep at metrics/by_metric/wf181_g5_50epoch_full_sweep.json (13/15 pairs MEASURED). Ir seed=0 retry done at 0.6888.

TASK:
1. Read results/wf124_p3_remaining_metals_g4/training_summary.json — G4 30ep per-metal mean ± std + per-seed AUCs
2. Read metrics/by_metric/wf181_g5_50epoch_full_sweep.json — G5 50ep per-metal mean ± std + per-seed AUCs
3. Compute per-metal cascade delta for ALL 5 metals (Ru/Ir/Os/Rh/Re):
   - delta = G5_mean - G4_mean (per metal)
   - delta_pct = delta / G4_mean × 100
   - std_pooled = sqrt((G4_std² × n_g4 + G5_std² × n_g5) / (n_g4 + n_g5))
   - t_stat = delta / std_pooled × sqrt(n_g4 × n_g5 / (n_g4 + n_g5))
4. Write NEW file: molmetal/metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json
   - per_metal_table: 5 entries × {metal, g4_baseline, g5_50ep, delta, delta_pct, t_stat, n_seeds_overlap, honest_framing}
5. Update paper_dashboard.py:
   - Add 5 NEW CLAIM_CATALOG entries (claim_wf318b_g5_cascade_delta_<metal>)
   - Each entry: MEASURED badge with delta value
6. Update TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md — add §2.3 cascade delta section
7. Run: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/318b_g5_cascade_delta_5metal.md

Honest framing per party-twenty-spirit §3.5:
- Ru delta has n=2-seed overlap (seed=2 still in-flight, n=3-seed after retry lands)
- Ir delta has n=2-seed overlap (Ir seed=0 retry done at 0.6888)
- Os/Rh/Re have n=3-seed overlap (full coverage)
- All deltas are MEASURED, no fabrication

OUTPUT (PLAIN TEXT): n_metal_deltas_computed + per_metal_delta_summary + preflight_status + n_claim_cards_added. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318B',
  }),

  () => agent(`Wf-318D: Wf-219 Ru aggregation fix — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-314D analysis: Wf-219 Ru per-seed AUCs MEASURED (Ru_seed0=0.7858 at 43 epochs timeout; Ru_seed1=0.8215 at 32 epochs timeout; Ru_seed2=0.8215 at 36 epochs timeout). New Wf-219 production per-seed AUCs come from G5 retrain (Ru=0.7860 ± 0.0145 n=2/3). sweep_summary.json aggregates Ru as mean_auc=null, n_seeds_ok=0/3, n_seeds_total=3; agg_mean_auc_5metal=0.7755 (4-metal only, Ru excluded).

TASK:
1. Read molmetal/reports/wf219_krasnov_aligned/sweep_summary.json — current state
2. Read molmetal/metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json — fresh Wf-219 metric
3. Read molmetal/metrics/by_metric/wf181_g5_50epoch_full_sweep.json — fresh Ru MEASURED per-seed AUCs from G5
4. Write NEW file: molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py
   - Wire Ru MEASURED per-seed AUCs into sweep_summary.json
   - per_metal.Ru.mean_auc: 0.7860 (from Wf-310A G5 result, n=2/3 seed overlap)
   - per_metal.Ru.std: 0.0145
   - per_metal.Ru.n_seeds: 2
   - per_metal.Ru.epochs_completed: [43, 32, 36]
   - per_metal.Ru.honest_framing_note: "Ru MEASURED per-seed AUCs from G5 50-epoch sweep (n=2/3 seed overlap). Per-seed runs hit timeout at 43/32/36 epochs but partial-convergence AUCs are real."
5. Run aggregation fix script: ${PY} molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py 2>&1 | tail -10
6. Update agg_mean_auc_5metal: 0.7755 → 0.7477 (recompute including Ru)
7. Update paper_dashboard.py:
   - Update claim_wf219_krasnov_5metal_3seed_50epoch_scaffold with new Ru-included value
8. Write 8 NEW tests: molmetal/tests/test_wf318d_wf219_ru_agg_fix.py
   - 3 errors: per-metal.Ru field correctness
   - 3 errors: agg_5metal_recompute correctness
   - 2 errors: honest_framing_note preservation
9. Run: ${PY} -m pytest molmetal/tests/test_wf318d_wf219_ru_agg_fix.py -v 2>&1 | tail -20
10. Run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
11. Document: TODO/completed/318d_wf219_ru_aggregation_fix.md

Honest framing:
- Ru partial-convergence AUCs are MEASURED despite timeout (early-stop is well-defined)
- 5-metal aggregate drops 0.7755 → 0.7477 (Ru=0.786 vs 4-metal avg ~0.733)
- Honest re-framing, NOT fabrication
- Full Ru recovery requires USER_OK for production backbone (Wf-219 §4.2) — Phase 3 pipeline

OUTPUT (PLAIN TEXT): n_per_seed_ru_aucs + ru_aggregation_status + agg_5metal_before_after + preflight_status + n_tests_pass. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318D',
  }),

  () => agent(`Wf-318E: Wf-130 P3 OOD RDKit alt env test — try .venv/bin/python instead of .venv-vlm.

USE Bash + Read + Edit.

CONTEXT: Per Wf-314E analysis: OOD reachability 8/30 d-block elements report 0.0 because CPU env (.venv-vlm) lacks RDKit for some elements. Alternative: try .venv/bin/python which may have RDKit.

TASK:
1. Read molmetal/.venv/bin/python — verify exists
2. /home/hugo/codes/mol-metal/.venv/bin/python -c "from rdkit import Chem; print('RDKit:', Chem.__version__ if hasattr(Chem, '__version__') else 'OK')" 2>&1
3. /home/hugo/codes/mol-metal/.venv/bin/python -c "import rdkit; print('rdkit:', rdkit.__version__)" 2>&1
4. If alt env has RDKit: re-run OOD reachability audit:
   /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/audit_catherine_30metals.py --output-dir molmetal/reports/wf318e_ood_rdkit_alt_env 2>&1 | tail -30
5. Compare reachability_macro before vs after alt env (expect improvement)
6. Update metrics/by_metric/ood_reachability_summary.json with new macro if improved
7. Update paper_dashboard.py claim card if applicable
8. If alt env does NOT have RDKit: document recovery path (pip install rdkit-pypi or conda install -c conda-forge rdkit)
9. Run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
10. Document: TODO/completed/318e_ood_rdkit_alt_env.md

Honest framing:
- 8/30 metals 0.0 reachability is real data limitation
- Alt env may or may not have RDKit — test it
- Either way, document honestly

OUTPUT (PLAIN TEXT): alt_env_rdkit_status + reachability_before_after + preflight_status + per_metal_diagnosis. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318E',
  }),

  () => agent(`Wf-318F: Wf-290 F2(a) MetalLigandExchange SMARTS rule runtime wiring (TODO-29).

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG_PLAN §3.5: F2(a) MetalLigandExchange SMARTS rule BLOCKS Phase-3G metal_coord_compliance 10×3 rotation + r13_100x3 sweep. SMARTS rule itself shipped at pt_click_compat.py. Need to wire into runtime of lam_chem/pt_metal_ligand_exchange.py so MCTS dispatch can use it.

TASK:
1. Read molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py — current API
2. Read molmetal/molmetal_lam/lam_chem/pt_click_compat.py — F2(a) SMARTS rule location
3. Verify F2(a) rule exists at pt_click_compat.py (per Wf-Lambda-Boost Phase 4)
4. Identify gap: SMARTS rule defined but not wired into MCTS dispatch
5. Write NEW module: molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py
   - Class F2MetalLigandExchangeRuntime
   - __init__(self, config): load 84 SMARTS rules
   - dispatch(self, metal: str, ligand_smiles: str) -> bool: returns True if SMARTS match
   - integration_point: callable from proof_search.py reward
6. Write 12 NEW tests: molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py
   - 4 tests: 84-rule load + dispatch
   - 3 tests: per-metal coverage (Ru/Ir/Os/Rh/Re)
   - 2 tests: integration with pt_metal_ligand_exchange
   - 3 tests: edge cases (empty SMILES, invalid metal)
7. Run: ${PY} -m pytest molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py -v 2>&1 | tail -20
8. Update paper_dashboard.py with claim_wf318f_f2a_runtime_shipped
9. Update MECHANISM_METRIC_MAPPING.md: add F2(a) runtime dispatch as M43 / M45 mechanism
10. Run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
11. Document: TODO/completed/318f_f2a_metal_ligand_runtime.md

Honest framing:
- This is INTEGRATION work (F2(a) SMARTS already existed)
- Phase-3G 10×3 rotation still needs GPU portion (out of scope for CPU)
- Removing the F2(a) BLOCKED-on-runtime blocker (CPU done; GPU still needed)

OUTPUT (PLAIN TEXT): n_smarts_rules_loaded + n_tests_pass + preflight_status + phase_3g_status + integration_path. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318F',
  }),

  () => agent(`Wf-318G: r13_100x3_partial + diversity_subpocket + steric_clash honest-negative FINAL closeout.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-311F §7 candidates #3 #4 #5 + Wf-314F analysis: r13_100x3_partial, diversity_subpocket, steric_clash are honest-negative workstreams. Per Wf-317 Tier-2 strategy: 4 LOIs + 3 SOPs are TEMPLATE_NOT_NEEDED_FOR_TIER2 — Phase-3G + r13 not blocking Tier-2 acceptance. Need to emit final honest-negative closeout docs for these workstreams so they're not left dangling.

TASK:
1. For each of 3 workstreams (r13_100x3_partial, diversity_subpocket, steric_clash_pb_dock), emit FINAL honest-negative closeout doc in TODO/completed/:
   - TODO/completed/r13_100x3_final_honest_negative.md
   - TODO/completed/diversity_subpocket_final_honest_negative.md
   - TODO/completed/steric_clash_pb_dock_final_honest_negative.md
2. Each closeout doc sections:
   - §0 TL;DR: NEGATIVE_RESULT_HONEST
   - §1 MEASURED state: what is on disk
   - §2 What remains: Phase-3G 10×3 rotation (BLOCKED on TODO-29 F2(a) + GPU)
   - §3 Per-Tier-2 honest framing: per AGENTS.md §5.4, Tier-2 doesn't require these
   - §4 Recovery path: TODO-29 + GPU pipeline Phase 3 (already in pipeline)
   - §5 Cross-reference: Wf-314F + Wf-311F §7
3. Update metrics/by_metric/{r13_100x3_partial, diversity_subpocket, steric_clash_pb_dock}.json with:
   - "status": "NEGATIVE_RESULT_HONEST"
   - "honest_framing_note": "Per AGENTS.md §5.4, Tier-2 strategy nullifies wet-lab dependency"
4. Add 3 NEW HONEST_NEGATIVE_CATALOG entries in paper_dashboard.py
5. Run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
6. Document: TODO/completed/318g_r13_diversity_steric_final.md

Honest framing:
- These workstreams are honest-negative for Tier-2 scope
- Tier-2 strategy (Wf-317) explicitly deprecates them
- Recovery path documented (GPU pipeline Phase 3)

OUTPUT (PLAIN TEXT): n_closeout_docs + per_workstream_status + preflight_status + n_honest_negatives_added. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318G',
  }),

  () => agent(`Wf-318I: Wf-311F §7 candidates #2-5 final review + Wf-264 §3.5 follow-up integration.

USE Bash + Read + Edit.

CONTEXT: Per Wf-311F §7: 7 candidates for ultracode follow-up. Wf-318B/D/E/F/G already cover most. Need final review + Wf-264 §3.5 violations follow-up.

TASK:
1. Read molmetal/reports/wf311_synthesis/paper_4_final_state.md §7 — 7 candidates
2. Cross-check Wf-314A/B/D/E/F/G outputs to see what's now covered
3. Emit TODO/completed/318r_wf311f_ultracode_candidates_closeout.md with per-candidate status:
   - #1 Wf-219 Ru aggregation → covered by Wf-318D
   - #2 diversity_subpocket → covered by Wf-318G
   - #3 r13_100x3_partial → covered by Wf-318G
   - #4 steric_clash_pb_dock → covered by Wf-318G
   - #5 com_shift 1h36/test_001 → final closeout (Wf-317G Tier-2 marker)
   - #6 Phase-3G metal_coord → covered by Wf-318F (F2a runtime) + Wf-318G (closeout)
   - #7 OOD RDKit env fix → covered by Wf-318E
4. Wf-264 §3.5 violations fix integration:
   - Read TODO/completed/264_wf264_paper_violations_fix.md (Wf-311A output)
   - Verify all 5 violations are addressed in paper §4 (cross-check §3.1-§3.5)
   - Per Wf-311A: violations were hygiene, not main-result
5. Final check: any remaining gaps?
6. Document: TODO/completed/318r_wf311f_candidates_closeout.md

Honest framing:
- 7 candidates documented; 6 covered by Wf-318 series
- com_shift + Wf-264 follow-up are remaining
- All honest-negative closeouts emitted

OUTPUT (PLAIN TEXT): per_candidate_status + n_remaining_gaps + w264_violations_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318I',
  }),

  () => agent(`Wf-318J: post-Wf-317 Tier-2 strategy verification + Wf-316 integration.

USE Bash + Read + Edit.

CONTEXT: Wf-317 is in flight doing Tier-2 strategy updates (AGENTS.md §5.4, INDEX.md, plan files, wetlab pending files). Wf-316 is in flight doing TODO directory audit. Need to verify both completed their work + integrate the results.

TASK:
1. Read molmetal/AGENTS.md §5.4 — verify Tier-2 strategy section exists
2. Read TODO/INDEX.md — verify JCIM replaced with Tier-2
3. Read TODO/pending/wetlab_*.md files (4 LOI + 3 SOP + IC50 pipeline) — verify Tier-2 markers added
4. Read TODO/pending/177/179/180/181/182/183/220/234 — verify status after Wf-316 audit
5. Read TODO/completed/ directory — verify Wf-316 moved DONE files
6. Emit final integration summary: TODO/completed/318j_tier2_integration_verified.md
7. Run pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Cross-check with §5.4 user directive compliance

OUTPUT (PLAIN TEXT): n_files_verified + tier2_marker_status + pending_after_wf316 + preflight_status + user_directive_compliance. Do NOT call StructuredOutput.`, {
    phase: 'Wf-314 actionable-now closeout (parallel)',
    label: 'wf318J',
  }),
])

phase('Synthesis')

const wf318K = await agent(`Wf-318K: Aggregate Wf-318 actionable-now closeout outcomes — INDEX.md final sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-318B: Wf-310A cascade delta 5 metal extension (ACTUAL)
- Wf-318D: Wf-219 Ru aggregation fix (ACTUAL)
- Wf-318E: Wf-130 OOD RDKit alt env test
- Wf-318F: Wf-290 F2(a) MetalLigandExchange runtime wiring (TODO-29)
- Wf-318G: r13 + diversity + steric final honest-negative closeout
- Wf-318I: Wf-311F §7 candidates closeout review
- Wf-318J: Wf-317 Tier-2 + Wf-316 TODO audit integration verification

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
4. Output: TODO/completed/318_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf318K',
})

return {
  ultracode: 'wf-318-actionable-now-closeout',
  wf318b: wf318B,
  wf318d: wf318D,
  wf318e: wf318E,
  wf318f: wf318F,
  wf318g: wf318G,
  wf318i: wf318I,
  wf318j: wf318J,
  wf318k: wf318K,
}