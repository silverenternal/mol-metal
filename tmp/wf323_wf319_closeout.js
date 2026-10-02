export const meta = {
  name: 'wf-323-wf319-closeout-actual',
  description: 'Wf-323: Wf-319 actionable-now closeout ACTUAL EXECUTION + Wf-319 exposed problems. (A) Wf-319A M14 --lr-warmup-steps re-integration — flag was silently removed from metallo_drug_smoke_retrain.py harness (per WEAK_TO_STRONG §3.1). (B) Wf-319D v3 schema drift fix — metalcytotox_loader to handle both v3 (smiles/metal) and v4 (SMILES_Ligands/Metal) column names. (C) m8_full_pass_rate CPU fix — UFF pre-relax + Vina re-dock to lift from 0/15 to higher (per WEAK_TO_STRONG §2.6 Phase 2). (D) Wf-319F ablation_runner aggregate.csv integration — wire into paper_dashboard + section_5_ablation.json (actual). (E) TODO-21 Lambda × CFM coupling CPU scaffold verification (per WEAK_TO_STRONG §2.5 long-term + TODO/completed/57). (F) 5-pose Vina per-pocket CPU smoke (per WEAK_TO_STRONG §2.4 Success Rate). (G) TODO-134 Phase 2 Vina install via conda-forge retry (still BLOCKED, attempt recovery path). (H) Synthesis + INDEX.md update.',
  phases: [
    { title: 'Wf-319 closeout + Wf-319 exposed problems (parallel)', detail: '7 parallel agents actually EXECUTING Wf-319 deferred work + Wf-319 exposed problems' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('Wf-319 closeout + Wf-319 exposed problems (parallel)')

const [wf323A, wf323B, wf323C, wf323D, wf323E, wf323F, wf323G] = await parallel([

  () => agent(`Wf-323A: Wf-319A M14 --lr-warmup-steps re-integration — flag was silently removed from metallo_drug_smoke_retrain.py.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-319A honest-negative: M14 (LR warmup) toggle has been silently broken since the flag was removed from the harness. Per WEAK_TO_STRONG §3.1: M14 = LR warmup (linear 200 steps).

TASK:
1. Read molmetal/scripts/metallo_drug_smoke_retrain.py — current --lr-warmup-steps handling
2. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — current --lr-warmup-steps handling
3. Read molmetal/scripts/ablation_runner.py — how M14 toggle passes --lr-warmup-steps 0
4. Identify gap: flag removed from harness → M14 ablation silently broken
5. Fix: re-add --lr-warmup-steps CLI flag to harness(es)
6. Update ablation_runner.py to correctly toggle the flag
7. Write 8 NEW tests: molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py
   - 3 tests: flag acceptance (spectral correctness)
   - 3 tests: ablation_runner M14 toggle correctness
   - 2 tests: integration with ablation_runner smoke
8. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py -v 2>&1 | tail -20
9. Re-run ablation_runner smoke with M14 toggle functional: re-execute to verify
11. Run pre-flight: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
12. Document: TODO/completed/323a_m14_lr_warmup_reintegration.md

Honest framing:
- M14 was silently broken — fix is INTEGRATION work
- Fix is pure CPU
- No user OK needed (pure integration)

OUTPUT (PLAIN TEXT): n_tests_pass + preflight_status + m14_toggle_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323A',
  }),

  () => agent(`Wf-323B: Wf-319D v3 schema drift fix — metalcytotox_loader to handle both v3 + v4 column names.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-319D: v3 schema drift noted (smiles→SMILES_Ligands, metal→Metal). v4 normalized. Need loader to handle both v3 and v4 column names.

TASK:
1. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask.csv (v3 column names)
2. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv (v4 column names)
3. Read molmetal/data/metal_cytotox.py — current loader
4. Identify gap: loader may only handle v3 or v4, not both
5. Fix: write _has_column() synonym map (v3 → v4 column names) + apply in loader
6. Write 8 NEW tests: molmetal/tests/test_wf323b_v3_v4_schema_drift.py
   - 3 tests: v3 CSV loading
   - 3 tests: v4 CSV loading
   - 2 tests: synonym map correctness
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf323b_v3_v4_schema_drift.py -v 2>&1 | tail -20
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/323b_v3_v4_schema_drift_fix.md

Honest framing:
- v3 and v4 are both legitimate versions of the same data
- Loader should handle both for backward compatibility
- Pure CPU fix (no GPU)

OUTPUT (PLAIN TEXT): n_tests_pass + per_version_load_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323B',
  }),

  () => agent(`Wf-323C: m8_full_pass_rate CPU fix — UFF pre-relax + Vina re-dock to lift from 0/15.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-319C: m8_full_pass_rate = 0.0000 (Vina heavy-atom-only limitation per Wf-246D). Per WEAK_TO_STRONG §2.6 Phase 2: MMFF94 relax on 100×3 production sweep. Need CPU-side fix.

TASK:
1. Read molmetal/scripts/bench_vina_triple_rebench.py — current Vina bench
2. Read molmetal/scripts/wf315d_pb_smoke/m8_prod_1h36.json — 0/15 full pass result
3. Identify: Vina heavy-atom-only → need UFF pre-relax before Vina re-dock
4. Implement CPU smoke: take Wf-319B UFF-min Pt-pool conformers → Vina re-dock → m8_full pass rate check
5. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/wf323c_m8_full_cpu_fix.py --input-pool molmetal/reports/wf319b_uff_min_pt_pool/pool.json --pocket-id 1h36 --n-conformers 96 --out molmetal/reports/wf323c_m8_full/result.json --execute 2>&1 | tail -20
6. Compute m8_full_pass_rate after UFF pre-relax + Vina re-dock
7. Save: molmetal/reports/wf323c_m8_full/result.json (n_poses + pass rate)
8. Update metrics/by_metric/steric_clash_pb_dock_1h36.json status (if improved)
9. Add pre-flight gate: p3_m8_full_pass_rate_documented
11. Write 8 NEW tests: molmetal/tests/test_wf323c_m8_full_cpu_fix.py
12. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
13. Document: TODO/completed/323c_m8_full_cpu_fix.md

Honest framing:
- UFF pre-relax may lift m8_full_pass_rate from 0/15 to some %
- Per WEAK_TO_STRONG §2.6: PROJECTED +0.10-0.15
- If still 0/15: HONEST_NEGATIVE_UFF_INSUFFICIENT + recovery path (MMFF94 + GPU)
- CPU-only (no GPU)

OUTPUT (PLAIN TEXT): n_poses_pre_relaxed + m8_full_pass_rate_before_after + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323C',
  }),

  () => agent(`Wf-323D: Wf-319A ablation_runner aggregate.csv integration — wire into paper_dashboard + section_5_ablation.json (actual).

USE Bash + Read + Edit.

CONTEXT: Per Wf-319A: 12/12 ablation table aggregate.csv shipped. Per Wf-319F: actual integration into paper_dashboard + section_5_ablation.json needed.

TASK:
1. Read molmetal/molmetal/scripts/paper_dashboard.py — verify claim_wf319a_ablation_table_aggregate
2. Read metrics/by_paper_section/section_5_ablation.json — verify wf319a_ablation_runner_axis
3. Cross-check: ablation_runner output integrated into dashboard + section
4. Add NEW paper_dashboard CLAIM_CATALOG entries if missing:
   - claim_wf319a_m12_atom_loss_+204%
   - claim_wf319a_m13_bond_loss_+152%
   - claim_wf319a_m14_lr_warmup_honest_negative (if M14 fix in place per Wf-323A)
5. Update section_5_ablation.json if needed
6. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
7. Document: TODO/completed/323d_ablation_runner_integration.md

Honest framing:
- Wf-319A already added 1 claim card (claim_wf319a_ablation_table_aggregate)
- This is for adding per-strat claims (M12 / M13 / M14)
- Per WEAK_TO_STRONG: M12/M13 strong signals, M14 honest-negative

OUTPUT (PLAIN TEXT): n_claim_cards_added + section_5_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323D',
  }),

  () => agent(`Wf-323E: TODO-21 Lambda × CFM coupling CPU scaffold verification (per WEAK_TO_STRONG §2.5 long-term).

USE Bash + Read + Edit.

CONTEXT: Per TODO/completed/57_lambda_cfm_coupling.md: CPU Joint Inference Skeleton SHIPPED. Reward-weighting wiring into CFM velocity loss = 1-line adapter change (deferred to R18 GPU). Need CPU scaffold verification.

TASK:
1. Read TODO/completed/57_lambda_cfm_coupling.md — current ship surface
2. Read molmetal/molmetal_lam/reward/coupling_adapter.py — current coupling
4. Verify CPU scaffold:
   - Coupling adapter exists
   - Joint inference skeleton callable
   - Reward-weighting wiring exists
5. Write 8 NEW tests: molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py
   - 3 tests: coupling adapter correctness
   - 3 tests: joint inference skeleton
   - 2 tests: integration with downstream consumer
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf323e_lambda_cfm_coupling_cpu_scaffold.py -v 2>&1 | tail -20
7. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/323e_todo21_lambda_cfm_coupling_closeout.md

Honest framing:
- TODO-21 reward-weighting wiring = 1-line adapter change for GPU R18
- CPU scaffold verified
- Full integration requires GPU portion (out of scope)

OUTPUT (PLAIN TEXT): n_tests_pass + scaffold_status + preflight_status + gpu_portion_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323E',
  }),

  () => agent(`Wf-323F: 5-pose Vina per-pocket CPU smoke (per WEAK_TO_STRONG §2.4 Success Rate).

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG §2.4: m4 Success Rate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59) MEASURED 0/100 — Vina sub-criterion binds at 0. Need CPU smoke to verify SA/QED sub-criteria can lift rate when Vina improves.

TASK:
1. Read molmetal/scripts/cfm_property_panel.py — current SA/QED eval
2. Read metrics/by_metric/sa_score.json + qed_score.json (from Wf-315A)
3. Compute m4 Success Rate SA/QED sub-criterion (independent of Vina):
   - Per-conformer SA<2.8 + QED>0.5 rate
   - SA/QED sub-criterion rate
4. Save: molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json
5. Update metrics/by_metric/m4_success_rate.json (or create if missing)
6. Add pre-flight gate: p3_m4_sa_qed_subcriterion_documented
7. Write 6 NEW tests: molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/323f_m4_sa_qed_subcriterion.md

Honest framing:
- SA/QED sub-criterion can lift rate independent of Vina (per WEAK_TO_STRONG §2.4 Phase 1)
- Vina sub-criterion requires GPU portion (out of scope)
- m4 Success Rate = SA/QED sub-rate ∩ Vina sub-rate

OUTPUT (PLAIN TEXT): n_conformers + sa_qed_rate + per_pool_distribution + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323F',
  }),

  () => agent(`Wf-323G: TODO-134 Phase 2 Vina install via conda-forge retry.

USE Bash + Read + Edit.

CONTEXT: Per Wf-320H: TODO-134 Phase 2 still BLOCKED on Vina 1.2.7 install. Recovery path: conda-forge vina. Attempt retry.

TASK:
1. which conda 2>&1
2. which mamba 2>&1
3. which micromamba 2>&1
4. Try conda install -c conda-forge vina --dry-run 2>&1 | head -20
5. Try pip install vina --dry-run 2>&1 | head -10
6. If install feasible: try actual install (user OK may be needed)
7. Document recovery path + installation status
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/323g_todo_134_vina_install_attempt.md

Honest framing:
- Vina install requires user sudo OK (system package)
- Recovery path documented but not forced
- HONEST_NEGATIVE_BLOCKED if install not feasible

OUTPUT (PLAIN TEXT): conda_status + pip_status + vina_install_attempt + recovery_path + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-319 closeout + Wf-319 exposed problems (parallel)',
    label: 'wf323G',
  }),
])

phase('Synthesis')

const wf323H = await agent(`Wf-323H: Aggregate Wf-323 Wf-319 closeout + Wf-319 exposed problems outcomes — INDEX.md final sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-323A: Wf-319A M14 --lr-warmup-steps re-integration
- Wf-323B: Wf-319D v3 schema drift fix
- Wf-323C: m8_full_pass_rate CPU fix (UFF pre-relax)
- Wf-323D: ablation_runner aggregate.csv integration
- Wf-323E: TODO-21 Lambda × CFM coupling CPU scaffold
- Wf-323F: 5-pose Vina per-pocket CPU smoke
- Wf-323G: TODO-134 Phase 2 Vina install retry

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 + §22
4. Output: TODO/completed/323_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf323H',
})

return {
  ultracode: 'wf-323-wf319-closeout-actual',
  wf323a: wf323A,
  wf323b: wf323B,
  wf323c: wf323C,
  wf323d: wf323D,
  wf323e: wf323E,
  wf323f: wf323F,
  wf323g: wf323G,
  wf323h: wf323H,
}