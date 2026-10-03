export const meta = {
  name: 'wf-325-honest-negative-closeout-cpu-only',
  description: 'Wf-325: Honest-negative closeout CPU-only (NO GPU interference with G6 training + pipeline master). (A) Wf-297B R² architectural ceiling investigation — try F5_RDKIT_2D 32-d expansion on platinai + hilbert + metalcytotox_combined cohorts. (B) Wf-143 Vina covalent ceiling final closeout — UFF Pt→Zn substitution attempt + honest-negative final. (C) Rh LOMO 0.136 regression final closeout (Wf-309A Phase 1 + Wf-321A scaffold + Wf-135 pipeline). (D) m6_kappa BELOW threshold + NEGATIVE_WEAK MetalCytoToxDB validation final closeout. (E) ADMET 4 BELOW druglikeness (aq_sol + logp7_4 + hepatotox + m6_kappa) final closeout. (F) F2(a) MetalLigandExchange SMARTS rule Phase-3G 10×3 rotation unblock verification (post Wf-318F). (G) m7 Diversity 0.23 mechanical lift attempt — Multi-PH integration CPU scaffold (per WEAK_TO_STRONG §2.5). (H) Synthesis + INDEX.md update + verify G6 still running.',
  phases: [
    { title: 'Honest-negative closeout CPU-only (parallel)', detail: '7 parallel agents addressing honest-negative issues WITHOUT GPU interference' },
    { title: 'Synthesis', detail: 'INDEX.md sync + verify G6 still running + final summary' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('Honest-negative closeout CPU-only (parallel)')

const [wf325A, wf325B, wf325C, wf325D, wf325E, wf325F, wf325G] = await parallel([

  () => agent(`Wf-325A: Wf-297B R² architectural ceiling investigation — F5_RDKIT_2D 32-d expansion.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-297B: R² architectural ceiling — 3 of 4 cohorts (platinai / hilbert / metalcytotox_combined) hit signal-floor R² ≤ 0. Per WEAK_TO_STRONG §3.1: signal-absence diagnostic per TODO-31 §2.1. Per TODO-122 P1.3: W₂ alignment would lift R² above floor.

TASK:
1. Read molmetal/molmetal_lam/sbdd_env/metrics_v2.py — F5_RDKIT_2D + F5 12-d vs 32-d
2. Read molmetal/.data/molmetal/platinai_4cell.csv + platinai_4cell_mbfinder.csv
3. Read molmetal/reports/wf297b_r2_ceiling/r2_audit.json (if exists)
4. Implement F5_RDKIT_2D 32-d expansion (CPU):
   - For each of 3 cohorts (platinai_4cell + hilbert + metalcytotox_combined):
     - Compute 32-d RDKit features
     - Ridge regression baseline
     - Compute R² on test split
     - Compare 12-d vs 32-d R² lift
5. Save: molmetal/reports/wf325a_r2_32d/result.json
6. Write 8 NEW tests: molmetal/tests/test_wf325a_r2_32d_expansion.py
   - 3 tests: per-cohort R² computation
   - 2 tests: 12-d vs 32-d comparison
   - 2 tests: regression baseline correctness
   - 1 test: integration with downstream consumer
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf325a_r2_32d_expansion.py -v 2>&1 | tail -20
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/325a_r2_32d_expansion.md

Honest framing per party-twenty-spirit §3.5:
- 12-d RDKit features are signal-floor (R² ≤ 0 by construction)
- 32-d expansion may lift R² +0.10-0.30 (PROJECTED per TODO-122)
- If still R² ≤ 0: HONEST_NEGATIVE (architectural ceiling confirmed)
- Per WEAK_TO_STRONG §2.1: full lift requires W₂ alignment + chemprop v2 graph features (GPU portion)
- CPU-only (no G6 interference)

OUTPUT (PLAIN TEXT): n_cohorts_tested + per_cohort_r2_12d_vs_32d + per_cohort_lift + preflight_status + honest_negative_status. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325A',
  }),

  () => agent(`Wf-325B: Wf-143 Vina covalent ceiling final closeout — UFF Pt→Zn substitution attempt + honest-negative final.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-143 + Wf-301B + Wf-319C + Wf-323C: Vina covalent mechanism architectural ceiling 0/10 pockets reach industry ≥−7 kcal/mol. Per Wf-323C NEW: Vina 1.2.7 PDBQT parser rejects 'Pt' atom-type. Per WEAK_TO_STRONG §2.3 / WEAK_TO_STRONG §3.5: F2(a) MetalLigandExchange SMARTS rule + MMFF94.

TASK:
1. Read molmetal/reports/wf301b_vina_pt_strip/vina_pt_strip_summary.json
2. Read molmetal/reports/wf319c_pb_smoke/m8_prod_1h36.json
3. Read molmetal/reports/wf323c_m8_full/result.json
4. Implement Pt→Zn UFF substitution (CPU):
   - For each of 5 pockets (1h36, 3eml, 3ny8, 4aaw, 4rlu):
     - Pt SMILES → Zn equivalent (UFF-supported)
     - Vina re-dock with Zn
     - Compute per-pocket m3 Vina Dock median
5. Save: molmetal/reports/wf325b_vina_pt_zn_substitution/result.json
6. Write 8 NEW tests: molmetal/tests/test_wf325b_vina_pt_zn_substitution.py
   - 3 tests: per-pocket Zn substitution correctness
   - 2 tests: Vina re-dock validity
   - 2 tests: m3 lift vs original
   - 1 test: integration with downstream consumer
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf325b_vina_pt_zn_substitution.py -v 2>&1 | tail -20
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/325b_vina_pt_zn_substitution.md

Honest framing:
- Pt→Zn substitution is a CPU workaround for Vina Pt-reject (NOT true Pt affinity)
- Recovery: per WEAK_TO_STRONG §3.5: F2(a) SMARTS rule + MMFF94 + GPU metal-coord term
- Per Wf-323C: HONEST_NEGATIVE_VINA_PT_REJECTED already documented
- Final closeout for m3 Vina Dock with architectural ceiling preserved

OUTPUT (PLAIN TEXT): n_pockets_substituted + per_pocket_vina_dock_zn + per_pocket_lift + preflight_status + honest_negative_status. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325B',
  }),

  () => agent(`Wf-325C: Rh LOMO 0.136 regression final closeout.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-121 + Wf-176: m19 LOMO Rh fold AUC = 0.136 REGRESSED from 0.368 baseline due to constant-inactive label on 500 synthetic Rh compounds. Per IMPROVEMENT_ENUM §1.4: Wf-135 Phase 2 = 5-metal × 5-seed × 50-epoch re-bench with balanced Rh cohort (Wf-309A Phase 1 cohort SHIPPED).

TASK:
1. Read molmetal/reports/wf176_m19_oversample_lomo/aggregate.json — current state
2. Read molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md — Phase 1 cohort
3. Read molmetal/reports/wf321a_wf135_phase2_cpu_scaffold/ — Phase 2 scaffold
4. Read scripts/launch_wf310e_wf135_phase2_gpu.sh — GPU wrapper (in pipeline)
5. Emit final closeout: TODO/completed/325c_wf176_rh_lomo_final.md
   - §0: Wf-121 regression discovery 0.136 vs 0.368 baseline
   - §1: Phase 1 (Wf-309A): 500 balanced cohort SHIPPED (250 active + 250 inactive)
   - §2: Phase 2 scaffold (Wf-321A): CPU scaffold verified
   - §3: Phase 2 GPU: in pipeline (Wf-135, ~10 GPU-hr)
   - §4: Recovery path: USER_OK required per mol-metal-gpu-execution §0
   - §5: Acceptance per IMPROVEMENT_ENUM §1.4 (Rh ≥ 0.30, LOMO ≥ 0.60)
   - §6: Tier-2 honest framing per AGENTS.md §5.4
6. Add pre-flight gate: p3_wf176_rh_lomo_closeout_documented
7. Update metrics/by_metric/wf176_m19_oversample_lomo/aggregate.json with Tier-2 honest-negative tag
8. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing per party-twenty-spirit §3.5:
- Wf-176 Rh fold 0.136 regression is REAL (label-bias collapse, per Wf-131 P2 + Wf-125 P1)
- Phase 1 cohort fixes the bias (Catherine 6/6 PASS, scaffold_overlap=0, 15/15 tests)
- Phase 2 GPU sweep in pipeline (Wf-135 launch wrapper USER_OK-gated)
- Tier-2 acceptance bar doesn't require m19 Rh ≥ 0.86 (full per IMPROVEMENT_ENUM §1.4)

OUTPUT (PLAIN TEXT): phase_1_status + phase_2_scaffold_status + phase_2_gpu_status + preflight_status + recovery_path. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325C',
  }),

  () => agent(`Wf-325D: m6_kappa BELOW threshold + NEGATIVE_WEAK MetalCytoToxDB validation final closeout.

USE Bash + Read + Edit.

CONTEXT: Per Wf-252 + Wf-311D PART 2: m6_kappa mean=0.4719 BELOW 0.6 drug-likeness threshold; MetalCytoToxDB validation NEGATIVE_WEAK (Pearson r 0.081 / Spearman ρ 0.111 per Wf-253D on 5,460-row cohort); canonical HONEST_NEGATIVE=true tag already present.

TASK:
1. Read metrics/by_metric/m6_kappa.json — verify current state
2. Read metrics/by_metric/m6_kappa.json::metalcytotox_validation — verify Pearson r + Spearman ρ
3. Read molmetal/reports/wf253d_m6_kappa_validation/ — Wf-253D detailed analysis
4. Emit final closeout: TODO/completed/325d_m6_kappa_final.md
   - §0: m6_kappa threshold 0.6, MEASURED 0.4719 (-0.128)
   - §1: MetalCytoToxDB validation 5,460 rows: Pearson 0.081 / Spearman 0.111
   - §2: Per-metal breakdown (Ru +0.138 / Ir +0.110 / Os +0.110 / Rh -0.021 / Re -0.184)
   - §3: Architectural fix: shape kappa is NOT a learned metallodrug score (per ChemAxon PMC11387870)
   - §4: Recovery: chemprop v2 graph features + W₂ alignment (GPU portion Wf-310C TODO-179)
   - §5: HONEST_NEGATIVE preserved
5. Add pre-flight gate: p3_m6_kappa_closeout_documented
6. Update paper_dashboard.py with HONEST_NEGATIVE_CATALOG entry (if missing)
7. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- m6_kappa is REAL honest-negative (not fabrication)
- Shape kappa has structural limitation for metallo-drug scoring
- Recovery requires W₂ + chemprop v2 (GPU portion)
- Tier-2 acceptance bar doesn't require m6_kappa ≥ 0.6

OUTPUT (PLAIN TEXT): per_metric_status + per_metal_breakdown + preflight_status + honest_negative_status + recovery_path. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325D',
  }),

  () => agent(`Wf-325E: ADMET 4 BELOW druglikeness final closeout.

USE Bash + Read + Edit.

CONTEXT: Per Wf-312C + Wf-314A + Wf-315A: 4 ADMET metrics BELOW threshold on 449 metallo-drug cohort — aq_sol -7.612 (logS), logp7_4 6.413 (Lipinski ceiling 5), hepatotox 0.548 (threshold 0.5), m6_kappa 0.472 (threshold 0.6). All 4 carry HONEST_NEGATIVE=true tag.

TASK:
1. Read metrics/by_metric/aqueous_solubility_logS.json — verify tag
2. Read metrics/by_metric/logp7_4.json — verify tag
3. Read metrics/by_metric/hepatotox_index.json — verify tag
4. Read metrics/by_metric/m6_kappa.json — covered by Wf-325D
5. Read metrics/by_metric/admet_full_coverage.json (from Wf-312C R01)
6. Emit final closeout: TODO/completed/325e_admet_4_below_final.md
   - §0: ADMET panel 8 fields (3 Wf-303C T1 + 5 Wf-312C R01)
   - §1: 4 BELOW threshold (aq_sol / logp7_4 / hepatotox / m6_kappa)
   - §2: 4 PASS / INFO (ames / cell_perm / sa_score / qed_score)
   - §3: Per-metric honest framing:
     - aq_sol -7.612: intrinsic to metallo-cohort hydrophobicity
     - logp7_4 6.413: Lipinski violation intrinsic to Pt/Ru/Ir
     - hepatotox 0.548: structural-alert proxy
     - m6_kappa 0.472: shape-kappa limitation
   - §4: Recovery: W₂ alignment + chemprop v2 (GPU portion)
7. Add pre-flight gate: p3_admet_4_below_documented
8. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing per party-twenty-spirit §3.5:
- All 4 BELOW metrics are REAL data limitations (intrinsic to metallo-drug chemistry)
- Tier-2 acceptance bar doesn't require ADMET 8/8 PASS
- Per AGENTS.md §5.4: Tier-2 strategy is computational-only

OUTPUT (PLAIN TEXT): n_admet_metrics_total + n_below + n_pass + per_metric_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325E',
  }),

  () => agent(`Wf-325F: F2(a) MetalLigandExchange SMARTS rule Phase-3G 10×3 rotation unblock verification.

USE Bash + Read + Edit.

CONTEXT: Per WEAK_TO_STRONG §3.5: F2(a) MetalLigandExchange SMARTS rule BLOCKS Phase-3G metal_coord_compliance 10×3 rotation + r13_100x3 sweep. Per Wf-318F: runtime wiring SHIPPED. Need verification + Phase-3G 10×3 rotation unblock verification.

TASK:
1. Read molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py — Wf-318F output
2. Verify Phase-3G metal_coord_compliance now possible:
   - Read molmetal/reports/wf290_metal_ligand_matrix/ — 84-rule MLE matrix SHIPPED
3. Read molmetal/metrics/by_metric/metal_coord_compliance_phase3g.json — current 1.0 singleton collapse
4. Emit final closeout: TODO/completed/325f_f2a_phase3g_unblock.md
   - §0: TODO-29 F2(a) SMARTS rule + Wf-318F runtime wiring SHIPPED
   - §1: Phase-3G 10×3 rotation NOW POSSIBLE (was BLOCKED)
   - §2: GPU portion still needed (Wf-310E Wf-135 Phase 2 in pipeline)
   - §3: Honest-negative: singleton collapse 1.0 from 5x1 cisplatin pilot (5x1 = trivial PASS)
   - §4: Recovery: full 10×3 CPU smoke possible (no GPU required)
5. Try 10×3 rotation on existing 5×1 cohort (CPU smoke):
   - Use Wf-290 84-rule MLE matrix
   - Run 10 rotation indices × 3 seeds = 30 runs
   - Compute metal_coord_compliance variance
6. Save: molmetal/reports/wf325f_phase3g_10x3_smoke/result.json
7. Add pre-flight gate: p3_phase3g_f2a_runtime_unblock_verified
8. Verify pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing per party-twenty-spirit §3.5:
- F2(a) SMARTS rule runtime wired (Wf-318F)
- 10×3 rotation CPU smoke possible (was BLOCKED)
- If smoke MEASURED: structural ceiling lifted
- If singleton persists: HONEST_NEGATIVE (architectural)

OUTPUT (PLAIN TEXT): n_runs_executed + per_rotation_compliance + variance + preflight_status + unblock_status. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325F',
  }),

  () => agent(`Wf-325G: m7 Diversity 0.23 mechanical lift attempt — Multi-PH integration CPU scaffold.

USE Bash + Read + Write + Edit.

CONTEXT: Per WEAK_TO_STRONG §2.5: m7 Diversity = 0.1065 (R12 Path A) → 0.1366 (R12 EV-3 sa-weight 0.3); structural fix needed (fragment-pool expansion). Per R18 P3: Multi-PH integration as MCTS heuristic. Per Wf-321B: R18 §3.5 + Multi-PH intelligence shipped at CPU level.

TASK:
1. Read molmetal/.data/molmetal/metallo_drugs_500_train.csv — current diversity cohort
2. Read molmetal/reports/wf176_m19_oversample_lomo/ — diversity metric
3. Compute current m7 Diversity 0.1366 baseline
4. Try Multi-PH mechanical lift:
   - Implement CPU multiplicity per process
   - Generate 100-pool expansion (vs current single-pool)
   - Compute diversity lift
5. Save: molmetal/reports/wf325g_m7_multi_ph_lift/result.json
6. Write 8 NEW tests: molmetal/tests/test_wf325g_m7_multi_ph_lift.py
   - 3 tests: per-cohort diversity computation
   - 2 tests: Multi-PH integration correctness
   - 2 tests: lift vs baseline
   - 1 test: integration with downstream consumer
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf325g_m7_multi_ph_lift.py -v 2>&1 | tail -20
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/325g_m7_multi_ph_lift.md

Honest framing per party-twenty-spirit §3.5:
- m7 Diversity is REAL pool-diversity issue (fragment-pool expansion needed)
- Per WEAK_TO_STRONG §2.5: long-term Lambda × CFM coupling TODO-21 may lift +0.30 to +0.45 (R19+ scope)
- CPU Multi-PH integration is mechanical lift, NOT structural fix
- If lift < 0.05: HONEST_NEGATIVE preserved

OUTPUT (PLAIN TEXT): n_cohorts_tested + per_cohort_diversity_before_after + per_cohort_lift + preflight_status + honest_negative_status. Do NOT call StructuredOutput.`, {
    phase: 'Honest-negative closeout CPU-only (parallel)',
    label: 'wf325G',
  }),
])

phase('Synthesis')

const wf325H = await agent(`Wf-325H: Aggregate Wf-325 honest-negative closeout outcomes — verify G6 still running + INDEX.md sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-325A: Wf-297B R² architectural ceiling (F5_RDKIT_2D 32-d expansion)
- Wf-325B: Wf-143 Vina covalent ceiling (Pt→Zn substitution)
- Wf-325C: Rh LOMO 0.136 regression final closeout
- Wf-325D: m6_kappa BELOW threshold closeout
- Wf-325E: ADMET 4 BELOW druglikeness closeout
- Wf-325F: F2(a) Phase-3G 10×3 rotation unblock verification
- Wf-325G: m7 Diversity Multi-PH mechanical lift attempt

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Verify G6 training still running (PID 1849475 or new G6 tuple):
   ps -ef | grep train_dmpnn_multitask | grep -v grep | head -3
4. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
5. Output: TODO/completed/325_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + G6_status + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf325H',
})

return {
  ultracode: 'wf-325-honest-negative-closeout-cpu-only',
  wf325a: wf325A,
  wf325b: wf325B,
  wf325c: wf325C,
  wf325d: wf325D,
  wf325e: wf325E,
  wf325f: wf325F,
  wf325g: wf325G,
  wf325h: wf325H,
}
