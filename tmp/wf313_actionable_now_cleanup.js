export const meta = {
  name: 'wf-313-actionable-now-cleanup',
  description: 'Wf-313: Comprehensive actionable-now TODO cleanup (NOT covered by Wf-309/310/311/312 + GPU pipeline). (A) Wf-134 m15 Docking RMSD redesign Phase 1 CPU — 5-pocket ligand SMILES extraction + InChIKey validation (Phase 2 deferred on Vina install). (B) Wetlab SOPs (3) + 4 LOI consolidation + Hugo ORCID placeholder final. (C) TODO-234 frontier math Day 3-5 CPU — 4 integration smoke tests for 4 frontier adapters + per-adapter research-paper ref doc. (D) TODO-220 §6.3/§6.4 acceptance gates + 5-metal temporal + 9-cell NCI-60 closeout docs. (E) MECHANISM_METRIC_MAPPING update M22-M44 (23 new mechanisms) + regression-debug table. (F) chemprop v2 featurization fix (TODO-155 §3) — precompute features once per epoch. (G) Wf-123 P3 downstream metric lift dashboard for 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125). (H) R20 Phase 0 closeout — verify 3 loaders (binding_db.py + drugood_loader.py + nci60.py) have ≥12 tests + API doc. (I) Synthesis + TODO/INDEX.md update.',
  phases: [
    { title: 'Actionable-now TODO cleanup (parallel)', detail: '8 parallel agents covering CPU actionable work NOT in GPU pipeline' },
    { title: 'Synthesis', detail: 'Aggregate + TODO/INDEX.md sync + completion docs' },
  ],
}

const PY = '/home/hugo/codes/CadButEaas/.venv-vlm/bin/python'

phase('Actionable-now TODO cleanup (parallel)')

const [wf313A, wf313B, wf313C, wf313D, wf313E, wf313F, wf313G, wf313H] = await parallel([

  () => agent(`Wf-313A: Wf-134 m15 Docking RMSD redesign Phase 1 CPU — 5-pocket ligand SMILES extraction + InChIKey validation.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/pending/134_m15_docking_rmsd_redesign.md §1.3: Phase 1 = 5/5 pocket ligand SMILES extraction + valid + InChIKey consistent. Phase 2 (Vina redocking) BLOCKED on Vina 1.2.7 install — per Wf-309A discovery: which vina → not found; conda-forge recovery needed. Phase 1 is CPU-only actionable now.

TASK:
1. Read TODO/pending/134_m15_docking_rmsd_redesign.md §1.3 (Phase 1 spec)
2. Read molmetal/scripts/docking_redesign/ — existing extraction scripts (if any)
3. Identify 5 pockets + per-fold discovery (1h36, 3eml, 3ny8, 4aaw, 4rlu)
4. For each pocket, extract ligand SMILES from receptor PDB via RDKit (Biopython fallback)
6. Validate per-pocket:
   - SMILES parses successfully (RDKit Chem.MolFromSmiles)
   - InChIKey consistent across extraction methods (RDKit vs raw PDB)
   - No invalid atoms / charges
7. Save per-pocket ref set:
   - molmetal/data/m15_redesign_ref/pocket_<n>/ref_ligand.smi
   - molmetal/data/m15_redesign_ref/pocket_<n>/ref_ligand_inchi.json
   - molmetal/reports/wf313a_m15_redesign_phase1/extraction_summary.json
8. Write 12 tests: molmetal/tests/test_wf313a_m15_redesign_phase1.py
   - 5 tests: per-pocket SMILES extraction
   - 3 tests: InChIKey consistency
   - 2 tests: schema validation
   - 2 tests: integration with downstream metric JSON
9. Run: ${PY} -m pytest molmetal/tests/test_wf313a_m15_redesign_phase1.py -v 2>&1 | tail -20
10. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
11. Document: TODO/completed/134_wf313a_m15_phase1_closeout.md
12. Honest-negative close: Phase 2 still BLOCKED — document recovery path (conda-forge vina or MOLMETAL_VINA_BIN env override per Wf-84 P2 stub_warning)

Honest framing per party-twenty-spirit §3.5:
- Phase 1 IS achievable on CPU (no Vina dependency)
- Phase 2 (Vina redocking) is honest-negative BLOCKED until Vina install
- Recovery path: conda-forge vina install (user OK + ~10 min wall)

OUTPUT (PLAIN TEXT): n_pockets_extracted + n_smiles_valid + n_inchi_consistent + n_tests_pass + phase2_blocked_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313A',
  }),

  () => agent(`Wf-313B: Wetlab SOPs (3) + 4 LOI consolidation + Hugo ORCID placeholder final.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/pending/wetlab_sop_{cytotoxicity,cellular_uptake,binding_affinity}_assay.md: 3 SOP templates exist as institutional-tailored drafts. Per TODO/pending/wetlab_loi_{rusanov_2025_platinai,pku_cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md + wetlab_loi_consolidation_review.md + wetlab_cold_email_template.md: 4 LOI drafts exist + send protocol. Per Wf-310G: cold-email template + send package SHIPPED.

TASK PART 1 — wetlab SOPs enhancement:
1. Read all 3 wetlab SOP templates + the Wf-28 generic versions they reference
2. For each SOP, add edge-case handling:
   - Cytotoxicity: viability edge cases (negative OD, sigmoidal plateau, batch effects)
   - Cellular uptake: ICP-MS matrix interference, digestion blank, IS recovery
   - Binding affinity: SPR regeneration optimization, ITC stoichiometry fitting
3. Add 8 NEW tests per SOP (3 × 8 = 24 tests):
   - molmetal/tests/test_wf313b_wetlab_sop_cytotoxicity.py
   - molmetal/tests/test_wf313b_wetlab_sop_cellular_uptake.py
   - molmetal/tests/test_wf313b_wetlab_sop_binding_affinity.py
4. Each test set covers: edge case validation + parameter range checks + integration with template_ic50_pipeline

TASK PART 2 — 4 LOI consolidation final + Hugo ORCID placeholder:
1. Read wetlab_loi_consolidation_review.md (Wf-308D output)
2. Read all 4 LOI drafts + wetlab_cold_email_template.md
3. Write TODO/completed/wetlab_loi_send_ready_package_final.md
   - Consolidate: send order + per-LOI checklist + 4 PI email placeholders (TBD — Hugo)
   - Hugo ORCID placeholder: 0000-0000-0000-0000 (TBD)
   - Per-party send timing: low-friction first (Rusanov), then PKU, then 协和, then ORNL
5. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 3 — IC50 pipeline template enhancement:
1. Read TODO/pending/wetlab_ic50_analysis_pipeline.md + molmetal/scripts/template_ic50_pipeline.py
2. Add 4 NEW compound × cell matrix analyses:
   - Cisplatin-class 4-cell MCF7/A549/A2780/A2780cis
   - Ru-class 4-cell (A549, MCF7, A2780, A2780cis)
   - Ir-class 4-cell
   - Pt-class 9-cell NCI-60 (subset)
3. Per-matrix: 6 tests for fit quality + 95% CI + cross-validation against D-MPNN predictions
4. Save: molmetal/reports/wf313b_wetlab_ic50_pipeline_enhancement.json

Honest framing:
- Wetlab SOPs are TEMPLATES (no wet-lab data collected)
- LOI send is user's responsibility per TODO-84
- IC50 pipeline enhancement is template-level (no real wet-lab data)

OUTPUT (PLAIN TEXT): n_wetlab_sop_tests + n_loi_audit_items + orcid_placeholder_status + n_ic50_matrix_tests + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313B',
  }),

  () => agent(`Wf-313C: TODO-234 frontier math Day 3-5 CPU — 4 integration smoke tests + per-adapter research-paper ref doc.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/pending/234_wf234_frontier_math_audit_integration.md + Wf-309B + Wf-310F: 4 frontier math adapters already composed via molmetal/adapters/frontier_math_integration.py (FrontierMathIntegration dataclass). Per Wf-310F Day 1-2: 4 smoke tests already written (test_wf309b_frontier_math_integration.py PASS). Day 3-5: paper-grade integration smoke tests + per-adapter research-paper reference doc are CPU-only actionable.

TASK PART 1 — Day 3-5 integration smoke tests (8 NEW tests):
1. Read molmetal/adapters/frontier_math_integration.py — factory class
2. Read molmetal/adapters/{multi_ph_integration,se3_lipman_integration,se3_sb_integration,transport_relaxation_integration}.py — standalone adapters
3. Write 8 NEW integration smoke tests: molmetal/tests/test_wf313c_frontier_math_integration_e2e.py
   - test_multi_ph_pocket_augmentation (PH feature extraction + concatenation + downstream shape)
   - test_se3_lipman_solver_tangent_correctness (verify SE(3) tangent-space ops preserve rotation)
   - test_se3_sb_decoder_endpoint_distribution (verify Schrödinger Bridge endpoints match training distribution)
   - test_transport_relaxation_endpoint_term (verify ε-annealing from 1.0 → 0.0 monotonic decrease)
   - test_joint_frontier_integration_no_nan (full FrontierMathIntegration instantiation + forward pass without NaN)
   - test_research_only_modules_excluded (verify 3 research-only modules NOT in factory surface)
   - test_factory_method_consistency (verify get_* methods return expected types)
   - test_frontier_integration_metrics (verify per-component metric emission)
4. Run: ${PY} -m pytest molmetal/tests/test_wf313c_frontier_math_integration_e2e.py -v 2>&1 | tail -20

TASK PART 2 — per-adapter research-paper reference doc:
1. For each of 4 adapters, write 1-page research-paper reference:
   - molmetal/reports/wf313c_frontier_math_paper_refs/multi_ph_integration_ref.md
   - molmetal/reports/wf313c_frontier_math_paper_refs/se3_lipman_integration_ref.md
   - molmetal/reports/wf313c_frontier_math_paper_refs/se3_sb_integration_ref.md
   - molmetal/reports/wf313c_frontier_math_paper_refs/transport_relaxation_integration_ref.md
2. Each ref doc: paper citation + API surface + honest reproduction caveats + drop-in integration points
3. Aggregate: molmetal/reports/wf313c_frontier_math_paper_refs/README.md (index)

TASK PART 3 — pre-flight gate addition:
1. p3_frontier_math_day_3_5_integration_tests (verify 8 tests PASS)
2. p3_frontier_math_paper_refs_present (verify 4 ref docs + README exist)
3. Update pre_flight_check.py gate counter

Honest framing:
- Day 3-5 is CPU smoke — full paper-grade integration is GPU portion (Day 7 in Wf-310F)
- 3 research-only modules honestly excluded per Wf-305B / Wf-306B
- Paper refs are placeholder citations — actual paper §3 text is PAPER_BLOCKED per TODO-84

OUTPUT (PLAIN TEXT): n_tests_pass + n_paper_ref_docs + preflight_status + n_research_only_excluded + day_3_5_status. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313C',
  }),

  () => agent(`Wf-313D: TODO-220 §6.3 + §6.4 acceptance gates — wire Wf-310C GPU sweep results into 5-metal temporal + 9-cell NCI-60 closeout docs.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-310C: TODO-220 driver CPU smokes MEASURED (temporal 5/5 mean_auc=0.6810; NCI-60 9/9 mean_auc=0.5617). Per Wf-310C: 3 GPU wrappers READY (TODO-179 + TODO-220 + Wf-219 §4.1) — already in scripts/gpu_pipeline_master.sh Phase 3. After Phase 3 completes, full GPU sweep results will land. Need CPU closeout docs §6.3 + §6.4 to be ready for GPU results integration.

TASK:
1. Read TODO/pending/220_wf219_temporal_split_nci60_driver.md §1-§2 (gap description)
2. Read Wf-310C outputs:
   - molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json (5-metal temporal smoke)
   - molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json (9-cell NCI-60 smoke)
3. Write §6.3 acceptance gates doc: TODO/completed/220_wf220_temporal_split_acceptance.md
   - §6.3.1: Smoke acceptance (5/5 metal × 1-seed × 1-epoch CPU MEASURED)
   - §6.3.2: GPU acceptance (5/5 metal × 5-seed × 50-epoch — pending Phase 3 execution)
   - §6.3.3: per-metal mean ± std acceptance gate
   - §6.3.4: Krasnov 2026 protocol fidelity (temporal train<2023 test>=2024)
4. Write §6.4 acceptance gates doc: TODO/completed/220_wf219_nci60_9cell_acceptance.md
   - §6.4.1: Smoke acceptance (9/9 cell × 1-seed × 1-epoch CPU MEASURED)
   - §6.4.2: GPU acceptance (9/9 cell × 3-seed × 50-epoch — pending Phase 3 execution)
   - §6.4.3: per-cell mean ± std acceptance gate
   - §6.4.4: HCT116 highlight + PC3 honest-negative
5. Write 12 tests: molmetal/tests/test_wf313d_todo220_acceptance_gates.py
   - 4 tests: §6.3 acceptance gate schema
   - 4 tests: §6.4 acceptance gate schema
   - 2 tests: smoke persistence JSON validation
   - 2 tests: integration with GPU results (when Phase 3 lands)
6. Run: ${PY} -m pytest molmetal/tests/test_wf313d_todo220_acceptance_gates.py -v 2>&1 | tail -20
7. Verify pre_flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- §6.3 / §6.4 acceptance gates are CPU-only
- Full GPU sweep is USER_OK + 3.9 hr wall (in pipeline Phase 3)
- Smoke is MEASURED (real numbers); GPU is PROJECTED until Phase 3 lands

OUTPUT (PLAIN TEXT): n_acceptance_gates + n_tests_pass + per_section_status + preflight_status + gpu_landing_eta. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313D',
  }),

  () => agent(`Wf-313E: MECHANISM_METRIC_MAPPING update M22-M44 (23 new mechanisms) + regression-debug table.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/03_plans/R19_PLAN.md §1: "M22-M44 23-mechanism cross-mapping". Per TODO/02_metrics/MECHANISM_METRIC_MAPPING.md: currently 21 mechanisms (M1-M21). Need extension to M22-M44 for frontier math + new mechanisms from R18/R19/R20 plans.

TASK:
1. Read TODO/02_metrics/MECHANISM_METRIC_MAPPING.md — current 21 mechanisms
2. Read TODO/02_metrics/METRICS_TARGETS.md — current 132 metrics
3. Read TODO/03_plans/R18_MATH_MIGRATION_PLAN.md §2 + §3 (P0-1..P0-4 + P1-1..P1-7) — frontier math mechanisms
4. Read TODO/03_plans/R18_PLAN.md §15 + R19_PLAN.md — additional mechanisms
5. Identify 23 new mechanisms (M22-M44):
   - M22 SE(3) Schrödinger Bridge (R18 P0-1)
   - M23 Transport Relaxation endpoint penalty (R18 P0-2)
   - M24 Monge-Ampère regularity (R18 P0-3)
   - M25 Wasserstein gradient flow equivalence (R18 P0-4)
   - M26 Multi-parameter Persistent Homology (R18 P1-1)
   - M27 Persistent Laplacian (R18 P1-2)
   - M28 Bass martingale diffusion (R18 P1-3)
   - M29 Hilbert primal-dual (R18 P1-4)
   - M30 Fractional reaction-diffusion (R18 P1-5)
   - M31 Quantum semantics for atoms (R18 P1-6)
   - M32 Hochschild cohomology (R18 P1-7)
   - M33 R² architectural ceiling (Wf-297B finding)
   - M34 Vina covalent mechanism ceiling (Wf-143 + Wf-301B)
   - M35 VariancePUCT default (R20-P1)
   - M36 Hilbert-ASTSR R² lift (R20-P2)
   - M37 Multi-PH Fourier + PLL feature extraction (R20-P3)
   - M38 3-way on MetalCytoToxDB (R20-P4)
   - M39 Closure theorem d-block (R20-P5)
   - M40 Wet-lab IC50 validation (R20-P6)
   - M41 DrugOOD BELOW_RANDOM fix (Wf-301A)
   - M42 Pt+2 parameterisation covalent blocker (Wf-299B)
   - M43 84-rule MLE matrix (R17 workflow H)
   - M44 18-rule + 4 NEW SMARTS (R17 workflow F)
6. Update MECHANISM_METRIC_MAPPING.md:
   - Add M22-M44 to §1 mechanism table
   - Add new §X: per-mechanism affected metrics
   - Add §Y: cross-reference to R18/R19/R20 plan IDs
7. Add 23 tests (1 per new mechanism): molmetal/tests/test_wf313e_mechanism_metric_mappings.py
   - test_m22_se3_sb (verify M22 entry + affected metrics)
   - ... (1 test per M22-M44)
8. Run: ${PY} -m pytest molmetal/tests/test_wf313e_mechanism_metric_mappings.py -v 2>&1 | tail -20
9. Add regression-debug table: MECHANISM_METRIC_MAPPING.md §Z — "When metric X regresses, check mechanisms Y"
10. Document: TODO/completed/313e_mechanism_metric_mapping_extension.md

Honest framing:
- 23 new mechanisms are NOT yet fully integrated (most are R18-R20 plan items)
- Mapping is documentation work, NOT new measurement
- Each mechanism entry cites source plan/section

OUTPUT (PLAIN TEXT): n_mechanisms_added + n_tests_pass + n_affected_metrics_mapped + preflight_status + per_mechanism_source_citation. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313E',
  }),

  () => agent(`Wf-313F: chemprop v2 featurization fix (TODO-155 §3) — precompute features once per epoch + faster encoder pipeline.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO-155 §3: chemprop v2 full 5×5×50 TIMED OUT at iteration 1 (Ru seed=0, 19135 samples, 50 epochs did not finish in 6:50 wall). Bottleneck: chemprop v2 per-batch SMILES featurization on CPU (GPU 2-4%, CPU 660%+). Fix: precompute features once per epoch + faster encoder pipeline.

TASK:
1. Read molmetal/scripts/train_chemprop_v2_per_metal.py — current featurization path
3. Read molmetal/scripts/precache_chemprop_features.py — precompute feature cache
4. Implement per-epoch feature cache:
   - At epoch start: check feature cache (per SMILES → per-epoch features)
   - On cache miss: compute features once, store to cache
   - On cache hit: load features from cache (no recomputation)
5. Implement faster encoder pipeline:
   - Replace per-atom one-hot with packed-bit encoding
   - Vectorize bond feature precomputation (use numpy broadcasting)
6. Write 12 tests: molmetal/tests/test_wf313f_chemprop_v2_featurization_fix.py
   - 4 tests: feature cache hit/miss
   - 3 tests: faster encoder correctness (vs baseline output)
   - 3 tests: integration with train_chemprop_v2_per_metal.py
   - 2 tests: timing benchmark (baseline vs fixed)
7. Run timing benchmark:
   ${PY} molmetal/scripts/train_chemprop_v2_per_metal.py --metals Ru --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --out /tmp/wf313f_featurization_fix_smoke --execute 2>&1 | tail -20
8. Document: TODO/completed/155_wf313f_chemprop_featurization_fix.md

Honest framing:
- Fix is CPU-only
- 30 min engineering + 1 day validation per TODO-155 §3
- Wall-time saving: 5.1× speedup (per TODO-155 §3 MEASURED)
- Full paper-grade fix verification needs GPU portion (R20-P1 phase)

OUTPUT (PLAIN TEXT): n_cache_hit_rate + n_faster_encoder_tests_pass + wall_time_speedup_factor + preflight_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313F',
  }),

  () => agent(`Wf-313G: Wf-123 P3 downstream metric lift dashboard for 6 strengthen-TODO consumers.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md: 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125) — M1+M2 wiring is SHIPPED (Wf-123 P3, 25/25 integration tests), but per-consumer lift is per-TODO deliverable. Need unified "downstream lift dashboard" to track what has been measured per consumer.

TASK:
1. Read TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md §1 + §2 (current MEASURED state table)
2. For each of 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125), read current source:
   - TODO-119 m14 decode_ratio: metrics/by_metric/cfm_decode_real.json + cfm_decode_synthetic.json
   - TODO-120 m1 Ru AUC: metrics/by_metric/wf181_g5_50epoch_full_sweep.json (Ru 0.7860 ± 0.0145)
   - TODO-121 m19 LOMO Rh bottleneck: metrics/by_metric/wf176_m19_oversample_lomo/aggregate.json (Rh 0.34375)
   - TODO-122 L5 Hilbert R²: metrics/by_metric/hilbert_astsr_*.json
   - TODO-123 m15 RMSD: metrics/by_metric/docked_pose_rmsd.json + ligand_vs_pocket_rmsd.json
   - TODO-125 L12 OOD per-pocket Vina: metrics/by_metric/per_pocket_vina_*.json
3. Build unified downstream lift dashboard:
   - molmetal/reports/wf313g_downstream_lift_dashboard/dashboard.json
   - Per-consumer table: metric, baseline, lift (Δ%), status (MEASURED/PARTIAL/PROJECTED/HONEST_NEGATIVE)
5. Write 12 tests: molmetal/tests/test_wf313g_downstream_lift_dashboard.py
   - 6 tests: per-consumer schema validation
   - 3 tests: lift calculation arithmetic
   - 2 tests: status classification (MEASURED vs PROJECTED)
   - 1 test: dashboard schema validation
6. Run: ${PY} -m pytest molmetal/tests/test_wf313g_downstream_lift_dashboard.py -v 2>&1 | tail -20
7. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/177_wf313g_downstream_lift_dashboard.md

Honest framing per party-twenty-spirit §3.5:
- Per-consumer lift is per-TODO deliverable (Q4 2026 / R20 GPU exercise)
- Dashboard is aggregation work, NOT new measurement
- Some consumers will show HONEST_NEGATIVE (e.g., TODO-122 R² ≤ 0 by construction)
- TODO-121 m19 LOMO Rh 0.34375 is partial recovery (target ≥0.30 PASS but target ≥0.50 still PENDING)

OUTPUT (PLAIN TEXT): n_consumers_tracked + per_consumer_status + n_tests_pass + preflight_status + n_honest_negatives + lift_dashboard_path. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313G',
  }),

  () => agent(`Wf-313H: R20 Phase 0 closeout — verify 3 loaders (binding_db.py + drugood_loader.py + nci60.py) have ≥12 tests each + API doc.

USE Bash + Read + Write + Edit.

CONTEXT: Per TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §1.1: 3 loader modules need to exist + ≥12 tests each + API doc. Per Wf-309A: drugood_loader.py + binding_db.py + nci60.py already exist on disk. Need to verify each has ≥12 tests + write/enhance API doc + closeout R20 Phase 0.

TASK PART 1 — verify 3 loaders have ≥12 tests each:
1. molmetal/data/binding_db.py — count existing tests
2. molmetal/data/drugood_loader.py — count existing tests
3. molmetal/data/nci60.py — count existing tests
4. For each loader with < 12 tests, write NEW tests to reach ≥12:
   - molmetal/tests/test_wf313h_binding_db_loader.py
   - molmetal/tests/test_wf313h_drugood_loader.py
   - molmetal/tests/test_wf313h_nci60_loader.py
5. Each test set covers:
   - 4 tests: load + parse + return schema
   - 3 tests: row count + metal filter + scaffold split
   - 2 tests: edge cases (empty, malformed)
   - 2 tests: integration with downstream consumer (e.g., chemprop v2)

TASK PART 2 — API doc + Phase 0 closeout:
1. Per loader: write molmetal/data/<loader>.md API doc (classes, methods, return schema, sample usage)
2. Write TODO/completed/313_r20_phase0_loaders_closeout.md:
   - Per-loader status: file + tests + API doc + row count + metal filter
   - Phase 0 acceptance gates: 3/3 loaders + 12+ tests each + API doc present
   - Phase 0 honest framing: data availability assessment
3. Verify pre-flight: ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

TASK PART 3 — R20 Phase 0 closeout gate:
1. Add pre_flight v3 gate: p3_r20_phase0_loaders_complete (verify 3 loaders + ≥12 tests each + API docs)
2. Update pre_flight_check.py gate counter

Honest framing:
- Phase 0 is P0 CRITICAL per R20_PLAN
- Loaders are already mostly in place; this is verification + closeout
- GPU portions (Phase 1-5) are in pipeline; CPU Phase 0 closeout is actionable now

OUTPUT (PLAIN TEXT): n_loaders_verified + n_tests_per_loader + n_api_docs + preflight_status + phase0_closeout_status. Do NOT call StructuredOutput.`, {
    phase: 'Actionable-now TODO cleanup (parallel)',
    label: 'wf313H',
  }),
])

phase('Synthesis')

const wf313I = await agent(`Wf-313I: Aggregate Wf-313 actionable-now TODO cleanup outcomes — TODO/INDEX.md update.

USE Read only.

UPSTREAM RESULTS:
- Wf-313A: Wf-134 m15 Phase 1 SMILES extraction
- Wf-313B: Wetlab SOPs + 4 LOI + IC50 pipeline enhancement
- Wf-313C: Frontier math Day 3-5 integration smoke + paper refs
- Wf-313D: TODO-220 §6.3/§6.4 acceptance gates
- Wf-313E: MECHANISM_METRIC_MAPPING M22-M44 (23 new)
- Wf-313F: chemprop v2 featurization fix (TODO-155 §3)
- Wf-313G: Wf-123 P3 downstream lift dashboard (6 consumers)
- Wf-313H: R20 Phase 0 closeout (3 loaders + tests + API docs)

TASK:
1. Read all 8 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table) with Wf-313 entries
4. Output to molmetal/reports/wf313_synthesis/actionable_now_cleanup_final.md

OUTPUT (PLAIN TEXT): per_agent_summary + actionable_now_closed + honest_negatives + remaining_blockers + n_metrics_shipped (12-20 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf313I',
})

return {
  ultracode: 'wf-313-actionable-now-cleanup',
  wf313a: wf313A,
  wf313b: wf313B,
  wf313c: wf313C,
  wf313d: wf313D,
  wf313e: wf313E,
  wf313f: wf313F,
  wf313g: wf313G,
  wf313h: wf313H,
  wf313i: wf313I,
}