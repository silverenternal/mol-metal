export const meta = {
  name: 'wf-320-wf313-actionable-now-closeout',
  description: 'Wf-320: Wf-313 actionable-now closeout ACTUAL EXECUTION + new concrete code tasks. (B) Wf-291F harness auto-fallback fix (Wf-315B recommendation #2 — Wf-291C pipeline regression residual). (C) TODO-220 stub driver full impl (Wf-305C stub → full CPU impl of temporal + NCI-60). (D) TODO-155 chemprop v2 featurization fix ACTUAL — precompute features once per epoch (Wf-313F actual; Wf-219P1 bottleneck). (E) Wf-123 P3 downstream metric lift dashboard ACTUAL — 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125) per-TODO lift table. (F) R20 Phase 0 closeout ACTUAL — verify 3 loaders (binding_db.py + drugood_loader.py + nci60.py) have ≥12 tests + API doc. (G) MECHANISM_METRIC_MAPPING M22-M44 ACTUAL — extend to 23 new mechanisms (R18/R19/R20 plan items). (H) TODO-134 Phase 2 attempt — try Vina install (conda-forge vina) to unblock m15 redesign. (I) Synthesis + INDEX.md update.',
  phases: [
    { title: 'Wf-313 closeout actual execution (parallel)', detail: '8 parallel agents actually EXECUTING Wf-313 deferred work + new concrete tasks' },
    { title: 'Synthesis', detail: 'Aggregate + INDEX.md final sync + pre-flight' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('Wf-313 closeout actual execution (parallel)')

const [wf320B, wf320C, wf320D, wf320E, wf320F, wf320G, wf320H] = await parallel([

  () => agent(`Wf-320B: Wf-291F harness auto-fallback fix — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315B recommendation #2: Wf-291F harness auto-fallback 真修. Wf-291C was the pipeline regression root cause. Wf-302C text-only template worked around it but the underlying auto-fallback isn't fully fixed.

TASK:
1. Read molmetal/scripts/wf302d_retry_with_correction.py — current retry helper
2. Read molmetal/scripts/abort_recovery.py — recovery helper
3. Identify auto-fallback gap: structured output retry vs schema mismatch
4. Write NEW module: molmetal/scripts/wf320b_harness_autofallback.py
   - Class HarnessAutoFallback
   - detect_schema_mismatch(error) → schema_coerce hint
   - retry_with_correction(task, max_retries=3) → result
   - Integration point: callable from Wf-302D helper
5. Write 12 NEW tests: molmetal/tests/test_wf320b_harness_autofallback.py
   - 4 tests: schema mismatch detection
   - 4 tests: retry with correction
   - 2 tests: integration with Wf-302D
   - 2 tests: edge cases (max retries exhausted)
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf320b_harness_autofallback.py -v 2>&1 | tail -20
7. Update paper_dashboard.py with claim_wf320b_harness_autofallback_shipped
8. Run pre-flight: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/320b_wf291f_harness_autofallback_fix.md

Honest framing:
- This is INTEGRATION work (CPU-only)
- Auto-fallback will reduce future pipeline regression frequency
- Bounded retry logic (max 3) prevents infinite loops

OUTPUT (PLAIN TEXT): n_fixes_shipped + n_tests_pass + preflight_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320B',
  }),

  () => agent(`Wf-320C: TODO-220 stub driver full impl — ACTUAL EXECUTION.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-315B recommendation #5: TODO-220 stub 实装. Wf-305C shipped dry-run stub; full impl missing. Wf-308C later shipped --execute path for both temporal and NCI-60 drivers. Need to verify + close out.

TASK:
1. Read molmetal/scripts/wf219_temporal_split_driver.py — verify --execute path exists
2. Read molmetal/scripts/nci60_9cell_chemprop_driver.py — verify --execute path exists
3. Per Wf-308C: both --execute paths SHIPPED + CPU smokes MEASURED (temporal 5/5 ok + NCI-60 9/9 ok)
4. Verify current state with --execute dry-run + --execute actual run
6. Write 10 NEW tests: molmetal/tests/test_wf320c_todo220_full_impl.py
   - 3 tests: temporal --execute path
   - 3 tests: NCI-60 --execute path
   - 2 tests: integration with downstream metrics
   - 2 tests: per-driver integration with Wf-313D acceptance gates
7. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf320c_todo220_full_impl.py -v 2>&1 | tail -20
8. Run pre-flight: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/320c_todo220_full_impl_closeout.md

Honest framing:
- Wf-308C + Wf-310C + Wf-313D already covered the --execute paths
- Wf-320C is verification + integration test (not new implementation)
- DO NOT re-implement what already works

OUTPUT (PLAIN TEXT): n_tests_pass + per_driver_status + integration_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320C',
  }),

  () => agent(`Wf-320D: TODO-155 chemprop v2 featurization fix ACTUAL — precompute features once per epoch (Wf-313F actual).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-313F (per TODO-155 §3): chemprop v2 full 5×5×50 TIMED OUT at iteration 1 (Ru seed=0, 19135 samples). Bottleneck: per-batch SMILES featurization on CPU (660%+ util). Fix: precompute features once per epoch + faster encoder pipeline.

TASK:
1. Read molmetal/scripts/train_chemprop_v2_per_metal.py — current featurization path
3. Read molmetal/scripts/precache_chemprop_features.py — precompute feature cache (if exists)
4. Implement per-epoch feature cache:
   - At epoch start: check feature cache (per SMILES → per-epoch features)
   - On cache miss: compute features once, store to cache
   - On cache hit: load features from cache (no recomputation)
5. Implement faster encoder pipeline:
   - Replace per-atom one-hot with packed-bit encoding
   - Vectorize bond feature precomputation (numpy broadcasting)
6. Write 12 NEW tests: molmetal/tests/test_wf320d_chemprop_featurization_fix.py
   - 4 tests: feature cache hit/miss
   - 3 tests: faster encoder correctness
   - 3 tests: integration with train_chemprop_v2_per_metal.py
   - 2 tests: timing benchmark (baseline vs fixed)
7. Run timing benchmark:
   /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/train_chemprop_v2_per_metal.py --metals Ru --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --out /tmp/wf320d_featurization_fix_smoke --execute 2>&1 | tail -20
8. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
9. Document: TODO/completed/320d_chemprop_featurization_fix.md

Honest framing:
- Fix is CPU-only
- 30 min engineering + 1 day validation per TODO-155 §3
- Wall-time saving: 5.1× speedup (per TODO-155 §3 MEASURED)
- Full paper-grade fix verification needs GPU portion (R20-P1 phase)

OUTPUT (PLAIN TEXT): n_cache_hit_rate + n_faster_encoder_tests_pass + wall_time_speedup_factor + preflight_status + integration_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320D',
  }),

  () => agent(`Wf-320E: Wf-123 P3 downstream metric lift dashboard ACTUAL — 6 strengthen-TODO consumers per-TODO lift table.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-313G: 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125) — M1+M2 wiring SHIPPED (Wf-123 P3) but per-consumer lift is per-TODO deliverable. Wf-313G may have analyzed but not executed the actual dashboard.

TASK:
1. Read TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md §1 + §2 (current MEASURED state table)
2. For each of 6 strengthen-TODO consumers (TODO-119/120/121/122/123/125), read current source:
   - TODO-119 m14 decode_ratio: metrics/by_metric/cfm_decode_real.json + cfm_decode_synthetic.json
   - TODO-120 m1 Ru AUC: metrics/by_metric/wf181_g5_50epoch_full_sweep.json
   - TODO-121 m19 LOMO Rh bottleneck: metrics/by_metric/wf176_m19_oversample_lomo/aggregate.json
   - TODO-122 L5 Hilbert R²: metrics/by_metric/hilbert_astsr_*.json
   - TODO-123 m15 RMSD: metrics/by_metric/docked_pose_rmsd.json + ligand_vs_pocket_rmsd.json
   - TODO-125 L12 OOD per-pocket Vina: metrics/by_metric/per_pocket_vina_*.json
3. Build unified downstream lift dashboard:
   - molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json
   - Per-consumer table: metric, baseline, lift (Δ%), status (MEASURED/PARTIAL/PROJECTED/HONEST_NEGATIVE)
4. Add 6 NEW claim cards to paper_dashboard.py (one per consumer)
5. Write 12 NEW tests: molmetal/tests/test_wf320e_downstream_lift_dashboard.py
   - 6 tests: per-consumer schema validation
   - 3 tests: lift calculation arithmetic
   - 2 tests: status classification
   - 1 test: dashboard schema validation
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf320e_downstream_lift_dashboard.py -v 2>&1 | tail -20
7. Run pre-flight: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
8. Document: TODO/completed/320e_downstream_lift_dashboard.md

Honest framing per party-twenty-spirit §3.5:
- Per-consumer lift is per-TODO deliverable (Q4 2026 / R20 GPU exercise)
- Dashboard is aggregation work, NOT new measurement
- Some consumers will show HONEST_NEGATIVE (e.g., TODO-122 R² ≤ 0 by construction)
- TODO-121 m19 LOMO Rh 0.34375 is partial recovery

OUTPUT (PLAIN TEXT): n_consumers_tracked + per_consumer_status + n_tests_pass + preflight_status + n_honest_negatives + lift_dashboard_path. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320E',
  }),

  () => agent(`Wf-320F: R20 Phase 0 closeout ACTUAL — verify 3 loaders + ≥12 tests + API doc.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-313H + WEAK_TO_STRONG §1.1: 3 loader modules need to exist + ≥12 tests each + API doc. Per Wf-313H: loaders mostly in place; this is verification + closeout.

TASK PART 1 — verify 3 loaders have ≥12 tests each:
1. molmetal/data/binding_db.py — count existing tests
2. molmetal/data/drugood_loader.py — count existing tests
3. molmetal/data/nci60.py — count existing tests
4. For each loader with < 12 tests, write NEW tests to reach ≥12:
   - molmetal/tests/test_wf320f_binding_db_loader.py
   - molmetal/tests/test_wf320f_drugood_loader.py
   - molmetal/tests/test_wf320f_nci60_loader.py
5. Each test set covers:
   - 4 tests: load + parse + return schema
   - 3 tests: row count + metal filter + scaffold split
   - 2 tests: edge cases (empty, malformed)
   - 2 tests: integration with downstream consumer
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf320f_*_loader.py -v 2>&1 | tail -20

TASK PART 2 — API doc + Phase 0 closeout:
1. Per loader: write molmetal/data/<loader>.md API doc (classes, methods, return schema, sample usage)
2. Write TODO/completed/320_r20_phase0_loaders_closeout.md:
   - Per-loader status: file + tests + API doc + row count + metal filter
   - Phase 0 acceptance gates: 3/3 loaders + 12+ tests each + API doc present
   - Phase 0 honest framing: data availability assessment

TASK PART 3 — pre-flight gate:
1. p3_r20_phase0_loaders_complete (verify 3 loaders + ≥12 tests each + API docs)
2. Update pre_flight_check.py gate counter

Honest framing:
- Phase 0 is P0 CRITICAL per R20_PLAN
- Loaders are already mostly in place; this is verification + closeout
- GPU portions (Phase 1-5) are in pipeline; CPU Phase 0 closeout is actionable now

OUTPUT (PLAIN TEXT): n_loaders_verified + n_tests_per_loader + n_api_docs + preflight_status + phase0_closeout_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320F',
  }),

  () => agent(`Wf-320G: MECHANISM_METRIC_MAPPING M22-M44 ACTUAL — 23 new mechanisms (Wf-313E actual).

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-313E: M22-M44 23-mechanism cross-mapping per R19 plan + R18 math migration plan. Need to extend the mapping.

TASK:
1. Read TODO/02_metrics/MECHANISM_METRIC_MAPPING.md — current 21 mechanisms (M1-M21)
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
   - M35 VariancePUCT default (R20-P1, Wf-309C shipped)
   - M36 Hilbert-ASTSR R² lift (R20-P2)
   - M37 Multi-PH Fourier + PLL feature extraction (R20-P3)
   - M38 3-way on MetalCytoToxDB (R20-P4)
   - M39 Closure theorem d-block (R20-P5)
   - M40 Wet-lab IC50 validation (R20-P6, DEPRECATED per Wf-317 Tier-2)
   - M41 DrugOOD BELOW_RANDOM fix (Wf-301A)
   - M42 Pt+2 parameterisation covalent blocker (Wf-299B)
   - M43 84-rule MLE matrix (R17 ship)
   - M44 18-rule + 4 NEW SMARTS (R17 ship)
6. Update MECHANISM_METRIC_MAPPING.md:
   - Add M22-M44 to §1 mechanism table
   - Add new §X: per-mechanism affected metrics
   - Add §Y: cross-reference to R18/R19/R20 plan IDs
7. Add 23 tests (1 per new mechanism): molmetal/tests/test_wf320g_mechanism_metric_mappings.py
8. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf320g_mechanism_metric_mappings.py -v 2>&1 | tail -20
9. Add regression-debug table: MECHANISM_METRIC_MAPPING.md §Z — "When metric X regresses, check mechanisms Y"
10. Document: TODO/completed/320g_mechanism_metric_mapping_extension.md

Honest framing:
- 23 new mechanisms are NOT yet fully integrated (most are R18-R20 plan items)
- Mapping is documentation work, NOT new measurement
- Per Wf-317 Tier-2: M40 wet-lab IC50 marked DEPRECATED

OUTPUT (PLAIN TEXT): n_mechanisms_added + n_tests_pass + n_affected_metrics_mapped + preflight_status + per_mechanism_source_citation. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320G',
  }),

  () => agent(`Wf-320H: TODO-134 Phase 2 attempt — try Vina install (conda-forge vina) to unblock m15 redesign.

USE Bash + Read + Edit.

CONTEXT: Per Wf-313A: TODO-134 Phase 2 (Vina redocking) BLOCKED on Vina 1.2.7 install. Recovery path: conda-forge vina. Need to verify if user can install + attempt.

TASK PART 1 — environment probe:
1. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import vina" 2>&1
2. which vina 2>&1
3. /home/hugo/codes/mol-metal/.venv/bin/python -c "import vina; print(vina.__version__)" 2>&1

TASK PART 2 — alternative paths to check:
1. Check pip packages: /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -m pip list 2>&1 | grep -i vina
2. Check conda availability: which conda 2>&1
3. Check if MOLMETAL_VINA_BIN env override is set: env | grep MOLMETAL_VINA_BIN

TASK PART 3 — verify Wf-313A outputs are ready:
1. ls /home/hugo/codes/mol-metal/molmetal/data/m15_redesign_ref/ 2>&1
2. Verify all 6 pockets have ref_ligand.smi + ref_ligand_inchi.json
4. If REF exists + Vina installed: write NEW m6_wf320h_vina_phase2.py driver
2. If REF exists + Vina NOT installed: write NEW recovery script
4. If Vina install blocked: HONEST_NEGATIVE_BLOCKED + clear recovery path

TASK PART 4 — pre-flight + documentation:
1. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
2. Document: TODO/completed/320h_todo134_phase2_attempt.md
3. Per Wf-317 Tier-2: Phase 2 may not be needed for paper; honest framing

OUTPUT (PLAIN TEXT): vina_install_status + recovery_path + n_pockets_ref_ready + phase2_status + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'Wf-313 closeout actual execution (parallel)',
    label: 'wf320H',
  }),
])

phase('Synthesis')

const wf320I = await agent(`Wf-320I: Aggregate Wf-320 Wf-313 closeout outcomes — INDEX.md final sync.

USE Read only.

UPSTREAM RESULTS:
- Wf-320B: Wf-291F harness auto-fallback fix
- Wf-320C: TODO-220 stub driver full impl
- Wf-320D: TODO-155 chemprop v2 featurization fix
- Wf-320E: Wf-123 P3 downstream metric lift dashboard
- Wf-320F: R20 Phase 0 closeout (3 loaders)
- Wf-320G: MECHANISM_METRIC_MAPPING M22-M44 (23 new)
- Wf-320H: TODO-134 Phase 2 Vina install attempt

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + honest-negative
3. Update TODO/INDEX.md §0 (recent ship wave) + §22 (detailed table)
4. Output: TODO/completed/320_final_synthesis.md

OUTPUT (PLAIN TEXT): per_agent_summary + n_completed + n_partial + n_honest_negatives + remaining_blockers + INDEX_md_status (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf320I',
})

return {
  ultracode: 'wf-320-wf313-actionable-now-closeout',
  wf320b: wf320B,
  wf320c: wf320C,
  wf320d: wf320D,
  wf320e: wf320E,
  wf320f: wf320F,
  wf320g: wf320G,
  wf320h: wf320H,
  wf320i: wf320I,
}