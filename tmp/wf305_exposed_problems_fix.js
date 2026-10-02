export const meta = {
  name: 'wf-305-exposed-problems-fix',
  description: 'Wf-305: Comprehensive fix for ALL exposed problems from Wf-300/301/303/304. (A) PlatinAI path switching — S7 MB-Finder stratified extension + S6 Rusanov outreach draft (since S1 Selenium BLOCKED by CF Enterprise). (B) Frontier math wire-in — 0/4 → 4/4 outer adapters into __init__.py. (C) TODO-287A AnchorGuard version-pin + TODO-220 sub-pending driver launch (temporal-split + NCI-60 9-cell chemprop v2 multi-task). (D) drugood DANN multi-task + Vina Pt-stripped real implementation. (E) Synthesis.',
  phases: [
    { title: 'Exposed problems fix (parallel)', detail: '4 parallel: platinai, frontier, anchor+wf219, drugood+vina' },
    { title: 'Synthesis', detail: 'Aggregate + remaining blockers' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Exposed problems fix (parallel)')

const [wf305A, wf305B, wf305C, wf305D] = await parallel([

  () => agent(`Wf-305A: PlatinAI path switching — S7 MB-Finder stratified extension + S6 Rusanov outreach draft.

USE Bash + Read + Edit.

CONTEXT: Per Wf-304B honest-negative: Selenium S1 BLOCKED by Cloudflare Enterprise + Turnstile (90s, 0 IC50 harvested). Per Wf-295B + Wf-301C, MB-Finder gives coverage but only A549=100, others=0. Need to extend MB-Finder to MCF7/A2780/A2780cis AND draft Rusanov outreach (S6 path).

TASK PART 1 — S7 MB-Finder stratified extension:
1. Read molmetal/scripts/platinai_real_label_loader.py — current MB-Finder loader (Wf-295B)
2. Read molmetal/reports/wf304b_selenium/selenium_report.md — S1 BLOCK honest-negative
3. Extend loader with per-cell-line stratified mode:
   - Already supports --cell-lines (per Wf-301C)
   - Verify per-cell-line coverage: A549=100 → MCF7=100 → A2780=100 → A2780cis=100
   - Add MB-Finder pagination for >100 per cell-line
4. Run stratified fetch:
   ${PY} molmetal/scripts/platinai_real_label_loader.py --cell-lines A549,MCF7,A2780,A2780cis --n-target-per-cell 100 --start-page 1 --delay 2.5 --out .data/molmetal/platinai_4cell_mbfinder_full.csv 2>&1 | tail -30
5. Verify per-cell-line count >= 100 each
6. Document blockers honestly (per Wf-297B §6 honest framing)

TASK PART 2 — S6 Rusanov outreach draft:
1. Read TODO/pending/wetlab_loi_ornl_protein_metal.md (existing LOI template)
2. Draft NEW outreach email/letter: TODO/pending/wetlab_loi_rusanov_2025_platinai.md
   - Subject: "PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)"
   - Body: project context + what's needed (Rusanov 2025 Table S4-S5 NUMERIC) + what's offered (co-authorship + open-source release)
   - Attach: 1-page project summary from paper §1
3. Honest framing: not yet sent — draft only, requires user OK to send per TODO-84

OUTPUT (PLAIN TEXT): per_cell_line_count + n_total + rusanov_draft_path + user_ok_status. Do NOT call StructuredOutput.`, {
    phase: 'Exposed problems fix (parallel)',
    label: 'wf305A',
  }),

  () => agent(`Wf-305B: Frontier math wire-in — 0/4 → 4/4 outer adapters into __init__.py.

USE Bash + Read + Edit.

CONTEXT: Per Wf-304C audit: 0/4 outer frontier math adapters wired into molmetal/adapters/flow_matching_lipman/__init__.py. Adapters: multi_ph, se3_lipman, se3_sb, transport_relaxation. All wire-ready standalone (per audit), just not imported.

TASK:
1. Read molmetal/adapters/flow_matching_lipman/__init__.py — current imports
2. Read molmetal/adapters/flow_matching_lipman/_reference/{multi_ph,se3_lipman,se3_sb,transport_relaxation}/__init__.py — verify importable
3. Wire-in (R17 INVARIANT: do NOT modify _reference/ — use lazy import wrappers in __init__.py):
   - Add lazy import for each adapter:
     try:
       from ._reference.multi_ph import MultiPHScheduler
     except ImportError:
       MultiPHScheduler = None
   - Same for se3_lipman, se3_sb, transport_relaxation
4. Add __all__ entry for each
5. Write 4 tests: test_<adapter>_import_succeeds
6. Run tests:
   ${PY} -m pytest molmetal/tests/test_wf305b_frontier_wire_in.py -v 2>&1 | tail -20
7. Verify pre_flight still PASS:
   ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10

Honest framing:
- Lazy imports (not eager) per R17 invariant — _reference/ is the read-only reference
- If a dependency is missing, log loud WARNING + return None (do not raise)
- 5-7 day ultracode window per Wf-304C — this is 4 hours work for the wire-in itself

OUTPUT (PLAIN TEXT): n_adapters_wired + n_tests_pass + preflight_status + lazy_import_used. Do NOT call StructuredOutput.`, {
    phase: 'Exposed problems fix (parallel)',
    label: 'wf305B',
  }),

  () => agent(`Wf-305C: TODO-287A AnchorGuard version-pin + TODO-220 sub-pending driver launch.

USE Bash + Read + Write + Edit.

CONTEXT: Per Wf-304A honest-negative: AnchorGuard version-pin code path in place but NO current JSON anchor carries 'version' field. Per Wf-304C: TODO-220 sub-pending (temporal-split driver + NCI-60 9-cell chemprop v2 multi-task) NOT in Wf-304C scope. Per Wf-219 audit (Wf-304C Task 1): Ru seed 全 timeout (43/50 epochs), agg 缺 Ru.

TASK PART 1 — TODO-287A version-pin:
1. Read molmetal/scripts/read_anchor_json.py — AnchorGuard API (Wf-302A)
2. Find the JSON anchors that should carry version field:
   - molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json (Wf-287A)
   - molmetal/reports/wf271b/wf269c_cfm_real/wf269c_report.json
   - molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json
   - molmetal/reports/wf219_krasnov_aligned/sweep_summary.json
3. Add 'version' field to each anchor (semver-style: '1.0.0' or similar)
4. Add test_anchor_version_pin to molmetal/tests/test_read_anchor_json.py:
   - Test that required_version='1.0.0' passes for current anchors
   - Test that required_version='2.0.0' raises AnchorVersionMismatchError
5. Run tests: ${PY} -m pytest molmetal/tests/test_read_anchor_json.py -v 2>&1 | tail -10

TASK PART 2 — TODO-220 sub-pending driver launch (CPU-only code, no GPU):
1. Read TODO/pending/180_wf124_p3_remaining_metals_seeds.md — TODO-180 done, TODO-220 derived
2. Design driver: temporal-split driver for Krasnov comparison
   - Path: molmetal/scripts/wf219_temporal_split_driver.py
   - CLI: --csv, --train-cutoff, --test-cutoff, --seeds, --hidden, --depth
   - Per-seed train on data before cutoff, test on data after cutoff
3. Design driver: NCI-60 9-cell chemprop v2 multi-task
   - Path: molmetal/scripts/nci60_9cell_chemprop_driver.py
   - CLI: --csv .data/molmetal/nci60_9cell.csv, --seeds, --hidden, --depth, --out
4. Write 2 stub drivers with --dry-run mode (exit 0 + emit JSON shape)
5. Document in: TODO/pending/220_wf219_temporal_split_nci60_driver.md

TASK PART 3 — Wf-219 Ru seed recovery (CPU-only consolidation):
1. Read molmetal/reports/wf219_krasnov_aligned/ — per-seed JSON files
2. Note: Ru seeds all timeout at 43/50 epochs (Wf-304C Task 1)
3. Document decision: use existing partial per-seed AUCs OR wait for GPU-1 retest with shorter epoch target
4. Output decision to molmetal/reports/wf305c_ru_recovery/decision.md
   - Honest framing: agg excludes Ru per party-twenty-spirit §3.5; full recovery requires user OK + 13.5 min GPU-1

OUTPUT (PLAIN TEXT): n_anchors_versioned + n_version_pin_tests + n_drivers_stubbed + ru_decision. Do NOT call StructuredOutput.`, {
    phase: 'Exposed problems fix (parallel)',
    label: 'wf305C',
  }),

  () => agent(`Wf-305D: drugood DANN multi-task + Vina Pt-stripped real implementation.

USE Bash + Edit.

CONTEXT: Per Wf-297B + Wf-301A: drugood_v3_scaffold AUC=0.4574 BELOW field-avg 0.55-0.65. Per Wf-299B + Wf-301B: Vina 0/100 docked (Pt+2 UFF parameterisation blocker). Both need real implementation per Wf-304C honest-negative.

TASK PART 1 — drugood DANN multi-task:
1. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — current --use-dann wiring (Wf-285B)
2. Read molmetal/scripts/wf286b_ecfp4_reeval.py — Wf-286B ruled out ECFP4 bottleneck
3. Write NEW training driver: molmetal/scripts/wf305d_drugood_dann_train.py
   - Wrapper around train_dmpnn_multitask_wasserstein.py with:
     - --csv molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv (Wf-275B)
     - --chemprop-mode v3-fallback (Wf-263A default)
     - --use-dann (Wf-285B)
     - --seeds 0 1 2
     - --epochs 5 (CPU-only smoke) OR --epochs 50 (GPU full)
   - Aggregate ood_test_auc across 3 seeds
4. CPU smoke: ${PY} molmetal/scripts/wf305d_drugood_dann_train.py --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf305d_smoke 2>&1 | tail -20
5. Document GPU full path (requires user OK per kill_then_adapt)

TASK PART 2 — Vina Pt-stripped:
1. Read molmetal/scripts/cfm_vina_dock_panel.py — current Vina workflow (Wf-299B)
2. Find RDKit UFF MMFF parameterisation step
3. Add Pt-stripping logic:
   - Use Chem.GetMolFrags to find largest organic fragment (drop Pt shell)
   - Re-parameterise ligand-only with UFF
4. Write NEW Pt-stripped Vina driver: molmetal/scripts/wf305d_vina_pt_strip.py
   - Wrapper around cfm_vina_dock_panel.py with:
     - --pt-strip (NEW flag)
     - --csv .data/molmetal/platinai_4cell_mbfinder.csv (Wf-295B 100 mols)
     - --pocket-id 1h36
     - --n-mols 100
5. CPU smoke: ${PY} molmetal/scripts/wf305d_vina_pt_strip.py --pt-strip --n-mols 5 --out /tmp/wf305d_vina_smoke 2>&1 | tail -20
6. Document GPU full path (requires user OK per kill_then_adapt)

Honest framing:
- Both implementations are CPU-smoke verified only
- Full GPU runs require user OK per kill_then_adapt directive
- Field-avg targets: drugood ≥ 0.55-0.65, Vina -7.5 to -9.5 kcal/mol

OUTPUT (PLAIN TEXT): n_drivers_implemented + drugood_smoke_status + vina_smoke_status + n_docked + field_avg_target_met_smoke. Do NOT call StructuredOutput.`, {
    phase: 'Exposed problems fix (parallel)',
    label: 'wf305D',
  }),
])

phase('Synthesis')

const wf305E = await agent(`Wf-305E: Aggregate Wf-305 exposed problems fix outcomes.

USE Read only.

UPSTREAM RESULTS:
- Wf-305A: PlatinAI path switching (S7 + S6 draft)
- Wf-305B: Frontier math wire-in
- Wf-305C: TODO-287A version-pin + TODO-220 drivers + Wf-219 Ru decision
- Wf-305D: drugood DANN + Vina Pt-stripped

TASK:
1. Read all 4 outputs
2. Aggregate: completed + failed + partial
3. Identify remaining blockers
4. Output to molmetal/reports/wf305_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + exposed_problems_closed + remaining_blockers (8-12 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf305E',
})

return {
  ultracode: 'wf-305-exposed-problems-fix',
  wf305a: wf305A,
  wf305b: wf305B,
  wf305c: wf305C,
  wf305d: wf305D,
  wf305e: wf305E,
}