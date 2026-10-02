export const meta = {
  name: 'wf-306-improvement-round2',
  description: 'Wf-306: Improvement round 2 — actual runs + integration + pre-flight v3 upgrade + honest framing expansion. (A) drugood + Vina real CPU smoke + GPU launch wrapper. (B) TODO-220 actual runs + frontier math actual smoke + Wf-219 Ru short-epoch wrapper. (C) Pre-flight v2 → v3 upgrade (25 → 30+ gates) + §9 honest framing catalog 9 → 13+ entries + AnchorGuard version-pin CI gate. (D) CFM property panel actual launch + G6 user-OK-gated launch wrapper. (E) Synthesis.',
  phases: [
    { title: 'Improvement round 2 (parallel)', detail: '4 parallel: drugood+vina run, TODO-220+frontier smoke, pre-flight v3+honest framing, CFM panel+G6 wrapper' },
    { title: 'Synthesis', detail: 'Aggregate + remaining blockers' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

// Plain text schema to avoid pipeline regression (Wf-291C root-cause + Wf-302C template)

phase('Improvement round 2 (parallel)')

const [wf306A, wf306B, wf306C, wf306D] = await parallel([

  () => agent(`Wf-306A: drugood + Vina real CPU smoke + GPU launch wrapper.

USE Bash + Edit.

CONTEXT: Per Wf-305D: drugood DANN driver (wf305d_drugood_dann_train.py) + Vina Pt-stripped driver (wf305d_vina_pt_strip.py) implemented but CPU smoke NOT RUN. Per Wf-297B: drugood_v3_scaffold AUC=0.4574 BELOW field-avg 0.55-0.65. Per Wf-299B: Vina 0/100 docked (Pt+2 parameterisation blocker). Need real CPU smoke + GPU launch wrapper.

TASK PART 1 — drugood DANN CPU smoke:
1. Read molmetal/scripts/wf305d_drugood_dann_train.py (Wf-305D output)
2. Read molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv — Wf-275B adapter (95,236 rows, 0 leak)
3. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — confirm --use-dann wiring (Wf-285B)
4. Run CPU smoke (1 seed, 1 epoch, hidden=64, depth=2):
   ${PY} molmetal/scripts/wf305d_drugood_dann_train.py --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu --out /tmp/wf306a_drugood_smoke --chemprop-mode v3-fallback --use-dann 2>&1 | tail -30
5. Verify: ood_test_auc emitted in JSON; no exceptions; smoke PASS verdict
6. Aggregate per-seed: even if 1-seed, emit JSON with n_seeds_completed=1 + per_seed_ood_test_auc

TASK PART 2 — Vina Pt-stripped CPU smoke:
1. Read molmetal/scripts/wf305d_vina_pt_strip.py (Wf-305D output)
2. Read molmetal/scripts/cfm_vina_dock_panel.py — Wf-299B original (BLOCKED on Pt+2)
3. Run CPU smoke (5 mols, pt-strip):
   ${PY} molmetal/scripts/wf305d_vina_pt_strip.py --pt-strip --n-mols 5 --csv .data/molmetal/platinai_4cell_mbfinder.csv --out /tmp/wf306a_vina_smoke 2>&1 | tail -30
4. Verify: at least 1/5 docked successfully (vs 0/5 without pt-strip); vina_score emitted

TASK PART 3 — drugood + Vina GPU launch wrapper:
1. Write NEW launch wrapper: scripts/launch_wf306a_drugood_vina_gpu.sh
   - USER_OK gate enforced (per Wf-300C pattern)
   - pre_flight_check.py first
   - wave_budget_tracker.py --cfm-yuelbond-required (defer if not CFM)
   - 2 phases: Phase-A drugood (3-4 hr GPU-1) + Phase-B Vina (1-2 hr GPU-1)
   - Total wall: ~5 hr
2. bash -n syntax check
3. Document in molmetal/reports/wf306a_drugood_vina_launch/launch_plan.md

Honest framing: CPU smoke = implementation verification; full field-avg target (drugood ≥ 0.55, Vina -7.5 to -9.5) requires GPU + user OK.

OUTPUT (PLAIN TEXT): drugood_smoke_status + drugood_auc_smoke + vina_smoke_status + vina_n_docked + wrapper_path + user_ok_status. Do NOT call StructuredOutput.`, {
    phase: 'Improvement round 2 (parallel)',
    label: 'wf306A',
  }),

  () => agent(`Wf-306B: TODO-220 actual runs + frontier math actual smoke + Wf-219 Ru short-epoch wrapper.

USE Bash + Edit.

CONTEXT: Per Wf-305C: TODO-220 stub drivers (temporal-split + NCI-60 9-cell chemprop v2 multi-task) implemented but NOT RUN. Per Wf-305B: 4 frontier math adapters wired but actual smoke (multi_ph + se3_lipman + se3_sb + transport_relaxation) NOT RUN. Per Wf-304C Task 1: Wf-219 Ru seed all timeout (43/50 epochs); needs short-epoch wrapper for 13.5 min GPU-1 retest.

TASK PART 1 — TODO-220 temporal-split driver actual run:
1. Read molmetal/scripts/wf219_temporal_split_driver.py (Wf-305C output)
2. Read molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv — 26,633 rows (Wf-261)
3. Run --dry-run (verify CLI shape):
   ${PY} molmetal/scripts/wf219_temporal_split_driver.py --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --train-cutoff 2020-01-01 --test-cutoff 2024-01-01 --seeds 0 --epochs 1 --hidden 64 --depth 2 --dry-run 2>&1 | tail -20
4. Verify: JSON shape emitted; no exceptions

TASK PART 2 — TODO-220 NCI-60 9-cell chemprop v2 multi-task driver actual run:
1. Read molmetal/scripts/nci60_9cell_chemprop_driver.py (Wf-305C output)
2. Read molmetal/.data/molmetal/nci60_9cell.csv (if exists; if not, note missing)
3. Run --dry-run:
   ${PY} molmetal/scripts/nci60_9cell_chemprop_driver.py --dry-run --out /tmp/wf306b_nci60_smoke 2>&1 | tail -20
4. Verify: JSON shape emitted

TASK PART 3 — frontier math actual smoke (4 adapters):
1. Read molmetal/adapters/flow_matching_lipman/__init__.py — wire-in (Wf-305B output)
2. Read molmetal/adapters/flow_matching_lipman/_reference/multi_ph/__init__.py
3. Read molmetal/adapters/flow_matching_lipman/_reference/se3_lipman/__init__.py
4. Read molmetal/adapters/flow_matching_lipman/_reference/se3_sb/__init__.py
5. Read molmetal/adapters/flow_matching_lipman/_reference/transport_relaxation/__init__.py
6. Run smoke test per adapter (CPU-only, 1 forward-pass):
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import MultiPHScheduler; m = MultiPHScheduler(); print('multi_ph OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import SE3LipmanScheduler; m = SE3LipmanScheduler(); print('se3_lipman OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import SE3SBScheduler; m = SE3SBScheduler(); print('se3_sb OK:', m)" 2>&1 | tail -5
   ${PY} -c "from molmetal.adapters.flow_matching_lipman import TransportRelaxationScheduler; m = TransportRelaxationScheduler(); print('transport_relaxation OK:', m)" 2>&1 | tail -5
7. Document: n_adapters_actually_loaded + any failures

TASK PART 4 — Wf-219 Ru short-epoch wrapper:
1. Write NEW GPU launch wrapper: scripts/launch_wf306b_wf219_ru_gpu.sh
   - USER_OK gate enforced
   - Reduce epoch target: 50 → 30 (per Wf-304C: Ru timeout at 43/50)
   - wall-budget: ~10 min (Ru × 3 seeds × 30 epochs on RTX 5090 cuda:1)
2. Document in molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md

OUTPUT (PLAIN TEXT): todo220_temporal_dry_run_status + todo220_nci60_dry_run_status + n_frontier_adapters_smoked + ru_wrapper_path + user_ok_status. Do NOT call StructuredOutput.`, {
    phase: 'Improvement round 2 (parallel)',
    label: 'wf306B',
  }),

  () => agent(`Wf-306C: Pre-flight v2 → v3 upgrade (25 → 30+ gates) + §9 honest framing catalog expansion (9 → 13+) + AnchorGuard version-pin CI gate.

USE Bash + Read + Edit.

CONTEXT: Per Wf-300D: harness auto-fallback SHIPPED at ~/.claude/wrapper.py. Per Wf-304A: AnchorGuard 6 sites adopted but version-pin path unused. Per TODO-207 §9: 9 honest framing entries (Wf-263B expansion). Pre-flight v2 25/25 PASS stale. Need v3 upgrade.

TASK PART 1 — Pre-flight v2 → v3 upgrade:
1. Read molmetal/scripts/pre_flight_check.py — current 25-gate structure
2. Add NEW gates (target: 30+ total):
   - Gate: frontier math wire-in (multi_ph + se3_lipman + se3_sb + transport_relaxation importable)
   - Gate: TODO-220 drivers present (wf219_temporal_split_driver.py + nci60_9cell_chemprop_driver.py)
   - Gate: AnchorGuard version-pin code path exercised (molmetal/scripts/read_anchor_json.py has version handling)
   - Gate: Wf-304A consumer adoption (paper_auto_update + paper_dashboard + pre_flight_check use AnchorGuard)
   - Gate: Wf-300D harness auto-fallback present (~/.claude/wrapper.py)
   - Gate: Wf-304B PlatinAI Selenium diagnostic present (block_state.html + .meta.json)
   - Gate: §9 honest framing catalog has ≥ 13 entries
3. Verify pre_flight_check.py runs:
   ${PY} molmetal/scripts/pre_flight_check.py 2>&1 | tail -10
4. Target: 30/30+ PASS (or honest disclosure of which gates FAIL)

TASK PART 2 — §9 honest framing catalog expansion (9 → 13+):
1. Read TODO/completed/207_unified_pre_flight_checklist_v2.md §9 — current 9 entries
2. Add NEW entries:
   - §9.7 drugood_v3_scaffold BELOW_RANDOM (Wf-300A + Wf-253D)
   - §9.8 Vina 0/100 docked on Pt+2 (Wf-299B + Wf-301B honest-negative)
   - §9.9 PlatinAI per-cell-line IC50 NUMERIC = 0/0/0/0 (Wf-304B Selenium CF BLOCK)
   - §9.10 Frontier math 0/4 wired (Wf-304C audit) → Wf-305B 4/4 wired (newly measurable)
   - §9.11 Wf-219 Ru seed all timeout (43/50 epochs) per Wf-304C Task 1
   - §9.12 TODO-220 temporal-split + NCI-60 drivers not yet run (Wf-305C stub-only)
3. Update entry count: 9 → 13+ entries

TASK PART 3 — AnchorGuard version-pin CI gate:
1. Read molmetal/scripts/read_anchor_json.py — AnchorGuard API
2. Read molmetal/tests/test_read_anchor_json.py — current tests
3. Add NEW CI gate to pre_flight_check.py:
   - Check that all anchors with 'version' field are pinned at expected version '1.0.0'
   - Add test: test_anchor_version_pin_blocks_regression
4. Run test: ${PY} -m pytest molmetal/tests/test_read_anchor_json.py -v 2>&1 | tail -10

OUTPUT (PLAIN TEXT): n_preflight_gates_total + preflight_status + n_honest_framing_entries + version_pin_test_pass. Do NOT call StructuredOutput.`, {
    phase: 'Improvement round 2 (parallel)',
    label: 'wf306C',
  }),

  () => agent(`Wf-306D: CFM property panel actual launch + G6 user-OK-gated launch wrapper.

USE Bash + Edit.

CONTEXT: Per Wf-264 audit: paper §4 claim audit SHIPPED (90 claims, 0 aspirational, 5 §3.5 violations). CFM property panel NOT STARTED at Wf-265-Collect time. Per Wf-296C: G6 launcher wired with 60/60 CONFIRMED tuples but no USER_OK gate (per Wf-300C pattern).

TASK PART 1 — CFM property panel launch:
1. Read molmetal/scripts/cfm_property_panel.py — Wf-264 panel script
2. Read molmetal/reports/wf188_m14_full_metalcytotox/ — m14 LEAD anchor (Wf-287A)
3. Run panel on n=100 stratified Wf-188 sample:
   ${PY} molmetal/scripts/cfm_property_panel.py --n-mols 100 --csv molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --out-dir molmetal/reports/wf306d_cfm_panel 2>&1 | tail -30
4. Verify: SA / QED / diversity stats emitted
5. Compare to field-avg: SA 2.65-2.86 (TargetDiff) / QED 0.5-0.7 (druglikeness)

TASK PART 2 — G6 user-OK-gated launch wrapper:
1. Read scripts/launch_wf124_p3_g6.sh — current G6 launcher (Wf-296C dry-run PASS, 60/60 CONFIRMED)
2. Add USER_OK gate (per Wf-300C pattern):
   - if [ -z "$USER_OK" ]; then echo "USER OK REQUIRED — kill_then_adapt standing directive"; exit 1; fi
3. Verify bash -n:
   bash -n scripts/launch_wf124_p3_g6.sh && echo SYNTAX_OK
4. Verify dry-run still works (no USER_OK → still dry-runs):
   bash scripts/launch_wf124_p3_g6.sh --dry-run 2>&1 | tail -20

TASK PART 3 — G6 wall-budget + per-wave checklist integration:
1. Read molmetal/scripts/wave_budget_tracker.py — current --no-enforce-budget + --cfm-yuelbond-required (Wf-291D/296C)
2. Read molmetal/scripts/per_wave_checklist.py — current per-wave emission (Wf-262 §10.1)
3. Verify G6 launches wire all 4 sub-waves (a..h per Wf-295D §1.3) through wave_budget_tracker + per_wave_checklist
4. Document: total wall ~7.5 hr; per-wave wall-budget tracking; abort recovery integration

Honest framing: CFM property panel may have stubs/placeholders (per Wf-264 audit); G6 wall is 7.5 hr so user OK gate is mandatory.

OUTPUT (PLAIN TEXT): cfm_panel_status + sa_qed_diversity_emitted + g6_wrapper_user_ok_gate + g6_total_wall + g6_n_waves. Do NOT call StructuredOutput.`, {
    phase: 'Improvement round 2 (parallel)',
    label: 'wf306D',
  }),
])

phase('Synthesis')

const wf306E = await agent(`Wf-306E: Aggregate Wf-306 improvement round 2 outcomes.

USE Read only.

UPSTREAM RESULTS:
- Wf-306A: drugood + Vina CPU smoke + GPU wrapper
- Wf-306B: TODO-220 + frontier math smoke + Wf-219 Ru wrapper
- Wf-306C: Pre-flight v3 + §9 expansion + AnchorGuard CI gate
- Wf-306D: CFM panel + G6 user-OK wrapper

TASK:
1. Read all 4 outputs
2. Aggregate: completed + failed + partial
3. Identify remaining blockers
4. Output to molmetal/reports/wf306_synthesis/final_state.md

OUTPUT (PLAIN TEXT): per_agent_summary + improvements_shipped + remaining_blockers (8-12 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf306E',
})

return {
  ultracode: 'wf-306-improvement-round2',
  wf306a: wf306A,
  wf306b: wf306B,
  wf306c: wf306C,
  wf306d: wf306D,
  wf306e: wf306E,
}