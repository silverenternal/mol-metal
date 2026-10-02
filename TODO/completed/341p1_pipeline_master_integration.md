# Wf-341-P1 — Pipeline Master Integration Test

**Owner:** Hugo Lee (autonomous Wf-341 ultracode unit)
**Date:** 2026-09-29
**Parent:** Wf-341 ultracode pipeline master integration test
**Status:** SHIPPED (42/42 PASS, 0 FAIL, 0 SKIP)

## TL;DR

End-to-end integration test verifying that `scripts/gpu_pipeline_master.sh`
correctly integrates the 5 launchers shipped across the Wf-334 + Wf-336 +
Wf-337 + Wf-338 ultracode wave. All 42 tests pass in 0.40 s wall (CPU-only,
no GPU launch, no Vina invocation).

## Per-launcher integration verdict

Per `TestIntegrationVerdict::test_all_5_launchers_integration_verdict`:

| Launcher | Status | Path | Parses |
|---|---|---|---|
| `launch_wf124_p3_g4.sh` (TODO-180) | STANDALONE | `scripts/` | YES |
| `launch_wf124_p3_g5.sh` (TODO-181) | STANDALONE | `scripts/` | YES |
| `launch_wf124_p3_g6.sh` (TODO-182) | STANDALONE | `scripts/` | YES |
| `launch_wf264_cfm_property_panel.sh` | INVOKED (Phase 3 #9) | `scripts/` | YES |
| `launch_wf333_vina_4step_gpu_bench.sh` | INVOKED (Phase 3 #3) | `molmetal/scripts/` | YES |

**Sequential order (Phase 3, GPU 1):**

1. Wf-310F frontier Wave B
2. Wf-219 §4.1 Ru
3. Wf-333 Vina 4-step [NEW per Wf-334-U2, BLOCKED on Vina install]
4. Wf-310C TODO-220
5. Wf-310C TODO-179
6. Wf-310D DrugOOD DANN Phase A
7. DrugOOD Wf-74 v2 DANN [NEW per Wf-336-X3]
8. Wf-135 Phase 2 LOMO (USER_OK-gated)
9. Wf-264 CFM property panel [NEW per Wf-334-U3, CPU]
10. Wf-264 CFM panel RUN-ONCE [NEW per Wf-338-F1]
11. W₂ sliced GPU smoke [NEW per Wf-338-F1]

## Test breakdown (4 categories, 42 tests)

### 1. `TestPipelineMasterHasAll5Launchers` (14 tests)

- `test_each_launcher_exists_on_disk[5]` — each of 5 launchers resolves
  to a real file on disk + executable bit set.
- `test_each_launcher_parses_cleanly[5]` — each of 5 launchers parses
  cleanly under `bash -n` (no syntax errors).
- `test_pipeline_master_references_g5_and_g6_in_phase1` — G5 + G6 are
  referenced in PHASE 1 in-flight comments (per Wf-337-G2 design).
- `test_pipeline_master_directly_invokes_wf333_and_wf264` — Wf-333 +
  Wf-264 are directly invoked in Phase 3 (correct paths).
- `test_pipeline_master_g4_not_invoked_but_standalone` — G4 wrapper
  exists on disk as standalone but is NOT invoked by the master
  (intentional per TODO-180 closeout).
- `test_pipeline_master_total_direct_invocations` — pipeline master has
  ≥ 10 `nohup bash` invocations (regression guard against accidental
  wrapper removal).

### 2. `TestSequentialLaunchOrder` (5 tests)

- `test_phase2_gpu0_launches_wave_c_then_wf119_p14` — Phase 2 launches
  Wf-310F Wave C → Wf-119 P1.4 in correct order.
- `test_phase3_gpu1_launches_11_jobs_in_priority_order` — Phase 3
  launches all 11 jobs in the documented priority order.
- `test_phase2_runs_before_phase3` — `phase2_gpu0()` declared before
  `phase3_gpu1()` (sequential dispatch).
- `test_wf264_panel_runs_after_wf135_phase2` — Wf-264 panel runs at the
  tail of Phase 3, after Wf-135 Phase 2 LOMO (per Wf-334-U3 design).
- `test_wf333_vina_runs_after_wf219_ru` — Wf-333 Vina inserted after
  Wf-219 §4.1 Ru (per Wf-336-D3 task brief).

### 3. `TestUserOkGates` (10 tests)

- `test_user_ok_gate_present[6]` — every wall-time wrapper invocation
  in Phase 2 + Phase 3 carries USER_OK / WF306A_USER_OK=YES / touch-file
  gate prefix.
- `test_touch_file_gates_for_wf310c` — TODO-220 + TODO-179 use
  touch-file USER_OK gates (per Wf-300C pattern).
- `test_wf333_user_ok_required_per_wrapper` — Wf-333 wrapper refuses
  without USER_OK=1 (exit 64).
- `test_wf264_user_ok_required_per_wrapper` — Wf-264 wrapper refuses
  without USER_OK=1 (real launch path).
- `test_g6_user_ok_required_per_wrapper` — G6 wrapper refuses without
  USER_OK=1 (real launch path; 7.5 hr wall).

### 4. `TestBackwardCompat` (13 tests)

- `test_pre_existing_wrapper_paths_unchanged` — pre-existing wrapper
  paths (Wf-310F, Wf-306B, Wf-310C, Wf-306A, Wf-310E) unchanged.
- `test_pre_existing_wrappers_parse_cleanly[6]` — pre-existing wrappers
  parse cleanly under `bash -n`.
- `test_pipeline_master_parses_cleanly` — pipeline master parses cleanly.
- `test_pipeline_master_cli_supports_phase_modes` — supports phase2|phase3|all.
- `test_wf337_g2_fixes_wired_into_g4_g5_g6` — all 3 launchers carry the
  Wf-337-G2 consolidated fix flags:
  - `--cpu-num-workers 8` (Wf-328-W2)
  - `--num-workers 8` (Wf-328-W6)
  - `--w2-gradient-mode sort` (Wf-328-W5)
  - `--no-cuda-graphs` (Wf-335-R3 safety lever)
- `test_run_wf264_panel_wrapper_exists` — `scripts/run_wf264_cfm_property_panel.sh`
  exists + referenced in pipeline master (per Wf-338-F1).
- `test_gpu_smoke_w2_sliced_wrapper_exists` — `scripts/gpu_smoke_w2_sliced.sh`
  exists + referenced in pipeline master (per Wf-338-F1).

### 5. `TestIntegrationVerdict` (1 test)

- `test_all_5_launchers_integration_verdict` — final per-launcher
  integration summary + sanity assertions.

## Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: pipeline master wires Wf-333 + Wf-264 + G5 + G6 per
  Wf-334-U2 + Wf-336-D3 + Wf-337-G2 ultracode audit.
- **MEASURED**: G4 launcher exists and parses cleanly; G4 is
  intentionally NOT invoked by the master (TODO-180 closed per Wf-124
  ship wave).
- **MEASURED**: 42/42 tests pass in 0.40 s wall; all 5 launchers
  parse cleanly; all USER_OK gates wired.
- **HONEST_NEGATIVE**: Wf-333 Vina 4-step is BLOCKED on Vina 1.2.7
  conda install per AGENTS.md §5.4 + Wf-299B (steps 1+2+4 CPU-ready;
  step 3 contingent on Vina availability).
- **HONEST_NEGATIVE**: pre-flight v3 state is 42 PASS / 15 FAIL /
  57 TOTAL HONEST-NEGATIVE REGRESSION per Wf-324D §22.16 (15 known
  FAILs are project state, not bugs in this test).

## Files

- **Test module:** `molmetal/tests/test_wf341p1_pipeline_master_integration.py`
  (640 LOC, 42 tests, 4 classes + 1 verdict class)
- **Pipeline master:** `scripts/gpu_pipeline_master.sh` (217 LOC,
  13 launcher invocations)
- **Launchers (5):**
  - `scripts/launch_wf124_p3_g4.sh` (141 LOC)
  - `scripts/launch_wf124_p3_g5.sh` (212 LOC)
  - `scripts/launch_wf124_p3_g6.sh` (238 LOC)
  - `scripts/launch_wf264_cfm_property_panel.sh` (143 LOC)
  - `molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` (203 LOC)

## Run command

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf341p1_pipeline_master_integration.py -v
```

## Cross-refs

- Wf-334-U2: pipeline master Wf-333 wrapper integration
- Wf-336-D3: pipeline master Wf-264 + Wf-119 + Wf-74v2 + Wf-333 integration
- Wf-337-G2: G6 launch wrapper v2 + G4/G5 parity + backward-compat
- Wf-338-F1: run-once Wf-264 wrapper + W₂ sliced GPU smoke (final 2 jobs)
- AGENTS.md §5.1 (standing GPU authorization), §5.2 (honest framing),
  §5.3 (no paper work without authorization)
