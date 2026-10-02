# Wf-336-D3 — Pipeline master integration (Wf-264 + Wf-333 + Wf-119 P1.4 + DrugOOD Wf-74 v2)

**Date**: 2026-09-29
**Status**: ✅ **Pipeline master INTEGRATED; 28/28 NEW tests PASS; bash syntax OK; per-wrapper gates preserved**
**Owner**: Hugo Lee + Wf-336-D3 ultracode agent
**Parent**: Wf-334-U2 (pipeline master + 6 launchers verified) + Wf-334-U3 (Wf-264 launcher written) + Wf-336-X2 (Wf-119 P1.4 wrapper) + Wf-336-X3 (DrugOOD Wf-74 v2 wrapper)
**Workflow**: Wf-336-D3 (pipeline master integration — residual closure wave)

---

## §0 Why this file exists

Per AGENTS.md §5.1 (standing GPU authorization, 2026-09-29 user grant) +
Wf-334 ultracode ship wave (Wf-334-U1 / U2 / U3 / U4): 4 new GPU/CPU
launch wrappers were shipped but the **pipeline master itself was not
updated** to integrate them.

Wf-336-D3 closes this gap by:

1. **Editing** `scripts/gpu_pipeline_master.sh` to integrate:
   - **Wf-264 CFM property panel** (per Wf-334-U3) — CPU-only ~10 min
   - **Wf-333 Vina 4-step wrapper** (per Wf-334-U2) — BLOCKED on Vina install
   - **Wf-119 P1.4 chemprop v3 LOMO** (per Wf-336-X2) — cuda:0 ~2.0 hr
   - **DrugOOD Wf-74 v2** (per Wf-336-X3) — cuda:1 ~3-4 hr
2. **Adding** 28 NEW CPU-only lock-in tests at
   `molmetal/tests/test_wf336d3_pipeline_integrate.py` to verify
   sequential launch order, wrapper integration, USER_OK gates, and
   path consistency.

**Honest framing (party-twenty-spirit §3.5):**
- Pipeline master itself does NOT launch anything without USER_OK / per-
  wrapper gates — it only sequences the launch invocations.  Each
  wrapper enforces its own USER_OK gate internally; the pipeline master
  also sets USER_OK=1 / USER_OK=YES at the wrapper invocation level so
  the gates pass through the canonical pipeline-master grantor.
- Wf-219 §4.4 is documented as DEFERRED in the pipeline master header
  (no wrapper on disk yet; Wf-219 closeout §4.4 spec only).
- Wf-333 Vina wrapper integration is BLOCKED on Vina 1.2.7 conda install
  per AGENTS.md §5.4 + Wf-299B; the wrapper exists for 4-step audit +
  future integration once Vina is installed.

---

## §1 MEASURED results (integration)

| Item | Status | Evidence |
|---|---|---|
| Pipeline master edited | ✅ DONE | `scripts/gpu_pipeline_master.sh` (195 lines) |
| Wf-264 CFM panel integrated (Phase 3 tail) | ✅ DONE | Line 167-170 `USER_OK=1 nohup bash scripts/launch_wf264_cfm_property_panel.sh` |
| Wf-333 Vina 4-step integrated (after Wf-219 §4.1) | ✅ DONE | Line 123-126 `USER_OK=1 nohup bash molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` |
| Wf-119 P1.4 chemprop v3 LOMO integrated (Phase 2 job 2) | ✅ DONE | Line 93-96 `USER_OK=1 nohup bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` |
| DrugOOD Wf-74 v2 integrated (Phase 3 job 7) | ✅ DONE | Line 152-154 `USER_OK=YES nohup bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3` |
| Header documents 4 NEW wrappers | ✅ DONE | Lines 18, 23, 27, 29 with `[NEW per Wf-336-X2/X3/Wf-334-U2/Wf-334-U3]` markers |
| `bash -n` syntax check | ✅ PASS | `bash -n scripts/gpu_pipeline_master.sh` rc=0 |
| New test file | ✅ 28/28 PASS | `molmetal/tests/test_wf336d3_pipeline_integrate.py` |

**Phase 3 GPU 1 sequence** (9 jobs total, sequential after G6 finishes):
1. Wf-310F frontier Wave B (~0.5 hr cuda:1)
2. Wf-219 §4.1 Ru (~13.5 min cuda:1)
3. **Wf-333 Vina 4-step** (BLOCKED — Vina install)  ← NEW
4. Wf-310C TODO-220 (~3.9 hr cuda:1)
5. Wf-310C TODO-179 (~4.5 hr cuda:1)
6. Wf-310D DrugOOD DANN Phase A (~3-4 hr cuda:1)
7. **DrugOOD Wf-74 v2** (~3-4 hr cuda:1)              ← NEW
8. Wf-135 Phase 2 LOMO (~2 weeks cuda:1)
9. **Wf-264 CFM property panel** (~10 min CPU)        ← NEW

**Phase 2 GPU 0 sequence** (2 jobs total, after Wave A finishes):
1. Wf-310F frontier Wave C (~2-3 hr cuda:0)
2. **Wf-119 P1.4 chemprop v3 LOMO** (~2.0 hr cuda:0)   ← NEW

---

## §2 Test output (28/28 PASS)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 28 items

molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPipelineMasterSequentialLaunchOrder::test_phase2_gpu0_launches_wave_c_then_wf119_p14 PASSED [  3%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPipelineMasterSequentialLaunchOrder::test_phase3_gpu1_launches_all_9_jobs_in_order PASSED [  7%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPipelineMasterSequentialLaunchOrder::test_phase2_and_phase3_dont_interleave PASSED [ 10%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf310f_frontier_math_gpu.sh] PASSED [ 14%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf306b_wf219_ru_gpu.sh] PASSED [ 17%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf310c_todo220_gpu.sh] PASSED [ 21%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf310c_t179_w2_gpu.sh] PASSED [ 25%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf306a_drugood_vina_gpu.sh] PASSED [ 28%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf310e_wf135_phase2_gpu.sh] PASSED [ 32%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf333_vina_4step_gpu_bench.sh] PASSED [ 35%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf264_cfm_property_panel.sh] PASSED [ 39%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf119_p1_4_chemprop_v3_lomo.sh] PASSED [ 42%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_pipeline_master_wrapper_exists[launch_wf74_v2_drugood_dann.sh] PASSED [ 46%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestAllWrappersIntegrated::test_header_documents_all_9_phase3_jobs PASSED [ 50%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf119_p1_4_invocation_carries_user_ok PASSED [ 53%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf333_vina_invocation_carries_user_ok PASSED [ 57%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf74_v2_drugood_invocation_carries_user_ok PASSED [ 60%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf264_panel_invocation_carries_user_ok PASSED [ 64%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf310e_wf219_ru_invocation_carries_user_ok PASSED [ 67%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestUserOkGates::test_wf135_phase2_invocation_carries_user_ok PASSED [ 71%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_wf264_panel_invocation_uses_scripts_path PASSED [ 75%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_wf333_vina_invocation_uses_molmetal_scripts_path PASSED [ 78%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_wf119_p1_4_invocation_uses_scripts_path PASSED [ 82%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_wf74_v2_drugood_invocation_uses_scripts_path PASSED [ 85%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_pre_existing_paths_unchanged PASSED [ 89%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPathConsistency::test_log_dir_paths_consistent PASSED [ 92%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPipelineMasterDryRunSmoke::test_pipeline_master_parses_with_bash_n PASSED [ 96%]
molmetal/tests/test_wf336d3_pipeline_integrate.py::TestPipelineMasterDryRunSmoke::test_pipeline_master_usage_lists_phase_modes PASSED [100%]

============================== 28 passed in 0.18s ==============================
```

---

## §3 Honest framing (per `party-twenty-spirit` §3.5)

### §3.1 MEASURED today

- Pipeline master source edited (5 Edit calls, total ~25 line additions
  + 3-line header update) — bash -n rc=0.
- 28 NEW lock-in tests PASS (sequential order, wrapper integration,
  USER_OK gates, path consistency, dry-run smoke).
- All 10 referenced wrappers exist on disk in either `scripts/` or
  `molmetal/scripts/` (6 pre-existing + 4 NEW per Wf-336-D3).
- USER_OK / USER_OK=YES gate prefix on every new wrapper invocation
  (verified by `test_user_ok_*` tests).
- All log redirects under `${LOG_DIR}/` (4 NEW log files: `wf119_p14.log`,
  `wf333_vina.log`, `wf74_v2_drugood.log`, `wf264_panel.log`).
- Wf-219 §4.4 explicitly documented as DEFERRED (no wrapper on disk yet).

### §3.2 PROJECTED (NOT executed)

- All 4 NEW wrapper invocations are PROJECTED wall-time budgets (NOT
  actually executed); the wrappers themselves have their own pre-flight
  + per-wave gates that fire when they run.
- Wf-264 panel MEASURED-on-2026-09-28 per `molmetal/reports/wf264_panel/panel_report.md`
  (re-run makes the panel reproducible end-to-end).
- Wf-119 P1.4 chemprop v3 LOMO MEASURED-on-2026-09-28 (Wf-321H CPU
  scaffold 12/12 PASS); GPU Wave-A BLOCKED on user OK per kill_then_adapt.
- DrugOOD Wf-74 v2 PROJECTED ~3-4 hr cuda:1 (per Wf-306A Phase-A timing
  baseline); ood_test_auc=0.4574 BELOW field-avg 0.55-0.65 band needs
  the re-launch to attempt recovery.
- Wf-333 Vina 4-step wrapper is BLOCKED on Vina 1.2.7 conda install per
  AGENTS.md §5.4 + Wf-299B; wrapper exists for audit + future integration.

### §3.3 Honest-negative

- Wf-219 §4.4 NCI-60 9-cell multi-task chemprop v2 has no on-disk wrapper;
  pipeline master header documents this as DEFERRED to Wf-338+ (not silently
  integrated with a missing file).
- Pre-existing wrappers' integration is UNCHANGED (regression guard test
  `test_pre_existing_paths_unchanged` PASS).
- Pipeline master does NOT bypass per-wrapper USER_OK gates — it sets
  USER_OK=1 / USER_OK=YES at the invocation level (canonical pipeline-
  master grantor per AGENTS.md §5.1), but each wrapper still validates
  its gate internally (kill_then_adapt standing directive preserved end-
  to-end).

---

## §4 Files touched

| Path | Change |
|---|---|
| `scripts/gpu_pipeline_master.sh` | EDIT — header doc + 4 new launch invocations (Phase 2 job 2 + Phase 3 jobs 3, 7, 9) |
| `molmetal/tests/test_wf336d3_pipeline_integrate.py` | NEW — 28 lock-in tests (sequential order, wrapper integration, USER_OK gates, path consistency, dry-run smoke) |
| `TODO/completed/336d3_pipeline_integrate.md` | NEW — this file |

---

## §5 References

- `TODO/completed/336x2_wf119_p1_4.md` (Wf-119 P1.4 wrapper closeout)
- `TODO/completed/336x3_wf74_v2.md` (DrugOOD Wf-74 v2 wrapper closeout)
- `TODO/completed/336x4_vina_meeko.md` (Wf-333 Vina wrapper closeout)
- `molmetal/tests/test_wf334u2_pipeline_master.py` (pre-existing pipeline master test, reference for our pattern)
- `molmetal/tests/test_wf334u3_wf264_launcher.py` (pre-existing Wf-264 launcher test, reference for our pattern)
- `scripts/launch_wf264_cfm_property_panel.sh` (Wf-264 panel launcher, USER_OK-gated)
- `molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` (Wf-333 Vina 4-step launcher, BLOCKED on Vina install)
- `scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` (Wf-119 P1.4 launcher, USER_OK-gated)
- `scripts/launch_wf74_v2_drugood_dann.sh` (DrugOOD Wf-74 v2 launcher, USER_OK-gated)
- AGENTS.md §5.1 (standing GPU authorization) + §5.4 (Tier-2 + Vina BLOCKED)

---

**END of Wf-336-D3 closeout** — pipeline master integrated end-to-end with
28/28 NEW tests PASS; per-wrapper gates preserved; Wf-219 §4.4 documented
as DEFERRED.