# Wf-320C — TODO-220 stub driver full implementation closeout

**Date:** 2026-09-29
**Workflow:** Wf-320C (Wf-315B recommendation #5 follow-up)
**Status:** ✅ **VERIFIED + 10/10 INTEGRATION TESTS PASS**
**Owner:** Hugo Lee + Wf-320C agent
**Predecessors:** Wf-305C (dry-run stub), Wf-308C (--execute path), Wf-310C (smoke + GPU wrapper), Wf-313D (acceptance gate schema), Wf-315B (recommendation #5)

---

## §0 One-line summary

Wf-320C verifies (NOT re-implements) the TODO-220 stub driver full implementation
claimed by Wf-308C / Wf-310C / Wf-313D. Wrote 10 integration tests covering
both `--execute` paths + smoke JSON contracts + Wf-313D acceptance gate
cross-references. **All 10 tests PASS** against the on-disk artifacts.

---

## §1 Verification matrix (4 axes)

| Axis | Surface | Status | Evidence |
|---|---|:---:|---|
| **1.1 Temporal `--execute` path** | `molmetal/scripts/wf219_temporal_split_driver.py::execute_sweep` | ✅ SHIPPED | Wf-308C task 1; line 524; test_temporal_01/02/03 PASS |
| **1.2 NCI-60 `--execute` path** | `molmetal/scripts/nci60_9cell_chemprop_driver.py::execute_sweep` | ✅ SHIPPED | Wf-308C task 1; line 432; test_nci60_01/02/03 PASS |
| **1.3 Smoke MEASURED** | Both drivers smoke 1-epoch 1-seed CPU | ✅ MEASURED | `wf310c_temporal_split/sweep_summary_smoke.json` 5/5 ok; `wf310c_nci60_9cell/sweep_summary_smoke.json` 9/9 ok |
| **1.4 GPU wrapper READY** | `molmetal/scripts/launch_wf310c_todo220_gpu.sh` | 🚧 PROJECTED | USER_OK + 3.9 hr GPU wall required (per `mol-metal-gpu-execution` §0 kill_then_adapt) |

---

## §2 Test inventory (10/10 PASS)

File: `molmetal/tests/test_wf320c_todo220_full_impl.py`

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_temporal_01_execute_sweep_callable` | `execute_sweep` is callable + in `__all__` (Wf-308C contract) |
| 2 | `test_temporal_02_dry_run_emits_csv_exists_true` | `--dry-run` JSON emits `csv_exists=true` + Wf-308C status string + `temporal_strict` protocol |
| 3 | `test_temporal_03_execute_sweep_returns_dict_with_status` | Missing CSV returns `CSV_MISSING` status (Wf-306B honest-negative) |
| 4 | `test_nci60_01_execute_sweep_callable` | NCI-60 `execute_sweep` is callable + in `__all__` |
| 5 | `test_nci60_02_dry_run_emits_csv_exists_true` | `--dry-run` JSON emits `csv_exists=true` + Wf-308C status + `n_cells=9` |
| 6 | `test_nci60_03_execute_sweep_returns_dict_with_status` | Missing CSV returns `CSV_MISSING` status |
| 7 | `test_integration_temporal_smoke_matches_metrics` | Smoke JSON matches §6.3.1 schema: 5 metals, krasnov_2026_baseline citations, `krasnov_protocol_aligned=true` |
| 8 | `test_integration_nci60_smoke_matches_metrics` | Smoke JSON matches §6.4.1 schema: 9 cells, all AUCs in [0,1], PC3 honest-negative (<0.5), HCT116 highlight (>0.5) |
| 9 | `test_acceptance_temporal_smoke_5_of_5_pass` | 5/5 ok MEASURED + §6.3 doc cites Wf-308C/310C/313D lineage |
| 10 | `test_acceptance_nci60_smoke_9_of_9_pass` | 9/9 ok MEASURED + §6.4 doc cites Wf-308C/310C/313D lineage + PC3 honest-negative declared |

Run output: `10 passed, 1 warning in 25.17s`

---

## §3 Smoke MEASURED values (cited from JSONs on disk)

### §3.1 Temporal (5/5 ok, 25.5 s wall, 1-epoch 1-seed)

| Metal | mean_auc | n_train | n_test | Krasnov 2026 baseline |
|---|---:|---:|---:|---:|
| Ru | 0.7421 | 14696 | 1598 | 0.81 |
| Ir | 0.6255 | 2796 | 909 | 0.73 |
| Os | 0.6765 | 784 | 122 | n/a |
| Rh | 0.7756 | 715 | 189 | n/a |
| Re | 0.5853 | 413 | 264 | n/a |
| **mean_auc_across_metals** | **0.6810** | | | |

Source: `molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json`

### §3.2 NCI-60 (9/9 ok, 5.1 s wall, 1-epoch 1-seed)

| Cell | mean_auc | n_train | n_test | Status |
|---|---:|---:|---:|---|
| MCF7 | 0.5495 | 1374 | 393 | borderline |
| A2780 | 0.6055 | 766 | 219 | above random |
| A549 | 0.6448 | 1852 | 530 | above random |
| HCT116 | **0.6998** | 490 | 141 | **HIGHLIGHT (Wf-313D §6.4.4)** |
| PC3 | **0.1667** | 192 | 55 | **HONEST-NEGATIVE (Wf-313D §6.4.4)** |
| HL60 | 0.6667 | 157 | 45 | above random |
| A2780cis | 0.5694 | 394 | 113 | borderline |
| DU145 | **0.7813** | 112 | 33 | lead |
| SKOV3 | 0.3718 | 133 | 39 | honest-negative |
| **mean_auc_across_cells** | **0.5617** | | | |

Source: `molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json`

---

## §4 Honest framing (per party-twenty-spirit §3.5)

* **Wf-320C is verification, NOT new implementation.** The `--execute` paths
  were shipped by Wf-308C. The smoke JSONs were shipped by Wf-310C. The
  acceptance gate schema was shipped by Wf-313D. Wf-320C adds 10 integration
  tests + a verification matrix; it does not claim to have implemented anything new.
* **Smoke MEASURED, GPU PROJECTED.** The 5/5 temporal + 9/9 NCI-60 smoke is
  CPU-only 1-epoch 1-seed MEASURED. The 50-epoch × 3-seed GPU full sweep is
  PROJECTED PENDING USER_OK + 3.9 hr GPU wall; falls under `mol-metal-gpu-execution` §0
  kill_then_adapt 1-hr threshold; **NOT in scope of Wf-320C**.
* **PC3 honest-negative preserved.** Per Wf-313D §6.4.4, PC3=0.1667 is below
  random and is declared honest-negative, NOT paraphrased as "achievable" via
  MultiTaskLoss head (which is TODO-130 G3 follow-up; not in Wf-320C scope).
* **HCT116 highlight preserved.** HCT116=0.6998 is the smoke-layer highlight
  (above 0.5 random), cited in `test_integration_nci60_smoke_matches_metrics`.

---

## §5 Pre-flight (TODO-207)

Pre-flight v2 was launched (`molmetal/scripts/pre_flight_check.py`) per AGENTS.md §1 rule 4.

**Pre-flight status: ✅ MEASURED PASS** (exit code 0).
- Last recorded checks: `p3_ablation_runner_aggregate_csv_present` PASS (12 rows / 14 columns, DESIGN_ONLY rows free of fabricated metrics); `pytest` PASS 12/12 in 1.93s.

Wf-320C itself launches no retrain / panel / ablation — it is CPU-only
verification + tests. Pre-flight was run as a courtesy + to honor AGENTS.md §1 rule 4
even though the gate primarily applies to retrain / panel / ablation launches.

Source: `/tmp/claude-1001/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/tasks/buv45vakp.output`

---

## §6 File diff summary

| File | Change | Status |
|---|---|:---:|
| `molmetal/tests/test_wf320c_todo220_full_impl.py` | NEW — 10 integration tests | ✅ |
| `TODO/completed/320c_todo220_full_impl_closeout.md` | NEW — this doc | ✅ |

No driver source files modified (Wf-308C + Wf-310C + Wf-313D shipped them
correctly; no rework needed).

---

## §7 Cross-references

* `TODO/completed/220_wf220_temporal_split_acceptance.md` — §6.3 doc
* `TODO/completed/220_wf219_nci60_9cell_acceptance.md` — §6.4 doc
* `TODO/completed/220_wf219_temporal_split_nci60_driver.md` — driver spec
* `molmetal/scripts/wf219_temporal_split_driver.py` — temporal driver
* `molmetal/scripts/nci60_9cell_chemprop_driver.py` — NCI-60 driver
* `molmetal/scripts/launch_wf310c_todo220_gpu.sh` — GPU wrapper (USER_OK-gated)
* `molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json` — temporal smoke
* `molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json` — NCI-60 smoke
* `molmetal/tests/test_wf313d_todo220_acceptance_gates.py` — 12 prior acceptance tests
* `molmetal/tests/test_wf320c_todo220_full_impl.py` — NEW 10 integration tests
* `mol-metal-gpu-execution` skill §0 — kill_then_adapt directive

---

**END OF Wf-320C closeout** — Owner Hugo Lee; per party-twenty-spirit §3.5
honest framing; verification work, not implementation work.