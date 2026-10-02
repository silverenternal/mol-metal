# Wf-294B: Pre-Flight Re-Verify (post-Wf-294A budget fix)

**Date:** 2026-09-29
**Workflow:** Wf-294B
**Baseline:** Wf-288C (24/25 PASS + 1 FAIL)
**Goal:** Verify that the Wf-294A budget fix yields 25/25 PASS on the unified pre-flight checklist.

---

## Command

```bash
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py
```

## Verdict

| Metric              | Wf-288C baseline | Wf-294B (post-budget-fix) |
| ------------------- | ---------------- | ------------------------- |
| n_pass              | 24               | **25**                    |
| n_fail              | 1                | **0**                     |
| total               | 25               | 25                        |
| preflight_walltime  | (baseline)       | **84.41 s**               |
| overall verdict     | FAIL             | **PASS**                  |

**Verdict: 25/25 PASS — Wf-294A budget fix successfully closes the last failing gate.**

## Summary (machine-readable)

```json
{
  "schema_version": "207.v1",
  "summary": {
    "pass": 25,
    "fail": 0,
    "skip": 0,
    "total": 25,
    "total_wall_sec": 84.4127
  }
}
```

## Per-Gate Breakdown (25/25 PASS)

### P1 — chemprop_v2_v3 (12/12 PASS)

| Gate                                | Status | wall_sec |
| ----------------------------------- | ------ | -------- |
| p1_chemprop_indices_fix             | PASS   | 0.000    |
| p1_chemprop_kendall_optimizer       | PASS   | 0.000    |
| p1_chemprop_dann_per_metal          | PASS   | 0.000    |
| p1_chemprop_nan_guard               | PASS   | 0.000    |
| p1_chemprop_grad_clip               | PASS   | 0.000    |
| p1_chemprop_epoch_avg               | PASS   | 0.000    |
| p1_chemprop_ablation_flags          | PASS   | 6.217    |
| p1_chemprop_seeds_min               | PASS   | 4.589    |
| p1_chemprop_master_wass_flag        | PASS   | 5.798    |
| p1_chemprop_master_cath_flag        | PASS   | 5.821    |
| p1_chemprop_click_aug_flag          | PASS   | 5.684    |
| p1_chemprop_smoke                   | PASS   | 45.888   |

### P1 — CFM backbone (6/6 PASS)

| Gate                                 | Status | wall_sec |
| ------------------------------------ | ------ | -------- |
| p1_cfm_script_exists                 | PASS   | 0.000    |
| p1_cfm_config_exists                 | PASS   | 0.000    |
| p1_cfm_decoder_exists                | PASS   | 0.000    |
| p1_cfm_wasserstein_loss_exists       | PASS   | 0.000    |
| p1_cfm_smoke_decoded                 | PASS   | 0.000    |
| p1_cfm_catherine_30metals            | PASS   | 0.000    |

### P2 — data / tests / vina (4/4 PASS)

| Gate                                  | Status | wall_sec |
| ------------------------------------- | ------ | -------- |
| p2_data_metalcytotox_exists           | PASS   | 0.003    |
| p2_data_metallo_drugs_exists          | PASS   | 0.001    |
| p2_tests_pass                         | PASS   | 9.392    |
| p2_vina_installed                     | PASS   | 0.002    |

### P0 — config / data invariants (3/3 PASS)

| Gate                                  | Status | wall_sec |
| ------------------------------------- | ------ | -------- |
| p0_config_schema_verify               | PASS   | 0.111    |
| p0_csv_hash_verify                    | PASS   | 0.046    |
| p0_scaffold_zero_leak                 | PASS   | 0.859    |

## Comparison vs Wf-288C Baseline

| Δ Metric            | Wf-288C → Wf-294B                   |
| ------------------- | ----------------------------------- |
| pass delta          | +1                                  |
| fail delta          | -1                                  |
| Previously failing gate | now PASS — closed by Wf-294A budget fix |

## Notes

- p2_tests_pass wall_sec 9.392 vs prior 4.31 — small variance within normal range; evidence: `pytest PASS: 24 passed, 1 warning in 4.31s` from earlier log; current run also PASS.
- p1_chemprop_smoke remains the dominant single-gate cost (45.9 s of 84.4 s ≈ 54%).
- All 12 chemprop_v2_v3 gates PASS; the budget fix is sufficient.
- All 6 CFM backbone gates PASS; no budget regression on CFM side.

## Files

- Script: `molmetal/scripts/pre_flight_check.py`
- Schema reference: `TODO/completed/207_unified_pre_flight_checklist_v2.md`
- This report: `molmetal/reports/wf294b_preflight_reverify/reverify_report.md`
- Raw output cache: `/tmp/wf294b_preflight.json`