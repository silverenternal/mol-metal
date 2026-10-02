# Wf-327-F1 — M14 (LR warmup) lever VERIFICATION

**Ship:** 2026-09-29
**Owner:** Claude Code (subagent of Wf-327 ultracode sweep)
**Skill:** `mol-metal-cfm-training` (M14 lever contract — WEAK_TO_STRONG §3.1)
**Status:** **VERIFIED** — Wf-323A integration intact; 8/8 NEW + 12/12 regression tests PASS; 3/3 ablation_runner pre-flight gates PASS

---

## 1.1 Verification outcome

Per Wf-327-T3 triage brief: *"verify Wf-323A status; if already fixed, confirm
+ skip; otherwise execute."*  **Verification confirms Wf-323A is fully fixed;
no further integration work required.**

This is a pure CPU verification step (no GPU launch, no retrain, no flag
re-introduction).  The M14 silent-break documented in
`TODO/completed/319a_ablation_runner_smoke.md` §3.5 was resolved by Wf-323A
on 2026-09-29; this Wf-327-F1 confirms the resolution holds under a fresh
test run on 2026-09-29.

---

## 1.2 What was re-verified

### Source files (intact from Wf-323A ship)

| File | Change | Status |
|---|---|---|
| `molmetal/scripts/metallo_drug_smoke_retrain.py` | Re-added `--lr-warmup-steps` CLI flag (default 200, per WEAK_TO_STRONG §3.1).  Forwarded to ``LipmanFlowMatchingAdapter(lr_warmup_steps=...)`` at line 365. | PRESENT |
| `molmetal/adapters/flow_matching_lipman/__init__.py` | Constructor accepts ``lr_warmup_steps: int = 200`` at line 1776 (negative rejected at line 1879-1882).  ``train_step`` applies linear LR ramp ``min(1.0, step / lr_warmup_steps)`` at line 2333-2335 (Goyal 2017, arXiv:1706.02677).  ``lr_warmup_steps == 0`` → factor = 1.0 (bit-exact legacy). | PRESENT |
| `molmetal/scripts/ablation_runner.py` | M14 AblationSpec at line 164-170 carries ``extra_args=["--lr-warmup-steps", "0"]`` (correct toggle = OFF for the ablation test). | PRESENT, UNCHANGED |

### Test files (existing — re-run, no NEW tests added)

Per brief: *"If Wf-323A already fixed: SKIP, mark as VERIFIED"* → no NEW
`test_wf327f1_m14_lr_warmup.py` written.  The 8 Wf-323A tests + 12
Wf-319A regression tests already cover the integration contract; this
verification re-runs them on a fresh venv invocation.

| File | Coverage |
|---|---|
| `molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py` | **8 NEW tests** (Wf-323A ship) — flag acceptance + spectral correctness + ablation_runner spec wiring + subprocess integration. |
| `molmetal/tests/test_wf319a_ablation_runner.py` | **12 EXISTING tests** (Wf-319A ship, sign-flipped) — toggleable spec classification + ablation_runner integration + honest-framing contract. |

---

## 1.3 Test results (re-run 2026-09-29)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py -v

molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_harness_accepts_lr_warmup_steps_flag PASSED [ 12%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_adapter_constructor_accepts_lr_warmup_steps_with_default_200 PASSED [ 25%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_adapter_lr_warmup_ramp_is_linear_with_spectral_correctness PASSED [ 37%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_spec_has_lr_warmup_steps_in_extra_args PASSED [ 50%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_spec_target_metric_is_m14_decode_ratio_label PASSED [ 62%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_cmd_construction_includes_lr_warmup_steps_zero PASSED [ 75%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_m14_subprocess_does_not_crash_on_unknown_flag PASSED [ 87%]
molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py::test_default_lr_warmup_steps_is_200 PASSED [100%]
============================== 8 passed in 38.57s ==============================
```

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf319a_ablation_runner.py -v

molmetal/tests/test_wf319a_ablation_runner.py::test_toggleable_specs_are_classified_toggleable PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_toggleable_specs_wire_env_vars_or_extra_args PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_toggleable_specs_lookup_by_id PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_design_only_specs_are_classified_design PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_design_only_run_returns_design_only_dict PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_design_only_specs_reference_correct_owner_workflow PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_aggregate_csv_writes_expected_columns_and_rows PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_aggregate_csv_handles_all_11_ablations PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_final_md_includes_summary_table PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_final_md_observes_honest_framing_contract PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_metallo_drug_smoke_retrain_accepts_required_cli_flags PASSED
molmetal/tests/test_wf319a_ablation_runner.py::test_toggleable_spec_cmd_includes_abl_harness_flags PASSED
============================== 12 passed in 1.30s ===============================
```

**Net:** 8/8 NEW (Wf-323A) + 12/12 EXISTING (Wf-319A regression) = **20/20 PASS**.

---

## 1.4 Pre-flight gate state (re-run 2026-09-29)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py
```

| Gate | Pipeline | Status |
|---|---|---|
| `p3_ablation_runner_tests_present` | wf319a | **PASS** |
| `p3_ablation_runner_aggregate_csv_present` | wf319a | **PASS** |
| `p3_ablation_runner_registry_complete` | wf321e | **PASS** |
| `p3_gate_count_consistency_audit` | all | **PASS** (62 actual == 62 expected) |

All 3 ablation_runner gates **PASS**.  The 12-experiment Wf-188 m14
result.json (`/home/hugo/codes/mol-metal/molmetal/reports/wf188_m14_full_metalcytotox/result.json:
value=0.995821 >= 0.5`) is referenced as the §5 prediction anchor.

**Aggregate pre-flight tally:** 61 PASS / 1 FAIL.

**The single FAIL** is `p3_wf327_cpu_num_workers_sufficient` — missing
launch scripts (`launch_wf124_p3_g4.sh`, `launch_wf124_p3_g5.sh`,
`launch_wf124_p3_g6.sh`, `launch_wf300_cfm_main.sh`,
`launch_wf310b_g6_final.sh`, `retry_g5_missing_pairs.sh`).  **This FAIL is
unrelated to M14**; it's a separate "launch-script availability" check
that pre-existed Wf-327-F1 and is in scope for a different workflow
(Wf-326 launch-script surface, not M14 verification).

---

## 1.5 Honest framing catalog (per Wf-263B §9 expansion)

| Item | Status |
|---|---|
| Was M14 silently broken pre-Wf-323A? | **YES** — flag removed from harness (Wf-319A §3.5) |
| Did Wf-323A fully fix the M14 silent-break? | **YES** — flag re-added, adapter wired, ablation_runner spec toggles it correctly (this Wf-327-F1 verification confirms the fix holds) |
| Is Wf-327-F1 pure CPU verification? | **YES** — no GPU launch, no retrain, no flag re-introduction |
| Was user OK needed for this verification? | **NO** — pure read-only verification (Wf-327-F1 brief §"Honest framing") |
| Are any numbers fabricated? | **NO** — every assertion reads a CLI flag / constructor param / param-group LR / test outcome |
| Does the M14 fix preserve bit-exact legacy behaviour? | **YES** — ``lr_warmup_steps == 0`` → factor=1.0 (no scaling) |
| Does the M14 fix match WEAK_TO_STRONG §3.1? | **YES** — linear 200 steps (Goyal 2017) |
| Does pre-flight show the M14-relevant gates passing? | **YES** — 3/3 ablation_runner gates PASS |
| Is the 1 pre-flight FAIL related to M14? | **NO** — unrelated launch-script availability check |

---

## 1.6 Wf-327-F1 vs Wf-323A delta

| Aspect | Wf-323A | Wf-327-F1 |
|---|---|---|
| **Purpose** | Fix the silent-break | Verify the fix holds |
| **Code change** | YES (flag re-add + adapter wiring) | NO (read-only verification) |
| **NEW tests** | 8 (`test_wf323a_m14_lr_warmup_reintegration.py`) | 0 (per brief: SKIP if already fixed) |
| **Test re-run** | n/a (first run) | 20 (8 NEW + 12 regression) |
| **Pre-flight re-run** | partial (p3_ablation_runner_tests_present FAIL with conftest-pycache issue per §1.4) | full (3/3 ablation_runner gates PASS, 61/62 total) |
| **Status** | INTEGRATION COMPLETE | **VERIFIED** |

The pre-flight `p3_ablation_runner_tests_present` FAIL noted in
Wf-323A §1.4 (conftest-pycache issue) is now resolved — the pre-flight
run on 2026-09-29 (this Wf-327-F1) reports `p3_ablation_runner_tests_present`
as **PASS**.  Root cause was a stale `.pyc` cache from a prior pytest
session; clearing the cache resolved it.  No code change needed.

---

## 1.7 Cross-references

* `molmetal/scripts/metallo_drug_smoke_retrain.py` lines 246-254, 286, 361-365 — harness (re-integration site)
* `molmetal/adapters/flow_matching_lipman/__init__.py` lines 1774-1776, 1872-1882, 2326-2335 — adapter constructor + train_step
* `molmetal/scripts/ablation_runner.py` lines 164-170 — M14 AblationSpec (untouched; toggles via existing `extra_args=["--lr-warmup-steps", "0"]`)
* `molmetal/tests/test_wf323a_m14_lr_warmup_reintegration.py` — 8 NEW tests (Wf-323A ship, this verification re-run)
* `molmetal/tests/test_wf319a_ablation_runner.py` — 12 EXISTING tests (regression re-run)
* `TODO/completed/323a_m14_lr_warmup_reintegration.md` — original ship doc
* `TODO/completed/319a_ablation_runner_smoke.md` — original honest-negative (Wf-319A)
* `TODO/completed/327f1_m14_lr_warmup_verified.md` — this verification doc
* `TODO/MECHANISM_METRIC_MAPPING.md` §5 — M14 row (LR warmup → m14 prediction)
* `WEAK_TO_STRONG_PLAN.md` §3.1 — M14 = LR warmup (linear 200 steps) contract
* `molmetal/reports/wf188_m14_full_metalcytotox/result.json` — §5 prediction anchor (m14 value=0.995821 >= 0.5)

---

## 1.8 Conclusion

**Wf-327-F1 verification result: VERIFIED.**

The M14 (LR warmup) lever integration is intact and operational.  The
8 NEW Wf-323A tests + 12 EXISTING Wf-319A regression tests all PASS,
and the 3 ablation_runner pre-flight gates all PASS.  No further
integration work is required for M14 — Wf-323A closed the
silent-break, and this Wf-327-F1 verification confirms the closure
holds under a fresh test run.

The single pre-flight FAIL (`p3_wf327_cpu_num_workers_sufficient`) is
out of scope for M14 verification and belongs to a separate workflow
tracking launch-script availability (Wf-326 or follow-up).
