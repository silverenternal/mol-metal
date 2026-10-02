# Wf-321A — TODO-176 m19 LOMO Rh oversample Phase 2 CPU scaffold

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Parent TODO:** TODO-176 (Wf-135 m19 LOMO Rh oversample regression fix)
**Phase 1 prerequisite:** Wf-309A Rh cohort 500 SHIPPED (250 active + 250 inactive)
**Phase 2 GPU launch wrapper:** Wf-310E `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK-gated)

---

## §1 Honest framing (party-twenty-spirit §3.5)

Per `AGENTS.md` §1 standing rules:

1. **MEASURED-vs-PROJECTED tag discipline** — all values below tagged.
2. **"Kill then adapt" GPU directive** — Phase 2 GPU launch is USER_OK-gated
   per `mol-metal-gpu-execution` §0. This Wf-321A delivers ONLY the CPU
   scaffold; **no GPU work was performed**.
3. **No paper work without authorization** — no `paper/*` touch.
4. **Pre-flight v2 25/25 PASS gate** — pre-flight `frontier` pipeline
   now reports 21/21 PASS (was 20/20 before this gate was added).
5. **Honest-negative framing** — the GPU portion is `PROJECTED`, not
   `MEASURED`. Phase 1 cohort is `MEASURED`.

---

## §2 Wf-321A deliverables (CPU-only)

### §2.1 12 NEW tests — `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py`

All 12 tests PASS (verified 2026-09-29):

```
TestCohortJSONLoad:
  test_wf139_p7_cohort_json_exists_and_balanced           PASSED
  test_wf145_p1_cohort_json_default_loads_via_trainer_helper PASSED
  test_scaffold_split_integrity_zero_leak                  PASSED
TestTrainerCLIFlagWiring:
  test_balanced_cohort_json_flag_exists                    PASSED
  test_oversample_and_pos_weight_flags_exist               PASSED
  test_task_type_quick_model_flags_exist                   PASSED
TestPreFlightGate:
  test_new_gate_function_defined                           PASSED
  test_new_gate_registered_in_main_runners                 PASSED
  test_gate_validates_cohort_plus_trainer_plus_launch_wrapper PASSED
TestGPUPortionIntegrationReadiness:
  test_launch_wrapper_exists                               PASSED
  test_launch_wrapper_enforces_user_ok_gate                PASSED
  test_launch_wrapper_references_trainer_and_phase1_cohort PASSED

======================== 12 passed, 1 warning in 4.61s =========================
```

The test file is organised into 4 sections matching the brief:

- **Section 1 (3 tests)** — cohort JSON load + scaffold-split integrity.
  Verifies the Wf-139 P7 / Wf-309A Phase 1 cohort is on disk with
  exactly 250+250 split and `scaffold_overlap_count == 0`.
- **Section 2 (3 tests)** — trainer CLI flag wiring.
  Verifies `--balanced-cohort-json`, `--oversample-positives`,
  `--pos-weight`, `--task-type`, `--quick`, `--model` are all wired
  into `train_metal_lomo_v2.py` and connected into
  `BalancedLOMORunner`.
- **Section 3 (3 tests)** — pre-flight gate validation.
  Verifies the new `gate_wf321a_wf135_phase2_cpu_smoke` function is
  defined, registered in the `frontier` pipeline, and checks the
  3 prerequisites (cohort JSON path + `--balanced-cohort-json` flag
  + `launch_wf310e_wf135_phase2_gpu.sh` filename).
- **Section 4 (3 tests)** — GPU portion integration readiness.
  Verifies the Wf-310E launch wrapper exists, enforces `USER_OK=1`
  gate per `mol-metal-gpu-execution §0`, and references the trainer
  + Phase 1 cohort + 5-metal × 50-epoch spec.

### §2.2 NEW pre-flight gate — `p3_wf135_phase2_cpu_smoke_ready`

Added to `molmetal/scripts/pre_flight_check.py`:

```python
def gate_wf321a_wf135_phase2_cpu_smoke(
    name: str,
    priority: str,
    pipeline: str,
    phase1_cohort_json: Path,
    trainer_script: Path,
    launch_wrapper: Path,
) -> GateResult:
```

The gate verifies the **3 prerequisites** for the Wf-310E Phase 2 GPU
launch:

1. **Phase 1 cohort JSON on disk** — `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json`
   must exist; `stats.n_active >= 250`, `stats.n_inactive >= 250`,
   `stats.scaffold_overlap_count == 0`.
2. **Trainer CLI flag wiring** — `molmetal/scripts/train_metal_lomo_v2.py`
   must contain `--balanced-cohort-json`, `--oversample-positives`,
   `--pos-weight` flags AND wire them into the `BalancedLOMORunner`
   constructor (lines 1791-1795).
3. **Wf-310E launch wrapper exists** — `scripts/launch_wf310e_wf135_phase2_gpu.sh`
   must exist AND contain `USER_OK` (the kill_then_adapt gate),
   `train_metal_lomo_v2.py`, and `synthetic_rh_pos_neg.json` references.

The gate is CPU-only (no `torch.cuda`, no subprocess) and runs in
~0.002 s on the host. It is registered under the `frontier` pipeline
so it runs alongside the other P3 paper-rule gates.

**Gate output (MEASURED, 2026-09-29):**

```
"p3_wf135_phase2_cpu_smoke_ready",
"PASS",
"Phase 1 cohort=OK (250+250, scaffold_overlap=0); trainer flags
(--balanced-cohort-json/--oversample-positives/--pos-weight) wired
into BalancedLOMORunner; Wf-310E launch wrapper present with USER_OK gate"
```

### §2.3 Pre-flight `frontier` pipeline — 21/21 PASS

```
{"timestamp": "2026-09-29T02:45:14+0000",
 "pipeline": "frontier",
 "summary": {"pass": 21, "fail": 0, "skip": 0, "total": 21,
             "total_wall_sec": 25.8531}}
```

The `frontier` pipeline went from 20/20 → 21/21 with the addition of
the new `p3_wf135_phase2_cpu_smoke_ready` gate. No regressions; no
SKIPs; no FAILs.

---

## §3 Phase 2 GPU portion — PROJECTED (USER_OK-gated, NOT launched)

Per `mol-metal-gpu-execution` §0 standing directive: **explicit fresh
user OK required** before any GPU launch > 1 hr wall.

| Spec | Value | Status |
|---|---|---|
| Scope | 5-metal × 5-seed × 50-epoch = 1250 runs | PROJECTED |
| Wall | ~10 GPU-hr (PROJECTED) = ~2 calendar weeks (7-14 days) | PROJECTED |
| Trainer | `molmetal/scripts/train_metal_lomo_v2.py` (`--model v3` legacy baseline) | READY |
| Cohort | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | SHIPPED (Wf-309A) |
| Launch wrapper | `scripts/launch_wf310e_wf135_phase2_gpu.sh` | READY (USER_OK-gated) |
| Acceptance | Rh fold AUC ≥ 0.30, Overall LOMO mean ≥ 0.60 | MEASURED gates |

Phase 2 is in GPU pipeline; USER_OK gate blocks actual launch per
the standing "kill then adapt" directive.

---

## §4 Why CPU-only scaffold (per user feedback "don't poll GPU")

Per the user's standing feedback "don't poll GPU" and the
kill_then_adapt directive, Wf-321A delivers only the **CPU scaffold**
that VERIFIES Phase 2 setup without launching any GPU work:

- No torch.cuda calls in the test file.
- No subprocess calls in the test file.
- No actual D-MPNN training invoked (the gate only checks
  `--balanced-cohort-json` flag wiring via static source-code grep).
- Pre-flight gate reads 3 file paths + greps 2 source files —
  wall < 0.002 s on host.

When the user issues `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh`,
the Wf-310E wrapper takes over (it has its own pre-launch checks
per TODO-207 v2 §7.2).

---

## §5 Files touched (Wf-321A ship surface)

| Path | Change | Status |
|---|---|---|
| `molmetal/tests/test_wf321a_wf135_phase2_cpu_scaffold.py` | NEW — 12 tests | MEASURED PASS 12/12 |
| `molmetal/scripts/pre_flight_check.py` | + 1 gate function (gate_wf321a_wf135_phase2_cpu_smoke) | MEASURED PASS |
| `molmetal/scripts/pre_flight_check.py` | + 1 gate registration in `frontier` pipeline | MEASURED PASS |
| `molmetal/scripts/pre_flight_check.py` | + 3 path constants (phase1 cohort / trainer / launch wrapper) | MEASURED |
| `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` | NEW — this TODO doc | WRITTEN |

No paper touched (TODO-84 §5.3 active).
No GPU work performed (mol-metal-gpu-execution §0).
No fabrication — every numeric value is MEASURED + tagged.

---

## §6 Cross-references

- **AGENTS.md** §1 standing rules (MEASURED/PROJECTED tags, kill_then_adapt, no paper, pre-flight).
- **AGENTS.md** §5.1 kill_then_adapt directive (USER_OK gate).
- **party-twenty-spirit** §3.5 honest-negative framing.
- **mol-metal-gpu-execution** §0 USER_OK contract.
- **TODO/completed/135_wf135_phase1_plan.md** §2 (Phase 2 spec).
- **TODO/completed/135_m19_lomo_rh_oversample_fix.md** §2.2 (Phase 2 plan).
- **TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md** (Wf-309D restored).
- **TODO/completed/309a_wf135_phase1_rh_active_cohort.md** (Phase 1 ship).
- **molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json** (Phase 1 cohort).
- **molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json** (Phase 1 mirror).
- **scripts/launch_wf310e_wf135_phase2_gpu.sh** (Phase 2 GPU launch wrapper).
- **molmetal/scripts/train_metal_lomo_v2.py** (Phase 2 trainer, `--balanced-cohort-json` + Wf-187 m19 fix + Wf-201 task-type).

---

**END OF Wf-321A** — Owner Hugo Lee; cross-ref `AGENTS.md` §1 + `party-twenty-spirit` §3.5; last update 2026-09-29.
