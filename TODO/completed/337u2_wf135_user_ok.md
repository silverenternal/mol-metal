# Wf-337-U2 — Wf-135 Phase 2 LOMO USER_OK-gate verify + status report

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent (Wf-337-U2 subagent)
**Status:** SHIPPED — 4/4 NEW tests PASS (CPU-only verification; USER_OK-gate verified end-to-end)
**Honest framing:** MEASURED (not PROJECTED); CPU-only; no GPU work performed; Wf-135 Phase 2 remains USER_OK-BLOCKED per the standing kill_then_adapt directive.

---

## §1 TL;DR

Verified the Wf-135 Phase 2 LOMO launch wrapper's USER_OK gate end-to-end:

1. ✅ **USER_OK gate REFUSES without explicit USER_OK=1** — exits 64 (kill_then_adapt directive enforced).
2. ✅ **Wrapper ACCEPTS with USER_OK=1 + --dry-run** — exits 0, emits `plan.json` ledger.
3. ✅ **500 balanced Rh cohort EXISTS** — `wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` (250 active + 250 inactive, scaffold_overlap=0).
4. ✅ **Path consistency after Wf-334-U2 path bug fix** — wrapper is at `scripts/launch_wf310e_wf135_phase2_gpu.sh` (NOT `molmetal/scripts/`).

**Wf-135 Phase 2 status**: USER_OK-BLOCKED per Wf-325C + Wf-310E. This Wf-337-U2 verifies the gate contract; it does NOT launch any GPU work. The actual 5-metal × 5-seed × 50-epoch re-bench (~10 GPU-hr PROJECTED = ~2 calendar weeks) requires explicit `USER_OK=1` from the user before the wrapper will accept the non-dry-run invocation.

---

## §2 Background

Per **Wf-325C**: Wf-135 Phase 2 LOMO is USER_OK-BLOCKED per the standing `mol-metal-gpu-execution` §0 kill_then_adapt directive (no per-launch user OK required for short jobs, but Phase 2 is 2 weeks wall — explicit OK mandatory).

Per **Wf-310E**: the launch wrapper `scripts/launch_wf310e_wf135_phase2_gpu.sh` is READY and emits a `plan.json` ledger on dry-run.

Per **Wf-309A**: the 500 balanced Rh cohort is SHIPPED at `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` (250 active + 250 inactive, scaffold_overlap=0).

Per **Wf-334-U2**: a path bug was previously fixed — `gpu_pipeline_master.sh` referenced `molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh` (FILE NOT FOUND) and was corrected to `scripts/launch_wf310e_wf135_phase2_gpu.sh`. This Wf-337-U2 verifies that the post-fix path is preserved.

Per **AGENTS.md §5.1** standing GPU authorization: VRAM budget check first, hyperparameters via config files, pre-flight PASS required, dry-run is the default. USER_OK gate is HARD-REQUIRED for > 1 hr GPU launches (Phase 2 is 2 weeks).

---

## §3 Wf-337-U2 deliverables (4 NEW tests — all CPU-only)

### §3.1 Test file — `molmetal/tests/test_wf337u2_wf135_user_ok.py`

All 4 tests PASS (verified 2026-09-29, wall < 0.15 s):

```
TestUserOKGateRefuses:
  test_user_ok_gate_enforces_exit_64                       PASSED
TestUserOKGateAccepts:
  test_accepts_exit_zero_with_user_ok_one                  PASSED
TestRhBalancedCohortExists:
  test_500_balanced_rh_cohort_exists                       PASSED
TestPathConsistencyAfterWf334U2:
  test_path_consistency_after_wf334u2_path_bug_fix         PASSED

======================== 4 passed in 0.15s =========================
```

The test file is organised into 4 sections matching the Wf-337-U2 brief:

- **Section 1 (1 test)** — USER_OK gate end-to-end: invokes the wrapper via subprocess WITHOUT `USER_OK=1` in env, asserts exit code 64 (per Wf-310E line 87), and verifies stderr contains USER_OK + gate/refus/fatal wording (proves the gate was hit, not a fixed exit code bypass).
- **Section 2 (1 test)** — Wrapper accept path: invokes the wrapper WITH `USER_OK=1` + `--dry-run`, asserts exit 0, and verifies `plan.json` was emitted with `user_ok=true` + `dry_run=1` + the 5-metal LOMO spec (Rh/Ru/Ir/Os/Re).
- **Section 3 (1 test)** — Phase 1 cohort on disk: verifies the Wf-309A / Wf-139 P7 cohort JSON has `n_total=500`, `n_active=250`, `n_inactive=250`, `scaffold_overlap_count=0`, and `len(actives)==len(inactives)==250` (cross-check the actual lists vs the stats block).
- **Section 4 (1 test)** — Path consistency: confirms the wrapper lives at `scripts/launch_wf310e_wf135_phase2_gpu.sh` (post-Wf-334-U2 path), the pre-fix path `molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh` does NOT exist (path bug NOT regressed), shebang is bash, and the wrapper references the canonical Phase 1 cohort path + trainer.

### §3.2 MEASURED evidence

The USER_OK gate contract is verified end-to-end, not just by source-code grep:

| Contract | MEASURED verification |
|---|---|
| Wrapper refuses without USER_OK | subprocess exit code = **64**; stderr contains "USER_OK" + "refus" + "FATAL" |
| Wrapper accepts with USER_OK=1 | subprocess exit code = **0**; `plan.json` emitted with `user_ok=true` + `dry_run=1` |
| 500 balanced cohort on disk | `wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` exists; `n_total=500`, `n_active=250`, `n_inactive=250`, `scaffold_overlap_count=0` |
| Path consistency (Wf-334-U2 fix) | `scripts/launch_wf310e_wf135_phase2_gpu.sh` EXISTS; `molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh` DOES NOT EXIST (bug NOT regressed) |

The end-to-end subprocess invocation is the strongest possible verification — it exercises the bash gate logic, env propagation, exit codes, and `plan.json` emission in the same way the actual launch path will.

---

## §4 Wf-135 Phase 2 status — USER_OK-BLOCKED

Per **AGENTS.md §5.1** standing GPU authorization + **Wf-325C + Wf-310E**:

| Spec | Value | Status |
|---|---|---|
| Scope | 5-metal × 5-seed × 50-epoch = 1250 runs | PROJECTED |
| Wall | ~10 GPU-hr (PROJECTED) = ~2 calendar weeks | PROJECTED |
| Trainer | `molmetal/scripts/train_metal_lomo_v2.py` | READY (Wf-201) |
| Cohort | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` | SHIPPED (Wf-309A) |
| Launch wrapper | `scripts/launch_wf310e_wf135_phase2_gpu.sh` | READY (USER_OK-gated, MEASURED) |
| Acceptance | Rh fold AUC ≥ 0.30, Overall LOMO mean ≥ 0.60 | MEASURED gates |
| **USER_OK gate** | **HARD-REQUIRED** | **VERIFIED end-to-end (this Wf-337-U2)** |

**To launch Phase 2**, the user must run:

```bash
USER_OK=1 bash /home/hugo/codes/mol-metal/scripts/launch_wf310e_wf135_phase2_gpu.sh
```

(without `--dry-run`). The wrapper will refuse to run unless `USER_OK=1` is set in env (this Wf-337-U2 verified the refusal path).

---

## §5 Why CPU-only verification (per "don't poll GPU" + kill_then_adapt)

Per the user's standing feedback "don't poll GPU" and the kill_then_adapt directive, Wf-337-U2 delivers only the **USER_OK-gate verification** without launching any GPU work:

- No `torch.cuda` calls in the test file.
- No D-MPNN training invoked (the wrapper's `--dry-run` path is the canonical safe invocation).
- Subprocess calls are limited to the wrapper's gate-check path; no GPU hardware is touched.
- Total wall < 0.15 s on host (4 subprocess invocations).

When the user issues `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh`, the wrapper takes over (it has its own pre-launch checks per TODO-207 v2 §7.2: per-phase timeout, NaN/OOM pattern detection, GPU util monitor, Rh fold AUC regression alarm).

---

## §6 Cross-references

- **AGENTS.md** §1 standing rules (MEASURED/PROJECTED tags, kill_then_adapt, no paper, pre-flight).
- **AGENTS.md** §5.1 kill_then_adapt directive (USER_OK gate, ~1 hr wall threshold).
- **party-twenty-spirit** §3.5 honest-negative framing.
- **mol-metal-gpu-execution** §0 USER_OK contract.
- **TODO/completed/135_wf135_phase1_plan.md** §2 (Phase 2 spec).
- **TODO/completed/135_m19_lomo_rh_oversample_fix.md** §2.2 (Phase 2 plan).
- **TODO/completed/309a_wf135_phase1_rh_active_cohort.md** (Phase 1 ship).
- **TODO/completed/321a_wf135_phase2_cpu_scaffold.md** (Wf-321A closeout — 12 tests + pre-flight gate).
- **TODO/completed/334u2_pipeline_master.md** §2.3 (path bug fix).
- **scripts/launch_wf310e_wf135_phase2_gpu.sh** (Phase 2 GPU launch wrapper, USER_OK-gated).
- **molmetal/scripts/train_metal_lomo_v2.py** (Phase 2 trainer, `--balanced-cohort-json` + Wf-187 m19 fix + Wf-201 task-type).
- **molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json** (Phase 1 cohort).

---

## §7 Files touched (Wf-337-U2 ship surface)

| Path | Change | Status |
|---|---|---|
| `molmetal/tests/test_wf337u2_wf135_user_ok.py` | NEW — 4 tests (USER_OK gate end-to-end) | MEASURED PASS 4/4 |
| `TODO/completed/337u2_wf135_user_ok.md` | NEW — this TODO doc | WRITTEN |

No paper touched (TODO-84 §5.3 active).
No GPU work performed (mol-metal-gpu-execution §0).
No fabrication — every numeric value is MEASURED + tagged.
No existing test files modified — Wf-337-U2 is purely additive.

---

**END OF Wf-337-U2** — Owner Hugo Lee; cross-ref `AGENTS.md` §1 + `party-twenty-spirit` §3.5; last update 2026-09-29.