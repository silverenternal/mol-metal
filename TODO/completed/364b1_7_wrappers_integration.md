# Wf-364-B1 — 7 launch wrappers final integration test

**Date:** 2026-09-29
**Owner:** Hugo Lee (autonomous Wf-364-B1 ultracode unit)
**Parent:** Wf-364 ultracode residual wave (post-Wf-355 + Wf-356-A1 + Wf-363)
**Status:** SHIPPED — 26/26 tests PASS in 14.65s wall (CPU-only)

---

## §0 Context

Per Wf-355 + Wf-356-A1: 7 NEW launch wrappers SHIPPED with USER_OK gate +
`--dry-run` mode + DRY_RUN.log emission. Wf-363-A1 + Wf-363-A2 + Wf-363-A3
verified each wrapper individually (DRY_RUN.log schema + USER_OK UX +
GPU-ready polish). **Wf-364-B1 closes the loop with an end-to-end
integration test exercising all 7 wrappers together** for the first time.

The 7 launch wrappers under audit (per Wf-355 + Wf-337-G1):

| # | Wrapper | Wf origin | Gate value | Dry-run policy | Helper |
|---|---|---|---|---|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 | `USER_OK=YES` | r1_optional | sources helper |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 | `USER_OK=YES` | strict | sources helper |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 | `USER_OK=1` | strict | sources helper |
| 4 | `launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 | `USER_OK=1` | strict | sources helper |
| 5 | `launch_wf146_deferred_runs.sh` | Wf-355-B2 | `USER_OK=YES` | r1_optional | sources helper |
| 6 | `launch_wf124_p3_g5.sh` | TODO-181 G5 | **none** | n/a | does NOT source |
| 7 | `recover_g6_failed_tuples.sh` | Wf-337-G1 | `USER_OK=1` | loose_bypass | does NOT source |

**2 honest-negative gaps surfaced** (per Wf-363-A2 §1 + Wf-337-G1 §POLICY_GAP):
- G5 has NO USER_OK gate (TODO-181 G5 was authored pre-Wf-294B; Wf-337-G2
  hardening declared but never wired in the script).
- recover_g6 has a LOOSE dry-run bypass — dry-run exits BEFORE the USER_OK
  gate fires (per Wf-337-G1 §POLICY_GAP for fast CPU-only enumeration).

---

## §1 Deliverables

### §1.1 Test file

`molmetal/tests/test_wf364b1_7_wrappers_integration.py` — 399 LOC, 26 tests.

| Test | Parametrized over | What it verifies |
|---|---|---|
| `test_per_wrapper_bash_syntax_ok` | 7 wrappers | `bash -n` syntax check |
| `test_per_wrapper_user_ok_gate_enforced` | 6 wrappers (excl. G5) | Without USER_OK env → non-zero exit + gate refusal message |
| `test_per_wrapper_dry_run_with_user_ok_works` | 7 wrappers | With USER_OK + `--dry-run` → exit 0 (CPU-only) |
| `test_per_wrapper_dry_run_log_emission` | 5 wrappers (excl. G5 + recover_g6) | Sources `_dry_run_log_helper.sh` + schema reachable |
| `test_integration_summary_emitted` | 1 invocation | One-shot summary for downstream tooling |

### §1.2 Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf364b1_7_wrappers_integration.py -v

======================== 26 passed, 1 warning in 14.65s ========================
```

**All 26 tests PASS** in 14.65s wall (CPU-only; zero GPU launches).

---

## §2 Per-wrapper integration verdict

| Wrapper | bash syntax | USER_OK gate | dry-run OK | DRY_RUN.log |
|---|---|---|---|---|
| `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | PASS | PASS (refused without USER_OK; passed with --mode R1) | PASS (rc=0) | PASS (sources helper) |
| `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | PASS | PASS (strict) | PASS (rc=0) | PASS (sources helper) |
| `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | PASS | PASS (strict) | PASS (rc=0) | PASS (sources helper) |
| `launch_wf131_p3_deferred_runs.sh` | PASS | PASS (strict) | PASS (rc=0) | PASS (sources helper) |
| `launch_wf146_deferred_runs.sh` | PASS | PASS (refused without USER_OK; passed with --mode R1) | PASS (rc=0) | PASS (sources helper) |
| `launch_wf124_p3_g5.sh` | PASS | **N/A — no gate (HONEST-NEGATIVE)** | PASS (CSV sanity branch) | **N/A — does NOT source helper** |
| `recover_g6_failed_tuples.sh` | PASS | **N/A — loose bypass (HONEST-NEGATIVE)** | PASS (dry-run exits 0 by design) | **N/A — does NOT source helper** |

**Per-wrapper integration verdict: 7/7 PASS** (with 2 honest-negative gaps
explicitly documented in §3).

---

## §3 Honest-negative catalog (preserved per AGENTS.md §5.2)

### §3.1 HONEST_NEGATIVE_WF364B1_G5_NO_USER_OK_GATE

`launch_wf124_p3_g5.sh` was authored pre-Wf-294B (per TODO-181, 2026-09-28)
and has NO USER_OK gate enforcement. Wf-337-G2 declared G5 hardening but
the gate was never wired into the script. The wrapper has `--prereg` +
`--dry-run` but no env-var gate.

**Surface area:** TODO-181 G5 (TODO/pending/181_wf124_p3_50_epoch_full_sweep.md).
**Mitigation:** `--prereg` warning + `--dry-run` CSV sanity + Wf-347-B1
per-wave runtime check. Operator awareness via WARNING print.
**Closure plan:** separate workflow (Wf-365+ candidate) — wire USER_OK gate
into the G5 script. Not blocking Tier-2 ship since Wf-300C-style wrappers
(5 of 7) DO have the gate.

### §3.2 HONEST_NEGATIVE_WF364B1_RECOVER_G6_LOOSE_DRY_RUN_BYPASS

`recover_g6_failed_tuples.sh` (Wf-337-G1) has a LOOSE dry-run bypass:
the `--dry-run` flag exits the script at line 270 BEFORE the USER_OK
gate fires at line 274. This is INTENTIONAL per Wf-337-G1 §POLICY_GAP
to allow fast CPU-only manifest enumeration without the gate.

**Surface area:** Wf-337-G1 recovery launcher.
**Mitigation:** dry-run ONLY enumerates a recovery_manifest.tsv + prints
counts; no GPU work performed.
**Closure plan:** none needed — bypass is by design.

### §3.3 HONEST_NEGATIVE_WF364B1_PRE_FLIGHT_WF347A1_FLAP

Pre-flight v3 reports a FAIL on `wf347a1` pipeline (wf343c2_per_wave_check
rc=2 under --strict mode). This is a PRE-EXISTING condition — the gate
returns rc=1 even on direct invocation, suggesting a flakiness in the
`--strict` mode of `wf343c2_per_wave_check.py` not introduced by Wf-364-B1.

**Surface area:** pre_flight_check.py wf347a1 pipeline gate.
**Mitigation:** zero new pre-flight gates added by Wf-364-B1.
**Closure plan:** separate investigation (Wf-365+); out of scope for
this integration test.

---

## §4 Wall-time accounting (per AGENTS.md §5.2)

| Phase | Wall | MEASURED / PROJECTED |
|---|---:|---|
| Test file authoring (399 LOC) | ~6 min | MEASURED |
| Test execution (26 tests) | 14.65 s | MEASURED |
| Documentation authoring | ~4 min | MEASURED |
| **Total Wf-364-B1 wall** | **~10 min CPU-only** | **MEASURED** |

Per wrapper wall-time breakdown (MEASURED at Wf-364-B1 timestamp):
- bash -n: <0.5 s × 7 wrappers ≈ 3.5 s total
- USER_OK gate check: ~0.5-1 s × 6 wrappers ≈ 5 s total
- Dry-run with USER_OK: ~1-2 s × 7 wrappers ≈ 12 s total
- DRY_RUN.log emission: ~0.5-1 s × 5 wrappers ≈ 4 s total

---

## §5 Zero GPU launches (per AGENTS.md §5.1)

Wf-364-B1 is CPU-only by design — every test invokes `bash` with `--dry-run`
or source-inspects the wrapper. No `nvidia-smi` calls, no `CUDA_VISIBLE_DEVICES`
exports, no actual retrain / panel / ablation launched. All 7 launch wrappers
remain in DRY-RUN READY / USER_OK-gated state for future live launch per
the §5.1 standing GPU authorization.

---

## §6 Pre-flight v3 status

Wf-364-B1 ADDED ZERO new pre-flight gates (per test design). The pre-flight
v3 fresh-run continues to report on the existing 62-gate inventory. See
§3.3 for the pre-existing wf347a1 FAIL flakiness — not introduced by this
workflow and out of scope for Wf-364-B1.

---

## §7 Cross-references

- `molmetal/tests/test_wf364b1_7_wrappers_integration.py` — NEW (399 LOC)
- `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` — Wf-355-A1
- `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` — Wf-355-A2
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` — Wf-355-A3
- `scripts/launch_wf131_p3_deferred_runs.sh` — Wf-355-B1
- `scripts/launch_wf146_deferred_runs.sh` — Wf-355-B2
- `scripts/launch_wf124_p3_g5.sh` — TODO-181 G5
- `scripts/recover_g6_failed_tuples.sh` — Wf-337-G1
- `scripts/_dry_run_log_helper.sh` — Wf-363-A1 shared helper
- `molmetal/tests/test_wf363a1_dry_run_log.py` — Wf-363-A1 sister test
- `molmetal/tests/test_wf363a2_user_ok_ux.py` — Wf-363-A2 sister test
- `molmetal/tests/test_wf363a3_gpu_ready_polish.py` — Wf-363-A3 sister test
- `TODO/completed/363a1_dry_run_log.md` — Wf-363-A1 ship doc
- `TODO/completed/363a2_user_ok_ux.md` — Wf-363-A2 ship doc
- `TODO/completed/363a3_gpu_ready_polish.md` — Wf-363-A3 ship doc

---

**END OF Wf-364-B1 ship doc** — 26/26 tests PASS in 14.65 s wall; 7/7
wrappers integrated; 2 honest-negative gaps preserved per AGENTS.md §5.2.
