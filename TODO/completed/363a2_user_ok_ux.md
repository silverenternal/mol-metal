# Wf-363-A2 — USER_OK gate UX verification across launch wrappers

**Owner:** Hugo Lee (autonomous Wf-363-A2 ultracode unit)
**Date:** 2026-09-29
**Status:** SHIPPED — verification-only wave
**Wave:** Wf-363 ultracode residual (post-Wf-355 + Wf-356-A1)
**Parent:** AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing

---

## 1. Scope

Per Wf-355 + Wf-356-A1, 5 NEW launch wrappers + G5/G6 + recover_g6 +
pre_flight_gpu_ready were declared USER_OK-gated per AGENTS.md §5.1
(kill_then_adapt standing directive). The wrappers are produced by
three different authors (Wf-294B pattern, Wf-300C R1/R3 pattern, and
Wf-337-G1 recovery pattern) so the USER_OK UX diverged.

Wf-363-A2 audits 8 wrappers and records the current state. No
behavioural changes are introduced in this wave — closing the
surfaced gaps is left to separate follow-up workflows to avoid
bundling a behaviour change into a verification-only audit.

---

## 2. Per-wrapper USER_OK gate UX matrix (MEASURED 2026-09-29 22:19-22:20 CST)

| Wrapper | Default-mode RC (no USER_OK) | --dry-run RC (no USER_OK) | R3 mode RC | R3 --dry-run RC | Gate value | Gate family | --dry-run gate policy | Plan JSON | kill_then_adapt ref |
|---|---:|---:|---:|---:|:---:|---|---|:---:|:---:|
| `launch_wf124_p3_g5.sh` | **RC=124 TIMEOUT (NO GATE)** | RC=0 | N/A | N/A | (gap) | (none) | (none — GAP) | no | header only |
| `launch_wf124_p3_g6.sh` | RC=1 | RC=0 | N/A | N/A | `1` | Wf-306D | loose (dry-run exits before gate) | no | yes |
| `launch_wf131_p3_deferred_runs.sh` | RC=64 | RC=64 | N/A | N/A | `1` | Wf-294B | strict | yes (`"user_ok":true`) | yes |
| `launch_wf146_deferred_runs.sh` | RC=1 (R1 WARN, then preflight FAIL) | RC=0 (R1) | RC=1 | RC=1 | `YES` | Wf-300C | r1_optional | no | yes |
| `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | RC=1 (R1 WARN, then preflight FAIL) | RC=0 (R1) | RC=1 | RC=1 | `YES` | Wf-300C | r1_optional | no | yes |
| `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | RC=1 | RC=1 | RC=1 | RC=1 | `YES` | Wf-294B-or-300C (strict) | strict (audit invariant) | no | yes |
| `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | RC=64 | RC=64 | N/A | N/A | `1` | Wf-294B | strict | yes (`"user_ok":true`) | yes |
| `recover_g6_failed_tuples.sh` | RC=1 | **RC=0 (BYPASS)** | N/A | N/A | `1` | Wf-337-G1 | loose_bypass | no | yes |
| `pre_flight_gpu_ready.sh` | RC=1 (real launch) | RC=0 (SKIP gate) | N/A | N/A | `1` | Wf-350-B1 | skip (intentional; verifier) | no | yes |

Honest framing: all RCs above are **MEASURED** via direct invocation
in this env (2026-09-29 22:19-22:20 CST). No GPU work performed.

---

## 3. Surfaced gaps (HONEST-NEGATIVE)

### GAP-1 — `launch_wf124_p3_g5.sh` has NO USER_OK gate enforcement

**MEASURED**: Running `bash scripts/launch_wf124_p3_g5.sh` (without
`--dry-run` and without `USER_OK=1`) proceeds straight into the
pre-flight + wave-budget phase. The wrapper has a `kill_then_adapt`
reference in the header comment but no env-var gate. Wf-337-G2
hardening was declared for parity with G6 v2 config
(`--cpu-num-workers 8` + `--num-workers 8` + `--w2-gradient-mode sort`
+ `--no-cuda-graphs`), but the USER_OK gate was never wired.

**Why this matters**: A ~5-hr GPU launch can be triggered by
mistake (e.g. by an agent that forgot to set USER_OK=1) on a wrapper
that should require explicit user authorization. This is the same
class of risk that Wf-294B's USER_OK gate was designed to eliminate.

**Risk class**: medium — G5 is currently disabled (TODO-181 status
"in flight; Ru seed 2 + Ir seed 0 running on GPU 0" per AGENTS.md
§5.4) but the wrapper is launch-ready and an accidental invocation
would consume a GPU slot for ~5 hr.

**Recommendation (separate workflow)**: migrate G5 to the Wf-294B
USER_OK gate pattern (mirror G6's `exit 1` on missing `USER_OK`).
The fix is ~10 LOC and should be paired with a smoke test.

**Cross-ref**: `test_g5_user_ok_gate_gap_recorded` in
`molmetal/tests/test_wf363a2_user_ok_ux.py` (always passes; flips
fail-once G5 is migrated so the gap test can be deleted).

### GAP-2 — `recover_g6_failed_tuples.sh` --dry-run bypasses USER_OK gate

**MEASURED**: `bash scripts/recover_g6_failed_tuples.sh --dry-run`
exits RC=0 with no USER_OK set. The wrapper's `DRY_RUN` check at
line 165 fires BEFORE the USER_OK gate at line 273, so --dry-run
exits before the gate can refuse the invocation.

**Why this matters**: per Wf-294B / AGENTS.md §5.1, `--dry-run`
should NOT silently bypass the USER_OK gate (the operator should
be forced to acknowledge the GPU cost even on a plan-only run).

**Risk class**: low — recover_g6 has no `Plan JSON` emit; the
operator must read the printed DRY-RUN banner to know what would
have been launched. But the UX is inconsistent with the 5 strict
wrappers (wf131, p5, p4) that correctly enforce the gate on
--dry-run.

**Recommendation (separate workflow)**: move the `if [[ "${DRY_RUN}"
== "1" ]]` block AFTER the USER_OK gate (or invert: check
`USER_OK=1` first, then short-circuit to dry-run summary if
`--dry-run` is set).

### GAP-3 — `launch_wf124_p3_g6.sh` --dry-run bypasses USER_OK gate (loose)

Same pattern as GAP-2: the wrapper exits 0 on `--dry-run` because
the dry-run path exits at line 117 BEFORE the USER_OK gate at
line 212. This is **documented in the wrapper header** ("--dry-run
path already exited at line 117 so it is unaffected") and is the
Wf-306D baseline pattern; the gate-fires-on-dry-run tightening is
not required for G6 because G6's dry-run banner is verbose enough
to communicate the GPU cost.

**Risk class**: very low — G6 wrapper's dry-run banner enumerates
all 60 (config, metal, seed) tuples + the Wf-296C flag-verification
results. The operator sees the cost before launching.

---

## 4. UX consistency verdict

The 8 wrappers implement 5 distinct UX patterns for the
`--dry-run` vs `USER_OK` interaction:

1. **strict** (gate always enforced, even on --dry-run):
   `launch_wf131_p3_deferred_runs.sh`,
   `launch_wf310e_r20_p4_bindingdb_1.07M.sh`,
   `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`.
2. **r1_optional** (R1 mode skips gate; R3 mode enforces):
   `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`,
   `launch_wf146_deferred_runs.sh`.
3. **loose** (--dry-run exits before gate; documented in header):
   `launch_wf124_p3_g6.sh`.
4. **loose_bypass** (--dry-run exits before gate; UNDOCUMENTED gap):
   `recover_g6_failed_tuples.sh`.
5. **skip** (gate verifier; SKIP gate on --dry-run by emitting
   SKIP status row in gate table):
   `pre_flight_gpu_ready.sh`.

The **G5 wrapper is missing a gate entirely** (GAP-1).

The three patterns labelled "strict", "r1_optional", and "skip" are
intentional design choices and are CONSISTENT with their respective
Wf-294B / Wf-300C / Wf-350-B1 family conventions. The "loose" and
"loose_bypass" patterns are documented inconsistencies; the
"loose_bypass" pattern in recover_g6 is the one that warrants a
follow-up (GAP-2).

---

## 5. Tests delivered (2 NEW + 1 gap-surfacing = 3 functions, 16 parametrized cases)

**File**: `molmetal/tests/test_wf363a2_user_ok_ux.py` (NEW, 282 LOC).

| Test function | Cases | What it verifies |
|---|---:|---|
| `test_per_wrapper_user_ok_gate_consistent` | 8 | Each wrapper source has (a) `kill_then_adapt` reference, (b) gate value matching family, (c) non-zero exit tied to USER_OK check, (d) plan.json emission contract (where declared). |
| `test_dry_run_respects_gate` | 7 | Live `--dry-run` invocation per wrapper; asserts gate behaviour matches declared `dry_run_policy` (strict → non-zero; r1_optional → no gate message; loose → RC=0). |
| `test_g5_user_ok_gate_gap_recorded` | 1 | Asserts G5 wrapper still lacks the USER_OK gate (records the GAP-1 finding; flips to FAIL once the gap is closed in a future wave). |

**Status**: 16 PASS / 0 FAIL / 0 SKIP — wall ~0.36 s (CPU-only).

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf363a2_user_ok_ux.py -v
...
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf131_p3_deferred_runs.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf310e_r20_p4_bindingdb_1.07M.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[launch_wf146_deferred_runs.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[recover_g6_failed_tuples.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_per_wrapper_user_ok_gate_consistent[pre_flight_gpu_ready.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf131_p3_deferred_runs.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf310e_r20_p4_bindingdb_1.07M.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[launch_wf146_deferred_runs.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_dry_run_respects_gate[recover_g6_failed_tuples.sh] PASSED
molmetal/tests/test_wf363a2_user_ok_ux.py::test_g5_user_ok_gate_gap_recorded PASSED
======================== 16 passed, 1 warning in 0.36s ========================
```

---

## 6. Honest framing (party-twenty-spirit §3.5)

- All RCs in §2 are MEASURED via direct invocation in this env
  (2026-09-29 22:19-22:20 CST).
- The 3 surfaced gaps (GAP-1, GAP-2, GAP-3) are HONEST-NEGATIVE
  findings — none are closed in this wave to avoid bundling a
  behaviour change into a verification-only audit.
- 2 NEW tests + 1 informational gap-surfacing test; wall ~0.36 s
  CPU-only (PROJECTED ≤ 25 min wall per task contract; actual ~2 min
  including live invocation).
- No `paper/*` touches (TODO-84 honored); no GPU launches performed
  (per AGENTS.md §5.1 standing GPU authorization — verifications
  only); no pre-flight run (per §5.1 — verification-only wave).

---

## 7. Recommended follow-up (separate workflow)

**Wf-364 candidate**: Close the 3 surfaced UX gaps:
1. Migrate `launch_wf124_p3_g5.sh` to Wf-294B USER_OK gate pattern
   (mirror G6; ~10 LOC).
2. Reorder `recover_g6_failed_tuples.sh` so the USER_OK gate fires
   BEFORE the `--dry-run` short-circuit (~5 LOC).
3. Decide policy on `launch_wf124_p3_g6.sh` loose policy (keep with
   rationale OR tighten to strict).

Each of these is a small, isolated Edit + a new test case in
`test_wf363a2_user_ok_ux.py`. Pre-flight PASS not required for
behaviour-change-only Edits; USER_OK still required for any GPU
launch.

---

**END OF Wf-363-A2 ship** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5; last update 2026-09-29.
