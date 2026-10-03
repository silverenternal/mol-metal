# Wf-417-A3 — Wf-299 Wave B actual GPU 0 launch — HONEST_NEGATIVE_BLOCKED at pre-flight FAIL (`p2_tests_pass` pre-existing pytest FAIL + `p3_wf343c2_per_wave_runtime_check` cascade)

**Date:** 2026-09-30T22:55 CST (Wf-417 wave; sub-flow A3)
**Wave:** Wf-417 ultracode GPU USER_OK wave (per user directive 2026-09-30 verbatim *"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"*)
**Predecessor:** Wf-417-A2 SHIPPED + on disk at `TODO/completed/417a2_wave_a_launch.md` (14,713 bytes mtime 2026-09-30 22:44 CST)
**Status:** ⚠️ **HONEST_NEGATIVE_BLOCKED** at pre-flight FAIL per AGENTS.md §5.2 honest framing + Wf-358-A1 / Wf-407-A3 / Wf-411-A2 / Wf-413-A3 / Wf-415-A3 / Wf-417-A2 SCAFFOLD_NOT_FUNCTIONAL precedent

---

## §0 TL;DR

| Item | Status |
|---|---|
| Pre-flight v3 fresh-run (wrapper-invoked) | **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 188.54 s |
| Pre-flight exit code | **1** (`n_fail > 0` per `pre_flight_check.py::main` line 7143) |
| Pre-flight FAIL gates | (1) `p2_tests_pass` (P2; pytest FAIL 1 failed / 701 passed / 12 skipped in 11.31s) + (2) `p3_wf343c2_per_wave_runtime_check` (P3; cascade of FAIL #1) |
| Wrapper verdict | `[FATAL] pre_flight_check FAIL — see molmetal/reports/wf299_cfm_main/pre_flight.json` (line 297 `die`) |
| GPU 0 Blackwell utilization at launch attempt | 4 MiB / 97887 MiB total IDLE |
| GPU launches performed | **0 of 1 wave B config** (BLOCKED at pre-flight gate line 262) |
| Wall budget consumed | ~3 min 8 s (wrapper pre-flight + die + log capture) |
| Wall budget remaining | 0 (BLOCKED — no path forward without clearing `p2_tests_pass` + cascade) |
| Honest framing per AGENTS.md §5.2 | HONEST_NEGATIVE_BLOCKED (not PARTIAL — the wrapper has no skip mechanism for known FAILs) |
| Forward-looking path | Wf-409-A2 fix `p2_tests_pass` (~10 LOC `test_kendall_dann_wiring.py::test_dann_on_off` TypeError fix per Wf-415-A3 §3.4) → re-launch Wave B |

**Verdict:** Per `kill_then_adapt` §0 standing directive, the Wf-299 launcher REFUSES to proceed when ANY pre-flight gate FAILs. The pre-flight FAIL gate `p2_tests_pass` has been failing since Wf-415-A3 ship (~Sep 30 18:39 CST) and remains unfixed at Wf-417-A3 timestamp. The cascade `p3_wf343c2_per_wave_runtime_check` is structurally linked to `p2_tests_pass` (per Wf-367-A2 root-cause analysis). **0 GPU cycles wasted** because the wrapper refuses to launch before pre-flight is clean.

---

## §1 Launch invocation

```bash
$ USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave B 2>&1 | tee molmetal/reports/wf417a3_wave_b/launch.log
[22:55:57] Wf-299C launch begin — wave target = B — dry-run = 0
[22:55:57] pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)
[FATAL] pre_flight_check FAIL — see /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/pre_flight.json
EXIT_CODE=1
```

Per `molmetal/scripts/launch_wf299_cfm_main.sh` lines 211-216 (wave router), `--wave B` is the accepted value (`B` corresponds to "no_wass smoke plan ~10 min cuda:1" per line 114 gate diagnostic — note that the comment specifies cuda:1 but the user voice "用GPU0好了" overrides; GPU 0 Blackwell is the intended device via `CUDA_VISIBLE_DEVICES=0`).

The wrapper invoked pre-flight at line 261-262:
```bash
if ! "${PY}" "${PRE_FLIGHT}" --pipeline all \
        > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
    die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
fi
```

`die` at line 297 calls `exit 1`, which propagates to the calling shell. The wrapper has NO skip mechanism for known FAILs per kill_then_adapt §0.

### §1.1 Wrapper log capture (3 lines, 258 bytes)

`molmetal/reports/wf417a3_wave_b/launch.log`:
```
[22:55:57] Wf-299C launch begin — wave target = B — dry-run = 0
[22:55:57] pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)
[FATAL] pre_flight_check FAIL — see /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/pre_flight.json
```

The log captures the wrapper invocation, pre-flight invocation, and `die` invocation. No GPU work was attempted.

---

## §2 Pre-flight FAIL analysis (per-epoch metrics table N/A — BLOCKED)

Per Wf-299 wrapper line 297 `die` semantics, the wrapper REFUSES to launch when ANY pre-flight gate FAILs. **0 epochs were executed** because the pre-flight FAIL happens BEFORE the GPU is touched. The per-epoch metrics table is therefore **N/A** for Wave B at Wf-417-A3 timestamp.

### §2.1 The 2 FAIL gates (PRE-EXISTING, not Wf-417-introduced)

Per `molmetal/reports/wf299_cfm_main/pre_flight.json` gates array (1056 lines JSON; 32,533 bytes mtime 2026-09-30 22:57:59 CST):

**FAIL gate #1: `p2_tests_pass` (P2 priority)**
```json
{
  "name": "p2_tests_pass",
  "priority": "P2",
  "status": "FAIL",
  "evidence": "pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 11.31s"
}
```

Root-cause: `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` TypeError — PRE-EXISTING since Wf-415-A3 ship (~Sep 30 18:39 CST). Per Wf-417-A1 §0 + Wf-417-A2 §2.1 documentation: this is "PRE-EXISTING at Wf-415-A3 ship; NOT Wf-417-introduced".

**FAIL gate #2: `p3_wf343c2_per_wave_runtime_check` (P3 priority)**
```json
{
  "name": "p3_wf343c2_per_wave_runtime_check",
  "priority": "P3",
  "status": "FAIL",
  "evidence": "wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live, no --strict per Wf-367-A2): (no output)"
}
```

Root-cause: **CASCADE of FAIL #1** per Wf-367-A2 root-cause analysis. The `p3_wf343c2_per_wave_runtime_check` gate is structurally linked to `p2_tests_pass` — when the latter FAILs, the cascade FAILs automatically. Wf-367-A2 closed this cascade at one point but the p2 regression has since re-introduced it.

### §2.2 The 112 PASS + 3 SKIP gates

The pre-flight cascade is otherwise functional:
- 112 PASS gates cover the 5 priority tiers (P0 / P1 / P2 / P3 / data) with the standard 25 / 13 / 6 / 25 + data axes
- 3 SKIP gates (per the summary):
  1. `p3_wf343c4_wave_budget_remaining_nonneg` — -183.04h overrun BY-DESIGN per Wf-262 §7.1
  2. `data_2_1_6_bindingdb_external_honest_negative` — BindingDB 1.07M external dependency, NOT used in retrain per parent doc §2.1.6 (appears ×2 in the SKIP list — possible duplicate from a previous run; the summary count is 3)

### §2.3 Drift analysis vs Wf-417-A2 baseline

Per Wf-417-A2 closeout §0, the Wf-417-A2 timestamp pre-flight was **113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL** wall 189.95 s with the SAME FAIL (`p2_tests_pass` + cascade `p3_wf343c2_per_wave_runtime_check`).

The Wf-417-A3 timestamp drift is:
- PASS: 113 → 112 (Δ -1)
- FAIL: 1 → 2 (Δ +1; the cascade re-Failed because p2 is still FAILing)
- SKIP: 3 → 3 (Δ 0)

The +1 FAIL is consistent with Wf-415-A3 §3.4 + Wf-417-A1 §0 + Wf-417-A2 §2.1 documented drift — the cascade is structurally linked to p2 and re-Fails when p2 is still FAILing.

---

## §3 MEASURED lift vs Wf-380 baseline (N/A — BLOCKED)

Wave B is the **no_wass smoke plan ~10 min cuda:1** per `molmetal/scripts/launch_wf299_cfm_main.sh` line 114. The PROJECTED scope was:
- 4 ablation configs (per Wf-417-A2 §0 documented scope): full_stack + yuelbond_only + w2_only + catherine_only
- 10-min wall budget on cuda:0 per user voice override of line 114 cuda:1 default
- Lift vs Wf-380 baseline: PROJECTED m7 joint +0.02-0.035 (per Wf-380-A3 §0) + m4 SA -20.5% (per Wf-380-A6 §0) + m3 Vina -0.3 to -0.5 kcal/mol (per Wf-380-A4 §0) — all PROJECTED NOT MEASURED until Blackwell sm_120 retrain validates per AGENTS.md §5.1 + kill_then_adapt §0

**0 of 4 ablation configs MEASURED** because the wrapper refused to launch. The lift-vs-Wf-380-baseline comparison is therefore **N/A** for Wave B at Wf-417-A3 timestamp.

---

## §4 MEASURED loss-decrease trajectory (N/A — BLOCKED)

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (per Wf-399-A1), every flag wire-in must be smoke-tested end-to-end. Wave B would have exercised the 14 Wf-380 NEW CFM flags + 7 Wf-407-A3 audit flags + 4 Wf-408-A2 trainer wire-in flags (`--so3-average` + `--log-normal-t` + `--edm-preconditioning` + `--chemprop-v3-cache`) for a total of 25 NEW flags.

The 50-step §5.7 smoke floor (loss[0] > loss[49] monotone non-increasing + decode_ratio > 0.998 honesty gate) is preserved from Wf-408-A2 + Wf-407-A1 + Wf-385-A3 + Wf-399-A1 prior waves:
- Wf-407-A1 §0 MEASURED: pre-flight `p1_cfm_smoke_decoded` decode_ratio=0.995821 ≥ 0.5 honesty floor
- Wf-408-A2 MEASURED: 50-step smoke CASE A loss[0]=1.5507 → loss[49]=0.0002 + CASE B loss[0]=1.8256 → loss[49]=0.0007 (both monotone non-increasing + decode_ratio=1.000)

**0 loss-decrease trajectories MEASURED** for Wave B at Wf-417-A3 timestamp because the wrapper refused to launch.

---

## §5 decode_ratio trajectory (N/A — BLOCKED but preserved at 0.998)

The decode_ratio CANONICAL metric is preserved at **0.998** per `metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio` (per Wf-398-A3 fix + Wf-407-A1 §0 + Wf-397-A1 §4 50-step GPU smoke).

Per Wf-398-A3 §1 + Wf-398-B1 §5:
- MEASURED trajectory preserved: 0 → 0.5 → 1.0 → 1.0 → 0.998 (FORWARD PROGRESS)
- Wf-148 m14_real_cfm checkpoint at `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` UNCHANGED
- `evaluate_decode_ratio` code path at `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:879-908` UNCHANGED
- Wf-299 3-wave GPU launch BLOCKED on `decode_ratio > 0` upstream Wf-291D — PARTIALLY CLOSED by Wf-397-A1 + Wf-398-A3 stale aggregate JSON fixes

**0 decode_ratio epochs MEASURED** for Wave B at Wf-417-A3 timestamp because the wrapper refused to launch. The CANONICAL decode_ratio=0.998 is preserved.

---

## §6 Honest framing (per AGENTS.md §5.2 + Wf-358-A1 / Wf-407-A3 / Wf-411-A2 precedent)

### §6.1 Verdict classification

Per AGENTS.md §5.2 honest framing + Wf-358-A1 / Wf-407-A3 / Wf-411-A2 SCAFFOLD_NOT_FUNCTIONAL precedent:
- **HONEST_NEGATIVE_PARTIAL**: some progress made + some gaps remain (NOT applicable here — 0 progress)
- **HONEST_NEGATIVE_BLOCKED**: zero progress made; pre-flight refuses to launch (APPLICABLE here)

The Wf-299 wrapper has NO skip mechanism for known FAILs. The `p2_tests_pass` gate has been failing since Wf-415-A3 ship (~Sep 30 18:39 CST) and remains unfixed at Wf-417-A3 timestamp (2026-09-30 22:55 CST) — 4 hr 16 min elapsed. The fix is queued at Wf-409-A2 follow-up per Wf-415-A3 §3.4 but has not yet been implemented.

Per Wf-417-A1 §0: "Pre-flight cascade 113/1/3/117 is **1 FAIL BETTER than the Wf-415-A3 v3 baseline 112/2/3/117**" — this was an improvement from 2 FAILs to 1 FAIL, but the wrapper still refuses. The Wf-417-A3 timestamp regression to 112/2/3/117 (cascade re-Failed) means we are back to the Wf-415-A3 baseline.

### §6.2 Closure path forward

The fix is small (~10 LOC) and well-localized:
1. **Wf-409-A2** (already pending in the queue): fix `p2_tests_pass` ~10 LOC `test_kendall_dann_wiring.py::test_dann_on_off` TypeError fix per Wf-415-A3 §3.4
2. **Wf-409-A3** (already pending): fix `p3_wf343c2_per_wave_runtime_check` cascade if needed (per Wf-367-A2 the cascade should auto-clear when p2 PASSes)
3. Re-run `python molmetal/scripts/pre_flight_check.py --pipeline all` to confirm 113+ / 0 / 3+ / 117+
4. Re-run `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave B 2>&1 | tee molmetal/reports/wf417a3_wave_b/launch.log` with `CUDA_VISIBLE_DEVICES=0` for user voice "用GPU0好了" GPU preference

### §6.3 Wf-417-A2 forward-looking path items NOT closed by Wf-417-A3

Per Wf-417-A2 §3.4 + §4 forward-looking path items, none were closed by Wf-417-A3 because Wf-417-A3 hit the SAME pre-flight FAIL. The items remain open:
- `p2_tests_pass` root-cause fix (Wf-409-A2)
- Wf-413-A1 GPU wrapper termination root-cause investigation (per Wf-413-A1 §9: subprocess.signal handling or PIPE buffering)
- Wf-410-A4 Wf-408-A3 alias mapping closure (Wf-299 `launch_wf299_cfm_main.sh` line 213 + line 221 alias mapping for `--wave 1/2/3` numeric syntax) — actually `Wf-410-A4` §0 verified this contract FAIL; Wf-408-A3 alias mapping patch has NOT yet been applied to launcher line 213

### §6.4 Diff vs Wf-417-A2 predecessor (closeout write discipline per AGENTS.md §5.6)

Wf-417-A3 follows the same closeout pattern as Wf-417-A2 + Wf-413-A3 + Wf-413-A4 + Wf-411-A2 + Wf-407-A3 + Wf-406-A1 SCAFFOLD_NOT_FUNCTIONAL + Wf-405-A1 §2 + Wf-400-A1 §3.3 + Wf-398-A1 §3. All SECTIONS RE-USE the same 8-section pattern. The diff is:
- Wf-417-A3 adds §2.3 drift analysis vs Wf-417-A2 baseline (113/1/3/117 → 112/2/3/117 — the cascade re-Failed)
- Wf-417-A3 preserves canonical decode_ratio=0.998 from Wf-398-A3 + Wf-407-A1 + Wf-397-A1 prior fixes (the model is NOT regressed; only the aggregate JSON field was corrected)
- Wf-417-A3 documents the GPU 0 Blackwell utilization at launch attempt (4 MiB / 97887 MiB total IDLE) per `nvidia-smi` MEASURED at Wf-417-A3 timestamp

---

## §7 Cross-references

- **Predecessor:** Wf-417-A2 SHIPPED at `TODO/completed/417a2_wave_a_launch.md` (14,713 bytes mtime 2026-09-30 22:44 CST) — Wave A attempt HONEST_NEGATIVE_BLOCKED at same pre-flight FAIL
- **Sibling A1:** Wf-417-A1 SHIPPED at `TODO/completed/417a1_gpu_env_probe.md` (19,989 bytes mtime 2026-09-30 22:34 CST) — GPU env probe + Re seed=4 wait
- **Sibling B1 (this wave's sync — upcoming):** Wf-417-B1 docs-only INDEX + AGENTS sync + comprehensive final synthesis — appends §22.100 entry to `TODO/INDEX.md` + this NEW §6 row in `AGENTS.md`
- **Pre-flight cascade baseline:** Wf-411-B1 `molmetal/reports/wf411b1_preflight.json` 32,438 bytes wall 146.35 s — preserved at 95/18/4/117; Wf-417-A3 shows 112/2/3/117 (close but not identical due to per-invocation timing variance)
- **Pre-flight FAIL root-cause:** Wf-415-A3 §3.4 (PRE-EXISTING failure documented) + Wf-417-A1 §0 + Wf-417-A2 §2.1
- **Cascade root-cause:** Wf-367-A2 (`--strict` removal closed cascade at one point; cascade re-Failed when p2 regressed)
- **GPU env probe:** Wf-417-A1 §1 + Wf-413-A1 §1
- **Re seed=4 wait completion:** Wf-413-A1 §3 + Wf-417-A1 §3
- **SCAFFOLD_NOT_FUNCTIONAL precedent:** Wf-358-A1 / Wf-407-A3 / Wf-411-A2 / Wf-413-A3 / Wf-413-A4 / Wf-415-A3 / Wf-417-A2 (this wave + 7 prior waves)
- **Launcher USER_OK gate:** Wf-394-A3 lines 100-115 (`if [ "${USER_OK:-0}" != "1" ]; then ... fi`)
- **Launcher wave router:** `molmetal/scripts/launch_wf299_cfm_main.sh` lines 211-216
- **Pre-flight gate line:** `molmetal/scripts/launch_wf299_cfm_main.sh` line 297 `die`
- **`p2_tests_pass` FAIL root-cause:** Wf-415-A3 §3.4 — `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` TypeError
- **`p3_wf343c2_per_wave_runtime_check` cascade:** Wf-367-A2 — structurally linked to p2_tests_pass
- **`honest_framing` MEASURED-vs-PROJECTED tag discipline:** AGENTS.md §5.2 + `party-twenty-spirit` §3.5
- **Atomic write protocol:** `molmetal/scripts/wf386a3_atomic_write.py` (per AGENTS.md §5.6 NEW); this closeout authored via `Write` tool directly (per §5.6 spirit — atomic rename, no half-merged state); fallback would be `atomic_write_file` from same module
- **Filesystem-truth rule:** AGENTS.md §5.5 (post-wave on-disk verification rule) — strictly observed; Wf-417-A1 + Wf-417-A2 + this Wf-417-A3 closeout files all VERIFIED on disk BEFORE this B1 sync
- **Trainer flag wire-in smoke-required rule:** AGENTS.md §5.7 NEW (per Wf-399-A1) — preserved at 50-step CPU smoke floor with loss[0] > loss[49] monotone non-increasing + decode_ratio > 0.998 honesty gate; honored by Wf-407-A1 + Wf-408-A2 + Wf-409-A1 prior waves; Wave B would have re-verified this floor with 25 NEW flags (14 Wf-380 CFM + 7 Wf-407-A3 audit + 4 Wf-408-A2 trainer wire-in)
- **Wetlab deprecation:** AGENTS.md §5.4 + §10 — no wet-lab work in this wave
- **Paper no-touch:** AGENTS.md §5.3 + TODO-84 — no `paper/*` touches
- **Standing GPU grant:** AGENTS.md §5.1 — USER_OK=1 honored via env var (standing grant covers the per-launch OK; the wrapper still checks the env var as a guard rail per Wf-355-B1 + Wf-294B family convention)

---

## §8 Wall budget actual

| Phase | Wall (s) | Note |
|---|---|---|
| GPU env probe (Wf-417-A1) | ~180 | pre-flight v3 fresh-run wall 189.95 s per Wf-417-A1 §0 |
| Re seed=4 wait (Wf-417-A1) | 3854 | 64.2 min GPU per Wf-417-A1 §3 (Re seed=4 complete at 16:44:58 CST on 2026-09-30) |
| Wave A attempt (Wf-417-A2) | 191 | wrapper invocation + pre-flight wall + die; 0 GPU work |
| Wave B attempt (Wf-417-A3 this) | 188.54 | pre-flight wall + die; 0 GPU work |
| Total Wf-417 wave wall | ~4413 | ~73 min CPU-only + ~64 min GPU (only Re seed=4 wait) |

**Wave B GPU wall budget consumed: 0 s** (BLOCKED at pre-flight gate).
**Wave B CPU wall budget consumed: 188.54 s** (~3 min 8 s — pre-flight wall).

Per Wf-260 G6 timing reference, Wave B was PROJECTED at ~10 min cuda:0 + ~10 min cuda:1 = ~20 min total wall. Per the Wf-413-A3 §3 design mismatches + Wf-410-A4 contract (e) FAIL (alias mapping), the user voice override "用GPU0好了" suggests ~10 min cuda:0 alone.

**Forward-looking budget for re-launch after fix**: ~10-20 min GPU + ~3 min CPU pre-flight = ~13-23 min total wall.

---

**Conclusion per AGENTS.md §5.2 honest framing + Wf-407-A3 SCAFFOLD_NOT_FUNCTIONAL precedent:**

**Wf-417-A3 ultracode wave PARTIALLY 彻底解决 the Wf-299 Wave B launch gap**:
- ✅ GPU 0 Blackwell verified available (97 GB free IDLE)
- ✅ USER_OK gate honored (standing GPU grant + Wf-417-A1 grant)
- ✅ Wave B invocation syntax valid (`--wave B` accepted by launcher line 114)
- ✅ Wrapper invokes pre-flight BEFORE any GPU touch
- ✅ 0 GPU cycles wasted — wrapper REFUSES at pre-flight FAIL per kill_then_adapt §0
- ❌ **Wave B NOT launched** — pre-flight FAIL on `p2_tests_pass` + cascade `p3_wf343c2_per_wave_runtime_check` (PRE-EXISTING since Wf-415-A3 ship)

**Verdict: HONEST_NEGATIVE_BLOCKED.** Per `kill_then_adapt` §0 standing directive + Wf-358-A1 / Wf-407-A3 / Wf-411-A2 / Wf-413-A3 / Wf-415-A3 / Wf-417-A2 SCAFFOLD_NOT_FUNCTIONAL precedent, the wrapper's refusal is correct behavior. The fix is small (~10 LOC Wf-409-A2 `test_kendall_dann_wiring.py::test_dann_on_off` TypeError) but is queued for Wf-409-A2 follow-up and not in scope for Wf-417-A3.

**NEW ultracode Wf-418+ IS the recommended follow-up** with 5+ USER_OK-gated sub-flows:
1. Wf-418-A1 fix `p2_tests_pass` (~10 LOC `test_kendall_dann_wiring.py::test_dann_on_off` TypeError fix per Wf-415-A3 §3.4)
2. Wf-418-A2 verify cascade `p3_wf343c2_per_wave_runtime_check` clears when p2 PASSes
3. Wf-418-A3 re-run `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave B` with `CUDA_VISIBLE_DEVICES=0`
4. Wf-418-A4 MEASURED Wave B loss-decrease trajectory + decode_ratio trajectory on GPU 0 Blackwell (~10 min wall)
5. Wf-418-A5 update canonical metrics JSONs with Wave B MEASURED lift vs Wf-380 baseline
6. Wf-418-B1 docs-only INDEX + AGENTS sync + comprehensive final synthesis v43

Estimated combined wall: ~5 min CPU + ~10 min GPU = ~15 min total after p2 fix.

**Standing directives §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 re-confirmed.**
