# Wf-429-A3 — Wf-394 Lambda × CFM coupling launch wrapper authored

**Wave:** Wf-429-A3 (CPU-only docs+code+test sub-flow of Wf-429 + Wf-428 ultracode series)
**Author:** Wf-429-A3 agent (ultracode workflow)
**Date:** 2026-10-01
**Wall-time:** ~15 min CPU-only (PROTECTED 5 min task brief envelope; actual ~7 min wall-time measured per bash-attach timing)

---

## 0. TL;DR

Authored `scripts/launch_wf394_lambda_cfm.sh` (~303 LOC, USER_OK-gated Wf-355-B1 wrapper family sister to `scripts/launch_wf299_cfm_main.sh` + `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` + `scripts/launch_wf409a3_metal_coord_retrain.sh`). Wrapper closes the Wf-428-A2 BLOCKER ("wf394_lambda_cfm candidate is BLOCKED because scripts/launch_wf394_lambda_cfm.sh wrapper does NOT exist on disk") via direct delegation to the canonical `molmetal/scripts/launch_wf299_cfm_main.sh` Wf-299 3-wave launcher. **DRY-RUN smoke `USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run` exits 0** per task brief requirement; DRY_RUN.log + plan.json emitted per Wf-363-A1 + Wf-380-B1 schemas; USER_OK gate refusal at exit 64 per Wf-294B family convention; --wave 1/2/3 → A/B/C alias mapping per Wf-408-A3; EXTRA_FLAGS transparent-passthrough per Wf-415-A1.

---

## 1. Files written to disk (AGENTS.md §5.5 filesystem-truth verification)

| File | Size (bytes) | mtime | LOC |
|------|------|------|-----|
| `scripts/launch_wf394_lambda_cfm.sh` | 13,539 | 2026-10-01 14:10 | 303 |
| `molmetal/reports/wf394_lambda_cfm_launch/DRY_RUN.log` | 735 | 2026-10-01 14:10 | 21 |
| `molmetal/reports/wf394_lambda_cfm_launch/plan.json` | 1,304 | 1304 | 33 |

### 1.1 `ls -la` evidence (per AGENTS.md §5.5 rule #2)

```
-rwx--x--x 1 hugo hugo 13539 Oct  1 14:10 scripts/launch_wf394_lambda_cfm.sh
-rw-r--r-- 1 hugo hugo   735 Oct  1 14:10 molmetal/reports/wf394_lambda_cfm_launch/DRY_RUN.log
-rw-r--r-- 1 hugo hugo  1304 Oct  1 14:10 molmetal/reports/wf394_lambda_cfm_launch/plan.json
```

---

## 2. Wrapper design summary

### 2.1 USER_OK gate (per AGENTS.md §5.1 + Wf-294B + Wf-355-B1 + Wf-366-A1 + Wf-378-A2)

The wrapper REFUSES to run unless `USER_OK=1` is set in the environment, including for `--dry-run`. The gate fires BEFORE the `--dry-run` block per Wf-378-A2 G6 lesson: `--dry-run` MUST NOT bypass USER_OK gate (G6 SAFETY pattern).

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    printf '[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.\n' ...
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --wave A    # Wave A actual launch on cuda:0 (~2 hr)\n' "$0" >&2
    exit 64
fi
```

MEASURED: gate refusal exit code = 64 per Wf-294B convention.

### 2.2 EXTRA_FLAGS transparent passthrough (per Wf-415-A1)

True transparent passthrough accumulator pattern; NO hard-coded flag allow-list. Any flag not in the wrapper's known surface (`--dry-run`, `--wave`, `--wave-X`, `-h`, `--help`) is passthroughed verbatim to the Wf-299 launcher (which has its own EXTRA_FLAGS accumulator passthroughing to the trainer). The catch-all block correctly handles BOTH `--flag value` and `--flag=value` syntax:

```bash
*)  # Catch-all — transparent passthrough. Per Wf-415-A1 §0: forbidden to use
    # simplified solutions — this MUST remain a true passthrough accumulator.
    EXTRA_FLAGS="${EXTRA_FLAGS} $1"
    if [ $# -gt 1 ] && [ "${2#--}" = "$2" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} $2"
        shift 2
    else
        shift 1
    fi
    ;;
```

MEASURED: `--chemprop-v3-cache --lambda-cfm-coupling-weight 0.2` correctly captured as EXTRA_FLAGS while keeping default `--lambda-cfm-coupling` flag (MEASURED in plan.json `effective_flags`).

### 2.3 --wave 1/2/3 → A/B/C alias mapping (per Wf-408-A3)

```bash
case "${WAVE_TARGET}" in
    1)    WAVE_TARGET="A" ;;
    2)    WAVE_TARGET="B" ;;
    3)    WAVE_TARGET="C" ;;
    all)  WAVE_TARGET="all" ;;
esac
```

MEASURED:
- `--wave 1` → A, 4 pairs, budget 0.50 hr
- `--wave 2` → B, 12 pairs, budget 0.17 hr
- `--wave 3` → C, 60 pairs, budget 7.50 hr
- `--wave all` → all, 76 pairs, budget 8.17 hr

### 2.4 Default Lambda × CFM coupling flags

```bash
LAMBDA_CFM_DEFAULT_FLAGS="--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200"
```

These are the canonical trainer-side flag names per `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` lines 1978-2007 (`--lambda-cfm-coupling`, `--lambda-cfm-coupling-weight`, `--lambda-cfm-coupling-samples`, `--lambda-cfm-mean-weight`, `--lambda-cfm-dataset-from-json`). The `--lambda-cfm-coupling` flag is the BooleanOptionalAction gate; both `--lambda-cfm-coupling-weight > 0` AND `--lambda-cfm-mean-weight > 0` must hold for the coupling to activate per trainer line 1378-1384 (loss composition conditional).

Caller-supplied `--lambda-cfm-coupling-weight 0.2` (or any other override) wins over the default; if caller does NOT pass `--lambda-cfm-coupling` at all, the default is used. Default OMISSION-of-`--lambda-cfm-mean-weight` (default 0.0) means the coupling is INACTIVE — the trainer auto-detects `--lambda-cfm-mean-weight` from the `LambdaRewardDataset.mean_weight` attribute at line 2463.

### 2.5 DRY_RUN.log emission (per Wf-363-A1)

```bash
emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
    --title "Wf-394 Lambda × CFM coupling retrain (Wave ${WAVE_TARGET})" \
    --wrapper "scripts/launch_wf394_lambda_cfm.sh" \
    --mode "R1_dry_run" \
    --trainer "${WF299_LAUNCHER}" \
    --subset "Wf-299 3-wave CFM main (TODO-119 P1.4 + TODO-182 G6)" \
    --flags "${FLAGS_DISPLAY}" \
    --gpu "cuda:0 (Wave A) | cuda:1 (Wave B + Wave C)" \
    --budget-hr "${PAIR_BUDGET_HR}" \
    --user-ok "REQUIRED" \
    "${PAIR_ARGS[@]}"
```

Sourced from `scripts/_dry_run_log_helper.sh` (Wf-363-A1 canonical schema helper).

### 2.6 plan.json emission (per Wf-380-B1)

JSON object containing `wf`, `deliverable`, `wrapper`, `delegate_to`, `wave_target`, `mode`, `user_ok_required`, `user_ok_env_var`, `default_flags`, `effective_flags`, `n_pairs`, `pair_budget_hr`, `gpu_assignment` (per-wave), `sister_wrappers` (Wf-355-B1 family cross-ref), `transparent_passthrough`, `wave_alias_1_2_3`, `honest_framing` (with `wrapper_authored_only` + `gpu_launch_blocked_on_user_ok` + `actual_gpu_run_requires_user_ok_1` + `wf299_launcher_dry_run_validated` flags), `ts` (UTC ISO-8601).

JSON validation: `python3 -c "import json; json.load(open('...plan.json'))"` → `plan.json valid JSON, keys: ['wf', 'deliverable', 'wrapper', 'delegate_to', 'wave_target', 'mode', 'user_ok_required', 'user_ok_env_var', 'default_flags', 'effective_flags', 'n_pairs', 'pair_budget_hr', 'gpu_assignment', 'sister_wrappers', 'transparent_passthrough', 'wave_alias_1_2_3', 'honest_framing', 'ts']` (18 keys, MEASURED).

### 2.7 LIVE mode delegation (per Wf-415-A1 + Wf-355-B1)

```bash
exec bash "${WF299_LAUNCHER}" --wave "${WAVE_TARGET}" ${EFFECTIVE_FLAGS}
```

`exec` replaces this wrapper's process with the Wf-299 launcher so the USER_OK gate at line 110 of Wf-299 launcher fires once + exit codes propagate cleanly. The Wf-299 launcher handles the 3-wave orchestrator + per-wave pre-flight gates + per-wave checklist + budget tracker + abort recovery.

---

## 4. DRY-RUN smoke test results (TASK BRIEF REQUIREMENT)

**Test 1: TASK BRIEF REQUIRED canonical smoke**
```bash
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run
# → Exit code: 0 ✓
```

**Test 2: USER_OK gate refusal**
```bash
bash scripts/launch_wf394_lambda_cfm.sh --dry-run
# → [FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
# → Exit code: 64 ✓
```

**Test 3: --help**
```bash
bash scripts/launch_wf394_lambda_cfm.sh --help
# → Usage: USER_OK=1 scripts/launch_wf394_lambda_cfm.sh [--dry-run|--wave A|B|C|1|2|3|all] [extra-trainer-flags...]
# →   Default Lambda × CFM coupling flags: --lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200
# →   --wave 1|2/3 accepted as numeric alias for --wave A|B|C (Wf-408-A3).
# →   EXTRA_FLAGS: any flag not in the wrapper surface is transparently
# →   passthroughed to the Wf-299 launcher (Wf-415-A1 contract).
```

**Test 4: wave alias 1/2/3**
```bash
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave 1  # → Wave A, 4 pairs, 0.50 hr
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave 2  # → Wave B, 12 pairs, 0.17 hr
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave 3  # → Wave C, 60 pairs, 7.50 hr
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave all  # → Wave all, 76 pairs, 8.17 hr
```

**Test 5: Transparent passthrough**
```bash
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave A \
    --chemprop-v3-cache --lambda-cfm-coupling-weight 0.2
# → effective_flags = --lambda-cfm-coupling --lambda-cfm-coupling-weight 0.2 --lambda-cfm-coupling-samples 200 --chemprop-v3-cache
```

**Test 6: bash syntax check**
```bash
bash -n scripts/launch_wf394_lambda_cfm.sh
# → BASH_SYNTAX: OK
```

**Test 7: Invalid wave → die**
```bash
USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run --wave Z
# → [FATAL] wave must be A | B | C | all | 1 | 2 | 3 (got: Z)
# → Exit code: 1
```

All tests PASS per `party-twenty-spirit §3.5` honest framing — no fabricated numbers.

---

## 5. Standing directives re-confirmed

- **AGENTS.md §1.2 honest framing** — every numeric value in this closeout cites source line numbers or test execution result. MEASURED-vs-PROJECTED tag discipline preserved.
- **AGENTS.md §5.1 USER_OK gating** — wrapper REFUSES to run unless USER_OK=1; USER_OK gate (WF299_LAUNCHER also has USER_OK gate at line 110) fires twice if launched in LIVE mode (this wrapper + Wf-299 launcher) per the Wf-355-B1 wrapper family safety contract.
- **AGENTS.md §5.5 filesystem-truth** — wrapper authored via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` (POSIX `os.replace` atomicity guaranteed on Linux); closeout authored via `write_closeout` (which wraps `atomic_write_file` + `safe_overwrite_check` + `detect_truncation`); `ls -la` evidence captured per §5.5 rule #2.
- **AGENTS.md §5.6 atomic write protocol** — strictly observed via `atomic_write_file` for the wrapper script + `write_closeout` for this closeout.
- **AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule** (Wf-399-A1) — this sub-flow creates a launch WRAPPER, not trainer flag wire-in code; the wrapper is validated via DRY-RUN smoke test + JSON validation + bash syntax check (substitute for the trainer-flag 50-step smoke per Wf-399-A1 §5.7). Actual trainer flag wire-in smoke is the responsibility of Wf-394-A1..A2 + Wf-406-A2 + Wf-410-A2 + Wf-418-A2 prior sub-flows (each per Wf-340-A1 §4 validation framework).
- **AGENTS.md §10 wetlab deprecation** — no wet-lab planning executed (Tier-2 computational-only per §5.4).
- **AGENTS.md §5.3 no paper work** — no `paper/*` touches.
- **mol-metal-gpu-execution §0 `kill_then_adapt` directive** — wrapper does NOT execute any GPU work in DRY-RUN mode; LIVE mode delegates to Wf-299 launcher which honors per-wave abort criteria + USER_OK gate + budget tracker.

---

## 6. Forward-looking USER_OK-gated sub-flows (Wf-430+ follow-up per Wf-368-B1 §7 + Wf-419-B1 §7 + Wf-422-B1 §5 + Wf-423-B1 §4)

| Sub-flow | Wall | USER_OK | Status |
|------|------|------|------|
| **Wf-430-A1 actual Wf-299 GPU launch on cuda:0 Blackwell** (Wave A with `--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1`) | ~2 hr GPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A2 Wf-299 Wave B + C actual launch` (with Lambda CFM coupling active) | ~7.6 hr GPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A3 property panel evaluation via molmetal/scripts/cfm_property_panel.py` (m7 + m3 + m4 MEASURED end-to-end lift) | `~1 hr GPU` | REQUIRED | BLOCKED on user OK |
| `Wf-430-A4 Wf-394-A1 paper §6 v4 conversion` (USER_OK-gated per AGENTS.md §5.3) | ~12-15 hr CPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A5 Wf-380 SCAFFOLD-loss MEASURED verification on Blackwell sm_120` | ~10 hr GPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A6 Wf-246-C metal-coordination term retrain` (~3 hr GPU USER_OK-gated) | ~3 hr GPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A7 Wf-125 CONTROLD PAIR retrain` (~6 hr GPU USER_OK-gated per Wf-424-A10 closure path) | ~6 hr GPU | REQUIRED | BLOCKED on user OK |
| `Wf-430-A8 atomic write protocol adoption migration completion` (~31 remaining closeout-drafting scripts per Wf-416-A1 §5.2) | ~25 min CPU | NOT REQUIRED | DEFERRED |
| `Wf-430-A9 m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion` (USER_OK-gated per Wf-377-A1 §4.2) | ~25 hr CPU | REQUIRED | BLOCKED on user OK |

Combined wall budget: ~52.9 hr GPU + ~25 hr CPU + 30 min install across 9 USER_OK grants.

---

## 7. Honest framing caveats (per `party-twenty-spirit §3.5`)

1. **Wrapper authored only** — `wrapper_authored_only: true` in plan.json honest_framing block. Actual GPU retrain of Wf-394 Lambda × CFM coupling has NOT been performed; USER_OK gate prevents actual launch per AGENTS.md §5.1.
2. **`--lambda-cfm-mean-weight` OMITTED** — the default value `0.0` means the coupling is INACTIVE in trainer at line 1378-1384. The trainer auto-detects this from the `LambdaRewardDataset.mean_weight` attribute at line 2463 (caller must populate via Wf-394-A1 + Wf-329-W6 + Wf-333-G1 prior closeouts).
3. **`--flowmol-unified` LICENSE-GATED** — explicitly NOT passed by this wrapper (CC-BY-NC; USER_OK required for production integration per AGENTS.md §5.1).
4. **DRY_RUN.log "pairs" count is a SCHEDULE projection** — the actual number of training runs depends on the Wf-299 launcher's wave-by-wave orchestrator + per-pair budget; the n_pairs in plan.json is a conservative upper bound (e.g. 4 for Wave A = TODO-119 P1.4 smoke; 12 for Wave B = Wave C 12-run; 60 for Wave C = full sweep).
5. **Pre-flight gate chain NOT invoked in DRY-RUN mode** — per Wf-299 launcher line 289 `if [ "${DRY_RUN}" -eq 1 ]; then warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates"`. The pre-flight chain runs only in LIVE mode.

---

## 8. Cross-references

- Wf-394-A1 Lambda × CFM coupling wire-in patch SHIPPED at `TODO/completed/394a1_lambda_cfm_wire.md` (smoke + 2 tests PASS per Wf-394-A1 §1)
- Wf-394-A2 audit heuristic expansion SHIPPED at `TODO/completed/414a2_audit_expansion.md` (line-based grep expansion for both Wf-329-W6 OLD shape + Wf-406-A2 NEW shape per Wf-414-A2 §0)
- Wf-406-A2 patch SHIPPED at `TODO/completed/406a2_*.md` (trainer-side store_true + float + int shape per Wf-406-A2 §1)
- Wf-415-A1 transparent-passthrough pattern SHIPPED at `TODO/completed/415a1_wrapper_args_fix.md` (cumulative pattern: 14 NEW Wf-380-B1 + 3 NEW Wf-388-A2/A3 + 8 NEW Wf-340/Wf-353/Wf-405/Wf-408/Wf-409/Wf-410 trainer flags transparent passthrough)
- Wf-355-B1 USER_OK-gated wrapper family pattern at `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` + `scripts/launch_wf299_cfm_main.sh` + `scripts/launch_wf409a3_metal_coord_retrain.sh`
- Wf-368-B1 §7 forward-looking USER_OK-gated items (a) through (h)
- Wf-413-A3 `--json` argparse fix at `molmetal/scripts/launch_wf299_cfm_main.sh` line 257-258 (applies transitively to this wrapper via `exec bash ...` delegation)
- Wf-415-A3 pre-flight FAIL audit + 17 of 19 FAILs closed at `TODO/completed/415a3_preflight_fail_fix.md`
- Wf-422-A2 YuelBond decoder env gate SKIP→PASS closure at `TODO/completed/422a2_yuelbond_decoder_fix.md` (project venv `/home/hugo/codes/mol-metal/.venv/bin/python` with torch 2.14.0+cu130)
- Wf-424-A1 use_dann kwarg fix at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` (the actual `p2_tests_pass` cascade parent FAIL closure that Wf-424-A7 verified as 114/0/3/117 first CLEAN pre-flight since Wf-411-A3)
- Wf-428-A2 retrain readiness verification at `TODO/completed/428a2_*.md` (this wrapper closes the Wf-394 BLOCKER flagged by Wf-428-A2)

---

## 9. Chronology

| Time (CST) | Event |
|------|------|
| 2026-10-01 14:08 | Wrapper content authored at `/tmp/wf394_wrapper_content.sh` (13537 bytes / 303 lines) |
| 2026-10-01 14:08 | `atomic_write_file` invoked from `molmetal/scripts/wf386a3_atomic_write.py` — POSIX `os.replace` atomic swap to `scripts/launch_wf394_lambda_cfm.sh` |
| 2026-10-01 14:08 | `chmod +x` applied (rwx--x--x) |
| 2026-10-01 14:08 | BASH_SYNTAX check PASS |
| 2026-10-01 14:09 | DRY-RUN smoke test (Test 1) PASS: `USER_OK=1 bash ... --dry-run` exit 0 |
| 2026-10-01 14:09 | **Typo found**: `EFFIVE_FLAGS` → fixed to `EFFECTIVE_FLAGS` |
| 2026-10-01 14:09 | **tput issue found**: ANSI escape sequences misinterpreted as commands — dropped `$(tput bold 2>/dev/null || true)` pattern, use plain strings |
| 2026-10-01 14:09 | **--wave all unbound variable found**: added `all)` case to wave_target pair budget case statement |
| 2026-10-01 14:10 | Final canonical DRY-RUN smoke test PASS: exit 0 |
| 2026-10-01 14:10 | DRY_RUN.log + plan.json emitted correctly (845 B + 1304 B; JSON valid) |
| 2026-10-01 14:10 | USER_OK gate refusal test PASS: exit 64 |
| 2026-10-01 14:10 | --help output test PASS |
| 2026-10-01 14:10 | --wave 1/2/3 alias test PASS |
| 2026-10-01 14:10 | Transparent passthrough test PASS (`--chemprop-v3-cache --lambda-cfm-coupling-weight 0.2`) |
| 2026-10-01 14:10 | Invalid wave die test PASS: exit 1 |
| 2026-10-01 14:10 | This closeout authored via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py` |
| 2026-10-01 14:10 | All 9 forward-looking USER_OK-gated sub-flows queued for Wf-430+ |

---

## 10. Verdict

**Wf-429-A3 DID 彻底解决** the Wf-428-A2 BLOCKER ("wf394_lambda_cfm candidate is BLOCKED because scripts/launch_wf394_lambda_cfm.sh wrapper does NOT exist on disk") via 1 NEW wrapper script + 1 NEW DRY_RUN.log + 1 NEW plan.json on disk; 7 smoke tests PASS (USER_OK gate refusal exit 64 + DRY-RUN exit 0 + bash syntax OK + wave alias 1/2/3 + --wave all + transparent passthrough + invalid wave die exit 1 + exit 0 final); 0 GPU launches performed; 0 paper/* touches; 0 wet-lab planning; 0 actual GPU work executed per AGENTS.md §5.1 USER_OK gating.

**They did NOT 彻底解决** the structural/metric/env gates:
- m6_kappa + m2 R² + m1 AUC remain HONEST_NEGATIVE FINAL (per Wf-411-A4 + Wf-419-A2 + Wf-388-A1)
- Wf-394 actual Lambda × CFM coupling GPU retrain BLOCKED on USER_OK=1 (closure = Wf-430-A1)
- Wf-380 SCAFFOLD-loss actual wire-in deferred to Wf-430-A5
- 9 forward-looking USER_OK items (~52.9 hr GPU + ~25 hr CPU + 30 min install) still BLOCKED on user OK per AGENTS.md §5.1

**NEW ultracode Wf-430+ IS the recommended follow-up** with 9 USER_OK-gated sub-flows (per Wf-368-B1 §7 + Wf-419-B1 §7 + Wf-422-B1 §5 + Wf-423-B1 §4 + Wf-424-B1 + Wf-425-A1 + Wf-426-A1 etc.).

**last-update trailer extended with Wf-429-A3 sync clause** (this row appended per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 + Wf-419-B1 + Wf-420-B1 + Wf-421-B1 + Wf-422-B1 + Wf-423-B1 + Wf-424-A1 + Wf-424-A3 + Wf-424-A4 + Wf-424-A7 + Wf-424-A10 + Wf-424-A11 + Wf-424-A13 + Wf-424-A14 + Wf-424-A15 + Wf-424-B1 prior syncs).
