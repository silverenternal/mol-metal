# Wf-366-A1 — launch_wf124_p3_g5.sh USER_OK gate gap closure (GAP-1)

**Date**: 2026-09-29 (Wf-366-A1 ship)
**Owner**: Hugo Lee + ultracode agent (Wf-366 ultracode residual wave)
**Status**: ✅ SHIP (CPU-only verification + 3 NEW tests + close-out)
**Parent**: Wf-363-A2 audit `TODO/completed/363a2_user_ok_ux.md` §3 GAP-1
**Sub-sections**: §A ship status / §B fix diff / §C integration verdict

---

## §0 TL;DR — Wf-366-A1 USER_OK gate fix

Wf-366-A1 ships the **USER_OK gate enforcement** for `scripts/launch_wf124_p3_g5.sh`
(TODO-181 G5 wrapper, 5 metals × 3 seeds × 50 epochs = 15 (metal, seed) pairs,
~5 hr GPU on RTX 5090 GPU-1 per Wf-260 G6 timing).

Per Wf-363-A2 audit finding GAP-1:
> "launch_wf124_p3_g5.sh has NO USER_OK gate entirely. RC=124 timeout means
>  wrapper proceeds straight into pre-flight without user OK. ~5-hr GPU
>  launch risk."

The fix closes the gap by adding the **canonical Wf-355-B1 USER_OK gate**
pattern (consistent with `scripts/launch_wf131_p3_deferred_runs.sh` lines
118-125 + `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` Wf-355-A3).

| Sub-task | Headline | Verdict |
|---|---|---|
| **A. Gate pattern wired** | Strict USER_OK=1 gate at lines 51-67; exit 64 + USER_OK gate refused FATAL | ✅ SHIP |
| **B. DRY_RUN.log + plan.json** | DRY_RUN.log emitted via Wf-363-A1 helper + plan.json with `user_ok: true` | ✅ SHIP |
| **C. Test migration** | Wf-363-A2 gap test REMOVED; G5 migrated into main WRAPPERS list | ✅ SHIP |

**3 NEW tests shipped**: `molmetal/tests/test_wf366a1_g5_user_ok_gate.py`
- `test_no_user_ok_exits_64` — no USER_OK → exit 64 + FATAL message
- `test_user_ok_proceeds_to_preflight` — USER_OK=1 → gate bypass + dry-run exit 0
- `test_dry_run_with_user_ok_emits_log_and_plan` — DRY_RUN.log + plan.json emitted

All 3 PASS in **0.36 s** (CPU-only, deterministic). Plus 16 Wf-363-A2 tests
re-verified (G5 now in strict policy list).

---

## §A — Ship status verification

**File changes**:

| File | Change | MEASURED |
|---|---|---|
| `scripts/launch_wf124_p3_g5.sh` | USER_OK gate wired at lines 51-67 (+17 LOC); DRY_RUN.log + plan.json emission in --dry-run block (+33 LOC) | bash -n SYNTAX_OK |
| `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` | NEW 3 tests (gate enforces, gate bypass, dry-run emission) | 3/3 PASS in 0.36 s |
| `molmetal/tests/test_wf363a2_user_ok_ux.py` | G5 migrated from GATE_GAP_WRAPPERS into main WRAPPERS list; gap test REMOVED | 9/9 wrappers PASS (G5 strict + 8 others) |

**Honest framing (party-twenty-spirit §3.5)**:
- 5-line gate patch CRITICAL SAFETY (prevents silent auto-launch of ~5 hr GPU job)
- 3 NEW tests PASS (CPU-only)
- Wall: ~10 min CPU-only (actual: 0.36 s test execution + 1 source patch + 1 test doc)
- 0 GPU launches performed
- Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (NOT re-run by Wf-366-A1 — CPU-only patch)

---

## §B — Fix diff (USER_OK gate at scripts/launch_wf124_p3_g5.sh lines 51-67)

```diff
 set -e
 cd /home/hugo/codes/mol-metal

+# Wf-366-A1 — USER_OK gate (kill_then_adapt standing directive; mirrors
+# scripts/launch_wf131_p3_deferred_runs.sh Wf-355-B1 pattern at lines
+# 118-125 + scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
+# Wf-355-A3 pattern). Per AGENTS.md §5.1 kill_then_adapt: any > 1 hr GPU
+# launch MUST be gated by USER_OK=1 in env, NOT silently auto-triggered.
+# G5 wall ~5 hr on RTX 5090 GPU-1 (per Wf-260 G6 timing) — gate is
+# CRITICAL SAFETY. Closes Wf-363-A2 HONEST-NEGATIVE gap (G5 was the
+# only launch wrapper missing a USER_OK gate; live-run audit showed
+# RC=124 timeout when invoked without --dry-run + without USER_OK).
+if [ "${USER_OK:-0}" != "1" ]; then
+    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
+        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
+    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
+    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run (15 pairs)\n' "$0" >&2
+    printf '       USER_OK=1 bash %s --prereg docs/prereg/<file>.md   # paper-grade + pre-reg\n' "$0" >&2
+    exit 64
+fi
+
+# Wf-366-A1 — source the shared DRY_RUN.log helper (consistent per-pair
+# schema per Wf-363-A1; used in --dry-run mode below)
+REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
+DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
+[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
+# shellcheck source=../_dry_run_log_helper.sh
+source "${DRY_RUN_LOG_HELPER}"
+
 # Per Wf-350-B2: parse --prereg <path> argument
```

**DRY-RUN block update (lines 144-220, +33 LOC for emit_dry_run_log + plan.json)**:
- `emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" --title "..." --wrapper ... --mode G5_50epoch_full_sweep --trainer ... --flags "..." --gpu ... --budget-hr 5 --user-ok REQUIRED --pair "Ru seed=0" ... --pair "Re seed=2"`
- `PLAN_JSON="${OUT_BASE}/plan.json"` with `user_ok: true`, `dry_run: true`, `total_runs: 15`, `metals: ["Ru", "Ir", "Os", "Rh", "Re"]`, `seeds: [0, 1, 2]`, `status: PLAN_ONLY`, plus `honest_framing` block (MEASURED_today + PROJECTED + kill_then_adapt_directive + wf337_g2_status)

---

## §C — Integration verdict

**Wf-363-A2 GAP-1 HONEST-NEGATIVE → CLOSED**.

Live-run verification (post-fix):
- `bash scripts/launch_wf124_p3_g5.sh` (no USER_OK) → exit 64, stderr `[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.` ✅
- `USER_OK=1 bash scripts/launch_wf124_p3_g5.sh --dry-run` → exit 0, DRY_RUN.log + plan.json emitted to `results/wf124_p3_50epoch_full_sweep_g5/` ✅
- `USER_OK=1 bash scripts/launch_wf124_p3_g5.sh` (real launch) → proceeds past gate to pre-flight (no auto-trigger without user OK) ✅

**Wf-363-A2 audit table update** (per Wf-366-A1):
- G5 added to WRAPPERS list at `molmetal/tests/test_wf363a2_user_ok_ux.py` with `gate_value="1"`, `dry_run_policy="strict"`, `family="Wf-294B"`, `wf_origin="Wf-366-A1 (gap closure post-Wf-363-A2 audit)"`
- G5 removed from `GATE_GAP_WRAPPERS` (list now empty)
- `test_g5_user_ok_gate_gap_recorded` REMOVED (replaced with cross-ref comment)

**Regression check**:
- 16/16 Wf-363-A2 tests PASS (9 wrappers × strict + 7 dry-run policies)
- 3/3 NEW Wf-366-A1 tests PASS
- 0 new pre-flight gates added (EXPECTED_GATE_COUNTS["all"] 62 PRESERVED)
- 0 GPU launches performed (CPU-only patch)

**Cross-references**:
- Wf-363-A2 audit: `TODO/completed/363a2_user_ok_ux.md` §3 GAP-1
- Reference pattern: `scripts/launch_wf131_p3_deferred_runs.sh` lines 118-125 (Wf-355-B1)
- Reference pattern: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` Wf-355-A3
- DRY_RUN helper: `scripts/_dry_run_log_helper.sh` (Wf-363-A1)
- AGENTS.md §5.1 standing GPU authorization + kill_then_adapt directive

**Wider Wf-366 ultracode residual wave status** (2026-09-29):
- Wf-366-A1: G5 USER_OK gate ✅ CLOSED
- Remaining Wf-366 closeouts: TBD per Wf-366 ultracode dispatch
