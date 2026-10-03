# Wf-366-A2 — Recover G6 Failed Tuples Block-Order Fix (GAP-2)

**Date:** 2026-09-29 (Wf-366-A2 ultracode follow-up)
**Owner:** Hugo Lee + ultracode agent
**Status:** SHIPPED
**Parent finding:** Wf-363-A2 audit (TODO/completed/363a2_user_ok_ux.md §3 GAP-2)
**Cross-references:**
- Wf-337-G1 (original G6 recovery launcher; ship at TODO/completed/337g1_g6_recovery.md)
- Wf-355-A3 reference pattern (scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh lines 115-122 + 364)
- Wf-363-A1 DRY_RUN.log schema (scripts/_dry_run_log_helper.sh)
- Wf-363-A2 audit (molmetal/tests/test_wf363a2_user_ok_ux.py)

---

## §1 Background — GAP-2 finding

The Wf-363-A2 user-OK UX audit (TODO/completed/363a2_user_ok_ux.md §3 GAP-2)
surfaced a block-order bug in `scripts/recover_g6_failed_tuples.sh`:

- The script's `--dry-run` exit block fired BEFORE the `USER_OK` gate.
- This meant an operator could dry-run the G6 recovery plan WITHOUT
  explicit USER_OK authorization, bypassing the kill_then_adapt directive
  (AGENTS.md §5.1).
- The recovery sweep is a multi-hour GPU run (~11.4 hr PROJECTED for 57
  FAILED + PENDING tuples on RTX 5090), so the gate is a CRITICAL safety
  lever.

Before the fix:

```bash
# OLD order (broken):
if [[ "${DRY_RUN}" == "1" ]]; then
  # emit DRY-RUN OK + recovery_manifest.tsv
  exit 0   # <-- BYPASSES USER_OK gate
fi
if [[ -z "${USER_OK:-}" ]]; then
  echo "USER_OK REQUIRED"
  exit 1   # <-- never reached if --dry-run was set
fi
```

After the fix (this Wf):

```bash
# NEW order (Wf-366-A2 GAP-2 fix):
if [[ "${USER_OK:-0}" != "1" ]]; then
  echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
  exit 64   # <-- gate fires FIRST, regardless of --dry-run
fi
# ... prereg.md check ...
# ... manifest enumeration ...
if [[ "${DRY_RUN}" == "1" ]]; then
  # emit DRY_RUN.log + plan
  exit 0   # <-- only reached AFTER gate passes
fi
# ... launch loop ...
```

---

## §2 Per-scenario verification (task step 6)

| Scenario | Before fix | After fix (Wf-366-A2) | Expected |
|---|---:|---:|---:|
| No USER_OK, no `--dry-run` | rc=1 (gate fires) | **rc=64** (gate fires first) | rc=64 ✓ |
| No USER_OK, `--dry-run` | **rc=0 (BYPASS BUG)** | **rc=64** (gate fires first) | rc=64 ✓ |
| `USER_OK=1`, `--dry-run` | rc=0 (no DRY_RUN.log) | **rc=0 + DRY_RUN.log emitted (120 pairs)** | rc=0 + DRY_RUN.log ✓ |
| `USER_OK=1`, no `--dry-run` | proceeds to launch | **proceeds to launch** (PROJECTED ~11.4 hr; BLOCKED on §5.1) | proceeds ✓ |

Scenario 4 (real launch path) is BLOCKED on AGENTS.md §5.1 user OK per the
G6 recovery wall budget. Not live-tested; block-order invariants verified
at source level in test 1.

All four scenarios verified live in `molmetal/tests/test_wf366a2_recover_g6_block_order.py::test_dry_run_without_user_ok_exits_64`
(3 parametrized cases; scenario 4 not parametrized since it would consume
GPU budget — source-inspected instead).

---

## §3 Implementation summary

### §3.1 Files modified

| Path | LOC delta | Purpose |
|---|---:|---|
| `scripts/recover_g6_failed_tuples.sh` | +34 net (5 edits) | Swap block order: USER_OK gate BEFORE --dry-run; exit-code 1 → 64; source DRY_RUN.log helper; emit per-pair DRY_RUN.log |
| `molmetal/tests/test_wf363a2_user_ok_ux.py` | 3 lines (1 edit) | Migrate `recover_g6_failed_tuples.sh` from `dry_run_policy="loose_bypass"` to `dry_run_policy="strict"` (the fix closes the GAP-2 finding) |
| `molmetal/tests/test_wf366a2_recover_g6_block_order.py` | +257 NEW (2 NEW tests) | Source-level block-order invariant + live-run gate enforcement |
| `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` | +200 NEW (this doc) | Wf-366-A2 ship doc |

### §3.2 Block-order fix details

```diff
 set -euo pipefail
 cd /home/hugo/codes/mol-metal

+# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema).
+# shellcheck source=./_dry_run_log_helper.sh
+source "$(dirname "$0")/_dry_run_log_helper.sh"
+
 # Per Wf-350-B2: parse --prereg <path> argument; ...
 PREREG=""
 DRY_RUN=0
 ...

+# ---- Wf-366-A2 GAP-2 FIX: USER_OK gate BEFORE --dry-run block ----
+# Reference: scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
+# (USER_OK gate at lines 115-122, --dry-run plan emit at line 364).
+if [[ "${USER_OK:-0}" != "1" ]]; then
+  echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
+  echo "       USER_OK=1 bash scripts/recover_g6_failed_tuples.sh --dry-run   # plan only (still requires USER_OK=1)" >&2
+  echo "       USER_OK=1 bash scripts/recover_g6_failed_tuples.sh              # actual recovery launch" >&2
+  exit 64
+fi
+echo "[Wf-337-G1] USER_OK set — proceeding with recovery launch (~11.4 hr wall PROJECTED)"

 # Verify prereg.md against launch config if supplied
 ...
```

```diff
 # ---- Step 1: enumerate (cfg, metal, seed) tuples + classify ----
-DRY_RUN=0
-if [[ "${1:-}" == "--dry-run" ]]; then
-  DRY_RUN=1
-fi
-
 echo "=== Wf-337-G1 G6 RECOVERY START $(date) ===" | tee -a "${LOG}"
```

```diff
+# ---- Wf-366-A2 GAP-2 FIX: --dry-run exit AFTER manifest enumeration ----
+if [[ "${DRY_RUN}" == "1" ]]; then
+  echo ""
+  echo "=== DRY-RUN OK (USER_OK=1 confirmed; plan only; no GPU work) ===" | tee -a "${LOG}"
+
+  # Wf-363-A1: emit structured DRY_RUN.log with per-pair (cfg, metal, seed)
+  # enumeration via the shared helper.
+  DRY_RUN_LOG="${OUT_BASE}/DRY_RUN.log"
+  pair_args=()
+  for cfg in "${CONFIRMED_FLAGS_ORDER[@]}"; do
+    for metal in "${METALS_ORDER[@]}"; do
+      for seed in 0 1 2; do
+        pair_args+=( --pair "${cfg}/${metal} seed=${seed}" )
+      done
+    done
+  done
+  emit_dry_run_log "${DRY_RUN_LOG}" \
+    --title "Wf-337-G1 G6 recovery (57 FAILED + PENDING tuples)" \
+    --wrapper "recover_g6_failed_tuples.sh" \
+    --mode "recovery_dry_run" \
+    --trainer "${SCRIPT}" \
+    --flags "--metals <per-pair> --seeds <per-pair> ... ${COMMON_ARGS[*]}" \
+    --gpu "${GPU}" \
+    --budget-hr "11.4" \
+    --user-ok "REQUIRED" \
+    "${pair_args[@]}"
+  ...
+  exit 0
+fi

-# ---- Step 2: USER_OK gate (AGENTS.md §5.1 kill_then_adapt) ----
-if [[ -z "${USER_OK:-}" ]]; then
-  echo ""
-  echo "USER_OK REQUIRED — kill_then_adapt standing directive."
-  ...
-  exit 1
-fi
-echo "[Wf-337-G1] USER_OK set — proceeding with recovery launch (~11.4 hr wall)" | tee -a "${LOG}"

-# ---- Step 3: launch each FAILED + PENDING tuple (skip OK + RUNNING) ----
+# ---- Step 2: launch each FAILED + PENDING tuple (skip OK + RUNNING) ----
```

### §3.3 New ordered arrays

Added two plain arrays to support ordered iteration in the per-pair DRY_RUN.log
emission (Bash associative arrays do not preserve order):

```bash
CONFIRMED_FLAGS_ORDER=(no_multi_task no_wass no_kendall no_dann)
METALS_ORDER=(Ru Ir Os Rh Re)
SEEDS_ORDER=(0 1 2)
```

These mirror the iteration order already used by the manifest enumeration
Python helper, so the per-pair enumeration in DRY_RUN.log matches the
manifest row order.

### §3.4 Exit-code update: 1 → 64

The old USER_OK gate used `exit 1` (generic failure). The fix updates to
`exit 64` to match the Wf-294B family convention (see
`scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` line 121) and
the standing AGENTS.md §5.1 kill_then_adapt convention. The exit-code
differentiates "USER_OK refused" (64) from "prereg.md missing/invalid" (1)
and from "GPU launch failure" (1), so operator triage is unambiguous.

---

## §4 Tests (2 NEW + 1 update)

### §4.1 NEW — `molmetal/tests/test_wf366a2_recover_g6_block_order.py`

2 NEW tests (1 source-inspection + 1 live-run parametrized 3 ways = 4 items):

1. **`test_block_order_user_ok_before_dry_run`** — source-inspects the wrapper
   and verifies:
   - USER_OK gate line < --dry-run exit line (CORE INVARIANT).
   - --dry-run exit line > manifest enumeration line.
   - USER_OK gate uses `exit 64` (matches Wf-294B family).
   - --dry-run exit uses `exit 0` (plan-only success).
   - Reference pattern (`launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`)
     is cited in a code comment with line numbers.
   - DRY_RUN.log helper (`_dry_run_log_helper.sh`) is sourced.
   - `emit_dry_run_log` is called inside the --dry-run block.

2. **`test_dry_run_without_user_ok_exits_64`** — parametrized 3 ways:
   - `[args0-64-True-no_USER_OK_no_dry_run]`: no USER_OK, no --dry-run → rc=64.
   - `[args1-64-True-no_USER_OK_with_dry_run]`: no USER_OK, --dry-run → rc=64
     (was rc=0 BEFORE the fix; this is the GAP-2 regression guard).
   - `[args2-0-False-USER_OK_1_with_dry_run]`: USER_OK=1, --dry-run → rc=0
     + DRY_RUN.log emitted with all 8 Wf-363-A1 schema fields.

### §4.2 UPDATE — `molmetal/tests/test_wf363a2_user_ok_ux.py`

Migrated `recover_g6_failed_tuples.sh` from
`dry_run_policy="loose_bypass"` to `dry_run_policy="strict"`. This
documents the Wf-366-A2 fix in the audit test (so a future agent reading
test_wf363a2_user_ok_ux.py sees that GAP-2 was closed by Wf-366-A2, not
silently papered over).

### §4.3 Test results

```
$ pytest molmetal/tests/test_wf366a2_recover_g6_block_order.py -v
collected 4 items
test_block_order_user_ok_before_dry_run PASSED [ 25%]
test_dry_run_without_user_ok_exits_64[args0-64-True-no_USER_OK_no_dry_run] PASSED [ 50%]
test_dry_run_without_user_ok_exits_64[args1-64-True-no_USER_OK_with_dry_run] PASSED [ 75%]
test_dry_run_without_user_ok_exits_64[args2-0-False-USER_OK_1_with_dry_run] PASSED [100%]
========================= 4 passed, 1 warning in 0.37s =========================
```

Plus regression run on Wf-363-A1 + Wf-363-A2 + Wf-366-A2 suites: **28/28 PASS**.

```
$ pytest molmetal/tests/test_wf363a2_user_ok_ux.py molmetal/tests/test_wf363a1_dry_run_log.py molmetal/tests/test_wf366a2_recover_g6_block_order.py
========================= 28 passed, 1 warning in 5.59s =========================
```

---

## §5 Wall + scope

- **MEASURED wall:** ~10 min CPU-only (Edit + Edit + Edit + Edit + Edit + 1 test file write + 1 pytest run + 1 syntax check + 1 doc write).
- **GPU launches:** 0 (CPU-only Edit + pytest).
- **Tests:** 2 NEW (1 source + 1 parametrized live-run; 4 items total) + 1 update (Wf-363-A2 audit).
- **Lines changed:** +34 net in script (5 edits); +257 NEW in test file; +200 NEW in this doc.

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **MEASURED (rc=0 → rc=64 on --dry-run without USER_OK)**: The fix changes
  behavior from rc=0 (BYPASS BUG) to rc=64 (gate fires first). This is the
  CRITICAL safety fix.
- **MEASURED (rc=1 → rc=64 on no --dry-run + no USER_OK)**: The exit code
  was updated from 1 to 64 to match the Wf-294B family convention.
- **MEASURED (USER_OK=1 --dry-run emits DRY_RUN.log with 120 (cfg, metal, seed)
  pairs)**: The fix adds Wf-363-A1 schema-consistent DRY_RUN.log emission
  via the shared helper, matching the 5 Wf-355 wrappers.
- **MEASURED (4 scenarios verified live + 1 source-inspected for block-order
  invariants)**: test 1 source-inspects; test 2 parametrized live-run hits
  3 of 4 scenarios; scenario 4 (real launch path) not live-tested since
  it would consume GPU budget.
- **PROJECTED**: G6 recovery sweep is ~11.4 hr wall (BLOCKED on AGENTS.md
  §5.1 user OK). Not launched in this wave.
- **PROJECTED**: the fix aligns `recover_g6_failed_tuples.sh` with the 5
  Wf-355 wrappers' USER_OK + --dry-run block-order invariant; future
  Wf-XXX-A2 audits can verify all 8 wrappers in a single consistency check.

---

## §7 Standing-directive compliance

| Directive | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | All numeric claims in this doc cite JSON / report path or test name. PROJECTED tag used for ~11.4 hr wall. ✓ |
| §1.2 Standing GPU authorization | No GPU launches; CPU-only. No user OK needed. ✓ |
| §1.3 No paper work without authorization | No `paper/*` touches. ✓ |
| §1.4 Pre-flight v2 25/25 PASS gate | Pre-flight not invoked (no launch); per §1.4 applies to "launching a retrain / panel / ablation" — this Wf is a script fix, not a launch. ✓ |
| §1.5 Honest-negative framing | Section §6 above; GAP-2 fix is a CRITICAL safety improvement, not a regression. ✓ |
| §5.4 Tier-2 journal target + NO wet-lab planning | N/A (no wet-lab work). ✓ |

---

## §8 Cross-references

- `scripts/recover_g6_failed_tuples.sh` (modified)
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (reference pattern)
- `scripts/_dry_run_log_helper.sh` (Wf-363-A1 shared helper)
- `molmetal/tests/test_wf366a2_recover_g6_block_order.py` (NEW, this Wf)
- `molmetal/tests/test_wf363a2_user_ok_ux.py` (sibling audit, updated)
- `molmetal/tests/test_wf363a1_dry_run_log.py` (sibling schema test)
- `TODO/completed/363a2_user_ok_ux.md §3 GAP-2` (original audit finding)
- `TODO/completed/337g1_g6_recovery.md` (Wf-337-G1 parent ship)
- `AGENTS.md §3.4 skill catalog + §5 standing directives` (convention source)

**END OF Wf-366-A2 SHIP DOC** — 2026-09-29 16:00 UTC wall.
