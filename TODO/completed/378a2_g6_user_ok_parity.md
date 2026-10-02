# Wf-378-A2 — launch_wf124_p3_g6.sh USER_OK Gate Parity Closure

**Date**: 2026-09-30 (Wf-378 ultracode residual wave)
**Owner**: Hugo Lee + Wf-378-A2 ultracode agent
**Status**: ✅ SHIPPED (CPU-only verification + 5 NEW tests + close-out)
**Parent findings**:
- Wf-365-B2 final integration test (TODO/completed/365b2_7_wrappers_final.md)
- Wf-368-A4 section 3 (cross-poll finding)
- Wf-368-B1 section 7.1 (forward-looking gap)
**Sub-sections**: §1 Background / §2 Fix / §3 Verification / §4 Tests / §5 Gate parity / §6 Standing directives / §7 Cross-references

---

## §1 Background — Wf-365-B2 + Wf-368-B1 §7.1 finding

The Wf-365-B2 final integration test surfaced that `scripts/launch_wf124_p3_g6.sh`
(TODO-182 G6 wrapper, 60 (config × metal × seed) tuples per-innovation
ablation, ~7.5 hr GPU on RTX 5090 GPU-1 per Wf-260 G6 timing) was
flagged as **the only legacy launch wrapper lacking USER_OK gate parity**.

Per Wf-368-B1 §7.1:
> "Wf-365-B2 G6 USER_OK parity OPEN — closure path = ~10 min CPU + 1 test;
> deferred to Wf-369+ docs-only sub-flow."

The root cause was a **block-order bug**:

| Wrapper | Gate exists? | Gate position | dry-run policy |
|---|---|---|---|
| `launch_wf124_p3_g5.sh` (Wf-366-A1 fix) | YES | BEFORE --dry-run block | `strict` ✅ |
| `recover_g6_failed_tuples.sh` (Wf-366-A2 fix) | YES | BEFORE --dry-run block | `strict` ✅ |
| `launch_wf124_p3_g6.sh` (Wf-378-A2 fix, this Wf) | YES (originally) | **AFTER --dry-run block** | `loose` (BYPASS BUG) ❌ |

The original gate at lines 208-218 was wired AFTER the --dry-run exit
block at lines 170-206, so `--dry-run` silently bypassed the gate.
Wf-363-A2 audit (`TODO/completed/363a2_user_ok_ux.md`) classified G6
with `dry_run_policy="loose"` and noted that this should be tightened
to `strict` per AGENTS.md §5.1.

---

## §2 Fix — Move USER_OK gate BEFORE --dry-run block (Wf-366-A2 lesson)

### §2.1 Canonical Wf-366-A1 pattern wired

The fix mirrors the canonical Wf-366-A1 pattern from
`scripts/launch_wf124_p3_g5.sh` lines 51-67 + Wf-366-A2 lesson from
`scripts/recover_g6_failed_tuples.sh` lines 95-109:

```bash
set -e
cd /home/hugo/codes/mol-metal

# Wf-378-A2 — USER_OK gate (kill_then_adapt standing directive; canonical
# Wf-366-A1 pattern ...). Closes Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1
# HONEST-NEGATIVE finding: the G6 wrapper's USER_OK gate was wired AFTER
# the --dry-run exit block at lines 208-218, so `--dry-run` silently
# bypassed the gate.
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~7.5 hr GPU run (60 tuples)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --prereg docs/prereg/<file>.md   # paper-grade + pre-reg\n' "$0" >&2
    exit 64
fi

# Wf-378-A2 — source the shared DRY_RUN.log helper (consistent per-pair
# schema per Wf-363-A1; used in --dry-run mode below)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"
```

### §2.2 Old late-wired gate removed

The original gate at lines 208-218 (after the --dry-run block) was
**removed** and replaced with a one-line echo to confirm the gate passed:

```diff
-# USER_OK gate (Wf-306D ...).  Set USER_OK=1 to authorize the real
-# launch; --dry-run path already exited at line 117 so it is unaffected.
-if [[ -z "${USER_OK:-}" ]]; then
-  echo "USER OK REQUIRED — kill_then_adapt standing directive."
-  ...
-  exit 1
-fi
-echo "[Wf-306D] USER_OK set — proceeding with 60-tuple launch (G6 wall ~7.5 hr)" | tee -a "${LOG}"
+# Wf-378-A2: USER_OK gate moved BEFORE --dry-run block (see lines ~60-75)
+# per Wf-366-A2 lesson. Old gate block at original lines 208-218 was
+# reached only on real-launch path, so `--dry-run` silently bypassed it.
+echo "[Wf-378-A2] USER_OK set — proceeding with 60-tuple launch (G6 wall ~7.5 hr)" | tee -a "${LOG}"
```

The new code path now uses `exit 64` (Wf-294B family convention) instead
of `exit 1`, matching the canonical pattern.

### §2.3 DRY_RUN.log + plan.json emission in --dry-run block

The --dry-run block was updated to emit structured `DRY_RUN.log` (via
`emit_dry_run_log` helper from `scripts/_dry_run_log_helper.sh`) and
`plan.json` consistent with Wf-355-A3 + Wf-366-A1 patterns:

```bash
emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
    --title "Wf-124 P3 G6 — 60 (config, metal, seed) tuples per-innovation ablation" \
    --wrapper "$(basename "$0")" \
    --mode "G6_per_innovation_ablation" \
    --trainer "${SCRIPT}" \
    --flags "..." \
    --gpu "${GPU}" \
    --budget-hr "7.5" \
    --user-ok "REQUIRED" \
    ${pair_args[@]+"${pair_args[@]}"}

PLAN_JSON="${OUT_BASE}/plan.json"
{
  printf '{\n'
  printf '  "wrapper": "launch_wf124_p3_g6.sh",\n'
  printf '  "wf_origin": "Wf-378-A2 (USER_OK gate parity closure per Wf-366-A2 lesson)",\n'
  printf '  "parent_plan": "TODO/completed/182_wf124_p3_per_innovation_ablation.md",\n'
  printf '  "user_ok": true,\n'
  printf '  "dry_run": true,\n'
  printf '  "configs": ["no_multi_task", "no_wass", "no_kendall", "no_dann"],\n'
  printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
  printf '  "seeds": [0, 1, 2],\n'
  printf '  "epochs": 50,\n'
  printf '  "total_tuples": 60,\n'
  ...
  printf '  "honest_framing": {\n'
  printf '    "MEASURED_today": "wrapper ready; USER_OK gate enforced BEFORE --dry-run per Wf-378-A2; 60 tuples enumerated; DRY_RUN.log emitted",\n'
  printf '    "PROJECTED": "G6 wall ~7.5 hr on RTX 5090 GPU-1; per-lever attribution %% NOT MEASURED until sweep completes",\n'
  printf '    "kill_then_adapt_directive": "7.5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
  printf '    "wf378_a2_status": "Wf-378-A2 closes Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1 HONEST-NEGATIVE finding..."\n'
  ...
} > "${PLAN_JSON}"
```

The plan.json carries `user_ok: true`, `dry_run: true`, `total_tuples: 60`,
`status: PLAN_ONLY`, plus the `honest_framing` block per the Wf-355-A3
convention.

### §2.4 Per-pair enumeration ordering

Added ordered arrays for the per-pair DRY_RUN.log enumeration (Bash
associative arrays do not preserve order):

```bash
CONFIRMED_FLAGS_ORDER=(no_multi_task no_wass no_kendall no_dann)
METALS_ORDER=(Ru Ir Os Rh Re)
SEEDS_ORDER=(0 1 2)
```

These mirror the iteration order already used in the `DRY-RUN` block
enumeration + the `for cfg in ...` launch loop at lines 286-292, so the
per-pair enumeration in DRY_RUN.log matches the actual launch order.

### §2.5 Files modified

| Path | LOC delta | Purpose |
|---|---:|---|
| `scripts/launch_wf124_p3_g6.sh` | +91 net (3 edits) | Add canonical Wf-366-A1 USER_OK gate BEFORE --dry-run; source DRY_RUN.log helper; emit DRY_RUN.log + plan.json in --dry-run block; remove old late-wired gate |
| `molmetal/tests/test_wf363a2_user_ok_ux.py` | +5 net (2 edits) | Migrate G6 from `dry_run_policy="loose"` to `dry_run_policy="strict"`; mark `emits_plan_json=True`; update `family="Wf-294B"` + `wf_origin="Wf-378-A2 (gap closure ...)"` |
| `molmetal/tests/test_wf378a2_g6_user_ok_gate.py` | +260 NEW | NEW test file with 5 tests (1 source-level + 3 parametrized + 1 dry-run emission) |
| `TODO/completed/378a2_g6_user_ok_parity.md` | +200 NEW | Wf-378-A2 ship doc (this file) |

---

## §3 Per-scenario verification (live-run, MEASURED)

| Scenario | Before fix | After fix (Wf-378-A2) | Expected |
|---|---:|---:|---:|
| No USER_OK, no `--dry-run` | rc=1 (gate fires) | **rc=64** (canonical Wf-366-A1) | rc=64 ✅ |
| No USER_OK, `--dry-run` | **rc=0 (BYPASS BUG)** | **rc=64** (gate fires first) | rc=64 ✅ |
| `USER_OK=1`, `--dry-run` | rc=0 (no DRY_RUN.log) | **rc=0 + DRY_RUN.log emitted (60 tuples) + plan.json emitted** | rc=0 + structured output ✅ |
| `USER_OK=1`, no `--dry-run` | proceeds to launch | **proceeds to launch** (PROJECTED ~7.5 hr; BLOCKED on §5.1) | proceeds ✅ |

Scenario 4 (real launch path) is BLOCKED on AGENTS.md §5.1 user OK per the
G6 wall budget. Not live-tested; block-order invariants verified at source
level in test 1.

All three executable scenarios verified live in
`molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_no_user_ok_exits_64`
(3 parametrized cases) + `test_dry_run_with_user_ok_emits_log_and_plan`
(emission verification with 60-pair enumeration + plan.json schema check).

---

## §4 Tests (5 NEW)

### §4.1 NEW — `molmetal/tests/test_wf378a2_g6_user_ok_gate.py`

5 NEW tests:

1. **`test_user_ok_gate_present_before_dry_run`** — source-inspects the
   wrapper and verifies:
   - Canonical Wf-366-A1 USER_OK gate pattern present.
   - `USER_OK:-0` env-var check + `exit 64` + `USER_OK gate refused`
     FATAL message + `kill_then_adapt` directive reference.
   - **CORE INVARIANT**: USER_OK gate block line < --dry-run block line
     (gate fires FIRST regardless of --dry-run, per Wf-366-A2 lesson).
   - Gate uses `exit 64` (Wf-294B family convention).
   - DRY_RUN.log helper is sourced.
   - DRY_RUN.log emission (`emit_dry_run_log`) present in --dry-run block.
   - plan.json emission present + `"user_ok": true` marker.
   - Wf-378-A2 / Wf-366-A2 reference in source comments.

2. **`test_no_user_ok_exits_64`** (parametrized 3 ways) — live-run gate
   enforcement:
   - `[no_USER_OK_no_dry_run]`: no USER_OK, no --dry-run → rc=64.
   - `[no_USER_OK_with_dry_run]`: no USER_OK, --dry-run → rc=64
     **(was rc=0 BEFORE Wf-378-A2 fix; the CORE gap closure)**.
   - `[USER_OK_1_with_dry_run]`: USER_OK=1, --dry-run → rc=0 + no
     gate refusal.

3. **`test_dry_run_with_user_ok_emits_log_and_plan`** — USER_OK=1 +
   --dry-run → exits 0 AND emits DRY_RUN.log AND plan.json in OUT_BASE:
   - DRY_RUN.log enumerates **60 (cfg, metal, seed) tuples** via --pair args.
   - DRY_RUN.log declares `user_ok: REQUIRED` + `budget: 7.5 hr`.
   - plan.json declares `user_ok: true`, `dry_run: true`,
     `total_tuples: 60`, `status: PLAN_ONLY`.
   - plan.json configs / metals / seeds sanity-checked.
   - plan.json honest_framing block contains `MEASURED_today`,
     `PROJECTED`, `kill_then_adapt_directive`, and
     `wf378_a2_status` referencing Wf-378-A2.

### §4.2 UPDATE — `molmetal/tests/test_wf363a2_user_ok_ux.py`

Migrated `launch_wf124_p3_g6.sh` from:
```python
"dry_run_policy": "loose",  # --dry-run path exits at line 117 BEFORE gate
"emits_plan_json": False,
"family": "Wf-306D",
"wf_origin": "Wf-306D",
```
to:
```python
"dry_run_policy": "strict",  # Wf-378-A2: gate moved BEFORE --dry-run; fires FIRST
"emits_plan_json": True,     # Wf-378-A2: plan.json now emitted in --dry-run block
"family": "Wf-294B",
"wf_origin": "Wf-378-A2 (gap closure post-Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1 audit)",
```

Also added explanatory comment in the GATE_GAP_WRAPPERS section noting
that Wf-378-A2 closed the G6 gap.

### §4.3 Test results

```
$ pytest molmetal/tests/test_wf378a2_g6_user_ok_gate.py -v
collected 5 items
molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_user_ok_gate_present_before_dry_run PASSED [ 20%]
molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_no_user_ok_exits_64[no_USER_OK_no_dry_run-None-args0-64-True] PASSED [ 40%]
molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_no_user_ok_exits_64[no_USER_OK_with_dry_run-None-args1-64-True] PASSED [ 60%]
molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_no_user_ok_exits_64[USER_OK_1_with_dry_run-env_extra2-args2-0-False] PASSED [ 80%]
molmetal/tests/test_wf378a2_g6_user_ok_gate.py::test_dry_run_with_user_ok_emits_log_and_plan PASSED [100%]
========================= 5 passed, 1 warning in 0.41s =========================
```

Plus regression run on related suites:

```
$ pytest molmetal/tests/test_wf363a2_user_ok_ux.py \
        molmetal/tests/test_wf366a1_g5_user_ok_gate.py \
        molmetal/tests/test_wf366a2_recover_g6_block_order.py \
        molmetal/tests/test_wf378a2_g6_user_ok_gate.py
collected 29 items
molmetal/tests/test_wf363a2_user_ok_ux.py .................              [ 58%]   (17/17 — G6 migrated to strict)
molmetal/tests/test_wf366a1_g5_user_ok_gate.py ...                       [ 68%]   (3/3)
molmetal/tests/test_wf366a2_recover_g6_block_order.py ....               [ 82%]   (4/4)
molmetal/tests/test_wf378a2_g6_user_ok_gate.py .....                     [100%]  (5/5 NEW)
========================= 29 passed, 1 warning in 0.95s =========================
```

---

## §5 Gate parity status — NOW 7/7 wrappers have USER_OK gate

Before Wf-378-A2: **6/7 wrappers** had USER_OK gate (G6 was the gap).
After Wf-378-A2: **7/7 wrappers** have USER_OK gate parity.

| # | Wrapper | Gate value | dry-run policy | Family | Origin |
|---|---|---|---|---|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | USER_OK=YES | r1_optional | Wf-300C | Wf-355-A1 |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | USER_OK=YES | strict | Wf-294B-or-300C | Wf-355-A2 |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | USER_OK=1 | strict | Wf-294B | Wf-355-A3 |
| 4 | `launch_wf131_p3_deferred_runs.sh` | USER_OK=1 | strict | Wf-294B | Wf-355-B1 |
| 5 | `launch_wf146_deferred_runs.sh` | USER_OK=YES | r1_optional | Wf-300C | Wf-355-B2 |
| 6 | `launch_wf124_p3_g5.sh` | USER_OK=1 | strict | Wf-294B | Wf-366-A1 (gap closure) |
| 7 | `launch_wf124_p3_g6.sh` | **USER_OK=1** | **strict** ✅ | **Wf-294B** | **Wf-378-A2** (gap closure — this Wf) |

Plus helper wrappers:
- `recover_g6_failed_tuples.sh`: USER_OK=1 strict (Wf-366-A2 GAP-2 fix)
- `pre_flight_gpu_ready.sh`: USER_OK=1 skip-on-dry-run (Wf-350-B1 verifier,
  intentional SKIP)

**G6 gate parity: CLOSED** (1/7 → 0/7). Wf-368-B1 §7.1 forward-looking
item (b) RESOLVED.

---

## §6 Standing-directive compliance

| Directive | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | All numeric claims cite JSON / report path or test name. PROJECTED tag used for ~7.5 hr G6 wall. ✓ |
| §1.2 Standing GPU authorization | No GPU launches; CPU-only. No user OK needed for this Wf. ✓ |
| §1.3 No paper work without authorization | No `paper/*` touches. ✓ |
| §1.4 Pre-flight v2 25/25 PASS gate | Pre-flight not invoked (no launch); per §1.4 applies to "launching a retrain / panel / ablation" — this Wf is a script fix + test, not a launch. ✓ |
| §1.5 Honest-negative framing | Section §3 above; gate fix is a CRITICAL safety improvement (closes silent bypass), not a regression. ✓ |
| §5.1 Standing GPU authorization | USER_OK gate enforcement STRENGTHENED per kill_then_adapt directive; G6 wall ~7.5 hr explicitly flagged in plan.json honest_framing. ✓ |
| §5.2 Honest framing mandate | MEASURED_today + PROJECTED + kill_then_adapt_directive + wf378_a2_status populated in plan.json per Wf-355-A3 convention. ✓ |
| §5.3 No paper work without authorization | No `paper/*` touches. ✓ |
| §5.4 Tier-2 journal target + NO wet-lab planning | N/A (no wet-lab work, no paper text). ✓ |

---

## §7 Cross-references

- `scripts/launch_wf124_p3_g6.sh` (modified; canonical Wf-366-A1 pattern)
- `scripts/launch_wf124_p3_g5.sh` (reference pattern; Wf-366-A1 fix)
- `scripts/recover_g6_failed_tuples.sh` (Wf-366-A2 lesson source)
- `scripts/_dry_run_log_helper.sh` (Wf-363-A1 shared helper)
- `molmetal/tests/test_wf378a2_g6_user_ok_gate.py` (NEW, this Wf)
- `molmetal/tests/test_wf363a2_user_ok_ux.py` (audit; G6 migrated strict)
- `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` (sibling pattern)
- `molmetal/tests/test_wf366a2_recover_g6_block_order.py` (sibling block-order pattern)
- `molmetal/tests/test_wf365b2_7_wrappers_final.py` (final integration; G6 gap flagged here)
- `TODO/completed/363a2_user_ok_ux.md` (Wf-363-A2 audit; G6 originally `loose`)
- `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (Wf-366-A1 canonical pattern)
- `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (Wf-366-A2 lesson)
- `TODO/completed/365b2_7_wrappers_final.md` (Wf-365-B2 final integration; gap surfaced)
- `TODO/completed/368b1_user_ok_final.md` §7.1 (forward-looking item closed)
- `TODO/completed/182_wf124_p3_per_innovation_ablation.md` (parent plan)
- `AGENTS.md §3.4 skill catalog + §5 standing directives` (convention source)

**END OF Wf-378-A2 SHIP DOC** — 2026-09-30.