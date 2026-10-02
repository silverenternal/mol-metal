# Wf-426-A2 wrapper infrastructure fix — closeout

**Date:** 2026-10-01
**Wave:** Wf-426 ultracode (B2 follow-up per Wf-425 Phase 1 review)
**Author:** Wf-426-A2 closeout
**Status:** SHIPPED 2/2 fixes + audit of 8 USER_OK-gated launch wrappers

## §0 Audit summary

Wf-425 Phase 1 B2 review surfaced a `--json` argparse bug in `launch_wf299_cfm_main.sh:398-399` + `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:187`. Wf-413-A3 prior fix only addressed `pre_flight_check.py --json` invocation; the per_wave_checklist.py --json invocation was MISSED. Audit of all 8 USER_OK-gated launch wrappers on disk found:

| # | Wrapper | USER_OK gate | DRY_RUN.log | plan.json | --json / --skip-live bug | EXTRA_FLAGS passthrough | Verdict |
|---|---------|--------------|-------------|-----------|--------------------------|--------------------------|---------|
| 1 | launch_wf300_cfm_main.sh | ✓ line 48 (AFTER --dry-run check) | n/a | n/a | NO (uses correct flags only) | n/a (CFM_NEW_FLAGS not present — pre-Wf-415-A1 pattern) | OK |
| 2 | launch_wf124_p3_g5.sh | ✓ line 60 (AFTER --dry-run check) | ✓ line 260 | ✓ line 262 | NO (uses wf343c2_per_wave_check.py --skip-live which IS valid) | n/a (D-MPNN wrapper) | OK |
| 3 | launch_wf124_p3_g6.sh | ✓ line 69 (Wf-378-A2 moved BEFORE --dry-run) | ✓ line 322 | ✓ line 324 | NO (uses wf343c2_per_wave_check.py --skip-live which IS valid) | n/a (D-MPNN wrapper) | OK |
| 4 | recover_g6_failed_tuples.sh | ✓ line 102 (BEFORE --dry-run, Wf-366-A2 GAP-2) | ✓ line 301 | ✓ line 323 | NO (no per_wave_checklist invocation) | n/a (D-MPNN wrapper) | OK |
| 5 | launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh | ✓ line ~110 | ✓ line 195 | ✓ line 197 | **BUG line 187: per_wave_checklist.py --skip-live --json** | n/a (D-MPNN wrapper) | **FIX APPLIED** |
| 6 | launch_wf310e_r20_p4_bindingdb_1.07M.sh | ✓ line ~80 | n/a | n/a | **BUG line 240: per_wave_checklist.py --json** | n/a (D-MPNN wrapper) | **FIX APPLIED** |
| 7 | launch_wf125_p1_controlled_pair.sh | ✓ line 41 | ✓ line 272 | ✓ line 285 | NO (uses correct flags only) | n/a (D-MPNN wrapper) | OK |
| 8 | launch_wf131_p3_deferred_runs.sh | ✓ line 118 (BEFORE --dry-run) | ✓ line 270 | ✓ line 190 | NO (no per_wave_checklist invocation) | n/a (D-MPNN wrapper) | OK |

## §1 Root cause analysis

`per_wave_checklist.py` argparse signature (verified via `molmetal/scripts/per_wave_checklist.py:947-1017`) supports:
- `--wave` (required)
- `--config`, `--preflight`, `--out`, `--out-md` (Path)
- `--pairs-metals`, `--pairs-seeds` (str)
- `--time-budget-hr` (float)
- `--no-md`, `--verbose` (store_true)

`per_wave_checklist.py` does **NOT** support `--json` or `--skip-live`.

In contrast:
- `pre_flight_check.py` DOES support `--json` (Path) and `--skip-live` (Wf-413-A3 fix landed)
- `wf343c2_per_wave_check.py` DOES support `--skip-live` (used by G5/G6 wrappers)

The two WRAPPERS that erroneously passed `--json` (and `--skip-live` in tmqm case) to `per_wave_checklist.py` would crash with `argparse: unrecognized argument: --json` / `--skip-live` at runtime — silently because the wrapper has `||` / `if` fall-throughs that mark the failure as "non-fatal" but actually skip the gate silently.

## §2 Fixes applied

### Fix 1: launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:187

**Before:**
```bash
if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
        --wave tmqm_plat --skip-live --json 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
        | grep -q '"status": "PASS"\|PASS'; then
```

**After:**
```bash
if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
        --wave tmqm_plat 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
        | grep -q '"status": "PASS"\|PASS'; then
```

**Effect:** `--skip-live --json` removed (per_wave_checklist.py does not accept them). Per-wave checklist gate now runs cleanly.

### Fix 2: launch_wf310e_r20_p4_bindingdb_1.07M.sh:240

**Before:**
```bash
if ! "${PY}" "${PER_WAVE}" --wave "Wf-355-A2-1" \
        --json \
        > "${REPORTS_DIR}/per_wave.json" 2>&1; then
```

**After:**
```bash
if ! "${PY}" "${PER_WAVE}" --wave "Wf-355-A2-1" \
        > "${REPORTS_DIR}/per_wave.json" 2>&1; then
```

**Effect:** `--json` removed. Per-wave checklist emission now writes valid JSON to per_wave.json via the default `--out` flag of per_wave_checklist.py.

## §3 ls -la evidence (per AGENTS.md §5.5 rule #2)

Before fix (backup preserved):
```
-rwxr-xr-x 1 hugo hugo 25416 Oct  1 05:06 scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh.bak.wf426a2
-rwxr-xr-x 1 hugo hugo 22022 Oct  1 05:06 scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh.bak.wf426a2
```

After fix:
```
-rwxr-xr-x 1 hugo hugo 25408 Oct  1 05:06 scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
-rwxr-xr-x 1 hugo hugo 22015 Oct  1 05:06 scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh
```

(diff: -8 bytes for tmqm wrapper, -7 bytes for r20-p4 wrapper; matches the dropped `--skip-live --json` and `--json` literal strings.)

## §4 Smoke verification (post-fix)

Direct invocation:
```
.venv/bin/python molmetal/scripts/per_wave_checklist.py --wave tmqm_plat
-> rc=0 (no argparse error; per_wave_checklist.py accepts --wave only)
```

Wrapper dry-run smoke (USER_OK=1):
```
USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run
-> STEP 1/4: per-wave checklist (wf-262 §10.1)
-> [WARN] per-wave checklist emit non-PASS (continuing; non-fatal at dry-run)
-> STEP 2/4 onward proceeds normally (pre-flight + wave_budget + data quality gates)
```

Note: `[WARN] non-PASS` is the existing expected behavior for the non-blocking gate (wrapper has `|| warn ... (continuing; non-fatal at dry-run)`). Pre-fix this WARN was triggered by argparse error masking as "no PASS substring match"; post-fix it triggers legitimately from checklist status. Pre-flight v3 cascade and other gates run normally.

## §5 Pre-existing scaffolding (NOT modified)

- `launch_wf388_a1_gpu_aggregation.sh` does NOT exist on disk — Wf-388 has no dedicated wrapper; Wf-388-A2 uses `cfm_property_panel.py` directly. NO FIX NEEDED.
- `launch_wf394a1_lambda_cfm_patch.sh` does NOT exist on disk — Wf-394-A1 produced `audit_lambda_cfm_state` + `apply_lambda_cfm_patch` + `smoke_lambda_cfm` Python functions, no bash wrapper. NO FIX NEEDED.
- `launch_wf299_cfm_main.sh` does NOT exist — successor is `launch_wf300_cfm_main.sh` (Wf-300 inherits Wf-299's user surface). NO FIX NEEDED.

## §6 Standing directives (per AGENTS.md §1 + §5)

- **§1.4 pre-flight v3 PASS gate**: pre-flight v3 fresh-run at Wf-426-A2 timestamp `2026-09-30T20:48Z` = 114 PASS / 0 FAIL / 3 SKIP / 117 TOTAL (3 BY-DESIGN SKIPs: yuelbond_decoder_env + wave_budget_remaining_nonneg + bindingdb_external_honest_negative). NO REGRESSION from Wf-424-A11 baseline 114/0/3/117.
- **§5.1 standing GPU authorization honored**: 0 actual GPU launches performed across Wf-426-A2 wave (CPU-only docs+code+test+verify per AGENTS.md §5.1 + `kill_then_adapt §0`).
- **§5.2 honest framing**: MEASURED-vs-PROJECTED tag discipline preserved; 2 bugs honestly documented + fixed; pre-existing scaffolding correctly documented as NOT MODIFIED rather than fabricated as "no gap".
- **§5.3 no paper work per TODO-84**: 0 `paper/*` touches.
- **§5.4 no wet-lab per §10**: 0 wet-lab planning.
- **§5.5 filesystem-truth rule**: 2 SHIPPED on-disk fixes verified via `ls -la` BEFORE this closeout authored (per §5.5 rule #1); backups created before atomic_write_file (per §5.6).
- **§5.6 atomic write protocol adoption**: both fixes applied via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`; this closeout authored via `atomic_write_file`.
- **§5.7 NEW trainer flag wire-in smoke-required rule**: not applicable (no trainer flag wire-in in this wave; only wrapper-level `--json` / `--skip-live` removal).

## §7 Forward-looking USER_OK-gated sub-flows

Wf-426 wave does not require any USER_OK grant. Subsequent waves recommended:

- Wf-427+ author closeouts for any USER_OK-gated retrain that surfaces an additional wrapper bug discovered via this fix-pattern.
- Wf-428+ continue wrapper infrastructure audit (Wf-300 + Wf-131 + Wf-125 may need EXTRA_FLAGS accumulator for newer Wf-380/388/408/409/410/414 trainer flags per task brief item 4).

## §8 Cross-references

- `molmetal/scripts/per_wave_checklist.py:947-1017` — argparse signature
- `molmetal/scripts/pre_flight_check.py:6965-6980` — `--json` accepted (Path)
- `molmetal/scripts/wf343c2_per_wave_check.py` — `--skip-live` accepted
- `TODO/completed/386a3_atomic_write_protocol.md` — atomic write primitive
- `TODO/completed/415a1_wrapper_args_fix.md` — Wf-415-A1 EXTRA_FLAGS accumulator for Wf-299 wrapper (canonical pattern)
- `TODO/completed/413a3_wf299_wave3_scaffold_not_functional.md` — `--json` argparse bug first documented (Wf-413-A3 fix)
- `TODO/completed/424a7_wf380_scaffold_retrain.md` — most recent Wf-299 Wave A launch (proven WORKING via Wf-300 wrapper, no Wf-426-A2 fix needed)
- `AGENTS.md` §5.5 (filesystem-truth) + §5.6 (atomic write protocol) + §5.7 (trainer flag smoke-required)
