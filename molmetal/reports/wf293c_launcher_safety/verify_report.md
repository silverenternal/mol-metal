# Wf-293C: Wf-279D retry — Wf-124 P3 launcher safety trio verify

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Refs:** Wf-288B P0 BLOCKER, Wf-283C, Wf-279D §5, Wf-280D recommendation #1
**Outcome:** GUARD INTACT — no re-add required.

## 1. Wrappers verified

| Wrapper | Path | Lines | bash -n |
|---|---|---|---|
| G4 | `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh` | 117 | OK |
| G5 | `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` | 157 | OK |
| G6 | `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh` | 186 | OK |

`n_wrappers_verified = 3`

## 2. Wf-283C guard block — present BEFORE (and AFTER, no edits needed)

### G4 (lines 62–66)
```bash
# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
${PY} molmetal/scripts/pre_flight_check.py --pipeline all || { echo "FAIL: pre_flight_check"; exit 1; }
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G4" || { echo "FAIL: per_wave_checklist"; exit 1; }
${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }
```

### G5 (lines 81–85)
```bash
# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
${PY} molmetal/scripts/pre_flight_check.py --pipeline all || { echo "FAIL: pre_flight_check"; exit 1; }
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G5" || { echo "FAIL: per_wave_checklist"; exit 1; }
${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }
```

### G6 (lines 111–115)
```bash
# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
${PY} molmetal/scripts/pre_flight_check.py --pipeline all || { echo "FAIL: pre_flight_check"; exit 1; }
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G6" || { echo "FAIL: per_wave_checklist"; exit 1; }
${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }
```

`guard_present_before = TRUE` (3/3)
`guard_present_after = TRUE` (3/3, no re-add needed)

**Note:** Wf-293C template used hard-coded `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`; existing wrappers use the `${PY}` variable (defined at line 21/27/31) which resolves to that exact path. Functionally equivalent; existing form is arguably better (centralized, with `FAIL: <tool>` diagnostic on exit 1).

## 3. Wf-262 safety trio scripts present

```
-rwxr-xr-x  32179 Sep 29 00:55  molmetal/scripts/pre_flight_check.py
-rw-r--r--  36180 Sep 28 20:25  molmetal/scripts/per_wave_checklist.py
-rwxr-xr-x  42824 Sep 29 00:48  molmetal/scripts/wave_budget_tracker.py
```

All three scripts present and reachable.

## 4. Dry-run verification

| Wrapper | --dry-run support | Verdict |
|---|---|---|
| G4 | NO (no `--dry-run` mode by design; pre-flight runs unconditionally before Wave A) | N/A (by design) |
| G5 | YES | PASS — 15 (metal, seed) pairs enumerated, CSV sanity OK (26633 rows, 11 cols) |
| G6 | YES | PASS — 60 (config, metal, seed) tuples enumerated; 30 CONFIRMED + 30 UNVERIFIED-skip-if-real |

`dry_run_verdict = PASS` (G5 + G6 dry-runs succeed; G4 has no dry-run mode by design, but its guard block sits before any GPU work so the trio always runs)

## 5. Verdict

`guard_present_before = TRUE`
`guard_present_after = TRUE` (zero edits required)
`dry_run_verdict = PASS` (G5 + G6 verified; G4 N/A by design)

P0 BLOCKER (Wf-288B) is cleared — Wf-124 P3 G4/G5/G6 launchers all wire the Wf-262 pre-launch safety trio. Wf-293C is a no-op confirmation.

## 6. Files inspected

- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh`
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh`
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh`
- `/home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/per_wave_checklist.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py`
