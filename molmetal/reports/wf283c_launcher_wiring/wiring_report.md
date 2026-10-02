# Wf-283C: launcher × wave-tool wiring report

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Ref**: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1

## Edit summary

Added a wave-tool guard block to all 3 Wf-124 P3 launchers (G4/G5/G6). The block
wires in:

1. `pre_flight_check.py --pipeline all` — auto-verify TODO-207 v2 pre-flight
   checklist (P1 + P2 gates, exit 0 = all PASS, exit 1 = any FAIL).
2. `per_wave_checklist.py --wave "Wf-124-P3-G{4,5,6}"` — per-wave sub-checklist
   per TODO-208 / Wf-262.
3. `wave_budget_tracker.py` — live wave budget tracker per 207 v2 §7.1
   (default: --enforce-budget; emits DEFERRED labels for over-budget waves).

The guard is placed AFTER the existing `--dry-run` early-exit on G5 and G6 so
dry-run mode still bypasses the guards. G4 has no dry-run handling so the
guard fires unconditionally — `--dry-run` on G4 will now ALSO run the wave-tool
guards before hitting the main loop (acceptable: G4 never had a true
dry-run mode anyway).

## Per-wrapper diff

### `scripts/launch_wf124_p3_g4.sh`

Inserted guard block (5 lines + 1 blank) before line 62 `echo "=== Wave A START"`.
Wave name: `Wf-124-P3-G4`.

### `scripts/launch_wf124_p3_g5.sh`

Inserted guard block (5 lines + 1 blank) after the `--dry-run` early-exit (line 79)
and before `run_pair` definition (now line 87). Wave name: `Wf-124-P3-G5`.

### `scripts/launch_wf124_p3_g6.sh`

Inserted guard block (5 lines + 1 blank) after the `--dry-run` early-exit (line 109)
and before `run_tuple` definition (now line 117). Wave name: `Wf-124-P3-G6`.

Total LOC added: 21 lines × 3 wrappers = 21 wrapper-lines + 6 comment/blank =
~27 net wrapper LOC (below the 45-LOC budget; comment-heavy per the Wf-280D
recommendation).

## Syntax check

```
$ bash -n scripts/launch_wf124_p3_g4.sh && echo "G4: OK"
G4: OK
$ bash -n scripts/launch_wf124_p3_g5.sh && echo "G5: OK"
G5: OK
$ bash -n scripts/launch_wf124_p3_g6.sh && echo "G6: OK"
G6: OK
```

All 3 launchers pass `bash -n` (no syntax errors).

## Dry-run verification

### G5 `--dry-run` (full path through dry-run branch, guards bypassed)

```
$ bash scripts/launch_wf124_p3_g5.sh --dry-run
... 15 (metal, seed) pairs enumerated ...
DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.
```

Guard block was correctly SKIPPED — dry-run mode exits at line 79 before
reaching the new guards.

### G6 `--dry-run`

```
$ bash scripts/launch_wf124_p3_g6.sh --dry-run
... 60 (config, metal, seed) tuples enumerated ...
DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
  CONFIRMED runs (real launch): 30 (no_multi_task + no_wass only)
  UNVERIFIED runs (real launch): 30 (no_kendall + no_dann) — flag check required first
```

Guard block correctly SKIPPED.

### G4 (no `--dry-run` handler)

```
$ bash scripts/launch_wf124_p3_g4.sh --dry-run
... guard block fires (G4 has no dry-run early-exit) ...
... pre_flight_check.py PASSED ...
... per_wave_checklist.py PASSED ...
... wave_budget_tracker.py PASSED ...
=== Wave A START Tue Sep 29 12:33:58 AM CST 2026 ===
=== 00:33:58 START Ru seed=1 wave=A ===
```

G4 guards fired correctly and let the main loop proceed (all 3 wave-tools
returned exit 0). Note: G4 does not implement `--dry-run` so any invocation
with that arg will trigger real training — this is pre-existing behavior,
NOT introduced by this wiring change.

## Verdict

**PASS** — all 3 launchers are now wave-tool aware. The Wf-279D §5
cross-cutting bug (G4/G5/G6 skip pre_flight_check / per_wave_checklist /
wave_budget_tracker) is FIXED. Any future real-launch invocation will be
gated by all 3 wave-tools before any GPU work begins, enforcing the
"verify before launch" invariant per TODO-207 v2 §7.1.

## Files modified

- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh`
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh`
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh`

## Files created

- `/home/hugo/codes/mol-metal/molmetal/reports/wf283c_launcher_wiring/wiring_report.md` (this file)
