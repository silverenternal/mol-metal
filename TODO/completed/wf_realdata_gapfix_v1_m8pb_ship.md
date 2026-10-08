# Wf-RealGap v1 — m8 PB fail-open threshold fix (2026-10-08)

## TL;DR

Workflow `wf_realdata_gapfix_v1` (Run ID `wf_94534c57-bf9`, Task ID `w0s4b5erm`,
duration ~15 min wall, ~174k subagent tokens) shipped a substantive CPU-only
gap fix on the m8 PB 🔴 MOCK_NEEDS_REMEASURE item: added
`M8_PB_PRODUCTION_MIN_POSES = 100` and tightened 3 PB gates so a sub-100
`n_poses` smoke can no longer silently PASS as "PB production". All 3 gates
now correctly SKIP with explicit "NOT PB production" evidence + cite the
Wf-323C honest-negative root cause (Vina 1.2.7 rejects Pt+2 octahedral).

## Files modified (filesystem-truth, §5.5)

| File | Change |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | +`M8_PB_PRODUCTION_MIN_POSES = 100` constant (line 136); 3 PB gates (`gate_wf315d_pb_smoke`, `gate_wf319c_pb_packages_status_present`, `gate_wf323c_m8_full_pass_rate_documented`) return SKIP when `n_poses < 100` |
| `molmetal/tests/test_wf315d_pb_smoke.py` | Contract widened (`PASS,FAIL`) → (`PASS,FAIL,SKIP`); assert SKIP carries "NOT PB production" |
| `molmetal/tests/test_wf319c_pb_smoke.py` | Same |
| `molmetal/tests/test_wf323c_m8_full_cpu_fix.py` | Same |
| `TODO/INDEX.md` | §17.1 row 3 + §17.2 Q5: m8 downgraded to "PB SMOKE-only, not production" with n_poses values cited; m16 reclassified EXTERNAL-DEPENDENCY (no Pt rows in MetalCytoToxDB; platinai_4cell.csv is SYNTHETIC — would be fabrication per §5.2) |

## Verification (§5.5 + §5.7)

- Pre-flight CPU re-run after the change: **103 PASS / 0 FAIL / 15 SKIP / 118 total** (no regressions; 15 SKIP includes the 3 newly-SKIP'd PB gates)
- Pytest on 3 modified test files: **19 passed + 1 skipped + 2 failed** (the 2 failures are pre-existing, NOT caused by this change — verified via `git stash` + re-run; they reflect stale expectations vs. evolved metric JSON, not gate defects)

## Why this matters

The m8 PB gate was a **fail-open** correctness hole: 3 PB gates returned PASS
on a 15-pose smoke while §17.1 called m8 "PB production". Tightening the gate
to require a production `n_poses ≥ 100` (per Wf-445-A1) closes the hole and
emits honest-negative evidence at gate-level, so the gap is now visible
upstream rather than masked.

## What is NOT closed (deferred)

- The actual m8 production re-measure is GPU + external-data gated (needs
  meeko Pt+2 parameters + receptor + USER_OK'd Vina GPU run, blocked by
  Wf-323C honest-negative that Vina 1.2.7 rejects Pt+2 octahedral)
- m4 SA/QED promotion (CPU-promotable, ~80-120 LOC; next fix candidate)
- m16 ckpt GAP (NOT CPU-actionable; EXTERNAL-DEPENDENCY per this fix's reclassification)

## Cross-refs

- TODO/completed/wf131_p3_wrapper_v4_silent_death_fix.md (sibling fix 2026-10-08: wrapper SIGPIPE silent death)
- AGENTS.md §5.2 (honest-negative framing), §5.5 (filesystem-truth), §5.6 (atomic write), §5.7 (smoke), §5.8 (wf token budget — 3 phases × 1 agent = 174k tokens, within 4-agent budget)
- TODO/INDEX.md §17 (5-category audit)
- Wf-323C (Vina Pt+2 honest-negative root cause cited in SKIP evidence)
- Wf-445-A1 (the source of the `M8_PB_PRODUCTION_MIN_POSES` threshold)
