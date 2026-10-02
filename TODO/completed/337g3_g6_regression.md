# Wf-337-G3 — G6 launch script regression test (prevent Wf-330-B4 class)

**Date:** 2026-09-29
**Wf:** Wf-337-G3 (post-Wf-337-G2; this Wf is the G6-specific regression test
locked-down after the Wf-337-G2 launcher v2 consolidation)
**Status:** SHIPPED — 4/4 NEW tests PASS on CPU-only path; no GPU touched.

---

## §1 TL;DR

Per Wf-330-B4 (2026-09-29): G6 launched with **stale args** that broke the
Ru seed=2 + Ir seed=0 runs (CUDA Graphs RuntimeError under
`--torch-compile-mode reduce-overhead`).  Wf-328-W2 + Wf-335-R3 patched the
launch wrapper, but **nothing locks down the post-fix state**.  Wf-337-G3
writes 4 NEW regression tests against `scripts/launch_wf124_p3_g6.sh` so
that any future PR that re-introduces stale args breaks the test.

| Item | Status |
|---|---|
| Test file `molmetal/tests/test_wf337g3_g6_regression.py` written | MEASURED — 4 tests |
| `test_g6_launcher_parses` (syntax + dry-run + 60 tuples) | MEASURED PASS |
| `test_g6_launcher_has_all_required_flags` (Wf-328 + Wf-335 + Wf-296C fixes) | MEASURED PASS |
| `test_g6_launcher_pre_flight_integration` (kill_then_adapt ordering) | MEASURED PASS |
| `test_g6_launcher_user_ok_gate` (USER_OK blocks real launch; --dry-run bypasses) | MEASURED PASS |
| CPU-only verification | MEASURED — 4/4 PASS, 0 GPU calls |
| Honest framing (party-twenty-spirit §3.5) | MEASURED-vs-PROJECTED tags per metric |

---

## §2 What this regression test prevents

### §2.1 The Wf-330-B4 regression class

Per `TODO/completed/337g2_launcher_v2.md` §2.1 + Wf-330-B4: G6 launch
wrapper leaked stale `--cuda-graphs default ON` + `--cpu-num-workers 0`
override.  Fixes shipped in Wf-328-W2 (num_workers 8) + Wf-335-R3
(`--no-cuda-graphs` safety lever).  Wf-296C additionally wired
`--no-kendall` + `--no-dann` so the per-innovation ablation actually
ablates per-config.

**Without Wf-337-G3**, any future PR that:
- Reverts `--cpu-num-workers 8` → `0` (Wf-326B-reverted override back)
- Drops `--no-cuda-graphs` from COMMON_ARGS
- Removes `--no-kendall` / `--no-dann` from CONFIRMED_FLAGS dict
- Re-orders pre_flight_check AFTER per_wave_checklist / wave_budget_tracker
- Removes the USER_OK gate (kill_then_adapt violation)

…would ship silently and trigger another Wf-330-B4-class regression.

### §2.2 What the 4 tests lock down

| Test | Locks down |
|---|---|
| `test_g6_launcher_parses` | bash syntax valid; --dry-run exits 0; reports 60 (config × metal × seed) tuples PLANNED |
| `test_g6_launcher_has_all_required_flags` | Wf-328-W2 (num_workers 8 in COMMON_ARGS, NOT 0); Wf-335-R3 (`--no-cuda-graphs`); Wf-296C (`--no-kendall`, `--no-dann` in CONFIRMED_FLAGS); UNVERIFIED_CONFIGS empty |
| `test_g6_launcher_pre_flight_integration` | pre_flight_check invoked BEFORE per_wave_checklist + wave_budget_tracker; all 3 have `\|\| { echo FAIL; exit 1; }` guards; pre_flight precedes the per-tuple runner for-loop |
| `test_g6_launcher_user_ok_gate` | USER_OK gate text present + checked BEFORE pre_flight; invocation without USER_OK exits non-zero with "USER OK REQUIRED" message; --dry-run bypasses gate (exits 0 without USER_OK) |

---

## §3 Test file location + invocation

**File:** `/home/hugo/codes/mol-metal/molmetal/tests/test_wf337g3_g6_regression.py`

**Invocation:**
```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf337g3_g6_regression.py -v
```

**Result:** 4 passed, 0.32s wall, 1 warning (hypothesis plugin — unrelated).

---

## §4 Honest framing

Per AGENTS.md §5.1 (standing GPU authorization) + §5.2 (honest framing
mandate):

- **MEASURED**: 4/4 tests PASS on 2026-09-29 against
  `scripts/launch_wf124_p3_g6.sh` current state.  CPU-only path.
  No GPU touched.  No actual training run.
- **PROJECTED**: Prevents re-introduction of Wf-330-B4-class regression.
  Will fail loudly if any of the 5 stale-arg classes above leak back.
- **Honest-negative close**: This test is structural-only.  It does NOT
  measure end-to-end G6 wall-time, MEASURED-vs-PROJECTED test_auc gap,
  or any per-metal lift.  Those are still gated by the Wf-306D USER_OK
  pre-flight + the 60-tuple launch (TODO-182 G6, 7.5 hr wall).

---

## §5 Cross-references

| Reference | Purpose |
|---|---|
| `TODO/completed/337g2_launcher_v2.md` | Wf-337-G2 (parent Wf; G6/G5/G4 launcher consolidation) |
| `TODO/completed/310b_wf310b_g6_launcher_test.md` | Wf-310B (different launcher — `launch_wf310b_g6_final.sh`; 8 tests) |
| `TODO/pending/182_wf124_p3_per_innovation_ablation.md` | TODO-182 G6 launch parent |
| `molmetal/scripts/launch_wf124_p3_g6.sh` | G6 launch wrapper (this regression test target) |
| `molmetal/scripts/pre_flight_check.py` | Pre-flight v2 25/25 PASS gate (TODO-207) |
| `molmetal/scripts/per_wave_checklist.py` | Per-wave sub-checklist (Wf-262 §10.1) |
| `molmetal/scripts/wave_budget_tracker.py` | Wave budget + `--no-enforce-budget` (Wf-262/291D) |
| `AGENTS.md` §1.4, §5.1, §5.2 | Pre-flight gate + standing GPU auth + honest framing |
| Wf-330-B4 | The regression class this test prevents |
| Wf-328-W2 | `--cpu-num-workers 8` fix (OPTIMAL_NUM_WORKERS on 24-core host) |
| Wf-335-R3 | `--no-cuda-graphs` safety lever (CUDA Graphs default ON in trainer) |
| Wf-296C | `--no-kendall` / `--no-dann` CONFIRMED_FLAGS wiring |

---

**END OF Wf-337-G3** — 4/4 PASS, regression prevention locked.
