# Wf-352-C1 — Full verification (EXPECTED_GATE_COUNTS bump + per-pipeline match + docs synced)

**Date:** 2026-09-29
**Workflow:** Wf-352 ultracode residual wave
**Task:** Wf-352-C1 — Verify EXPECTED_GATE_COUNTS bumped + per-pipeline match + docs synced
**Status:** SHIPPED (CPU-only, no GPU launches; wall 197.97s for pre-flight + 0.39s for pytest)

---

## §1 Outcome

**Pre-flight v3 fresh-run (Wf-352-C1 timestamp 2026-09-29 09:01 UTC):**
```
pass = 89,  fail = 3,  skip = 1,  total = 93,  wall = 197.97s
```

**EXPECTED_GATE_COUNTS** (current state at Wf-352-C1 timestamp):
```
{'all': 93, 'frontier': 25, 'chemprop_v2_v3': 14, 'cfm': 6}
```

**Per-pipeline match verdict:**

| Pipeline | Actual | Expected | Match |
|---|---|---|---|
| `all` | 93 | 93 | MATCH |
| `frontier` | 25 | 25 | MATCH |
| `chemprop_v2_v3` | 14 | 14 | MATCH |
| `cfm` | 6 | 6 | MATCH |

All 4 pipelines aligned. Integration verified.

---

## §2 Delta from Wf-352-A1 brief

The Wf-352-C1 brief stated: *"After Wf-352: EXPECTED_GATE_COUNTS[all] = 71 (or dynamic per Wf-345-B3)"*.

The brief's 71 baseline was the **Wf-352-A1 post-state** (66 → 71). At Wf-352-C1
timestamp, the constant has subsequently been bumped to **93** by **Wf-353-B1**
(+22 NEW gates). Per `pre_flight_check.py` line 3126 comment:

```
"all": 93,  # Wf-353-B1: 71 pre-Wf-353-B1 baseline + 22 NEW Wf-343c1
            # auto-verifiable gates (10 code + 3 ablation + 7 data +
            # 1 test + 1 honest_framing; the 4 §5.11-§5.14 post-retrain
            # SKIP gates are intentionally NOT wired individually —
            # they remain SKIP per honest-framing convention).
```

Wf-353-B1 closeout is `TODO/completed/353b2_wf343c2_c3_verify.md` (the verify
+ bump path); Wf-353-A1..A4 are downstream gate-cohort work (Triton-minibatch-OT,
shortcut-distillation, consistency-regularization, unified-atom-bond-flow).

**Net effective delta from Wf-321D audit (66) to Wf-352-C1 (93):** +27.

---

## §3 Pre-flight FAIL decomposition

The 3 FAILs are preserved from Wf-352-A1 audit + Wf-353 evolution; none of these
are gate-count drift (gate-count is correct; gates are PASS/FAIL by content).

| Gate | Status | Reason |
|---|---|---|
| `p3_wf343c4_anchor_freshness_24h` | FAIL | 4/11 paper-grade anchors STALE (> 24h) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | FAIL | remaining_hr=-183.04h (wave budget overrun) |
| `p3_wf343c2_per_wave_runtime_check` | FAIL | wf343c2_per_wave_check.py FAIL rc=1 |

These are pre-existing honest-negatives from Wf-266 + Wf-343A2; not in scope
for Wf-352-C1. Per Wf-352-A1 §3 honest framing preserved.

The 1 SKIP is the §5.11-§5.14 post-retrain SKIP gate (honest-framing convention).

---

## §4 Docs sync verification

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| `grep -c "71" TODO/completed/321d_preflight_gate_audit.md` | preserved (≥5) | 9 | PRESERVED |
| `grep -c "Wf-352\|Wf-350\|Wf-346" AGENTS.md` | ≥ 2 | 2 | PRESERVED |
| `grep -c "Wf-353" AGENTS.md` | (post-Wf-353-B1 state, ideally synced) | 0 | **HONEST_GAP** |
| `grep -c "Wf-353" TODO/INDEX.md` | (post-Wf-353-B1 state, ideally synced) | 0 | **HONEST_GAP** |

**Sync gap finding (HONEST, not BLOCKING):** Wf-353-B1 closeout
(`TODO/completed/353b2_wf343c2_c3_verify.md` 2026-09-29 16:54) bumped
`EXPECTED_GATE_COUNTS["all"]` 71 → 93 with +22 NEW Wf-343c1 gates, but neither
AGENTS.md §6 nor TODO/INDEX.md §0/§22 has been updated to reflect the Wf-353
ship yet. This is the next docs-sync Wf (e.g. **Wf-353-D1** or **Wf-353-C2**)
to schedule. Per AGENTS.md §5.2 honest framing, it is NOT BLOCKING — the
EXPECTED_GATE_COUNTS / build_gates / pre-flight state is internally consistent
(audit-gate passing on all 4 pipelines).

Wf-352 references in AGENTS.md are at 2 locations (the AGENTS.md §6 timeline
extension + Wf-352-A1 mention) — preserved per Wf-352-B2 docs-sync.

TODO/INDEX.md has Wf-352 references at lines 614, 616, 628, 629, 633, 635
(§0.21.1 post-Wf-338-F2 timeline extension + baseline evolution table) —
preserved per Wf-352-B2 docs-sync.

---

## §5 Test results

```
molmetal/tests/test_wf352a1_expected_gate_counts.py::test_expected_gate_counts_all_matches_build_gates PASSED [ 50%]
molmetal/tests/test_wf352a1_expected_gate_counts.py::test_expected_gate_counts_per_pipeline_consistency PASSED [100%]
2 passed, 1 warning in 0.39s
```

Both tests pass — the `EXPECTED_GATE_COUNTS` invariant is pinned.

---

## §6 Files verified (no edits in this Wf-352-C1)

| File | Status |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | READ-ONLY verification; line 3126 EXPECTED_GATE_COUNTS = 93 MATCHES build_gates |
| `molmetal/tests/test_wf352a1_expected_gate_counts.py` | 2 NEW tests PASS (from Wf-352-A1) |
| `TODO/completed/352a1_expected_gate_counts_bump.md` | READ-ONLY verification |
| `TODO/completed/352a2_per_pipeline_audit.md` | READ-ONLY verification |
| `TODO/completed/352b1_wf321d_audit_doc_sync.md` | READ-ONLY verification |
| `TODO/completed/352b2_synthesis_doc_sync.md` | READ-ONLY verification |
| `TODO/completed/352b3_agmd_sync.md` | READ-ONLY verification |
| `TODO/completed/353b2_wf343c2_c3_verify.md` | READ-ONLY verification (Wf-353-B1 source of +22 gate bump) |
| `TODO/completed/352c1_full_verify.md` | NEW (this file) |

---

## §7 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **Measured:**
  - Pre-flight v3 fresh-run at Wf-352-C1 timestamp: pass=89, fail=3, skip=1, total=93, wall=197.97s.
  - EXPECTED_GATE_COUNTS = {'all': 93, 'frontier': 25, 'chemprop_v2_v3': 14, 'cfm': 6}.
  - All 4 pipelines: actual == expected (MATCH).
  - Tests: 2/2 PASS (test_wf352a1_*) in 0.39s.
  - Docs sync: Wf-352 references preserved in AGENTS.md (2 occurrences) + TODO/INDEX.md (6 line hits).
- **Projected:**
  - Wall for future pre-flight+pytest cycle: 5-10 min CPU-only (consistent with Wf-352-A1 estimate).
- **Source:**
  - `/tmp/wf352c1_preflight.log` (pre-flight fresh-run capture).
  - `molmetal/scripts/pre_flight_check.py:3125-3134` (EXPECTED_GATE_COUNTS constant).
  - `molmetal/scripts/pre_flight_check.py` lines 4403-4487 (audit gate) +
    lines 5665-5839 (NEW 'all'-only gate registrations) + line 6147 (`return runners`).
  - `TODO/completed/352a1_expected_gate_counts_bump.md` §1 (Wf-352-A1 baseline 66 → 71).
  - `TODO/completed/353b2_wf343c2_c3_verify.md` (Wf-353-B1 bump 71 → 93, +22 gates).
  - `TODO/INDEX.md` lines 614, 616, 628, 629, 633, 635 (Wf-352-B2 docs-sync).

- **Honest-negative caveats:**
  1. **Docs sync gap (HONEST_GAP, NOT BLOCKING):** Wf-353 ship (2026-09-29 16:54)
     bumped `EXPECTED_GATE_COUNTS["all"]` 71 → 93 but AGENTS.md §6 + TODO/INDEX.md
     §0/§22 have NOT been synced. Pre-flight + gate-count are internally consistent,
     so this is a docs-only sync task for the next Wf-353-D1 / Wf-353-C2.
  2. **3 pre-flight FAILs preserved** from Wf-352-A1 (anchor freshness + wave budget +
     wf343c2 companion); not in scope for Wf-352-C1 (gate-count verification task).
  3. **Wf-343c1 +22 gate comment** notes 4 §5.11-§5.14 post-retrain SKIP gates are
     NOT wired individually; remain SKIP per honest-framing convention.

---

## §8 Integration verdict

**INTEGRATION VERIFIED.** All Wf-352-A + Wf-352-B agents are integrated with the live `pre_flight_check.py` + `build_gates(...)` machinery:

- EXPECTED_GATE_COUNTS bumped 66 → 71 (Wf-352-A1) → 93 (Wf-353-B1) — internally consistent.
- All 4 pipelines (all / frontier / chemprop_v2_v3 / cfm): actual == expected (MATCH).
- 2 NEW tests pin the EXPECTED_GATE_COUNTS invariant — pass on re-invocation.
- Wf-352-B2 docs-sync preserved in AGENTS.md (2 refs) + TODO/INDEX.md (6 line hits).
- Wf-353 ship deserves the next docs-sync Wf (Wf-353-D1 / Wf-353-C2) — flagged as HONEST_GAP above.

**Wall budget consumed (Wf-352-C1):** 197.97s (pre-flight) + 0.39s (pytest) = ~3.3 min CPU-only.

---

**END OF Wf-352-C1 CLOSEOUT**
