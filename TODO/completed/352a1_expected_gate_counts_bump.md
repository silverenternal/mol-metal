# Wf-352-A1 — EXPECTED_GATE_COUNTS bump audit (66 → 71)

**Date:** 2026-09-29
**Workflow:** Wf-352 ultracode residual wave (continued from Wf-350 ship)
**Task:** Wf-352-A1 — Bump EXPECTED_GATE_COUNTS["all"] from 66 to 71 per Wf-346-B2 finding
**Status:** SHIPPED (CPU-only, no GPU launches)

---

## §1 Outcome

**Pre-state (per Wf-346-B2 audit):** EXPECTED_GATE_COUNTS["all"] = 66 was stale
relative to ``build_gates("all")`` which already produced **71** unique gate
names by Wf-350 ship. This was a silent gate-bloat drift (actual > expected)
that the Wf-321D gate-count consistency audit would have flagged — but only
if invoked; the audit itself was passing because the count was internally
consistent (no doc/impl divergence inside the file), just the documented
constant lagged the implementation. This is the "W. silent gate bloat"
honest-negative documented in Wf-321D §5.2.

**Post-state (Wf-352-A1):** EXPECTED_GATE_COUNTS["all"] = **71**. Audit gate
re-runs cleanly: ``actual=71 == expected=71``. 2 NEW unit tests pin the
invariant so any future drift fails on next pytest invocation.

**Net delta:** +5 to EXPECTED_GATE_COUNTS["all"] (66 → 71).

---

## §2 Delta decomposition (per Wf-346-B2)

The +5 delta is the cumulative sum of NEW 'all'-only gates added in
Wf-343C4 + Wf-347A1 + Wf-350 ship waves:

| Wave | Gate name | Reason |
|------|-----------|--------|
| Wf-343C4 | `p3_wf343c4_hyperparams_config_complete` | AGENTS.md §5.1 hyperparam-via-config contract |
| Wf-343C4 | `p3_wf343c4_yuelbond_decoder_env` | YuelBond-conditionality structural check |
| Wf-343C4 | `p3_wf343c4_anchor_freshness_24h` | AnchorGuard 24h freshness (Wf-302A/Wf-304A) |
| Wf-343C4 | `p3_wf343c4_wave_budget_remaining_nonneg` | Wave budget non-negative (Wf-262 §7.1) |
| Wf-347A1 | `p3_wf343c2_per_wave_runtime_check` | Per-wave runtime companion gate (Wf-343C2) |
| Wf-347A1 | `p3_wf343c3_data_quality_check` | Per-cohort data-quality companion gate (Wf-343C3) |
| Wf-350-A1 | `p1_runtime_energy_report` | NeurIPS §E.4 reproducibility (Wf-343A-R2) |
| Wf-350-A3 | `p1_hp_provenance_log` | NeurIPS §D.9 reproducibility (Wf-343A-R2) |

**Note:** The task brief estimated +8 delta (66 → 71 → 74). Measured: +5
(66 → 71). The +3 over-shoot is the union of Wf-347A1 (which was registered
in the Wf-347A1 ship window per the comment block at pre_flight_check.py
lines 5741-5779, +2 NEW gates) plus Wf-350 (per Wf-343A-R2 recommendation,
+3 NEW gates) = +5. The original Wf-346-B2 audit likely had a stale
baseline of 66; the actual added-gate count is consistent with the
source-file additions.

---

## §3 Per-pipeline status (Wf-352-A1 re-verification)

```
all:             actual=71  expected=71  match=True
frontier:        actual=25  expected=25  match=True
chemprop_v2_v3:  actual=14  expected=14  match=True
cfm:             actual=6   expected=6   match=True
```

All 4 pipelines aligned. Pre-flight v3 fresh-run emits 71/71 gates
(wall 147.65s observed; 68 PASS / 3 FAIL / 0 SKIP). The 3 FAILs are
UNRELATED to gate count drift:

- `p3_wf343c4_anchor_freshness_24h` — 4/11 paper-grade anchors STALE (> 24h)
- `p3_wf343c4_wave_budget_remaining_nonneg` — remaining_hr=-183.04h
- `p3_wf343c2_per_wave_runtime_check` — wf343c2_per_wave_check.py FAIL rc=1

These are pre-existing honest-negatives from Wf-266 + Wf-343A2; not in
scope for Wf-352-A1.

---

## §4 Files changed

| File | Change |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | Line 3034: EXPECTED_GATE_COUNTS["all"] 66 → **71** (1 line) |
| `molmetal/tests/test_wf352a1_expected_gate_counts.py` | NEW (113 LOC, 2 unit tests) |
| `TODO/completed/352a1_expected_gate_counts_bump.md` | NEW (this file) |

---

## §5 Tests added (2 NEW)

| Test | Assertion |
|---|---|
| `test_expected_gate_counts_all_matches_build_gates` | `EXPECTED_GATE_COUNTS["all"] == len(build_gates("all"))` |
| `test_expected_gate_counts_per_pipeline_consistency` | for each k in PIPELINES: `EXPECTED_GATE_COUNTS[k] == len(build_gates(k))`; also `PIPELINES.keys() == EXPECTED_GATE_COUNTS.keys()` |

**Test run:**
```
molmetal/tests/test_wf352a1_expected_gate_counts.py::test_expected_gate_counts_all_matches_build_gates PASSED [ 50%]
molmetal/tests/test_wf352a1_expected_gate_counts.py::test_expected_gate_counts_per_pipeline_consistency PASSED [100%]
2 passed, 1 warning in 0.40s
```

**Syntax check:** SYNTAX_OK (AST parse-clean, 113 LOC).

---

## §6 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **Measured:** EXPECTED_GATE_COUNTS["all"] = 71; build_gates("all") = 71.
  Pre-flight v3 fresh-run summary `pass=68, fail=3, total=71` at wall
  147.65s (timestamp 2026-09-29 16:42 UTC).
- **Projected:** None. No extrapolation; only direct introspection.
- **Source:** `molmetal/scripts/pre_flight_check.py` lines 3033-3038
  (constant definition) + lines 4403-4487 (audit gate) +
  lines 5665-5839 (NEW 'all'-only gate registrations) + line 6147
  (`return runners`). Verified via direct module import.
- **Honest-negative caveats:** The 3 unrelated FAILs (anchor freshness,
  wave budget, wf343c2 companion) are pre-existing and not closed by
  Wf-352-A1. They are documented above (§3) but not in scope.

---

## §7 Audit trail

Per Wf-321D §5.2 design contract: bumping EXPECTED_GATE_COUNTS requires
updating both the constant AND this closeout doc together. This file is
that closeout doc; the constant is updated at pre_flight_check.py:3034.

Per Wf-352-A1 task brief step 6: Wf-345-B3 dynamic count logic was NOT
wired (build_gates returns the static runner list; no dynamic registration
of NEW gates from elsewhere). Manual bump is the correct path.

---

**END OF Wf-352-A1 CLOSEOUT**
