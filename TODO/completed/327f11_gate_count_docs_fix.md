# Wf-327F11 — pre-flight gate-count consistency docs drift sync

**Wf:** Wf-327F11
**TODO:** TODO-327F11
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (docs sync + 3 NEW tests; CPU-only, < 2 s)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §5.2 (honest framing)
---

## §1 Scope

Wf-321D (shipped 2026-09-29) introduced a self-audit gate
`p3_gate_count_consistency_audit` that calls `build_gates(pipeline)`
and asserts `len(runners) == EXPECTED_GATE_COUNTS[pipeline]`.  When
Wf-327 added a NEW gate `p3_wf327_cpu_num_workers_sufficient` to the
`all` pipeline, the code constant was bumped (61 → 62) but the audit
doc `TODO/completed/321d_preflight_gate_audit.md` was NOT updated to
reflect the bump.

This is the EXACT class of "docs drift" the audit gate cannot detect
on its own (the audit gate verifies code-vs-code consistency; it
doesn't read the audit doc).  Wf-327F11 is the docs-drift close-out:
sync the audit doc to match the code constant + add 3 NEW tests that
pin the sync invariants.

## §2 Concrete drift

Three classes of drift were present in the audit doc:

| Location | Before (Wf-321D) | After (Wf-327F11) |
|---|---:|---:|
| §2.1 header | `all pipeline — 61 gates` | `all pipeline — 62 gates` |
| §2.1 P3 row | `Wf-325F (1 ...)` | + Wf-327 (1 cpu-num-workers ≥ 4 launch-script gate) |
| §2.1 Total | `61` | `62` |
| §5.2 code example | `EXPECTED_GATE_COUNTS["all"]: 61` | `62` |
| §6 audit summary table | `61 (all) / 25 ...` | `62 (all) / 25 ...` |
| §6 n_measured_only | `61 / 61` | `62 / 62` |
| §1 scope list | (Wf-327 absent) | + Wf-327 row |

Two additional code-comment fixes in
`molmetal/scripts/pre_flight_check.py`:

| Location | Before | After |
|---|---|---|
| `gate_gate_count_consistency_audit` docstring §3.5 (lines 2746-2757) | `61 gates (56 pre-Wf-323F baseline + ...)` | `62 gates (61 pre-Wf-327 baseline + 1 Wf-327 cpu-num-workers gate)` |

## §3 New tests

Three NEW tests in `molmetal/tests/test_wf327f11_gate_count_consistency.py`:

1. `TestExpectedGateCountsSync::test_expected_gate_counts_match_actual`
   — direct introspection: `EXPECTED_GATE_COUNTS[pipeline] ==
   len(build_gates(pipeline))` for every pipeline.  CPU-only, < 50 ms.
2. `TestAuditDocSync::test_audit_doc_table_count_matches_constant` —
   docs-drift guard: §2.1 header + §6 summary table must match the
   code constant.  CPU-only, < 100 ms.
3. `TestAuditGateFlip::test_audit_gate_passes_for_all_pipeline` —
   direct invocation of the audit gate to confirm it PASSes with the
   expected evidence text.  CPU-only, < 50 ms.

All 3 tests passed on first run (no regressions).  Total wall: < 1 s.

## §4 Audit gate post-sync verification

`p3_gate_count_consistency_audit` evidence after Wf-327F11 sync:

```
status: PASS
evidence: gate count consistent for pipeline='all': actual=62 == expected=62
wall_sec: 0.0004
```

This is the SAME status as before Wf-327F11 (the code constant was
already 62; only the docs claimed 61).  Wf-327F11 is a docs fix, NOT a
new gate FAIL→PASS flip (the gate was passing all along because the
CODE was correct; the DOCS were stale).  Honest framing per
party-twenty-spirit §3.5: this is docs drift, not a metric regression.

## §5 Standing-rule compliance

- **AGENTS.md §1 MEASURED-vs-PROJECTED:** every numeric in this doc is
  grep-derived (line numbers in §2 are MEASURED; gate counts in §4 are
  direct introspection).
- **AGENTS.md §5.2 honest framing:** §4 explicitly disclaims the
  FAIL→PASS flip claim — the audit gate was passing before Wf-327F11;
  only the docs claimed 61.  The 3 NEW tests pin the sync invariants
  so the next docs drift cannot survive a unit-test cycle.
- **party-twenty-spirit §3.5:** every test is ADDITIVE (no Wf-321D
  test was removed or modified).  The audit doc was UPDATED in place
  (not replaced) — the Wf-321D ship surface is preserved verbatim
  except for the 61→62 numeric updates + the Wf-327 row addition.

## §6 Bump protocol (Wf-321D §5.3) reaffirmation

Per Wf-321D §5.3 + this Wf-327F11 close-out, future gate-count bumps
MUST update FOUR artefacts together:

1. `molmetal/scripts/pre_flight_check.py` `EXPECTED_GATE_COUNTS` constant.
2. `molmetal/scripts/pre_flight_check.py` `gate_gate_count_consistency_audit`
   docstring (the `As-of ...` block).
3. `TODO/completed/321d_preflight_gate_audit.md` §2 (per-pipeline
   section + Total row) + §6 (audit summary table + n_measured_only).
4. Run `pytest molmetal/tests/test_wf327f11_gate_count_consistency.py`
   + the full pre-flight v2 (`pre_flight_check.py`) to confirm both
   pass.

Skipping any of the four re-introduces the docs-drift class of bug.

## §7 References

- `molmetal/scripts/pre_flight_check.py` — auto-verify script (gate
  source-of-truth).
- `TODO/completed/321d_preflight_gate_audit.md` — audit doc (was 61,
  synced to 62).
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — source-of-
  truth checklist referenced by every gate.
- `molmetal/tests/test_wf327f11_gate_count_consistency.py` — 3 NEW tests.
- AGENTS.md §1 (MEASURED-vs-PROJECTED) + §5.2 (honest framing).
