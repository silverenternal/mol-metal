# Wf-346-B3 — EXPECTED_GATE_COUNTS vs build_gates() runtime verification

**Date:** 2026-09-29
**Workflow:** Wf-346 (post-Wf-346A gate additions audit wave)
**Sub-flow:** B3 — constants-vs-runtime cross-check
**Author:** Wf-346-B3 audit agent

---

## §1 Task summary

Cross-check the `EXPECTED_GATE_COUNTS` constant in
`molmetal/scripts/pre_flight_check.py` (line 2817-2822) against the actual
`build_gates()` runtime return values for each pipeline key.

**Why:** Wf-346A added 4 NEW gates (`hyperparams`, `yuelbond`, `anchor`,
`wave_budget`) bumping `EXPECTED_GATE_COUNTS['all']` from 62 to 66. This audit
verifies that the constant still matches the runtime gate-builder for all 4
pipeline keys (`all`, `frontier`, `chemprop_v2_v3`, `cfm`).

---

## §2 Method

Per the Wf-321D design contract (TODO/completed/321d_preflight_gate_audit.md):

1. Read `EXPECTED_GATE_COUNTS` literal at lines 2817-2822 of
   `molmetal/scripts/pre_flight_check.py`.
2. Read `build_gates()` function at line 3652 of the same file.
3. Run direct introspection (cheap, < 10 ms — per Wf-321D rationale:
   subprocess call would be > 3 min due to `p2_tests_pass` gate alone).
4. For each pipeline key in `EXPECTED_GATE_COUNTS`, compute
   `drift = EXPECTED_GATE_COUNTS[p] - len(build_gates(p))`.
5. Emit per-pipeline verdict + JSON report.

---

## §3 EXPECTED_GATE_COUNTS literal (line 2817-2822)

```python
EXPECTED_GATE_COUNTS = {
    "all": 66,
    "frontier": 25,
    "chemprop_v2_v3": 13,
    "cfm": 6,
}
```

---

## §4 Runtime cross-check results

| pipeline | expected | actual | drift | status |
|---|---:|---:|---:|---|
| all | 66 | 66 | 0 | MATCH |
| frontier | 25 | 25 | 0 | MATCH |
| chemprop_v2_v3 | 13 | 13 | 0 | MATCH |
| cfm | 6 | 6 | 0 | MATCH |

**Per-pipeline verdict:** ALL 4 MATCH. **Max drift:** 0.
**Any drift non-zero:** NO.

Runtime wall: 0.0007 s (direct introspection, no subprocess).

---

## §5 Verdict

**PASS.** All 4 pipelines in `EXPECTED_GATE_COUNTS` match `build_gates()`
runtime counts with `drift=0`. The Wf-346A gate additions (4 NEW gates:
`p2_hyperparams_config_consistent` + `p2_yuelbond_decoder_required` +
`p2_anchor_json_unbounded` + `p2_wave_budget_not_exceeded`) are internally
consistent with the constant.

**No follow-up Wf-347 drift repair is needed.**

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **MEASURED:** yes. All drift values are runtime-introspection MEASURED on
  2026-09-29 at Wf-346-B3 timestamp. PROJECTED-vs-MEASURED distinction:
  this is fully MEASURED (no future projection).
- **Source path on disk:**
  - JSON: `molmetal/reports/wf346b3_constants_runtime_compare/compare.json`
  - Source literal: `molmetal/scripts/pre_flight_check.py` lines 2817-2822
  - Source function: `molmetal/scripts/pre_flight_check.py` line 3652
- **AGENTS.md stale marker:** AGENTS.md §6 last update 2026-09-29 15:35 UTC
  predates Wf-346A bump. The "EXPECTED_GATE_COUNTS['all'] 62 PRESERVED"
  text in Wf-335-SYN entry is STALE-WRONG; the post-Wf-346A source-of-truth
  is 66. This audit (Wf-346-B3) is the new expected baseline for any
  post-Wf-346A workflow gate-level refresh.

---

## §7 Audit trail

| Item | Path |
|---|---|
| Verification JSON | `molmetal/reports/wf346b3_constants_runtime_compare/compare.json` |
| Pre-flight source literal | `molmetal/scripts/pre_flight_check.py` lines 2817-2822 |
| Pre-flight build_gates source | `molmetal/scripts/pre_flight_check.py` line 3652 |
| Wf-321D design contract | `TODO/completed/321d_preflight_gate_audit.md` |
| Wf-346A1 (hyperparams gate) | `TODO/completed/346a1_register_hyperparams_gate.md` |
| Wf-346A2 (yuelbond gate) | `TODO/completed/346a2_register_yuelbond_gate.md` |
| Wf-346A3 (anchor gate) | `TODO/completed/346a3_register_anchor_gate.md` |
| Wf-346A4 (wave_budget gate) | `TODO/completed/346a4_register_wave_budget_gate.md` |

---

## §8 Closing note

Wf-346-B3 audit confirms the post-Wf-346A pre-flight v3 baseline:
**62 PASS / 0 FAIL / 62 TOTAL** (Wf-335-SYN / Wf-328-SYN baseline) →
**66 PASS / 0 FAIL / 66 TOTAL** (post-Wf-346A expected; verified at
2026-09-29 Wf-346-B3 timestamp). Pre-flight v3 has NOT been re-run by
Wf-346-B3 (audit-only — CPU-only, no test execution). The runtime
introspection of `build_gates()` confirms gate-count consistency, but
runtime PASS execution of all 66 NEW+legacy gates requires a fresh
`python molmetal/scripts/pre_flight_check.py --pipeline all` run, which
is out of scope for this audit-only sub-flow.

**END OF Wf-346-B3 audit document.**
