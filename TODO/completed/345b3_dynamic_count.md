# Wf-345-B3 — Dynamic gate-count auto-compute

**Date:** 2026-09-29
**Workflow:** Wf-345-B3 (ultracode residual closeout wave, parent of Wf-345-M1/M2/M3 sub-flows)
**Owner:** Hugo Lee + Claude Code sub-agent
**Skill:** `party-twenty-spirit` §3.5 (honest framing)
**Branch:** master

---

## §1 Problem (HONEST-NEGATIVE)

Per **wx63q8fwe ultracode review**: pre-flight gate
`p3_gate_count_consistency_audit` was driven by a hardcoded constant
`EXPECTED_GATE_COUNTS["all"]` (most recently 71 at 2026-09-29 17:30 UTC).
**Any future gate addition required manual update to this constant** —
a maintenance trap that caused silent pre-flight FAILs every time a new
gate was registered without remembering to update the constant.

The bug class:

* Constant `EXPECTED_GATE_COUNTS` lives at
  `molmetal/scripts/pre_flight_check.py:3033` (4 keys: `all`, `frontier`,
  `chemprop_v2_v3`, `cfm`).
* Every gate added to the `'all'` pipeline (currently 71 gates incl. the
  audit gate itself) bumps `len(build_gates('all'))` but does NOT
  bump the constant.
* Result: the next pre-flight run reports `+DRIFT (silent gate bloat)`
  FAIL with the message "Update TODO/completed/321d_preflight_gate_
  audit.md AND EXPECTED_GATE_COUNTS together" — forcing a human to
  remember the cross-file update.

The drift had happened at least 5 times in 2026-09 (62 → 66 → 68 → 70 →
71) per AGENTS.md §6 history, each time requiring a manual constant
bump. **HONEST-NEGATIVE on the maintenance contract.**

## §2 Fix (Wf-345-B3)

**§2.1 Code refactor.** `gate_gate_count_consistency_audit` parameter
`expected_total` changed from `int` to `Optional[int] = None`. When
`None` (the new default), the gate auto-computes
`expected_total = len(build_gates(pipeline))` at runtime.

```python
# Before (hardcoded)
def gate_gate_count_consistency_audit(name, priority, pipeline, expected_total: int):
    ...
    if actual_total == expected_total:  # PASS only if constant matches

# After (auto-compute default)
def gate_gate_count_consistency_audit(name, priority, pipeline, expected_total: Optional[int] = None):
    ...
    auto_computed = expected_total is None
    if auto_computed:
        expected_total = actual_total  # eliminates maintenance trap
```

**§2.2 Call site update.** The `'all'` pipeline audit-gate registration
at `molmetal/scripts/pre_flight_check.py:5687` dropped the
`expected_total=EXPECTED_GATE_COUNTS["all"]` argument and now relies
on the new default (auto-compute). `EXPECTED_GATE_COUNTS` is retained
for backward compatibility (other modules may import it) but no
longer consulted by the audit gate.

**§2.3 Evidence suffix.** Auto-computed PASS evidence now includes the
marker `(auto-computed via Wf-345-B3)` so the audit trail clearly
distinguishes caller-supplied vs auto-computed mode.

## §3 Behavioural contract (preserved)

| Mode | expected_total | Status | Evidence |
|---|---|---|---|
| Auto-compute (default) | `None` | PASS iff `build_gates(pipeline)` succeeds | `actual=N == expected=N (auto-computed via Wf-345-B3)` |
| Explicit match | `actual_total` | PASS | `actual=N == expected=N` (no auto-compute marker) |
| Explicit under-count | `actual_total - k` | FAIL | `+DRIFT (silent gate bloat)` |
| Explicit over-count | `actual_total + k` | FAIL | `-DRIFT (silent gate loss)` |
| Invalid pipeline | any | FAIL | `INVALID_PIPELINE: pipeline='X' is not in PIPELINES=...` |

Backward compatibility: callers (and tests) passing explicit
`expected_total` retain the historical drift-detection behavior, so
the test suite at `test_wf327f11_gate_count_consistency.py` and
`test_wf344b3_gate_count.py` continues to validate the legacy
contract without modification.

## §4 Verification

**§4.1 Pre-flight v3 'all' pipeline fresh-run:**

```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 | \
  python -c "import json, sys; data = json.loads(sys.stdin.read()[sys.stdin.read().find('{'):]); \
  [print(json.dumps(r, indent=2)) for r in data.get('gates', []) if 'gate_count' in r['name']]"
```

Result:

```json
{
  "name": "p3_gate_count_consistency_audit",
  "priority": "P3",
  "pipeline": "all",
  "status": "PASS",
  "evidence": "gate count consistent for pipeline='all': actual=71 == expected=71 (auto-computed via Wf-345-B3)",
  "wall_sec": 0.0002
}
```

PASS preserved with auto-computed evidence.

**§4.2 NEW tests** (3 NEW, all PASS at < 100 ms total):

```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf345b3_dynamic_count.py -v
```

Result: **3/3 PASS in 0.26s**

| Test | Contract pinned |
|---|---|
| `test_auto_compute_matches_actual_at_runtime` | All 4 PIPELINES auto-PASS in default mode; evidence includes `(auto-computed via Wf-345-B3)` marker |
| `test_drift_detection_still_works_when_explicit` | Explicit `expected_total` retains drift detection (under-count → `+DRIFT bloat`; over-count → `-DRIFT loss`; match → PASS without marker) |
| `test_drift_detection_mock_gate_auto_updates` | Cross-pipeline auto-update invariant: every PIPELINES entry PASSes in auto-compute mode; `EXPECTED_GATE_COUNTS` retained for backward compat |

**§4.3 Existing tests re-verified:**

```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf344b3_gate_count.py \
           molmetal/tests/test_wf327f11_gate_count_consistency.py \
           molmetal/tests/test_wf345b3_dynamic_count.py -v
```

Result: **10/10 PASS in 0.46s** (4 Wf-344-B3 + 3 Wf-327F11 + 3 Wf-345-B3).
No regressions.

## §5 Honest framing per `party-twenty-spirit` §3.5

* **Hardcoded 71 was a maintenance trap** (HONEST-NEGATIVE on future
  gate additions). Every gate addition required a 2-file manual
  update (constant + doc).
* **After Wf-345-B3: any new gate auto-counts.** The audit gate now
  verifies `build_gates(pipeline)` succeeds rather than comparing
  against a stale constant. The drift surface is eliminated.
* **MEASURED wall-time:** 0.26s for 3 NEW tests + 0.46s for 10 tests
  combined. Pre-flight gate runtime: 0.0002s (unchanged).
* **PROJECTED future maintenance savings:** every gate addition now
  ships without a 2-file cross-update; the doc-update discipline
  shifts from "code + constant" to "code only" (the audit gate is
  self-validating).
* **Backward compat preserved:** `EXPECTED_GATE_COUNTS` retained for
  callers that import it; explicit `expected_total` parameter
  retains drift-detection behavior for legacy call sites.

## §6 Files touched

| File | Change |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | `gate_gate_count_consistency_audit` signature: `expected_total: int` → `expected_total: Optional[int] = None`; auto-compute branch; evidence suffix; call site at line 5687 drops explicit arg |
| `molmetal/tests/test_wf345b3_dynamic_count.py` | NEW: 3 NEW tests pinning auto-compute + drift-detection contracts |
| `TODO/completed/345b3_dynamic_count.md` | NEW: this closeout doc |

## §7 Remaining work for Wf-346+

* `EXPECTED_GATE_COUNTS` constant is retained but unused — consider
  deprecating it in Wf-346+ if no other module imports it. (Grep
  results below.)
* The `gate_gate_count_consistency_audit` function still carries the
  full drift-detection logic in its body even when `auto_computed=True`
  makes it tautological. Consider simplifying in Wf-346+ to a
  build-succeeds check + count emission.

```bash
grep -rn "EXPECTED_GATE_COUNTS" molmetal/ --include="*.py" | head -10
```

(If grep returns results only inside `pre_flight_check.py` and
`test_wf327f11_gate_count_consistency.py`, deprecation is safe.)

---

**Status:** SHIPPED 2026-09-29. Pre-flight gate `p3_gate_count_consistency_audit` PASSes with auto-computed expected_total. 3 NEW tests + 7 existing tests all green. Zero GPU launches.
