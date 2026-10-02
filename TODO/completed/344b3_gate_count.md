# Wf-344-B3 — Pre-flight gate-count consistency audit drift regression guard

**Wf:** wf-344-b3
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-344-B3 agent
**Status:** SHIPPED
**Refs:** Wf-338-A3 finding 11 (pre-flight `p3_gate_count_consistency_audit`
FAIL on `pipeline='audit' actual=0 vs expected=57`); Wf-321D gate
contract; Wf-327F11 docs drift guard; Wf-343C4 EXPECTED_GATE_COUNTS
bump to 66

---

## §1 Context

Per Wf-338-A3 finding 11 (TODO/INDEX.md §22.16 root cause #6 +
§22.16 gate-fail #15): pre-flight gate `p3_gate_count_consistency_audit`
emitted the misleading drift message

```
gate count drift for pipeline='audit': actual=0 vs expected=57
[-DRIFT (silent gate loss)]
```

Root cause: the audit gate was invoked with `pipeline='audit'` (a
pre-Wf-321D stale registration value) which is NOT in
`molmetal.scripts.pre_flight_check.PIPELINES = ("chemprop_v2_v3", "cfm",
"all", "frontier")`.  Because `build_gates('audit')` returns `[]`
silently, the gate FAILed with the misleading `-DRIFT` message that
pointed the debugger at the wrong layer (the EXPECTED_GATE_COUNTS
constant) when the real bug was the caller-side pipeline arg.

The pre-Wf-344-B3 fix was already partially shipped (the current
`build_gates(pipeline)` lambda correctly passes `pipeline='all'` per
the post-Wf-343C4 wire-up).  What was missing was a **fail-fast guard**
inside the gate itself so any future caller-side regression surfaces
with a clear `INVALID_PIPELINE` marker instead of the misleading
silent-zero drift message.

**Wf-344-B3** ships:

1. A 30-line code fix in `molmetal/scripts/pre_flight_check.py`
   that adds a `PIPELINES` membership guard to
   `gate_gate_count_consistency_audit` — invalid pipeline args now
   return `FAIL` with an `INVALID_PIPELINE` evidence string that
   lists the valid pipelines.
2. A NEW 4-test regression layer at
   `molmetal/tests/test_wf344b3_gate_count.py` that pins the
   `INVALID_PIPELINE` contract + the `EXPECTED_GATE_COUNTS ⊆ PIPELINES`
   invariant + the docs-drift guard for §2.1.

The Wf-321D audit doc (`TODO/completed/321d_preflight_gate_audit.md`)
is already in sync with `EXPECTED_GATE_COUNTS['all'] = 66` (per
Wf-343C4 ship surface — §2.1 + §5.2 + §6 all carry the 66 value).

---

## §2 Code fix — `gate_gate_count_consistency_audit` INVALID_PIPELINE guard

The fix is a 25-line Edit to `molmetal/scripts/pre_flight_check.py`
adding a `PIPELINES` membership check at the top of the gate
function.  Before any `build_gates()` call, the gate now verifies
that `pipeline in PIPELINES`; if not, it short-circuits with a
clear FAIL message.

Diff summary (full Edit at `pre_flight_check.py:3555-3587`):

```python
# Wf-344-B3 — invalid pipeline guard (regression for Wf-338-A3
# finding 11 where pipeline='audit' caused actual=0 silently).
# PIPELINES is the canonical list of supported pipeline names;
# EXPECTED_GATE_COUNTS must use only those keys.
if pipeline not in PIPELINES:
    return GateResult(
        name, priority, pipeline, "FAIL",
        f"INVALID_PIPELINE: pipeline={pipeline!r} is not in PIPELINES="
        f"{PIPELINES!r}.  Per Wf-321D §5.2 design contract, the audit "
        f"gate may only be invoked with one of {PIPELINES!r}.  "
        f"This is a caller-side bug (likely a stale registration "
        f"from a pre-Wf-321D workflow that pre-dates the "
        f"EXPECTED_GATE_COUNTS constant).  Update the lambda "
        f"registration to use one of {PIPELINES!r}.",
        time.perf_counter() - t0,
    )
```

The fix is intentionally **minimal** — it preserves the
`gate_gate_count_consistency_audit` signature, the existing drift
detection logic for valid pipelines, and the existing
`EXPECTED_GATE_COUNTS` constant.  Only the new invalid-pipeline
short-circuit is added.

---

## §3 Test layer — `molmetal/tests/test_wf344b3_gate_count.py`

The 4 NEW tests cover 4 distinct contracts that the Wf-338-A3
finding 11 drift bug violated:

| # | Test | Contract |
|---|---|---|
| 1 | `TestAuditPipelineInvalid::test_audit_gate_fails_with_invalid_pipeline_marker` | `pipeline='audit'` → FAIL with `INVALID_PIPELINE` evidence. Direct regression for Wf-338-A3 finding 11. |
| 2 | `TestAuditPipelineInvalid::test_audit_gate_does_not_silently_return_zero` | Invalid pipeline FAIL must NOT use the misleading `actual=0` drift message (Wf-338-A3 finding 11 silent-zero regression). Tested across 6 invalid pipeline names × 5 expected counts = 30 combinations. |
| 3 | `TestExpectedGateCountsSync::test_expected_gate_counts_keys_match_pipelines` | `EXPECTED_GATE_COUNTS.keys() ⊆ PIPELINES` + reverse invariant + positive integer counts + actual==expected for every pipeline. Prevents orphan keys (e.g. legacy `'audit'`) from re-introducing the silent-zero bug. |
| 4 | `TestAuditDocSync::test_audit_doc_section_2_1_count_matches_constant` | Audit doc §2.1 `### §2.1 \`all\` pipeline — N gates` count + §6 summary table match `EXPECTED_GATE_COUNTS['all']`. The exact docs-drift surface the Wf-338-A3 finding 11 surfaced. |

All 4 tests run in **< 0.5 s** on the host (CPU-only, no torch, no
subprocess).  The existing 3-test `test_wf327f11_gate_count_consistency.py`
suite remains unchanged; Wf-344-B3 is purely additive.

---

## §4 Verification — integration status

| Pipeline | Expected | Actual | Audit gate status |
|---|---:|---:|:---:|
| `all` | 66 | 66 | **PASS** |
| `frontier` | 25 | 25 | **PASS** |
| `chemprop_v2_v3` | 13 | 13 | **PASS** |
| `cfm` | 6 | 6 | **PASS** |
| `audit` (invalid) | n/a | n/a | **FAIL with INVALID_PIPELINE marker** (Wf-344-B3 regression guard) |

Test results (`PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf344b3_gate_count.py -v`):

```
test_audit_gate_fails_with_invalid_pipeline_marker PASSED [ 25%]
test_audit_gate_does_not_silently_return_zero      PASSED [ 50%]
test_expected_gate_counts_keys_match_pipelines     PASSED [ 75%]
test_audit_doc_section_2_1_count_matches_constant  PASSED [100%]
========================= 4 passed in 0.34s =========================
```

Combined run with the existing Wf-327F11 suite (`test_wf327f11_gate_count_consistency.py`):

```
test_expected_gate_counts_match_actual            PASSED [ 14%]
test_audit_doc_table_count_matches_constant       PASSED [ 28%]
test_audit_gate_passes_for_all_pipeline           PASSED [ 42%]
test_audit_gate_fails_with_invalid_pipeline_marker PASSED [ 57%]
test_audit_gate_does_not_silently_return_zero     PASSED [ 71%]
test_expected_gate_counts_keys_match_pipelines    PASSED [ 85%]
test_audit_doc_section_2_1_count_matches_constant PASSED [100%]
========================= 7 passed in 0.46s =========================
```

The Wf-321D audit doc `TODO/completed/321d_preflight_gate_audit.md`
is already in sync: §2.1 declares `66 gates`, §5.2 code block shows
`"all": 66`, and §6 summary table shows `66 (\`all\`) / 25
(\`frontier\`) / 13 (\`chemprop_v2_v3\`) / 6 (\`cfm\`)`.  No doc
edit was needed.

---

## §5 Honoured standing rules (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every numeric cites on-disk JSON/MD path; gate count 66 MEASURED via `len(build_gates('all'))` |
| **§1.5 Honest-negative framing** | ✅ Wf-338-A3 finding 11 is a docs/config drift (NOT a real metric regression); declared honestly |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; CPU-only Edit + tests |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — no `paper/*` touches |
| **§5.4 Tier-2 + no wet-lab** | ✅ Zero wet-lab touches; in-silico-only code |

---

## §6 Honest framing per `party-twenty-spirit` §3.5

* This is a **docs / config drift fix** (NOT a real metric regression).
  The pre-flight gate count was already 66 in code (the Wf-343C4
  ship surface bumped EXPECTED_GATE_COUNTS from 62 → 66); the
  Wf-321D audit doc was already in sync at 66.
* The Wf-338-A3 finding 11 was a **historical** silent-zero drift
  that the Wf-321D `pipeline='all'` lambda fix (already shipped)
  resolved.  Wf-344-B3 ships the **fail-fast defense** so a
  future caller-side regression cannot silently re-introduce the
  misleading `actual=0 vs expected=N` drift message.
* The INVALID_PIPELINE guard short-circuits BEFORE `build_gates()`
  is called, so the silent-zero bug is structurally impossible.
* CPU-only fix; runs in < 0.5 s.
* Wf-321D audit doc already in sync; no doc edit needed.

---

## §7 Files touched

| File | Type | LOC |
|---|---|---:|
| `molmetal/scripts/pre_flight_check.py` | Edit | +25 lines (invalid pipeline guard + docstring) |
| `molmetal/tests/test_wf344b3_gate_count.py` | NEW | 313 lines (4-test regression layer + module docstring) |
| `TODO/completed/344b3_gate_count.md` | NEW | this file |

**Total**: 25 Edit + 313 NEW + this doc.  Zero deletions; zero
GPU launches; zero paper touches.

---

## §8 References

* `molmetal/scripts/pre_flight_check.py:3555-3587` — `gate_gate_count_consistency_audit`
  with Wf-344-B3 INVALID_PIPELINE guard
* `molmetal/scripts/pre_flight_check.py:2817-2822` —
  `EXPECTED_GATE_COUNTS` constant (already at 66 per Wf-343C4)
* `molmetal/scripts/pre_flight_check.py:101` — `PIPELINES` constant
* `molmetal/tests/test_wf327f11_gate_count_consistency.py` — existing
  3-test docs-drift guard (unchanged)
* `molmetal/tests/test_wf344b3_gate_count.py` — NEW 4-test regression layer
* `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D ship surface
  (already at 66 per Wf-343C4)
* `TODO/INDEX.md` §22.16 — pre-flight v3 15-FAIL gate breakdown
  (Wf-338-A3 finding 11 root cause #6 + gate-fail #15)
* `TODO/completed/343c4_integration.md` — Wf-343C4 ship (4 NEW gates;
  EXPECTED_GATE_COUNTS bump 62 → 66)
* `TODO/completed/327f11_gate_count_docs_fix.md` — Wf-327F11 ship
  (3-test docs-drift guard; unchanged)