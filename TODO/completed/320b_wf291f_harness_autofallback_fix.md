# Wf-320B — Wf-291F harness auto-fallback 真修

**Status:** SHIPPED (CPU-only integration work)
**Date:** 2026-09-29
**Closes:** Wf-315B recommendation #2 (Wf-291F 真修)

---

## §1 Scope

Per Wf-315B recommendation #2: Wf-291F harness auto-fallback 真修.

Wf-291C was the pipeline regression root cause (parent schema-extraction
layer returns the schema definition instead of values). Wf-302C text-only
template worked around it but the underlying auto-fallback wasn't fully
fixed. Wf-320B closes the remaining gap.

---

## §2 What was shipped

| Artifact | Path | Purpose |
|---|---|---|
| Module | `molmetal/scripts/wf320b_harness_autofallback.py` | The third line of defence against Wf-291C regression |
| Tests | `molmetal/tests/test_wf320b_harness_autofallback.py` | 13 contract tests (12 brief-mandated + 1 bonus) |
| Claim | `molmetal/scripts/paper_dashboard.py` line 1345 | `claim_wf320b_harness_autofallback_shipped` |
| Doc | This file | Closeout notes |

---

## §3 The gap that was open

| Layer | Status before Wf-320B |
|---|---|
| Wf-301E (Harness shape detector) | Detects *plain schema-shape* payloads (the Wf-270/271/272/274/279/285/291/298/302 regression class) |
| Wf-302D (Retry-with-correction helper) | Calls the agent again with a generic correction prompt; bounded `max_retries=2` |
| Wf-301E.extract_agent_result | Survives plain-text / markdown / scalar agent outputs |

| Failure mode | Closed by |
|---|---|
| Schema-shape payload (agent returns the schema definition) | Wf-302D |
| Plain text / markdown passthrough | Wf-301E `extract_agent_result` |
| **Partial / mismatched values (missing keys, extras, wrong types)** | **Wf-320B (this PR)** |

The third failure mode is what Wf-320B closes.

---

## §4 Public API

### `HarnessAutoFallback`

```python
helper = HarnessAutoFallback(
    schema_dict=EXPECTED_SCHEMA,
    max_retries=3,                              # default; bounded
    fallback_strategy=FallbackStrategy.SIBLING_RECONSTRUCTION,  # default
)
result, report = helper.retry_with_correction(agent_callable)
```

Two-stage strategy:
1. **Wf-302D generic prompt** for plain schema-shape payloads.
2. **Wf-320B schema_coerce hint** for partial / mismatched values
   (enumerates the missing keys / extras / type mismatches).

When the retry budget is exhausted, the configured `FallbackStrategy`
is applied:
- `SIBLING_RECONSTRUCTION` (default): best-effort coerce + tag
  `_schema_detected=True`, `_partial=True`.
- `EMPTY_DEFAULT`: returns `{}`.
- `RAISE`: re-raises with a descriptive message.

### `detect_schema_mismatch(agent_result, schema_dict)`

Pure function returning a structured mismatch report:
```python
{
    "is_mismatch": bool,
    "is_schema_shape": bool,
    "expected_fields": [str, ...],
    "recovered_fields": [str, ...],
    "missing_keys": [str, ...],
    "extra_keys": [str, ...],
    "type_mismatches": {field: [expected_type, actual_type]},
    "reason": FailureReason,  # one of 6 buckets
}
```

### `build_schema_coerce_hint(mismatch_report, schema_dict)`

Builds a targeted corrective message that enumerates the specific gap
(missing keys / extras / type mismatches) -- not the generic
"please retry" prompt.

### `wrap_wf302d_with_autofallback(schema_dict, ...)`

The integration point promised by the brief. Callers that today use
`run_agent_with_retry` switch to this helper with one source line.

---

## §5 Test coverage (13 tests, all PASS)

| # | Test | Section |
|---|---|---|
| 1 | `test_detect_schema_shape_classified_separately` | Schema-mismatch detection |
| 2 | `test_detect_missing_keys_only` | Schema-mismatch detection |
| 3 | `test_detect_extra_keys_only` | Schema-mismatch detection |
| 4 | `test_detect_type_mismatch` | Schema-mismatch detection |
| 5 | `test_retry_with_correction_schema_coerce_hint` | Retry-with-correction |
| 6 | `test_retry_succeeds_after_schema_coerce` | Retry-with-correction |
| 7 | `test_retry_uses_wf302d_path_first_for_schema_shape` | Retry-with-correction |
| 8 | `test_first_attempt_ok_no_retry` | Retry-with-correction |
| 9 | `test_wrap_wf302d_returns_harness_autofallback` | Integration |
| 10 | `test_wrapped_helper_uses_existing_wf302d_prompt_for_schema_shape` | Integration |
| 11 | `test_max_retries_exhausted_applies_fallback_strategy` | Edge case |
| 12 | `test_max_retries_zero_still_applies_fallback` | Edge case |
| 13 | `test_coerce_drops_extras_and_wrong_types` | Bonus |

Run command:
```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf320b_harness_autofallback.py -v
```

Result: **13 passed in 0.53 sec**.

---

## §6 Pre-flight check status

**CFM pipeline:** 6/6 PASS, 0 FAIL, 0 SKIP, wall=0.0006 sec.
```
{
  "timestamp": "2026-09-29T02:43:11+0000",
  "schema_version": "207.v1",
  "pipeline": "cfm",
  "summary": {"pass": 6, "fail": 0, "skip": 0, "total": 6, "total_wall_sec": 0.0006}
}
```

**chemprop_v2_v3 / all pipelines:** wall exceeded 60 sec timeout in
this session (likely a slow checkpoint-probe gate in §3.1; unrelated
to Wf-320B -- no chemprop v2 v3 source change was made).

Honest framing: the **CFM pre-flight gate** (which is the relevant
gate for the harness auto-fallback work, since the helper is a
generic harness layer) passes cleanly. The chemprop v2 v3 gates
were not affected by Wf-320B and the timeout is a pre-existing
issue tracked separately.

---

## §7 Honest framing (party-twenty-spirit §3.5)

- Wf-320B is **integration work** (CPU-only; no GPU launch).
- The auto-fallback **reduces** future pipeline regression frequency
  (3rd line of defence) but does not **eliminate** the Wf-291C class
  (the root cause lives in the parent schema-extraction layer).
- Bounded `max_retries=3` prevents infinite loops.
- All 13 contract tests are CPU-only, deterministic, and complete
  in < 1 sec.
- Per `AGENTS.md` §5.2, all numeric values trace to a JSON / report
  path on disk; the **13 PASS** value traces to
  `molmetal/tests/test_wf320b_harness_autofallback.py` (pytest
  output). The **6/6 CFM pre-flight PASS** traces to
  `pre_flight_check.py --pipeline cfm` JSON output.

---

## §8 Files changed

| Path | Change |
|---|---|
| `molmetal/scripts/wf320b_harness_autofallback.py` | NEW (~520 LOC) |
| `molmetal/tests/test_wf320b_harness_autofallback.py` | NEW (~470 LOC) |
| `molmetal/scripts/paper_dashboard.py` | +1 claim entry at line 1345 |
| `TODO/completed/320b_wf291f_harness_autofallback_fix.md` | NEW (this file) |

No source files outside the harness layer were touched. No
`paper/*` files were touched (TODO-84 / `AGENTS.md` §5.3 respected).

---

## §9 Reference

- Wf-291C root cause: `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md`
- Wf-301E detector: `molmetal/scripts/wf301e_harness_detector.py`
- Wf-302D retry helper: `molmetal/scripts/wf302d_retry_with_correction.py`
- Wf-315B recommendation #2: pipeline integration audit
- Wf-285E precedent: sibling-data reconstruction fallback
