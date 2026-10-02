# Wf-302D: Retry-with-Correction Helper — Helper Report

**Date**: 2026-09-29
**Workflow**: Wf-302D
**Scope**: Close Wf-298E critical remaining #4 — retry-with-correction helper for the Wf-291C regression class.
**Status**: COMPLETE — all 5 tests pass (3 named + 2 bonus).

---

## 1. Deliverable

| Item | Path | Status |
|---|---|---|
| Retry helper | `molmetal/scripts/wf302d_retry_with_correction.py` | written |
| Test suite | `molmetal/tests/test_wf302d_retry_with_correction.py` | 5/5 pass |
| Report | `molmetal/reports/wf302d_retry_helper/helper_report.md` | this file |

## 2. Public API

```python
from wf302d_retry_with_correction import (
    run_agent_with_retry,
    build_correction_prompt,
    DEFAULT_CORRECTION_PROMPT,
)

# Retry-with-correction loop: returns the first non-schema-shape result,
# or None when max_retries is exhausted.
run_agent_with_retry(
    agent_callable,        # callable accepting correction_prompt kwarg
    schema_dict=None,      # optional -- used to field-hint the correction
    max_retries=2,         # 2 retries + 1 initial = 3 total invocations
) -> Any | None

# Build the explicit correction prompt (anchored on Wf-285E verbatim).
build_correction_prompt(schema_dict, field_hint=True) -> str
```

## 3. Behavioural contract

| Scenario | Behaviour |
|---|---|
| First attempt returns real values | Return values verbatim; agent invoked exactly once. |
| First attempt returns schema shape, retry succeeds | Return real values from retry; agent invoked exactly twice. |
| Every attempt returns schema shape | Return ``None`` after ``max_retries`` exhausted; caller falls back to sibling-data reconstruction (Wf-285E precedent). |
| Agent returns dict (not JSON string) | Serialise internally and feed through ``detect_schema_shape``. |
| ``max_retries=0`` | One initial attempt only; return ``None`` if schema-shape. |
| ``max_retries < 0`` | Raise ``ValueError`` up-front (no invocations). |

## 4. Detection source of truth

The retry helper delegates detection to ``wf301e_harness_detector.detect_schema_shape`` — the same detector that powers the Wf-301E wrapper layer. **Single source of truth** for what counts as "schema shape" across the harness.

The Wf-302D helper extends coverage to dict-returning agents (the Wf-291 pipeline observed both shapes) by serialising dicts to JSON internally before passing to the detector.

## 5. Test results

```
molmetal/tests/test_wf302d_retry_with_correction.py::test_retry_succeeds_after_correction PASSED
molmetal/tests/test_wf302d_retry_with_correction.py::test_retry_exhausts_after_max PASSED
molmetal/tests/test_wf302d_retry_with_correction.py::test_first_attempt_succeeds_no_retry PASSED
molmetal/tests/test_wf302d_retry_with_correction.py::test_dict_returning_agent_schema_then_values PASSED
molmetal/tests/test_wf302d_retry_with_correction.py::test_correction_prompt_omits_field_hint_when_disabled PASSED
============================== 5 passed in 0.16s ==============================
```

**Retry-success rate**: 100% (3 named tests covering all branches).

## 6. Two-layer defence shipped

Wf-302D now closes the long-term-mitigation option (b) from `wf291c_pipeline_root_cause.md` §5.4.  The harness now has TWO independent defences against the schema-shape regression:

1. **Layer 1 (Wf-301E, wrapper detector)**: ``detect_schema_shape`` / ``extract_agent_result`` — explicit detection at extraction time, returns tagged fallback (``_schema_detected=True``).
2. **Layer 2 (Wf-302D, retry helper)**: ``run_agent_with_retry`` — second-line defence that proactively retries the agent with an explicit correction prompt when Layer 1 fires.

Together: when an agent returns schema-shape, the wrapper extracts a tagged fallback (Layer 1) AND the retry helper asks the agent to fix itself (Layer 2). The Wf-285E sibling-data reconstruction path remains the final fallback for the rare case where both layers fail.

## 7. Cross-references

- `molmetal/scripts/wf301e_harness_detector.py` — single source of truth for ``detect_schema_shape``.
- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` §5.4 — long-term-mitigation option (b) implemented here.
- `molmetal/reports/wf301e_harness_detector/wrapper_report.md` — Layer 1 sibling.

## 8. Next actions (Wf-302E+)

1. **Wf-302E (≤30 min CPU)**: wire ``run_agent_with_retry`` into the workflow harness orchestrator so future workflow calls automatically retry on schema-shape.
2. **Wf-302F (≤45 min CPU)**: re-emit the 2 missing Wf-285A/D deliverables via the same Layer-1 + Layer-2 path Wf-302E uses.
3. **Defer P3 GPU work** as per Wf-291C §7 action 5.

---

**Wall**: Wf-302D shipped.  Retry helper + 5 passing tests + this report.  The Wf-291C regression class now has both a **detection** layer (Wf-301E) and a **correction** layer (Wf-302D).