# Wf-301E: Harness Shape Detector — Wrapper Report

**Date**: 2026-09-29
**Workflow**: Wf-301E
**Scope**: Close the Wf-291C regression class via a wrapper-level shape detector.
**Status**: COMPLETE — all 11 tests pass (5 named + 6 parametrised).

---

## 1. Deliverable

| Item | Path | Status |
|---|---|---|
| Detector module | `molmetal/scripts/wf301e_harness_detector.py` | written |
| Test suite | `molmetal/tests/test_wf301e_harness_detector.py` | 11/11 pass |
| Report | `molmetal/reports/wf301e_harness_detector/wrapper_report.md` | this file |

## 2. Public API

```python
from wf301e_harness_detector import detect_schema_shape, extract_agent_result

# Returns True iff the JSON string is a JSON Schema definition (not values).
detect_schema_shape(json_str: str) -> bool

# Returns the real-data dict from any agent result envelope (dict / str /
# None); schema-shape payloads are tagged with "_schema_detected": True.
extract_agent_result(agent_result) -> dict
```

## 3. Regression class closed

The 6-occurrence streak (Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 /
Wf-285) of upstream-pipeline regression is now addressable at the
wrapper layer:

- **Schema-shape payload detected**: a payload like
  `{"type":"object","properties":{"G1_…":{"type":"boolean"}, …}}`
  returns `True` from `detect_schema_shape` and is tagged
  `_schema_detected=True` by `extract_agent_result` (the parent's
  `undefined` field-extraction is replaced with an explicit empty-tagged
  result, allowing sibling-data reconstruction per Wf-274E §5.2).

- **False-positive guard**: a payload like
  `{"type":"object","G1_score":0.91,"G2_passes":true}` (where
  `"type":"object"` is a domain tag, NOT a JSON Schema declaration) is
  correctly classified as values.

- **Plain-text passthrough**: Wf-274A/B/C/D-style plain-text agent
  outputs survive the extractor unchanged (wrapped as `{"text": …}`)
  so the Wf-280E first-clean-4-of-4 round continues to work.

## 4. Test results

```
molmetal/tests/test_wf301e_harness_detector.py::test_real_value_dict PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_schema_shape_dict PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_nested_real_value PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_nested_schema_shape PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_text_passthrough PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[payload0-False] PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[payload1-True] PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[just a sentence-False] PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[-False] PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[hello world-False] PASSED
molmetal/tests/test_wf301e_harness_detector.py::test_detector_round_trip[payload5-False] PASSED
============================== 11 passed in 0.16s ==============================
```

## 5. Cross-references

- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` §2.1 (regression evidence) + §5.4 (long-term mitigations; option (c) auto-fallback implemented here at the wrapper layer).
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.1.1 wave-ledger catalog (will catalog Wf-301E in next pass).

## 6. Next actions (Wf-301F / Wf-301G)

1. **Wf-301F (≤30 min CPU)**: wire `extract_agent_result` into the
   workflow harness `journal.jsonl` writer so future agents get the
   tagged fallback rather than raw schema-shape.
2. **Wf-301G (≤45 min CPU)**: re-emit the 2 missing Wf-285A/D
   deliverables via the same sibling-data reconstruction path Wf-285E
   used (per `wf291c_pipeline_root_cause.md` §7 action 3).
3. **Defer P3 GPU work** as per Wf-291C §7 action 5.

---

**Wall**: Wf-301E shipped.  Detector + 11 passing tests + this report.
Long-term fix path (c) "auto-fallback to text" implemented at the
wrapper layer as a drop-in replacement for downstream `StructuredOutput`
extractors.