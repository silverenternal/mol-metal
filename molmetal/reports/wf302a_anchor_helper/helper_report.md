# Wf-302A — AnchorGuard helper for read_anchor_json

**Date**: 2026-09-29
**Ship surface**: `molmetal/scripts/read_anchor_json.py` + `molmetal/tests/test_read_anchor_json.py`
**Closes**: Wf-271B + Wf-287A m14 anchor re-use risk

## Motivation

The M14 lead-deployment precondition (TODO-207 §9.AF + TODO-207 §9.6.3) reuses a paper-grade anchor from Wf-287A across multiple downstream consumers (paper §4, m14 LEAD claim, dashboard, auto-update). Historically three bug classes have surfaced:

1. **Stale read** — the anchor was last regenerated weeks ago but a consumer silently re-emitted the old number.
2. **Missing read** — the anchor was renamed / archived; the consumer crashed with a bare `FileNotFoundError` whose traceback pointed at the wrong call site.
3. **Version drift** — a downstream consumer pinned `v1` but the anchor was upgraded to `v2`, causing a silent `KeyError` deep in the consumer.

`AnchorGuard` makes all three failure modes loud and explicit with named exception types.

## Reference anchors examined

* `molmetal/reports/wf269c_cfm_real/wf269c_report.json` — current m14 anchor pattern (per-config CFM ablation results, 51/51 decode_ratio)
* `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` — Wf-287A formal anchor (`metric`, `wf_origin`, `date`, `value`, `methodology`, `verdict`, `use_when`, `cross_references`, `status`)

## Public API

```python
from molmetal.scripts.read_anchor_json import AnchorGuard

guard = AnchorGuard(
    path="molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json",
    max_age_sec=86400,             # 24 hr
    required_version="m14_v1",     # optional
)
anchor = guard.require()
print(anchor["value"])
```

### Named exception types

* `AnchorMissingError` (subclass of `FileNotFoundError` for back-compat)
* `AnchorStaleError`
* `AnchorJSONError`
* `AnchorVersionMismatchError`

## Test results

```
molmetal/tests/test_read_anchor_json.py::test_fresh_anchor_passes PASSED                          [ 20%]
molmetal/tests/test_read_anchor_json.py::test_stale_anchor_raises PASSED                          [ 40%]
molmetal/tests/test_read_anchor_json.py::test_wrong_version_raises PASSED                         [ 60%]
molmetal/tests/test_read_anchor_json.py::test_missing_anchor_raises PASSED                        [ 80%]
molmetal/tests/test_read_anchor_json.py::test_malformed_json_raises_anchor_json_error PASSED    [100%]

5 passed, 1 warning in 0.40s
```

### Behaviors covered

| Behavior | Test |
|---|---|
| Fresh anchor passes (4 API surfaces: read / require / convenience) | `test_fresh_anchor_passes` |
| Stale mtime raises `AnchorStaleError` (with age + max_age_sec in message) | `test_stale_anchor_raises` |
| Version mismatch raises `AnchorVersionMismatchError` (incl. missing-field case) | `test_wrong_version_raises` |
| `required_version=None` skips version check | `test_wrong_version_raises` (Case B) |
| Missing path raises `AnchorMissingError` (incl. back-compat with `FileNotFoundError`) | `test_missing_anchor_raises` |
| Malformed JSON raises `AnchorJSONError` with path-prefixed message | `test_malformed_json_raises_anchor_json_error` |

## Honest framing (party-twenty-spirit §3.5)

* **MEASURED**: 5 tests pass against the new helper on CPU.
* **MEASURED**: helper imports cleanly, no external deps.
* **PROJECTED** (not measured here): downstream consumers (`paper_auto_update.py`, `paper_dashboard.py`, `pre_flight_check.py`) currently read these anchor JSON files ad-hoc — adopting `AnchorGuard.require()` at each call site is a follow-up and is NOT shipped in this workflow run. Wf-302B (if dispatched) would do the wire-in.
* **PARTIAL**: `m14_anchor.json` does not yet carry a `version` field, so callers wanting version pinning must either supply `version_field="wf_origin"` (legacy) or first add `"version": "m14_v1"` to the anchor itself. Out of scope here.

## Files shipped

* `molmetal/scripts/read_anchor_json.py` — helper (220 LOC)
* `molmetal/tests/test_read_anchor_json.py` — 5 tests
* `molmetal/reports/wf302a_anchor_helper/helper_report.md` — this report
