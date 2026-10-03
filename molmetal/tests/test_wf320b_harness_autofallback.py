"""Tests for the Wf-320B harness auto-fallback helper.

================================================================
Closes Wf-315B recommendation #2 (Wf-291F 真修).

Twelve named tests pin the Wf-320B contract:

  Schema-mismatch detection (4 tests)
  ------------------------------------
  1. test_detect_schema_shape_classified_separately
        ``detect_schema_mismatch`` must classify schema-shape payloads
        distinctly from value-shape payloads.
  2. test_detect_missing_keys_only
        When values are returned but with missing keys, the report
        must enumerate the missing keys.
  3. test_detect_extra_keys_only
        When values are returned with extra (unexpected) keys, the
        report must enumerate the extras.
  4. test_detect_type_mismatch
        When a value has the wrong JSON type, the report must record
        the expected / actual type tuple.

  Retry-with-correction (4 tests)
  -------------------------------
  5. test_retry_with_correction_schema_coerce_hint
        When the agent returns a partial / mismatched payload, the
        helper must invoke it again with a schema_coerce hint that
        enumerates the specific gap.
  6. test_retry_succeeds_after_schema_coerce
        The first attempt returns values with missing keys; the second
        attempt (after the schema_coerce hint) returns the corrected
        values; the helper surfaces the corrected values.
  7. test_retry_uses_wf302d_path_first_for_schema_shape
        When the agent returns the schema definition (plain schema-shape,
        not mismatch), the helper must use the Wf-302D generic prompt
        first, not the Wf-320B schema_coerce hint.
  8. test_first_attempt_ok_no_retry
        When the initial attempt returns values matching the schema,
        the helper must short-circuit (no second invocation).

  Integration with Wf-302D (2 tests)
  ---------------------------------
  9. test_wrap_wf302d_returns_harness_autofallback
        ``wrap_wf302d_with_autofallback`` returns a configured
        :class:`HarnessAutoFallback` instance.
  10. test_wrapped_helper_uses_existing_wf302d_prompt_for_schema_shape
        When the agent returns a schema, the helper must use the
        Wf-302D-style generic prompt (mentions "schema definition")
        rather than the schema_coerce prompt (mentions "missing keys").

  Edge cases (2 tests)
  --------------------
  11. test_max_retries_exhausted_applies_fallback_strategy
        When every attempt fails, the helper applies the configured
        fallback strategy and the report records budget_exhausted=True.
  12. test_max_retries_zero_still_applies_fallback
        ``max_retries=0`` means the helper applies the fallback even
        if the initial attempt returned values; this guards against
        infinite-loop regressions.

All tests are CPU-only, deterministic, and complete in < 5 seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Path bootstrap -- mirror the Wf-302D test layout.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from wf320b_harness_autofallback import (  # noqa: E402
    FailureReason,
    FallbackStrategy,
    HarnessAutoFallback,
    detect_schema_mismatch,
    wrap_wf302d_with_autofallback,
)


# ---------------------------------------------------------------------------
# Fixtures -- canonical regression payload (Wf-270A verbatim schema).
# ---------------------------------------------------------------------------

_SCHEMA_PAYLOAD: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "G1_cuda_fix_verified": {"type": "boolean"},
        "G2_config_yaml_exists": {"type": "boolean"},
        "G3_yaml_md_valid": {"type": "boolean"},
        "G4_n_pre_flight_items": {"type": "integer"},
        "G5_n_open_violations": {"type": "integer"},
        "honest_framing_note": {"type": "string"},
    },
    "required": [
        "G1_cuda_fix_verified",
        "G2_config_yaml_exists",
        "G3_yaml_md_valid",
    ],
}

_REAL_VALUES: Dict[str, Any] = {
    "G1_cuda_fix_verified": True,
    "G2_config_yaml_exists": True,
    "G3_yaml_md_valid": True,
    "G4_n_pre_flight_items": 25,
    "G5_n_open_violations": 0,
    "honest_framing_note": "MEASURED, single seed, single epoch",
}


# ===========================================================================
# Section 1 -- Schema-mismatch detection (4 tests)
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. test_detect_schema_shape_classified_separately
# ---------------------------------------------------------------------------


def test_detect_schema_shape_classified_separately() -> None:
    """Schema-shape payloads must be classified distinctly from values.

    The Wf-291C regression class returns the *JSON Schema definition*
    itself.  The mismatch detector must recognise this case AND tag it
    with ``is_schema_shape=True`` so the caller can route to the
    Wf-302D generic prompt path (rather than the Wf-320B schema_coerce
    hint, which would be misleading for a schema-shape payload).
    """
    report = detect_schema_mismatch(_SCHEMA_PAYLOAD, _SCHEMA_PAYLOAD)

    assert report["is_mismatch"] is True
    assert report["is_schema_shape"] is True
    assert report["reason"] == FailureReason.SCHEMA_SHAPE.value, (
        "Schema-shape payload was not classified as SCHEMA_SHAPE: %r" % report
    )

    # Real values must NOT be flagged as schema-shape.
    report_values = detect_schema_mismatch(_REAL_VALUES, _SCHEMA_PAYLOAD)
    assert report_values["is_schema_shape"] is False, (
        "Real values were misclassified as schema-shape: %r" % report_values
    )
    assert report_values["is_mismatch"] is False, (
        "Real values that match the schema were flagged as mismatched: %r"
        % report_values
    )
    assert report_values["reason"] == FailureReason.OK.value


# ---------------------------------------------------------------------------
# 2. test_detect_missing_keys_only
# ---------------------------------------------------------------------------


def test_detect_missing_keys_only() -> None:
    """Values with missing keys -> report enumerates them.

    The Wf-302D generic prompt cannot help when the agent returned
    *values* but the wrong ones.  The Wf-320B mismatch detector must
    surface the specific gap so the schema_coerce hint can target it.
    """
    partial = {
        "G1_cuda_fix_verified": True,
        "G2_config_yaml_exists": True,
        # G3_yaml_md_valid MISSING
        "G4_n_pre_flight_items": 25,
        # G5_n_open_violations MISSING
        "honest_framing_note": "partial",
    }
    report = detect_schema_mismatch(partial, _SCHEMA_PAYLOAD)

    assert report["is_mismatch"] is True
    assert report["is_schema_shape"] is False
    assert "G3_yaml_md_valid" in report["missing_keys"], (
        "Missing key G3_yaml_md_valid not enumerated: %r" % report["missing_keys"]
    )
    assert "G5_n_open_violations" in report["missing_keys"], (
        "Missing key G5_n_open_violations not enumerated: %r" % report["missing_keys"]
    )
    assert report["reason"] in {
        FailureReason.MISSING_KEYS.value,
        FailureReason.SCHEMA_MISMATCH.value,
    }, "Unexpected reason: %r" % report["reason"]


# ---------------------------------------------------------------------------
# 3. test_detect_extra_keys_only
# ---------------------------------------------------------------------------


def test_detect_extra_keys_only() -> None:
    """Values with extra (unexpected) keys -> report enumerates them.

    The reverse failure mode: the agent populates the right fields but
    ALSO emits a sibling-artefact field (e.g. ``"_wave"`` or ``"_ts"``)
    that pollutes the dict.  The mismatch detector must flag the extra
    key so the caller can decide whether to drop it or accept it.
    """
    noisy = {
        "G1_cuda_fix_verified": True,
        "G2_config_yaml_exists": True,
        "G3_yaml_md_valid": True,
        "G4_n_pre_flight_items": 25,
        "G5_n_open_violations": 0,
        "honest_framing_note": "with noise",
        "_wave": "Wf-320B",  # extra: sibling-artefact field
        "_ts": "2026-09-29",  # extra: timestamp pollution
    }
    report = detect_schema_mismatch(noisy, _SCHEMA_PAYLOAD)

    assert report["is_mismatch"] is True
    assert "_wave" in report["extra_keys"], (
        "Extra key _wave not enumerated: %r" % report["extra_keys"]
    )
    assert "_ts" in report["extra_keys"], (
        "Extra key _ts not enumerated: %r" % report["extra_keys"]
    )
    assert report["missing_keys"] == [], (
        "No missing keys expected; got %r" % report["missing_keys"]
    )
    assert report["reason"] in {
        FailureReason.EXTRA_KEYS.value,
        FailureReason.SCHEMA_MISMATCH.value,
    }, "Unexpected reason: %r" % report["reason"]


# ---------------------------------------------------------------------------
# 4. test_detect_type_mismatch
# ---------------------------------------------------------------------------


def test_detect_type_mismatch() -> None:
    """Value with the wrong JSON type -> report records (expected, actual).

    The Wf-302D generic prompt cannot help when the *types* are wrong
    (the agent populated all the fields but ``G4_n_pre_flight_items``
    came back as a string instead of an integer).  The mismatch
    detector must record the (expected, actual) tuple so the
    schema_coerce hint can tell the agent "please re-emit this field
    with the correct type".
    """
    wrong_types = {
        "G1_cuda_fix_verified": True,
        "G2_config_yaml_exists": True,
        "G3_yaml_md_valid": True,
        "G4_n_pre_flight_items": "25",  # WRONG: should be integer
        "G5_n_open_violations": "0",  # WRONG: should be integer
        "honest_framing_note": 42,  # WRONG: should be string
    }
    report = detect_schema_mismatch(wrong_types, _SCHEMA_PAYLOAD)

    assert report["is_mismatch"] is True
    type_mm = report["type_mismatches"]
    assert "G4_n_pre_flight_items" in type_mm, (
        "Type mismatch on G4_n_pre_flight_items not recorded: %r" % type_mm
    )
    assert type_mm["G4_n_pre_flight_items"] == ["integer", "str"], (
        "Wrong (expected, actual) tuple for G4_n_pre_flight_items: %r"
        % type_mm["G4_n_pre_flight_items"]
    )
    assert "honest_framing_note" in type_mm, (
        "Type mismatch on honest_framing_note not recorded: %r" % type_mm
    )
    assert report["reason"] in {
        FailureReason.TYPE_MISMATCH.value,
        FailureReason.SCHEMA_MISMATCH.value,
    }


# ===========================================================================
# Section 2 -- Retry-with-correction (4 tests)
# ===========================================================================


# ---------------------------------------------------------------------------
# 5. test_retry_with_correction_schema_coerce_hint
# ---------------------------------------------------------------------------


def test_retry_with_correction_schema_coerce_hint() -> None:
    """Partial payload triggers a retry with a schema_coerce hint.

    The schema_coerce hint must enumerate the specific gap (missing
    keys, extra keys, type mismatches) -- NOT just the generic
    "please retry" prompt.  This is the *core contract* of Wf-320B.
    """
    calls: List[Any] = []

    def fake_agent(*, correction_prompt=None):
        calls.append(correction_prompt)
        if correction_prompt is None:
            # Initial attempt: partial payload (missing G3 + G5).
            return json.dumps(
                {
                    "G1_cuda_fix_verified": True,
                    "G2_config_yaml_exists": True,
                    "G4_n_pre_flight_items": 25,
                    "honest_framing_note": "partial",
                }
            )
        # Retry attempt: full values.
        return json.dumps(_REAL_VALUES)

    helper = HarnessAutoFallback(schema_dict=_SCHEMA_PAYLOAD, max_retries=3)
    result, report = helper.retry_with_correction(fake_agent)

    assert (
        result is not None
    ), "Helper returned None even though the corrected retry returned values."
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES, (
        "Helper returned %r instead of the corrected real values." % parsed
    )

    # Find the schema_coerce hint among the recorded prompts.
    coerce_prompts = [p for p in calls if p is not None]
    assert coerce_prompts, "No correction prompt was used at all."
    last_prompt = coerce_prompts[-1]
    assert "MISSING KEYS" in last_prompt or "SCHEMA_MISMATCH" in last_prompt, (
        "Schema_coerce hint did not enumerate the missing keys.  Got: %r" % last_prompt
    )
    assert "G3_yaml_md_valid" in last_prompt or "G5_n_open_violations" in last_prompt, (
        "Schema_coerce hint did not name the missing field.  Got: %r" % last_prompt
    )

    # Report records the coerce hint used.
    assert report.coerce_hints_used, "Report did not record the schema_coerce hint."
    assert (
        report.recovered is True
    ), "Report.recovered should be True after successful correction."


# ---------------------------------------------------------------------------
# 6. test_retry_succeeds_after_schema_coerce
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_schema_coerce() -> None:
    """First attempt partial; second attempt after coerce succeeds.

    End-to-end happy path: the agent returns a partial dict, the
    helper drives the agent again with the schema_coerce hint, the
    second attempt returns the corrected values, and the helper
    surfaces them.
    """
    calls: List[Any] = []

    def fake_agent(*, correction_prompt=None):
        calls.append(correction_prompt)
        if correction_prompt is None:
            # Partial: missing one required field.
            return json.dumps(
                {
                    "G1_cuda_fix_verified": True,
                    "G2_config_yaml_exists": True,
                    "G3_yaml_md_valid": True,
                    # G4 + G5 missing
                    "honest_framing_note": "needs coerce",
                }
            )
        # After coerce -> corrected values.
        return json.dumps(_REAL_VALUES)

    helper = HarnessAutoFallback(schema_dict=_SCHEMA_PAYLOAD, max_retries=3)
    result, report = helper.retry_with_correction(fake_agent)

    assert result is not None
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES

    # The helper must have invoked the agent at least twice
    # (initial + at least one retry carrying the coerce hint).
    assert len(calls) >= 2, (
        "Helper invoked the agent %d times; expected at least 2." % len(calls)
    )
    assert report.recovered is True
    assert report.n_attempts >= 2
    assert report.budget_exhausted is False


# ---------------------------------------------------------------------------
# 7. test_retry_uses_wf302d_path_first_for_schema_shape
# ---------------------------------------------------------------------------


def test_retry_uses_wf302d_path_first_for_schema_shape() -> None:
    """Schema-shape payload -> Wf-302D generic prompt first, not coerce.

    When the agent returns the schema definition (plain schema-shape,
    not a partial / mismatched payload), the Wf-302D generic prompt is
    the right corrective message ("please return VALUES, not the
    schema").  The Wf-320B schema_coerce hint is misleading here
    (it would say "MISSING KEYS" when in fact the agent returned the
    whole schema definition by mistake).
    """
    calls: List[Any] = []

    def fake_agent(*, correction_prompt=None):
        calls.append(correction_prompt)
        if correction_prompt is None:
            # Return the schema-shape payload.
            return json.dumps(_SCHEMA_PAYLOAD)
        # After generic prompt -> return values.
        return json.dumps(_REAL_VALUES)

    helper = HarnessAutoFallback(schema_dict=_SCHEMA_PAYLOAD, max_retries=3)
    result, report = helper.retry_with_correction(fake_agent)

    assert result is not None
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES

    # The first correction prompt used must be the Wf-302D generic
    # prompt -- NOT a schema_coerce hint.
    correction_prompts = [p for p in calls if p is not None]
    assert correction_prompts, "No correction prompt was used."
    first_correction = correction_prompts[0]
    assert "SCHEMA definition" in first_correction or "SCHEMA" in first_correction, (
        "First correction prompt for schema-shape should be the Wf-302D "
        "generic prompt mentioning 'SCHEMA definition'.  Got: %r" % first_correction
    )
    assert "MISSING KEYS" not in first_correction, (
        "First correction prompt should NOT be a schema_coerce hint; got: %r"
        % first_correction
    )
    assert report.recovered is True


# ---------------------------------------------------------------------------
# 8. test_first_attempt_ok_no_retry
# ---------------------------------------------------------------------------


def test_first_attempt_ok_no_retry() -> None:
    """First attempt is good -> no second invocation.

    Efficiency guarantee: when the agent returns real values on the
    first try, the helper must short-circuit and not waste budget on
    unnecessary invocations.
    """
    calls: List[Any] = []

    def good_agent(*, correction_prompt=None):
        calls.append(correction_prompt)
        return json.dumps(_REAL_VALUES)

    helper = HarnessAutoFallback(schema_dict=_SCHEMA_PAYLOAD, max_retries=3)
    result, report = helper.retry_with_correction(good_agent)

    assert result is not None
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES
    # Exactly one invocation: the initial attempt.
    assert len(calls) == 1, (
        "First-attempt-OK path should invoke the agent exactly once; got %d."
        % len(calls)
    )
    assert calls[0] is None, (
        "First attempt should NOT receive a correction prompt; got %r." % calls[0]
    )
    assert report.recovered is True
    assert report.n_attempts == 1
    assert report.budget_exhausted is False


# ===========================================================================
# Section 3 -- Integration with Wf-302D (2 tests)
# ===========================================================================


# ---------------------------------------------------------------------------
# 9. test_wrap_wf302d_returns_harness_autofallback
# ---------------------------------------------------------------------------


def test_wrap_wf302d_returns_harness_autofallback() -> None:
    """``wrap_wf302d_with_autofallback`` returns a configured instance.

    The integration point promised by the Wf-320B brief: callers that
    today use ``run_agent_with_retry`` can switch to the wrapped
    helper with one line of source change and gain the mismatch layer.
    """
    helper = wrap_wf302d_with_autofallback(schema_dict=_SCHEMA_PAYLOAD)
    assert isinstance(helper, HarnessAutoFallback), (
        "wrap_wf302d_with_autofallback did not return a HarnessAutoFallback; "
        "got %r." % type(helper)
    )
    assert helper.schema_dict == _SCHEMA_PAYLOAD
    assert helper.max_retries == 3, (
        "Default max_retries should be 3; got %r." % helper.max_retries
    )
    assert helper.fallback_strategy == FallbackStrategy.SIBLING_RECONSTRUCTION, (
        "Default fallback_strategy should be SIBLING_RECONSTRUCTION; got %r."
        % helper.fallback_strategy
    )

    # Custom config also flows through.
    custom = wrap_wf302d_with_autofallback(
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=5,
        fallback_strategy=FallbackStrategy.EMPTY_DEFAULT,
    )
    assert custom.max_retries == 5
    assert custom.fallback_strategy == FallbackStrategy.EMPTY_DEFAULT


# ---------------------------------------------------------------------------
# 10. test_wrapped_helper_uses_existing_wf302d_prompt_for_schema_shape
# ---------------------------------------------------------------------------


def test_wrapped_helper_uses_existing_wf302d_prompt_for_schema_shape() -> None:
    """Wrapped helper re-uses the Wf-302D prompt for plain schema-shape.

    Confirms that the integration point does NOT regress the Wf-302D
    behaviour -- the generic prompt ("your previous output returned
    the JSON SCHEMA definition...") must still be the first prompt
    used for schema-shape payloads.
    """
    calls: List[Any] = []

    def fake_agent(*, correction_prompt=None):
        calls.append(correction_prompt)
        if correction_prompt is None:
            return json.dumps(_SCHEMA_PAYLOAD)
        return json.dumps(_REAL_VALUES)

    helper = wrap_wf302d_with_autofallback(schema_dict=_SCHEMA_PAYLOAD)
    result, report = helper.retry_with_correction(fake_agent)

    assert result is not None
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES

    # First correction prompt must be the Wf-302D generic prompt.
    correction_prompts = [p for p in calls if p is not None]
    assert correction_prompts
    first = correction_prompts[0]
    assert "JSON SCHEMA definition" in first or "SCHEMA" in first, (
        "First correction prompt should be the Wf-302D generic prompt; got: %r" % first
    )
    assert report.recovered is True


# ===========================================================================
# Section 4 -- Edge cases (2 tests)
# ===========================================================================


# ---------------------------------------------------------------------------
# 11. test_max_retries_exhausted_applies_fallback_strategy
# ---------------------------------------------------------------------------


def test_max_retries_exhausted_applies_fallback_strategy() -> None:
    """Every attempt fails -> fallback strategy applied.

    The bounded-retry guarantee: when every attempt fails (whether
    schema-shape or mismatch), the helper applies the configured
    fallback strategy and records ``budget_exhausted=True`` on the
    report.
    """

    def always_fails(*, correction_prompt=None):
        # Always return the schema-shape payload -> Wf-302D path keeps failing.
        return json.dumps(_SCHEMA_PAYLOAD)

    helper = HarnessAutoFallback(
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=2,
        fallback_strategy=FallbackStrategy.EMPTY_DEFAULT,
    )
    result, report = helper.retry_with_correction(always_fails)

    assert result == {}, (
        "EMPTY_DEFAULT fallback should return {} on exhaustion; got %r." % result
    )
    assert (
        report.budget_exhausted is True
    ), "Report.budget_exhausted should be True when retries are exhausted."
    assert report.fallback_strategy == FallbackStrategy.EMPTY_DEFAULT
    assert report.recovered is False
    assert report.n_attempts >= 1

    # SIBLING_RECONSTRUCTION fallback returns a tagged dict (not None).
    helper_sib = HarnessAutoFallback(
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=2,
        fallback_strategy=FallbackStrategy.SIBLING_RECONSTRUCTION,
    )
    result_sib, report_sib = helper_sib.retry_with_correction(always_fails)
    assert isinstance(result_sib, dict)
    assert result_sib.get("_schema_detected") is True
    assert result_sib.get("_partial") is True
    assert report_sib.fallback_strategy == FallbackStrategy.SIBLING_RECONSTRUCTION


# ---------------------------------------------------------------------------
# 12. test_max_retries_zero_still_applies_fallback
# ---------------------------------------------------------------------------


def test_max_retries_zero_still_applies_fallback() -> None:
    """``max_retries=0`` -> fallback applied even on the initial attempt.

    Edge case: when the caller explicitly sets ``max_retries=0`` (no
    retries at all), the helper must still apply the fallback strategy
    rather than loop forever.  This guards against a future regression
    where the helper accidentally skips the fallback path on
    ``max_retries=0``.
    """

    def always_returns_partial(*, correction_prompt=None):
        # Always return a partial dict -- never matches the schema.
        return json.dumps(
            {
                "G1_cuda_fix_verified": True,
                # everything else missing
            }
        )

    helper = HarnessAutoFallback(
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=0,
        fallback_strategy=FallbackStrategy.EMPTY_DEFAULT,
    )
    result, report = helper.retry_with_correction(always_returns_partial)

    # With max_retries=0 + EMPTY_DEFAULT, the result should be the
    # fallback (an empty dict), not the partial dict.
    assert result == {}, (
        "max_retries=0 + EMPTY_DEFAULT must return {}; got %r." % result
    )
    assert report.budget_exhausted is True
    assert report.recovered is False

    # And the fallback strategy is recorded on the report.
    assert report.fallback_strategy == FallbackStrategy.EMPTY_DEFAULT


# ---------------------------------------------------------------------------
# Bonus: coerce() helper -- best-effort shape coercion (not in the
# count of 12 but covers the public surface promised by the brief).
# ---------------------------------------------------------------------------


def test_coerce_drops_extras_and_wrong_types() -> None:
    """HarnessAutoFallback.coerce() drops extras + wrong-type values.

    Pin the best-effort coercion contract: extras are dropped, wrong
    types are dropped, missing keys stay missing (we never fabricate).
    """
    helper = HarnessAutoFallback(schema_dict=_SCHEMA_PAYLOAD)
    noisy = {
        "G1_cuda_fix_verified": True,
        "G2_config_yaml_exists": True,
        "G3_yaml_md_valid": True,
        "G4_n_pre_flight_items": "25",  # wrong type -> dropped
        "G5_n_open_violations": 0,
        "honest_framing_note": "kept",
        "_wave": "Wf-320B",  # extra -> dropped
    }
    coerced = helper.coerce(noisy)

    assert "_wave" not in coerced, "Extras should be dropped; got %r." % coerced
    assert "G4_n_pre_flight_items" not in coerced, (
        "Wrong-type values should be dropped; got %r." % coerced
    )
    assert coerced.get("G5_n_open_violations") == 0
    assert coerced.get("honest_framing_note") == "kept"
    assert coerced.get("G1_cuda_fix_verified") is True
