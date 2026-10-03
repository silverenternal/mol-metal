"""Tests for the Wf-302D retry-with-correction helper.

================================================================
Closes Wf-298E critical remaining #4.

These are the pytest mirrors of
:file:`molmetal/scripts/wf302d_retry_with_correction.py`.  Three named
tests pin the retry contract:

  1. test_retry_succeeds_after_correction
        The first attempt returns schema-shape; the retry with the
        explicit correction prompt returns real values.  The helper
        must return the values from the retry, NOT the schema from the
        initial attempt, and NOT ``None``.

  2. test_retry_exhausts_after_max
        Every attempt returns schema-shape.  The helper must respect
        ``max_retries`` and return ``None`` so the caller can fall back
        to the Wf-285E sibling-data reconstruction path.

  3. test_first_attempt_succeeds_no_retry
        The initial attempt returns real values.  The helper must
        return those values immediately and must NOT invoke the agent
        a second time (verified via a call counter).

All tests are CPU-only, deterministic, and complete in < 5 seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap -- mirror the script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from wf302d_retry_with_correction import (  # noqa: E402
    DEFAULT_CORRECTION_PROMPT,
    build_correction_prompt,
    run_agent_with_retry,
)


# ---------------------------------------------------------------------------
# Fixtures -- canonical regression payload (Wf-270A verbatim).
# ---------------------------------------------------------------------------

_SCHEMA_PAYLOAD = {
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

_REAL_VALUES = {
    "G1_cuda_fix_verified": True,
    "G2_config_yaml_exists": True,
    "G3_yaml_md_valid": True,
    "G4_n_pre_flight_items": 25,
    "G5_n_open_violations": 0,
    "honest_framing_note": "MEASURED, single seed, single epoch",
}


# ---------------------------------------------------------------------------
# 1. test_retry_succeeds_after_correction -- the happy retry path.
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_correction() -> None:
    """First attempt schema-shape; retry returns values.

    This is the **core contract** of Wf-302D: when the agent initially
    returns the schema definition (the Wf-270/271/272/274/279/285/291
    regression class), the retry-with-explicit-correction-prompt path
    must surface the real values.

    To make the success deterministic without mocking out the agent
    stack, we record the correction prompts that were passed to the
    callable so we can assert on:

      * the initial attempt received NO correction prompt,
      * the retry attempt received a prompt that **mentions the schema
        fields** (so the agent knows which keys to populate),
      * the returned value is the real-values dict (NOT the schema),
      * the agent was invoked exactly twice (initial + 1 retry).
    """
    seen_correction_prompts: List[Any] = []
    seen_call_kwargs: List[dict] = []

    def fake_agent(*, correction_prompt=None):
        seen_correction_prompts.append(correction_prompt)
        seen_call_kwargs.append({"correction_prompt": correction_prompt})
        if correction_prompt is None:
            # Initial attempt -- return the regression-case schema.
            return json.dumps(_SCHEMA_PAYLOAD)
        # Retry attempt -- return the real values.
        return json.dumps(_REAL_VALUES)

    result = run_agent_with_retry(
        agent_callable=fake_agent,
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=2,
    )

    # 1. Result must be the real values, NOT the schema.
    assert result is not None, (
        "Retry helper returned None even though the retry succeeded."
    )
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES, (
        "Retry helper returned the schema-shape payload instead of "
        "the real values.  Got: %r" % (parsed,)
    )

    # 2. Initial attempt: no correction prompt.
    assert seen_correction_prompts[0] is None, (
        "Initial attempt must NOT receive a correction prompt.  Got: %r"
        % (seen_correction_prompts[0],)
    )

    # 3. Retry attempt: prompt must mention the schema fields.
    assert seen_correction_prompts[1] is not None, (
        "Retry attempt did not receive a correction prompt."
    )
    retry_prompt = seen_correction_prompts[1]
    for field in ("G1_cuda_fix_verified", "G4_n_pre_flight_items"):
        assert field in retry_prompt, (
            "Correction prompt missing schema field %r.  Got: %r"
            % (field, retry_prompt)
        )

    # 4. Exactly 2 invocations: 1 initial + 1 retry.
    assert len(seen_correction_prompts) == 2, (
        "Expected exactly 2 agent invocations (initial + 1 retry), "
        "got %d." % (len(seen_correction_prompts),)
    )


# ---------------------------------------------------------------------------
# 2. test_retry_exhausts_after_max -- the budget-exhaustion path.
# ---------------------------------------------------------------------------


def test_retry_exhausts_after_max() -> None:
    """Every attempt returns schema-shape; helper must return None.

    The helper MUST honour ``max_retries`` and return ``None`` so the
    caller can fall back to the Wf-285E sibling-data reconstruction
    path (per ``wf291c_pipeline_root_cause.md`` §5.2).
    """
    call_count = 0

    def always_returns_schema(*, correction_prompt=None):
        nonlocal call_count
        call_count += 1
        # Always return the regression-case schema -- never real values.
        return json.dumps(_SCHEMA_PAYLOAD)

    result = run_agent_with_retry(
        agent_callable=always_returns_schema,
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=2,
    )

    # 1. Result must be None (budget exhausted).
    assert result is None, (
        "Helper returned %r even though every attempt returned "
        "schema-shape.  Caller would not be able to fall back."
        % (result,)
    )

    # 2. Total invocations = 1 initial + max_retries retries = 3.
    assert call_count == 3, (
        "Expected 3 total invocations (1 initial + 2 retries), "
        "got %d." % (call_count,)
    )

    # 3. Sanity: max_retries=0 must give exactly 1 invocation (initial only).
    single_call_count = 0

    def single_schema(*, correction_prompt=None):
        nonlocal single_call_count
        single_call_count += 1
        return json.dumps(_SCHEMA_PAYLOAD)

    result_zero = run_agent_with_retry(
        agent_callable=single_schema,
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=0,
    )
    assert result_zero is None, (
        "max_retries=0 + schema-shape must still return None."
    )
    assert single_call_count == 1, (
        "max_retries=0 must yield exactly 1 invocation (initial), "
        "got %d." % (single_call_count,)
    )

    # 4. Sanity: a negative max_retries is rejected up-front.
    with pytest.raises(ValueError):
        run_agent_with_retry(
            agent_callable=always_returns_schema,
            schema_dict=_SCHEMA_PAYLOAD,
            max_retries=-1,
        )


# ---------------------------------------------------------------------------
# 3. test_first_attempt_succeeds_no_retry -- the no-op path.
# ---------------------------------------------------------------------------


def test_first_attempt_succeeds_no_retry() -> None:
    """Initial attempt is good; helper must not invoke the agent twice.

    This is the **efficiency guarantee**: when the agent returns real
    values on the first try, the retry helper must short-circuit and
    not spend budget on unnecessary invocations.
    """
    call_count = 0
    seen_kwargs: List[dict] = []

    def good_agent(*, correction_prompt=None):
        nonlocal call_count
        call_count += 1
        seen_kwargs.append({"correction_prompt": correction_prompt})
        return json.dumps(_REAL_VALUES)

    result = run_agent_with_retry(
        agent_callable=good_agent,
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=2,
    )

    # 1. Result must be the real values verbatim.
    parsed = json.loads(result)
    assert parsed == _REAL_VALUES, (
        "First-attempt-succeeds path returned wrong payload: %r" % (parsed,)
    )

    # 2. Helper must NOT invoke the agent a second time.
    assert call_count == 1, (
        "First-attempt-succeeds path should invoke the agent exactly "
        "once, got %d." % (call_count,)
    )

    # 3. The single invocation must have had NO correction prompt.
    assert seen_kwargs[0]["correction_prompt"] is None, (
        "First attempt must NOT receive a correction prompt."
    )


# ---------------------------------------------------------------------------
# Bonus: dict-returning agents (the regression case sometimes returns
# the schema as a dict, not a JSON string).
# ---------------------------------------------------------------------------


def test_dict_returning_agent_schema_then_values() -> None:
    """Agents that return dicts (not JSON strings) are also handled.

    The Wf-291 pipeline observed both str-returning agents and
    dict-returning agents.  The retry helper must serialise dicts
    internally and feed them through :func:`detect_schema_shape`.
    """
    attempts: List[Any] = []

    def dict_agent(*, correction_prompt=None):
        attempts.append(correction_prompt)
        if correction_prompt is None:
            return _SCHEMA_PAYLOAD  # dict form of the schema
        return _REAL_VALUES        # dict form of the real values

    result = run_agent_with_retry(
        agent_callable=dict_agent,
        schema_dict=_SCHEMA_PAYLOAD,
        max_retries=1,
    )

    assert result == _REAL_VALUES, (
        "Dict-returning agent's real values were not surfaced.  Got: %r"
        % (result,)
    )
    assert len(attempts) == 2


def test_correction_prompt_omits_field_hint_when_disabled() -> None:
    """``build_correction_prompt(field_hint=False)`` must omit the field list.

    Useful when the schema is large and a verbatim field enumeration
    would blow the agent's context window.
    """
    base = build_correction_prompt(_SCHEMA_PAYLOAD, field_hint=False)
    assert base == DEFAULT_CORRECTION_PROMPT
    for field in ("G1_cuda_fix_verified", "G4_n_pre_flight_items"):
        assert field not in base, (
            "field_hint=False should omit %r; got: %r" % (field, base)
        )
