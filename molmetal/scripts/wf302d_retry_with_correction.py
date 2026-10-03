"""Wf-302D: Retry-with-correction helper.

================================================================
Closes Wf-298E critical remaining #4.

Per Wf-298E long-term-mitigation plan, when an upstream agent tasked
with returning complex structured JSON via the ``StructuredOutput`` tool
returns the JSON **schema definition** instead of the values, the
detection-and-retry loop must:

  1. Detect the schema shape (delegated to
     :func:`wf301e_harness_detector.detect_schema_shape`).
  2. Retry the agent call with an **explicit correction prompt** that
     tells the agent "the previous output was the schema, please return
     VALUES matching the schema".
  3. Cap the retry budget (``max_retries``) so we never loop forever;
     return ``None`` when the budget is exhausted so callers can fall
     back to the same sibling-data reconstruction path Wf-285E used
     (per ``wf291c_pipeline_root_cause.md`` §5.2).

The retry helper is **pure-Python**, has **no third-party deps**, and
is safe to call from any harness layer (workflow orchestrator, CI
synthesiser, journal writer).

Public API
----------

* :func:`run_agent_with_retry` -- the canonical retry loop.
* :func:`build_correction_prompt` -- the explicit correction prompt
  template (exported so callers can inspect / customise it).

Usage
-----

::

    from wf302d_retry_with_correction import run_agent_with_retry
    from wf301e_harness_detector import detect_schema_shape

    def my_agent_callable(correction_prompt=None):
        # If correction_prompt is given, prepend it to the system message.
        return some_agent.invoke(...)

    result = run_agent_with_retry(
        agent_callable=my_agent_callable,
        schema_dict=EXPECTED_SCHEMA,
        max_retries=2,
    )
    if result is None:
        # Budget exhausted -- caller falls back to sibling-data reconstruction.
        ...

Five rounds of evidence (Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 /
Wf-285 / Wf-291) confirm the regression is in the schema-extraction
layer and that a retry loop with explicit correction is the right
**second-line defence** behind the Wf-301E wrapper detector.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

# Re-use the Wf-301E detector so the two helpers cannot drift out of
# sync on what counts as "schema shape".
from wf301e_harness_detector import detect_schema_shape

# ---------------------------------------------------------------------------
# Logging -- informational only; callers may silence via ``logging.disable``.
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("wf302d_retry_with_correction")

# ---------------------------------------------------------------------------
# Default correction prompt template.
#
# Anchored on the Wf-285E (per root_cause.md §5.4) verbatim corrective
# phrasing: "the previous output returned the schema definition, please
# return VALUES matching the schema".  We expose this as a function so
# callers can customise the wording for their own agent stack without
# having to fork the retry loop.
# ---------------------------------------------------------------------------

DEFAULT_CORRECTION_PROMPT = (
    "Your previous output returned the JSON SCHEMA definition (with "
    "keys like 'type', 'properties', 'required') rather than actual "
    "VALUES matching that schema. Please re-emit a JSON object "
    "containing the VALUES for every field listed in the schema -- "
    "for example {\"G1_score\": 25, \"G2_label\": \"passed\"} -- not "
    "the schema definition itself."
)


def build_correction_prompt(
    schema_dict: Optional[Dict[str, Any]] = None,
    field_hint: bool = True,
) -> str:
    """Build the explicit correction prompt.

    Parameters
    ----------
    schema_dict:
        The schema the agent was originally tasked to fill.  When given
        AND ``field_hint`` is True, the field names are appended to the
        prompt so the agent knows which keys to populate.
    field_hint:
        When True (default), append the schema's top-level property
        names to the correction prompt.  When False, omit the field list
        (useful when the schema is too large to enumerate).

    Returns
    -------
    str
        The full correction prompt text, ready to be passed back into
        the agent via ``correction_prompt=<this>``.
    """
    base = DEFAULT_CORRECTION_PROMPT

    if field_hint and isinstance(schema_dict, dict):
        properties = schema_dict.get("properties")
        if isinstance(properties, dict) and properties:
            field_names = sorted(properties.keys())
            base = (
                base
                + "  The expected fields are: "
                + ", ".join(field_names)
                + "."
            )
    return base


# ---------------------------------------------------------------------------
# The retry loop.
# ---------------------------------------------------------------------------

# Type alias: an ``agent_callable`` takes an optional ``correction_prompt``
# keyword argument and returns whatever the agent produced -- usually a
# string (raw JSON / text) but may also be a dict.
AgentCallable = Callable[..., Any]


def run_agent_with_retry(
    agent_callable: AgentCallable,
    schema_dict: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> Any:
    """Run *agent_callable* and retry on schema-shape regressions.

    Behaviour
    ---------
    1. Invoke ``agent_callable()`` with no correction prompt.
    2. If the result is **not** schema-shape (per
       :func:`wf301e_harness_detector.detect_schema_shape`), return it
       immediately.
    3. Otherwise, loop up to ``max_retries`` times: invoke
       ``agent_callable(correction_prompt=<explicit prompt>)`` and
       return as soon as a non-schema-shape result is obtained.
    4. If the budget is exhausted, return ``None`` so the caller can
       fall back to the sibling-data reconstruction path (Wf-285E
       precedent).

    Parameters
    ----------
    agent_callable:
        A zero-argument (or ``correction_prompt`` keyword-only)
        callable that invokes the agent and returns its raw output.
        Must accept ``correction_prompt`` as a keyword argument; the
        helper will only pass it on retry attempts.
    schema_dict:
        Optional schema.  When provided, the schema is appended to the
        correction prompt as a field-name hint so the agent knows
        exactly which keys to populate.
    max_retries:
        Maximum number of **retry** attempts (NOT counting the initial
        attempt).  Default ``2`` -- i.e. up to 3 total invocations.

    Returns
    -------
    Any
        The first non-schema-shape result from ``agent_callable``, or
        ``None`` when every attempt returned schema-shape.

    Notes
    -----
    * This helper never raises on a schema-shape result; it returns
      ``None`` so callers can decide their own fallback policy.
    * The detection is delegated to :func:`detect_schema_shape` -- the
      same detector that powers the Wf-301E wrapper layer.  There is
      therefore a **single source of truth** for what counts as
      "schema shape" across the harness.
    * The function is sync; callers that want async retry semantics
      should wrap it in ``asyncio.to_thread``.
    """
    if max_retries < 0:
        raise ValueError(
            "max_retries must be >= 0 (got %r).  Use 0 for "
            "single-attempt-no-retry semantics." % (max_retries,)
        )

    # Attempt 0: the initial call -- no correction prompt.
    _LOG.debug("Wf-302D: initial attempt (no correction prompt).")
    result = agent_callable()
    if not _is_schema_shape(result):
        _LOG.debug("Wf-302D: initial attempt succeeded (non-schema).")
        return result

    # The initial attempt returned schema-shape; enter the retry loop.
    for attempt in range(1, max_retries + 1):
        correction_prompt = build_correction_prompt(
            schema_dict=schema_dict, field_hint=True
        )
        _LOG.debug(
            "Wf-302D: retry attempt %d / %d with correction prompt.",
            attempt,
            max_retries,
        )
        result = agent_callable(correction_prompt=correction_prompt)
        if not _is_schema_shape(result):
            _LOG.debug(
                "Wf-302D: retry attempt %d succeeded (non-schema).",
                attempt,
            )
            return result

    _LOG.warning(
        "Wf-302D: exhausted %d retries; returning None.  Caller "
        "should fall back to sibling-data reconstruction (Wf-285E "
        "precedent, wf291c_pipeline_root_cause.md §5.2).",
        max_retries,
    )
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_schema_shape(result: Any) -> bool:
    """Return True iff *result* is a schema-shape payload.

    Mirrors :func:`wf301e_harness_detector.detect_schema_shape` but
    accepts the wider set of types that agents may return (dict, str,
    None, scalars).

    Parameters
    ----------
    result:
        The raw value returned by the agent.

    Returns
    -------
    bool
        ``True`` when the value is a JSON Schema definition,
        ``False`` otherwise (real values, plain text, ``None``, etc.).
    """
    if result is None:
        return False

    # Path 1: already a string -- feed straight into the detector.
    if isinstance(result, str):
        return detect_schema_shape(result)

    # Path 2: dict -- serialise to JSON and detect.  Real data
    # serialisation always produces JSON with non-empty fields;
    # schemas always have type=object + properties=...
    if isinstance(result, dict):
        import json as _json

        try:
            serialised = _json.dumps(result)
        except (TypeError, ValueError):
            return False
        return detect_schema_shape(serialised)

    # Path 3: scalar (int / float / bool / list) -- never schema-shape.
    return False


__all__ = [
    "run_agent_with_retry",
    "build_correction_prompt",
    "DEFAULT_CORRECTION_PROMPT",
]
