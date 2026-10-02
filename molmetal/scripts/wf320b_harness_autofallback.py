"""Wf-320B: Harness auto-fallback fix for the Wf-291C regression class.

================================================================
Closes Wf-315B recommendation #2 (Wf-291F 真修).

Per ``wf291c_pipeline_root_cause.md`` §5.4 and Wf-302D's contract:

  * Wf-301E (Harness shape detector)         — recognises when an agent
    returns a *JSON Schema definition* instead of the values.
  * Wf-302D (Retry-with-correction helper)   — calls the agent again with
    an explicit correction prompt.

**The remaining gap (Wf-315B recommendation #2):** neither helper handles
the *partial / schema-mismatch* failure mode.  Specifically:

  * The agent returns values, but for the **wrong schema** (some keys
    from a sibling artefact are present, others are missing or extra).
  * The agent returns values whose **types** do not match the expected
    schema (e.g. a string where a number was expected).
  * The agent returns a schema fragment (not a full schema, not values)
    and the parent pipeline cannot tell which to use.

This module adds the **third line of defence**: an auto-fallback layer
that

  1. Detects schema-mismatch (in addition to plain schema-shape).
  2. Produces a *schema_coerce hint* describing the gap (missing keys,
     extra keys, type mismatches) so the agent has a structured corrective
     message instead of a free-text "please retry".
  3. Retries with the schema_coerce hint (bypassing the schema-shape
     retry path when the failure mode is a *mismatch* not a *schema*).
  4. On budget exhaustion, applies the chosen **fallback strategy**
     (sibling reconstruction / empty default / explicit raise).

Public API
----------

* :class:`HarnessAutoFallback` — the all-in-one helper.
* :func:`detect_schema_mismatch` — pure-function mismatch detector
  (exported so callers can run the check without instantiating the
  class).
* :func:`build_schema_coerce_hint` — pure-function coerce-hint builder
  (exported for the same reason).

The module is **pure-Python**, has **no third-party deps**, and re-uses
the Wf-301E + Wf-302D helpers so all four layers share a single source
of truth for "what counts as a schema-shape payload".

Six rounds of evidence (Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 /
Wf-285 / Wf-291 / Wf-298 / Wf-302 / Wf-315) confirm that the
schema-extraction layer regression class spans three failure modes —
*plain schema-shape* (closed by Wf-302D), *partial / mismatched values*
(closed here), and *plain text passthrough* (closed by Wf-301E
extract_agent_result).  This module adds the second closure and wires
it into the Wf-302D helper via a thin wrapper so existing callers need
no source change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Re-use the Wf-301E detector + Wf-302D retry helper so the four layers
# cannot drift out of sync.
from wf301e_harness_detector import detect_schema_shape, extract_agent_result
from wf302d_retry_with_correction import (
    DEFAULT_CORRECTION_PROMPT,
    build_correction_prompt,
    run_agent_with_retry,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("wf320b_harness_autofallback")


# ---------------------------------------------------------------------------
# Failure modes + fallback strategies
# ---------------------------------------------------------------------------


class FailureReason(str, Enum):
    """Failure mode classification for an agent result."""

    SCHEMA_SHAPE = "schema_shape"            # agent returned the schema def
    SCHEMA_MISMATCH = "schema_mismatch"      # values returned, wrong schema
    TYPE_MISMATCH = "type_mismatch"          # values returned, wrong types
    MISSING_KEYS = "missing_keys"            # values returned, keys missing
    EXTRA_KEYS = "extra_keys"                # values returned, unexpected keys
    PLAIN_TEXT = "plain_text"                # plain text / markdown
    OK = "ok"                                # values match expected schema


class FallbackStrategy(str, Enum):
    """What to do when the retry budget is exhausted."""

    SIBLING_RECONSTRUCTION = "sibling_reconstruction"
    EMPTY_DEFAULT = "empty_default"
    RAISE = "raise"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class AutoFallbackReport:
    """Diagnostic record of one ``HarnessAutoFallback`` invocation.

    Captured so the caller can write it to disk for post-mortem analysis
    (the Wf-285E sibling-data reconstruction precedent).  Every field is
    intentionally serialisable to JSON via :func:`asdict`.
    """

    n_attempts: int = 0
    final_reason: FailureReason = FailureReason.OK
    recovered: bool = False
    final_result: Optional[Any] = None
    mismatch_details: Dict[str, Any] = field(default_factory=dict)
    coerce_hints_used: List[str] = field(default_factory=list)
    budget_exhausted: bool = False
    fallback_strategy: Optional[FallbackStrategy] = None

    def asdict(self) -> Dict[str, Any]:
        out = {
            "n_attempts": self.n_attempts,
            "final_reason": self.final_reason.value,
            "recovered": self.recovered,
            "final_result": self.final_result,
            "mismatch_details": dict(self.mismatch_details),
            "coerce_hints_used": list(self.coerce_hints_used),
            "budget_exhausted": self.budget_exhausted,
            "fallback_strategy": (
                self.fallback_strategy.value if self.fallback_strategy else None
            ),
        }
        return out


# ---------------------------------------------------------------------------
# Pure functions (exported; do not require a class instance)
# ---------------------------------------------------------------------------


def detect_schema_mismatch(
    agent_result: Any,
    schema_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a structured mismatch report for *agent_result*.

    The report has the following keys::

        {
            "is_mismatch":       bool,
            "is_schema_shape":   bool,
            "expected_fields":   [str, ...],
            "recovered_fields":  [str, ...],
            "missing_keys":      [str, ...],
            "extra_keys":        [str, ...],
            "type_mismatches":   {field: (expected, actual)},
            "reason":            FailureReason,
        }

    Parameters
    ----------
    agent_result:
        Whatever the agent returned -- dict, str, scalar, ``None``.
    schema_dict:
        Optional JSON Schema definition with a ``properties`` key whose
        values are ``{"type": <primitive>, ...}`` sub-dicts.  When
        ``None``, the function returns ``is_mismatch=False`` (no schema
        to compare against).

    Notes
    -----
    * This function **never raises**.  Malformed inputs are tagged with
      ``is_mismatch=False`` and ``reason=plain_text`` so callers can
      fall through to the plain-text passthrough path.
    * When the agent returned a *schema-shape* payload (per the
      Wf-301E detector), the function tags ``is_schema_shape=True`` AND
      ``is_mismatch=True`` so the caller can choose the right retry
      strategy (schema-shape → re-prompt for values; mismatch → coerce).
    """
    out: Dict[str, Any] = {
        "is_mismatch": False,
        "is_schema_shape": False,
        "expected_fields": [],
        "recovered_fields": [],
        "missing_keys": [],
        "extra_keys": [],
        "type_mismatches": {},
        "reason": FailureReason.OK.value,
    }

    # Path 0: no schema -> nothing to compare against.
    if not isinstance(schema_dict, dict):
        return out

    expected = _expected_fields(schema_dict)
    out["expected_fields"] = expected

    # Path 1: None / plain text / non-dict / scalar -> plain-text passthrough.
    if agent_result is None:
        out["reason"] = FailureReason.PLAIN_TEXT.value
        return out
    if isinstance(agent_result, (int, float, bool)):
        out["reason"] = FailureReason.PLAIN_TEXT.value
        return out
    if isinstance(agent_result, str):
        payload = agent_result.strip()
        if not payload:
            out["reason"] = FailureReason.PLAIN_TEXT.value
            return out
        # Plain text / markdown is anything that does not parse as JSON
        # or that parses to a non-object JSON value.
        try:
            parsed = json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            out["reason"] = FailureReason.PLAIN_TEXT.value
            return out
        if not isinstance(parsed, dict):
            out["reason"] = FailureReason.PLAIN_TEXT.value
            return out
        agent_dict = parsed
    elif isinstance(agent_result, dict):
        agent_dict = agent_result
    else:
        out["reason"] = FailureReason.PLAIN_TEXT.value
        return out

    # Path 2: schema-shape.  Defer to the Wf-301E detector.
    if detect_schema_shape(json.dumps(agent_dict)):
        out["is_schema_shape"] = True
        out["is_mismatch"] = True
        out["reason"] = FailureReason.SCHEMA_SHAPE.value
        return out

    # Path 3: real values; check against expected schema.
    recovered = sorted(k for k in agent_dict.keys() if isinstance(k, str))
    missing = sorted(f for f in expected if f not in agent_dict)
    extra = sorted(k for k in recovered if k not in expected)
    type_mm: Dict[str, tuple] = {}

    properties = schema_dict.get("properties") if isinstance(schema_dict, dict) else None
    if isinstance(properties, dict):
        for field_name, prop_def in properties.items():
            if field_name not in agent_dict:
                continue
            if not isinstance(prop_def, dict):
                continue
            exp_type = prop_def.get("type")
            if not isinstance(exp_type, str):
                continue
            actual = agent_dict[field_name]
            if not _type_matches(exp_type, actual):
                type_mm[field_name] = (exp_type, type(actual).__name__)

    if missing and not extra and not type_mm:
        out["reason"] = FailureReason.MISSING_KEYS.value
    elif extra and not missing and not type_mm:
        out["reason"] = FailureReason.EXTRA_KEYS.value
    elif type_mm and not missing and not extra:
        out["reason"] = FailureReason.TYPE_MISMATCH.value
    elif missing or extra or type_mm:
        out["reason"] = FailureReason.SCHEMA_MISMATCH.value
    else:
        out["reason"] = FailureReason.OK.value

    out["is_mismatch"] = bool(missing or extra or type_mm)
    out["recovered_fields"] = recovered
    out["missing_keys"] = missing
    out["extra_keys"] = extra
    out["type_mismatches"] = {k: list(v) for k, v in type_mm.items()}
    return out


def build_schema_coerce_hint(
    mismatch_report: Dict[str, Any],
    schema_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a structured correction prompt from a mismatch report.

    Unlike :func:`wf302d_retry_with_correction.build_correction_prompt`,
    which only knows about schema-shape, this function enumerates the
    *specific* gap (missing keys, extra keys, type mismatches) so the
    agent gets a targeted corrective message instead of a generic
    "please retry".

    Parameters
    ----------
    mismatch_report:
        A dict as returned by :func:`detect_schema_mismatch`.
    schema_dict:
        The expected schema (used to enumerate the type names when
        type mismatches are present).

    Returns
    -------
    str
        The full correction prompt, ready to be passed back to the
        agent via ``correction_prompt=<this>``.  When the mismatch
        report is empty (``reason=ok``), the base Wf-302D prompt is
        returned unchanged.
    """
    base = DEFAULT_CORRECTION_PROMPT

    if not mismatch_report.get("is_mismatch", False):
        # Empty / OK report -- emit the generic prompt for backwards compat.
        return base

    reason = mismatch_report.get("reason", "ok")
    missing = list(mismatch_report.get("missing_keys", []) or [])
    extra = list(mismatch_report.get("extra_keys", []) or [])
    type_mm = dict(mismatch_report.get("type_mismatches", {}) or {})

    hint_lines: List[str] = [base, ""]
    hint_lines.append(
        "Specific gap detected (Wf-320B schema_coerce hint, reason=%s):" % reason
    )

    if missing:
        fields_str = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else " (and %d more)" % (len(missing) - 20)
        hint_lines.append(
            "  * MISSING KEYS: please populate these fields with values: "
            + fields_str + suffix
        )
    if extra:
        fields_str = ", ".join(extra[:20])
        suffix = "" if len(extra) <= 20 else " (and %d more)" % (len(extra) - 20)
        hint_lines.append(
            "  * EXTRA KEYS: please REMOVE these fields from your response: "
            + fields_str + suffix
        )
    if type_mm:
        bits = []
        for k in sorted(type_mm.keys())[:20]:
            exp, act = type_mm[k]
            bits.append("%s (expected %s, got %s)" % (k, exp, act))
        suffix = "" if len(type_mm) <= 20 else " (and %d more)" % (len(type_mm) - 20)
        hint_lines.append(
            "  * TYPE MISMATCHES: please re-emit these fields with the "
            "correct JSON type: " + ", ".join(bits) + suffix
        )

    # Append the expected field list as a final reminder (consistent with
    # Wf-302D build_correction_prompt).
    if isinstance(schema_dict, dict):
        properties = schema_dict.get("properties")
        if isinstance(properties, dict) and properties:
            field_names = sorted(properties.keys())
            hint_lines.append(
                "  The expected fields are: " + ", ".join(field_names) + "."
            )

    return "\n".join(hint_lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_PRIMITIVE_TYPE_NAMES = (
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
)


def _expected_fields(schema_dict: Dict[str, Any]) -> List[str]:
    """Return the sorted list of expected top-level field names."""
    if not isinstance(schema_dict, dict):
        return []
    properties = schema_dict.get("properties")
    if not isinstance(properties, dict):
        return []
    return sorted(p for p in properties.keys() if isinstance(p, str))


def _type_matches(expected_type: str, actual: Any) -> bool:
    """Return True iff *actual* satisfies JSON Schema *expected_type*."""
    if expected_type not in _PRIMITIVE_TYPE_NAMES:
        return True  # unknown type -- be permissive

    if expected_type == "object":
        return isinstance(actual, dict)
    if expected_type == "array":
        return isinstance(actual, list)
    if expected_type == "string":
        return isinstance(actual, str)
    if expected_type == "integer":
        # JSON Schema "integer" accepts bool=False but not bool=True (bools are ints).
        return isinstance(actual, int) and not isinstance(actual, bool)
    if expected_type == "number":
        return isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if expected_type == "boolean":
        return isinstance(actual, bool)
    if expected_type == "null":
        return actual is None
    return True


def _coerce_to_schema(
    agent_result: Any, schema_dict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Best-effort coercion of *agent_result* into the shape of *schema_dict*.

    Strategy (per Wf-285E precedent):

      * If *agent_result* is a dict, keep all keys that appear in the
        expected schema (drop extras).
      * Drop keys with wrong type; never fabricate new values.
      * Return ``{}`` when nothing is recoverable -- the caller tags
        this as ``{"_schema_detected": True, "_partial": True}`` via
        :func:`extract_agent_result`.
    """
    out: Dict[str, Any] = {}
    if not isinstance(schema_dict, dict):
        if isinstance(agent_result, dict):
            return dict(agent_result)
        return {}

    expected = _expected_fields(schema_dict)
    properties = schema_dict.get("properties") if isinstance(schema_dict, dict) else None
    if not isinstance(properties, dict):
        properties = {}

    # Best-effort: try dict path first.
    candidate_dict: Optional[Dict[str, Any]] = None
    if isinstance(agent_result, dict):
        candidate_dict = agent_result
    elif isinstance(agent_result, str):
        try:
            parsed = json.loads(agent_result.strip())
            if isinstance(parsed, dict):
                candidate_dict = parsed
        except (ValueError, json.JSONDecodeError):
            candidate_dict = None

    if candidate_dict is None:
        return {}

    for field_name in expected:
        if field_name not in candidate_dict:
            continue
        actual = candidate_dict[field_name]
        prop_def = properties.get(field_name)
        if isinstance(prop_def, dict):
            exp_type = prop_def.get("type")
            if isinstance(exp_type, str) and _type_matches(exp_type, actual):
                out[field_name] = actual
            # Else: drop the value (wrong type).
        else:
            out[field_name] = actual
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HarnessAutoFallback:
    """All-in-one harness auto-fallback helper.

    Wraps the Wf-301E detector, the Wf-302D retry helper, and the
    schema-mismatch layer added by this module into a single object
    that callers can drive with one method:
    :meth:`retry_with_correction`.

    Parameters
    ----------
    schema_dict:
        The JSON Schema definition the agent is supposed to fill.  When
        ``None``, only the Wf-302D behaviour is exposed (the mismatch
        detection is a no-op).
    max_retries:
        Maximum number of **retry** attempts beyond the initial call.
        Default ``3`` (one more than Wf-302D's default ``2`` to leave
        room for the mismatch-specific retry after the schema-shape
        retry).
    fallback_strategy:
        What to do when the retry budget is exhausted.  Default
        :attr:`FallbackStrategy.SIBLING_RECONSTRUCTION` (the Wf-285E
        precedent).  :attr:`FallbackStrategy.EMPTY_DEFAULT` returns
        ``{}``; :attr:`FallbackStrategy.RAISE` re-raises the last
        agent error.

    Usage
    -----

    ::

        from wf320b_harness_autofallback import HarnessAutoFallback

        helper = HarnessAutoFallback(schema_dict=EXPECTED_SCHEMA)
        result, report = helper.retry_with_correction(
            agent_callable=my_agent,
            max_retries=3,
        )
        if result is None:
            # Budget exhausted -- report.fallback_strategy tells you which
            # fallback was applied.
            ...
    """

    def __init__(
        self,
        schema_dict: Optional[Dict[str, Any]] = None,
        *,
        max_retries: int = 3,
        fallback_strategy: FallbackStrategy = FallbackStrategy.SIBLING_RECONSTRUCTION,
    ) -> None:
        self.schema_dict = schema_dict
        self.max_retries = max_retries
        self.fallback_strategy = fallback_strategy
        # Re-use the Wf-302D retry helper under the hood so we don't
        # duplicate the retry loop logic.
        self._wf302d_max_retries = max(1, max_retries - 1) if max_retries > 0 else 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def retry_with_correction(
        self,
        agent_callable: Callable[..., Any],
        *,
        max_retries: Optional[int] = None,
    ) -> tuple:
        """Drive the agent with retry + schema-coerce fallback.

        Two-stage strategy:

          1. Run the Wf-302D retry path (handles plain schema-shape) with
             the Wf-302D-style generic correction prompt.
          2. If the result is still a mismatch (not a plain schema-shape,
             but values that do not match the expected schema), invoke
             the agent one final time with a schema_coerce hint that
             enumerates the missing / extra / type-mismatched fields.

        On budget exhaustion, apply :attr:`fallback_strategy`.

        Parameters
        ----------
        agent_callable:
            A keyword-only callable accepting ``correction_prompt``.
        max_retries:
            Override :attr:`max_retries` for this call only.

        Returns
        -------
        ``(result, report)`` where ``result`` is whatever the agent
        returned (or the fallback artefact) and ``report`` is an
        :class:`AutoFallbackReport`.
        """
        max_retries = self.max_retries if max_retries is None else max_retries
        if max_retries < 0:
            raise ValueError(
                "max_retries must be >= 0 (got %r)." % (max_retries,)
            )

        report = AutoFallbackReport(fallback_strategy=None)

        # Track real invocation count via a thin wrapper so the report
        # records what actually happened (vs the budget cap).
        wf302d_max_retries = max(0, max_retries - 1) if max_retries > 0 else 0

        # Stage 1: Wf-302D retry (schema-shape path).  Wrap to count
        # the actual number of agent invocations.
        invocation_count = {"n": 0}

        def _counting_callable(*args, **kwargs):
            invocation_count["n"] += 1
            return agent_callable(*args, **kwargs)

        result = run_agent_with_retry(
            agent_callable=_counting_callable,
            schema_dict=self.schema_dict,
            max_retries=wf302d_max_retries,
        )
        report.n_attempts = invocation_count["n"]

        # Stage 2: mismatch-specific retry (if any retries remain).
        if result is not None:
            mismatch = detect_schema_mismatch(result, self.schema_dict)
            if mismatch.get("is_mismatch", False) and not mismatch.get(
                "is_schema_shape", False
            ):
                if max_retries > 0:
                    # Use one of the remaining retries for the schema_coerce hint.
                    hint = build_schema_coerce_hint(mismatch, self.schema_dict)
                    report.coerce_hints_used.append(hint)
                    report.mismatch_details = {
                        k: v for k, v in mismatch.items() if k != "expected_fields"
                    }
                    result = agent_callable(correction_prompt=hint)
                    invocation_count["n"] += 1
                    report.n_attempts = invocation_count["n"]
                    mismatch = detect_schema_mismatch(result, self.schema_dict)

        # Final classification.
        if result is None:
            report.budget_exhausted = True
            report.final_reason = FailureReason.SCHEMA_SHAPE  # conservative default
            result = self._apply_fallback(report)
        else:
            mismatch = detect_schema_mismatch(result, self.schema_dict)
            report.mismatch_details = (
                {k: v for k, v in mismatch.items() if k != "expected_fields"}
                if mismatch.get("is_mismatch", False)
                else report.mismatch_details
            )
            if mismatch.get("is_mismatch", False):
                # Still mismatched after retries -- apply fallback.
                report.budget_exhausted = True
                report.final_reason = FailureReason(mismatch.get("reason", "schema_mismatch"))
                result = self._apply_fallback(report)
            else:
                report.final_reason = FailureReason.OK
                report.recovered = True

        report.final_result = result
        report.fallback_strategy = (
            self.fallback_strategy if report.budget_exhausted else None
        )
        return result, report

    def coerce(
        self, agent_result: Any,
    ) -> Dict[str, Any]:
        """Coerce *agent_result* to the expected schema (best-effort).

        Thin wrapper around the module-level :func:`_coerce_to_schema`
        so callers can use the instance to inspect what was kept /
        dropped.
        """
        return _coerce_to_schema(agent_result, self.schema_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_fallback(self, report: AutoFallbackReport) -> Any:
        """Apply :attr:`fallback_strategy` to produce a final result."""
        strategy = self.fallback_strategy
        report.fallback_strategy = strategy

        if strategy == FallbackStrategy.EMPTY_DEFAULT:
            return {}

        if strategy == FallbackStrategy.RAISE:
            raise RuntimeError(
                "Wf-320B HarnessAutoFallback: budget exhausted "
                "(reason=%s, attempts=%d); RAISE strategy in effect."
                % (report.final_reason.value, report.n_attempts)
            )

        # SIBLING_RECONSTRUCTION: best-effort coercion of any cached
        # prior result; falls back to {} when nothing is recoverable.
        # We do NOT have access to the original raw agent_result here
        # (only the post-retry final_result), so we coerce what we have.
        coerced = _coerce_to_schema(report.final_result, self.schema_dict)
        if coerced:
            coerced["_schema_detected"] = True
            coerced["_partial"] = True
            coerced["_fallback_strategy"] = strategy.value
            return coerced
        return {"_schema_detected": True, "_partial": True, "_fallback_strategy": strategy.value}


# ---------------------------------------------------------------------------
# Convenience integration with the Wf-302D helper.
# ---------------------------------------------------------------------------


def wrap_wf302d_with_autofallback(
    schema_dict: Optional[Dict[str, Any]] = None,
    *,
    max_retries: int = 3,
    fallback_strategy: FallbackStrategy = FallbackStrategy.SIBLING_RECONSTRUCTION,
) -> HarnessAutoFallback:
    """Return a :class:`HarnessAutoFallback` configured for Wf-302D callers.

    This is the **integration point** the Wf-320B brief asks for:
    callers that today use ``run_agent_with_retry`` can switch to this
    helper with one line of source change and gain the mismatch-specific
    retry + auto-fallback layer without otherwise touching their code.
    """
    return HarnessAutoFallback(
        schema_dict=schema_dict,
        max_retries=max_retries,
        fallback_strategy=fallback_strategy,
    )


__all__ = [
    "HarnessAutoFallback",
    "FailureReason",
    "FallbackStrategy",
    "AutoFallbackReport",
    "detect_schema_mismatch",
    "build_schema_coerce_hint",
    "wrap_wf302d_with_autofallback",
]
