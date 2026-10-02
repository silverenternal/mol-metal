"""Wf-301E: Harness shape detector.

================================================================
Closes the Wf-291C regression class.

Per Wf-291C root-cause investigation: when workflow agents are tasked with
returning complex structured JSON via the `StructuredOutput` tool, the
agents return the JSON **schema definition** itself rather than the actual
values matching that schema. The parent pipeline then tries to extract
fields (e.g., ``G1_n_violations_fixed``, ``G2_existing_entries_preserved``)
and receives ``undefined`` for all fields.

This module provides a wrapper-level detector:

    * :func:`detect_schema_shape` -- returns ``True`` when a JSON string
      is the schema definition rather than the values.
    * :func:`extract_agent_result` -- pulls the real data dict out of an
      agent result envelope, regardless of whether the agent returned
      values or a schema.

Both functions are pure-Python (no external deps beyond the stdlib) so they
can run inside any harness layer without setup overhead.

Six rounds of evidence (Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 /
Wf-285) confirm that the regression is in the parent pipeline's
schema-extraction layer, NOT in the underlying work products -- so a
wrapper-level detector is the right place to fix it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Union

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

# Substring markers that, when ALL present in a JSON object, indicate the
# payload is the *schema definition* the agent was supposed to fill rather
# than the values themselves.  These markers are anchored on the Wf-270
# / Wf-271 / Wf-272 / Wf-274 / Wf-279 / Wf-285 verbatim evidence (see
# molmetal/reports/wf291c_pipeline_root_cause/root_cause.md §2.1).
# We use compiled regexes so that both compact JSON and ``json.dumps(...,
# indent=...)`` (with whitespace) qualify.
_OBJECT_TYPE_RE = re.compile(r'"type"\s*:\s*"object"')
_PROPERTIES_KEY_RE = re.compile(r'"properties"\s*:\s*\{')

# A nested ``type`` field at any depth that holds a *primitive* type name
# is a strong indicator of a JSON Schema definition.  Primitive type
# names per JSON Schema: object, array, string, integer, number, boolean,
# null.
_PRIMITIVE_TYPE_NAMES = (
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
)
_PRIMITIVE_TYPE_RE = re.compile(
    r'"type"\s*:\s*"(' + "|".join(_PRIMITIVE_TYPE_NAMES) + r')"'
)

# A schema-shaped result almost always contains a ``required`` array OR
# a ``fields`` list of {name, type} dicts.
_SCHEMA_OPTIONAL_MARKERS = (
    '"required"',
    '"fields"',
)


def detect_schema_shape(json_str: str) -> bool:
    """Return True iff *json_str* is a JSON Schema definition, not values.

    The detector is intentionally conservative: it only fires when the
    payload exhibits **both** the mandatory markers (object + properties)
    AND at least one of the optional markers (nested primitive ``type``
    fields, a ``required`` array, or a ``fields`` list).

    Parameters
    ----------
    json_str:
        The candidate string.  May be a raw JSON string, a Python
        ``str(json.dumps(...))`` representation, or any string-shaped
        payload.  If it is not valid JSON, the function returns
        ``False`` (we never want a parse error to be misclassified as a
        schema).

    Returns
    -------
    bool
        ``True`` when the string matches the schema-shape signature;
        ``False`` otherwise (real values, malformed JSON, plain text,
        etc.).
    """
    if not isinstance(json_str, str):
        return False
    if not json_str:
        return False

    # Strip leading whitespace just in case the JSON encoder emitted any.
    payload = json_str.lstrip()
    if not payload or payload[0] not in "{[":
        # Real agent outputs that are not JSON objects/arrays are plain
        # text -- never confuse them with a schema definition.
        return False

    # Both mandatory markers must be present.  JSON Schema only describes
    # an "object" via {"type": "object", "properties": {...}} -- a real
    # data payload may legitimately contain {"type": "object"} but
    # almost never *also* a sibling "properties" key.  We allow optional
    # whitespace so both compact JSON and ``json.dumps(..., indent=...)``
    # qualify.
    if not _OBJECT_TYPE_RE.search(payload):
        return False
    if not _PROPERTIES_KEY_RE.search(payload):
        return False

    # At least one optional marker must be present.  This eliminates the
    # false-positive case where an agent happens to return a payload with
    # the substring "type":"object" (e.g., {"type": "object", "G1": true}).
    if not any(marker in payload for marker in _SCHEMA_OPTIONAL_MARKERS):
        # Check for nested primitive ``type`` fields -- a strong schema
        # signature but missing the optional markers above.
        if not _PRIMITIVE_TYPE_RE.search(payload):
            return False

    # Final confirm: parse the JSON and inspect the structure.  A schema
    # always has a top-level "type" of "object" AND a "properties" dict.
    try:
        parsed = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        return False

    if not isinstance(parsed, dict):
        return False

    if parsed.get("type") != "object":
        return False

    properties = parsed.get("properties")
    if not isinstance(properties, dict):
        return False

    # The properties dict's values must look like JSON Schema property
    # definitions (dicts with their own "type" field) -- this is the
    # strongest disambiguator against real data like
    # {"type":"object", "G1_score": 0.91} where "G1_score" is a leaf
    # value, not a nested schema.
    nested_type_count = 0
    for prop_value in properties.values():
        if isinstance(prop_value, dict) and "type" in prop_value:
            nested_type_count += 1

    # If the majority of property values are dicts containing "type",
    # treat as schema.  Threshold of >=1 is sufficient for the regression
    # evidence we have; raise it if false-positives appear.
    return nested_type_count >= 1 or _PRIMITIVE_TYPE_RE.search(payload) is not None


def extract_agent_result(
    agent_result: Union[Dict[str, Any], str, None],
) -> Dict[str, Any]:
    """Extract a real-data dict from an agent result envelope.

    This is the wrapper-level counterpart to
    :func:`detect_schema_shape`.  Given whatever shape the agent
    returned (``dict``, ``str``, ``None``), produce the dictionary the
    downstream pipeline would have wanted.  If the agent returned a
    schema definition, the function falls back to an empty dict and tags
    the response with ``"_schema_detected": True`` so the caller can log
    / reconstruct from sibling artefacts.

    Parameters
    ----------
    agent_result:
        Whatever the agent returned.  Accepted shapes:

        * ``dict`` already containing the values -- returned as-is
          (shallow-copied).
        * ``dict`` containing the schema shape -- the values are
          reconstructed by looking for non-schema keys (a best-effort
          fallback) or, when nothing is recoverable, an empty dict
          tagged with ``"_schema_detected": True``.
        * ``str`` that parses to JSON -- same logic as the dict case.
        * ``str`` that does not parse to JSON (plain text / markdown) --
          returned as ``{"text": <raw_string>}``.
        * ``None`` -- returned as ``{}``.

    Returns
    -------
    dict
        The extracted values.  Never raises.
    """
    if agent_result is None:
        return {}

    # Path 1: already a dict.
    if isinstance(agent_result, dict):
        if _is_schema_dict(agent_result):
            # Best-effort salvage: surface any sibling keys that look
            # like real data (i.e. NOT the schema scaffolding itself).
            salvaged = {
                k: v
                for k, v in agent_result.items()
                if k not in {"type", "properties", "required", "fields"}
            }
            return {"_schema_detected": True, **salvaged}
        return dict(agent_result)

    # Path 2: string -- either JSON or plain text.
    if isinstance(agent_result, str):
        payload = agent_result.strip()
        if not payload:
            return {}

        # First try to parse as JSON.  If it fails, treat as plain text.
        try:
            parsed = json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            return {"text": agent_result}

        if isinstance(parsed, dict):
            if _is_schema_dict(parsed):
                salvaged = {
                    k: v
                    for k, v in parsed.items()
                    if k not in {"type", "properties", "required", "fields"}
                }
                return {"_schema_detected": True, **salvaged}
            return dict(parsed)

        if isinstance(parsed, list):
            return {"_list": list(parsed)}

        # Scalars returned by the agent -- wrap them.
        return {"value": parsed}

    # Path 3: scalar number / bool from the agent.
    if isinstance(agent_result, (int, float, bool)):
        return {"value": agent_result}

    # Path 4: unknown type -- coerce to string and wrap.
    return {"text": str(agent_result)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_schema_dict(payload: Dict[str, Any]) -> bool:
    """Return True iff *payload* looks like a JSON Schema definition.

    Mirrors the signature in :func:`detect_schema_shape` but works on an
    already-parsed dict (no JSON re-parsing).
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("type") != "object":
        return False
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        return False
    if not properties:
        return False
    # At least one property must itself be a dict with a "type" field --
    # otherwise the payload looks like real data that happens to use
    # {"type": "object"} as a domain tag.
    for prop_value in properties.values():
        if isinstance(prop_value, dict) and "type" in prop_value:
            return True
    return False


__all__ = [
    "detect_schema_shape",
    "extract_agent_result",
]