"""Wf-298D harness-layer StructuredOutput auto-fallback shape detector.

Context
-------
Per Wf-291C root-cause investigation (see
molmetal/reports/wf291c_pipeline_root_cause/root_cause.md), workflow agents
tasked with returning complex structured JSON via the harness `StructuredOutput`
tool sometimes return the JSON-schema DEFINITION (e.g.
``{"type":"object","properties":{...},"required":[...]}``) instead of the actual
values matching that schema. The parent pipeline then tries to extract named
fields (e.g. ``G1_cuda_fix_verified``) and receives ``undefined`` for every key.

This module is a USER-SIDE workaround (the StructuredOutput tool itself lives
inside the closed-source claude-code native binary, which is NOT user-modifiable
from this workspace). The wrapper implements option **(c) auto-fallback to text**
from Wf-291C §5.4: it inspects a candidate ``StructuredOutput`` payload, decides
whether the JSON is the schema SHAPE or actual data VALUES, and:

  - if data: returns the parsed object verbatim;
  - if schema: extracts the agent's free-text commentary (typically a sibling
    1-line summary) and returns a ``FallbackExtraction`` with the recovered text
    plus a ``schema_shape_detected=True`` flag for the synthesis layer to react.

Why "is `properties` a dict" is the discriminator
------------------------------------------------
A real JSON value for an object schema looks like:

    {"G1_cuda_fix_verified": true, "G2_config_yaml_exists": false, ...}

A schema-shape payload (what the buggy agents return) looks like:

    {"type":"object",
     "properties": {"G1_cuda_fix_verified": {"type":"boolean"}, ...},
     "required": ["G1_cuda_fix_verified", ...]}

The diagnostic signal is the TYPE of the JSON object values:

  - **values payload**: every value is a primitive (bool/int/float/str/array/null)
    or a list of primitives. Nested objects contain REAL DATA, not schemas.
  - **schema payload**: at least one top-level value is itself an object whose
    own value is a string matching a JSON-Schema type keyword (one of
    ``"string"``, ``"integer"``, ``"number"``, ``"boolean"``, ``"array"``,
    ``"object"``, ``"null"``), AND that nested object also carries one of the
    schema meta-keys (``type`` / ``properties`` / ``required`` / ``items`` /
    ``enum`` / ``$ref`` / ``format`` / ``description``).

We additionally catch the doubly-malformed hybrid seen in Wf-271D where the
"properties" field is the empty string and the field VALUES are empty strings
(see root_cause.md §2.1) — those are detected as "schema-shape + values empty".

The fallback path returns the *raw text* the agent emitted alongside the schema
payload. Most Wf-274/280/282/284 wave agents that hit this regression still
wrote a useful free-text report via Edit/Read (see root_cause.md §3.1 table);
the synthesis layer can feed that text straight into its reconstruction tree.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# JSON-Schema primitive type keywords. These are the canonical type names from
# https://json-schema.org/draft-07/draft-handrews-json-schema-01#rfc.section.4.2.1
# and they are the most reliable signal that a nested object is a schema node
# rather than a payload value.
_JSON_SCHEMA_TYPE_KEYWORDS = frozenset(
    {
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "null",
    }
)

# JSON-Schema meta keys. If a nested object carries ANY of these, it is almost
# certainly a schema node, not a payload value.
_JSON_SCHEMA_META_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "$ref",
        "format",
        "description",
        "additionalProperties",
        "patternProperties",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "definitions",
        "$defs",
        "$schema",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)

# String-identifier keys. A real JSON-Schema row almost always carries one of
# these alongside its "type" keyword (e.g. ``{"name":"G1","type":"boolean"}``
# from Wf-273B). Real data payloads rarely carry a key called ``name``/``id``/
# ``$ref``/``title``/etc. whose value is a short string identifier.
_STRING_IDENTIFIER_KEYS = frozenset(
    {"name", "$ref", "title", "description", "id", "key", "const", "$id"}
)

# Top-level markers that signal a schema-shape payload at the root.
_SCHEMA_ROOT_MARKERS = frozenset({"properties", "required", "$ref", "$defs"})

# Minimum length of a payload-shape candidate for the detector to consider it
# data at all. Empty / 1-key payloads are too ambiguous and the caller should
# decide.
_MIN_PAYLOAD_KEYS = 1

# Cap on how aggressively we scan nested arrays/objects. Schema-shape payloads
# seen in the Wf-270/271/272/274/279/285 regression class were always flat
# (single nesting level), so a 2-level walk is plenty.
_MAX_NEST_DEPTH = 2


@dataclass
class FallbackExtraction:
    """Result of HarnessOutputExtractor.extract() when schema-shape is detected.

    Attributes
    ----------
    schema_shape_detected : bool
        Always True for this dataclass (the extract() method returns either
        a parsed payload dict or a FallbackExtraction — schema_shape_detected
        distinguishes the two).
    recovered_text : str
        The free-text fallback that the synthesis layer should feed into its
        reconstruction tree. Default: empty string.
    raw_payload : Any
        The original JSON-decoded value (preserved so callers can log / audit
        exactly what the agent returned).
    fallback_reason : str
        Human-readable explanation of WHY the detector flagged the payload.
        One of:
          - "properties_is_schema_object"
          - "properties_has_schema_meta_keys"
          - "nested_object_with_schema_type_keyword"
          - "hybrid_empty_string_properties"
          - "root_has_schema_root_markers"
          - "values_empty_under_schema_wrapper"
    payload_key_count : int
        How many top-level keys the JSON payload had. Useful for the synthesis
        layer to decide whether to re-dispatch vs. reconstruct.
    """

    schema_shape_detected: bool = True
    recovered_text: str = ""
    raw_payload: Any = None
    fallback_reason: str = ""
    payload_key_count: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class HarnessOutputExtractor:
    """Detect & work around StructuredOutput schema-shape regression.

    Public API
    ----------
    detect_schema_shape(json_string): bool
        Pure boolean detector — see class docstring §"Why ..." for the rule.
    extract(json_string, *, sibling_text: str = "") -> dict | FallbackExtraction
        Full extraction. If the payload is a value-shaped JSON object, returns
        the parsed dict. If the payload is schema-shaped, returns a
        FallbackExtraction wrapping the ``sibling_text`` (or the raw string if
        none provided).
    extract_from_agent_message(message: dict, *, sibling_text: str = "") -> ...
        Convenience wrapper that pulls the candidate payload out of a typical
        claude-1001 task-output ``{"type":"result", "result": ...}`` envelope.

    Notes for callers
    -----------------
    The detector is intentionally CONSERVATIVE: it only flags a payload as
    schema-shape when there is unambiguous evidence (a nested object that is
    itself a schema node, or the root carries schema root markers, or the
    hybrid empty-string pattern from Wf-271D). False negatives (saying "data"
    when it was really schema) leave the caller exactly where it would have
    been without this wrapper — so the cost of a miss is bounded.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_schema_shape(self, json_string: str | bytes | bytearray) -> bool:
        """Return True iff ``json_string`` is a JSON-Schema definition, not values.

        Accepts ``str``, ``bytes``, or ``bytearray``. Strings are stripped of
        surrounding whitespace before parsing. Non-JSON input returns False
        (the harness extractor should fall back to plain-text in that case
        upstream of this method).
        """
        if isinstance(json_string, (bytes, bytearray)):
            json_string = json_string.decode("utf-8", errors="replace")
        if not isinstance(json_string, str):
            return False
        payload = self._safe_parse(json_string)
        if payload is None:
            return False
        return self._is_schema_shape(payload)

    def extract(
        self,
        json_string: str | bytes | bytearray,
        *,
        sibling_text: str = "",
    ) -> dict[str, Any] | FallbackExtraction:
        """Extract values or fall back to sibling_text when schema-shape detected.

        Parameters
        ----------
        json_string : str | bytes | bytearray
            The candidate StructuredOutput payload from the agent.
        sibling_text : str, optional
            The free-text companion output (1-line summary the agent emitted
            via Bash / Edit / Read in Wf-274+ workaround style). Used as the
            fallback payload when schema-shape is detected.

        Returns
        -------
        dict
            Parsed JSON object — what the synthesis layer expects in the
            normal (non-regression) case.
        FallbackExtraction
            Wrapped fallback with ``schema_shape_detected=True`` and the
            recovered_text from ``sibling_text`` (or the raw string).
        """
        if isinstance(json_string, (bytes, bytearray)):
            json_string = json_string.decode("utf-8", errors="replace")
        if not isinstance(json_string, str):
            # Non-string input → can't be schema-shape, return raw envelope.
            return {"_raw": json_string}

        payload = self._safe_parse(json_string)
        if payload is None:
            # Parse failure → likely raw text. Fall back to sibling_text.
            return FallbackExtraction(
                recovered_text=(sibling_text or json_string).strip(),
                raw_payload=json_string,
                fallback_reason="json_parse_failed",
            )

        if not self._is_schema_shape(payload):
            # Happy path — parsed JSON object that LOOKS like data.
            if isinstance(payload, dict):
                return payload
            return {"_value": payload}

        # Schema-shape detected. Build the fallback.
        reason = self._diagnose_reason(payload)
        return FallbackExtraction(
            recovered_text=sibling_text.strip(),
            raw_payload=payload,
            fallback_reason=reason,
            payload_key_count=len(payload) if isinstance(payload, dict) else 0,
        )

    def extract_from_agent_message(
        self,
        message: dict[str, Any],
        *,
        sibling_text: str = "",
    ) -> dict[str, Any] | FallbackExtraction:
        """Convenience extractor for a ``{"type":"result", "result": ...}`` envelope.

        Per Wf-291C §2.1, the claude-1001 task-output transcripts store agent
        results inside ``message["result"]``. This method unwraps that and
        re-serialises it for ``self.extract``.
        """
        # The schema-shape payloads we observed ALWAYS lived under message["result"].
        if not isinstance(message, dict):
            return FallbackExtraction(
                recovered_text=sibling_text,
                raw_payload=message,
                fallback_reason="message_not_dict",
            )
        candidate = message.get("result", message)
        if isinstance(candidate, str):
            serialised = candidate
        else:
            serialised = json.dumps(candidate, ensure_ascii=False)
        return self.extract(serialised, sibling_text=sibling_text)

    # ------------------------------------------------------------------
    # Internal: parsing + classification
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse(json_string: str) -> Any:
        """Parse JSON; return None on any decode error."""
        try:
            return json.loads(json_string)
        except (ValueError, TypeError):
            return None

    def _is_schema_shape(self, payload: Any) -> bool:
        """Return True iff ``payload`` looks like a JSON-Schema def, not data."""
        if not isinstance(payload, dict) or len(payload) < _MIN_PAYLOAD_KEYS:
            return False
        return self._classify_node(payload, depth=0) == "schema"

    def _classify_node(self, node: Any, *, depth: int) -> str:
        """Return 'schema' if the node matches the schema-shape signature.

        Walks the node up to ``_MAX_NEST_DEPTH`` levels deep. Returns 'data' as
        soon as a clearly-value-shaped subtree is found.
        """
        if not isinstance(node, dict):
            return "data"
        if depth > _MAX_NEST_DEPTH:
            return "data"

        # Single-node fingerprint (catches Wf-273B `{"fields":[{"name":...,
        # "type":"boolean"}]}` because the array-item dict itself is a
        # schema node). Real data payloads with short dicts AND a "type"
        # JSON-Schema keyword are extremely rare — the cost of a false-
        # positive here is the caller falling back to sibling_text (safe).
        if self._is_likely_schema_node(node):
            return "schema"

        # Hybrid empty-string pattern from Wf-271D: top-level values are
        # empty strings, AND the object carries schema root markers. We catch
        # this BEFORE the meta-keys check below because every value is "" so
        # the per-value signal won't fire.
        empty_strs = sum(1 for v in node.values() if v == "")
        if empty_strs >= max(2, len(node) // 2):
            if any(k in _SCHEMA_ROOT_MARKERS for k in node):
                return "schema"

        # Per-value signal: a value that is a dict carrying schema meta-keys
        # OR whose own "type" value is a JSON-Schema type keyword.
        for value in node.values():
            if isinstance(value, dict):
                if value.keys() & _JSON_SCHEMA_META_KEYS:
                    return "schema"
                inner_type = value.get("type")
                if isinstance(inner_type, str) and inner_type in _JSON_SCHEMA_TYPE_KEYWORDS:
                    # Nested "type": "string" inside a dict that has NO other
                    # content is very strong schema signal.
                    if len(value) <= 3:
                        return "schema"
                # Recurse one level deeper — the malformed payload from
                # Wf-272A had nested {"type":"integer"} inside {"properties":{}}.
                child = self._classify_node(value, depth=depth + 1)
                if child == "schema":
                    return "schema"
            elif isinstance(value, list):
                # Arrays of schema-nodes (Wf-273B/C/D used {"fields":[{...}]})
                for item in value:
                    if isinstance(item, dict):
                        # Quick fingerprint check: short dict with a JSON-
                        # Schema "type" keyword is almost certainly a schema
                        # node. Avoids the false-negative on Wf-273B where
                        # the array items have no nested dicts to recurse
                        # into — only flat {"name":..., "type":"boolean"}.
                        if self._is_likely_schema_node(item):
                            return "schema"
                        child = self._classify_node(item, depth=depth + 1)
                        if child == "schema":
                            return "schema"

        # Top-level schema-root marker (no nested object needed). The Wf-271A
        # payload had {"type":"object","properties":"","items":"", ...} where
        # "properties" was the empty string — caught by the empty-string rule
        # above. The Wf-272A payload had {"type":"object","properties":{...}}
        # — caught by the per-value rule above.
        if "type" in node and isinstance(node["type"], str) and node["type"] in _JSON_SCHEMA_TYPE_KEYWORDS:
            # If "type" is one of the schema keywords AND there is at least
            # one sibling that smells like a schema (properties/required/items),
            # we have a schema root.
            if any(k in node for k in _SCHEMA_ROOT_MARKERS):
                return "schema"
            # Otherwise "type" might be a real data field — keep scanning.

        return "data"

    @classmethod
    def _is_likely_schema_node(cls, node: Any) -> bool:
        """Single-dict fingerprint for a JSON-Schema definition node.

        Used both as the top-level check in ``_classify_node`` (catches
        ``{"type":"object","fields":[...]}`` wrappers) and for individual
        array elements (catches ``{"name":"G1_...","type":"boolean"}``
        schema rows inside a ``fields`` list).

        Three signals — ANY one is sufficient:

        1. The dict has a non-``type`` schema meta-key (``properties`` /
           ``required`` / ``items`` / ``$ref`` / ``enum`` / ``format`` /
           ``description`` / ...). Catches Wf-270A / Wf-272A wrappers.
        2. The dict has ``"type"`` whose value is a JSON-Schema type keyword
           AND at least one schema-root marker sibling (``properties`` /
           ``required`` / ``items`` / ``$ref`` / ``$defs``). Catches the
           Wf-272A outer shape. The sibling requirement prevents false-
           positives on Wf-272B working payloads like
           ``{"type":"object", "G1_n_new_entries": 4, ...}``.
        3. The dict is SHORT (≤5 keys), has a ``"type"`` JSON-Schema keyword,
           AND carries a string-identifier key (``name`` / ``$ref`` /
           ``title`` / ``description`` / ``id`` / ``key`` / ``const`` /
           ``$id``). Catches the Wf-273B array-element rows like
           ``{"name":"G1_pre_flight_25_25","type":"boolean"}``. Real data
           payloads rarely carry these identifier-style keys.
        """
        if not isinstance(node, dict):
            return False

        non_type_meta = set(node.keys()) & (_JSON_SCHEMA_META_KEYS - {"type"})
        if non_type_meta:
            return True

        t = node.get("type")
        if isinstance(t, str) and t in _JSON_SCHEMA_TYPE_KEYWORDS:
            # Signal 2: "type" keyword + schema-root marker sibling.
            if node.keys() & _SCHEMA_ROOT_MARKERS:
                return True
            # Signal 3: very short + "type" keyword + string-identifier key.
            if len(node) <= 5 and (node.keys() & _STRING_IDENTIFIER_KEYS):
                return True

        return False

    def _diagnose_reason(self, payload: Any) -> str:
        """Return a stable human-readable reason code for logging / dashboards."""
        if not isinstance(payload, dict):
            return "unknown"
        empty_strs = sum(1 for v in payload.values() if v == "")
        if empty_strs >= max(2, len(payload) // 2) and any(
            k in _SCHEMA_ROOT_MARKERS for k in payload
        ):
            return "hybrid_empty_string_properties"
        for value in payload.values():
            if isinstance(value, dict):
                if value.keys() & _JSON_SCHEMA_META_KEYS:
                    return "properties_has_schema_meta_keys"
                inner_type = value.get("type")
                if isinstance(inner_type, str) and inner_type in _JSON_SCHEMA_TYPE_KEYWORDS:
                    if len(value) <= 3:
                        return "nested_object_with_schema_type_keyword"
                if self._classify_node(value, depth=1) == "schema":
                    return "properties_is_schema_object"
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and self._classify_node(item, depth=1) == "schema":
                        return "nested_object_with_schema_type_keyword"
        if any(
            isinstance(payload.get("type"), str)
            and payload["type"] in _JSON_SCHEMA_TYPE_KEYWORDS
            and any(k in payload for k in _SCHEMA_ROOT_MARKERS)
            for _ in [0]
        ):
            return "root_has_schema_root_markers"
        return "unknown"


# ----------------------------------------------------------------------
# Convenience: regex-based fast pre-filter (use only when full JSON parse
# would be too expensive; e.g. scanning thousands of journal.jsonl lines).
# ----------------------------------------------------------------------

# A "looks like JSON schema" pre-filter. This is intentionally looser than the
# structural detector — it can false-positive, so use it only to short-circuit
# before invoking the full parser.
_SCHEMA_FAST_PRE_FILTER = re.compile(
    r'"type"\s*:\s*"(?:string|integer|number|boolean|array|object|null)"'
    r'|"\$ref"\s*:|"properties"\s*:\s*\{'
)


def fast_filter_says_schema(json_string: str | bytes | bytearray) -> bool:
    """Return True if the string contains schema-shape fingerprints.

    Faster than HarnessOutputExtractor.detect_schema_shape() because it
    operates on the raw string with a single regex pass. Use this to short-
    circuit before invoking the full parser when scanning large journal
    transcripts.

    False-positive rate: moderate (e.g. literal string content that happens
    to contain the pattern). False-negative rate: very low — the regression
    class ALWAYS carries at least one of these markers.
    """
    if isinstance(json_string, (bytes, bytearray)):
        json_string = json_string.decode("utf-8", errors="replace")
    if not isinstance(json_string, str):
        return False
    return bool(_SCHEMA_FAST_PRE_FILTER.search(json_string))


__all__ = [
    "HarnessOutputExtractor",
    "FallbackExtraction",
    "fast_filter_says_schema",
]