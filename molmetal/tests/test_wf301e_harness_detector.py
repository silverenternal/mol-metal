"""Tests for the Wf-301E harness shape detector.

================================================================
Closes the Wf-291C regression class.

These are the pytest mirrors of
:file:`molmetal/scripts/wf301e_harness_detector.py`.  Five tests pin the
detector contract:

  1. test_real_value_dict         -- real values returned by an agent are
                                     classified as NOT-schema and the
                                     extractor returns them verbatim.
  2. test_schema_shape_dict       -- the regression case: an agent
                                     returns the JSON Schema definition
                                     (Wf-270 / Wf-271 / Wf-272 / Wf-274
                                     / Wf-279 / Wf-285 evidence) and the
                                     detector must flag it; the
                                     extractor must return a tagged
                                     fallback.
  3. test_nested_real_value       -- real values that contain
                                     {"type": "object"} as a domain
                                     tag (NOT a schema definition) must
                                     still be classified as values.
  4. test_nested_schema_shape     -- a deeply nested schema definition
                                     with extra scaffolding must also be
                                     detected.
  5. test_text_passthrough        -- plain-text / markdown agent
                                     outputs (the Wf-274 / Wf-280 /
                                     Wf-282 workaround path) must
                                     survive the extractor unchanged.

All tests are CPU-only, deterministic, and complete in < 5 seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

from wf301e_harness_detector import (  # noqa: E402
    detect_schema_shape,
    extract_agent_result,
)


# ---------------------------------------------------------------------------
# 1. test_real_value_dict -- the happy path.
# ---------------------------------------------------------------------------


def test_real_value_dict() -> None:
    """A dict of real values must NOT be flagged as schema-shape."""
    agent_payload = {
        "G1_cuda_fix_verified": True,
        "G2_config_yaml_exists": True,
        "G3_yaml_md_valid": True,
        "G4_n_pre_flight_items": 25,
        "G5_n_open_violations": 0,
        "honest_framing_note": "MEASURED, single seed, single epoch",
    }

    # 1. Detector must NOT flag it.
    assert (
        detect_schema_shape(json.dumps(agent_payload)) is False
    ), "Real-value dict was misclassified as a schema-shape payload."

    # 2. Extractor must return the values verbatim.
    extracted = extract_agent_result(agent_payload)
    assert extracted == agent_payload, (
        "Extractor mangled the real-value dict: %r" % extracted
    )
    assert "_schema_detected" not in extracted


# ---------------------------------------------------------------------------
# 2. test_schema_shape_dict -- the regression case.
# ---------------------------------------------------------------------------


def test_schema_shape_dict() -> None:
    """The Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 / Wf-285 regression.

    An agent returns the JSON Schema definition instead of values.  The
    detector must flag it; the extractor must NOT pretend values exist
    where they don't.
    """
    # Wf-270A verbatim (per root_cause.md §2.1).
    schema_payload = {
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

    # 1. Detector MUST flag it.
    assert (
        detect_schema_shape(json.dumps(schema_payload)) is True
    ), "Regression-case schema payload was not detected as schema-shape."

    # 2. Extractor MUST mark it as schema-detected and not silently
    #    fabricate values.
    extracted = extract_agent_result(schema_payload)
    assert extracted.get("_schema_detected") is True, (
        "Extractor failed to tag schema-shape result.  Got: %r" % extracted
    )

    # 3. None of the expected data fields should have leaked through --
    #    if they did, the synthesis pipeline would silently use schema
    #    booleans/integers as data values.
    for key in (
        "G1_cuda_fix_verified",
        "G2_config_yaml_exists",
        "G4_n_pre_flight_items",
    ):
        assert key not in extracted, (
            "Schema key %r leaked into extracted values: %r" % (key, extracted)
        )


# ---------------------------------------------------------------------------
# 3. test_nested_real_value -- false-positive guard.
# ---------------------------------------------------------------------------


def test_nested_real_value() -> None:
    """A real payload that incidentally contains ``"type": "object"``.

    This is the hardest false-positive: a payload like
    ``{"type":"object","G1_score":0.91}`` where ``"type": "object"`` is
    a domain tag (e.g. recording an object's class), NOT a JSON Schema
    declaration.  The detector must NOT flag this as a schema definition
    because there is no ``"properties"`` dict.
    """
    agent_payload = {
        "type": "object",
        "G1_score": 0.91,
        "G2_passes": True,
        "G3_label": "validated",
    }

    # 1. Detector must NOT flag it (no "properties" key).
    assert detect_schema_shape(json.dumps(agent_payload)) is False, (
        "Real payload with 'type':'object' tag was misclassified as "
        "schema-shape -- the false-positive we explicitly guard against."
    )

    # 2. Extractor must return the values.
    extracted = extract_agent_result(agent_payload)
    assert extracted == agent_payload
    assert "_schema_detected" not in extracted


# ---------------------------------------------------------------------------
# 4. test_nested_schema_shape -- deeper schemas.
# ---------------------------------------------------------------------------


def test_nested_schema_shape() -> None:
    """A deeply nested schema definition (Wf-272A verbatim evidence).

    The Wf-272A regression carried a schema with both a ``properties``
    dict AND a ``required`` object (not array).  The detector must still
    flag it.
    """
    # Wf-272A verbatim (per root_cause.md §2.1).
    nested_schema = {
        "type": "object",
        "properties": {
            "G1_n_violations_fixed": {"type": "integer"},
            "G2_no_new_violations": {"type": "boolean"},
            "G3_yaml_md_valid": {"type": "boolean"},
        },
        "required": {
            "G1_n_violations_fixed": {},
            "G2_no_new_violations": {},
        },
    }

    # 1. Detector MUST flag it.
    assert (
        detect_schema_shape(json.dumps(nested_schema)) is True
    ), "Nested-schema payload (Wf-272A verbatim) was not detected."

    # 2. Extractor MUST mark it as schema-detected.
    extracted = extract_agent_result(nested_schema)
    assert extracted.get("_schema_detected") is True


# ---------------------------------------------------------------------------
# 5. test_text_passthrough -- the workaround path.
# ---------------------------------------------------------------------------


def test_text_passthrough() -> None:
    """Plain-text agent outputs must survive the extractor unchanged.

    Per Wf-274E (root_cause.md §5.1), the workaround is to force agents
    to return plain text via Edit.  The extractor must wrap such text
    but preserve the original string.
    """
    raw_text = (
        "Wf-274A complete. Added regression_disclosure top-level field "
        "to metrics/by_metric/cfm_decode_real.json documenting the "
        "Wf-271B silent regression."
    )

    # 1. Detector must NOT flag plain text.
    assert (
        detect_schema_shape(raw_text) is False
    ), "Plain-text agent output was misclassified as schema-shape."

    # 2. Extractor must wrap as {"text": <raw>}.
    extracted = extract_agent_result(raw_text)
    assert "text" in extracted
    assert extracted["text"] == raw_text
    assert "_schema_detected" not in extracted


# ---------------------------------------------------------------------------
# Bonus: stringify-then-detect round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected_schema",
    [
        # Real values -- NOT schema.
        ({"G1": True, "G2": 5}, False),
        # Schema -- IS schema.
        (
            {
                "type": "object",
                "properties": {"G1": {"type": "boolean"}},
            },
            True,
        ),
        # Plain text -- NOT schema.
        ("just a sentence", False),
        # Empty -- NOT schema.
        ("", False),
        # Non-JSON -- NOT schema.
        ("hello world", False),
        # Real data with "type":"object" tag (no properties) -- NOT schema.
        ({"type": "object", "name": "foo"}, False),
    ],
)
def test_detector_round_trip(payload, expected_schema) -> None:
    """Sanity check across a small parametrised grid."""
    assert detect_schema_shape(json.dumps(payload)) is expected_schema
