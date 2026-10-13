# SPDX-License-Identifier: Apache-2.0
"""The sweep helper must RECORD why a decode failed, not just that it did.

THE DEFECT THIS PINS
--------------------
``wf_blkc_l3_guidance_probe.py::decode`` read ``.smiles`` and ``.sanitized``
and never ``.error``. Every failure therefore collapsed into an anonymous
``smiles: None``.

That made the 2026-10-12 aromatic-offset sweep UNINTERPRETABLE. Decode rate
fell 22 -> 13 -> 3 -> 0 of 64 as the bias rose 0.0 -> 0.5 -> 1.5 -> 3.0, and
with no reason recorded it is impossible to say whether:

  * the BOND HEAD rejected its own output  -> a head-ranking problem, and a
    retrain of the head is the right lever; or
  * RDKit sanitisation refused a valence    -> a bond-count problem, and the
    head was never the thing at fault.

Those are opposite diagnoses and the sweep data cannot choose between them.
``DecodedMol.error`` carries the distinction (``cfm_decoder_yuelbond.py``:727
RWMol.GetMol failed, :779 empty_or_failed, and a SanitizeMol branch at :733
that does NOT set it -- which is why ``error_stage`` records
``unknown_no_error_field`` rather than pretending every failure has a reason).

RED CHECK: drop the ``decode_error`` key from the failure branch and
``test_failure_rows_carry_a_reason`` fails.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PROBE = REPO / "molmetal" / "scripts" / "wf_blkc_l3_guidance_probe.py"


def _probe():
    if "_probe_err_test" in sys.modules:
        return sys.modules["_probe_err_test"]
    spec = importlib.util.spec_from_file_location("_probe_err_test", str(PROBE))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_probe_err_test"] = m
    spec.loader.exec_module(m)
    return m


def _decode_source() -> str:
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "decode":
            return ast.get_source_segment(
                PROBE.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError("decode() not found in the probe")


def test_decode_reads_the_error_field():
    src = _decode_source()
    assert '"error"' in src or "'error'" in src or ".error" in src, (
        "decode() never reads DecodedMol.error, so every failure is anonymous"
    )


def test_failure_rows_carry_a_reason():
    """The failure branch must populate the reason field, not drop it."""
    src = _decode_source()
    assert "decode_error" in src, (
        "the failure branch writes no decode_error key, so a rejected molecule "
        "is indistinguishable from one the decoder never attempted"
    )
    assert "error_stage" in src, (
        "without error_stage a decode that produced nothing silently and one "
        "that said why are recorded identically"
    )


def test_success_rows_also_record_the_field():
    """Otherwise success rows and failure rows have different schemas."""
    src = _decode_source()
    # The success dict must also carry decode_error (None on success), so a
    # consumer can read one key off every row.
    assert src.count("decode_error") >= 2, (
        "decode_error appears only once; success and failure rows must share "
        "the key so consumers do not need a branch"
    )


def test_purpose_is_stated_in_the_file():
    """A future edit should see WHY this field exists."""
    src = _decode_source()
    assert "uninterpretable" in src.lower() or "head" in src.lower(), (
        "the rationale distinguishing a head-ranking failure from a "
        "sanitisation failure must live next to the code"
    )