"""Wf-352-A1 — unit tests for EXPECTED_GATE_COUNTS vs build_gates() reality.

Two tests (per Wf-352-A1 task brief):

* ``test_expected_gate_counts_all_matches_build_gates`` — the documented
  EXPECTED_GATE_COUNTS["all"] must equal ``len(build_gates("all"))`` exactly.
  This is the primary invariant: a drift in either direction (actual >
  expected = silent gate bloat; actual < expected = silent gate loss)
  is an actionable honest-negative.

* ``test_expected_gate_counts_per_pipeline_consistency`` — every pipeline
  listed in :data:`PIPELINES` must have its EXPECTED_GATE_COUNTS[k] value
  equal to ``len(build_gates(k))``. Covers "frontier" / "chemprop_v2_v3" /
  "cfm" alongside "all".

The script under test (``molmetal/scripts/pre_flight_check.py``) is CPU-only
and does not touch any training path. Both tests import the module directly
to introspect ``build_gates`` and ``EXPECTED_GATE_COUNTS`` — no subprocess.

Wf-352-A1 context (Wf-346-B2 audit): 71 unique gate names in dispatcher
vs 66 EXPECTED at Wf-346-B2 baseline; +5 delta. After Wf-350 added 3 NEW
gates (Wf-350-A1 + Wf-350-A3 + Wf-350-A2 = 4-5) the count moved to 71.
This file pins the bump at 71 so any future drift is detected on next
``pytest`` invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make molmetal importable when invoked from repo root or any other cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.pre_flight_check import (  # noqa: E402
    EXPECTED_GATE_COUNTS,
    PIPELINES,
    build_gates,
)


def test_expected_gate_counts_all_matches_build_gates() -> None:
    """EXPECTED_GATE_COUNTS["all"] must equal len(build_gates("all")).

    Hard invariant: drift in either direction is an honest-negative.
    """
    actual_all = len(build_gates("all"))
    expected_all = EXPECTED_GATE_COUNTS["all"]
    assert expected_all == actual_all, (
        f"EXPECTED_GATE_COUNTS['all'] drift: expected={expected_all} "
        f"actual={actual_all}. Update EXPECTED_GATE_COUNTS in "
        f"molmetal/scripts/pre_flight_check.py AND "
        f"TODO/completed/352a1_expected_gate_counts_bump.md together "
        f"(per Wf-321D §5.2 design contract)."
    )


def test_expected_gate_counts_per_pipeline_consistency() -> None:
    """Every PIPELINES key must have EXPECTED_GATE_COUNTS[k] == build_gates(k)."""
    assert set(PIPELINES) == set(EXPECTED_GATE_COUNTS.keys()), (
        f"PIPELINES != EXPECTED_GATE_COUNTS keys: "
        f"PIPELINES={sorted(PIPELINES)}, "
        f"EXPECTED={sorted(EXPECTED_GATE_COUNTS.keys())}. "
        f"Both must enumerate the same pipeline universe per "
        f"Wf-321D §5.2 design contract."
    )
    for pipeline in sorted(PIPELINES):
        expected = EXPECTED_GATE_COUNTS[pipeline]
        actual = len(build_gates(pipeline))
        assert expected == actual, (
            f"EXPECTED_GATE_COUNTS[{pipeline!r}] drift: "
            f"expected={expected} actual={actual}. Update "
            f"EXPECTED_GATE_COUNTS in molmetal/scripts/pre_flight_check.py "
            f"AND TODO/completed/352a1_expected_gate_counts_bump.md together."
        )


if __name__ == "__main__":
    # Allow direct invocation via `python test_wf352a1_expected_gate_counts.py`.
    raise SystemExit(pytest.main([__file__, "-v"]))