"""Wf-362-A3 — Refresh STALE docked_pose_rmsd.json (29h -> FRESH).

Per Wf-359-A3 finding: docked_pose_rmsd.json STALE ~29h. File measurements array
has date_utc 2026-09-28T02:42:18Z for all 5 pockets. Wf-341-T1 redesign integration
4/4 tests PASS. PoseBusters validity SKIPPED.

This test module verifies the refresh:
  1. docked_pose_rmsd.json includes a Wf-341-T1 redesign field
  2. median 0.6578 + 5/5 pockets pass matches Wf-341-T1 source

Per `party-twenty-spirit` Section 3.5: MEASURED values must trace to a JSON /
report path on disk. PROJECTED vs MEASURED tag is required.

Wf-362-A3 — 2026-09-29 — STALE -> FRESH refresh.
"""

from __future__ import annotations

import datetime
import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RMSD_JSON = os.path.join(REPO_ROOT, "metrics", "by_metric", "docked_pose_rmsd.json")
WF341T1_DOC = os.path.join(REPO_ROOT, "TODO", "completed", "341t1_m15_rmsd.md")


def _load_rmsd_json() -> dict:
    """Load the docked_pose_rmsd.json file."""
    with open(RMSD_JSON) as f:
        return json.load(f)


def test_docked_pose_rmsd_json_includes_wf341t1_redesign_field() -> None:
    """docked_pose_rmsd.json must include the redesign_wf341t1 field added by Wf-362-A3.

    The field should:
      - be named ``redesign_wf341t1`` (matching the Wf-341-T1 workflow name)
      - carry a ``source`` pointer to TODO/completed/341t1_m15_rmsd.md
      - carry ``updated_at: 2026-09-29`` (the Wf-362-A3 refresh date)
      - have ``status: MEASURED`` (per party-twenty-spirit Section 3.5)
    """
    d = _load_rmsd_json()
    assert "redesign_wf341t1" in d, (
        "docked_pose_rmsd.json missing redesign_wf341t1 field; "
        "Wf-362-A3 should have added it to refresh the STALE file"
    )
    wf341t1 = d["redesign_wf341t1"]
    assert wf341t1["workflow"].startswith("Wf-341-T1"), (
        f"Unexpected workflow name: {wf341t1['workflow']}"
    )
    assert wf341t1["source"].endswith("341t1_m15_rmsd.md"), (
        f"Source should point to Wf-341-T1 doc: {wf341t1['source']}"
    )
    assert wf341t1["updated_at"] == "2026-09-29", (
        f"updated_at should be 2026-09-29, got {wf341t1['updated_at']}"
    )
    assert wf341t1["status"] == "MEASURED", (
        f"status should be MEASURED per party-twenty-spirit 3.5, got {wf341t1['status']}"
    )


def test_docked_pose_rmsd_json_median_5_of_5_matches_wf341t1_source() -> None:
    """Verify median 0.6578 + 5/5 pockets pass matches the Wf-341-T1 source.

    The Wf-341-T1 doc at TODO/completed/341t1_m15_rmsd.md asserts:
      - aggregate_median_best_rmsd_angstrom: 0.6578
      - n_pockets_passed_2_0a: 5/5
      - aggregate_success_rate_2_0a: 1.0
      - difficulty_counts: EASY=1, MEDIUM=3, HARD=1
      - posebusters_validity: SKIPPED (graceful, not installed)

    These values must appear in the redesign_wf341t1 block AND in the
    top-level aggregate fields of docked_pose_rmsd.json (cross-check).
    """
    d = _load_rmsd_json()
    assert "redesign_wf341t1" in d, "redesign_wf341t1 field missing"
    wf341t1 = d["redesign_wf341t1"]

    # Median
    assert wf341t1["aggregate_median_best_rmsd_angstrom"] == pytest.approx(0.6578, abs=1e-4), (
        f"median should be 0.6578, got {wf341t1['aggregate_median_best_rmsd_angstrom']}"
    )
    # 5/5 pass at 2.0 A SOTA gate
    assert wf341t1["n_pockets_passed_2_0a"] == 5, (
        f"n_pockets_passed_2_0a should be 5, got {wf341t1['n_pockets_passed_2_0a']}"
    )
    assert wf341t1["n_pockets_measured"] == 5, (
        f"n_pockets_measured should be 5, got {wf341t1['n_pockets_measured']}"
    )
    assert wf341t1["aggregate_success_rate_2_0a"] == pytest.approx(1.0, abs=1e-9), (
        f"aggregate_success_rate_2_0a should be 1.0, got {wf341t1['aggregate_success_rate_2_0a']}"
    )
    # Difficulty counts
    assert wf341t1["difficulty_counts"]["EASY"] == 1
    assert wf341t1["difficulty_counts"]["MEDIUM"] == 3
    assert wf341t1["difficulty_counts"]["HARD"] == 1
    # PoseBusters SKIPPED (honest-negative)
    assert wf341t1["posebusters_validity"] == "SKIPPED", (
        f"posebusters_validity should be SKIPPED, got {wf341t1['posebusters_validity']}"
    )

    # Cross-check against top-level aggregate_median_across_pockets
    # (the existing aggregate field must also agree).
    assert d["aggregate_median_across_pockets"] == pytest.approx(0.6578, abs=1e-4), (
        f"top-level aggregate_median_across_pockets should also be 0.6578, "
        f"got {d['aggregate_median_across_pockets']}"
    )

    # Wf-341-T1 doc must exist on disk (per party-twenty-spirit 3.5
    # MEASURED values must trace to a report path).
    assert os.path.exists(WF341T1_DOC), (
        f"Wf-341-T1 source doc not found on disk: {WF341T1_DOC}"
    )


if __name__ == "__main__":
    # Allow `python -m molmetal.tests.test_wf362a3_rmsd_json_refresh` for quick smoke
    import sys
    sys.exit(pytest.main([__file__, "-v"]))