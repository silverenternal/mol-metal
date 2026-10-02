"""Wf-362-A1 — Update metalcytotox_combined_r2.json with Wf-335-M2 CEILING_BREAK result.

Per Wf-359-A3 finding: ``metalcytotox_combined_r2.json`` did NOT include the
Wf-335-M2 CEILING_BREAK result (R² best = +0.2227, lift = +0.1172 over Wf-325A
baseline of +0.1055). Documentation gap closed by Wf-362-A1.

This test module verifies the update:
  1. metalcytotox_combined_r2.json includes a ``wf335m2_ceiling_break`` field
  2. r2_best + lift_vs_wf325a values match the Wf-335-M2 source on disk

Per ``party-twenty-spirit`` Section 3.5: MEASURED values must trace to a JSON /
report path on disk. PROJECTED vs MEASURED tag is required.

Wf-362-A1 — 2026-09-29 — documentation gap closed.
"""

from __future__ import annotations

import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2_JSON = os.path.join(
    REPO_ROOT, "metrics", "by_metric", "metalcytotox_combined_r2.json"
)
WF335M2_SOURCE = os.path.join(
    REPO_ROOT, "molmetal", "reports", "wf335m2_r2", "result.json"
)


def _load_r2_json() -> dict:
    """Load the metalcytotox_combined_r2.json file."""
    with open(R2_JSON) as f:
        return json.load(f)


def _load_wf335m2_source() -> dict:
    """Load the Wf-335-M2 result.json source-of-truth."""
    with open(WF335M2_SOURCE) as f:
        return json.load(f)


def test_metalcytotox_combined_r2_json_includes_wf335m2_ceiling_break_field() -> None:
    """metalcytotox_combined_r2.json must include the wf335m2_ceiling_break field.

    The field should:
      - be named ``wf335m2_ceiling_break`` (matching the Wf-335-M2 workflow)
      - carry a ``source`` pointer to molmetal/reports/wf335m2_r2/result.json
      - carry ``updated_at: 2026-09-29`` (the Wf-362-A1 refresh date)
      - have ``status: MEASURED`` (per party-twenty-spirit Section 3.5)
      - have ``verdict: CEILING_BREAK``
    """
    d = _load_r2_json()
    assert "wf335m2_ceiling_break" in d, (
        "metalcytotox_combined_r2.json missing wf335m2_ceiling_break field; "
        "Wf-362-A1 should have added it to close the documentation gap"
    )
    block = d["wf335m2_ceiling_break"]
    assert block["workflow"].startswith("Wf-335-M2"), (
        f"Unexpected workflow name: {block['workflow']}"
    )
    assert block["source"].endswith("wf335m2_r2/result.json"), (
        f"Source should point to Wf-335-M2 result.json: {block['source']}"
    )
    assert block["updated_at"] == "2026-09-29", (
        f"updated_at should be 2026-09-29, got {block['updated_at']}"
    )
    assert block["status"] == "MEASURED", (
        f"status should be MEASURED per party-twenty-spirit 3.5, "
        f"got {block['status']}"
    )
    assert block["verdict"] == "CEILING_BREAK", (
        f"verdict should be CEILING_BREAK, got {block['verdict']}"
    )
    assert block["cohort"] == "metalcytotox_combined", (
        f"cohort should be metalcytotox_combined, got {block['cohort']}"
    )


def test_metalcytotox_combined_r2_json_r2_best_and_lift_match_wf335m2_source() -> None:
    """Verify r2_best + lift_vs_wf325a values match the Wf-335-M2 source.

    The Wf-335-M2 result.json asserts (metalcytotox_combined best_config =
    full_static_combined_plus_w2_best_axis):
      - r2_best: +0.22268113216745766 (multi-seed CV mean)
      - lift_vs_wf325a_baseline: +0.11715430073107957
      - r2_std: 0.32098447223005655
      - wf325a_baseline_r2: 0.10552683143637809

    These values must appear in the wf335m2_ceiling_break block. Cross-check
    against the source-of-truth result.json on disk (party-twenty-spirit 3.5).
    """
    d = _load_r2_json()
    assert "wf335m2_ceiling_break" in d, "wf335m2_ceiling_break field missing"
    block = d["wf335m2_ceiling_break"]

    # r2_best
    assert block["r2_best"] == pytest.approx(0.22268113216745766, abs=1e-9), (
        f"r2_best should be 0.22268113216745766, got {block['r2_best']}"
    )
    # lift
    assert block["lift_vs_wf325a"] == pytest.approx(0.11715430073107957, abs=1e-9), (
        f"lift_vs_wf325a should be 0.11715430073107957 (+0.1172), "
        f"got {block['lift_vs_wf325a']}"
    )
    # std
    assert block["r2_std"] == pytest.approx(0.32098447223005655, abs=1e-9), (
        f"r2_std should be 0.32098447223005655, got {block['r2_std']}"
    )
    # wf325a baseline
    assert block["wf325a_baseline_r2"] == pytest.approx(0.10552683143637809, abs=1e-9), (
        f"wf325a_baseline_r2 should be 0.10552683143637809, "
        f"got {block['wf325a_baseline_r2']}"
    )

    # Cross-check against the Wf-335-M2 source-of-truth result.json on disk.
    # The metalcytotox_combined cohort's best_config must agree bit-for-bit.
    src = _load_wf335m2_source()
    cohort_block = None
    for c in src.get("per_cohort", []):
        if c.get("cohort") == "metalcytotox_combined":
            cohort_block = c
            break
    assert cohort_block is not None, (
        "Wf-335-M2 source missing metalcytotox_combined cohort entry"
    )
    src_best = cohort_block["best_config"]
    assert block["r2_best"] == pytest.approx(src_best["r2_mean"], abs=1e-12), (
        f"r2_best ({block['r2_best']}) does not match source "
        f"({src_best['r2_mean']})"
    )
    assert block["lift_vs_wf325a"] == pytest.approx(
        src_best["lift_vs_wf325a_baseline"], abs=1e-12
    ), (
        f"lift_vs_wf325a ({block['lift_vs_wf325a']}) does not match source "
        f"({src_best['lift_vs_wf325a_baseline']})"
    )

    # The source file must exist on disk (per party-twenty-spirit 3.5
    # MEASURED values must trace to a report path).
    assert os.path.exists(WF335M2_SOURCE), (
        f"Wf-335-M2 source result.json not found on disk: {WF335M2_SOURCE}"
    )


if __name__ == "__main__":
    # Allow `python -m molmetal.tests.test_wf362a1_r2_json_update` for quick smoke
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
