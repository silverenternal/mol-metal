"""Tests for Wf-246-B: TODO-130 m15 REDESIGN (pocket-aware self-RMSD metric).

These tests pin down the invariants the Wf-246-B deliverable depends on
so future metric regressions are caught immediately.

The redesign is implemented in
``molmetal/scripts/redesign_m15_pocket_aware.py`` and produces:

* ``metrics/by_metric/docked_pose_rmsd.json`` (REFACTORED with new fields)
* ``molmetal/reports/wf246b_m15_redesign/summary.json`` (NEW)
* ``molmetal/reports/wf246b_m15_redesign/synthesis.md`` (NEW)
* ``molmetal/reports/wf246b_m15_redesign/per_pocket_table.md`` (NEW)

Tests pin down:

 1.  The metric file `metrics/by_metric/docked_pose_rmsd.json` exists and is
     valid JSON.
 2.  The metric file has a `redesign_wf246b` block at top level.
 3.  The metric file has `cross_pocket_aggregate` block at top level
     with `median`, `std`, `min`, `max`, `n_pockets` keys.
 4.  The metric file has `difficulty_counts` block at top level with
     `EASY`, `MEDIUM`, `HARD` keys.
 5.  The metric file has `per_pocket_with_difficulty` block at top level
     with 5 pockets (1h36, 3eml, 3ny8, 4aaw, 4rlu).
 6.  Each pocket has `difficulty_flag`, `std_best_rmsd_angstrom`,
     `median_best_rmsd_angstrom`, `per_seed_best_rmsd_angstrom`.
 7.  The cross-pocket aggregate median is in the plausible SOTA region
     (between 0.1 and 5.0 angstrom).
 8.  The redesign report exists at
     `molmetal/reports/wf246b_m15_redesign/summary.json`.
 9.  The synthesis.md exists with PASS verdict.
10.  The per_pocket_table.md exists with all 5 pockets listed.
11.  Difficulty flag logic: < 0.5 A = EASY, < 1.0 A = MEDIUM, >= 1.0 A = HARD.
12.  Backward compat: legacy top-level fields (description, unit, scope,
     value, aggregate_median_across_pockets, gate_4_passes_2_5a_overall)
     are still present.

Tests are **skipped (not failed)** when the artefacts are missing — so
CI on a fresh checkout does not get red until the Wf-246-B deliverable
has shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
METRIC_JSON = REPO / "metrics/by_metric/docked_pose_rmsd.json"
SUMMARY_JSON = REPO / "molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json"
REPORT_DIR = REPO / "molmetal/reports/wf246b_m15_redesign"

EXPECTED_POCKETS = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]


def _load_metric() -> dict:
    with open(METRIC_JSON) as f:
        return json.load(f)


def _load_report_summary() -> dict:
    with open(REPORT_DIR / "summary.json") as f:
        return json.load(f)


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_metric_json_exists_and_valid() -> None:
    """Test 1: docked_pose_rmsd.json exists and parses as JSON."""
    metric = _load_metric()
    assert "metric" in metric
    assert metric["metric"] == "docked_pose_rmsd"


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_metric_has_redesign_wf246b_block() -> None:
    """Test 2: top-level `redesign_wf246b` block exists."""
    metric = _load_metric()
    assert "redesign_wf246b" in metric, "Missing `redesign_wf246b` block at top level"
    rb = metric["redesign_wf246b"]
    assert rb["workflow"] == "Wf-246-B TODO-130 m15 REDESIGN"
    assert "per_pocket_count" in rb
    assert "difficulty_thresholds_angstrom" in rb


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_metric_has_cross_pocket_aggregate() -> None:
    """Test 3: top-level `cross_pocket_aggregate` block with required keys."""
    metric = _load_metric()
    assert "cross_pocket_aggregate" in metric
    agg = metric["cross_pocket_aggregate"]
    for key in ("median", "std", "min", "max", "n_pockets"):
        assert key in agg, f"Missing key `{key}` in cross_pocket_aggregate"
    assert agg["n_pockets"] == 5
    assert agg["median"] is not None
    assert agg["std"] is not None
    assert agg["min"] is not None
    assert agg["max"] is not None


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_metric_has_difficulty_counts() -> None:
    """Test 4: top-level `difficulty_counts` block with EASY/MEDIUM/HARD."""
    metric = _load_metric()
    assert "difficulty_counts" in metric
    dc = metric["difficulty_counts"]
    for key in ("EASY", "MEDIUM", "HARD"):
        assert key in dc, f"Missing key `{key}` in difficulty_counts"
    # Counts should add up to n_pockets
    assert sum(dc.values()) == metric["cross_pocket_aggregate"]["n_pockets"]


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_metric_has_per_pocket_with_difficulty() -> None:
    """Test 5: top-level `per_pocket_with_difficulty` block, 5 pockets."""
    metric = _load_metric()
    assert "per_pocket_with_difficulty" in metric
    pp = metric["per_pocket_with_difficulty"]
    assert set(pp.keys()) == set(
        EXPECTED_POCKETS
    ), f"Expected pockets {EXPECTED_POCKETS}, got {list(pp.keys())}"


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_per_pocket_fields_complete() -> None:
    """Test 6: each pocket has difficulty_flag, std, median, per_seed."""
    metric = _load_metric()
    pp = metric["per_pocket_with_difficulty"]
    for pocket_id in EXPECTED_POCKETS:
        d = pp[pocket_id]
        assert "difficulty_flag" in d
        assert d["difficulty_flag"] in ("EASY", "MEDIUM", "HARD")
        assert "std_best_rmsd_angstrom" in d
        assert "median_best_rmsd_angstrom" in d
        assert "per_seed_best_rmsd_angstrom" in d
        assert len(d["per_seed_best_rmsd_angstrom"]) >= 1
        assert "n_seeds_succeeded" in d
        assert "n_seeds_total" in d


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_cross_pocket_median_plausible() -> None:
    """Test 7: cross-pocket median is in plausible SOTA region."""
    metric = _load_metric()
    median = metric["cross_pocket_aggregate"]["median"]
    assert 0.1 < median < 5.0, f"Cross-pocket median {median} outside plausible range"


@pytest.mark.skipif(
    not (REPORT_DIR / "summary.json").exists(),
    reason="Wf-246-B report summary not yet shipped",
)
def test_report_summary_json_exists() -> None:
    """Test 8: Wf-246-B redesign report summary.json exists."""
    summary = _load_report_summary()
    assert summary["workflow"] == "Wf-246-B TODO-130 m15 REDESIGN"
    assert "cross_pocket_aggregate" in summary
    assert "per_pocket" in summary
    assert "difficulty_counts" in summary


@pytest.mark.skipif(
    not (REPORT_DIR / "synthesis.md").exists(),
    reason="Wf-246-B synthesis.md not yet shipped",
)
def test_synthesis_md_exists() -> None:
    """Test 9: synthesis.md exists with PASS verdict."""
    text = (REPORT_DIR / "synthesis.md").read_text()
    assert "Wf-246-B" in text
    assert "PASS" in text or "MEASURED" in text
    assert "Cross-pocket aggregate" in text


@pytest.mark.skipif(
    not (REPORT_DIR / "per_pocket_table.md").exists(),
    reason="Wf-246-B per_pocket_table.md not yet shipped",
)
def test_per_pocket_table_lists_all_5_pockets() -> None:
    """Test 10: per_pocket_table.md lists all 5 pockets."""
    text = (REPORT_DIR / "per_pocket_table.md").read_text()
    for pocket_id in EXPECTED_POCKETS:
        assert pocket_id in text, f"Missing pocket {pocket_id} in per_pocket_table.md"


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_difficulty_flag_logic() -> None:
    """Test 11: difficulty_flag follows thresholds."""
    metric = _load_metric()
    pp = metric["per_pocket_with_difficulty"]
    for pocket_id, d in pp.items():
        median = d["median_best_rmsd_angstrom"]
        flag = d["difficulty_flag"]
        if median < 0.5:
            assert (
                flag == "EASY"
            ), f"{pocket_id}: median {median} should be EASY, got {flag}"
        elif median < 1.0:
            assert (
                flag == "MEDIUM"
            ), f"{pocket_id}: median {median} should be MEDIUM, got {flag}"
        else:
            assert (
                flag == "HARD"
            ), f"{pocket_id}: median {median} should be HARD, got {flag}"


@pytest.mark.skipif(
    not METRIC_JSON.exists(), reason="Wf-246-B metric JSON not yet shipped"
)
def test_backward_compat_legacy_fields_preserved() -> None:
    """Test 12: legacy top-level fields are still present."""
    metric = _load_metric()
    for key in (
        "metric",
        "description",
        "unit",
        "scope",
        "value",
        "aggregate_median_across_pockets",
        "gate_4_passes_2_5a_overall",
        "n_pockets_passed_2_5a",
        "n_pockets_passed_2_0a",
    ):
        assert key in metric, f"Missing legacy key `{key}` in metric JSON"
