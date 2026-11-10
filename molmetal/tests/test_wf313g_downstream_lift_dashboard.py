"""Wf-313G — Downstream lift dashboard tests for 6 Wf-123 P3 M1+M2 strengthen-TODO consumers.

====================================================================
Tests for the Wf-313G aggregation dashboard that tracks the
per-consumer MEASURED state for TODO-119/120/121/122/123/125 (the 6
strengthen-TODO consumers wired into the Wf-123 P3 M1+M2 integration
layer SHIP).

This is **aggregation-only** work per party-twenty-spirit §3.5:
the dashboard JSON is a metadata artefact; no new measurements are
produced. Tests verify:

* Per-consumer schema validation (6 tests, one per consumer)
* Lift calculation arithmetic (3 tests)
* Status classification (MEASURED / MEASURED_PARTIAL / HONEST_NEGATIVE) (2 tests)
* Dashboard schema validation (1 test)

Total: 12 tests.

Honest framing (party-twenty-spirit §3.5):
Per-consumer lift is per-TODO deliverable (Q4 2026 / R20 GPU exercise).
Some consumers will show HONEST_NEGATIVE (TODO-122 R² ≤ 0 by
construction; TODO-121 m19 LOMO Rh fold absent; TODO-125 L12 per-pocket
OOD Vina files not on disk). TODO-121 m19 LOMO Rh 0.6220 is on the
overall target 0.6 (incidental hit) but the Rh-specific target ≥ 0.5
is NOT measured (Rh fold absent from per_fold_auc_mean).

Source: TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md §1+§2
+ molmetal/reports/wf313g_downstream_lift_dashboard/dashboard.json

Run via::

    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf313g_downstream_lift_dashboard.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


DASHBOARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "molmetal"
    / "reports"
    / "wf313g_downstream_lift_dashboard"
    / "dashboard.json"
)

EXPECTED_TODOS: tuple = (
    "TODO-119",
    "TODO-120",
    "TODO-121",
    "TODO-122",
    "TODO-123",
    "TODO-125",
)
EXPECTED_CONSUMER_METRICS: tuple = (
    "m14_decode_ratio",
    "m1_ru_auc",
    "m19_lomo_rh_bottleneck",
    "L5_hilbert_r2_4cell",
    "m15_rmsd_redocking",
    "L12_ood_per_pocket_vina",
)
EXPECTED_STATUS_VALUES: frozenset = frozenset(
    {"MEASURED_PASS", "MEASURED_PARTIAL", "HONEST_NEGATIVE", "PROJECTED"}
)


@pytest.fixture(scope="module")
def dashboard() -> dict:
    """Load the Wf-313G downstream lift dashboard JSON."""
    assert DASHBOARD_PATH.exists(), f"dashboard missing: {DASHBOARD_PATH}"
    with DASHBOARD_PATH.open("r") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def consumer_index(dashboard: dict) -> dict:
    """Map todo_id -> consumer dict from the dashboard."""
    return {c["todo_id"]: c for c in dashboard["consumers"]}


# ---------------------------------------------------------------------------
# Tests 1-6: per-consumer schema validation (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "todo_id,metric",
    [
        ("TODO-119", "m14_decode_ratio"),
        ("TODO-120", "m1_ru_auc"),
        ("TODO-121", "m19_lomo_rh_bottleneck"),
        ("TODO-122", "L5_hilbert_r2_4cell"),
        ("TODO-123", "m15_rmsd_redocking"),
        ("TODO-125", "L12_ood_per_pocket_vina"),
    ],
)
def test_per_consumer_schema(todo_id: str, metric: str, consumer_index: dict) -> None:
    """Each of the 6 strengthen-TODO consumers must carry the required schema."""
    consumer = consumer_index[todo_id]
    assert consumer["todo_id"] == todo_id
    assert consumer["consumer_metric"] == metric
    # Required fields per consumer schema
    assert "baseline_value" in consumer
    assert "baseline_source" in consumer
    assert "measured_source" in consumer
    assert "status" in consumer
    assert consumer["status"] in EXPECTED_STATUS_VALUES
    # Lift arithmetic (if both baseline and measured are numeric)
    if (
        consumer["baseline_value"] is not None
        and consumer["measured_value"] is not None
    ):
        assert "lift_abs" in consumer
        assert "lift_pct" in consumer
        # Lift pct must be computed correctly:
        # lift_pct = 100 * (measured - baseline) / baseline
        baseline = consumer["baseline_value"]
        measured = consumer["measured_value"]
        expected_pct = 100.0 * (measured - baseline) / baseline
        # Compare with small tolerance for floating-point
        assert math.isclose(
            consumer["lift_pct"], expected_pct, rel_tol=1e-3, abs_tol=1e-6
        ), f"{todo_id}: lift_pct={consumer['lift_pct']} vs expected={expected_pct}"
    # Consumer follow-up must be documented
    assert "consumer_followup_wf" in consumer
    assert "wall_time_estimate_hr" in consumer
    assert "ready_for_launch" in consumer


# ---------------------------------------------------------------------------
# Tests 7-9: lift calculation arithmetic (3 tests)
# ---------------------------------------------------------------------------


def test_lift_calc_ru_auc_over_single_task_baseline(consumer_index: dict) -> None:
    """TODO-120 lift_abs = 0.7860 - 0.7123 = 0.0737; lift_pct = 10.35%."""
    c = consumer_index["TODO-120"]
    baseline = c["baseline_value"]
    measured = c["measured_value"]
    expected_abs = measured - baseline
    expected_pct = 100.0 * (measured - baseline) / baseline
    assert math.isclose(c["lift_abs"], expected_abs, rel_tol=1e-3, abs_tol=1e-6)
    assert math.isclose(c["lift_pct"], expected_pct, rel_tol=1e-3, abs_tol=1e-6)
    # Specific check: 0.7860 vs 0.7123 = 10.35% positive lift
    assert c["lift_pct"] > 10.0
    assert c["lift_pct"] < 11.0


def test_lift_calc_hilbert_r2_negative_delta(consumer_index: dict) -> None:
    """TODO-122 lift_abs = -5.66 - (-1.50) = -4.16 (post-fix R² worse than pre-fix)."""
    c = consumer_index["TODO-122"]
    baseline = c["baseline_value"]
    measured = c["measured_value"]
    expected_abs = measured - baseline
    expected_pct = 100.0 * (measured - baseline) / baseline
    assert math.isclose(c["lift_abs"], expected_abs, rel_tol=1e-3, abs_tol=1e-6)
    assert math.isclose(c["lift_pct"], expected_pct, rel_tol=1e-3, abs_tol=1e-6)
    # Specific check: post-fix R² is worse than pre-fix baseline, but both are
    # negative, so lift_pct is positive (denominator -1.50 flips sign).
    assert c["lift_abs"] < -4.0
    assert c["lift_pct"] > 200.0
    assert c["lift_pct"] < 300.0


def test_lift_calc_m15_rmsd_relative_to_gate(consumer_index: dict) -> None:
    """TODO-123 m15 RMSD: measured 0.6578 Å vs 2.0 Å SOTA gate = -67%."""
    c = consumer_index["TODO-123"]
    assert c["measured_value"] is not None
    assert c["gate_target_value"] == 2.0
    # Lift vs gate: 0.6578 vs 2.0 = -67.11%
    assert math.isclose(c["lift_vs_gate_pct"], -67.11, rel_tol=1e-3, abs_tol=0.01)
    # m15 should be PASS (5/5 pockets)
    assert c["gate_passed"] is True
    assert c["n_pockets_passed_2_0a"] == 5


# ---------------------------------------------------------------------------
# Tests 10-11: status classification (2 tests)
# ---------------------------------------------------------------------------


def test_status_classification_measured_vs_projected(
    dashboard: dict, consumer_index: dict
) -> None:
    """Verify status values match MEASURED/PROJECTED honest framing.

    Expected:
        TODO-119 -> MEASURED_PARTIAL
        TODO-120 -> MEASURED_PARTIAL
        TODO-121 -> HONEST_NEGATIVE (Rh fold absent)
        TODO-122 -> HONEST_NEGATIVE (architectural ceiling)
        TODO-123 -> MEASURED_PASS
        TODO-125 -> HONEST_NEGATIVE (no per-pocket OOD data)
    """
    expected = {
        "TODO-119": "MEASURED_PARTIAL",
        "TODO-120": "MEASURED_PARTIAL",
        "TODO-121": "HONEST_NEGATIVE",
        "TODO-122": "HONEST_NEGATIVE",
        "TODO-123": "MEASURED_PASS",
        "TODO-125": "HONEST_NEGATIVE",
    }
    for todo_id, exp_status in expected.items():
        actual = consumer_index[todo_id]["status"]
        assert actual == exp_status, f"{todo_id}: expected {exp_status}, got {actual}"

    # Count HONEST_NEGATIVE consumers
    hn = [
        tid
        for tid in EXPECTED_TODOS
        if consumer_index[tid]["status"] == "HONEST_NEGATIVE"
    ]
    assert len(hn) == 3
    # Dashboard-level counts
    assert dashboard["n_honest_negatives"] == 3
    assert dashboard["n_consumers_honest_negative"] == 3
    assert dashboard["n_consumers_tracked"] == 6


def test_status_classification_status_detail_required(dashboard: dict) -> None:
    """Every consumer must have a non-empty status_detail field documenting why."""
    for c in dashboard["consumers"]:
        assert "status_detail" in c, f"{c.get('todo_id')} missing status_detail"
        assert isinstance(c["status_detail"], str)
        assert len(c["status_detail"]) > 50, (
            f"{c.get('todo_id')} status_detail too short ({len(c['status_detail'])} chars); "
            "must document the honest framing per party-twenty-spirit §3.5"
        )


# ---------------------------------------------------------------------------
# Test 12: dashboard schema validation (1 test)
# ---------------------------------------------------------------------------


def test_dashboard_schema_validation(dashboard: dict) -> None:
    """Top-level dashboard JSON schema must match the Wf-313G contract."""
    # Required top-level fields
    required = {
        "dashboard_id",
        "workflow",
        "purpose",
        "parent_todo",
        "honest_framing_policy",
        "generated_utc",
        "n_consumers_tracked",
        "consumers",
        "status_summary_table",
        "engine_tag_invariant",
        "wf123_p3_ship_status",
        "honest_negative_closeout_precedent",
        "no_best_seed_selection",
        "no_paper_work_disclosure",
        "pending_actions",
    }
    missing = required - set(dashboard.keys())
    assert not missing, f"missing top-level fields: {missing}"

    # consumers list length = 6
    assert len(dashboard["consumers"]) == 6
    # status_summary_table length = 6
    assert len(dashboard["status_summary_table"]) == 6

    # Engine-tag invariant must be preserved (Wf-123 P3 ship contract)
    et = dashboard["engine_tag_invariant"]
    assert et["claim"] == "wasserstein_reg ∩ closure_catherine_bounded = ∅"
    assert et["source"].endswith("test_combined_engine_set_is_disjoint PASS")

    # Wf-123 P3 ship status: 25/25 integration tests PASS
    ship = dashboard["wf123_p3_ship_status"]
    assert "25/25 PASS" in ship["m1m2_cross_module_integration"]

    # All 6 expected TODOs are present in status_summary_table
    summary_todos = {row["todo_id"] for row in dashboard["status_summary_table"]}
    assert summary_todos == set(EXPECTED_TODOS)

    # All 6 expected consumer_metric strings are present
    summary_metrics = [row["metric"] for row in dashboard["status_summary_table"]]
    assert set(summary_metrics) == set(EXPECTED_CONSUMER_METRICS)

    # Honest-negative close-out precedent must cite Wf-86 + Wf-139 P4 + Wf-122 P3
    prec = dashboard["honest_negative_closeout_precedent"]
    assert "wf86_honest_negative_close" in prec
    assert "wf139_p4_l5_l7_l10_arch_ceiling_close" in prec
    assert "wf122_p3_4cell_combined_r2_close" in prec

    # No best-seed selection honored
    nbs = dashboard["no_best_seed_selection"]
    assert nbs["honored"] is True
    assert "no best-seed selection" in nbs["rule"].lower()

    # No paper work disclosure must be present (TODO-84 active)
    assert "TODO-84" in dashboard["no_paper_work_disclosure"]

    # All pending_actions must declare user_ok_required=True (kill_then_adapt directive)
    for action in dashboard["pending_actions"]:
        assert (
            action["user_ok_required"] is True
        ), f"pending_action {action['action']} must require user OK per kill_then_adapt §0"
