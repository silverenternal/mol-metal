"""Tests for Wf-320E downstream lift dashboard.

Wf-320E builds the per-TODO lift table for the 6 strengthen-TODO consumers
that consume M1 (Wasserstein regularizer) + M2 (Catherine closure-bounded).
This dashboard is aggregation-only; no new measurements.

Per party-twenty-spirit §3.5 honest framing: per-consumer lift is per-TODO
deliverable (Q4 2026 / R20 GPU exercise). The dashboard surfaces HONEST_
NEGATIVE entries without silently masking them.

Test layout (12 tests):
  - 6 per-consumer schema validation tests
  - 3 lift calculation arithmetic tests
  - 2 status classification tests
  - 1 dashboard schema validation test
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf320e_downstream_lift_dashboard"
    / "dashboard.json"
)

EXPECTED_TODO_IDS = {"TODO-119", "TODO-120", "TODO-121", "TODO-122", "TODO-123", "TODO-125"}
EXPECTED_BADGES = {"MEASURED", "HONEST_NEGATIVE"}
EXPECTED_STATUSES = {"MEASURED", "PARTIAL", "PROJECTED", "HONEST_NEGATIVE"}


@pytest.fixture(scope="module")
def dashboard() -> dict:
    """Load the dashboard JSON once per module."""
    assert DASHBOARD_JSON.is_file(), f"dashboard.json missing at {DASHBOARD_JSON}"
    with open(DASHBOARD_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 6 per-consumer schema validation tests
# ---------------------------------------------------------------------------


def test_todo119_m14_decode_ratio_consumer_schema(dashboard: dict) -> None:
    """TODO-119 m14 cfm_decode_ratio consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-119")
    assert row["consumer_metric"] == "m14_cfm_decode_ratio"
    assert row["badge"] in EXPECTED_BADGES
    assert row["status"] in EXPECTED_STATUSES
    assert "engine_tags" in row and isinstance(row["engine_tags"], list)
    assert "yuelbond_decoder" in row["engine_tags"]
    assert "wasserstein_reg" in row["engine_tags"]
    assert "closure_catherine_bounded" in row["engine_tags"]
    assert isinstance(row["baseline_value"], (int, float))
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    assert row["honest_framing_note"], "HONEST_NEGATIVE requires honest_framing_note"
    assert "Wf-274A" in row["honest_framing_note"] or "regression_disclosure" in row["honest_framing_note"]


def test_todo120_m1_ru_auc_consumer_schema(dashboard: dict) -> None:
    """TODO-120 m1 Ru AUC consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-120")
    assert row["consumer_metric"] == "m1_Ru_AUC"
    assert row["badge"] == "MEASURED"
    assert row["status"] == "MEASURED"
    assert isinstance(row["baseline_value"], (int, float))
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    assert row["baseline_value"] < row["m1m2_enabled_value"], (
        "MEASURED positive lift requires m1m2 > baseline"
    )
    assert row["gate_hit"] is True, "MEASURED lift should hit internal 0.7 gate"
    assert row["krasnov_gate_hit"] is False, "Krasnov 0.81 gate NOT yet hit (honest)"


def test_todo121_m19_lomo_rh_consumer_schema(dashboard: dict) -> None:
    """TODO-121 m19 LOMO Rh bottleneck consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-121")
    assert row["consumer_metric"] == "m19_LOMO_Rh_bottleneck"
    assert row["badge"] in EXPECTED_BADGES
    assert row["status"] in EXPECTED_STATUSES
    assert isinstance(row["baseline_value"], (int, float))
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    assert row["m1m2_enabled_value"] < row["baseline_value"], (
        "HONEST_NEGATIVE: BELOW_RANDOM test_auc < 0.5 baseline"
    )
    assert row["gate_hit"] is False
    assert "BELOW_RANDOM" in row["honest_framing_note"]


def test_todo122_l5_hilbert_r2_consumer_schema(dashboard: dict) -> None:
    """TODO-122 L5 Hilbert R² ≥ 0.85 consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-122")
    assert row["consumer_metric"] == "L5_Hilbert_R2_4cell"
    assert row["badge"] in EXPECTED_BADGES
    assert row["status"] in EXPECTED_STATUSES
    assert row["baseline_value"] == 0.0, "signal-floor baseline is 0.0 by construction"
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    assert row["m1m2_enabled_value"] <= 0, (
        "R² on signal-floor regime cannot exceed 0 by 12-d RDKit |ρ| ≤ 0.019 audit"
    )
    assert "ARCHITECTURAL_CEILING" in row["honest_framing_note"]
    assert row["gate_hit"] is False


def test_todo123_m15_rmsd_consumer_schema(dashboard: dict) -> None:
    """TODO-123 m15 RMSD docked pose consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-123")
    assert row["consumer_metric"] == "m15_RMSD_docked_pose"
    assert row["badge"] == "MEASURED"
    assert row["status"] == "MEASURED"
    assert isinstance(row["baseline_value"], (int, float))
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    # For RMSD lower-is-better: m1m2 < baseline indicates improvement
    assert row["m1m2_enabled_value"] < row["baseline_value"]
    assert row["m1m2_enabled_value"] < 2.0, "5/5 pockets < 2.0 Å SOTA gate"
    assert row["gate_hit"] is True
    assert row["n_pockets_passed_sota"] == 5
    assert row["n_pockets_total"] == 5


def test_todo125_l12_per_pocket_vina_consumer_schema(dashboard: dict) -> None:
    """TODO-125 L12 OOD per-pocket Vina consumer has valid schema."""
    row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-125")
    assert row["consumer_metric"] == "L12_OOD_per_pocket_Vina"
    assert row["badge"] in EXPECTED_BADGES
    assert row["status"] in EXPECTED_STATUSES
    assert isinstance(row["baseline_value"], (int, float))
    assert isinstance(row["m1m2_enabled_value"], (int, float))
    assert row["n_pockets_passed_uniform_floor"] == 1, (
        "only 1h36 PASSES the 0.30 uniform floor; 4/5 pockets honest-negative"
    )
    assert row["n_pockets_total"] == 5
    assert "STUB_FALLBACK" in str(row.get("engine_tags", [])).upper(), (
        "Wf-248C per-pocket_vina.csv STUB_FALLBACK must be in engine_tags"
    )
    assert "stub_fallback_unrun_for_5pocket_wf248c" in row["engine_tags"]


# ---------------------------------------------------------------------------
# 3 lift calculation arithmetic tests
# ---------------------------------------------------------------------------


def test_lift_abs_calculation_matches_formula(dashboard: dict) -> None:
    """Per-consumer lift_abs equals m1m2_enabled_value - baseline_value."""
    for row in dashboard["per_consumer"]:
        expected_lift_abs = row["m1m2_enabled_value"] - row["baseline_value"]
        assert math.isclose(row["lift_abs"], expected_lift_abs, abs_tol=1e-9), (
            f"{row['todo_id']}: lift_abs {row['lift_abs']} != "
            f"{row['m1m2_enabled_value']} - {row['baseline_value']}"
        )


def test_lift_pct_calculation_matches_formula(dashboard: dict) -> None:
    """Per-consumer lift_pct equals (m1m2 - baseline) / baseline * 100.

    For the signal-floor baseline (TODO-122 baseline=0.0), lift_pct is
    mathematically undefined; the dashboard records None.
    """
    for row in dashboard["per_consumer"]:
        baseline = row["baseline_value"]
        m1m2 = row["m1m2_enabled_value"]
        if baseline == 0.0:
            assert row["lift_pct"] is None, (
                f"{row['todo_id']}: signal-floor baseline → lift_pct must be None"
            )
            continue
        expected_lift_pct = (m1m2 - baseline) / baseline * 100.0
        if math.isinf(expected_lift_pct):
            assert math.isinf(row["lift_pct"]), f"{row['todo_id']}: lift_pct should be inf"
        else:
            # Use 1e-2 tolerance to accept rounded JSON values (e.g. 3.473 ≈ 3.473302)
            assert math.isclose(row["lift_pct"], expected_lift_pct, abs_tol=1e-2, rel_tol=1e-3), (
                f"{row['todo_id']}: lift_pct {row['lift_pct']} != "
                f"computed {expected_lift_pct}"
            )


def test_lift_sign_convention_for_lower_is_better_metrics(dashboard: dict) -> None:
    """For RMSD (lower better), NEGATIVE lift_pct = improvement.

    The dashboard must carry the sign convention explicitly. Per the
    lift_calculation_notes block: 'positive lift = improvement; for
    RMSD/Vina-receptor-error metrics, NEGATIVE lift = IMPROVEMENT (lower
    is better)'. TODO-123 (RMSD) and TODO-125 (Vina-relaxed-success-rate
    with different scale) verify the convention.
    """
    rmse_row = next(c for c in dashboard["per_consumer"] if c["todo_id"] == "TODO-123")
    assert rmse_row["lift_pct"] < 0, (
        "TODO-123 RMSD: lower is better → NEGATIVE lift_pct expected"
    )
    assert rmse_row["badge"] == "MEASURED", (
        "NEGATIVE RMSD lift is the IMPROVEMENT direction → MEASURED badge"
    )

    # Verify the lift_calculation_notes block documents the sign convention
    notes = dashboard.get("lift_calculation_notes", {})
    assert "lift_sign_convention" in notes
    assert "lower is better" in notes["lift_sign_convention"].lower()


# ---------------------------------------------------------------------------
# 2 status classification tests
# ---------------------------------------------------------------------------


def test_status_classification_rules_are_explicit(dashboard: dict) -> None:
    """Status classification rules block must enumerate MEASURED/PARTIAL/PROJECTED/HONEST_NEGATIVE."""
    notes = dashboard.get("lift_calculation_notes", {})
    rules = notes.get("status_classification_rules", {})
    for expected in EXPECTED_STATUSES:
        assert expected in rules, f"missing status rule: {expected}"
    # Each rule must be a non-empty string
    for status, rule in rules.items():
        assert isinstance(rule, str) and rule.strip(), (
            f"status rule {status!r} must be non-empty string"
        )


def test_status_counts_match_n_consumers_tracked(dashboard: dict) -> None:
    """Top-level n_me + n_partial + n_projected + n_honest_negative must sum to n_consumers_tracked.

    The status field is independent of the badge field (status reflects the
    methodology of the comparison; badge reflects the verdict). Both must
    match their respective categorical counts.
    """
    n_total = dashboard["n_consumers_tracked"]
    n_sum = (
        dashboard["n_me"]
        + dashboard["n_partial"]
        + dashboard["n_projected"]
        + dashboard["n_honest_negative"]
    )
    assert n_total == n_sum, (
        f"n_consumers_tracked={n_total} != "
        f"n_me + n_partial + n_projected + n_honest_negative = {n_sum}"
    )
    # Per-consumer badge must agree with the categorical counts
    badge_counts = {"MEASURED": 0, "HONEST_NEGATIVE": 0}
    status_counts = {"MEASURED": 0, "PARTIAL": 0, "PROJECTED": 0, "HONEST_NEGATIVE": 0}
    for row in dashboard["per_consumer"]:
        if row["badge"] in badge_counts:
            badge_counts[row["badge"]] += 1
        status_counts[row["status"]] += 1
    assert dashboard["n_me"] == status_counts["MEASURED"]
    assert dashboard["n_partial"] == status_counts["PARTIAL"]
    assert dashboard["n_projected"] == status_counts["PROJECTED"]
    assert dashboard["n_honest_negative"] == status_counts["HONEST_NEGATIVE"]
    # Badge counts have separate fields
    assert dashboard["n_me_badge"] == badge_counts["MEASURED"]
    assert dashboard["n_honest_negative_badge"] == badge_counts["HONEST_NEGATIVE"]


# ---------------------------------------------------------------------------
# 1 dashboard schema validation test
# ---------------------------------------------------------------------------


def test_dashboard_schema_validation(dashboard: dict) -> None:
    """Validate the top-level dashboard schema fields."""
    # Required keys
    for key in (
        "dashboard_id",
        "title",
        "schema_version",
        "generated_utc",
        "wf_tag",
        "honest_framing_disclaimer",
        "n_consumers_tracked",
        "per_consumer",
        "summary_status_table",
        "lift_calculation_notes",
        "wf_follow_up_summary",
        "integration_notes",
    ):
        assert key in dashboard, f"missing top-level key: {key}"

    assert dashboard["dashboard_id"] == "wf320e_downstream_lift_dashboard"
    assert dashboard["schema_version"] == "320e.v1"
    assert dashboard["wf_tag"] == "Wf-320E"

    # per_consumer must have all 6 expected TODO ids
    todo_ids = {row["todo_id"] for row in dashboard["per_consumer"]}
    assert todo_ids == EXPECTED_TODO_IDS, (
        f"per_consumer TODO ids {todo_ids} != expected {EXPECTED_TODO_IDS}"
    )

    # summary_status_table must have a row per consumer (6 + header = 7 rows)
    assert len(dashboard["summary_status_table"]) == 7, (
        "summary_status_table must have header + 6 consumer rows"
    )

    # honest_framing_disclaimer must mention party-twenty-spirit §3.5
    assert "party-twenty-spirit" in dashboard["honest_framing_disclaimer"]

    # wf_follow_up_summary must enumerate pending GPU launches
    follow_up = dashboard["wf_follow_up_summary"]
    assert follow_up["n_pending_gpu_launches"] >= 1
    assert "wall_budget_total_hr" in follow_up

    # integration_notes must reflect 6 claim cards + 12 tests
    integration = dashboard["integration_notes"]
    assert integration["paper_dashboard_claim_cards_added"] == 6
    assert integration["tests_added"] == 12
