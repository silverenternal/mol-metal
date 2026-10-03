"""Wf-31 Phase 5 lift analysis tests.

These tests verify that:
1. The lift analysis JSON exists and is well-formed
2. The lift summary MD exists and references the JSON
3. All 3 R17 baselines are preserved in their respective metrics JSONs
4. All 3 metrics JSONs have a wf31_phase5_lift_update entry
5. No fabricated MEASURED headline values exist (m1 + m2 = null; m7 = real value with comparability caveat)
6. The R20 plan items are referenced (P2 + P3 + P8 NEW)
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIFT_ANALYSIS_JSON = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf31_m1_m2_m7_lift_analysis.json"
)
LIFT_SUMMARY_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf31_m1_m2_m7_lift_summary.md"
)
METRICS_RU = PROJECT_ROOT / "metrics" / "by_metric" / "metalcytotox_ru_auc.json"
METRICS_NCI60 = PROJECT_ROOT / "metrics" / "by_metric" / "nci60_precious_9cell_auc.json"
METRICS_DIVERSITY = PROJECT_ROOT / "metrics" / "by_metric" / "diversity_tanimoto.json"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    return path.read_text()


class TestLiftAnalysisJsonExists:
    def test_lift_analysis_json_exists(self):
        assert (
            LIFT_ANALYSIS_JSON.exists()
        ), f"Wf-31 Phase 5 lift analysis JSON not found at {LIFT_ANALYSIS_JSON}"

    def test_lift_analysis_json_is_valid_json(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert isinstance(data, dict)

    def test_lift_analysis_top_level_keys(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        for key in (
            "metric",
            "description",
            "generated_utc",
            "workflow",
            "phase",
            "metrics",
            "aggregate_lift_summary",
            "honest_negative_provenance",
            "r20_plan_summary",
            "ship_ledger",
        ):
            assert key in data, f"missing top-level key {key}"

    def test_lift_analysis_has_three_metrics(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert len(data["metrics"]) == 3
        metric_ids = [m["metric_id"] for m in data["metrics"]]
        assert metric_ids == ["m1", "m2", "m7"]

    def test_lift_analysis_workflow_phase(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert data["workflow"] == "Wf-31"
        assert "Phase 5" in data["phase"]


class TestLiftSummaryMdExists:
    def test_lift_summary_md_exists(self):
        assert LIFT_SUMMARY_MD.exists()

    def test_lift_summary_md_references_json(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "wf31_m1_m2_m7_lift_analysis.json" in text

    def test_lift_summary_md_has_tl_dr(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "TL;DR" in text
        assert "m1" in text
        assert "m2" in text
        assert "m7" in text

    def test_lift_summary_md_honest_negative_provenance(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert (
            "honest-negative" in text.lower()
            or "HONEST-NEGATIVE" in text
            or "HONEST-NO-MEASURED" in text
        )


class TestM1Metric:
    def test_ru_metrics_has_wf31_entry(self):
        data = _load_json(METRICS_RU)
        assert (
            "wf31_phase5_lift_update" in data
        ), "metrics/by_metric/metalcytotox_ru_auc.json must have wf31_phase5_lift_update"

    def test_ru_r17_baseline_preserved(self):
        data = _load_json(METRICS_RU)
        # R17 baseline must remain 0.7123 (value field)
        assert abs(data["value"] - 0.7122770919067215) < 1e-9

    def test_ru_wf31_no_measured_value(self):
        data = _load_json(METRICS_RU)
        u = data["wf31_phase5_lift_update"]
        assert u["r18_wf31_status"] == "DEFER_NO_MEASURED_VALUE"
        assert u["r18_wf31_measured_value"] is None
        assert u["lift_r18_minus_r17"] is None

    def test_ru_wf22_history_recorded(self):
        data = _load_json(METRICS_RU)
        u = data["wf31_phase5_lift_update"]
        # Wf-22 history shows honest-negative 0.7123 -> 0.630
        assert u["wf22_history"]["value"] == 0.630
        assert u["wf22_history"]["lift_vs_r17"] < 0

    def test_ru_wf31_defer_reason_includes_smoke(self):
        data = _load_json(METRICS_RU)
        u = data["wf31_phase5_lift_update"]
        assert "smoke" in u["r18_wf31_defer_reason"].lower()


class TestM2Metric:
    def test_nci60_metrics_has_wf31_entry(self):
        data = _load_json(METRICS_NCI60)
        assert "wf31_phase5_lift_update" in data

    def test_nci60_overall_mean_auc_preserved(self):
        data = _load_json(METRICS_NCI60)
        # R17 broader 9-cell mean AUC must remain 0.8007
        assert abs(data["value_overall_mean_auc"] - 0.80072017249287) < 1e-6

    def test_nci60_A2780cis_per_cell_preserved(self):
        data = _load_json(METRICS_NCI60)
        # R17 broader A2780cis per-cell AUC = 0.7278 (different cohort from m2 R17 baseline 0.6476)
        assert (
            abs(data["per_cell_line"]["A2780cis"]["auc_mean"] - 0.7277591973244147)
            < 1e-6
        )

    def test_nci60_wf31_no_measured_value(self):
        data = _load_json(METRICS_NCI60)
        u = data["wf31_phase5_lift_update"]
        assert u["r18_wf31_status"] == "DEFER_NO_MEASURED_VALUE"
        assert u["r18_wf31_measured_value"] is None
        assert u["lift_r18_minus_r17"] is None

    def test_nci60_wf31_cross_reference_platinai_4cell(self):
        data = _load_json(METRICS_NCI60)
        u = data["wf31_phase5_lift_update"]
        # The m2 R17 baseline 0.6476 is in platinai_4cell_auc.json, NOT in this file
        assert u["r17_baseline_value_platinai_4cell"] == 0.6475694444444444
        assert u["r17_baseline_value_nci60_9cell_A2780cis"] == 0.7277591973244147


class TestM7Metric:
    def test_diversity_metrics_has_wf31_entry(self):
        data = _load_json(METRICS_DIVERSITY)
        assert "wf31_phase5_lift_update" in data

    def test_diversity_r17_baseline_preserved(self):
        data = _load_json(METRICS_DIVERSITY)
        # R17 baseline 0.1065 lives in measurements[4] (wf_round12_lambda_patha_10x3)
        r17_entry = data["measurements"][4]
        assert abs(r17_entry["value"] - 0.1065) < 1e-6

    def test_diversity_wf31_measured_value_added(self):
        data = _load_json(METRICS_DIVERSITY)
        # Wf-31 MEASURED 0.904 should be in measurements as the last entry
        last = data["measurements"][-1]
        assert abs(last["value"] - 0.9036262631416321) < 1e-6
        assert "wf31_phase4_multi_ph_diversity_bench" in last["run"]

    def test_diversity_wf31_floor_pass(self):
        data = _load_json(METRICS_DIVERSITY)
        u = data["wf31_phase5_lift_update"]
        assert u["r18_wf31_status"] == "MEASURED"
        assert u["r18_wf31_measured_value"] == 0.9036262631416321
        assert u["ship_gate_passed"] is True
        assert u["r18_section_3_6_t5_multi_ph_theoretical_lift_passed"] is True

    def test_diversity_apples_to_oranges_caveat(self):
        data = _load_json(METRICS_DIVERSITY)
        u = data["wf31_phase5_lift_update"]
        # The naive lift is reported as APPLES_TO_ORANGES, not as a real lift
        assert "APPLES_TO_ORANGES" in u["lift_r18_minus_r17_interpretation"]
        # Honest verdict: floor PASS, but cohort mismatch means we don't claim real lift
        assert u["verdict"] == "HONEST_PASS_ON_FLOOR_APPLES_TO_ORANGES_VS_R17"


class TestR20PlanItems:
    def test_lift_analysis_has_r20_plan_summary(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert "r20_plan_items" in data["r20_plan_summary"]
        items = data["r20_plan_summary"]["r20_plan_items"]
        for key in ("R20·P2", "R20·P3", "R20·P8_new"):
            assert key in items, f"missing R20 plan item {key}"

    def test_lift_summary_md_has_r20_plan_section(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "R20·P2" in text
        assert "R20·P3" in text
        assert "R20·P8" in text


class TestAggregateLiftSummary:
    def test_aggregate_lift_summary_zero_apples_to_apples_lift(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        agg = data["aggregate_lift_summary"]
        assert agg["n_lift_above_baseline_apples_to_apples"] == 0
        assert agg["r17_baselines_unchanged"] is True
        assert agg["r17_baseline_retained_as_canonical"] is True

    def test_aggregate_verdict_partial_honest_negative(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        verdict = data["aggregate_lift_summary"]["verdict"]
        assert "HONEST" in verdict.upper()
        assert "PARTIAL" in verdict.upper() or "honest" in verdict.lower()

    def test_aggregate_count_breakdown(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        agg = data["aggregate_lift_summary"]
        assert agg["n_metrics_compared"] == 3
        assert agg["n_metrics_with_measured_r18"] == 1
        assert agg["n_honest_negative"] == 2
        assert agg["n_honest_pass_on_floor"] == 1


class TestNoFabricatedHeadlines:
    """Verify no fake MEASURED headline was invented — the lift analysis must be honest."""

    def test_ru_metrics_no_fabricated_m1_value(self):
        data = _load_json(METRICS_RU)
        u = data["wf31_phase5_lift_update"]
        # m1 must have value=null, not a fake AUC
        assert u["r18_wf31_measured_value"] is None

    def test_nci60_metrics_no_fabricated_m2_value(self):
        data = _load_json(METRICS_NCI60)
        u = data["wf31_phase5_lift_update"]
        # m2 must have value=null, not a fake AUC
        assert u["r18_wf31_measured_value"] is None

    def test_diversity_metrics_m7_value_is_real(self):
        data = _load_json(METRICS_DIVERSITY)
        u = data["wf31_phase5_lift_update"]
        # m7 has a real MEASURED value from on-disk metrics.json
        assert u["r18_wf31_measured_value"] == 0.9036262631416321
        # But apples-to-oranges caveat must be present
        assert "APPLES_TO_ORANGES" in u["lift_r18_minus_r17_interpretation"]

    def test_honest_negative_provenance_section_present(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        hnp = data["honest_negative_provenance"]
        assert "policy" in hnp
        assert "applied" in hnp
        assert len(hnp["applied"]) == 3  # one entry per metric


class TestCrossReferences:
    def test_lift_analysis_references_wf31_summaries(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        preds = data["predecessor_summaries"]
        assert any("wf31_m1_summary.json" in p for p in preds)
        assert any("wf31_m2_summary.json" in p for p in preds)
        assert any("wf31_m7_summary.json" in p for p in preds)

    def test_lift_summary_references_phase4_doc(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "TODO/completed/86_wf31_phase4_dual_line.md" in text

    def test_lift_analysis_references_m7_metrics_json(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        # The m7 metric should reference the on-disk metrics.json
        m7_entry = [m for m in data["metrics"] if m["metric_id"] == "m7"][0]
        assert "wf31_m7/metrics.json" in m7_entry["r18_wf31_measured"]["source"]

    def test_lift_analysis_r20_p8_new_flagged(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        items = data["r20_plan_summary"]["r20_plan_items"]
        # R20·P8 must be flagged as NEW (created by this Phase 5)
        assert "NEW" in items["R20·P8_new"]


class TestWf22HistoryPreserved:
    """Verify Wf-22 honest-negative history is preserved as cautionary tale."""

    def test_m1_wf22_history_in_lift_analysis(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        m1_entry = [m for m in data["metrics"] if m["metric_id"] == "m1"][0]
        assert "wf22_history" in m1_entry
        assert m1_entry["wf22_history"]["value"] == 0.630
        assert m1_entry["wf22_history"]["lift_vs_r17"] < 0

    def test_m1_wf22_history_in_ru_metrics(self):
        data = _load_json(METRICS_RU)
        u = data["wf31_phase5_lift_update"]
        assert u["wf22_history"]["value"] == 0.630


class TestComparabilitySummary:
    """Verify the apples-to-oranges comparability is correctly documented."""

    def test_m7_cohort_mismatch_documented(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        m7_entry = [m for m in data["metrics"] if m["metric_id"] == "m7"][0]
        assert m7_entry["r18_wf31_measured"]["comparability"] == "PARTIALLY_COMPARABLE"
        # Reason should mention generated-molecule vs dataset-intrinsic cohort difference
        reason = m7_entry["r18_wf31_measured"]["reason"]
        assert "GENERATED molecules" in reason
        assert "FIXED DATASET" in reason
        assert "model's ability to generate diverse molecules" in reason
        assert "dataset's intrinsic diversity" in reason

    def test_m7_lift_not_claimed_as_real(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        agg = data["aggregate_lift_summary"]
        assert agg["n_lift_above_baseline_apples_to_apples"] == 0
