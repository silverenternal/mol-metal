"""Wf-32 Phase 5 lift analysis tests.

These tests verify that:
1. The lift analysis JSON exists and is well-formed
2. The lift summary MD exists and references the JSON
3. All 3 R17 baselines are preserved in their respective metrics JSONs
4. All 3 metrics JSONs have a wf32_phase5_lift_update entry
5. No fabricated MEASURED headline values exist (lift_r18_minus_r17 is None for m19, negative for m16, null + smoke-caveat for m8)
6. The R20 plan items are referenced
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIFT_ANALYSIS_JSON = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf32_m16_m19_m8_lift_analysis.json"
)
LIFT_SUMMARY_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf32_m16_m19_m8_lift_summary.md"
)
METRICS_PEARSON = PROJECT_ROOT / "metrics" / "by_metric" / "pearson_r_pic50.json"
METRICS_LOMO = PROJECT_ROOT / "metrics" / "by_metric" / "metal_lomo_auc.json"
METRICS_PB = PROJECT_ROOT / "metrics" / "by_metric" / "pb_pass_rate.json"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    return path.read_text()


class TestLiftAnalysisJsonExists:
    def test_lift_analysis_json_exists(self):
        assert (
            LIFT_ANALYSIS_JSON.exists()
        ), f"Wf-32 Phase 5 lift analysis JSON not found at {LIFT_ANALYSIS_JSON}"

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
        assert metric_ids == ["m16", "m19", "m8"]

    def test_lift_analysis_workflow_phase(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert data["workflow"] == "Wf-32"
        assert "Phase 5" in data["phase"]


class TestLiftSummaryMdExists:
    def test_lift_summary_md_exists(self):
        assert LIFT_SUMMARY_MD.exists()

    def test_lift_summary_md_references_json(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "wf32_m16_m19_m8_lift_analysis.json" in text

    def test_lift_summary_md_has_tl_dr(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "TL;DR" in text
        assert "m16" in text
        assert "m19" in text
        assert "m8" in text

    def test_lift_summary_md_honest_negative_provenance(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "honest-negative" in text.lower() or "HONEST-NEGATIVE" in text


class TestM16Metric:
    def test_pearson_metrics_has_wf32_entry(self):
        data = _load_json(METRICS_PEARSON)
        assert (
            "wf32_phase5_lift_update" in data
        ), "metrics/by_metric/pearson_r_pic50.json must have wf32_phase5_lift_update"

    def test_pearson_r17_baseline_preserved(self):
        data = _load_json(METRICS_PEARSON)
        # R17 baseline must remain 0.207 (aggregate.best_neural)
        assert data["aggregate"]["best_neural"] == 0.207

    def test_pearson_measurements_appended(self):
        data = _load_json(METRICS_PEARSON)
        assert len(data["measurements"]) >= 5  # was 4 before Phase 5
        last = data["measurements"][-1]
        assert "wf32_phase4_smoke" in last["run"]

    def test_pearson_wf32_update_negative_lift(self):
        data = _load_json(METRICS_PEARSON)
        u = data["wf32_phase5_lift_update"]
        assert u["r17_baseline_retained_as_canonical"] is True
        assert u["lift_r18_minus_r17"] < 0  # -0.247


class TestM19Metric:
    def test_lomo_metrics_has_wf32_entry(self):
        data = _load_json(METRICS_LOMO)
        assert "wf32_phase5_lift_update" in data

    def test_lomo_r17_baseline_preserved(self):
        data = _load_json(METRICS_LOMO)
        # R17 baseline must remain 0.5346 (value field)
        assert abs(data["value"] - 0.5346340730251975) < 1e-6

    def test_lomo_wf32_no_measured_value(self):
        data = _load_json(METRICS_LOMO)
        u = data["wf32_phase5_lift_update"]
        assert u["r18_wf32_status"] == "DEFER_NO_MEASURED_VALUE"
        assert u["r18_wf32_measured_value"] is None
        assert u["lift_r18_minus_r17"] is None

    def test_lomo_wf32_defer_reason_includes_gpu1(self):
        data = _load_json(METRICS_LOMO)
        u = data["wf32_phase5_lift_update"]
        assert (
            "GPU 1" in u["r18_wf32_defer_reason"] or "GPU" in u["r18_wf32_defer_reason"]
        )


class TestM8Metric:
    def test_pb_metrics_has_wf32_entry(self):
        data = _load_json(METRICS_PB)
        assert "wf32_phase5_lift_update" in data

    def test_pb_r17_baseline_preserved(self):
        data = _load_json(METRICS_PB)
        # R17 baseline must remain 0.846 (aggregate.best_measured_smoke)
        assert data["aggregate"]["best_measured_smoke"] == 0.846

    def test_pb_wf32_smoke_status(self):
        data = _load_json(METRICS_PB)
        u = data["wf32_phase5_lift_update"]
        assert u["r18_wf32_status"] == "MEASURED_SMOKE_NOT_COMPARABLE"
        assert u["r17_baseline_retained_as_canonical"] is True

    def test_pb_wf32_naive_lift_artifact_flag(self):
        data = _load_json(METRICS_PB)
        u = data["wf32_phase5_lift_update"]
        # The naive smoke-to-smoke lift is +0.154 but is artifact of reduced check scope
        assert u["naive_smoke_to_smoke_lift"] == 0.154
        assert u["naive_lift_is_artifact_of_reduced_check_scope"] is True
        assert u["lift_r18_minus_r17"] is None  # no honest lift


class TestR20PlanItems:
    def test_lift_analysis_has_r20_plan_summary(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        assert "r20_plan_items" in data["r20_plan_summary"]
        items = data["r20_plan_summary"]["r20_plan_items"]
        for key in ("R20·P5", "R20·P6", "R20·P7_revised"):
            assert key in items, f"missing R20 plan item {key}"

    def test_lift_summary_md_has_r20_plan_section(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "R20·P5" in text
        assert "R20·P6" in text
        assert "R20·P7" in text


class TestAggregateLiftSummary:
    def test_aggregate_lift_summary_zero_lift(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        agg = data["aggregate_lift_summary"]
        assert agg["n_lift_above_baseline"] == 0
        assert agg["n_metrics_passed_r17_baseline"] == 0
        assert agg["r17_baselines_unchanged"] is True
        assert agg["r17_baseline_retained_as_canonical"] is True

    def test_aggregate_verdict_full_honest_negative(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        verdict = data["aggregate_lift_summary"]["verdict"]
        assert "FULL_HONEST_NEGATIVE" in verdict or "honest-negative" in verdict.lower()


class TestNoFabricatedHeadlines:
    """Verify no fake MEASURED headline was invented — the lift analysis must be honest-negative."""

    def test_pearson_metrics_aggregate_has_wf32_update(self):
        data = _load_json(METRICS_PEARSON)
        assert "wf32_phase5_update" in data["aggregate"]

    def test_no_fabricated_m19_value(self):
        data = _load_json(METRICS_LOMO)
        u = data["wf32_phase5_lift_update"]
        # m19 must have value=null, not a fake AUC
        assert u["r18_wf32_measured_value"] is None

    def test_no_fabricated_m8_value(self):
        data = _load_json(METRICS_PB)
        u = data["wf32_phase5_lift_update"]
        # m8 R18 value is the 1-cell smoke; honest framing says it's NOT comparable to R17
        assert u["r18_wf32_smoke_value"] == 1.0
        assert u["r18_wf32_status"] == "MEASURED_SMOKE_NOT_COMPARABLE"
        assert u["lift_r18_minus_r17"] is None

    def test_honest_negative_provenance_section_present(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        hnp = data["honest_negative_provenance"]
        assert "policy" in hnp
        assert "applied" in hnp
        assert len(hnp["applied"]) == 3  # one entry per metric


class TestCrossReferences:
    def test_lift_analysis_references_wf32_summaries(self):
        data = _load_json(LIFT_ANALYSIS_JSON)
        preds = data["predecessor_summaries"]
        assert any("wf32_m16_summary.json" in p for p in preds)
        assert any("wf32_m19_summary.json" in p for p in preds)
        assert any("wf32_m8_summary.json" in p for p in preds)

    def test_lift_summary_references_phase4_doc(self):
        text = _load_text(LIFT_SUMMARY_MD)
        assert "TODO/completed/87_wf32_phase4_honest_negative.md" in text
