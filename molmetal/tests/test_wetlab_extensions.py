"""test_wetlab_extensions.py — unit tests for the TODO-30 P5.2 Tier-1
wet-lab EXTENSIONS (scheduling, cost, collaborators, parsers, feedback).

These tests cover the contract:

1. :class:`WetLabSchedulingHelper` — capacity-aware bucketing, priority
   ordering, graceful empty input.
2. :class:`CostEstimator` — volume-discount curve + non-negative input.
3. :class:`CollaboratorTracker` — add / dedup / list / len.
4. :class:`AssayDataParser` — CSV + JSON parsing, both on-disk shapes.
5. :class:`WetLabFeedbackIntegrator` — recommendation → delta mapping,
   opt-in contract (no aggregator mutation).

All tests are CPU-only and never touch the GPU / chem stack.

Run from the project root::

    uv run pytest molmetal/tests/test_wetlab_extensions.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Subject under test — imported lazily so the test module is collectible
# even when the heavier lam_chem / search_alg stack is broken.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.lam_chem.wetlab_extensions import (
        WetLabSchedulingHelper,
        CostEstimator,
        CollaboratorTracker,
        AssayDataParser,
        WetLabFeedbackIntegrator,
    )
    _HAS_EXT = True
except Exception:  # pragma: no cover
    WetLabSchedulingHelper = None  # type: ignore[assignment]
    CostEstimator = None  # type: ignore[assignment]
    CollaboratorTracker = None  # type: ignore[assignment]
    AssayDataParser = None  # type: ignore[assignment]
    WetLabFeedbackIntegrator = None  # type: ignore[assignment]
    _HAS_EXT = False


pytestmark = pytest.mark.skipif(
    not _HAS_EXT,
    reason="molmetal_lam.lam_chem.wetlab_extensions unavailable",
)


# ===========================================================================
# WetLabSchedulingHelper
# ===========================================================================


def test_scheduling_helper_schedule():
    """Priority list is bucketed into weeks with capacity respected."""
    helper = WetLabSchedulingHelper(assay_capacity_per_week=3)
    priority_list = [
        {"smiles": "CCO", "priority": "high"},
        {"smiles": "CCN", "priority": "medium"},
        {"smiles": "CCC", "priority": "low"},
        {"smiles": "CCCl", "priority": "high"},
        {"smiles": "CCBr", "priority": 0.7},
    ]
    scheduled = helper.schedule_assays(priority_list)
    assert isinstance(scheduled, list)
    assert len(scheduled) == 5
    # All scheduled items must have week, label, priority, raw.
    for item in scheduled:
        assert {"label", "week", "priority", "raw"} <= set(item.keys())
        assert isinstance(item["week"], int)
        assert item["week"] >= 1


def test_scheduling_helper_capacity():
    """Capacity cap is respected within each week bucket."""
    helper = WetLabSchedulingHelper(assay_capacity_per_week=2)
    priority_list = [
        {"smiles": f"S{i}", "priority": 1.0} for i in range(5)
    ]
    scheduled = helper.schedule_assays(priority_list)
    # 5 items, capacity 2 → at least 3 distinct weeks used.
    weeks = sorted({it["week"] for it in scheduled})
    assert len(weeks) >= 3
    # Capacity respected: no week has more than 2 items.
    from collections import Counter
    counts = Counter(it["week"] for it in scheduled)
    for w, n in counts.items():
        assert n <= 2


def test_scheduling_helper_priority_sort():
    """High-priority items land in earlier weeks than low-priority."""
    helper = WetLabSchedulingHelper(assay_capacity_per_week=1)
    priority_list = [
        {"smiles": "LOW1", "priority": "low"},
        {"smiles": "HIGH1", "priority": "high"},
    ]
    scheduled = helper.schedule_assays(priority_list)
    high = next(it for it in scheduled if it["label"] == "HIGH1")
    low = next(it for it in scheduled if it["label"] == "LOW1")
    assert high["week"] <= low["week"]


def test_scheduling_helper_empty_input():
    """Empty input returns an empty list."""
    helper = WetLabSchedulingHelper()
    assert helper.schedule_assays([]) == []


def test_scheduling_helper_default_capacity():
    """Default capacity is 10 per week (sanity)."""
    helper = WetLabSchedulingHelper()
    assert helper.assay_capacity_per_week == 10


def test_scheduling_helper_invalid_capacity_clamped():
    """Non-positive capacity is clamped to 1."""
    helper = WetLabSchedulingHelper(assay_capacity_per_week=0)
    assert helper.assay_capacity_per_week == 1
    helper2 = WetLabSchedulingHelper(assay_capacity_per_week=-5)
    assert helper2.assay_capacity_per_week == 1
    # Also a string that can't parse → 10.
    helper3 = WetLabSchedulingHelper(assay_capacity_per_week="not_a_number")
    assert helper3.assay_capacity_per_week == 10


def test_scheduling_helper_priority_string_mapping():
    """String priorities map to numeric values per the documented table."""
    helper = WetLabSchedulingHelper(assay_capacity_per_week=1)
    priority_list = [
        {"smiles": "H", "priority": "high"},
        {"smiles": "M", "priority": "medium"},
        {"smiles": "L", "priority": "low"},
    ]
    scheduled = helper.schedule_assays(priority_list)
    pmap = {it["label"]: it["priority"] for it in scheduled}
    assert pmap["H"] == pytest.approx(1.0)
    assert pmap["M"] == pytest.approx(0.5)
    assert pmap["L"] == pytest.approx(0.1)


# ===========================================================================
# CostEstimator
# ===========================================================================


def test_cost_estimator_basic():
    """Baseline cost is n_assays * cost_per_assay_usd (no discount)."""
    est = CostEstimator(cost_per_assay_usd=100.0)
    assert est.estimate_total_cost(10) == 1000.0


def test_cost_estimator_volume_discount():
    """Large N triggers the volume discount brackets."""
    est = CostEstimator(cost_per_assay_usd=100.0)
    # < 50 → 0 % discount
    assert est.estimate_total_cost(40) == 4000.0
    # >= 50 → 5 % discount
    cost_50 = est.estimate_total_cost(50)
    assert cost_50 == pytest.approx(50 * 100 * 0.95)
    # >= 200 → 15 % discount
    cost_200 = est.estimate_total_cost(200)
    assert cost_200 == pytest.approx(200 * 100 * 0.85)
    # >= 1000 → 30 % discount
    cost_1000 = est.estimate_total_cost(1000)
    assert cost_1000 == pytest.approx(1000 * 100 * 0.70)


def test_cost_estimator_zero_assays():
    """n_assays == 0 returns 0.0."""
    est = CostEstimator()
    assert est.estimate_total_cost(0) == 0.0


def test_cost_estimator_negative_assays_clamped():
    """Negative n_assays returns 0.0 (clamped)."""
    est = CostEstimator()
    assert est.estimate_total_cost(-5) == 0.0


def test_cost_estimator_default_cost():
    """Default per-assay cost is $200 (sanity)."""
    est = CostEstimator()
    assert est.cost_per_assay_usd == 200.0


def test_cost_estimator_invalid_cost_falls_back():
    """Non-numeric / negative cost falls back to the default $200."""
    est = CostEstimator(cost_per_assay_usd="oops")
    assert est.cost_per_assay_usd == 200.0
    est2 = CostEstimator(cost_per_assay_usd=-100)
    assert est2.cost_per_assay_usd == 200.0


def test_cost_estimator_rounding_two_decimals():
    """Cost is rounded to 2 decimal places."""
    est = CostEstimator(cost_per_assay_usd=33.333)
    total = est.estimate_total_cost(3)  # 99.999 → 100.0
    assert isinstance(total, float)
    assert total == 100.0


# ===========================================================================
# CollaboratorTracker
# ===========================================================================


def test_collaborator_tracker_add_list():
    """Adding collaborators then listing returns them."""
    tr = CollaboratorTracker()
    a = tr.add_collaborator("Alice", "MIT", "medicinal_chem")
    assert isinstance(a, dict)
    assert a["name"] == "Alice"
    assert a["institution"] == "MIT"
    assert a["expertise"] == "medicinal_chem"
    listed = tr.list_collaborators()
    assert len(listed) == 1
    assert listed[0]["name"] == "Alice"


def test_collaborator_tracker_dedup_same_person():
    """Same (name, institution) → dedup; existing record is kept."""
    tr = CollaboratorTracker()
    tr.add_collaborator("Alice", "MIT", "medicinal_chem")
    tr.add_collaborator("Alice", "MIT", "biochem")
    assert len(tr.list_collaborators()) == 1
    # Original expertise is preserved (first-loaded wins).
    assert tr.list_collaborators()[0]["expertise"] == "medicinal_chem"


def test_collaborator_tracker_new_institution_appends():
    """Same name but new institution → new record (no dedup)."""
    tr = CollaboratorTracker()
    tr.add_collaborator("Alice", "MIT", "medchem")
    tr.add_collaborator("Alice", "Stanford", "medchem")
    assert len(tr.list_collaborators()) == 2


def test_collaborator_tracker_empty_name_noop():
    """Empty / non-string name returns {} and is NOT added."""
    tr = CollaboratorTracker()
    rec = tr.add_collaborator("", "MIT", "x")
    assert rec == {}
    assert len(tr.list_collaborators()) == 0


def test_collaborator_tracker_list_is_copy():
    """list_collaborators() returns a defensive copy."""
    tr = CollaboratorTracker()
    tr.add_collaborator("Bob", "Caltech", "biology")
    listed = tr.list_collaborators()
    listed.clear()
    # The tracker itself is unchanged.
    assert len(tr.list_collaborators()) == 1


def test_collaborator_tracker_len_dunder():
    """len(tracker) reflects the count."""
    tr = CollaboratorTracker()
    assert len(tr) == 0
    tr.add_collaborator("Eve", "Berkeley", "chemistry")
    tr.add_collaborator("Frank", "Harvard", "biology")
    assert len(tr) == 2


# ===========================================================================
# AssayDataParser
# ===========================================================================


def test_assay_data_parser_csv():
    """CSV file is parsed into a list of dict rows."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "assays.csv")
        Path(path).write_text(
            "smiles,pIC50,cell_line\n"
            "CCO,5.0,HeLa\n"
            "CCN,6.5,A2780\n"
        )
        rows = AssayDataParser.parse_csv_assay_data(path)
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0]["smiles"] == "CCO"
    assert rows[0]["pIC50"] == "5.0"
    assert rows[0]["cell_line"] == "HeLa"
    assert rows[1]["smiles"] == "CCN"


def test_assay_data_parser_csv_missing_file():
    """Missing CSV file returns an empty list."""
    rows = AssayDataParser.parse_csv_assay_data("/nonexistent/path.csv")
    assert rows == []


def test_assay_data_parser_json():
    """JSON list-of-dicts is parsed into a list of dict rows."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "assays.json")
        Path(path).write_text(json.dumps([
            {"smiles": "CCO", "pIC50": 5.0},
            {"smiles": "CCN", "pIC50": 6.5},
        ]))
        rows = AssayDataParser.parse_json_assay_data(path)
    assert len(rows) == 2
    assert rows[0]["smiles"] == "CCO"
    assert rows[1]["smiles"] == "CCN"


def test_assay_data_parser_json_dict_shape():
    """JSON dict-of-dicts (keyed by SMILES) is normalised."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "assays.json")
        Path(path).write_text(json.dumps({
            "CCO": {"pIC50": 5.0, "cell_line": "HeLa"},
            "CCN": {"pIC50": 6.5},
        }))
        rows = AssayDataParser.parse_json_assay_data(path)
    assert len(rows) == 2
    # The 'smiles' key is injected from the dict key.
    smis = sorted(r["smiles"] for r in rows)
    assert smis == ["CCN", "CCO"]
    cco = next(r for r in rows if r["smiles"] == "CCO")
    assert cco["pIC50"] == "5.0"
    assert cco["cell_line"] == "HeLa"


def test_assay_data_parser_json_missing_file():
    """Missing JSON file returns an empty list (no exception)."""
    rows = AssayDataParser.parse_json_assay_data("/nonexistent/path.json")
    assert rows == []


def test_assay_data_parser_json_invalid():
    """Malformed JSON returns an empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        Path(path).write_text("{not valid json")
        rows = AssayDataParser.parse_json_assay_data(path)
    assert rows == []


# ===========================================================================
# WetLabFeedbackIntegrator
# ===========================================================================


def _fake_recalibrate_ok(results):
    """Recalibrate stub that returns an OK recommendation."""
    return {"n_overlap": 20, "pearson_r": 0.85, "recommendation": "OK — well-calibrated"}


def _fake_recalibrate_retrain(results):
    return {"n_overlap": 10, "recommendation": "RE-TRAIN with weight += 0.5"}


def _fake_recalibrate_bias(results):
    return {"n_overlap": 12, "recommendation": "RE-TRAIN with bias-correction — intercept=+1.2"}


def _fake_recalibrate_marginal(results):
    return {"n_overlap": 8, "recommendation": "MARGINAL — Pearson r=+0.5"}


def _fake_recalibrate_insufficient(results):
    return {"n_overlap": 2, "recommendation": "INSUFFICIENT_DATA — collect more"}


def test_wetlab_feedback_integrator_ok_returns_zero_delta():
    """OK recommendation → calibration delta = 0.0."""
    integ = WetLabFeedbackIntegrator(_fake_recalibrate_ok)
    delta = integ.integrate_assay_results([{"smiles": "CCO", "pIC50": 5.0}])
    assert delta == 0.0


def test_wetlab_feedback_integrator_retrain_weight_returns_delta():
    """RE-TRAIN with weight → delta = 0.5."""
    integ = WetLabFeedbackIntegrator(_fake_recalibrate_retrain)
    delta = integ.integrate_assay_results([])
    assert delta == pytest.approx(0.5)


def test_wetlab_feedback_integrator_bias_correction_returns_one():
    """RE-TRAIN with bias-correction → delta = 1.0."""
    integ = WetLabFeedbackIntegrator(_fake_recalibrate_bias)
    delta = integ.integrate_assay_results([])
    assert delta == pytest.approx(1.0)


def test_wetlab_feedback_integrator_marginal_returns_quarter():
    """MARGINAL recommendation → delta = 0.25."""
    integ = WetLabFeedbackIntegrator(_fake_recalibrate_marginal)
    delta = integ.integrate_assay_results([])
    assert delta == pytest.approx(0.25)


def test_wetlab_feedback_integrator_insufficient_returns_zero():
    """INSUFFICIENT_DATA → delta = 0.0 (no automatic promotion)."""
    integ = WetLabFeedbackIntegrator(_fake_recalibrate_insufficient)
    delta = integ.integrate_assay_results([])
    assert delta == 0.0


def test_wetlab_feedback_integrator_recalibrate_raises_returns_zero():
    """If recalibrate_fn raises, integrator returns 0.0 (graceful)."""
    def bad_fn(results):
        raise RuntimeError("kaboom")

    integ = WetLabFeedbackIntegrator(bad_fn)
    assert integ.integrate_assay_results([]) == 0.0


def test_wetlab_feedback_integrator_non_callable_raises():
    """Constructor raises TypeError on non-callable recalibrate_fn."""
    with pytest.raises(TypeError):
        WetLabFeedbackIntegrator("not_callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-cutting sanity: extensions module does NOT mutate R17 surface.
# ---------------------------------------------------------------------------


def test_extensions_no_mutation_of_r17_modules():
    """Importing wetlab_extensions does not touch the R17 modules.

    Sanity check — the only public names exposed by
    :mod:`wetlab_extensions` are the 5 helper classes; no symbol
    from :mod:`wetlab_protocol` or :mod:`wetlab_reward_channel` is
    re-exported.
    """
    from molmetal_lam.lam_chem import wetlab_extensions as ext

    assert hasattr(ext, "WetLabSchedulingHelper")
    assert hasattr(ext, "CostEstimator")
    assert hasattr(ext, "CollaboratorTracker")
    assert hasattr(ext, "AssayDataParser")
    assert hasattr(ext, "WetLabFeedbackIntegrator")
    # These are R17 symbols and should NOT have been re-exported.
    assert not hasattr(ext, "Assay")
    assert not hasattr(ext, "compute_wetlab_signed_error")
    assert not hasattr(ext, "register_wetlab_channel")
