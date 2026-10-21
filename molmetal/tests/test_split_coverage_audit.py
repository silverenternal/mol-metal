"""The split-coverage audit must describe the split that was actually trained.

`metrics/by_metric/split_coverage_4cell.json` explains why
`platinai_3cell_resistance_real` reports `MEASURED_2_OF_3_CELLS`.  It re-derives
the split from the cohort file with the production splitter, so it can drift
from the trained artifact in two ways: the cohort file changes, or the splitter
changes.  Either would leave an audit that confidently explains a split nobody
ran.  These checks are cheap because they read the two committed JSONs rather
than re-running the 500-permutation measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "metrics" / "by_metric" / "split_coverage_4cell.json"
RUN = ROOT / "metrics" / "by_metric" / "platinai_3cell_resistance_real.json"
STRAT = ROOT / "metrics" / "by_metric" / "platinai_3cell_resistance_real_strat.json"


@pytest.fixture(scope="module")
def audit():
    if not AUDIT.exists():
        pytest.skip("split coverage audit not generated")
    return json.loads(AUDIT.read_text())


@pytest.fixture(scope="module")
def run():
    if not RUN.exists():
        pytest.skip("real-label resistance run not present")
    return json.loads(RUN.read_text())


def test_audit_reproduces_the_trained_split_sizes(audit, run):
    tv = audit["trainer_view"]
    sizes = run["split_sizes"]
    assert tv["train"]["rows"] == sizes["n_train"]
    assert tv["val"]["rows"] == sizes["n_val"]
    assert tv["test"]["rows"] == sizes["n_test"]


def test_audit_reproduces_the_dropped_rows(audit, run):
    # The trainer models three of four columns; A2780-only compounds fall out
    # after the split.  Both files must agree on how many.
    as_is_total = sum(audit["as_is"][s]["rows"] for s in ("train", "val", "test"))
    tv_total = sum(audit["trainer_view"][s]["rows"] for s in ("train", "val", "test"))
    assert as_is_total - tv_total == audit["trainer_view"]["dropped_rows"]
    assert tv_total == run["n_rows"] - audit["trainer_view"]["dropped_rows"]


def test_audit_agrees_with_the_run_that_a2780cis_was_not_evaluated(audit, run):
    cell = run["per_cell_line"]["A2780cis"]
    assert cell["n_seeds_evaluated"] == 0
    assert cell["auc_mean"] is None
    assert audit["as_is"]["test"]["cells"]["A2780cis"]["evaluable"] is False
    assert audit["value"] == audit["as_is"]["test"]["cells"]["A2780cis"]["labelled"]


def test_audit_bar_matches_the_trainer_constant(audit):
    from molmetal.scripts.train_cisplatin_resistance import MIN_EVAL_ROWS

    assert audit["evaluation_bar"]["min_labelled_rows"] == MIN_EVAL_ROWS


def test_applied_split_is_the_one_the_stratified_run_trained(audit):
    """The audit calls the production splitter; this pins it to the artifact."""
    if not STRAT.exists():
        pytest.skip("coverage-stratified run not present")
    strat = json.loads(STRAT.read_text())
    applied = audit["applied_coverage_split"]
    dropped = audit["trainer_view"]["dropped_rows"]
    assert strat["split"] == "scaffold_70_10_20_by_cell_coverage"
    for name, key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test")):
        # The audit measures the split before the trainer's three-column
        # filter; the difference is the A2780-only rows, counted once.
        assert strat["split_sizes"][key] <= applied[name]["rows"]
        assert applied[name]["rows"] - strat["split_sizes"][key] <= dropped
    assert strat["status"] == "MEASURED"
    for cell, info in strat["per_cell_line"].items():
        assert info["n_seeds_evaluated"] == 5, cell
        assert info["auc_mean"] is not None, cell
