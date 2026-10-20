#!/usr/bin/env python3
"""Tests for the Wf-31 block restore.

``metalcytotox_ru_auc.json`` and ``diversity_tanimoto.json`` were re-written by
later runs (Wf-124 P3, Wf-129 P2) and each rewrite dropped the
``wf31_phase5_lift_update`` block, leaving 12 tests red with no gate noticing.
These tests pin the restore: it copies from the analysis, it is idempotent, and
it refuses rather than inventing a row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import wf_a56_restore_wf31_blocks as restore  # noqa: E402


def _analysis():
    return json.loads((PROJECT_ROOT / restore.ANALYSIS).read_text(encoding="utf-8"))


def _entry(metric_id):
    return next(m for m in _analysis()["metrics"] if m["metric_id"] == metric_id)


# --- block construction ---------------------------------------------------


def test_m1_block_is_honest_no_measured():
    block = restore._build_block(_entry("m1"), restore.ANALYSIS, 0)
    assert block["r18_wf31_status"] == "DEFER_NO_MEASURED_VALUE"
    assert block["r18_wf31_measured_value"] is None
    assert block["lift_r18_minus_r17"] is None
    # The Wf-22 retrain history is the reason the metric was deferred.
    assert block["wf22_history"]["value"] == 0.630
    assert block["wf22_history"]["lift_vs_r17"] < 0
    assert "smoke" in block["r18_wf31_defer_reason"].lower()


def test_m7_block_carries_the_apples_to_oranges_caveat():
    block = restore._build_block(_entry("m7"), restore.ANALYSIS, 2)
    assert block["r18_wf31_status"] == "MEASURED"
    assert block["r18_wf31_measured_value"] == 0.9036262631416321
    assert "APPLES_TO_ORANGES" in block["lift_r18_minus_r17_interpretation"]
    assert block["ship_gate_passed"] is True


def test_every_block_points_at_its_source():
    for mid, idx in (("m1", 0), ("m7", 2)):
        block = restore._build_block(_entry(mid), restore.ANALYSIS, idx)
        assert block["source_analysis"] == f"{restore.ANALYSIS}::metrics[{idx}]"


# --- main() behaviour -----------------------------------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A minimal tree: the analysis, the m7 summary, and two metric files."""
    (tmp_path / "metrics/by_metric").mkdir(parents=True)
    (tmp_path / "molmetal/reports").mkdir(parents=True)
    for rel in (restore.ANALYSIS, restore.M7_SUMMARY):
        src = PROJECT_ROOT / rel
        (tmp_path / rel).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "metrics/by_metric/metalcytotox_ru_auc.json").write_text(
        json.dumps({"metric": "m1", "value": 0.7613}), encoding="utf-8"
    )
    (tmp_path / "metrics/by_metric/diversity_tanimoto.json").write_text(
        json.dumps(
            {
                "metric": "m7",
                "measurements": [{"run": "wf_round12_lambda_patha_10x3", "value": 0.1065}],
                "aggregate": {"best_measured": 0.13656567595469193},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(restore, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _read(root, rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def test_dry_run_writes_nothing(sandbox):
    rc = restore.main(["--dry-run"])
    assert rc == 0
    assert restore.BLOCK_KEY not in _read(
        sandbox, "metrics/by_metric/metalcytotox_ru_auc.json"
    )


def test_restore_adds_block_and_measurement_row(sandbox):
    rc = restore.main([])
    assert rc == 0

    m1 = _read(sandbox, "metrics/by_metric/metalcytotox_ru_auc.json")
    assert m1[restore.BLOCK_KEY]["r18_wf31_status"] == "DEFER_NO_MEASURED_VALUE"
    # The live value is untouched — this script restores provenance only.
    assert m1["value"] == 0.7613

    m7 = _read(sandbox, "metrics/by_metric/diversity_tanimoto.json")
    assert m7[restore.BLOCK_KEY]["r18_wf31_measured_value"] == 0.9036262631416321
    assert any(restore.M7_RUN in m["run"] for m in m7["measurements"])
    assert "scope_note" in m7["aggregate"]


def test_restore_is_idempotent(sandbox):
    assert restore.main([]) == 0
    first = _read(sandbox, "metrics/by_metric/diversity_tanimoto.json")
    assert restore.main([]) == 0
    second = _read(sandbox, "metrics/by_metric/diversity_tanimoto.json")
    assert first == second
    runs = [m["run"] for m in second["measurements"]]
    assert sum(restore.M7_RUN in r for r in runs) == 1


def test_missing_summary_refuses_to_invent_the_m7_row(sandbox):
    (sandbox / restore.M7_SUMMARY).unlink()
    rc = restore.main([])
    assert rc == 2
    m7 = _read(sandbox, "metrics/by_metric/diversity_tanimoto.json")
    assert not any(restore.M7_RUN in m["run"] for m in m7["measurements"])
