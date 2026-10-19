#!/usr/bin/env python3
"""Tests for the NCI-60 aggregate rebuild and its gate cross-check.

The artifact carried two tables that disagreed: a real ``per_cell_line`` block
(mean 0.7294, A2780cis 0.3356) and a ``wf312b_nci60_smoke_aggregate`` block of
placeholders (7 of 9 cells at the identical 0.80072, DU145 included only to
reach the 9-key floor).  The pre-flight gate validated the placeholder block's
internal consistency and passed, so every document in the repo quoted 0.8007.

These tests pin both halves of the fix: the block is derived from the real
table, and the gate fails when it is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import pre_flight_check as pf  # noqa: E402
from molmetal.scripts import wf_a55_nci60_aggregate_refresh as refresh  # noqa: E402

CELLS = {
    "A2780": 0.5104, "A2780cis": 0.3356, "A549": 0.6951, "DU145": 1.0,
    "HCT116": 0.7517, "HL60": 0.8, "MCF7": 0.6690, "PC3": 0.8028, "SKOV3": 1.0,
}
N_TEST = {
    "A2780": 48, "A2780cis": 28, "A549": 107, "DU145": 7,
    "HCT116": 31, "HL60": 11, "MCF7": 90, "PC3": 18, "SKOV3": 3,
}


def _per_cell():
    return {
        c: {"auc_mean": v, "auc_std": 0.01, "n_train": 100, "n_test": N_TEST[c]}
        for c, v in CELLS.items()
    }


def _doc():
    return {"per_cell_line": _per_cell()}


def _write(tmp_path, doc):
    p = tmp_path / "nci60.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# --- block construction ---------------------------------------------------


def test_block_is_derived_from_per_cell_line():
    block = refresh.build_block(_per_cell())
    assert set(block["per_cell_breakdown"]) == set(CELLS)
    for cell, v in CELLS.items():
        assert block["per_cell_breakdown"][cell]["mean_auc"] == pytest.approx(v)
    assert block["overall_mean"] == pytest.approx(sum(CELLS.values()) / 9, abs=1e-9)


def test_block_is_not_a_placeholder_table():
    """The exact shape of the defect: 7 of 9 cells at one identical number.

    Two cells sharing a value is legitimate (two test sets can both score 1.0);
    seven sharing one is a table that was never measured.
    """
    block = refresh.build_block(_per_cell())
    values = [v["mean_auc"] for v in block["per_cell_breakdown"].values()]
    modal = max(values.count(v) for v in set(values))
    assert modal <= 2, f"{modal} of 9 cells share one AUC — looks like a placeholder"


def test_honest_negative_is_the_lowest_cell_below_chance():
    block = refresh.build_block(_per_cell())
    assert block["honest_negative"]["cell"] == "A2780cis"
    assert block["honest_negative"]["value"] < 0.5
    assert block["honest_negative"]["badge"] == "honest-negative"
    assert block["honest_negative"]["honest_negative_reason"].strip()


def test_resolvable_subset_excludes_single_digit_test_sets():
    rs = refresh.build_block(_per_cell())["resolvable_subset"]
    assert rs["cells_excluded"] == ["DU145", "SKOV3"]
    assert rs["n_cells"] == 7
    # 7-cell mean of the resolvable cells, not the 9-cell mean.
    expected = sum(v for c, v in CELLS.items() if N_TEST[c] >= 10) / 7
    assert rs["unweighted_mean"] == pytest.approx(expected)
    assert rs["unweighted_mean"] < refresh.build_block(_per_cell())["overall_mean"]


def test_n_weighted_mean_uses_test_sizes():
    rs = refresh.build_block(_per_cell())["resolvable_subset"]
    num = sum(v * N_TEST[c] for c, v in CELLS.items())
    assert rs["n_weighted_mean"] == pytest.approx(num / sum(N_TEST.values()))


# --- gate cross-check -----------------------------------------------------


def test_gate_passes_on_a_derived_block(tmp_path):
    doc = _doc()
    doc[refresh.BLOCK_KEY] = refresh.build_block(_per_cell())
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "PASS"


def test_gate_fails_when_block_disagrees_with_real_table(tmp_path):
    """Self-consistency is not enough — the block must match the measurement."""
    doc = _doc()
    block = refresh.build_block(_per_cell())
    block["per_cell_breakdown"]["DU145"]["mean_auc"] = 0.80072
    block["overall_mean"] = sum(
        v["mean_auc"] for v in block["per_cell_breakdown"].values()
    ) / 9
    doc[refresh.BLOCK_KEY] = block
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "FAIL"
    assert "DU145" in r.evidence


def test_gate_still_passes_without_a_real_table(tmp_path):
    """Backwards compatible: a file with only the block is still checked."""
    doc = {refresh.BLOCK_KEY: refresh.build_block(_per_cell())}
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "PASS"
