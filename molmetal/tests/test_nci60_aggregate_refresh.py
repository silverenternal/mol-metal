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


def test_threshold_is_taken_from_the_funnel_when_present():
    """One file must not carry two different 'honest' subsets.

    The artifact's own ``test_set_size_funnel.primary_filter`` is the declared
    primary (``n_test_ge_20``).  The first rebuild hardcoded 10 and wrote
    0.6521 beside a funnel that said 0.5923.
    """
    doc = _doc()
    doc["test_set_size_funnel"] = {
        "primary_filter": "n_test_ge_20",
        "primary_mean_auc": 0.5923288209839935,
    }
    assert refresh._resolve_threshold(doc) == (
        20,
        "test_set_size_funnel.primary_filter",
    )
    assert refresh._resolve_threshold(_doc())[0] == refresh.MIN_RESOLVABLE_N_TEST


def test_build_block_honours_an_explicit_threshold():
    rs = refresh.build_block(_per_cell(), 20)["resolvable_subset"]
    assert rs["min_n_test"] == 20
    assert rs["cells_excluded"] == ["DU145", "HL60", "PC3", "SKOV3"]
    expected = sum(v for c, v in CELLS.items() if N_TEST[c] >= 20) / 5
    assert rs["unweighted_mean"] == pytest.approx(expected)


def test_refresh_refuses_to_write_when_funnel_disagrees(tmp_path):
    doc = _doc()
    doc["test_set_size_funnel"] = {
        "primary_filter": "n_test_ge_20",
        "primary_mean_auc": 0.6521,  # the stale n>=10 number
    }
    rc = refresh.main(["--artifact", str(_write(tmp_path, doc))])
    assert rc == 2
    # Nothing was written: the artifact still has no rebuilt block.
    assert refresh.BLOCK_KEY not in json.loads(
        (tmp_path / "nci60.json").read_text(encoding="utf-8")
    )


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


def _doc_with_funnel():
    doc = _doc()
    doc["test_set_size_funnel"] = {
        "primary_filter": "n_test_ge_20",
        "primary_mean_auc": 0.5923288209839935,
    }
    doc[refresh.BLOCK_KEY] = refresh.build_block(_per_cell(), 20, "funnel")
    return doc


def test_gate_passes_when_subset_and_funnel_agree(tmp_path):
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, _doc_with_funnel())
    )
    assert r.status == "PASS"


def test_gate_fails_when_subset_mean_disagrees_with_funnel(tmp_path):
    doc = _doc_with_funnel()
    doc[refresh.BLOCK_KEY]["resolvable_subset"]["unweighted_mean"] = 0.6521
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "FAIL"
    assert "primary_mean_auc" in r.evidence


def test_gate_fails_when_threshold_does_not_match_primary_filter(tmp_path):
    doc = _doc_with_funnel()
    doc[refresh.BLOCK_KEY]["resolvable_subset"]["min_n_test"] = 10
    r = pf.gate_nci60_9cell_per_cell_aggregate(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "FAIL"
    assert "primary_filter" in r.evidence


# --- the "REAL on 3 seeds" gate reads the live field, not the legacy one ---


def _real_doc():
    doc = _doc()
    doc["value_overall_mean_auc"] = 0.7293802091886383
    doc["n_seeds"] = 5
    doc["status"] = "MEASURED_HONEST_NEGATIVE"
    # The 3-seed degenerate-split value is still in the file as history.
    doc["aggregate_mean_auc_REAL"] = 0.80072017249287
    return doc


def test_real_gate_reports_the_live_value_and_names_the_legacy_one(tmp_path):
    r = pf.gate_p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds(
        "g", "P3", "x", _write(tmp_path, _real_doc())
    )
    assert r.status == "PASS"
    assert "0.7294" in r.evidence
    assert "legacy" in r.evidence and "0.8007" in r.evidence
    assert "5 seeds" in r.evidence


def test_real_gate_falls_back_to_the_legacy_field_when_live_is_absent(tmp_path):
    """A file without the live field still passes — it just says which it read."""
    doc = {"aggregate_mean_auc_REAL": 0.80072017249287, "n_seeds": 3}
    r = pf.gate_p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds(
        "g", "P3", "x", _write(tmp_path, doc)
    )
    assert r.status == "PASS"
    assert "0.8007" in r.evidence
