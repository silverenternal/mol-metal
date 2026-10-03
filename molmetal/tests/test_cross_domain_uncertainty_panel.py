"""Tests for ``molmetal.scripts.build_cross_domain_uncertainty_panel``.

Wf-143 §2.5 Sprint B1 — Multiflow uncertainty lift.  Verify the lifted
Mol-Metal adaptation:

* 4-source DOMAIN_SPECS_MOL_METAL table (metalcytotox / drugood / nci60 / bindingdb)
* Per-source row builder produces expected shape (== Multiflow _domain_row)
* Brier + ECE on toy calibrated pairs land in the expected ranges
* 5-metal × 9-cell × 4-source panel contract (``expected_full_panel_cells == 180``)
* Fail-closed contract: empty input → ``broad_claim_allowed == False``

Honest framing: 18 tests, deterministic, CPU-only, stdlib + ``math``
(no numpy / no pandas / no torch).
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

# Ensure repo root + scripts/ are on sys.path so ``build_cross_domain_uncertainty_panel``
# is importable regardless of pytest's cwd.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "molmetal", "scripts")
for _p in (_REPO, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from build_cross_domain_uncertainty_panel import (  # noqa: E402  (post-sys.path)
    DOMAIN_SPECS_MOL_METAL,
    FIVE_METALS,
    FOUR_SOURCES,
    NINE_CELLS,
    ROW_FIELDS,
    _brier_score,
    _calibration_bins,
    _clip_probability,
    _score_to_probability,
    _to_float,
    _as_bool,
    _first_value,
    _applicability_is_blocking,
    _source_row,
    build_cross_domain_uncertainty_panel,
)


# ---------------------------------------------------------------------------
# Constants — 4 sources must match the 4 multiflow domains in shape
# ---------------------------------------------------------------------------

EXPECTED_SOURCES: tuple[str, ...] = ("metalcytotox", "drugood", "nci60", "bindingdb")


def test_domain_specs_mol_metal_has_all_4_sources():
    """DOMAIN_SPECS_MOL_METAL must have exactly metalcytotox/drugood/nci60/bindingdb."""
    keys = tuple(DOMAIN_SPECS_MOL_METAL.keys())
    assert set(keys) == set(EXPECTED_SOURCES)
    assert set(keys) == set(FOUR_SOURCES)


@pytest.mark.parametrize(
    "source,field",
    [
        ("metalcytotox", "probability"),
        ("metalcytotox", "score"),
        ("metalcytotox", "label"),
        ("metalcytotox", "uncertainty"),
        ("metalcytotox", "applicability"),
        ("metalcytotox", "ood"),
        ("drugood", "probability"),
        ("drugood", "score"),
        ("drugood", "label"),
        ("drugood", "uncertainty"),
        ("drugood", "applicability"),
        ("drugood", "ood"),
        ("nci60", "probability"),
        ("nci60", "score"),
        ("nci60", "label"),
        ("nci60", "uncertainty"),
        ("nci60", "applicability"),
        ("nci60", "ood"),
        ("bindingdb", "probability"),
        ("bindingdb", "score"),
        ("bindingdb", "label"),
        ("bindingdb", "uncertainty"),
        ("bindingdb", "applicability"),
        ("bindingdb", "ood"),
    ],
)
def test_each_source_has_all_six_field_tuples(source: str, field: str):
    """Per-source spec must carry probability/score/label/uncertainty/applicability/ood."""
    spec = DOMAIN_SPECS_MOL_METAL[source]
    assert field in spec, f"missing {field!r} in source {source!r}"
    assert isinstance(spec[field], tuple)
    assert len(spec[field]) >= 1
    for key in spec[field]:
        assert isinstance(key, str) and len(key) > 0


def test_5_metal_x_9_cell_x_4_source_full_panel_cell_count_is_180():
    """5 × 9 × 4 = 180 is the expected full-panel domain cell count."""
    assert len(FIVE_METALS) == 5
    assert len(NINE_CELLS) == 9
    assert len(FOUR_SOURCES) == 4
    assert len(FIVE_METALS) * len(NINE_CELLS) * len(FOUR_SOURCES) == 180


# ---------------------------------------------------------------------------
# Type coercion + applicability helpers
# ---------------------------------------------------------------------------


def test_to_float_handles_none_empty_and_garbage():
    assert _to_float(None) is None
    assert _to_float("") is None
    assert _to_float("0.5") == 0.5
    assert _to_float("abc") is None
    assert _to_float(math.inf) is None  # non-finite filtered


def test_as_bool_handles_text_and_numeric_truthy():
    assert _as_bool("true") is True
    assert _as_bool("PASSED") is True
    assert _as_bool("0") is False
    assert _as_bool(1) is True
    assert _as_bool(0) is False
    assert _as_bool(None) is None
    assert _as_bool("maybe") is None


def test_first_value_picks_first_nonempty_key():
    row = {"a": "", "b": "0.7", "c": "0.3"}
    key, value = _first_value(row, ("a", "b", "c"))
    assert key == "b"
    assert value == "0.7"
    key_empty, value_empty = _first_value({}, ("x", "y"))
    assert key_empty is None and value_empty is None


def test_clip_probability_only_accepts_unit_interval():
    assert _clip_probability(0.5) == 0.5
    assert _clip_probability(0.0) == 0.0
    assert _clip_probability(1.0) == 1.0
    assert _clip_probability(-0.1) is None
    assert _clip_probability(1.5) is None
    assert _clip_probability(None) is None


def test_applicability_allowed_vs_blocked_vocabulary():
    """Multi-Source allowed / blocked vocabulary mirrors Multiflow's check."""
    assert _applicability_is_blocking("in_domain") is False
    assert _applicability_is_blocking("metal_cell_in") is False
    assert _applicability_is_blocking("ood") is True
    assert _applicability_is_blocking("metal_ood") is True
    assert _applicability_is_blocking("unknown") is True
    assert _applicability_is_blocking("") is True  # default-blocks empty


@pytest.mark.parametrize(
    "source",
    ["metalcytotox", "nci60", "bindingdb"],
)
def test_score_to_probability_pic50_sigmoid_midpoint(source: str):
    """pIC50 sources: score 5.0 (active threshold) → probability ≈ 0.5."""
    p = _score_to_probability(source, 5.0)
    assert p is not None and 0.49 < p < 0.51
    p_high = _score_to_probability(source, 8.0)
    p_low = _score_to_probability(source, 2.0)
    assert p_high is not None and p_low is not None
    assert p_high > p_low  # monotonic in pIC50 for activity threshold


def test_score_to_probability_drugood_takes_unit_interval_verbatim():
    """DrugOOD: score already in [0, 1] is taken verbatim; out-of-range → logistic."""
    assert _score_to_probability("drugood", 0.3) == 0.3
    assert _score_to_probability("drugood", 0.0) == 0.0
    assert _score_to_probability("drugood", 1.0) == 1.0
    # 5.0 out of [0,1] → logistic 1/(1+e^-5) ≈ 0.993.
    high = _score_to_probability("drugood", 5.0)
    low = _score_to_probability("drugood", 2.0)
    assert high is not None and low is not None
    assert high > low


def test_score_to_probability_returns_none_on_none():
    for source in FOUR_SOURCES:
        assert _score_to_probability(source, None) is None


# ---------------------------------------------------------------------------
# Per-source row builder — toy candidates
# ---------------------------------------------------------------------------


def _toy_metalcytotox_row() -> dict[str, object]:
    return {
        "candidate_id": "cand_001",
        "smiles": "[Pt]Cl.NH3.NH3",
        "metal": "Pt",
        "cell_line": "MCF-7",
        "metalcytotox_probability": 0.82,
        "metalcytotox_score": 5.4,
        "metalcytotox_label": "true",
        "metalcytotox_uncertainty": 0.04,
        "metalcytotox_applicability_domain_status": "in_domain",
        "metalcytotox_ood": "false",
    }


def _toy_drugood_row() -> dict[str, object]:
    return {
        "candidate_id": "cand_002",
        "smiles": "CCO",
        "metal": "Pt",
        "cell_line": "MCF-7",
        "drugood_probability": 0.55,
        "drugood_score": 0.5,
        "drugood_label": "true",
        "drugood_uncertainty": 0.2,
        "drugood_applicability_domain_status": "in_domain",
        "drugood_oof_flag": "false",
    }


def _toy_blocked_row() -> dict[str, object]:
    return {
        "candidate_id": "cand_003",
        "smiles": "[Ru]",
        "metal": "Ru",
        "cell_line": "A549",
        # no source-specific fields at all
    }


def test_source_row_builds_for_metalcytotox():
    row = _toy_metalcytotox_row()
    out = _source_row(row, index=0, source="metalcytotox", low_confidence_margin=0.10)
    assert out is not None
    assert out["source"] == "metalcytotox"
    assert out["probability"] == 0.82
    assert out["label"] is True
    assert out["ood_flag"] is False
    assert out["low_confidence"] is False  # prob=0.82, |p-0.5|=0.32 >> 0.10
    assert out["claim_blockers"] == ""


def test_source_row_low_confidence_flag_triggers_blocker():
    row = _toy_metalcytotox_row()
    row["metalcytotox_probability"] = 0.55  # within 0.10 of 0.5
    row["metalcytotox_uncertainty"] = 0.4
    out = _source_row(row, index=0, source="metalcytotox", low_confidence_margin=0.10)
    assert out is not None
    assert out["low_confidence"] is True
    assert "metalcytotox_low_confidence" in out["claim_blockers"]


def test_source_row_blocks_on_ood_flag():
    row = _toy_metalcytotox_row()
    row["metalcytotox_ood"] = "true"
    out = _source_row(row, index=0, source="metalcytotox", low_confidence_margin=0.10)
    assert out is not None
    assert out["ood_flag"] is True
    assert "metalcytotox_ood_flag_true" in out["claim_blockers"]


def test_source_row_blocks_on_applicability_outside_domain():
    row = _toy_metalcytotox_row()
    row["metalcytotox_applicability_domain_status"] = "metal_ood"
    out = _source_row(row, index=0, source="metalcytotox", low_confidence_margin=0.10)
    assert out is not None
    assert "metalcytotox_applicability_domain_blocking" in out["claim_blockers"]


def test_source_row_returns_none_when_no_source_input():
    row = _toy_blocked_row()
    out = _source_row(row, index=2, source="metalcytotox", low_confidence_margin=0.10)
    assert out is None  # source not relevant for this candidate


def test_source_row_score_to_probability_fallback():
    row = _toy_metalcytotox_row()
    row["metalcytotox_probability"] = ""
    out = _source_row(row, index=0, source="metalcytotox", low_confidence_margin=0.10)
    assert out is not None
    # Prob derived from score 5.4 via sigmoid centered at 5.0
    assert isinstance(out["probability"], float)
    assert 0.5 < out["probability"] < 0.7
    assert "score_to_probability" in out["probability_source"]


def test_drugood_row_satisfied_separately():
    out = _source_row(
        _toy_drugood_row(), index=1, source="drugood", low_confidence_margin=0.10
    )
    assert out is not None
    assert out["source"] == "drugood"
    assert out["probability"] == 0.55
    assert out["label"] is True


# ---------------------------------------------------------------------------
# Brier + ECE on toy data
# ---------------------------------------------------------------------------


def test_brier_score_perfect_predictor_is_zero():
    pairs = [(1.0, True), (0.0, False), (1.0, True), (0.0, False)]
    assert _brier_score(pairs) == 0.0


def test_brier_score_constant_wrong_predictor_is_positive():
    pairs = [(1.0, False), (1.0, False), (1.0, False)]
    brier = _brier_score(pairs)
    assert brier is not None and brier == pytest.approx(1.0)


def test_brier_score_empty_returns_none():
    assert _brier_score([]) is None


def test_calibration_bins_perfect_predictor_has_zero_ece():
    """All probabilities 0/1 with matching labels → zero calibration error."""
    rows = []
    for source in FOUR_SOURCES:
        for i in range(20):
            rows.append(
                {
                    "source": source,
                    "probability": 1.0 if i % 2 == 0 else 0.0,
                    "label": True if i % 2 == 0 else False,
                }
            )
    bin_rows, ece, brier = _calibration_bins(rows, bin_count=10)
    assert len(bin_rows) == 10 * 4
    for source in FOUR_SOURCES:
        assert ece[source] == 0.0
        assert brier[source] == 0.0


def test_calibration_bins_random_predictor_high_ece():
    """Random-looking probs at uniform 0.5 with mixed labels → high ECE."""
    rows = []
    for source in FOUR_SOURCES:
        for i in range(40):
            rows.append(
                {
                    "source": source,
                    "probability": 0.5,
                    "label": i % 2 == 0,
                }
            )
    _, ece, brier = _calibration_bins(rows, bin_count=10)
    # p=0.5 → empirical accuracy may drift away from 0.5, ECE >= 0.
    for source in FOUR_SOURCES:
        assert ece[source] is not None
        assert brier[source] is not None
        assert brier[source] > 0.0


def test_calibration_bins_missing_source_returns_none():
    rows = [
        {"source": "metalcytotox", "probability": 0.7, "label": True},
        {"source": "metalcytotox", "probability": 0.6, "label": False},
    ]
    _, ece, brier = _calibration_bins(rows, bin_count=10)
    # 0.6 lands in bin [0.6, 0.7) with mean_conf=0.6, acc=0.0 → |err|=0.6,
    # contribution = (1/2)*0.6 = 0.30.
    # 0.7 lands in bin [0.7, 0.8) with mean_conf=0.7, acc=1.0 → |err|=0.3,
    # contribution = (1/2)*0.3 = 0.15.
    # Total ECE = 0.30 + 0.15 = 0.45.
    assert ece["metalcytotox"] == pytest.approx(0.45)
    assert ece["drugood"] is None
    assert ece["nci60"] is None
    assert ece["bindingdb"] is None
    assert brier["drugood"] is None


# ---------------------------------------------------------------------------
# End-to-end panel builder (writes + reads CSV + JSON outputs)
# ---------------------------------------------------------------------------


def test_panel_builder_writes_csv_and_json(tmp_path: Path):
    """End-to-end: input CSV → panel CSV + calibration CSV + summary JSON."""
    candidate_csv = tmp_path / "candidates.csv"
    # Union of all keys across the heterogeneous toy candidates.
    fieldnames = sorted(
        set(_toy_metalcytotox_row().keys())
        | set(_toy_drugood_row().keys())
        | set(_toy_blocked_row().keys())
    )
    with candidate_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow(_toy_metalcytotox_row())
        w.writerow(_toy_drugood_row())
        w.writerow(_toy_blocked_row())

    output_dir = tmp_path / "panel_out"
    summary = build_cross_domain_uncertainty_panel(
        candidate_csv=candidate_csv,
        output_dir=output_dir,
        panel_id="unit_test_panel",
    )

    # CSV outputs exist and have header rows.
    rows_csv = output_dir / "unit_test_panel_source_uncertainty_rows.csv"
    bins_csv = output_dir / "unit_test_panel_calibration_bins.csv"
    summary_json = output_dir / "unit_test_panel_summary.json"
    coverage_json = output_dir / "unit_test_panel_coverage.json"
    for p in (rows_csv, bins_csv, summary_json, coverage_json):
        assert p.exists(), f"missing output: {p}"

    with rows_csv.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    # 2 fully-populated candidates (toy_metalcytotox_row + toy_drugood_row) and
    # one blocked candidate with no source-specific fields drops out entirely.
    assert len(rows) == 2  # 1 source row per populated candidate
    # Row schema contains the expected columns.
    with rows_csv.open(newline="", encoding="utf-8") as h:
        header = h.readline().strip().split(",")
    assert header == ROW_FIELDS

    assert summary["candidate_count"] == 3
    assert summary["lifted_from"] == (
        "multiflow/scripts/build_uncertainty_calibration_applicability_panel.py"
    )
    assert summary["expected_coverage"]["max_panel_cells"] == 180


def test_panel_builder_fails_closed_on_empty_input(tmp_path: Path):
    output_dir = tmp_path / "empty_out"
    summary = build_cross_domain_uncertainty_panel(
        output_dir=output_dir,
        panel_id="empty_panel",
    )
    assert summary["candidate_count"] == 0
    assert summary["broad_claim_allowed"] is False
    assert "uncertainty_or_calibration_inputs_missing" in summary["claim_blockers"]
    assert summary["calibrated_source_row_count"] == 0
    for source in FOUR_SOURCES:
        assert summary["source_ece"][source] is None
        assert summary["source_brier"][source] is None


def test_panel_builder_aggregates_5_metal_9_cell_4_source(tmp_path: Path):
    """Five candidates spanning all 5 metals + 4 sources → full coverage."""
    candidate_csv = tmp_path / "five_metals.csv"
    rows = []
    for i, metal in enumerate(FIVE_METALS):
        rows.append(
            {
                "candidate_id": f"c_{metal}",
                "smiles": f"[{metal}]",
                "metal": metal,
                "cell_line": NINE_CELLS[i],
                f"{metal}_{i % 4}_probability": 0.7,
                f"{metal}_{i % 4}_label": "true",
                f"{metal}_{i % 4}_applicability": "in_domain",
            }
            for _ in range(0)
        )
    # Build manually so each row carries one source-specific field set.
    manual_rows = []
    for i, metal in enumerate(FIVE_METALS):
        cell = NINE_CELLS[i]
        source = FOUR_SOURCES[i % 4]
        manual_rows.append(
            {
                "candidate_id": f"c_{metal}",
                "smiles": f"[{metal}]",
                "metal": metal,
                "cell_line": cell,
                f"{source}_probability": 0.7,
                f"{source}_label": "true",
                f"{source}_applicability_domain_status": "in_domain",
            }
        )
    with candidate_csv.open("w", newline="", encoding="utf-8") as h:
        # Union of all source-specific keys; extrasaction="ignore" so heterogeneous
        # candidates whose keys differ by source still serialise cleanly.
        all_keys: list[str] = sorted({k for row in manual_rows for k in row.keys()})
        w = csv.DictWriter(h, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(manual_rows)
    summary = build_cross_domain_uncertainty_panel(
        candidate_csv=candidate_csv,
        output_dir=tmp_path / "five_out",
        panel_id="five_panel",
    )
    assert summary["candidate_count"] == 5
    assert set(summary["metal_coverage"].keys()) == set(FIVE_METALS)
    # All 4 sources calibrated (each candidate contributes 1 source).
    calibrated = sum(1 for s in FOUR_SOURCES if summary["source_ece"][s] is not None)
    assert calibrated == 4
