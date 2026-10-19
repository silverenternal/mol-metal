"""FIX-L4 tests — the cell-feature blocker must be measured, not assumed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from molmetal.scripts.msrs_cell_features import (
    build_artifact,
    coverage_table,
    min_lines_for_coverage,
    read_identity_ceiling,
    scan_for_annotation_files,
)


# ---------------------------------------------------------------------------
# disk scan
# ---------------------------------------------------------------------------


def test_scan_finds_a_known_annotation_dataset(tmp_path: Path):
    (tmp_path / "ccle_expression.csv").write_text("a,b\n1,2\n")
    hits = scan_for_annotation_files([tmp_path])
    assert [Path(h["path"]).name for h in hits] == ["ccle_expression.csv"]


def test_scan_ignores_unrelated_files(tmp_path: Path):
    (tmp_path / "measurements.parquet").write_text("x")
    assert scan_for_annotation_files([tmp_path]) == []


def test_scan_matches_case_insensitively(tmp_path: Path):
    (tmp_path / "GDSC2_MUTATIONS.csv").write_text("x")
    assert len(scan_for_annotation_files([tmp_path])) == 1


def test_scan_respects_the_depth_limit(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "ccle.csv").write_text("x")
    assert scan_for_annotation_files([tmp_path], max_depth=2) == []
    assert len(scan_for_annotation_files([tmp_path], max_depth=6)) == 1


# ---------------------------------------------------------------------------
# coverage arithmetic
# ---------------------------------------------------------------------------


def test_coverage_table_reports_cumulative_shares():
    counts = pd.Series([50, 30, 20], index=["a", "b", "c"])
    out = coverage_table(counts, ks=(1, 2, 3))
    assert [row["share"] for row in out] == [0.5, 0.8, 1.0]


def test_coverage_table_skips_k_larger_than_the_corpus():
    counts = pd.Series([1, 1], index=["a", "b"])
    assert len(coverage_table(counts, ks=(1, 5))) == 1


def test_min_lines_for_coverage_returns_a_position_not_a_label():
    # The first version returned the cell-line key instead of k.
    counts = pd.Series([50, 30, 15, 5], index=["a", "b", "c", "d"])
    assert min_lines_for_coverage(counts, 0.80) == 2
    assert min_lines_for_coverage(counts, 0.95) == 3
    assert min_lines_for_coverage(counts, 0.99) == 4


def test_min_lines_for_coverage_returns_none_on_an_empty_series():
    assert min_lines_for_coverage(pd.Series(dtype=float), 0.8) is None


# ---------------------------------------------------------------------------
# identity ceiling
# ---------------------------------------------------------------------------


def test_missing_ablation_artifact_is_reported_not_crashed(tmp_path: Path):
    assert read_identity_ceiling(tmp_path / "nope.json") == {"available": False}


def test_identity_ceiling_reads_both_protocols(tmp_path: Path):
    path = tmp_path / "abl.json"
    path.write_text(
        json.dumps(
            {
                "protocols": {
                    "ligand": {"arms": {"cell_blind": {"delta_r2_vs_full": 0.06, "delta_ci": {"ci95": [0.05, 0.08]}}}},
                    "doi": {"arms": {"cell_blind": {"delta_r2_vs_full": 0.04, "delta_ci": {"ci95": [0.02, 0.06]}}}},
                }
            }
        )
    )
    out = read_identity_ceiling(path)
    assert out["available"] is True
    assert out["ligand"]["delta_r2_vs_full"] == 0.06
    assert out["doi"]["delta_r2_vs_full"] == 0.04


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def _msrs_fixture(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "endpoint": ["potency"] * 100,
            "value_log10": [1.0] * 100,
            "censoring": ["none"] * 100,
            "ligand_main_fragment": ["lig"] * 100,
            "cell_line_key": ["A549"] * 60 + ["MCF7"] * 40,
        }
    )
    path = tmp_path / "msrs.parquet"
    frame.to_parquet(path)
    return path


def test_no_annotation_file_blocks_the_work_package(tmp_path: Path):
    msrs = _msrs_fixture(tmp_path)
    art = build_artifact(msrs, tmp_path / "missing_abl.json", roots=[tmp_path / "data"])
    assert art["verdict"] == "BLOCKED-PENDING-DATA"
    assert art["on_disk_annotation_files"] == []
    assert art["corpus"]["min_lines_for_80pct"] == 2


def test_one_annotation_file_is_not_enough_without_the_gain_measurement(tmp_path: Path):
    msrs = _msrs_fixture(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "ccle_expression.csv").write_text("x")
    art = build_artifact(msrs, tmp_path / "missing_abl.json", roots=[data])
    # The row gate is met, so the blocker is now the unmeasured +0.005 gain.
    assert art["on_disk_annotation_files"]
    assert art["corpus"]["min_lines_for_80pct"] == 2
    assert art["identity_ceiling"] == {"available": False}
    # No measured gain was supplied, so availability alone cannot PASS.
    assert art["verdict"] == "BLOCKED-PENDING-DATA"
