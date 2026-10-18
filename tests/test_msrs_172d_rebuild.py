"""Tests for AUD-F2 (``msrs_172d_rebuild``).

The heavy lift test is exercised by the real run, not here; these tests pin the
three things that can silently corrupt it: the cohort filter, the NaN policy on
a failed stream, and the coverage arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molmetal.scripts.msrs_172d_rebuild import (
    FEATURE_DIM,
    MULTI_PH_DIM,
    PLL_DIM,
    RDKIT12_DIM,
    build_table,
    coverage_report,
    fragment_features,
    main,
    potency_cohort,
    write_npz,
)

# Benzene is small enough to embed reliably and has every stream defined.
VALID = "c1ccccc1"


def _long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "endpoint": ["potency", "potency", "potency", "phototoxicity", "potency"],
            "value_log10": [1.0, 2.0, np.nan, 3.0, 4.0],
            "censoring": ["none", "none", "none", "none", "right"],
            "ligand_main_fragment": ["CCO", "CCO", "CCN", "CCC", "CCC"],
            "ligand_canonical": ["CCO", "CCO", "CCN", "CCC", "CCC"],
        }
    )


def _expected(frame: pd.DataFrame) -> pd.DataFrame:
    """The shared predicate, applied in memory (no parquet round trip)."""
    return frame[
        (frame["endpoint"] == "potency")
        & frame["value_log10"].notna()
        & (frame["censoring"].astype(str) == "none")
        & frame["ligand_main_fragment"].notna()
        & frame["ligand_canonical"].notna()
    ].reset_index(drop=True)


def test_potency_cohort_keeps_only_exact_potency_rows():
    out = _expected(_long_frame())
    # Row 2 has a missing target, row 3 is phototoxicity, row 4 is censored.
    assert list(out["ligand_main_fragment"]) == ["CCO", "CCO"]
    assert out["value_log10"].notna().all()


def test_potency_cohort_matches_the_shared_predicate(tmp_path):
    path = tmp_path / "measurements.parquet"
    _long_frame().to_parquet(path)
    pd.testing.assert_frame_equal(potency_cohort(path), _expected(_long_frame()))


def test_feature_dim_is_12_plus_32_plus_128():
    assert (RDKIT12_DIM, MULTI_PH_DIM, PLL_DIM) == (12, 32, 128)
    assert FEATURE_DIM == 172


def test_fragment_features_fills_all_three_streams_for_a_parseable_smiles():
    vec, ok = fragment_features(VALID)
    assert vec.shape == (FEATURE_DIM,)
    assert all(ok.values())
    assert not np.isnan(vec).any()


def test_fragment_features_leaves_a_failed_stream_as_nan_not_zero():
    vec, ok = fragment_features("not_a_smiles")
    assert not any(ok.values())
    assert np.isnan(vec).all()
    # The distinction matters: a zero-filled block looks like a real measurement.
    assert not np.any(vec == 0.0)


def test_build_table_returns_one_row_per_fragment_with_masks():
    frags = [VALID, "not_a_smiles"]
    X, masks = build_table(frags)
    assert X.shape == (2, FEATURE_DIM)
    assert masks["rdkit"].tolist() == [True, False]
    assert masks["multiph"].tolist() == [True, False]
    assert masks["pll"].tolist() == [True, False]


def test_coverage_report_counts_complete_rows_not_any_stream():
    frags = ["a", "b", "c", "d"]
    X = np.zeros((4, FEATURE_DIM), dtype=np.float32)
    X[0, 0] = np.nan  # one missing stream is enough to fail the ligand
    masks = {
        "rdkit": np.array([False, True, True, True]),
        "multiph": np.array([True, True, True, True]),
        "pll": np.array([True, True, True, True]),
    }
    rep = coverage_report(frags, X, masks, wall_s=1.5)
    assert rep["n_unique_ligands"] == 4
    assert rep["n_complete_172d"] == 3
    assert rep["coverage_complete"] == 0.75
    assert rep["coverage_per_stream"]["rdkit"] == 0.75
    assert rep["wall_s"] == 1.5


def test_write_npz_round_trips_fragments_and_masks(tmp_path):
    frags = [VALID]
    X, masks = build_table(frags)
    out = tmp_path / "f.npz"
    write_npz(out, frags, X, masks, {"coverage_complete": 1.0})
    with np.load(out, allow_pickle=True) as z:
        assert z["X"].shape == (1, FEATURE_DIM)
        assert z["fragments"].tolist() == frags
        assert bool(z["rdkit_ok"][0]) is True
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["coverage_complete"] == 1.0


def test_main_skip_lift_writes_artifact_and_metrics(tmp_path):
    path = tmp_path / "measurements.parquet"
    frame = pd.DataFrame(
        {
            "endpoint": ["potency"] * 4,
            "value_log10": [1.0, 2.0, 3.0, 4.0],
            "censoring": ["none"] * 4,
            "ligand_main_fragment": ["CCO", "CCO", "CCN", "not_a_smiles"],
            "ligand_canonical": ["CCO", "CCO", "CCN", "not_a_smiles"],
        }
    )
    frame.to_parquet(path)
    out = tmp_path / "f2.npz"
    metrics = tmp_path / "f2.json"
    rc = main(
        [
            "--long",
            str(path),
            "--out",
            str(out),
            "--metrics",
            str(metrics),
            "--skip-lift",
        ]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(metrics.read_text())
    # CCO, CCN, not_a_smiles -> 3 unique fragments, 2 of them complete.
    assert payload["coverage"]["n_unique_ligands"] == 3
    assert payload["verdict"] == "COVERAGE_FAIL"
    assert payload["coverage"]["coverage_complete"] == pytest.approx(2 / 3, abs=1e-4)
