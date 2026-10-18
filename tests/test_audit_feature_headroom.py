"""Tests for the feature-headroom audit (which covariates never reach the model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from molmetal.scripts import audit_feature_headroom as A


# ---------------------------------------------------------------------------
# time_bucket
# ---------------------------------------------------------------------------
def test_time_bucket_keeps_the_four_dense_durations():
    values = pd.Series([24, 48, 72, 96, 24.0])
    assert A.time_bucket(values).tolist() == ["24h", "48h", "72h", "96h", "24h"]


def test_time_bucket_pools_sparse_durations_and_keeps_missing_separate():
    values = pd.Series([6, 12, 120, None], dtype=object)
    out = A.time_bucket(values)
    assert out.tolist() == ["other", "other", "other", "missing"]
    # "missing" must not be silently folded into "other": the two mean different
    # things (duration unknown vs. a duration that is rare).
    assert out.nunique() == 2


def test_time_bucket_does_not_mutate_input():
    values = pd.Series([24, 48])
    A.time_bucket(values)
    assert values.tolist() == [24, 48]


# ---------------------------------------------------------------------------
# categorical_codes
# ---------------------------------------------------------------------------
def test_categorical_codes_missing_is_its_own_level():
    df = pd.DataFrame({"metal": ["Pt", None, "Ru", "Pt"]})
    codes, levels = A.categorical_codes(df, ["metal"])
    assert levels == {"metal": 3}
    # sort=True is plain string order: "Pt" < "Ru" < "__missing__" (underscore sorts last)
    assert codes[:, 0].tolist() == [0, 2, 1, 0]


def test_categorical_codes_empty_column_list():
    df = pd.DataFrame({"metal": ["Pt"]})
    codes, levels = A.categorical_codes(df, [])
    assert codes.shape == (1, 0)
    assert levels == {}


def test_categorical_codes_are_dense_integers_per_column():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", None]})
    codes, levels = A.categorical_codes(df, ["a", "b"])
    assert codes.dtype == np.int32
    assert codes.shape == (2, 2)
    assert levels == {"a": 2, "b": 2}


# ---------------------------------------------------------------------------
# build_stage
# ---------------------------------------------------------------------------
def test_build_stage_ligand_only_has_no_categoricals():
    df = pd.DataFrame({"metal": ["Pt", "Ru"]})
    fp = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    X, cat_idx, levels = A.build_stage(df, ["ligand_ecfp4"], fp)
    assert X.shape == (2, 2)
    assert cat_idx == []
    assert levels == {}


def test_build_stage_appends_categoricals_after_the_fingerprint():
    df = pd.DataFrame({"metal": ["Pt", "Ru"], "cell_line_key": ["A549", "A549"]})
    fp = sparse.csr_matrix(np.ones((2, 5)))
    X, cat_idx, levels = A.build_stage(df, ["ligand_ecfp4", "metal", "cell_line_key"], fp)
    assert X.shape == (2, 7)
    # categorical indices must point at the two dense columns, not the fingerprint
    assert cat_idx == [5, 6]
    assert levels == {"metal": 2, "cell_line_key": 1}
    # a single-level categorical carries no information but must still be built
    assert X[:, 6].toarray().ravel().tolist() == [0.0, 0.0]


def test_build_stage_without_fingerprint_starts_at_zero():
    df = pd.DataFrame({"metal": ["Pt", "Ru"]})
    fp = sparse.csr_matrix(np.ones((2, 5)))
    X, cat_idx, _ = A.build_stage(df, ["metal"], fp)
    assert X.shape == (2, 1)
    assert cat_idx == [0]


# ---------------------------------------------------------------------------
# oof_regression
# ---------------------------------------------------------------------------
def _toy_regression(n_groups: int = 40, per_group: int = 20):
    # The production LGB_PARAMS require min_child_samples=20, so a 40-row toy set
    # cannot split at all and every stage would predict the mean.  Keep the toy set
    # large enough for the real hyper-parameters to be exercised.
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=(len(groups), 3))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.05, size=len(groups))
    return sparse.csr_matrix(x), y, groups


def test_oof_regression_returns_a_prediction_for_every_row():
    X, y, groups = _toy_regression()
    oof, fold_id = A.oof_regression(X, y, groups, [])
    assert oof.shape == y.shape
    assert not np.isnan(oof).any()
    assert set(fold_id.tolist()) == set(range(A.N_FOLDS))


def test_oof_regression_learns_a_linear_signal():
    X, y, groups = _toy_regression()
    oof, _ = A.oof_regression(X, y, groups, [])
    assert A.score(y, oof)["r2"] > 0.8


def test_oof_regression_accepts_categorical_indices():
    X, y, groups = _toy_regression()
    Xc = sparse.hstack([X, sparse.csr_matrix((X.shape[0], 1))], format="csr")
    oof, _ = A.oof_regression(Xc, y, groups, [3])
    assert not np.isnan(oof).any()


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def test_score_known_values():
    y = np.array([0.0, 2.0])
    assert A.score(y, y) == {"r2": 1.0, "mae": 0.0, "rmse": 0.0}
    out = A.score(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert out["mae"] == 1.0
    assert out["rmse"] == 1.0
    assert out["r2"] == 0.0


def test_score_r2_is_undefined_for_a_constant_target():
    # A degenerate target has zero total variance; R² is 0/0.  Recorded, not hidden.
    out = A.score(np.zeros(3), np.zeros(3))
    assert np.isnan(out["r2"])
    assert out["mae"] == 0.0


# ---------------------------------------------------------------------------
# censoring / reliability / coverage audits
# ---------------------------------------------------------------------------
def test_censoring_audit_separates_exact_from_censored():
    # Censored rows are KEPT at the bound (value_log10 = log10(bound)), not dropped.
    long = pd.DataFrame(
        {
            "endpoint": ["potency"] * 4 + ["phototoxicity"],
            "metal": ["Ru", "Ru", "Re", "Re", "Ru"],
            "value_log10": [0.5, 1.0, 2.0, 2.0, 2.0],
            "censor_bound": [np.nan, np.nan, 100.0, 200.0, np.nan],
            "censoring": ["none", "none", "right", "right", "none"],
            "all_censored": [False, False, True, True, False],
        }
    )
    out = A.censoring_audit(long)
    assert out["n_rows_total"] == 4
    assert out["n_exact"] == 4
    assert out["n_censored"] == 2
    assert out["censored_share"] == 0.5
    assert out["censoring_kinds"] == {"right": 2}
    assert out["n_all_censored_groups"] == 2
    # two rows sit exactly at the 100 uM ceiling (log10 = 2.0)
    assert out["n_rows_at_the_ceiling"] == 2
    assert out["exact_median_log10"] == 1.5
    # median bound 150 -> log10(150) = 2.1761
    assert out["censored_bound_median_log10"] == pytest.approx(2.1761, abs=1e-4)
    # differential censoring: Re 100% vs Ru 0% -> spread 1.0
    assert out["censored_share_by_metal"]["Re"]["censored_share"] == 1.0
    assert out["censored_share_by_metal"]["Ru"]["censored_share"] == 0.0
    assert out["censored_share_spread"] == 1.0


def test_censoring_audit_handles_no_censored_rows():
    long = pd.DataFrame(
        {
            "endpoint": ["potency"],
            "metal": ["Ru"],
            "value_log10": [1.0],
            "censor_bound": [np.nan],
            "censoring": ["none"],
            "all_censored": [False],
        }
    )
    out = A.censoring_audit(long)
    assert out["n_censored"] == 0
    assert out["censored_bound_median_log10"] is None
    assert out["censoring_kinds"] == {}
    assert out["censored_share_spread"] == 0.0


def test_reliability_audit_counts_replicates():
    dedup = pd.DataFrame(
        {
            "endpoint": ["potency"] * 4 + ["cisplatin_ratio"],
            "value_log10": [1.0, 1.1, np.nan, 0.9, 0.3],
            "n_exact": [1, 1, 3, 2, 5],
        }
    )
    out = A.reliability_audit(dedup)
    assert out["n_rows"] == 3
    assert out["n_exact_distribution"] == {"1": 2, "2": 1}
    assert out["share_single_measurement"] == pytest.approx(0.6667, abs=1e-4)


def test_coverage_audit_reports_cardinality_and_missingness():
    df = pd.DataFrame(
        {
            "metal": ["Pt", "Ru", None],
            "cell_line_key": ["A549", "A549", "HeLa"],
            "time_h": [48, 48, 24],
            "ligand_main_fragment": ["C", "CC", "CCC"],
            "ligand_canonical": ["C", "CC", "CCC"],
        }
    )
    out = A.coverage_audit(df)
    assert out["metal"] == {"n_unique": 2, "missing_share": pytest.approx(0.3333, abs=1e-4)}
    assert out["cell_line_key"]["n_unique"] == 2
    assert out["cell_line_top40_share"] == 1.0
    assert out["time_top_values"] == {"48": 2, "24": 1}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_end_to_end_writes_json_with_headroom(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 80
    long = pd.DataFrame(
        {
            "endpoint": ["potency"] * n,
            "value_log10": rng.normal(size=n),
            "censor_bound": np.nan,
            "censoring": "none",
            "all_censored": False,
            "ligand_main_fragment": [f"C{i % 10}" for i in range(n)],
            "ligand_canonical": [f"C{i % 10}" for i in range(n)],
            "metal": rng.choice(["Pt", "Ru", "Rh"], size=n),
            "oxidation_state": rng.choice([2, 3], size=n),
            "charge_complex": rng.choice([0, 1], size=n),
            "counterion": rng.choice(["Cl", None], size=n),
            "cell_line_key": rng.choice(["A549", "HeLa"], size=n),
            "time_h": rng.choice([24, 48, 72], size=n),
            "year": rng.choice([2015, 2020], size=n),
            "doi": rng.choice(["10.1/a", "10.1/b"], size=n),
        }
    )
    dedup = long.assign(n_exact=1)
    long_path = tmp_path / "long.parquet"
    dedup_path = tmp_path / "dedup.parquet"
    long.to_parquet(long_path)
    dedup.to_parquet(dedup_path)
    out_path = tmp_path / "out.json"

    # 10 ligand groups < N_FOLDS is not the case here, but keep the run cheap
    monkeypatch.setattr(A, "morgan_matrix", lambda smiles: sparse.csr_matrix((len(smiles), 8)))
    monkeypatch.setattr(A, "LGB_PARAMS", {**A.LGB_PARAMS, "n_estimators": 5})

    rc = A.main(["--long", str(long_path), "--dedup", str(dedup_path), "--out", str(out_path)])
    assert rc == 0
    import json

    payload = json.loads(out_path.read_text())
    assert payload["artifact"] == "feature_headroom_audit"
    assert set(payload["stages"]) == {name for name, _ in A.STAGES}
    assert payload["headroom"]["production_r2"] == payload["stages"][
        "+metal+cell (production)"
    ]["r2"]
    assert payload["censoring"]["n_censored"] == 0
