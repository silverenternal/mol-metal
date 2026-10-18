"""Tests for the MSRS-7 external / temporal / scaffold validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts import msrs_external_validation as V


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------
def test_parse_year_accepts_delimited_years():
    assert V.parse_year("10.1016/j.ejmech.2018.04.044") == 2018
    assert V.parse_year("10.1021/acs.jmedchem.2020.01234") == 2020
    assert V.parse_year("10.1039/D0DT01234A") is None


def test_parse_year_rejects_volume_and_page_numbers():
    # 00319 is a page-like token, not a year; the delimiter rule must not fire.
    assert V.parse_year("10.1021/acs.inorgchem.0c00319") is None
    # A manuscript number glued to the year is also rejected: the rule requires a
    # delimiter after the four digits, which costs coverage but keeps the labels
    # plausible (measured coverage on potency is 30.6%).
    assert V.parse_year("10.1002/anie.201912345") is None


def test_parse_year_rejects_out_of_range_and_non_strings():
    assert V.parse_year("10.1000/journal.1899.001") is None
    assert V.parse_year("10.1000/journal.2091.001") is None
    assert V.parse_year(None) is None
    assert V.parse_year(12345) is None


def test_parse_year_takes_the_first_valid_hit():
    # Two year-like tokens; the first delimited one wins.
    assert V.parse_year("10.1000/abc.1999.suppl.2001.001") == 1999
    # An out-of-range token before a valid one must not win.
    assert V.parse_year("10.1000/abc.1899.x.2003.001") == 2003


# ---------------------------------------------------------------------------
# Year pooling
# ---------------------------------------------------------------------------
def test_year_group_pools_sparse_years():
    counts = {2020: 500, 2019: 50}
    assert V.year_group("10.1000/x.2020.001", counts, min_rows=100) == "2020"
    assert V.year_group("10.1000/x.2019.001", counts, min_rows=100) == V.EARLY_YEAR_LABEL
    assert V.year_group("10.1000/x.0c001", counts) is None


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------
def test_murcko_scaffold_strips_decorations():
    assert V.murcko_scaffold("c1ccccc1CCO") == "c1ccccc1"
    assert V.murcko_scaffold("CCO") == "ACYCLIC"
    assert V.murcko_scaffold(None) == "UNPARSED"
    assert V.murcko_scaffold("not-a-smiles") == "UNPARSED"


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------
def _frame() -> pd.DataFrame:
    rows = []
    for i in range(30):
        for metal, offset in (("Ru", 0.0), ("Ir", 0.5)):
            rows.append(
                {
                    "ligand_main_fragment": f"lig{i % 10}",
                    "ligand_canonical": f"L{i}",
                    "metal": metal,
                    "cell_line_key": "A549",
                    "doi": f"10.1000/x.20{20 + i % 3}.00{i % 6}",
                    "time_h": 48.0,
                    "scaffold": f"scaf{i % 6}",
                    "year_group": str(2020 + i % 3),
                    "value_log10": float(i % 4) * 0.2 + offset,
                    "endpoint": "potency",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("scheme,key", [("random", "ligand_main_fragment"), ("doi", "doi"),
                                       ("scaffold", "scaffold"), ("year", "year_group")])
def test_fold_ids_never_split_a_group(scheme, key):
    df = _frame()
    folds = V.fold_ids(df, scheme)
    assert (folds >= 0).all()
    for g in df[key].unique():
        assert len(set(folds[df[key] == g])) == 1


def test_fold_ids_leaves_unparsed_years_unassigned():
    df = _frame()
    df.loc[:3, "year_group"] = None
    folds = V.fold_ids(df, "year")
    assert (folds[:4] == -1).all()
    assert (folds[4:] >= 0).all()


def test_fold_ids_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        V.fold_ids(_frame(), "metal")


# ---------------------------------------------------------------------------
# OOF machinery
# ---------------------------------------------------------------------------
def test_oof_by_folds_leaves_unassigned_rows_nan():
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(40, 5)))
    y = rng.normal(size=40)
    fold_id = np.array([0] * 20 + [-1] * 20)
    # A single fold has no training data, so give it a real second fold.
    fold_id[10:20] = 1
    pred = V.oof_by_folds(X, y, fold_id, None)
    assert np.isnan(pred[20:]).all()
    assert not np.isnan(pred[:20]).any()


def test_baseline_by_folds_uses_training_folds():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    metals = np.array(["Ru", "Ir", "Ru", "Ir"])
    fold_id = np.array([0, 0, 1, 1])
    out = V._baseline_by_folds(y, metals, fold_id)
    assert out.tolist() == pytest.approx([2.0, 3.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def _synthetic() -> pd.DataFrame:
    smiles = [
        "CCO", "c1ccccc1", "CCN", "c1ccncc1", "CC(=O)O",
        "CC(C)O", "c1ccc(O)cc1", "CCCl", "c1ccsc1", "CC(C)(C)O",
    ]
    dois = [
        "10.1000/a.2019.001", "10.1000/a.2019.002", "10.1000/b.2020.001",
        "10.1000/b.2020.002", "10.1000/c.2021.001", "10.1000/c.2021.002",
    ]
    rows = []
    for i in range(60):
        for metal, offset in (("Ru", 0.0), ("Ir", 0.5)):
            rows.append(
                {
                    "endpoint": "potency",
                    "value_log10": float(i % 5) * 0.2 + offset,
                    "ligand_main_fragment": smiles[i % len(smiles)],
                    "ligand_canonical": f"L{i:03d}",
                    "metal": metal,
                    "cell_line_key": "A549",
                    "doi": dois[i % len(dois)],
                    "time_h": 48.0,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_cell_reports_every_scheme():
    df = _synthetic()
    descriptors = pd.DataFrame(
        {
            "metal_symbol": list(V.METALS),
            "ie1_kj": [710.2, 880.0, 719.7, 840.0, 760.0],
            "ie2_kj": [1620.0, 1600.0, 1740.0, 1600.0, 1260.0],
            "pauling_en": [2.20, 2.20, 2.28, 2.20, 1.90],
        }
    )
    for scheme in ("random", "doi", "scaffold", "year"):
        out = V.evaluate_cell(df, "potency", scheme, descriptors, verbose=False, min_year_rows=0)
        assert out["status"] == "OK", (scheme, out)
        assert out["n_evaluated"] == 120
        assert out["n_folds"] >= 2
        assert set(out["models"]) == {"additive", "interaction", "per_metal_mean"}
        assert out["delta_sign"]["n_pairs"] > 0


def test_evaluate_cell_drops_unparseable_years():
    df = _synthetic()
    df.loc[df["doi"] == "10.1000/a.2019.001", "doi"] = "10.1000/xyz"
    descriptors = pd.DataFrame(
        {
            "metal_symbol": list(V.METALS),
            "ie1_kj": [710.2, 880.0, 719.7, 840.0, 760.0],
            "ie2_kj": [1620.0, 1600.0, 1740.0, 1600.0, 1260.0],
            "pauling_en": [2.20, 2.20, 2.28, 2.20, 1.90],
        }
    )
    out = V.evaluate_cell(df, "potency", "year", descriptors, verbose=False, min_year_rows=0)
    assert out["status"] == "OK"
    assert out["n_unassigned"] > 0
    assert out["assignment_coverage"] < 1.0
    assert out["n_evaluated"] < 120
