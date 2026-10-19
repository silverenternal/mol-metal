"""MET-2 tests — the partitions and the pair metrics must be honest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts import msrs_generalization as gen
from molmetal.scripts.msrs_generalization import (
    bootstrap_r2_ci,
    pair_sign_accuracy,
    pair_table,
    scheme_folds,
)

# Valid SMILES so the Murcko scaffold is defined; one per DOI.
SMILES = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccc(cc1)C(=O)O",
]


def _frame(n_doi: int = 6, per_doi: int = 4) -> pd.DataFrame:
    rows = []
    for d in range(n_doi):
        for i in range(per_doi):
            rows.append(
                {
                    "ligand_canonical": SMILES[d % len(SMILES)],
                    "ligand_main_fragment": SMILES[d % len(SMILES)],
                    "cell_line_key": "A549",
                    "doi": f"10.1000/{2018 + d % 5}.p{d}",
                    "time_h": 48.0,
                    "metal": ["Ru", "Ir", "Os", "Rh"][i % 4],
                    "value_log10": float(d + i),
                }
            )
    return pd.DataFrame(rows)


def test_doi_folds_never_split_a_paper():
    frame = _frame()
    fold_id, groups, info = scheme_folds(frame, "doi", n_splits=3)
    assert (fold_id >= 0).all()
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g].tolist())) == 1
    assert info["n_groups"] == frame["doi"].nunique()


def test_ligand_folds_never_split_a_complex():
    frame = _frame()
    fold_id, groups, _ = scheme_folds(frame, "ligand", n_splits=3)
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g].tolist())) == 1


def test_scaffold_folds_group_by_murcko_scaffold():
    frame = _frame()
    fold_id, groups, info = scheme_folds(frame, "scaffold", n_splits=3)
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g].tolist())) == 1
    # Six distinct molecules must not collapse to a single scaffold group.
    assert info["n_groups"] >= 3


def test_year_folds_leave_unparseable_dois_unscored(monkeypatch):
    monkeypatch.setattr(gen, "MIN_YEAR_ROWS", 1)
    frame = _frame()
    frame.loc[frame.index[:4], "doi"] = "no-year-here"
    fold_id, _, info = scheme_folds(frame, "year")
    assert (fold_id[:4] == -1).all()
    assert info["n_rows_unscored"] == 4
    assert info["n_rows_evaluated"] == len(frame) - 4


def test_year_folds_hold_out_a_whole_year(monkeypatch):
    monkeypatch.setattr(gen, "MIN_YEAR_ROWS", 1)
    frame = _frame(n_doi=4, per_doi=4)
    frame["doi"] = np.repeat(
        ["10.1000/2018.x", "10.1000/2019.x", "10.1000/2020.x", "10.1000/2020.y"], 4
    )
    fold_id, groups, info = scheme_folds(frame, "year")
    # 2018 / 2019 / 2020 -> three held-out years, one fold each.
    assert info["n_groups"] == 3
    assert info["n_folds"] == 3
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g].tolist())) == 1


def test_pair_table_skips_single_metal_groups():
    frame = _frame(n_doi=2, per_doi=4)
    frame.loc[frame.index[:2], "metal"] = "Ru"
    pairs = pair_table(frame)
    assert not pairs.empty
    for _, row in pairs.iterrows():
        assert row["metal_a"] != row["metal_b"]
        assert frame.at[row["idx_a"], "metal"] != frame.at[row["idx_b"], "metal"]


def test_pair_sign_accuracy_is_one_for_a_perfect_model():
    frame = _frame(n_doi=2, per_doi=4)
    pairs = pair_table(frame)
    y = frame["value_log10"].to_numpy(dtype=float)
    fold_id = np.zeros(len(frame), dtype=int)
    out = pair_sign_accuracy(y, y.copy(), pairs, fold_id)
    for th, block in out["by_threshold"].items():
        assert block["sign_accuracy"] == 1.0, th


def test_pair_sign_accuracy_is_zero_for_a_reversed_model():
    frame = _frame(n_doi=2, per_doi=4)
    pairs = pair_table(frame)
    y = frame["value_log10"].to_numpy(dtype=float)
    fold_id = np.zeros(len(frame), dtype=int)
    out = pair_sign_accuracy(y, -y, pairs, fold_id)
    assert out["by_threshold"]["0"]["sign_accuracy"] == 0.0


def test_pair_sign_accuracy_ignores_unscored_rows():
    frame = _frame(n_doi=2, per_doi=4)
    pairs = pair_table(frame)
    y = frame["value_log10"].to_numpy(dtype=float)
    fold_id = np.zeros(len(frame), dtype=int)
    fold_id[0] = -1
    out = pair_sign_accuracy(y, y.copy(), pairs, fold_id)
    touched = (pairs["idx_a"] == 0) | (pairs["idx_b"] == 0)
    assert out["n_pairs"] == int((~touched).sum())


def test_bootstrap_r2_ci_reports_too_few_groups():
    out = bootstrap_r2_ci(np.arange(4.0), np.arange(4.0), np.array(["a", "a", "b", "b"]))
    assert out["status"] == "TOO_FEW_GROUPS"


def test_bootstrap_r2_ci_brackets_a_perfect_fit():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    groups = np.repeat(np.arange(10), 20)
    out = bootstrap_r2_ci(y, y.copy(), groups, n_boot=200, seed=0)
    lo, hi = out["ci95"]
    assert lo >= 0.99 and hi >= lo


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
