"""Tests for STR-3 censored-training plumbing (no fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.msrs_censored_training import (
    group_key,
    hinge_objective,
    inherit_folds,
    select_rows,
)


class _Dataset:
    def __init__(self, y):
        self._y = np.asarray(y, dtype=float)

    def get_label(self):
        return self._y


def test_hinge_gradient_is_zero_above_the_bound():
    censored = np.array([True, True, False])
    y = np.array([2.0, 2.0, 1.0])
    pred = np.array([2.5, 1.0, 1.5])  # above bound, below bound, exact
    grad, hess = hinge_objective(censored)(pred, _Dataset(y))
    assert grad[0] == 0.0          # already above the bound: no push
    assert grad[1] == -1.0         # below the bound: push up
    assert grad[2] == pytest.approx(1.0)  # 2 * (1.5 - 1.0)
    assert hess[0] == 1.0 and hess[2] == 2.0


def test_hinge_ignores_exact_rows_above_and_below():
    censored = np.array([False])
    y = np.array([1.0])
    pred = np.array([0.5])
    grad, _ = hinge_objective(censored)(pred, _Dataset(y))
    assert grad[0] == pytest.approx(-1.0)


def _frame():
    return pd.DataFrame(
        {
            "ligand_canonical": [f"L{i}" for i in range(8)] * 2,
            "ligand_main_fragment": [f"L{i}" for i in range(8)] * 2,
            "doi": [f"10.1/{i // 2}" for i in range(8)] * 2,
        }
    )


def test_group_key_ligand_and_doi():
    frame = _frame()
    lig = group_key(frame, "ligand")
    doi = group_key(frame, "doi")
    assert lig[0] == "L0" and doi[0] == "10.1/0"
    assert len(lig) == len(frame)


def test_group_key_year_needs_counts():
    with pytest.raises(ValueError, match="year scheme"):
        group_key(_frame(), "year")


def test_inherit_folds_gives_censored_rows_their_group_fold():
    exact = _frame().iloc[:8].reset_index(drop=True)
    combined = _frame()  # second copy of every ligand is "censored"
    folds, stats = inherit_folds(exact, combined, "ligand")
    assert (folds[:8] >= 0).all()
    # A censored row sharing a ligand with an exact row inherits that fold.
    for i in range(8, 16):
        same = np.flatnonzero(exact["ligand_canonical"].to_numpy() == combined["ligand_canonical"][i])
        assert folds[i] == folds[same[0]]
    assert stats["censored_rows"] == 8


def test_inherit_folds_unseen_group_trains_everywhere():
    exact = _frame().iloc[:4].reset_index(drop=True)
    combined = pd.concat(
        [exact, pd.DataFrame(
            {
                "ligand_canonical": ["NEW1", "NEW2"],
                "ligand_main_fragment": ["NEW1", "NEW2"],
                "doi": ["10.9/x", "10.9/y"],
            }
        )],
        ignore_index=True,
    )
    folds, stats = inherit_folds(exact, combined, "ligand")
    assert folds[-2] == -1 and folds[-1] == -1
    assert stats["censored_with_unseen_group"] == 2


def test_select_rows_puts_exact_first_and_marks_censored(tmp_path):
    frame = pd.DataFrame(
        {
            "endpoint": ["potency"] * 4,
            "value_log10": [1.0, 2.0, 2.0, 1.5],
            "censoring": ["none", "none", "right", "left"],
            "ligand_main_fragment": ["a", "b", "c", "d"],
            "ligand_canonical": ["a", "b", "c", "d"],
            "oxidation_state": [2, 2, 2, 2],
        }
    )
    path = tmp_path / "m.parquet"
    frame.to_parquet(path, index=False)
    rows, censored = select_rows(path)
    assert len(rows) == 4
    assert list(censored) == [False, False, True, True]
    assert list(rows["ligand_canonical"]) == ["a", "b", "c", "d"]


def test_select_rows_limit_keeps_groups_together(tmp_path):
    frame = pd.DataFrame(
        {
            "endpoint": ["potency"] * 6,
            "value_log10": [1.0] * 6,
            "censoring": ["none"] * 6,
            "ligand_main_fragment": ["a", "a", "b", "b", "c", "c"],
            "ligand_canonical": ["a", "a", "b", "b", "c", "c"],
            "oxidation_state": [2] * 6,
        }
    )
    path = tmp_path / "m.parquet"
    frame.to_parquet(path, index=False)
    rows, _ = select_rows(path, limit=1)
    assert set(rows["ligand_canonical"]) == {"a"}
