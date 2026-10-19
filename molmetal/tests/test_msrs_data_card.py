"""MET-10 tests — the leakage numbers must measure what they claim."""

from __future__ import annotations

import numpy as np
import pandas as pd

from molmetal.scripts.msrs_data_card import (
    cross_fold_rate,
    duplicate_audit,
    duplicates_cross_folds,
)


def test_cross_fold_rate_is_zero_when_each_group_owns_a_fold():
    values = np.array(["a", "a", "b", "b", "c", "c"])
    fold = np.array([0, 0, 1, 1, 2, 2])
    out = cross_fold_rate(values, fold)
    assert out["rate"] == 0.0
    assert out["n_test_rows"] == 6


def test_cross_fold_rate_is_one_when_every_test_value_was_trained_on():
    values = np.array(["a", "b", "a", "b", "a", "b"])
    fold = np.array([0, 0, 1, 1, 2, 2])
    out = cross_fold_rate(values, fold)
    assert out["rate"] == 1.0


def test_cross_fold_rate_ignores_unscored_rows():
    values = np.array(["a", "b", "c"])
    fold = np.array([0, 1, -1])
    out = cross_fold_rate(values, fold)
    # Row 2 is never evaluated and is not a test row.
    assert out["n_test_rows"] == 2
    assert out["rate"] == 0.0


def _frame():
    return pd.DataFrame(
        {
            "ligand_canonical": ["L1", "L1", "L2", "L3"],
            "metal": ["Ru", "Ru", "Ir", "Os"],
            "cell_line_key": ["A549"] * 4,
            "doi": ["10.1/x"] * 4,
            "time_h": [48.0] * 4,
            "value_log10": [1.0, 1.0, 2.0, 3.0],
        }
    )


def test_duplicate_audit_counts_repeats_conflicts_and_exact_copies():
    out = duplicate_audit(_frame())
    assert out["n_rows"] == 4
    assert out["n_unique_keys"] == 3
    assert out["n_keys_repeated"] == 1
    assert out["n_keys_conflicting_values"] == 0
    assert out["n_exact_duplicate_rows"] == 1


def test_duplicate_audit_flags_conflicting_values():
    frame = _frame()
    frame.loc[1, "value_log10"] = 9.0
    out = duplicate_audit(frame)
    assert out["n_keys_conflicting_values"] == 1
    assert out["n_rows_in_conflicting_keys"] == 2
    assert out["n_exact_duplicate_rows"] == 0


def test_duplicates_cross_folds_when_a_key_straddles_folds():
    frame = _frame()
    folds = {"ligand": np.array([0, 1, 1, 0])}
    out = duplicates_cross_folds(frame, folds)
    assert out["ligand"]["crossing_keys"] == 1
    assert out["ligand"]["nan_keys"] == 0


def test_duplicates_do_not_cross_folds_when_grouped():
    frame = _frame()
    folds = {"ligand": np.array([0, 0, 1, 0])}
    out = duplicates_cross_folds(frame, folds)
    assert out["ligand"]["crossing_keys"] == 0
