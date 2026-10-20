#!/usr/bin/env python3
"""Tests for the C6 fix: resolvability audit + seed ensemble.

Two things decide the verdict and are easy to get wrong silently: the standard
error used to call a fold "unresolvable", and the difference between averaging
AUCs and averaging scores.  Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.lomo_resolvability_audit import fold_counts, hanley_se
from molmetal.scripts.lomo_seed_ensemble import load_fold


def test_hanley_se_shrinks_as_the_fold_grows():
    small = hanley_se(0.6, 25, 25)
    large = hanley_se(0.6, 500, 500)
    assert large < small


def test_hanley_se_matches_the_known_magnitude_at_n_36():
    # the project's own note: "at n=36 a 95% CI on AUC is roughly +-0.16"
    se = hanley_se(0.6, 15, 21)
    assert 0.13 < 1.96 * se < 0.20


def test_hanley_se_is_zero_for_a_perfect_auc():
    assert hanley_se(1.0, 20, 20) == pytest.approx(0.0)


def test_fold_counts_dedups_per_metal(tmp_path):
    frame = pd.DataFrame(
        {
            "SMILES_Ligands": ["CCO", "CCO", "CCN", "CCC", "CCC"],
            "Metal": ["Ru", "Ru", "Ru", "Ir", "Ir"],
            "active": [1, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    counts = fold_counts(path)
    # Ru: CCO + CCN -> 2 unique, both majority-active 1 + 0
    assert counts["Ru"]["n_unique"] == 2
    assert counts["Ru"]["n_pos"] == 1
    # Ir: CCC appears twice with labels 1 and 0 -> majority ties go ACTIVE
    assert counts["Ir"]["n_unique"] == 1
    assert counts["Ir"]["n_pos"] == 1


def _write_fold(directory, metal, y, scores):
    """One file per row of ``scores`` (seeds x rows)."""
    for index in range(scores.shape[0]):
        np.savez(
            directory / f"{metal}_seed{index}.npz",
            y=np.asarray(y, dtype=np.int64),
            score=np.asarray(scores[index], dtype=np.float64),
            metal=metal,
            seed=index,
        )


def test_load_fold_rejects_a_row_order_change(tmp_path):
    y = np.array([0, 1, 0, 1])
    _write_fold(tmp_path, "Ru", y, np.tile(y.astype(float), (2, 1)))
    np.savez(
        tmp_path / "Ru_seed9.npz",
        y=np.array([1, 0, 0, 1]),
        score=np.ones(4),
        metal="Ru",
        seed=9,
    )
    with pytest.raises(ValueError, match="row order"):
        load_fold(tmp_path, "Ru")


def test_load_fold_stacks_seeds_in_name_order(tmp_path):
    y = np.array([0, 1, 0, 1])
    _write_fold(tmp_path, "Ru", y, np.tile(y.astype(float), (3, 1)))
    fold = load_fold(tmp_path, "Ru")
    assert fold["scores"].shape == (4, 3)
    assert fold["paths"] == ["Ru_seed0.npz", "Ru_seed1.npz", "Ru_seed2.npz"]


def test_averaging_scores_beats_averaging_aucs_when_one_seed_inverts():
    """Four good seeds and one inverted seed: the ensemble still ranks."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    good = [y + rng.normal(0, 0.5, len(y)) for _ in range(4)]
    bad = -y.astype(float) + rng.normal(0, 0.5, len(y))
    scores = np.column_stack(good + [bad])
    mean_of_aucs = float(np.mean([roc_auc_score(y, scores[:, j]) for j in range(5)]))
    auc_of_mean = float(roc_auc_score(y, scores.mean(axis=1)))
    assert mean_of_aucs < auc_of_mean
    assert auc_of_mean > 0.9
