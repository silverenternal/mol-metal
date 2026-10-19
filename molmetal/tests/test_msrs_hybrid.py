"""Tests for FIX-M2's hybrid blend (``msrs_hybrid``).

The script is a runner; what must be right is the estimator contract:

* a one-parameter blend recovers a known weight,
* the fold-wise fit never sees the fold it predicts,
* the rank blend breaks ties deterministically instead of inheriting row order,
* alignment refuses two OOF files that are not the same cohort.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_hybrid import (  # noqa: E402
    align,
    fit_linear,
    fit_one_weight,
    foldwise,
    rank_blend,
    rank_tie_stats,
)


def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ligand_canonical": [f"L{i % 8}" for i in range(n)],
            "ligand_main_fragment": [f"F{i % 4}" for i in range(n)],
            "metal": [["Pt", "Ru", "Ir", "Os"][i % 4] for i in range(n)],
            "cell_line_key": ["A549"] * n,
            "doi": [f"10.1/{i % 5}" for i in range(n)],
            "y": rng.normal(size=n),
            "tree": rng.normal(size=n),
            "gnn": rng.normal(size=n),
        }
    )


def test_one_weight_recovers_known_mixture():
    rng = np.random.default_rng(1)
    p_tree = rng.normal(size=500)
    p_gnn = rng.normal(size=500)
    y = 0.7 * p_tree + 0.3 * p_gnn
    assert fit_one_weight(y, p_tree, p_gnn) == pytest.approx(0.7, abs=1e-6)


def test_one_weight_is_clipped_to_the_simplex():
    p_tree = np.array([1.0, 1.0, 1.0])
    p_gnn = np.array([0.0, 0.0, 0.0])
    # w would be 2.0 and -2.0 respectively.
    assert fit_one_weight(np.array([2.0, 2.0, 2.0]), p_tree, p_gnn) == 1.0
    assert fit_one_weight(np.array([-2.0, -2.0, -2.0]), p_tree, p_gnn) == 0.0


def test_linear_recovers_coefficients():
    rng = np.random.default_rng(2)
    p_tree = rng.normal(size=800)
    p_gnn = rng.normal(size=800)
    y = 0.6 * p_tree + 0.35 * p_gnn - 0.12
    a, b, c = fit_linear(y, p_tree, p_gnn)
    assert (a, b, c) == pytest.approx((0.6, 0.35, -0.12), abs=1e-8)


def test_foldwise_never_uses_the_fold_it_predicts():
    frame = _frame()
    frame["fold"] = np.tile(np.arange(4), 10)
    base, _ = foldwise(frame, "one_weight")

    perturbed = frame.copy()
    perturbed.loc[perturbed["fold"] == 2, "y"] += 100.0
    shifted, _ = foldwise(perturbed, "one_weight")

    # Fold 2 is predicted from the other three folds, whose labels did not move.
    assert np.allclose(base[frame["fold"] == 2], shifted[perturbed["fold"] == 2])
    # Fold 0 is predicted from a training set that contains the perturbed fold 2.
    assert not np.allclose(base[frame["fold"] == 0], shifted[perturbed["fold"] == 0])


def test_rank_blend_breaks_ties_toward_the_tree():
    # One cohort, two metals.  The ranks are 0.5/1.0 for both models, so the
    # mean rank ties exactly; the tree prefers "a" (lower prediction).
    frame = pd.DataFrame(
        {
            "ligand_canonical": ["L", "L"],
            "cell_line_key": ["A", "A"],
            "doi": ["d", "d"],
            "time_h": [24.0, 24.0],
            "metal": ["a", "b"],
            "y": [1.0, 0.0],
        }
    )
    score = rank_blend(frame, np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    # Tree ranks a=0.5, b=1.0; gnn ranks a=1.0, b=0.5 -> both mean 0.75.
    assert score[0] == pytest.approx(score[1], abs=1e-8)
    assert score[0] < score[1]  # 1e-9 * tree rank breaks the tie toward "a"


def test_rank_tie_stats_counts_ties_on_evaluable_cohorts():
    frame = pd.DataFrame(
        {
            "ligand_canonical": ["L1", "L1", "L2"],
            "cell_line_key": ["A", "A", "A"],
            "doi": ["d", "d", "d"],
            "time_h": [24.0, 24.0, 24.0],
            "metal": ["a", "b", "a"],
            "y": [1.0, 0.0, 0.0],
        }
    )
    stats = rank_tie_stats(frame, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert stats["n_cohorts"] == 1  # the singleton cohort is not evaluable
    assert stats["n_tied_at_min"] == 1
    assert stats["first_row_is_best_when_tied"] == 0.0


def test_align_rejects_different_labels():
    frame = _frame()
    tree = {
        "y": frame["y"].to_numpy(),
        "ligand_canonical": frame["ligand_canonical"].to_numpy(),
        "ligand_main_fragment": frame["ligand_main_fragment"].to_numpy(),
        "metal": frame["metal"].to_numpy(),
        "cell_line_key": frame["cell_line_key"].to_numpy(),
        "doi": frame["doi"].to_numpy(),
        "canonical_grouped__ligand+both": frame["tree"].to_numpy(),
    }
    gnn = dict(tree)
    gnn["y"] = tree["y"] + 1.0
    gnn["canonical_grouped__metal_state"] = frame["gnn"].to_numpy()
    with pytest.raises(ValueError, match="labels differ"):
        align(tree, gnn, "canonical_grouped")


def test_align_rejects_different_row_order():
    frame = _frame()
    tree = {
        "y": frame["y"].to_numpy(),
        "ligand_canonical": frame["ligand_canonical"].to_numpy(),
        "ligand_main_fragment": frame["ligand_main_fragment"].to_numpy(),
        "metal": frame["metal"].to_numpy(),
        "cell_line_key": frame["cell_line_key"].to_numpy(),
        "doi": frame["doi"].to_numpy(),
        "canonical_grouped__ligand+both": frame["tree"].to_numpy(),
    }
    gnn = dict(tree)
    gnn["metal"] = np.roll(tree["metal"], 1)
    gnn["canonical_grouped__metal_state"] = frame["gnn"].to_numpy()
    with pytest.raises(ValueError, match="column 'metal' differs"):
        align(tree, gnn, "canonical_grouped")
