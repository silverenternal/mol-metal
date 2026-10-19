"""MET-1 — tests for the seed-stability aggregator."""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.msrs_seed_stability import paired_vs_tree, summarise


def _per_seed(values, split="canonical_grouped"):
    return {
        seed: {split: {"r2": {"lgb_tree": 0.50, "blend_equal": v}}}
        for seed, v in enumerate(values)
    }


def test_summarise_matches_numpy():
    out = summarise([0.50, 0.52, 0.54])
    assert out["mean"] == pytest.approx(0.52, abs=1e-4)
    assert out["std"] == pytest.approx(float(np.std([0.50, 0.52, 0.54], ddof=1)), abs=1e-4)
    assert out["n"] == 3
    assert out["per_seed"] == [0.5, 0.52, 0.54]


def test_summarise_single_seed_has_zero_std():
    out = summarise([0.53])
    assert out["std"] == 0.0
    assert out["n"] == 1


def test_paired_vs_tree_counts_direction():
    per_seed = _per_seed([0.55, 0.56, 0.49, 0.57, 0.54])
    out = paired_vs_tree(per_seed, "canonical_grouped", "blend_equal")
    assert out["n"] == 5
    assert out["n_seeds_better"] == 4
    assert out["mean_delta"] == pytest.approx(0.042, abs=1e-4)


def test_paired_vs_tree_consistent_gain_is_significant():
    per_seed = _per_seed([0.55, 0.56, 0.54, 0.57, 0.545])
    out = paired_vs_tree(per_seed, "canonical_grouped", "blend_equal")
    assert out["t"] > 5.0
    assert out["p_one_sided"] < 0.01


def test_paired_vs_tree_negative_gain_has_large_p():
    per_seed = _per_seed([0.45, 0.46, 0.44, 0.47, 0.445])
    out = paired_vs_tree(per_seed, "canonical_grouped", "blend_equal")
    assert out["n_seeds_better"] == 0
    assert out["p_one_sided"] > 0.5
