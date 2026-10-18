"""Tests for the MSRS-4 leave-one-metal-out surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from molmetal.scripts import msrs_lomo_surface as L
from molmetal.scripts import msrs_surface_model as M


def _descriptors(ie1=None) -> pd.DataFrame:
    ie1 = ie1 or [710.2, 880.0, 719.7, 840.0, 760.0]
    return pd.DataFrame(
        {
            "metal_symbol": list(M.METALS),
            "ie1_kj": ie1,
            "ie2_kj": [1620.0, 1600.0, 1740.0, 1600.0, 1260.0],
            "pauling_en": [2.20, 2.20, 2.28, 2.20, 1.90],
        }
    )


def _frame(n_per_metal: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for metal in M.METALS:
        for i in range(n_per_metal):
            rows.append(
                {
                    "ligand_main_fragment": ["CCO", "c1ccccc1", "CCN", "c1ccncc1", "CC(=O)O"][i % 5],
                    "cell_line_key": "A549",
                    "metal": metal,
                    "doi": f"d{i % 5}",
                    "time_h": 24.0,
                    "value_log10": 0.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Design assembly
# ---------------------------------------------------------------------------
def test_assemble_shapes_and_groups():
    df = _frame(10)
    blocks = L.build_blocks(df, _descriptors(), ["A549"])
    rows = np.arange(len(df))
    for rep, n_metal in (("descriptors", 3), ("one_hot", 5), ("none", 0)):
        X, groups = L.assemble(blocks, rep, rows)
        assert X.shape[0] == len(df)
        assert X.shape[1] == sum(len(g) for g in groups)
        if n_metal:
            assert len(groups) == 3 and len(groups[1]) == n_metal
        else:
            assert len(groups) == 2
        flat = [i for g in groups for i in g]
        assert sorted(flat) == list(range(X.shape[1]))


def test_one_hot_holdout_column_is_unseen_in_training():
    df = _frame(10)
    blocks = L.build_blocks(df, _descriptors(), ["A549"])
    train = np.where(df["metal"].to_numpy() != "Os")[0]
    test = np.where(df["metal"].to_numpy() == "Os")[0]
    X_tr, _ = L.assemble(blocks, "one_hot", train)
    X_te, _ = L.assemble(blocks, "one_hot", test)
    os_col = M.METALS.index("Os")
    # The held-out metal's column is constant zero in training and 1 at test:
    # a pattern the model has never seen, which is the point of the baseline.
    assert np.all(X_tr[:, 2048 + os_col].toarray() == 0.0)
    assert np.all(X_te[:, 2048 + os_col].toarray() == 1.0)


# ---------------------------------------------------------------------------
# Aggregation and rank agreement
# ---------------------------------------------------------------------------
def test_aggregate_split_is_weighted():
    per = {
        "Ru": {"status": "OK", "n_test": 100, "models": {r: {"mae": 1.0, "r2": -0.1, "bias": 0.2} for r in L.REPRESENTATIONS}},
        "Ir": {"status": "OK", "n_test": 300, "models": {r: {"mae": 2.0, "r2": -0.3, "bias": -0.4} for r in L.REPRESENTATIONS}},
    }
    agg = L.aggregate_split(per)
    assert agg["descriptors"]["weighted_mae"] == pytest.approx(1.75)
    assert agg["descriptors"]["weighted_r2"] == pytest.approx(-0.25)
    assert agg["descriptors"]["mean_abs_bias"] == pytest.approx(0.3)
    assert agg["descriptors"]["n_metals_negative_r2"] == 2


def test_rank_agreement_perfect_and_reversed():
    def entry(pred, obs):
        return {
            "status": "OK",
            "observed_mean": obs,
            "pred_mean_by_representation": {r: pred for r in L.REPRESENTATIONS},
        }

    perfect = {"Ru": entry(1.0, 1.0), "Ir": entry(2.0, 2.0), "Rh": entry(3.0, 3.0)}
    assert L.rank_agreement(perfect)["descriptors"]["spearman_rho"] == pytest.approx(1.0)

    reversed_ = {"Ru": entry(3.0, 1.0), "Ir": entry(2.0, 2.0), "Rh": entry(1.0, 3.0)}
    assert L.rank_agreement(reversed_)["descriptors"]["spearman_rho"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Can the pipeline detect a learnable metal effect?
# ---------------------------------------------------------------------------
def test_evaluate_split_finds_a_descriptor_signal():
    # ie1 is interior for the held-out metal, so a tree can interpolate.
    desc = _descriptors(ie1=[700.0, 720.0, 740.0, 760.0, 780.0])
    df = _frame(n_per_metal=60, seed=1)
    rng = np.random.default_rng(1)
    level = {m: (v - 740.0) / 100.0 * 2.0 for m, v in zip(M.METALS, desc["ie1_kj"])}
    y = np.array([level[m] for m in df["metal"]]) + rng.normal(0, 0.05, len(df))
    df["value_log10"] = y
    blocks = L.build_blocks(df, desc, ["A549"])
    out = L.evaluate_split(df, blocks, y, "Ir", False, verbose=False)
    assert out["status"] == "OK"
    assert out["models"]["descriptors"]["mae"] < out["models"]["none"]["mae"]


def test_evaluate_split_cannot_find_an_idiosyncratic_offset():
    desc = _descriptors(ie1=[700.0, 720.0, 740.0, 760.0, 780.0])
    df = _frame(n_per_metal=60, seed=2)
    rng = np.random.default_rng(2)
    # The held-out metal has a level that is not a function of the descriptors.
    level = {"Ru": 0.0, "Ir": 0.3, "Rh": 1.2, "Os": -0.2, "Re": 0.1}
    y = np.array([level[m] for m in df["metal"]]) + rng.normal(0, 0.05, len(df))
    df["value_log10"] = y
    blocks = L.build_blocks(df, desc, ["A549"])
    out = L.evaluate_split(df, blocks, y, "Rh", False, verbose=False)
    assert out["status"] == "OK"
    # The offset is 1.2 while the noise is 0.05; no representation can recover
    # it, so both errors stay an order of magnitude above the noise floor.
    assert out["models"]["descriptors"]["mae"] > 0.5
    assert out["models"]["none"]["mae"] > 0.5
