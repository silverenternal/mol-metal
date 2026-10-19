#!/usr/bin/env python3
"""Tests for the STR-5 readout (msrs_delta_direct).

The readout decides a pre-registered gate, so the three things that can move
it silently are pinned here: pair orientation (a flipped pair turns a correct
prediction into a wrong one), the gate filter, and the cluster bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD
from molmetal.scripts.msrs_delta_direct import (
    canonical_pairs,
    cluster_bootstrap,
    delta_stats,
)


def test_canonical_pairs_keeps_one_orientation_by_metal_rank():
    metal = np.array(["Pt", "Ru", "Pt", "Ru", "Ir"])
    # cohort {0,1}: Pt(rank 1) vs Ru(rank 2); cohort {2,3} same; row 4 is alone
    pair_a = np.array([0, 1, 2, 3])
    pair_b = np.array([1, 0, 3, 2])
    keep = canonical_pairs(pair_a, pair_b, metal)
    # rank(Pt)=1 < rank(Ru)=2 -> the (0,1) and (2,3) orientations survive
    assert keep.tolist() == [True, False, True, False]


def test_delta_stats_gate_and_arithmetic():
    obs = np.array([1.0, -0.5, 0.05, 0.4])
    pred = np.array([0.8, -0.1, 0.9, -0.4])
    gated = delta_stats(obs, pred, GATE_THRESHOLD)
    # only |obs| >= 0.2091 survives: rows 0, 1, 3
    assert gated["n_pairs"] == 3
    # signs: (+,+) ok, (-,-) ok, (+,-) wrong
    assert gated["sign_accuracy"] == pytest.approx(round(2 / 3, 4))
    assert gated["mae_delta"] == pytest.approx(round((0.2 + 0.4 + 0.8) / 3, 4))
    assert gated["bias_delta"] == pytest.approx(round((-0.2 + 0.4 - 0.8) / 3, 4))


def test_delta_stats_ungated_keeps_everything():
    obs = np.array([1.0, -0.5, 0.05, 0.4])
    pred = np.array([0.8, -0.1, 0.9, -0.4])
    ungated = delta_stats(obs, pred, None)
    assert ungated["n_pairs"] == 4


def test_cluster_bootstrap_is_deterministic_and_brackets_the_point():
    rng = np.random.default_rng(0)
    n = 120
    obs = rng.normal(0.0, 0.5, n)
    pred = 0.5 * obs + rng.normal(0.0, 0.3, n)
    clusters = np.array([f"doi{i % 12}" for i in range(n)])
    first = cluster_bootstrap(obs, pred, clusters, GATE_THRESHOLD, n_boot=200, seed=7)
    second = cluster_bootstrap(obs, pred, clusters, GATE_THRESHOLD, n_boot=200, seed=7)
    assert first == second
    point = delta_stats(obs, pred, GATE_THRESHOLD)["sign_accuracy"]
    low, high = first["sign_accuracy"]["ci95"]
    assert low <= point <= high
    assert first["n_dois"] == 12
