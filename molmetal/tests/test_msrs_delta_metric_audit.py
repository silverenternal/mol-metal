#!/usr/bin/env python3
"""Tests for the delta-sign metric audit (C3).

The whole point of the audit is that raw sign accuracy is unreadable on an
imbalanced pair set, so the two things that must be right are the balanced
definition and its behaviour on a constant predictor.
"""

from __future__ import annotations

import numpy as np

from molmetal.scripts.msrs_delta_metric_audit import balanced_sign, bootstrap_balanced


def test_balanced_is_half_for_a_constant_predictor():
    obs = np.array([1.0, 1.0, 1.0, -1.0])
    pred = np.ones(4)
    out = balanced_sign(obs, pred)
    assert out["tpr"] == 1.0
    assert out["tnr"] == 0.0
    assert out["balanced"] == 0.5


def test_balanced_is_one_for_a_perfect_predictor():
    obs = np.array([1.0, -1.0, 1.0, -1.0])
    out = balanced_sign(obs, obs.copy())
    assert out["balanced"] == 1.0


def test_raw_accuracy_of_a_constant_predictor_equals_the_base_rate():
    obs = np.array([1.0] * 6 + [-1.0] * 4)
    pred = np.ones(10)
    raw = float(np.mean(np.sign(pred) == np.sign(obs)))
    assert raw == 0.6
    assert balanced_sign(obs, pred)["balanced"] == 0.5


def test_balanced_ignores_missing_sign_classes():
    obs = np.array([1.0, 1.0])
    out = balanced_sign(obs, np.ones(2))
    assert out["tnr"] != out["tnr"]  # NaN: no negatives to score


def test_bootstrap_interval_brackets_the_point():
    rng = np.random.default_rng(0)
    obs = np.where(rng.random(200) > 0.6, 1.0, -1.0)
    pred = np.where(rng.random(200) > 0.5, 1.0, -1.0)
    clusters = np.array([f"d{i % 20}" for i in range(200)])
    point = balanced_sign(obs, pred)["balanced"]
    low, high = bootstrap_balanced(obs, pred, clusters, n_boot=200, seed=3)
    assert low <= point <= high
