"""Tests for the MSRS-3 conformal / actionability layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from molmetal.scripts import msrs_actionability as A
from molmetal.scripts import msrs_surface_model as M


# ---------------------------------------------------------------------------
# Conformal quantile
# ---------------------------------------------------------------------------
def test_conformal_quantile_uses_finite_sample_level():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    # n=4, alpha=0.5 -> level = ceil(5*0.5)/4 = 3/4 -> "higher" gives 4.0
    assert A.conformal_quantile(scores, 0.5) == 4.0
    # alpha=0.25 -> level = ceil(5*0.75)/4 = 4/4 = 1.0 -> 4.0
    assert A.conformal_quantile(scores, 0.25) == 4.0


def test_conformal_quantile_empty_is_infinite():
    assert np.isinf(A.conformal_quantile(np.array([]), 0.1))


# ---------------------------------------------------------------------------
# Actionability
# ---------------------------------------------------------------------------
def _actionability_frame():
    rows = []
    for lig in ("L1", "L2", "L3", "L4"):
        for metal in ("Ru", "Ir"):
            rows.append(
                {
                    "ligand_main_fragment": lig,
                    "cell_line_key": "A",
                    "doi": "d1",
                    "time_h": 24.0,
                    "metal": metal,
                }
            )
    return pd.DataFrame(rows)


def test_actionability_rewards_narrow_and_correct():
    df = _actionability_frame()
    y = np.array([2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0])
    # L1/L2: Ir is more potent (gain +1), predicted correctly, narrow intervals.
    # L3/L4: Ru is more potent (gain -1), predicted with the wrong sign, wide.
    y_hat = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    half = np.array([0.1, 0.1, 0.1, 0.1, 1.2, 1.2, 1.2, 1.2])
    out = A.actionability_report(df, y, y_hat, half)
    prim = out["primary"]
    assert prim["n_actionable"] == 2
    assert prim["accuracy_actionable"] == 1.0
    assert prim["accuracy_rest"] == 0.0


def test_actionability_handles_no_pairs():
    df = _actionability_frame().iloc[[0, 3, 4, 7]].copy()  # one metal per ligand
    out = A.actionability_report(df, np.zeros(4), np.zeros(4), np.ones(4))
    assert out["status"] == "NO_MATCHED_PAIRS"


def test_stratified_accuracy_removes_composition_bias():
    n_act, n_rest = 12, 10
    pairs = pd.DataFrame(
        {
            "metal_a": ["Ru"] * (n_act + n_rest),
            "metal_b": ["Ir"] * (n_act + n_rest),
        }
    )
    correct = np.array([1.0] * 10 + [0.0] * 2 + [1.0] * 2 + [0.0] * 8)
    act = np.array([True] * n_act + [False] * n_rest)
    informative = np.ones(n_act + n_rest, dtype=bool)
    out = A._stratified_accuracy(pairs, correct, act, informative)
    assert out["Ru-Ir"]["n_actionable"] == n_act
    assert out["Ru-Ir"]["accuracy_actionable"] == pytest.approx(10 / 12, abs=1e-4)
    assert out["Ru-Ir"]["accuracy_rest"] == pytest.approx(0.2, abs=1e-4)
    assert out["Ru-Ir"]["diff"] == pytest.approx(10 / 12 - 0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# End-to-end nested conformal on synthetic data
# ---------------------------------------------------------------------------
def test_nested_conformal_coverage_is_near_nominal():
    rng = np.random.default_rng(0)
    n_lig, per_lig, n_bits = 80, 4, 24
    n = n_lig * per_lig
    lig_id = np.repeat(np.arange(n_lig), per_lig)
    metals = np.array(["Ru", "Ir", "Rh", "Os"] * (n // 4))[:n]
    bits = rng.integers(0, 2, size=(n, n_bits)).astype(np.float32)
    X = sparse.hstack(
        [
            sparse.csr_matrix(bits),
            sparse.csr_matrix(np.eye(5, dtype=np.float32)[[M.METALS.index(m) for m in metals]]),
            sparse.csr_matrix(np.ones((n, 1), dtype=np.float32)),
        ],
        format="csr",
    )
    groups = np.array([f"L{i}" for i in lig_id])
    y = bits[:, 0] * 0.8 + np.array([{"Ru": 0.0, "Ir": 0.5, "Rh": -0.3, "Os": 0.2}[m] for m in metals])
    y = y + rng.normal(0, 0.4, n)
    constraints = [list(range(n_bits)), list(range(n_bits, n_bits + 5)), [n_bits + 5]]

    conf = A.nested_conformal(X, y, groups, metals, constraints)
    half = conf["half_width_global"]
    coverage = float((np.abs(y - conf["y_hat"]) <= half).mean())
    assert 0.75 <= coverage <= 0.99
    assert np.all(half > 0)
