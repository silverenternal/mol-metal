#!/usr/bin/env python3
"""Tests for the STR-6 SA yield audit.

The audit's verdict is a rate, so the failure modes are arithmetic and
silent dropping: a Wilson interval that returns a point mass at k=0, or an
unparseable SMILES that disappears instead of being counted.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.cfm_sa_yield_audit import SA_GATE, score_smiles, wilson


def test_wilson_at_zero_has_a_nonzero_upper_bound():
    low, high = wilson(0, 66)
    assert low == 0.0
    assert 0.0 < high < 0.10


def test_wilson_brackets_the_point_estimate():
    low, high = wilson(16, 2396)
    assert low < 16 / 2396 < high
    assert high - low < 0.02


def test_wilson_handles_empty_sample():
    assert wilson(0, 0) == [0.0, 0.0]


def test_score_smiles_counts_unparseable_rows():
    out = score_smiles(["CCO", "not-a-smiles", "c1ccccc1"])
    assert len(out["sas"]) == 2
    assert out["n_unscored"] == 1


def test_gate_smiles_only_holds_molecules_at_or_below_the_gate():
    # benzene is trivially easy; a fused polycyclic is not
    out = score_smiles(["c1ccccc1", "C1CC2CCC3CCCC4CCCC1C234"])
    assert len(out["sas"]) == 2
    for smiles in out["gate_smiles"]:
        assert smiles
    n_gate = len(out["gate_smiles"])
    assert n_gate == sum(1 for s in out["sas"] if s <= SA_GATE)


def test_gate_threshold_is_the_documented_one():
    assert SA_GATE == 4.5
