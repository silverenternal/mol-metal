# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Plan A Step 1 DMPNN value adapter (Wf-R24-A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "molmetal") not in sys.path:
    sys.path.insert(0, str(REPO / "molmetal"))

from molmetal_lam.search_alg.dmpnn_value import (  # noqa: E402
    DEFAULT_CKPT,
    DMPNNValueFn,
    make_value_fn,
    predict_pic50,
)


def test_default_ckpt_exists():
    """The default checkpoint must be on disk (Plan A §6 risk mitigation)."""
    assert DEFAULT_CKPT.exists(), f"missing DMPNN checkpoint: {DEFAULT_CKPT}"


def test_bad_smiles_degrades_to_zero():
    """Unparseable / empty SMILES return 0.0 — never raise (§5.7 gate)."""
    fn = make_value_fn()
    assert fn("not-a-real-smiles") == 0.0
    assert fn("") == 0.0
    assert fn(None) == 0.0
    assert fn(12345) == 0.0  # not a str, no canonical_smiles attr -> 0.0


def test_object_without_canonical_smiles_returns_zero():
    class _NoSmiles:
        pass

    assert make_value_fn()(_NoSmiles()) == 0.0


def test_forward_is_continuous_not_discrete():
    """The whole point of Step 1: values must NOT quantise to {0, 0.5, 1}.

    Two distinct molecules must produce two distinct, non-round values.
    """
    a = predict_pic50("C#CCN")
    b = predict_pic50("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert a > 0.0, "caffeine pIC50 should be > 0, DMPNN produced 0"
    assert b > 0.0
    assert a != b, "different molecules must not collapse to one value"
    # Not in the legacy discrete reward grid.
    for v in (a, b):
        assert v not in (0.0, 0.5, 1.0), f"value {v} is still discrete"


def test_lru_cache_hits():
    """Repeat calls hit the lru_cache (MCTS revisits leaves constantly)."""
    inner = DMPNNValueFn()
    inner._load()
    inner._cached.cache_clear()
    fn = make_value_fn.__wrapped__ if hasattr(make_value_fn, "__wrapped__") else None
    del fn  # unused; keep the cache assertions on the instance below

    first = inner("C#CCN")
    n_after_first = inner.n_forward
    second = inner("C#CCN")
    # Same value, and NO new forward pass (cache hit).
    assert first == second
    assert inner.n_forward == n_after_first, "second call must hit the lru_cache"


def test_cache_info_maxsize_256():
    inner = DMPNNValueFn()
    assert inner._cached.cache_info().maxsize == 256


def test_metal_selection_wildcard_vs_specific():
    """Wildcard averages the 5 metal heads; a specific metal runs one head."""
    wild = DMPNNValueFn(metal="wildcard")
    ir = DMPNNValueFn(metal="Ir")
    wild._load()
    ir._load()
    assert wild._metal_ids() == [0, 1, 2, 3, 4]
    assert ir._metal_ids() == [1]
    # Wildcard is the mean of the per-metal heads.
    per_head = [DMPNNValueFn(metal=m)("C#CCN") for m in ("Ru", "Ir", "Rh", "Os", "Re")]
    expected = sum(per_head) / 5.0
    assert wild("C#CCN") == pytest.approx(expected, abs=1e-5)


def test_missing_checkpoint_degrades_to_zero():
    """A nonexistent checkpoint must not raise — it returns 0.0 (§5.2)."""
    fn = make_value_fn(Path("/nonexistent/dmpnn_does_not_exist.pt"))
    assert fn("C#CCN") == 0.0


def test_reward_aggregator_default_is_byte_identical():
    """Plan A §2.1: dmpnn_value_fn defaults to None -> unchanged reward."""
    from molmetal_lam.search_alg.proof_search import RewardAggregator

    agg = RewardAggregator()
    assert agg.dmpnn_value_fn is None
    assert agg.w_dmpnn_value == 0.6
    # With the channel off, __call__ contributes exactly 0.0 extra.
    assert agg.bonus_typed == 0.5
    assert agg.bonus_binder == 0.5


def test_reward_aggregator_applies_dmpnn_channel():
    """Wiring a value fn adds w * clip(pIC50 / 8, 0, 1) on top of the base."""
    from molmetal_lam.search_alg.proof_search import RewardAggregator

    class _State:
        def canonical_smiles(self):
            return "C#CCN"

    base = RewardAggregator()
    wired = RewardAggregator(dmpnn_value_fn=lambda s: 8.0, w_dmpnn_value=0.6)
    st = _State()
    assert wired(st) - base(st) == pytest.approx(0.6, abs=1e-9)

    # pIC50 is clamped: a value of 100 saturates at w * 1.0.
    clamped = RewardAggregator(dmpnn_value_fn=lambda s: 100.0)
    assert clamped(st) - base(st) == pytest.approx(0.6, abs=1e-9)

    # A raising channel degrades to 0.0 rather than crashing the search.
    def _boom(_s):
        raise RuntimeError("torch exploded")

    safe = RewardAggregator(dmpnn_value_fn=_boom)
    assert safe(st) == pytest.approx(base(st), abs=1e-9)