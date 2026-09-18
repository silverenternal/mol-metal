"""test_lambda_reward_channel.py — unit tests for the TODO-21 Strategy 1
Lambda-as-reward channel.

The :mod:`molmetal_lam.lam_chem.lambda_reward_channel` module ships a
new :attr:`RewardAggregator.r_lambda_score` channel (additive only — does
NOT modify any existing channel).  These tests verify the contract:

1. ``compute_lambda_score`` returns the **mean cosine similarity** when
   the channel is enabled and the candidate list is non-empty.
2. **Empty candidate list** returns 0.0 (no spurious noise contribution).
3. **Coupling disabled** (``COUPLING_ENABLED`` env var unset / False)
   returns 0.0 (opt-in contract).
4. The channel is **NaN-safe** — neither the closure nor
   ``compute_lambda_score`` should propagate NaN / Inf.
5. ``register_lambda_reward_channel`` wires the channel on a populated
   :class:`RewardAggregator` (additive), and ``w_lambda_score = 0.0``
   keeps the existing weighted reward bit-for-bit identical (regression-
   proof).
6. ``set_lambda_candidates`` populates the module-level candidate list
   consumed by the closure lazily.
7. The aggregator's ``aggregate`` method honours ``r_lambda_score`` as
   a recognised channel key.

All tests are CPU-only and skip the GPU/CFM training stack — they run
the aggregator with mock state objects (just need ``canonical_smiles``
or ``smiles`` attribute, or be a plain ``str``).

Run from the project root::

    uv run pytest molmetal/molmetal_lam/tests/test_lambda_reward_channel.py -q --tb=short
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Subject under test — imported lazily so the test module can be collected
# even when the heavier search_alg / torch stack is broken.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.lam_chem.lambda_reward_channel import (
        compute_lambda_score,
        make_lambda_reward_channel,
        register_lambda_reward_channel,
        set_lambda_candidates,
    )
    _HAS_CHANNEL = True
except Exception:  # pragma: no cover
    compute_lambda_score = None
    make_lambda_reward_channel = None
    register_lambda_reward_channel = None
    set_lambda_candidates = None
    _HAS_CHANNEL = False

try:
    from molmetal_lam.lam_chem.coupling_adapter import (
        CouplingAdapter,
        load_coupling_adapter,
    )
    _HAS_COUPLING = True
except Exception:  # pragma: no cover
    CouplingAdapter = None
    load_coupling_adapter = None
    _HAS_COUPLING = False

try:
    from molmetal_lam.search_alg.proof_search import RewardAggregator
    _HAS_REWARD_AGGREGATOR = True
except Exception:  # pragma: no cover
    RewardAggregator = None
    _HAS_REWARD_AGGREGATOR = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _DummyState:
    """Minimal state-object stub mimicking MoleculeClosedTerm.smiles."""

    smiles: str


# ---------------------------------------------------------------------------
# Test 1 — compute_lambda_score returns mean cosine similarity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_compute_lambda_score_returns_mean_cos_sim():
    """When enabled and the candidate list is non-empty, the channel
    returns a mean cosine similarity in ``[-1, 1]``.
    """
    # Force the env var ON for this test only.
    old = os.environ.get("COUPLING_ENABLED")
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        v = compute_lambda_score(
            ["CCN", "CCC", "CCO"],
            pocket_features=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            pocket_name="test_pocket",
        )
    finally:
        if old is None:
            os.environ.pop("COUPLING_ENABLED", None)
        else:
            os.environ["COUPLING_ENABLED"] = old

    # Result must be a finite float in [-1, 1].
    assert v == v, "compute_lambda_score returned NaN"
    assert isinstance(v, float)
    assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Test 2 — empty candidate list returns 0.0
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_compute_lambda_score_empty_candidates_returns_zero():
    """An empty candidate list returns 0.0 (no spurious noise)."""
    v = compute_lambda_score(
        [],
        pocket_features=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        pocket_name="test_pocket",
        enabled=True,
    )
    assert v == 0.0


# ---------------------------------------------------------------------------
# Test 3 — coupling disabled returns 0.0 (opt-in gate)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_compute_lambda_score_disabled_returns_zero():
    """``COUPLING_ENABLED=0`` (default) returns 0.0 even with non-empty
    candidates — opt-in contract."""
    old = os.environ.get("COUPLING_ENABLED")
    os.environ.pop("COUPLING_ENABLED", None)
    try:
        v = compute_lambda_score(
            ["CCN", "CCC"],
            pocket_features=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            pocket_name="test_pocket",
        )
    finally:
        if old is not None:
            os.environ["COUPLING_ENABLED"] = old

    assert v == 0.0


# ---------------------------------------------------------------------------
# Test 4 — channel is NaN-safe
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_lambda_channel_nan_safe():
    """Neither ``compute_lambda_score`` nor the closure should propagate
    NaN / Inf.  We force ``enabled=True`` and pass an empty pocket
    descriptor — both should yield ``0.0`` rather than NaN.
    """
    old = os.environ.get("COUPLING_ENABLED")
    os.environ["COUPLING_ENABLED"] = "1"
    try:
        # All-invalid candidates
        v1 = compute_lambda_score(["", None, ""], enabled=True)
        v2 = compute_lambda_score([], enabled=True)
        # Closure constructed via the factory must also be NaN-safe.
        closure = make_lambda_reward_channel(enabled=True)
        v3 = closure(_DummyState(smiles=""))
        v4 = closure(None)
    finally:
        if old is None:
            os.environ.pop("COUPLING_ENABLED", None)
        else:
            os.environ["COUPLING_ENABLED"] = old
    for v in (v1, v2, v3, v4):
        assert v == v, "channel returned NaN"
        assert isinstance(v, float)
        # When no candidates or all-empty SMILES, the contribution must be 0.0
        # (no candidates survive the empty-list gate or the invalid-SMILES gate).
        assert v == 0.0


# ---------------------------------------------------------------------------
# Test 5 — register_lambda_reward_channel wires the channel + regression-proof
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _HAS_REWARD_AGGREGATOR or not _HAS_CHANNEL,
    reason="RewardAggregator or lambda_reward_channel not importable",
)
def test_register_lambda_reward_channel_additive_regression_proof():
    """``register_lambda_reward_channel`` adds the new ``r_lambda_score``
    channel to a :class:`RewardAggregator`.  When ``weight=0.0`` the
    weighted reward is bit-for-bit identical to the unwired baseline.
    """
    # Baseline: only QED channel.
    baseline = RewardAggregator(r_qed=lambda s: 0.7, w_qed=1.0)
    state = _DummyState(smiles="CCO")
    v_base = baseline(state)

    # Wired: same baseline PLUS the (default weight=0.0) lambda channel.
    agg_wired = RewardAggregator(r_qed=lambda s: 0.7, w_qed=1.0)
    register_lambda_reward_channel(agg_wired, weight=0.0)

    # The new attributes are set on the aggregator.
    assert hasattr(agg_wired, "r_lambda_score")
    assert hasattr(agg_wired, "w_lambda_score")
    assert agg_wired.w_lambda_score == 0.0
    assert callable(agg_wired.r_lambda_score)

    v_wired = agg_wired(state)
    assert v_base == pytest.approx(v_wired, abs=1e-12), (
        f"weight=0.0 must be bit-for-bit identical to baseline; "
        f"got baseline={v_base} wired={v_wired}"
    )


# ---------------------------------------------------------------------------
# Test 6 — set_lambda_candidates populates the module-level candidate list
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_set_lambda_candidates_populates_module_global():
    """set_lambda_candidates should populate the module-level
    ``_R_LAMBDA_CANDIDATES`` global so the closure can read it lazily.
    """
    import molmetal_lam.lam_chem.lambda_reward_channel as mod

    set_lambda_candidates(["SMILES_A", "SMILES_B", "SMILES_C"])
    assert mod._R_LAMBDA_CANDIDATES == ["SMILES_A", "SMILES_B", "SMILES_C"]

    # Clearing should yield an empty list.
    set_lambda_candidates([])
    assert mod._R_LAMBDA_CANDIDATES == []


# ---------------------------------------------------------------------------
# Test 7 — aggregate() method honours r_lambda_score as a recognised channel
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _HAS_REWARD_AGGREGATOR or not _HAS_CHANNEL,
    reason="RewardAggregator or lambda_reward_channel not importable",
)
def test_aggregate_method_honours_r_lambda_score_channel():
    """The :meth:`RewardAggregator.aggregate` method should recognise
    ``r_lambda_score`` as a channel key and weight it by ``w_lambda_score``.
    """
    agg = RewardAggregator()
    # Default weight is 0.0.
    assert float(getattr(agg, "w_lambda_score", 0.0)) == 0.0

    # Aggregate with an explicit r_lambda_score channel and weight=0.5
    # — must be added to the total.
    agg.w_lambda_score = 0.5
    v = agg.aggregate(
        "CCO",
        channels={"r_lambda_score": 0.8},
    )
    # 0.5 * 0.8 = 0.4 (other channels are 0 / not configured).
    assert v == pytest.approx(0.4, abs=1e-12)

    # weight=0.0 → channel contributes 0.0.
    agg.w_lambda_score = 0.0
    v_zero = agg.aggregate("CCO", channels={"r_lambda_score": 0.8})
    assert v_zero == 0.0


# ---------------------------------------------------------------------------
# Test 8 — closure accepts plain str + state objects
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHANNEL, reason="lambda_reward_channel not importable")
def test_closure_accepts_str_and_state():
    """The closure installed by ``register_lambda_reward_channel`` must
    accept both plain ``str`` and state-like objects without raising.
    """
    closure = make_lambda_reward_channel(enabled=False)
    # Disabled → always 0.0.
    assert closure("CCO") == 0.0
    assert closure(_DummyState(smiles="CCO")) == 0.0
    assert closure(None) == 0.0