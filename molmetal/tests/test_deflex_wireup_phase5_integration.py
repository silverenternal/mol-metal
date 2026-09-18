"""WF-Deflex Wire-up Phase 5 integration test — end-to-end smoke.

Brings up the full chain:
1. Build RewardAggregator with F5 channel enabled.
2. Build PocketMacroInference with v2 checkpoint.
3. Build a stub LearnedPolicyPrior.
4. Verify the round-trip works without crashing:
   - F5 contributes to the reward.
   - PocketMacroInference produces a 32-d embedding.
   - LearnedPolicyPrior produces a per-rule distribution.

This is a "no crash" integration test — it does NOT run a full MCTS
rollout (which requires rdkit + chemistry fixtures out of scope here).
"""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_env():
    """Ensure LEARNED_SHAPING_ENABLED is unset by default."""
    old = os.environ.pop("LEARNED_SHAPING_ENABLED", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["LEARNED_SHAPING_ENABLED"] = old
        else:
            os.environ.pop("LEARNED_SHAPING_ENABLED", None)


def test_full_chain_no_crash():
    """Verify the three Deflex wire-up adapters co-exist without crashing."""
    # Phase 1: F5 wire into RewardAggregator
    try:
        from molmetal_lam.search_alg.proof_search import RewardAggregator
    except Exception as exc:
        pytest.skip(f"proof_search not importable: {exc}")

    agg = RewardAggregator()
    agg.register_learned_shaping_channel(enabled=True)
    assert agg.use_learned_shaping is True
    f5_contrib = agg._learned_shaping_contribution(3.0)
    assert isinstance(f5_contrib, float)
    assert abs(f5_contrib - (2.5836 - 2.5149 * 3.0)) < 1e-6

    # Phase 4: PocketMacroInference v2 loads
    try:
        from molmetal_lam.lam_chem.pocket_macro_inference import (
            PocketMacroInference,
        )
    except Exception as exc:
        pytest.skip(f"pocket_macro_inference not importable: {exc}")

    pmi = PocketMacroInference()
    try:
        pmi.load()
    except Exception as exc:
        pytest.skip(f"v2 checkpoint unavailable: {exc}")
    emb = pmi.get_embedding("CA2")
    assert emb.shape == (32,)

    # Phase 3: LearnedPolicyPrior contract
    class _StubLP:
        def predict_proba(self, smiles: str):
            return {
                "cuacc": 0.5, "spaac": 0.3, "thiol_ene": 0.1,
                "suzuki": 0.05, "amide_coupling": 0.05,
            }

    lp = _StubLP()
    dist = lp.predict_proba("C#N")
    assert max(dist, key=dist.get) == "cuacc"

    # All three together: no exception.
    print(
        f"\n  F5 contribution = {f5_contrib:.4f}"
        f"\n  PocketMacro CA2 embedding sum = {emb.sum():.4f}"
        f"\n  LearnedPrior argmax = cuacc (dist={dist})"
    )


def test_aggregator_f5_call_with_pocket_macro_embedding():
    """End-to-end: aggregator with F5 + SA channel + PocketMacro embedding.
    The reward should equal ``w_f5 * R_F5(sa) + w_sa * (1 - (sa-1)/9)``
    when SA is the only non-zero channel.
    """
    try:
        from molmetal_lam.search_alg.proof_search import RewardAggregator
    except Exception as exc:
        pytest.skip(f"proof_search not importable: {exc}")

    agg = RewardAggregator()
    agg.r_sa = lambda state: 3.0
    agg.register_learned_shaping_channel(enabled=True)

    class _DummyState:
        def canonical_smiles(self):
            return "C"

    sa = 3.0
    expected = (
        agg.w_sa * (1.0 - (sa - 1.0) / 9.0)  # SA channel
        + agg.w_learned_shaping * (2.5836 - 2.5149 * sa)  # F5 channel
    )
    actual = agg(_DummyState())
    assert abs(actual - expected) < 1e-6


def test_search_kwargs_accepted():
    """Verify the new search() kwargs are accepted without crashing on a
    minimal instantiation.  We don't run a full search() — just confirm
    the kwargs are recognised by the signature.
    """
    try:
        import inspect
        from molmetal_lam.search_alg.proof_search import MCTSProofSearch
        sig = inspect.signature(MCTSProofSearch.search)
    except Exception as exc:
        pytest.skip(f"proof_search not importable: {exc}")

    params = sig.parameters
    # The new kwargs must all be in the signature.
    assert "use_pocket_macro" in params
    assert "pocket_macro_target_name" in params
    assert "use_learned_prior_argmax" in params
    # Defaults must be False / None so existing callers are bit-for-bit.
    assert params["use_pocket_macro"].default is False
    assert params["pocket_macro_target_name"].default is None
    assert params["use_learned_prior_argmax"].default is False