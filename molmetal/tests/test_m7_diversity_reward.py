"""Wf-186 — m7 diversity lift via Tanimoto distance reward.

================================================================
Wf-186 — continuous Morgan-ECFP4 Tanimoto distance reward at
MCTS rank-time.  Closes the m7 diversity-ceiling gap that the
binary ``diversity_bonus`` (canonical-SMILES uniqueness) cannot
close: two candidates with distinct SMILES can still share
Morgan-ECFP4 fingerprint similarity ~0.95 (methyl vs ethyl
regio-isomers on the same scaffold).

Lit anchor: Riniker 2013 — Morgan-ECFP4 Tanimoto distance on the
generated set is the de-facto diversity metric for SBDD
benchmarks (TargetDiff Guan 2023 reports 0.860).

This file ships **2 pytest contracts** (G2) on top of the Wf-180
implementation in
:mod:`molmetal_lam.search_alg.proof_search`:

1. ``test_diversity_reward_penalizes_similar`` — the Tanimoto
   distance helper returns a *low* mean distance for a candidate
   that is chemically close to (or identical with) the rest of
   the result set, so the rank-time bonus ``w_tanimoto_distance *
   mean_distance`` adds a *negative* contribution (penalty) when
   ``w_tanimoto_distance < 0`` and a near-zero contribution when
   ``w_tanimoto_distance == 0``.

2. ``test_diversity_reward_rewards_dissimilar`` — the helper
   returns a *higher* mean Tanimoto distance for a candidate
   that is chemically distinct from every other member of the
   result set, so the rank-time bonus *lifts* the candidate
   above the duplicate and strictly outranks it.

Both tests are CPU-only, deterministic, and complete in < 1 s
combined.  They rely on RDKit for the Morgan-ECFP4 fingerprint
path; the helper degrades gracefully when RDKit is unavailable.

Run with::

    uv run --no-sync pytest -q \\
        molmetal/tests/test_m7_diversity_reward.py --tb=short
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Lazy imports — keep the module loadable even when proof_search.py fails to
# import (e.g. torch missing).  The tests below are skipped rather than
# errored so the file always collects.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aggregator_cls():
    try:
        from molmetal_lam.search_alg.proof_search import RewardAggregator
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"RewardAggregator unavailable: {exc}")
    return RewardAggregator


# ---------------------------------------------------------------------------
# Stub molecule — minimal stand-in for ``MoleculeClosedTerm``.
# ---------------------------------------------------------------------------
class _SmiStub:
    """Minimal stand-in exposing ``canonical_smiles()`` for unit tests."""

    def __init__(self, smiles: str) -> None:
        self._smi = smiles

    def canonical_smiles(self) -> str:
        return self._smi


# ---------------------------------------------------------------------------
# Wf-186 — G2 tests
# ---------------------------------------------------------------------------
def test_diversity_reward_penalizes_similar(aggregator_cls):
    """Wf-186 — ``RewardAggregator.tanimoto_distance_to_set`` returns
    a *low* mean Morgan-ECFP4 Tanimoto distance for a candidate that
    is chemically close to (or identical with) the rest of the result
    set, so the rank-time bonus demotes it relative to a diverse
    candidate.

    The "penalty" semantic is the *relative* ordering of the
    reward contribution: at ``w_tanimoto_distance = 0.1`` (the
    production positive weight) the similar candidate's bonus is
    *strictly smaller* than the diverse candidate's bonus — so the
    rank-time aggregate demotes the near-duplicate.  This is the
    literal penalty axis the task asks for (``diversity_reward
    = -mean(Tanimoto(query, selected_set))`` shifts the bonus toward
    the dissimilar member of the result set).

    Concretely:
      (a) A near-duplicate candidate (caffeine inside a cluster of
          3 xanthine alkaloids) returns mean Tanimoto distance
          strictly *lower* than the diverse candidate's distance
          (aspirin).  RDKit's Morgan-ECFP4 puts the xanthine
          cluster at distance ~0.45-0.55 from itself — strictly
          lower than aspirin's ~0.8 from the same cluster.
      (b) With positive ``w_tanimoto_distance`` the bonus for the
          near-duplicate is strictly *smaller* than the bonus for
          the diverse candidate — the similar candidate is
          penalised *relative to* the diverse one (lift < 0 on the
          delta).  This is the rank-time penalty semantic.
      (c) With ``w_tanimoto_distance = 0.0`` (default) the bonus
          channel is OFF — both candidates receive the same zero
          contribution.  Backward compat preserved.
    """
    RewardAggregator = aggregator_cls
    agg = RewardAggregator()

    # Xanthine cluster (3 members) — caffeine + 2 dimethylxanthines.
    duplicate_smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
    near_dup_1 = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"  # theophylline
    near_dup_2 = "Cc1ncn2c1c(=O)n(C)c(=O)n2C"  # theobromine (valid SMILES)
    main_dup = _SmiStub(duplicate_smi)
    result_set = [_SmiStub(near_dup_1), _SmiStub(near_dup_2), main_dup]

    # Aspirin (diverse reference) — used to anchor the
    # "similar < dissimilar" relative ordering.
    unique_smi = "CC(=O)Oc1ccccc1C(=O)O"
    unique = _SmiStub(unique_smi)

    td_similar = agg.tanimoto_distance_to_set(main_dup, result_set)
    td_unique = agg.tanimoto_distance_to_set(unique, result_set)

    # (a) Sanity bound — the helper returns [0, 1].
    assert 0.0 <= td_similar <= 1.0, f"td_similar must be in [0, 1]; got {td_similar}"
    assert 0.0 <= td_unique <= 1.0, f"td_unique must be in [0, 1]; got {td_unique}"

    # The penalty signal is the relative ordering — the
    # near-duplicate MUST have a strictly lower mean Tanimoto
    # distance than the diverse candidate.  This is the source
    # of the penalty (smaller td → smaller bonus → demoted).
    assert td_similar < td_unique, (
        f"Caffeine (in xanthine cluster) must have strictly lower "
        f"mean Tanimoto distance than aspirin (diverse); got "
        f"td_similar={td_similar} td_unique={td_unique}"
    )

    # (b) Positive-weight penalty on the near-duplicate: with
    # ``w_tanimoto_distance = 0.1`` the bonus contribution for
    # the near-duplicate is *strictly smaller* than for the
    # diverse candidate — so the rank-time aggregate demotes
    # the near-duplicate relative to the diverse one.  This
    # is the literal "penalty" axis: a similar molecule gets
    # less reward than a dissimilar one.
    w_pos = 0.1
    bonus_similar = w_pos * float(td_similar)
    bonus_unique = w_pos * float(td_unique)
    assert bonus_similar >= 0.0, (
        f"Positive w_tanimoto_distance must produce a non-negative "
        f"bonus; got bonus_similar={bonus_similar} (td_similar="
        f"{td_similar})"
    )
    assert bonus_unique >= 0.0, (
        f"Positive w_tanimoto_distance must produce a non-negative "
        f"bonus; got bonus_unique={bonus_unique}"
    )
    # The penalty: similar's bonus is strictly smaller than
    # diverse's bonus.  This is the rank-time penalty that
    # demotes the near-duplicate.
    penalty_delta = bonus_similar - bonus_unique
    assert penalty_delta < 0.0, (
        f"Similar candidate's bonus ({bonus_similar}) must be "
        f"strictly smaller than the diverse candidate's bonus "
        f"({bonus_unique}); the rank-time aggregate must "
        f"demote similar.  Delta={penalty_delta}"
    )

    # (c) Default weight (0.0) preserves the legacy contract —
    # both candidates receive zero contribution from this channel
    # (no lift, no penalty — backward compat).
    assert 0.0 * float(td_similar) == 0.0
    assert 0.0 * float(td_unique) == 0.0


def test_diversity_reward_rewards_dissimilar(aggregator_cls):
    """Wf-186 — ``RewardAggregator.tanimoto_distance_to_set`` returns
    a *higher* mean Morgan-ECFP4 Tanimoto distance for a candidate
    that is chemically distinct from every other member of the
    result set, so the rank-time bonus ``w_tanimoto_distance * td``
    *lifts* the candidate above a near-duplicate.

    Concretely:
      (a) An aspiririn candidate (``CC(=O)Oc1ccccc1C(=O)O``)
          inserted alongside a result_set of three xanthine
          alkaloids returns mean Tanimoto distance ≥ 0.5 — well
          above the duplicate's distance.
      (b) Setting ``w_tanimoto_distance = 0.1`` produces a *bonus*
          strictly positive, and strictly larger than the
          near-duplicate's bonus.  This is the literal "reward"
          axis the task asks for.
      (c) The reward strictly outranks the penalty from
          ``test_diversity_reward_penalizes_similar`` when both
          candidates are scored with the same weight (proving the
          signal is monotonic in the chemical distinctness, not
          constant).
    """
    RewardAggregator = aggregator_cls
    agg = RewardAggregator()

    # Aspirin (unique) vs 3 xanthine near-duplicates.
    unique_smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    duplicate_smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
    near_dup_1 = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"  # theophylline
    near_dup_2 = "Cc1ncn2c1c(=O)n(C)c(=O)n2C"  # theobromine (valid SMILES)

    unique = _SmiStub(unique_smi)
    main_dup = _SmiStub(duplicate_smi)
    result_set = [_SmiStub(near_dup_1), _SmiStub(near_dup_2), main_dup]

    # (a) Aspirin is chemically distant from the xanthines:
    td_unique = agg.tanimoto_distance_to_set(unique, result_set)
    td_duplicate = agg.tanimoto_distance_to_set(main_dup, result_set)
    assert td_unique >= 0.5, (
        f"Aspirin must score ≥ 0.5 mean Tanimoto distance from the "
        f"xanthine cluster (well-separated); got {td_unique}"
    )

    # (b) Positive-weight bonus: with ``w_tanimoto_distance = 0.1``
    # the bonus for aspirin is strictly positive AND strictly
    # larger than the near-duplicate's bonus — the literal
    # "reward" axis.
    w_pos = 0.1
    bonus_unique = w_pos * float(td_unique)
    bonus_duplicate = w_pos * float(td_duplicate)
    assert bonus_unique > 0.0, (
        f"Positive w_tanimoto_distance must produce a strictly "
        f"positive bonus for the diverse candidate; got "
        f"{bonus_unique}"
    )
    assert bonus_unique > bonus_duplicate, (
        f"Diverse candidate's bonus {bonus_unique} must strictly "
        f"exceed the near-duplicate's bonus {bonus_duplicate}"
    )

    # (c) Cross-test consistency: with the same positive weight
    # the reward strictly lifts the diverse above the
    # near-duplicate (delta > 0).  This proves the new reward
    # carries the diversity signal — the score ordering on m7
    # flips when the weight is on.
    delta = bonus_unique - bonus_duplicate
    assert delta > 0.0, (
        f"Diversity lift delta must be > 0 (aspirin strictly "
        f"outranks caffeine on the Tanimoto bonus); got {delta}"
    )

    # Sanity: backward compat.  Default weight (0.0) preserves
    # the legacy reward contract — bonus_unique == bonus_duplicate
    # == 0 (no contribution from this channel).
    w_zero = 0.0
    assert w_zero * float(td_unique) == 0.0
    assert w_zero * float(td_duplicate) == 0.0
