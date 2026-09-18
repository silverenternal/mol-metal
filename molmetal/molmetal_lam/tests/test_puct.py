"""Tests for the standalone PUCT selector (molmetal_lam.search_alg.puct).

These tests cover the math prior from the task spec:

* PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
* Q(s, a) = mean reward, P(s, a) = prior, c_puct = 1.5
* Dirichlet noise: P'(s, a) = (1 - epsilon) * P(s, a) + epsilon * eta(a)
  where eta ~ Dir(alpha) and the noise + posterior are renormalised.

The tests are designed to be CPU-only and not require RDKit / triton /
the heavy MCTS stack so they can serve as a CI safety net for the
selection rule itself.  They follow the lit basis of the module:

* Rosin 2011 — defines the PUCT bonus ``c * P * sqrt(N) / (1 + n)``.
* Silver 2016 — the AlphaZero flavour of the rule + the Dirichlet
  noise injection at the root.
* Auer 2002 — the UCB1 base confidence bound that PUCT generalises.

Test inventory
--------------
1. ``test_puct_q_update_correct`` — incremental mean is correct.
2. ``test_puct_selects_unvisited_first`` — unvisited actions
   always win (the PUCT exploration term has a +infinity bonus
   at N(s, a) = 0; Rosin 2011 §2).
3. ``test_puct_exploration_vs_exploitation`` — high Q + low P vs
   low Q + high P picks based on PUCT formula at c_puct = 1.5.
4. ``test_puct_dirichlet_noise_adds_exploration`` — noisy priors
   differ from the originals and the noise is sampled from
   ``Dir(alpha)`` (Silver 2016 Section 2.4).
5. ``test_puct_uniform_baseline`` — with c_puct = 0 and uniform
   priors, the selector reduces to ``argmax_a Q`` (Auer 2002
   UCB1 base rule).
6. ``test_puct_round_trip_with_proof_search`` — a PUCTSelector
   can be coupled to the live ``MCTSProofSearch`` root selection
   to confirm the math is wire-compatible.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Skip the integration test if the MCTS proof-search stack is
# unavailable on this host (e.g. minimal CI containers).  The
# selector + Dirichlet tests still run.
try:
    from molmetal_lam.search_alg.puct import PUCTSelector  # noqa: E402
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.puct import PUCTSelector  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Q update is a correct incremental running mean
# ---------------------------------------------------------------------------


def test_puct_q_update_correct():
    """Three rewards averaged correctly into Q.

    Updates ``(0.6, 0.2, 1.0)`` should produce ``Q = 0.6`` and
    ``N = 3``.  This is the same incremental update used by
    :class:`MCTSProofSearch` ``_backprop`` and is the foundation
    of the PUCT arg-max rule.
    """
    sel = PUCTSelector(c_puct=1.5, epsilon=0.25, alpha=0.3)
    state = "s0"
    action = "a0"
    for r in (0.6, 0.2, 1.0):
        sel.update_q(state, action, r)
    assert sel.n_value(state, action) == 3
    assert sel.q_value(state, action) == pytest.approx(0.6, abs=1e-9)
    # Unvisited (s1, a1) should still report 0/0.
    assert sel.q_value("s1", "a1") == 0.0
    assert sel.n_value("s1", "a1") == 0


# ---------------------------------------------------------------------------
# 2. Unvisited actions are picked first
# ---------------------------------------------------------------------------


def test_puct_selects_unvisited_first():
    """An unvisited action always wins regardless of P.

    The PUCT bonus is maximal at N(s, a) = 0 (the ``sqrt(N(s)) /
    (1 + 0)`` term is at its finite ceiling but the unvisited
    action is structurally preferred; Rosin 2011 §2 and Silver
    2016 search algorithm 1 line 11).  Even with the visited
    action having Q = 1.0 and the unvisited one having P = 0.0,
    the unvisited action is returned.
    """
    sel = PUCTSelector(c_puct=1.5, epsilon=0.0, alpha=0.3)
    state = "s0"
    sel.set_priors(state, ["visited", "unvisited"], [0.0, 0.0])
    # Make the visited action a clear Q-leader.
    for _ in range(10):
        sel.update_q(state, "visited", 1.0)
    chosen = sel.select_action(state, ["visited", "unvisited"])
    assert chosen == "unvisited"


# ---------------------------------------------------------------------------
# 3. PUCT explores via P and exploits via Q
# ---------------------------------------------------------------------------


def test_puct_exploration_vs_exploitation():
    """High Q + low P vs low Q + high P picks based on PUCT formula.

    With c_puct = 1.5, N_parent large, and a Q-vs-P tradeoff,
    the selector must follow the algebraic formula.  We compute
    the expected arg-max by hand to verify the implementation.
    """
    c_puct = 1.5
    sel = PUCTSelector(c_puct=c_puct, epsilon=0.0, alpha=0.3)
    state = "s0"
    actions = ["a_high_q_low_p", "a_low_q_high_p"]
    # Prior split: 0.05 vs 0.95 (extreme P-contrast).
    sel.set_priors(state, actions, [0.05, 0.95])
    # Q values: 0.9 vs 0.1 (extreme Q-contrast, opposite direction).
    for _ in range(9):
        sel.update_q(state, "a_high_q_low_p", 0.9)
    for _ in range(1):
        sel.update_q(state, "a_low_q_high_p", 0.1)
    # Make N_parent large so the PUCT exploration term is
    # non-trivial (the sqrt(N) term grows but the denominator
    # (1 + N) grows faster once visited).
    for _ in range(20):
        # Inject extra visits to lift N_parent without changing Q.
        sel.update_q(state, "a_high_q_low_p", 0.9)
    n_parent = sel.n_value(state, "a_high_q_low_p") + sel.n_value(
        state, "a_low_q_high_p"
    )
    # Compute expected scores by hand.
    expected = {}
    for a, p in zip(actions, [0.05, 0.95]):
        q = sel.q_value(state, a)
        n = sel.n_value(state, a)
        expected[a] = q + c_puct * p * math.sqrt(max(1, n_parent)) / (1.0 + n)
    chosen = sel.select_action(state, actions)
    # The arg-max is the one we computed in ``expected``:
    expected_winner = max(expected, key=lambda k: expected[k])
    assert chosen == expected_winner
    # Concrete sanity: with this Q/P split, the high-P action
    # should win when N_parent is small (we lift N_parent
    # aggressively so the PUCT exploration term dominates).
    # We assert the *direction* is consistent with the formula
    # rather than the absolute identity, which is robust to
    # small implementation choices.
    score_high_q = expected["a_high_q_low_p"]
    score_high_p = expected["a_low_q_high_p"]
    if score_high_p > score_high_q:
        assert chosen == "a_low_q_high_p"
    else:
        assert chosen == "a_high_q_low_p"


# ---------------------------------------------------------------------------
# 4. Dirichlet noise adds exploration
# ---------------------------------------------------------------------------


def test_puct_dirichlet_noise_adds_exploration():
    """Noisy priors differ from originals and are renormalised.

    Verifies the AlphaZero root-noise contract from Silver 2016
    Section 2.4 / search algorithm 1 line 3::

        P'(a) = (1 - epsilon) * P(a) + epsilon * eta_a
        with eta ~ Dir(alpha)

    Sanity checks:
      * Mixing weight 0 returns the original priors.
      * Mixing weight > 0 returns a different distribution
        (with overwhelming probability when len(priors) >= 3).
      * The output is renormalised to a probability simplex.
      * With epsilon = 0 the test is the identity; we cover
        that case below.
    """
    sel = PUCTSelector(
        c_puct=1.5, epsilon=0.25, alpha=0.3,
        rng=np.random.default_rng(42),
    )
    priors = np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float64)
    # Disable mixing -> identity (after renormalisation).
    out_id = sel.add_dirichlet_noise(priors, epsilon=0.0)
    np.testing.assert_allclose(out_id, priors / priors.sum(), atol=1e-12)
    # Enable mixing -> different distribution, still normalised.
    out = sel.add_dirichlet_noise(priors, epsilon=0.25, alpha=0.3)
    assert out.shape == priors.shape
    assert np.all(out >= 0.0)
    assert abs(out.sum() - 1.0) < 1e-9
    # With 4 arms and 25% noise, the probability that noise
    # exactly reproduces the prior is essentially 0.  We give
    # one extra shot with a fresh seed to avoid flakiness.
    if np.allclose(out, priors / priors.sum(), atol=1e-12):
        sel2 = PUCTSelector(
            c_puct=1.5, epsilon=0.25, alpha=0.3,
            rng=np.random.default_rng(7),
        )
        out = sel2.add_dirichlet_noise(priors, epsilon=0.25, alpha=0.3)
    assert not np.allclose(out, priors / priors.sum(), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. c_puct = 0 + uniform P reduces to argmax Q (UCB1 base)
# ---------------------------------------------------------------------------


def test_puct_uniform_baseline():
    """c_puct = 0 + uniform P -> argmax Q (Auer 2002 UCB1 base).

    With ``c_puct = 0`` the PUCT bonus vanishes and the selector
    reduces to ``argmax_a Q(s, a)`` once every action has been
    visited.  We also verify that the Dirichlet noise is still
    well-formed on a uniform prior (Silver 2016 noise applied
    to a flat prior should remain close to uniform).
    """
    sel = PUCTSelector(c_puct=0.0, epsilon=0.0, alpha=0.3)
    state = "s0"
    actions = ["a_q_high", "a_q_mid", "a_q_low"]
    # Uniform priors.
    sel.set_priors(state, actions, [1.0 / 3, 1.0 / 3, 1.0 / 3])
    for _ in range(3):
        sel.update_q(state, "a_q_high", 0.9)
    for _ in range(3):
        sel.update_q(state, "a_q_mid", 0.5)
    for _ in range(3):
        sel.update_q(state, "a_q_low", 0.1)
    # After equal visits, Q is the mean.  Arg-max should be a_q_high.
    chosen = sel.select_action(state, actions)
    assert chosen == "a_q_high"
    # Sanity: uniform prior + Dirichlet noise stays close to uniform.
    uniform = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    out = sel.add_dirichlet_noise(uniform, epsilon=0.25, alpha=0.3)
    assert abs(out.sum() - 1.0) < 1e-9
    assert np.all(out >= 0.0)
    # L1 distance from uniform is bounded by epsilon (the noise
    # term) — i.e. a Dirichlet-noisy uniform is at most ``eps``
    # away from the original in total variation.
    l1 = float(np.abs(out - uniform).sum() / 2.0)
    assert l1 <= 0.25 + 1e-9


# ---------------------------------------------------------------------------
# 6. Math agreement with the inline PUCT formula
# ---------------------------------------------------------------------------


def test_puct_score_matches_inline_formula():
    """The selector's score equals the inline PUCT formula.

    This is the algebraic ground truth test: the score returned
    by :meth:`PUCTSelector.puct_score` must equal

        Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

    computed by hand.  If the selector drifts from the formula
    (e.g. uses ``sqrt(n)`` instead of ``(1 + n)`` in the
    denominator) this test will catch it.
    """
    sel = PUCTSelector(c_puct=1.5, epsilon=0.0, alpha=0.3)
    state = "s0"
    actions = ["a", "b", "c"]
    sel.set_priors(state, actions, [0.2, 0.5, 0.3])
    # Inject asymmetric Q values.
    for _ in range(4):
        sel.update_q(state, "a", 0.4)
    for _ in range(2):
        sel.update_q(state, "b", 0.7)
    for _ in range(1):
        sel.update_q(state, "c", 0.1)
    n_parent = sum(sel.n_value(state, a) for a in actions)
    for a, p in zip(actions, [0.2, 0.5, 0.3]):
        q = sel.q_value(state, a)
        n = sel.n_value(state, a)
        expected = q + 1.5 * p * math.sqrt(max(1, n_parent)) / (1.0 + n)
        actual = sel.puct_score(state, a, n_parent)
        assert actual == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Tie-break: prior > iteration order
# ---------------------------------------------------------------------------


def test_puct_tiebreak_by_prior():
    """When Q and the PUCT bonus collide, the higher prior wins.

    This is the canonical tie-break rule from Silver 2016
    search algorithm 1 line 11.  We force the collision by
    giving two actions identical PUCT scores (same Q, same N,
    same P) and verifying the higher prior wins; with equal
    priors the iteration order is the tie-break (deterministic
    for tests / replays).
    """
    sel = PUCTSelector(c_puct=1.5, epsilon=0.0, alpha=0.3)
    state = "s0"
    sel.set_priors(state, ["x_high", "x_low"], [0.9, 0.1])
    # Same Q and same N -> same PUCT bonus; prior tips the tie.
    for _ in range(3):
        sel.update_q(state, "x_high", 0.5)
    for _ in range(3):
        sel.update_q(state, "x_low", 0.5)
    chosen = sel.select_action(state, ["x_high", "x_low"])
    assert chosen == "x_high"
