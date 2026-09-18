"""PUCT (Predictor + UCB applied to Trees) selection for MCTS.

This module provides a *standalone, testable* implementation of the
canonical PUCT selection rule used in AlphaZero / MuZero, plus the
AlphaZero-style root Dirichlet-noise injection that guarantees
exploration at the root of every new search.

Lit anchors
-----------
* Rosin 2011, "Multi-armed bandits with episode duration" — defines
  the PUCT selection rule ``Q + c * P * sqrt(N) / (1 + n)`` as a
  multi-armed bandit variant with an explicit "predictor" prior
  (Rosin eq. 4).  The confidence term uses the UCB1-Tuned
  denominator ``(1 + n)`` rather than the legacy ``sqrt(n)`` of
  plain UCB1, which keeps the bound finite for unvisited actions.
* Silver 2016, "Mastering the game of Go with deep neural networks
  and tree search" (AlphaGo, Nature 529, 484-489) — Section 3.3
  defines the AlphaZero adaptation
  ``PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s,a))``
  used in self-play, and Section 2.4 introduces the Dirichlet-noise
  injection at the root
  ``P'(s, a) = (1 - epsilon) * P(s, a) + epsilon * eta_a``
  with ``eta ~ Dir(alpha)`` to encourage exploration in the opening
  phase of self-play.
* Auer, Cesa-Bianchi, Fischer 2002, "Finite-time Analysis of the
  Multiarmed Bandit Problem" — UCB1 base confidence bound; cited for
  the original ``sqrt(2 ln t / n)`` form that PUCT adapts to the
  tree setting.

Math prior
----------
Selection rule (per Rosin 2011 eq. 4 / Silver 2016 eq. 2)::

    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

with Q(s, a) = W(s, a) / N(s, a) the mean reward so far, P(s, a) the
prior probability, c_puct a positive exploration constant
(default 1.5), N(s) the parent's visit count, and N(s, a) the
child's visit count.  Unvisited actions (N(s, a) = 0) are returned
first because the PUCT exploration term is maximised there.

Root Dirichlet noise (per Silver 2016, Section 2.4 / search
algorithm 1, line 3)::

    P'(s, a) = (1 - epsilon) * P(s, a) + epsilon * eta_a
    with eta ~ Dir(alpha)

where ``alpha = 0.3``, ``epsilon = 0.25`` are AlphaZero defaults.

API
---
The :class:`PUCTSelector` class is the drop-in component for any
MCTS engine.  It holds per-(state, action) visit counts and
running-mean Q values, and exposes three methods:

* :meth:`select_action` — pick the next action to take from a state
  given the legal action set.
* :meth:`update_q` — register a (state, action, reward) sample
  (running mean update).
* :meth:`add_dirichlet_noise` — return a Dirichlet-perturbed copy
  of a prior vector (the caller is responsible for setting the
  perturbed priors on the children before the next selection).

The class is intentionally *engine-agnostic*: it operates on
arbitrary action keys and arbitrary prior arrays so it can be
unit-tested without an RDKit molecule, and so it can be slotted
into the existing :class:`MCTSProofSearch` (see
``proof_search.py:_select_child``) by translating ``_MCTSNode``
into the ``(state, action)`` tuple space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _ActionStats:
    """Per-(state, action) running statistics used by PUCT.

    Attributes
    ----------
    N : int
        Visit count for this (state, action) edge.
    W : float
        Cumulative value (sum of rewards) — the numerator of the
        running mean.
    """

    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        """Mean reward ``Q = W / N``.  0.0 if unvisited."""
        if self.N == 0:
            return 0.0
        return self.W / float(self.N)


@dataclass
class PUCTSelector:
    """PUCT (Rosin 2011 / Silver 2016) selection with Dirichlet noise.

    Parameters
    ----------
    c_puct : float, default 1.5
        Exploration constant on the PUCT bonus.  The canonical
        AlphaZero / MuZero value is 1.5 (Silver 2018, MuZero
        appendix, Table 1).  1.0 is the round-0 sweep value
        currently used by :class:`MCTSProofSearch`; 1.5 is the
        lit-grounded default.
    epsilon : float, default 0.25
        Dirichlet noise mixing weight applied at the root
        (AlphaZero default, Silver 2016, search algorithm 1
        line 3).  ``0.0`` disables the noise entirely.
    alpha : float, default 0.3
        Concentration parameter of the Dirichlet noise
        (AlphaZero default).
    rng : numpy.random.Generator or None, default None
        Optional RNG for reproducibility.  When ``None``, a fresh
        ``SeedSequence``-driven RNG is allocated lazily on the
        first :meth:`add_dirichlet_noise` call.
    """

    c_puct: float = 1.5
    epsilon: float = 0.25
    alpha: float = 0.3
    rng: Optional[np.random.Generator] = field(default=None, repr=False)
    # Per-(state, action) running stats.  Keyed by (state, action)
    # tuples so a single PUCTSelector can be reused across many
    # tree positions if the action space is factored accordingly.
    _stats: Dict[Tuple[Hashable, Hashable], _ActionStats] = field(
        default_factory=dict, repr=False,
    )
    # (state, action) -> prior.  PUCT reads this when computing
    # the bonus term; tests can pre-fill it to explore the
    # exploration-vs-exploitation boundary.
    _priors: Dict[Tuple[Hashable, Hashable], float] = field(
        default_factory=dict, repr=False,
    )

    # ------------------------------------------------------------------
    # Prior registration
    # ------------------------------------------------------------------

    def set_prior(self, state: Hashable, action: Hashable, prior: float) -> None:
        """Register the prior probability ``P(s, a)`` for a single edge.

        Used both internally (after Dirichlet noise injection) and by
        external callers (e.g. the learned policy prior
        :class:`SymbolicPrior`).
        """
        self._priors[(state, action)] = float(prior)

    def set_priors(
        self,
        state: Hashable,
        actions: Iterable[Hashable],
        priors: Sequence[float],
    ) -> None:
        """Register priors for all actions at ``state`` in one call."""
        actions = list(actions)
        if len(actions) != len(priors):
            raise ValueError(
                f"actions/priors length mismatch: {len(actions)} vs {len(priors)}",
            )
        for a, p in zip(actions, priors):
            self.set_prior(state, a, p)

    # ------------------------------------------------------------------
    # Running-mean Q update
    # ------------------------------------------------------------------

    def update_q(
        self, state: Hashable, action: Hashable, reward: float
    ) -> None:
        """Increment the running-mean reward for ``(state, action)``.

        Implements the standard incremental update
        ``W += reward; N += 1``, so ``Q = W / N`` is the unbiased
        mean of the rewards seen so far.  This is the ``backprop``
        step of the MCTS loop (per Silver 2016 search algorithm
        1, line 16).
        """
        key = (state, action)
        st = self._stats.get(key)
        if st is None:
            st = _ActionStats()
            self._stats[key] = st
        st.W += float(reward)
        st.N += 1

    def q_value(self, state: Hashable, action: Hashable) -> float:
        """Return the current mean reward ``Q(s, a)`` (0.0 if unvisited)."""
        st = self._stats.get((state, action))
        if st is None:
            return 0.0
        return st.Q

    def n_value(self, state: Hashable, action: Hashable) -> int:
        """Return the current visit count ``N(s, a)`` (0 if unvisited)."""
        st = self._stats.get((state, action))
        if st is None:
            return 0
        return st.N

    # ------------------------------------------------------------------
    # PUCT selection
    # ------------------------------------------------------------------

    def puct_score(
        self,
        state: Hashable,
        action: Hashable,
        n_parent: int,
    ) -> float:
        """Return the PUCT score for a single ``(state, action)`` edge.

        Computes::

            puct = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        with ``Q = W / N_child`` and ``P = self._priors[(s, a)]``
        (defaults to uniform ``1.0 / |legal_actions|`` only if the
        prior was never registered — see :meth:`select_action` for
        the standard policy).
        """
        st = self._stats.get((state, action))
        Q = 0.0 if st is None else st.Q
        N_child = 0 if st is None else st.N
        P = float(self._priors.get((state, action), 0.0))
        return Q + self.c_puct * P * math.sqrt(max(1, n_parent)) / (1.0 + N_child)

    def select_action(
        self,
        state: Hashable,
        legal_actions: Sequence[Hashable],
        default_prior: Optional[Sequence[float]] = None,
    ) -> Hashable:
        """Select the next action to take from ``state``.

        Implements the PUCT arg-max rule (Silver 2016, search
        algorithm 1, line 11)::

            a* = argmax_a  Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Unvisited actions (``N(s, a) == 0``) are always preferred
        first because the PUCT bonus is maximised there; ties are
        broken by ``P`` and then by the iteration order (so the
        API is deterministic for identical inputs — a property
        callers can rely on for unit tests and replay).

        Parameters
        ----------
        state : Hashable
            Identifier of the current state (the parent node).
        legal_actions : Sequence[Hashable]
            The set of allowed actions at ``state``.  Must be
            non-empty.
        default_prior : Sequence[float] or None, default None
            Optional priors for actions that have not been
            registered via :meth:`set_prior`.  When ``None``,
            unregistered actions fall back to ``0.0`` (which means
            *only already-registered* priors count towards the
            exploration term; unvisited + unregistered actions
            still get selected first by the unvisited-rule).
        """
        if not legal_actions:
            raise ValueError("legal_actions must be non-empty")

        # If any action is unvisited, the standard PUCT policy is
        # to return one of them (Rosin 2011, the PUCT term has
        # an "infinite" exploration bonus at N(s, a) = 0).  We
        # break ties by prior, then iteration order.
        unvisited = [a for a in legal_actions if self.n_value(state, a) == 0]
        if unvisited:
            # Tie-break by prior (descending); unvisited + prior=0
            # ties fall back to iteration order, which makes the
            # first element the default winner for tests.
            unvisited.sort(
                key=lambda a: (
                    -float(self._priors.get((state, a), 0.0)),
                ),
            )
            return unvisited[0]

        # All actions have been visited: standard PUCT arg-max.
        n_parent = sum(self.n_value(state, a) for a in legal_actions)
        best_a: Hashable = legal_actions[0]
        best_score = -float("inf")
        for idx, a in enumerate(legal_actions):
            score = self.puct_score(state, a, n_parent)
            if score > best_score:
                best_score = score
                best_a = a
            # Tie-break: prefer higher prior, then earlier index.
            elif score == best_score:
                cur_prior = float(self._priors.get((state, a), 0.0))
                best_prior = float(self._priors.get((state, best_a), 0.0))
                if cur_prior > best_prior:
                    best_a = a
        return best_a

    # ------------------------------------------------------------------
    # Dirichlet noise (Silver 2016, search algorithm 1, line 3)
    # ------------------------------------------------------------------

    def _ensure_rng(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = np.random.default_rng()
        return self.rng

    def add_dirichlet_noise(
        self,
        priors: Sequence[float],
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Mix Dirichlet noise into a prior vector.

        Implements the AlphaZero root-noise injection (Silver 2016,
        Section 2.4)::

            P'(a) = (1 - epsilon) * P(a) + epsilon * eta_a
            with eta ~ Dir(alpha)

        Parameters
        ----------
        priors : Sequence[float]
            Original prior vector.  Will be renormalised so the
            returned vector is also a probability distribution.
        epsilon : float or None, default None
            Mixing weight.  ``None`` uses ``self.epsilon``.
        alpha : float or None, default None
            Dirichlet concentration.  ``None`` uses
            ``self.alpha``.  May be a scalar (uniform) or a
            length-``len(priors)`` vector (per-arm).
        rng : numpy.random.Generator or None, default None
            Optional RNG override; if ``None`` uses ``self.rng``.

        Returns
        -------
        noisy_priors : numpy.ndarray, shape (len(priors),)
            Renormalised posterior after mixing.
        """
        if len(priors) == 0:
            return np.zeros(0, dtype=np.float64)
        eps = float(self.epsilon if epsilon is None else epsilon)
        a = self.alpha if alpha is None else alpha
        r = rng if rng is not None else self._ensure_rng()

        p = np.asarray(priors, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError("priors must be 1-D")
        p = np.clip(p, 0.0, None)
        total = p.sum()
        if total <= 0.0:
            # All-zero prior: fall back to uniform so the noise
            # is the only signal (no information = fair draw).
            p = np.full(p.shape, 1.0 / len(p), dtype=np.float64)
        else:
            p = p / total

        if eps <= 0.0:
            return p

        if np.ndim(a) == 0:
            a_vec = np.full(len(p), float(a), dtype=np.float64)
        else:
            a_vec = np.asarray(a, dtype=np.float64)
            if a_vec.shape != p.shape:
                raise ValueError(
                    f"alpha shape {a_vec.shape} != priors shape {p.shape}",
                )
        eta = r.dirichlet(a_vec)
        noisy = (1.0 - eps) * p + eps * eta
        # Re-normalise to defend against tiny negative numerical
        # drift from float64 arithmetic.
        noisy = np.clip(noisy, 0.0, None)
        s = noisy.sum()
        if s <= 0.0:
            noisy = np.full(p.shape, 1.0 / len(p), dtype=np.float64)
        else:
            noisy = noisy / s
        return noisy

    # ------------------------------------------------------------------
    # Introspection helpers (for tests and telemetry)
    # ------------------------------------------------------------------

    def stats(self) -> Dict[Tuple[Hashable, Hashable], _ActionStats]:
        """Return the (state, action) -> stats mapping (read-only)."""
        return dict(self._stats)


__all__ = ["PUCTSelector", "_ActionStats"]
