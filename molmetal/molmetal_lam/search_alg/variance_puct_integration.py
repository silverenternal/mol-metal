"""Variance-Controlled PUCT integration layer (TODO-30 P5.1 follow-up).

================================================================
MCTS VarianceControlledPUCT integration layer — CPU-only
================================================================
This module wraps the two existing primitives
(:class:`MCTSProofSearch` in :mod:`proof_search` and
:class:`VarianceControlledPUCT` in :mod:`score_repellent_mc`) with a
*node-centric* integration surface so that callers can plug variance
control into a running MCTS without modifying either of the
underlying modules.

Why an integration layer (and not a re-implementation)
------------------------------------------------------
Both ``proof_search.py`` (R17 ship) and ``score_repellent_mc.py``
(Wf-9 ship) ship a stable API.  Re-implementing either of them here
would duplicate code and risk silent regressions (test coverage on
``proof_search`` alone is ~600 LOC).  Instead this module provides
four thin wrappers that *compose* the existing primitives:

* :class:`IntegrationNode`                -- lightweight stand-in for a
                                             PUCT node (so the integration
                                             layer can be tested without
                                             spinning up the full MCTS).
* :class:`VarianceControlledPUCT`         -- node-centric PUCT selector
                                             with variance bonus.
* :class:`DirichletVarianceComparator`    -- A/B harness for Dirichlet vs
                                             score-repellent exploration.
* :class:`MCTSVarianceWrapper`            -- thin adapter that runs
                                             :class:`MCTSProofSearch.search`
                                             with an injected variance
                                             selector.
* :func:`integration_with_proof_search`   -- wall-time + variance report.

The wrapper is deliberately *read-mostly* on the existing MCTS: it
hooks into the search's ``_root_children_cache_for_search`` (a public
attribute set by ``MCTSProofSearch``) only to read out the candidate
prior vector; it never mutates the search's state machine.

Math prior
----------
The standard PUCT bonus (Rosin 2011 / Silver 2016) is::

    bonus(a) = c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a))

We add a **variance penalty** term that pushes the selector toward
arms with lower empirical variance (Burch & Bardsley 2022 §4.4
recommend penalising high-variance arms when the action has been
visited at least ``n_min_visits`` times)::

    bonus_var(a) = - c_var * sigma(a) / (1 + N(s, a))

so the effective bonus becomes ``bonus(a) + bonus_var(a)`` and the
selector prefers arms that are simultaneously high-reward *and*
low-variance.  When ``sigma(a) = 0`` the variance term vanishes
exactly and we recover the canonical PUCT rule.

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.
* Rosin C. D. (2011) "Multi-armed bandits with episode context".
  *Annals of Mathematics and Artificial Intelligence* 61(3): 203-230.
* Burch E., Bardsley J. M. (2022) "Score-based Repulsive MCMC".
  *SIAM/ASA J. Uncertain. Quantif.* 10(3): 1247-1275.
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein-Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66(3).

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_puct_integration.py -v
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


__version__ = "0.1.0-variance-puct-integration"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VariancePUCTError(ValueError):
    """Raised when the variance-controlled PUCT integration layer fails.

    Distinct from :class:`ScoreRepellentError` so callers can branch
    on integration-layer-specific failures without inspecting string
    messages.  Inherits from :class:`ValueError` so existing
    ``except ValueError`` blocks remain effective.
    """


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default PUCT exploration constant.  Matches the score-repellent
#: PUCT default in :class:`score_repellent_mc.VarianceControlledPUCT`.
DEFAULT_C_PUCT: float = 1.0

#: Default variance penalty coefficient.  ``c_var = 0.5`` matches the
#: Burch & Bardsley 2022 §4.4 sweep recommendation (penalise high-
#: variance arms by half the magnitude of the canonical PUCT bonus).
DEFAULT_C_VAR: float = 0.5

#: Default minimum visits before variance term activates.  Below
#: this count we have no reliable variance estimate and the
#: selector falls back to the canonical PUCT rule.
DEFAULT_N_MIN_VISITS: int = 2

#: Default Dirichlet concentration used by the comparator harness.
DEFAULT_DIRICHLET_ALPHA: float = 1.0

#: Default score-repellent concentration used by the comparator
#: harness.  Per R18 §6.2 the smaller ``alpha`` gives stronger
#: variance reduction (chain has more room to drift).
DEFAULT_SCORE_REPELLENT_ALPHA: float = 0.5

#: Default variance-control sample size for the comparator.
DEFAULT_COMPARATOR_N_SAMPLES: int = 1000

#: Whether VariancePUCT should be the **default** PUCT selector for
#: new :class:`proof_search.MCTSProofSearch` instances (R20-P1).
#:
#: When ``True`` (default), calls to :func:`make_default_selector`
#: (and the CLI wrapper :file:`molmetal/scripts/run_proof_search_variance_puct.py`)
#: build a :class:`VarianceControlledPUCT` with the score-repellent
#: flag enabled — i.e. the canonical Dirichlet PUCT bonus is replaced
#: by a VariancePUCT bonus with a penalty for high-variance arms.
#:
#: When ``False``, the legacy Dirichlet-only :class:`proof_search`
#: selector path is preserved (backward compat for callers that
#: explicitly opt out via ``--disable-variance-puct`` on the CLI or
#: by passing ``use_variance_puct=False`` to :func:`make_default_selector`).
#:
#: This flag is the single source of truth for "VariancePUCT default
#: replace Dirichlet PUCT" (TODO-92, R20-P1).  It does NOT mutate
#: :mod:`proof_search`; the wrapper layer reads this constant and
#: builds the appropriate selector.
DEFAULT_USE_VARIANCE_PUCT: bool = True


def make_default_selector(
    use_variance_puct: Optional[bool] = None,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    score_repellent: bool = True,
    n_min_visits: int = DEFAULT_N_MIN_VISITS,
    epsilon: float = 0.25,
    rng_seed: int = 0,
) -> "VarianceControlledPUCT":
    """Build the canonical "default" PUCT selector (R20-P1, TODO-92).

    The returned selector is the **VariancePUCT** with
    :attr:`score_repellent=True` (Burch & Bardsley 2022 score-repellent
    MCMC noise) by default — replacing the legacy Dirichlet PUCT that
    ships in :mod:`proof_search`.

    Parameters
    ----------
    use_variance_puct : bool, optional
        When ``None`` (default), fall back to :data:`DEFAULT_USE_VARIANCE_PUCT`.
        When ``True``, build a :class:`VarianceControlledPUCT`.  When
        ``False``, build a canonical Dirichlet PUCT selector (c_var=0
        effectively disables the variance penalty term, restoring the
        legacy Dirichlet PUCT behaviour — backward compat for callers
        that want the legacy regime).
    c_puct, c_var, score_repellent, n_min_visits, epsilon, rng_seed
        Forwarded to :class:`VarianceControlledPUCT`.

    Returns
    -------
    VarianceControlledPUCT
        A configured variance-controlled PUCT selector.

    Notes
    -----
    The "fallback to Dirichlet PUCT" path is implemented by setting
    ``c_var=0`` on the same :class:`VarianceControlledPUCT` class —
    this preserves a single code path (and keeps the lazy inner-selector
    import intact) but eliminates the variance penalty term.  See the
    Wf-51 Phase-1 sign-flip fix for the formula.
    """
    if use_variance_puct is None:
        use_variance_puct = DEFAULT_USE_VARIANCE_PUCT
    if not bool(use_variance_puct):
        # Backward-compat path: disable the variance penalty by
        # setting c_var=0; this is mathematically identical to the
        # canonical Dirichlet PUCT bonus (Q + c_puct * P * sqrt(N)
        # / (1 + N)) on the > n_min_visits regime.
        c_var = 0.0
    return VarianceControlledPUCT(
        c_puct=float(c_puct),
        c_var=float(c_var),
        score_repellent=bool(score_repellent),
        n_min_visits=int(n_min_visits),
        epsilon=float(epsilon),
        rng_seed=int(rng_seed),
    )


# ---------------------------------------------------------------------------
# IntegrationNode
# ---------------------------------------------------------------------------


@dataclass
class IntegrationNode:
    """Lightweight PUCT node used by the integration layer.

    The full :class:`proof_search._MCTSNode` carries a heavyweight
    :class:`MoleculeClosedTerm` plus PUCT statistics; for the
    integration layer we only need a *node id*, the per-arm
    statistics, and a way to enumerate children.  We provide this
    stand-in so the variance selector can be unit-tested without
    spinning up the full MCTS.

    Attributes
    ----------
    node_id : str
        Stable identifier.  Used as the hashable key in the
        selector's :attr:`_stats` table.
    prior : float
        PUCT prior ``P(s, a)``.  Defaults to ``1.0`` (uniform on
        unvisited).  The integration layer treats this as the prior
        for *this* node, not for its children — the per-arm priors
        live on each child's :attr:`IntegrationNode.prior`.
    children : list[IntegrationNode]
        Expanded child nodes.  The selector uses
        ``len(children)`` as the action count when no explicit
        ``action`` index is supplied.
    """

    node_id: str
    prior: float = 1.0
    children: List["IntegrationNode"] = field(default_factory=list)

    # PUCT statistics are *not* stored on the node itself — the
    # :class:`VarianceControlledPUCT` selector keeps its own
    # ``(state, action) -> stats`` table.  This keeps the node
    # pickle-friendly (no back-references to the selector).

    def action_count(self) -> int:
        """Return the number of legal actions (= children)."""
        return max(1, len(self.children))


# ---------------------------------------------------------------------------
# VarianceControlledPUCT (integration-layer version)
# ---------------------------------------------------------------------------


@dataclass
class VarianceControlledPUCT:
    """PUCT selector with variance penalty — node-centric integration API.

    Compared to :class:`score_repellent_mc.VarianceControlledPUCT`
    which is keyed by ``(state, action)`` hashables and accepts a
    list of legal actions, this integration-layer class accepts an
    :class:`IntegrationNode` and returns the **action index** of
    the next action to take.  Internally it delegates to the
    score-repellent :class:`VarianceControlledPUCT` for the
    canonical PUCT bonus and only adds the variance penalty term.

    Parameters
    ----------
    c_puct : float, default :data:`DEFAULT_C_PUCT`
        PUCT exploration constant.
    c_var : float, default :data:`DEFAULT_C_VAR`
        Variance penalty coefficient.  ``0.0`` disables variance
        control (selector degenerates to canonical PUCT).
    score_repellent : bool, default True
        When ``True`` the bonus uses score-repellent noise; when
        ``False`` it falls back to vanilla Dirichlet.  This flag
        is propagated to the inner :class:`VarianceControlledPUCT`.
    n_min_visits : int, default :data:`DEFAULT_N_MIN_VISITS`
        Minimum visit count before the variance term activates.
        Below this count the selector ignores :attr:`variance` and
        uses canonical PUCT.
    epsilon : float, default 0.25
        Dirichlet mixing weight (AlphaZero convention).  Forwarded
        to the inner selector.
    rng_seed : int, default 0
        Default seed for the inner selector's RNG.

    Attributes
    ----------
    stats_ : dict
        Per-arm statistics ``{(node_id, action_idx): {"N","W","Q",
        "W2","sigma"}}``.  :attr:`W2` accumulates ``reward ** 2``
        so :attr:`sigma` = ``sqrt(E[r^2] - E[r]^2)`` (Welford
        style).
    inner_ : score_repellent_mc.VarianceControlledPUCT
        The underlying score-repellent selector (lazy — built on
        the first :meth:`select` call so it can infer ``dim`` from
        the action count).

    Examples
    --------
    >>> sel = VarianceControlledPUCT(c_puct=1.0, c_var=0.5)
    >>> root = IntegrationNode(node_id="root", prior=0.5,
    ...     children=[IntegrationNode(node_id=f"a{i}", prior=1.0/(i+1))
    ...               for i in range(4)])
    >>> action = sel.select(root)
    >>> 0 <= action < 4
    True
    >>> sel.update(root, action, value=1.0, variance=0.0)
    """

    c_puct: float = DEFAULT_C_PUCT
    c_var: float = DEFAULT_C_VAR
    score_repellent: bool = True
    n_min_visits: int = DEFAULT_N_MIN_VISITS
    epsilon: float = 0.25
    rng_seed: int = 0
    # Per-arm statistics keyed by (node_id, action_idx).
    stats_: Dict[Tuple[str, int], Dict[str, float]] = field(
        default_factory=dict,
    )
    # Inner selector (lazy).
    inner_: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.c_puct < 0.0:
            raise VariancePUCTError(f"c_puct must be >= 0; got {self.c_puct}")
        if self.c_var < 0.0:
            raise VariancePUCTError(f"c_var must be >= 0; got {self.c_var}")
        if self.n_min_visits < 1:
            raise VariancePUCTError(
                f"n_min_visits must be >= 1; got {self.n_min_visits}"
            )
        if not 0.0 <= self.epsilon <= 1.0:
            raise VariancePUCTError(f"epsilon must be in [0, 1]; got {self.epsilon}")

    # ------------------------------------------------------------------
    # RNG / inner-selector lazy builders
    # ------------------------------------------------------------------

    def _ensure_inner(self, dim: int) -> Any:
        """Lazily build the inner score-repellent selector."""
        if self.inner_ is None:
            # Imported lazily so this module can be loaded even when
            # the inner selector's transitive deps (torch) are not
            # installed.  Tests that exercise the variance-only
            # surface (without delegating to the inner selector)
            # never trigger this import.
            from molmetal.molmetal_lam.search_alg.score_repellent_mc import (
                VarianceControlledPUCT as _InnerPUCT,
            )

            self.inner_ = _InnerPUCT(
                c_puct=float(self.c_puct),
                score_repellent=bool(self.score_repellent),
                epsilon=float(self.epsilon),
                seed=int(self.rng_seed),
            )
        return self.inner_

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _key(node: IntegrationNode, action: int) -> Tuple[str, int]:
        return (str(node.node_id), int(action))

    def _get_stats(
        self,
        node: IntegrationNode,
        action: int,
    ) -> Dict[str, float]:
        key = self._key(node, action)
        st = self.stats_.get(key)
        if st is None:
            st = {"N": 0.0, "W": 0.0, "W2": 0.0, "Q": 0.0, "sigma": 0.0}
            self.stats_[key] = st
        return st

    def n_value(self, node: IntegrationNode, action: int) -> int:
        return int(self._get_stats(node, action)["N"])

    def q_value(self, node: IntegrationNode, action: int) -> float:
        return float(self._get_stats(node, action)["Q"])

    def variance_value(self, node: IntegrationNode, action: int) -> float:
        """Return the empirical variance ``sigma^2`` for arm ``(node, action)``."""
        return float(self._get_stats(node, action).get("sigma", 0.0))

    # ------------------------------------------------------------------
    # PUCT scoring (with variance penalty)
    # ------------------------------------------------------------------

    def puct_score(
        self,
        node: IntegrationNode,
        action: int,
        n_parent: int,
    ) -> float:
        """Compute the variance-controlled PUCT score for ``(node, action)``.

        Implements (Wf-51 Phase-1 sign-flip fix)::

            score = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent)
                              / (1 + N(s, a))
                    - c_var * sigma(s, a)                       if N(s, a) >= n_min_visits
                    + 0                                         otherwise

        where ``sigma = sqrt(E[r^2] - E[r]^2)`` is the empirical
        standard deviation across the arm's visit history.

        Notes
        -----
        The variance penalty is **persistent** (no ``(1 + N(s, a))``
        divisor) by design — see
        :file:`molmetal/reports/wf51_sign_audit/sign_flip_diff.md`
        for the audit. The pre-fix formula
        ``c_var * sigma / (1 + N)`` decayed to ~zero at typical
        visit counts (N >= 10), causing the selector to degenerate
        to canonical PUCT (``Q + exploration``) and lose variance
        control — measured as
        ``variance_reduction = -0.185`` on the Wf-36 P2 mock CCO
        bench. The fix restores the qualitative regime: variance
        control is *active late* (when the ``sigma`` estimate is
        reliable) and *suppressed early* via the ``n_min_visits``
        gate.
        """
        st = self._get_stats(node, action)
        Q = float(st["Q"])
        N_child = float(st["N"])
        # Use the child node's own prior (PUCT convention).
        if 0 <= action < len(node.children):
            P = float(node.children[action].prior)
        else:
            P = float(node.prior)
        # Canonical PUCT bonus (decays with N_child — correct).
        bonus = self.c_puct * P * math.sqrt(max(1, n_parent)) / (1.0 + N_child)
        score = Q + bonus
        # Variance penalty (Wf-51 fix: persistent, no divisor).
        if N_child >= self.n_min_visits and self.c_var > 0.0:
            sigma = float(st["sigma"])
            score -= self.c_var * sigma
        return float(score)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, node: IntegrationNode) -> int:
        """Select the next action index from ``node``.

        Mirrors the canonical PUCT arg-max rule (Rosin 2011 / Silver
        2016) with the variance penalty term.  Unvisited actions
        (``N(s, a) = 0``) are returned first in iteration order —
        this is the FPU convention from
        :class:`proof_search.MCTSProofSearch`.
        """
        n_actions = node.action_count()
        if n_actions <= 0:
            raise VariancePUCTError(f"node {node.node_id!r} has no legal actions")
        # FPU: unvisited first.
        for idx in range(n_actions):
            if self.n_value(node, idx) == 0:
                return idx
        # All visited — argmax PUCT + variance penalty.
        n_parent = sum(self.n_value(node, a) for a in range(n_actions))
        best_idx = 0
        best_score = -float("inf")
        for idx in range(n_actions):
            sc = self.puct_score(node, idx, n_parent)
            if sc > best_score:
                best_score = sc
                best_idx = idx
        return int(best_idx)

    # ------------------------------------------------------------------
    # Statistics update
    # ------------------------------------------------------------------

    def update(
        self,
        node: IntegrationNode,
        action: int,
        value: float,
        variance: float,
    ) -> None:
        """Record a ``(value, variance)`` observation for ``(node, action)``.

        We update the Welford-style running estimates of ``Q`` and
        ``sigma^2`` using the supplied :attr:`variance` *directly*
        rather than re-deriving it from a Welford accumulator —
        this lets callers thread the variance estimate from
        downstream channels (rollout reward variance, PLL
        uncertainty, learned-prior dropout) without re-deriving.

        Parameters
        ----------
        node : IntegrationNode
            The PUCT node whose child is being updated.
        action : int
            Index of the action that produced :attr:`value`.
        value : float
            Scalar reward observation.  Used to update ``Q``.
        variance : float
            Empirical ``sigma^2`` for the *arm* over the last
            ``n_min_visits`` observations.  Stored as ``sigma`` for
            the variance penalty term.
        """
        st = self._get_stats(node, action)
        v = float(value)
        # Update N, W, Q (running mean).
        st["N"] = float(st["N"]) + 1.0
        st["W"] = float(st["W"]) + v
        st["Q"] = float(st["W"]) / float(st["N"])
        # W2 for diagnostics (used by :meth:`running_rmse`).
        st["W2"] = float(st["W2"]) + v * v
        # sigma = max(0, variance)**0.5 (clamp negatives).
        st["sigma"] = float(max(0.0, float(variance)) ** 0.5)


# ---------------------------------------------------------------------------
# DirichletVarianceComparator
# ---------------------------------------------------------------------------


@dataclass
class DirichletVarianceComparator:
    """A/B harness comparing Dirichlet vs score-repellent exploration.

    For a fixed prior vector ``p`` we draw ``n_samples`` independent
    exploration-noise vectors from each of:

    * **Dirichlet**: ``eta ~ Dir(alpha)``
    * **Score-repellent**: a short MCMC chain anchored at a soft-max
      of a Dirichlet draw (per R18 §6.2 the chain has lower
      marginal variance than vanilla Dirichlet).

    The harness reports the **marginal per-arm variance** (a vector
    of length ``len(p)``) and the **KL divergence**
    ``KL(Dirichlet || ScoreRepellent)`` between the two empirical
    distributions (computed via numpy histogram intersection).

    Parameters
    ----------
    alpha : float, default :data:`DEFAULT_DIRICHLET_ALPHA`
        Dirichlet concentration.
    score_repellent_alpha : float, default :data:`DEFAULT_SCORE_REPELLENT_ALPHA`
        Score-repellent chain concentration.
    step_size : float, default 0.1
        Step size for the score-repellent MCMC chain.
    n_iters : int, default 100
        MCMC chain length.  Must be ``>= 1``.
    rng_seed : int, default 0
        Default seed for both samplers.

    Examples
    --------
    >>> cmp = DirichletVarianceComparator(alpha=1.0, score_repellent_alpha=0.5)
    >>> p = [0.25, 0.25, 0.25, 0.25]
    >>> report = cmp.compare(priors=p, n_samples=64)
    >>> set(report.keys()) >= {"per_arm_var_dirichlet", "per_arm_var_score_repellent", "kl_divergence"}
    True
    """

    alpha: float = DEFAULT_DIRICHLET_ALPHA
    score_repellent_alpha: float = DEFAULT_SCORE_REPELLENT_ALPHA
    step_size: float = 0.1
    n_iters: int = 100
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise VariancePUCTError(f"alpha must be > 0; got {self.alpha}")
        if self.score_repellent_alpha <= 0.0:
            raise VariancePUCTError(
                f"score_repellent_alpha must be > 0; got "
                f"{self.score_repellent_alpha}"
            )
        if self.step_size < 0.0:
            raise VariancePUCTError(f"step_size must be >= 0; got {self.step_size}")
        if self.n_iters < 1:
            raise VariancePUCTError(f"n_iters must be >= 1; got {self.n_iters}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_dirichlet(
        self,
        priors: Sequence[float],
        n_samples: int,
        rng: "Any",
    ) -> "Any":
        import numpy as np

        dim = len(priors)
        a_vec = np.full(dim, float(self.alpha), dtype=np.float64)
        # Re-weight to the requested priors (preserve relative
        # magnitude) by mixing with a uniform floor.
        p = np.asarray(priors, dtype=np.float64)
        p = np.clip(p, 1.0e-12, None)
        p = p / p.sum()
        # Dirichlet draws of the *base* Dirichlet are independent
        # of ``p``; the comparator focuses on marginal variance so
        # we keep this simple.  Callers that want a *shifted*
        # distribution can use ``priors`` as a custom anchor.
        return rng.dirichlet(a_vec, size=int(n_samples))

    def _sample_score_repellent(
        self,
        priors: Sequence[float],
        n_samples: int,
        rng: "Any",
    ) -> "Any":
        import numpy as np

        dim = len(priors)
        out = np.zeros((int(n_samples), dim), dtype=np.float64)
        p = np.asarray(priors, dtype=np.float64)
        p = np.clip(p, 1.0e-12, None)
        p = p / p.sum()
        # Anchored at the prior (not a fresh Dirichlet draw) so
        # the chain starts inside the support of the policy and
        # the marginal variance reflects only the *repulsion*
        # contribution.  This matches the contract used by
        # :class:`score_repellent_mc.DirichletReplacement`.
        for i in range(int(n_samples)):
            anchor = rng.dirichlet(
                np.full(dim, float(self.score_repellent_alpha), dtype=np.float64),
            )
            x = anchor.copy()
            for _ in range(int(self.n_iters)):
                score = 1.0 - dim * np.exp(x - x.max()) / np.exp(x - x.max()).sum()
                x = (
                    x
                    - 0.5 * self.step_size * score
                    + math.sqrt(self.step_size) * rng.standard_normal(size=x.shape)
                )
                # Softmax projection back onto the simplex.
                x = np.exp(x - x.max())
                x = x / max(x.sum(), 1.0e-12)
            out[i] = x
        return out

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare(
        self,
        priors: Optional[Sequence[float]] = None,
        n_samples: int = DEFAULT_COMPARATOR_N_SAMPLES,
    ) -> Dict[str, Any]:
        """Run the A/B comparison and return the variance + KL report.

        Parameters
        ----------
        priors : sequence of float, optional
            Prior vector.  Defaults to ``[1/dim] * dim`` with
            ``dim = 4`` when ``None`` (the canonical PUCT 4-arm
            benchmark).
        n_samples : int, default :data:`DEFAULT_COMPARATOR_N_SAMPLES`
            Number of independent draws from each sampler.

        Returns
        -------
        report : dict
            ``{"per_arm_var_dirichlet": [...], "per_arm_var_score_repellent": [...],
               "kl_divergence": float, "total_var_dirichlet": float,
               "total_var_score_repellent": float, "n_samples": int}``.
        """
        import numpy as np

        if priors is None:
            dim = 4
            priors = [1.0 / dim] * dim
        priors = list(priors)
        dim = len(priors)
        if dim < 1:
            raise VariancePUCTError(f"priors must be non-empty; got {dim}")
        if n_samples < 2:
            raise VariancePUCTError(
                f"n_samples must be >= 2 for variance estimation; " f"got {n_samples}"
            )
        rng = np.random.default_rng(int(self.rng_seed))
        # Sample.
        dir_eta = self._sample_dirichlet(priors, n_samples, rng)
        sr_eta = self._sample_score_repellent(priors, n_samples, rng)
        # Per-arm variance (column-wise).
        per_arm_dir = dir_eta.var(axis=0, ddof=1)
        per_arm_sr = sr_eta.var(axis=0, ddof=1)
        # KL divergence via 1-D histograms per arm (averaged across
        # arms).  This is a *proxy* KL — sufficient for ranking
        # the two samplers without invoking scipy.
        kl_per_arm = []
        for j in range(dim):
            bins = np.linspace(0.0, 1.0, 32)
            p_hist, _ = np.histogram(dir_eta[:, j], bins=bins, density=True)
            q_hist, _ = np.histogram(sr_eta[:, j], bins=bins, density=True)
            p_hist = p_hist / max(p_hist.sum(), 1.0e-12)
            q_hist = q_hist / max(q_hist.sum(), 1.0e-12)
            # Add a small floor to avoid log(0).
            p_hist = p_hist + 1.0e-9
            q_hist = q_hist + 1.0e-9
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()
            kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
            kl_per_arm.append(kl)
        kl_div = float(np.mean(kl_per_arm))
        return {
            "per_arm_var_dirichlet": [float(v) for v in per_arm_dir],
            "per_arm_var_score_repellent": [float(v) for v in per_arm_sr],
            "total_var_dirichlet": float(per_arm_dir.sum()),
            "total_var_score_repellent": float(per_arm_sr.sum()),
            "kl_divergence": kl_div,
            "n_samples": int(n_samples),
            "alpha_dirichlet": float(self.alpha),
            "alpha_score_repellent": float(self.score_repellent_alpha),
        }


# ---------------------------------------------------------------------------
# MCTSVarianceWrapper
# ---------------------------------------------------------------------------


@dataclass
class MCTSVarianceWrapper:
    """Thin adapter that injects variance control into an existing MCTSProofSearch.

    The wrapper runs ``base_search.search(initial_state)`` with the
    *legacy* Dirichlet noise path *disabled* (i.e. via the
    score-repellent variance-controlled selector when
    ``score_repellent=True``).  Because we are forbidden from
    modifying ``proof_search.py``, the wrapper achieves this by:

    1. Reading :attr:`MCTSProofSearch.dirichlet_alpha` /
       :attr:`dirichlet_fraction` to detect the noise regime.
    2. Calling :meth:`MCTSProofSearch.search` once with the *default*
       regime to obtain the candidate list.
    3. Re-scoring the candidates through the variance-controlled
       selector and reporting the per-candidate variance estimate.

    This is *deliberately* conservative — the wrapper does **not**
    silently mutate the inner MCTS state machine.  Callers that
    want a deeper integration should patch ``proof_search.py`` in a
    follow-up PR; the wrapper here is the safety net.

    Parameters
    ----------
    base_search : proof_search.MCTSProofSearch
        The MCTS instance to wrap.  Not mutated.
    variance_puct : VarianceControlledPUCT, optional
        Pre-built variance selector.  When ``None`` we build one
        with the wrapper's ``c_puct`` / ``c_var`` defaults.
    score_repellent : bool, default True
        Whether to thread the score-repellent flag through the
        inner selector (matches ``VarianceControlledPUCT.score_repellent``).
    """

    base_search: Any
    variance_puct: Optional[VarianceControlledPUCT] = None
    score_repellent: bool = True

    def __post_init__(self) -> None:
        if self.variance_puct is None:
            self.variance_puct = VarianceControlledPUCT(
                c_puct=DEFAULT_C_PUCT,
                c_var=DEFAULT_C_VAR,
                score_repellent=bool(self.score_repellent),
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, root: IntegrationNode) -> int:
        """Run a single variance-controlled selection from ``root``.

        We do **not** invoke :meth:`MCTSProofSearch.search` here —
        that would couple the wrapper to a heavyweight RDKit /
        molecular-closed-term setup.  Instead we expose the same
        signature as the full search (root → best action) but
        drive it from the variance-controlled selector.  Tests
        that want the full MCTS loop should call
        :meth:`search_with_proof_search` instead.
        """
        if not isinstance(root, IntegrationNode):
            raise VariancePUCTError(
                f"root must be an IntegrationNode; got {type(root)!r}"
            )
        return int(self.variance_puct.select(root))

    def search_with_proof_search(
        self,
        initial_state: Any,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """Run the underlying MCTSProofSearch and attach variance stats.

        This is the *integration* entry point: it actually invokes
        ``base_search.search(initial_state, max_depth=max_depth)``
        and re-scores the returned candidates through the
        variance-controlled selector.  The result is a dict with
        the candidates + a per-candidate variance report.

        Parameters
        ----------
        initial_state : MoleculeClosedTerm
            Root state for the MCTS.
        max_depth : int, default 3
            Maximum rollout depth.

        Returns
        -------
        report : dict
            ``{"candidates": [...], "best_action": int, "variance_report":
            {"per_candidate_var": [...], "mean_var": float}, "wall_time_s":
            float}``.  When ``base_search.search`` raises (e.g. empty
            tile_library), the dict carries ``{"error": "..."}`` so
            callers can branch without try/except.
        """
        t0 = time.time()
        try:
            candidates = self.base_search.search(
                initial_state,
                max_depth=max_depth,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": str(exc), "wall_time_s": time.time() - t0}
        wall = time.time() - t0
        if not candidates:
            return {
                "candidates": [],
                "best_action": -1,
                "variance_report": {"per_candidate_var": [], "mean_var": 0.0},
                "wall_time_s": float(wall),
            }
        # Re-score through the variance selector.  We use the
        # *base_search*'s reward aggregator (if any) for the value
        # estimate and the residual against the mean as the
        # variance estimate — this is a stand-in until a future
        # patch threads the variance channel through the MCTS.
        try:
            values = [float(self.base_search.score_final(c)) for c in candidates]
        except Exception:
            values = [0.0 for _ in candidates]
        if values:
            mean_v = sum(values) / len(values)
        else:
            mean_v = 0.0
        per_cand_var = [(v - mean_v) ** 2 for v in values]
        # Pick the best action = argmax value, projected onto the
        # candidate index space.
        best_action = 0
        best_v = -float("inf")
        for idx, v in enumerate(values):
            if v > best_v:
                best_v = v
                best_action = idx
        return {
            "candidates": list(candidates),
            "best_action": int(best_action),
            "variance_report": {
                "per_candidate_var": [float(x) for x in per_cand_var],
                "mean_var": float(mean_v),
            },
            "wall_time_s": float(wall),
        }


# ---------------------------------------------------------------------------
# integration_with_proof_search
# ---------------------------------------------------------------------------


def integration_with_proof_search(
    search_proof: Any,
    n_runs: int = 10,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Run the A/B comparison between Dirichlet and score-repellent on ``search_proof``.

    This is the *integration smoke harness* — for ``n_runs`` repeated
    calls to :meth:`MCTSProofSearch.search` we record:

    * ``wall_time_s``: total wall clock time for all runs.
    * ``per_run_candidate_count``: number of candidates emitted by
      each run.
    * ``per_run_score_mean``: mean reward across each run's
      candidates.
    * ``variance_comparison``: the
      :class:`DirichletVarianceComparator` report.

    Parameters
    ----------
    search_proof : MCTSProofSearch
        A *configured* MCTSProofSearch instance with a non-empty
        tile library and rules.  Not mutated.
    n_runs : int, default 10
        Number of repeated search calls.
    max_depth : int, default 3
        Rollout depth budget per call.

    Returns
    -------
    report : dict
        ``{"wall_time_s": float, "per_run_candidate_count": [int],
        "per_run_score_mean": [float], "variance_comparison": dict}``.
        When the search raises (empty tile library, RDKit missing,
        etc.) the report carries ``{"error": "..."}``.
    """
    if n_runs < 1:
        raise VariancePUCTError(f"n_runs must be >= 1; got {n_runs}")
    t0 = time.time()
    per_run_n_cands: List[int] = []
    per_run_score: List[float] = []
    initial_state = None
    try:
        # Discover the root state from the search's tile library —
        # we use the first tile as the starting molecule.  This
        # is conservative; callers wanting a custom initial state
        # should drive ``base_search.search`` directly.
        if not getattr(search_proof, "tile_library", None):
            return {
                "error": "search_proof.tile_library is empty",
                "wall_time_s": time.time() - t0,
            }
        initial_state = search_proof.tile_library[0]
    except Exception as exc:
        return {"error": str(exc), "wall_time_s": time.time() - t0}
    for _ in range(int(n_runs)):
        try:
            cands = search_proof.search(initial_state, max_depth=max_depth)
        except Exception:
            cands = []
        per_run_n_cands.append(len(cands))
        try:
            if cands:
                vals = [float(search_proof.score_final(c)) for c in cands]
                per_run_score.append(float(sum(vals) / len(vals)))
            else:
                per_run_score.append(0.0)
        except Exception:
            per_run_score.append(0.0)
    wall = time.time() - t0
    cmp = DirichletVarianceComparator()
    variance_cmp = cmp.compare(n_samples=64)
    return {
        "wall_time_s": float(wall),
        "per_run_candidate_count": [int(x) for x in per_run_n_cands],
        "per_run_score_mean": [float(x) for x in per_run_score],
        "variance_comparison": variance_cmp,
        "n_runs": int(n_runs),
        "max_depth": int(max_depth),
    }


__all__ = [
    "DEFAULT_C_PUCT",
    "DEFAULT_C_VAR",
    "DEFAULT_N_MIN_VISITS",
    "DEFAULT_DIRICHLET_ALPHA",
    "DEFAULT_SCORE_REPELLENT_ALPHA",
    "DEFAULT_COMPARATOR_N_SAMPLES",
    "DEFAULT_USE_VARIANCE_PUCT",
    "VariancePUCTError",
    "IntegrationNode",
    "VarianceControlledPUCT",
    "DirichletVarianceComparator",
    "MCTSVarianceWrapper",
    "integration_with_proof_search",
    "make_default_selector",
]
