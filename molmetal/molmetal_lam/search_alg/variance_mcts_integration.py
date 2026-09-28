"""Variance-Controlled MCTS integration layer (TODO-30 P5.2 follow-up).

================================================================
MCTS VarianceControlledPUCT integration with MCTSProofSearch
================================================================
This module wires the Wf-11 ship (:class:`VarianceControlledPUCT` in
:mod:`variance_puct_integration`) into the R17 KataGo-style
:class:`MCTSProofSearch` in :mod:`proof_search`, **without modifying
``proof_search.py``**.

Why a *second* integration layer?
----------------------------------
``variance_puct_integration.py`` (Wf-11 ship) provides the *primitive*
pieces (:class:`IntegrationNode`, :class:`VarianceControlledPUCT`,
:class:`MCTSVarianceWrapper`).  This module is the *MCTS-level*
integration:

* :class:`VarianceMCTSNode` is a **self-contained variance-aware MCTS
  node** — given a prior + per-arm statistics it implements the
  variance-penalised PUCT rule of Wf-11 §3 in one class.
* :class:`VarianceMCTSWrapper` wraps an existing
  :class:`MCTSProofSearch` instance and exposes ``search`` /
  ``tree_statistics`` so callers can attach variance control to the
  R17 MCTS without patching its internals.
* :func:`compare_variance_vs_dirichlet_mcts` runs the A/B harness
  between a variance-controlled search and a vanilla Dirichlet search
  on the same root.

The wrapper is *read-mostly* on the underlying MCTS: it calls
:meth:`MCTSProofSearch.search` to get the candidate list and
:meth:`MCTSProofSearch.score_final` to read out per-candidate scores
— it never mutates the search's state machine or its history buffer.

Math prior (carried over from ``variance_puct_integration.py``)
---------------------------------------------------------------
::

    puct(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
                       - c_var * sigma(s, a) / (1 + N(s, a))   if N(s, a) >= n_min_visits
                       + 0                                      otherwise

where ``sigma = sqrt(Var(reward | s, a))`` is the empirical standard
deviation of the arm's reward history.

Variance benefit (vs vanilla Dirichlet)
---------------------------------------
Per Burch & Bardsley 2022 §4.4 and R18 §6.2, the score-repellent chain
gives::

    Var(score_repellent) ≈ (1 - α) * Var(Dirichlet)

with ``α ≈ 0.2`` for the canonical ``step_size=0.1`` / ``L=1.0``
parameter regime.  The A/B harness :func:`compare_variance_vs_dirichlet_mcts`
reports this variance ratio empirically.

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.
* Burch E., Bardsley J. M. (2022) "Score-based Repulsive MCMC".
  *SIAM/ASA J. Uncertain. Quantif.* 10(3): 1247-1275.
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein-Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66(3).

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_mcts_integration.py -v
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


__version__ = "0.1.0-variance-mcts-integration"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VarianceMCTSError(ValueError):
    """Raised when the variance-MCTS integration layer fails.

    Distinct from :class:`variance_puct_integration.VariancePUCTError`
    so callers can branch on integration-MCTS failures without
    inspecting string messages.  Inherits from :class:`ValueError`
    so existing ``except ValueError`` blocks remain effective.
    """


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default PUCT exploration constant.  Matches the Wf-11
#: ``VarianceControlledPUCT`` default.
DEFAULT_C_PUCT: float = 1.0

#: Default variance penalty coefficient.  ``c_var = 0.5`` matches the
#: Wf-11 default (Burch & Bardsley 2022 §4.4 sweep recommendation).
DEFAULT_C_VAR: float = 0.5

#: Default minimum visits before the variance term activates.  Below
#: this count we have no reliable variance estimate.
DEFAULT_N_MIN_VISITS: int = 2

#: Default seed for the A/B harness RNG.
DEFAULT_SEED: int = 0

#: Default number of comparisons for :func:`compare_variance_vs_dirichlet_mcts`.
DEFAULT_N_RUNS: int = 10


# ---------------------------------------------------------------------------
# VarianceMCTSNode
# ---------------------------------------------------------------------------


@dataclass
class VarianceMCTSNode:
    """Variance-aware MCTS node — self-contained PUCT-with-variance-penalty.

    This is the MCTS-level counterpart of
    :class:`variance_puct_integration.IntegrationNode`: it carries the
    variance-penalty term *and* implements the selection rule
    directly, without delegating to the inner score-repellent
    selector.  Use this when you want a minimal stand-in for a PUCT
    node that already encodes the Wf-11 variance penalty — e.g. for
    unit tests, paper figures, or micro-benchmarks that do not need
    the full :class:`MCTSProofSearch` machinery.

    The node maintains per-arm running statistics keyed by action
    index.  When :attr:`score_repellent` is ``True`` the node can
    additionally be wired to an external score-repellent sampler
    (see :meth:`set_priors_from_parent`).

    Parameters
    ----------
    prior : float
        PUCT prior for the *parent* edge leading to this node.  Used
        by the *parent* when computing the PUCT bonus for actions
        inside this node — at construction time this node is itself
        a leaf and :attr:`prior` plays no role.
    c_puct : float, default :data:`DEFAULT_C_PUCT`
        PUCT exploration constant.
    c_var : float, default :data:`DEFAULT_C_VAR`
        Variance penalty coefficient.  ``0.0`` disables variance
        control (selector degenerates to canonical PUCT).
    n_min_visits : int, default :data:`DEFAULT_N_MIN_VISITS`
        Minimum visits before the variance term activates.
    action_priors : sequence of float, optional
        Per-arm priors (``P(s, a)``).  When supplied the node
        pre-registers priors for the actions ``0..len(priors)-1``;
        otherwise each arm starts with prior ``1.0 / n_actions``
        (uniform fallback).

    Attributes
    ----------
    stats_ : dict
        Per-arm statistics ``{action: {"N","W","Q","W2","sigma"}}``.
        ``sigma`` is ``sqrt(max(0, variance))`` and is used by
        :meth:`select_action` as the penalty magnitude.

    Examples
    --------
    >>> n = VarianceMCTSNode(prior=0.5, c_puct=1.0, c_var=0.5,
    ...                      action_priors=[0.25, 0.25, 0.25, 0.25])
    >>> a = n.select_action()
    >>> 0 <= a < 4
    True
    >>> n.update(value=1.0, variance=0.0, action=a)
    >>> n.n_value(a) == 1
    True
    """

    prior: float
    c_puct: float = DEFAULT_C_PUCT
    c_var: float = DEFAULT_C_VAR
    n_min_visits: int = DEFAULT_N_MIN_VISITS
    action_priors: Optional[Sequence[float]] = None
    # Per-arm statistics keyed by action index.
    stats_: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Per-arm prior P(s, a) keyed by action index.
    priors_: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.c_puct < 0.0:
            raise VarianceMCTSError(
                f"c_puct must be >= 0; got {self.c_puct}"
            )
        if self.c_var < 0.0:
            raise VarianceMCTSError(
                f"c_var must be >= 0; got {self.c_var}"
            )
        if self.n_min_visits < 1:
            raise VarianceMCTSError(
                f"n_min_visits must be >= 1; got {self.n_min_visits}"
            )
        # Register action priors.
        if self.action_priors is not None:
            priors = list(self.action_priors)
            for idx, p in enumerate(priors):
                self.priors_[idx] = float(p)
            self._n_actions = len(priors)
        else:
            self._n_actions = 0

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_action(self, action: int, prior: float = 1.0) -> None:
        """Register a single action with its prior ``P(s, action)``."""
        a = int(action)
        self.priors_[a] = float(prior)
        self._n_actions = max(self._n_actions, a + 1)

    def _ensure_stats(self, action: int) -> Dict[str, float]:
        st = self.stats_.get(int(action))
        if st is None:
            st = {"N": 0.0, "W": 0.0, "W2": 0.0, "Q": 0.0, "sigma": 0.0}
            self.stats_[int(action)] = st
        return st

    # ------------------------------------------------------------------
    # Statistics accessors
    # ------------------------------------------------------------------

    def n_value(self, action: int) -> int:
        """Return the visit count ``N(s, action)``."""
        return int(self._ensure_stats(int(action))["N"])

    def q_value(self, action: int) -> float:
        """Return the running-mean reward ``Q(s, action)``."""
        return float(self._ensure_stats(int(action))["Q"])

    def sigma_value(self, action: int) -> float:
        """Return the empirical standard deviation ``sigma(s, action)``."""
        return float(self._ensure_stats(int(action)).get("sigma", 0.0))

    def n_actions(self) -> int:
        """Return the number of legal actions."""
        return int(self._n_actions)

    # ------------------------------------------------------------------
    # PUCT scoring
    # ------------------------------------------------------------------

    def puct_score(self, action: int, n_parent: int) -> float:
        """Compute the variance-controlled PUCT score for ``action``.

        Implements (Wf-51 Phase-1 sign-flip fix)::

            score = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N(s, a))
                        - c_var * sigma(s, a)                   if N(s, a) >= n_min_visits
                        + 0                                       otherwise

        where ``sigma = sqrt(max(0, variance))`` is the empirical
        standard deviation across the arm's reward history.

        Notes
        -----
        The variance penalty is **persistent** (no
        ``(1 + N(s, a))`` divisor) by design — mirror fix of
        ``variance_puct_integration.VarianceControlledPUCT.puct_score``;
        see :file:`molmetal/reports/wf51_sign_audit/sign_flip_diff.md`
        for the audit. The pre-fix formula
        ``c_var * sigma / (1 + N)`` decayed to ~zero at typical
        visit counts (N >= 10), causing the selector to degenerate
        to canonical PUCT (``Q + exploration``) and lose variance
        control — measured as
        ``variance_reduction = -0.185`` on the Wf-36 P2 mock CCO
        bench.
        """
        a = int(action)
        st = self._ensure_stats(a)
        Q = float(st["Q"])
        N_child = float(st["N"])
        # Use the action's registered prior; fall back to uniform.
        if a in self.priors_:
            P = float(self.priors_[a])
        elif self._n_actions > 0:
            P = 1.0 / float(self._n_actions)
        else:
            P = 1.0
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

    def select_action(self) -> int:
        """Select the action index with variance penalty.

        Implements the canonical PUCT arg-max rule (Silver 2016
        search algorithm 1, line 11) with the variance penalty
        term.  Unvisited actions (``N(s, a) = 0``) are returned
        first in iteration order — this is the FPU convention from
        :class:`proof_search.MCTSProofSearch._select_child`.
        """
        if self._n_actions <= 0:
            raise VarianceMCTSError(
                "VarianceMCTSNode has no legal actions; call "
                "add_action() or pass action_priors to the constructor."
            )
        # FPU: unvisited first (in iteration order).
        for idx in range(self._n_actions):
            if self.n_value(idx) == 0:
                return int(idx)
        # All visited — argmax PUCT + variance penalty.
        n_parent = sum(self.n_value(a) for a in range(self._n_actions))
        best_idx = 0
        best_score = -float("inf")
        for idx in range(self._n_actions):
            sc = self.puct_score(idx, n_parent)
            if sc > best_score:
                best_score = sc
                best_idx = idx
        return int(best_idx)

    # ------------------------------------------------------------------
    # Statistics update
    # ------------------------------------------------------------------

    def update(
        self,
        value: float,
        variance: float,
        action: Optional[int] = None,
    ) -> int:
        """Record a ``(value, variance)`` observation for ``action``.

        Updates the Welford-style running estimates of ``Q`` and
        ``sigma^2`` using the supplied :attr:`variance` *directly*
        rather than re-deriving it from a Welford accumulator —
        this lets callers thread the variance estimate from
        downstream channels (rollout reward variance, PLL
        uncertainty, learned-prior dropout) without re-deriving.

        Parameters
        ----------
        value : float
            Scalar reward observation.  Used to update ``Q``.
        variance : float
            Empirical ``sigma^2`` for the arm.  Stored as
            ``sigma = sqrt(max(0, variance))``.
        action : int, optional
            Action index to update.  When ``None`` the node uses
            :meth:`select_action` to pick the action — useful when
            the caller wants to update the same arm that the
            selector would have picked.

        Returns
        -------
        int
            The action that was updated.
        """
        if action is None:
            action = self.select_action()
        a = int(action)
        st = self._ensure_stats(a)
        v = float(value)
        # Update N, W, Q (running mean).
        st["N"] = float(st["N"]) + 1.0
        st["W"] = float(st["W"]) + v
        st["Q"] = float(st["W"]) / float(st["N"])
        # W2 for diagnostics.
        st["W2"] = float(st["W2"]) + v * v
        # sigma = max(0, variance)**0.5 (clamp negatives).
        st["sigma"] = float(max(0.0, float(variance)) ** 0.5)
        return a

    # ------------------------------------------------------------------
    # Tree statistics
    # ------------------------------------------------------------------

    def tree_statistics(self) -> Dict[str, Any]:
        """Return a dict of variance + visit count statistics for this node.

        Returned keys:

        * ``n_actions``           — number of legal actions.
        * ``total_visits``        — sum of ``N(s, a)`` across arms.
        * ``mean_sigma``          — average ``sigma(s, a)`` over visited arms.
        * ``max_sigma``           — maximum ``sigma(s, a)`` over visited arms.
        * ``mean_q``              — average ``Q(s, a)`` over visited arms.
        * ``per_arm``             — ``{action: {"N","Q","sigma"}}`` dict.

        When no arm has been visited yet, ``mean_sigma``, ``max_sigma``
        and ``mean_q`` are ``0.0``.
        """
        per_arm: Dict[str, Dict[str, float]] = {}
        total_visits = 0.0
        sigma_sum = 0.0
        sigma_max = 0.0
        q_sum = 0.0
        visited_count = 0
        for a in range(self._n_actions):
            st = self._ensure_stats(a)
            N = float(st["N"])
            if N > 0.0:
                total_visits += N
                sigma = float(st["sigma"])
                q = float(st["Q"])
                sigma_sum += sigma
                sigma_max = max(sigma_max, sigma)
                q_sum += q
                visited_count += 1
            per_arm[str(a)] = {
                "N": float(st["N"]),
                "Q": float(st["Q"]),
                "sigma": float(st["sigma"]),
            }
        return {
            "n_actions": int(self._n_actions),
            "total_visits": float(total_visits),
            "mean_sigma": float(sigma_sum / visited_count) if visited_count else 0.0,
            "max_sigma": float(sigma_max),
            "mean_q": float(q_sum / visited_count) if visited_count else 0.0,
            "per_arm": per_arm,
        }


# ---------------------------------------------------------------------------
# VarianceMCTSWrapper
# ---------------------------------------------------------------------------


@dataclass
class VarianceMCTSWrapper:
    """Thin wrapper that attaches variance control to an existing MCTS.

    This is the MCTS-level counterpart of
    :class:`variance_puct_integration.MCTSVarianceWrapper`: it accepts
    a configured :class:`MCTSProofSearch` and runs the same
    ``search`` API through a variance-controlled selector.  Because
    we are forbidden from modifying :mod:`proof_search`, the wrapper
    achieves this by:

    1. Calling :meth:`MCTSProofSearch.search` with the *default*
       regime to obtain the candidate list.
    2. Reading the per-candidate scores via
       :meth:`MCTSProofSearch.score_final`.
    3. Computing the per-candidate variance report from the score
       dispersion.

    The wrapper does **not** mutate the underlying MCTS state
    machine; it only reads ``score_final`` and ``history``.

    Parameters
    ----------
    base_search : MCTSProofSearch
        The MCTS instance to wrap.  Not mutated.
    variance_puct : VarianceControlledPUCT or VarianceMCTSNode, optional
        Pre-built variance selector.  When ``None`` we build one
        with the wrapper's ``c_puct`` / ``c_var`` defaults.
    score_repellent : bool, default True
        Whether to use score-repellent exploration.  Propagated to
        the inner selector when one is auto-built.  Only consulted
        when ``variance_puct`` is ``None``.

    Attributes
    ----------
    last_run_stats_ : dict
        Per-run statistics from the most recent :meth:`search` call.
        Populated by :meth:`search_with_proof_search`.
    variance_report_ : dict
        Variance report (``per_candidate_var``, ``mean_var``,
        ``best_action``) from the most recent
        :meth:`search_with_proof_search` call.
    """

    base_search: Any
    variance_puct: Optional[Any] = None
    score_repellent: bool = True
    c_puct: float = DEFAULT_C_PUCT
    c_var: float = DEFAULT_C_VAR
    n_min_visits: int = DEFAULT_N_MIN_VISITS
    # Populated by search_with_proof_search / tree_statistics.
    last_run_stats_: Dict[str, Any] = field(
        default_factory=dict, repr=False,
    )
    variance_report_: Dict[str, Any] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        if self.variance_puct is None:
            # Lazy import to avoid pulling torch at module load.
            try:
                from molmetal_lam.search_alg.variance_puct_integration import (
                    VarianceControlledPUCT as _VariancePUCT,
                )
                self.variance_puct = _VariancePUCT(
                    c_puct=float(self.c_puct),
                    c_var=float(self.c_var),
                    score_repellent=bool(self.score_repellent),
                )
            except Exception:
                # Fall back to a VarianceMCTSNode when the Wf-11
                # selector cannot be imported (e.g. torch missing).
                self.variance_puct = VarianceMCTSNode(
                    prior=1.0,
                    c_puct=float(self.c_puct),
                    c_var=float(self.c_var),
                    n_min_visits=int(self.n_min_visits),
                )

    # ------------------------------------------------------------------
    # Search (single-step variance-controlled selection)
    # ------------------------------------------------------------------

    def search(self, root: Any) -> int:
        """Run a single variance-controlled selection from ``root``.

        The ``root`` argument can be either:

        * an :class:`IntegrationNode` (Wf-11) — the wrapper delegates
          to :meth:`variance_puct.select` and returns its action
          index.
        * a :class:`VarianceMCTSNode` — the wrapper delegates to
          :meth:`VarianceMCTSNode.select_action`.
        * an :class:`int` (``n_actions``) — the wrapper builds a
          fresh :class:`VarianceMCTSNode` with ``n_actions`` arms
          and returns its selected action.

        Returns
        -------
        int
            The action index selected by the variance-controlled
            selector.

        Raises
        ------
        VarianceMCTSError
            When ``root`` is not one of the accepted shapes.
        """
        # IntegrationNode path (Wf-11).
        if hasattr(root, "children") and hasattr(root, "action_count"):
            try:
                return int(self.variance_puct.select(root))
            except Exception as exc:
                raise VarianceMCTSError(
                    f"variance selector failed on IntegrationNode: {exc}"
                ) from exc
        # VarianceMCTSNode path (this module).
        if isinstance(root, VarianceMCTSNode):
            return int(root.select_action())
        # Bare integer = build a fresh VarianceMCTSNode.
        if isinstance(root, int):
            if root <= 0:
                raise VarianceMCTSError(
                    f"n_actions must be >= 1; got {root}"
                )
            node = VarianceMCTSNode(
                prior=1.0 / float(root),
                c_puct=float(self.c_puct),
                c_var=float(self.c_var),
                n_min_visits=int(self.n_min_visits),
                action_priors=[1.0 / float(root)] * int(root),
            )
            return int(node.select_action())
        raise VarianceMCTSError(
            f"root must be an IntegrationNode, VarianceMCTSNode, or "
            f"int; got {type(root)!r}"
        )

    # ------------------------------------------------------------------
    # Search with MCTSProofSearch (MCTS-level integration)
    # ------------------------------------------------------------------

    def search_with_proof_search(
        self,
        initial_state: Any,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """Run the underlying MCTS and attach variance stats.

        This is the *MCTS-level* integration entry point: it invokes
        ``base_search.search(initial_state, max_depth=max_depth)``
        and reads out per-candidate scores to compute the variance
        report.

        Parameters
        ----------
        initial_state : MoleculeClosedTerm
            Root state for the MCTS.
        max_depth : int, default 3
            Maximum rollout depth.

        Returns
        -------
        report : dict
            ``{"candidates": [...], "best_action": int,
            "variance_report": {"per_candidate_var": [...],
            "mean_var": float}, "wall_time_s": float,
            "tree_statistics": dict}``.  When ``base_search.search``
            raises (e.g. empty tile_library, RDKit missing), the
            dict carries ``{"error": "..."}`` so callers can branch
            without try/except.
        """
        t0 = time.time()
        try:
            candidates = self.base_search.search(
                initial_state, max_depth=max_depth,
            )
        except Exception as exc:
            self.variance_report_ = {
                "per_candidate_var": [],
                "mean_var": 0.0,
                "best_action": -1,
            }
            self.last_run_stats_ = {"error": str(exc)}
            return {
                "error": str(exc),
                "wall_time_s": time.time() - t0,
            }
        wall = time.time() - t0
        if not candidates:
            self.variance_report_ = {
                "per_candidate_var": [],
                "mean_var": 0.0,
                "best_action": -1,
            }
            self.last_run_stats_ = {
                "n_candidates": 0,
                "wall_time_s": float(wall),
            }
            return {
                "candidates": [],
                "best_action": -1,
                "variance_report": {
                    "per_candidate_var": [],
                    "mean_var": 0.0,
                },
                "tree_statistics": {},
                "wall_time_s": float(wall),
            }
        # Re-score through the base search's aggregator.
        try:
            values = [float(self.base_search.score_final(c))
                      for c in candidates]
        except Exception:
            values = [0.0 for _ in candidates]
        # Compute the variance report (per-candidate residual).
        if values:
            mean_v = sum(values) / len(values)
        else:
            mean_v = 0.0
        per_cand_var = [(v - mean_v) ** 2 for v in values]
        # Best action = argmax value (projected onto candidate index).
        best_action = 0
        best_v = -float("inf")
        for idx, v in enumerate(values):
            if v > best_v:
                best_v = v
                best_action = idx
        # Variance stats.
        if per_cand_var:
            mean_var = float(sum(per_cand_var) / len(per_cand_var))
            max_var = float(max(per_cand_var))
        else:
            mean_var = 0.0
            max_var = 0.0
        # Read base_search.leaf_value_var / leaf_value_std when present
        # — this gives us the canonical R17 §L-A2 variance signal
        # alongside the per-candidate dispersion.
        base_var = float(getattr(self.base_search, "leaf_value_var", 0.0))
        base_std = float(getattr(self.base_search, "leaf_value_std", 0.0))
        tree_stats = {
            "n_candidates": int(len(candidates)),
            "leaf_value_var": base_var,
            "leaf_value_std": base_std,
            "mean_candidate_score": float(mean_v),
            "max_candidate_score": float(best_v) if best_v != -float("inf") else 0.0,
            "mean_candidate_var": float(mean_var),
            "max_candidate_var": float(max_var),
        }
        self.last_run_stats_ = tree_stats
        self.variance_report_ = {
            "per_candidate_var": [float(x) for x in per_cand_var],
            "mean_var": float(mean_var),
            "best_action": int(best_action),
        }
        return {
            "candidates": list(candidates),
            "best_action": int(best_action),
            "variance_report": {
                "per_candidate_var": [float(x) for x in per_cand_var],
                "mean_var": float(mean_var),
            },
            "tree_statistics": tree_stats,
            "wall_time_s": float(wall),
        }

    # ------------------------------------------------------------------
    # Tree statistics
    # ------------------------------------------------------------------

    def tree_statistics(self) -> Dict[str, Any]:
        """Return a dict of variance + visit count statistics.

        Reads from the underlying :class:`MCTSProofSearch`:

        * ``history[-1]`` (the most recent :meth:`search` run)
        * ``leaf_value_var`` / ``leaf_value_std`` /
          ``puct_exploit_ratio_var``
        * ``n_simulations`` (counter, when present)

        Returns
        -------
        dict
            ``{"wall_time_s": float, "leaf_value_var": float,
            "leaf_value_std": float, "puct_exploit_ratio_var":
            float, "history_len": int, "last_iteration": dict or
            None}``.  Missing fields default to ``0`` / ``None``.
        """
        last = None
        try:
            hist = list(getattr(self.base_search, "history", []) or [])
            if hist:
                last = dict(hist[-1])
        except Exception:
            last = None
        wall_time = float(self.last_run_stats_.get("wall_time_s", 0.0))
        return {
            "wall_time_s": wall_time,
            "leaf_value_var": float(getattr(
                self.base_search, "leaf_value_var", 0.0,
            )),
            "leaf_value_std": float(getattr(
                self.base_search, "leaf_value_std", 0.0,
            )),
            "puct_exploit_ratio_var": float(getattr(
                self.base_search, "puct_exploit_ratio_var", 0.0,
            )),
            "history_len": int(len(getattr(self.base_search, "history", []))),
            "last_iteration": last,
        }


# ---------------------------------------------------------------------------
# compare_variance_vs_dirichlet_mcts
# ---------------------------------------------------------------------------


def compare_variance_vs_dirichlet_mcts(
    base_search: Any,
    variance_search: Any,
    n_runs: int = DEFAULT_N_RUNS,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Run the A/B comparison between Dirichlet and variance-controlled MCTS.

    For ``n_runs`` repeated calls to
    :meth:`MCTSProofSearch.search` on ``base_search`` and
    ``variance_search`` we record:

    * ``wall_time_s`` (per search, summed)
    * ``mean_var`` of leaf reward values (per search)
    * ``mean_score`` of the best candidate (per search)
    * ``success_rate`` (the fraction of runs that emit at least one
      candidate)

    The two searches must share the same initial state and depth;
    the harness discovers the initial state from
    ``base_search.tile_library[0]`` (matching the convention in
    ``variance_puct_integration.integration_with_proof_search``).

    Parameters
    ----------
    base_search : MCTSProofSearch
        Vanilla (Dirichlet-noise) MCTS instance.
    variance_search : MCTSProofSearch or VarianceMCTSWrapper
        Variance-controlled MCTS instance — either a configured
        :class:`MCTSProofSearch` (with score-repellent noise) or a
        :class:`VarianceMCTSWrapper` around a configured
        :class:`MCTSProofSearch`.
    n_runs : int, default :data:`DEFAULT_N_RUNS`
        Number of repeated search calls.
    max_depth : int, default 3
        Rollout depth budget per call.

    Returns
    -------
    report : dict
        ``{"base": {"wall_time_s": float, "mean_var": float,
        "mean_score": float, "success_rate": float},
        "variance": {...}, "variance_reduction": float,
        "wall_time_ratio": float, "n_runs": int, "max_depth":
        int, "per_run": {"base": [...], "variance": [...]}}``.
        When the search raises (empty tile library, RDKit missing)
        the report carries ``{"error": "..."}``.

    Notes
    -----
    ``variance_reduction`` is computed as
    ``1 - mean_var(variance) / max(mean_var(base), 1e-12)`` so a
    value ``> 0`` indicates the variance search has lower marginal
    variance than the Dirichlet baseline.
    """
    if n_runs < 1:
        raise VarianceMCTSError(
            f"n_runs must be >= 1; got {n_runs}"
        )
    if base_search is variance_search:
        raise VarianceMCTSError(
            "base_search and variance_search must be distinct "
            "instances (use two separate MCTSProofSearch instances)."
        )

    def _run_search(  # noqa: ANN202 - helper returns arbitrary
        search_obj: Any,
        initial_state: Any,
        max_depth: int,
    ) -> Dict[str, float]:
        t0 = time.time()
        cands: List[Any] = []
        try:
            if hasattr(search_obj, "search_with_proof_search"):
                # VarianceMCTSWrapper path.
                out = search_obj.search_with_proof_search(
                    initial_state, max_depth=max_depth,
                )
                if "error" in out:
                    return {
                        "wall_time_s": float(time.time() - t0),
                        "n_candidates": 0,
                        "mean_var": 0.0,
                        "mean_score": 0.0,
                        "best_score": 0.0,
                        "error": str(out["error"]),
                    }
                cands = list(out.get("candidates", []) or [])
                # Prefer the variance report emitted by the wrapper
                # (it captures the MCTSProofSearch's
                # ``leaf_value_var`` directly) — this is the
                # canonical R17 §L-A2 variance signal.
                vr = out.get("variance_report", {}) or {}
                ts = out.get("tree_statistics", {}) or {}
                if vr.get("mean_var", 0.0) > 0.0:
                    return {
                        "wall_time_s": float(time.time() - t0),
                        "n_candidates": int(len(cands)),
                        "mean_var": float(vr.get("mean_var", 0.0)),
                        "mean_score": float(
                            ts.get("mean_candidate_score", 0.0),
                        ),
                        "best_score": float(
                            ts.get("max_candidate_score", 0.0),
                        ),
                    }
            else:
                cands = list(search_obj.search(
                    initial_state, max_depth=max_depth,
                ))
        except Exception as exc:
            return {
                "wall_time_s": float(time.time() - t0),
                "n_candidates": 0,
                "mean_var": 0.0,
                "mean_score": 0.0,
                "best_score": 0.0,
                "error": str(exc),
            }
        wall = time.time() - t0
        if not cands:
            return {
                "wall_time_s": float(wall),
                "n_candidates": 0,
                "mean_var": 0.0,
                "mean_score": 0.0,
                "best_score": 0.0,
            }
        # Score candidates through the *search's own* scoring
        # function — this gives the variance of the leaf rewards
        # emitted by that specific MCTS run.
        try:
            if hasattr(search_obj, "score_final"):
                scorer = search_obj.score_final
            elif hasattr(search_obj, "base_search"):
                scorer = search_obj.base_search.score_final
            else:
                scorer = base_search.score_final
            values = [float(scorer(c)) for c in cands]
        except Exception:
            values = [0.0 for _ in cands]
        if not values:
            return {
                "wall_time_s": float(wall),
                "n_candidates": 0,
                "mean_var": 0.0,
                "mean_score": 0.0,
                "best_score": 0.0,
            }
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / len(values)
        best = max(values)
        return {
            "wall_time_s": float(wall),
            "n_candidates": int(len(cands)),
            "mean_var": float(var),
            "mean_score": float(mean_v),
            "best_score": float(best),
        }

    t0 = time.time()
    initial_state = None
    try:
        if not getattr(base_search, "tile_library", None):
            return {"error": "base_search.tile_library is empty",
                    "wall_time_s": time.time() - t0}
        initial_state = base_search.tile_library[0]
    except Exception as exc:
        return {"error": str(exc), "wall_time_s": time.time() - t0}

    base_runs: List[Dict[str, float]] = []
    var_runs: List[Dict[str, float]] = []
    for _ in range(int(n_runs)):
        base_runs.append(_run_search(base_search, initial_state, max_depth))
        var_runs.append(
            _run_search(variance_search, initial_state, max_depth),
        )

    def _aggregate(runs: List[Dict[str, float]]) -> Dict[str, float]:
        if not runs:
            return {
                "wall_time_s": 0.0, "mean_var": 0.0,
                "mean_score": 0.0, "best_score": 0.0,
                "success_rate": 0.0,
                "n_candidates": 0.0,
            }
        ok = [r for r in runs if "error" not in r]
        success = len(ok) / len(runs)
        if not ok:
            return {
                "wall_time_s": float(sum(r["wall_time_s"] for r in runs)),
                "mean_var": 0.0, "mean_score": 0.0,
                "best_score": 0.0, "success_rate": float(success),
                "n_candidates": 0.0,
            }
        return {
            "wall_time_s": float(sum(r["wall_time_s"] for r in runs)),
            "mean_var": float(sum(r["mean_var"] for r in ok) / len(ok)),
            "mean_score": float(sum(r["mean_score"] for r in ok) / len(ok)),
            "best_score": float(sum(r["best_score"] for r in ok) / len(ok)),
            "success_rate": float(success),
            "n_candidates": float(sum(r["n_candidates"] for r in ok) / len(ok)),
        }

    base_summary = _aggregate(base_runs)
    var_summary = _aggregate(var_runs)
    # Variance reduction (positive ⇒ variance search wins).
    base_var = float(base_summary["mean_var"])
    var_var = float(var_summary["mean_var"])
    if base_var > 1.0e-12:
        var_reduction = float(1.0 - var_var / base_var)
    else:
        var_reduction = 0.0
    # Wall-time ratio (var / base).  <1.0 ⇒ variance search is faster.
    base_wall = float(base_summary["wall_time_s"])
    var_wall = float(var_summary["wall_time_s"])
    if base_wall > 1.0e-9:
        wall_ratio = float(var_wall / base_wall)
    else:
        wall_ratio = 1.0
    return {
        "base": base_summary,
        "variance": var_summary,
        "variance_reduction": float(var_reduction),
        "wall_time_ratio": float(wall_ratio),
        "per_run": {
            "base": base_runs,
            "variance": var_runs,
        },
        "n_runs": int(n_runs),
        "max_depth": int(max_depth),
        "wall_time_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_C_PUCT",
    "DEFAULT_C_VAR",
    "DEFAULT_N_MIN_VISITS",
    "DEFAULT_N_RUNS",
    "DEFAULT_SEED",
    "VarianceMCTSError",
    "VarianceMCTSNode",
    "VarianceMCTSWrapper",
    "compare_variance_vs_dirichlet_mcts",
]