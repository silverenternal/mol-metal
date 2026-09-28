"""Counterfactual Replay Mining — Advanced Dream-RSI application #1 (TODO-153).

================================================================
Counterfactual Replay Mining for Off-Policy Evaluation
================================================================
Dream-RSI (arXiv:2609.14858, Zheng, Wu, Zhang et al., 2026) is built
on a *deterministic replay buffer* — every past (state, action,
reward, next_state) tuple is preserved, so alternative branches can
be re-walked without paying for fresh agent calls.  mol-metal has
adapted that idea (TODO-150 §A2) as a *cache-hit + surrogate ranker*
for Vina.  This module is the load-bearing next step: it answers
*counterfactual* questions like "what would have happened if we had
picked candidate B instead of A?" by replaying the discovery tree.

Two primitives are exposed:

1. ``CounterfactualReplay.replay_counterfactual``  — given a target
   candidate (smiles, pocket, seed) and a list of *alternative*
   candidates, return the surrogate's predicted Vina score for each
   alternative *as if* the alternative had been picked instead of
   the target.  This is the load-bearing primitive for closed-loop
   self-improvement: it lets a training-loop controller ask "of the
   three HPs I almost picked, which one would have done better?"
   without paying for a fresh Vina call.

2. ``CounterfactualReplay.best_alternative``  — convenience helper
   that returns the top-K alternatives ranked by counterfactual
   predicted score.  This is the natural integration point with the
   ``proof_search.py`` MCTS rollout — at each branch the rollout
   can ask "of the unexplored children, which would have scored
   best in the historical record?" and bias exploration accordingly.

Design contract (per §3.5 honest-framing discipline):

* Counterfactual predictions are *deterministic* given the replay
  pool + a fitted surrogate; the surrogate is the
  :class:`ReplayPoolSurrogate` from the sibling
  :mod:`vina_replay_pool` module so the math is identical to what
  TODO-151 verified (Spearman ρ = 0.9659 on wf126 P3 cells).
* When the target candidate has a *cache hit* (was docked before),
  the cache hit's score is returned verbatim with
  ``confidence="high"``; the counterfactual branch is the
  load-bearing oracle-call-saver.
* When the cache is empty or the surrogate is unfitted, every
  prediction is returned with ``confidence="low"`` and the caller
  is expected to fall back to the real oracle.  No silent promotion
  of under-informed counterfactuals.
* Counterfactual predictions are NOT a substitute for the oracle —
  every ``confidence="low"`` row should still be confirmed with a
  real Vina call before being acted on.  The savings come from
  shortlisting alternatives cheaply, not from skipping confirmation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from molmetal.reports.dream_rsi_replay_pool.vina_replay_pool import (
    ReplayPoolSurrogate,
    VinaReplayPool,
)


log = logging.getLogger(__name__)


#: Minimum pool size required to issue *medium*-confidence
#: counterfactual predictions.  Below this threshold every row is
#: flagged ``confidence="low"``.  Mirrors
#: ``MIN_HPOVER_POOL_SIZE`` in the sibling vina module so the two
#: confidence tiers line up.
MIN_COUNTERFACTUAL_POOL: int = 32


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualRow:
    """One row in a counterfactual replay report.

    Attributes
    ----------
    smiles:
        The candidate whose counterfactual Vina score is reported.
    predicted_score:
        The surrogate-predicted Vina kcal/mol if this candidate
        had been picked in place of the target.  More negative
        == stronger binder, matches Vina convention.
    confidence:
        ``"high"`` when the surrogate has an exact-pool match for
        this candidate in the target pocket (cache hit);
        ``"medium"`` when the pool exceeds ``MIN_COUNTERFACTUAL_POOL``
        entries; ``"low"`` otherwise.
    n_pool_used:
        Number of pool entries the surrogate used to fit its
        prediction.  Useful for audit reports.
    cache_hit:
        Convenience flag — True iff the candidate was already in
        the pool.
    metadata:
        Free-form provenance blob.
    """

    smiles: str
    predicted_score: float
    confidence: str
    n_pool_used: int
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": str(self.smiles),
            "predicted_score": float(self.predicted_score),
            "confidence": str(self.confidence),
            "n_pool_used": int(self.n_pool_used),
            "cache_hit": bool(self.cache_hit),
            "metadata": dict(self.metadata) if self.metadata else {},
        }


@dataclass
class CounterfactualReport:
    """Result of :meth:`CounterfactualReplay.replay_counterfactual`.

    Attributes
    ----------
    target_smiles:
        The (counterfactual) "actual" candidate the alternatives are
        being compared against.
    pocket_id:
        The pocket the alternatives would have been docked against.
    seed:
        Seed passed to the surrogate encoder (so the report is
        reproducible).
    rows:
        Per-alternative counterfactual predictions, ordered the
        same as the input ``alternative_smiles`` argument (NOT
        ranked ascending — the caller decides the ranking policy).
    n_pool_used:
        Total pool size across pockets.  Useful for audit.
    surrogate_class:
        String name of the surrogate backend.
    """

    target_smiles: str
    pocket_id: str
    seed: int
    rows: List[CounterfactualRow]
    n_pool_used: int
    surrogate_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_smiles": str(self.target_smiles),
            "pocket_id": str(self.pocket_id),
            "seed": int(self.seed),
            "rows": [r.to_dict() for r in self.rows],
            "n_pool_used": int(self.n_pool_used),
            "surrogate_class": str(self.surrogate_class),
        }

    def ranked(self) -> List[CounterfactualRow]:
        """Return ``self.rows`` sorted ascending by ``predicted_score``.

        Convenience helper for callers that just want "which
        alternative would have done best?".
        """
        return sorted(self.rows, key=lambda r: r.predicted_score)


# ---------------------------------------------------------------------------
# Counterfactual Replay
# ---------------------------------------------------------------------------


class CounterfactualReplay:
    """Off-policy counterfactual evaluator for Vina-replay candidates.

    The class wraps a :class:`VinaReplayPool` and a fitted
    :class:`ReplayPoolSurrogate`.  The replay buffer is treated as
    the immutable "world" from Dream-RSI §3 stage ❷; counterfactual
    questions about the world are answered by re-walking the buffer
    deterministically through the surrogate.

    Parameters
    ----------
    pool:
        The :class:`VinaReplayPool` instance to query.  When ``None``
        the object is built but every prediction is flagged
        ``confidence="low"``.
    surrogate:
        The :class:`ReplayPoolSurrogate` to use for novel-candidate
        predictions.  When ``None`` a default ridge-regression
        surrogate is constructed and fitted on first use.
    backend:
        Surrogate backend string; only consulted when ``surrogate``
        is None.  Defaults to ``"ridge"`` for CPU-only determinism.
    alpha:
        Ridge regularisation strength; only consulted when
        ``surrogate`` is None.
    """

    def __init__(
        self,
        pool: Optional[VinaReplayPool] = None,
        surrogate: Optional[ReplayPoolSurrogate] = None,
        backend: str = "ridge",
        alpha: float = 1.0,
    ) -> None:
        self.pool = pool
        self.surrogate = surrogate
        self.backend = str(backend)
        self.alpha = float(alpha)
        # Cache the target pocket's pool size so callers don't have to
        # import VinaReplayPool internals.
        self._last_pool_size: int = 0
        self._last_pool_low_coverage: bool = True

    # ------------------------------------------------------------------
    # Lazy surrogate construction
    # ------------------------------------------------------------------
    def _ensure_surrogate(self) -> ReplayPoolSurrogate:
        if self.surrogate is None:
            if self.pool is None:
                raise RuntimeError(
                    "CounterfactualReplay needs either a pool or a "
                    "pre-fitted surrogate"
                )
            self.surrogate = ReplayPoolSurrogate(
                self.pool, backend=self.backend, alpha=self.alpha
            )
            self.surrogate.fit()
        return self.surrogate

    # ------------------------------------------------------------------
    # Pool-size helpers
    # ------------------------------------------------------------------
    def _pool_size_for(self, pocket_id: str) -> int:
        if self.pool is None:
            return 0
        try:
            return int(self.pool.total_for_pocket(pocket_id))
        except Exception:  # pragma: no cover — defensive
            log.warning("CounterfactualReplay: pool size lookup failed")
            return 0

    def _is_low_coverage(self, pocket_id: str) -> bool:
        return self._pool_size_for(pocket_id) < MIN_COUNTERFACTUAL_POOL

    # ------------------------------------------------------------------
    # Counterfactual prediction
    # ------------------------------------------------------------------
    def replay_counterfactual(
        self,
        target_smiles: str,
        alternative_smiles: Sequence[str],
        pocket_id: str,
        seed: int = 0,
    ) -> CounterfactualReport:
        """Predict the Vina score for each alternative "as if" the
        alternative had been picked instead of ``target_smiles``.

        The target is preserved in the report so the caller can
        include the actual observed score (if it was a cache hit)
        for comparison.

        Parameters
        ----------
        target_smiles:
            The candidate that was actually picked in the recorded
            history.  Used only as a label — its predicted score
            is *not* computed (callers can look it up directly via
            :meth:`VinaReplayPool.lookup_exact`).
        alternative_smiles:
            The candidates to evaluate counterfactually.  Order is
            preserved in ``report.rows``.
        pocket_id:
            Pocket for the counterfactual dock.
        seed:
            Encoder seed forwarded to the surrogate so the report
            is reproducible.

        Returns
        -------
        :class:`CounterfactualReport`
            One :class:`CounterfactualRow` per alternative; sorted
            ascending by ``predicted_score`` via :meth:`ranked`
            but the raw ``report.rows`` preserve input order.
        """
        if not alternative_smiles:
            return CounterfactualReport(
                target_smiles=str(target_smiles),
                pocket_id=str(pocket_id),
                seed=int(seed),
                rows=[],
                n_pool_used=self._pool_size_for(pocket_id),
                surrogate_class="CounterfactualReplay[empty]",
            )

        surrogate = None
        try:
            surrogate = self._ensure_surrogate()
        except RuntimeError:
            # No pool and no pre-fitted surrogate — every prediction
            # will be flagged low-confidence and scored as 0.0.
            surrogate = None
        n_pool = self._pool_size_for(pocket_id)
        low_cov = self._is_low_coverage(pocket_id)

        rows: List[CounterfactualRow] = []
        for alt in alternative_smiles:
            if not isinstance(alt, str) or not alt:
                raise ValueError(
                    f"alternative_smiles entries must be non-empty str; "
                    f"got {alt!r}"
                )
            # Cache-hit path: if the alternative was docked before
            # in this pocket, return the cached score with high
            # confidence.  This is the load-bearing oracle-call
            # reduction — same logic as TODO-151's cache-hit path.
            if self.pool is not None:
                exact = self.pool.lookup_exact(alt, pocket_id)
                if exact is not None:
                    rows.append(
                        CounterfactualRow(
                            smiles=alt,
                            predicted_score=float(exact.vina_score),
                            confidence="high",
                            n_pool_used=n_pool,
                            cache_hit=True,
                            metadata={
                                "method": "cache_hit",
                                "cached_seed": int(exact.seed),
                            },
                        )
                    )
                    continue
            # Surrogate path: predict the counterfactual score.  When
            # the pool is below the coverage threshold the prediction
            # is flagged ``"low"``; otherwise it is ``"medium"``.
            if surrogate is None:
                score = 0.0
            else:
                score = float(
                    surrogate.predict_score(alt, pocket_id, seed=int(seed))
                )
            confidence = "low" if (low_cov or surrogate is None) else "medium"
            rows.append(
                CounterfactualRow(
                    smiles=alt,
                    predicted_score=score,
                    confidence=confidence,
                    n_pool_used=n_pool,
                    cache_hit=False,
                    metadata={"method": "surrogate_predict"},
                )
            )

        return CounterfactualReport(
            target_smiles=str(target_smiles),
            pocket_id=str(pocket_id),
            seed=int(seed),
            rows=rows,
            n_pool_used=n_pool,
            surrogate_class=(
                f"CounterfactualReplay[{self.surrogate.backend}]"
                if self.surrogate is not None
                else "CounterfactualReplay[unfitted]"
            ),
        )

    def best_alternative(
        self,
        target_smiles: str,
        alternative_smiles: Sequence[str],
        pocket_id: str,
        seed: int = 0,
        top_k: int = 5,
        min_confidence: str = "low",
    ) -> List[CounterfactualRow]:
        """Return the top-K alternatives ranked by counterfactual score.

        ``min_confidence`` filters out low-confidence rows; pass
        ``"medium"`` to keep only medium+high confidence rows
        (useful when the pool is small and you don't want to act
        on speculative predictions).

        Parameters
        ----------
        target_smiles, alternative_smiles, pocket_id, seed:
            Forwarded to :meth:`replay_counterfactual`.
        top_k:
            Maximum number of alternatives to return.
        min_confidence:
            One of ``"low"``, ``"medium"``, ``"high"``.  Rows below
            the threshold are dropped before ranking.

        Returns
        -------
        list[CounterfactualRow]
            Top-K alternatives sorted ascending by ``predicted_score``.
        """
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        threshold = confidence_rank.get(str(min_confidence), 0)

        report = self.replay_counterfactual(
            target_smiles=target_smiles,
            alternative_smiles=alternative_smiles,
            pocket_id=pocket_id,
            seed=seed,
        )
        filtered = [
            row for row in report.rows
            if confidence_rank.get(row.confidence, 0) >= threshold
        ]
        filtered.sort(key=lambda r: r.predicted_score)
        return filtered[: int(top_k)]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def coverage_summary(self) -> Dict[str, Any]:
        """Per-pocket coverage diagnostics, mirroring the parent
        :class:`VinaReplayPool` API surface.

        Returns
        -------
        dict
            ``{"pocket_id": {"n": int, "is_low_coverage": bool}, ...}``.
        """
        if self.pool is None:
            return {}
        summary: Dict[str, Any] = {}
        for pocket_id in sorted(self.pool.n_per_pocket().keys()):
            n = self.pool.total_for_pocket(pocket_id)
            summary[str(pocket_id)] = {
                "n": int(n),
                "is_low_coverage": bool(n < MIN_COUNTERFACTUAL_POOL),
            }
        return summary


__all__ = [
    "MIN_COUNTERFACTUAL_POOL",
    "CounterfactualReplay",
    "CounterfactualReport",
    "CounterfactualRow",
]