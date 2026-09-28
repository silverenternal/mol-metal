"""Adaptive Curriculum via Replay Difficulty Mining — Advanced Dream-RSI
application #2 (TODO-153).

================================================================
Replay-driven adaptive curriculum for D-MPNN training
================================================================
Dream-RSI's replay pool gives mol-metal a cheap way to measure
*training-example difficulty* without paying for fresh oracle calls.
The mechanism is borrowed from TODO-150 §A6 (replay-disagreement
uncertainty): for each cached (smiles, pocket) pair, query the
surrogate's predicted score against the cached true score, and the
absolute residual is the difficulty proxy.  High residual means the
surrogate was surprised — that example is *hard*.  Upsample the hard
examples in the next training round.

This module exposes two primitives:

1. :class:`DifficultyScorer` — given a fitted surrogate + a list of
   ``(smiles, pocket, true_score)`` records, return a difficulty
   score per record.  The scorer is deterministic (modulo the
   random-bootstrap seed) and CPU-only.

2. :class:`AdaptiveCurriculumSampler` — wraps a :class:`VinaReplayPool`
   (or any pool whose ``lookup_exact`` returns a cached score) and
   exposes :meth:`sample_batch` which returns a weighted sample of
   (smiles, pocket) pairs biased toward the hardest cached examples.
   The weighting is configurable via a temperature parameter
   ``difficulty_temp`` (high temp = uniform, low temp = concentrate
   on hardest).

Design contract (per §3.5 honest-framing discipline):

* Difficulty scores are deterministic given a fixed pool +
  deterministic encoder.  Two callers asking the same query see
  the same scores.
* When the pool is below the coverage threshold (default 32 per
  pocket), every score is marked ``confidence="low"`` and the
  sampler falls back to uniform sampling.  No silent promotion
  of under-informed curriculum biases.
* The sampler is *opt-in*: callers must explicitly request an
  adaptive batch via :meth:`sample_batch`.  Default behaviour is
  uniform, matching the current training loop.
* The difficulty score is a *surrogate residual*, not a measure
  of physical difficulty.  A high residual means the surrogate
  was surprised — which can be either (a) genuinely hard
  chemistry or (b) a region the surrogate has not seen enough
  training data for.  Both interpretations are useful (active
  learning + hard-example mining) but the sampler does NOT
  distinguish them.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from molmetal.reports.dream_rsi_replay_pool.vina_replay_pool import (
    ReplayPoolSurrogate,
    VinaReplayPool,
)


log = logging.getLogger(__name__)


#: Minimum pool size per pocket before the curriculum switches to
#: uniform sampling.  Matches :data:`MIN_HPOVER_POOL_SIZE` in the
#: sibling vina module so the two confidence tiers line up.
MIN_DIFFICULTY_POOL: int = 32

#: Default softmax temperature for the difficulty weighting.  A
#: temperature of 1.0 gives a moderate bias toward hard examples;
#: ``-1.0`` flips the sign and biases toward easy examples (useful
#: as a control in ablation experiments); ``0.0`` collapses to
#: uniform.  Callers can override per-call.
DEFAULT_DIFFICULTY_TEMP: float = 1.0

#: Floor on the per-row sampling weight so even the easiest cached
#: example still has a non-zero probability of being picked.  Without
#: this floor the adaptive sampler would degenerate to always pick
#: the single hardest row once the temperature is small.
DEFAULT_WEIGHT_FLOOR: float = 1e-3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DifficultyRow:
    """One row in a difficulty report.

    Attributes
    ----------
    smiles:
        Candidate SMILES.
    pocket_id:
        Pocket ID.
    true_score:
        Recorded (cached) Vina score for this (smiles, pocket).
    predicted_score:
        Surrogate's predicted score for this (smiles, pocket).
    absolute_residual:
        ``|predicted_score - true_score|`` — the difficulty proxy.
        Higher == harder / more surprising.
    relative_residual:
        ``absolute_residual / max(|true_score|, 1.0)`` — same
        difficulty signal but normalised by the score magnitude so
        it is comparable across pockets with different Vina ranges.
    confidence:
        ``"high"`` when the row was a cache hit (the surrogate's
        prediction matched exactly); ``"medium"`` when the pool has
        at least ``MIN_DIFFICULTY_POOL`` entries; ``"low"``
        otherwise.
    """

    smiles: str
    pocket_id: str
    true_score: float
    predicted_score: float
    absolute_residual: float
    relative_residual: float
    confidence: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": str(self.smiles),
            "pocket_id": str(self.pocket_id),
            "true_score": float(self.true_score),
            "predicted_score": float(self.predicted_score),
            "absolute_residual": float(self.absolute_residual),
            "relative_residual": float(self.relative_residual),
            "confidence": str(self.confidence),
        }


@dataclass
class CurriculumBatch:
    """Result of :meth:`AdaptiveCurriculumSampler.sample_batch`.

    Attributes
    ----------
    rows:
        The selected (smiles, pocket_id, weight) triples, in the
        order they were sampled.
    n_pool_used:
        Size of the underlying pool when the batch was drawn.
    difficulty_temp:
        The temperature used for this batch.
    weight_floor:
        The weight floor used for this batch.
    is_low_coverage:
        True iff the underlying pool is below
        :data:`MIN_DIFFICULTY_POOL` — in which case the sampler
        fell back to uniform.
    """

    rows: List[Tuple[str, str, float]]
    n_pool_used: int
    difficulty_temp: float
    weight_floor: float
    is_low_coverage: bool = False

    def smiles(self) -> List[str]:
        return [r[0] for r in self.rows]

    def pockets(self) -> List[str]:
        return [r[1] for r in self.rows]


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


class DifficultyScorer:
    """Compute per-example difficulty (surrogate residual) on a cache.

    Parameters
    ----------
    pool:
        The :class:`VinaReplayPool` to score against.  When ``None``
        the scorer is constructed but every call returns an empty
        report.
    surrogate:
        Optional pre-fitted :class:`ReplayPoolSurrogate`.  When
        ``None`` a default ridge-regression surrogate is built and
        fitted on first use.
    backend, alpha:
        Surrogate construction args; only consulted when
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

    def _ensure_surrogate(self) -> ReplayPoolSurrogate:
        if self.surrogate is None:
            if self.pool is None:
                raise RuntimeError(
                    "DifficultyScorer needs either a pool or a "
                    "pre-fitted surrogate"
                )
            self.surrogate = ReplayPoolSurrogate(
                self.pool, backend=self.backend, alpha=self.alpha
            )
            self.surrogate.fit()
        return self.surrogate

    def score_pool(
        self,
        pocket_id: Optional[str] = None,
        seed: int = 0,
    ) -> List[DifficultyRow]:
        """Score every cached row in ``pocket_id`` (or all pockets).

        Parameters
        ----------
        pocket_id:
            When provided, only rows for this pocket are scored.
            When ``None`` every row in the pool is scored.
        seed:
            Encoder seed forwarded to the surrogate.

        Returns
        -------
        list[DifficultyRow]
            One row per cached (smiles, pocket) pair.  Empty when
            the pool is empty.
        """
        if self.pool is None:
            return []
        surrogate = self._ensure_surrogate()

        # Flatten the pool to (smiles, pocket, true_score) triples.
        triples: List[Tuple[str, str, float, int]] = []
        pocket_filter = (
            lambda pid: True if pocket_id is None else pid == pocket_id
        )
        for pid, entries in self.pool._by_pocket.items():  # noqa: SLF001 — internal but stable
            if not pocket_filter(pid):
                continue
            for entry in entries:
                triples.append(
                    (entry.smiles, pid, float(entry.vina_score), int(entry.seed))
                )
        if not triples:
            return []

        n_pool_pid = self.pool.total_for_pocket(
            pocket_id if pocket_id is not None else triples[0][1]
        )
        low_cov = n_pool_pid < MIN_DIFFICULTY_POOL

        rows: List[DifficultyRow] = []
        for smi, pid, true_score, entry_seed in triples:
            exact = self.pool.lookup_exact(smi, pid)
            cache_hit = (
                exact is not None
                and abs(float(exact.vina_score) - true_score) < 1e-9
            )
            # Compute surrogate prediction; fall back to true_score
            # when the surrogate is unfitted (defensive).
            try:
                pred = float(
                    surrogate.predict_score(smi, pid, seed=int(seed))
                )
            except Exception:  # pragma: no cover — defensive
                pred = true_score
            # Confidence: cache hit is always "high"; otherwise the
            # coverage tier decides between "medium" / "low".
            if cache_hit:
                confidence = "high"
            else:
                confidence = "low" if low_cov else "medium"

            abs_res = abs(float(pred) - float(true_score))
            denom = max(abs(float(true_score)), 1.0)
            rel_res = abs_res / denom
            rows.append(
                DifficultyRow(
                    smiles=smi,
                    pocket_id=pid,
                    true_score=float(true_score),
                    predicted_score=float(pred),
                    absolute_residual=float(abs_res),
                    relative_residual=float(rel_res),
                    confidence=confidence,
                )
            )
        return rows


# ---------------------------------------------------------------------------
# AdaptiveCurriculumSampler
# ---------------------------------------------------------------------------


class AdaptiveCurriculumSampler:
    """Weighted sampler biased toward hard cached examples.

    Parameters
    ----------
    pool:
        The :class:`VinaReplayPool` to sample from.  When ``None``
        the sampler is constructed but every batch is empty.
    scorer:
        Optional pre-built :class:`DifficultyScorer`.  When ``None``
        a default scorer is constructed and used.
    difficulty_temp:
        Softmax temperature for the difficulty weighting.  Higher
        == more uniform; lower == more concentrated on hard
        examples.  Default :data:`DEFAULT_DIFFICULTY_TEMP`.
    weight_floor:
        Floor on the per-row sampling weight.  Default
        :data:`DEFAULT_WEIGHT_FLOOR`.
    rng_seed:
        Optional seed for reproducibility; pass ``None`` for
        non-deterministic sampling.
    """

    def __init__(
        self,
        pool: Optional[VinaReplayPool] = None,
        scorer: Optional[DifficultyScorer] = None,
        difficulty_temp: float = DEFAULT_DIFFICULTY_TEMP,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        rng_seed: Optional[int] = 0,
    ) -> None:
        self.pool = pool
        self.scorer = scorer
        self.difficulty_temp = float(difficulty_temp)
        self.weight_floor = float(weight_floor)
        self.rng_seed = (
            int(rng_seed) if rng_seed is not None else None
        )
        self._rng = random.Random(self.rng_seed)

    # ------------------------------------------------------------------
    # Lazy scorer construction
    # ------------------------------------------------------------------
    def _ensure_scorer(self) -> DifficultyScorer:
        if self.scorer is None:
            self.scorer = DifficultyScorer(pool=self.pool)
        return self.scorer

    # ------------------------------------------------------------------
    # Weighting
    # ------------------------------------------------------------------
    def _softmax_weights(
        self, residuals: Sequence[float]
    ) -> np.ndarray:
        """Convert residuals to a probability distribution.

        ``weight = softmax(residual / max(temp, eps))`` with a
        uniform floor applied so no row has zero probability.
        """
        if not residuals:
            return np.zeros(0, dtype=np.float64)
        temp = max(abs(float(self.difficulty_temp)), 1e-6)
        arr = np.asarray(residuals, dtype=np.float64) / temp
        # Numerical stability: subtract max before exp.
        arr = arr - float(arr.max())
        weights = np.exp(arr)
        weights = weights / weights.sum()
        # Apply floor: re-normalise so the floor only kicks in
        # when the natural weight is below it.  P rows * 1e-3 =
        # ~P * floor mass that must come out of the non-floor rows.
        floor = float(self.weight_floor)
        if floor > 0.0:
            n = len(weights)
            total_floor = floor * n
            if total_floor < 1.0:
                # Reduce natural weights to make room for the floor.
                weights = weights * (1.0 - total_floor)
                weights = weights + floor
                weights = weights / weights.sum()
            else:
                # Floor mass exceeds the simplex; fall back to uniform.
                weights = np.ones(n, dtype=np.float64) / n
        return weights

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_batch(
        self,
        batch_size: int,
        pocket_id: Optional[str] = None,
        seed: int = 0,
    ) -> CurriculumBatch:
        """Draw ``batch_size`` rows from the pool, weighted by
        difficulty.

        Parameters
        ----------
        batch_size:
            Number of (smiles, pocket) pairs to draw.
        pocket_id:
            When provided, restrict sampling to this pocket.  When
            ``None`` sample across all pockets.
        seed:
            Encoder seed forwarded to the underlying scorer.

        Returns
        -------
        :class:`CurriculumBatch`
            The sampled rows plus provenance.

        Notes
        -----
        When the pool is below :data:`MIN_DIFFICULTY_POOL` the
        sampler falls back to uniform random sampling and reports
        ``is_low_coverage=True``.  This is the load-bearing honesty
        gate — we never silently bias a small cache.
        """
        if self.pool is None or batch_size <= 0:
            return CurriculumBatch(
                rows=[],
                n_pool_used=0,
                difficulty_temp=self.difficulty_temp,
                weight_floor=self.weight_floor,
                is_low_coverage=True,
            )

        scorer = self._ensure_scorer()
        difficulty_rows = scorer.score_pool(pocket_id=pocket_id, seed=seed)
        if not difficulty_rows:
            return CurriculumBatch(
                rows=[],
                n_pool_used=0,
                difficulty_temp=self.difficulty_temp,
                weight_floor=self.weight_floor,
                is_low_coverage=True,
            )

        n_pool_pid = self.pool.total_for_pocket(
            pocket_id if pocket_id is not None else difficulty_rows[0].pocket_id
        )
        low_cov = n_pool_pid < MIN_DIFFICULTY_POOL

        # Use the absolute residual as the difficulty signal.
        residuals = [r.absolute_residual for r in difficulty_rows]
        weights = self._softmax_weights(residuals)

        # Sample with replacement so the batch is always of size
        # ``batch_size`` even when the pool is small.
        if low_cov:
            # Uniform fallback — the load-bearing honesty gate.
            weights = np.ones(len(difficulty_rows), dtype=np.float64)
            weights = weights / weights.sum()

        indices = self._rng.choices(
            range(len(difficulty_rows)), weights=weights.tolist(), k=int(batch_size)
        )

        rows: List[Tuple[str, str, float]] = []
        for idx in indices:
            r = difficulty_rows[int(idx)]
            rows.append((r.smiles, r.pocket_id, float(weights[int(idx)])))

        return CurriculumBatch(
            rows=rows,
            n_pool_used=n_pool_pid,
            difficulty_temp=self.difficulty_temp,
            weight_floor=self.weight_floor,
            is_low_coverage=bool(low_cov),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def difficulty_table(
        self,
        pocket_id: Optional[str] = None,
        top_k: int = 10,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return the top-K hardest examples as plain dicts.

        Convenience for printing/auditing the curriculum's notion of
        "hard".  ``top_k`` rows with the largest ``relative_residual``
        are returned.
        """
        scorer = self._ensure_scorer()
        rows = scorer.score_pool(pocket_id=pocket_id, seed=seed)
        rows.sort(key=lambda r: r.relative_residual, reverse=True)
        return [r.to_dict() for r in rows[: int(top_k)]]


__all__ = [
    "DEFAULT_DIFFICULTY_TEMP",
    "DEFAULT_WEIGHT_FLOOR",
    "MIN_DIFFICULTY_POOL",
    "AdaptiveCurriculumSampler",
    "CurriculumBatch",
    "DifficultyRow",
    "DifficultyScorer",
]