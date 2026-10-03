"""Dream-RSI replay pool primitives for mol-metal training (TODO-150).

================================================================
ReplayPool — cheap off-policy evaluator for D-MPNN config search
================================================================
This module implements the load-bearing P0 piece of the
Dream-RSI integration plan:

* ``TrajectoryRecord``     — one (config, hp_dict, per-epoch metrics,
  final score, metal, seed) row per training run.
* ``ReplayPool``           — append/query/coverage/predict interface
  used by A1 (config-search via replay), A3 (per-metal bootstrap), and
  A6 (replay-disagreement uncertainty).
* ``pessimistic_shrinkage``— §4 R1 mitigation that down-weights replay
  predictions on metals with low replay coverage, so the pool cannot
  silently produce optimistic predictions for under-sampled cells
  (Ru has ~2500 training rows; Re has ~121 — a 20× gap).

Design contract (per §3.5 honest-framing discipline):

* All methods are CPU-only — no GPU, no RDKit, no D-MPNN weights.
* :meth:`ReplayPool.replay_predict` is *deterministic* given a fixed
  pool and ``hp_delta`` (modulo the optional bootstrap seed).
* Predictions carry a ``confidence`` field that callers MUST honour:
  ``confidence == "low"`` means the cell is ``LOW_COVERAGE`` and the
  prediction is pessimistically shrunk.  Promotion of a replay-only
  decision on a low-confidence cell without a held-out verify-run is
  explicitly disallowed by the plan's exit gate.
* No claim of headline 50×–162× call reduction (the paper's claim is in
  a different bottleneck regime — LLM agent calls, not GPU training).
  The mechanism is borrowed; the speedups are project-local projections
  per TODO-150 §3.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


#: Per-metal cell-count below which a metal is flagged
#: ``LOW_COVERAGE`` and replay predictions are pessimistically shrunk.
#: Default 10 matches the threshold cited in TODO-150 §P0.4.
LOW_COVERAGE_THRESHOLD: int = 10

#: Default baseline prior used for shrinkage when ``n_records == 0``.
#: ``0.5`` is the midpoint of AUC; for regression metrics (Pearson r,
#: MAE) callers should override ``baseline`` via :meth:`replay_predict`.
_DEFAULT_BASELINE: float = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryRecord:
    """One training trajectory appended to the replay pool.

    Mirrors the Dream-RSI "discovery-tree node" primitive: every past
    (config, hp_dict, per-epoch metrics, final test score, metal, seed)
    run becomes a leaf that can be re-walked deterministically.

    Attributes
    ----------
    config_hash:
        Stable hash of the resolved training config (typically
        ``hashlib.sha256(json.dumps(hp_dict, sort_keys=True)).hexdigest()
        [:16]``).  Two runs with identical hp_dict should produce
        identical hashes so query-by-config works.
    hp_dict:
        Resolved training hyperparameters (lr, hidden_dim, depth,
        alpha, dropout, …).  Stored as a plain dict for cheap
        equality / subset matching.
    per_epoch_metrics:
        Optional per-epoch trajectory — list of dicts, one per epoch.
        Each dict carries keys like ``val_auc`` / ``val_mae`` /
        ``val_pearson_r`` / ``train_loss``.  Empty list is allowed
        when the run did not emit per-epoch metrics (older ckpts).
    final_test_score:
        The metric the trainer reported as final (test_auc for
        classification, test_pearson_r / test_rmse for regression).
    metal:
        Metal centre — one of ``{Ru, Ir, Rh, Os, Re, Pt}``.
    seed:
        RNG seed for the run.
    timestamp:
        Wall-clock time the record was added (seconds since epoch).
    metadata:
        Free-form provenance blob (commit sha, workflow id, sweep
        preset, …).  Not used by the predictor; preserved through
        save/load for audit.
    """

    config_hash: str
    hp_dict: Dict[str, Any]
    per_epoch_metrics: List[Dict[str, float]] = field(default_factory=list)
    final_test_score: float = 0.0
    metal: str = "Ru"
    seed: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config_hash, str) or not self.config_hash:
            raise ValueError(
                f"config_hash must be non-empty str; got {self.config_hash!r}"
            )
        if not isinstance(self.hp_dict, dict):
            raise ValueError(
                f"hp_dict must be dict; got {type(self.hp_dict).__name__}"
            )
        if not isinstance(self.final_test_score, (int, float)):
            raise ValueError(
                f"final_test_score must be numeric; "
                f"got {type(self.final_test_score).__name__}"
            )
        if math.isnan(self.final_test_score) or math.isinf(self.final_test_score):
            # NaN/Inf cannot be used by nearest-neighbour arithmetic;
            # reject so the caller fixes the source rather than silently
            # poisoning the pool.
            raise ValueError(
                f"final_test_score must be finite; got {self.final_test_score}"
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict representation."""
        return {
            "config_hash": self.config_hash,
            "hp_dict": dict(self.hp_dict),
            "per_epoch_metrics": [
                {k: float(v) for k, v in ep.items()}
                for ep in self.per_epoch_metrics
            ],
            "final_test_score": float(self.final_test_score),
            "metal": str(self.metal),
            "seed": int(self.seed),
            "timestamp": float(self.timestamp),
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrajectoryRecord":
        """Inverse of :meth:`to_dict`."""
        return cls(
            config_hash=str(d["config_hash"]),
            hp_dict=dict(d.get("hp_dict", {})),
            per_epoch_metrics=[
                {k: float(v) for k, v in ep.items()}
                for ep in d.get("per_epoch_metrics", [])
            ],
            final_test_score=float(d.get("final_test_score", 0.0)),
            metal=str(d.get("metal", "Ru")),
            seed=int(d.get("seed", 0)),
            timestamp=float(d.get("timestamp", time.time())),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class CoverageStats:
    """Per-metal replay-coverage diagnostics (P0.4 in TODO-150).

    Attributes
    ----------
    metal:
        Metal the report is for.  ``"ALL"`` aggregates across all
        metals.
    n_records:
        Total number of TrajectoryRecord entries for this metal.
    n_seeds:
        Distinct seed values observed.
    config_hashes:
        Distinct config_hash values (one per unique HP signature).
    score_mean, score_std, score_min, score_max:
        Distribution of ``final_test_score`` across records.
    is_low_coverage:
        True iff ``n_records < LOW_COVERAGE_THRESHOLD``.  Per §4 R1,
        these cells get pessimistic-shrunk predictions.
    """

    metal: str
    n_records: int = 0
    n_seeds: int = 0
    config_hashes: List[str] = field(default_factory=list)
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    is_low_coverage: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metal": self.metal,
            "n_records": int(self.n_records),
            "n_seeds": int(self.n_seeds),
            "config_hashes": list(self.config_hashes),
            "score_mean": float(self.score_mean),
            "score_std": float(self.score_std),
            "score_min": float(self.score_min),
            "score_max": float(self.score_max),
            "is_low_coverage": bool(self.is_low_coverage),
            "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
        }


@dataclass
class ScoreEstimate:
    """Result of :meth:`ReplayPool.replay_predict`.

    Carries both the point estimate and the provenance (which records
    were used, what confidence the cell has) so callers can audit
    before promoting a replay-only decision.

    Attributes
    ----------
    score:
        Predicted final_test_score for the (hp_delta, metal) cell.
        On low-coverage cells, this is the pessimistically-shrunk
        value (see :func:`pessimistic_shrinkage`).
    confidence:
        ``"low"`` iff the metal is LOW_COVERAGE — caller MUST verify
        before promotion.  ``"medium"`` when n_records >= threshold
        but no exact-hp match.  ``"high"`` when an exact-hp-dict match
        exists in the pool.
    n_neighbors:
        Number of pool records the prediction was averaged across.
    baseline:
        The shrinkage baseline used (default 0.5 — midpoint of AUC).
    shrinkage_weight:
        Weight applied to the replay prediction in
        ``coverage / threshold * replay + (1 - …) * baseline``.
        Zero on empty pools; ``>= 1.0`` (i.e. no shrinkage) once the
        cell has at least ``threshold`` records.
    supporting_records:
        List of ``config_hash`` strings that contributed.  Useful for
        audit reports.
    """

    score: float
    confidence: str = "low"
    n_neighbors: int = 0
    baseline: float = _DEFAULT_BASELINE
    shrinkage_weight: float = 0.0
    supporting_records: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "confidence": str(self.confidence),
            "n_neighbors": int(self.n_neighbors),
            "baseline": float(self.baseline),
            "shrinkage_weight": float(self.shrinkage_weight),
            "supporting_records": list(self.supporting_records),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_hash_from_hp(hp_dict: Mapping[str, Any]) -> str:
    """Stable 16-char hash of an hp_dict.

    Used by callers that don't pre-compute the hash.  Sort_keys=True
    so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce the same
    hash.
    """
    import hashlib

    blob = json.dumps(dict(hp_dict), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _hp_distance(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> float:
    """L1 distance over the union of keys, treating missing keys as 0.

    Cheap, deterministic, and good-enough for HP ranking when there are
    <500 trajectories.  All numeric values are scaled to [0, 1] by
    dividing by max(|a|, |b|, 1) so LR=1e-4 vs LR=1e-3 contributes a
    finite, comparable distance rather than dominating via raw scale.
    """
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    dist = 0.0
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            scale = max(abs(float(va)), abs(float(vb)), 1.0)
            dist += abs(float(va) - float(vb)) / scale
        elif va != vb:
            dist += 1.0
    return dist


def pessimistic_shrinkage(
    replay_pred: float,
    n_records: int,
    threshold: int = LOW_COVERAGE_THRESHOLD,
    baseline: float = _DEFAULT_BASELINE,
) -> Tuple[float, float]:
    """Apply §4 R1 pessimistic shrinkage to a replay prediction.

    The cell-coverage factor is ``min(n_records / threshold, 1.0)`` so
    a cell with exactly ``threshold`` records gets full weight (no
    shrinkage) and a cell with 0 records collapses to ``baseline``.

    Returns
    -------
    (shrunk_score, weight)
        ``shrunk_score`` is the bias-corrected prediction;
        ``weight`` is the replay-prediction weight (so callers can
        log it for audit).
    """
    if n_records <= 0:
        return float(baseline), 0.0
    weight = min(float(n_records) / float(max(threshold, 1)), 1.0)
    shrunk = weight * float(replay_pred) + (1.0 - weight) * float(baseline)
    return shrunk, weight


# ---------------------------------------------------------------------------
# ReplayPool
# ---------------------------------------------------------------------------


class ReplayPool:
    """Append-only trajectory store with cheap off-policy queries.

    The pool is the single source of truth for the Dream-RSI
    primitives A1/A3/A6:

    * :meth:`add`            — append a finished trajectory.
    * :meth:`query`          — filter & project records to a dict-list
      (lightweight pandas-style API; no pandas import required).
    * :meth:`coverage_report`— per-metal diagnostics (P0.4).
    * :meth:`replay_predict` — A1: nearest-neighbour + pessimistic
      shrinkage for a candidate ``hp_delta`` on a given ``metal``.
    * :meth:`per_metal_bootstrap` — A3: leave-one-out within-metal
      estimator of (candidate_hp → expected final score).
    * :meth:`disagreement_uncertainty` — A6: variance of replay
      predictions across ``K`` sampled HP perturbations as a proxy
      for epistemic uncertainty.
    * :meth:`save` / :meth:`load` — JSON-lines persistence (one record
      per line) for round-to-round carry-over.

    The pool is intentionally CPU-only and zero-dependency (no pandas,
    no numpy, no RDKit).  Tests should run on any CI container in
    <500 ms.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        low_coverage_threshold: int = LOW_COVERAGE_THRESHOLD,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1; got {max_size}")
        if low_coverage_threshold < 1:
            raise ValueError(
                f"low_coverage_threshold must be >= 1; got {low_coverage_threshold}"
            )
        self.max_size = int(max_size)
        self.low_coverage_threshold = int(low_coverage_threshold)
        self._records: List[TrajectoryRecord] = []
        self._by_metal: Dict[str, List[int]] = defaultdict(list)
        self._n_added: int = 0
        self._n_evicted: int = 0

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def n_added(self) -> int:
        return self._n_added

    @property
    def n_evicted(self) -> int:
        return self._n_evicted

    def metals(self) -> List[str]:
        """Distinct metal labels currently in the pool."""
        return sorted(self._by_metal.keys())

    # ------------------------------------------------------------------
    # Append + eviction
    # ------------------------------------------------------------------
    def add(self, record: TrajectoryRecord) -> None:
        """Append a trajectory.  FIFO-evicts the oldest record if full."""
        if not isinstance(record, TrajectoryRecord):
            raise TypeError(
                f"record must be TrajectoryRecord; "
                f"got {type(record).__name__}"
            )
        if len(self._records) >= self.max_size:
            evicted = self._records.pop(0)
            # Remove evicted from the per-metal index.  Linear scan is
            # fine because we only evict one record at a time and the
            # pool is bounded by ``max_size``.
            bucket = self._by_metal.get(evicted.metal, [])
            if bucket:
                bucket.pop(0)
                if not bucket:
                    del self._by_metal[evicted.metal]
            self._n_evicted += 1
        self._records.append(record)
        self._by_metal[record.metal].append(len(self._records) - 1)
        self._n_added += 1

    def extend(self, records: Iterable[TrajectoryRecord]) -> int:
        """Bulk-add; returns the number of records actually added."""
        n_before = self._n_added
        for r in records:
            self.add(r)
        return self._n_added - n_before

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        metal: Optional[str] = None,
        config_hash: Optional[str] = None,
        seed: Optional[int] = None,
        hp_subset: Optional[Mapping[str, Any]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter the pool; return each surviving record as a dict.

        All filters are AND-combined.  ``hp_subset`` matches records
        whose ``hp_dict`` has every key in ``hp_subset`` with the
        exact same value (cheap subset match; deep equality is not
        performed on non-scalar values).
        """
        out: List[Dict[str, Any]] = []
        for r in self._records:
            if metal is not None and r.metal != metal:
                continue
            if config_hash is not None and r.config_hash != config_hash:
                continue
            if seed is not None and r.seed != seed:
                continue
            if min_score is not None and r.final_test_score < min_score:
                continue
            if max_score is not None and r.final_test_score > max_score:
                continue
            if hp_subset is not None:
                if not all(
                    k in r.hp_dict and r.hp_dict[k] == v
                    for k, v in hp_subset.items()
                ):
                    continue
            out.append(r.to_dict())
        return out

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    def coverage_report(self, metal: Optional[str] = None) -> CoverageStats:
        """Per-metal coverage diagnostics.

        When ``metal`` is None, returns the aggregate (``metal="ALL"``)
        stats.  When ``metal`` is specified, returns the per-metal
        slice.
        """
        if metal is None:
            return self._aggregate_report()
        return self._per_metal_report(metal)

    def _per_metal_report(self, metal: str) -> CoverageStats:
        bucket = self._by_metal.get(metal, [])
        if not bucket:
            return CoverageStats(
                metal=metal,
                n_records=0,
                n_seeds=0,
                config_hashes=[],
                score_mean=0.0,
                score_std=0.0,
                score_min=0.0,
                score_max=0.0,
                is_low_coverage=True,
            )
        records = [self._records[i] for i in bucket]
        scores = [r.final_test_score for r in records]
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
        std = math.sqrt(var)
        return CoverageStats(
            metal=metal,
            n_records=n,
            n_seeds=len({r.seed for r in records}),
            config_hashes=sorted({r.config_hash for r in records}),
            score_mean=mean,
            score_std=std,
            score_min=min(scores),
            score_max=max(scores),
            is_low_coverage=n < self.low_coverage_threshold,
        )

    def _aggregate_report(self) -> CoverageStats:
        if not self._records:
            return CoverageStats(metal="ALL", is_low_coverage=True)
        scores = [r.final_test_score for r in self._records]
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
        std = math.sqrt(var)
        return CoverageStats(
            metal="ALL",
            n_records=n,
            n_seeds=len({r.seed for r in self._records}),
            config_hashes=sorted({r.config_hash for r in self._records}),
            score_mean=mean,
            score_std=std,
            score_min=min(scores),
            score_max=max(scores),
            is_low_coverage=n < self.low_coverage_threshold,
        )

    def coverage_table(self) -> Dict[str, Dict[str, Any]]:
        """Per-metal + aggregate coverage as a JSON-serialisable dict.

        Convenience for ``MASTER.md`` reports.
        """
        out: Dict[str, Dict[str, Any]] = {"ALL": self._aggregate_report().to_dict()}
        for m in sorted(self._by_metal.keys()):
            out[m] = self._per_metal_report(m).to_dict()
        return out

    # ------------------------------------------------------------------
    # A1 — replay_predict
    # ------------------------------------------------------------------
    def replay_predict(
        self,
        hp_delta: Mapping[str, Any],
        metal: str,
        baseline: float = _DEFAULT_BASELINE,
        k: int = 5,
    ) -> ScoreEstimate:
        """Cheap off-policy score estimate for a candidate HP delta.

        Algorithm
        ---------
        1. Filter the pool to ``metal``.
        2. If empty → return ``ScoreEstimate(baseline, "low", 0, ...)``.
        3. Compute L1 distance from ``hp_delta`` to every record's
           ``hp_dict``; pick the ``k`` nearest (ties broken by larger
           ``final_test_score`` — assuming higher is better).
        4. Mean their ``final_test_score``.
        5. Apply :func:`pessimistic_shrinkage` per §4 R1.

        The returned ``confidence`` is:

        * ``"high"``   — at least one neighbour with exact-equality
          ``hp_dict`` (post-shim with the same key/value pairs).
        * ``"medium"`` — pool has ≥ ``low_coverage_threshold`` records
          for the metal but no exact match.
        * ``"low"``    — fewer than ``low_coverage_threshold`` records
          (shrinkage applied).

        Returns
        -------
        ScoreEstimate
        """
        bucket = self._by_metal.get(metal, [])
        if not bucket:
            return ScoreEstimate(
                score=float(baseline),
                confidence="low",
                n_neighbors=0,
                baseline=float(baseline),
                shrinkage_weight=0.0,
                supporting_records=[],
            )
        candidates: List[Tuple[float, float, str, TrajectoryRecord]] = []
        target = dict(hp_delta)
        exact_match = False
        for idx in bucket:
            r = self._records[idx]
            dist = _hp_distance(target, r.hp_dict)
            # Detect exact match on every (target) key.
            if target and all(
                k in r.hp_dict and r.hp_dict[k] == v for k, v in target.items()
            ):
                exact_match = True
            candidates.append((dist, -float(r.final_test_score), r.config_hash, r))
        # Sort: smallest distance first; on tie, larger score first
        # (negated score keeps ascending-sort tiebreaker descending).
        candidates.sort(key=lambda t: (t[0], t[1]))
        top = candidates[: max(1, int(k))]
        mean_score = sum(r.final_test_score for *_, r in top) / len(top)
        n_records = len(bucket)
        shrunk, weight = pessimistic_shrinkage(
            mean_score,
            n_records=n_records,
            threshold=self.low_coverage_threshold,
            baseline=float(baseline),
        )
        if exact_match:
            confidence = "high"
        elif n_records >= self.low_coverage_threshold:
            confidence = "medium"
        else:
            confidence = "low"
        return ScoreEstimate(
            score=float(shrunk),
            confidence=confidence,
            n_neighbors=len(top),
            baseline=float(baseline),
            shrinkage_weight=float(weight),
            supporting_records=[c[2] for c in top],
        )

    # ------------------------------------------------------------------
    # A3 — per_metal_bootstrap
    # ------------------------------------------------------------------
    def per_metal_bootstrap(
        self,
        hp_delta: Mapping[str, Any],
        metal: str,
        baseline: float = _DEFAULT_BASELINE,
        k: int = 5,
    ) -> ScoreEstimate:
        """Leave-one-out within-metal estimator (A3).

        For each record ``r`` in the metal bucket, predict the held-out
        ``r.final_test_score`` from the remaining ``k`` nearest
        neighbours of ``r.hp_dict`` (within the same metal).  Return
        the mean held-out prediction as the score estimate.

        This is the A3 primitive used by
        :mod:`molmetal.scripts.r4_dmpnn_replay_sweep` to validate the
        pool before committing a real run: if leave-one-out MAE on
        ``final_test_score`` is <0.05 (or analogous threshold for the
        metric), the pool is trustworthy for that metal; otherwise the
        sweep should bail per TODO-150 §P1.3.

        Notes
        -----
        * If the metal bucket has fewer than 2 records, falls back to
          :meth:`replay_predict` so the call is always well-defined.
        * Confidence semantics match :meth:`replay_predict` —
          ``"high"`` requires an exact-match neighbour.
        """
        bucket = self._by_metal.get(metal, [])
        if len(bucket) < 2:
            return self.replay_predict(
                hp_delta=hp_delta, metal=metal, baseline=baseline, k=k
            )
        # Collect all hp_dict + final_test_score for the metal.
        items: List[Tuple[Dict[str, Any], float]] = [
            (self._records[i].hp_dict, self._records[i].final_test_score)
            for i in bucket
        ]
        held_out_predictions: List[float] = []
        for held_idx in range(len(items)):
            held_hp, held_score = items[held_idx]
            # k nearest neighbours of held_hp among the rest.
            others = [
                (items[j][0], items[j][1])
                for j in range(len(items))
                if j != held_idx
            ]
            others.sort(key=lambda t: _hp_distance(held_hp, t[0]))
            top = others[: max(1, int(k))]
            if not top:
                held_out_predictions.append(float(baseline))
                continue
            held_out_predictions.append(
                sum(s for _, s in top) / len(top)
            )
        mean_estimate = sum(held_out_predictions) / len(held_out_predictions)
        n_records = len(bucket)
        shrunk, weight = pessimistic_shrinkage(
            mean_estimate,
            n_records=n_records,
            threshold=self.low_coverage_threshold,
            baseline=float(baseline),
        )
        # Detect exact match on ``hp_delta`` vs any record in the metal.
        target = dict(hp_delta)
        exact_match = any(
            target
            and all(
                k in self._records[bucket[i]].hp_dict
                and self._records[bucket[i]].hp_dict[k] == v
                for k, v in target.items()
            )
            for i in range(len(bucket))
        )
        if exact_match:
            confidence = "high"
        elif n_records >= self.low_coverage_threshold:
            confidence = "medium"
        else:
            confidence = "low"
        return ScoreEstimate(
            score=float(shrunk),
            confidence=confidence,
            n_neighbors=len(bucket) - 1,
            baseline=float(baseline),
            shrinkage_weight=float(weight),
            supporting_records=[
                self._records[bucket[i]].config_hash
                for i in range(min(len(bucket), k))
            ],
        )

    # ------------------------------------------------------------------
    # A6 — replay_disagreement
    # ------------------------------------------------------------------
    def disagreement_uncertainty(
        self,
        hp_delta: Mapping[str, Any],
        metal: str,
        k_perturbations: int = 5,
        seed: int = 0,
        baseline: float = _DEFAULT_BASELINE,
    ) -> float:
        """A6: variance of replay predictions under K small perturbations.

        The Dream-RSI paper uses replay-disagreement as a cheap
        epistemic-uncertainty estimator (variance of replay outcomes
        across alternative sampling policies).  Here we approximate
        "alternative sampling policies" by adding bounded Gaussian noise
        (relative scale 0.1) to numeric hp values and rerunning
        :meth:`replay_predict` for each perturbation.

        Returns
        -------
        float
            Standard deviation of the K perturbation predictions.
            Zero on an empty pool; clamped to be non-negative.
        """
        import random as _random

        bucket = self._by_metal.get(metal, [])
        if not bucket:
            return 0.0
        rng = _random.Random(int(seed))
        scores: List[float] = []
        target = dict(hp_delta)
        for _ in range(max(1, int(k_perturbations))):
            perturbed: Dict[str, Any] = {}
            for k, v in target.items():
                if isinstance(v, (int, float)):
                    scale = max(abs(float(v)) * 0.1, 1e-3)
                    perturbed[k] = float(v) + rng.gauss(0.0, scale)
                else:
                    perturbed[k] = v
            est = self.replay_predict(
                hp_delta=perturbed, metal=metal, baseline=baseline, k=5
            )
            scores.append(est.score)
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return max(0.0, math.sqrt(var))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> str:
        """Persist the pool as JSON-lines.  Returns the path written."""
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(r.to_dict(), sort_keys=True))
                fh.write("\n")
        return path

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        max_size: int = 10_000,
        low_coverage_threshold: int = LOW_COVERAGE_THRESHOLD,
    ) -> "ReplayPool":
        """Load a JSON-lines pool from disk."""
        pool = cls(max_size=max_size, low_coverage_threshold=low_coverage_threshold)
        path = Path(path)
        if not path.exists():
            return pool
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pool.add(TrajectoryRecord.from_dict(d))
        return pool
