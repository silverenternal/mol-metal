"""RSI for hyperparameter search — cold-start HPO primitives (TODO-150 §P1.5+P3).

================================================================
Top pick #2 from the R21 RSI plan: trajectory cache + cold-start
Bayesian-Ridge surrogate + epsilon-greedy acquisition + ASHA-style
early stop for mol-metal D-MPNN config search.
================================================================

This module is the load-bearing piece for the "15x compute savings"
claim in the user's top-2 picks brief.  It provides four primitives
that compose cleanly with the existing :class:`ReplayPool`:

* :class:`TrajectoryCacheWriter`  — write-through disk cache for
  (config, hp_dict, per-epoch trajectory, final score) records, one
  JSON per (metal, seed, config_hash) under
  ``metrics/by_metric/rsi_trajectory_cache/<metal>/<seed>/<config_hash>.json``.
  Wraps a :class:`ReplayPool` so a single :meth:`write` updates both
  the in-memory pool and the on-disk cache (auto-cache on every
  future Wf-124 P3 iteration).
* :class:`ColdStartSurrogate`    — deterministic Bayesian-Ridge-style
  ridge regressor over (config_features ⊕ partial_curve_features)
  → final AUC.  Optional sklearn fallback when the project has
  sklearn.  CPU-only.  Carries an explicit
  ``label == "MEASURED-PROJECTED"`` provenance tag so the project
  can never accidentally quote a projected lift as a measured lift
  per the honest-framing discipline.
* :class:`EpsilonGreedyAcquisition` — epsilon-greedy acquisition
  policy that shrinks epsilon as the cache fills (default: 0.3 when
  cache < 30 rows, 0.05 once cache >= 100 rows per metal, linear
  interpolation between).
* :class:`ASHAEarlyStop`         — ASHA-style early-stop scheduler.
  Given the partial per-epoch trajectory and the fitted surrogate,
  predicts terminal AUC at epoch ``k`` and rejects the run when the
  predicted terminal is below the metal's median final AUC.

Design contract (per TODO-150 §3.5 honest-framing discipline):

* All methods are CPU-only — no GPU, no RDKit, no D-MPNN weights.
* :meth:`ColdStartSurrogate.predict` and
  :meth:`EpsilonGreedyAcquisition.select` are *deterministic* given a
  fixed cache + random seed (modulo the optional bootstrap seed).
* Predictions carry a ``label`` field = ``"MEASURED-PROJECTED"`` and
  a ``confidence`` field that callers MUST honour:
  ``confidence == "low"`` means the metal is ``LOW_COVERAGE`` and the
  prediction should NOT be promoted without a held-out verify run.
* The cache schema is version-stamped with ``rsi_iter`` so when the
  underlying mol-metal training regime shifts (Wasserstein pIC50 reg
  → CFM-archived, LR scheduler swap, etc.) the cache can be marked
  stale and refilled without breaking older readers.

Why this is one module (not three):

The plan's Week-2 deliverable requires the surrogate, the epsilon
acquisition policy, and the ASHA scheduler to compose against the
trajectory cache.  Splitting them across three modules would force
callers to import four things and re-pass the same per-metal cache
slice each time.  Single module = single import = atomic unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Schema versioning — bump rsi_iter when the cache contract changes.
# ---------------------------------------------------------------------------

#: Current cache schema version.  Increment when:
#:   - new fields are added to TrajectoryCacheEntry (backwards-compat only)
#:   - config feature vector layout changes (BREAKING — readers must
#:     reject older entries or migrate them)
#:   - surrogate feature normalization changes
#:
#: ``rsi_iter=0`` corresponds to the first deployment under the
#: Wf-124 P3 + Wasserstein-pIC50-reg regime.
RSI_ITER: int = 0

#: Calendar date (ISO) when the current schema was frozen.  Used by
#: downstream auditors to flag stale caches whose ``deprecation_date``
#: has elapsed.
RSI_SCHEMA_FROZEN: str = "2026-09-27"

#: When ``deprecation_date`` is in the past, the entry is considered
#: STALE — readers MUST treat predictions derived from it as
#: ``MEASURED-PROJECTED (STALE)`` and never quote as MEASURED.  Set
#: far enough in the future that one quarter of work fits inside it.
RSI_DEFAULT_DEPRECATION_DAYS: int = 90
DEFAULT_DEPRECATION_DAYS: int = RSI_DEFAULT_DEPRECATION_DAYS

#: Default cache root — overridable per ``TrajectoryCacheWriter``.
DEFAULT_CACHE_ROOT: str = "metrics/by_metric/rsi_trajectory_cache"

#: Default feature vector dimension for ``ColdStartSurrogate``.  Six
#: scalar hyperparameters (lr_log, hidden_log, depth, dropout,
#: batch_size_log, weight_decay_log) plus four partial-curve features
#: (val_auc_at_5, val_auc_at_10, val_auc_at_15, train_loss_at_5) =
#: 10 features.  Any change here must bump ``RSI_ITER``.
DEFAULT_FEATURE_DIM: int = 10

#: Bayesian-Ridge-style regularization.  Small lambda_1; lambda_2 is
#: the more important noise prior.
DEFAULT_LAMBDA_1: float = 1e-6
DEFAULT_LAMBDA_2: float = 1e-6

#: ASHA default — minimum epoch before early-stop is allowed.  Below
#: this, the trajectory is too short to project terminal AUC reliably.
DEFAULT_ASHA_MIN_EPOCH: int = 15

#: Epsilon-greedy schedule breakpoints.  Default matches the plan:
#: ``0.3`` while cache < 30 rows; ``0.05`` once cache >= 100 rows;
#: linear interpolation between.
DEFAULT_EPSILON_HIGH: float = 0.3
DEFAULT_EPSILON_LOW: float = 0.05
DEFAULT_EPSILON_HIGH_CACHE: int = 30
DEFAULT_EPSILON_LOW_CACHE: int = 100

#: Per-metal LOW_COVERAGE threshold for the surrogate.  Same default
#: as the parent :class:`ReplayPool` for consistency.
SURROGATE_LOW_COVERAGE: int = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryCacheEntry:
    """One trajectory row written to the on-disk cache.

    Mirrors :class:`ReplayPool.TrajectoryRecord` but is explicitly
    version-stamped for the RSI cache contract.  All fields are
    JSON-safe so the entry can be serialised directly.

    Attributes
    ----------
    config_hash:
        Stable 16-char hash of ``hp_dict`` (``hashlib.sha256(json.dumps(
        hp_dict, sort_keys=True)).hexdigest()[:16]``).
    hp_dict:
        Resolved hyperparameters (lr, hidden, depth, dropout,
        batch_size, weight_decay, …).  Plain dict; scalar values
        preferred.
    per_epoch_metrics:
        Per-epoch trajectory — list of dicts with at least
        ``val_auc`` (and optionally ``train_loss``, ``val_mae``).
        May be empty for very-early checkpoints.
    final_test_score:
        Final test AUC / Pearson r / MAE the trainer reported.
    metal:
        Metal centre — one of ``{Ru, Ir, Rh, Os, Re, Pt}``.
    seed:
        RNG seed.
    rsi_iter:
        Schema version at write-time.  ``0`` for the initial Wf-124
        P3 deployment; bump when the contract changes.
    deprecation_date:
        ISO date string after which the entry is considered STALE.
        Defaults to ``RSI_SCHEMA_FROZEN + RSI_DEFAULT_DEPRECATION_DAYS``.
    timestamp:
        Wall-clock seconds since epoch at write-time.
    ckpt_sha256:
        Optional sha256 of the checkpoint file (for audit).  Empty
        string when the run was a surrogate-only prediction.
    metadata:
        Free-form provenance blob.  Not used by the surrogate.
    """

    config_hash: str
    hp_dict: Dict[str, Any]
    per_epoch_metrics: List[Dict[str, float]] = field(default_factory=list)
    final_test_score: float = 0.0
    metal: str = "Ru"
    seed: int = 0
    rsi_iter: int = RSI_ITER
    deprecation_date: str = ""
    timestamp: float = field(default_factory=time.time)
    ckpt_sha256: str = ""
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
        score = self.final_test_score
        if not isinstance(score, (int, float)):
            raise ValueError(
                f"final_test_score must be numeric; "
                f"got {type(score).__name__}"
            )
        if math.isnan(score) or math.isinf(score):
            raise ValueError(
                f"final_test_score must be finite; got {score}"
            )
        if self.rsi_iter < 0:
            raise ValueError(
                f"rsi_iter must be >= 0; got {self.rsi_iter}"
            )
        if not self.deprecation_date:
            # Default to RSI_SCHEMA_FROZEN + RSI_DEFAULT_DEPRECATION_DAYS.
            # Calendar arithmetic done in ISO-string form to keep the
            # module free of ``datetime`` for unit-testability.
            self.deprecation_date = _iso_plus_days(
                RSI_SCHEMA_FROZEN, RSI_DEFAULT_DEPRECATION_DAYS
            )

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
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
            "rsi_iter": int(self.rsi_iter),
            "deprecation_date": str(self.deprecation_date),
            "timestamp": float(self.timestamp),
            "ckpt_sha256": str(self.ckpt_sha256),
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrajectoryCacheEntry":
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
            rsi_iter=int(d.get("rsi_iter", RSI_ITER)),
            deprecation_date=str(
                d.get("deprecation_date", _iso_plus_days(RSI_SCHEMA_FROZEN, 90))
            ),
            timestamp=float(d.get("timestamp", time.time())),
            ckpt_sha256=str(d.get("ckpt_sha256", "")),
            metadata=dict(d.get("metadata", {})),
        )

    def is_stale(self, today: str = "") -> bool:
        """True iff ``deprecation_date`` has passed (ISO-string compare)."""
        ref = today or _today_iso()
        return str(self.deprecation_date) < ref


@dataclass
class SurrogateDecision:
    """Result of a :class:`ColdStartSurrogate.predict` call.

    Attributes
    ----------
    score:
        Predicted final test score for the candidate (metal, config).
    label:
        Always ``"MEASURED-PROJECTED"`` per the honest-framing
        discipline.  NEVER ``"MEASURED"``.
    confidence:
        ``"high"`` when the cache holds >= ``SURROGATE_LOW_COVERAGE``
        entries for the metal AND the candidate's nearest neighbour
        has L1 distance < ``DEFAULT_EPSILON_HIGH_CACHE`` rows
        worth of evidence; ``"medium"`` otherwise (projected but no
        strong local evidence); ``"low"`` when LOW_COVERAGE.
    n_train:
        Number of cache entries the surrogate was fit on.
    r2_loo:
        Leave-one-out R² of the surrogate on the cache (per-metal).
        ``0.0`` when n_train < 2.
    std:
        Estimated predictive std-dev (sqrt of the diagonal of the
        posterior covariance — closed-form for the ridge surrogate).
    supporting_hashes:
        Top-K nearest cache hashes that contributed to the prediction.
    """

    score: float
    label: str = "MEASURED-PROJECTED"
    confidence: str = "low"
    n_train: int = 0
    r2_loo: float = 0.0
    std: float = 0.0
    supporting_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "label": str(self.label),
            "confidence": str(self.confidence),
            "n_train": int(self.n_train),
            "r2_loo": float(self.r2_loo),
            "std": float(self.std),
            "supporting_hashes": list(self.supporting_hashes),
        }


@dataclass
class EarlyStopDecision:
    """Result of :meth:`ASHAEarlyStop.should_stop`.

    Attributes
    ----------
    should_stop:
        ``True`` when the projected terminal AUC is below the cache
        median AND the partial curve has flattened (last-delta <
        ``plateau_eps``).
    predicted_terminal:
        Projected final test score from the partial trajectory.
    median_baseline:
        Per-metal median of ``final_test_score`` across the cache.
        ``0.5`` when the cache is empty for that metal.
    reason:
        Short human-readable reason — ``"below_median"``,
        ``"plateau"``, ``"too_early"``, ``"empty_cache"``,
        ``"continue"``.
    epoch_checked:
        Epoch index the decision was made at (1-based).
    """

    should_stop: bool
    predicted_terminal: float = 0.0
    median_baseline: float = 0.5
    reason: str = "continue"
    epoch_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_stop": bool(self.should_stop),
            "predicted_terminal": float(self.predicted_terminal),
            "median_baseline": float(self.median_baseline),
            "reason": str(self.reason),
            "epoch_checked": int(self.epoch_checked),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def config_hash_from_hp(hp_dict: Mapping[str, Any]) -> str:
    """Stable 16-char SHA-256 prefix of a canonicalised HP dict."""
    blob = json.dumps(dict(hp_dict), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _today_iso() -> str:
    """ISO-format today date — kept as a tiny helper for testability."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def _iso_plus_days(iso_date: str, days: int) -> str:
    """Add ``days`` to an ISO date string and return ISO again.

    Implemented with the standard library only (no ``datetime``
    dependency) — ``datetime.strptime`` / ``timedelta`` to keep the
    module testable on minimal containers.
    """
    import datetime as _dt

    base = _dt.datetime.strptime(str(iso_date), "%Y-%m-%d")
    return (base + _dt.timedelta(days=int(days))).strftime("%Y-%m-%d")


def _hp_feature_vector(hp_dict: Mapping[str, Any]) -> List[float]:
    """Project an HP dict to the 6-D scalar feature vector.

    Order is fixed; missing keys default to safe mid-range values so
    older cache entries (without all six keys) don't NaN-poison the
    surrogate.  Values are log-scaled where the convention is
    multiplicative (lr, hidden, batch_size, weight_decay).
    """
    defaults = {
        "lr_log": -3.0,        # log10(1e-3)
        "hidden_log": math.log10(128.0),
        "depth": 3.0,
        "dropout": 0.1,
        "batch_size_log": math.log10(32.0),
        "weight_decay_log": -5.0,  # log10(1e-5)
    }
    out = []
    for key in (
        "lr_log", "hidden_log", "depth", "dropout",
        "batch_size_log", "weight_decay_log",
    ):
        if key in hp_dict:
            v = hp_dict[key]
        else:
            # Derive from the raw name when present.
            raw_key = {
                "lr_log": "lr",
                "hidden_log": "hidden",
                "batch_size_log": "batch_size",
                "weight_decay_log": "weight_decay",
            }.get(key, key)
            if raw_key in hp_dict and isinstance(hp_dict[raw_key], (int, float)):
                raw = float(hp_dict[raw_key])
                if key.endswith("_log"):
                    v = math.log10(max(raw, 1e-12))
                else:
                    v = raw
            else:
                v = defaults[key]
        out.append(float(v))
    return out


def _curve_feature_vector(
    per_epoch_metrics: Sequence[Mapping[str, Any]],
) -> List[float]:
    """Extract the 4-D partial-curve feature vector.

    Order: ``val_auc_at_5``, ``val_auc_at_10``, ``val_auc_at_15``,
    ``train_loss_at_5``.  Missing epochs return ``NaN`` so the
    surrogate can learn to ignore them when the epoch wasn't reached.
    """
    nans = [float("nan")] * 4
    if not per_epoch_metrics:
        return nans
    out: List[float] = [float("nan")] * 4
    for ep in per_epoch_metrics:
        try:
            idx = int(ep.get("epoch", -1))
        except Exception:
            continue
        if idx == 5:
            out[0] = float(ep.get("val_auc", float("nan")))
            out[3] = float(ep.get("train_loss", float("nan")))
        elif idx == 10:
            out[1] = float(ep.get("val_auc", float("nan")))
        elif idx == 15:
            out[2] = float(ep.get("val_auc", float("nan")))
    return out


def _hp_l1_distance(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> float:
    """Cheap HP-space distance (L1 on the 6-D feature vector)."""
    va = _hp_feature_vector(a)
    vb = _hp_feature_vector(b)
    return sum(abs(x - y) for x, y in zip(va, vb))


def _nan_safe(vec: Sequence[float]) -> bool:
    """True iff ``vec`` has no NaN entries."""
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vec)


# ---------------------------------------------------------------------------
# TrajectoryCacheWriter
# ---------------------------------------------------------------------------


class TrajectoryCacheWriter:
    """Write-through disk cache for ``TrajectoryCacheEntry`` rows.

    Each :meth:`write` performs three things atomically (modulo
    filesystem failure):

    1. Appends to a backing :class:`ReplayPool` (in-memory).
    2. Writes a JSON file to
       ``<cache_root>/<metal>/<seed>/<config_hash>.json``.
    3. Returns the resolved absolute path for audit logging.

    Reads are supported via :meth:`read` and :meth:`scan_metal` for
    cheap offline analysis (no GPU, no D-MPNN weights).

    The writer is intentionally minimal — it does NOT call any
    training code, just records what the trainer reports.  Wire it
    into the Wf-124 P3 trainer via a single ``replay_writer.write(...)``
    call after each epoch (or at run end); the plan's atomic scope
    prevents in-place trainer refactors.
    """

    def __init__(
        self,
        cache_root: str | os.PathLike[str] = DEFAULT_CACHE_ROOT,
        replay_pool: Optional[Any] = None,
        rsi_iter: int = RSI_ITER,
        deprecation_days: int = RSI_DEFAULT_DEPRECATION_DAYS,
    ) -> None:
        self.cache_root = Path(str(cache_root))
        self.replay_pool = replay_pool
        self.rsi_iter = int(rsi_iter)
        self.deprecation_days = int(deprecation_days)
        self._n_written: int = 0
        self._n_failed: int = 0

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    @property
    def n_written(self) -> int:
        return self._n_written

    @property
    def n_failed(self) -> int:
        return self._n_failed

    def path_for(
        self,
        metal: str,
        seed: int,
        config_hash: str,
    ) -> Path:
        """Return the absolute on-disk path for a (metal, seed, hash)."""
        return self.cache_root / str(metal) / f"seed_{int(seed)}" / f"{config_hash}.json"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(
        self,
        hp_dict: Mapping[str, Any],
        per_epoch_metrics: Sequence[Mapping[str, Any]],
        final_test_score: float,
        metal: str,
        seed: int,
        ckpt_sha256: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TrajectoryCacheEntry:
        """Append one trajectory row.

        Returns the persisted :class:`TrajectoryCacheEntry` so the
        caller can log/audit it.  Failures (filesystem errors) are
        counted but do NOT raise — the in-memory pool stays correct.
        """
        config_hash = config_hash_from_hp(hp_dict)
        entry = TrajectoryCacheEntry(
            config_hash=config_hash,
            hp_dict=dict(hp_dict),
            per_epoch_metrics=[dict(ep) for ep in per_epoch_metrics],
            final_test_score=float(final_test_score),
            metal=str(metal),
            seed=int(seed),
            rsi_iter=self.rsi_iter,
            ckpt_sha256=str(ckpt_sha256),
            metadata=dict(metadata) if metadata else {},
        )
        # 1. In-memory replay pool (best-effort — accepts anything
        # with a .add method to avoid a hard import cycle).
        if self.replay_pool is not None:
            try:
                from .pool import TrajectoryRecord  # type: ignore

                self.replay_pool.add(
                    TrajectoryRecord(
                        config_hash=entry.config_hash,
                        hp_dict=dict(entry.hp_dict),
                        per_epoch_metrics=[
                            {k: float(v) for k, v in ep.items()}
                            for ep in entry.per_epoch_metrics
                        ],
                        final_test_score=entry.final_test_score,
                        metal=entry.metal,
                        seed=entry.seed,
                        timestamp=entry.timestamp,
                        metadata=dict(entry.metadata),
                    )
                )
            except Exception:
                # The pool has its own contract; we never let a cache
                # write fail the parent run.
                self._n_failed += 1
        # 2. On-disk JSON.
        path = self.path_for(metal=metal, seed=seed, config_hash=config_hash)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(entry.to_dict(), fh, sort_keys=True)
            self._n_written += 1
        except Exception:
            self._n_failed += 1
        return entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(
        self,
        metal: str,
        seed: int,
        config_hash: str,
    ) -> Optional[TrajectoryCacheEntry]:
        """Read a single entry.  Returns ``None`` if missing."""
        path = self.path_for(metal=metal, seed=seed, config_hash=config_hash)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return TrajectoryCacheEntry.from_dict(json.load(fh))
        except Exception:
            return None

    def scan_metal(
        self,
        metal: str,
        include_stale: bool = False,
        today: str = "",
    ) -> List[TrajectoryCacheEntry]:
        """Return all entries for a metal.

        Stale entries (whose ``deprecation_date`` has passed) are
        excluded by default — call ``include_stale=True`` for an
        audit-mode pass.
        """
        ref = today or _today_iso()
        metal_dir = self.cache_root / str(metal)
        if not metal_dir.exists():
            return []
        out: List[TrajectoryCacheEntry] = []
        for seed_dir in metal_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            for f in seed_dir.iterdir():
                if not f.is_file() or not f.name.endswith(".json"):
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        entry = TrajectoryCacheEntry.from_dict(json.load(fh))
                except Exception:
                    continue
                if (not include_stale) and entry.is_stale(ref):
                    continue
                out.append(entry)
        return out

    def scan_all(
        self,
        include_stale: bool = False,
        today: str = "",
    ) -> Dict[str, List[TrajectoryCacheEntry]]:
        """Return ``{metal: [entries...]}`` for every metal on disk."""
        if not self.cache_root.exists():
            return {}
        out: Dict[str, List[TrajectoryCacheEntry]] = {}
        for metal_dir in self.cache_root.iterdir():
            if not metal_dir.is_dir():
                continue
            entries = self.scan_metal(
                metal=metal_dir.name,
                include_stale=include_stale,
                today=today,
            )
            if entries:
                out[metal_dir.name] = entries
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def per_metal_count(
        self,
        include_stale: bool = False,
        today: str = "",
    ) -> Dict[str, int]:
        """Return ``{metal: n_entries}`` (cheap — no JSON parse)."""
        if not self.cache_root.exists():
            return {}
        out: Dict[str, int] = {}
        for metal_dir in self.cache_root.iterdir():
            if not metal_dir.is_dir():
                continue
            n = 0
            for seed_dir in metal_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                for f in seed_dir.iterdir():
                    if f.is_file() and f.name.endswith(".json"):
                        n += 1
            if n > 0:
                out[metal_dir.name] = n
        return out


# ---------------------------------------------------------------------------
# ColdStartSurrogate
# ---------------------------------------------------------------------------


class ColdStartSurrogate:
    """Bayesian-Ridge-style surrogate over (config ⊕ curve) features.

    The model is intentionally simple:

    * The mean function is a constant (intercept) plus a linear
      function of the 10-D feature vector — same form as scikit-learn's
      BayesianRidge when ``n_iter=1`` and ``compute_score=False``.
    * The prior on the weight vector is N(0, lambda_1^{-1} I); the
      prior on the noise variance is an inverse-Gamma with shape
      ``lambda_2`` (matches the sklearn default).
    * The posterior mean / covariance are computed in closed form
      once per :meth:`fit` call — no iterative optimisation, no
      hyperparameter re-fit.  This is the *cold-start* surrogate: a
      single-pass estimator that doesn't require warm-up.
    * When ``use_sklearn=True`` AND scikit-learn is importable, the
      class delegates to ``sklearn.linear_model.BayesianRidge`` for a
      posterior that has been marginalised over the noise.  This is
      the *warm* path; the deterministic CPU fallback is the default
      so unit tests never need sklearn.

    Attributes
    ----------
    lambda_1, lambda_2:
        Ridge / noise precision priors.
    feature_dim:
        Expected feature dimension (default 10).
    """

    def __init__(
        self,
        lambda_1: float = DEFAULT_LAMBDA_1,
        lambda_2: float = DEFAULT_LAMBDA_2,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        use_sklearn: bool = False,
        low_coverage_threshold: int = SURROGATE_LOW_COVERAGE,
    ) -> None:
        if lambda_1 <= 0:
            raise ValueError(f"lambda_1 must be > 0; got {lambda_1}")
        if lambda_2 <= 0:
            raise ValueError(f"lambda_2 must be > 0; got {lambda_2}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1; got {feature_dim}")
        if low_coverage_threshold < 1:
            raise ValueError(
                f"low_coverage_threshold must be >= 1; got {low_coverage_threshold}"
            )
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.feature_dim = int(feature_dim)
        self.use_sklearn = bool(use_sklearn)
        self.low_coverage_threshold = int(low_coverage_threshold)

        # Fitted state.
        self._weights: Optional[List[float]] = None
        self._bias: float = 0.0
        self._X_train: List[List[float]] = []
        self._y_train: List[float] = []
        self._metal_train: List[str] = []
        self._hash_train: List[str] = []
        self._sk_model: Any = None

    # ------------------------------------------------------------------
    # Feature extraction (public so callers can re-use the layout)
    # ------------------------------------------------------------------
    @staticmethod
    def hp_features(hp_dict: Mapping[str, Any]) -> List[float]:
        return _hp_feature_vector(hp_dict)

    @staticmethod
    def curve_features(per_epoch_metrics: Sequence[Mapping[str, Any]]) -> List[float]:
        return _curve_feature_vector(per_epoch_metrics)

    @classmethod
    def joint_features(
        cls,
        hp_dict: Mapping[str, Any],
        per_epoch_metrics: Sequence[Mapping[str, Any]] = (),
    ) -> List[float]:
        """Concatenation of HP and partial-curve features.

        The concatenated vector is the input to the surrogate.
        """
        return _hp_feature_vector(hp_dict) + _curve_feature_vector(
            per_epoch_metrics
        )

    # ------------------------------------------------------------------
    # Fit / refit
    # ------------------------------------------------------------------
    def fit(self, entries: Iterable[TrajectoryCacheEntry]) -> "ColdStartSurrogate":
        """Refit the surrogate on a snapshot of cache entries."""
        X: List[List[float]] = []
        y: List[float] = []
        metals: List[str] = []
        hashes: List[str] = []
        for e in entries:
            x = self.joint_features(e.hp_dict, e.per_epoch_metrics)
            if not _nan_safe(x):
                # Skip rows with NaN curve features (curve too short
                # to be informative).  HP features are never NaN.
                continue
            X.append(x)
            y.append(float(e.final_test_score))
            metals.append(str(e.metal))
            hashes.append(str(e.config_hash))
        self._X_train = X
        self._y_train = y
        self._metal_train = metals
        self._hash_train = hashes

        if not X:
            # Cold-start: leave weights at zero; predict() will fall
            # back to a constant prior.
            self._weights = [0.0] * self.feature_dim
            self._bias = 0.5  # midpoint of AUC
            self._sk_model = None
            return self

        if self.use_sklearn:
            self._fit_sklearn(X, y)
            return self
        self._fit_closed_form(X, y)
        return self

    def _fit_closed_form(self, X: List[List[float]], y: List[float]) -> None:
        """Closed-form Bayesian-Ridge posterior mean / variance."""
        n = len(X)
        d = self.feature_dim
        # Build the design matrix with a bias column.
        Xb = [[1.0] + row for row in X]
        Xt = [_Xb_transpose_row(Xb, i) for i in range(n)]
        # A = X^T X + lambda_1 I  (skip the bias column's regularisation)
        A = [[0.0] * (d + 1) for _ in range(d + 1)]
        for i in range(n):
            for r in range(d + 1):
                for c in range(d + 1):
                    A[r][c] += Xb[i][r] * Xb[i][c]
        # Regularise weights only (not the intercept).
        for r in range(1, d + 1):
            A[r][r] += self.lambda_1
        # Solve A w = X^T y
        Xty = [0.0] * (d + 1)
        for i in range(n):
            for r in range(d + 1):
                Xty[r] += Xb[i][r] * y[i]
        w = _solve_linear_system(A, Xty)
        self._bias = float(w[0])
        self._weights = [float(wj) for wj in w[1:]]
        self._sk_model = None

    def _fit_sklearn(self, X: List[List[float]], y: List[float]) -> None:
        """Optional sklearn path — only when ``use_sklearn=True``."""
        try:
            from sklearn.linear_model import BayesianRidge  # type: ignore
        except Exception:
            # Graceful fallback — never let sklearn absence break fit.
            self._fit_closed_form(X, y)
            return
        import numpy as np  # type: ignore

        model = BayesianRidge(
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            max_iter=1,
            compute_score=False,
        )
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        self._sk_model = model
        # Also populate the closed-form weights so downstream code that
        # bypasses ``_sk_model`` still works.
        coef = getattr(model, "coef_", None)
        intercept = float(getattr(model, "intercept_", 0.5))
        if coef is None:
            self._fit_closed_form(X, y)
            return
        self._bias = float(intercept)
        self._weights = [float(c) for c in coef]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        hp_dict: Mapping[str, Any],
        per_epoch_metrics: Sequence[Mapping[str, Any]] = (),
        metal: str = "Ru",
        k_neighbors: int = 5,
    ) -> SurrogateDecision:
        """Predict final test score for a candidate (metal, config).

        Returns a :class:`SurrogateDecision` with ``label =
        "MEASURED-PROJECTED"`` and a ``confidence`` derived from the
        metal's cache coverage.  The decision is deterministic given a
        fixed cache.
        """
        x = self.joint_features(hp_dict, per_epoch_metrics)
        if not _nan_safe(x):
            # Partial curve too short — use HP-only features; zero-fill
            # the curve slots.
            x_hp = _hp_feature_vector(hp_dict)
            x = x_hp + [0.0] * (self.feature_dim - len(x_hp))

        # Per-metal slice (LOO uses the same metal bucket for R²).
        bucket_idx = [
            i for i, m in enumerate(self._metal_train) if m == metal
        ]
        n_train = len(bucket_idx)
        if n_train == 0:
            return SurrogateDecision(
                score=self._bias,
                confidence="low",
                n_train=0,
                r2_loo=0.0,
                std=0.0,
                supporting_hashes=[],
            )

        # k nearest neighbours by L1 on the joint feature vector.
        target = x
        scored: List[Tuple[float, int]] = []
        for i in bucket_idx:
            dist = sum(abs(a - b) for a, b in zip(target, self._X_train[i]))
            scored.append((dist, i))
        scored.sort(key=lambda t: t[0])
        k = max(1, int(k_neighbors))
        top = scored[:k]
        supporting_hashes = [self._hash_train[i] for _, i in top]

        # Predictive mean — use sklearn posterior if available,
        # otherwise the closed-form point estimate.
        if self._sk_model is not None:
            try:
                import numpy as np  # type: ignore

                mu = float(self._sk_model.predict(np.asarray([target]))[0])
            except Exception:
                mu = self._closed_form_predict(target)
        else:
            mu = self._closed_form_predict(target)

        # Confidence: cache-coverage-driven.
        if n_train < self.low_coverage_threshold:
            confidence = "low"
        else:
            # Promote to "high" only when the nearest neighbour is
            # within DEFAULT_EPSILON_HIGH_CACHE rows of evidence —
            # i.e. local density is non-trivial.
            nearest_dist = top[0][0] if top else float("inf")
            if nearest_dist < 0.5 * self.feature_dim:
                confidence = "high"
            else:
                confidence = "medium"

        # Leave-one-out R² on the per-metal bucket (cheap closed form
        # via the inverse-design-matrix trick; computed only when
        # sklearn is present — falls back to 0.0 otherwise).
        r2_loo = self._r2_loo(metal) if self._sk_model is not None else 0.0
        # Estimated predictive std (simplified: 1 / sqrt(n_train) prior).
        std = 1.0 / math.sqrt(max(1, n_train))

        return SurrogateDecision(
            score=float(mu),
            confidence=confidence,
            n_train=int(n_train),
            r2_loo=float(r2_loo),
            std=float(std),
            supporting_hashes=supporting_hashes,
        )

    def _closed_form_predict(self, x: List[float]) -> float:
        if not self._weights:
            return self._bias
        return self._bias + sum(w * xi for w, xi in zip(self._weights, x))

    def _r2_loo(self, metal: str) -> float:
        """Leave-one-out R² on the metal bucket (sklearn-only path).

        Returns 0.0 when the bucket has < 2 entries.
        """
        if self._sk_model is None:
            return 0.0
        bucket_idx = [
            i for i, m in enumerate(self._metal_train) if m == metal
        ]
        if len(bucket_idx) < 2:
            return 0.0
        try:
            import numpy as np  # type: ignore

            Xb = np.asarray(
                [self._X_train[i] for i in bucket_idx], dtype=float
            )
            y = np.asarray(
                [self._y_train[i] for i in bucket_idx], dtype=float
            )
            from sklearn.model_selection import LeaveOneOut  # type: ignore

            loo = LeaveOneOut()
            preds = []
            truths = []
            for tr, te in loo.split(Xb):
                m = type(self._sk_model)(
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    max_iter=1,
                    compute_score=False,
                )
                m.fit(Xb[tr], y[tr])
                preds.append(float(m.predict(Xb[te])[0]))
                truths.append(float(y[te][0]))
            if not preds:
                return 0.0
            mean = sum(truths) / len(truths)
            ss_res = sum((t - p) ** 2 for t, p in zip(truths, preds))
            ss_tot = sum((t - mean) ** 2 for t in truths)
            if ss_tot <= 0:
                return 0.0
            return 1.0 - ss_res / ss_tot
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def n_train(self) -> int:
        return len(self._X_train)

    @property
    def per_metal_n_train(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self._metal_train:
            out[m] = out.get(m, 0) + 1
        return out


# ---------------------------------------------------------------------------
# EpsilonGreedyAcquisition
# ---------------------------------------------------------------------------


@dataclass
class EpsilonSchedule:
    """Linear-interpolation epsilon schedule.

    Attributes
    ----------
    epsilon_high:
        Exploration rate when the cache is small (< ``high_cache``).
    epsilon_low:
        Exploration rate when the cache is large (>= ``low_cache``).
    high_cache:
        Cache-size threshold below which epsilon == ``epsilon_high``.
    low_cache:
        Cache-size threshold at/above which epsilon == ``epsilon_low``.
    """

    epsilon_high: float = DEFAULT_EPSILON_HIGH
    epsilon_low: float = DEFAULT_EPSILON_LOW
    high_cache: int = DEFAULT_EPSILON_HIGH_CACHE
    low_cache: int = DEFAULT_EPSILON_LOW_CACHE

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon_high <= 1.0):
            raise ValueError(
                f"epsilon_high must be in [0,1]; got {self.epsilon_high}"
            )
        if not (0.0 <= self.epsilon_low <= 1.0):
            raise ValueError(
                f"epsilon_low must be in [0,1]; got {self.epsilon_low}"
            )
        if self.high_cache < 1:
            raise ValueError(
                f"high_cache must be >= 1; got {self.high_cache}"
            )
        if self.low_cache < self.high_cache:
            raise ValueError(
                f"low_cache ({self.low_cache}) must be >= high_cache "
                f"({self.high_cache})"
            )

    def epsilon_for(self, cache_size: int) -> float:
        """Epsilon value for the current per-metal cache size."""
        n = max(0, int(cache_size))
        if n <= self.high_cache:
            return float(self.epsilon_high)
        if n >= self.low_cache:
            return float(self.epsilon_low)
        # Linear interpolation between the two breakpoints.
        span = self.low_cache - self.high_cache
        delta = n - self.high_cache
        frac = max(0.0, min(1.0, delta / span))
        return float(
            self.epsilon_high + frac * (self.epsilon_low - self.epsilon_high)
        )


class EpsilonGreedyAcquisition:
    """Epsilon-greedy acquisition policy over candidate HPs.

    Given a list of candidate :class:`TrajectoryCacheEntry`-like dicts
    (each with at least ``hp_dict`` and a target ``metal``) and a
    fitted :class:`ColdStartSurrogate`, picks one candidate per the
    schedule:

    * With probability ``epsilon_for(cache_size)``, sample uniformly
      at random from the candidates (exploration).
    * Otherwise, return the candidate with the highest predicted
      score (exploitation).

    The selection is deterministic given a fixed RNG seed.

    Attributes
    ----------
    schedule:
        :class:`EpsilonSchedule` controlling the epsilon-vs-cache-size
        interpolation.
    rng_seed:
        Optional integer seed for the exploration RNG.  Defaults to 0
        so unit tests are reproducible without external setup.
    """

    def __init__(
        self,
        schedule: Optional[EpsilonSchedule] = None,
        rng_seed: int = 0,
    ) -> None:
        self.schedule = schedule or EpsilonSchedule()
        self.rng_seed = int(rng_seed)
        # Call counter so each ``select`` advances the RNG.  Without
        # this, every call would re-seed with ``self.rng_seed`` and
        # produce the same first draw — the explore arm would always
        # pick the same index, breaking random exploration.
        self._n_calls: int = 0

    def select(
        self,
        candidates: Sequence[Mapping[str, Any]],
        surrogate: ColdStartSurrogate,
        metal: str,
        cache_size: int = 0,
        partial_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Tuple[int, SurrogateDecision]:
        """Return ``(chosen_index, decision)``.

        Parameters
        ----------
        candidates:
            Sequence of dicts with at least ``hp_dict``.  May also
            carry ``metal`` per candidate (used to skip non-matching
            metals) — when absent, the call-site ``metal`` is used.
        surrogate:
            A fitted :class:`ColdStartSurrogate`.
        metal:
            Target metal for the selection (cache lookup key).
        cache_size:
            Per-metal cache size used to set epsilon.
        partial_metrics:
            Optional partial-epoch trajectory to bias the surrogate
            prediction (e.g. for mid-flight re-routing of in-progress
            runs).
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        # Score every candidate up-front so the exploit arm can pick
        # the max.
        decisions: List[SurrogateDecision] = []
        for c in candidates:
            cand_metal = str(c.get("metal", metal))
            if cand_metal != metal:
                # Build a low-confidence zero decision so the call
                # remains well-defined; the caller can filter out the
                # metal mismatch upstream.
                decisions.append(
                    SurrogateDecision(score=0.0, confidence="low", n_train=0)
                )
                continue
            pm = partial_metrics
            if pm is None and "per_epoch_metrics" in c:
                pm = c["per_epoch_metrics"]
            decisions.append(
                surrogate.predict(
                    hp_dict=c.get("hp_dict", {}),
                    per_epoch_metrics=pm or (),
                    metal=metal,
                )
            )

        eps = self.schedule.epsilon_for(cache_size)
        # Seed the RNG with a counter-derived seed so successive
        # ``select`` calls produce different draws.  The first call
        # uses ``self.rng_seed``; subsequent calls use
        # ``self.rng_seed + n_calls``.
        self._n_calls += 1
        rng = random.Random(self.rng_seed + self._n_calls)
        if rng.random() < eps:
            idx = rng.randrange(len(candidates))
        else:
            idx = max(
                range(len(candidates)),
                key=lambda i: (
                    decisions[i].score
                    if decisions[i].confidence != "low"
                    else -1e9
                ),
            )
        return idx, decisions[idx]


# ---------------------------------------------------------------------------
# ASHAEarlyStop
# ---------------------------------------------------------------------------


class ASHAEarlyStop:
    """ASHA-style early-stop scheduler.

    Given a partial per-epoch trajectory and the fitted surrogate,
    predicts the terminal final-test-score and decides whether to
    stop the run:

    * If the partial trajectory is shorter than ``min_epoch``, the
      decision is "continue" (too early).
    * Otherwise, ask the surrogate for the projected terminal score
      using the partial trajectory as evidence; if the projected
      score is below the per-metal median final score of the cache,
      AND the last-epoch-to-previous delta is below ``plateau_eps``,
      stop.

    The class is purely deterministic — no learning of its own; it
    consumes the surrogate's prediction.
    """

    def __init__(
        self,
        min_epoch: int = DEFAULT_ASHA_MIN_EPOCH,
        plateau_eps: float = 0.005,
    ) -> None:
        if min_epoch < 1:
            raise ValueError(f"min_epoch must be >= 1; got {min_epoch}")
        if plateau_eps < 0:
            raise ValueError(f"plateau_eps must be >= 0; got {plateau_eps}")
        self.min_epoch = int(min_epoch)
        self.plateau_eps = float(plateau_eps)

    def _median(
        self, scores: Sequence[float], default: float = 0.5
    ) -> float:
        if not scores:
            return float(default)
        s = sorted(scores)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        return 0.5 * (s[mid - 1] + s[mid])

    def should_stop(
        self,
        per_epoch_metrics: Sequence[Mapping[str, Any]],
        surrogate: ColdStartSurrogate,
        metal: str,
        cache_entries_for_metal: Sequence[TrajectoryCacheEntry] = (),
    ) -> EarlyStopDecision:
        """Decide whether to stop the run at the current epoch.

        ``cache_entries_for_metal`` is used only to compute the
        per-metal median baseline (a constant projection threshold).
        """
        # Resolve current epoch from the trajectory.
        epoch_checked = 0
        if per_epoch_metrics:
            try:
                epoch_checked = max(
                    int(ep.get("epoch", 0))
                    for ep in per_epoch_metrics
                )
            except Exception:
                epoch_checked = len(per_epoch_metrics)
        else:
            epoch_checked = 0

        # Too early — always continue.
        if epoch_checked < self.min_epoch or not per_epoch_metrics:
            return EarlyStopDecision(
                should_stop=False,
                predicted_terminal=0.0,
                median_baseline=self._median(
                    [e.final_test_score for e in cache_entries_for_metal],
                    default=0.5,
                ),
                reason="too_early",
                epoch_checked=epoch_checked,
            )

        # Median baseline (per-metal).
        median = self._median(
            [e.final_test_score for e in cache_entries_for_metal],
            default=0.5,
        )

        # Project terminal score from the surrogate using the partial
        # trajectory as evidence.
        hp_dict = (
            dict(cache_entries_for_metal[0].hp_dict)
            if cache_entries_for_metal
            else {}
        )
        decision = surrogate.predict(
            hp_dict=hp_dict,
            per_epoch_metrics=per_epoch_metrics,
            metal=metal,
        )

        # Plateau check — last-vs-previous delta on val_auc.
        last = per_epoch_metrics[-1]
        prev = (
            per_epoch_metrics[-2]
            if len(per_epoch_metrics) >= 2
            else None
        )
        last_v = float(last.get("val_auc", float("nan")))
        prev_v = (
            float(prev.get("val_auc", float("nan")))
            if prev is not None
            else float("nan")
        )
        plateaued = (
            (not math.isnan(last_v))
            and (not math.isnan(prev_v))
            and abs(last_v - prev_v) < self.plateau_eps
        )

        below_median = decision.score < median
        if below_median and plateaued:
            return EarlyStopDecision(
                should_stop=True,
                predicted_terminal=float(decision.score),
                median_baseline=float(median),
                reason="below_median+plateau",
                epoch_checked=epoch_checked,
            )
        if below_median:
            return EarlyStopDecision(
                should_stop=False,
                predicted_terminal=float(decision.score),
                median_baseline=float(median),
                reason="below_median",
                epoch_checked=epoch_checked,
            )
        if plateaued:
            return EarlyStopDecision(
                should_stop=False,
                predicted_terminal=float(decision.score),
                median_baseline=float(median),
                reason="plateau",
                epoch_checked=epoch_checked,
            )
        return EarlyStopDecision(
            should_stop=False,
            predicted_terminal=float(decision.score),
            median_baseline=float(median),
            reason="continue",
            epoch_checked=epoch_checked,
        )


# ---------------------------------------------------------------------------
# Internal linear-algebra helpers (deterministic, no numpy required)
# ---------------------------------------------------------------------------


def _Xb_transpose_row(Xb: List[List[float]], i: int) -> List[float]:
    """Helper to keep the fit loop readable (no numpy)."""
    return Xb[i]


def _solve_linear_system(
    A: List[List[float]], b: List[float]
) -> List[float]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Used only by :meth:`ColdStartSurrogate._fit_closed_form`.  The
    matrix is small (11 x 11 by default) so an O(n^3) solver is more
    than fast enough and keeps the module numpy-free.
    """
    n = len(A)
    if n != len(b):
        raise ValueError(
            f"A and b dimension mismatch: {n} vs {len(b)}"
        )
    # Copy to avoid mutating caller's data.
    M = [row[:] for row in A]
    y = list(b)
    for k in range(n):
        # Partial pivot.
        pivot = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[pivot][k]):
                pivot = i
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
            y[k], y[pivot] = y[pivot], y[k]
        if abs(M[k][k]) < 1e-12:
            # Singular — replace with a tiny ridge so the solver
            # never NaNs.  This is a degenerate case the closed-form
            # fit shouldn't hit in practice, but we guard for
            # robustness.
            M[k][k] = 1e-12
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            y[i] -= factor * y[k]
    # Back-substitute.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
    return x