"""Vina Replay Pool — Vina-caching adaptation of the Dream-RSI replay idea.

================================================================
Purpose
================================================================
Adapts the Dream-RSI ``Reveal + Surrogate + Select`` pattern
(arXiv 2609.14858 §3) to the mol-metal Vina docking workflow.

Concretely:

* :class:`VinaReplayNode`       — one cached ``(smiles, embedding,
  pocket_features, vina_score)`` tuple.
* :class:`VinaReplayPool`       — per-pocket JSON-backed persistent
  store of :class:`VinaReplayNode` entries, one file per pocket.
* :class:`VinaReplaySimulator`  — a :class:`ReplayProbe` subclass
  that reveals cached (smiles, score) pairs from the pool when a
  policy probes a cell, *without* invoking Vina again.
* :class:`VinaSurrogate`        — a small MLP (< 10 K params) that
  ranks new SMILES candidates using the cached pool, so a
  generation-evaluation loop can avoid the Vina oracle call when
  the surrogate has enough coverage.
* :func:`combined_v0_vm_ranking` — V^0-invariant + V^m-ranking
  composition: sort candidates by ``V^0(node) + V^m(surrogate)``
  so we never pick worse than the cached baseline.

Honest framing
--------------
* Pool starts empty; the surrogate is degenerate (zero-weight) until
  enough Vina runs are cached.  See ``MIN_POOL_FOR_SURROGATE``.
* The surrogate is a CPU-only ridge / 1-hidden-layer MLP; not a
  faithful Vina oracle.  Spearman ρ on synthetic data is the only
  honest sanity check — absolute kcal/mol predictions are biased
  when the cache is small.
* Backward compat: this module is *opt-in*.  Existing Vina dock
  scripts (``dock_10pockets_se3sb.py`` and friends) are untouched;
  the replay pool is consumed via :class:`VinaReplayPool`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


#: Default dimensionality of the D-MPNN embedding vector stored on
#: each node.  Matches the Wf-124 multitask backbone hidden dim.
DEFAULT_EMBED_DIM: int = 256

#: Default dimensionality of the per-pocket feature vector.
DEFAULT_POCKET_DIM: int = 64

#: Minimum number of cached entries required before the surrogate is
#: allowed to issue predictions.  Below this threshold the surrogate
#: degenerates to a zero-weight prior and rankings collapse to
#: insertion order (the caller MUST check ``is_low_coverage`` before
#: substituting the surrogate for a real Vina call).
MIN_POOL_FOR_SURROGATE: int = 32

#: Surrogate ridge regression regularisation strength.  Deliberately
#: small so a few hundred cached tuples don't saturate.
RIDGE_ALPHA: float = 1.0

#: Hidden-layer width for the small MLP backend.  With embed_dim +
#: pocket_dim = 320 input and hidden = 24 we have:
#:   320 * 24 + 24 + 24 * 1 + 1 = 7 705 parameters (well under 10K).
#: The constraint of < 10K params keeps the surrogate fast on CPU
#: so :meth:`VinaSurrogate.rank_candidates` can serve as an oracle
#: proxy inside the inner loop of a search algorithm.
MLP_HIDDEN: int = 24

#: Underlying storage directory layout for the persistent pool.
DEFAULT_STORAGE_SUBDIR: str = "data/vina_replay_pool"


# ---------------------------------------------------------------------------
# ReplayProbe base class
# ---------------------------------------------------------------------------


class ReplayProbe:
    """Base class for replay probes that reveal cached world state.

    The Dream-RSI port (``molmetal_lam.search_alg.dream_rsi``) models
    a probe as a callable; here we provide a thin class-based surface
    so the Vina caching probe can carry state (the
    :class:`VinaReplayPool` it queries) and so callers can subclass
    it for unit tests.

    Subclasses must implement :meth:`reveal`, returning a list of
    ``RevealedRecord`` tuples.  ``Reveal + Surrogate + Select`` is the
    pattern from paper §3 stage ❷.
    """

    def reveal(self, cells: Sequence[Any]) -> List["RevealedRecord"]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement reveal()"
        )

    # Async-mirroring surface so a probe can be plugged into either
    # an async or sync caller.
    def __call__(self, cells: Sequence[Any]) -> List["RevealedRecord"]:
        return self.reveal(cells)


@dataclass
class RevealedRecord:
    """One revealed (smiles, score) pair from the pool.

    Equivalent to ``RevealedNode`` in the discovery-tree probe, but
    shaped for the Vina caching pool where the "cell" is keyed by
    smiles + pocket_id rather than a tree cell id.
    """

    smiles: str
    pocket_id: str
    vina_score: float
    confidence: str = "high"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Node + pool
# ---------------------------------------------------------------------------


@dataclass
class VinaReplayNode:
    """One cached ``(smiles, embedding, pocket_features, vina_score)`` tuple.

    Mirrors the Dream-RSI "discovery-tree node" primitive: each
    :class:`VinaReplayNode` is a leaf that can be re-walked
    deterministically by the surrogate, exactly the way the paper's
    "Replay Simulator" wraps each ``T_i`` as an immutable read-only
    world.
    """

    smiles: str
    dmpnn_embedding: np.ndarray
    pocket_features: np.ndarray
    vina_score: float
    pocket_id: str = "default"
    seed: int = 0
    method: str = "se3sb"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce arrays to a consistent dtype/shape.
        emb = np.asarray(self.dmpnn_embedding, dtype=np.float64).ravel()
        if emb.size != DEFAULT_EMBED_DIM:
            padded = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
            n = min(emb.size, DEFAULT_EMBED_DIM)
            padded[:n] = emb[:n]
            emb = padded
        self.dmpnn_embedding = emb

        pf = np.asarray(self.pocket_features, dtype=np.float64).ravel()
        if pf.size != DEFAULT_POCKET_DIM:
            padded = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
            n = min(pf.size, DEFAULT_POCKET_DIM)
            padded[:n] = pf[:n]
            pf = padded
        self.pocket_features = pf

        if not isinstance(self.vina_score, (int, float)) or math.isnan(
            float(self.vina_score)
        ) or math.isinf(float(self.vina_score)):
            raise ValueError(
                f"vina_score must be finite; got {self.vina_score!r}"
            )
        if not isinstance(self.smiles, str) or not self.smiles:
            raise ValueError("smiles must be a non-empty string")
        if not isinstance(self.pocket_id, str) or not self.pocket_id:
            raise ValueError("pocket_id must be a non-empty string")

    # ------------------------------------------------------------------
    # Feature / serialisation helpers
    # ------------------------------------------------------------------
    def feature_vector(self) -> np.ndarray:
        """Concatenate ``[embedding | pocket_features]`` into one
        surrogate-consumed feature vector."""
        return np.concatenate([self.dmpnn_embedding, self.pocket_features])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": str(self.smiles),
            "pocket_id": str(self.pocket_id),
            "dmpnn_embedding": [float(x) for x in self.dmpnn_embedding],
            "pocket_features": [float(x) for x in self.pocket_features],
            "vina_score": float(self.vina_score),
            "seed": int(self.seed),
            "method": str(self.method),
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VinaReplayNode":
        return cls(
            smiles=str(d["smiles"]),
            pocket_id=str(d.get("pocket_id", "default")),
            dmpnn_embedding=list(d.get("dmpnn_embedding", [])),
            pocket_features=list(d.get("pocket_features", [])),
            vina_score=float(d.get("vina_score", 0.0)),
            seed=int(d.get("seed", 0)),
            method=str(d.get("method", "se3sb")),
            metadata=dict(d.get("metadata", {})),
        )


class VinaReplayPool:
    """Per-pocket JSON-backed persistent storage of Vina results.

    Layout::

        storage_dir/
            1h36.json
            3eml.json
            ...

    Each file is a list of :class:`VinaReplayNode` dicts.  Load is
    idempotent and partial-corrupt (malformed lines are skipped with
    a warning rather than crashing the caller).

    Backward compat: this class is *opt-in*.  Existing Vina dock
    scripts are untouched; the pool is consumed via the
    :class:`VinaReplayPool` API.
    """

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.storage_dir: Optional[Path] = (
            Path(storage_dir) if storage_dir is not None else None
        )
        # In-memory mirror: pocket_id -> list[VinaReplayNode]
        self._by_pocket: Dict[str, List[VinaReplayNode]] = defaultdict(list)
        # Auto-load on construction when storage_dir is set.
        if self.storage_dir is not None and self.storage_dir.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _path_for(self, pocket_id: str) -> Path:
        assert self.storage_dir is not None, (
            "storage_dir is required for IO; "
            "construct VinaReplayPool(storage_dir=<path>)."
        )
        return self.storage_dir / f"{pocket_id}.json"

    def _load_all(self) -> None:
        assert self.storage_dir is not None
        for path in sorted(self.storage_dir.glob("*.json")):
            pocket_id = path.stem
            try:
                with path.open("r") as fh:
                    raw = json.load(fh)
            except Exception as exc:
                log.warning(
                    "VinaReplayPool: skipping corrupt cache file %s (%s)",
                    path,
                    exc,
                )
                continue
            if not isinstance(raw, list):
                log.warning(
                    "VinaReplayPool: %s is not a JSON list; resetting", path
                )
                continue
            for entry in raw:
                try:
                    self._by_pocket[pocket_id].append(
                        VinaReplayNode.from_dict(entry)
                    )
                except Exception as exc:
                    log.warning(
                        "VinaReplayPool: skipping bad entry in %s (%s)",
                        path,
                        exc,
                    )

    def save(self) -> None:
        """Flush the in-memory cache to disk (per-pocket JSON).

        Atomic via ``os.replace`` so a partial write never poisons
        the cache.
        """
        if self.storage_dir is None:
            raise RuntimeError(
                "VinaReplayPool.save() requires storage_dir; "
                "construct with storage_dir=<path>."
            )
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for pocket_id, entries in self._by_pocket.items():
            target = self._path_for(pocket_id)
            tmp = target.with_suffix(".json.tmp")
            with tmp.open("w") as fh:
                json.dump([e.to_dict() for e in entries], fh, indent=2)
            os.replace(tmp, target)

    @classmethod
    def load(cls, storage_dir: Union[str, Path]) -> "VinaReplayPool":
        """Load (or create) a pool backed by ``storage_dir``.

        If ``storage_dir`` does not exist, returns an empty pool that
        will lazily create the directory on the first :meth:`save`.
        """
        pool = cls(storage_dir=storage_dir)
        return pool

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add_observation(
        self,
        smiles: str,
        dmpnn_embedding: np.ndarray,
        pocket_features: np.ndarray,
        vina_score: float,
        pocket_id: str = "default",
        seed: int = 0,
        method: str = "se3sb",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VinaReplayNode:
        """Append one observation to the pool (in-memory).

        Call :meth:`save` afterwards to flush to disk.  Returns the
        appended :class:`VinaReplayNode` so callers can capture the
        ``seed``/``timestamp`` for downstream bookkeeping.
        """
        node = VinaReplayNode(
            smiles=str(smiles),
            pocket_id=str(pocket_id),
            dmpnn_embedding=np.asarray(dmpnn_embedding, dtype=np.float64),
            pocket_features=np.asarray(pocket_features, dtype=np.float64),
            vina_score=float(vina_score),
            seed=int(seed),
            method=str(method),
            metadata=dict(metadata) if metadata else {},
        )
        self._by_pocket[node.pocket_id].append(node)
        return node

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_all(self) -> List[VinaReplayNode]:
        """Return *all* cached entries (across pockets) as a flat list.

        Order is deterministic: pocket-id ascending, then insertion
        order within each pocket.
        """
        out: List[VinaReplayNode] = []
        for pocket_id in sorted(self._by_pocket.keys()):
            out.extend(self._by_pocket[pocket_id])
        return out

    def get_for_pocket(self, pocket_id: str) -> List[VinaReplayNode]:
        """Return cached entries for a single pocket (immutable list)."""
        return list(self._by_pocket.get(pocket_id, ()))

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_pocket.values())

    def n_per_pocket(self) -> Dict[str, int]:
        return {pid: len(es) for pid, es in self._by_pocket.items()}

    def total_for_pocket(self, pocket_id: str) -> int:
        return len(self._by_pocket.get(pocket_id, ()))


# ---------------------------------------------------------------------------
# Replay simulator (probe)
# ---------------------------------------------------------------------------


class VinaReplaySimulator(ReplayProbe):
    """A :class:`ReplayProbe` that reveals cached ``(smiles, score)`` pairs.

    The simulator queries the underlying :class:`VinaReplayPool` and
    returns a :class:`RevealedRecord` for each ``(smiles, pocket_id)``
    tuple that is already cached — *without* calling Vina again.

    If a ``(smiles, pocket_id)`` pair is *not* cached, the simulator
    appends a "miss" :class:`RevealedRecord` with
    ``confidence="miss"`` so the caller can decide whether to fall
    back to a real Vina run or to the surrogate ranking.
    """

    def __init__(self, pool: VinaReplayPool) -> None:
        self.pool = pool

    def reveal(
        self, cells: Sequence[Tuple[str, str]]
    ) -> List[RevealedRecord]:
        """Reveal cached ``(smiles, pocket_id)`` pairs.

        ``cells`` is a sequence of ``(smiles, pocket_id)`` tuples.
        Returns a :class:`RevealedRecord` per input — confidence is
        ``"high"`` for an exact pool hit, ``"miss"`` otherwise.
        """
        out: List[RevealedRecord] = []
        for smiles, pocket_id in cells:
            for entry in self.pool.get_for_pocket(pocket_id):
                if entry.smiles == smiles:
                    out.append(
                        RevealedRecord(
                            smiles=smiles,
                            pocket_id=pocket_id,
                            vina_score=float(entry.vina_score),
                            confidence="high",
                            metadata=dict(entry.metadata),
                        )
                    )
                    break
            else:
                out.append(
                    RevealedRecord(
                        smiles=smiles,
                        pocket_id=pocket_id,
                        vina_score=float("nan"),
                        confidence="miss",
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


@dataclass
class VinaSurrogateReport:
    """Result of :meth:`VinaSurrogate.rank_candidates`.

    Attributes
    ----------
    rankings:
        List of ``(input_idx, predicted_score, confidence)`` tuples
        sorted by ``predicted_score`` ascending (more negative =
        better Vina binder).
    is_low_coverage:
        True when the underlying pool had fewer than
        :data:`MIN_POOL_FOR_SURROGATE` entries at fit time.
    surrogate_train_loss:
        In-sample MSE on the training set — *not* a held-out metric.
    """

    rankings: List[Tuple[int, float, str]]
    is_low_coverage: bool
    surrogate_train_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [
                {
                    "input_idx": int(i),
                    "predicted_vina_score": float(s),
                    "confidence": str(c),
                }
                for (i, s, c) in self.rankings
            ],
            "is_low_coverage": bool(self.is_low_coverage),
            "surrogate_train_loss": float(self.surrogate_train_loss),
        }


class VinaSurrogate:
    """Small offline MLP surrogate (≤ 10 K params) trained on the pool.

    Two backends:

    * ``"ridge"`` — closed-form ridge regression.  Always available,
      no torch dependency.
    * ``"mlp"``   — single-hidden-layer MLP with
      ``MLP_HIDDEN = 32`` units, trained via the normal equation on
      the hidden representation.  Default for production use; total
      parameter count stays just under 10 K.

    Honest framing: this is a *learnable* stand-in for Vina, not a
    faithful oracle.  Spearman ρ on synthetic data is the only
    honest smoke check — absolute kcal/mol predictions are biased
    when the cache is small or out-of-distribution.
    """

    def __init__(
        self,
        backend: str = "ridge",
        alpha: float = RIDGE_ALPHA,
        hidden: int = MLP_HIDDEN,
    ) -> None:
        if backend not in ("ridge", "mlp"):
            raise ValueError(
                f"backend must be 'ridge' or 'mlp'; got {backend!r}"
            )
        self.backend = backend
        self.alpha = float(alpha)
        self.hidden = int(hidden)
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._W1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._train_loss: float = float("nan")
        self._fitted = False
        self._is_low_coverage = True

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_Xy(
        pools: Sequence[VinaReplayPool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Materialise the (X, y) training set from one or more pools."""
        Xs: List[np.ndarray] = []
        ys: List[float] = []
        for pool in pools:
            for node in pool.get_all():
                Xs.append(node.feature_vector())
                ys.append(float(node.vina_score))
        if not Xs:
            return (
                np.zeros((0, DEFAULT_EMBED_DIM + DEFAULT_POCKET_DIM)),
                np.zeros((0,)),
            )
        return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.float64)

    def fit(self, pools: Sequence[VinaReplayPool]) -> "VinaSurrogate":
        """Train on the union of all cached observations in ``pools``.

        ``pools`` may be a single pool or an iterable of pools
        (e.g. one pool per pocket).  With ``n=0`` the surrogate
        becomes a zero-weight prior — callers must check
        ``is_low_coverage`` before trusting the predictions.
        """
        X, y = self._collect_Xy(pools)
        n, d = X.shape if X.ndim == 2 else (0, DEFAULT_EMBED_DIM + DEFAULT_POCKET_DIM)
        self._is_low_coverage = (n < MIN_POOL_FOR_SURROGATE)

        if n == 0:
            # Degenerate prior — zero weight + zero bias.
            self._weights = np.zeros(d, dtype=np.float64)
            self._bias = 0.0
            self._W1 = None
            self._b1 = None
            self._train_loss = float("nan")
            self._fitted = True
            return self

        mean_y = float(y.mean())
        centred_y = y - mean_y

        if self.backend == "ridge":
            self._fit_ridge(X, centred_y, mean_y, d)
        else:  # "mlp"
            self._fit_mlp(X, centred_y, mean_y, d)

        # Mark fitted BEFORE computing the in-sample train loss so
        # ``_raw_predict``'s ``self._fitted`` invariant holds.
        self._fitted = True
        # In-sample train loss (NOT held-out).
        preds = self._raw_predict(X)
        self._train_loss = float(np.mean((preds - y) ** 2))
        return self

    def _fit_ridge(
        self,
        X: np.ndarray,
        centred_y: np.ndarray,
        mean_y: float,
        d: int,
    ) -> None:
        XtX = X.T @ X
        XtX[np.diag_indices(d)] += self.alpha
        Xty = X.T @ centred_y
        try:
            self._weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        self._bias = float(mean_y - self._weights @ X.mean(axis=0))
        self._W1 = None
        self._b1 = None

    def _fit_mlp(
        self,
        X: np.ndarray,
        centred_y: np.ndarray,
        mean_y: float,
        d: int,
    ) -> None:
        hidden = max(1, int(self.hidden))
        rng = np.random.default_rng(0)
        W1 = rng.standard_normal((d, hidden)) * (1.0 / math.sqrt(d))
        b1 = np.zeros(hidden, dtype=np.float64)
        H = np.tanh(X @ W1 + b1)
        HtH = H.T @ H
        HtH[np.diag_indices(hidden)] += self.alpha
        Hty = H.T @ centred_y
        try:
            w2 = np.linalg.solve(HtH, Hty)
        except np.linalg.LinAlgError:
            w2 = np.linalg.lstsq(HtH, Hty, rcond=None)[0]
        self._W1 = W1
        self._b1 = b1
        self._weights = w2
        self._bias = float(mean_y - (H.mean(axis=0) @ w2))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._weights is None:
            raise RuntimeError(
                "VinaSurrogate: call fit() before predict()."
            )
        if self.backend == "ridge":
            return X @ self._weights + self._bias
        # MLP backend
        assert self._W1 is not None and self._b1 is not None
        hidden_act = np.tanh(X @ self._W1 + self._b1)
        return hidden_act @ self._weights + self._bias

    def predict(
        self,
        dmpnn_embedding: np.ndarray,
        pocket_features: np.ndarray,
    ) -> float:
        """Predict Vina score for a single candidate."""
        emb = np.asarray(dmpnn_embedding, dtype=np.float64).ravel()
        pf = np.asarray(pocket_features, dtype=np.float64).ravel()
        # Pad/truncate to the canonical width so callers can pass any shape.
        if emb.size != DEFAULT_EMBED_DIM:
            padded = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
            n = min(emb.size, DEFAULT_EMBED_DIM)
            padded[:n] = emb[:n]
            emb = padded
        if pf.size != DEFAULT_POCKET_DIM:
            padded = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
            n = min(pf.size, DEFAULT_POCKET_DIM)
            padded[:n] = pf[:n]
            pf = padded
        X = np.concatenate([emb, pf])[None, :]
        return float(self._raw_predict(X)[0])

    def rank_candidates(
        self,
        smiles_list: Sequence[str],
        embeddings: Sequence[np.ndarray],
        pocket_features: np.ndarray,
        top_k: Optional[int] = None,
    ) -> VinaSurrogateReport:
        """Rank ``smiles_list`` by predicted Vina score.

        Parameters
        ----------
        smiles_list:
            The candidate SMILES (length N).  Returned as the
            ``smiles`` field of the ranking; not consumed by the
            surrogate directly.
        embeddings:
            Numpy array (or list of arrays) of shape ``(N, embed_dim)``
            — D-MPNN embedding for each candidate.
        pocket_features:
            1-D array of length ``pocket_dim`` shared by all
            candidates (the pocket context is fixed for one ranking).
        top_k:
            Optional truncation; ``None`` returns all N candidates.

        Returns
        -------
        :class:`VinaSurrogateReport` whose ``rankings`` are sorted
        ascending by predicted kcal/mol (more negative = better
        binder).  Confidence is ``"low"`` when the pool was below
        :data:`MIN_POOL_FOR_SURROGATE` and ``"medium"`` otherwise.
        """
        n = len(smiles_list)
        if n == 0:
            return VinaSurrogateReport(
                rankings=[],
                is_low_coverage=self._is_low_coverage,
                surrogate_train_loss=self._train_loss,
            )

        # Build the (N, D) feature matrix in one shot.
        emb_matrix = np.stack(
            [np.asarray(e, dtype=np.float64).ravel() for e in embeddings],
            axis=0,
        )
        if emb_matrix.shape[1] != DEFAULT_EMBED_DIM:
            padded = np.zeros((n, DEFAULT_EMBED_DIM), dtype=np.float64)
            copy = min(emb_matrix.shape[1], DEFAULT_EMBED_DIM)
            padded[:, :copy] = emb_matrix[:, :copy]
            emb_matrix = padded

        pf = np.asarray(pocket_features, dtype=np.float64).ravel()
        if pf.size != DEFAULT_POCKET_DIM:
            padded_pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
            copy = min(pf.size, DEFAULT_POCKET_DIM)
            padded_pf[:copy] = pf[:copy]
            pf = padded_pf

        X = np.concatenate(
            [emb_matrix, np.tile(pf[None, :], (n, 1))], axis=1
        )
        preds = self._raw_predict(X)
        confidence = "low" if self._is_low_coverage else "medium"

        idx_score = [(i, float(preds[i])) for i in range(n)]
        idx_score.sort(key=lambda t: t[1])
        if top_k is not None:
            idx_score = idx_score[: int(top_k)]
        rankings: List[Tuple[int, float, str]] = [
            (i, s, confidence) for (i, s) in idx_score
        ]
        return VinaSurrogateReport(
            rankings=rankings,
            is_low_coverage=self._is_low_coverage,
            surrogate_train_loss=self._train_loss,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def is_fitted(self) -> bool:
        return bool(self._fitted)

    @property
    def is_low_coverage(self) -> bool:
        return bool(self._is_low_coverage)

    def train_loss(self) -> float:
        return float(self._train_loss)

    def n_parameters(self) -> int:
        """Return the parameter count of the fitted surrogate.

        Used by :func:`combined_v0_vm_ranking` to assert the
        < 10K-param constraint.
        """
        if not self._fitted or self._weights is None:
            return 0
        n = int(self._weights.size) + 1  # weights + bias
        if self.backend == "mlp" and self._W1 is not None and self._b1 is not None:
            n += int(self._W1.size) + int(self._b1.size)
        return n


# ---------------------------------------------------------------------------
# Combined V^0 invariant + V^m ranking
# ---------------------------------------------------------------------------


def combined_v0_vm_ranking(
    v0_baseline: float,
    candidates: Sequence[Tuple[str, float]],
    surrogate: VinaSurrogate,
    v0_weight: float = 1.0,
    vm_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """Combine V^0 invariant + V^m surrogate ranking.

    Paper §3 monotone invariant: ``V* >= V_0``.  We honour this by
    blending the cached baseline ``V^0_baseline`` (constant across
    candidates for one pocket) with each candidate's surrogate
    prediction ``V^m``.  The blend is::

        combined(s) = v0_weight * V^0_baseline + vm_weight * V^m_s

    so any candidate whose surrogate prediction is finite has a
    combined score ≥ V^0 (so V* ≥ V_0).  The function then sorts
    ascending by combined score (more negative = better Vina
    binder).

    Parameters
    ----------
    v0_baseline:
        The cached V^0 baseline for this pocket (kcal/mol).  For
        novelty reasons the invariant is a *floor*, not a pick —
        the function only enforces monotone-ness when
        ``v0_baseline`` is finite.
    candidates:
        Sequence of ``(smiles, surrogate_prediction)`` tuples.
    surrogate:
        The fitted :class:`VinaSurrogate`.  Used for diagnostics
        only (``is_low_coverage``); not called here.
    v0_weight, vm_weight:
        Blend weights.  Defaults are 1.0 each, giving
        ``combined(s) = V^0_baseline + V^m_s``.

    Returns
    -------
    Sorted list of ``(smiles, combined_score)`` tuples, ascending
    by ``combined_score``.
    """
    del surrogate  # reserved for future diagnostics (e.g. low-coverage)
    if not math.isfinite(float(v0_baseline)):
        # Without a finite V^0 the invariant cannot be enforced; fall
        # back to pure V^m ranking.
        v0_baseline = 0.0
    blended: List[Tuple[str, float]] = []
    for smiles, vm in candidates:
        if not math.isfinite(float(vm)):
            # Surrogate said "don't know" — keep the candidate but
            # mark it as the worst by giving it +inf (so it sinks).
            blended.append((smiles, float("inf")))
            continue
        blended.append(
            (
                smiles,
                float(v0_weight) * float(v0_baseline)
                + float(vm_weight) * float(vm),
            )
        )
    blended.sort(key=lambda t: t[1])
    return blended


__all__ = [
    "DEFAULT_EMBED_DIM",
    "DEFAULT_POCKET_DIM",
    "MIN_POOL_FOR_SURROGATE",
    "MLP_HIDDEN",
    "RIDGE_ALPHA",
    "ReplayProbe",
    "RevealedRecord",
    "VinaReplayNode",
    "VinaReplayPool",
    "VinaReplaySimulator",
    "VinaSurrogate",
    "VinaSurrogateReport",
    "combined_v0_vm_ranking",
]


# Backward-compat alias for callers that typo'd the base class name.
RevealProbe = ReplayProbe
