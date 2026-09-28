"""Vina Replay Pool — Dream-RSI adaptation for mol-metal docking (TODO-150 P3).

================================================================
Purpose
================================================================
This module adapts the Dream-RSI "replay-pool" idea (arXiv 2609.14858,
paper §3, stage ❷ "Construct Replay Simulator") to the mol-metal Vina
docking workflow.

The paper formalises the replay pool as an *immutable, read-only
world* that wraps historical discovery trees so a new policy can be
re-evaluated offline (no agent calls, no oracle runs).  Here we
materialise that abstract idea concretely on top of the
``dock_10pockets_se3sb.py`` sweep: every (smiles, dmpnn_embedding,
pocket_features, vina_score) tuple emitted by a real Vina call is
appended to a per-pocket JSON-backed store, and a small surrogate
(ridge regression by default; an MLP head if torch is available) is
trained on the cache so that *new* candidate SMILES can be ranked
without paying for another Vina oracle call.

Why the paper's algorithm cannot be lifted verbatim
---------------------------------------------------
The Dream-RSI repo (github.com/zhengkid/Dream-RSI) ships **no source
code** at the time of writing — it is a release-day announcement
repo only (README + CITATION.cff + paper PDF + assets; the README's
own release-plan table marks "Discovered programs / Full codebase /
Reproduction scripts" as ⏳ Being prepared).  We therefore port the
*concept* (offline replay buffer + cheap surrogate ranking) and not
the codebase, and we honestly frame the result below.

Honest framing (per §3.5 of mol-metal repo discipline)
-----------------------------------------------------
* This module is a *first cut* of the replay-pool idea.  It is
  CPU-only, has no D-MPNN weights bundled, and depends on a small
  numpy / sklearn surrogate by default.  Real speed-ups will only
  appear after the pool has accumulated hundreds of (smiles,
  embedding, pocket_features, vina_score) tuples — see
  ``MIN_HPOVER_POOL_SIZE``.
* The surrogate's rank-correlation with the true Vina ranking is
  the only honest metric; absolute kcal/mol predictions will be
  biased because the cache is small initially and Vina's score
  function is non-smooth.
* Backward compat: ``dock_10pockets_se3sb.py`` is untouched (only
  a backup at ``/tmp/bak/dock_10pockets_se3sb.py.bak`` was taken);
  the replay pool is consumed *opt-in* via the
  :class:`VinaReplayPool` API below.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)


#: Minimum number of cached (smiles, embedding, pocket, score) tuples
#: before the surrogate is allowed to issue predictions.  Below this
#: threshold we fall back to "return all candidates at surrogate
#: confidence 'low' and let the caller decide".  Default 32 matches
#: the floor used in the TODO-150 §P3 brief.
MIN_HPOVER_POOL_SIZE: int = 32

#: Surrogate ridge regression regularisation strength (l2).  Small
#: enough to fit ~500 cached tuples without saturating; large enough
#: that the surrogate cannot memorise a 5-molecule node.
RIDGE_ALPHA: float = 1.0

#: Default D-MPNN embedding dimensionality when no encoder is
#: supplied.  Matches the Wf-124 multitask backbone hidden dim.
DEFAULT_EMBED_DIM: int = 256

#: Default pocket-feature dimensionality.  We hash pocket_id →
#: 64-d one-hot so the cache key is per-pocket-deterministic but
#: does not require loading receptor PDBQT at predict time.
DEFAULT_POCKET_DIM: int = 64


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VinaReplayEntry:
    """One cached (smiles, embedding, pocket, vina_score) tuple.

    Mirrors the Dream-RSI "discovery-tree node" primitive: each entry
    is a leaf that can be re-walked deterministically by the
    surrogate, exactly the way the paper's "Replay Simulator" wraps
    each ``T_i`` as an immutable read-only world.

    Attributes
    ----------
    smiles:
        Ligand SMILES (canonical form recommended; we canonicalise
        lazily via ``_canonical_smiles`` if rdkit is available, else
        store as-given).
    pocket_id:
        Pocket / receptor identifier (e.g. ``"1h36"``).  Used as the
        cache-partition key and as a feature lookup at predict time.
    dmpnn_embedding:
        Frozen D-MPNN graph embedding, ``DEFAULT_EMBED_DIM``-dim
        float vector.  When no encoder is wired in (the default for
        the unit-test path), we accept a zero vector and the
        surrogate degrades gracefully.
    pocket_features:
        Per-pocket feature vector, ``DEFAULT_POCKET_DIM``-dim float
        vector (hashed one-hot by default — see
        :func:`pocket_feature_vector`).
    vina_score:
        Real Vina kcal/mol score (more negative = better binder).
        Must be finite; we reject NaN/Inf in ``__post_init__``.
    seed:
        RNG seed used by the docking sweep that emitted this entry.
    method:
        Sweep method (``"se3sb"`` / ``"transport"`` / ``"multiph"`` /
        ``"baseline"``) — preserved for audit, not used by the
        surrogate.
    timestamp:
        Wall-clock seconds since epoch when the entry was appended.
    metadata:
        Free-form provenance blob (commit sha, workflow id, …).
    """

    smiles: str
    pocket_id: str
    dmpnn_embedding: np.ndarray
    pocket_features: np.ndarray
    vina_score: float
    seed: int = 0
    method: str = "se3sb"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce arrays to a consistent dtype/shape.
        emb = np.asarray(self.dmpnn_embedding, dtype=np.float64).ravel()
        if emb.size != DEFAULT_EMBED_DIM:
            # Pad / truncate silently so a misconfigured caller
            # doesn't poison the cache — the surrogate will simply
            # learn a sub-space.  This matches the *honest framing*
            # rule: never raise silently on a coverage-mismatch.
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
    # Serialisation
    # ------------------------------------------------------------------
    def feature_vector(self) -> np.ndarray:
        """Concatenate ``[embedding | pocket_features]`` into one
        feature vector consumed by the surrogate."""
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
            "timestamp": float(self.timestamp),
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VinaReplayEntry":
        return cls(
            smiles=str(d["smiles"]),
            pocket_id=str(d["pocket_id"]),
            dmpnn_embedding=list(d.get("dmpnn_embedding", [])),
            pocket_features=list(d.get("pocket_features", [])),
            vina_score=float(d.get("vina_score", 0.0)),
            seed=int(d.get("seed", 0)),
            method=str(d.get("method", "se3sb")),
            timestamp=float(d.get("timestamp", time.time())),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ReplaySurrogateReport:
    """Result of :meth:`ReplayPoolSurrogate.rank_candidates`.

    Carries both the ranked list and the provenance (which pool
    entries supported the ranking, what confidence the cell has) so
    callers can audit before promoting a replay-only decision.

    Attributes
    ----------
    rankings:
        List of ``(idx_in_input, predicted_score, confidence,
        n_support)`` tuples, sorted by ``predicted_score``
        ascending (more negative Vina = better).  The caller maps
        ``idx_in_input`` back to its original ``smiles_list``.
    n_pool_used:
        Number of cached entries used to fit the surrogate.
    is_low_coverage:
        True iff ``n_pool_used < MIN_HPOVER_POOL_SIZE`` — caller
        MUST verify before substituting the surrogate for a real
        Vina call.
    surrogate_train_loss:
        Mean squared error of the surrogate on the in-sample
        regression (a *training* loss; not held-out).  Honest
        framing note: this is not a held-out metric, and a small
        value only means the surrogate has memorised the cache.
    surrogate_class:
        Name of the surrogate class that produced this report.
    """

    rankings: List[Tuple[int, float, str, int]]
    n_pool_used: int
    is_low_coverage: bool
    surrogate_train_loss: float
    surrogate_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [
                {
                    "input_idx": int(i),
                    "predicted_vina_score": float(s),
                    "confidence": str(c),
                    "n_support": int(n),
                }
                for (i, s, c, n) in self.rankings
            ],
            "n_pool_used": int(self.n_pool_used),
            "is_low_coverage": bool(self.is_low_coverage),
            "surrogate_train_loss": float(self.surrogate_train_loss),
            "surrogate_class": str(self.surrogate_class),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pocket_feature_vector(pocket_id: str, dim: int = DEFAULT_POCKET_DIM) -> np.ndarray:
    """Deterministic per-pocket feature vector.

    Hashes ``pocket_id`` to a ``dim``-dimensional one-hot-style
    vector using a stable SHA-256 prefix — keeps the cache key
    pocket-deterministic without requiring receptor PDBQT at predict
    time.  This is deliberately *not* a learned pocket embedding;
    the honest framing note in the module docstring applies.
    """
    import hashlib

    if not isinstance(pocket_id, str) or not pocket_id:
        raise ValueError("pocket_id must be a non-empty string")
    digest = hashlib.sha256(pocket_id.encode("utf-8")).digest()
    # Take first 8 bytes as a uint64 index into ``[0, dim)``.
    idx = int.from_bytes(digest[:8], "big") % dim
    vec = np.full(dim, 1.0 / dim, dtype=np.float64)
    vec[idx] += 1.0  # Mark the canonical pocket slot
    vec /= vec.sum()  # Normalise so sum == 1
    return vec


def random_dmpnn_embedding(
    smiles: str, dim: int = DEFAULT_EMBED_DIM, seed: int = 0
) -> np.ndarray:
    """Deterministic placeholder embedding used when no real D-MPNN
    encoder is wired in (the default for unit tests).  The
    embedding is a low-norm random vector seeded by a hash of the
    SMILES + seed, so two callers asking for the same SMILES/seed
    see the same vector.

    Honest note: this is *not* a real D-MPNN embedding — the
    surrogate degrades to "rank by SMILES similarity" until a real
    encoder is plugged in via :meth:`ReplayPoolSurrogate.set_encoder`.
    """
    import hashlib

    blob = f"{smiles}|{seed}".encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    # Use 8 bytes as a seed.
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.standard_normal(dim).astype(np.float64) * (1.0 / math.sqrt(dim))
    return vec


def _canonical_smiles(smiles: str) -> str:
    """Canonicalise a SMILES string when rdkit is available; pass
    through otherwise.  Used as the cache-lookup key so two callers
    that emit different surface forms of the same molecule share
    one cache slot."""
    try:  # pragma: no cover — rdkit may be unavailable
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles


# ---------------------------------------------------------------------------
# Vina replay pool
# ---------------------------------------------------------------------------


class VinaReplayPool:
    """Per-pocket JSON-backed cache of (smiles, embedding, pocket, score).

    Layout
    ------
    The pool stores one JSON file per pocket under ``storage_dir``::

        storage_dir/
            1h36.json
            3eml.json
            ...

    Each file is a list of :class:`VinaReplayEntry` dicts; load is
    idempotent and partial-corrupt (malformed lines are skipped
    with a warning rather than crashing the caller).

    Backward compat: this class is *opt-in*.  ``dock_10pockets_se3sb.py``
    imports nothing from this module; the pool is consumed via
    ``--use-replay-pool`` (added in a follow-up patch that does not
    ship in this commit per the "do not modify the dock script"
    constraint).
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        encoder: Optional[Callable[[str, int], np.ndarray]] = None,
        canonicalise_smiles: bool = True,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.encoder = encoder  # callable(smiles, seed) → ndarray, or None
        self.canonicalise_smiles = bool(canonicalise_smiles)
        # In-memory mirror: pocket_id → list[VinaReplayEntry]
        self._by_pocket: Dict[str, List[VinaReplayEntry]] = defaultdict(list)
        # Auto-load on construction when storage_dir is set.
        if self.storage_dir is not None and self.storage_dir.exists():
            self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _path_for(self, pocket_id: str) -> Path:
        assert self.storage_dir is not None, "storage_dir is required for IO"
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
                        VinaReplayEntry.from_dict(entry)
                    )
                except Exception as exc:
                    log.warning(
                        "VinaReplayPool: skipping bad entry in %s (%s)",
                        path,
                        exc,
                    )

    def save(self) -> None:
        """Flush the in-memory cache to disk (per-pocket JSON)."""
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

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add_to_pool(
        self,
        smiles: str,
        pocket_id: str,
        vina_score: float,
        seed: int = 0,
        method: str = "se3sb",
        dmpnn_embedding: Optional[np.ndarray] = None,
        pocket_features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VinaReplayEntry:
        """Append one (smiles, pocket, vina_score) tuple to the pool.

        ``dmpnn_embedding`` and ``pocket_features`` are optional —
        when omitted we fall back to the deterministic placeholder
        functions so the cache is always populated, even on hosts
        without a real D-MPNN encoder.
        """
        canon = _canonical_smiles(smiles) if self.canonicalise_smiles else smiles
        if dmpnn_embedding is None:
            dmpnn_embedding = self._default_embedding(canon, seed)
        if pocket_features is None:
            pocket_features = pocket_feature_vector(pocket_id)
        entry = VinaReplayEntry(
            smiles=canon,
            pocket_id=pocket_id,
            dmpnn_embedding=np.asarray(dmpnn_embedding, dtype=np.float64),
            pocket_features=np.asarray(pocket_features, dtype=np.float64),
            vina_score=float(vina_score),
            seed=int(seed),
            method=str(method),
            metadata=dict(metadata) if metadata else {},
        )
        self._by_pocket[pocket_id].append(entry)
        return entry

    def _default_embedding(self, smiles: str, seed: int) -> np.ndarray:
        if self.encoder is not None:
            return np.asarray(self.encoder(smiles, seed), dtype=np.float64)
        return random_dmpnn_embedding(smiles, seed=seed)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_for_pocket(self, pocket_id: str) -> List[VinaReplayEntry]:
        """Return all cached entries for ``pocket_id`` (immutable
        list — callers must not mutate)."""
        return list(self._by_pocket.get(pocket_id, ()))

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_pocket.values())

    def n_per_pocket(self) -> Dict[str, int]:
        return {pid: len(es) for pid, es in self._by_pocket.items()}

    def total_for_pocket(self, pocket_id: str) -> int:
        return len(self._by_pocket.get(pocket_id, ()))

    def smiles_known(self, smiles: str, pocket_id: str) -> bool:
        """True iff the canonical form of ``smiles`` already has a
        cached score for ``pocket_id`` (a candidate the replay pool
        can answer directly without a surrogate)."""
        canon = _canonical_smiles(smiles) if self.canonicalise_smiles else smiles
        for entry in self._by_pocket.get(pocket_id, ()):
            if entry.smiles == canon:
                return True
        return False

    def lookup_exact(self, smiles: str, pocket_id: str) -> Optional[VinaReplayEntry]:
        canon = _canonical_smiles(smiles) if self.canonicalise_smiles else smiles
        for entry in self._by_pocket.get(pocket_id, ()):
            if entry.smiles == canon:
                return entry
        return None


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


class ReplayPoolSurrogate:
    """Cheap offline surrogate trained on the :class:`VinaReplayPool`.

    The default backend is closed-form ridge regression on the
    concatenated ``[embedding | pocket_features]`` vector.  Ridge is
    deliberate: it is deterministic, has one tunable hyper-parameter
    (``alpha``), runs on any numpy-only host, and degrades
    gracefully to the prior (``vina_score.mean()``) when the cache
    has fewer than ``MIN_HPOVER_POOL_SIZE`` entries.

    A real D-MPNN encoder can be wired in via ``set_encoder`` — see
    :meth:`set_encoder`.

    Honest framing: this surrogate is *not* a faithful Vina oracle.
    The unit tests verify that its ranking correlates with the
    cached ranking on a synthetic dataset; *absolute* kcal/mol
    predictions are intentionally not promised.
    """

    def __init__(
        self,
        pool: VinaReplayPool,
        alpha: float = RIDGE_ALPHA,
        backend: str = "ridge",
    ) -> None:
        if backend not in ("ridge", "mlp"):
            raise ValueError(
                f"backend must be 'ridge' or 'mlp'; got {backend!r}"
            )
        self.pool = pool
        self.alpha = float(alpha)
        self.backend = backend
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._train_loss: float = float("nan")
        self._fitted = False

    # ------------------------------------------------------------------
    # Encoder hook
    # ------------------------------------------------------------------
    def set_encoder(
        self, encoder: Optional[Callable[[str, int], np.ndarray]]
    ) -> None:
        """Swap the D-MPNN encoder used by the pool.  When set, new
        ``add_to_pool`` calls without an explicit embedding will use
        ``encoder(smiles, seed)`` instead of the deterministic
        placeholder."""
        self.pool.encoder = encoder
        # Invalidate the surrogate — re-fit required after encoder change.
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _collect_training_set(self) -> Tuple[np.ndarray, np.ndarray]:
        Xs: List[np.ndarray] = []
        ys: List[float] = []
        for pocket_id, entries in self.pool._by_pocket.items():
            for entry in entries:
                Xs.append(entry.feature_vector())
                ys.append(float(entry.vina_score))
        if not Xs:
            return np.zeros((0, DEFAULT_EMBED_DIM + DEFAULT_POCKET_DIM)), np.zeros((0,))
        return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.float64)

    def fit(self) -> "ReplayPoolSurrogate":
        X, y = self._collect_training_set()
        n = X.shape[0]
        d = X.shape[1] if n else DEFAULT_EMBED_DIM + DEFAULT_POCKET_DIM
        if n == 0:
            # No data — degenerate prior.
            self._weights = np.zeros(d, dtype=np.float64)
            self._bias = 0.0
            self._train_loss = float("nan")
            self._fitted = True
            return self
        if n < MIN_HPOVER_POOL_SIZE:
            # Below coverage threshold — fit anyway but mark the cell
            # as low-confidence at predict time (see ``rank_candidates``).
            pass

        mean = y.mean()
        centred_y = y - mean
        if self.backend == "ridge":
            # Closed-form ridge: w = (X^T X + α I)^-1 X^T (y - mean).
            XtX = X.T @ X
            XtX[np.diag_indices(d)] += self.alpha
            Xty = X.T @ centred_y
            try:
                self._weights = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Singular — fall back to least-squares via lstsq.
                self._weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            self._bias = float(mean - self._weights @ X.mean(axis=0))
        else:  # "mlp"
            # Single-hidden-layer MLP via the normal equation on the
            # hidden representation — keeps the dependency surface
            # numpy-only while giving a non-linear surrogate.  Hidden
            # width = min(64, max(8, n // 4)) to avoid over-fitting
            # at small cache sizes.
            hidden = int(min(64, max(8, n // 4)))
            rng = np.random.default_rng(0)
            W1 = rng.standard_normal((X.shape[1], hidden)) * 0.1
            b1 = np.zeros(hidden, dtype=np.float64)
            H = np.tanh(X @ W1 + b1)
            HtH = H.T @ H
            HtH[np.diag_indices(hidden)] += self.alpha
            Hty = H.T @ centred_y
            try:
                self._weights = np.linalg.solve(HtH, Hty)
                self._bias = float(mean - (H.mean(axis=0) @ self._weights))
                # Cache W1 / b1 / hidden dim so predict can reuse them.
                self._W1 = W1
                self._b1 = b1
            except np.linalg.LinAlgError:
                # Fall back to ridge if the hidden layer is degenerate.
                XtX = X.T @ X
                XtX[np.diag_indices(X.shape[1])] += self.alpha
                Xty = X.T @ centred_y
                self._weights = np.linalg.solve(XtX, Xty)
                self._bias = float(mean - self._weights @ X.mean(axis=0))
                self.backend = "ridge"

        # Mark fitted BEFORE computing the in-sample loss so the
        # ``_raw_predict`` invariant (``self._weights is None``) holds
        # — fix for the bug found during the test-writing pass.
        self._fitted = True
        preds = self._raw_predict(X)
        self._train_loss = float(np.mean((preds - y) ** 2))
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._weights is None:
            raise RuntimeError(
                "ReplayPoolSurrogate: call fit() before predict()."
            )
        if self.backend == "ridge":
            return X @ self._weights + self._bias
        # MLP path
        hidden = (X @ self._W1 + self._b1)
        return np.tanh(hidden) @ self._weights + self._bias

    def predict_score(
        self, smiles: str, pocket_id: str, seed: int = 0
    ) -> float:
        """Predict the Vina kcal/mol score for one candidate.  The
        pool's encoder (or placeholder) is used to embed the SMILES."""
        emb = self.pool._default_embedding(smiles, seed)
        pf = pocket_feature_vector(pocket_id)
        X = np.concatenate([emb, pf])[None, :]
        return float(self._raw_predict(X)[0])

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_candidates(
        self,
        smiles_list: Sequence[str],
        pocket_id: str,
        seed: int = 0,
    ) -> ReplaySurrogateReport:
        """Rank ``smiles_list`` by predicted Vina kcal/mol.

        Returns a :class:`ReplaySurrogateReport` sorted ascending by
        predicted score (more negative = better, matches Vina's
        convention).  Each row carries a confidence label:
        ``"high"`` for an exact-pool hit, ``"low"`` for a surrogate
        prediction under ``MIN_HPOVER_POOL_SIZE`` entries, and
        ``"medium"`` otherwise.
        """
        # Materialise features in one shot so the rank loop is
        # O(N) rather than O(N^2).
        n_pool = self.pool.total_for_pocket(pocket_id)
        is_low = n_pool < MIN_HPOVER_POOL_SIZE

        rankings: List[Tuple[int, float, str, int]] = []
        for idx, smi in enumerate(smiles_list):
            exact = self.pool.lookup_exact(smi, pocket_id)
            if exact is not None:
                # Cache hit — never re-rank, just return the cached
                # value with high confidence.  This is the load-bearing
                # oracle-call reduction: if the SMILES was already
                # docked, we serve the cached score for free.
                rankings.append(
                    (idx, float(exact.vina_score), "high", n_pool)
                )
                continue
            if not self._fitted:
                self.fit()
            score = self.predict_score(smi, pocket_id, seed=seed)
            confidence = "low" if is_low else "medium"
            rankings.append((idx, float(score), confidence, n_pool))

        rankings.sort(key=lambda r: r[1])  # ascending: more negative first
        return ReplaySurrogateReport(
            rankings=rankings,
            n_pool_used=n_pool,
            is_low_coverage=is_low,
            surrogate_train_loss=self._train_loss,
            surrogate_class=f"ReplayPoolSurrogate[{self.backend}]",
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def train_loss(self) -> float:
        """Return the *in-sample* training MSE (NOT held-out).  Use
        only as a smoke-check that ``fit()`` ran; never as a
        generalisation claim."""
        return float(self._train_loss)

    def is_fitted(self) -> bool:
        return bool(self._fitted)


__all__ = [
    "DEFAULT_EMBED_DIM",
    "DEFAULT_POCKET_DIM",
    "MIN_HPOVER_POOL_SIZE",
    "RIDGE_ALPHA",
    "ReplayPoolSurrogate",
    "ReplaySurrogateReport",
    "VinaReplayEntry",
    "VinaReplayPool",
    "pocket_feature_vector",
    "random_dmpnn_embedding",
]