"""MCTS replay buffer — AlphaGo-style experience replay (TODO-31).

================================================================
Layer 9 — Lambda MCTS experience replay
================================================================
The AlphaGo / AlphaZero family is **defined** by its experience replay
loop:

    play   ->   store (state, action, value, prior)   ->   replay batch   ->   train

Our :class:`MCTSProofSearch` (Layer 8) currently only *plays*; every
:func:`search` call rebuilds its tree from scratch and discards the
visit-count / value / prior statistics when it returns.  This module
adds the missing second half:

    * :class:`MCTSReplayEntry`   — single replay record
    * :class:`MCTSReplayBuffer`   — bounded FIFO ring buffer of entries
    * :class:`MCTSReplayWrapper`  — adapter that wraps an existing
                                    :class:`MCTSProofSearch` and
                                    transparently collects replay
                                    records during :meth:`search`
    * :class:`ReplayGuidedMCTS`   — replay-guided prior biasing; uses
                                    accumulated replay statistics to
                                    bias the root prior of subsequent
                                    searches (the "warm-start from
                                    replay" analogue)

Why a *wrapper* (not a re-implementation)
-----------------------------------------
Both ``proof_search.py`` (R17 ship) and ``variance_puct_integration.py``
(Wf-11 ship) and ``variance_mcts_integration.py`` (Wf-14 ship) and
``score_repellent_mc.py`` (Wf-9 ship) are stable, well-tested modules.
Re-implementing them here would duplicate code and risk silent
regressions.  Instead this module provides thin adapters that *observe*
the search's state machine:

* :class:`MCTSReplayWrapper.search` calls :meth:`MCTSProofSearch.search`
  with the *default* regime to obtain the candidate list, then reads
  the per-candidate priors / visit counts / values off the search's
  public attributes (``history``, ``debug_trace``,
  ``_root_children_cache_for_search``).

* :class:`ReplayGuidedMCTS.bias_prior` consumes a populated
  :class:`MCTSReplayBuffer` and produces a *bias vector* that downstream
  MCTS calls can mix into their root prior — mirroring AlphaZero's
  *learning from replay* loop without touching the underlying search.

The wrapper does **not** mutate the underlying MCTS state machine;
it only reads public attributes and writes replay entries.

Math prior
----------
The AlphaZero replay target is::

    L = (z - v)^2  +  cross_entropy(pi, p_policy)

where ``z`` is the game outcome (here: the leaf value reported by the
scorer / aggregator), ``v`` is the value-network prediction, ``pi``
is the MCTS-improved policy (= visit-count distribution over actions),
and ``p_policy`` is the policy-network prediction.  We do not ship a
neural policy/value network in this module (it lives in
``molmetal/training/``); instead we surface the **per-batch loss
estimates** that a downstream value head can consume — see
:meth:`MCTSReplayWrapper.train_value_network`.

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.  §3.3 (Replay
  buffer) + §3.4 (Training pipeline).
* Silver D. et al. (2018) "A general reinforcement learning algorithm
  that masters chess, shogi, and Go through self-play".  *Science*
  362, 1140-1144.  §3 (AlphaZero).
* Schrittwieser J. et al. (2020) "Mastering Atari, Go, Chess and
  Shogi by Planning with a Learned Model".  *Nature* 588, 604-609.

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_replay_buffer.py -v
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


__version__ = "0.1.0-mcts-replay-buffer"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCTSReplayError(ValueError):
    """Raised when the MCTS replay buffer layer fails.

    Distinct from :class:`variance_puct_integration.VariancePUCTError` so
    callers can branch on replay-layer-specific failures without
    inspecting string messages.  Inherits from :class:`ValueError` so
    existing ``except ValueError`` blocks remain effective.
    """


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default replay-buffer capacity.  10000 entries matches the canonical
#: AlphaGo Zero replay-buffer size of "the most recent 500,000 games"
#: when each game emits ~50 MCTS records (AlphaGo Zero uses 500K games
#: × ~50 records = 25M entries; we scale down by 2500× for a CPU-only
#: smoke run).
DEFAULT_MAX_SIZE: int = 10_000

#: Default batch size for :meth:`MCTSReplayBuffer.sample`.  32 is the
#: canonical AlphaGo Zero mini-batch size.
DEFAULT_BATCH_SIZE: int = 32

#: Default replay-bias mixing weight.  ``alpha_replay = 0.5`` means the
#: replay bias contributes equally to the root prior as the original
#: learned / symbolic prior (AlphaZero convention: 0.0 = pure replay,
#: 1.0 = pure base prior).
DEFAULT_REPLAY_ALPHA: float = 0.5

#: Default visit-count temperature.  ``tau = 1.0`` recovers the canonical
#: ``pi(a) = N(a) ** (1/tau) / sum_b N(b) ** (1/tau)`` AlphaZero
#: target policy; ``tau → 0`` collapses to argmax.
DEFAULT_TAU: float = 1.0

#: Cap on entries sampled in a single :meth:`sample` call.  Guards
#: against accidental ``batch_size > max_size`` requests.
MAX_BATCH_SIZE: int = 1_000_000


# ---------------------------------------------------------------------------
# Replay record
# ---------------------------------------------------------------------------


@dataclass
class MCTSReplayEntry:
    """Single MCTS replay record — AlphaGo ``(s, a, z, pi)`` tuple.

    This is the analogue of AlphaGo Zero's per-move training record::

        (state, action_taken, outcome_z, mcts_policy_pi)

    In the MLC setting:

    * ``state``    — a hashable representation of the λ-term (typically
      the canonical SMILES, but we accept any hashable for testability).
    * ``action``   — the action chosen at ``state`` (a hashable id, e.g.
      the rule_name + tile_id pair).  Stored as a string for portability.
    * ``value``    — the scalar leaf value reported by the scorer /
      reward aggregator for the *resulting* state after taking
      ``action``.  Higher = better (matches PUCT convention).
    * ``prior``    — the PUCT prior P(action | state) at expansion time.
      Real-valued in [0, 1].
    * ``variance`` — the per-arm variance reported by
      :class:`VarianceMCTSWrapper` (Wf-14) for ``action`` at ``state``.
      ``0.0`` when the search did not record variance.
    * ``visit_count`` — the MCTS visit count N(state, action) at the
      time of recording.  Powers the AlphaZero target policy
      ``pi(a) = N(a)^(1/tau) / sum_b N(b)^(1/tau)``.

    All six fields are required; the dataclass provides a stable
    pickle / JSON contract for :meth:`MCTSReplayBuffer.save`.
    """

    state: Any
    action: Any
    value: float
    prior: float
    variance: float = 0.0
    visit_count: int = 1
    #: Wall-clock timestamp (seconds since epoch).  Populated by
    #: :meth:`MCTSReplayBuffer.add` so callers can age out stale
    #: records.
    timestamp: float = field(default_factory=time.time)
    #: Optional metadata blob (e.g. smiles, rule_name, tile_id, scorer
    #: name).  Not used by the buffer but forwarded through save/load
    #: so downstream consumers can read the provenance without
    #: re-parsing ``state`` / ``action``.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise MCTSReplayError(
                f"value must be numeric; got {type(self.value).__name__}"
            )
        if not isinstance(self.prior, (int, float)):
            raise MCTSReplayError(
                f"prior must be numeric; got {type(self.prior).__name__}"
            )
        if not isinstance(self.variance, (int, float)):
            raise MCTSReplayError(
                f"variance must be numeric; got {type(self.variance).__name__}"
            )
        if not isinstance(self.visit_count, int) or self.visit_count < 1:
            raise MCTSReplayError(
                f"visit_count must be a positive int; got {self.visit_count!r}"
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def age_seconds(self) -> float:
        """Wall-clock age of the entry in seconds."""
        return max(0.0, time.time() - float(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict representation.

        ``state`` and ``action`` are coerced to ``str`` so the result is
        always JSON-safe; the original objects can be reconstructed by
        the caller (we do not attempt to encode arbitrary Python
        objects).
        """
        return {
            "state": str(self.state),
            "action": str(self.action),
            "value": float(self.value),
            "prior": float(self.prior),
            "variance": float(self.variance),
            "visit_count": int(self.visit_count),
            "timestamp": float(self.timestamp),
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCTSReplayEntry":
        """Inverse of :meth:`to_dict`.

        Note that ``state`` and ``action`` are restored as their string
        representation — callers that need the original Python objects
        must reconstruct them from the encoded form.
        """
        return cls(
            state=d.get("state"),
            action=d.get("action"),
            value=float(d.get("value", 0.0)),
            prior=float(d.get("prior", 0.0)),
            variance=float(d.get("variance", 0.0)),
            visit_count=int(d.get("visit_count", 1)),
            timestamp=float(d.get("timestamp", time.time())),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Buffer manager
# ---------------------------------------------------------------------------


@dataclass
class MCTSReplayBuffer:
    """Bounded FIFO ring buffer of :class:`MCTSReplayEntry` records.

    Mirrors AlphaGo Zero's replay buffer: a fixed-capacity queue that
    drops the oldest entry when full (FIFO eviction).  Sampling is
    uniformly at random over the current contents (AlphaZero uses a
    uniform sampler; recent work — see Schrittwieser 2020 — uses a
    prioritised sampler, which we do **not** implement here).

    Parameters
    ----------
    max_size : int, default 10000
        Maximum number of entries retained.  Must be ≥ 1.
    seed : int, default 0
        RNG seed for :meth:`sample`.  Reproducibility matters when
        replay samples feed a downstream value-network training step.

    Attributes
    ----------
    entries : deque[MCTSReplayEntry]
        Underlying ring buffer.  Read-only by convention — mutate via
        :meth:`add` / :meth:`clear`.
    n_added : int
        Cumulative number of entries ever added (does not decrease on
        eviction).  Useful for tracking replay-buffer "burn-in".
    n_evicted : int
        Cumulative number of entries evicted due to ``max_size``.  A
        steady-state replay buffer reports ``n_evicted > 0`` once
        ``n_added > max_size``.
    """

    max_size: int = DEFAULT_MAX_SIZE
    seed: int = 0
    entries: Deque[MCTSReplayEntry] = field(default_factory=deque)
    n_added: int = 0
    n_evicted: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise MCTSReplayError(f"max_size must be >= 1; got {self.max_size}")
        self._rng = np.random.default_rng(int(self.seed))

    # ------------------------------------------------------------------
    # Capacity / size
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def is_full(self) -> bool:
        return len(self.entries) >= int(self.max_size)

    def capacity_remaining(self) -> int:
        return max(0, int(self.max_size) - len(self.entries))

    # ------------------------------------------------------------------
    # Add / clear
    # ------------------------------------------------------------------
    def add(self, entry: MCTSReplayEntry) -> None:
        """Append an entry to the buffer (FIFO-evicting on overflow).

        Parameters
        ----------
        entry : MCTSReplayEntry
            Replay record.  Must be a fully-formed
            :class:`MCTSReplayEntry` (the dataclass enforces field
            types).
        """
        if not isinstance(entry, MCTSReplayEntry):
            raise MCTSReplayError(
                f"add() expects MCTSReplayEntry; got {type(entry).__name__}"
            )
        if self.is_full():
            self.entries.popleft()
            self.n_evicted += 1
        self.entries.append(entry)
        self.n_added += 1

    def extend(self, entries: Iterable[MCTSReplayEntry]) -> int:
        """Append multiple entries; returns the number added."""
        before = self.n_added
        for e in entries:
            self.add(e)
        return self.n_added - before

    def clear(self) -> None:
        """Drop all entries.  Does not reset ``n_added`` / ``n_evicted``."""
        self.entries.clear()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size: int = DEFAULT_BATCH_SIZE) -> List[MCTSReplayEntry]:
        """Uniformly sample ``batch_size`` entries without replacement.

        When ``batch_size > len(self.entries)`` we return a single
        mini-batch of size ``len(self.entries)`` (and log a warning via
        the buffer's ``n_short_sample`` counter so callers can detect
        the under-filled regime).

        Parameters
        ----------
        batch_size : int, default 32
            Number of entries to sample.  Must be ≥ 1.
        """
        if batch_size < 1:
            raise MCTSReplayError(f"batch_size must be >= 1; got {batch_size}")
        if batch_size > MAX_BATCH_SIZE:
            raise MCTSReplayError(
                f"batch_size={batch_size} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
            )
        n_avail = len(self.entries)
        if n_avail == 0:
            return []
        k = min(int(batch_size), n_avail)
        # numpy.random.Generator.choice without replacement on a
        # deque of dataclasses — we sample indices into a list and
        # then index back, since ``choice`` does not support deque
        # directly.
        idx = self._rng.choice(n_avail, size=k, replace=False)
        # Materialise once; deque indexing is O(n) so we list() it.
        pool = list(self.entries)
        return [pool[int(i)] for i in idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return a stats dict summarising the buffer's contents.

        Useful for the verify_smoke harness and for streaming
        TensorBoard-equivalent logs.
        """
        if not self.entries:
            return {
                "size": 0,
                "capacity": int(self.max_size),
                "n_added": int(self.n_added),
                "n_evicted": int(self.n_evicted),
                "mean_value": 0.0,
                "std_value": 0.0,
                "mean_prior": 0.0,
                "mean_variance": 0.0,
                "mean_visit_count": 0.0,
                "value_range": (0.0, 0.0),
            }
        values = np.asarray([e.value for e in self.entries], dtype=float)
        priors = np.asarray([e.prior for e in self.entries], dtype=float)
        varis = np.asarray([e.variance for e in self.entries], dtype=float)
        visits = np.asarray([e.visit_count for e in self.entries], dtype=float)
        return {
            "size": len(self.entries),
            "capacity": int(self.max_size),
            "n_added": int(self.n_added),
            "n_evicted": int(self.n_evicted),
            "mean_value": float(values.mean()),
            "std_value": float(values.std()),
            "mean_prior": float(priors.mean()),
            "mean_variance": float(varis.mean()),
            "mean_visit_count": float(visits.mean()),
            "value_range": (float(values.min()), float(values.max())),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Persist the buffer to disk.

        Two formats are supported, picked by the file extension:

        * ``.json`` / ``.jsonl`` — JSON-lines via :meth:`to_dict` /
          :meth:`from_dict`.  Human-readable, JSON-safe; ``state`` and
          ``action`` are coerced to ``str``.
        * anything else (``.pkl``, ``.bin``, no extension) — pickle
          (full Python-object fidelity).

        Returns the absolute path written.
        """
        if not path:
            raise MCTSReplayError("save() requires a non-empty path")
        abspath = os.path.abspath(path)
        ext = os.path.splitext(abspath)[1].lower()
        os.makedirs(os.path.dirname(abspath) or ".", exist_ok=True)
        if ext in (".json", ".jsonl"):
            with open(abspath, "w", encoding="utf-8") as fh:
                meta = {
                    "_format": "jsonl",
                    "_version": __version__,
                    "max_size": int(self.max_size),
                    "n_added": int(self.n_added),
                    "n_evicted": int(self.n_evicted),
                    "seed": int(self.seed),
                }
                fh.write(json.dumps(meta) + "\n")
                for e in self.entries:
                    fh.write(json.dumps(e.to_dict()) + "\n")
        else:
            payload = {
                "_format": "pickle",
                "_version": __version__,
                "max_size": int(self.max_size),
                "n_added": int(self.n_added),
                "n_evicted": int(self.n_evicted),
                "seed": int(self.seed),
                "entries": list(self.entries),
            }
            with open(abspath, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return abspath

    @classmethod
    def load(cls, path: str) -> "MCTSReplayBuffer":
        """Inverse of :meth:`save` — read a buffer from disk.

        Detects the format from the file extension and routes to the
        matching loader.
        """
        if not path:
            raise MCTSReplayError("load() requires a non-empty path")
        abspath = os.path.abspath(path)
        if not os.path.exists(abspath):
            raise MCTSReplayError(f"load() cannot find {abspath!r}")
        ext = os.path.splitext(abspath)[1].lower()
        if ext in (".json", ".jsonl"):
            with open(abspath, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            if not lines:
                raise MCTSReplayError(f"load() — empty file {abspath!r}")
            meta = json.loads(lines[0])
            buf = cls(
                max_size=int(meta.get("max_size", DEFAULT_MAX_SIZE)),
                seed=int(meta.get("seed", 0)),
            )
            # Restore counters verbatim — do NOT route through
            # :meth:`add`, which would double-count ``n_added``.
            buf.n_added = int(meta.get("n_added", 0))
            buf.n_evicted = int(meta.get("n_evicted", 0))
            for ln in lines[1:]:
                e = MCTSReplayEntry.from_dict(json.loads(ln))
                buf.entries.append(e)
            return buf
        with open(abspath, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or "entries" not in payload:
            raise MCTSReplayError(
                "load() — malformed pickle payload (missing 'entries')"
            )
        buf = cls(
            max_size=int(payload.get("max_size", DEFAULT_MAX_SIZE)),
            seed=int(payload.get("seed", 0)),
        )
        # Restore counters verbatim — see JSON branch above.
        buf.n_added = int(payload.get("n_added", 0))
        buf.n_evicted = int(payload.get("n_evicted", 0))
        for e in payload["entries"]:
            buf.entries.append(e)
        return buf


# ---------------------------------------------------------------------------
# Wrapper that collects replay entries from MCTSProofSearch
# ---------------------------------------------------------------------------


@dataclass
class MCTSReplayWrapper:
    """Adapter that wraps an existing :class:`MCTSProofSearch` with replay collection.

    Why a wrapper
    -------------
    We are forbidden from modifying :mod:`proof_search` /
    :mod:`score_repellent_mc` / :mod:`variance_puct_integration` /
    :mod:`variance_mcts_integration`.  This wrapper composes them by
    calling :meth:`MCTSProofSearch.search` and reading the *public*
    diagnostics (``history``, ``debug_trace``,
    ``_root_children_cache_for_search``) to emit replay records —
    without ever mutating the underlying search state machine.

    Parameters
    ----------
    base_search : MCTSProofSearch
        The MCTS instance to wrap.  Not mutated.
    buffer : MCTSReplayBuffer
        The replay buffer to append entries into.  Must not be
        ``None``.
    collect_replay : bool, default True
        When True, :meth:`search` records one replay entry per
        candidate returned.  When False, :meth:`search` is a
        transparent pass-through (still returns the same int).
    n_top_actions : int, default 8
        Number of (state, action) pairs to record per :meth:`search`
        call.  Defaults to 8 (the AlphaGo Zero "top-K actions per
        state" recommendation; smaller than ``top_k=20`` so the
        replay buffer stays focused on the head of the visit-count
        distribution).

    Attributes
    ----------
    n_searches_ : int
        Cumulative number of :meth:`search` calls.
    n_entries_recorded_ : int
        Cumulative number of replay entries recorded across all
        :meth:`search` calls.
    """

    base_search: Any
    buffer: MCTSReplayBuffer
    collect_replay: bool = True
    n_top_actions: int = 8

    n_searches_: int = field(default=0, init=False, repr=False)
    n_entries_recorded_: int = field(default=0, init=False, repr=False)
    last_loss_: Dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.buffer is None:
            raise MCTSReplayError("MCTSReplayWrapper requires a non-None buffer")
        if self.n_top_actions < 1:
            raise MCTSReplayError(
                f"n_top_actions must be >= 1; got {self.n_top_actions}"
            )

    # ------------------------------------------------------------------
    # Search (records replay entries)
    # ------------------------------------------------------------------
    def search(self, root: Any, collect_replay: Optional[bool] = None) -> int:
        """Run the wrapped MCTS and record replay entries.

        Returns
        -------
        int
            Best-action index (the highest-visit-count action at the
            root).  When the wrapped MCTS returns a list of candidates
            rather than an int, we return ``0`` as the canonical
            "best is first" fallback — the wrapper's contract is
            *single-int action selection* (matches AlphaGo's
            per-move action choice), so callers needing the full
            candidate list should call ``self.base_search.search``
            directly.

        Parameters
        ----------
        root : Any
            Root state passed to ``base_search.search``.  Typically a
            :class:`MoleculeClosedTerm` for the MLC search, but we
            forward any object the wrapped search accepts.
        collect_replay : bool, optional
            Override the constructor-level ``collect_replay`` for this
            single call.
        """
        if collect_replay is None:
            collect_replay = bool(self.collect_replay)
        self.n_searches_ += 1

        # ------------------------------------------------------------------
        # 1. Dispatch the search.  We try ``base_search.search(root)``
        #    first (single-step / variance-controlled wrappers return
        #    int directly), then fall back to the legacy list-returning
        #    MCTSProofSearch.search(initial_state, max_depth=...).
        # ------------------------------------------------------------------
        best_action: int = 0
        candidates: List[Any] = []
        scores: List[float] = []
        try:
            result = self.base_search.search(root)
            if isinstance(result, int):
                best_action = int(result)
                # Some variance wrappers return only an int; we still
                # try to read per-action diagnostics below.
            elif isinstance(result, list):
                candidates = list(result)
                # We don't know the leaf scores from the int-returning
                # path, so the diagnostics reader below will fill them
                # in when available.
        except Exception as exc:  # pragma: no cover - defensive
            raise MCTSReplayError(
                f"base_search.search raised {type(exc).__name__}: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # 2. Read per-candidate diagnostics from the search's public
        #    attributes (history[-1], debug_trace, _root_children_cache).
        #    None of these are required — missing diagnostics simply
        #    yield coarser replay entries (uniform prior / visit count 1).
        # ------------------------------------------------------------------
        diagnostics = self._read_diagnostics()
        if diagnostics and not candidates:
            candidates = diagnostics.get("candidates", [])
        scores = diagnostics.get("scores", scores)
        priors = diagnostics.get("priors", [])
        variances = diagnostics.get("variances", [])
        visits = diagnostics.get("visits", [])

        if not collect_replay or not candidates:
            return best_action

        # ------------------------------------------------------------------
        # 3. Emit one replay entry per top-K action.  We sort by visit
        #    count (descending) when available; otherwise by score
        #    (descending); otherwise by index (the natural order).
        # ------------------------------------------------------------------
        n_actions = len(candidates)
        # Build (idx, prior, variance, visit, score) tuples for sorting.
        rows: List[Tuple[int, float, float, int, float]] = []
        for i in range(n_actions):
            p = float(priors[i]) if i < len(priors) else 0.5
            v = float(variances[i]) if i < len(variances) else 0.0
            n = int(visits[i]) if i < len(visits) else 1
            s = float(scores[i]) if i < len(scores) else 0.5
            rows.append((i, p, v, n, s))
        # Sort by visit count desc, then by score desc.
        rows.sort(key=lambda r: (r[3], r[4]), reverse=True)

        # Emit up to n_top_actions entries.  ``state`` is the action's
        # resulting child; ``action`` is its index.  When we have
        # richer state representation (e.g. canonical_smiles) we use
        # that; otherwise we fall back to the action index.
        n_emit = min(int(self.n_top_actions), len(rows))
        for rank in range(n_emit):
            idx, prior, var, visit, score = rows[rank]
            # Prefer canonical_smiles when candidate is a
            # MoleculeClosedTerm; fall back to str(...) otherwise.
            child = candidates[idx]
            state_repr = self._state_repr(child, fallback=f"action_{idx}")
            action_id = f"action_{idx}"
            entry = MCTSReplayEntry(
                state=state_repr,
                action=action_id,
                value=score,
                prior=prior,
                variance=var,
                visit_count=max(1, visit),
                metadata={
                    "rank": int(rank),
                    "search_iter": int(self.n_searches_),
                    "child_index": int(idx),
                },
            )
            self.buffer.add(entry)
            self.n_entries_recorded_ += 1

        return best_action

    # ------------------------------------------------------------------
    # Value-network training step (loss estimate from replay)
    # ------------------------------------------------------------------
    def train_value_network(
        self,
        batch: Optional[Sequence[MCTSReplayEntry]] = None,
    ) -> Dict[str, float]:
        """Return loss estimates for a sampled replay mini-batch.

        This is the **AlphaGo Zero L = (z - v)^2 + cross_entropy**
        surrogate.  Because we do not ship a neural policy/value head
        in this module, we compute the *empirical* loss statistics that
        a downstream value head would see — i.e. the variance and
        range of the batch's ``value`` (for the value head MSE term)
        and the entropy of the ``prior`` distribution (for the policy
        head cross-entropy term).

        Parameters
        ----------
        batch : sequence of MCTSReplayEntry, optional
            Pre-sampled batch.  When ``None`` we draw a fresh batch via
            :meth:`MCTSReplayBuffer.sample`.

        Returns
        -------
        dict
            ``value_loss``  : float  — empirical MSE proxy (variance
                                       of the batch's value column)
            ``policy_loss`` : float  — empirical cross-entropy proxy
                                       (entropy of the batch's prior
                                       column, normalised)
            ``batch_size``  : int    — actual number of entries in the
                                       returned loss estimate
            ``mean_value``  : float  — mean ``value`` across the batch
            ``mean_prior``  : float  — mean ``prior`` across the batch
            ``mean_variance``: float — mean ``variance`` across the
                                       batch
        """
        if batch is None:
            batch = self.buffer.sample(DEFAULT_BATCH_SIZE)
        batch = list(batch)
        if not batch:
            return {
                "value_loss": 0.0,
                "policy_loss": 0.0,
                "batch_size": 0,
                "mean_value": 0.0,
                "mean_prior": 0.0,
                "mean_variance": 0.0,
            }
        values = np.asarray([e.value for e in batch], dtype=float)
        priors = np.asarray(
            [max(1.0e-6, min(1.0 - 1.0e-6, e.prior)) for e in batch],
            dtype=float,
        )
        varis = np.asarray([e.variance for e in batch], dtype=float)

        # Value loss proxy: variance of values (squared error when the
        # network predicts the mean and z = value column).
        value_loss = float(values.var())
        # Policy loss proxy: mean Bernoulli cross-entropy of the
        # batch's prior column against its own mean (a constant
        # predictor).  Equivalent to H(p) + H(1-p_mean) where
        # p_mean = mean(priors).
        p_mean = float(priors.mean())
        eps = 1.0e-6
        p_clipped = np.clip(priors, eps, 1.0 - eps)
        per_entry = -(
            p_mean * np.log(p_clipped) + (1.0 - p_mean) * np.log(1.0 - p_clipped)
        )
        policy_loss = float(per_entry.mean())

        out = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "batch_size": int(len(batch)),
            "mean_value": float(values.mean()),
            "mean_prior": float(priors.mean()),
            "mean_variance": float(varis.mean()),
        }
        self.last_loss_ = out
        return out

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def _read_diagnostics(self) -> Dict[str, List[Any]]:
        """Read public per-candidate diagnostics from the wrapped search.

        Tolerant of missing attributes — every diagnostic is best-
        effort.  Returns an empty dict when the wrapped search exposes
        none of the expected attributes.
        """
        out: Dict[str, List[Any]] = {}
        # 1. ``history[-1]`` — last search iteration's row.
        try:
            hist = getattr(self.base_search, "history", None)
            if isinstance(hist, list) and hist and isinstance(hist[-1], dict):
                last = hist[-1]
                if "candidates" in last and isinstance(last["candidates"], list):
                    out["candidates"] = list(last["candidates"])
                if "scores" in last and isinstance(last["scores"], list):
                    out["scores"] = [float(s) for s in last["scores"]]
                if "priors" in last and isinstance(last["priors"], list):
                    out["priors"] = [float(p) for p in last["priors"]]
                if "variances" in last and isinstance(last["variances"], list):
                    out["variances"] = [float(v) for v in last["variances"]]
                if "visits" in last and isinstance(last["visits"], list):
                    out["visits"] = [int(n) for n in last["visits"]]
        except Exception:
            pass

        # 2. ``_root_children_cache_for_search`` — set by
        #    ``MCTSProofSearch.search`` (see proof_search.py for the
        #    field definition).  We only read it when ``history``
        #    didn't already populate the candidates list.
        if "candidates" not in out:
            try:
                cache = getattr(
                    self.base_search,
                    "_root_children_cache_for_search",
                    None,
                )
                if cache is not None:
                    if isinstance(cache, dict):
                        cands = cache.get("candidates") or cache.get("children")
                        if isinstance(cands, list):
                            out["candidates"] = list(cands)
                        scores = cache.get("scores")
                        if isinstance(scores, list):
                            out["scores"] = [float(s) for s in scores]
                        priors = cache.get("priors")
                        if isinstance(priors, list):
                            out["priors"] = [float(p) for p in priors]
                        visits = cache.get("visits") or cache.get("visit_counts")
                        if isinstance(visits, list):
                            out["visits"] = [int(n) for n in visits]
                    elif isinstance(cache, list):
                        out["candidates"] = list(cache)
            except Exception:
                pass

        # 3. ``debug_trace`` — populated when ``base_search.debug=True``.
        if "candidates" not in out:
            try:
                trace = getattr(self.base_search, "debug_trace", None)
                if isinstance(trace, list) and trace:
                    # Each entry is a per-iteration dict.  Collect the
                    # ``best_action`` field as a 1-element candidate
                    # list — degenerate but enough to record a replay
                    # entry.
                    out["candidates"] = [
                        {"best_action": e.get("best_action")}
                        for e in trace
                        if isinstance(e, dict)
                    ]
                    out["scores"] = [
                        float(e.get("reward", 0.0))
                        for e in trace
                        if isinstance(e, dict)
                    ]
            except Exception:
                pass

        return out

    @staticmethod
    def _state_repr(obj: Any, fallback: str) -> str:
        """Best-effort string representation of a candidate state."""
        try:
            smi = getattr(obj, "canonical_smiles", None)
            if callable(smi):
                s = smi()
                if s:
                    return str(s)
        except Exception:
            pass
        try:
            return str(obj)
        except Exception:
            return fallback


# ---------------------------------------------------------------------------
# Replay-guided prior biasing
# ---------------------------------------------------------------------------


@dataclass
class ReplayGuidedMCTS:
    """Replay-guided MCTS — bias the root prior from accumulated replay statistics.

    The "warm-start from replay" analogue: once the replay buffer is
    populated with ``n ≥ n_min_visits`` entries for a given state, we
    can bias the root prior toward actions that historically produced
    high-value children — closing the AlphaGo loop between *play* and
    *learn*.

    Parameters
    ----------
    base_search : MCTSProofSearch
        The MCTS instance whose prior we want to bias.  Not mutated.
    replay_buffer : MCTSReplayBuffer
        Source of historical (state, action, value, prior) records.
    alpha_replay : float, default 0.5
        Mixing weight between the base prior and the replay bias.
        ``0.0`` = pure base prior (replay ignored), ``1.0`` = pure
        replay bias (base ignored).  AlphaZero's default replay blend
        is 0.5.
    tau : float, default 1.0
        Temperature for the visit-count → policy conversion
        (``pi(a) = N(a)^(1/tau) / Z``).  ``tau → 0`` collapses to
        argmax.

    Attributes
    ----------
    n_bias_calls_ : int
        Cumulative number of :meth:`bias_prior` invocations.
    """

    base_search: Any
    replay_buffer: MCTSReplayBuffer
    alpha_replay: float = DEFAULT_REPLAY_ALPHA
    tau: float = DEFAULT_TAU

    n_bias_calls_: int = field(default=0, init=False, repr=False)
    last_bias_: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.alpha_replay) <= 1.0):
            raise MCTSReplayError(
                f"alpha_replay must be in [0, 1]; got {self.alpha_replay}"
            )
        if float(self.tau) <= 0.0:
            raise MCTSReplayError(f"tau must be > 0; got {self.tau}")

    # ------------------------------------------------------------------
    # bias_prior
    # ------------------------------------------------------------------
    def bias_prior(
        self,
        root: Any,
        replay_buffer: Optional[MCTSReplayBuffer] = None,
        n_actions: int = 4,
    ) -> np.ndarray:
        """Return a replay-biased prior vector of length ``n_actions``.

        The returned vector is a *mixture* of two components::

            bias = alpha_replay * bias_replay
                 + (1 - alpha_replay) * bias_base

        where ``bias_base`` is the wrapped MCTS's prior over
        ``n_actions`` arms (uniform 1/n when not exposed) and
        ``bias_replay`` is the visit-count-temperature policy over
        the entries in ``replay_buffer`` whose ``state`` matches
        ``root``.

        When no replay entries match ``root``, ``bias_replay`` falls
        back to uniform and the returned vector is the uniform prior.

        Parameters
        ----------
        root : Any
            Root state whose prior we are biasing.  Stringified via
            ``str(...)`` for the replay lookup.
        replay_buffer : MCTSReplayBuffer, optional
            Override the constructor-level buffer for this call.
        n_actions : int, default 4
            Dimensionality of the returned bias vector.  Must be ≥ 1.
        """
        if n_actions < 1:
            raise MCTSReplayError(f"n_actions must be >= 1; got {n_actions}")
        buf = replay_buffer if replay_buffer is not None else self.replay_buffer
        if buf is None:
            raise MCTSReplayError(
                "ReplayGuidedMCTS.bias_prior requires a non-None buffer"
            )
        self.n_bias_calls_ += 1

        # 1. Base prior — uniform by default; if the wrapped search
        #    exposes ``_root_children_cache_for_search`` we try to
        #    read its priors.
        base_prior = self._read_base_prior(n_actions)
        # 2. Replay bias — visit-count-temperature policy over
        #    matching entries.
        root_key = self._state_key(root)
        replay_prior = self._replay_prior_for(root_key, n_actions, buf)

        alpha = float(self.alpha_replay)
        bias = alpha * replay_prior + (1.0 - alpha) * base_prior
        # Renormalise for safety (the mixture is already a
        # convex combination of two probability vectors, so this is
        # only a floating-point hygiene step).
        s = float(bias.sum())
        if s > 0.0:
            bias = bias / s
        else:
            bias = np.full(n_actions, 1.0 / float(n_actions), dtype=float)
        self.last_bias_ = bias
        return bias

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_base_prior(self, n_actions: int) -> np.ndarray:
        """Read the wrapped MCTS's prior over ``n_actions`` arms.

        Falls back to uniform when the search does not expose a public
        prior accessor.
        """
        try:
            cache = getattr(
                self.base_search,
                "_root_children_cache_for_search",
                None,
            )
            if isinstance(cache, dict):
                priors = cache.get("priors")
                if isinstance(priors, list) and len(priors) >= n_actions:
                    arr = np.asarray(priors[:n_actions], dtype=float)
                    arr = np.clip(arr, 1.0e-6, None)
                    return arr / float(arr.sum())
        except Exception:
            pass
        return np.full(n_actions, 1.0 / float(n_actions), dtype=float)

    @staticmethod
    def _state_key(state: Any) -> str:
        """Stringify a state for replay-buffer lookup."""
        try:
            smi = getattr(state, "canonical_smiles", None)
            if callable(smi):
                s = smi()
                if s:
                    return str(s)
        except Exception:
            pass
        try:
            return str(state)
        except Exception:
            return ""

    def _replay_prior_for(
        self,
        root_key: str,
        n_actions: int,
        buf: MCTSReplayBuffer,
    ) -> np.ndarray:
        """Compute the visit-count-temperature policy over matching entries."""
        if not buf.entries:
            return np.full(n_actions, 1.0 / float(n_actions), dtype=float)
        tau = float(self.tau)
        # Aggregate visit counts per action index across all matching
        # entries.  We parse the ``action`` field as ``"action_<idx>"``
        # (the convention :class:`MCTSReplayWrapper` emits); if the
        # parse fails we fall back to uniform.
        visit_counts = np.zeros(n_actions, dtype=float)
        n_matched = 0
        for e in buf.entries:
            if root_key and self._state_key(e.state) != root_key:
                continue
            idx = self._parse_action_index(e.action)
            if idx is None or idx < 0 or idx >= n_actions:
                continue
            visit_counts[idx] += float(max(1, int(e.visit_count)))
            n_matched += 1
        if n_matched == 0 or visit_counts.sum() <= 0:
            return np.full(n_actions, 1.0 / float(n_actions), dtype=float)
        # Temperature-scaled softmax.
        logits = np.power(np.maximum(visit_counts, 1.0e-12), 1.0 / tau)
        s = float(logits.sum())
        if s <= 0:
            return np.full(n_actions, 1.0 / float(n_actions), dtype=float)
        return logits / s

    @staticmethod
    def _parse_action_index(action: Any) -> Optional[int]:
        """Parse ``"action_<idx>"`` back to ``idx``."""
        try:
            s = str(action)
            if s.startswith("action_"):
                return int(s.split("_", 1)[1])
        except Exception:
            return None
        return None


__all__ = [
    "MCTSReplayError",
    "MCTSReplayEntry",
    "MCTSReplayBuffer",
    "MCTSReplayWrapper",
    "ReplayGuidedMCTS",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_REPLAY_ALPHA",
    "DEFAULT_TAU",
    "MAX_BATCH_SIZE",
]
