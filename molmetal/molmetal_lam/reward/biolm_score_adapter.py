"""biolm_score_adapter.py — BioLM-Score adapter as `r_biolm_score` reward channel.

================================================================
What this is
================================================================
Adapters the BioLM-Score binding-affinity predictor behind
:mod:`molmetal_lam.sbdd_env.biomlm_sota_scoring` as a
:class:`RewardChannel` Protocol-conformant callable that can be
wired into :class:`molmetal_lam.search_alg.proof_search.RewardAggregator`
via :meth:`RewardAggregator.register_biolm_score_channel` (or directly
via ``aggregator.r_biolm_score = ch.register_channel()``).

Output semantics:

* :meth:`BiolmScoreChannel.score` returns a finite ``float`` in
  ``[0, 1]`` where higher = more confident binding (i.e. more drug-
  like per the BioLM-Score confidence convention — see
  :mod:`molmetal_lam.sbdd_env.biomlm_sota_scoring`
  ("Honest framing" section) for the sign convention).
* The function is graceful-degrading: when the upstream BioLM-Score
  CLI is unavailable, when torch or the ``.pth`` checkpoint is
  missing, or when the candidate SMILES does not parse, the
  channel returns ``0.0``.  This mirrors the existing PlatinAI /
  REINVENT4 / ADMET reward-channel contract.

Wiring contract
===============

* :class:`BiolmScoreChannel` exposes :meth:`score` and
  :meth:`register_channel`, where :meth:`register_channel` returns
  a ``(state) -> float`` closure suitable for
  :attr:`RewardAggregator.r_biolm_score`.
* The module-import path does NOT spawn subprocesses or load
  upstream model weights; weight load happens once at
  :meth:`BiolmScoreChannel.__init__` (or via the first subprocess
  invocation — depends on the upstream CLI contract).
* Tests are designed to be CPU-only and LIGHTWEIGHT: setting the
  env var ``BIOLM_DISABLE_WEIGHTS=1`` (or, equivalently, default
  pytest behaviour on this CPU-only host) makes
  :meth:`BiolmScoreChannel.score` return ``0.0`` without
  loading any model weights.  This guarantees the unit-test
  surface is reproducible and fast — the upstream BioLM-Score
  inference is GPU-bound (torch_scatter + ESM-C + Chemformer
  embeddings) and is not available in the local RX 7800 XT /
  Blackwell smoke-test environments.

Lit anchors
===========

* BioLM-Score, the upstream protein–ligand binding-affinity
  predictor trained on CASF-2016; see the vendored repo at
  ``molmetal/references/BioLM-Score`` (script
  ``scripts/casf2016_scoring_ranking.py``) and
  :mod:`molmetal_lam.sbdd_env.biomlm_sota_scoring` for the
  upstream wire-up.
* Dayan 1997 (potential-based reward shaping is policy-
  preserving) — this module implements a *reward channel*
  sitting on top of the existing MCTS aggregator so adding
  the BioLM-Score term keeps existing reward bit-for-bit
  identical when unused.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default BioLM-Score repository root — explicit candidates walked at
# import time.  Order matters: the first existing directory wins.
# ---------------------------------------------------------------------------
def _first_existing_repo_root(candidates: Tuple[Path, ...]) -> str:
    """Return the absolute string of the first directory in ``candidates``
    that exists on disk.

    Behaviour
    ---------
    * Walks ``candidates`` in order; the first directory that
      :py:meth:`pathlib.Path.is_dir` reports as existing is returned as
      ``str(cand.resolve())``.
    * When *no* candidate exists, fall back to ``str(candidates[0].resolve())``
      so callers always receive a string — downstream
      :class:`BiolmScoreChannel` treats this as a discovery hint and
      degrades to ``0.0`` when the path is still missing at ``score()``
      time.  An empty tuple yields ``""``.
    * Individual probes never raise; ``OSError`` / ``ValueError`` from
      bad symlinks / unreadable parents are swallowed so import-time
      discovery is robust on hostile filesystems.
    """
    for cand in candidates:
        try:
            if cand.is_dir():
                return str(cand.resolve())
        except Exception:
            continue
    if candidates:
        try:
            return str(candidates[0].resolve())
        except Exception:
            return str(candidates[0])
    return ""


#: Tuple of candidate BioLM-Score repository roots, in priority order.
#: The first existing directory wins at import time.  Two canonical
#: locations are checked:
#:
#: 1. The host-absolute checkout
#:    (``/home/hugo/codes/mol-metal/references/BioLM-Score``) — primary
#:    on this development host.
#: 2. The in-repo vendored location
#:    (``<repo-root>/molmetal/references/BioLM-Score``) — primary in
#:    distributable wheels / other hosts where the host-absolute path
#:    doesn't exist.
_BIOLM_CANDIDATES: Tuple[Path, ...] = (
    Path(__file__).resolve().parents[3] / "references" / "BioLM-Score",
    Path(__file__).resolve().parents[2] / "references" / "BioLM-Score",
)

#: Module-level default — the absolute path string of the first
#: existing candidate in :data:`_BIOLM_CANDIDATES`.  When no candidate
#: exists, this is the first candidate resolved as a string
#: (best-effort); an empty tuple yields ``""``.  Callers should treat
#: this as a discovery hint and degrade gracefully when the path is
#: missing at ``score()`` time.
_DEFAULT_BIOLM_REPO_ROOT: str = _first_existing_repo_root(_BIOLM_CANDIDATES)


# ---------------------------------------------------------------------------
# Lazy import guard — BioLM-Score is heavy (torch + torch_scatter + ESM3 +
# Chemformer).  If the upstream package is not importable we never raise —
# we fall back to a permanent 0.0 score.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.sbdd_env.biomlm_sota_scoring import (
        score_candidates as _biomlm_score_candidates,
        biomlm_cli_available as _biomlm_cli_available,
        BioLMSScoreColumn as _BioLMSScoreColumn,
    )

    _HAS_BIOLM = True
except Exception as exc:  # pragma: no cover - defensive
    # We deliberately swallow the import error so the channel can be
    # constructed in any environment.  The lazy-import guard at score()
    # time falls back to 0.0.  ``_DEFAULT_BIOLM_REPO_ROOT`` is defined
    # at module level above (via the candidates walk) so callers can
    # construct a :class:`BiolmScoreChannel` even when the upstream
    # package is unimportable — the channel degrades to ``0.0`` on
    # missing paths.
    _HAS_BIOLM = False
    _biomlm_score_candidates = None  # type: ignore[assignment]
    _biomlm_cli_available = None  # type: ignore[assignment]
    _BioLMSScoreColumn = None  # type: ignore[assignment]
    _BIOLM_IMPORT_ERROR = repr(exc)
else:  # pragma: no cover - reported in diagnostics only
    _BIOLM_IMPORT_ERROR = ""


# ---------------------------------------------------------------------------
# Constants — module init / weight loading
# ---------------------------------------------------------------------------
#: Env-var name used to disable weight loads in the test surface.  When
#: this env var is set to a truthy value (``"1"``, ``"true"``, ``"yes"``,
#: ``"on"``), :meth:`BiolmScoreChannel.score` returns ``0.0`` without
#: spawning the upstream CLI subprocess or loading any model weights.
#: The contract is enforced at ``score()`` time so a misconfigured
#: production host that still has the env var set will fall back to
#: 0.0 — never crash MCTS.
_DISABLE_WEIGHTS_ENV: str = "BIOLM_DISABLE_WEIGHTS"


def _truthy_env(name: str) -> bool:
    """Return True iff ``os.environ[name]`` is one of ``{1, true, yes, on}``."""
    try:
        v = str(os.environ.get(name, "") or "").strip().lower()
    except Exception:
        return False
    return v in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Protocol — matches the RewardChannel shape used elsewhere in the reward
# package (see ``molmetal_lam/reward/platinai_oracle.py`` for the closest
# analogue).  Not a typing.Protocol because we want to keep the import
# surface zero (no typing_extensions dependency).
# ---------------------------------------------------------------------------
class RewardChannel:
    """Minimal protocol every reward-channel adapter implements.

    Subclasses MUST implement:

    * :meth:`score` — ``(smiles: str, pocket: Any | None) -> float``
      returning a finite value in ``[0, 1]`` where higher = better.
    * :meth:`register_channel` — returns a ``(state) -> float``
      closure suitable for ``RewardAggregator.r_biolm_score``.

    The base class provides graceful-degradation fallbacks (``0.0``)
    so subclasses can selectively override the heavy-lifting parts.
    """

    #: Per-instance flag set by :meth:`__init__` — ``True`` iff the
    #: upstream BioLM-Score backend is reachable (CLI script + package
    #: marker present, no ``BIOLM_DISABLE_WEIGHTS`` env-var override).
    available: bool = False

    def score(self, smiles: str, pocket: Any = None) -> float:
        """Return BioLM-Score-derived binding-affinity score in ``[0, 1]``.

        Parameters
        ----------
        smiles : str
            Canonical SMILES of the candidate molecule.  Empty / unparsable
            SMILES fall back to ``0.0``.
        pocket : Any, default ``None``
            Free-form pocket identifier (typically a PDB path or
            ``PocketId`` enum-like).  Forwarded to the upstream
            CLI as ``protein_path`` for diagnostics; the upstream
            scoring CLI reads its own ``data_dir`` and ignores
            the protein path beyond the dataclass record.

        Returns
        -------
        float
            Finite value in ``[0, 1]``; higher = more confident binding.
        """
        raise NotImplementedError  # subclasses MUST override

    def register_channel(self) -> Callable[[Any], float]:
        """Return a ``(state) -> float`` closure for ``RewardAggregator``.

        The closure accepts both plain SMILES strings and state objects
        (anything with a ``canonical_smiles()`` method or ``smiles``
        attribute) — exactly the protocol used by the PlatinAI oracle
        channel closure in :mod:`platinai_oracle`.

        Returns
        -------
        callable
            ``(state) -> float`` suitable for ``RewardAggregator.r_biolm_score``.
        """
        channel_self = self

        def _closure(state: Any) -> float:
            if state is None:
                return 0.0
            if isinstance(state, str):
                smi = state
            else:
                try:
                    fn = getattr(state, "canonical_smiles", None)
                    if callable(fn):
                        smi = str(fn() or "")
                    else:
                        smi = str(getattr(state, "smiles", "") or "")
                except Exception:
                    return 0.0
            try:
                return channel_self.score(smi, pocket=None)
            except Exception:
                return 0.0

        return _closure


# ---------------------------------------------------------------------------
# Concrete adapter — wraps the upstream BioLM-Score CLI subprocess wrapper
# ---------------------------------------------------------------------------
@dataclass
class BiolmScoreChannel(RewardChannel):
    """In-process BioLM-Score reward channel.

    Parameters
    ----------
    repo_root : str, default ``DEFAULT_BIOLM_REPO_ROOT``
        Path to the cloned BioLM-Score checkout.  Used only as a
        *discovery hint* — when the upstream CLI cannot be located
        the channel silently degrades to ``0.0`` (never crashes
        MCTS).
    encoder : str, default ``"gatedgcn"``
        Which upstream graph encoder to invoke.  Mirrors the
        ``--encoder`` CLI flag.
    model_type : str, default ``"biolm"``
        Which upstream model variant.  ``"biolm"`` requires the
        protein + ligand embedding pipeline; ``"genscore"`` uses
        graph features only.
    ckpt_path : str | None, default ``None``
        Optional path to a BioLM-Score ``.pth`` checkpoint.  When
        ``None`` the upstream CLI complains at runtime (we
        deliberately do NOT synthesise weights).
    timeout_sec : float, default ``60.0``
        Per-call subprocess wall-clock budget.  Smaller than the
        600s default in :mod:`biomlm_sota_scoring` because the
        channel is called once per MCTS leaf and we want it to
        degrade quickly when the GPU backend is unavailable.
    pocket : Any, default ``None``
        Default pocket identifier to forward to the upstream CLI as
        ``protein_path``.  Can be overridden per-call via
        :meth:`score(smiles, pocket=...)`.

    Notes
    -----
    * **Weight loading semantics** — the constructor does NOT load
      any model weights in-process; the upstream CLI is spawned per
      ``score()`` call (or batched, when the caller batch-wraps the
      channel).  Setting ``BIOLM_DISABLE_WEIGHTS=1`` in the
      environment (or, equivalently, instantiating the channel with
      ``disable_weights=True``) short-circuits the channel to
      return ``0.0`` without spawning any subprocess.
    * **Thread safety** — the channel acquires an internal lock
      around the subprocess invocation so concurrent MCTS workers
      do not collide on the upstream CLI's tempdir / ``.dat``
      output file.

    Honest framing
    --------------
    * BioLM-Score inference is GPU-bound.  On CPU-only smoke-test
      hosts the upstream CLI is unavailable and the channel returns
      ``0.0`` regardless of ``disable_weights``.  This matches the
      honest behaviour documented in
      :mod:`molmetal_lam.sbdd_env.biomlm_sota_scoring`.
    """

    repo_root: str = _DEFAULT_BIOLM_REPO_ROOT
    encoder: str = "gatedgcn"
    model_type: str = "biolm"
    ckpt_path: Optional[str] = None
    timeout_sec: float = 60.0
    pocket: Any = None
    #: Per-call override; defaults to deriving from the
    #: ``BIOLM_DISABLE_WEIGHTS`` env var at construction time.
    disable_weights: bool = field(
        default_factory=lambda: _truthy_env(_DISABLE_WEIGHTS_ENV)
    )

    # ---- internal state ------------------------------------------------
    #: Instance-level lock guarding subprocess invocation.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Whether the upstream CLI was reachable at construction time.
    _cli_available: bool = field(default=False, init=False, repr=False)
    #: Per-instance cache: ``{smiles: score}``.  Set to ``None`` to
    #: bypass the cache (useful in tests).
    _score_cache: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    #: Number of ``score()`` calls dispatched (diagnostic).
    _n_scored: int = field(default=0, init=False, repr=False)
    #: Number of ``score()`` calls that fell back to ``0.0`` (diagnostic).
    _n_degraded: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Probe the upstream CLI once at construction (no model load).

        * Sets :attr:`available` to ``True`` iff the upstream
          BioLM-Score CLI package marker + scoring script are
          present AND ``disable_weights`` is False.
        * The probe NEVER spawns a subprocess (that's deferred to
          :meth:`score`); ``available`` only reflects whether the
          vendored repo can be located.  Honest framing: this
          guards a fast-discovery path, not a real e2e test of
          GPU availability.
        """
        try:
            cli_ok = (
                bool(_biomlm_cli_available(self.repo_root)) if _HAS_BIOLM else False
            )
        except Exception:
            cli_ok = False
        self._cli_available = bool(cli_ok) and not bool(self.disable_weights)
        self.available = bool(self._cli_available)
        if self.disable_weights:
            logger.info(
                "BiolmScoreChannel: BIOLM_DISABLE_WEIGHTS=1 — channel "
                "returns 0.0 (no model load, no subprocess).",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Return a JSON-safe dict of diagnostics for reporting."""
        return {
            "available": bool(self.available),
            "cli_available": bool(self._cli_available),
            "disable_weights": bool(self.disable_weights),
            "has_upstream_import": bool(_HAS_BIOLM),
            "encoder": str(self.encoder),
            "model_type": str(self.model_type),
            "repo_root": str(self.repo_root),
            "ckpt_path": str(self.ckpt_path) if self.ckpt_path else None,
            "n_scored": int(self._n_scored),
            "n_degraded": int(self._n_degraded),
        }

    def score(self, smiles: str, pocket: Any = None) -> float:
        """Return BioLM-Score binding confidence in ``[0, 1]``.

        Behaviour matrix:

        ======================  =====================================
        Inputs                  Behaviour
        ======================  =====================================
        ``smiles`` empty/None   returns ``0.0`` (degraded)
        ``disable_weights`` T   returns ``0.0`` (no subprocess)
        CLI unavailable         returns ``0.0`` (degraded)
        subprocess failure      returns ``0.0`` (degraded)
        finite ``mean``         clamped to ``[0, 1]``
        ======================  =====================================
        """
        # --- (a) cheap degrade paths (no IO, no subprocess) ---------
        if not isinstance(smiles, str) or not smiles.strip():
            self._n_degraded += 1
            return 0.0
        if self.disable_weights or not _HAS_BIOLM or not self._cli_available:
            self._n_degraded += 1
            return 0.0

        smi = smiles.strip()
        # --- (b) cache lookup (cheaper than invoking the CLI) -------
        # The cache key is the (smiles, pocket) pair so callers can
        # re-evaluate with a different pocket without poisoning the
        # cache.  ``pocket`` is part of the key only when supplied;
        # the default ``None`` collapses to a single bucket.
        cache_token = f"{smi}::{pocket!r}"
        if cache_token in self._score_cache:
            self._n_scored += 1
            return float(self._score_cache[cache_token])

        # --- (c) subprocess path ------------------------------------
        # Wrap the upstream ``score_candidates`` API — see
        # :mod:`biomlm_sota_scoring` for the full CLI semantics.
        candidate = {"smiles": smi}
        # ``protein_path`` is purely diagnostic; the upstream CLI
        # reads its own ``data_dir`` argument.  We honour
        # BioLM-Score's contract: empty / None protein path is fine
        # in the smoke-test environment.
        protein_path = ""
        try:
            protein_path = str(pocket) if pocket is not None else ""
        except Exception:
            protein_path = ""
        if not protein_path:
            protein_path = str(self.pocket) if self.pocket is not None else ""

        # Acquire the lock so concurrent MCTS workers don't race
        # on the upstream CLI's tempdir / .dat output file.  We
        # batch a single candidate per call — callers wanting a
        # batched path should wrap the channel externally.
        with self._lock:
            try:
                column = _biomlm_score_candidates(
                    [candidate],
                    protein_path=protein_path or ".",
                    repo_root=self.repo_root,
                    encoder=self.encoder,
                    model_type=self.model_type,
                    ckpt_path=self.ckpt_path,
                    timeout_sec=float(self.timeout_sec),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "BiolmScoreChannel.score: upstream CLI raised %s; "
                    "falling back to 0.0",
                    exc,
                )
                self._n_degraded += 1
                return 0.0

        # --- (d) result extraction + clamping -----------------------
        if column is None:
            self._n_degraded += 1
            return 0.0

        # ``BioLMSScoreColumn`` exposes ``biomlm_score_mean`` (float
        # | None) and ``per_smiles`` (dict[str, float]); we prefer
        # the per-SMILES slot so different candidates in the same
        # batch don't share a single (already-averaged) score.
        raw: Optional[float] = None
        try:
            per = getattr(column, "per_smiles", None) or {}
            if isinstance(per, dict) and smi in per:
                raw = float(per[smi])
        except Exception:
            raw = None
        if raw is None or (raw != raw):  # NaN-safe
            try:
                mean = getattr(column, "biomlm_score_mean", None)
                if mean is not None and mean == mean:
                    raw = float(mean)
            except Exception:
                raw = None
        if raw is None or (raw != raw):  # NaN-safe
            self._n_degraded += 1
            return 0.0

        try:
            clamped = float(max(0.0, min(1.0, raw)))
        except Exception:
            self._n_degraded += 1
            return 0.0

        self._score_cache[cache_token] = clamped
        self._n_scored += 1
        return clamped

    def register_channel(self) -> Callable[[Any], float]:
        """Return a ``(state) -> float`` closure for ``RewardAggregator``.

        The closure accepts both plain SMILES strings and state objects
        (``canonical_smiles()`` / ``smiles`` attribute) — it never
        raises; the underlying :meth:`score` already swallows every
        exception and returns ``0.0`` on degradation.
        """
        # Inherit the base-class implementation (it routes ``state``
        # → ``score(smi, pocket=None)``).
        return super().register_channel()


# ---------------------------------------------------------------------------
# RewardAggregator hook — symmetric with register_platinai_oracle_channel
# ---------------------------------------------------------------------------
def register_biolm_score_channel(
    aggregator: Any,
    *,
    channel: Optional[BiolmScoreChannel] = None,
    repo_root: str = _DEFAULT_BIOLM_REPO_ROOT,
    encoder: str = "gatedgcn",
    model_type: str = "biolm",
    ckpt_path: Optional[str] = None,
    timeout_sec: float = 60.0,
    disable_weights: Optional[bool] = None,
    weight: float = 0.0,
) -> BiolmScoreChannel:
    """Wire the ``r_biolm_score`` channel on a :class:`RewardAggregator`.

    Behaviour
    ---------
    * Constructs a :class:`BiolmScoreChannel` (or accepts a
      pre-built one via ``channel=``), installs it as
      ``aggregator.r_biolm_score`` (NEW attribute, additive only —
      the existing ``r_vina_proxy``, ``r_platinai``, ``r_qed``, ``r_sa``
      slots are not touched), and stashes a handle on
      ``aggregator._biolm_channel`` for diagnostics.
    * Default weight is ``0.0`` so existing reward is bit-for-bit
      identical when the channel is not enabled.  Set
      ``weight=1.0`` to make the BioLM-Score contribution compete on
      the same footing as the other reward channels.
    * The function is safe to call repeatedly — subsequent
      invocations REPLACE the channel closure (matches the
      :meth:`RewardAggregator.register_platinai_oracle_channel`
      contract).

    Parameters
    ----------
    aggregator : RewardAggregator
        The aggregator instance.  Mutated in place.
    channel : BiolmScoreChannel, optional
        Pre-built channel; ``None`` → construct a default one.
    repo_root, encoder, model_type, ckpt_path, timeout_sec :
        Forwarded to :class:`BiolmScoreChannel` when ``channel is None``.
    disable_weights : bool | None, default ``None``
        Override for ``disable_weights``; ``None`` → derive from the
        ``BIOLM_DISABLE_WEIGHTS`` env var via the dataclass default.
    weight : float, default ``0.0``
        Per-channel weight.  ``0.0`` = opt-in / silent.

    Returns
    -------
    BiolmScoreChannel
        The (possibly freshly constructed) channel instance.
    """
    if channel is None:
        # Walk the candidates list if the configured ``repo_root`` is
        # missing — mirrors the import-time discovery so a caller that
        # passes an explicit ``repo_root`` which has since been moved /
        # deleted still gets a working fallback.  ``_DEFAULT_BIOLM_REPO_ROOT``
        # already encodes the first existing candidate at import time;
        # the explicit walk below additionally covers the case where
        # the caller passed a *different* path that happens to be
        # missing on this host.
        resolved_root = str(repo_root) if repo_root else ""
        try:
            if not resolved_root or not Path(resolved_root).is_dir():
                resolved_root = _first_existing_repo_root(_BIOLM_CANDIDATES)
        except Exception:
            resolved_root = str(_DEFAULT_BIOLM_REPO_ROOT or "")

        kwargs: Dict[str, Any] = dict(
            repo_root=resolved_root,
            encoder=encoder,
            model_type=model_type,
            timeout_sec=float(timeout_sec),
        )
        if ckpt_path is not None:
            kwargs["ckpt_path"] = ckpt_path
        if disable_weights is not None:
            kwargs["disable_weights"] = bool(disable_weights)
        channel = BiolmScoreChannel(**kwargs)

    setattr(aggregator, "r_biolm_score", channel.register_channel())
    setattr(aggregator, "w_biolm_score", float(weight))
    setattr(aggregator, "_biolm_channel", channel)
    return channel


__all__ = [
    "BiolmScoreChannel",
    "RewardChannel",
    "register_biolm_score_channel",
    "_DEFAULT_BIOLM_REPO_ROOT",
    "_DISABLE_WEIGHTS_ENV",
    "_HAS_BIOLM",
]
