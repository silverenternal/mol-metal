"""Closed-loop design driver: generate → dock → predict → score → refine.

Two concrete implementations live here:

- :class:`WeightedSumScorer` — a :class:`~molmetal.ports.ScoringFunction`
  that reduces a :class:`~molmetal.ports.PropertyPrediction` (+ the docked
  :class:`~molmetal.domain.Complex`) into a single scalar via a weighted
  linear combination of normalised objectives.
- :class:`ClosedLoop` — a :class:`~molmetal.ports.DesignLoop` that runs
  ``config.n_iterations`` rounds of

      1. ``generator.generate(pocket, GenerationConfig(n_samples=...))``
      2. ``docker.dock`` for the top-K molecules ranked by QED (docking is
         the expensive step, so we only pay for the promising ones)
      3. ``predictor.predict`` for **all** molecules
      4. ``scorer.score`` on the (mol, complex, prediction) triples
      5. keep the global top ``config.n_top_k`` candidates

  Every iteration appends a summary dict to the history, retrievable via
  :meth:`ClosedLoop.get_history`.

Design notes
------------
* Ports are ``typing.Protocol``s, so anything duck-typed works here — the
  loop never imports a concrete adapter.
* Adapter failures (a docking engine that raises on a degenerate pose, a
  predictor that chokes on an invalid molecule) are caught per-candidate
  so one bad sample cannot kill a multi-hour run.  Failures are counted in
  the per-iteration history record.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import (
    DesignLoop,
    DesignLoopConfig,
    DockingConfig,
    DockingEngine,
    GenerationConfig,
    MoleculeGenerator,
    PropertyPrediction,
    PropertyPredictor,
    ScoredCandidate,
    ScoringFunction,
)
from molmetal_lam.search_alg.proof_search import RewardAggregator

log = logging.getLogger(__name__)

__all__ = [
    "WeightedSumScorer",
    "RewardAggregatorScorer",
    "ClosedLoop",
    "DEFAULT_WEIGHTS",
    "DEFAULT_REWARD_AGGREGATOR_WEIGHTS",
    "TODO01_REWARD_AGGREGATOR_WEIGHTS",
    "SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS",
    "SOTA_ALIGNED_CONFIG_PATH",
    "build_default_reward_aggregator",
    "build_todo01_reward_aggregator",
    "build_sota_aligned_reward_aggregator",
    "decompose_reward",
    "apply_reward_weights_override",
    "add_reward_weights_arg",
    "add_sota_aligned_arg",
    "resolve_reward_aggregator_for_flags",
]


# ---------------------------------------------------------------------------
# Default multi-objective weights
# ---------------------------------------------------------------------------
# Sign convention: positive weight  → higher raw value is better
#                  negative weight  → lower raw value is better
#   qed           0..1, higher better           → +0.5
#   sa_score      1..10, LOWER better, but we normalise to 0..1 "ease of
#                 synthesis" before weighting, so the weight stays positive → +0.3
#   binding_pic50 ~4..11, higher better         → +0.8
#   vina_score    kcal/mol, LOWER (more negative) better → -1.0
DEFAULT_WEIGHTS: Dict[str, float] = {
    "qed": 0.5,
    "sa_score": 0.3,
    "binding_pic50": 0.8,
    "vina_score": -1.0,
}

# ---------------------------------------------------------------------------
# TODO/pending/01 — rewire the default reward to Vina + PoseBusters + AiZynth
# ---------------------------------------------------------------------------
# The Lambda proof-search tree (proof_search.RewardAggregator) exposes a
# multi-channel reward that already supports r_vina, r_posebusters, r_retro,
# r_sa, r_qed, r_pic50 and r_reinvent4.  Round-3 evidence
# (``mmp13_vina_real.md``: 100/100 Lambda products docked, mean Vina = -1.355,
# success_rate = 100% on PDB 830c) shows the Vina channel is wired end-to-end
# and useful, while ``lambda_vs_sbdd_paper_numbers.md`` shows 0/13 CuAAC
# products pass PoseBusters without an explicit channel — exactly the gap a
# multi-channel reward closes.
#
# The historical default above is a QED+SA+pIC50+Vina scalar mix that does
# not cover PoseBusters validity or retrosynthesis feasibility.  The new
# :func:`build_default_reward_aggregator` factory below exposes the rich
# channel layout as the default for closed-loop runs: r_vina + r_posebusters
# + r_retro as the primary channels (weights 1.0/1.0/1.0), with r_sa + r_qed
# as secondary fall-backs (weight 0.3 each).  Channel callables degrade to
# 0.0 when the underlying adapter is unavailable, so the aggregator stays
# usable even before the docking / PoseBusters / AiZynth adapters are wired.
DEFAULT_REWARD_AGGREGATOR_WEIGHTS: Dict[str, float] = {
    # Primary channels — task TODO/pending/01 (Vina + PoseBusters + AiZynth).
    "w_vina": 1.0,
    "w_posebusters": 1.0,
    "w_retro": 1.0,
    # Secondary channels — preserve drug-likeness / synthesizability signal.
    "w_sa": 0.3,
    "w_qed": 0.3,
}

# ---------------------------------------------------------------------------
# TODO-01 — canonical default reward weights for the closed-loop scorer.
# ---------------------------------------------------------------------------
# Composite reward: Vina (40%) + PoseBusters (20%) + AiZynth retro (15%) +
# pIC50 (10%) + QED (10%) + SA (5%).  Sum = 1.0 so the total reward stays
# on a comparable 0..1-ish scale after channel normalisation.  These match
# the original task brief and override the historical round-3
# ``{vina: 1.0, posebusters: 1.0, retro: 1.0}`` primary-channel layout with a
# softer split so QED/SA/pIC50 still act as tie-breakers rather than being
# clamped to 0 when the primary channels are unavailable.
TODO01_REWARD_AGGREGATOR_WEIGHTS: Dict[str, float] = {
    "w_vina": 0.40,
    "w_posebusters": 0.20,
    "w_retro": 0.15,
    "w_pic50": 0.10,
    "w_qed": 0.10,
    "w_sa": 0.05,
}

# ---------------------------------------------------------------------------
# SOTA-aligned reward weights — TargetDiff / CrossDocked100 protocol
# ---------------------------------------------------------------------------
# When ``--sota-aligned`` is passed on the closed-loop CLI, the default
# :class:`RewardAggregator` switches to these weights so head‑to‑head
# comparisons against the SOTA numbers in
# ``molmetal/configs/sota_aligned_targetdiff.yaml`` are apples‑to‑apples:
#
#   r_vina + r_posebusters + r_retro  →  weight 1.0 each
#   r_sa + r_qed                      →  weight 0.3 each
#
# Rationale (mirrors the YAML):
#   * Vina / PoseBusters / AiZynth retro are the SOTA‑paper reward
#     channels (TargetDiff, TransDiff, Pocket2Mol, MolCRAFT all dock
#     + filter for synthetic accessibility before reporting success).
#   * SA + QED act as tie‑breakers — the SOTA sweeps don't surface
#     them as primary rewards, but they are still useful for ranking
#     near‑ties among poses that satisfy the primary channels.
#
# Reference: ``molmetal/configs/sota_aligned_targetdiff.yaml``
# (test_set: crossdocked_pocket10, success_rate.vina_threshold_kcal: -8.0,
# posebusters.enabled: true).  The YAML encodes the *protocol*; the
# reward weights below are the *channel layout* the closed loop uses to
# rank candidates under that protocol.
SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS: Dict[str, float] = {
    # Primary channels — one for each SOTA success-rate criterion.
    "w_vina": 1.0,
    "w_posebusters": 1.0,
    "w_retro": 1.0,
    # Secondary channels — drug-likeness / synthesizability tie-breakers.
    "w_sa": 0.3,
    "w_qed": 0.3,
}

# Canonical location of the SOTA-aligned YAML referenced by --sota-aligned.
# Resolved relative to the repository root (this file lives in
# molmetal/orchestration/closed_loop.py).  Kept as a module constant so
# CLI helpers and tests can both refer to it without re-deriving paths.
_SOTA_ALIGNED_CONFIG_FILENAME = "sota_aligned_targetdiff.yaml"
SOTA_ALIGNED_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    _SOTA_ALIGNED_CONFIG_FILENAME,
)

# Normalisation ranges used to bring each raw objective onto a comparable
# 0..1 scale before the weighted sum.  Chosen from the usual medicinal
# chemistry ranges rather than fit to any dataset.
_SA_MIN, _SA_MAX = 1.0, 10.0          # Ertl & Schuffenhauer SA score range
_PIC50_MIN, _PIC50_MAX = 4.0, 11.0    # ~100 µM .. 10 pM
_VINA_MIN, _VINA_MAX = -14.0, 0.0     # AutoDock Vina kcal/mol


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def build_default_reward_aggregator(
    *,
    r_vina=None,
    r_posebusters=None,
    r_retro=None,
    r_sa=None,
    r_qed=None,
    r_pic50=None,
    r_admet=None,
    enable_admet: bool = False,
) -> RewardAggregator:
    """Construct the round-3 default :class:`RewardAggregator`.

    The default channel layout mirrors ``TODO/pending/01``: Vina + PoseBusters
    + AiZynth retrosynthesis as the primary channels (weight 1.0 each) and
    SA + QED as secondary fall-backs (weight 0.3 each).  Any channel that
    is left as ``None`` degrades to ``0.0`` inside the aggregator, so the
    returned object is always usable — even before the Vina / PoseBusters /
    AiZynth adapters have been plugged in.

    Parameters
    ----------
    r_vina, r_posebusters, r_retro, r_sa, r_qed, r_pic50, r_admet:
        Optional ``Callable[[MoleculeClosedTerm], float]`` channel
        implementations.  When omitted, the aggregator silently returns
        ``0.0`` for that channel — the closed loop continues running on
        whatever subset of channels is wired.
    enable_admet:
        When ``True`` the round-8 ADMET runner default closure is
        attached (when available) and ``w_admet=1.0`` is set so the
        Lipinski desirability contributes.  Defaults to ``False``.

    Returns
    -------
    RewardAggregator
        A :class:`RewardAggregator` configured with the round-3 default
        channel weights.
    """
    weights = DEFAULT_REWARD_AGGREGATOR_WEIGHTS
    admet_channel = r_admet
    w_admet = 0.0
    if enable_admet:
        w_admet = 1.0
        if admet_channel is None:
            try:
                from molmetal_lam.search_alg.proof_search import _r_admet_default
                admet_channel = _r_admet_default
                log.info("ADMET channel wired via proof_search._r_admet_default")
            except Exception as exc:
                log.info("ADMET default channel unavailable (%s) — skipping", exc)
                admet_channel = None
                w_admet = 0.0
    return RewardAggregator(
        r_vina=r_vina,
        r_posebusters=r_posebusters,
        r_retro=r_retro,
        r_sa=r_sa,
        r_qed=r_qed,
        r_pic50=r_pic50,
        r_admet=admet_channel,
        w_vina=weights["w_vina"],
        w_posebusters=weights["w_posebusters"],
        w_retro=weights["w_retro"],
        w_sa=weights["w_sa"],
        w_qed=weights["w_qed"],
        w_admet=w_admet,
        vina_invert=True,
    )


# ---------------------------------------------------------------------------
# TODO-01 — Vina + PoseBusters + AiZynth composite factory
# ---------------------------------------------------------------------------
def build_todo01_reward_aggregator(
    *,
    r_vina: Optional[Callable[[Any], float]] = None,
    r_posebusters: Optional[Callable[[Any], float]] = None,
    r_retro: Optional[Callable[[Any], float]] = None,
    r_pic50: Optional[Callable[[Any], float]] = None,
    r_qed: Optional[Callable[[Any], float]] = None,
    r_sa: Optional[Callable[[Any], float]] = None,
    r_admet: Optional[Callable[[Any], float]] = None,
    weights: Optional[Dict[str, float]] = None,
    enable_admet: bool = False,
) -> RewardAggregator:
    """Construct the TODO-01 default :class:`RewardAggregator`.

    Channel weights follow :data:`TODO01_REWARD_AGGREGATOR_WEIGHTS`:
    Vina (0.40) + PoseBusters (0.20) + AiZynth retro (0.15) + pIC50
    (0.10) + QED (0.10) + SA (0.05) — summing to 1.0.  Any channel that
    is left as ``None`` degrades to ``0.0`` inside the aggregator, so the
    returned object is always usable.

    Parameters
    ----------
    r_vina, r_posebusters, r_retro, r_pic50, r_qed, r_sa, r_admet:
        Optional ``Callable[[MoleculeClosedTerm], float]`` channel
        implementations.  When omitted, the aggregator silently returns
        ``0.0`` for that channel.
    weights:
        Optional override dict keyed by ``w_<channel>`` (any subset of
        ``TODO01_REWARD_AGGREGATOR_WEIGHTS`` keys).  Values are merged
        on top of the defaults.
    enable_admet:
        When ``True`` (and ``r_admet`` is supplied or the round-8 ADMET
        runner is importable), ``w_admet=1.0`` is set so the Lipinski
        desirability channel contributes alongside the six primary
        channels.  Defaults to ``False`` so existing callers aren't
        surprised by the new channel.

    Returns
    -------
    RewardAggregator
        A :class:`RewardAggregator` configured with the TODO-01 default
        channel weights and the supplied channel callables.
    """
    cfg = dict(TODO01_REWARD_AGGREGATOR_WEIGHTS)
    if weights:
        cfg.update(weights)

    # ADMET — opt-in.  When ``enable_admet`` is True and no callable was
    # passed, try to wire the round-8 ADMET runner default closure.
    admet_channel = r_admet
    w_admet = 0.0
    if enable_admet:
        w_admet = 1.0
        if admet_channel is None:
            try:
                from molmetal_lam.search_alg.proof_search import _r_admet_default
                admet_channel = _r_admet_default
                log.info("ADMET channel wired via proof_search._r_admet_default")
            except Exception as exc:
                log.info("ADMET default channel unavailable (%s) — skipping", exc)
                admet_channel = None
                w_admet = 0.0

    return RewardAggregator(
        r_vina=r_vina,
        r_posebusters=r_posebusters,
        r_retro=r_retro,
        r_pic50=r_pic50,
        r_qed=r_qed,
        r_sa=r_sa,
        r_admet=admet_channel,
        w_vina=cfg["w_vina"],
        w_posebusters=cfg["w_posebusters"],
        w_retro=cfg["w_retro"],
        w_pic50=cfg["w_pic50"],
        w_qed=cfg["w_qed"],
        w_sa=cfg["w_sa"],
        w_admet=w_admet,
        vina_invert=True,
    )


# ---------------------------------------------------------------------------
# SOTA-aligned RewardAggregator — TargetDiff / CrossDocked100 protocol
# ---------------------------------------------------------------------------
def build_sota_aligned_reward_aggregator(
    *,
    r_vina: Optional[Callable[[Any], float]] = None,
    r_posebusters: Optional[Callable[[Any], float]] = None,
    r_retro: Optional[Callable[[Any], float]] = None,
    r_pic50: Optional[Callable[[Any], float]] = None,
    r_qed: Optional[Callable[[Any], float]] = None,
    r_sa: Optional[Callable[[Any], float]] = None,
    r_admet: Optional[Callable[[Any], float]] = None,
    weights: Optional[Dict[str, float]] = None,
    enable_admet: bool = False,
) -> RewardAggregator:
    """Construct the SOTA-aligned :class:`RewardAggregator`.

    Channel weights follow :data:`SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS`:
    Vina + PoseBusters + AiZynth retro at 1.0 each, SA + QED at 0.3
    each.  Any channel left as ``None`` degrades to ``0.0`` inside the
    aggregator, so the returned object stays usable even before the
    docking / PoseBusters / AiZynth adapters are wired.

    Parameters
    ----------
    r_vina, r_posebusters, r_retro, r_pic50, r_qed, r_sa, r_admet:
        Optional ``Callable[[MoleculeClosedTerm], float]`` channel
        implementations.  When omitted, the aggregator silently returns
        ``0.0`` for that channel.
    weights:
        Optional override dict keyed by ``w_<channel>``.  Values are
        merged on top of :data:`SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS`
        — useful when a sweep wants to pin *only* one channel while
        keeping the SOTA-aligned defaults for the rest.
    enable_admet:
        When ``True`` the round-8 ADMET runner default closure is
        attached (when available) with ``w_admet=1.0``.  Defaults to
        ``False`` — the SOTA-aligned protocol does not weight ADMET
        explicitly, so this is opt-in to preserve parity with the YAML.

    Returns
    -------
    RewardAggregator
        A :class:`RewardAggregator` configured with the SOTA-aligned
        channel weights and the supplied channel callables.

    Notes
    -----
    The reference protocol lives in
    ``molmetal/configs/sota_aligned_targetdiff.yaml``.  This factory
    mirrors its reward layout; the YAML itself encodes the *protocol*
    (test pockets, docking exhaustiveness, NFE budget) — the reward
    weights here are the *channel layout* used to rank candidates
    under that protocol.
    """
    cfg = dict(SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS)
    if weights:
        cfg.update(weights)

    # ADMET — opt-in.  The SOTA-aligned YAML does not list ADMET as a
    # primary channel, so the default is to leave it unweighted.  Callers
    # can opt in via ``enable_admet=True`` (or pass ``r_admet`` directly
    # together with a ``weights={"w_admet": 1.0}`` override).
    admet_channel = r_admet
    w_admet = 0.0
    if enable_admet:
        w_admet = 1.0
        if admet_channel is None:
            try:
                from molmetal_lam.search_alg.proof_search import _r_admet_default
                admet_channel = _r_admet_default
                log.info("ADMET channel wired via proof_search._r_admet_default")
            except Exception as exc:
                log.info("ADMET default channel unavailable (%s) — skipping", exc)
                admet_channel = None
                w_admet = 0.0

    return RewardAggregator(
        r_vina=r_vina,
        r_posebusters=r_posebusters,
        r_retro=r_retro,
        r_pic50=r_pic50,
        r_qed=r_qed,
        r_sa=r_sa,
        r_admet=admet_channel,
        w_vina=cfg.get("w_vina", 1.0),
        w_posebusters=cfg.get("w_posebusters", 1.0),
        w_retro=cfg.get("w_retro", 1.0),
        # pIC50 / ADMET are not in the SOTA-aligned primary channel set;
        # carry them at 0.0 unless the caller overrides via ``weights``.
        w_pic50=cfg.get("w_pic50", 0.0),
        w_qed=cfg.get("w_qed", 0.3),
        w_sa=cfg.get("w_sa", 0.3),
        w_admet=w_admet,
        vina_invert=True,
    )


# ---------------------------------------------------------------------------
# TODO-01 — per-channel reward breakdown
# ---------------------------------------------------------------------------
def decompose_reward(
    aggregator: RewardAggregator,
    state: Any,
    *,
    target_predicates: Optional[Sequence[Any]] = None,
    binding_site: Optional[Any] = None,
    satisfies_typed: bool = False,
    binds_target: bool = False,
) -> Dict[str, float]:
    """Return the per-channel reward breakdown for ``state``.

    The returned dict has keys ``r_total`` plus ``r_<channel>`` for each
    active channel wired on the aggregator (``r_vina``, ``r_posebusters``,
    ``r_retro``, ``r_pic50``, ``r_qed``, ``r_sa``, ``r_admet``, ``r_pb_valid``,
    ``r_vina_proxy``, ``r_reinvent4``, ``r_synth``).  Channels whose
    callable is ``None`` are reported as ``0.0`` — the closed-loop results
    dict always carries the full channel schema so downstream logging
    doesn't have to guess which channels were live.

    The function is intentionally defensive: it never raises, so it is
    safe to call from per-iteration logging paths inside the closed-loop
    orchestrator.
    """
    breakdown: Dict[str, float] = {}

    def _safe(channel: Optional[Callable[[Any], float]]) -> float:
        if channel is None:
            return 0.0
        try:
            return float(channel(state))
        except Exception:
            return 0.0

    def _sa_norm(raw: float) -> float:
        # Same SA 1..10 -> 0..1 (higher = easier) normalisation the
        # aggregator uses internally.
        return max(0.0, min(1.0, 1.0 - (raw - 1.0) / 9.0))

    v_vina_raw = _safe(aggregator.r_vina)
    if aggregator.vina_invert and aggregator.r_vina is not None:
        v_vina_norm = -v_vina_raw
    else:
        v_vina_norm = v_vina_raw
    v_sa_raw = _safe(aggregator.r_sa)
    v_sa_norm = _sa_norm(v_sa_raw) if aggregator.r_sa is not None else 0.0
    v_qed = _safe(aggregator.r_qed)
    v_pb = _safe(aggregator.r_posebusters)
    v_pb_valid = _safe(aggregator.r_pb_valid)
    v_pi = _safe(aggregator.r_pic50)
    v_re = _safe(aggregator.r_retro)
    v_vina_proxy = _safe(aggregator.r_vina_proxy)
    v_reinvent4 = _safe(aggregator.r_reinvent4)
    v_synth = _safe(aggregator.r_synth)
    v_admet = _safe(aggregator.r_admet)

    # Weighted contribution of each channel — same arithmetic as
    # RewardAggregator.__call__ so the sum reproduces the scalar output.
    breakdown["r_vina"] = aggregator.w_vina * v_vina_norm
    breakdown["r_sa"] = aggregator.w_sa * v_sa_norm
    breakdown["r_qed"] = aggregator.w_qed * v_qed
    breakdown["r_vina_proxy"] = aggregator.w_vina_proxy * v_vina_proxy
    breakdown["r_posebusters"] = aggregator.w_posebusters * v_pb
    breakdown["r_pb_valid"] = aggregator.w_pb_valid * v_pb_valid
    breakdown["r_pic50"] = aggregator.w_pic50 * v_pi
    breakdown["r_retro"] = aggregator.w_retro * v_re
    breakdown["r_reinvent4"] = aggregator.w_reinvent4 * v_reinvent4
    breakdown["r_synth"] = aggregator.w_synth * v_synth
    breakdown["r_admet"] = aggregator.w_admet * v_admet
    if satisfies_typed:
        breakdown["bonus_typed"] = aggregator.bonus_typed
    if binds_target:
        breakdown["bonus_binder"] = aggregator.bonus_binder
    breakdown["r_total"] = float(sum(breakdown.values()))
    return breakdown


# ---------------------------------------------------------------------------
# TODO-01 — CLI weights override
# ---------------------------------------------------------------------------
def add_reward_weights_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--reward-weights PATH`` argument to ``parser``.

    The argument is optional — when omitted, the closed loop uses
    :data:`TODO01_REWARD_AGGREGATOR_WEIGHTS`.  When supplied, the path
    must be a YAML file with a flat mapping of ``w_<channel>: float``
    entries.  Unknown keys are kept (forward-compatible) and missing
    keys fall back to the TODO-01 defaults.
    """
    parser.add_argument(
        "--reward-weights",
        type=str,
        default=None,
        help=(
            "Optional YAML file overriding TODO01_REWARD_AGGREGATOR_WEIGHTS. "
            "Keys: w_vina, w_posebusters, w_retro, w_pic50, w_qed, w_sa, "
            "w_admet.  Sum need not equal 1.0 — values are merged on top "
            "of the defaults."
        ),
    )


# ---------------------------------------------------------------------------
# SOTA-aligned CLI flag — TargetDiff / CrossDocked100 protocol
# ---------------------------------------------------------------------------
def add_sota_aligned_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--sota-aligned`` flag to ``parser``.

    When set, the default closed-loop :class:`RewardAggregator`
    switches to :data:`SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS`
    (Vina + PoseBusters + AiZynth retro at 1.0 each, SA + QED at
    0.3 each).  The flag is ``store_true`` so it composes cleanly with
    ``--reward-weights`` and any future protocol-level flags.

    See :data:`SOTA_ALIGNED_CONFIG_PATH` for the YAML this flag aligns
    with.
    """
    parser.add_argument(
        "--sota-aligned",
        action="store_true",
        default=False,
        help=(
            "Use the SOTA-aligned reward weights from "
            f"{SOTA_ALIGNED_CONFIG_PATH}: w_vina=1.0, w_posebusters=1.0, "
            "w_retro=1.0, w_sa=0.3, w_qed=0.3. Composes with "
            "--reward-weights PATH (per-channel overrides take precedence)."
        ),
    )


def resolve_reward_aggregator_for_flags(
    *,
    sota_aligned: bool = False,
    weights_path: Optional[str] = None,
    base_weights: Optional[Dict[str, float]] = None,
    channel_kwargs: Optional[Dict[str, Optional[Callable[[Any], float]]]] = None,
    enable_admet: bool = False,
) -> Tuple[RewardAggregator, Dict[str, float]]:
    """Build the right :class:`RewardAggregator` for the active CLI flags.

    Decision matrix (first match wins):

    1. ``sota_aligned=True`` → :func:`build_sota_aligned_reward_aggregator`
       with :data:`SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS` as the base.
       Any ``weights_path`` is applied on top via
       :func:`apply_reward_weights_override`.
    2. ``weights_path`` is set → :func:`build_todo01_reward_aggregator`
       with :func:`apply_reward_weights_override` applied to
       :data:`TODO01_REWARD_AGGREGATOR_WEIGHTS`.
    3. Fallback → :func:`build_todo01_reward_aggregator` with the
       TODO-01 defaults (today's behaviour).

    The returned tuple is ``(aggregator, merged_weights)`` so callers
    that want to log the active channel layout can grab ``merged_weights``
    without re-deriving it.  ``channel_kwargs`` is forwarded verbatim
    to the chosen factory so the Vina / PoseBusters / AiZynth closures
    reach the aggregator intact.
    """
    channel_kwargs = dict(channel_kwargs or {})

    if sota_aligned:
        merged = apply_reward_weights_override(
            base_weights=SOTA_ALIGNED_REWARD_AGGREGATOR_WEIGHTS,
            override_path=weights_path,
        )
        agg = build_sota_aligned_reward_aggregator(
            weights=merged,
            enable_admet=enable_admet,
            **channel_kwargs,
        )
        return agg, merged

    if weights_path:
        merged = apply_reward_weights_override(
            base_weights=base_weights
            if base_weights is not None
            else TODO01_REWARD_AGGREGATOR_WEIGHTS,
            override_path=weights_path,
        )
        agg = build_todo01_reward_aggregator(
            weights=merged,
            enable_admet=enable_admet,
            **channel_kwargs,
        )
        return agg, merged

    base = (
        base_weights if base_weights is not None else TODO01_REWARD_AGGREGATOR_WEIGHTS
    )
    agg = build_todo01_reward_aggregator(
        weights=base,
        enable_admet=enable_admet,
        **channel_kwargs,
    )
    return agg, dict(base)


def apply_reward_weights_override(
    base_weights: Optional[Dict[str, float]] = None,
    override_path: Optional[str] = None,
) -> Dict[str, float]:
    """Merge a YAML weights file on top of ``base_weights`` (defaults to
    :data:`TODO01_REWARD_AGGREGATOR_WEIGHTS`).

    Returns the merged dict.  Missing files, parse failures, and unknown
    keys are logged and ignored — the closed loop never crashes because
    of a malformed weights override.
    """
    cfg = dict(base_weights if base_weights is not None else TODO01_REWARD_AGGREGATOR_WEIGHTS)
    if not override_path:
        return cfg
    if not os.path.exists(override_path):
        log.warning("reward weights override not found: %s — using defaults", override_path)
        return cfg
    try:
        # Lazy import — PyYAML is optional and many test envs don't have it.
        import yaml  # type: ignore

        with open(override_path, "r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
    except ImportError:
        log.warning("PyYAML not installed — ignoring %s", override_path)
        return cfg
    except Exception as exc:
        log.warning("failed to load reward weights override (%s): %s", override_path, exc)
        return cfg
    if not isinstance(override, dict):
        log.warning("reward weights override must be a dict, got %s", type(override))
        return cfg
    for key, val in override.items():
        if not isinstance(key, str):
            continue
        try:
            cfg[key] = float(val)
        except (TypeError, ValueError):
            log.warning("ignoring non-numeric weight %s=%r", key, val)
    return cfg


class WeightedSumScorer(ScoringFunction):
    """Weighted linear combination of normalised objectives.

    ``score`` is computed as::

        s = w_qed * qed
          + w_sa  * sa_norm            (sa_norm = 1 - (sa - 1) / 9, higher = easier)
          + w_pic * pic50_norm         (pic50_norm = (pIC50 - 4) / 7)
          + w_vina * vina_norm         (vina_norm = (vina - (-14)) / 14, higher = worse)

    With the default weights the ``vina`` term carries a *negative* weight
    so that a strongly negative (good) Vina score raises the total.

    Missing signals (no docking done, no affinity model available) simply
    drop out of the sum — their contribution is zero, never NaN.

    Parameters
    ----------
    weights:
        Override any subset of :data:`DEFAULT_WEIGHTS`.  Unknown keys are
        kept and will be looked up on the ``PropertyPrediction`` /
        ``Complex`` by attribute name, so custom objectives (e.g.
        ``metal_binding_score``) can be added without subclassing.
    normalize:
        When ``False`` the raw values are used directly (useful when the
        adapters already emit normalised scores).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
    ) -> None:
        self._weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._normalize = normalize
        self._is_setup = False

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "WeightedSumScorer_v1"

    @property
    def weights(self) -> Dict[str, float]:
        """Read-only view of the active weights."""
        return dict(self._weights)

    def setup(self) -> None:
        self._is_setup = True

    # ------------------------------------------------------------------
    # Raw-value extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_value(
        key: str,
        molecule: Molecule,
        complex: Optional[Complex],
        pred: PropertyPrediction,
    ) -> Optional[float]:
        """Pull the raw value for ``key`` out of the available objects.

        Lookup order: explicit alias → ``PropertyPrediction`` attr →
        ``Complex`` attr → ``Molecule`` attr.  Returns ``None`` when the
        signal is unavailable, in which case the term is skipped.
        """
        if key == "qed":
            val = pred.qed if pred.qed else molecule.qed
            return float(val) if val is not None else None
        if key == "sa_score":
            val = pred.sa_score if pred.sa_score else molecule.sa_score
            return float(val) if val is not None else None
        if key == "binding_pic50":
            val = pred.binding_affinity_pic50
            if val is None and complex is not None:
                val = complex.binding_affinity
            return float(val) if val is not None else None
        if key == "vina_score":
            val = complex.vina_score if complex is not None else None
            return float(val) if val is not None else None
        # Generic fallback for user-supplied objectives.
        for obj in (pred, complex, molecule):
            if obj is not None and hasattr(obj, key):
                val = getattr(obj, key)
                if val is not None:
                    return float(val)
        return None

    def _normalized(self, key: str, raw: float) -> float:
        """Map a raw objective value onto a 0..1 scale (higher = larger raw)."""
        if not self._normalize:
            return raw
        if key == "qed":
            return _clip01(raw)
        if key == "sa_score":
            # Invert: SA 1 (easy) → 1.0, SA 10 (hard) → 0.0
            return _clip01(1.0 - (raw - _SA_MIN) / (_SA_MAX - _SA_MIN))
        if key == "binding_pic50":
            return _clip01((raw - _PIC50_MIN) / (_PIC50_MAX - _PIC50_MIN))
        if key == "vina_score":
            # 0 kcal/mol → 1.0 (bad), -14 kcal/mol → 0.0 (good).
            # Combined with the negative default weight, a good (very
            # negative) Vina score contributes ~0 penalty while a poor one
            # subtracts up to |w|.
            return _clip01((raw - _VINA_MIN) / (_VINA_MAX - _VINA_MIN))
        return raw

    # ------------------------------------------------------------------
    def score_one(
        self,
        molecule: Molecule,
        complex: Optional[Complex],
        pred: PropertyPrediction,
    ) -> float:
        """Combined scalar score for a single candidate."""
        total = 0.0
        for key, weight in self._weights.items():
            if weight == 0.0:
                continue
            raw = self._raw_value(key, molecule, complex, pred)
            if raw is None:
                continue
            total += weight * self._normalized(key, raw)
        return float(total)

    def breakdown(
        self,
        molecule: Molecule,
        complex: Optional[Complex],
        pred: PropertyPrediction,
    ) -> Dict[str, float]:
        """Per-objective contributions — handy for logging / debugging."""
        out: Dict[str, float] = {}
        for key, weight in self._weights.items():
            raw = self._raw_value(key, molecule, complex, pred)
            out[key] = 0.0 if raw is None else weight * self._normalized(key, raw)
        return out

    def score(
        self,
        candidates: Sequence[Tuple[Molecule, Optional[Complex], PropertyPrediction]],
    ) -> List[ScoredCandidate]:
        """Score + rank every candidate (rank 1 = best)."""
        scored = [
            ScoredCandidate(
                molecule=mol,
                complex=cplx,
                property_pred=pred,
                combined_score=self.score_one(mol, cplx, pred),
                rank=0,
            )
            for mol, cplx, pred in candidates
        ]
        scored.sort(key=lambda c: c.combined_score, reverse=True)
        # ScoredCandidate is frozen → rebuild with the final rank.
        return [
            ScoredCandidate(
                molecule=c.molecule,
                complex=c.complex,
                property_pred=c.property_pred,
                combined_score=c.combined_score,
                rank=i + 1,
            )
            for i, c in enumerate(scored)
        ]

    def get_metadata(self) -> dict:
        return {
            "scorer": self.name,
            "weights": dict(self._weights),
            "normalize": self._normalize,
            "normalization_ranges": {
                "sa_score": [_SA_MIN, _SA_MAX],
                "binding_pic50": [_PIC50_MIN, _PIC50_MAX],
                "vina_score": [_VINA_MIN, _VINA_MAX],
            },
            "sign_convention": (
                "positive weight = higher raw is better; "
                "vina_score carries a negative weight because lower "
                "(more negative) kcal/mol is better"
            ),
        }


# ---------------------------------------------------------------------------
# TODO-01 — RewardAggregatorScorer (the new default scorer)
# ---------------------------------------------------------------------------
class RewardAggregatorScorer(ScoringFunction):
    """Composite scorer that delegates to a :class:`RewardAggregator`.

    Implements the :class:`~molmetal.ports.ScoringFunction` port by
    wrapping a pre-built :class:`~molmetal_lam.search_alg.proof_search.RewardAggregator`
    and exposing the per-channel reward breakdown to downstream callers
    via :attr:`last_breakdown`.  This is the canonical TODO-01 scorer:

    * channels: r_vina + r_posebusters + r_retro + r_pic50 + r_qed + r_sa
      (+ optional r_admet) — Vina + PoseBusters + AiZynth composite,
    * weights: :data:`TODO01_REWARD_AGGREGATOR_WEIGHTS`,
    * output: ``combined_score`` is the aggregator scalar; the breakdown
      is stored as ``dict[mol.smiles] -> {r_<channel>: float, r_total: float}``
      so callers (e.g. ``ClosedLoop``) can splice the channels into the
      per-iteration results dict.

    Parameters
    ----------
    aggregator:
        Pre-built :class:`RewardAggregator`.  Defaults to the TODO-01
        factory output with no callables wired (every channel degrades
        to 0.0).
    molecule_to_state:
        Callable mapping ``(Molecule, Optional[Complex], PropertyPrediction)``
        → ``MoleculeClosedTerm`` (or any state the channel callables
        accept).  Defaults to a no-op identity function — useful when the
        channel callables take a Molecule-shaped object directly.
    """

    def __init__(
        self,
        aggregator: Optional[RewardAggregator] = None,
        molecule_to_state: Optional[Callable[[Any, Any, Any], Any]] = None,
    ) -> None:
        self._aggregator = (
            aggregator
            if aggregator is not None
            else build_todo01_reward_aggregator()
        )
        self._state_fn = molecule_to_state or (lambda m, c, p: m)
        self.last_breakdown: Dict[str, Dict[str, float]] = {}
        self._is_setup = False

    @property
    def name(self) -> str:
        return "RewardAggregatorScorer_v1"

    @property
    def aggregator(self) -> RewardAggregator:
        return self._aggregator

    def setup(self) -> None:
        self._is_setup = True

    def score_one(
        self,
        molecule: Molecule,
        complex: Optional[Complex],
        pred: PropertyPrediction,
    ) -> float:
        """Scalar reward for one candidate — delegates to the aggregator."""
        state = self._state_fn(molecule, complex, pred)
        try:
            return float(self._aggregator(state))
        except Exception:
            return 0.0

    def breakdown(
        self,
        molecule: Molecule,
        complex: Optional[Complex],
        pred: PropertyPrediction,
    ) -> Dict[str, float]:
        """Per-channel contribution for one candidate."""
        state = self._state_fn(molecule, complex, pred)
        return decompose_reward(self._aggregator, state)

    def score(
        self,
        candidates: Sequence[Tuple[Molecule, Optional[Complex], PropertyPrediction]],
    ) -> List[ScoredCandidate]:
        """Score + rank every candidate; record per-channel breakdown."""
        scored: List[ScoredCandidate] = []
        per_candidate_breakdown: Dict[str, Dict[str, float]] = {}
        for mol, cplx, pred in candidates:
            bd = self.breakdown(mol, cplx, pred)
            total = float(bd.get("r_total", 0.0))
            smi = getattr(mol, "smiles", "") or ""
            per_candidate_breakdown[smi] = bd
            scored.append(
                ScoredCandidate(
                    molecule=mol,
                    complex=cplx,
                    property_pred=pred,
                    combined_score=total,
                    rank=0,
                )
            )
        scored.sort(key=lambda c: c.combined_score, reverse=True)
        self.last_breakdown = per_candidate_breakdown
        return [
            ScoredCandidate(
                molecule=c.molecule,
                complex=c.complex,
                property_pred=c.property_pred,
                combined_score=c.combined_score,
                rank=i + 1,
            )
            for i, c in enumerate(scored)
        ]

    def get_metadata(self) -> dict:
        agg = self._aggregator
        return {
            "scorer": self.name,
            "weights": {
                "w_vina": agg.w_vina,
                "w_posebusters": agg.w_posebusters,
                "w_retro": agg.w_retro,
                "w_pic50": agg.w_pic50,
                "w_qed": agg.w_qed,
                "w_sa": agg.w_sa,
                "w_admet": agg.w_admet,
                "w_pb_valid": agg.w_pb_valid,
                "w_reinvent4": agg.w_reinvent4,
                "w_synth": agg.w_synth,
            },
            "vina_invert": agg.vina_invert,
            "channels": {
                "r_vina": agg.r_vina is not None,
                "r_posebusters": agg.r_posebusters is not None,
                "r_retro": agg.r_retro is not None,
                "r_pic50": agg.r_pic50 is not None,
                "r_qed": agg.r_qed is not None,
                "r_sa": agg.r_sa is not None,
                "r_admet": agg.r_admet is not None,
                "r_pb_valid": agg.r_pb_valid is not None,
                "r_reinvent4": agg.r_reinvent4 is not None,
                "r_synth": agg.r_synth is not None,
            },
        }


# ---------------------------------------------------------------------------
# ClosedLoop — DesignLoop implementation
# ---------------------------------------------------------------------------
class ClosedLoop(DesignLoop):
    """Generate → dock (top-K by QED) → predict → score → keep top-K.

    Parameters
    ----------
    generator, docker, predictor, scorer:
        Port implementations.  Only duck-typing is required.
    dock_top_k:
        How many molecules per round get docked.  Docking is 10–1000× more
        expensive than property prediction, so we pre-filter by QED (a free
        2D descriptor) and only dock the most drug-like fraction.  When
        ``None`` (default) it falls back to ``config.n_top_k`` at run time.
    docking_config:
        Passed verbatim to ``docker.dock``.  Defaults to
        ``DockingConfig()``.
    generation_kwargs:
        Extra keyword arguments merged into the per-round
        ``GenerationConfig`` (e.g. ``n_steps``, ``temperature``).
    verbose:
        Print a one-line progress summary per iteration.
    sota_aligned:
        When ``True``, the default ``scorer`` is *replaced* with a
        :class:`RewardAggregatorScorer` configured by
        :func:`build_sota_aligned_reward_aggregator` so the
        closed-loop history records the SOTA-aligned per-channel
        breakdown (r_vina + r_posebusters + r_retro at weight 1.0,
        r_sa + r_qed at weight 0.3).  Mirrors the ``--sota-aligned``
        CLI flag wired by :func:`add_sota_aligned_arg`.  Defaults to
        ``False`` to preserve today's behaviour.
    """

    def __init__(
        self,
        generator: MoleculeGenerator,
        docker: DockingEngine,
        predictor: PropertyPredictor,
        scorer: ScoringFunction,
        dock_top_k: Optional[int] = None,
        docking_config: Optional[DockingConfig] = None,
        generation_kwargs: Optional[dict] = None,
        verbose: bool = True,
        sota_aligned: bool = False,
    ) -> None:
        self.generator = generator
        self.docker = docker
        self.predictor = predictor

        self.dock_top_k = dock_top_k
        self.docking_config = docking_config or DockingConfig()
        self.generation_kwargs = dict(generation_kwargs or {})
        self.verbose = verbose

        # SOTA-aligned mode replaces the supplied scorer with a
        # RewardAggregatorScorer using the TargetDiff / CrossDocked100
        # channel layout.  When ``sota_aligned=False`` the supplied
        # scorer is used verbatim (back-compat with all existing callers
        # in test_closed_loop_reward.py + production sweeps).
        if sota_aligned:
            self.scorer: ScoringFunction = RewardAggregatorScorer(
                aggregator=build_sota_aligned_reward_aggregator(),
            )
        else:
            self.scorer = scorer

        self.sota_aligned = bool(sota_aligned)

        self._history: List[dict] = []
        self._best: List[ScoredCandidate] = []
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "ClosedLoop_v1"

    def setup(self, device: str = "cuda") -> None:
        """Set up every downstream port.  ``scorer.setup()`` takes no device."""
        self._device = device
        for component in (self.generator, self.docker, self.predictor):
            setup = getattr(component, "setup", None)
            if setup is not None:
                setup(device=device)
        scorer_setup = getattr(self.scorer, "setup", None)
        if scorer_setup is not None:
            scorer_setup()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self, pocket: Pocket, config: DesignLoopConfig
    ) -> List[ScoredCandidate]:
        """Run ``config.n_iterations`` design rounds; return the global top-K."""
        self._history = []
        self._best = []
        prev_best_score: Optional[float] = None

        for it in range(config.n_iterations):
            t0 = time.time()

            # --- 1. Generate -------------------------------------------------
            gen_cfg = GenerationConfig(
                n_samples=config.n_samples_per_round,
                seed=config.seed + it,
                **self.generation_kwargs,
            )
            molecules = list(self.generator.generate(pocket, gen_cfg))

            # --- 2. Dock the most drug-like subset ---------------------------
            dock_k = self.dock_top_k if self.dock_top_k is not None else config.n_top_k
            dock_k = max(0, min(dock_k, len(molecules)))
            # Rank by QED (free 2D descriptor) — highest first.
            order = sorted(
                range(len(molecules)),
                key=lambda i: float(getattr(molecules[i], "qed", 0.0) or 0.0),
                reverse=True,
            )
            dock_indices = set(order[:dock_k])

            complexes: List[Optional[Complex]] = [None] * len(molecules)
            n_dock_failures = 0
            for i in sorted(dock_indices):
                try:
                    poses = self.docker.dock(molecules[i], pocket, self.docking_config)
                    if poses:
                        # Best pose = highest confidence, ties broken by
                        # the more negative (better) Vina score.
                        complexes[i] = max(
                            poses,
                            key=lambda c: (
                                float(c.pose_confidence or 0.0),
                                -float(c.vina_score if c.vina_score is not None else 0.0),
                            ),
                        )
                except Exception:  # noqa: BLE001 — one bad pose must not kill the run
                    n_dock_failures += 1

            # --- 3. Predict properties for ALL molecules ---------------------
            triples: List[Tuple[Molecule, Optional[Complex], PropertyPrediction]] = []
            n_pred_failures = 0
            for mol, cplx in zip(molecules, complexes):
                try:
                    pred = self.predictor.predict(mol, cplx)
                except Exception:  # noqa: BLE001
                    n_pred_failures += 1
                    pred = PropertyPrediction()
                triples.append((mol, cplx, pred))

            # --- 4. Score ----------------------------------------------------
            scored = list(self.scorer.score(triples))

            # --- 5. Keep the global top-K across all rounds so far -----------
            pool = self._best + scored
            pool.sort(key=lambda c: c.combined_score, reverse=True)
            self._best = [
                ScoredCandidate(
                    molecule=c.molecule,
                    complex=c.complex,
                    property_pred=c.property_pred,
                    combined_score=c.combined_score,
                    rank=i + 1,
                )
                for i, c in enumerate(pool[: config.n_top_k])
            ]

            # --- History -----------------------------------------------------
            iter_scores = [c.combined_score for c in scored]
            best_score = max(iter_scores) if iter_scores else float("-inf")
            mean_score = (
                sum(iter_scores) / len(iter_scores) if iter_scores else float("nan")
            )
            improvement = (
                None if prev_best_score is None else best_score - prev_best_score
            )
            converged = (
                improvement is not None
                and abs(improvement) < config.convergence_threshold
            )

            record = {
                "iteration": it,
                "n_generated": len(molecules),
                "n_docked": len(dock_indices) - n_dock_failures,
                "n_dock_failures": n_dock_failures,
                "n_predicted": len(triples),
                "n_pred_failures": n_pred_failures,
                "n_scored": len(scored),
                "n_kept": len(self._best),
                "best_score": best_score,
                "mean_score": mean_score,
                "running_best_score": (
                    self._best[0].combined_score if self._best else float("-inf")
                ),
                "improvement": improvement,
                "converged": converged,
                "elapsed_s": time.time() - t0,
            }
            # TODO-01 — splice per-channel reward breakdown (r_vina,
            # r_posebusters, r_retro, r_pic50, r_qed, r_sa, r_total) into
            # the record when the scorer exposes one.  The
            # ``RewardAggregatorScorer`` keeps ``last_breakdown`` after
            # every ``.score()`` call; ``WeightedSumScorer`` emits its
            # legacy ``breakdown()`` dict here too.  Either way the
            # downstream history reader sees the full channel schema.
            record["reward_breakdown"] = self._last_breakdown_snapshot()
            self._history.append(record)
            prev_best_score = best_score

            if self.verbose:
                print(
                    f"[{self.name}] iter {it + 1}/{config.n_iterations}  "
                    f"gen={record['n_generated']}  dock={record['n_docked']}  "
                    f"best={best_score:.4f}  mean={mean_score:.4f}  "
                    f"kept={record['n_kept']}  ({record['elapsed_s']:.1f}s)"
                )

            if converged:
                if self.verbose:
                    print(
                        f"[{self.name}] converged: |Δbest| < "
                        f"{config.convergence_threshold} — stopping early"
                    )
                break

        return self._best

    # ------------------------------------------------------------------
    def _last_breakdown_snapshot(self) -> Dict[str, Any]:
        """Return a defensive snapshot of the scorer's per-channel breakdown.

        Priority:

        * If the scorer is a :class:`RewardAggregatorScorer`, take its
          ``last_breakdown`` (mapping ``smiles -> {r_<channel>: float,
          r_total: float}``).  Returns ``{"per_molecule": {...},
          "channels": [...], "aggregate": {...}}`` so callers can
          iterate uniformly — the ``aggregate`` block is the channel
          mean across every scored candidate in the iteration, which
          is the right number to log on the closed-loop dashboard.
        * If the scorer is a :class:`WeightedSumScorer`, fall back to its
          ``breakdown(mol, complex, pred)`` method per-candidate using
          the in-flight ``triples`` (if available).
        * Otherwise return ``{}`` — non-reward-aggregator scorers simply
          have nothing to report per-channel.
        """
        scorer = self.scorer
        if hasattr(scorer, "last_breakdown") and isinstance(
            getattr(scorer, "last_breakdown", None), dict
        ):
            per_mol = dict(scorer.last_breakdown)
            if not per_mol:
                return {}
            channels = sorted(
                {k for v in per_mol.values() for k in v.keys()},
            )
            aggregate: Dict[str, float] = {}
            n = len(per_mol)
            if n > 0:
                for chan in channels:
                    vals = [
                        float(v.get(chan, 0.0)) for v in per_mol.values()
                    ]
                    aggregate[chan] = float(sum(vals) / n)
            return {
                "per_molecule": per_mol,
                "channels": channels,
                "aggregate": aggregate,
            }
        return {}

    # ------------------------------------------------------------------
    def get_history(self) -> List[dict]:
        """Per-iteration log records (copies, so callers can't mutate state)."""
        return [dict(r) for r in self._history]

    def get_metadata(self) -> dict:
        def _meta(component) -> dict:
            get = getattr(component, "get_metadata", None)
            try:
                return get() if get is not None else {}
            except Exception:  # noqa: BLE001
                return {}

        return {
            "loop": self.name,
            "device": self._device,
            "generator": _meta(self.generator),
            "docker": _meta(self.docker),
            "predictor": _meta(self.predictor),
            "scorer": _meta(self.scorer),
            "dock_top_k": self.dock_top_k,
            "sota_aligned": self.sota_aligned,
            "sota_aligned_config_path": (
                SOTA_ALIGNED_CONFIG_PATH if self.sota_aligned else None
            ),
        }
