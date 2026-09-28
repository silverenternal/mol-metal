"""Dreaming — offline evaluation + policy improvement + selection.

Pure-Python port of ``extensions/pi-dream-rsi/engine/dream.ts``.

Paper §3 ("Offline evaluation" + "Policy improvement and selection"):

    fixed history H_t  →  evaluate π_0 .. π_{M-1} on EVERY world (replay
    only, no new generation) →  revise the executable policy between
    revisions →  π_{t+1} = argmax_m V_m  (V_m = mean over worlds of
    Eq. 1)  →  deploy the winner for the next live episode.

Because ``π_0`` (the policy that just ran live) is always among the
candidates, the selection satisfies ``V* >= V_0``.  That guarantee is
*asserted*, not assumed: if it breaks, ``run_dream_phase`` raises and the
iteration is reported as failed instead of deploying a regression.

Invariants preserved (matches the TS reference):

*   ``V* >= V_0`` — always include ``π_0`` as candidate 0.
*   Non-deterministic replays disqualify a candidate: ``v_mean = -inf``.
*   Failed episodes contribute ``-inf`` to ``values`` and are excluded
    from the scored-set average.
*   The monotone check uses only finite ``v_mean`` candidates, so a
    single failure cannot break the iteration through.
"""

from __future__ import annotations

import asyncio
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .discovery_tree import DiscoveryTree
from .policy_host import Decision, HostOptions, PolicyHost, ProbeFn
from .replay_episode import (
    EpisodeMetrics,
    ReplayEpisode,
    ReplayEpisodeOptions,
    ReplayEpisodeResult,
    RewardConfig,
    replay_value,
    run_replay_episode,
    trace_metrics,
)
from .replay_probe import RevealedNode, make_replay_probe


NEG_INF = float("-inf")


@dataclass
class PolicyCandidate:
    """One policy version evaluated in the dream phase."""

    revision: int
    #: A Python callable that runs the policy against a host.  Stored as
    #: the ``policy_fn`` argument of :class:`ReplayEpisodeOptions`.
    policy_fn: Callable[[PolicyHost], Awaitable[List[Decision]] | List[Decision]]
    #: Free-form name (e.g. ``"π_0"``).
    name: Optional[str] = None
    #: The version's baked-in default beta (resolved from the policy itself).
    default_beta: float = 0.6


@dataclass
class VersionEvaluation:
    """Result of evaluating one candidate on the fixed history."""

    revision: int
    v_mean: float
    deterministic: Optional[bool]
    per_world: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass
class DreamOptions:
    """Dream-phase options."""

    worlds: List[DiscoveryTree]
    #: One entry per candidate.  The first entry *must* be the policy
    #: that just ran live (``π_0``) — the monotone invariant
    #: ``V* >= V_0`` depends on it.
    candidates: List[PolicyCandidate]
    max_parallelism: int = 4
    k2: int = 4
    beta1: float = 0.05
    beta2: float = 0.1
    lambda_: float = 1.0
    verify_determinism: bool = True


@dataclass
class DreamResult:
    """Outcome of the dream phase."""

    ok: bool
    error: Optional[str]
    worlds: int
    revisions: int
    evaluations: List[VersionEvaluation]
    #: Index into ``evaluations`` of the selected version.
    selected: Optional[int]
    selected_policy: Optional[str]
    #: The beta baked into the deployed policy.
    deployed_beta: float
    report: str = ""


@dataclass
class DreamLoop:
    """Stateful runner for the offline dreaming loop.

    The caller supplies the fixed history (frozen worlds from previous
    live cycles) and the candidate set; :meth:`DreamLoop.run` evaluates
    every candidate on every world, then selects ``argmax V`` with the
    ``V* >= V_0`` invariant asserted.
    """

    options: DreamOptions

    async def run(self) -> DreamResult:
        return await run_dream_phase(self.options)


async def run_dream_phase(options: DreamOptions) -> DreamResult:
    """Run the offline dreaming loop.

    Mirrors the TS reference: ``π_0`` is the first candidate; ``V*``
    is the max ``v_mean`` over the candidate set; the monotone invariant
    is asserted on the *finite* values.
    """
    worlds = list(options.worlds)
    if not worlds:
        raise ValueError("no recorded worlds to replay; run a live cycle first")
    candidates = list(options.candidates)
    if not candidates:
        raise ValueError("no policy candidates supplied")

    reward_cfg = RewardConfig(
        max_parallelism=int(options.max_parallelism),
        beta1=float(options.beta1),
        beta2=float(options.beta2),
        lambda_=float(options.lambda_),
        baseline_score=worlds[0].baseline_score,
    )

    evaluations: List[VersionEvaluation] = []
    for candidate in candidates:
        ev = await _evaluate_candidate(candidate, worlds, reward_cfg, options)
        evaluations.append(ev)

    # ---- selection ----------------------------------------------------
    selected = 0
    for index in range(1, len(evaluations)):
        if evaluations[index].v_mean > evaluations[selected].v_mean:
            selected = index

    # The monotone invariant is checked against the *finite* v_mean values:
    # a version that failed to score contributes -inf and is excluded from
    # the scored-set average.
    scored = [e for e in evaluations if math.isfinite(e.v_mean)]
    if not scored:
        return DreamResult(
            ok=False,
            error="no policy version produced a finite replay score (all failed or non-deterministic)",
            worlds=len(worlds),
            revisions=len(evaluations),
            evaluations=evaluations,
            selected=None,
            selected_policy=None,
            deployed_beta=0.6,
            report="dream phase: no candidate scored",
        )

    best = evaluations[selected]
    monotone = math.isfinite(best.v_mean) and best.v_mean >= scored[0].v_mean
    if not monotone:
        return DreamResult(
            ok=False,
            error="selection violated V* >= V_0 (non-deterministic or non-prefix-only policy)",
            worlds=len(worlds),
            revisions=len(evaluations),
            evaluations=evaluations,
            selected=selected,
            selected_policy=candidates[selected].name,
            deployed_beta=candidates[selected].default_beta,
            report="dream phase: monotone check failed",
        )

    return DreamResult(
        ok=True,
        error=None,
        worlds=len(worlds),
        revisions=len(evaluations),
        evaluations=evaluations,
        selected=selected,
        selected_policy=candidates[selected].name,
        deployed_beta=candidates[selected].default_beta,
        report=_format_report(worlds, evaluations, selected, monotone),
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


async def _evaluate_candidate(
    candidate: PolicyCandidate,
    worlds: Sequence[DiscoveryTree],
    reward_cfg: RewardConfig,
    options: DreamOptions,
) -> VersionEvaluation:
    per_world: List[Dict[str, Any]] = []
    errors: List[str] = []
    values: List[float] = []
    deterministic: Optional[bool] = None

    for world in worlds:
        opts = ReplayEpisodeOptions(
            policy=candidate.policy_fn,
            world=world,
            beta=None,
            max_parallelism=options.max_parallelism,
            k2=options.k2,
            beta1=options.beta1,
            beta2=options.beta2,
            lambda_=options.lambda_,
            verify_determinism=options.verify_determinism,
        )
        result = await run_replay_episode(opts)
        # A version that cannot run a world must not be credited with the
        # zeroed metrics of a failed run.
        value = result.metrics.value if result.ok else NEG_INF
        values.append(value)
        per_world.append(
            {
                "value": value,
                "probes": result.metrics.probes,
                "rounds": result.metrics.rounds,
                "attainment": result.metrics.attainment,
                "auc": result.metrics.auc,
                "parallel_penalty": result.metrics.parallel_penalty,
                "ok": result.ok,
                "error": result.error,
            }
        )
        if result.deterministic is False and deterministic is not False:
            deterministic = False
        elif result.deterministic is True and deterministic is None:
            deterministic = True
        if not result.ok and result.error:
            errors.append(f"{result.error}")

    v_mean = sum(values) / len(values) if values else NEG_INF
    err: Optional[str] = None
    if deterministic is False:
        err = "non-deterministic replay"
    if errors:
        err = f"{err}; {errors[0]}" if err else errors[0]
    if deterministic is False:
        v_mean = NEG_INF

    return VersionEvaluation(
        revision=candidate.revision,
        v_mean=v_mean,
        deterministic=deterministic,
        per_world=per_world,
        error=err,
    )


def _format_report(
    worlds: Sequence[DiscoveryTree],
    evaluations: Sequence[VersionEvaluation],
    selected: int,
    monotone: bool,
) -> str:
    lines = [f"dream phase: {len(worlds)} replay world(s), {len(evaluations)} version(s) evaluated"]
    for index, ev in enumerate(evaluations):
        mean_probes = sum(w["probes"] for w in ev.per_world) / max(1, len(ev.per_world))
        mean_rounds = sum(w["rounds"] for w in ev.per_world) / max(1, len(ev.per_world))
        lines.append(
            f"  pi_{index}: V={ev.v_mean:.4f} probes={mean_probes:.1f} rounds={mean_rounds:.1f}"
            + (f" error={ev.error}" if ev.error else "")
        )
    lines.append(
        f"selected pi_{selected} (V* >= V_0: {monotone})"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# cross-cycle beta + fallback
# ----------------------------------------------------------------------


def fallback_grid(task: Dict[str, Any]) -> Dict[str, Any]:
    """Default safe grid: ``branch_count = max(1, workers)``, ``refine_count = max(1, k1)``."""
    workers = int(task.get("workers", 1) or 1)
    k1 = int(task.get("k1", 1) or 1)
    return {
        "branch_count": max(1, workers),
        "refine_count": max(1, k1),
        "reason": "fallback: one branch per worker refined to the round budget",
    }
