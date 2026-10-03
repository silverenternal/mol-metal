"""Replay episode — replay one policy version against one frozen world.

Pure-Python port of ``extensions/pi-dream-rsi/engine/replay.ts``
(``replayEpisode``) and ``reward.ts``.

Reward functions:

*   **Replay objective (Eq. 1)** — used to *select* the next policy:

        V = max s_v  −  β₁ · N  +  β₂ · N / max(1, k*)

    where ``N`` is the number of revealed non-root nodes (each one is a
    generation-evaluation request) and ``k*`` is the number of completed
    decision rounds.

*   **Sweep ranking** — judges a policy's whole beta curve:

        pareto.reward = pareto.auc  −  λ · parallel_penalty

    ``parallel_penalty`` is the mean of
    ``effective_sequential_rounds / total_probes``; a serial policy sits
    near 1, full batches approach ``1/W``.

All values are derived from the host-recorded decision trace, never from
numbers a policy reports about itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

from .discovery_tree import DiscoveryTree
from .policy_host import Decision, HostOptions, PolicyHost
from .replay_probe import ProbeResult, RevealedNode, make_replay_probe


@dataclass
class RewardConfig:
    """Reward configuration — explicit fields, no hidden defaults."""

    max_parallelism: int
    beta1: float
    beta2: float
    #: Lambda for the *sweep* ranking; **not** the Eq. 1 reward.
    lambda_: float
    baseline_score: Optional[float]


@dataclass
class EpisodeMetrics:
    """Metrics derived from one replay episode's decision trace."""

    #: Revealed non-root nodes = N.
    probes: int
    #: Completed decision rounds = k*.
    rounds: int
    #: Max s_v over everything revealed so far (including the root's baseline).
    attainment: Optional[float]
    #: Eq. 1.
    value: float
    #: Mean per-probe attainment (diagnostic).
    auc: float
    #: effective_sequential_rounds / total_probes.
    parallel_penalty: float
    #: Mean revealed cells per decision round.
    batch_size: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probes": self.probes,
            "rounds": self.rounds,
            "attainment": self.attainment,
            "value": self.value,
            "auc": self.auc,
            "parallel_penalty": self.parallel_penalty,
            "batch_size": self.batch_size,
        }


def replay_value(
    attainment: Optional[float],
    probes: int,
    rounds: int,
    cfg: RewardConfig,
) -> float:
    """Eq. 1: ``max(s_v) − β₁ · N + β₂ · N / max(1, k*)``."""
    quality = float(attainment) if attainment is not None else 0.0
    parallelism_bonus = (cfg.beta2 * probes) / max(1, rounds) if probes > 0 else 0.0
    return quality - cfg.beta1 * probes + parallelism_bonus


def trace_metrics(trace: Sequence[Decision], cfg: RewardConfig) -> EpisodeMetrics:
    """Derive everything the sweep needs from one replay episode's decision trace."""
    probes = 0
    attainment: Optional[float] = cfg.baseline_score
    area = 0.0
    effective_sequential_rounds = 0
    W = max(1, cfg.max_parallelism)

    for decision in trace:
        revealed = list(decision.revealed.values())
        # A batch of size k costs ceil(k / W) effective sequential rounds, regardless of how many
        # of the selected cells turned out to have a recorded continuation.
        effective_sequential_rounds += math.ceil(len(decision.batch) / W)
        for observation in revealed:
            probes += 1
            if (
                observation.evaluated
                and observation.error is None
                and observation.fail_class == "ok"
                and isinstance(observation.score, (int, float))
            ):
                if attainment is None or float(observation.score) > float(attainment):
                    attainment = float(observation.score)
            area += float(attainment) if attainment is not None else 0.0

    auc = (
        float(cfg.baseline_score)
        if cfg.baseline_score is not None
        else 0.0
        if probes == 0
        else area / probes
    )
    parallel_penalty = 0.0 if probes == 0 else effective_sequential_rounds / probes
    return EpisodeMetrics(
        probes=probes,
        rounds=len(trace),
        attainment=attainment,
        value=replay_value(attainment, probes, len(trace), cfg),
        auc=float(auc),
        parallel_penalty=float(parallel_penalty),
        batch_size=(probes / len(trace)) if trace else 0.0,
    )


# A policy module is anything callable that returns a list of decisions
# given a host.  In the TS reference this is loaded as a Node module; for
# our purposes a plain Python callable suffices (and the tests use one).
PolicyFn = Callable[..., Any]  # callable accepting a host (and optional args)


@dataclass
class ReplayEpisodeOptions:
    """Replay-episode options."""

    #: The policy to evaluate.  Accepts a Python callable or a string path
    #: (the string path is reserved for the eventual integration with the
    #: real Dream-RSI worker — for the unit-tested port a callable is
    #: enough).
    policy: Union[PolicyFn, str]
    world: DiscoveryTree
    #: The one scalar baked into the policy config; ``None`` uses the policy's
    #: own default.
    beta: Optional[float]
    max_parallelism: int
    #: K2 — maximum decision rounds replay allows.
    k2: int
    beta1: float
    beta2: float
    #: Lambda for the sweep ranking (not Eq. 1).
    lambda_: float = 1.0
    timeout_ms: Optional[float] = None
    verify_determinism: bool = False


@dataclass
class ReplayEpisodeResult:
    """Outcome of one replay episode."""

    ok: bool
    error: Optional[str]
    timed_out: bool
    metrics: EpisodeMetrics
    trace: List[Decision]
    deterministic: Optional[bool]
    #: The beta the policy actually ran at.
    beta: Optional[float]
    duration_ms: float


@dataclass
class ReplayEpisode:
    """Replay-episode runner — stateful only via its ``options``.

    Mirrors the TS reference: a fresh :class:`PolicyHost` is built in
    ``"replay"`` mode, the policy is invoked once with K2 round cap, then
    :func:`trace_metrics` derives the Eq. 1 score from the recorded trace.
    """

    options: ReplayEpisodeOptions

    async def run(self, policy_fn: Optional[PolicyFn] = None) -> ReplayEpisodeResult:
        return await run_replay_episode(self.options, policy_fn=policy_fn)


async def run_replay_episode(
    options: ReplayEpisodeOptions,
    policy_fn: Optional[PolicyFn] = None,
) -> ReplayEpisodeResult:
    """Replay one policy version against one frozen world.

    ``policy_fn`` overrides ``options.policy`` when supplied — used by tests
    that pass in synthetic policies.  When ``options.policy`` is itself a
    callable, that callable is used.
    """
    started = time.monotonic()
    reward_cfg = RewardConfig(
        max_parallelism=int(options.max_parallelism),
        beta1=float(options.beta1),
        beta2=float(options.beta2),
        lambda_=float(options.lambda_),
        baseline_score=options.world.baseline_score,
    )
    host = _build_replay_host(options)
    callable_to_use: Optional[PolicyFn] = policy_fn
    if callable_to_use is None and callable(options.policy):
        callable_to_use = options.policy

    error: Optional[str] = None
    timed_out = False
    trace: List[Decision] = []
    deterministic: Optional[bool] = None
    actual_beta: Optional[float] = options.beta
    try:
        if callable_to_use is None:
            raise ValueError(
                "replay_episode requires a policy callable (string paths are not "
                "supported in the pure-Python port)"
            )
        result = callable_to_use(host)
        if hasattr(result, "__await__"):
            trace = await result  # type: ignore[assignment]
        else:
            trace = result  # type: ignore[assignment]
    except TimeoutError:
        timed_out = True
        error = "policy timed out"
    except Exception as exc:  # noqa: BLE001 — surface to the caller
        error = f"{type(exc).__name__}: {exc}"

    # Determiism check: re-run and compare decision traces.
    if options.verify_determinism and error is None:
        try:
            host2 = _build_replay_host(options)
            result2 = callable_to_use(host2)
            if hasattr(result2, "__await__"):
                trace2 = await result2  # type: ignore[assignment]
            else:
                trace2 = result2  # type: ignore[assignment]
            deterministic = _decisions_equal(trace, trace2)
            if not deterministic:
                error = "non-deterministic replay"
        except Exception as exc:  # noqa: BLE001
            deterministic = False
            error = f"determinism check failed: {exc}"
    elif deterministic is None and error is None:
        deterministic = True

    metrics = trace_metrics(trace, reward_cfg)
    duration_ms = (time.monotonic() - started) * 1000.0
    return ReplayEpisodeResult(
        ok=error is None,
        error=error,
        timed_out=timed_out,
        metrics=metrics,
        trace=list(trace),
        deterministic=deterministic,
        beta=actual_beta,
        duration_ms=duration_ms,
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _build_replay_host(options: ReplayEpisodeOptions) -> PolicyHost:
    """Construct a replay-mode host with a probe that reads its own revealed set.

    The probe's ``is_revealed`` predicate points at the *host's* revealed
    set so the simulator and the host cannot disagree about what is
    revealed.  Without this pairing, replay could double-reveal the same
    frontier, distorting the sweep.
    """
    world = options.world
    host = PolicyHost(
        HostOptions(
            mode="replay",
            max_parallelism=int(options.max_parallelism),
            probe=lambda cells, _round: [],  # placeholder; replaced below
            grid={
                "branch_count": int(world.branch_count or 0),
                "refine_count": int(world.refine_count or 0),
            },
            max_rounds=int(options.k2),
            baseline_score=world.baseline_score,
            refine_count=int(world.refine_count or 0),
        )
    )
    host.probe_fn = make_replay_probe(world, host.is_revealed)
    return host


def _decisions_equal(a: Sequence[Decision], b: Sequence[Decision]) -> bool:
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        if da.round != db.round:
            return False
        if list(da.batch) != list(db.batch):
            return False
        if list(da.observed) != list(db.observed):
            return False
        if set(da.revealed.keys()) != set(db.revealed.keys()):
            return False
        for cid, obs_a in da.revealed.items():
            obs_b = db.revealed[cid]
            if obs_a.to_dict() != obs_b.to_dict():
                return False
        if da.best_after != db.best_after:
            return False
    return True
