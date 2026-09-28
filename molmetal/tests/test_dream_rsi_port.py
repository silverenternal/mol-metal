"""Tests for the pi-Dream-RSI Python port (molmetal/molmetal_lam/search_alg/dream_rsi/).

Mirrors the contract described in ``paper §3`` ("Offline evaluation" +
"Policy improvement and selection"):

*   :class:`DiscoveryTree` structure and observation semantics.
*   :func:`make_replay_probe` deterministic rules (slot / frontier /
    empty).
*   :class:`ReplayEpisode` runs to completion with a synthetic policy.
*   :class:`DreamLoop` selection monotone invariant ``V* >= V_0``.
*   Replay determinism: same world → same result.
"""

from __future__ import annotations

import asyncio
import math
from typing import List

from molmetal.molmetal_lam.search_alg.dream_rsi import (
    ROOT_ID,
    Decision,
    DiscoveryTree,
    DreamLoop,
    DreamOptions,
    HostOptions,
    PolicyCandidate,
    PolicyHost,
    ReplayEpisode,
    ReplayEpisodeOptions,
    RewardConfig,
    is_success,
    make_replay_probe,
    replay_value,
)


# ======================================================================
# helpers
# ======================================================================


def _run(coro):
    """Run an async coroutine in tests."""
    return asyncio.run(coro)


def _make_world(
    baseline: float = 0.0,
    branch_count: int = 2,
    refine_count: int = 2,
    branch_scores: dict | None = None,
) -> DiscoveryTree:
    """Build a frozen world with two branches × one refinement.

    Each branch root gets an executed (evaluated) outcome with score
    ``branch_scores.get(branch, 1.0)``, plus a refinement child
    ``b{branch}a1`` with score ``branch_scores.get((branch, 1), 0.5)``.
    """
    branch_scores = branch_scores or {}
    tree = DiscoveryTree(
        baseline_score=baseline,
        branch_count=branch_count,
        refine_count=refine_count,
    )
    for branch in range(branch_count):
        slot = tree.add_branch_slot(branch)
        # Mark slot as opened (executed branch root).
        slot.evaluated = True
        slot.score = branch_scores.get(branch, 1.0)
        slot.valid = True
        slot.fail_class = "ok"
        # First refinement child.
        child = tree.add_child(slot.meta.cell_id, branch, 1, ["refine"])
        child.evaluated = True
        child.score = branch_scores.get((branch, 1), 0.5)
        child.valid = True
        child.fail_class = "ok"
    return tree


def _closed_world(baseline: float = 0.0) -> DiscoveryTree:
    """Build a world whose every cell is already revealed (so the probe
    has nothing left to reveal)."""
    tree = DiscoveryTree(
        baseline_score=baseline, branch_count=1, refine_count=1
    )
    slot = tree.add_branch_slot(0)
    slot.evaluated = True
    slot.score = 0.5
    slot.valid = True
    slot.fail_class = "ok"
    return tree


# ======================================================================
# DiscoveryTree
# ======================================================================


def test_discovery_tree_add_branch_slot():
    """addBranchSlot is idempotent and tags as 'root-slot'."""
    tree = DiscoveryTree(baseline_score=1.0, branch_count=2, refine_count=2)
    node_a = tree.add_branch_slot(0)
    node_a_again = tree.add_branch_slot(0)
    assert node_a is node_a_again, "addBranchSlot must be idempotent"
    assert node_a.meta.cell_id == "b0a0"
    assert node_a.meta.parent_id == ROOT_ID
    assert node_a.meta.attempt == 0
    assert "root-slot" in node_a.meta.tags

    # Adding a different branch slot assigns the next seq.
    node_b = tree.add_branch_slot(1)
    assert node_b.meta.cell_id == "b1a0"
    assert node_b.meta.seq > node_a.meta.seq


def test_discovery_tree_add_child():
    """addChild creates a refinement child with parent_id pointing at the caller."""
    tree = DiscoveryTree(baseline_score=0.5, branch_count=2, refine_count=2)
    slot = tree.add_branch_slot(0)
    child = tree.add_child(slot.meta.cell_id, 0, 1, ["refine"])
    assert child.meta.cell_id == "b0a1"
    assert child.meta.parent_id == slot.meta.cell_id
    assert child.meta.attempt == 1
    assert child.meta.branch == 0
    assert "refine" in child.meta.tags
    # The parent's chain is now non-empty.
    assert tree.frontier_of(0) is not None
    assert tree.frontier_of(0).meta.cell_id == child.meta.cell_id


# ======================================================================
# Replay probe rules
# ======================================================================


def test_replay_probe_unopened_root():
    """Rule 1: an unopened-root-slot selector reveals the earliest-created recorded child."""
    world = _make_world(baseline=0.0, branch_count=2, refine_count=2)
    revealed: set = set()  # empty: nothing is revealed yet.

    def is_revealed(cell: str) -> bool:
        return cell in revealed

    # Selecting an unopened root slot ``b0a0`` reveals the recorded
    # ``b0a0`` child (the slot itself is also the recorded child of ROOT
    # when branch 0 is opened).
    probe = make_replay_probe(world, is_revealed)
    out = probe(["b0a0"], 1)
    # The probe returns revealed nodes that were *recorded* in the world.
    # For an unopened slot selector, it returns the recorded opened-root
    # child of ROOT.
    assert any(r.cell == "b0a0" for r in out), out


def test_replay_probe_opened_frontier():
    """Rule 2: an opened-frontier selector reveals that node's recorded child."""
    world = _make_world(baseline=0.0, branch_count=2, refine_count=2)
    revealed = {"b0a0"}  # the branch-0 slot has been revealed.

    def is_revealed(cell: str) -> bool:
        return cell in revealed

    probe = make_replay_probe(world, is_revealed)
    out = probe(["b0a0"], 1)
    assert any(r.cell == "b0a1" for r in out), out


def test_replay_probe_empty_returns_none():
    """Rule 3: Child(v)=∅ when no recorded continuation remains."""
    world = _closed_world(baseline=0.0)
    revealed = {"b0a0"}  # the only recorded cell.

    def is_revealed(cell: str) -> bool:
        return cell in revealed

    probe = make_replay_probe(world, is_revealed)
    out = probe(["b0a0"], 1)
    assert out == [], out


# ======================================================================
# Replay episode
# ======================================================================


async def _serial_policy_fn(host: PolicyHost) -> List[Decision]:
    """A simple policy: open one branch at a time, then keep refining."""
    while True:
        if host.rounds_exhausted:
            break
        legal = host.legal_actions()
        if not legal:
            break
        # Probe the first legal action.
        cell = legal[0]
        await host.probe_batch([cell])
    return host.decisions


def test_replay_episode_runs_to_completion():
    """A replay episode runs to completion and produces finite Eq. 1 metrics."""
    world = _make_world(baseline=0.0, branch_count=2, refine_count=2)
    opts = ReplayEpisodeOptions(
        policy=_serial_policy_fn,
        world=world,
        beta=0.5,
        max_parallelism=2,
        k2=4,
        beta1=0.05,
        beta2=0.1,
        lambda_=1.0,
    )
    episode = ReplayEpisode(opts)
    result = _run(episode.run())
    assert result.ok, f"replay failed: {result.error}"
    assert result.metrics.probes > 0
    assert result.metrics.rounds >= 1
    assert math.isfinite(result.metrics.value)
    assert result.metrics.attainment is not None
    assert result.metrics.attainment >= 0.0


# ======================================================================
# Dream loop invariants
# ======================================================================


def test_dream_loop_v0_equals_baseline():
    """When only π_0 is a candidate, V* == V_0 (monotone invariant)."""
    world = _make_world(baseline=0.0, branch_count=2, refine_count=2)

    async def pi_0(host: PolicyHost) -> List[Decision]:
        while True:
            if host.rounds_exhausted:
                break
            legal = host.legal_actions()
            if not legal:
                break
            await host.probe_batch([legal[0]])
        return host.decisions

    options = DreamOptions(
        worlds=[world],
        candidates=[PolicyCandidate(revision=0, policy_fn=pi_0, name="π_0")],
        max_parallelism=2,
        k2=4,
        beta1=0.05,
        beta2=0.1,
        lambda_=1.0,
        verify_determinism=False,
    )
    loop = DreamLoop(options)
    result = _run(loop.run())
    assert result.ok, f"dream phase failed: {result.error}"
    # V* == V_0 when there's only one candidate.
    assert result.selected == 0
    assert math.isfinite(result.evaluations[0].v_mean)


def test_dream_loop_selects_better_candidate():
    """When a better candidate exists, the dream loop picks it (V* > V_0)."""
    # World: branch 0's refinement child has the highest recorded score.
    # A policy that reaches that child (π_1) beats one that doesn't (π_0).
    world = _make_world(
        baseline=0.0,
        branch_count=2,
        refine_count=2,
        branch_scores={
            0: 0.5,            # branch-0 root scores 0.5
            (0, 1): 2.0,       # branch-0 first refinement scores 2.0
            1: 0.4,            # branch-1 root scores 0.4
            (1, 1): 0.3,       # branch-1 first refinement scores 0.3
        },
    )

    async def pi_0(host: PolicyHost) -> List[Decision]:
        """Stall policy: opens branch 0 then stops (max attainment 0.5)."""
        legal = host.legal_actions()
        if not legal:
            return host.decisions
        await host.probe_batch([legal[0]])
        return host.decisions

    async def pi_1(host: PolicyHost) -> List[Decision]:
        """Aggressive policy: opens branch 0, then refines it (max attainment 2.0)."""
        # Round 1: open branch 0 (the first unopened root slot).
        legal = host.legal_actions()
        await host.probe_batch([legal[0]])
        # Round 2: refine the branch-0 frontier (skip unopened roots).
        legal = host.legal_actions()
        frontier_cells = [cid for cid in legal if host.is_revealed(cid)]
        if frontier_cells:
            await host.probe_batch([frontier_cells[0]])
        return host.decisions

    options = DreamOptions(
        worlds=[world],
        candidates=[
            PolicyCandidate(revision=0, policy_fn=pi_0, name="π_0"),
            PolicyCandidate(revision=1, policy_fn=pi_1, name="π_1"),
        ],
        max_parallelism=1,
        k2=4,
        beta1=0.05,
        beta2=0.1,
        lambda_=1.0,
        verify_determinism=False,
    )
    loop = DreamLoop(options)
    result = _run(loop.run())
    assert result.ok, f"dream phase failed: {result.error}"
    # π_1 must be selected — it reaches the b0a1 child with score 2.0.
    assert result.selected == 1
    assert result.evaluations[1].v_mean > result.evaluations[0].v_mean


# ======================================================================
# Replay determinism
# ======================================================================


def test_replay_determinism_same_world_same_result():
    """Replaying the same world twice produces the same Eq. 1 metrics."""
    world = _make_world(baseline=0.0, branch_count=2, refine_count=2)
    opts = ReplayEpisodeOptions(
        policy=_serial_policy_fn,
        world=world,
        beta=0.5,
        max_parallelism=2,
        k2=4,
        beta1=0.05,
        beta2=0.1,
        lambda_=1.0,
    )
    episode_a = ReplayEpisode(opts)
    episode_b = ReplayEpisode(opts)
    result_a = _run(episode_a.run())
    result_b = _run(episode_b.run())
    assert result_a.ok and result_b.ok
    assert result_a.metrics.value == result_b.metrics.value
    assert result_a.metrics.attainment == result_b.metrics.attainment
    assert result_a.metrics.probes == result_b.metrics.probes
    assert result_a.metrics.rounds == result_b.metrics.rounds


# ======================================================================
# Reward function
# ======================================================================


def test_reward_function_eq1():
    """Eq. 1: V = max(s_v) − β₁ · N + β₂ · N / max(1, k*)."""
    cfg = RewardConfig(max_parallelism=2, beta1=0.05, beta2=0.1, lambda_=1.0, baseline_score=0.0)
    # N=4 probes, k*=2 rounds, attainment=1.0
    v = replay_value(attainment=1.0, probes=4, rounds=2, cfg=cfg)
    assert math.isclose(v, 1.0 - 0.05 * 4 + 0.1 * 4 / 2)
    # No probes: quality still counted, parallelism bonus is 0.
    v0 = replay_value(attainment=0.5, probes=0, rounds=0, cfg=cfg)
    assert math.isclose(v0, 0.5)
