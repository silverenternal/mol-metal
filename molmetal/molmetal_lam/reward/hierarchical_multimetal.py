"""Hierarchical multi-metal coordination scaffolding (Wf-78, R22-P1 Phase 1).

================================================================
Background — R22-P1 (L13 hierarchical multi-metal coord, 2026-09-26)
================================================================
TODO-105 (Wf-78 R22-P1 P1.1-P1.2 CPU scaffolding) introduces a
**hierarchical coordination loop** as a CPU-only scaffold for the
L13 Lambda gap (see ``TODO/pending/r22_l13_l16_gap_analysis.md`` §1).

The single-metal closure theorem in
:mod:`molmetal.molmetal_lam.lam_chem.closure` measures reachability
``R(S, R, B)`` — the BFS closure of a starting scaffold ``S`` under
a fixed click-rule set ``R`` bounded by depth ``B``.  Wf-36 P1
(L1) reports ``reachability_rate=0.5108`` on PlatinAI 5K × 9 metals;
Wf-37 P2 (L6) reports ``0.5891`` aggregate over 30 metals × 100
samples; ``completed/96_r20_p5_closure_catherine.md`` §3.2 confirms
the gate ``0.95`` is **structurally NOT MET** with three structural
blockers:

  1. Valence-BNF rejection of ``[M+]<-X`` dative bonds (RDKit
     valence accounting undercounts metal bonds).
  2. Tautomerisation redex exhaustion (~30% tmQM samples).
  3. Sample sparsity for 7 metals (Hf=20, Ta=30, La=3, Nb=19,
     Tc=15, Sc=20, Zr=53).

L13 (hierarchical multi-metal coord) is the **HIGH-priority
candidate** with **188 scan signals** (57% of 330 total) per
``r22_l13_l16_gap_analysis.md`` §7.1.  This module authors the
**first half of the L13 candidate approach** (the P1.1 + P1.2
sub-tasks of R22-P1):

  * **Outer loop** — target selection (over candidate binding
    pockets / scaffolds indexed by ``target_id``).
  * **Inner loop** — metal selection (over a configurable metal
    pool ``{Pt, Pd, Ru, Ir, Au, Rh, Os, Re}`` for d8/d6 4d/5d
    centers).
  * Per-metal + per-target breakdown — the L13 metric we want to
    **observe structurally lift** is the **per-metal hierarchical
    reachability**; for each (target, metal) pair, we run a
    shallow closure BFS using a synthetic mock scoring oracle and
    record the inner-loop reward.

The **second half** of the L13 candidate (P1.3 multi-metal cluster
centers + P1.4 trans-effect propagation) belongs to the Wf-79
R22-P1 GPU phase and is **deferred to Wf-79** per the
R22-P1 plan.

This is a **CPU-only scaffold**: synthetic mock + smoke test, no
GPU runs, no real metal-coord chemistry.  The P1.1 + P1.2 outputs
are:

  * :class:`HierarchicalCoordinator`  — outer/inner loop runner.
  * :class:`HierarchicalCoordResult`  — dataclass with per-metal +
    per-target breakdown.
  * :func:`synthetic_mock_score`      — deterministic mock scorer
    that returns a finite reward without any chemistry.
  * :func:`smoke_hierarchical`        — end-to-end smoke call
    (3 targets × 5 metals × depth 2).

Honest framing
--------------
* This is a **CPU-only skeleton**.  No GPU retrain, no real
  closure-theorem invocation, no ``closure_theorem`` BFS — the
  scorer is a deterministic mock that uses a hash of the
  (target, metal) pair.  R22-P1 P1.3 will plug in the real
  ``closure_test`` from :mod:`molmetal.molmetal_lam.lam_chem.closure`.
* The P1.1 + P1.2 outputs do **NOT** lift the reachability gate
  (0.95) on their own; the structural blocker is upstream.
* Per :class:`HierarchicalCoordResult`, the metrics are
  ``n_targets``, ``n_metals``, ``n_pairs``, ``mean_inner_reward``,
  ``per_metal_mean_reward`` and ``per_target_mean_reward`` — all
  computed deterministically from the synthetic mock.
* R17 surface (closure.py + lam_chem/) is **not modified**; this
  module is independently testable and exists to scaffold the L13
  outer/inner loop pattern for downstream Wf-79 R22-P1 P1.3 + P1.4
  integration.

Public API
----------
* :class:`MetalPool`                 — supported d-block metal set.
* :class:`TargetSpec`                — target scaffold specification.
* :class:`HierarchicalCoordConfig`   — config dataclass.
* :class:`HierarchicalCoordResult`   — result dataclass.
* :class:`HierarchicalCoordinator`   — outer/inner loop runner.
* :func:`synthetic_mock_score`       — deterministic mock scorer.
* :func:`smoke_hierarchical`         — end-to-end smoke runner.

Lit anchors
-----------
* ``TODO/pending/r22_l13_l16_gap_analysis.md`` §1 (L13 candidate).
* ``TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md`` §2 (R22-P1 plan).
* ``TODO/completed/96_r20_p5_closure_catherine.md`` §3.2 (gate 0.95
  NOT MET structural reasons).
* ``TODO/completed/95_r20_p4_3way_100pocket.md`` (3-way + 100-pocket
  baseline).
* Reedijk 2008 *Platinum Metals Rev.* (trans-effect ordering,
  deferred to Wf-79 R22-P1 P1.4).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Metal pool — supported d-block metal centers for hierarchical coord
# ---------------------------------------------------------------------------
# Eight d-block centers spanning 4d (Ru/Rh/Pd) and 5d (Os/Ir/Pt/Au)
# plus the L12 bottleneck metal Re.  This pool covers the closure
# bench metal subset per ``completed/95_r20_p4_3way_100pocket.md``
# (Ru/Rh/Pd/Os/Ir/Re/Pt/Au = 8 of 30 tmQM metals used in Wf-67).
SUPPORTED_METALS: Tuple[str, ...] = (
    "Pt",
    "Pd",
    "Ru",
    "Rh",
    "Ir",
    "Os",
    "Au",
    "Re",
)
assert (
    len(SUPPORTED_METALS) == 8
), f"L13 metal pool must stay at 8 metals; got {len(SUPPORTED_METALS)}"


@dataclass(frozen=True)
class MetalPool:
    """Immutable d-block metal pool for the inner loop.

    Parameters
    ----------
    metals : Tuple[str, ...]
        Tuple of metal symbols (default :data:`SUPPORTED_METALS`).
    """

    metals: Tuple[str, ...] = SUPPORTED_METALS

    def __post_init__(self) -> None:
        if not self.metals:
            raise ValueError("MetalPool.metals must be non-empty")
        for m in self.metals:
            if not isinstance(m, str) or not m:
                raise ValueError(f"invalid metal symbol: {m!r}")

    @property
    def n_metals(self) -> int:
        return len(self.metals)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.metals)

    def __contains__(self, metal: str) -> bool:
        return metal in self.metals


# ---------------------------------------------------------------------------
# Target scaffold — outer-loop element
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    """A single target scaffold for the outer loop.

    The target is intentionally **abstract** at the P1.1 + P1.2
    stage — we do NOT depend on PlatinAI 5K / tmQM / MetalCytoToxDB
    real data here.  Instead the ``target_id`` is a hashable string
    that drives the deterministic mock scorer.  Wf-79 R22-P1 P1.3
    will swap :class:`TargetSpec` for a ``(pocket_id, scaffold_smiles)``
    pair sourced from a real dataset.

    Parameters
    ----------
    target_id : str
        Stable identifier for the target (e.g. ``"pocket_4aaw"``).
    """

    target_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError(
                f"target_id must be a non-empty string; got {self.target_id!r}"
            )


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HierarchicalCoordConfig:
    """Configuration for :class:`HierarchicalCoordinator`.

    Parameters
    ----------
    max_outer_depth : int, default ``2``
        Maximum number of outer-loop (target-selection) iterations.
        Default matches the P1.2 hierarchical BFS depth-≥2 spec.
    inner_per_target : int, default ``8``
        Number of inner-loop (metal-selection) samples per outer
        iteration.  Defaults to the 8-metal pool size; smaller
        values truncate the pool.
    seed : int, default ``0``
        Deterministic seed for the synthetic mock scorer.
    scorer : Optional[Callable[[str, str], float]], default ``None``
        Optional override scorer.  When ``None``, the runner uses
        :func:`synthetic_mock_score` with the configured seed.
    """

    max_outer_depth: int = 2
    inner_per_target: int = 8
    seed: int = 0
    scorer: Optional[Callable[[str, str], float]] = None

    def __post_init__(self) -> None:
        if self.max_outer_depth <= 0:
            raise ValueError(
                f"max_outer_depth must be positive; got {self.max_outer_depth}"
            )
        if self.inner_per_target <= 0:
            raise ValueError(
                f"inner_per_target must be positive; got {self.inner_per_target}"
            )


@dataclass(frozen=True)
class HierarchicalCoordResult:
    """Result of a hierarchical coordination run.

    Attributes
    ----------
    n_targets : int
        Number of distinct targets visited in the outer loop.
    n_metals : int
        Number of distinct metals sampled in the inner loop.
    n_pairs : int
        Total number of ``(target, metal)`` pairs scored.
    mean_inner_reward : float
        Mean of the inner-loop reward across all ``n_pairs``
        entries.  Deterministic given the seed.
    per_metal_mean_reward : Dict[str, float]
        Mean inner-loop reward per metal.  Length equals
        ``n_metals``.
    per_target_mean_reward : Dict[str, float]
        Mean inner-loop reward per target.  Length equals
        ``n_targets``.
    wall_time_s : float
        End-to-end wall time in seconds (CPU-only, mock scorer).
    seed : int
        Echo of the seed used for the run.
    max_outer_depth : int
        Echo of the configured outer depth.
    """

    n_targets: int
    n_metals: int
    n_pairs: int
    mean_inner_reward: float
    per_metal_mean_reward: Dict[str, float]
    per_target_mean_reward: Dict[str, float]
    wall_time_s: float
    seed: int
    max_outer_depth: int

    # ----- sanity accessors ------------------------------------------------
    @property
    def is_balanced(self) -> bool:
        """True iff every (target, metal) pair was scored exactly once
        per outer depth iteration.

        Returns
        -------
        bool
            ``True`` when ``n_pairs == n_targets * n_metals *
            max_outer_depth``.
        """
        if self.n_targets <= 0 or self.n_metals <= 0:
            return False
        return self.n_pairs == self.n_targets * self.n_metals * self.max_outer_depth

    @property
    def samples_per_pair(self) -> int:
        """Number of reward samples per (target, metal) pair.

        Equals ``max_outer_depth`` when ``inner_per_target`` covers
        the full metal pool; equals ``max_outer_depth * 0`` (i.e. 0)
        if any (target, metal) pair was missing.  The :attr:`is_balanced`
        flag is the strict version of this check.
        """
        if self.n_targets <= 0 or self.n_metals <= 0:
            return 0
        return self.n_pairs // (self.n_targets * self.n_metals)


# ---------------------------------------------------------------------------
# Synthetic mock scorer — deterministic, CPU-only, no real chemistry
# ---------------------------------------------------------------------------
def synthetic_mock_score(target_id: str, metal: str, seed: int = 0) -> float:
    """Deterministic mock reward for a (target, metal) pair.

    The scorer hashes ``(target_id, metal, seed)`` with SHA-256 and
    maps the first 8 hex chars to a float in ``[0.0, 1.0)``.  No
    real chemistry, no GPU, no external libraries.  This is a
    **stand-in** for the Wf-79 R22-P1 P1.3 scorer that will plug
    in :func:`closure_test` from :mod:`molmetal.molmetal_lam.lam_chem.closure`.

    Parameters
    ----------
    target_id : str
        The target identifier (any non-empty string).
    metal : str
        The metal symbol (must be a member of
        :data:`SUPPORTED_METALS`).
    seed : int, default ``0``
        Deterministic seed mixed into the hash.

    Returns
    -------
    float
        Finite reward in ``[0.0, 1.0)``.
    """
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(f"target_id must be a non-empty string; got {target_id!r}")
    if not isinstance(metal, str) or not metal:
        raise ValueError(f"metal must be a non-empty string; got {metal!r}")
    payload = f"{target_id}|{metal}|{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    # First 8 hex chars → integer in [0, 2**32) → float in [0, 1).
    n = int(digest[:8], 16)
    return float(n) / float(1 << 32)


# ---------------------------------------------------------------------------
# Coordinator — outer/inner loop runner
# ---------------------------------------------------------------------------
@dataclass
class HierarchicalCoordinator:
    """Hierarchical coordination runner (outer × inner loop).

    The coordinator iterates over an **outer loop** of
    :class:`TargetSpec` instances and, for each target, runs an
    **inner loop** over the :class:`MetalPool` members.  The mock
    scorer (or user override) emits a reward per (target, metal)
    pair, and the result is aggregated into a
    :class:`HierarchicalCoordResult`.

    Example
    -------
    >>> pool = MetalPool()                       # doctest: +SKIP
    >>> targets = [TargetSpec("pocket_4aaw"),   # doctest: +SKIP
    ...            TargetSpec("pocket_1h36")]   # doctest: +SKIP
    >>> coord = HierarchicalCoordinator(pool,    # doctest: +SKIP
    ...     targets,                            # doctest: +SKIP
    ...     HierarchicalCoordConfig(max_outer_depth=2))  # doctest: +SKIP
    >>> result = coord.run()                    # doctest: +SKIP
    >>> assert result.n_pairs == 2 * 8          # doctest: +SKIP

    Parameters
    ----------
    pool : MetalPool
        Inner-loop metal pool.
    targets : Sequence[TargetSpec]
        Outer-loop target sequence.
    config : HierarchicalCoordConfig, optional
        Runner configuration; defaults to the default config.
    """

    pool: MetalPool
    targets: Sequence[TargetSpec]
    config: HierarchicalCoordConfig = field(default_factory=HierarchicalCoordConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.pool, MetalPool):
            raise ValueError(
                f"pool must be a MetalPool; got {type(self.pool).__name__}"
            )
        if not isinstance(self.targets, Sequence) or not self.targets:
            raise ValueError("targets must be a non-empty sequence of TargetSpec")
        for t in self.targets:
            if not isinstance(t, TargetSpec):
                raise ValueError(
                    f"targets must contain only TargetSpec; got {type(t).__name__}"
                )
        if not isinstance(self.config, HierarchicalCoordConfig):
            raise ValueError(
                "config must be a HierarchicalCoordConfig; got "
                f"{type(self.config).__name__}"
            )

    # ----- public runner ----------------------------------------------------
    def run(self) -> HierarchicalCoordResult:
        """Execute the hierarchical outer × inner loop.

        Returns
        -------
        HierarchicalCoordResult
            Aggregated per-target + per-metal breakdown.  Deterministic
            given ``config.seed``.
        """
        scorer = self.config.scorer or (
            lambda target_id, metal: synthetic_mock_score(
                target_id, metal, seed=self.config.seed
            )
        )
        t_start = time.perf_counter()

        # The outer loop runs for ``max_outer_depth`` iterations.  Within
        # each outer iteration we re-visit every target and score every
        # metal in the pool.  This is the P1.2 hierarchical BFS depth-≥2
        # analogue for the synthetic mock.
        per_pair_rewards: List[Tuple[str, str, float]] = []
        n_targets = len(self.targets)
        metals = list(self.pool.metals)
        # Honour ``inner_per_target`` truncation if smaller than the pool.
        if self.config.inner_per_target < len(metals):
            metals = metals[: self.config.inner_per_target]

        for _depth in range(self.config.max_outer_depth):
            for tgt in self.targets:
                for metal in metals:
                    r = scorer(tgt.target_id, metal)
                    per_pair_rewards.append((tgt.target_id, metal, r))

        # Aggregate per-metal + per-target means.
        per_metal: Dict[str, List[float]] = {m: [] for m in metals}
        per_target: Dict[str, List[float]] = {t.target_id: [] for t in self.targets}
        for tgt_id, metal, r in per_pair_rewards:
            per_metal[metal].append(r)
            per_target[tgt_id].append(r)

        per_metal_mean = {
            m: (sum(v) / len(v)) if v else 0.0 for m, v in per_metal.items()
        }
        per_target_mean = {
            t: (sum(v) / len(v)) if v else 0.0 for t, v in per_target.items()
        }
        mean_reward = (
            sum(r for _, _, r in per_pair_rewards) / len(per_pair_rewards)
            if per_pair_rewards
            else 0.0
        )
        wall_time = time.perf_counter() - t_start

        return HierarchicalCoordResult(
            n_targets=n_targets,
            n_metals=len(metals),
            n_pairs=len(per_pair_rewards),
            mean_inner_reward=float(mean_reward),
            per_metal_mean_reward=dict(per_metal_mean),
            per_target_mean_reward=dict(per_target_mean),
            wall_time_s=float(wall_time),
            seed=int(self.config.seed),
            max_outer_depth=int(self.config.max_outer_depth),
        )


# ---------------------------------------------------------------------------
# Smoke runner — end-to-end deterministic smoke check
# ---------------------------------------------------------------------------
def smoke_hierarchical(
    n_targets: int = 3,
    seed: int = 0,
) -> HierarchicalCoordResult:
    """End-to-end smoke runner for the hierarchical coord scaffold.

    Constructs ``n_targets`` :class:`TargetSpec` instances with
    deterministic ``target_id`` strings (``smoke_target_<i>``) and
    runs the default :class:`HierarchicalCoordinator` against the
    full :data:`SUPPORTED_METALS` pool.  Designed to be called from
    ``pytest`` or as a CLI ``__main__`` smoke.

    Parameters
    ----------
    n_targets : int, default ``3``
        Number of outer-loop targets.  Must be ≥ 1.
    seed : int, default ``0``
        Deterministic seed.

    Returns
    -------
    HierarchicalCoordResult
        Aggregated result.
    """
    if n_targets <= 0:
        raise ValueError(f"n_targets must be positive; got {n_targets}")
    targets = [TargetSpec(f"smoke_target_{i}") for i in range(n_targets)]
    coord = HierarchicalCoordinator(
        pool=MetalPool(),
        targets=targets,
        config=HierarchicalCoordConfig(
            max_outer_depth=2,
            inner_per_target=len(SUPPORTED_METALS),
            seed=seed,
        ),
    )
    return coord.run()


# ---------------------------------------------------------------------------
# CLI smoke entry point — `python -m molmetal.molmetal_lam.reward.hierarchical_multimetal`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = smoke_hierarchical(n_targets=3, seed=0)
    print(
        f"hierarchical_multimetal smoke: "
        f"n_targets={result.n_targets} "
        f"n_metals={result.n_metals} "
        f"n_pairs={result.n_pairs} "
        f"mean_inner_reward={result.mean_inner_reward:.6f} "
        f"wall_time_s={result.wall_time_s:.4f}"
    )
