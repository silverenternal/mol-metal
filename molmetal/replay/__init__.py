"""molmetal.replay — Dream-RSI style replay pool for mol-metal training.

Implements the foundation primitives from TODO-150 (Dream-RSI integration
plan) so that downstream techniques (A1 config-search via replay, A3
per-metal bootstrap, A6 replay-disagreement uncertainty) can consume a
single source of truth.

The replay pool is the mol-metal analogue of the paper's "discovery-tree
replay buffer" — every (config_hash, hp_dict, per_epoch_metrics,
final_test_score, metal, seed) trajectory from past training runs is
appended and replayed deterministically to score new HP candidates
without paying for fresh GPU runs.

Why a new module (and not an extension of
:mod:`molmetal.molmetal_lam.search_alg.mcts_replay_buffer`)?  The MCTS
replay buffer is tightly scoped to the AlphaGo-style ``(s, a, z, pi)``
tuple contract used by ``MCTSProofSearch``.  The Dream-RSI primitive
operates at the *training-loop* level — config + per-epoch metrics +
final test score — and so warrants its own package.  The two are
complementary: MCTS replay feeds action selection during proof search;
this pool feeds HP selection during training.

Backfill source (P0.2 in TODO-150): existing ``dmpnn_ckpt_status.json``
(25 ckpts from Wf-120 P2) + ``metal_lomo_auc_wasserstein.json`` +
``metalcytotox_*_auc.json`` should yield >200 trajectories once
expanded with ``metal``, ``seed``, ``hp_dict``, ``per_epoch_metrics``.
Until that backfill is done, the pool starts empty and
:meth:`ReplayPool.coverage_report` will flag every metal as
``LOW_COVERAGE`` — predictions for those cells are pessimistic-shrunk
per §4 R1 mitigation.
"""

from __future__ import annotations

from .pool import (
    CoverageStats,
    LOW_COVERAGE_THRESHOLD,
    ReplayPool,
    ScoreEstimate,
    TrajectoryRecord,
    pessimistic_shrinkage,
)
from .hpo import (
    ASHAEarlyStop,
    DEFAULT_CACHE_ROOT,
    DEFAULT_FEATURE_DIM,
    ColdStartSurrogate,
    EarlyStopDecision,
    EpsilonGreedyAcquisition,
    EpsilonSchedule,
    RSI_ITER,
    RSI_SCHEMA_FROZEN,
    SurrogateDecision,
    TrajectoryCacheEntry,
    TrajectoryCacheWriter,
    config_hash_from_hp,
)

__all__ = [
    # P0 pool primitives
    "CoverageStats",
    "LOW_COVERAGE_THRESHOLD",
    "ReplayPool",
    "ScoreEstimate",
    "TrajectoryRecord",
    "pessimistic_shrinkage",
    # P1 HPO primitives (this module)
    "ASHAEarlyStop",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_FEATURE_DIM",
    "ColdStartSurrogate",
    "EarlyStopDecision",
    "EpsilonGreedyAcquisition",
    "EpsilonSchedule",
    "RSI_ITER",
    "RSI_SCHEMA_FROZEN",
    "SurrogateDecision",
    "TrajectoryCacheEntry",
    "TrajectoryCacheWriter",
    "config_hash_from_hp",
]
