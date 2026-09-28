"""Tests for Wf-36 Phase-3 — joint ScoreRepellent + VariancePUCT + MCTS pipeline.

================================================================
CPU-only smoke / contract tests for ``joint_score_repellent_variance_puct_mcts``
================================================================
This module validates the four-stage joint pipeline that wires
together the Wf-9 / Wf-11 / Wf-14 / Wf-21 ships:

* Stage 1 — ScoreRepellent MCMC initial-state sampling
* Stage 2 — VariancePUCT action selection
* Stage 3 — MCTS trajectory + VarianceMCTSWrapper
* Stage 4 — Replay buffer write

Total: **15 tests** matching the Wf-36 Phase-3 task brief.  All
CPU-only; completes in <10 s on any CI container.  None of the
tests mutate :mod:`proof_search` / :mod:`score_repellent_mc` /
:mod:`variance_puct_integration` / :mod:`variance_mcts_integration` /
:mod:`mcts_replay_buffer`.

Test inventory
--------------
* :class:`TestJointPipelineImports`     (3 tests) — module surface
* :class:`TestStage1ScoreRepellent`     (2 tests) — Stage 1 contracts
* :class:`TestStage2VariancePuct`       (2 tests) — Stage 2 contracts
* :class:`TestStage3MctsTrajectory`     (2 tests) — Stage 3 contracts
* :class:`TestStage4ReplayWrite`        (2 tests) — Stage 4 contracts
* :class:`TestFullPipeline`             (4 tests) — full pipeline

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_joint_3stage_mcts.py -v
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Any, Dict, List

import pytest

# Allow the test to run both as a top-level module and as part of
# the molmetal test suite.  Match the convention in
# ``test_variance_mcts_proof_search.py``.
_PROJECT_ROOT = "/home/hugo/codes/mol-metal"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
    )
    from molmetal_lam.search_alg.score_repellent_mc import (
        ScoreRepellentSampler,
    )
    from molmetal_lam.search_alg.variance_puct_integration import (
        IntegrationNode,
        VarianceControlledPUCT,
    )
    from molmetal_lam.search_alg.variance_mcts_integration import (
        VarianceMCTSWrapper,
    )
    from molmetal_lam.search_alg.proof_search import (
        EXPLORATION_STRATEGIES,
        MCTSProofSearch,
    )
    from molmetal_lam.binding.types import BindingSite
    from molmetal_lam.bonds.application import FreeSiteLedger
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import ReactionRule
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
        MCTSReplayBuffer,
        MCTSReplayEntry,
    )
    from molmetal.molmetal_lam.search_alg.score_repellent_mc import (  # type: ignore
        ScoreRepellentSampler,
    )
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore
        IntegrationNode,
        VarianceControlledPUCT,
    )
    from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
        VarianceMCTSWrapper,
    )
    from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore
        EXPLORATION_STRATEGIES,
        MCTSProofSearch,
    )
    from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
    from molmetal.molmetal_lam.bonds.application import (  # type: ignore
        FreeSiteLedger,
    )
    from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
        MoleculeClosedTerm,
    )
    from molmetal.molmetal_lam.reactions.beta_reductions import (  # type: ignore
        ReactionRule,
    )


try:
    from molmetal.scripts.joint_score_repellent_variance_puct_mcts import (
        DEFAULT_BUFFER_SIZE,
        DEFAULT_N_ITERS,
        DEFAULT_N_ROUNDS,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_STEP_SIZE,
        run_joint_pipeline,
        stage1_score_repellent,
        stage2_variance_puct,
        stage3_mcts_trajectory,
        stage4_replay_write,
    )
except Exception:  # pragma: no cover - import path fallback
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "molmetal", "scripts"))
    try:
        import importlib
        _joint_mod = importlib.import_module(
            "joint_score_repellent_variance_puct_mcts",
        )
        DEFAULT_BUFFER_SIZE = _joint_mod.DEFAULT_BUFFER_SIZE
        DEFAULT_N_ITERS = _joint_mod.DEFAULT_N_ITERS
        DEFAULT_N_ROUNDS = _joint_mod.DEFAULT_N_ROUNDS
        DEFAULT_N_TRAJECTORIES = _joint_mod.DEFAULT_N_TRAJECTORIES
        DEFAULT_STEP_SIZE = _joint_mod.DEFAULT_STEP_SIZE
        run_joint_pipeline = _joint_mod.run_joint_pipeline
        stage1_score_repellent = _joint_mod.stage1_score_repellent
        stage2_variance_puct = _joint_mod.stage2_variance_puct
        stage3_mcts_trajectory = _joint_mod.stage3_mcts_trajectory
        stage4_replay_write = _joint_mod.stage4_replay_write
    except Exception:
        # If the script is unavailable in this environment, fall back
        # to direct path injection so the import surface still works.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "joint_pipeline_fallback",
            os.path.join(
                _PROJECT_ROOT, "molmetal", "scripts",
                "joint_score_repellent_variance_puct_mcts.py",
            ),
        )
        _mod = _ilu.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        DEFAULT_BUFFER_SIZE = _mod.DEFAULT_BUFFER_SIZE
        DEFAULT_N_ITERS = _mod.DEFAULT_N_ITERS
        DEFAULT_N_ROUNDS = _mod.DEFAULT_N_ROUNDS
        DEFAULT_N_TRAJECTORIES = _mod.DEFAULT_N_TRAJECTORIES
        DEFAULT_STEP_SIZE = _mod.DEFAULT_STEP_SIZE
        run_joint_pipeline = _mod.run_joint_pipeline
        stage1_score_repellent = _mod.stage1_score_repellent
        stage2_variance_puct = _mod.stage2_variance_puct
        stage3_mcts_trajectory = _mod.stage3_mcts_trajectory
        stage4_replay_write = _mod.stage4_replay_write


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors the unit-test fixtures in
# :file:`molmetal/tests/test_persistent_mcts_tree.py`.
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Identity concatenation: state + tile → bigger state."""

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            return [
                MoleculeClosedTerm(
                    atoms=list(state.atoms) + list(tile.atoms),
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []


def _tiles() -> List[MoleculeClosedTerm]:
    return [
        MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
        for smi in ("CCO", "CCN", "CCCl", "CCS")
    ]


def _binding_site() -> BindingSite:
    return BindingSite(
        name="wf36_3stage_test", constraints=[], geometry_hints={},
    )


def _scorer(state: MoleculeClosedTerm) -> float:
    """Mock scorer: SMILES-hash + n_atoms — gives distinct leaves distinct scores."""
    try:
        n_atoms = int(getattr(state, "n_atoms", 0) or 0)
    except Exception:
        n_atoms = 0
    try:
        smi = str(state.canonical_smiles() or "")
    except Exception:
        smi = ""
    if not smi:
        try:
            smi = ",".join(
                sorted(str(a) for a in getattr(state, "atoms", []) or []),
            )
        except Exception:
            smi = ""
    smi_hash = float(abs(hash(smi)) % 1000) / 1000.0
    return float(
        min(1.0, 0.05 * float(n_atoms) + 0.30 * smi_hash + 0.05 * float(n_atoms % 7)),
    )


def _make_mcts(
    *,
    seed: int = 0,
    n_simulations: int = 5,
    strategy: str = "variance",
    c_var: float = 0.5,
) -> MCTSProofSearch:
    """Build a minimal :class:`MCTSProofSearch` for one strategy."""
    kwargs: Dict[str, Any] = dict(
        tile_library=_tiles(),
        rules={"identity": _IdentityRule()},
        target_predicates=[],
        binding_site=_binding_site(),
        scorer=_scorer,
        n_simulations=int(n_simulations),
        c_puct=1.4,
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,
        debug=False,
    )
    if strategy in ("score_repellent", "variance"):
        kwargs["exploration_strategy"] = strategy
    if strategy == "variance":
        kwargs["c_var"] = float(c_var)
    return MCTSProofSearch(**kwargs)


def _stub_mcts_report(
    *,
    candidates: int = 3,
    leaf_value_var: float = 0.04,
    best_score: float = 0.7,
    with_candidates: bool = True,
) -> Dict[str, Any]:
    """Build a stub Stage-3-like report dict."""
    out: Dict[str, Any] = {
        "candidates": (
            [_StubMol(f"CC{i}") for i in range(candidates)]
            if with_candidates else []
        ),
        "best_action": 0 if candidates > 0 else -1,
        "leaf_value_var": float(leaf_value_var),
        "leaf_value_std": float(math.sqrt(max(0.0, leaf_value_var))),
        "mean_candidate_score": float(best_score),
        "max_candidate_score": float(best_score),
        "mean_candidate_var": float(leaf_value_var * 0.5),
        "wall_time_s": 0.01,
        "n_states_explored": int(candidates),
        "best_score": float(best_score),
        "strategy": "variance",
        "seed": 0,
        "tree_statistics": {},
    }
    return out


class _StubMol:
    """Mock molecule with a canonical_smiles() method."""

    def __init__(self, smiles: str):
        self._smi = smiles

    def canonical_smiles(self) -> str:
        return self._smi

    def __repr__(self) -> str:
        return f"_StubMol({self._smi!r})"

    def __str__(self) -> str:
        return self._smi


# ===========================================================================
# TestJointPipelineImports (3 tests)
# ===========================================================================


class TestJointPipelineImports:
    def test_joint_imports(self) -> None:
        """All four pipeline stages + run_joint_pipeline importable."""
        assert callable(stage1_score_repellent)
        assert callable(stage2_variance_puct)
        assert callable(stage3_mcts_trajectory)
        assert callable(stage4_replay_write)
        assert callable(run_joint_pipeline)

    def test_joint_defaults(self) -> None:
        """Defaults match the Wf-36 Phase-3 brief."""
        assert isinstance(DEFAULT_N_TRAJECTORIES, int)
        assert isinstance(DEFAULT_N_ROUNDS, int)
        assert isinstance(DEFAULT_BUFFER_SIZE, int)
        assert isinstance(DEFAULT_N_ITERS, int)
        assert isinstance(DEFAULT_STEP_SIZE, float)
        # Reasonable ranges — sanity check, not exhaustive.
        assert DEFAULT_N_TRAJECTORIES > 0
        assert DEFAULT_N_ROUNDS > 0
        assert DEFAULT_BUFFER_SIZE > 0
        assert DEFAULT_N_ITERS >= 1
        assert 0.0 < DEFAULT_STEP_SIZE <= 1.0

    def test_joint_exploration_strategies_contract(self) -> None:
        """EXPLORATION_STRATEGIES includes ``variance``."""
        assert "variance" in EXPLORATION_STRATEGIES
        assert "score_repellent" in EXPLORATION_STRATEGIES
        assert "dirichlet" in EXPLORATION_STRATEGIES


# ===========================================================================
# TestStage1ScoreRepellent (2 tests)
# ===========================================================================


class TestStage1ScoreRepellent:
    def test_stage1_score_repellent(self) -> None:
        """Stage 1 returns samples on the simplex + variance bound."""
        report = stage1_score_repellent(
            priors=[0.5, 0.3, 0.15, 0.05],
            n_samples=8,
            step_size=0.1,
            n_iters=8,
            seed=42,
        )
        assert "samples" in report
        assert "chain_mean" in report
        assert "chain_std" in report
        assert "variance_bound_alpha" in report
        assert "wall_time_s" in report
        # 0 < variance_bound_alpha < 1 for the canonical parameters.
        assert 0.0 <= report["variance_bound_alpha"] < 1.0
        # chain_mean has the expected dimension.
        assert len(report["chain_mean"]) == 4
        # chain_std has the expected dimension.
        assert len(report["chain_std"]) == 4
        # wall_time_s is non-negative.
        assert report["wall_time_s"] >= 0.0

    def test_stage1_score_repellent_deterministic(self) -> None:
        """Same seed → same chain_mean."""
        rep_a = stage1_score_repellent(
            priors=[0.5, 0.3, 0.15, 0.05],
            n_samples=4,
            step_size=0.1,
            n_iters=4,
            seed=7,
        )
        rep_b = stage1_score_repellent(
            priors=[0.5, 0.3, 0.15, 0.05],
            n_samples=4,
            step_size=0.1,
            n_iters=4,
            seed=7,
        )
        assert rep_a["chain_mean"] == rep_b["chain_mean"]
        assert rep_a["chain_std"] == rep_b["chain_std"]


# ===========================================================================
# TestStage2VariancePuct (2 tests)
# ===========================================================================


class TestStage2VariancePuct:
    def test_stage2_variance_puct(self) -> None:
        """Stage 2 returns a valid action + PUCT scores."""
        report = stage2_variance_puct(
            root_smiles="CCO",
            n_actions=4,
            priors=[0.5, 0.3, 0.15, 0.05],
            c_puct=1.4,
            c_var=0.5,
            seed=0,
        )
        assert "selected_action" in report
        assert "puct_scores" in report
        assert "wall_time_s" in report
        assert "root_smiles" in report
        # Selected action is in range.
        assert 0 <= int(report["selected_action"]) < 4
        # PUCT scores list has the expected length.
        assert len(report["puct_scores"]) == 4
        # Wall-time is non-negative.
        assert report["wall_time_s"] >= 0.0
        # Root SMILES echoed back.
        assert report["root_smiles"] == "CCO"

    def test_stage2_variance_puct_handles_zero_prior(self) -> None:
        """Stage 2 handles degenerate (zero / uniform) priors."""
        report = stage2_variance_puct(
            root_smiles="CCN",
            n_actions=3,
            priors=[0.0, 0.0, 0.0],  # degenerate
            c_puct=1.0,
            c_var=0.5,
            seed=0,
        )
        assert 0 <= int(report["selected_action"]) < 3
        assert len(report["puct_scores"]) == 3


# ===========================================================================
# TestStage3MctsTrajectory (2 tests)
# ===========================================================================


class TestStage3MctsTrajectory:
    def test_stage3_mcts_trajectory(self) -> None:
        """Stage 3 returns a well-formed MCTS trajectory report."""
        report = stage3_mcts_trajectory(
            seed=0,
            n_simulations=4,
            max_depth=2,
            c_var=0.5,
            start_smiles="CCO",
            strategy="variance",
        )
        # Required keys.
        for k in [
            "wall_time_s", "leaf_value_var", "leaf_value_std",
            "best_action", "best_score", "n_states_explored",
            "candidates", "strategy", "seed", "tree_statistics",
        ]:
            assert k in report, f"missing key: {k}"
        assert report["strategy"] == "variance"
        assert report["seed"] == 0
        # Variance non-negative.
        assert report["leaf_value_var"] >= 0.0
        assert report["leaf_value_std"] >= 0.0
        assert report["wall_time_s"] >= 0.0

    def test_stage3_mcts_trajectory_handles_failure(self) -> None:
        """Stage 3 returns a well-formed report on failure."""
        # Force a failure by using an invalid start_smiles that the
        # MoleculeClosedTerm constructor cannot parse.
        report = stage3_mcts_trajectory(
            seed=0,
            n_simulations=4,
            max_depth=2,
            c_var=0.5,
            start_smiles="",  # invalid
            strategy="variance",
        )
        # Either it returns an error or it returns zeroed defaults —
        # both are valid contracts.
        assert "wall_time_s" in report
        if "error" in report:
            assert isinstance(report["error"], str)
        # Variance non-negative even on failure.
        assert report.get("leaf_value_var", 0.0) >= 0.0


# ===========================================================================
# TestStage4ReplayWrite (2 tests)
# ===========================================================================


class TestStage4ReplayWrite:
    def test_stage4_replay_write(self) -> None:
        """Stage 4 appends one entry per candidate to the buffer."""
        buffer = MCTSReplayBuffer(max_size=16, seed=0)
        mcts_report = _stub_mcts_report(candidates=4)
        out = stage4_replay_write(
            mcts_report=mcts_report,
            root_smiles="CCO",
            selected_action=1,
            buffer=buffer,
            seed=0,
        )
        assert "n_added" in out
        assert "mean_value" in out
        assert "mean_variance" in out
        assert "buffer_size" in out
        assert "wall_time_s" in out
        # Should have added 4 entries (one per candidate).
        assert out["n_added"] == 4
        assert out["buffer_size"] == 4
        assert len(buffer) == 4
        assert out["wall_time_s"] >= 0.0

    def test_stage4_replay_write_empty_candidates(self) -> None:
        """Stage 4 with no candidates adds zero entries."""
        buffer = MCTSReplayBuffer(max_size=16, seed=0)
        mcts_report = _stub_mcts_report(candidates=0, with_candidates=False)
        out = stage4_replay_write(
            mcts_report=mcts_report,
            root_smiles="CCO",
            selected_action=0,
            buffer=buffer,
            seed=0,
        )
        assert out["n_added"] == 0
        assert out["buffer_size"] == 0
        assert len(buffer) == 0


# ===========================================================================
# TestFullPipeline (4 tests)
# ===========================================================================


class TestFullPipeline:
    def test_full_pipeline_smoke(self) -> None:
        """Full pipeline runs end-to-end on a tiny smoke config."""
        report = run_joint_pipeline(
            n_trajectories=2,
            n_rounds=1,
            buffer_size=16,
            n_simulations=4,
            max_depth=2,
            seed=2025,
            start_smiles="CCO",
            step_size=0.1,
            n_iters=8,
            c_var=0.5,
            c_puct=1.4,
        )
        # Top-level keys.
        for k in [
            "config", "per_round", "aggregate",
            "replay_buffer_stats", "wall_time_s",
        ]:
            assert k in report, f"missing key: {k}"
        # Aggregate stats.
        agg = report["aggregate"]
        assert agg["n_runs"] == 2
        assert agg["success_rate"] > 0.0
        assert "stage_wall_time_s" in agg
        # Stage wall-time dict has all four stages.
        for stage_key in ("stage1", "stage2", "stage3", "stage4"):
            assert stage_key in agg["stage_wall_time_s"]
            assert agg["stage_wall_time_s"][stage_key] >= 0.0
        # Replay buffer stats.
        rbs = report["replay_buffer_stats"]
        assert rbs["size"] >= 0
        assert rbs["capacity"] == 16
        # Per-round records.
        assert len(report["per_round"]) == 1
        assert len(report["per_round"][0]["trajectories"]) == 2
        # Total wall-time.
        assert report["wall_time_s"] >= 0.0

    def test_full_pipeline_replay_buffer_populated(self) -> None:
        """Replay buffer records the per-trajectory variance."""
        report = run_joint_pipeline(
            n_trajectories=2,
            n_rounds=1,
            buffer_size=32,
            n_simulations=4,
            max_depth=2,
            seed=99,
            start_smiles="CCN",
            step_size=0.1,
            n_iters=4,
            c_var=0.5,
            c_puct=1.4,
        )
        rbs = report["replay_buffer_stats"]
        # We should have populated the buffer with at least one entry
        # per trajectory (the replay step emits one per candidate).
        assert rbs["size"] >= 1
        assert rbs["mean_variance"] >= 0.0
        assert rbs["mean_value"] >= 0.0

    def test_full_pipeline_deterministic(self) -> None:
        """Same seed produces identical aggregate metrics."""
        kwargs = dict(
            n_trajectories=2,
            n_rounds=1,
            buffer_size=16,
            n_simulations=4,
            max_depth=2,
            seed=2026,
            start_smiles="CCO",
            step_size=0.1,
            n_iters=4,
            c_var=0.5,
            c_puct=1.4,
        )
        rep_a = run_joint_pipeline(**kwargs)
        rep_b = run_joint_pipeline(**kwargs)
        # Aggregate stats match exactly (same seed).
        agg_a = rep_a["aggregate"]
        agg_b = rep_b["aggregate"]
        assert agg_a["n_runs"] == agg_b["n_runs"]
        assert agg_a["mean_best_score"] == agg_b["mean_best_score"]
        assert agg_a["mean_leaf_value_var"] == agg_b["mean_leaf_value_var"]
        # Replay buffer sizes match.
        assert rep_a["replay_buffer_stats"]["size"] == rep_b["replay_buffer_stats"]["size"]

    def test_full_pipeline_stage_wall_time_pct_sums_to_one(self) -> None:
        """Stage wall-time percentages sum to ≈ 1.0."""
        report = run_joint_pipeline(
            n_trajectories=2,
            n_rounds=1,
            buffer_size=16,
            n_simulations=4,
            max_depth=2,
            seed=2027,
            start_smiles="CCO",
            step_size=0.1,
            n_iters=4,
            c_var=0.5,
            c_puct=1.4,
        )
        pct = report["aggregate"]["stage_wall_time_pct"]
        total = sum(pct.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)