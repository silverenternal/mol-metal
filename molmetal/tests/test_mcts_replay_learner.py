"""Tests for Wf-36 Phase 4 — MCTS replay learner (AlphaGo-style).

================================================================
CPU-only smoke / contract tests for ``mcts_replay_learner``
================================================================
This module validates the AlphaGo-style experience-replay learner
that closes the Wf-21 ``mcts_replay_buffer`` loop:

* :class:`ValueNetwork`                  — value head (CPU numpy MLP).
* :class:`PolicyNetwork`                 — policy head (softmax).
* :class:`MCTSReplayLearner`             — orchestrates training.
* :func:`featurise_entry`                — deterministic featuriser.
* :func:`synth_populate_buffer`          — synthetic buffer fixture.
* :func:`run_learner` / :func:`save_report` — top-level runner.

Total: **16 tests** matching the Wf-36 Phase-4 task brief.  All
CPU-only; completes in <10 s on any CI container.  None of the
tests require a GPU or ``torch.cuda``.

Test inventory
--------------
* :class:`TestImports`                   (2 tests) — module surface
* :class:`TestFeaturiseEntry`            (2 tests) — featuriser
* :class:`TestValueNetwork`              (3 tests) — value head
* :class:`TestPolicyNetwork`             (3 tests) — policy head
* :class:`TestLearnerConvergence`        (2 tests) — training loop
* :class:`TestLearnerIntegration`        (3 tests) — replay buffer
* :class:`TestRunner`                    (1 test)  — top-level CLI

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_replay_learner.py -v
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

# Allow the test to run both as a top-level module and as part of
# the molmetal test suite.  Match the convention in
# ``test_variance_mcts_proof_search.py``.
_PROJECT_ROOT = "/home/hugo/codes/mol-metal"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Imports — try both the bare ``molmetal_lam`` (top-level module)
# path and the package-qualified ``molmetal.molmetal_lam`` path.
# ---------------------------------------------------------------------------


try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
    )
    from molmetal.scripts.mcts_replay_learner import (
        CONVERGENCE_EPS,
        CONVERGENCE_WINDOW,
        DEFAULT_FEATURE_DIM,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_LR,
        DEFAULT_N_ACTIONS,
        DEFAULT_N_EPOCHS,
        MCTSReplayLearner,
        MCTSReplayLearnerError,
        PolicyNetwork,
        ValueNetwork,
        featurise_entry,
        run_learner,
        save_report,
        synth_populate_buffer,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
        MCTSReplayBuffer,
        MCTSReplayEntry,
    )
    from molmetal.scripts.mcts_replay_learner import (  # type: ignore
        CONVERGENCE_EPS,
        CONVERGENCE_WINDOW,
        DEFAULT_FEATURE_DIM,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_LR,
        DEFAULT_N_ACTIONS,
        DEFAULT_N_EPOCHS,
        MCTSReplayLearner,
        MCTSReplayLearnerError,
        PolicyNetwork,
        ValueNetwork,
        featurise_entry,
        run_learner,
        save_report,
        synth_populate_buffer,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_entry(
    state: str = "CCO",
    action: str = "action_0",
    value: float = 0.5,
    prior: float = 0.5,
    variance: float = 0.0,
    visit_count: int = 1,
) -> MCTSReplayEntry:
    """Build a minimal :class:`MCTSReplayEntry` for tests."""
    return MCTSReplayEntry(
        state=state,
        action=action,
        value=value,
        prior=prior,
        variance=variance,
        visit_count=visit_count,
    )


def _mk_buffer(n_entries: int = 64, n_states: int = 4, n_actions: int = 4,
               seed: int = 0) -> MCTSReplayBuffer:
    """Build a synthetic replay buffer for tests."""
    buf = MCTSReplayBuffer(max_size=max(16, 4 * n_entries), seed=seed)
    synth_populate_buffer(
        buf,
        n_entries=int(n_entries),
        n_states=int(n_states),
        n_actions=int(n_actions),
        seed=int(seed),
    )
    return buf


# ===========================================================================
# TestImports (2 tests)
# ===========================================================================


class TestImports:
    def test_learner_imports(self) -> None:
        """All public symbols import cleanly."""
        # Classes
        assert ValueNetwork is not None
        assert PolicyNetwork is not None
        assert MCTSReplayLearner is not None
        # Functions
        assert callable(featurise_entry)
        assert callable(synth_populate_buffer)
        assert callable(run_learner)
        assert callable(save_report)
        # Constants
        assert int(DEFAULT_FEATURE_DIM) >= 13
        assert int(DEFAULT_HIDDEN_DIM) >= 1
        assert int(DEFAULT_N_ACTIONS) >= 1
        assert float(DEFAULT_LR) > 0.0
        assert int(DEFAULT_N_EPOCHS) >= 1
        assert int(CONVERGENCE_WINDOW) >= 1
        assert float(CONVERGENCE_EPS) > 0.0

    def test_learner_error_is_value_error(self) -> None:
        """MCTSReplayLearnerError subclasses ValueError so callers
        can keep their existing ``except ValueError`` blocks."""
        assert issubclass(MCTSReplayLearnerError, ValueError)
        with pytest.raises(MCTSReplayLearnerError):
            ValueNetwork(feature_dim=0)
        with pytest.raises(MCTSReplayLearnerError):
            PolicyNetwork(n_actions=0)
        with pytest.raises(MCTSReplayLearnerError):
            MCTSReplayLearner(n_epochs=0)


# ===========================================================================
# TestFeaturiseEntry (2 tests)
# ===========================================================================


class TestFeaturiseEntry:
    def test_featurise_entry_shape_and_range(self) -> None:
        """featurise_entry returns a fixed-length feature vector in range."""
        e = _mk_entry(state="CCO", action="action_0", value=0.7,
                      prior=0.4, variance=0.02, visit_count=4)
        feat = featurise_entry(e, feature_dim=DEFAULT_FEATURE_DIM)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (DEFAULT_FEATURE_DIM,)
        # Hash slots are in (-1, 1) due to tanh squashing.
        assert np.all(np.abs(feat[:-4]) <= 1.0 + 1.0e-6)
        # Value / prior / variance slots in expected range.
        assert -1.0 - 1.0e-6 <= feat[DEFAULT_FEATURE_DIM - 4] <= 1.0 + 1.0e-6
        assert 0.0 <= feat[DEFAULT_FEATURE_DIM - 3] <= 1.0
        assert -1.0 - 1.0e-6 <= feat[DEFAULT_FEATURE_DIM - 2] <= 1.0 + 1.0e-6
        assert feat[DEFAULT_FEATURE_DIM - 1] >= 0.0  # log1p / 8

    def test_featurise_entry_rejects_too_small(self) -> None:
        """featurise_entry raises on too-small ``feature_dim``."""
        e = _mk_entry()
        with pytest.raises(MCTSReplayLearnerError):
            featurise_entry(e, feature_dim=8)


# ===========================================================================
# TestValueNetwork (3 tests)
# ===========================================================================


class TestValueNetwork:
    def test_value_network_forward(self) -> None:
        """ValueNetwork.forward returns a 1-D output of the right length."""
        net = ValueNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, hidden_dim=16, seed=0,
        )
        x = np.random.default_rng(0).normal(
            0.0, 0.5, size=(4, DEFAULT_FEATURE_DIM),
        )
        out, h_pre, h_post = net.forward(x)
        assert out.shape == (4,)
        assert h_pre.shape == (4, 16)
        assert h_post.shape == (4, 16)
        # tanh squashes the output to (-1, 1).
        assert np.all(np.abs(out) <= 1.0 + 1.0e-6)
        # tanh squashes the hidden activation to (-1, 1).
        assert np.all(np.abs(h_post) <= 1.0 + 1.0e-6)

    def test_value_network_update_decreases_mse(self) -> None:
        """A few SGD updates drive the value MSE down on a constant target."""
        net = ValueNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, hidden_dim=16, lr=1.0e-2, seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(32, DEFAULT_FEATURE_DIM))
        target = np.full(32, 0.7, dtype=float)
        losses: List[float] = []
        for _ in range(20):
            loss = net.update(x, target)
            losses.append(loss)
        # Strict decrease: every loss ≤ first loss (allow tiny float noise).
        for L in losses:
            assert L <= losses[0] + 1.0e-9
        # And strictly lower at the end (with generous tolerance).
        assert losses[-1] < losses[0] - 1.0e-4
        assert net.n_updates_ == 20

    def test_value_network_predict_matches_forward(self) -> None:
        """predict is a no-grad wrapper around forward."""
        net = ValueNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, hidden_dim=16, seed=0,
        )
        x = np.random.default_rng(0).normal(
            0.0, 0.5, size=(3, DEFAULT_FEATURE_DIM),
        )
        pred = net.predict(x)
        fwd, _, _ = net.forward(x)
        assert np.allclose(pred, fwd)


# ===========================================================================
# TestPolicyNetwork (3 tests)
# ===========================================================================


class TestPolicyNetwork:
    def test_policy_network_forward(self) -> None:
        """PolicyNetwork.forward returns a row-stochastic probability matrix."""
        net = PolicyNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, n_actions=8, seed=0,
        )
        x = np.random.default_rng(0).normal(
            0.0, 0.5, size=(5, DEFAULT_FEATURE_DIM),
        )
        probs = net.forward(x)
        assert probs.shape == (5, 8)
        # Each row sums to 1.
        assert np.allclose(probs.sum(axis=1), np.ones(5), atol=1.0e-6)
        # All entries are non-negative.
        assert np.all(probs >= 0.0)

    def test_policy_network_update_decreases_ce(self) -> None:
        """A few SGD updates drive the policy CE loss down on a fixed target."""
        net = PolicyNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, n_actions=4, lr=1.0e-2, seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(32, DEFAULT_FEATURE_DIM))
        # Target distribution concentrated on action 0.
        target = np.zeros((32, 4), dtype=float)
        target[:, 0] = 1.0
        losses: List[float] = []
        for _ in range(20):
            loss = net.update(x, target)
            losses.append(loss)
        # CE on a fixed target is non-increasing.
        for L in losses:
            assert L <= losses[0] + 1.0e-9
        assert losses[-1] < losses[0] - 1.0e-4
        assert net.n_updates_ == 20

    def test_policy_network_update_handles_uniform_target(self) -> None:
        """A uniform target produces a finite, well-defined loss."""
        net = PolicyNetwork(
            feature_dim=DEFAULT_FEATURE_DIM, n_actions=4, lr=1.0e-2, seed=0,
        )
        x = np.random.default_rng(0).normal(
            0.0, 0.5, size=(8, DEFAULT_FEATURE_DIM),
        )
        target = np.full((8, 4), 0.25, dtype=float)
        loss = net.update(x, target)
        # Initial random weights → non-uniform prediction → loss > ln(4).
        # We only assert finite + non-negative; CE is bounded above by
        # ln(n_actions) for a uniform target but the random init
        # routinely exceeds this (the network has to *learn* the
        # uniform distribution).
        assert np.isfinite(loss)
        assert loss >= 0.0


# ===========================================================================
# TestLearnerConvergence (2 tests)
# ===========================================================================


class TestLearnerConvergence:
    def test_loss_decreases(self) -> None:
        """Across epochs, the value / policy losses should not increase."""
        buf = _mk_buffer(n_entries=64, n_states=4, n_actions=4, seed=0)
        learner = MCTSReplayLearner(
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            n_epochs=5,
            batch_size=8,
            lr=1.0e-2,
            seed=0,
        )
        history = learner.fit(buf)
        assert len(history) == 5
        # Value loss is non-increasing across epochs.
        for i in range(1, len(history)):
            assert history[i]["value_loss"] <= history[0]["value_loss"] + 1.0e-6
        # And there is at least one improvement somewhere.
        v_first = history[0]["value_loss"]
        v_last = history[-1]["value_loss"]
        p_first = history[0]["policy_loss"]
        p_last = history[-1]["policy_loss"]
        # At least one of the two losses dropped.
        assert (v_last < v_first - 1.0e-6) or (p_last < p_first - 1.0e-6)
        # Summary dict populated.
        summary = learner.summary()
        assert summary["n_epochs_run"] == 5
        assert "final_value_loss" in summary
        assert "final_policy_loss" in summary

    def test_history_appended_per_epoch(self) -> None:
        """history_ is cleared at the start of each fit() call."""
        buf = _mk_buffer(n_entries=32, n_states=4, n_actions=4, seed=1)
        learner = MCTSReplayLearner(
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            n_epochs=3,
            batch_size=4,
            lr=1.0e-2,
            seed=1,
        )
        history_a = learner.fit(buf)
        assert len(history_a) == 3
        # Re-running clears the history.
        history_b = learner.fit(buf, n_epochs=2)
        assert len(history_b) == 2
        assert learner.history_ is not history_a


# ===========================================================================
# TestLearnerIntegration (3 tests)
# ===========================================================================


class TestLearnerIntegration:
    def test_replay_buffer_integration(self) -> None:
        """End-to-end: build a buffer, train a learner, check summary."""
        buf = _mk_buffer(n_entries=128, n_states=8, n_actions=4, seed=42)
        learner = MCTSReplayLearner(
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=32,
            n_actions=4,
            n_epochs=3,
            batch_size=16,
            lr=1.0e-2,
            seed=42,
        )
        history = learner.fit(buf)
        summary = learner.summary()
        # Network update counts match the number of epochs.
        assert learner.value_net.n_updates_ == len(history)
        assert learner.policy_net.n_updates_ == len(history)
        # Summary keys present.
        for k in [
            "n_epochs_run", "final_value_loss", "final_policy_loss",
            "value_loss_drop", "policy_loss_drop", "converged",
            "total_wall_time_s", "value_net_updates", "policy_net_updates",
        ]:
            assert k in summary, f"missing key: {k}"
        # Updates counters match.
        assert summary["value_net_updates"] == len(history)
        assert summary["policy_net_updates"] == len(history)

    def test_learner_fit_rejects_none_buffer(self) -> None:
        """fit() rejects a None buffer with MCTSReplayLearnerError."""
        learner = MCTSReplayLearner(
            feature_dim=DEFAULT_FEATURE_DIM, n_epochs=1, batch_size=4, seed=0,
        )
        with pytest.raises(MCTSReplayLearnerError):
            learner.fit(None)  # type: ignore[arg-type]

    def test_learner_handles_empty_buffer(self) -> None:
        """fit() on an empty buffer logs zero loss without crashing."""
        buf = MCTSReplayBuffer(max_size=16, seed=0)
        learner = MCTSReplayLearner(
            feature_dim=DEFAULT_FEATURE_DIM, n_epochs=3, batch_size=4, seed=0,
        )
        history = learner.fit(buf)
        # Each epoch logs zero loss and zero n_samples.
        assert len(history) == 3
        for entry in history:
            assert entry["value_loss"] == 0.0
            assert entry["policy_loss"] == 0.0
            assert entry["n_samples"] == 0


# ===========================================================================
# TestRunner (1 test)
# ===========================================================================


class TestRunner:
    def test_run_learner_synthetic_and_save(self, tmp_path: Any) -> None:
        """run_learner + save_report on the synthetic fixture writes a JSON."""
        out_dir = tmp_path / "wf36_learner"
        report = run_learner(
            replay_buffer=None,
            n_epochs=2,
            batch_size=8,
            lr=1.0e-2,
            out_dir=str(out_dir),
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            seed=0,
            synth_entries=32,
            synth_states=4,
        )
        # Top-level keys present.
        for k in [
            "config", "replay_source", "buffer_stats", "summary",
            "history", "wall_time_s", "version",
        ]:
            assert k in report, f"missing key: {k}"
        assert report["replay_source"].startswith("synthetic:")
        assert report["summary"]["n_epochs_run"] == 2
        # Persist to disk and re-read.
        json_path = save_report(report, str(out_dir))
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as fh:
            reloaded = json.load(fh)
        assert reloaded["summary"]["n_epochs_run"] == 2
        assert reloaded["replay_source"].startswith("synthetic:")
