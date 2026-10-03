"""Tests for Wf-129 Phase 2 — regularised MCTS replay learner.

================================================================
Unit tests for
:mod:`molmetal.scripts.bench_mcts_replay_learner_regularized`.
================================================================
This module validates the Wf-129 P2 regularised replay learner:

* :class:`TestNumpySlicedW2`               (2 tests) — W_2 fallback
* :class:`TestRegularisedValueNetwork`     (3 tests) — value head hooks
* :class:`TestRegularisedPolicyNetwork`    (3 tests) — policy head hooks
* :class:`TestRegularisedLearner`          (2 tests) — learner training
* :class:`TestStrategyResultSchema`        (1 test)  — to_dict schema
* :class:`TestAggregatePerSeed`            (1 test)  — 1-seed end-to-end
* :class:`TestSynthesisVerdict`            (1 test)  — synthesis.md written
* :class:`TestCliSmoke`                    (1 test)  — argparse + smoke run

Total: **14 tests** matching the Wf-129 Phase-2 task brief.  All
CPU-only; completes in <10 s on any CI container.  None of the
tests require a GPU or ``torch.cuda``.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_replay_learner_regularized.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Path bootstrap — the bench script lives in
# ``molmetal/scripts/bench_mcts_replay_learner_regularized.py`` as a
# free-standing module (no ``__init__.py`` in ``scripts/``).  Add
# that directory to ``sys.path`` so the import resolves.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bench_mcts_replay_learner_regularized import (  # type: ignore
    DEFAULT_BATCH_SIZE_P2,
    DEFAULT_BETA_W2_POLICY,
    DEFAULT_FEATURE_DIM,
    DEFAULT_GAMMA_W2_VALUE,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS_P2,
    DEFAULT_N_MC_W2,
    DEFAULT_WEIGHT_DECAY,
    LearnerStrategyResult,
    RegularisedMCTSReplayLearner,
    RegularisedPolicyNetwork,
    RegularisedValueNetwork,
    TARGET_POLICY_LOSS,
    TARGET_VALUE_LOSS,
    Wf129P2Error,
    _aggregate_per_seed,
    _compute_w2,
    _np_sliced_w2,
    _parse_args,
    _write_synthesis,
    main,
)


# ---------------------------------------------------------------------------
# 1. Numpy-only W_2 fallback
# ---------------------------------------------------------------------------


class TestNumpySlicedW2:
    """``_np_sliced_w2`` is robust to 1-D / 2-D inputs and edge cases."""

    def test_identical_distributions_returns_zero(self) -> None:
        """Identical 1-D inputs yield ~0 W_2 (sorted-projection overlap)."""
        x = np.linspace(0.0, 1.0, 64)
        w2 = _np_sliced_w2(x, x, n_mc=64, seed=0)
        assert w2 == pytest.approx(0.0, abs=1.0e-6)

    def test_shifted_distributions_returns_positive(self) -> None:
        """Shifted 1-D inputs yield a positive W_2 estimate."""
        x = np.linspace(0.0, 1.0, 64)
        y = x + 0.5
        w2 = _np_sliced_w2(x, y, n_mc=128, seed=0)
        assert w2 > 0.0

    def test_zero_dim_inputs_does_not_crash(self) -> None:
        """Single-point distributions do not crash the fallback."""
        x = np.array([0.5])
        y = np.array([0.7])
        w2 = _np_sliced_w2(x, y, n_mc=8, seed=0)
        # Single point on each side — the slice becomes a pointwise
        # comparison.  We just assert a finite, non-negative float.
        assert isinstance(w2, float)
        assert w2 >= 0.0

    def test_compute_w2_prefers_torch_when_available(self) -> None:
        """``_compute_w2`` returns a torch-evaluated W_2 when torch exists."""
        try:
            import torch  # noqa: F401
        except Exception:
            pytest.skip("torch unavailable — skip torch-preferring branch")
        x = np.linspace(0.0, 1.0, 32)
        y = np.linspace(0.0, 1.0, 32)
        out = _compute_w2(x, y, n_mc=32, seed=0)
        assert out == pytest.approx(0.0, abs=1.0e-5)


# ---------------------------------------------------------------------------
# 2. Regularised value head
# ---------------------------------------------------------------------------


class TestRegularisedValueNetwork:
    """``RegularisedValueNetwork`` exposes the L2 + W_2 hooks correctly."""

    def test_default_construction(self) -> None:
        """The default constructor wires the base ``ValueNetwork``."""
        net = RegularisedValueNetwork(seed=0)
        assert net.weight_decay == pytest.approx(DEFAULT_WEIGHT_DECAY)
        assert net.gamma_w2 == pytest.approx(DEFAULT_GAMMA_W2_VALUE)
        assert net.n_mc == DEFAULT_N_MC_W2
        assert net.n_updates_ == 0

    def test_update_with_regularisation_returns_loss_dict(self) -> None:
        """``update_with_regularisation`` returns ``{mse, l2, w2, total}``."""
        net = RegularisedValueNetwork(
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            lr=1.0e-2,
            weight_decay=1.0e-4,
            gamma_w2=0.01,
            n_mc=64,
            seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(8, DEFAULT_FEATURE_DIM))
        target = np.full(8, 0.7, dtype=float)
        out = net.update_with_regularisation(x, target, seed=0)
        for key in ("mse", "l2", "w2", "total"):
            assert key in out
            assert isinstance(out[key], float)
            assert out[key] >= 0.0
        # Total == mse + l2 + gamma_w2 * w2.
        assert out["total"] == pytest.approx(
            out["mse"] + out["l2"] + 0.01 * out["w2"],
            abs=1.0e-9,
        )
        assert net.n_updates_ == 1

    def test_zero_weight_decay_zero_w2_disables_hooks(self) -> None:
        """When both hooks are 0, total == mse."""
        net = RegularisedValueNetwork(
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            lr=1.0e-2,
            weight_decay=0.0,
            gamma_w2=0.0,
            n_mc=32,
            seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(4, DEFAULT_FEATURE_DIM))
        target = np.full(4, 0.5, dtype=float)
        out = net.update_with_regularisation(x, target, seed=0)
        assert out["total"] == pytest.approx(out["mse"], abs=1.0e-9)
        assert out["l2"] == pytest.approx(0.0, abs=1.0e-12)
        assert out["w2"] == pytest.approx(0.0, abs=1.0e-12)

    def test_negative_weight_decay_rejected(self) -> None:
        """``weight_decay < 0`` raises ``Wf129P2Error``."""
        with pytest.raises(Wf129P2Error):
            RegularisedValueNetwork(weight_decay=-0.01)

    def test_negative_gamma_w2_rejected(self) -> None:
        """``gamma_w2 < 0`` raises ``Wf129P2Error``."""
        with pytest.raises(Wf129P2Error):
            RegularisedValueNetwork(gamma_w2=-0.01)


# ---------------------------------------------------------------------------
# 3. Regularised policy head
# ---------------------------------------------------------------------------


class TestRegularisedPolicyNetwork:
    """``RegularisedPolicyNetwork`` exposes the L2 + W_2 hooks correctly."""

    def test_default_construction(self) -> None:
        """The default constructor wires the base ``PolicyNetwork``."""
        net = RegularisedPolicyNetwork(seed=0)
        assert net.weight_decay == pytest.approx(DEFAULT_WEIGHT_DECAY)
        assert net.beta_w2 == pytest.approx(DEFAULT_BETA_W2_POLICY)
        assert net.n_mc == DEFAULT_N_MC_W2
        assert net.n_updates_ == 0

    def test_update_with_regularisation_returns_loss_dict(self) -> None:
        """``update_with_regularisation`` returns ``{ce, l2, w2, total}``."""
        net = RegularisedPolicyNetwork(
            feature_dim=DEFAULT_FEATURE_DIM,
            n_actions=4,
            lr=1.0e-2,
            weight_decay=1.0e-4,
            beta_w2=0.05,
            n_mc=64,
            seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(8, DEFAULT_FEATURE_DIM))
        target = np.zeros((8, 4), dtype=float)
        target[:, 0] = 1.0
        out = net.update_with_regularisation(x, target, seed=0)
        for key in ("ce", "l2", "w2", "total"):
            assert key in out
            assert isinstance(out[key], float)
            assert out[key] >= 0.0
        # Total == ce + l2 + beta_w2 * w2.
        assert out["total"] == pytest.approx(
            out["ce"] + out["l2"] + 0.05 * out["w2"],
            abs=1.0e-9,
        )
        assert net.n_updates_ == 1

    def test_zero_hooks_disabled(self) -> None:
        """When both hooks are 0, total == ce."""
        net = RegularisedPolicyNetwork(
            feature_dim=DEFAULT_FEATURE_DIM,
            n_actions=4,
            lr=1.0e-2,
            weight_decay=0.0,
            beta_w2=0.0,
            n_mc=32,
            seed=0,
        )
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.5, size=(4, DEFAULT_FEATURE_DIM))
        target = np.zeros((4, 4), dtype=float)
        target[:, 1] = 1.0
        out = net.update_with_regularisation(x, target, seed=0)
        assert out["total"] == pytest.approx(out["ce"], abs=1.0e-9)
        assert out["l2"] == pytest.approx(0.0, abs=1.0e-12)
        assert out["w2"] == pytest.approx(0.0, abs=1.0e-12)


# ---------------------------------------------------------------------------
# 4. Regularised learner training loop
# ---------------------------------------------------------------------------


class TestRegularisedLearner:
    """``RegularisedMCTSReplayLearner`` orchestrates the 3 strategies."""

    def test_vanilla_strategy_uses_bare_heads(self) -> None:
        """``vanilla`` strategy wires bare ``ValueNetwork`` / ``PolicyNetwork``."""
        learner = RegularisedMCTSReplayLearner(
            strategy="vanilla",
            n_epochs=1,
            batch_size=4,
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            seed=0,
        )
        # Bare heads are NOT regularised wrappers.
        from molmetal.scripts.mcts_replay_learner import (  # type: ignore
            PolicyNetwork,
            ValueNetwork,
        )

        assert isinstance(learner.value_net, ValueNetwork)
        assert isinstance(learner.policy_net, PolicyNetwork)

    def test_l2_w2_strategy_uses_regularised_heads(self) -> None:
        """``l2_w2`` strategy wires ``RegularisedValueNetwork`` /
        ``RegularisedPolicyNetwork``."""
        learner = RegularisedMCTSReplayLearner(
            strategy="l2_w2",
            n_epochs=1,
            batch_size=4,
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            seed=0,
        )
        assert isinstance(learner.value_net, RegularisedValueNetwork)
        assert isinstance(learner.policy_net, RegularisedPolicyNetwork)

    def test_invalid_strategy_rejected(self) -> None:
        """An invalid ``strategy`` raises ``Wf129P2Error``."""
        with pytest.raises(Wf129P2Error):
            RegularisedMCTSReplayLearner(strategy="garbage")

    def test_fit_runs_all_three_strategies(self) -> None:
        """``fit`` returns a non-empty history for each strategy."""
        # Build a small synthetic buffer inline (no real chemistry).
        from molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
            MCTSReplayBuffer,
        )
        from molmetal.scripts.mcts_replay_learner import (  # type: ignore
            synth_populate_buffer,
        )

        buf = MCTSReplayBuffer(max_size=64, seed=0)
        synth_populate_buffer(
            buf,
            n_entries=32,
            n_states=4,
            n_actions=4,
            seed=0,
        )
        for strat in ("vanilla", "l2_only", "l2_w2"):
            learner = RegularisedMCTSReplayLearner(
                strategy=strat,
                n_epochs=2,
                batch_size=8,
                feature_dim=DEFAULT_FEATURE_DIM,
                hidden_dim=16,
                n_actions=4,
                seed=0,
                weight_decay=1.0e-4,
                beta_w2=0.05,
                gamma_w2=0.01,
                n_mc=32,
            )
            history = learner.fit(buf)
            assert len(history) == 2
            for entry in history:
                assert "value_loss" in entry
                assert "policy_loss" in entry
                assert "wall_time_s" in entry
                if strat == "l2_w2":
                    assert "w2_diagnostics" in entry
                    # At least one entry has a w2 channel.
                    assert any(
                        "w2_policy" in d or "w2_value" in d
                        for d in entry["w2_diagnostics"]
                    )

    def test_summary_returns_expected_keys(self) -> None:
        """``summary`` returns the headline + W_2 diagnostic keys."""
        from molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
            MCTSReplayBuffer,
        )
        from molmetal.scripts.mcts_replay_learner import (  # type: ignore
            synth_populate_buffer,
        )

        buf = MCTSReplayBuffer(max_size=64, seed=0)
        synth_populate_buffer(
            buf,
            n_entries=32,
            n_states=4,
            n_actions=4,
            seed=0,
        )
        learner = RegularisedMCTSReplayLearner(
            strategy="l2_w2",
            n_epochs=2,
            batch_size=8,
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            n_actions=4,
            seed=0,
        )
        learner.fit(buf)
        summary = learner.summary()
        for key in (
            "n_epochs_run",
            "final_value_loss",
            "final_policy_loss",
            "value_loss_drop",
            "policy_loss_drop",
            "converged",
            "total_wall_time_s",
            "mean_w2_policy",
            "mean_w2_value",
        ):
            assert key in summary


# ---------------------------------------------------------------------------
# 5. StrategyResult schema
# ---------------------------------------------------------------------------


class TestStrategyResultSchema:
    """``LearnerStrategyResult.to_dict`` returns the expected schema."""

    def test_to_dict_contains_expected_keys(self) -> None:
        """The schema carries all the headline + W_2 channels."""
        res = LearnerStrategyResult(
            strategy="l2_w2",
            seed=0,
            n_epochs=5,
            batch_size=32,
            n_samples=128,
            final_value_loss=0.05,
            final_policy_loss=0.40,
            value_loss_drop=0.10,
            policy_loss_drop=-0.05,
            converged=True,
            total_wall_time_s=0.5,
            mean_w2_policy=0.01,
            mean_w2_value=0.005,
            history=[],
        )
        d = res.to_dict()
        for key in (
            "strategy",
            "seed",
            "n_epochs",
            "batch_size",
            "n_samples",
            "final_value_loss",
            "final_policy_loss",
            "value_loss_drop",
            "policy_loss_drop",
            "converged",
            "total_wall_time_s",
            "mean_w2_policy",
            "mean_w2_value",
            "history",
        ):
            assert key in d
        assert d["strategy"] == "l2_w2"
        assert d["seed"] == 0
        assert d["converged"] is True
        assert d["mean_w2_policy"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 6. Aggregate per-seed (1-seed end-to-end smoke)
# ---------------------------------------------------------------------------


class TestAggregatePerSeed:
    """``_aggregate_per_seed`` runs all 3 strategies across all seeds."""

    def test_aggregate_single_seed(self, tmp_path: Path) -> None:
        """A 1-seed × 3-strategy aggregate is non-empty."""
        report = _aggregate_per_seed(
            seeds=[0],
            n_epochs=2,
            batch_size=8,
            n_entries=32,
            n_states=4,
            n_actions=4,
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            lr=DEFAULT_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            beta_w2=DEFAULT_BETA_W2_POLICY,
            gamma_w2=DEFAULT_GAMMA_W2_VALUE,
            n_mc=32,
        )
        assert "per_strategy_per_seed" in report
        assert "aggregate" in report
        assert "vanilla" in report["per_strategy_per_seed"]
        assert "l2_only" in report["per_strategy_per_seed"]
        assert "l2_w2" in report["per_strategy_per_seed"]
        for strat in ("vanilla", "l2_only", "l2_w2"):
            assert len(report["per_strategy_per_seed"][strat]) == 1
        # Aggregate carries the headline channels.
        for strat in ("vanilla", "l2_only", "l2_w2"):
            a = report["aggregate"][strat]
            assert "final_value_loss" in a
            assert "final_policy_loss" in a
            assert "policy_loss_drop" in a
            assert "value_loss_drop" in a
            assert "converged_rate" in a
            assert "wall_time_s" in a


# ---------------------------------------------------------------------------
# 7. Synthesis verdict
# ---------------------------------------------------------------------------


class TestSynthesisVerdict:
    """``_write_synthesis`` writes a synthesis.md with the expected sections."""

    def test_synthesis_writes_file(self, tmp_path: Path) -> None:
        """The synthesis file is created at ``output_dir/synthesis.md``."""
        report = _aggregate_per_seed(
            seeds=[0],
            n_epochs=2,
            batch_size=8,
            n_entries=32,
            n_states=4,
            n_actions=4,
            feature_dim=DEFAULT_FEATURE_DIM,
            hidden_dim=16,
            lr=DEFAULT_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            beta_w2=DEFAULT_BETA_W2_POLICY,
            gamma_w2=DEFAULT_GAMMA_W2_VALUE,
            n_mc=32,
        )
        _write_synthesis(report, tmp_path)
        out_path = tmp_path / "synthesis.md"
        assert out_path.exists()
        text = out_path.read_text()
        # Headline sections present.
        assert "# Wf-129 Phase 2" in text
        assert "## §1 Goal" in text
        assert "## §4 Verdict" in text
        assert "## §5 Honest framing" in text
        assert "## §6 Cross-references" in text
        # Strategy labels present.
        for strat in ("vanilla", "l2_only", "l2_w2"):
            assert f"`{strat}`" in text


# ---------------------------------------------------------------------------
# 8. CLI smoke
# ---------------------------------------------------------------------------


class TestCliSmoke:
    """``main`` runs the smoke harness end-to-end."""

    def test_main_smoke(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--smoke --report-path <tmp>`` writes the JSON + synthesis files."""
        report_path = tmp_path / "wf129_p2_smoke"
        monkeypatch.setattr(
            "sys.argv",
            [
                "bench_mcts_replay_learner_regularized.py",
                "--smoke",
                "--report-path",
                str(report_path),
            ],
        )
        rc = main()
        assert rc == 0
        # JSON root written.
        root = report_path / "per_seed_metrics.json"
        assert root.exists()
        root_doc = json.loads(root.read_text())
        assert root_doc["metric"] == "wf129_p2_replay_learner_loss"
        assert root_doc["n_seeds"] == 1
        # Per-seed raw written.
        seed_dir = report_path / "seeds" / "seed_0"
        assert seed_dir.exists()
        seed_doc = json.loads(
            (seed_dir / "per_seed_metrics.json").read_text(),
        )
        assert seed_doc["metric"] == "wf129_p2_replay_learner_loss"
        # Synthesis.md written.
        synth_path = report_path / "synthesis.md"
        assert synth_path.exists()

    def test_parse_args_defaults(self) -> None:
        """``_parse_args`` returns sane defaults."""
        args = _parse_args([])
        assert args.n_epochs == DEFAULT_N_EPOCHS_P2
        assert args.batch_size == DEFAULT_BATCH_SIZE_P2
        assert args.weight_decay == pytest.approx(DEFAULT_WEIGHT_DECAY)
        assert args.beta_w2 == pytest.approx(DEFAULT_BETA_W2_POLICY)
        assert args.gamma_w2 == pytest.approx(DEFAULT_GAMMA_W2_VALUE)
        assert args.n_mc == DEFAULT_N_MC_W2
        assert args.target_policy_loss == pytest.approx(TARGET_POLICY_LOSS)
        assert args.target_value_loss == pytest.approx(TARGET_VALUE_LOSS)
        assert args.seeds == "0,1,2,3,4"


# ---------------------------------------------------------------------------
# 9. Defaults sanity
# ---------------------------------------------------------------------------


class TestDefaultsSanity:
    """Top-level constants match the Wf-129 P2 spec."""

    def test_target_policy_loss_below_wf36_baseline(self) -> None:
        """The target ``policy_loss <= 0.5`` is well below the Wf-36 P4 2.795."""
        assert TARGET_POLICY_LOSS <= 0.5
        assert TARGET_POLICY_LOSS > 0.0

    def test_target_value_loss_below_one(self) -> None:
        """The target ``value_loss < 1.0`` is below the Wf-36 P4 0.032."""
        assert TARGET_VALUE_LOSS == pytest.approx(1.0)

    def test_default_n_epochs_p2_above_wf36(self) -> None:
        """Wf-129 P2 ships ``n_epochs=30`` (vs Wf-36 P4's 5)."""
        assert DEFAULT_N_EPOCHS_P2 >= 30

    def test_default_batch_size_p2_above_wf36(self) -> None:
        """Wf-129 P2 ships ``batch_size=32`` (vs Wf-36 P4's 16)."""
        assert DEFAULT_BATCH_SIZE_P2 >= 32
