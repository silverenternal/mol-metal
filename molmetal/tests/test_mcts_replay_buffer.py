"""Tests for the MCTS replay buffer layer (TODO-31).

================================================================
CPU-only smoke / contract tests for ``mcts_replay_buffer``
================================================================
This module validates the four classes that comprise the
AlphaGo-style experience-replay layer:

* :class:`MCTSReplayEntry`                            (3 tests)
* :class:`MCTSReplayBuffer`                           (8 tests)
* :class:`MCTSReplayWrapper`                          (7 tests)
* :class:`ReplayGuidedMCTS`                           (7 tests)

Total: **25 tests** matching the task brief.  All CPU-only; completes
in <5 s on any CI container.  None of the tests mutate
:mod:`proof_search` / :mod:`score_repellent_mc` /
:mod:`variance_puct_integration` / :mod:`variance_mcts_integration`.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_replay_buffer.py -v
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_SIZE,
        DEFAULT_REPLAY_ALPHA,
        DEFAULT_TAU,
        MAX_BATCH_SIZE,
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
        MCTSReplayWrapper,
        ReplayGuidedMCTS,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore  # noqa: E402
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_SIZE,
        DEFAULT_REPLAY_ALPHA,
        DEFAULT_TAU,
        MAX_BATCH_SIZE,
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
        MCTSReplayWrapper,
        ReplayGuidedMCTS,
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
    """Build a minimal MCTSReplayEntry for tests."""
    return MCTSReplayEntry(
        state=state,
        action=action,
        value=value,
        prior=prior,
        variance=variance,
        visit_count=visit_count,
    )


def _stub_base_search(candidates=None, scores=None, priors=None,
                      visits=None, variances=None, raise_exc=None,
                      has_history=True, return_int=False):
    """Return a minimal stub that mimics ``MCTSProofSearch``.

    The stub exposes the public attributes the wrapper consults
    (``history``, ``_root_children_cache_for_search``,
    ``debug_trace``).  No real MCTS is required.
    """
    if raise_exc is not None:
        class _Broken:
            def search(self, *args, **kwargs):
                raise RuntimeError(raise_exc)

            history = []

        return _Broken()

    if candidates is None:
        candidates = [
            _StubMol("CCO"), _StubMol("CCN"),
            _StubMol("CCC"), _StubMol("CCCC"),
        ]
    if scores is None:
        scores = [0.3, 0.7, 0.5, 0.9]
    if priors is None:
        priors = [0.25, 0.25, 0.25, 0.25]
    if visits is None:
        visits = [4, 8, 2, 16]
    if variances is None:
        variances = [0.01, 0.02, 0.015, 0.005]

    # NOTE: build the class bodies via ``type()`` rather than
    # ``class ...:`` so the local ``history`` list is captured into
    # the namespace via the explicit dict — otherwise Python's
    # scoping rules hide the local variable from the class body.
    history_list = []
    if has_history:
        history_list.append({
            "candidates": list(candidates),
            "scores": list(scores),
            "priors": list(priors),
            "visits": list(visits),
            "variances": list(variances),
        })

    if return_int:
        def _search_int(self, root):
            return int(np.argmax(np.asarray(scores)))

        return type(
            "_IntSearch",
            (),
            {
                "history": history_list,
                "debug_trace": [],
                "_root_children_cache_for_search": None,
                "search": _search_int,
            },
        )()

    def _search_list(self, root):
        return list(candidates)

    return type(
        "_ListSearch",
        (),
        {
            "history": history_list,
            "debug_trace": [],
            "_root_children_cache_for_search": None,
            "search": _search_list,
        },
    )()


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
# MCTSReplayEntry — 3 tests
# ===========================================================================


class TestMCTSReplayEntry:
    """Tests for the :class:`MCTSReplayEntry` dataclass."""

    def test_replay_entry_init(self):
        """A minimal entry round-trips through its constructor."""
        e = MCTSReplayEntry(
            state="CCO", action="action_0",
            value=0.7, prior=0.25, variance=0.01, visit_count=8,
        )
        assert e.state == "CCO"
        assert e.action == "action_0"
        assert e.value == pytest.approx(0.7)
        assert e.prior == pytest.approx(0.25)
        assert e.variance == pytest.approx(0.01)
        assert e.visit_count == 8
        assert isinstance(e.timestamp, float)
        assert e.timestamp > 0.0
        assert e.metadata == {}

    def test_replay_entry_rejects_non_numeric(self):
        """Non-numeric value / prior / variance raise MCTSReplayError."""
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value="not_a_number",
                prior=0.5, variance=0.0, visit_count=1,
            )
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value=0.5,
                prior="nope", variance=0.0, visit_count=1,
            )
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value=0.5,
                prior=0.5, variance=[0.0], visit_count=1,
            )

    def test_replay_entry_rejects_invalid_visit_count(self):
        """visit_count must be a positive int."""
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value=0.5,
                prior=0.5, variance=0.0, visit_count=0,
            )
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value=0.5,
                prior=0.5, variance=0.0, visit_count=-3,
            )
        with pytest.raises(MCTSReplayError):
            MCTSReplayEntry(
                state="x", action="y", value=0.5,
                prior=0.5, variance=0.0, visit_count=1.5,  # type: ignore[arg-type]
            )


# ===========================================================================
# MCTSReplayBuffer — 8 tests
# ===========================================================================


class TestMCTSReplayBuffer:
    """Tests for the :class:`MCTSReplayBuffer` ring buffer."""

    def test_replay_buffer_add(self):
        """Add entries and verify they are stored."""
        buf = MCTSReplayBuffer(max_size=8)
        assert len(buf) == 0
        buf.add(_mk_entry(value=0.1))
        buf.add(_mk_entry(value=0.2))
        buf.add(_mk_entry(value=0.3))
        assert len(buf) == 3
        assert buf.n_added == 3
        assert buf.n_evicted == 0

    def test_replay_buffer_max_size(self):
        """FIFO eviction kicks in once max_size is reached."""
        buf = MCTSReplayBuffer(max_size=4)
        for i in range(7):
            buf.add(_mk_entry(value=float(i)))
        assert len(buf) == 4
        assert buf.n_added == 7
        assert buf.n_evicted == 3
        # Oldest surviving entry is the 4th one added (i=3 → value=3.0).
        assert buf.entries[0].value == pytest.approx(3.0)
        assert buf.entries[-1].value == pytest.approx(6.0)

    def test_replay_buffer_sample(self):
        """Sample returns batch_size entries."""
        buf = MCTSReplayBuffer(max_size=64, seed=42)
        for i in range(40):
            buf.add(_mk_entry(value=float(i) / 40.0))
        batch = buf.sample(batch_size=16)
        assert len(batch) == 16
        # Sample is without replacement.
        ids = [id(e) for e in batch]
        assert len(set(ids)) == 16

    def test_replay_buffer_sample_underfilled(self):
        """Sample shrinks when batch_size > len(buffer)."""
        buf = MCTSReplayBuffer(max_size=16)
        for i in range(3):
            buf.add(_mk_entry(value=float(i)))
        batch = buf.sample(batch_size=16)
        assert len(batch) == 3

    def test_replay_buffer_sample_empty(self):
        """Sample on empty buffer returns [].  No exception."""
        buf = MCTSReplayBuffer(max_size=4)
        assert buf.sample(batch_size=8) == []

    def test_replay_buffer_sample_rejects_invalid(self):
        """Invalid batch_size raises MCTSReplayError."""
        buf = MCTSReplayBuffer(max_size=4)
        with pytest.raises(MCTSReplayError):
            buf.sample(batch_size=0)
        with pytest.raises(MCTSReplayError):
            buf.sample(batch_size=-5)

    def test_replay_buffer_save_load_pickle(self):
        """Save + load round-trip via pickle preserves contents."""
        buf = MCTSReplayBuffer(max_size=8, seed=0)
        for i in range(5):
            buf.add(_mk_entry(value=float(i), state=f"s{i}", action=f"action_{i}"))
        with tempfile.TemporaryDirectory() as td:
            path = buf.save(os.path.join(td, "buf.pkl"))
            assert os.path.exists(path)
            buf2 = MCTSReplayBuffer.load(path)
            assert len(buf2) == 5
            assert buf2.n_added == 5
            assert buf2.n_evicted == 0
            assert buf2.max_size == 8
            for i, e in enumerate(buf2.entries):
                assert e.state == f"s{i}"
                assert e.value == pytest.approx(float(i))

    def test_replay_buffer_save_load_jsonl(self):
        """Save + load round-trip via JSON-lines preserves scalar fields."""
        buf = MCTSReplayBuffer(max_size=8, seed=0)
        for i in range(3):
            buf.add(_mk_entry(value=float(i), state=f"s{i}", action=f"action_{i}"))
        with tempfile.TemporaryDirectory() as td:
            path = buf.save(os.path.join(td, "buf.jsonl"))
            assert os.path.exists(path)
            buf2 = MCTSReplayBuffer.load(path)
            assert len(buf2) == 3
            assert buf2.n_added == 3
            for i, e in enumerate(buf2.entries):
                assert str(e.state) == f"s{i}"
                assert e.value == pytest.approx(float(i))


# ===========================================================================
# MCTSReplayWrapper — 7 tests
# ===========================================================================


class TestMCTSReplayWrapper:
    """Tests for the :class:`MCTSReplayWrapper` adapter."""

    def test_replay_wrapper_search_runs(self):
        """search() returns an int and bumps n_searches_."""
        base = _stub_base_search(return_int=True)
        buf = MCTSReplayBuffer(max_size=16)
        w = MCTSReplayWrapper(base_search=base, buffer=buf)
        a = w.search(root="CCO")
        assert isinstance(a, int)
        assert w.n_searches_ == 1

    def test_replay_wrapper_records_entries(self):
        """search() with collect_replay=True appends to the buffer."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        w = MCTSReplayWrapper(
            base_search=base, buffer=buf, n_top_actions=3,
        )
        w.search(root="CCO")
        # Should have recorded up to n_top_actions entries.
        assert len(buf) == 3
        assert w.n_entries_recorded_ == 3
        # And the entries should be sorted by visit count desc.
        visits = [e.visit_count for e in buf.entries]
        assert visits == sorted(visits, reverse=True)

    def test_replay_wrapper_search_no_collect(self):
        """search() with collect_replay=False does not append entries."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        w = MCTSReplayWrapper(base_search=base, buffer=buf)
        w.search(root="CCO", collect_replay=False)
        assert len(buf) == 0
        assert w.n_entries_recorded_ == 0

    def test_replay_wrapper_train_value_network(self):
        """train_value_network returns the expected loss keys."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=64)
        for i in range(48):
            buf.add(_mk_entry(value=float(i) / 48.0, prior=0.1 + 0.012 * i))
        w = MCTSReplayWrapper(base_search=base, buffer=buf)
        out = w.train_value_network()
        assert "value_loss" in out
        assert "policy_loss" in out
        assert "batch_size" in out
        assert "mean_value" in out
        assert "mean_prior" in out
        assert "mean_variance" in out
        assert out["batch_size"] == DEFAULT_BATCH_SIZE
        assert 0.0 <= out["value_loss"] <= 1.0
        assert out["policy_loss"] >= 0.0

    def test_replay_wrapper_train_value_network_empty(self):
        """train_value_network on an empty buffer returns zeroed losses."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=8)
        w = MCTSReplayWrapper(base_search=base, buffer=buf)
        out = w.train_value_network()
        assert out["value_loss"] == 0.0
        assert out["policy_loss"] == 0.0
        assert out["batch_size"] == 0
        assert out["mean_value"] == 0.0

    def test_replay_wrapper_train_value_network_with_batch(self):
        """train_value_network accepts an explicit batch."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        batch = [
            _mk_entry(value=0.1, prior=0.2),
            _mk_entry(value=0.3, prior=0.4),
            _mk_entry(value=0.5, prior=0.6),
        ]
        w = MCTSReplayWrapper(base_search=base, buffer=buf)
        out = w.train_value_network(batch=batch)
        assert out["batch_size"] == 3
        assert out["mean_value"] == pytest.approx((0.1 + 0.3 + 0.5) / 3)
        assert out["mean_prior"] == pytest.approx((0.2 + 0.4 + 0.6) / 3)

    def test_replay_wrapper_rejects_invalid(self):
        """Constructor validates buffer / n_top_actions."""
        base = _stub_base_search()
        with pytest.raises(MCTSReplayError):
            MCTSReplayWrapper(base_search=base, buffer=None)  # type: ignore[arg-type]
        buf = MCTSReplayBuffer(max_size=4)
        with pytest.raises(MCTSReplayError):
            MCTSReplayWrapper(base_search=base, buffer=buf, n_top_actions=0)


# ===========================================================================
# ReplayGuidedMCTS — 7 tests
# ===========================================================================


class TestReplayGuidedMCTS:
    """Tests for the :class:`ReplayGuidedMCTS` prior biasing."""

    def test_replay_guided_mcts_bias_prior(self):
        """bias_prior returns a length-n_actions probability vector."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        rg = ReplayGuidedMCTS(base_search=base, replay_buffer=buf)
        out = rg.bias_prior(root="CCO", n_actions=4)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4,)
        # Probability vector (sums to 1).
        assert out.sum() == pytest.approx(1.0, abs=1.0e-6)
        # All entries non-negative.
        assert (out >= 0.0).all()

    def test_replay_guided_mcts_bias_prior_empty_buffer(self):
        """With no replay entries the bias is uniform."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=8)
        rg = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=0.7,
        )
        out = rg.bias_prior(root="CCO", n_actions=4)
        assert np.allclose(out, np.full(4, 0.25))

    def test_replay_guided_mcts_bias_prior_uses_replay(self):
        """With matching entries the bias shifts toward high-visit arms."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        # 10 visits on action_3, 2 visits on action_0, 1 each on others.
        buf.add(_mk_entry(state="CCO", action="action_3",
                          value=0.9, prior=0.25, visit_count=10))
        buf.add(_mk_entry(state="CCO", action="action_0",
                          value=0.1, prior=0.25, visit_count=2))
        for i in [1, 2]:
            buf.add(_mk_entry(state="CCO", action=f"action_{i}",
                             value=0.5, prior=0.25, visit_count=1))
        rg = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=1.0,
        )
        out = rg.bias_prior(root="CCO", n_actions=4)
        # action_3 has the highest visit count → highest probability.
        assert int(np.argmax(out)) == 3
        # And the probability mass should concentrate on action_3.
        assert out[3] > out[0]

    def test_replay_guided_mcts_alpha_mixing(self):
        """alpha_replay controls the bias / base prior mix."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        buf.add(_mk_entry(state="X", action="action_2",
                          value=0.9, prior=0.25, visit_count=10))
        # alpha=0 → pure base prior → uniform.
        rg_pure_base = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=0.0,
        )
        out_pure_base = rg_pure_base.bias_prior(root="X", n_actions=4)
        assert np.allclose(out_pure_base, np.full(4, 0.25))
        # alpha=1 → pure replay bias → action_2 dominates.
        rg_pure_replay = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=1.0,
        )
        out_pure_replay = rg_pure_replay.bias_prior(root="X", n_actions=4)
        assert int(np.argmax(out_pure_replay)) == 2

    def test_replay_guided_consistency(self):
        """bias_prior is deterministic given a fixed replay buffer."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16, seed=0)
        buf.add(_mk_entry(state="CCO", action="action_0",
                          value=0.1, prior=0.25, visit_count=4))
        buf.add(_mk_entry(state="CCO", action="action_1",
                          value=0.2, prior=0.25, visit_count=8))
        rg = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=0.5,
        )
        out_a = rg.bias_prior(root="CCO", n_actions=4)
        out_b = rg.bias_prior(root="CCO", n_actions=4)
        assert np.allclose(out_a, out_b)

    def test_replay_guided_no_match_returns_uniform(self):
        """When no replay entries match, the bias is uniform."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=16)
        buf.add(_mk_entry(state="OTHER", action="action_0",
                          value=0.1, prior=0.25, visit_count=4))
        rg = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf, alpha_replay=0.9,
        )
        out = rg.bias_prior(root="CCO", n_actions=4)
        assert np.allclose(out, np.full(4, 0.25))

    def test_replay_guided_rejects_invalid(self):
        """Constructor validates alpha_replay / tau / buffer."""
        base = _stub_base_search()
        buf = MCTSReplayBuffer(max_size=4)
        with pytest.raises(MCTSReplayError):
            ReplayGuidedMCTS(base_search=base, replay_buffer=buf,
                             alpha_replay=1.5)
        with pytest.raises(MCTSReplayError):
            ReplayGuidedMCTS(base_search=base, replay_buffer=buf,
                             alpha_replay=-0.1)
        with pytest.raises(MCTSReplayError):
            ReplayGuidedMCTS(base_search=base, replay_buffer=buf,
                             tau=0.0)
        # n_actions must be >= 1.
        rg = ReplayGuidedMCTS(base_search=base, replay_buffer=buf)
        with pytest.raises(MCTSReplayError):
            rg.bias_prior(root="CCO", n_actions=0)
        # Buffer must not be None.
        rg_none = ReplayGuidedMCTS(
            base_search=base, replay_buffer=buf,
        )
        rg_none.replay_buffer = None  # type: ignore[assignment]
        with pytest.raises(MCTSReplayError):
            rg_none.bias_prior(root="CCO", n_actions=4)
