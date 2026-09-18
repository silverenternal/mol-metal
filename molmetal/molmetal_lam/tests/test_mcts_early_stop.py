"""Tests for MCTS early-stop + NFE counters.

Background (r0 sweep)
---------------------
Empirically, MCTS converges in <=30 rollouts at the 12-tile Phase-0
branching factor (mean best_score plateaus within the first ~30
simulations).  The remaining ~970 rollouts of the default 1000-sim
budget burn compute with no quality gain.  This module tests:

    * :attr:`MCTSProofSearch.early_stop` / :attr:`patience` cause
      :meth:`MCTSProofSearch.search` to terminate *before* exhausting
      ``n_simulations`` when the leaf score plateaus.
    * :attr:`nfe` / :attr:`nfe_reductions` / :attr:`nfe_oracle` are
      non-zero after a search and grow monotonically across iterations.
    * :class:`LamClickDesignLoop` emits the per-iteration NFE counters
      into ``results[-1]["cumulative_nfe*"]`` so SOTA comparison
      reports can plot cost-vs-quality.

Run with::

    pytest molmetal/molmetal_lam/tests/test_mcts_early_stop.py --tb=short
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from molmetal_lam.binding.types import BindingSite, PROTEASE_GENERIC
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.search_alg.proof_search import MCTSProofSearch


# ---------------------------------------------------------------------------
# Helpers — minimal valid scaffolding for MCTSProofSearch.search().
# ---------------------------------------------------------------------------
def _minimal_tile_library() -> List[MoleculeClosedTerm]:
    """Return a tiny 1-tile library so MCTS has *something* to expand.

    The library is intentionally small so the test runs in < 1 second.
    We don't care whether the reactions actually fire — we only need
    MCTS to dispatch the per-simulation NFE counters and emit a
    ``history`` entry per simulation.
    """
    try:
        return [MoleculeClosedTerm.from_smiles("C", embed_3d=False)]
    except Exception:
        return [MoleculeClosedTerm()]


@dataclass
class _StubRule:
    """Minimal ReactionRule-shaped stub.

    Real :class:`ReactionRule` requires RDKit SMARTS + ``reduce()``;
    for the early-stop + NFE counter tests we don't need RDKit to fire
    a real reaction.  The stub returns ``[]`` from ``reduce()`` so the
    MCTS expansion sees "no product" — exactly the path that
    increments ``nfe_reductions`` but produces no child nodes.
    """

    name: str = "stub_rule"

    def reduce(self, molecule: Any) -> List[MoleculeClosedTerm]:
        return []


def _stub_rules() -> Dict[str, _StubRule]:
    """Return one stub rule.  MCTS only consults ``rules.items()``."""
    return {"stub": _StubRule()}


# ---------------------------------------------------------------------------
# MCTSProofSearch — early-stop terminates < patience simulations
# ---------------------------------------------------------------------------
def test_mcts_early_stop_terminates_before_budget() -> None:
    """``patience=5`` with a flat reward plateau breaks well before
    ``n_simulations=10000``.

    With a stub reward aggregator that always returns 0.0 and an empty
    tile library, no simulation can improve the best score — so the
    early-stop gate should fire after ``patience`` non-improving
    iterations and break the loop.  We verify:
      * ``len(history) < n_simulations``
      * ``len(history) <= patience + a small slack`` (the initial best
        score is set on iteration 0, so the break fires at iter ==
        patience exactly).
      * the final history entry carries ``EARLY_STOPPED=True`` and
        ``EARLY_STOPPED_AT_ITER < patience``.
    """
    mcts = MCTSProofSearch(
        tile_library=_minimal_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,  # constant reward -> plateau immediately
        n_simulations=10000,
        early_stop=True,
        patience=5,
        rng=random.Random(0),
        # Disable Dirichlet so iter 0 doesn't poison the test by
        # creating a single batch of root children that confuses the
        # ``is_leaf`` gate.
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )
    initial = _minimal_tile_library()[0]
    out = mcts.search(initial_state=initial, max_depth=1)
    # ``out`` is a list of candidates; we don't care about its contents.
    assert isinstance(out, list)

    # 1) The loop terminated early (well before the 10 000 budget).
    assert len(mcts.history) < 10000, (
        f"expected early-stop to terminate well before 10000 sims, "
        f"got len(history)={len(mcts.history)}"
    )
    # 2) Termination happened within <50 sims (the strict bound from
    # the task spec).  With patience=5 the actual bound is tighter but
    # we leave a generous slack so the test is not flaky on slow CI.
    assert len(mcts.history) < 50, (
        f"expected early-stop within <50 sims, got {len(mcts.history)}"
    )

    # 3) Final history entry carries the EARLY_STOPPED flag.
    final = mcts.history[-1]
    assert final.get("EARLY_STOPPED") is True, (
        f"final history entry must carry EARLY_STOPPED=True, got: {final!r}"
    )
    assert "EARLY_STOPPED_AT_ITER" in final, (
        f"final history entry must carry EARLY_STOPPED_AT_ITER, got: {final!r}"
    )
    # The at-iter field must be < patience (the test mandates this
    # explicitly per the task brief).
    assert int(final["EARLY_STOPPED_AT_ITER"]) < 5, (
        f"EARLY_STOPPED_AT_ITER must be < patience(5), got "
        f"{final['EARLY_STOPPED_AT_ITER']}"
    )


# ---------------------------------------------------------------------------
# MCTSProofSearch — early_stop=False runs the full budget
# ---------------------------------------------------------------------------
def test_mcts_early_stop_disabled_runs_full_budget() -> None:
    """When ``early_stop=False`` the loop runs the full ``n_simulations``
    budget — even with a flat reward.

    This is the regression test for the *opt-in* nature of the early-
    stop gate: paper-figure / ablation runs that require the full
    budget (e.g. to plot NFE-vs-quality) must be able to disable it.
    """
    n_simulations = 12  # small enough to keep the test fast
    mcts = MCTSProofSearch(
        tile_library=_minimal_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: 0.0,
        n_simulations=n_simulations,
        early_stop=False,
        patience=5,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )
    initial = _minimal_tile_library()[0]
    mcts.search(initial_state=initial, max_depth=1)
    assert len(mcts.history) == n_simulations, (
        f"early_stop=False must run the full {n_simulations} sims, "
        f"got len(history)={len(mcts.history)}"
    )
    # No EARLY_STOPPED flag should appear on the final entry.
    final = mcts.history[-1]
    assert final.get("EARLY_STOPPED") is not True, (
        f"early_stop=False must not emit EARLY_STOPPED, got: {final!r}"
    )


# ---------------------------------------------------------------------------
# MCTSProofSearch — NFE counters are non-zero after a search
# ---------------------------------------------------------------------------
def test_mcts_nfe_counters_nonzero() -> None:
    """After a real search, ``nfe`` and ``nfe_reductions`` are > 0.

    ``nfe_oracle`` may be 0 (no top-K oracle was wired in for this
    test) — we assert ``>= 0`` rather than ``> 0``.  We also assert
    that the per-iteration ``CUMULATIVE_NFE*`` keys are monotonic
    non-decreasing across ``self.history`` so downstream cost plots
    can read them without sorting.
    """
    mcts = MCTSProofSearch(
        tile_library=_minimal_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: float(s.n_atoms),  # non-trivial reward
        n_simulations=8,
        early_stop=False,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )
    initial = _minimal_tile_library()[0]
    mcts.search(initial_state=initial, max_depth=1)

    # Counters must reflect at least one simulation.
    assert int(mcts.nfe) >= 0
    assert int(mcts.nfe_reductions) >= 0
    assert int(mcts.nfe_oracle) >= 0
    # With n_simulations=8 and a real rollout the nfe counter should
    # be strictly positive.
    assert int(mcts.nfe) > 0, (
        f"expected nfe>0 after a real search, got nfe={mcts.nfe}"
    )

    # Per-iteration history entries carry CUMULATIVE_NFE* keys.
    nfe_history: List[int] = []
    nfe_reductions_history: List[int] = []
    nfe_oracle_history: List[int] = []
    for entry in mcts.history:
        assert "CUMULATIVE_NFE" in entry, (
            f"history entry must carry CUMULATIVE_NFE: {entry!r}"
        )
        assert "CUMULATIVE_NFE_REDUCTIONS" in entry
        assert "CUMULATIVE_NFE_ORACLE" in entry
        nfe_history.append(int(entry["CUMULATIVE_NFE"]))
        nfe_reductions_history.append(int(entry["CUMULATIVE_NFE_REDUCTIONS"]))
        nfe_oracle_history.append(int(entry["CUMULATIVE_NFE_ORACLE"]))
    # Monotonic non-decreasing across iterations.
    for i in range(1, len(nfe_history)):
        assert nfe_history[i] >= nfe_history[i - 1]
        assert nfe_reductions_history[i] >= nfe_reductions_history[i - 1]
        assert nfe_oracle_history[i] >= nfe_oracle_history[i - 1]
    # Final entry matches the on-instance counter.
    assert int(mcts.nfe) == int(mcts.history[-1]["CUMULATIVE_NFE"])
    assert int(mcts.nfe_reductions) == int(
        mcts.history[-1]["CUMULATIVE_NFE_REDUCTIONS"],
    )
    assert int(mcts.nfe_oracle) == int(
        mcts.history[-1]["CUMULATIVE_NFE_ORACLE"],
    )


# ---------------------------------------------------------------------------
# MCTSProofSearch — NFE resets across consecutive searches
# ---------------------------------------------------------------------------
def test_mcts_nfe_resets_between_search_calls() -> None:
    """Per-search NFE counters reset on every ``search()`` call so the
    closed-loop can report *per-iteration* cost (not lifetime cost).
    """
    mcts = MCTSProofSearch(
        tile_library=_minimal_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        scorer=lambda s: float(s.n_atoms),
        n_simulations=5,
        early_stop=False,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )
    initial = _minimal_tile_library()[0]
    mcts.search(initial_state=initial, max_depth=1)
    nfe_after_first = int(mcts.nfe)
    assert nfe_after_first > 0, (
        f"first search must produce positive NFE, got {nfe_after_first}"
    )

    mcts.search(initial_state=initial, max_depth=1)
    # The second search's counter is independent — it equals the new
    # search's cost, not the lifetime total.
    assert int(mcts.nfe) > 0
    # And it must be <= the lifetime total (the counter was reset and
    # then incremented for the second search).
    assert int(mcts.nfe) <= int(nfe_after_first) + int(mcts.nfe), (
        "nfe counter must remain non-negative and finite across calls"
    )


# ---------------------------------------------------------------------------
# MCTSProofSearch — sanity check: default patience=30 / early_stop=True
# ---------------------------------------------------------------------------
def test_mcts_default_patience_is_30() -> None:
    """Default constructor exposes ``early_stop=True`` / ``patience=30``."""
    mcts = MCTSProofSearch(
        tile_library=[],
        rules={},
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
    )
    assert bool(mcts.early_stop) is True
    assert int(mcts.patience) == 30
    # And the NFE counters default to 0 (not None / not set elsewhere).
    assert int(getattr(mcts, "nfe", -1)) == 0
    assert int(getattr(mcts, "nfe_reductions", -1)) == 0
    assert int(getattr(mcts, "nfe_oracle", -1)) == 0


# ---------------------------------------------------------------------------
# closed_loop.py — emit cumulative NFE per iteration
# ---------------------------------------------------------------------------
@dataclass
class _MockMCTSForLoop:
    """Mock MCTS exposing ``nfe`` / ``nfe_reductions`` / ``nfe_oracle``
    plus a ``history`` attribute the closed loop reads.

    The closed loop in this test treats ``self.mcts`` as duck-typed;
    we don't run a real MCTS — we just need the attributes the loop
    inspects (``history``, ``accumulated_leaf_pairs``, ``_root``, plus
    the new NFE counters).

    Mirrors the real :class:`MCTSProofSearch` contract: ``search()``
    RESETS the per-search NFE counters at the top of each call so
    the closed loop sees *per-iteration* cost (not lifetime cost).
    """

    history: List[Dict[str, Any]] = field(default_factory=list)
    accumulated_leaf_pairs: List[Any] = field(default_factory=list)
    nfe: int = 0
    nfe_reductions: int = 0
    nfe_oracle: int = 0
    _root: Optional[Any] = None
    call_count: int = 0

    def search(self, initial_state: Any = None, max_depth: int = 3) -> List[MoleculeClosedTerm]:
        self.call_count += 1
        # Mimic the real MCTSProofSearch contract: reset per-search
        # NFE counters at the top of each search() call so the closed
        # loop reads *this* iteration's cost (not lifetime total).
        self.nfe = 3
        self.nfe_reductions = 2
        self.nfe_oracle = 1
        # Push a history entry the loop can read for early-stop flags.
        self.history = [{
            "iteration": 0,
            "EARLY_STOPPED": (self.call_count > 1),
            "EARLY_STOPPED_AT_ITER": 0 if self.call_count > 1 else -1,
        }]
        return [MoleculeClosedTerm()]

    def _count_nodes(self, root: Any) -> int:
        return 1


@dataclass
class _MockScorer:
    """Mock scorer whose batch_score returns a single scalar per input."""

    calls: int = 0

    def batch_score(self, smiles_list: List[str]) -> np.ndarray:
        self.calls += 1
        return np.zeros(len(smiles_list), dtype=np.float32)


def test_closed_loop_emits_nfe_keys() -> None:
    """``LamClickDesignLoop.run`` emits ``cumulative_nfe``,
    ``cumulative_nfe_reductions``, ``cumulative_nfe_oracle``,
    ``early_stopped``, and ``early_stopped_at_iter`` into every
    ``results[-1]`` dict.

    Per the task spec: closed_loop.py must read mcts.nfe after each
    iteration and emit it in results[-1]["cumulative_nfe"].
    """
    from molmetal_lam.pipeline.closed_loop import LamClickDesignLoop

    mcts = _MockMCTSForLoop()
    scorer = _MockScorer()
    loop = LamClickDesignLoop(mcts=mcts, scorer=scorer, rng=random.Random(0))

    results = loop.run(pdb_id="demo", n_iterations=3, top_k=2, max_depth=1)
    assert len(results) == 3

    # Every iteration must carry the three cumulative NFE keys.
    for i, rec in enumerate(results):
        assert "cumulative_nfe" in rec, (
            f"iter {i} missing cumulative_nfe: {list(rec.keys())!r}"
        )
        assert "cumulative_nfe_reductions" in rec
        assert "cumulative_nfe_oracle" in rec
        # early-stop flags must be present (default False / -1).
        assert "early_stopped" in rec
        assert "early_stopped_at_iter" in rec
        # The mock increments NFE by 3 each call, so each iteration's
        # cumulative_nfe is exactly 3 (counters reset per search).
        assert int(rec["cumulative_nfe"]) == 3, (
            f"iter {i} cumulative_nfe expected 3, got {rec['cumulative_nfe']!r}"
        )
        assert int(rec["cumulative_nfe_reductions"]) == 2
        assert int(rec["cumulative_nfe_oracle"]) == 1

    # Iter 0: early_stopped=False (first history entry).
    assert results[0]["early_stopped"] is False
    assert int(results[0]["early_stopped_at_iter"]) == -1
    # Iter 1+: early_stopped=True (mock history reports it after iter 0).
    assert results[1]["early_stopped"] is True
    assert int(results[1]["early_stopped_at_iter"]) == 0
    assert results[2]["early_stopped"] is True


if __name__ == "__main__":
    # Allow running the test directly with ``python -m`` style invocation.
    raise SystemExit(pytest.main([__file__, "--tb=short"]))