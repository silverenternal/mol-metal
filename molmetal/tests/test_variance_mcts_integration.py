"""Tests for the variance-controlled MCTS integration layer (TODO-30 P5.2).

================================================================
CPU-only smoke / contract tests for ``variance_mcts_integration``
================================================================
This module validates the three classes / functions that comprise
the MCTS-level variance-controlled integration stack:

* :class:`VarianceMCTSNode`                          (10 tests)
* :class:`VarianceMCTSWrapper`                       (8 tests)
* :func:`compare_variance_vs_dirichlet_mcts`         (7 tests)

The tests are CPU-only and complete in <5 s on any CI container.
None of them mutate :mod:`proof_search`.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_mcts_integration.py -v
"""

from __future__ import annotations

import math

import pytest

try:
    from molmetal_lam.search_alg.variance_mcts_integration import (
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        DEFAULT_N_MIN_VISITS,
        DEFAULT_N_RUNS,
        VarianceMCTSError,
        VarianceMCTSNode,
        VarianceMCTSWrapper,
        compare_variance_vs_dirichlet_mcts,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore  # noqa: E402
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        DEFAULT_N_MIN_VISITS,
        DEFAULT_N_RUNS,
        VarianceMCTSError,
        VarianceMCTSNode,
        VarianceMCTSWrapper,
        compare_variance_vs_dirichlet_mcts,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_node(
    n_actions: int = 4,
    c_puct: float = 1.0,
    c_var: float = 0.5,
    n_min_visits: int = 2,
    priors: "list[float] | None" = None,
) -> VarianceMCTSNode:
    """Build a VarianceMCTSNode with uniform priors."""
    if priors is None:
        priors = [1.0 / float(n_actions)] * int(n_actions)
    return VarianceMCTSNode(
        prior=1.0 / float(n_actions),
        c_puct=float(c_puct),
        c_var=float(c_var),
        n_min_visits=int(n_min_visits),
        action_priors=list(priors),
    )


def _stub_proof_search(candidates=None, scores=None, error=None):
    """Return a minimal stub that mimics ``MCTSProofSearch``."""
    if error is not None:

        class _Broken:
            tile_library = ["stub_root"]

            def score_final(self, state):
                raise RuntimeError(error)

            def search(self, *args, **kwargs):
                raise RuntimeError(error)

        return _Broken()

    if candidates is None:
        candidates = [object(), object(), object()]
    if scores is None:
        scores = [0.3, 0.7, 0.5]

    class _Stub:
        tile_library = ["stub_root"]
        leaf_value_var = 0.04
        leaf_value_std = 0.2
        puct_exploit_ratio_var = 0.97
        history = [
            {"step": 0, "PUCT_EXPLOIT_RATIO_VAR": 0.97, "LEAF_VALUE_VAR": 0.04},
        ]

        def search(self, initial_state, max_depth: int = 3):
            return list(candidates)

        def score_final(self, state):
            try:
                idx = list(candidates).index(state)
                return float(scores[idx % len(scores)])
            except ValueError:
                return 0.0

    return _Stub()


# ===========================================================================
# VarianceMCTSNode (10 tests)
# ===========================================================================


class TestVarianceMCTSNode:
    def test_variance_mcts_node_select_returns_valid_action(self) -> None:
        """``select_action`` returns an index in ``[0, n_actions)``."""
        node = _mk_node(n_actions=4)
        a = node.select_action()
        assert 0 <= a < 4

    def test_variance_mcts_node_update_stats(self) -> None:
        """``update(value, variance)`` increments visits + stores sigma."""
        node = _mk_node(n_actions=3)
        node.update(value=1.0, variance=0.04, action=0)
        assert node.n_value(0) == 1
        assert math.isclose(node.q_value(0), 1.0)
        assert math.isclose(node.sigma_value(0), math.sqrt(0.04))

    def test_variance_mcts_node_running_mean(self) -> None:
        """``q_value`` is the running mean of all updates."""
        node = _mk_node(n_actions=2)
        node.update(value=1.0, variance=0.0, action=0)
        node.update(value=3.0, variance=0.0, action=0)
        assert math.isclose(node.q_value(0), 2.0)
        assert node.n_value(0) == 2

    def test_variance_mcts_node_unvisited_first(self) -> None:
        """FPU-style: unvisited arm wins before arg-max."""
        node = _mk_node(n_actions=3, c_puct=0.0, c_var=0.0)
        # Pre-populate arms 0,1 to make arm 2 the only unvisited.
        node.update(value=0.0, variance=0.0, action=0)
        node.update(value=0.0, variance=0.0, action=1)
        a = node.select_action()
        assert a == 2

    def test_variance_mcts_node_high_variance_penalty(self) -> None:
        """High variance on high-Q arm → selector down-weights it."""
        node = _mk_node(
            n_actions=2,
            c_puct=0.0,
            c_var=5.0,
            n_min_visits=2,
        )
        # Arm 0: high value but high variance (sigma=1).
        for _ in range(3):
            node.update(value=1.0, variance=1.0**2, action=0)
        # Arm 1: same value, zero variance.
        for _ in range(3):
            node.update(value=1.0, variance=0.0**2, action=1)
        # arm 0: penalty = 5 * 1 / 4 = 1.25 > Q=1 → arm 1 wins.
        a = node.select_action()
        assert a == 1

    def test_variance_mcts_node_zero_variance_no_penalty(self) -> None:
        """Zero variance → selector behaves like canonical PUCT."""
        node = _mk_node(
            n_actions=3,
            c_puct=0.0,
            c_var=0.5,
            n_min_visits=2,
        )
        for _ in range(3):
            node.update(value=0.5, variance=0.0, action=0)
            node.update(value=0.3, variance=0.0, action=1)
            node.update(value=0.7, variance=0.0, action=2)
        # No variance penalty → argmax Q picks arm 2 (Q=0.7).
        assert node.select_action() == 2

    def test_variance_mcts_node_rejects_bad_init(self) -> None:
        """Negative ``c_puct`` / ``c_var`` / ``n_min_visits`` are rejected."""
        with pytest.raises(VarianceMCTSError):
            VarianceMCTSNode(prior=0.5, c_puct=-0.1)
        with pytest.raises(VarianceMCTSError):
            VarianceMCTSNode(prior=0.5, c_var=-0.1)
        with pytest.raises(VarianceMCTSError):
            VarianceMCTSNode(prior=0.5, n_min_visits=0)

    def test_variance_mcts_node_puct_formula(self) -> None:
        """``puct_score`` matches canonical PUCT + variance penalty.

        Wf-51 Phase-1 sign-flip fix: the variance penalty is
        *persistent* (no ``(1 + N_child)`` divisor).  See
        ``molmetal/reports/wf51_sign_audit/sign_flip_diff.md`` for
        the audit.
        """
        node = _mk_node(
            n_actions=2,
            c_puct=1.0,
            c_var=0.5,
            n_min_visits=1,
        )
        node.update(value=0.5, variance=0.04, action=0)
        # n_parent = 1, Q=0.5, N=1, P=0.5 (uniform), sigma=0.2
        # score = 0.5 + 1*0.5*sqrt(1)/(2) - 0.5*0.2
        #       = 0.5 + 0.25 - 0.1 = 0.65
        score = node.puct_score(action=0, n_parent=1)
        assert math.isclose(score, 0.65, rel_tol=1e-6)

    def test_variance_mcts_node_tree_statistics(self) -> None:
        """``tree_statistics`` reports variance + visit count correctly."""
        node = _mk_node(n_actions=3, n_min_visits=2)
        node.update(value=0.5, variance=0.04, action=0)
        node.update(value=0.7, variance=0.16, action=1)
        node.update(value=0.5, variance=0.04, action=0)
        stats = node.tree_statistics()
        assert stats["n_actions"] == 3
        assert stats["total_visits"] == 3.0
        # sigma(0) = sqrt(0.04) = 0.2, sigma(1) = sqrt(0.16) = 0.4.
        assert math.isclose(stats["max_sigma"], 0.4, rel_tol=1e-6)
        # mean_sigma over arm 0 + arm 1 = (0.2 + 0.4) / 2 = 0.3.
        assert math.isclose(stats["mean_sigma"], 0.3, rel_tol=1e-6)
        assert set(stats["per_arm"].keys()) == {"0", "1", "2"}

    def test_variance_mcts_node_no_actions_raises(self) -> None:
        """``select_action`` raises on a node with zero actions."""
        node = VarianceMCTSNode(prior=0.5)
        with pytest.raises(VarianceMCTSError):
            node.select_action()


# ===========================================================================
# VarianceMCTSWrapper (8 tests)
# ===========================================================================


class TestVarianceMCTSWrapper:
    def test_variance_mcts_wrapper_constructs(self) -> None:
        """Wrapper accepts a stub ``base_search`` and a selector."""
        stub = _stub_proof_search()
        wrap = VarianceMCTSWrapper(base_search=stub)
        assert wrap.base_search is stub
        assert wrap.variance_puct is not None

    def test_variance_mcts_wrapper_search_runs(self) -> None:
        """``search(root)`` returns a valid action index."""
        wrap = VarianceMCTSWrapper(base_search=_stub_proof_search())
        a = wrap.search(root=4)
        assert 0 <= a < 4

    def test_variance_mcts_wrapper_returns_best_action(self) -> None:
        """``search_with_proof_search`` picks argmax-score candidate."""
        wrap = VarianceMCTSWrapper(base_search=_stub_proof_search())
        report = wrap.search_with_proof_search(initial_state="stub")
        assert report["best_action"] in (0, 1, 2)
        # Scores: [0.3, 0.7, 0.5] → best_action == 1.
        assert report["best_action"] == 1
        assert report["variance_report"]["mean_var"] > 0.0

    def test_variance_mcts_wrapper_tree_statistics(self) -> None:
        """``tree_statistics`` returns the canonical R17 §L-A2 fields."""
        wrap = VarianceMCTSWrapper(base_search=_stub_proof_search())
        wrap.search_with_proof_search(initial_state="stub")
        stats = wrap.tree_statistics()
        # The stub exposes leaf_value_var / leaf_value_std /
        # puct_exploit_ratio_var.
        assert "leaf_value_var" in stats
        assert "leaf_value_std" in stats
        assert "puct_exploit_ratio_var" in stats
        assert "wall_time_s" in stats

    def test_variance_mcts_wrapper_rejects_bad_root(self) -> None:
        """``search`` raises on non-IntegrationNode / non-int root."""
        wrap = VarianceMCTSWrapper(base_search=object())
        with pytest.raises(VarianceMCTSError):
            wrap.search(root="not a node")
        with pytest.raises(VarianceMCTSError):
            wrap.search(root=0)  # int <= 0 also rejected.

    def test_variance_mcts_wrapper_error_path(self) -> None:
        """``search_with_proof_search`` returns ``error`` on broken search."""
        wrap = VarianceMCTSWrapper(
            base_search=_stub_proof_search(error="boom"),
        )
        report = wrap.search_with_proof_search(initial_state="stub")
        assert "error" in report
        assert "boom" in report["error"]

    def test_variance_mcts_wrapper_empty_candidates(self) -> None:
        """``search_with_proof_search`` handles empty candidate list."""
        wrap = VarianceMCTSWrapper(
            base_search=_stub_proof_search(candidates=[], scores=[]),
        )
        report = wrap.search_with_proof_search(initial_state="stub")
        assert report["best_action"] == -1
        assert report["variance_report"]["mean_var"] == 0.0

    def test_variance_mcts_wrapper_variance_report_populated(self) -> None:
        """``variance_report_`` is populated after a search call."""
        wrap = VarianceMCTSWrapper(base_search=_stub_proof_search())
        wrap.search_with_proof_search(initial_state="stub")
        assert "per_candidate_var" in wrap.variance_report_
        assert "best_action" in wrap.variance_report_
        assert "mean_var" in wrap.variance_report_


# ===========================================================================
# compare_variance_vs_dirichlet_mcts (7 tests)
# ===========================================================================


class TestCompareVarianceVsDirichlet:
    def test_compare_returns_dict(self) -> None:
        """``compare_variance_vs_dirichlet_mcts`` returns a documented dict."""
        base = _stub_proof_search()
        var = _stub_proof_search()
        report = compare_variance_vs_dirichlet_mcts(
            base_search=base,
            variance_search=var,
            n_runs=2,
        )
        for key in (
            "base",
            "variance",
            "variance_reduction",
            "wall_time_ratio",
            "n_runs",
            "max_depth",
        ):
            assert key in report

    def test_compare_variance_lower_variance(self) -> None:
        """When variance_search emits tighter scores, mean_var is lower."""
        # Base: scores span [0, 1] (high variance).
        base = _stub_proof_search(
            candidates=["c1", "c2", "c3", "c4"],
            scores=[0.0, 0.3, 0.7, 1.0],
        )
        # Variance: scores cluster around 0.5 (low variance).
        var = _stub_proof_search(
            candidates=["c1", "c2", "c3", "c4"],
            scores=[0.45, 0.50, 0.55, 0.50],
        )
        report = compare_variance_vs_dirichlet_mcts(
            base_search=base,
            variance_search=var,
            n_runs=2,
        )
        assert report["variance"]["mean_var"] < report["base"]["mean_var"]
        # variance_reduction is positive when variance search wins.
        assert report["variance_reduction"] > 0.0

    def test_compare_wall_time(self) -> None:
        """``wall_time_ratio`` is a finite positive scalar."""
        base = _stub_proof_search()
        var = _stub_proof_search()
        report = compare_variance_vs_dirichlet_mcts(
            base_search=base,
            variance_search=var,
            n_runs=3,
        )
        assert report["wall_time_ratio"] >= 0.0
        assert "wall_time_s" in report["base"]
        assert "wall_time_s" in report["variance"]

    def test_compare_success_rate(self) -> None:
        """``success_rate`` is the fraction of runs that emit candidates."""
        base = _stub_proof_search()
        var = _stub_proof_search()
        report = compare_variance_vs_dirichlet_mcts(
            base_search=base,
            variance_search=var,
            n_runs=3,
        )
        assert 0.0 <= report["base"]["success_rate"] <= 1.0
        assert 0.0 <= report["variance"]["success_rate"] <= 1.0

    def test_compare_rejects_same_search(self) -> None:
        """Base and variance search must be distinct instances."""
        s = _stub_proof_search()
        with pytest.raises(VarianceMCTSError):
            compare_variance_vs_dirichlet_mcts(
                base_search=s,
                variance_search=s,
                n_runs=1,
            )

    def test_compare_rejects_bad_n_runs(self) -> None:
        """``n_runs < 1`` raises ``VarianceMCTSError``."""
        with pytest.raises(VarianceMCTSError):
            compare_variance_vs_dirichlet_mcts(
                base_search=_stub_proof_search(),
                variance_search=_stub_proof_search(),
                n_runs=0,
            )

    def test_compare_with_empty_tile_library(self) -> None:
        """Empty tile library surfaces an ``error`` key in the report."""

        class Empty:
            tile_library = []

        report = compare_variance_vs_dirichlet_mcts(
            base_search=Empty(),
            variance_search=_stub_proof_search(),
            n_runs=1,
        )
        assert "error" in report
