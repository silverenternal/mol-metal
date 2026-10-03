"""Tests for the variance-controlled PUCT integration layer (TODO-30 P5.1).

================================================================
CPU-only smoke / contract tests for ``variance_puct_integration``
================================================================
This module validates the four classes and one helper that comprise
the R18 variance-controlled PUCT integration stack:

* :class:`IntegrationNode`                (1 test)
* :class:`VarianceControlledPUCT`         (8 tests)
* :class:`DirichletVarianceComparator`    (4 tests)
* :class:`MCTSVarianceWrapper`            (4 tests)
* :func:`integration_with_proof_search`   (3 tests)

The tests are CPU-only and complete in <2 s on any CI container.
None of them mutate :mod:`proof_search` or :mod:`score_repellent_mc`.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_puct_integration.py -v
"""

from __future__ import annotations

import math

import pytest

# Lazy import so missing torch / score_repellent_mc is caught at
# use-time, not at module load (matches the convention in
# test_score_repellent_mc.py).
try:
    from molmetal_lam.search_alg.variance_puct_integration import (
        DirichletVarianceComparator,
        IntegrationNode,
        MCTSVarianceWrapper,
        VarianceControlledPUCT,
        VariancePUCTError,
        integration_with_proof_search,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore  # noqa: E402
        DirichletVarianceComparator,
        IntegrationNode,
        MCTSVarianceWrapper,
        VarianceControlledPUCT,
        VariancePUCTError,
        integration_with_proof_search,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_root(n_children: int = 4, prior: float = 0.5) -> IntegrationNode:
    """Return a root :class:`IntegrationNode` with ``n_children`` children.

    All children carry a uniform prior ``1.0`` so the PUCT bonus
    is constant across arms and Q (the running-mean reward) is
    the dominant signal after a few visits.
    """
    return IntegrationNode(
        node_id="root",
        prior=float(prior),
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0) for i in range(int(n_children))
        ],
    )


def _mk_root_decreasing(n_children: int = 4, prior: float = 0.5) -> IntegrationNode:
    """Variant with decreasing priors — used for tests that probe PUCT bonus."""
    return IntegrationNode(
        node_id="root",
        prior=float(prior),
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0 / (i + 1))
            for i in range(int(n_children))
        ],
    )


# ===========================================================================
# IntegrationNode (1 test)
# ===========================================================================


class TestIntegrationNode:
    def test_integration_node_action_count(self) -> None:
        """``action_count()`` returns the number of children."""
        root = _mk_root(n_children=5)
        assert root.action_count() == 5
        leaf = IntegrationNode(node_id="leaf")
        assert leaf.action_count() == 1  # at-least-1 floor


# ===========================================================================
# VarianceControlledPUCT (8 tests)
# ===========================================================================


class TestVarianceControlledPUCT:
    def test_variance_puct_select_returns_valid_action(self) -> None:
        """``select`` returns an index in ``[0, n_actions)``."""
        sel = VarianceControlledPUCT(c_puct=1.0, c_var=0.5)
        root = _mk_root(n_children=4)
        action = sel.select(root)
        assert 0 <= action < 4

    def test_variance_puct_zero_variance(self) -> None:
        """Zero variance → selector behaves like canonical PUCT (no penalty)."""
        sel = VarianceControlledPUCT(c_puct=1.0, c_var=0.5, n_min_visits=2)
        root = _mk_root(n_children=3)
        # Run a few iterations to populate Q.
        for _ in range(3):
            sel.update(root, 0, value=0.5, variance=0.0)
            sel.update(root, 1, value=0.3, variance=0.0)
            sel.update(root, 2, value=0.7, variance=0.0)
        # Without variance penalty the highest-Q arm should win.
        a = sel.select(root)
        # Q values: arm 0 = 0.5, arm 1 = 0.3, arm 2 = 0.7 → arm 2 wins.
        assert a == 2

    def test_variance_puct_high_variance(self) -> None:
        """High variance on the high-Q arm → selector down-weights it."""
        sel = VarianceControlledPUCT(
            c_puct=0.0,
            c_var=1.0,
            n_min_visits=2,
        )
        root = _mk_root(n_children=2)
        # Arm 0: high value but high variance.
        for _ in range(3):
            sel.update(root, 0, value=1.0, variance=0.5**2)
        # Arm 1: lower value, low variance.
        for _ in range(3):
            sel.update(root, 1, value=0.5, variance=0.0**2)
        # With c_var=1, c_puct=0 the penalty should push arm 1
        # above arm 0 even though Q(0) > Q(1).
        # Need many visits to activate penalty (n_min_visits=2).
        # arm 0: N=3, sigma=0.5, penalty = 1 * 0.5 / 4 = 0.125
        # arm 1: N=3, sigma=0.0, penalty = 0
        # Q(0)=1, Q(1)=0.5; net: arm 0=0.875, arm 1=0.5 → arm 0 still wins.
        # Bump c_var and zero out Q via many visits to force penalty:
        sel_big = VarianceControlledPUCT(
            c_puct=0.0,
            c_var=5.0,
            n_min_visits=2,
        )
        for _ in range(3):
            sel_big.update(root, 0, value=1.0, variance=1.0**2)
            sel_big.update(root, 1, value=1.0, variance=0.0**2)
        # arm 0: penalty = 5 * 1 / 4 = 1.25 > Q=1
        # arm 1: penalty = 0; Q=1
        a = sel_big.select(root)
        assert a == 1

    def test_variance_puct_unvisited_first(self) -> None:
        """Unvisited arms win FPU-style before any arg-max."""
        sel = VarianceControlledPUCT(c_puct=0.0, c_var=0.0)
        root = _mk_root(n_children=3)
        a = sel.select(root)
        assert a == 0  # first unvisited in iteration order

    def test_variance_puct_q_running_mean(self) -> None:
        """``q_value`` is the running mean of updates."""
        sel = VarianceControlledPUCT()
        root = _mk_root(n_children=2)
        sel.update(root, 0, value=1.0, variance=0.0)
        sel.update(root, 0, value=3.0, variance=0.0)
        assert math.isclose(sel.q_value(root, 0), 2.0)
        assert sel.n_value(root, 0) == 2

    def test_variance_puct_update_increments_visits(self) -> None:
        """Each ``update`` increments the visit count by 1."""
        sel = VarianceControlledPUCT()
        root = _mk_root(n_children=2)
        for i in range(1, 6):
            sel.update(root, 0, value=float(i), variance=0.0)
            assert sel.n_value(root, 0) == i

    def test_variance_puct_rejects_bad_init(self) -> None:
        """Negative ``c_puct`` / ``c_var`` / ``epsilon`` are rejected."""
        with pytest.raises(VariancePUCTError):
            VarianceControlledPUCT(c_puct=-0.1)
        with pytest.raises(VariancePUCTError):
            VarianceControlledPUCT(c_var=-0.1)
        with pytest.raises(VariancePUCTError):
            VarianceControlledPUCT(n_min_visits=0)
        with pytest.raises(VariancePUCTError):
            VarianceControlledPUCT(epsilon=1.5)

    def test_variance_puct_puct_formula(self) -> None:
        """``puct_score`` matches the canonical PUCT + variance penalty.

        Wf-51 Phase-1 sign-flip fix: the variance penalty is
        *persistent* (no ``(1 + N_child)`` divisor).  See
        ``molmetal/reports/wf51_sign_audit/sign_flip_diff.md`` for
        the audit.
        """
        sel = VarianceControlledPUCT(
            c_puct=1.0,
            c_var=0.5,
            n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        # One visit so variance term is active.
        sel.update(root, 0, value=0.5, variance=0.04)
        # n_parent = 1 (only arm 0 visited).
        # arm 0: Q=0.5, N=1, P=1.0 (children[0].prior), sigma=0.2
        # score = 0.5 + 1*1*sqrt(1)/(2) - 0.5*0.2
        #       = 0.5 + 0.5 - 0.1 = 0.9
        score = sel.puct_score(root, 0, n_parent=1)
        assert math.isclose(score, 0.9, rel_tol=1e-6)


# ===========================================================================
# DirichletVarianceComparator (4 tests)
# ===========================================================================


class TestDirichletVarianceComparator:
    def test_dirichlet_vs_score_repellent_variance(self) -> None:
        """Score-repellent samples have lower marginal variance than Dirichlet."""
        cmp = DirichletVarianceComparator(
            alpha=1.0,
            score_repellent_alpha=0.5,
            n_iters=20,
        )
        report = cmp.compare(priors=[0.25, 0.25, 0.25, 0.25], n_samples=256)
        assert "per_arm_var_dirichlet" in report
        assert "per_arm_var_score_repellent" in report
        assert report["n_samples"] == 256
        # The score-repellent chain anchors at the prior so its
        # marginal variance should be ≤ Dirichlet's per arm.
        for v_dir, v_sr in zip(
            report["per_arm_var_dirichlet"],
            report["per_arm_var_score_repellent"],
        ):
            assert v_sr <= v_dir + 1e-3

    def test_dirichlet_variance_returns_dict(self) -> None:
        """``compare`` returns a dict with the documented keys."""
        cmp = DirichletVarianceComparator()
        report = cmp.compare(n_samples=64)
        for key in (
            "per_arm_var_dirichlet",
            "per_arm_var_score_repellent",
            "kl_divergence",
            "total_var_dirichlet",
            "total_var_score_repellent",
            "n_samples",
        ):
            assert key in report

    def test_dirichlet_variance_kl_non_negative(self) -> None:
        """KL divergence is a non-negative scalar."""
        cmp = DirichletVarianceComparator()
        report = cmp.compare(n_samples=128)
        assert report["kl_divergence"] >= 0.0

    def test_dirichlet_variance_rejects_bad_alpha(self) -> None:
        """Non-positive alphas are rejected at construction."""
        with pytest.raises(VariancePUCTError):
            DirichletVarianceComparator(alpha=0.0)
        with pytest.raises(VariancePUCTError):
            DirichletVarianceComparator(score_repellent_alpha=-0.1)


# ===========================================================================
# MCTSVarianceWrapper (4 tests)
# ===========================================================================


class TestMCTSVarianceWrapper:
    def test_mcts_variance_wrapper_constructs(self) -> None:
        """Wrapper accepts a stub ``base_search`` and a selector."""
        stub = type("Stub", (), {})()
        sel = VarianceControlledPUCT()
        wrap = MCTSVarianceWrapper(base_search=stub, variance_puct=sel)
        assert wrap.base_search is stub
        assert wrap.variance_puct is sel

    def test_mcts_variance_wrapper_search(self) -> None:
        """``search(root)`` returns the selector's action index."""
        sel = VarianceControlledPUCT()
        wrap = MCTSVarianceWrapper(base_search=object(), variance_puct=sel)
        root = _mk_root(n_children=3)
        a = wrap.search(root)
        assert 0 <= a < 3

    def test_mcts_variance_wrapper_rejects_bad_root(self) -> None:
        """``search`` raises ``VariancePUCTError`` on non-IntegrationNode root."""
        wrap = MCTSVarianceWrapper(base_search=object())
        with pytest.raises(VariancePUCTError):
            wrap.search("not a node")

    def test_mcts_variance_wrapper_search_with_proof_search_error(self) -> None:
        """``search_with_proof_search`` returns ``error`` key on failure."""

        class BrokenSearch:
            def search(self, *args, **kwargs):
                raise RuntimeError("simulated MCTS failure")

        wrap = MCTSVarianceWrapper(base_search=BrokenSearch())
        report = wrap.search_with_proof_search(initial_state=None)
        assert "error" in report
        assert "simulated MCTS failure" in report["error"]
        assert "wall_time_s" in report


# ===========================================================================
# integration_with_proof_search (3 tests)
# ===========================================================================


class TestIntegrationWithProofSearch:
    def test_integration_wall_time(self) -> None:
        """Integration smoke harness records wall time + per-run counts."""

        # Provide a stub that returns 3 candidates each call.
        class StubProof:
            tile_library = ["stub_root"]

            def search(self, initial_state, max_depth: int = 3):
                return ["c1", "c2", "c3"]

            def score_final(self, state):
                return 0.5

        report = integration_with_proof_search(
            search_proof=StubProof(),
            n_runs=3,
            max_depth=2,
        )
        assert report["wall_time_s"] >= 0.0
        assert report["per_run_candidate_count"] == [3, 3, 3]
        assert report["n_runs"] == 3
        assert report["max_depth"] == 2
        assert "variance_comparison" in report

    def test_integration_with_empty_tile_library(self) -> None:
        """Empty tile library surfaces an ``error`` key in the report."""

        class EmptyProof:
            tile_library = []

        report = integration_with_proof_search(
            search_proof=EmptyProof(),
            n_runs=1,
        )
        assert "error" in report

    def test_integration_rejects_bad_n_runs(self) -> None:
        """``n_runs < 1`` raises ``VariancePUCTError``."""
        with pytest.raises(VariancePUCTError):
            integration_with_proof_search(
                search_proof=object(),
                n_runs=0,
            )
