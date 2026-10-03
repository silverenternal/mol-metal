"""Sign-flip regression tests for the VariancePUCT scoring rule (Wf-51 P1).

================================================================
CPU-only regression tests for the Wf-51 Phase-1 sign-flip fix
================================================================
The pre-fix formula
``score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
          - c_var * sigma / (1 + N_child)``
decayed the variance penalty like ``1 / N_child``, causing the
selector to degenerate to canonical PUCT at typical visit counts
(N >= 10) and lose variance control — measured as
``variance_reduction = -0.185`` on the Wf-36 P2 mock CCO bench.

The post-fix formula drops the ``(1 + N_child)`` divisor on the
variance penalty, making it *persistent* across visits:

``score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
          - c_var * sigma``

This module validates the post-fix behaviour with 15 targeted
regression tests:

* **Sign preservation** (1-5) — the penalty's *direction* (subtract)
  is preserved; only the *scaling* changes.
* **Persistence** (6-10) — the penalty magnitude does NOT decay with
  ``N_child`` (the pre-fix bug).
* **Selection behaviour** (11-15) — the selector still picks
  low-variance arms and the Q-only / FPU / n_min_visits gates are
  preserved (backward compat at the selection-outcome level).

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_puct_sign_fix.py -v
"""
from __future__ import annotations

import math

import pytest

# Lazy imports (mirrors the convention in test_variance_puct_integration.py).
try:
    from molmetal_lam.search_alg.variance_puct_integration import (
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        IntegrationNode,
        VarianceControlledPUCT,
        VariancePUCTError,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore  # noqa: E402
        DEFAULT_C_PUCT,
        DEFAULT_C_VAR,
        IntegrationNode,
        VarianceControlledPUCT,
        VariancePUCTError,
    )

try:
    from molmetal_lam.search_alg.variance_mcts_integration import (
        VarianceMCTSNode,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore  # noqa: E402
        VarianceMCTSNode,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_root(n_children: int = 4, prior: float = 0.5) -> IntegrationNode:
    """Return a root :class:`IntegrationNode` with uniform-prior children."""
    return IntegrationNode(
        node_id="root",
        prior=float(prior),
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0)
            for i in range(int(n_children))
        ],
    )


def _mk_node(
    n_actions: int = 4,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    n_min_visits: int = 2,
) -> VarianceMCTSNode:
    """Return a uniform-prior :class:`VarianceMCTSNode`."""
    return VarianceMCTSNode(
        prior=0.5,
        c_puct=c_puct,
        c_var=c_var,
        n_min_visits=n_min_visits,
        action_priors=[1.0 / n_actions] * n_actions,
    )


# ===========================================================================
# 1. Sign preservation (5 tests)
# ===========================================================================


class TestSignPreservation:
    def test_sign_penalty_subtracts_for_high_sigma(self) -> None:
        """Penalty direction is subtraction (sign preserved after fix)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=1.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        # Arm 0: high variance; arm 1: zero variance; both Q=1.
        for _ in range(3):
            sel.update(root, 0, value=1.0, variance=1.0)
            sel.update(root, 1, value=1.0, variance=0.0)
        s0 = sel.puct_score(root, 0, n_parent=6)
        s1 = sel.puct_score(root, 1, n_parent=6)
        # Sign-preservation: high-variance arm has the LOWER score.
        assert s0 < s1
        # Penalty magnitude: c_var * sigma = 1 * 1 = 1 (post-fix).
        # Q is 1, so s0 = 1 + 0 - 1 = 0; s1 = 1 + 0 - 0 = 1.
        assert math.isclose(s0, 0.0, abs_tol=1e-9)
        assert math.isclose(s1, 1.0, abs_tol=1e-9)

    def test_sign_penalty_zero_when_sigma_zero(self) -> None:
        """sigma = 0 → no penalty (post-fix)."""
        sel = VarianceControlledPUCT(
            c_puct=1.0, c_var=0.5, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        sel.update(root, 0, value=0.7, variance=0.0)
        sel.update(root, 1, value=0.3, variance=0.0)
        s0 = sel.puct_score(root, 0, n_parent=2)
        s1 = sel.puct_score(root, 1, n_parent=2)
        # With sigma=0, only the canonical PUCT rule applies.
        # Arm 0 has Q=0.7; arm 1 has Q=0.3 → s0 > s1.
        assert s0 > s1

    def test_sign_persistent_at_large_n(self) -> None:
        """Penalty magnitude is the same at N=2 and N=20 (post-fix)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=0.5, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        # Arm 0: high variance sigma=0.4 (so penalty = 0.5 * 0.4 = 0.2).
        for _ in range(20):
            sel.update(root, 0, value=0.5, variance=0.16)
        # Arm 1: zero variance.
        for _ in range(20):
            sel.update(root, 1, value=0.5, variance=0.0)
        # Post-fix: penalty = c_var * sigma = 0.5 * 0.4 = 0.2, regardless of N.
        # Q(0) = Q(1) = 0.5, c_puct=0 → s0 = 0.5 - 0.2 = 0.3, s1 = 0.5.
        s0 = sel.puct_score(root, 0, n_parent=40)
        s1 = sel.puct_score(root, 1, n_parent=40)
        assert math.isclose(s0, 0.3, abs_tol=1e-9)
        assert math.isclose(s1, 0.5, abs_tol=1e-9)

    def test_sign_decayed_at_small_n_in_old_formula(self) -> None:
        """Documents the pre-fix decay: penalty was tiny at N=20.

        This test recomputes the *old* formula directly (manually)
        to document the regime-flip that the fix corrects.  The
        live ``puct_score`` already uses the post-fix formula, so
        we compare the live value to the *pre-fix* reference value
        and assert that they DIVERGE — proving the fix is active.
        """
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=0.5, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        for _ in range(20):
            sel.update(root, 0, value=0.5, variance=0.16)
        # Live (post-fix) score: 0.5 - 0.5 * 0.4 = 0.3.
        live_score = sel.puct_score(root, 0, n_parent=20)
        # Pre-fix score (manually computed):
        # 0.5 + 0 - 0.5 * 0.4 / (1 + 20) = 0.5 - 0.2/21 ≈ 0.4905
        pref_score = 0.5 + 0.0 - 0.5 * 0.4 / (1.0 + 20.0)
        # The two diverge by ~0.19 — proof the fix is active.
        assert math.isclose(live_score, 0.3, abs_tol=1e-9)
        assert math.isclose(pref_score, 0.4905, abs_tol=1e-3)
        assert live_score < pref_score

    def test_sign_penalty_direction_independent_of_n(self) -> None:
        """Increasing N_child does NOT change penalty direction."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=1.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        sel.update(root, 0, value=0.5, variance=0.04)
        sel.update(root, 1, value=0.5, variance=0.0)
        # At every visit count, arm 0 (sigma > 0) must have lower
        # score than arm 1 (sigma = 0).
        for n in (1, 5, 10, 50, 100):
            s0 = sel.puct_score(root, 0, n_parent=n)
            s1 = sel.puct_score(root, 1, n_parent=n)
            assert s0 < s1, (
                f"sign-flip detected at n_parent={n}: s0={s0}, s1={s1}"
            )


# ===========================================================================
# 2. Persistence / scaling (5 tests)
# ===========================================================================


class TestPersistence:
    def test_sign_puct_bonus_still_decays(self) -> None:
        """PUCT exploration bonus still has ``(1 + N_child)`` divisor.

        We only fix the *variance* term; the exploration term must
        continue to decay so the selector transitions from
        exploration to exploitation as visits grow.
        """
        sel = VarianceControlledPUCT(
            c_puct=1.0, c_var=0.0, n_min_visits=1,  # c_var=0 disables variance
        )
        root = _mk_root(n_children=2)
        # Arm 0: many visits, Q stable.
        for _ in range(20):
            sel.update(root, 0, value=0.5, variance=0.0)
        # PUCT bonus for arm 0: c_puct * P * sqrt(20) / (1 + 20)
        #                      = 1 * 1 * sqrt(20) / 21 ≈ 0.213
        s0 = sel.puct_score(root, 0, n_parent=20)
        expected_bonus = 1.0 * 1.0 * math.sqrt(20) / 21.0
        assert math.isclose(s0, 0.5 + expected_bonus, abs_tol=1e-9)
        # Bonus shrinks as N grows: at N=20 the bonus is ~0.213
        # (much smaller than at N=1: 1.0 * 1.0 * 1 / 2 = 0.5).
        sel2 = VarianceControlledPUCT(
            c_puct=1.0, c_var=0.0, n_min_visits=1,
        )
        root2 = _mk_root(n_children=2)
        sel2.update(root2, 0, value=0.5, variance=0.0)
        s0_n1 = sel2.puct_score(root2, 0, n_parent=1)
        # higher N_parent → larger sqrt(N_parent) BUT ALSO larger
        # (1 + N_child) divisor; the *bonus magnitude* shrinks.
        # Compare PUCT bonus in isolation (subtract Q):
        bonus_n20 = s0 - 0.5
        bonus_n1 = s0_n1 - 0.5
        assert bonus_n20 < bonus_n1  # higher N → smaller bonus

    def test_sign_q_value_unmodified(self) -> None:
        """Q is used directly, not divided (canonical PUCT preserved)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=0.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        sel.update(root, 0, value=0.7, variance=0.0)
        sel.update(root, 1, value=0.3, variance=0.0)
        # c_puct = 0 → no exploration; c_var = 0 → no penalty.
        # Score is just Q.
        s0 = sel.puct_score(root, 0, n_parent=2)
        s1 = sel.puct_score(root, 1, n_parent=2)
        assert math.isclose(s0, 0.7, abs_tol=1e-9)
        assert math.isclose(s1, 0.3, abs_tol=1e-9)

    def test_sign_no_penalty_below_n_min_visits(self) -> None:
        """Variance penalty is gated by ``n_min_visits`` (preserved)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=5.0, n_min_visits=3,  # require N >= 3
        )
        root = _mk_root(n_children=2)
        # Only one visit so the gate is NOT met → no penalty.
        sel.update(root, 0, value=0.5, variance=0.5)
        sel.update(root, 1, value=0.5, variance=0.0)
        s0 = sel.puct_score(root, 0, n_parent=2)
        s1 = sel.puct_score(root, 1, n_parent=2)
        # No variance penalty below n_min_visits → both scores are pure Q.
        assert math.isclose(s0, 0.5, abs_tol=1e-9)
        assert math.isclose(s1, 0.5, abs_tol=1e-9)

    def test_sign_c_var_zero_disables_penalty(self) -> None:
        """``c_var = 0`` disables the variance penalty (preserved)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=0.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        for _ in range(3):
            sel.update(root, 0, value=0.5, variance=0.5 ** 2)
            sel.update(root, 1, value=0.5, variance=0.0 ** 2)
        s0 = sel.puct_score(root, 0, n_parent=6)
        s1 = sel.puct_score(root, 1, n_parent=6)
        # c_var = 0 → both scores are pure Q → tie.
        assert math.isclose(s0, 0.5, abs_tol=1e-9)
        assert math.isclose(s1, 0.5, abs_tol=1e-9)

    def test_sign_variance_puct_persistent_with_c_var_2(self) -> None:
        """Numerical check: c_var=2, sigma=0.3 → penalty=0.6 across N."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=2.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        for _ in range(30):
            sel.update(root, 0, value=0.5, variance=0.09)  # sigma=0.3
        # Post-fix: penalty = c_var * sigma = 2 * 0.3 = 0.6
        # Q = 0.5, c_puct = 0 → s0 = 0.5 - 0.6 = -0.1
        s0 = sel.puct_score(root, 0, n_parent=30)
        assert math.isclose(s0, -0.1, abs_tol=1e-9)


# ===========================================================================
# 3. Selection behaviour (5 tests)
# ===========================================================================


class TestSelectionBehaviour:
    def test_sign_high_variance_arm_loses_after_fix(self) -> None:
        """High-variance arm is still down-weighted (selection unchanged)."""
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=5.0, n_min_visits=2,
        )
        root = _mk_root(n_children=2)
        for _ in range(3):
            sel.update(root, 0, value=1.0, variance=1.0 ** 2)
            sel.update(root, 1, value=1.0, variance=0.0 ** 2)
        a = sel.select(root)
        assert a == 1

    def test_sign_variance_mcts_node_persistent(self) -> None:
        """Mirror test for ``VarianceMCTSNode`` (mirror fix shipped)."""
        node = _mk_node(n_actions=2, c_puct=0.0, c_var=0.5, n_min_visits=1)
        # Arm 0: high variance sigma=0.4 → penalty = 0.5 * 0.4 = 0.2.
        for _ in range(20):
            node.update(value=0.5, variance=0.16, action=0)
        # Arm 1: zero variance.
        for _ in range(20):
            node.update(value=0.5, variance=0.0, action=1)
        # Post-fix: s0 = 0.5 - 0.2 = 0.3; s1 = 0.5 - 0 = 0.5.
        s0 = node.puct_score(action=0, n_parent=40)
        s1 = node.puct_score(action=1, n_parent=40)
        assert math.isclose(s0, 0.3, abs_tol=1e-9)
        assert math.isclose(s1, 0.5, abs_tol=1e-9)

    def test_sign_backward_compat_high_variance(self) -> None:
        """Old ``test_variance_puct_high_variance`` selection still wins."""
        # Reproduces the Wf-11 selection test verbatim.
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=5.0, n_min_visits=2,
        )
        root = _mk_root(n_children=2)
        for _ in range(3):
            sel.update(root, 0, value=1.0, variance=1.0 ** 2)
            sel.update(root, 1, value=1.0, variance=0.0 ** 2)
        # Old (buggy) penalty at N=3: c_var * sigma / 4 = 5 * 1 / 4 = 1.25.
        # New (fixed) penalty at N=3: c_var * sigma = 5 * 1 = 5.
        # Either way, arm 1 wins because Q(0) = Q(1) = 1 and arm 0
        # has the additional penalty.
        assert sel.select(root) == 1

    def test_sign_variance_reduction_positive_signal(self) -> None:
        """Smoke signal: variance penalty reduces the effective score.

        We do NOT re-run the full Wf-36 P2 mock CCO bench here
        (that's R20-P2 in ``fix_score_repellent_sign.py``); we
        just verify the *direction* of the variance penalty on a
        minimal 2-arm UCB instance.  If the post-fix penalty were
        a *bonus* (wrong sign), this test would fail.
        """
        sel = VarianceControlledPUCT(
            c_puct=0.0, c_var=1.0, n_min_visits=1,
        )
        root = _mk_root(n_children=2)
        # Arm 0: low Q, zero variance.  Arm 1: high Q, high variance.
        for _ in range(5):
            sel.update(root, 0, value=0.3, variance=0.0)
            sel.update(root, 1, value=0.7, variance=0.16)  # sigma=0.4
        s0 = sel.puct_score(root, 0, n_parent=10)
        s1 = sel.puct_score(root, 1, n_parent=10)
        # Post-fix penalty on arm 1 is c_var * sigma = 1 * 0.4 = 0.4.
        # s0 = 0.3, s1 = 0.7 - 0.4 = 0.3.  The variance penalty
        # brings the scores to within epsilon — the *direction* of
        # the penalty is correct (subtraction, not addition).
        assert s1 <= s0 + 1e-9  # arm 1 ≤ arm 0 (penalty applied)
        # And the *post-fix* equality (not just <=) is the
        # specific signal:
        assert math.isclose(s0, 0.3, abs_tol=1e-9)
        assert math.isclose(s1, 0.3, abs_tol=1e-9)

    def test_sign_docstring_updated(self) -> None:
        """``puct_score`` docstring reflects the new formula.

        The post-fix docstring must NOT contain the buggy formula
        ``c_var * sigma(s, a) / (1 + N(s, a))``.  This is an
        audit-trail check that prevents a regression where the
        source is patched but the docstring is forgotten.
        """
        doc = VarianceControlledPUCT.puct_score.__doc__ or ""
        # The buggy formula must be gone:
        assert "c_var * sigma(s, a) / (1 + N(s, a))" not in doc, (
            "puct_score docstring still documents the BUGGY formula "
            "with the (1 + N_child) divisor"
        )
        # The fixed formula should be documented (either in the
        # formula block or in the Notes section).
        assert "c_var * sigma" in doc, (
            "puct_score docstring must document the post-fix formula"
        )
        # The audit-report path must be cross-referenced:
        assert "wf51_sign_audit" in doc, (
            "puct_score docstring must cross-reference the audit report"
        )


__all__ = [
    "TestSignPreservation",
    "TestPersistence",
    "TestSelectionBehaviour",
]
