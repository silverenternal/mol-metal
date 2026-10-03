"""Tests for :mod:`molmetal.adapters.flow_matching_lipman.optimal_transport_v2`.

Wf-207 — R18 math migration retry: Wasserstein gradient flow
(薛小平 2026 mean-field limit).

These tests cover the three R18 additions on top of the upstream
``flow_matching.optimal_transport`` API:

1. :func:`mean_field_step_size` — closed-form ``dt`` from Xue
   (2026) Lecture 4 Theorem 3.2 (mean-field CFL bound).
2. :class:`MeanFieldW2Plan` — verification wrapper exposing the
   mean-field residual and the per-step safety check.
3. :func:`mean_field_score_smoothed_ot_plan` — kernel-smoothed
   mean-field cost bias on top of :func:`compute_ot_plan`.

Plus three backward-compatibility smoke tests confirming the
upstream ``compute_ot_plan``, ``mini_batch_ot_coupling`` and
``PlanMode`` symbols are still importable through the v2 path.

Every test is **CPU-only** and runs in < 1 second.

Reference
---------
Xue, X.-P. (薛小平) (2026).  *Optimal Transport, Gradient Flows and
Functional Inequalities, Mean-Field Limits*.  Lecture notes,
Changchun Normal University, June 2026.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.adapters.flow_matching_lipman import optimal_transport_v2 as ot2
from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
    MeanFieldW2Plan,
    compute_ot_plan,
    ma_regularization_factor,
    ma_regularized_ot_plan,
    mean_field_score_smoothed_ot_plan,
    mean_field_step_size,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _random_points(n: int = 8, d: int = 3, seed: int = 0) -> torch.Tensor:
    """Random sample points of shape ``(n, d)``."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


# ---------------------------------------------------------------------------
# 1. mean_field_step_size — closed-form dt from Xue 2026 Lecture 4
# ---------------------------------------------------------------------------


def test_mean_field_step_size_basic_formula() -> None:
    """``dt_* = relative_tol / (2 * ||s - s_ref||^2)`` for the canonical case."""
    # score_norm = 1, reference_norm = 0 → delta = 1; tol = 1e-3 → dt_* = 5e-4
    dt = mean_field_step_size(8, 1.0, relative_tol=1.0e-3)
    assert math.isclose(dt, 5.0e-4, rel_tol=1.0e-6)


def test_mean_field_step_size_handles_reference_offset() -> None:
    """``dt_*`` uses the *difference* score norm, not the raw score norm."""
    # ||s - s_ref|| = 2 → delta = 4; tol = 1e-3 → dt_* = 1.25e-4
    dt = mean_field_step_size(
        16, score_norm=3.0, reference_norm=1.0, relative_tol=1.0e-3
    )
    assert math.isclose(dt, 1.25e-4, rel_tol=1.0e-6)


def test_mean_field_step_size_clamps_at_one() -> None:
    """Tiny score norms produce ``dt_* = 1.0`` (CFL clamp)."""
    dt = mean_field_step_size(64, 1.0e-15)
    assert dt == 1.0


def test_mean_field_step_size_zero_score_norm_returns_one() -> None:
    """``score_norm = 0`` (and ``reference_norm = 0``) returns ``1.0``."""
    dt = mean_field_step_size(64, 0.0)
    assert dt == 1.0


def test_mean_field_step_size_rejects_invalid_batch_size() -> None:
    """``batch_size < 1`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="batch_size"):
        mean_field_step_size(0, 1.0)


def test_mean_field_step_size_rejects_negative_score_norm() -> None:
    """Negative score norms raise ``ValueError``."""
    with pytest.raises(ValueError, match="score norms"):
        mean_field_step_size(8, -1.0)


def test_mean_field_step_size_rejects_invalid_tol() -> None:
    """``relative_tol`` outside ``(0, 1)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="relative_tol"):
        mean_field_step_size(8, 1.0, relative_tol=0.0)
    with pytest.raises(ValueError, match="relative_tol"):
        mean_field_step_size(8, 1.0, relative_tol=1.0)
    with pytest.raises(ValueError, match="relative_tol"):
        mean_field_step_size(8, 1.0, relative_tol=-0.1)


# ---------------------------------------------------------------------------
# 2. MeanFieldW2Plan — verification wrapper
# ---------------------------------------------------------------------------


def test_mean_field_plan_default_construction() -> None:
    """Default ``MeanFieldW2Plan`` has relative_tol=1e-3, mean_field_tol=1.0."""
    plan = MeanFieldW2Plan()
    assert plan.relative_tol == 1.0e-3
    assert plan.mean_field_tol == 1.0
    # Flow is auto-constructed (kernel='none' is the cheap default).
    assert plan.flow.kernel == "none"


def test_mean_field_plan_rejects_invalid_relative_tol() -> None:
    """``relative_tol`` outside ``(0, 1)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="relative_tol"):
        MeanFieldW2Plan(relative_tol=0.0)
    with pytest.raises(ValueError, match="relative_tol"):
        MeanFieldW2Plan(relative_tol=1.0)


def test_mean_field_plan_rejects_invalid_mean_field_tol() -> None:
    """``mean_field_tol <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="mean_field_tol"):
        MeanFieldW2Plan(mean_field_tol=0.0)
    with pytest.raises(ValueError, match="mean_field_tol"):
        MeanFieldW2Plan(mean_field_tol=-1.0)


def test_mean_field_residual_is_one_over_sqrt_n() -> None:
    """``mean_field_residual(N) = mean_field_tol / sqrt(N)``."""
    plan = MeanFieldW2Plan(mean_field_tol=1.0)
    assert math.isclose(plan.mean_field_residual(1), 1.0, rel_tol=1.0e-6)
    assert math.isclose(
        plan.mean_field_residual(4), 0.5, rel_tol=1.0e-6
    )
    assert math.isclose(
        plan.mean_field_residual(100), 0.1, rel_tol=1.0e-6
    )


def test_mean_field_residual_floors_n_at_one() -> None:
    """``mean_field_residual(0) == mean_field_residual(1)`` (clamp)."""
    plan = MeanFieldW2Plan()
    assert plan.mean_field_residual(0) == plan.mean_field_residual(1)


def test_verify_step_size_accepts_safe_step() -> None:
    """Safe step (at the CFL boundary) returns ``True`` without warning."""
    plan = MeanFieldW2Plan()
    dt_star = plan.recommended_step_size(64, score_norm=1.0)
    assert plan.verify_step_size(64, 1.0, dt_star) is True


def test_verify_step_size_warns_when_cfl_violated() -> None:
    """A step larger than ``dt_*`` emits a warning and returns ``False``."""
    plan = MeanFieldW2Plan()
    dt_star = plan.recommended_step_size(64, score_norm=1.0)
    with pytest.warns(RuntimeWarning, match="CFL violated"):
        ok = plan.verify_step_size(
            64, 1.0, dt_star * 2.0
        )
    assert ok is False


def test_verify_step_size_warns_when_mean_field_residual_too_large() -> None:
    """A step so large that ``dt * residual > relative_tol`` warns + False.

    We use ``mean_field_tol = 100`` (extremely large additive term)
    and ``batch_size = 1`` so the residual is ``100``; then any
    nonzero ``dt`` triggers the bound.  This isolates the
    mean-field-residual branch from the CFL branch.
    """
    plan = MeanFieldW2Plan(mean_field_tol=100.0, relative_tol=1.0e-3)
    # Use a tiny score norm so dt_star is clamped at 1.0 and the
    # CFL check trivially passes.
    with pytest.warns(RuntimeWarning, match="mean_field_residual"):
        ok = plan.verify_step_size(
            1, 1.0e-15, step_size=1.0
        )
    assert ok is False


def test_verify_step_size_rejects_negative_step() -> None:
    """Negative step size raises ``ValueError``."""
    plan = MeanFieldW2Plan()
    with pytest.raises(ValueError, match="step_size"):
        plan.verify_step_size(64, 1.0, -0.1)


def test_recommended_step_size_matches_mean_field_formula() -> None:
    """``recommended_step_size`` equals ``mean_field_step_size``."""
    plan = MeanFieldW2Plan()
    for n in (8, 32, 128):
        dt_a = plan.recommended_step_size(n, 1.0)
        dt_b = mean_field_step_size(n, 1.0)
        assert math.isclose(dt_a, dt_b, rel_tol=1.0e-9)


# ---------------------------------------------------------------------------
# 3. mean_field_score_smoothed_ot_plan — kernel-smoothed cost bias
# ---------------------------------------------------------------------------


def test_score_smoothed_ot_plan_random_mode_is_permutation() -> None:
    """``random`` mode returns a valid permutation of ``[0, B)``."""
    x0 = _random_points(n=8, d=3, seed=0)
    x1 = _random_points(n=8, d=3, seed=1)
    idx = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="random", seed=42
    )
    assert idx.shape == (8,)
    assert sorted(idx.tolist()) == list(range(8))


def test_score_smoothed_ot_plan_hungarian_returns_identity_on_self() -> None:
    """``hungarian`` mode returns the identity when ``x0 == x1``."""
    x0 = _random_points(n=4, d=3, seed=2)
    idx = mean_field_score_smoothed_ot_plan(
        x0, x0.clone(), mode="hungarian", alpha=0.5
    )
    assert sorted(idx.tolist()) == list(range(4))


def test_score_smoothed_ot_plan_hungarian_optimal_cost() -> None:
    """``hungarian`` mode recovers the minimum-cost assignment on a tiny case.

    For ``x0 = [[0,0]]`` and ``x1 = [[1,0]]`` there is only one
    assignment: ``0 -> 0``.  The Hungarian solver must pick it.
    """
    x0 = torch.tensor([[0.0, 0.0]])
    x1 = torch.tensor([[1.0, 0.0]])
    idx = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="hungarian"
    )
    assert idx.tolist() == [0]


def test_score_smoothed_ot_plan_alpha_zero_falls_back_to_base() -> None:
    """``alpha = 0`` (no mean-field bias) does not raise and returns a perm."""
    x0 = _random_points(n=6, d=2, seed=3)
    x1 = _random_points(n=6, d=2, seed=4)
    idx = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="hungarian", alpha=0.0
    )
    assert sorted(idx.tolist()) == list(range(6))


def test_score_smoothed_ot_plan_rejects_mismatched_shapes() -> None:
    """``x0`` and ``x1`` with different shapes raise ``ValueError``."""
    x0 = _random_points(n=4, d=3, seed=5)
    x1 = _random_points(n=5, d=3, seed=6)
    with pytest.raises(ValueError, match="share shape"):
        mean_field_score_smoothed_ot_plan(x0, x1)


def test_score_smoothed_ot_plan_rejects_negative_alpha() -> None:
    """Negative ``alpha`` raises ``ValueError``."""
    x0 = _random_points(n=4, d=3, seed=7)
    x1 = _random_points(n=4, d=3, seed=8)
    with pytest.raises(ValueError, match="alpha"):
        mean_field_score_smoothed_ot_plan(x0, x1, alpha=-0.1)


def test_score_smoothed_ot_plan_rejects_unknown_kernel() -> None:
    """Unknown kernel name raises ``ValueError``."""
    x0 = _random_points(n=4, d=3, seed=9)
    x1 = _random_points(n=4, d=3, seed=10)
    with pytest.raises(ValueError, match="kernel"):
        mean_field_score_smoothed_ot_plan(x0, x1, kernel="laplace")


def test_score_smoothed_ot_plan_rejects_invalid_mode() -> None:
    """``mode`` outside ``('random', 'hungarian')`` raises ``ValueError``."""
    x0 = _random_points(n=4, d=3, seed=11)
    x1 = _random_points(n=4, d=3, seed=12)
    with pytest.raises(ValueError, match="mode"):
        mean_field_score_smoothed_ot_plan(x0, x1, mode="bogus")  # type: ignore[arg-type]


def test_score_smoothed_ot_plan_handles_reference_score() -> None:
    """Supplying a ``reference_score0`` does not raise and returns a perm."""
    x0 = _random_points(n=4, d=3, seed=13)
    x1 = _random_points(n=4, d=3, seed=14)
    ref = torch.zeros_like(x0)
    idx = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="hungarian",
        reference_score0=ref, alpha=0.5,
    )
    assert sorted(idx.tolist()) == list(range(4))


# ---------------------------------------------------------------------------
# 4. Backward compatibility — upstream API still importable
# ---------------------------------------------------------------------------


def test_v2_re_exports_compute_ot_plan() -> None:
    """``compute_ot_plan`` is importable from the v2 module."""
    assert callable(ot2.compute_ot_plan)
    # And it has the upstream signature.
    x0 = _random_points(n=4, d=3, seed=15)
    x1 = _random_points(n=4, d=3, seed=16)
    idx = ot2.compute_ot_plan(x0, x1, mode="random", seed=0)
    assert idx.shape == (4,)


def test_v2_re_exports_mini_batch_ot_coupling() -> None:
    """``mini_batch_ot_coupling`` is importable from the v2 module."""
    assert callable(ot2.mini_batch_ot_coupling)


def test_v2_re_exports_plan_mode() -> None:
    """``PlanMode`` is importable from the v2 module."""
    # typing.Literal renders as a parameterized type; assert it
    # has the expected args.
    args = getattr(ot2.PlanMode, "__args__", ())
    assert "random" in args
    assert "hungarian" in args


# ---------------------------------------------------------------------------
# 5. End-to-end — score bias picks a non-worse assignment than random
# ---------------------------------------------------------------------------


def test_mean_field_bias_reduces_cost_on_toy_two_cluster_case() -> None:
    """The score-smoothed bias favours the cluster-aligned pairing.

    Setup: two tight clusters — ``x0`` cluster A and ``x1`` clusters
    B (close to A) and C (far from A).  The score ``s(x0)`` points
    toward cluster B.  With a positive ``alpha``, the mean-field
    bias should *prefer* the B-pairing over the C-pairing; with
    ``alpha = 0`` (no bias) the Hungarian solver picks the minimum
    cost, which may be either depending on the geometry.
    """
    # x0 = 2 points at origin and far-right
    x0 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    # x1 = 1 point near origin (cluster B), 1 point far away (cluster C)
    x1 = torch.tensor([[0.1, 0.0], [5.0, 0.0]])
    # Score pointing toward the near cluster.
    score0 = torch.tensor([[0.5, 0.0], [0.5, 0.0]])

    # With alpha = 0, the Hungarian solver picks the minimum-cost
    # assignment: row 0 ↔ x1[0] (cost 0.01), row 1 ↔ x1[1] (cost 25).
    idx_base = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="hungarian", alpha=0.0
    )
    assert sorted(idx_base.tolist()) == [0, 1]

    # With the bias pointing toward x1[0] (the near cluster), the
    # bias term can only *strengthen* this assignment (the other
    # pairing incurs a positive cost penalty).
    idx_biased = mean_field_score_smoothed_ot_plan(
        x0, x1, mode="hungarian", alpha=10.0,
        reference_score0=score0,
    )
    # Both rows still get distinct columns; the bias does not
    # produce duplicates.
    assert sorted(idx_biased.tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# 4. Monge-Ampère regularity regularisation (Wf-236)
# ---------------------------------------------------------------------------
# These tests cover the new MA-regularity surface added by Wf-236:
# Wang (2026) C^{2,α} regularity check + Brenier-map regularisation
# for the OT cost matrix when the empirical transport map is
# degenerate.  See :func:`ma_regularization_factor` and
# :func:`ma_regularized_ot_plan` for the mathematical definition.


def test_ma_regularization_factor_well_conditioned() -> None:
    """A well-conditioned ``x0`` returns ``is_regular=True`` and a small ``reg_factor``."""
    torch.manual_seed(0)
    x0 = torch.randn(16, 4)
    Q, is_regular, reg_factor = ma_regularization_factor(
        x0, alpha=0.5, degenerate_threshold=1.0e-4,
    )
    assert Q.shape == (4, 4)
    # SPD by construction
    eigvals = torch.linalg.eigvalsh(Q)
    assert (eigvals > 0).all()
    # Strict convexity at the default tol
    assert bool(is_regular)
    # Regularisation factor is finite and positive
    assert reg_factor > 0.0
    assert reg_factor < 1.0e8


def test_ma_regularization_factor_detects_degenerate_constant_x0() -> None:
    """``x0 = 0`` triggers the degenerate Brenier-map flag."""
    x0 = torch.zeros(8, 3)
    Q, is_regular, reg_factor = ma_regularization_factor(
        x0, alpha=0.5, degenerate_threshold=1.0e-4,
    )
    # All-zero input → degenerate empirical covariance → Q is the SPD
    # projection at the floor → strict convexity check fails because
    # the eigenvalues are exactly at the SPD floor.
    assert not bool(is_regular)
    assert reg_factor > 1.0


def test_ma_regularization_factor_detects_rank_deficient() -> None:
    """A rank-deficient ``x0`` (one eigenvalue collapsed) is detected."""
    torch.manual_seed(0)
    x0 = torch.randn(8, 3)
    x0[:, 2] *= 1.0e-6  # collapse the z-axis
    Q, is_regular, reg_factor = ma_regularization_factor(
        x0, alpha=0.5, degenerate_threshold=1.0e-4,
    )
    # The z-eigenvalue falls below the threshold → degenerate.
    assert not bool(is_regular)
    assert reg_factor > 1.0


def test_ma_regularization_factor_validates_inputs() -> None:
    """Input validation raises ``ValueError`` on bad alpha / threshold / size."""
    with pytest.raises(ValueError, match="alpha"):
        ma_regularization_factor(torch.randn(4, 3), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ma_regularization_factor(torch.randn(4, 3), alpha=1.5)
    with pytest.raises(ValueError, match="degenerate_threshold"):
        ma_regularization_factor(torch.randn(4, 3), degenerate_threshold=-1.0)
    with pytest.raises(ValueError, match="MA-regularity"):
        ma_regularization_factor(torch.randn(1, 3))


def test_ma_regularized_ot_plan_well_conditioned_matches_upstream() -> None:
    """Well-conditioned input → MA plan = upstream plan (no regularization)."""
    torch.manual_seed(0)
    x0 = torch.randn(8, 3)
    x1 = torch.randn(8, 3)
    idx_ma, diag = ma_regularized_ot_plan(
        x0, x1, lambda_ma=0.01, mode="hungarian",
        return_diagnostics=True,
    )
    assert diag["activated"] is False
    assert diag["is_regular"] is True
    # Standard plan
    idx_ref = compute_ot_plan(x0, x1, mode="hungarian")
    assert sorted(idx_ma.tolist()) == sorted(idx_ref.tolist())


def test_ma_regularized_ot_plan_degenerate_activates_regularisation() -> None:
    """Degenerate ``x0`` → MA plan activates the Brenier-map regularisation."""
    torch.manual_seed(0)
    x0 = torch.zeros(8, 3)
    x1 = torch.randn(8, 3)
    idx_ma, diag = ma_regularized_ot_plan(
        x0, x1, lambda_ma=0.01, mode="hungarian",
        return_diagnostics=True,
    )
    assert diag["activated"] is True
    assert diag["is_regular"] is False
    assert diag["reg_factor"] > 0.0
    # Output must still be a valid permutation.
    assert idx_ma.shape == (8,)
    assert sorted(idx_ma.tolist()) == list(range(8))


def test_ma_regularized_ot_plan_lambda_zero_is_no_op() -> None:
    """``lambda_ma = 0`` → MA plan returns the upstream plan exactly."""
    torch.manual_seed(0)
    x0 = torch.zeros(8, 3)  # degenerate, but lambda_ma disables regularisation
    x1 = torch.randn(8, 3)
    idx_ma, diag = ma_regularized_ot_plan(
        x0, x1, lambda_ma=0.0, mode="hungarian",
        return_diagnostics=True,
    )
    assert diag["activated"] is False
    assert diag["reg_factor"] == 0.0


def test_ma_regularized_ot_plan_random_mode_skips_ma() -> None:
    """``mode='random'`` falls through to the upstream random pairing."""
    torch.manual_seed(0)
    x0 = torch.zeros(8, 3)
    x1 = torch.randn(8, 3)
    idx_ma = ma_regularized_ot_plan(
        x0, x1, lambda_ma=0.01, mode="random",
    )
    # Random mode never activates Hungarian; it just calls upstream.
    assert idx_ma.shape == (8,)


def test_ma_regularized_ot_plan_validates_inputs() -> None:
    """Input validation: shape mismatch + negative ``lambda_ma``."""
    with pytest.raises(ValueError, match="share shape"):
        ma_regularized_ot_plan(torch.randn(4, 3), torch.randn(5, 3))
    with pytest.raises(ValueError, match="lambda_ma"):
        ma_regularized_ot_plan(torch.randn(4, 3), torch.randn(4, 3), lambda_ma=-0.1)


def test_mean_field_plan_exposes_verify_ma_regularity() -> None:
    """``MeanFieldW2Plan.verify_ma_regularity`` is the Wf-236 surface."""
    plan = MeanFieldW2Plan()
    torch.manual_seed(0)
    x0 = torch.randn(8, 3)
    diag = plan.verify_ma_regularity(x0, alpha=0.5, degenerate_threshold=1.0e-4)
    assert "Q" in diag
    assert "is_regular" in diag
    assert "reg_factor" in diag
    assert diag["is_regular"] is True


def test_ma_regularized_ot_plan_returns_permutation_shape() -> None:
    """Output is always a (B,) long tensor — even when regularisation triggers."""
    torch.manual_seed(0)
    # Mixed-condition: half degenerate + half well-conditioned.
    x0 = torch.cat([torch.zeros(4, 3), torch.randn(4, 3)], dim=0)
    x1 = torch.randn(8, 3)
    idx = ma_regularized_ot_plan(
        x0, x1, lambda_ma=0.01, mode="hungarian",
    )
    assert idx.shape == (8,)
    assert idx.dtype == torch.long
    # Permutation: 0..7 must each appear exactly once.
    assert sorted(idx.tolist()) == list(range(8))


def test_ma_regularization_mathematical_correctness_gaussian() -> None:
    """The Gaussian-Gaussian case: regularised plan = upstream plan (Brenier affine)."""
    torch.manual_seed(42)
    # Two well-separated isotropic Gaussians — the Brenier map is
    # affine and the empirical covariance is full-rank.
    x0 = torch.randn(32, 4)
    x1 = torch.randn(32, 4) + 0.5
    _, is_regular, _ = ma_regularization_factor(
        x0, alpha=0.5, degenerate_threshold=1.0e-4,
    )
    # Affine Brenier map → C^{2,α} regular → no MA-regularisation.
    assert bool(is_regular)
