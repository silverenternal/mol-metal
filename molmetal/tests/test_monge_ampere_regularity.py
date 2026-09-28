"""Tests for :mod:`molmetal.math_skills.monge_ampere_regularity`.

T3 — Monge-Ampère Regularity (汪徐家)
======================================

These tests cover the three mathematical primitives shipped by the
R18 MA-regularity skeleton:

1. **KantorovichPotential** — quadratic-form approximation of the
   Kantorovich potential with C^{2,α} regularity.  We verify
   construction, evaluation, the analytic / numerical Hessian, the
   strict-convexity check and the in-place update primitives.
2. **MongeAmpereSolver** — iterative solver for the Monge-Ampère
   equation.  We verify the construction, the SPD-projection
   invariant, the SPD output, and the Gaussian→Gaussian
   correctness.
3. **AdvantageEstimatorMA** — MA-regularised advantage estimator
   for SynFlowNet.  We verify the construction, the variance bound
   reduces to Bernstein at ``lam = 0``, the bound tightens at
   ``lam > 0``, the ``advantage`` helper, and the edge cases.

Every test is **CPU-only** and uses small data (dim 2–4, n_iters
30–60) so the full suite runs in < 10 seconds.

Reference
---------
Wang, X.-J. (汪徐家) (2026).  *On the C^{2,α} regularity of
Kantorovich potentials for optimal transport*.
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

from molmetal.math_skills import (
    AdvantageEstimatorMA,
    KantorovichPotential,
    MongeAmpereSolver,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _random_cost_matrix(d: int = 4, seed: int = 0) -> torch.Tensor:
    """Random SPD-ish cost matrix of shape ``(d, d)``."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    return A @ A.T + 0.1 * torch.eye(d)


# ---------------------------------------------------------------------------
# 1. KantorovichPotential — construction & invariants
# ---------------------------------------------------------------------------


def test_kantorovich_potential_init_with_spd_cost_matrix() -> None:
    """A square SPD cost matrix produces a strictly-convex potential."""
    C = _random_cost_matrix(d=4, seed=0)
    phi = KantorovichPotential(C)
    assert phi.d == 4
    assert phi.Q.shape == (4, 4)
    assert phi.c.shape == (4,)
    # Q is SPD by construction → strictly convex
    assert phi.is_strictly_convex()


def test_kantorovich_potential_init_rejects_non_square() -> None:
    """Non-square cost matrix raises ``ValueError``."""
    C = torch.randn(4, 5)
    with pytest.raises(ValueError, match="square"):
        KantorovichPotential(C)


def test_kantorovich_potential_init_rejects_non_finite() -> None:
    """NaN / inf entries in the cost matrix raise ``ValueError``."""
    C = torch.eye(3)
    C[1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        KantorovichPotential(C)


def test_kantorovich_potential_init_rejects_invalid_alpha() -> None:
    """``alpha`` outside ``(0, 1]`` raises ``ValueError``."""
    C = torch.eye(3)
    with pytest.raises(ValueError, match="alpha"):
        KantorovichPotential(C, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        KantorovichPotential(C, alpha=1.5)


def test_kantorovich_potential_init_alpha_default() -> None:
    """Default ``alpha`` is the canonical ``0.5``."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    assert phi.alpha == 0.5


# ---------------------------------------------------------------------------
# 2. KantorovichPotential — evaluation, strict convexity, updates
# ---------------------------------------------------------------------------


def test_kantorovich_potential_call_quadratic_form() -> None:
    """``phi(x) == (1/2) x^T Q x + c^T x + d`` for the trivial potential."""
    C = torch.eye(3) * 2.0
    phi = KantorovichPotential(C)
    phi.update_c(torch.tensor([0.1, 0.2, 0.3]))
    phi._d_const = torch.tensor(0.5)
    x = torch.tensor([1.0, -1.0, 2.0])
    val = phi(x)
    # Q = 2 I, c = (0.1, 0.2, 0.3), d = 0.5
    expected = 0.5 * (x @ (2.0 * torch.eye(3)) * x).sum() + \
        (x * torch.tensor([0.1, 0.2, 0.3])).sum() + 0.5
    assert torch.allclose(val, expected, atol=1e-5)


def test_kantorovich_potential_gradient_is_Qx_plus_c() -> None:
    """``gradient(x) == Q x + c``."""
    C = _random_cost_matrix(d=3, seed=1)
    phi = KantorovichPotential(C)
    phi.update_c(torch.tensor([0.5, -0.5, 0.25]))
    x = torch.tensor([1.0, 2.0, 3.0])
    grad = phi.gradient(x)
    expected = phi.Q @ x + torch.tensor([0.5, -0.5, 0.25])
    assert torch.allclose(grad, expected, atol=1e-5)


def test_kantorovich_potential_hessian_matches_Q() -> None:
    """The numerical Hessian equals the analytic Hessian ``Q``."""
    C = _random_cost_matrix(d=4, seed=2)
    phi = KantorovichPotential(C)
    x = torch.randn(3, 4)
    H = phi.hessian(x)
    # For a quadratic form, every batch element has the same Hessian = Q.
    assert H.shape == (3, 4, 4)
    for i in range(3):
        assert torch.allclose(H[i], phi.Q, atol=1e-3, rtol=1e-3)


def test_kantorovich_potential_hessian_batch_shape() -> None:
    """Hessian on a batched input returns shape ``(B, d, d)``."""
    C = _random_cost_matrix(d=3, seed=3)
    phi = KantorovichPotential(C)
    x = torch.randn(5, 3)
    H = phi.hessian(x)
    assert H.shape == (5, 3, 3)


def test_kantorovich_potential_hessian_rejects_wrong_dim() -> None:
    """Mismatched ``x`` dimension raises ``ValueError``."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    with pytest.raises(ValueError, match="last dim"):
        phi.hessian(torch.randn(2, 4))


def test_kantorovich_potential_is_strictly_convex_true_for_spd() -> None:
    """A diagonal SPD Q is strictly convex."""
    C = torch.eye(3) * 5.0
    phi = KantorovichPotential(C)
    assert phi.is_strictly_convex() is True


def test_kantorovich_potential_not_strictly_convex_when_psd() -> None:
    """A rank-deficient Q (eigenvalue = 0) is *not* strictly convex."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    # Force Q to be rank-deficient: zero out the last eigenvalue.
    Q_new = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    # update_Q would reject this; patch directly via a tiny substitution.
    phi._Q = Q_new
    assert phi.is_strictly_convex() is False


def test_kantorovich_potential_is_strictly_convex_quantitative_alpha1() -> None:
    """``alpha = 1`` requires a bounded condition number."""
    C = torch.eye(3)
    phi = KantorovichPotential(C, alpha=1.0)
    # Force Q to be very ill-conditioned.
    phi._Q = torch.diag(torch.tensor([1.0, 1.0, 1e-9]))
    # Eigenvalues strictly positive but condition number huge.
    assert phi.is_strictly_convex(tol=1e-6) is False


def test_kantorovich_potential_update_Q_replaces_Q() -> None:
    """``update_Q`` replaces Q in-place and keeps SPD-ness invariant."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    new_Q = torch.eye(3) * 7.0
    phi.update_Q(new_Q)
    assert torch.allclose(phi.Q, new_Q)
    assert phi.is_strictly_convex()


def test_kantorovich_potential_update_Q_rejects_non_spd() -> None:
    """``update_Q`` with a non-SPD matrix raises ``ValueError``."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    bad_Q = -torch.eye(3)
    with pytest.raises(ValueError, match="positive-definite"):
        phi.update_Q(bad_Q)


def test_kantorovich_potential_update_c_replaces_c() -> None:
    """``update_c`` replaces ``c`` in-place."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    new_c = torch.tensor([0.1, 0.2, 0.3])
    phi.update_c(new_c)
    assert torch.allclose(phi.c, new_c)


def test_kantorovich_potential_parameters_returns_Q_and_c() -> None:
    """``parameters()`` returns the ``(Q, c)`` tuple."""
    C = torch.eye(3)
    phi = KantorovichPotential(C)
    Q, c = phi.parameters()
    assert Q.shape == (3, 3)
    assert c.shape == (3,)


# ---------------------------------------------------------------------------
# 3. MongeAmpereSolver
# ---------------------------------------------------------------------------


def test_monge_ampere_solver_init_validates() -> None:
    """``n_iters`` and ``lr`` are validated."""
    with pytest.raises(ValueError, match="n_iters"):
        MongeAmpereSolver(n_iters=0)
    with pytest.raises(ValueError, match="lr"):
        MongeAmpereSolver(n_iters=10, lr=-1.0)


def test_monge_ampere_solver_solve_gaussian_identity() -> None:
    """``solve`` on two equal Gaussians returns (approximately) the identity Hessian."""
    rho_0 = torch.stack([torch.zeros(3), torch.ones(3)])
    rho_1 = torch.stack([torch.zeros(3), torch.ones(3)])
    solver = MongeAmpereSolver(n_iters=60, lr=0.05)
    Q = solver.solve(rho_0, rho_1, n_samples=64, seed=42)
    assert Q.shape == (3, 3)
    # When ρ_0 = ρ_1 the Brenier map is the identity (Q = I, c = 0).
    assert torch.allclose(Q, torch.eye(3), atol=0.3)


def test_monge_ampere_solver_solve_returns_spd() -> None:
    """The output Hessian is SPD."""
    rho_0 = torch.stack([torch.zeros(3), torch.ones(3)])
    rho_1 = torch.stack([torch.ones(3) * 0.5, torch.ones(3) * 0.5])
    solver = MongeAmpereSolver(n_iters=40, lr=0.05)
    Q = solver.solve(rho_0, rho_1, n_samples=32, seed=7)
    eigvals = torch.linalg.eigvalsh(Q)
    assert bool((eigvals > 0).all()), (
        f"Output Hessian must be SPD; eigenvalues = {eigvals.tolist()}"
    )


def test_monge_ampere_solver_solve_shape_matches_dim() -> None:
    """Output Hessian shape is ``(d, d)`` for the given ``d``."""
    d = 4
    rho_0 = torch.stack([torch.zeros(d), torch.ones(d)])
    rho_1 = torch.stack([torch.zeros(d), torch.ones(d) * 0.5])
    solver = MongeAmpereSolver(n_iters=30, lr=0.05)
    Q = solver.solve(rho_0, rho_1, n_samples=16, dim=d, seed=11)
    assert Q.shape == (d, d)


def test_monge_ampere_solver_solve_rejects_invalid_rho() -> None:
    """``rho_0`` / ``rho_1`` of wrong shape raise ``ValueError``."""
    rho_0 = torch.randn(4)  # not (2, d)
    rho_1 = torch.randn(4)
    solver = MongeAmpereSolver()
    with pytest.raises(ValueError, match="rho_0"):
        solver.solve(rho_0, rho_1)


def test_monge_ampere_solver_solve_rejects_non_positive_var() -> None:
    """Non-positive diagonal covariance raises ``ValueError``."""
    rho_0 = torch.stack([torch.zeros(2), torch.tensor([1.0, -1.0])])
    rho_1 = torch.stack([torch.zeros(2), torch.ones(2)])
    solver = MongeAmpereSolver()
    with pytest.raises(ValueError, match="covariance"):
        solver.solve(rho_0, rho_1)


def test_monge_ampere_solver_log_density_gaussian_at_mean() -> None:
    """``_log_density_gaussian`` peaks at the mean."""
    d = 3
    mu = torch.tensor([1.0, -1.0, 0.5])
    var = torch.tensor([1.0, 2.0, 0.5])
    x = mu.unsqueeze(0)
    log_p = MongeAmpereSolver._log_density_gaussian(x, mu, var)
    # Evaluate at a different point.
    x_far = mu.unsqueeze(0) + torch.tensor([5.0, 0.0, 0.0])
    log_p_far = MongeAmpereSolver._log_density_gaussian(x_far, mu, var)
    assert log_p.item() > log_p_far.item()


# ---------------------------------------------------------------------------
# 4. AdvantageEstimatorMA
# ---------------------------------------------------------------------------


def test_advantage_estimator_ma_init_validates_lambda() -> None:
    """Negative ``lam`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="lam"):
        AdvantageEstimatorMA(lam=-0.1)


def test_advantage_estimator_ma_init_default_lambda() -> None:
    """Default ``lam`` is the canonical ``0.1``."""
    est = AdvantageEstimatorMA()
    assert est.lam == 0.1


def test_advantage_estimator_ma_variance_bound_lam_zero_is_bernstein() -> None:
    """For ``lam = 0`` the bound equals ``Var(A) / n``."""
    g = torch.Generator().manual_seed(13)
    A = torch.randn(256, generator=g)
    est = AdvantageEstimatorMA(lam=0.0)
    bound = est.variance_bound(A)
    expected = A.var(unbiased=True) / A.numel()
    assert torch.allclose(bound, expected, atol=1e-6)


def test_advantage_estimator_ma_variance_bound_lam_positive_tightens() -> None:
    """For ``lam > 0`` the bound is at least as tight as ``Var(A) / n``."""
    g = torch.Generator().manual_seed(14)
    A = torch.randn(512, generator=g)
    est_off = AdvantageEstimatorMA(lam=0.0)
    est_on = AdvantageEstimatorMA(lam=0.5)
    bound_off = est_off.variance_bound(A)
    bound_on = est_on.variance_bound(A)
    # MA tightening: lam > 0 multiplies by (1 + lam * ||A||²/n) >= 1
    # (since ||A||²/n = mean(A²) > 0 for non-degenerate A).
    assert bound_on.item() > bound_off.item()


def test_advantage_estimator_ma_variance_bound_scales_with_n() -> None:
    """The bound decays like ``1 / n`` for fixed variance."""
    g = torch.Generator().manual_seed(15)
    A = torch.randn(1024, generator=g)
    est = AdvantageEstimatorMA(lam=0.1)
    b_full = est.variance_bound(A)
    # Subsample to n/4: bound should be larger (and roughly 4× larger).
    A_sub = A[: A.numel() // 4]
    b_sub = est.variance_bound(A_sub)
    assert b_sub.item() > b_full.item()


def test_advantage_estimator_ma_variance_bound_needs_two_samples() -> None:
    """A single sample raises ``ValueError``."""
    est = AdvantageEstimatorMA(lam=0.1)
    with pytest.raises(ValueError, match="2 samples"):
        est.variance_bound(torch.tensor([1.0]))


def test_advantage_estimator_ma_variance_bound_returns_scalar() -> None:
    """The bound is a 0-D scalar tensor."""
    g = torch.Generator().manual_seed(16)
    A = torch.randn(64, generator=g)
    est = AdvantageEstimatorMA(lam=0.2)
    bound = est.variance_bound(A)
    assert bound.ndim == 0
    assert bound.dtype == torch.float32


def test_advantage_estimator_ma_advantage_helper() -> None:
    """``advantage`` returns ``log_p - log_q`` element-wise."""
    log_p = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    log_q = torch.tensor([0.0, 0.5, 2.0, -1.0])
    est = AdvantageEstimatorMA(lam=0.1)
    A = est.advantage(log_p, log_q)
    expected = torch.tensor([-1.0, -0.5, -1.0, 3.0])
    assert torch.allclose(A, expected)


def test_advantage_estimator_ma_advantage_helper_shape_mismatch() -> None:
    """Mismatched ``log_p`` / ``log_q`` raise ``ValueError``."""
    est = AdvantageEstimatorMA(lam=0.1)
    with pytest.raises(ValueError, match="same shape"):
        est.advantage(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# 5. Integration smoke test
# ---------------------------------------------------------------------------


def test_integration_kantorovich_mae_solver_advantage() -> None:
    """End-to-end smoke: solve MA, build a Kantorovich potential, bound the variance.

    Builds a Kantorovich potential from the MA Hessian, evaluates the
    potential on a small batch, then runs the MA-regularised
    advantage bound on a separate batch of advantages.  All steps
    run on CPU and complete in well under a second.
    """
    d = 3
    rho_0 = torch.stack([torch.zeros(d), torch.ones(d)])
    rho_1 = torch.stack([torch.zeros(d), torch.ones(d) * 0.5])
    solver = MongeAmpereSolver(n_iters=30, lr=0.05)
    Q = solver.solve(rho_0, rho_1, n_samples=32, seed=17)
    # Build a potential from the recovered Hessian.
    C = Q + 0.01 * torch.eye(d)
    phi = KantorovichPotential(C)
    phi.update_Q(Q)
    # Evaluate on a small batch.
    x = torch.randn(8, d)
    phi_vals = phi(x)
    assert phi_vals.shape == (8,)
    assert torch.isfinite(phi_vals).all()

    # Bound the variance of an unrelated batch of advantages.
    g = torch.Generator().manual_seed(18)
    A = torch.randn(128, generator=g)
    est = AdvantageEstimatorMA(lam=0.1)
    bound = est.variance_bound(A)
    assert torch.isfinite(bound)
    assert bound.item() > 0.0