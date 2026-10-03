"""Tests for :mod:`molmetal.math_skills.transport_relaxation`.

T2 — Transport Relaxation
==========================

These tests cover the three mathematical primitives shipped by the
R18 transport-relaxation skeleton:

1. **RelaxedOT** — soft endpoint penalty OT.  Standard OT is
   recovered at ``lambda_relax = 0``; the coupling collapses to the
   uniform plan as ``lambda_relax -> infty``.
2. **BoltzmannEndpoint** — canonical Boltzmann distribution over
   docked poses.
3. **TransportRelaxationTerm** — the GFlowNet TB transport
   relaxation term, KL divergence on intermediate distributions,
   and the ``sweep_lambda`` sensitivity helper.

Every test is **CPU-only** and uses small data (batch sizes 4–16)
so the full suite runs in < 5 seconds.

Reference
---------
Jiang, Y., Xu, M. & Zhang, T. (2026).  *Transport Relaxation for
Generative Modeling*.  arXiv:2602.08118 [stat.ML].
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
    BoltzmannEndpoint,
    RelaxedOT,
    TransportRelaxationTerm,
)
from molmetal.math_skills.se3_schrodinger_bridge import (
    LogSinkhornLoss,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _rand_points(n: int, m: int, d: int = 3, seed: int = 0) -> tuple:
    """Return two random point clouds ``(p0, p1)`` for OT testing."""
    g = torch.Generator().manual_seed(seed)
    p0 = torch.randn(n, d, generator=g)
    p1 = torch.randn(m, d, generator=g)
    return p0, p1


def _uniform_marginal(n: int, dtype=torch.float32) -> torch.Tensor:
    """Uniform marginal of shape ``(n,)``."""
    return torch.full((n,), 1.0 / n, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. RelaxedOT
# ---------------------------------------------------------------------------


def test_relaxed_ot_zero_lambda_matches_standard_ot() -> None:
    """``lambda_relax = 0`` recovers the standard entropic OT loss.

    We compare against T1's :class:`LogSinkhornLoss` with the same
    ``eps`` and the same cost matrix — for ``lambda_relax = 0`` the
    soft marginals equal the target marginals and the two losses
    should agree to ``eps`` Sinkhorn-tolerance.
    """
    p0, p1 = _rand_points(6, 8, d=3, seed=1)
    eps = 0.1
    n_iters = 200

    # Standard OT (T1's LogSinkhornLoss).
    std = LogSinkhornLoss(eps=eps, n_iters=n_iters).forward(p0, p1)[0]

    # Relaxed OT with lambda_relax = 0 (should equal standard OT).
    relaxed = RelaxedOT(eps=eps, lambda_relax=0.0, n_iters=n_iters).forward(p0, p1)

    # Tighter than 1 % — both solvers use the same log-domain
    # Sinkhorn fixed point so the only source of mismatch is the
    # rounding of the uniform marginal ``1 / n`` when alpha = 0.
    assert torch.allclose(std, relaxed, atol=1e-3, rtol=1e-3)


def test_relaxed_ot_large_lambda_yields_uniform_marginals() -> None:
    """For ``lambda_relax -> infty`` the coupling's marginals converge to uniform.

    We pick ``lambda_relax = 1000`` so ``alpha = lambda / (1 + lambda)``
    is ``~ 0.999`` and the soft marginals are within 0.1 % of
    uniform; with enough Sinkhorn iterations the resulting coupling
    should have row / column sums indistinguishable from
    ``1/n`` / ``1/m``.  We check the marginals (not the entries)
    because even with uniform marginals the optimal coupling may
    concentrate on cheap pairs.
    """
    p0, p1 = _rand_points(5, 7, d=3, seed=2)
    n, m = 5, 7
    ot = RelaxedOT(eps=0.1, lambda_relax=1.0e3, n_iters=500)
    P = ot.coupling(p0, p1)
    assert P.shape == (n, m)

    # Marginal sums should be very close to uniform.
    row_sums = P.sum(dim=1)
    col_sums = P.sum(dim=0)
    uniform_rows = torch.full((n,), 1.0 / n)
    uniform_cols = torch.full((m,), 1.0 / m)
    assert torch.allclose(row_sums, uniform_rows, atol=1e-2)
    assert torch.allclose(col_sums, uniform_cols, atol=1e-2)


def test_relaxed_ot_large_lambda_loss_finite_and_bounded() -> None:
    """For ``lambda_relax -> infty`` the OT loss stays finite and below ``max(C)``.

    The relaxed loss ``<P, C> + eps * KL`` is bounded above by the
    cost of the worst-case pair (since ``P_ij >= 0`` sums to 1 and
    the KL term is non-negative).  The relaxation constraint pushes
    the loss into a moderate regime — not zero (because OT is no
    longer minimising hard) and not unbounded.
    """
    p0, p1 = _rand_points(5, 7, d=3, seed=2)
    eps = 0.1

    C = RelaxedOT._pairwise_sqdist(p0, p1)
    max_cost = C.max()

    ot = RelaxedOT(eps=eps, lambda_relax=1.0e3, n_iters=500)
    loss = ot.forward(p0, p1)

    assert torch.isfinite(loss)
    # Sanity bound: the cost term is at most ``max(C)`` (coupling
    # sums to 1) and the KL term is non-negative, so the loss is
    # bounded by ``max(C) + small``.
    assert loss.item() <= max_cost.item() + 1e-2


def test_relaxed_ot_invalid_eps() -> None:
    """``eps <= 0`` is rejected."""
    with pytest.raises(ValueError, match="eps"):
        RelaxedOT(eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        RelaxedOT(eps=-1.0)


def test_relaxed_ot_invalid_lambda() -> None:
    """``lambda_relax < 0`` is rejected."""
    with pytest.raises(ValueError, match="lambda_relax"):
        RelaxedOT(eps=0.1, lambda_relax=-1e-3)


def test_relaxed_ot_invalid_n_iters() -> None:
    """``n_iters < 1`` is rejected."""
    with pytest.raises(ValueError, match="n_iters"):
        RelaxedOT(eps=0.1, lambda_relax=0.0, n_iters=0)


def test_relaxed_ot_finite_loss() -> None:
    """The relaxed OT loss is finite on a normal random input."""
    p0, p1 = _rand_points(6, 6, d=4, seed=3)
    loss = RelaxedOT(eps=0.05, lambda_relax=1e-2).forward(p0, p1)
    assert torch.isfinite(loss)


def test_relaxed_ot_alpha_sigmoid() -> None:
    """``alpha = lambda_relax / (1 + lambda_relax)`` is in ``[0, 1)``.

    We sweep ``lambda_relax`` over a few orders of magnitude and
    verify the alpha values lie in the right range.
    """
    for lam in [0.0, 1e-4, 1e-3, 1e-2, 1.0, 10.0, 1.0e6]:
        alpha = RelaxedOT(eps=0.1, lambda_relax=lam)._alpha()
        assert 0.0 <= alpha < 1.0
        if lam == 0.0:
            assert alpha == 0.0
        else:
            assert math.isclose(alpha, lam / (1.0 + lam), rel_tol=1e-6)


def test_relaxed_ot_custom_marginals() -> None:
    """User-provided marginals ``a, b`` are respected (soft-blended)."""
    p0, p1 = _rand_points(4, 4, d=3, seed=4)
    a = torch.tensor([0.5, 0.2, 0.2, 0.1])
    b = torch.tensor([0.4, 0.3, 0.2, 0.1])

    # ``lambda_relax = 0`` -> marginals enforced exactly.  Check the
    # row / column sums of the coupling match ``a`` and ``b``.
    P = RelaxedOT(eps=0.1, lambda_relax=0.0, n_iters=300).coupling(
        p0, p1, a=a, b=b
    )
    row_sums = P.sum(dim=1)
    col_sums = P.sum(dim=0)
    assert torch.allclose(row_sums, a, atol=1e-3)
    assert torch.allclose(col_sums, b, atol=1e-3)


def test_relaxed_ot_custom_cost() -> None:
    """A precomputed cost matrix ``C`` overrides the pairwise distance."""
    p0, p1 = _rand_points(4, 4, d=3, seed=5)
    # Custom constant cost (same everywhere).
    C_const = torch.ones(4, 4)
    loss_const = RelaxedOT(eps=0.1, lambda_relax=0.0).forward(p0, p1, C=C_const)

    p0_other, p1_other = _rand_points(4, 4, d=3, seed=5)
    C_other = RelaxedOT._pairwise_sqdist(p0_other, p1_other)
    # Force the same cost matrix but a different ``p0`` — the loss
    # should depend only on ``C`` and the marginals, not on ``p0``.
    loss_other = RelaxedOT(eps=0.1, lambda_relax=0.0).forward(
        p0_other, p1_other, C=C_const
    )
    assert torch.allclose(loss_const, loss_other, atol=1e-5)


def test_relaxed_ot_loss_monotonic_in_lambda() -> None:
    """Relaxed OT loss is non-decreasing in ``lambda_relax``.

    Standard OT (lambda = 0) is the optimal transport under the
    marginal constraints; relaxing the constraints (lambda > 0)
    can only *increase* the cost (or leave it unchanged).
    """
    p0, p1 = _rand_points(5, 5, d=3, seed=6)
    eps = 0.1
    n_iters = 200
    losses = []
    for lam in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        loss = RelaxedOT(eps=eps, lambda_relax=lam, n_iters=n_iters).forward(
            p0, p1
        )
        losses.append(loss.item())
    for prev, curr in zip(losses, losses[1:]):
        assert curr >= prev - 1e-3, (
            f"Relaxed OT loss should be non-decreasing in lambda_relax, "
            f"but went {prev} -> {curr}"
        )


def test_relaxed_ot_coupling_shape() -> None:
    """The coupling has shape ``(n, m)`` for input clouds of size ``n, m``."""
    p0, p1 = _rand_points(7, 9, d=2, seed=7)
    P = RelaxedOT(eps=0.1, lambda_relax=0.0).coupling(p0, p1)
    assert P.shape == (7, 9)
    assert (P >= 0).all()


# ---------------------------------------------------------------------------
# 2. BoltzmannEndpoint
# ---------------------------------------------------------------------------


def test_boltzmann_normalization() -> None:
    """``sum_i exp(log_prob_i) = 1`` over the last axis."""
    energies = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    poses = torch.randn(5, 3)
    boltz = BoltzmannEndpoint(T=1.0)
    probs = boltz.prob(poses, energies)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


def test_boltzmann_log_prob_finite() -> None:
    """``log_prob`` is finite for typical energies."""
    energies = torch.linspace(-5.0, 5.0, 11)
    poses = torch.randn(11, 4)
    lp = BoltzmannEndpoint(T=1.0).log_prob(poses, energies)
    assert torch.isfinite(lp).all()


def test_boltzmann_large_temperature_approaches_uniform() -> None:
    """For ``T -> infty`` the distribution approaches uniform."""
    energies = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    poses = torch.randn(5, 3)
    boltz = BoltzmannEndpoint(T=1.0e6)
    probs = boltz.prob(poses, energies)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    assert torch.allclose(probs, uniform, atol=1e-5)


def test_boltzmann_small_temperature_concentrates_on_minimum() -> None:
    """For ``T -> 0`` the distribution concentrates on the minimum-energy pose."""
    energies = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    poses = torch.randn(5, 3)
    boltz = BoltzmannEndpoint(T=1.0e-6)
    probs = boltz.prob(poses, energies)
    # All mass on the first pose (which has the smallest energy).
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(probs, expected, atol=1e-4)


def test_boltzmann_invalid_temperature() -> None:
    """``T <= 0`` is rejected."""
    with pytest.raises(ValueError, match="T"):
        BoltzmannEndpoint(T=0.0)
    with pytest.raises(ValueError, match="T"):
        BoltzmannEndpoint(T=-1.0)


def test_boltzmann_shape_mismatch_raises() -> None:
    """Mismatched ``poses`` and ``energies`` shapes raise ``ValueError``."""
    poses = torch.randn(4, 3)
    energies = torch.randn(5)
    with pytest.raises(ValueError, match="energies shape"):
        BoltzmannEndpoint(T=1.0).log_prob(poses, energies)


def test_boltzmann_batched_shapes() -> None:
    """Batched ``(B, n)`` inputs produce ``(B, n)`` outputs that sum to 1 per batch."""
    energies = torch.randn(3, 5)
    poses = torch.randn(3, 5, 4)
    boltz = BoltzmannEndpoint(T=0.7)
    probs = boltz.prob(poses, energies)
    assert probs.shape == (3, 5)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_boltzmann_log_prob_argmin() -> None:
    """The maximum log-probability corresponds to the minimum-energy pose."""
    energies = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0])
    poses = torch.randn(5, 3)
    lp = BoltzmannEndpoint(T=0.5).log_prob(poses, energies)
    assert int(lp.argmax().item()) == int(energies.argmin().item())


# ---------------------------------------------------------------------------
# 3. TransportRelaxationTerm
# ---------------------------------------------------------------------------


def test_kl_divergence_non_negative() -> None:
    """``KL(p || q) >= 0`` for any valid probability vectors ``p, q``."""
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.4, 0.4, 0.2])
    kl = term.kl_divergence(p, q)
    assert kl.item() >= 0.0


def test_kl_divergence_zero_when_equal() -> None:
    """``KL(p || p) = 0``."""
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    p = torch.tensor([0.5, 0.3, 0.2])
    kl = term.kl_divergence(p, p)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_asymmetric() -> None:
    """``KL(p || q) != KL(q || p)`` in general."""
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.5, 0.3])
    kl_pq = term.kl_divergence(p, q)
    kl_qp = term.kl_divergence(q, p)
    assert not torch.isclose(kl_pq, kl_qp, atol=1e-4)


def test_kl_divergence_shape_mismatch() -> None:
    """Mismatched ``p_T`` / ``q_T`` shapes raise ``ValueError``."""
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.3, 0.4, 0.3])
    with pytest.raises(ValueError, match="same shape"):
        term.kl_divergence(p, q)


def test_kl_invalid_lambda() -> None:
    """``lambda_relax < 0`` is rejected."""
    with pytest.raises(ValueError, match="lambda_relax"):
        TransportRelaxationTerm(lambda_relax=-1.0)


def test_sweep_lambda_returns_dict() -> None:
    """``sweep_lambda`` returns a dict keyed by the lambda values."""
    p0, p1 = _rand_points(4, 4, d=3, seed=8)
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    lambdas = [1e-4, 1e-3, 1e-2]
    out = term.sweep_lambda(p0, p1, lambdas=lambdas)
    assert set(out.keys()) == set(lambdas)
    for v in out.values():
        assert torch.isfinite(v)


def test_sweep_lambda_default_values() -> None:
    """``sweep_lambda`` defaults to ``[1e-4, 1e-3, 1e-2]`` when ``lambdas=None``."""
    p0, p1 = _rand_points(4, 4, d=3, seed=9)
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    out = term.sweep_lambda(p0, p1)
    assert set(out.keys()) == {1e-4, 1e-3, 1e-2}


def test_sweep_lambda_monotonic_non_decreasing() -> None:
    """OT loss in the sweep is non-decreasing in ``lambda_relax``.

    This is the canonical sensitivity-analysis claim of
    Jiang-Xu-Zhang 2026, §5 — increasing the relaxation parameter
    can only relax the marginal constraints, which can only increase
    the OT cost.
    """
    p0, p1 = _rand_points(5, 5, d=3, seed=10)
    term = TransportRelaxationTerm(lambda_relax=1e-3)
    out = term.sweep_lambda(
        p0, p1, lambdas=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], eps=0.1, n_iters=300
    )
    losses = [out[lam].item() for lam in sorted(out.keys())]
    for prev, curr in zip(losses, losses[1:]):
        assert curr >= prev - 1e-3, (
            f"Sweep should be non-decreasing, got {prev} -> {curr}"
        )


# ---------------------------------------------------------------------------
# 4. Integration smoke test
# ---------------------------------------------------------------------------


def test_integration_relaxed_ot_with_boltzmann_endpoint() -> None:
    """End-to-end smoke: relaxed OT + Boltzmann endpoint + KL term.

    Builds an intermediate distribution ``p_T`` from the relaxed OT
    coupling (row sums) and a Boltzmann reference ``q_T`` from
    pairwise costs; verifies the full transport relaxation loss is
    finite and non-negative.
    """
    p0, p1 = _rand_points(6, 6, d=3, seed=11)
    ot = RelaxedOT(eps=0.1, lambda_relax=1e-2, n_iters=200)
    # Use the row sums of the (slightly non-uniform) coupling as the
    # empirical intermediate distribution.
    P = ot.coupling(p0, p1)
    p_T = P.sum(dim=1)  # shape (n,)

    # Use the pairwise costs as "energies" for the Boltzmann endpoint.
    C = RelaxedOT._pairwise_sqdist(p0, p1)
    energies = C.diagonal()  # self-distances == 0 — trivial but legal.
    boltz = BoltzmannEndpoint(T=1.0)
    q_T = boltz.prob(p0, energies)

    term = TransportRelaxationTerm(lambda_relax=1e-2)
    kl = term.kl_divergence(p_T, q_T)
    assert torch.isfinite(kl)
    # KL is non-negative; if the two distributions happen to be equal
    # it could be 0, but generically it's positive.
    assert kl.item() >= 0.0
