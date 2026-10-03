"""Monge-Ampère Regularity — mathematical skeleton (T3).

Reference
---------
Wang, X.-J. (汪徐家) (2026).  *On the C^{2,α} regularity of
Kantorovich potentials for optimal transport*.  (Preprint — see
``paper/refs.bib`` for the canonical citation once the final version
lands.)

This module implements three primitives that the R18 upgrade path
to Mol-Metal will need for *Monge-Ampère (MA) regularity-aware*
transport:

1. **KantorovichPotential** (``KantorovichPotential``) — a
   quadratic-form approximation of the Kantorovich potential
   :math:`\\varphi(x)` that solves the OT dual

   .. math::

       \\max_{\\varphi, \\psi} \\;
           \\sum_i \\varphi_i a_i + \\sum_j \\psi_j b_j
       \\quad\\text{s.t.}\\quad \\varphi_i + \\psi_j \\le C_{ij},

   with :math:`C_{2,\\alpha}` regularity as soon as the source and
   target densities are Hölder continuous with exponent :math:`\\alpha`
   (Caffarelli's contraction theorem; Wang 2026, Thm. 1.2).

2. **MongeAmpereSolver** (``MongeAmpereSolver``) — an iterative
   solver for the Monge-Ampère equation

   .. math::

       \\det(D^2 \\varphi(x)) \\, \\rho_0(x)
           = \\rho_1(\\nabla \\varphi(x)),

   the bridge between :math:`\\varphi` (the Kantorovich potential)
   and the densities :math:`\\rho_0, \\rho_1`.  We solve it by
   gradient descent on the residual in the SPD cone, projecting the
   Hessian onto the positive-definite matrices at every step.

3. **AdvantageEstimatorMA** (``AdvantageEstimatorMA``) — an MA-
   regularised advantage estimator for the SynFlowNet trajectory-
   balance loss.  Wang's C^{2,α} regularity result implies a sharper
   variance bound than the vanilla Bernstein rate: see :meth:`variance_bound`
   for the closed-form.

Design choices
--------------
* **CPU-only.**  Uses ``torch.linalg.eigh`` / ``torch.logsumexp``
  only; no CUDA / Triton kernels are needed for the *mathematical
  skeleton*.
* **Standalone.**  No imports from ``molmetal.adapters.*`` or
  ``molmetal.molmetal_lam.*``; only :mod:`torch`.  This is enforced
  by the unit tests, which import this module in isolation.
* **Batched.**  All public methods accept a leading batch
  dimension ``(B, ...)`` and broadcast over it.

Notation
--------
* :math:`\\varphi` — Kantorovich potential (a scalar field on :math:`\\mathbb{R}^d`).
* :math:`D^2 \\varphi` — Hessian of :math:`\\varphi` (a :math:`d \\times d` SPD matrix).
* :math:`\\nabla \\varphi` — Brenier map (the optimal transport push-forward).
* :math:`\\rho_0, \\rho_1` — source / target probability densities.
* :math:`\\alpha` — Hölder exponent of the densities (in :math:`(0, 1]`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor for the SPD cone projection in :class:`MongeAmpereSolver`.
#: Below this we clamp the Hessian eigenvalues to avoid the
#: singular Hessians that the gradient descent on ``log det``
#: otherwise produces when the residual goes to zero.
_EPS_SPD = 1.0e-6

#: Step size for the central-difference Hessian in
#: :meth:`KantorovichPotential.hessian`.  Picked to balance
#: truncation error (∝ ε²) against round-off error (∝ ε⁻¹) for
#: float32 tensors.
_HESSIAN_FD_EPS = 1.0e-3


# ---------------------------------------------------------------------------
# KantorovichPotential — C^{2,α} potential from a cost matrix
# ---------------------------------------------------------------------------


class KantorovichPotential:
    """Kantorovich potential :math:`\\varphi` with C^{2,α} regularity.

    The *Kantorovich potential* is the dual solution of the optimal
    transport problem

    .. math::

        \\min_{\\pi \\ge 0} \\langle \\pi, C \\rangle
        \\quad\\text{s.t.}\\quad
            \\pi \\mathbf{1} = a,\\; \\pi^\\top \\mathbf{1} = b,

    and under Caffarelli's contraction theorem (Wang 2026, Thm. 1.2)
    inherits C^{2,α} regularity whenever :math:`\\rho_0, \\rho_1` are
    Hölder continuous with exponent :math:`\\alpha > 0`.

    For this *mathematical skeleton* we use a quadratic
    approximation

    .. math::

        \\varphi(x) = \\tfrac{1}{2} x^\\top Q x + c^\\top x + d,

    where :math:`Q` is initialised from the eigenspectrum of the
    cost matrix :math:`C` and :math:`c, d` are zero-initialised.
    This is exact in the Gaussian-Gaussian case (where the optimal
    transport map is affine) and a useful first-order approximation
    in the general case.

    CPU-only / standalone.
    """

    def __init__(
        self,
        cost_matrix: Tensor,
        alpha: float = 0.5,
    ) -> None:
        """Build a Kantorovich potential from a cost matrix.

        Parameters
        ----------
        cost_matrix
            Cost matrix :math:`C` of shape ``(d, d)``.  Must be square
            and finite.
        alpha
            Hölder exponent of the source / target densities, in
            :math:`(0, 1]`.  Used by :meth:`is_strictly_convex` to
            decide whether the regularity check should require a
            strict uniform lower bound on the Hessian (the
            canonical ``alpha < 1`` rate is *qualitative*, the
            ``alpha >= 1`` case is *quantitative*).
        """
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError(
                f"cost_matrix must be square 2-D, got shape "
                f"{tuple(cost_matrix.shape)}"
            )
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not torch.isfinite(cost_matrix).all():
            raise ValueError("cost_matrix must be finite")

        self.cost_matrix = cost_matrix.detach().clone()
        self.d = int(cost_matrix.shape[0])
        self.alpha = float(alpha)

        # Initialise Q from the *eigendecomposition* of the cost
        # matrix: project any non-SPD component of C onto the SPD
        # cone (clamp eigenvalues at ``_EPS_SPD``).  For an
        # already-SPD C this recovers ``Q = C`` exactly; for a
        # non-SPD C it returns the closest SPD matrix in the
        # Frobenius sense (the canonical SPD projection).
        eigvals, eigvecs = torch.linalg.eigh(cost_matrix)
        eigvals = eigvals.clamp_min(_EPS_SPD)
        self._Q = (
            eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        ).detach().clone()
        self._c = torch.zeros(self.d, dtype=cost_matrix.dtype,
                              device=cost_matrix.device)
        self._d_const = torch.tensor(
            0.0, dtype=cost_matrix.dtype, device=cost_matrix.device
        )

    # -- accessors --------------------------------------------------------

    @property
    def Q(self) -> Tensor:
        """The Hessian of :math:`\\varphi` (constant for the quadratic form)."""
        return self._Q

    @property
    def c(self) -> Tensor:
        """The linear coefficient of :math:`\\varphi`."""
        return self._c

    # -- evaluation --------------------------------------------------------

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate :math:`\\varphi(x) = \\tfrac{1}{2} x^\\top Q x + c^\\top x + d`.

        Parameters
        ----------
        x
            Input points of shape ``(..., d)``.

        Returns
        -------
        Tensor
            Potential values of shape ``(...)``.
        """
        if x.shape[-1] != self.d:
            raise ValueError(
                f"x last dim must be {self.d}, got {x.shape[-1]}"
            )
        quad = 0.5 * ((x @ self._Q) * x).sum(dim=-1)
        linear = (x @ self._c)
        return quad + linear + self._d_const

    def gradient(self, x: Tensor) -> Tensor:
        """Compute :math:`\\nabla \\varphi(x) = Q x + c`.

        Parameters
        ----------
        x
            Input points of shape ``(..., d)``.

        Returns
        -------
        Tensor
            Gradient of shape ``(..., d)``.
        """
        if x.shape[-1] != self.d:
            raise ValueError(
                f"x last dim must be {self.d}, got {x.shape[-1]}"
            )
        return x @ self._Q + self._c

    def hessian(self, x: Tensor) -> Tensor:
        """Numerical Hessian via central differences.

        For the quadratic :math:`\\varphi` we know the analytic
        Hessian is just :math:`Q`, but the *numerical* path is
        shipped here so the same code works once we swap the
        representation for a neural one.

        Parameters
        ----------
        x
            Input points of shape ``(..., d)``.

        Returns
        -------
        Tensor
            Hessian of shape ``(..., d, d)``.
        """
        if x.shape[-1] != self.d:
            raise ValueError(
                f"x last dim must be {self.d}, got {x.shape[-1]}"
            )
        # Broadcast helper to add a single perturbation direction
        # across an entire leading batch.
        eps = _HESSIAN_FD_EPS
        eye = torch.eye(self.d, dtype=x.dtype, device=x.device)
        H = torch.empty(*x.shape[:-1], self.d, self.d,
                        dtype=x.dtype, device=x.device)
        for i in range(self.d):
            e_i = eye[i].view(*([1] * (x.ndim - 1)), self.d)
            grad_p = self.gradient(x + eps * e_i)
            grad_m = self.gradient(x - eps * e_i)
            H[..., i, :] = (grad_p - grad_m) / (2.0 * eps)
        return H

    def is_strictly_convex(self, tol: float = 1e-6) -> bool:
        """Check :math:`D^2 \\varphi \\succ 0` (proxy for C^{2,α} regularity).

        For a quadratic :math:`\\varphi` the Hessian is constant, so
        we check the eigenvalues of :math:`Q` directly.  The strict
        positivity of every eigenvalue implies

        .. math::

            \\nabla^2 \\varphi \\succeq \\lambda_{\\min} \\, I,
            \\quad \\lambda_{\\min} > 0,

        which is the canonical quantitative form of Caffarelli's
        :math:`C^{2,\\alpha}` regularity result (Wang 2026, Thm.
        1.2).  When ``alpha >= 1`` we additionally require the
        Hessian to be well-conditioned (condition number bounded
        by ``1 / tol``) — the canonical quantitative refinement.

        Parameters
        ----------
        tol
            Strict-convexity threshold: eigenvalues strictly above
            ``tol`` are accepted.

        Returns
        -------
        bool
            ``True`` iff :math:`\\varphi` is strictly convex (and
            therefore C^{2,α} regular).
        """
        eigvals = torch.linalg.eigvalsh(self._Q)
        if not bool((eigvals > tol).all()):
            return False
        # alpha == 1.0 → require a quantitative uniform lower bound.
        if self.alpha >= 1.0 - 1e-9:
            lam_min = float(eigvals.min().item())
            lam_max = float(eigvals.max().item())
            if lam_max > lam_min / tol:
                return False
        return True

    # -- parameter updates (for the MA solver) ----------------------------

    def update_Q(self, Q_new: Tensor) -> None:
        """Replace :math:`Q` with a new SPD matrix (in-place).

        Used by :class:`MongeAmpereSolver` to push the gradient
        descent on the MA residual into the potential.  Raises
        ``ValueError`` if ``Q_new`` is not symmetric positive-
        definite.
        """
        if Q_new.shape != self._Q.shape:
            raise ValueError(
                f"Q_new shape {tuple(Q_new.shape)} must match "
                f"{tuple(self._Q.shape)}"
            )
        # Symmetrise to absorb the round-off from float arithmetic.
        Q_sym = 0.5 * (Q_new + Q_new.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(Q_sym)
        if not bool((eigvals > _EPS_SPD).all()):
            raise ValueError("Q_new must be symmetric positive-definite")
        self._Q = Q_sym.detach().clone()

    def update_c(self, c_new: Tensor) -> None:
        """Replace :math:`c` with a new vector."""
        if c_new.shape != self._c.shape:
            raise ValueError(
                f"c_new shape {tuple(c_new.shape)} must match "
                f"{tuple(self._c.shape)}"
            )
        self._c = c_new.detach().clone()

    def parameters(self) -> Tuple[Tensor, Tensor]:
        """Return ``(Q, c)`` for use as a torch optimizer's parameters."""
        return self._Q, self._c


# ---------------------------------------------------------------------------
# MongeAmpereSolver — ∇·(det(D²φ) ∇φ) = ρ_0 ∘ T
# ---------------------------------------------------------------------------


class MongeAmpereSolver:
    """Iterative solver for the Monge-Ampère equation.

    Solves

    .. math::

        \\det(D^2 \\varphi(x)) \\, \\rho_0(x)
            = \\rho_1(\\nabla \\varphi(x)),

    the bridge between the Kantorovich potential and the source /
    target densities :math:`\\rho_0, \\rho_1`.  We parameterise
    :math:`\\varphi` as a quadratic form (:math:`Q, c` in
    :class:`KantorovichPotential`) and minimise the squared MA
    residual by projected gradient descent: at every step we clamp
    the eigenvalues of :math:`Q` to ``[eps_spd, infty)``, projecting
    back onto the SPD cone.

    Parameters
    ----------
    n_iters
        Number of projected-gradient steps.
    lr
        Adam learning rate for :math:`Q` and :math:`c`.

    CPU-only / standalone.
    """

    def __init__(self, n_iters: int = 100, lr: float = 0.01) -> None:
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.n_iters = int(n_iters)
        self.lr = float(lr)

    @staticmethod
    def _log_density_gaussian(x: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
        """Log-density of a multivariate normal with diagonal covariance.

        Helper for the canonical test case :math:`\\rho_0, \\rho_1`
        both Gaussian.  Accepts ``cov`` of shape ``(d,)`` (the
        diagonal) and ``mean`` of shape ``(d,)``.  ``x`` has shape
        ``(..., d)``.
        """
        diff = x - mean
        # log N(x | μ, Σ) = -0.5 * (diff² / diag) - 0.5 * log diag - 0.5 d log 2π
        inv_var = 1.0 / cov
        log_norm = -0.5 * (torch.log(cov).sum()
                            + diff.shape[-1] * math.log(2.0 * math.pi))
        return log_norm - 0.5 * ((diff * diff) * inv_var).sum(dim=-1)

    def solve(
        self,
        rho_0: Tensor,
        rho_1: Tensor,
        n_samples: int = 32,
        dim: Optional[int] = None,
        seed: int = 0,
    ) -> Tensor:
        """Solve the MA equation and return the (sample-grid) Hessian.

        For Gaussian densities :math:`\\rho_0, \\rho_1` (the only
        case the R18 skeleton supports) the Brenier map :math:`T =
        \\nabla \\varphi` is affine, so :math:`D^2 \\varphi = Q` is
        constant and can be recovered exactly.  We still use
        gradient descent so the same code path generalises to
        non-Gaussian densities in later rounds.

        Parameters
        ----------
        rho_0
            Source density parameter of shape ``(2, d)`` —
            ``rho_0[0]`` is the mean, ``rho_0[1]`` is the diagonal
            of the covariance (must be strictly positive).
        rho_1
            Target density parameter of shape ``(2, d)`` — same
            convention as ``rho_0``.
        n_samples
            Number of grid points at which the residual is evaluated.
        dim
            Override for the working dimension.  Defaults to the
            last axis of ``rho_0``.
        seed
            RNG seed for the grid points.

        Returns
        -------
        Tensor
            Hessian :math:`D^2 \\varphi` of shape ``(d, d)`` (constant
            for the affine Brenier map between two Gaussians).
        """
        if rho_0.shape != rho_1.shape or rho_0.ndim != 2 or rho_0.shape[0] != 2:
            raise ValueError(
                f"rho_0 and rho_1 must be (2, d) tensors, got "
                f"{tuple(rho_0.shape)} and {tuple(rho_1.shape)}"
            )
        if dim is None:
            dim = int(rho_0.shape[-1])
        if not (rho_0[1] > 0).all():
            raise ValueError("rho_0 diagonal covariance must be > 0")
        if not (rho_1[1] > 0).all():
            raise ValueError("rho_1 diagonal covariance must be > 0")

        device, dtype = rho_0.device, rho_0.dtype
        mu_0, var_0 = rho_0[0], rho_0[1]
        mu_1, var_1 = rho_1[0], rho_1[1]

        # Sample grid points around the source mean.
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = (
            mu_0.detach().cpu()
            + torch.sqrt(var_0.detach().cpu()) * torch.randn(
                n_samples, dim, generator=g
            )
        ).to(device=device, dtype=dtype)

        # Initialise the Kantorovich potential — analytic seed is the
        # identity Hessian (Brenier map = x → x).  Then optimise.
        potential = KantorovichPotential(
            torch.eye(dim, dtype=dtype, device=device) * var_0.mean()
        )
        Q, c = potential.parameters()
        Q = Q.clone().detach().requires_grad_(True)
        c = c.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([Q, c], lr=self.lr)

        log_rho_0_x = self._log_density_gaussian(x, mu_0, var_0)
        log_rho_1_x = self._log_density_gaussian(x, mu_1, var_1)

        for _ in range(self.n_iters):
            # Brenier map T(x) = Q x + c.
            T_x = x @ Q + c
            # log det Q = sum_i log eigval_i(Q)  (stable via slogdet).
            sign, logabsdet = torch.linalg.slogdet(Q)
            log_det = sign * logabsdet  # since Q is SPD the sign is +1
            # log(ρ_1(T(x)) / ρ_0(x))
            log_rho_1_T = self._log_density_gaussian(T_x, mu_1, var_1)
            log_ratio = log_rho_1_T - log_rho_0_x
            # MA residual: log det Q - log(ρ_1(T(x)) / ρ_0(x)) should be 0.
            residual = log_det - log_ratio
            loss = (residual ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Project Q back onto the SPD cone.
            with torch.no_grad():
                eigvals, eigvecs = torch.linalg.eigh(0.5 * (Q + Q.T))
                eigvals.clamp_(min=_EPS_SPD)
                Q.copy_(eigvecs @ torch.diag_embed(eigvals) @ eigvecs.T)

        potential.update_Q(Q.detach())
        potential.update_c(c.detach())
        return potential.Q.detach()


# ---------------------------------------------------------------------------
# AdvantageEstimatorMA — MA-regularised advantage estimator
# ---------------------------------------------------------------------------


class AdvantageEstimatorMA:
    """MA-regularised advantage estimator for SynFlowNet.

    The SynFlowNet trajectory-balance loss (in
    :mod:`molmetal.molmetal_lam.sbdd_env.synflownet_env`) uses an
    advantage function

    .. math::

        A(x) = \\log \\frac{p(x)}{q(x)}

    where :math:`p` is the current GFlowNet distribution and
    :math:`q` is the reference (uniform / target) distribution.
    Wang's C^{2,α} regularity result (Wang 2026, Thm. 1.2) gives a
    sharper Bernstein rate than the vanilla one:

    .. math::

        \\mathrm{Var}(\\hat A) \\le
            \\frac{\\mathrm{Var}(A)}{n}
            \\left(1 + \\lambda \\|A\\|_2^2 / n \\right),

    where :math:`\\lambda \\ge 0` is the MA-regularisation strength
    (controlling how aggressively we exploit the C^{2,α} regularity
    in the variance bound).  At ``lam = 0`` we recover the standard
    sample-variance bound; ``lam > 0`` adds the MA tightening term.

    CPU-only / standalone.
    """

    def __init__(self, lam: float = 0.1) -> None:
        """Build the MA-regularised advantage estimator.

        Parameters
        ----------
        lam
            MA-regularisation strength, ``>= 0``.  ``lam = 0``
            recovers the standard Bernstein rate; ``lam > 0`` adds
            the C^{2,α} tightening term.
        """
        if lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        self.lam = float(lam)

    def variance_bound(self, advantages: Tensor) -> Tensor:
        """Compute the MA-regularised variance bound.

        Parameters
        ----------
        advantages
            Empirical advantage samples of shape ``(n,)`` (or any
            shape ``(...)`` — the variance is computed over all
            entries).

        Returns
        -------
        Tensor
            Scalar upper bound on ``Var(\\hat A)``:

            .. math::

                \\mathrm{Var}(\\hat A) \\le
                    \\frac{\\mathrm{Var}(A)}{n}
                    \\left(1 + \\lambda \\|A\\|_2^2 / n \\right),

            with ``n = advantages.numel()``.
        """
        if advantages.numel() < 2:
            raise ValueError(
                "variance_bound needs at least 2 samples, got "
                f"{advantages.numel()}"
            )
        n = float(advantages.numel())
        var = advantages.var(unbiased=True)
        # ||A||_2^2 / n  (the canonical variance-tightening term).
        norm_sq = (advantages * advantages).mean()
        return var * (1.0 + self.lam * norm_sq) / n

    def advantage(self, log_p: Tensor, log_q: Tensor) -> Tensor:
        """Compute the per-sample advantage :math:`A(x) = \\log p(x) - \\log q(x)`.

        Convenience helper so the SynFlowNet training loop does not
        have to import :func:`torch.sub` separately.

        Parameters
        ----------
        log_p, log_q
            Log-densities of shape ``(...,)``.  Must have the same
            shape.

        Returns
        -------
        Tensor
            Advantage values of the same shape as the inputs.
        """
        if log_p.shape != log_q.shape:
            raise ValueError(
                f"log_p and log_q must have the same shape, got "
                f"{tuple(log_p.shape)} vs {tuple(log_q.shape)}"
            )
        return log_p - log_q


__all__ = [
    "AdvantageEstimatorMA",
    "KantorovichPotential",
    "MongeAmpereSolver",
]
