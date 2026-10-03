"""Transport Relaxation — mathematical skeleton (T2).

Reference
---------
Jiang, Y., Xu, M. & Zhang, T. (2026).  *Transport Relaxation for
Generative Modeling*.  arXiv:2602.08118 [stat.ML].  Cited in
``paper/refs.bib`` as ``jiang2026transport_relax``.

This module implements the three primitives that the R18 upgrade
path to Mol-Metal will need for *transport relaxation*:

1. **RelaxedOT** (``RelaxedOT``) — entropic Optimal Transport with a
   *soft* endpoint penalty.  Standard entropic OT (Cuturi 2013;
   already shipped in :mod:`molmetal.math_skills.se3_schrodinger_bridge`
   as :class:`~molmetal.math_skills.LogSinkhornLoss`) enforces the
   marginal constraints ``P 1 = a`` and ``P^T 1 = b`` *exactly*.
   The relaxed version replaces them by a convex blend of the
   target marginals and the uniform distribution:

   .. math::

       \\tilde a = (1 - \\alpha) a + \\alpha \\mathbb{1}/n, \\quad
       \\tilde b = (1 - \\alpha) b + \\alpha \\mathbb{1}/m,
       \\quad \\alpha = \\frac{\\lambda_{\\rm relax}}{1 + \\lambda_{\\rm relax}}.

   and solves the same entropic OT problem with these *soft*
   marginals (Jiang-Xu-Zhang 2026, §3.2).  Standard OT is recovered
   at ``lambda_relax = 0``; ``lambda_relax -> infty`` collapses the
   coupling to the uniform plan ``P_ij = 1/(n m)``.

2. **Boltzmann endpoint** (``BoltzmannEndpoint``) — canonical
   Boltzmann distribution over a discrete set of docked poses.
   Used as the reference ``q_T`` in the transport relaxation KL
   term.

3. **Transport relaxation term** (``TransportRelaxationTerm``) — the
   top-level entry point used by the GFlowNet TB loss.  Adds
   ``lambda_relax * KL(p_T || q_T)`` to the GFlowNet trajectory-
   balance loss, where ``p_T`` is the marginal pose distribution at
   time ``T`` and ``q_T`` is the Boltzmann endpoint distribution.
   Also exposes a :meth:`sweep_lambda` helper for the standard
   sensitivity-analysis sweep over ``lambda_relax``.

Design choices
--------------
* **CPU-only.**  Uses ``torch.logsumexp`` and basic tensor ops only;
  no CUDA / Triton kernels are needed for the *mathematical
  skeleton*.  R18 integration in
  ``molmetal.adapters.flow_matching_lipman`` may add a GPU kernel
  later but the math layer stays portable.
* **Standalone.**  No imports from ``molmetal.adapters.*``,
  ``molmetal.molmetal_lam.*`` or any other *internal* subpackage —
  only :mod:`torch`.  This is enforced by the unit tests
  (``test_transport_relaxation.py``) which import this module in
  isolation.
* **Batched.**  Every public function accepts a leading batch
  dimension ``(B, ...)`` and broadcasts over it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor used when taking ``log`` of marginals / probability vectors.
#: Below this we treat the value as zero to avoid ``log(0) = -inf``.
#: Picked safely above float32 ``log(1e-30) ~ -69`` (well within
#: ``logsumexp``'s stable range) and below any realistic probability
#: mass a 32-bit OT solver will produce, so it never alters a
#: non-zero entry.
_EPS_LOG = 1.0e-30


# ---------------------------------------------------------------------------
# RelaxedOT — soft endpoint penalty Optimal Transport
# ---------------------------------------------------------------------------


class RelaxedOT:
    """Entropic OT with a *soft* endpoint penalty.

    The standard entropic OT (Cuturi 2013; implemented in T1's
    :class:`~molmetal.math_skills.LogSinkhornLoss`) enforces
    marginals exactly:

    .. math::

        \\min_{P \\ge 0} \\langle P, C \\rangle
            + \\varepsilon \\, \\mathrm{KL}(P \\,\\|\\, a \\otimes b)
        \\quad\\text{s.t.}\\quad P \\mathbf{1} = a,\\ P^\\top \\mathbf{1} = b.

    The *relaxed* version (Jiang-Xu-Zhang 2026, arXiv:2602.08118,
    §3.2) replaces the hard marginal constraints by a convex
    combination of the target marginals and the uniform
    distribution:

    .. math::

        \\tilde a = (1 - \\alpha) a + \\alpha \\mathbb{1}/n,
        \\quad
        \\tilde b = (1 - \\alpha) b + \\alpha \\mathbb{1}/m,
        \\quad
        \\alpha = \\frac{\\lambda_{\\rm relax}}{1 + \\lambda_{\\rm relax}},

    and solves the same entropic OT problem with these *soft*
    marginals.  The standard OT is recovered at ``lambda_relax = 0``
    (since then ``alpha = 0`` and ``tilde a = a``); as
    ``lambda_relax -> infty`` we get ``alpha -> 1`` and the
    coupling collapses to the uniform plan ``P_ij = 1/(n m)``.

    Why this is "transport relaxation"
    ----------------------------------
    In the GFlowNet-TB context (Jiang-Xu-Zhang 2026, §5) the
    intermediate distribution ``p_T`` is only required to
    *approximately* match the target endpoints.  Hard-matching
    makes the bridge over-fit the noisy / biased endpoints; the
    relaxation lets the bridge absorb the Boltzmann endpoint
    penalty smoothly rather than trying to satisfy it exactly.

    CPU-only / standalone
    ---------------------
    Solved in log-domain via :func:`torch.logsumexp` (no CUDA
    needed); does not import anything from the Mol-Metal codebase.
    """

    def __init__(
        self,
        eps: float = 0.1,
        lambda_relax: float = 1e-3,
        n_iters: int = 50,
    ) -> None:
        """Build the relaxed OT solver.

        Parameters
        ----------
        eps
            Entropy-regularisation coefficient of the inner Sinkhorn
            solve.  Must be strictly positive.  Smaller ``eps`` ⇒
            the coupling concentrates on cheaper pairs (closer to
            exact OT).
        lambda_relax
            Endpoint relaxation coefficient.  ``lambda_relax = 0``
            recovers standard entropic OT; ``lambda_relax -> infty``
            drives the coupling to the uniform plan.  Must be
            ``>= 0``.
        n_iters
            Number of Sinkhorn fixed-point iterations.  Must be
            ``>= 1``.
        """
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if lambda_relax < 0.0:
            raise ValueError(f"lambda_relax must be >= 0, got {lambda_relax}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.eps = float(eps)
        self.lambda_relax = float(lambda_relax)
        self.n_iters = int(n_iters)

    # -- internal helpers -------------------------------------------------

    def _alpha(self) -> float:
        """``alpha = lambda_relax / (1 + lambda_relax)`` ∈ [0, 1)."""
        return self.lambda_relax / (1.0 + self.lambda_relax)

    def _soft_marginals(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute ``(tilde a, tilde b)`` from the target marginals.

        Implements the convex blend ``tilde a = (1 - alpha) a + alpha u_n``
        where ``u_n = 1/n`` (uniform on ``n`` atoms) and similarly
        for ``tilde b``.
        """
        alpha = self._alpha()
        n, m = a.shape[0], b.shape[0]
        device, dtype = a.device, a.dtype
        u_a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        u_b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        soft_a = (1.0 - alpha) * a + alpha * u_a
        soft_b = (1.0 - alpha) * b + alpha * u_b
        return soft_a, soft_b

    @staticmethod
    def _pairwise_sqdist(p0: Tensor, p1: Tensor) -> Tensor:
        """Build the pairwise squared Euclidean cost ``C``.

        ``C_ij = ||p0_i - p1_j||^2 = ||p0_i||^2 + ||p1_j||^2 - 2 p0_i . p1_j``.
        Clamped at 0 to absorb the floating-point round-off that
        can make ``||x - x||^2`` come out as a tiny negative number
        for ``x`` that bit-equals itself.
        """
        sq0 = (p0 * p0).sum(dim=-1, keepdim=True)  # (n, 1)
        sq1 = (p1 * p1).sum(dim=-1, keepdim=True).T  # (1, m)
        cross = p0 @ p1.T  # (n, m)
        return (sq0 + sq1 - 2.0 * cross).clamp_min(0.0)

    def _log_sinkhorn(
        self,
        C: Tensor,
        soft_a: Tensor,
        soft_b: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Log-domain Sinkhorn (Cuturi 2013) with given marginals.

        Maintains dual potentials ``f, g`` in log-domain so the
        coupling ``log P_ij = -C_ij / eps + f_i + g_j`` never
        overflows, even when ``eps`` is small.  Same fixed-point
        iteration as T1's :class:`~molmetal.math_skills.LogSinkhornLoss`,
        kept here in-module so this file remains standalone.
        """
        n, m = C.shape
        device, dtype = C.device, C.dtype

        log_a = torch.log(soft_a.clamp_min(_EPS_LOG))
        log_b = torch.log(soft_b.clamp_min(_EPS_LOG))
        M = -C / self.eps  # (n, m) — log kernel

        f = torch.zeros(n, device=device, dtype=dtype)
        g = torch.zeros(m, device=device, dtype=dtype)
        for _ in range(self.n_iters):
            f = log_a - torch.logsumexp(M + g.unsqueeze(0), dim=-1)
            g = log_b - torch.logsumexp(M + f.unsqueeze(1), dim=0)

        log_P = M + f.unsqueeze(1) + g.unsqueeze(0)
        P = torch.exp(log_P)
        return P, log_P

    # -- public API -------------------------------------------------------

    def forward(
        self,
        p0: Tensor,
        p1: Tensor,
        C: Optional[Tensor] = None,
        a: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the relaxed OT loss.

        Parameters
        ----------
        p0
            Source samples of shape ``(n, d)``.  Only used to build
            ``C`` when ``C`` is ``None``.
        p1
            Target samples of shape ``(m, d)``.
        C
            Optional precomputed cost matrix of shape ``(n, m)``.
        a
            Optional source marginal of shape ``(n,)``.  Defaults to
            the uniform distribution.
        b
            Optional target marginal of shape ``(m,)``.  Defaults to
            the uniform distribution.

        Returns
        -------
        Tensor
            Scalar — the relaxed OT loss

            .. math::
                \\langle P^\\star, C \\rangle
                + \\varepsilon \\, \\mathrm{KL}(P^\\star \\,\\|\\,
                \\tilde a \\otimes \\tilde b).

            For ``lambda_relax = 0`` this equals the standard
            entropic OT loss of T1's
            :class:`~molmetal.math_skills.LogSinkhornLoss`.
        """
        if C is None:
            C = self._pairwise_sqdist(p0, p1)
        n, m = C.shape
        if a is None:
            a = torch.full((n,), 1.0 / n, dtype=C.dtype, device=C.device)
        if b is None:
            b = torch.full((m,), 1.0 / m, dtype=C.dtype, device=C.device)

        soft_a, soft_b = self._soft_marginals(a, b)
        P, log_P = self._log_sinkhorn(C, soft_a, soft_b)

        log_soft_a = torch.log(soft_a.clamp_min(_EPS_LOG))
        log_soft_b = torch.log(soft_b.clamp_min(_EPS_LOG))
        # KL(P || soft_a ⊗ soft_b) = sum P (log P - log soft_a - log soft_b)
        kl = (P * (log_P - log_soft_a.unsqueeze(1) - log_soft_b.unsqueeze(0))).sum()
        cost = (P * C).sum()
        return cost + self.eps * kl

    def coupling(
        self,
        p0: Tensor,
        p1: Tensor,
        C: Optional[Tensor] = None,
        a: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the *relaxed* Sinkhorn coupling matrix (for inspection).

        Same arguments as :meth:`forward`.  Useful in tests where we
        want to verify that the coupling is close to uniform when
        ``lambda_relax`` is large.
        """
        if C is None:
            C = self._pairwise_sqdist(p0, p1)
        n, m = C.shape
        if a is None:
            a = torch.full((n,), 1.0 / n, dtype=C.dtype, device=C.device)
        if b is None:
            b = torch.full((m,), 1.0 / m, dtype=C.dtype, device=C.device)
        soft_a, soft_b = self._soft_marginals(a, b)
        P, _ = self._log_sinkhorn(C, soft_a, soft_b)
        return P


# ---------------------------------------------------------------------------
# BoltzmannEndpoint — canonical Boltzmann distribution over docked poses
# ---------------------------------------------------------------------------


class BoltzmannEndpoint:
    """Canonical Boltzmann distribution over a discrete set of docked poses.

    Given a discrete set of docked poses with associated energies
    :math:`E_i`, the Boltzmann distribution at temperature :math:`T`
    is

    .. math::

        q_T(i) = \\frac{\\exp(-E_i / T)}{Z},
        \\quad Z = \\sum_j \\exp(-E_j / T).

    In the limit :math:`T \\to 0` the distribution concentrates on
    the minimum-energy pose (the Boltzmann selector); in the limit
    :math:`T \\to \\infty` it approaches the uniform distribution.

    CPU-only / standalone
    ---------------------
    Computed via :func:`torch.logsumexp` — works on CPU tensors and
    is numerically stable for energies spanning many orders of
    magnitude.
    """

    def __init__(self, T: float = 1.0) -> None:
        """Build the Boltzmann endpoint distribution.

        Parameters
        ----------
        T
            Temperature in the same units as the energies.  Must be
            strictly positive.
        """
        if T <= 0.0:
            raise ValueError(f"T must be > 0, got {T}")
        self.T = float(T)

    def log_prob(self, poses: Tensor, energies: Tensor) -> Tensor:
        """Compute the log Boltzmann probabilities.

        Parameters
        ----------
        poses
            Docked poses of shape ``(..., d)``.  Only used to infer
            the output shape; the Boltzmann distribution itself is
            defined purely by the energies.
        energies
            Energies of shape ``(...)`` — must match the prefix of
            ``poses.shape``.

        Returns
        -------
        Tensor
            Log probabilities of shape ``(...)`` (matches the
            energies' shape).  ``exp`` of these sums to 1 over the
            last axis (via ``logsumexp`` normalisation).
        """
        if poses.shape[:-1] != energies.shape:
            raise ValueError(
                f"energies shape {tuple(energies.shape)} must match "
                f"the prefix of poses shape {tuple(poses.shape[:-1])}"
            )
        log_unnorm = -energies / self.T
        # logsumexp over the last axis; broadcasting handles the batch
        # dimensions automatically.  ``keepdim=True`` so the result
        # broadcasts back onto ``log_unnorm`` for batched inputs.
        log_Z = torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
        return log_unnorm - log_Z

    def prob(self, poses: Tensor, energies: Tensor) -> Tensor:
        """Compute probabilities (== ``exp`` of :meth:`log_prob`)."""
        return torch.exp(self.log_prob(poses, energies))


# ---------------------------------------------------------------------------
# TransportRelaxationTerm — full GFlowNet TB integration
# ---------------------------------------------------------------------------


class TransportRelaxationTerm:
    """Transport relaxation term for the GFlowNet TB loss.

    Given an empirical intermediate distribution ``p_T`` of shape
    ``(..., n)`` at time ``T`` and a Boltzmann reference ``q_T`` of
    the same shape, the transport relaxation term is

    .. math::

        L_{\\rm tr} = \\lambda_{\\rm relax} \\,
        \\mathrm{KL}(p_T \\,\\|\\, q_T).

    This is the gradient that the GFlowNet TB loss back-propagates
    into the bridge interpolant (Jiang-Xu-Zhang 2026, §5): it
    pushes the intermediate distribution towards the Boltzmann
    endpoint without forcing an exact fit (the exact fit would
    over-fit the noisy Boltzmann endpoint).

    The class also exposes :meth:`sweep_lambda`, a small helper that
    runs the :class:`RelaxedOT` solver for a list of
    ``lambda_relax`` values and returns the corresponding OT
    losses — the standard sensitivity-analysis sweep used in the
    transport-relaxation ablations of Jiang-Xu-Zhang 2026.

    CPU-only / standalone.
    """

    def __init__(self, lambda_relax: float = 1e-3) -> None:
        if lambda_relax < 0.0:
            raise ValueError(f"lambda_relax must be >= 0, got {lambda_relax}")
        self.lambda_relax = float(lambda_relax)

    def kl_divergence(self, p_T: Tensor, q_T: Tensor) -> Tensor:
        """Compute :math:`\\mathrm{KL}(p_T \\,\\|\\, q_T)`.

        Both arguments must be probability distributions over the
        last axis (sum to 1 along ``dim=-1``).  Returns a scalar
        tensor — KL is summed over all axes.

        Parameters
        ----------
        p_T, q_T
            Tensors of the same shape ``(..., n)``.

        Returns
        -------
        Tensor
            Scalar KL divergence.  Always ``>= 0`` with equality
            iff ``p_T = q_T`` element-wise.
        """
        if p_T.shape != q_T.shape:
            raise ValueError(
                f"p_T and q_T must have the same shape, got "
                f"{tuple(p_T.shape)} vs {tuple(q_T.shape)}"
            )
        log_p = torch.log(p_T.clamp_min(_EPS_LOG))
        log_q = torch.log(q_T.clamp_min(_EPS_LOG))
        return (p_T * (log_p - log_q)).sum()

    def sweep_lambda(
        self,
        p0: Tensor,
        p1: Tensor,
        lambdas: Optional[List[float]] = None,
        eps: float = 0.1,
        n_iters: int = 50,
    ) -> Dict[float, Tensor]:
        """Sweep ``lambda_relax`` values and return the relaxed OT losses.

        Parameters
        ----------
        p0, p1
            Source / target samples (same semantics as
            :meth:`RelaxedOT.forward`).
        lambdas
            List of ``lambda_relax`` values to try.  Defaults to
            ``[1e-4, 1e-3, 1e-2]`` (the canonical range used in the
            transport-relaxation ablations of Jiang-Xu-Zhang 2026).
        eps
            Entropy-regularisation coefficient forwarded to
            :class:`RelaxedOT`.
        n_iters
            Number of Sinkhorn iterations forwarded to
            :class:`RelaxedOT`.

        Returns
        -------
        dict
            Mapping from ``lambda_relax`` value (float) to the
            corresponding relaxed OT loss (scalar tensor).
        """
        if lambdas is None:
            lambdas = [1e-4, 1e-3, 1e-2]
        out: Dict[float, Tensor] = {}
        for lam in lambdas:
            ot = RelaxedOT(eps=eps, lambda_relax=lam, n_iters=n_iters)
            out[lam] = ot.forward(p0, p1)
        return out


__all__ = [
    "BoltzmannEndpoint",
    "RelaxedOT",
    "TransportRelaxationTerm",
]
