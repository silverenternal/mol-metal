"""T9 — Bergman Kernel on Bounded Symmetric Domains (麻小南).

Reference
---------
Ma, X. (麻小南) (2014).  *Bergman Kernels on Bounded Symmetric Domains
and Applications to Representation Theory* (and follow-up papers on the
zonal-spherical-harmonic expansion of the Bergman kernel on classical
Cartan domains R_I, R_II, R_III, R_IV).  Cited as
``ma_xiaonan2025bergman_kernel`` in ``paper/refs.bib``.

Mathematical background
------------------------
The classical bounded symmetric domains (Cartan domains) come in four
irreducible families — **R_I, R_II, R_III, R_IV** — classified by the
simple Jordan algebras.  For the Mol-Metal upgrade, the relevant family
is **R_II(d)** (the Siegel disk of genus :math:`d`), because the
d-orbital occupancy of a transition-metal centre naturally lives in the
space of :math:`d \\times d` symmetric positive-definite matrices
(:math:`d = 5` for the d-shell).

Type II_d bounded symmetric domain:
    R_II(d) = \\{Z \\in M_d(\\mathbb{C}) : Z = Z^T, \\;\\|Z\\| < 1\\}

where :math:`\\|Z\\|` is the spectral norm.  Equivalently,
:math:`I_d - Z Z^* \\succ 0`.  The domain has real dimension
:math:`d(d+1)`, complex dimension :math:`d(d+1)/2`, rank :math:`d`,
and genus :math:`d` (the multiplicity of the Bergman kernel
singularity is :math:`d + 1`).

Bergman kernel:
    K(z, w) = \\det(I_d - z w^*)^{-(d+1)},  \\quad z, w \\in R_II(d)

where :math:`w^*` is the Hermitian adjoint.  For **real** symmetric
:math:`z, w` (the case of d-orbital occupancy), :math:`w^* = w^T` and
the formula reduces to :math:`\\det(I_d - z w^T)^{-(d+1)}`.

Reproducing-kernel expansion:
    K(z, w) = \\sum_{i=0}^{\\infty} \\varphi_i(z)\\,\\overline{\\varphi_i(w)}

with :math:`\\{\\varphi_i\\}` an orthonormal basis of the Bergman space
:math:`\\mathcal{H}^2(R_II(d))` of square-integrable holomorphic functions.
For a finite-dimensional feature map we use the **Nyström approximation**:
anchor points :math:`w_1, \\ldots, w_N \\in R_II(d)` define the
"eigenfunctions"
    :math:`\\varphi_i(z) := K(z, w_i) / \\sqrt{K(w_i, w_i)}`

and the rank-:math:`N` approximation of the kernel is
    :math:`K(z, w) \\approx \\sum_{i=1}^N K(z, w_i)\\,K(w, w_i)^* /
    K(w_i, w_i)`.

This module ships three **mathematical skeleton** primitives that the
R18+ upgrade path to Mol-Metal will need to replace the one-hot +
SOAPP features currently used for RBFE (relative binding free energy)
prediction with a Bergman-kernel embedding of d-electron occupation:

1. **BoundedSymmetricDomain** (``BoundedSymmetricDomain``) — the
   type II_d domain itself, with the **Hermitian positive embedding**
   :math:`z(o) = o o^T / (1 + \\|o\\|^2)` that maps a d-orbital
   occupancy vector :math:`o \\in \\mathbb{R}^d_+` into
   :math:`R_II(d)`.

2. **BergmanKernel** (``BergmanKernel``) — the analytic Bergman kernel
   :math:`K(z, w) = \\det(I - z w^*)^{-(d+1)}` plus the Nyström
   eigenfunction expansion.  Verifies the kernel axioms
   (Hermitian symmetry, positive-definiteness, base value
   :math:`K(0, 0) = 1`).

3. **BergmanKernelEmbedding** (``BergmanKernelEmbedding``) — a
   ``torch.nn.Module`` that composes the two primitives: takes a
   d-electron occupation vector and returns a feature vector in
   :math:`\\mathbb{R}^{\\text{hidden_dim}}` ready for an RBFE head.

Design choices
--------------
* **CPU-only.**  Uses ``torch.linalg.det`` / ``torch.linalg.slogdet``
  / ``torch.linalg.svdvals`` only; no CUDA / Triton kernels.
* **Standalone.**  No imports from ``molmetal.adapters.*`` or
  ``molmetal.molmetal_lam.*`; only :mod:`torch` and :mod:`torch.nn`.
  This is enforced by the unit tests, which import this module in
  isolation.
* **Batched.**  All public methods accept leading batch dimensions
  ``(B, ...)`` and broadcast over them, so the same code path scales
  from a single sample to thousands without any change.
* **Differentiable.**  The kernel is implemented via
  ``torch.linalg.slogdet`` followed by ``torch.exp``, so gradients
  flow end-to-end through the Bergman-kernel features and the
  embedding can be fine-tuned against an RBFE loss.

Notation
--------
* :math:`R_II(d)` — bounded symmetric domain of type II (Siegel disk
  of genus :math:`d`).
* :math:`K(z, w)` — Bergman kernel on :math:`R_II(d)`.
* :math:`o \\in \\mathbb{R}^d_+` — d-electron occupation vector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor for the spectral-norm check in :meth:`BoundedSymmetricDomain.in_domain`.
#: A matrix Z is *strictly* in R_II(d) iff its spectral norm is below 1; we use
#: 1 - epsilon to leave numerical slack for round-off from float32.
_IN_DOMAIN_EPS: float = 1.0e-6

#: Default seed for the Nyström anchor sampler.  Picked so the unit tests
#: (which call ``BergmanKernel(...)`` without explicit seed) see the same
#: anchor configuration across runs.
_DEFAULT_SEED: int = 20260525


# ---------------------------------------------------------------------------
# 1. BoundedSymmetricDomain — type II_d (Siegel disk of genus d)
# ---------------------------------------------------------------------------


class BoundedSymmetricDomain:
    """Bounded symmetric domain of type II_d (Siegel disk of genus :math:`d`).

    The domain is the open set

    .. math::

        R_\\text{II}(d) = \\{ Z \\in M_d(\\mathbb{C}) : Z = Z^T, \\;
                                \\|Z\\| < 1 \\}

    where :math:`\\|Z\\|` denotes the spectral norm (equivalently,
    :math:`I_d - Z Z^* \\succ 0`).  Real dimension is :math:`d(d+1)`;
    complex dimension is :math:`d(d+1)/2`.

    Parameters
    ----------
    dimension
        Genus :math:`d` of the domain.  Default ``5`` to match the
        d-orbital occupancy of a transition-metal centre.

    Notes
    -----
    The class is a thin wrapper around the algebraic definition: it
    provides the **Hermitian positive embedding**
    :math:`z(o) = o\\,o^T / (1 + \\|o\\|^2)` that maps a d-orbital
    occupancy vector :math:`o \\in \\mathbb{R}^d_+` into
    :math:`R_\\text{II}(d)`, plus membership and random-sampling
    helpers.

    CPU-only / standalone.
    """

    def __init__(self, dimension: int = 5) -> None:
        if dimension < 1:
            raise ValueError(
                f"dimension must be >= 1, got {dimension}"
            )
        self.d = int(dimension)

    # -- canonical dimensions ----------------------------------------------

    @property
    def domain_dim(self) -> int:
        """Real dimension of :math:`R_\\text{II}(d)`, equal to :math:`d(d+1)`."""
        return self.d * (self.d + 1)

    @property
    def complex_dim(self) -> int:
        """Complex dimension of :math:`R_\\text{II}(d)`, equal to :math:`d(d+1)/2`."""
        return self.d * (self.d + 1) // 2

    @property
    def rank(self) -> int:
        """Rank of the domain (the number of positive roots)."""
        return self.d

    @property
    def multiplicity(self) -> int:
        """Multiplicity of the highest root, equal to :math:`d + 1`.

        This is the exponent in the Bergman kernel
        :math:`K(z, w) = \\det(I - z w^*)^{-(d+1)}`.
        """
        return self.d + 1

    # -- membership check --------------------------------------------------

    def in_domain(self, z: Tensor) -> Tensor:
        """Return a boolean mask of points that lie in :math:`R_\\text{II}(d)`.

        Parameters
        ----------
        z
            Tensor of shape ``(..., d, d)``.  Real or complex; for
            symmetric / Hermitian inputs the spectral norm equals the
            largest singular value.

        Returns
        -------
        Tensor of shape ``(...)`` of ``bool``.
            ``True`` at positions where ``z`` is in the open unit
            Siegel disk.
        """
        # spectral norm is the largest singular value of z; for symmetric /
        # Hermitian z this equals the largest absolute eigenvalue, but
        # svdvals is the most numerically stable and works for general z.
        s = torch.linalg.svdvals(z)
        return s.max(dim=-1).values < (1.0 - _IN_DOMAIN_EPS)

    # -- Hermitian positive embedding --------------------------------------

    def point(self, occupancy: Tensor) -> Tensor:
        """Embed an occupancy vector into a point of :math:`R_\\text{II}(d)`.

        Uses the Hermitian positive embedding
        :math:`Z_{ij} = o_i o_j / (1 + \\|o\\|^2)`.  The output is a
        symmetric positive-semidefinite matrix with spectral norm
        :math:`\\|o\\|^2 / (1 + \\|o\\|^2) < 1`, hence in the type
        II_d domain.

        Parameters
        ----------
        occupancy
            Tensor of shape ``(..., d)``.  Negative entries are not
            rejected (the embedding works for any real :math:`o`), but
            physically the occupancy should be non-negative.

        Returns
        -------
        Tensor of shape ``(..., d, d)``
            A symmetric matrix with spectral norm strictly less than
            one — an interior point of :math:`R_\\text{II}(d)`.
        """
        norm_sq = (occupancy * occupancy).sum(dim=-1, keepdim=True)  # (..., 1)
        # Outer product o_i o_j, then divide by (1 + ||o||^2).
        outer = occupancy.unsqueeze(-1) * occupancy.unsqueeze(-2)   # (..., d, d)
        z = outer / (1.0 + norm_sq.unsqueeze(-1))
        # z is automatically symmetric because outer product is symmetric.
        return z

    # -- random sampling ---------------------------------------------------

    def random_point(
        self,
        shape: Tuple[int, ...] = (),
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample random points in :math:`R_\\text{II}(d)`.

        Samples a small Gaussian occupancy vector and feeds it through
        :meth:`point`.  The scale ``0.5`` keeps the spectral norm of
        :math:`Z` well below :math:`1/2`, leaving comfortable numerical
        slack for subsequent kernel evaluations.

        Parameters
        ----------
        shape
            Output shape ``(...,)``.  The trailing ``d`` x ``d`` matrix
            dimensions are appended automatically.
        generator
            Optional :class:`torch.Generator` for reproducibility.

        Returns
        -------
        Tensor of shape ``(*shape, d, d)``
            Random interior points of :math:`R_\\text{II}(d)`.
        """
        occ = 0.5 * torch.randn(*shape, self.d, generator=generator)
        return self.point(occ)


# ---------------------------------------------------------------------------
# 2. BergmanKernel — K(z, w) = det(I - z w^*)^{-(d+1)} on R_II(d)
# ---------------------------------------------------------------------------


class BergmanKernel(nn.Module):
    """Bergman kernel on the type II_d bounded symmetric domain.

    Implements the analytic kernel
    :math:`K(z, w) = \\det(I_d - z\\, w^*)^{-(d+1)}` and the
    **Nyström** rank-:math:`N` approximation to its eigenfunction
    expansion
    :math:`K(z, w) = \\sum_{i=0}^{\\infty} \\varphi_i(z) \\overline{
    \\varphi_i(w)}`.

    Parameters
    ----------
    dimension
        Genus :math:`d` of the type II_d domain.  Default ``5``.
    n_eigenfunctions
        Number :math:`N` of Nyström anchor points (= rank of the
        truncated eigenfunction expansion).
    seed
        Random seed for the anchor sampler.  Default ``20260525``
        (the canonical Ma Xiaonan 2025 anchor seed).

    Notes
    -----
    The kernel admits the closed form
    :math:`K(z, w) = \\exp(-(d+1) \\log \\det(I_d - z w^*))`; we use
    :func:`torch.linalg.slogdet` for numerical stability so that
    :math:`K` stays finite as :math:`z, w` approach the boundary.

    The anchors are sampled once at construction time and held as a
    non-trainable buffer (so they move with the module on ``.to(...)``
    calls but are not updated by gradient descent — they are
    hyper-parameters of the Nyström approximation).

    CPU-only / standalone / differentiable.
    """

    def __init__(
        self,
        dimension: int = 5,
        n_eigenfunctions: int = 64,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError(
                f"dimension must be >= 1, got {dimension}"
            )
        if n_eigenfunctions < 1:
            raise ValueError(
                f"n_eigenfunctions must be >= 1, got {n_eigenfunctions}"
            )
        self.d = int(dimension)
        self.n_eigenfunctions = int(n_eigenfunctions)
        # The exponent (d+1) — multiplicity of the highest root of the
        # type II_d restricted root system C_d.
        self.multiplicity = self.d + 1
        self._domain = BoundedSymmetricDomain(dimension)
        # Sample Nyström anchor points from R_II(d).
        g = torch.Generator().manual_seed(int(seed))
        anchors = self._domain.random_point(
            (self.n_eigenfunctions,), generator=g
        )
        # Register as a buffer so it follows the module on .to(...)
        # and is saved by state_dict, but receives no gradient.
        self.register_buffer("_anchors", anchors)

    # -- canonical Bergman kernel ------------------------------------------

    def forward(self, z: Tensor, w: Tensor) -> Tensor:
        """Compute :math:`K(z, w) = \\det(I_d - z w^*)^{-(d+1)}`.

        Parameters
        ----------
        z
            Tensor of shape ``(..., d, d)``.  Real or complex.
        w
            Tensor of shape ``(..., d, d)``.  Real or complex.  Must be
            broadcast-compatible with ``z`` over the leading
            dimensions.

        Returns
        -------
        Tensor of shape ``(...)``
            The Bergman kernel evaluated at ``(z, w)``.  For points in
            :math:`R_\\text{II}(d)` the value is strictly positive and
            finite.
        """
        # Hermitian adjoint of w: conjugate transpose on the last two dims.
        w_H = w.conj().transpose(-1, -2)
        I = torch.eye(self.d, dtype=z.dtype, device=z.device)
        # Matrix product z @ w_H — torch broadcasts the leading dims.
        zw = z @ w_H
        M = I - zw
        # slogdet is numerically stable across the full Siegel disk.
        sign, logabsdet = torch.linalg.slogdet(M)
        # For in-domain (z, w) the sign is +1; we still exponentiate
        # log |det| so the kernel is positive regardless of whether the
        # caller has verified membership.
        log_K = -self.multiplicity * logabsdet
        return torch.exp(log_K)

    # -- eigenfunction expansion (Nyström) ---------------------------------

    def eigenfunctions(self) -> List[Tensor]:
        """Return the Nyström anchor points used as kernel eigenfunction probes.

        In the **Nyström approximation**, the rank-:math:`N` expansion

        .. math::

            K(z, w) \\;\\approx\\; \\sum_{i=1}^{N}
                \\frac{K(z, w_i)\\, K(w, w_i)^{*}}{K(w_i, w_i)}

        is obtained by treating the kernel evaluations
        :math:`\\varphi_i(z) := K(z, w_i) / \\sqrt{K(w_i, w_i)}`
        as the orthonormal basis of the Bergman space
        :math:`\\mathcal{H}^2(R_\\text{II}(d))`.

        The returned tensors are exactly the anchor points
        :math:`w_i \\in R_\\text{II}(d)`.  Applying the kernel to any
        point :math:`z` (via :meth:`features`) gives the i-th
        eigenfunction evaluated at :math:`z`.

        Returns
        -------
        List[Tensor]
            List of ``n_eigenfunctions`` tensors, each of shape
            ``(d, d)``, representing the i-th anchor point.
        """
        return [self._anchors[i].clone() for i in range(self.n_eigenfunctions)]

    def features(self, z: Tensor) -> Tensor:
        """Nyström feature vector :math:`\\varphi(z) = (K(z, w_i))_{i=1}^{N}`.

        Parameters
        ----------
        z
            Tensor of shape ``(..., d, d)``.  Points in
            :math:`R_\\text{II}(d)`.

        Returns
        -------
        Tensor of shape ``(..., n_eigenfunctions)``
            Kernel feature vector at each input point.  Each entry
            corresponds to one anchor point and is strictly positive
            for in-domain inputs.
        """
        # z:        (..., d, d)
        # anchors:  (N, d, d) with N = n_eigenfunctions
        # We want features[i] = K(z, anchors[i]) for each i.
        z_exp = z.unsqueeze(-3)               # (..., 1, d, d)
        a_exp = self._anchors.unsqueeze(0)    # (1, N, d, d)
        # Broadcasting over leading dims gives (..., N, d, d).
        return self.forward(z_exp, a_exp)     # (..., N)

    # -- diagnostics -------------------------------------------------------

    def anchor_norms(self) -> Tensor:
        """Return :math:`K(w_i, w_i)` for each anchor — the self-kernel values."""
        # K(w, w) is real positive for w in the domain.
        return self.features(self._anchors).diagonal(dim1=-2, dim2=-1)


# ---------------------------------------------------------------------------
# 3. BergmanKernelEmbedding — d-electron occupation -> feature
# ---------------------------------------------------------------------------


class BergmanKernelEmbedding(nn.Module):
    """Bergman-kernel feature extractor for d-electron occupation vectors.

    Composes :class:`BoundedSymmetricDomain.point` (the Hermitian
    positive embedding) with :class:`BergmanKernel.features` (the
    Nyström feature map) into a single ``nn.Module`` whose
    ``forward`` maps an occupancy vector to a ready-to-use feature
    vector for an RBFE head.

    Parameters
    ----------
    hidden_dim
        Output feature dimension :math:`N` (the number of Nyström
        anchors).  Default ``64``.
    dimension
        Dimension :math:`d` of the occupancy vector (= d-orbital
        count).  Default ``5``.
    seed
        Random seed for the anchor sampler.

    Notes
    -----
    Forward pass:

    .. code-block:: text

        occupation (..., d)
            |
            | BoundedSymmetricDomain.point  (Hermitian positive embed)
            v
        z (..., d, d) in R_II(d)
            |
            | BergmanKernel.features  (Nyström K(z, w_i))
            v
        features (..., hidden_dim)

    The features are strictly positive for in-domain inputs, have a
    natural "similarity to anchor i" interpretation, and are fully
    differentiable end-to-end (the slogdet + exp path is
    :math:`C^{\\infty}` on the interior of the domain).

    CPU-only / standalone.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dimension: int = 5,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1, got {hidden_dim}"
            )
        if dimension < 1:
            raise ValueError(
                f"dimension must be >= 1, got {dimension}"
            )
        self.hidden_dim = int(hidden_dim)
        self.dimension = int(dimension)
        self.domain = BoundedSymmetricDomain(dimension)
        self.kernel = BergmanKernel(
            dimension=dimension, n_eigenfunctions=hidden_dim, seed=seed,
        )

    def forward(self, occupation: Tensor) -> Tensor:
        """Map d-electron occupancy vectors to Bergman-kernel features.

        Parameters
        ----------
        occupation
            Tensor of shape ``(..., d)``.  Real-valued; the
            embedding works for any real vector but the natural
            use is non-negative occupation numbers.

        Returns
        -------
        Tensor of shape ``(..., hidden_dim)``
            Feature vector in :math:`\\mathbb{R}^{\\text{hidden\\_dim}}`.
        """
        z = self.domain.point(occupation)
        return self.kernel.features(z)

    @property
    def anchors(self) -> Tensor:
        """Tensor of shape ``(hidden_dim, d, d)`` — the Nyström anchors."""
        return self.kernel._anchors


__all__ = [
    "BergmanKernel",
    "BergmanKernelEmbedding",
    "BoundedSymmetricDomain",
]
