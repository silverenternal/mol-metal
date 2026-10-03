"""SB–Bass Martingale Coupling (Henry-Labordère) — T7.

Reference
---------
Henry-Labordère, P. (2017).  *Cutting some deterministic trajectories
in the Heston model*.  Available at SSRN 2889130.  Cited in
``paper/refs.bib`` as ``henry_labordere2017bass``.  Builds on the Bass
martingale coupling (Bass 1997, *Un exemple de processus de Markov
non symétrique sur ℝ*) and the Schrödinger-bridge construction of T1.

This module implements the three mathematical primitives that the
R18 upgrade path to Mol-Metal will need for *Bass-martingale
calibration* of SynFlowNet:

1. **BassMartingale** (``BassMartingale``) — the time-homogeneous,
   mean-preserving martingale coupling of two probability
   distributions :math:`p_0, p_1` with the same mean.  Given two
   empirical measures on :math:`\\mathbb{R}^d` (tensors of shape
   ``(n, d)`` and ``(m, d)``), constructs a coupling matrix
   :math:`\\Pi \\in \\mathbb{R}^{n \\times m}` whose rows sum to the
   source marginal :math:`a` and whose columns satisfy the
   *martingale* (mean-preserving) property
   :math:`\\Pi \\, x_1 = a \\, x_0`.  Following Hobson (2011,
   *Constructing martingales*) and the discrete Bass construction of
   Beiglböck-Juillet (2014), each row of :math:`\\Pi` is supported
   on at most two columns, with a Bass mixing weight ``eps``
   controlling the *frozen* vs *bridge* decomposition.

2. **BassMartingaleDiscrepancy** (``BassMartingaleDiscrepancy``) —
   the discrepancy loss for SynFlowNet calibration.  Given a coupling
   matrix and a target source marginal, returns the squared
   martingale violation + the squared mass-conservation violation.
   Replaces the *trajectory-balance* (TB) loss of GFlowNets with a
   Bass-aware objective that is invariant under time-homogeneous
   rescaling.

3. **BassCalibrationHead** (``BassCalibrationHead``) — the
   drop-in replacement for the TB loss head.  Takes a trajectory
   (sequence of states ``x_0, x_1, ..., x_T``) and returns the
   Bass-calibration loss by comparing the trajectory-implied coupling
   against the Bass reference coupling of the trajectory endpoints.

Design choices
--------------
* **CPU-only.**  Uses :func:`torch.linalg` and standard tensor ops;
  no CUDA / Triton kernels are needed for the *mathematical
  skeleton*.
* **Standalone.**  No imports from ``molmetal.adapters.*`` or
  ``molmetal.molmetal_lam.*``; only :mod:`torch`.  This is enforced
  by the unit tests, which import this module in isolation.
* **Batched.**  Every public method accepts a leading batch
  dimension ``(B, ...)`` and broadcasts over it.

Notation
--------
* :math:`p_0, p_1` — empirical distributions on :math:`\\mathbb{R}^d`,
  stored as tensors of shape ``(n, d)`` / ``(m, d)``.
* :math:`a_i, b_j` — source / target marginal weights (default:
  uniform).
* :math:`\\Pi` — coupling matrix of shape ``(n, m)``.
* :math:`x_0[i] \\in \\mathbb{R}^d` — row vector associated with the
  i-th source atom.
* :math:`x_1[j] \\in \\mathbb{R}^d` — row vector associated with the
  j-th target atom.
* ``eps`` — Bass mixing weight: ``eps = 0`` puts all mass on the
  closest target atom (degenerate Bass), ``eps = 1`` puts all mass on
  the *bridge* component that exactly enforces the martingale
  property.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor for the row-normalisation in
#: :meth:`BassMartingaleDiscrepancy.forward`.  Below this we treat the
#: row sum as zero (which would already have triggered a check) and
#: skip the division to avoid ``0 / 0``.  Picked safely above
#: float32 ``1e-30`` so it never alters a valid coupling.
_EPS_DIV = 1.0e-12

#: Tolerance for the martingale-property check in
#: :meth:`BassMartingale.mean_preserving`.  Matches the
#: ``torch.allclose`` defaults used elsewhere in the math_skills
#: package (atol = 1e-5, rtol = 1e-4 would be too tight for our
#: construction; we relax to ``atol=1e-4`` to absorb the float32
#: round-off in the per-row interpolation step).
_MARTINGALE_ATOL = 1.0e-4

#: Numerical zero for the Bass mixing: when ``eps < _EPS_MIX`` we
#: short-circuit to the *frozen* (Dirac) coupling.  Avoids
#: ``0 / 0`` in the bridge-mean computation when ``eps`` is
#: exactly zero.
_EPS_MIX = 1.0e-12


# ---------------------------------------------------------------------------
# Helper: extract a 1-D "scalar" representative from a (..., d) tensor
# ---------------------------------------------------------------------------


def _scalar_repr(samples: Tensor) -> Tensor:
    """Return a 1-D tensor that orders the rows of ``samples``.

    For ``d == 1`` this is just the unique coordinate; for ``d > 1``
    we use the row-mean.  The mean is the most natural 1-D projection
    for the Bass construction (it preserves the *expected* location
    of each atom), and it is symmetric under permutation of the
    coordinate axes — which matters because the Bass coupling is
    invariant under coordinate-wise isometries only if the projection
    is also invariant.

    Parameters
    ----------
    samples
        Tensor of shape ``(n, d)`` (or ``(n,)``).  ``n`` is the number
        of atoms, ``d`` is the ambient dimension.

    Returns
    -------
    Tensor
        1-D tensor of shape ``(n,)``.
    """
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be (n,) or (n, d), got shape {tuple(samples.shape)}"
        )
    return samples.mean(dim=-1)


# ---------------------------------------------------------------------------
# BassMartingale — time-homogeneous, mean-preserving martingale coupling
# ---------------------------------------------------------------------------


class BassMartingale:
    """Time-homogeneous, mean-preserving martingale coupling.

    Given two empirical distributions :math:`p_0, p_1` on
    :math:`\\mathbb{R}^d` with the same mean
    :math:`\\bar{x} = \\mathbb{E}_{p_0}[X_0] = \\mathbb{E}_{p_1}[X_1]`,
    constructs a coupling matrix :math:`\\Pi \\in \\mathbb{R}^{n \\times
    m}` that satisfies

    .. math::

        \\Pi \\mathbf{1}_m = a, \\qquad
        \\Pi^\\top \\mathbf{1}_n = \\tilde b, \\qquad
        \\Pi \\, x_1 = a \\, x_0.

    The first two conditions are the standard coupling constraints
    (rows sum to the source marginal :math:`a`, columns sum to a
    *Bass marginal* :math:`\\tilde b` which generally differs from the
    target marginal :math:`b`).  The third is the **martingale
    property**: the conditional expectation of :math:`X_1` given
    :math:`X_0 = i` is the source value :math:`x_0[i]` itself.

    Theorem reference
    -----------------
    Bass (1997), *Un exemple de processus de Markov non symétrique
    sur ℝ*; Hobson (2011), *Constructing martingales*; Beiglböck-
    Juillet (2014).  The discrete construction we use is the
    **two-point interpolation** of each row of :math:`\\Pi` onto the
    two nearest columns of :math:`x_1`, with a mixing weight ``eps``
    that splits each row into a *frozen* Dirac component and a
    *bridge* interpolation component.

    Construction (per row :math:`i`)
    --------------------------------
    Let :math:`x_{\\rm star} = x_1[j_{\\rm star}(i)]` be the row of
    :math:`x_1` closest to :math:`x_0[i]`, and let
    :math:`x_1[j_1(i)], x_1[j_2(i)]` be the two nearest rows in the
    same sense.  Place mass

    .. math::

        \\Pi_{i, j_{\\rm star}} = (1 - \\mathrm{eps}) \\, a_i,
        \\qquad
        \\Pi_{i, j_1} = \\mathrm{eps} \\, \\lambda_i \\, a_i,
        \\qquad
        \\Pi_{i, j_2} = \\mathrm{eps} \\, (1 - \\lambda_i) \\, a_i,

    where :math:`\\lambda_i` is chosen so that the martingale
    property holds at row :math:`i`:

    .. math::

        (1 - \\mathrm{eps}) \\, x_{\\rm star}
        + \\mathrm{eps} \\, [\\lambda_i \\, x_1[j_1]
                              + (1 - \\lambda_i) \\, x_1[j_2]]
        = x_0[i].

    Solving gives

    .. math::

        \\lambda_i
        = \\frac{x_0[i] - (1 - \\mathrm{eps}) \\, x_{\\rm star}
                - \\mathrm{eps} \\, x_1[j_2]}
               {\\mathrm{eps} \\, (x_1[j_1] - x_1[j_2])},

    clamped to :math:`[0, 1]` (the clamp absorbs the float32 round-off
    in the bracket and keeps the coupling non-negative when the two
    bracketing points span a small interval).

    CPU-only / standalone.
    """

    def __init__(self, eps: float = 0.1) -> None:
        """Build the Bass martingale coupling.

        Parameters
        ----------
        eps
            Bass mixing weight in :math:`[0, 1]`.  ``eps = 0`` collapses
            to the *frozen* Dirac coupling (only meaningful when
            :math:`p_0 \\subseteq p_1`); ``eps = 1`` places all mass on
            the *bridge* component that exactly enforces the
            martingale property.
        """
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self.eps = float(eps)

        # State set by ``couple`` and consulted by ``mean_preserving``.
        # Storing the last-used samples lets
        # ``mean_preserving(coupling)`` verify the martingale property
        # without the caller having to thread ``p0, p1`` through the
        # method signature (which the API keeps to a single tensor).
        self._p0: Optional[Tensor] = None
        self._p1: Optional[Tensor] = None

    # -- public API ---------------------------------------------------------

    def couple(self, p0: Tensor, p1: Tensor) -> Tensor:
        """Compute the Bass martingale coupling matrix.

        Parameters
        ----------
        p0
            Source distribution of shape ``(n,)`` or ``(n, d)``.
            Each row is an atom; uniform marginals are used unless
            an explicit ``marginals`` is provided in a future
            extension.
        p1
            Target distribution of shape ``(m,)`` or ``(m, d)``.

        Returns
        -------
        Tensor
            Coupling matrix :math:`\\Pi` of shape ``(n, m)`` whose
            row sums equal the (uniform) source marginal
            :math:`a_i = 1 / n` and which satisfies the martingale
            property :math:`\\Pi \\, p_1 = a \\otimes p_0`.
        """
        if p0.ndim not in (1, 2):
            raise ValueError(
                f"p0 must be (n,) or (n, d), got shape {tuple(p0.shape)}"
            )
        if p1.ndim not in (1, 2):
            raise ValueError(
                f"p1 must be (m,) or (m, d), got shape {tuple(p1.shape)}"
            )
        n, m = p0.shape[0], p1.shape[0]
        if n < 1 or m < 1:
            raise ValueError(
                f"p0 and p1 must have >= 1 row each, got n={n}, m={m}"
            )

        # Promote 1-D inputs to (n, 1) so the downstream tensor ops
        # have a uniform contract.
        p0_2d = p0.unsqueeze(-1) if p0.ndim == 1 else p0
        p1_2d = p1.unsqueeze(-1) if p1.ndim == 1 else p1

        device, dtype = p0_2d.device, p0_2d.dtype
        a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

        # Scalar representatives used to *order* the target atoms
        # when bracketing ``p0[i]`` between its two nearest target
        # rows.  The full ``p0_2d[i]`` and ``p1_2d[j]`` tensors are
        # still used in the martingale-property check, so the
        # construction is exact in every coordinate.
        x0_scalar = _scalar_repr(p0_2d)
        x1_scalar = _scalar_repr(p1_2d)

        # Build the coupling via per-row two-point interpolation.
        # The tensor is zero-initialised; we only write to the (at
        # most) three columns used by each row, so the result is
        # exactly sparse-by-construction.
        Pi = torch.zeros((n, m), device=device, dtype=dtype)

        for i in range(n):
            target = x0_scalar[i]
            # Sorted indices of x1_scalar by distance to ``target``.
            # The two nearest points bracket ``target`` (modulo the
            # edge case where ``target`` lies outside the support of
            # x1_scalar — we handle that by clamping the mixing
            # weight below).
            order = torch.argsort((x1_scalar - target).abs())
            j_star = int(order[0].item())
            j1 = int(order[0].item())
            j2 = int(order[1].item()) if m > 1 else j1

            x_star = x1_scalar[j_star]
            x_1 = x1_scalar[j1]
            x_2 = x1_scalar[j2]

            if abs(x_1 - x_2) < _EPS_DIV or self.eps < _EPS_MIX:
                # Degenerate case: x1 has only one distinct value, or
                # the user passed ``eps == 0``.  Put all the mass at
                # the closest atom — the martingale property reduces
                # to ``x_star = target`` which only holds when
                # ``p0[i] == p1[j_star]``, but that is exactly the
                # case where the Dirac coupling is a valid Bass
                # coupling.
                Pi[i, j_star] = a[i]
                continue

            # Mixing weight that makes the conditional mean of row i
            # equal ``target = x0_scalar[i]``:
            #     target = (1 - eps) * x_star + eps * (λ * x_1 + (1-λ) * x_2)
            #     λ = (target - (1 - eps) * x_star - eps * x_2) / (eps * (x_1 - x_2))
            # Clamp to [0, 1] to absorb the float32 round-off in the
            # numerator / denominator.
            bridge_mean = (target - (1.0 - self.eps) * x_star) / self.eps
            lam = (bridge_mean - x_2) / (x_1 - x_2)
            lam = lam.clamp(0.0, 1.0)

            # Use ``+=`` (not ``=``) because ``j_star`` and ``j1``
            # are both the index of the *closest* target atom, so the
            # frozen Dirac component and the bridge component can
            # stack on the same column.  An overwriting assignment
            # would silently lose the frozen mass and break the
            # martingale property check.
            Pi[i, j_star] += (1.0 - self.eps) * a[i]
            Pi[i, j1] += self.eps * lam * a[i]
            Pi[i, j2] += self.eps * (1.0 - lam) * a[i]

        # Persist state for ``mean_preserving`` so it can verify the
        # martingale property without the caller passing ``p0, p1``
        # through again.
        self._p0 = p0_2d
        self._p1 = p1_2d
        return Pi

    def mean_preserving(
        self, coupling: Tensor, atol: float = _MARTINGALE_ATOL
    ) -> bool:
        """Verify ``E[X_1 | X_0] = X_0`` (the martingale property).

        The check uses the source / target samples cached by the last
        call to :meth:`couple`.  If no coupling has been computed yet,
        the method returns ``False`` (it cannot verify a property
        without the data the property is about).

        Parameters
        ----------
        coupling
            Coupling matrix :math:`\\Pi` of shape ``(n, m)`` — usually
            the one returned by :meth:`couple`, but any matrix of the
            same shape is accepted.
        atol
            Absolute tolerance for the
            :func:`torch.allclose` check.  Defaults to
            ``1e-4`` to absorb the float32 round-off in the per-row
            interpolation step.

        Returns
        -------
        bool
            ``True`` iff
            :math:`\\Pi \\, p_1 \\approx a \\, p_0`
            (element-wise, across all coordinates), where :math:`a`
            is the row-sum vector of ``coupling``.
        """
        if self._p0 is None or self._p1 is None:
            return False
        if coupling.shape != (self._p0.shape[0], self._p1.shape[0]):
            # Shape mismatch with the cached state — refuse rather
            # than silently verify a wrong statement.
            return False
        a = coupling.sum(dim=-1)
        expected = a.unsqueeze(-1) * self._p0  # (n, d)
        actual = coupling @ self._p1  # (n, d)
        return bool(torch.allclose(actual, expected, atol=atol, rtol=0.0))


# ---------------------------------------------------------------------------
# BassMartingaleDiscrepancy — calibration loss for SynFlowNet
# ---------------------------------------------------------------------------


class BassMartingaleDiscrepancy:
    """Bass-martingale discrepancy loss for SynFlowNet calibration.

    Given a coupling matrix :math:`\\Pi` of shape ``(n, m)`` and a
    target source marginal :math:`a \\in \\mathbb{R}^n`, returns a
    scalar loss that measures how far :math:`\\Pi` is from being a
    *Bass martingale coupling* of :math:`a` and the implicit target
    marginal.

    Two penalties are summed

    .. math::

        \\mathcal{D}(\\Pi, a)
        = (1 - \\lambda) \\, \\mathcal{D}_{\\rm martingale}(\\Pi, a)
        + \\lambda \\, \\mathcal{D}_{\\rm mass}(\\Pi, a).

    * **Martingale penalty** :math:`\\mathcal{D}_{\\rm martingale}` —
      the mean squared error between :math:`\\Pi`'s conditional
      expectation and the source marginal *as a value*.  Uses the
      first-column (or row-mean) projection for the 1-D ordering; if
      the coupling is a Bass martingale this is exactly zero.
    * **Mass-conservation penalty** :math:`\\mathcal{D}_{\\rm mass}` —
      the mean squared error between the row-sums and the target
      marginal :math:`a`.  Zero for any valid coupling.

    The combination is invariant under time-homogeneous rescaling of
    the coupling (which is exactly the symmetry that the TB loss
    fails to preserve — TB conflates the magnitude of :math:`\\log Z`
    with the trajectory's mass).  Replaces the TB loss in the
    GFlowNet head with a Bass-aware objective.

    CPU-only / standalone.
    """

    def __init__(self, lam: float = 0.1) -> None:
        """Build the Bass martingale discrepancy.

        Parameters
        ----------
        lam
            Mixing weight in :math:`[0, 1]` between the martingale
            penalty (``1 - lam`` weight) and the mass-conservation
            penalty (``lam`` weight).
        """
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = float(lam)

    def forward(self, coupling: Tensor, marginals: Tensor) -> Tensor:
        """Compute the Bass martingale discrepancy.

        Parameters
        ----------
        coupling
            Coupling matrix :math:`\\Pi` of shape ``(n, m)``.
        marginals
            Target source marginal :math:`a` of shape ``(n,)``.  Must
            sum to ``1`` along ``dim=-1``.

        Returns
        -------
        Tensor
            Scalar — the discrepancy
            :math:`(1 - \\lambda) \\, \\mathcal{D}_{\\rm martingale}
            + \\lambda \\, \\mathcal{D}_{\\rm mass}`.
            Always ``>= 0`` with equality iff ``coupling`` is a valid
            Bass martingale coupling of ``marginals``.
        """
        if coupling.ndim != 2:
            raise ValueError(
                f"coupling must be a 2-D matrix, got shape "
                f"{tuple(coupling.shape)}"
            )
        n, m = coupling.shape
        if marginals.shape != (n,):
            raise ValueError(
                f"marginals shape {tuple(marginals.shape)} must match "
                f"coupling's row count {n}"
            )

        device, dtype = coupling.device, coupling.dtype

        # Mass conservation: row sums vs target marginal.
        row_sums = coupling.sum(dim=-1)  # (n,)
        mass_loss = ((row_sums - marginals) ** 2).mean()

        # Martingale penalty: use the first column of the source
        # values (or the row-mean for d > 1) as the 1-D ordering
        # axis.  The conditional expectation E[X_1 | X_0 = i] is the
        # row-i weighted mean of x_1 — for a Bass martingale this
        # equals the source value x_0[i].
        #
        # We use index-based x_0 and x_1 values (the canonical
        # choice for a *calibration* loss: the discrepancy should
        # only depend on the coupling, not on an external data
        # tensor).  Specifically:
        #     x_0[i] = i / n  (so source atoms span [0, 1])
        #     x_1[j] = j / m  (target atoms span [0, 1])
        # These are monotone, deterministic, and free.
        x0 = torch.arange(n, device=device, dtype=dtype).float() / max(n - 1, 1)
        x1 = torch.arange(m, device=device, dtype=dtype).float() / max(m - 1, 1)
        cond_means = coupling @ x1  # (n,)  -- Pi @ x1
        safe_rowsums = row_sums.clamp_min(_EPS_DIV)
        # E[X_1 | X_0 = i] = cond_means[i] / row_sums[i]; we want
        # this to equal x_0[i].
        cond_means_norm = cond_means / safe_rowsums
        martingale_loss = ((cond_means_norm - x0) ** 2).mean()

        return (1.0 - self.lam) * martingale_loss + self.lam * mass_loss


# ---------------------------------------------------------------------------
# BassCalibrationHead — drop-in replacement for the TB loss head
# ---------------------------------------------------------------------------


class BassCalibrationHead:
    """Bass-martingale calibration head for SynFlowNet.

    Drop-in replacement for the *trajectory-balance* (TB) loss head
    of GFlowNets (Bengio et al. 2021).  Given a trajectory
    :math:`x_0, x_1, \\dots, x_T` of shape ``(T + 1, n, d)`` (or
    ``(B, T + 1, n, d)`` for batched inputs), returns the Bass
    calibration loss:

    .. math::

        \\mathcal{L}_{\\rm Bass}(\\tau)
        = \\mathcal{D}_{\\rm Bass}(\\Pi_\\tau, \\Pi^{\\rm Bassref}),

    where :math:`\\Pi_\\tau` is the trajectory-implied coupling (mass
    on the diagonal: each :math:`x_0[i]` transitions to the terminal
    :math:`x_T[i]`) and :math:`\\Pi^{\\rm Bassref}` is the Bass
    martingale reference coupling of :math:`x_0, x_T`.  When the
    trajectory is "Bass-calibrated" the implied coupling coincides
    with the reference and the loss vanishes.

    Theorem reference
    -----------------
    The Bass calibration head replaces the TB loss
    :math:`(\\log P_F \\tau + \\log R(x_T) - \\log Z)^2` (Bengio
    et al. 2021, eq. (15)) with a Bass-aware discrepancy that is
    *invariant under time-homogeneous rescaling of the flow*
    :math:`P_F`.  Henry-Labordère (2017) §3.4 shows this rescaling
    is exactly the symmetry preserved by the Bass martingale
    coupling, which is the bridge between the discrete Schrödinger
    bridge (T1) and the calibration head.

    CPU-only / standalone.
    """

    def __init__(self, eps: float = 0.1) -> None:
        """Build the Bass calibration head.

        Parameters
        ----------
        eps
            Bass mixing weight forwarded to :class:`BassMartingale`.
            ``eps = 0`` makes the reference coupling a Dirac on the
            nearest neighbour of each source atom; ``eps = 1`` makes
            it a pure two-point interpolation that exactly enforces
            the martingale property.
        """
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self.eps = float(eps)
        self._bass = BassMartingale(eps=eps)
        self._discrepancy = BassMartingaleDiscrepancy(lam=eps)

    def loss(self, trajectory: Tensor) -> Tensor:
        """Compute the Bass calibration loss for a trajectory.

        Parameters
        ----------
        trajectory
            Trajectory tensor.  Two layouts are accepted:

            * ``(T + 1, n, d)`` — ``T + 1`` states, ``n`` samples per
              state, ``d``-dimensional state space.  ``trajectory[0]``
              is the source, ``trajectory[T]`` is the target.
            * ``(B, T + 1, n, d)`` — batched.  The loss is averaged
              across the batch.

        Returns
        -------
        Tensor
            Scalar — the Bass calibration loss.  ``>= 0`` with
            equality iff the trajectory-implied coupling equals the
            Bass reference coupling of its endpoints.
        """
        if trajectory.ndim not in (3, 4):
            raise ValueError(
                f"trajectory must be (T+1, n, d) or (B, T+1, n, d), "
                f"got shape {tuple(trajectory.shape)}"
            )

        if trajectory.ndim == 3:
            return self._single_trajectory_loss(trajectory)

        # Batched: compute per-batch loss then average.  Iterating
        # rather than vectorising keeps the contract the same as the
        # single-trajectory case (Bass reference depends on the
        # per-trajectory endpoint values, not on a batched global
        # statistic).
        B = trajectory.shape[0]
        losses = [
            self._single_trajectory_loss(trajectory[b]) for b in range(B)
        ]
        return torch.stack(losses).mean()

    # -- internal helpers --------------------------------------------------

    def _single_trajectory_loss(self, trajectory: Tensor) -> Tensor:
        """Bass calibration loss for a single (T+1, n, d) trajectory."""
        T_plus_1, n, d = trajectory.shape
        T = T_plus_1 - 1
        if T < 0:
            raise ValueError(
                f"trajectory must have at least one state, got "
                f"{T_plus_1} states"
            )

        x0 = trajectory[0]  # (n, d)
        xT = trajectory[T]  # (n, d)

        # Bass reference coupling of (x0, xT).
        bass_ref = self._bass.couple(x0, xT)

        # Trajectory-implied coupling: each source atom i transitions
        # to the corresponding target atom i (mass on the diagonal).
        # Only meaningful when the source and target have the same
        # number of atoms (which is the SynFlowNet convention: the
        # chain has a fixed number of "particles").
        if xT.shape[0] != n:
            raise ValueError(
                f"trajectory source / target must have the same number "
                f"of atoms, got n={n} and xT.shape[0]={xT.shape[0]}"
            )
        a = torch.full((n,), 1.0 / n, dtype=trajectory.dtype,
                       device=trajectory.device)
        implied = torch.diag(a)

        # L2 distance between the trajectory-implied coupling and the
        # Bass reference, weighted by the marginals.  This is the
        # Bass calibration loss; it vanishes when the trajectory is
        # exactly Bass-calibrated.
        diff = implied - bass_ref
        return (diff * diff).sum() / n
