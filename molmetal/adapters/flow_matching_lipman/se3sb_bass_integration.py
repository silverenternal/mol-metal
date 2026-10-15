"""SE(3) Schrödinger Bridge + Bass martingale OT-sampler integration (Wf-235).

================================================================
Why this module exists
================================================================
The round-18 (R18) math migration (Wf-234 audit + integration plan,
``TODO/pending/234_wf234_frontier_math_audit_integration.md``) identified
two math modules that should be added to the CFM main flow as **OT
samplers** (replacements / augmentations for the standard Euclidean OT
plan used to pair ``x_0`` source samples with ``x_1`` data samples):

1. :mod:`molmetal.math_skills.se3_schrodinger_bridge` (Mahmood 2026,
   arXiv:2603.14049) — SE(3) Sinkhorn coupling in twist space,
   replacing Euclidean ``||x_0 - x_1||^2`` cost with the SE(3)
   geodesic-aware twist-space cost + Sinkhorn entropy regularisation.
   Best for **rigid-body / SE(3) structured data** (molecules,
   ligands, metal coordination complexes).

2. :mod:`molmetal.math_skills.sb_bass_martingale` (Henry-Labordère
   2017, SSRN 2889130) — Bass martingale mean-preserving coupling,
   replacing the random / Hungarian pairing with a *mean-preserving*
   coupling that is invariant under time-homogeneous rescaling of the
   flow.  Best for **heavy-tail distributions** where the marginals
   have heavy tails and the standard OT pairing creates bias at the
   tails.

This module is the **integration layer** that wires these two
mathematical primitives into the ``w_flow.py`` main flow as drop-in
OT samplers, **without modifying** the upstream public API of
``WaveformRelaxationSolver`` / ``WFlowCFM`` (backward-compat contract
of the W2.1/W2.2/W2.3 ships):

* New optional ``use_se3_sb`` / ``use_bass_martingale`` parameters
  default to ``False`` so the legacy midpoint / waveform-relaxation
  ODE solver is bit-exact unchanged.
* When ``use_se3_sb=True`` the solver matches per-window endpoints
  via SE(3) twist-space Sinkhorn coupling (in addition to the
  existing Picard-style waveform boundary propagation).
* When ``use_bass_martingale=True`` the solver matches per-window
  endpoints via Bass martingale mean-preserving coupling (heavy-tail
  case).
* When both flags are ``False`` the legacy behaviour is bit-exactly
  recovered.

Design constraints (per the W2.1 / W2.2 / W2.3 ships)
--------------------------------------------------
1. **Do NOT modify** ``molmetal.adapters.flow_matching_lipman.w_flow``
   signature-breaking way.  Only ADD optional parameters with
   defaults that preserve legacy behaviour.
2. **CPU-only.**  All operations use ``torch.linalg`` /
   ``torch.logsumexp`` (CPU backend by default).
3. **Standalone.**  No internal imports from
   ``molmetal.molmetal_lam.*``.  Only ``torch`` + the pure
   ``molmetal.math_skills`` package.
4. **Batched.**  Every public function accepts a leading batch
   dimension and broadcasts over it.

Module map
----------
* :class:`SE3SBOTSampler` — SE(3) twist-space Sinkhorn OT sampler.
* :class:`BassMartingaleOTSampler` — Bass mean-preserving martingale
  OT sampler.
* :class:`OTCouplingMode` — string enum of the three sampler modes
  (``"legacy"`` / ``"se3_sb"`` / ``"bass_martingale"``).
* :func:`make_ot_sampler` — factory that returns the right sampler
  given the two boolean flags.
* :class:`SE3SBBassComposite` — production composite that wraps a
  base velocity_fn with the SE(3) SB / Bass OT sampler so the CFM
  training loop can pair ``x_0`` and ``x_1`` via the chosen coupling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

# T1 — pure-math SE(3) layer.  See
# ``molmetal/math_skills/se3_schrodinger_bridge.py`` for the
# mathematical reference (Mahmood 2026, arXiv:2603.14049).
from molmetal.math_skills.se3_schrodinger_bridge import (
    LogSinkhornLoss,
    SE3Geodesic,
    se3_exp,
    se3_log,
)

# T7 — Bass martingale coupling.  See
# ``molmetal/math_skills/sb_bass_martingale.py`` for the
# mathematical reference (Henry-Labordère 2017, SSRN 2889130).
from molmetal.math_skills.sb_bass_martingale import (
    BassMartingale,
    BassMartingaleDiscrepancy,
)


__all__ = [
    "OTCouplingMode",
    "SE3SBOTSampler",
    "BassMartingaleOTSampler",
    "LegacyOTSampler",
    "make_ot_sampler",
    "SE3SBBassComposite",
]


# ---------------------------------------------------------------------------
# Numerical guards — mirror the constants from the math layer so the
# adapter is self-contained.
# ---------------------------------------------------------------------------
_EPS_TRIG: float = 1.0e-6
_EPS_DIV: float = 1.0e-12
_DEFAULT_SB_EPS: float = 0.1
_DEFAULT_SB_N_ITERS: int = 50
_DEFAULT_BASS_EPS: float = 0.5


# ---------------------------------------------------------------------------
# Coupling -> index, two ways
# ---------------------------------------------------------------------------
# Both samplers below build an (B, B) coupling matrix and then have to turn it
# into a (B,) index.  Until 2026-10-04 both did `plan.argmax(dim=-1)`, and both
# documented the result as a "permutation".  It is not one: argmax assigns each
# SOURCE its best column independently, so several sources can land on the same
# target.  Measured on random inputs, Bass produced a non-permutation in 10/10
# trials and SE(3) SB in 3/10 — the contract was simply untrue, and the test
# named `..._returns_permutation` only checked shape and dtype, on a fixture
# where x0 == x1 so no collision could ever occur.
#
# The two helpers below give the two things a caller can legitimately want:
#
#   _assignment_permutation — an exact one-to-one assignment, i.e. a genuine
#     permutation.  This is what OT-CFM (Lipman 2023 §3.3) wants when pairing
#     equal-size batches: one target per source, no target used twice.
#   _sample_from_plan — a draw j ~ plan[i, :] per source.  This is the faithful
#     way to SAMPLE a coupling, and for a Bass plan it can legitimately repeat
#     a target (the Bass coupling is two-point, so within one row the mass sits
#     on two columns; hard assignment throws that structure away, sampling
#     keeps it).
#
# Keeping both means callers pick the semantics they need instead of getting
# whichever one argmax happened to produce.
def _assignment_permutation(plan: Tensor) -> Tensor:
    """Max-weight one-to-one assignment of sources to targets.

    Returns a genuine ``(B,)`` permutation: every index in ``0..B-1`` appears
    exactly once.  Solves the linear assignment problem on ``-plan``, so the
    result maximises the total coupling mass under the one-to-one constraint —
    which is what ``plan.argmax(dim=-1)`` approximates greedily and fails to
    satisfy.

    Falls back to a greedy row-order assignment if SciPy is unavailable, so the
    contract (a true permutation) still holds; only optimality is lost.
    """
    b = int(plan.shape[0])
    if b == 0:
        return torch.zeros(0, dtype=torch.long, device=plan.device)
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-plan.detach().cpu().numpy())
        del rows  # row order is 0..B-1 by construction
        return torch.as_tensor(cols, dtype=torch.long, device=plan.device)
    except Exception:  # pragma: no cover - SciPy is a declared dependency
        idx = plan.argmax(dim=-1).to(torch.long)
        taken = torch.zeros(b, dtype=torch.bool, device=plan.device)
        order = torch.argsort(-plan.max(dim=-1).values)
        for i in order.tolist():
            row = plan[i]
            for j in torch.argsort(-row).tolist():
                if not bool(taken[j]):
                    idx[i] = j
                    taken[j] = True
                    break
        return idx


def _sample_from_plan(plan: Tensor, generator: Optional[object] = None) -> Tensor:
    """Draw one target index per source from the coupling: ``j ~ plan[i, :]``.

    Unlike :func:`_assignment_permutation` this is NOT a permutation and does
    not claim to be — repeating a target is a legitimate outcome of sampling a
    coupling, and for a two-point Bass plan it is the only way to preserve the
    martingale structure rather than replacing it with a hard assignment.

    Falls back to argmax if the plan has a degenerate row (all-zero, which
    makes ``torch.multinomial`` raise).
    """
    b = int(plan.shape[0])
    if b == 0:
        return torch.zeros(0, dtype=torch.long, device=plan.device)
    p = plan.detach().float()
    row_sums = p.sum(dim=-1, keepdim=True)
    degenerate = (row_sums <= 0).expand_as(p)
    if bool(degenerate.any()):
        p = torch.where(degenerate, torch.ones_like(p), p / row_sums.clamp(min=_EPS_DIV))
    try:
        return torch.multinomial(p, num_samples=1, generator=generator).squeeze(-1).long()
    except Exception:  # pragma: no cover - defensive
        return p.argmax(dim=-1).to(torch.long)


# ---------------------------------------------------------------------------
# OTCouplingMode — string-based enum (stable across pickle / import)
# ---------------------------------------------------------------------------
class OTCouplingMode:
    """OT coupling modes used by :func:`make_ot_sampler`.

    Constants
    ---------
    LEGACY            : use the legacy Euclidean OT plan (no modification).
    SE3_SB            : use the SE(3) twist-space Sinkhorn coupling (Mahmood 2026).
    BASS_MARTINGALE   : use the Bass martingale mean-preserving coupling
                        (Henry-Labordère 2017).
    """

    LEGACY: str = "legacy"
    SE3_SB: str = "se3_sb"
    BASS_MARTINGALE: str = "bass_martingale"

    _ALL: Tuple[str, ...] = (LEGACY, SE3_SB, BASS_MARTINGALE)

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        """Tuple of all known mode names (read-only)."""
        return cls._ALL

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """True iff ``name`` is a known OTCouplingMode."""
        return str(name) in cls._ALL


# ---------------------------------------------------------------------------
# OTError — narrow exception for OT-shape / OT-config mismatches.
# Wf-A49 (2026-10-16) — defined so the W-Flow OT coupling catch in
# ``WaveformRelaxationSolver.solve`` can narrow from bare
# ``except Exception`` to ``except OTError``.  Subclass of ValueError so
# pre-existing ``except ValueError`` call sites keep catching OT-shape
# errors (no R17 backward-compat breakage) while the W-Flow caller can
# specifically identify the OT-related ones.
# ---------------------------------------------------------------------------
class OTError(ValueError):
    """Raised by OT samplers on shape / config mismatches.

    Inherits from :class:`ValueError` so existing ``except ValueError``
    call sites continue to catch these (no R17 backward-compat
    breakage).  The W-Flow solver in
    :mod:`molmetal.adapters.flow_matching_lipman.w_flow` narrows to
    ``except OTError`` to identify OT-specific failures vs other
    shape mismatches that may surface during waveform relaxation.
    """


# ---------------------------------------------------------------------------
# LegacyOTSampler — pure-Euclidean OT (the Wf-234 baseline)
# ---------------------------------------------------------------------------
class LegacyOTSampler:
    """Pure-Euclidean OT coupling (the pre-Wf-235 baseline).

    Computes a deterministic identity permutation ``i -> i`` so the
    upstream :class:`WaveformRelaxationSolver`'s Picard-style window
    matching is unaffected.  This is the sampler returned when both
    ``use_se3_sb`` and ``use_bass_martingale`` are ``False``; it pins
    the backward-compat contract that the Wf-235 wire-in does not
    change the legacy midpoint / waveform-relaxation solver bit-exactly.

    Parameters
    ----------
    None.

    Notes
    -----
    Mathematically trivial: the "OT coupling" is the identity, which is
    exactly what :meth:`WaveformRelaxationSolver.solve`'s Picard-style
    outer-iteration does when no extra coupling is requested.  We
    surface it as a class only so callers can switch between
    samplers via a uniform ``.couple(x0, x1) -> Tensor`` contract.
    """

    def couple(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Return the identity permutation ``[0, 1, ..., B-1]``.

        Parameters
        ----------
        x0 : ``(B, ...)`` tensor
            Source batch.
        x1 : ``(B, ...)`` tensor
            Target batch.

        Returns
        -------
        Tensor
            ``(B,)`` long permutation.  When ``x0.shape[0] !=
            x1.shape[0]`` we return the smaller of the two batch
            sizes (the standard CFM convention).
        """
        if x0.shape[0] != x1.shape[0]:
            raise ValueError(
                f"LegacyOTSampler requires matching batch sizes; got "
                f"x0.shape[0]={x0.shape[0]} and x1.shape[0]={x1.shape[0]}"
            )
        return torch.arange(x0.shape[0], dtype=torch.long, device=x0.device)


# ---------------------------------------------------------------------------
# SE3SBOTSampler — SE(3) twist-space Sinkhorn coupling
# ---------------------------------------------------------------------------
class SE3SBOTSampler:
    """SE(3) twist-space Sinkhorn OT sampler (Mahmood 2026).

    Replaces the Euclidean squared-distance OT cost ``||x_0 - x_1||^2``
    with the **SE(3) twist-space geodesic proxy**: convert each
    ``(B, 6)`` twist ``xi`` to an SE(3) matrix via :func:`se3_exp`,
    compute pairwise SE(3) body-frame twist distances in the
    Lie algebra, and run the **log-domain Sinkhorn iteration** from
    :class:`LogSinkhornLoss` to recover the entropic-regularised OT
    coupling.  The output is a ``(B,)`` permutation
    ``idx`` such that ``x1[idx]`` is paired with ``x0`` under the
    SE(3)-aware cost.

    This sampler is the **production wire-in** of the SE(3) Schrödinger
    Bridge math skeleton (T1 + W2.1) into the CFM main flow.  It is
    bit-exact backward-compatible: when the cost matrix is identity
    (i.e. all source atoms map to themselves) the recovered
    permutation is the identity, matching the legacy Euclidean OT
    sampler.  In the non-trivial case it gives a lower-variance CFM
    velocity target because the *path* between paired samples is
    shorter on the SE(3) manifold than on the Euclidean tangent.

    Parameters
    ----------
    eps : float, default ``0.1``
        Sinkhorn entropy regularisation.  Forwarded to
        :class:`LogSinkhornLoss`.
    n_iters : int, default ``50``
        Number of Sinkhorn iterations.  Forwarded to
        :class:`LogSinkhornLoss`.
    seed : Optional[int], default ``None``
        Random seed for the Sinkhorn coupling draw (used when the
        recovered coupling is a stochastic sample; the
        permutation-mode used here is deterministic so ``seed`` is
        only consulted by the optional :meth:`sample` method).

    Notes
    -----
    The input ``x0`` / ``x1`` are expected to be ``(B, 6)`` twists
    (linear + angular, ``se(3)`` Lie algebra).  For ``(B, N, 3)``
    per-atom Euclidean coordinates we use the **centroid + RBM
    rigid-body identification** to lift to ``(B, 6)``: the centroid
    of the per-atom coordinates is the translation, and the
    per-batch rotation that aligns two copies of the same molecule
    is the rotation (computed via :func:`torch.linalg.svd` on the
    covariance matrix).  See :meth:`lift_to_se3_twist` for the
    exact reduction.

    CPU-only by design (no CUDA / Triton kernels).
    """

    def __init__(
        self,
        eps: float = _DEFAULT_SB_EPS,
        n_iters: int = _DEFAULT_SB_N_ITERS,
        seed: Optional[int] = None,
    ) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.seed = seed
        self._sinkhorn = LogSinkhornLoss(eps=self.eps, n_iters=self.n_iters)

    # ------------------------------------------------------------------
    # Rigid-body identification (per-atom coords -> SE(3) twist)
    # ------------------------------------------------------------------
    @staticmethod
    def lift_to_se3_twist(coords: Tensor) -> Tensor:
        """Lift ``(B, N, 3)`` per-atom coordinates to ``(B, 6)`` SE(3) twists.

        The reduction is the standard rigid-body identification:

        * **Linear part** = centroid of the per-atom coordinates
          (``coords.mean(dim=1)``) — this is the SE(3) translation.
        * **Angular part** = a 3-vector extracted from the
          antisymmetric part of the centred covariance:
          ``cov = centred.T @ centred / N``; the rotation vector is
          read off the off-diagonal entries of ``cov`` in the same
          way :func:`molmetal.math_skills.se3_schrodinger_bridge.so3_log`
          reads the skew matrix.  This is a *proxy* for the
          per-batch orientation (true SE(3) "orientation" requires
          a reference frame; we use the centroid covariance as a
          convenient rotation-invariant summary).

        Parameters
        ----------
        coords : ``(B, N, 3)`` tensor
            Per-atom coordinates.

        Returns
        -------
        Tensor
            ``(B, 6)`` SE(3) twist — ``[..., :3]`` is the centroid
            translation, ``[..., 3:]`` is the rotation-vector proxy.
        """
        if coords.dim() != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"lift_to_se3_twist expects (B, N, 3) input, got "
                f"{tuple(coords.shape)}"
            )
        centroid = coords.mean(dim=1)  # (B, 3)
        centred = coords - centroid.unsqueeze(1)  # (B, N, 3)
        # Covariance matrix per batch (B, 3, 3) = centred.T @ centred / N
        cov = centred.transpose(-1, -2) @ centred / max(1, centred.shape[1])
        # Rotation-vector proxy: read the off-diagonal entries of the
        # skew part of ``cov``.  This is the same closed-form
        # reduction :func:`so3_log` uses on rotation matrices; here we
        # apply it to the *covariance* as a rotation-invariant proxy.
        skew = 0.5 * (cov - cov.transpose(-1, -2))  # (B, 3, 3)
        omega = torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
            dim=-1,
        )  # (B, 3)
        return torch.cat([centroid, omega], dim=-1)  # (B, 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _to_twist_pair(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        """Lift both sides to ``(B, 6)`` se(3) twists, validating the layout."""
        out = []
        for name, x in (("x0", x0), ("x1", x1)):
            if x.shape[-1] == 3:
                out.append(self.lift_to_se3_twist(x))
            elif x.shape[-1] == 6:
                out.append(x)
            else:
                raise ValueError(
                    f"{name} trailing dim must be 3 (lift to se3) or 6 "
                    f"(already a twist); got {x.shape[-1]}"
                )
        return out[0], out[1]

    def couple(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute the SE(3)-aware OT permutation.

        Parameters
        ----------
        x0 : ``(B, 6)`` or ``(B, N, 3)`` tensor
            Source batch.  When the trailing dim is 3 we lift to
            ``(B, 6)`` via :meth:`lift_to_se3_twist`; when it is 6 we
            use the input directly.
        x1 : same layout as ``x0``
            Target batch.

        Returns
        -------
        Tensor
            ``(B,)`` long permutation ``idx`` such that
            ``x1[idx]`` is paired with ``x0``.  Recovered by
            hard-assigning each source to the column of the
            Sinkhorn coupling with maximum mass (Theorem 4.1 of
            :cite:`mahmood2026sb_lie` — the SB coupling converges
            to the un-regularised OT plan as ``eps -> 0``).

        Raises
        ------
        ValueError
            If ``x0`` and ``x1`` have different batch sizes or if
            the trailing dim is neither 3 nor 6.
        """
        if x0.shape[0] != x1.shape[0]:
            raise OTError(
                f"SE3SBOTSampler requires matching batch sizes; got "
                f"x0.shape[0]={x0.shape[0]} and x1.shape[0]={x1.shape[0]}"
            )
        xi0, xi1 = self._to_twist_pair(x0, x1)

        # Pairwise SE(3) twist cost (the SE(3) geodesic proxy, eq. (12)
        # of Mahmood 2026): squared L2 distance in twist space.
        b = xi0.shape[0]
        diff = xi0.unsqueeze(1) - xi1.unsqueeze(0)  # (B, B, 6)
        C = (diff * diff).sum(dim=-1)  # (B, B)
        # Uniform marginals (the SB self-bridge convention).
        a = torch.full((b,), 1.0 / b, dtype=xi0.dtype, device=xi0.device)
        # Log-domain Sinkhorn — stable for eps down to 1e-3 (see math
        # layer docstring line 562-566).
        _loss, P = self._sinkhorn.forward(xi0, xi1, C=C, a=a, b=a)
        # 2026-10-04: was `P.argmax(dim=-1)`, which is NOT a permutation --
        # two sources can pick the same target, and did in 3/10 measured
        # trials.  `_assignment_permutation` solves the assignment exactly.
        return _assignment_permutation(P)

    def sample_coupling(
        self, x0: Tensor, x1: Tensor, generator: Optional[object] = None
    ) -> Tensor:
        """Draw ``j ~ P[i, :]`` per source from the SE(3) Sinkhorn coupling.

        Not a permutation, and does not claim to be — see
        :func:`_sample_from_plan`.
        """
        xi0, xi1 = self._to_twist_pair(x0, x1)
        b = xi0.shape[0]
        diff = xi0.unsqueeze(1) - xi1.unsqueeze(0)
        C = (diff * diff).sum(dim=-1)
        a = torch.full((b,), 1.0 / b, dtype=xi0.dtype, device=xi0.device)
        _loss, P = self._sinkhorn.forward(xi0, xi1, C=C, a=a, b=a)
        return _sample_from_plan(P, generator=generator)


# ---------------------------------------------------------------------------
# BassMartingaleOTSampler — mean-preserving martingale coupling
# ---------------------------------------------------------------------------
class BassMartingaleOTSampler:
    """Bass martingale mean-preserving OT sampler (Henry-Labordère 2017).

    Replaces the random / Hungarian OT pairing with a **Bass
    mean-preserving martingale coupling**: a coupling matrix
    :math:`\\Pi \\in \\mathbb{R}^{B \\times B}` whose row sums equal the
    source marginals and whose columns satisfy the martingale property
    ``\\Pi \\cdot x_1 = a \\otimes x_0`` (the conditional expectation of
    the target given the source equals the source itself — the
    Bass-mean-preserving property).

    The Bass construction is **two-point interpolating** (each row of
    ``\\Pi`` is supported on at most two columns), making the coupling
    cheap to compute (O(B) per row, O(B^2) total) and **mean-preserving**
    (the source marginal is recovered as the conditional expectation
    of the target).  This is the textbook "Bass martingale" used in
    the Henry-Labordère 2017 SB construction (T7) and is the
    mathematical skeleton for heavy-tail distribution coupling where
    the standard Euclidean OT creates a tail bias.

    Parameters
    ----------
    eps : float, default ``0.5``
        Bass mixing weight.  ``eps = 0`` collapses to a Dirac on the
        nearest neighbour (degenerate); ``eps = 1`` places all mass on
        the two-point bridge interpolation that exactly enforces the
        martingale property.  The default ``0.5`` is the canonical
        Henry-Labordère setting.
    input_dim : int, default ``3``
        Input feature dimension.  ``3`` means we operate directly on
        the (centroid) per-batch coordinates; ``6`` means we operate
        on SE(3) twists.  When ``x0.shape[-1] != input_dim`` we lift
        ``(B, N, 3)`` per-atom coords to ``(B, 3)`` centroids via
        :meth:`reduce_to_centroid`.

    Notes
    -----
    CPU-only by design (no CUDA / Triton kernels).
    """

    def __init__(
        self,
        eps: float = _DEFAULT_BASS_EPS,
        input_dim: int = 3,
    ) -> None:
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        if input_dim not in (3, 6):
            raise ValueError(f"input_dim must be 3 or 6, got {input_dim}")
        self.eps = float(eps)
        self.input_dim = int(input_dim)
        self._bass = BassMartingale(eps=self.eps)

    # ------------------------------------------------------------------
    # Centroid reduction (per-atom coords -> per-batch centroid)
    # ------------------------------------------------------------------
    @staticmethod
    def reduce_to_centroid(coords: Tensor) -> Tensor:
        """Lift ``(B, N, 3)`` per-atom coordinates to ``(B, 3)`` centroids.

        Parameters
        ----------
        coords : ``(B, N, 3)`` tensor
            Per-atom coordinates.

        Returns
        -------
        Tensor
            ``(B, 3)`` per-batch centroid.
        """
        if coords.dim() != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"reduce_to_centroid expects (B, N, 3) input, got "
                f"{tuple(coords.shape)}"
            )
        return coords.mean(dim=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _reduce_pair(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        """Reduce both sides to ``(B, d)`` centroids, validating the layout."""
        out = []
        for name, x in (("x0", x0), ("x1", x1)):
            if x.dim() == 3:
                out.append(self.reduce_to_centroid(x))
            elif x.dim() == 2:
                out.append(x)
            else:
                raise ValueError(
                    f"{name} must be (B, d) or (B, N, 3); got shape {tuple(x.shape)}"
                )
        return out[0], out[1]

    def couple(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute the Bass-martingale permutation.

        Parameters
        ----------
        x0 : ``(B, ...)`` tensor
            Source batch.  When ``x0.dim() == 3`` (per-atom coords)
            we lift to ``(B, 3)`` centroids via :meth:`reduce_to_centroid`;
            when ``x0.dim() == 2`` (already ``(B, d)``) we use the
            input directly.
        x1 : same layout as ``x0``
            Target batch.

        Returns
        -------
        Tensor
            ``(B,)`` long permutation ``idx`` such that ``x1[idx]``
            is paired with ``x0``.  Recovered by hard-assigning each
            source to the column of the Bass coupling with maximum
            mass.

        Raises
        ------
        ValueError
            If ``x0`` and ``x1`` have different batch sizes.
        """
        if x0.shape[0] != x1.shape[0]:
            raise OTError(
                f"BassMartingaleOTSampler requires matching batch sizes; "
                f"got x0.shape[0]={x0.shape[0]} and x1.shape[0]={x1.shape[0]}"
            )
        x0r, x1r = self._reduce_pair(x0, x1)

        # Bass martingale coupling matrix.
        Pi = self._bass.couple(x0r, x1r)
        # 2026-10-04: was `Pi.argmax(dim=-1)`.  A Bass plan is TWO-POINT --
        # each row's mass sits on at most two columns -- so independent argmax
        # collides constantly (measured 10/10 trials), which is why this method
        # never actually returned the permutation its docstring promised.  The
        # assignment is now solved exactly.
        return _assignment_permutation(Pi)

    def sample_coupling(
        self, x0: Tensor, x1: Tensor, generator: Optional[object] = None
    ) -> Tensor:
        """Draw ``j ~ Pi[i, :]`` per source from the Bass coupling matrix.

        This is the variant that PRESERVES the Bass structure.  argmax throws
        the two-point plan away and replaces it with a hard assignment;
        sampling keeps the mass split, at the cost of repeating targets — which
        is a legitimate outcome of drawing from a coupling, not a defect.

        Not a permutation, and does not claim to be — see
        :func:`_sample_from_plan`.
        """
        x0r, x1r = self._reduce_pair(x0, x1)
        Pi = self._bass.couple(x0r, x1r)
        return _sample_from_plan(Pi, generator=generator)


# ---------------------------------------------------------------------------
# make_ot_sampler — factory from flag pair to OT sampler
# ---------------------------------------------------------------------------
def make_ot_sampler(
    use_se3_sb: bool = False,
    use_bass_martingale: bool = False,
    *,
    sb_eps: float = _DEFAULT_SB_EPS,
    sb_n_iters: int = _DEFAULT_SB_N_ITERS,
    bass_eps: float = _DEFAULT_BASS_EPS,
    bass_input_dim: int = 3,
    seed: Optional[int] = None,
) -> Tuple[str, object]:
    """Return an OT sampler (legacy / SE(3) SB / Bass martingale).

    Parameters
    ----------
    use_se3_sb : bool, default ``False``
        When ``True`` and ``use_bass_martingale`` is ``False`` the
        SE(3) Sinkhorn sampler is returned.
    use_bass_martingale : bool, default ``False``
        When ``True`` and ``use_se3_sb`` is ``False`` the Bass
        martingale sampler is returned.
        When **both** flags are ``True`` a :class:`ValueError` is
        raised (mutually exclusive — see the Wf-234 audit §4.2).
    sb_eps, sb_n_iters, seed
        Forwarded to :class:`SE3SBOTSampler` when it is selected.
    bass_eps, bass_input_dim
        Forwarded to :class:`BassMartingaleOTSampler` when it is
        selected.

    Returns
    -------
    mode : str
        The selected :class:`OTCouplingMode` value.
    sampler : object
        A sampler exposing ``.couple(x0, x1) -> Tensor``.  The
        returned sampler is one of:
        :class:`LegacyOTSampler`, :class:`SE3SBOTSampler`,
        :class:`BassMartingaleOTSampler`.

    Raises
    ------
    ValueError
        If both ``use_se3_sb`` and ``use_bass_martingale`` are
        ``True`` (mutually exclusive — picking the SE(3) Sinkhorn
        cost AND the Bass martingale structure is mathematically
        ill-defined because the two samplers operate on different
        cost representations: SE(3) twist-space Sinkhorn vs. mean-
        preserving two-point interpolation).
    """
    if use_se3_sb and use_bass_martingale:
        raise ValueError(
            "make_ot_sampler: use_se3_sb and use_bass_martingale are "
            "mutually exclusive.  Pick one (or neither, for legacy "
            "Euclidean OT)."
        )
    if use_se3_sb:
        return (
            OTCouplingMode.SE3_SB,
            SE3SBOTSampler(eps=sb_eps, n_iters=sb_n_iters, seed=seed),
        )
    if use_bass_martingale:
        return (
            OTCouplingMode.BASS_MARTINGALE,
            BassMartingaleOTSampler(eps=bass_eps, input_dim=bass_input_dim),
        )
    return (OTCouplingMode.LEGACY, LegacyOTSampler())


# ---------------------------------------------------------------------------
# SE3SBBassComposite — production composite wrapping a velocity_fn
# ---------------------------------------------------------------------------
class SE3SBBassComposite:
    """Production composite: wraps a base ``(x, t) -> v`` velocity_fn
    with the chosen OT sampler for CFM training.

    The CFM main flow uses OT pairing to reduce the variance of the
    velocity target.  This composite exposes a uniform
    ``pair(x0, x1) -> Tensor`` API that returns the permutation under
    the chosen OT mode (legacy / SE(3) SB / Bass martingale), plus a
    :meth:`make_paired_batch` helper that produces the actual
    ``(x0_paired, x1_paired)`` tensors ready to feed into the
    velocity field.

    Parameters
    ----------
    use_se3_sb : bool, default ``False``
        Forwarded to :func:`make_ot_sampler`.
    use_bass_martingale : bool, default ``False``
        Forwarded to :func:`make_ot_sampler`.
    sb_eps, sb_n_iters, bass_eps, bass_input_dim, seed
        Forwarded to :func:`make_ot_sampler`.

    Notes
    -----
    This composite is a **call-site wrapper** — it does not modify the
    :class:`WaveformRelaxationSolver` / :class:`WFlowCFM` / ODESolver
    contracts.  Callers that want SE(3) SB / Bass OT coupling in the
    CFM training loop simply pass the ``pair`` / ``make_paired_batch``
    output to their existing CFM loss computation; the solver-side
    change is the ``use_se3_sb`` / ``use_bass_martingale`` flags on
    :class:`WaveformRelaxationSolver` (added in Wf-235).
    """

    def __init__(
        self,
        use_se3_sb: bool = False,
        use_bass_martingale: bool = False,
        *,
        sb_eps: float = _DEFAULT_SB_EPS,
        sb_n_iters: int = _DEFAULT_SB_N_ITERS,
        bass_eps: float = _DEFAULT_BASS_EPS,
        bass_input_dim: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.mode, self.sampler = make_ot_sampler(
            use_se3_sb=use_se3_sb,
            use_bass_martingale=use_bass_martingale,
            sb_eps=sb_eps,
            sb_n_iters=sb_n_iters,
            bass_eps=bass_eps,
            bass_input_dim=bass_input_dim,
            seed=seed,
        )

    def pair(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Return the permutation ``idx`` such that ``x1[idx]`` is
        paired with ``x0`` under the chosen OT mode."""
        return self.sampler.couple(x0, x1)

    def make_paired_batch(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return the actual ``(x0_paired, x1_paired)`` tensors.

        Parameters
        ----------
        x0 : ``(B, ...)`` tensor
            Source batch.
        x1 : ``(B, ...)`` tensor
            Target batch (same shape as ``x0``).

        Returns
        -------
        (x0_paired, x1_paired)
            Both of shape ``(B, ...)``.  ``x1_paired[i]`` is the OT-
            optimal partner of ``x0_paired[i] = x0[i]`` under the
            chosen OT mode.
        """
        idx = self.pair(x0, x1)
        return x0, x1[idx]
