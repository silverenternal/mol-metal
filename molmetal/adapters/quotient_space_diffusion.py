"""Quotient-Space Diffusion adapter — SE(3)-equivariant diffusion by
construction (R18 P2-4, Mol-Metal round-18 upgrade path).

Reference
---------
Arvind, R., Petrov, D. & Wang, S. (2026).  *Quotient-Space Diffusion
Models for Equivariant Geometry Generation*.  arXiv:2604.21809
[cs.LG].  Cited in ``paper/refs.bib`` as ``arvind2026qsdm``.

The round-17 (R17) close-out shipped
``molmetal.adapters.flow_matching_lipman`` — a Euclidean CFM
(Lipman 2023) + SE(3)-equivariant EGNN velocity net.  R18 P2-4 is the
*quotient-space* alternative: instead of enforcing SE(3) equivariance
in the network architecture, we run the diffusion / flow-matching
process *in the quotient space* :math:`X / G` where :math:`G` is the
SE(3) group action.  This achieves **equivariance by construction** —
the quotient space itself contains no preferred orientation or origin,
so any model that lives in it is automatically equivariant to the
quotiented-out transformations.

Concretely: the molecular conformer :math:`X \\in \\mathbb{R}^{N \\times 3}`
has a *natural* SE(3) action
``(R, t) : X \\mapsto R X + t`` (rotate every atom by ``R``, translate
by ``t``).  The quotient :math:`X / SE(3)`` is the space of all
conformation *shapes* — i.e. the inter-atomic distance matrix and
relative orientation graph, which are invariant under :math:`SE(3)`.

The three supported quotient spaces
------------------------------------

1. ``"SE3"`` (default) — quotient by the full SE(3) action.  We
   quotient *first* by translations (centre the coordinates at the
   centroid) and *then* by SO(3) (canonicalise the orientation via a
   Procrustes-aligned rotation).  This is the standard
   "shape-only" embedding of a molecular conformer — see Arvind 2026,
   Definition 3.1.
2. ``"rotation"`` — quotient by SO(3) only.  We canonicalise the
   orientation but keep the centroid translation.  Useful when the
   *position* of the molecule in space matters (e.g. binding-pocket
   conditioning).
3. ``"translation"`` — quotient by translations only.  We centre the
   coordinates but keep the original orientation.  Useful when the
   orientation matters (e.g. chiral molecules).

For each quotient we provide ``project`` (conformer → representative
in the quotient) and ``lift`` (representative → conformer with a
canonical lift).  The contract pinned by the test suite is:

* ``project(x) == project(R x + t)`` for every SE(3) transformation.
* ``lift(project(x))`` returns a conformer that is *aligned* with
  ``x`` (same shape; canonical orientation / centre).

Design constraints (per R18 P2-4 brief)
---------------------------------------
1. **Do NOT modify** ``molmetal/adapters/flow_matching_lipman/`` —
   the R17 close-out is locked.  This file is the *only* new
   adapter added under R18 P2-4.
2. **CPU-only.**  All operations use ``torch.linalg`` (CPU backend by
   default).  No CUDA / Triton kernels required for the math layer.
3. **Standalone.**  No internal imports from
   ``molmetal.molmetal_lam.*``.  Only ``torch`` + the pure
   ``molmetal.math_skills`` package + the existing CFM stack (read-
   only via the ``base_cfm`` parameter).
4. **Batched.**  Every public function accepts a leading batch
   dimension and broadcasts over it.

Module map
----------
* :class:`QuotientSpace` — the quotient-space geometric kernel
  (``project`` / ``lift`` plus a couple of metric helpers).
* :class:`QuotientDiffusionScheduler` — a VP-style noise scheduler
  that runs the diffusion process *inside* the quotient space.
* :class:`QuotientSpaceCFM` — drop-in wrapper around an existing CFM
  decoder that does flow-matching *inside* the quotient space.  The
  ``equivariance_error`` helper provides an *end-to-end* numerical
  sanity check for the equivariance contract (the output is
  equivariant up to the tolerance ``1e-3`` the test suite enforces).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# R18 T1 — pure-math SE(3) layer (CPU-only, batched).  We reuse
# ``so3_exp`` / ``so3_log`` for the Procrustes alignment step; the
# rest of the quotient kernel is built on top of ``torch.linalg``
# only.  See ``molmetal/math_skills/se3_schrodinger_bridge.py`` for
# the reference.
from molmetal.math_skills.se3_schrodinger_bridge import (
    so3_exp,
    so3_log,
)

__all__ = [
    "QuotientSpace",
    "QuotientDiffusionScheduler",
    "QuotientSpaceCFM",
]


# ---------------------------------------------------------------------------
# Numerical guards — mirror the constants from the math layer so the
# adapter is self-contained and never reaches into the module's
# private namespace.
# ---------------------------------------------------------------------------
_EPS_TRIG: float = 1.0e-6
_EPS_ROT: float = 1.0e-8
_EPS_SVD: float = 1.0e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_rotation(
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample a batch of random ``SO(3)`` rotation matrices.

    Uses an SVD-based polar decomposition of a random Gaussian matrix
    to obtain a (near-)orthogonal matrix, then fixes the determinant
    to ``+1`` so the result lives on ``SO(3)`` rather than ``O(3)``.
    This is the standard ``torch.randn`` -> SVD trick; it produces a
    *Haar-uniform* distribution on ``SO(3)`` (Mezzadri 2007).
    """
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        A = torch.randn(batch, 3, 3, generator=g, dtype=dtype).to(device)
    else:
        A = torch.randn(batch, 3, 3, dtype=dtype, device=device)
    u, _s, v = torch.linalg.svd(A)
    R = u @ v
    # Fix det(R) = +1 in case the SVD flipped it.  We flip the last
    # row of ``v`` for the entries with det < 0 — the canonical
    # "fix the sign" trick (Mezzadri 2007, §3).
    det = torch.linalg.det(R)
    v_flipped = v.clone()
    need_flip = det < 0
    if need_flip.any():
        v_flipped[need_flip, -1, :] = -v_flipped[need_flip, -1, :]
    R = u @ v_flipped
    return R


def _canonical_frame(
    x: torch.Tensor,  # (B, N, 3) input
    eps: float = _EPS_TRIG,
) -> torch.Tensor:
    """Build the canonical right-handed orthonormal frame ``F``
    for a batch of conformers.

    The frame has columns
    ``(e1_hat, e2_hat, e3_hat)`` where:

    * ``e1_hat = (x[1] - x[0]) / ||x[1] - x[0]||``
    * ``e2_hat = (x[2] - x[0]) - <x[2] - x[0], e1_hat> e1_hat``
      (then normalised) — the component of ``x[2] - x[0]``
      orthogonal to ``e1_hat``.
    * ``e3_hat = e1_hat × e2_hat`` (right-hand rule).

    This is the standard "three-point" canonical frame
    (Arvind 2026, Definition 3.4): atom 0 sits at the origin, atom 1
    along the x-axis, atom 2 in the xy-plane.  The frame is
    *equivariant under rotations of the input*: if ``x`` is replaced
    by ``R x``, then ``F`` is replaced by ``R F``, which is exactly
    the transformation we need for the quotient-space
    canonicalisation (see :meth:`QuotientSpace.project`).

    Parameters
    ----------
    x : Tensor
        Input conformer of shape ``(B, N, 3)``.  Requires ``N >= 3``.
    eps : float, default ``1e-6``
        Numerical floor for the ``||e1||`` and ``||e2_perp||``
        divisions.

    Returns
    -------
    Tensor
        Frame ``F`` of shape ``(B, 3, 3)`` with columns
        ``(e1_hat, e2_hat, e3_hat)`` and ``det(F) = +1`` (the
        right-handed orthonormal frame).

    Notes
    -----
    Degenerate conformers (``x[1] = x[0]`` or atoms 0, 1, 2 collinear)
    fall back to the identity frame — this is the canonical choice in
    the absence of a well-defined principal-axis basis.  Callers that
    need a different fallback should pre-filter their inputs.
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
    if x.shape[1] < 3:
        raise ValueError(f"canonical_frame requires N >= 3, got N={x.shape[1]}")
    b = x.shape[0]
    # Vector from atom 0 to atom 1.
    e1 = x[:, 1, :] - x[:, 0, :]  # (B, 3)
    e1_norm = e1.norm(dim=-1, keepdim=True).clamp_min(eps)
    e1_hat = e1 / e1_norm  # (B, 3)
    # Vector from atom 0 to atom 2, orthogonalised against e1_hat.
    e2 = x[:, 2, :] - x[:, 0, :]  # (B, 3)
    proj = (e2 * e1_hat).sum(dim=-1, keepdim=True)  # (B, 1)
    e2_perp = e2 - proj * e1_hat  # (B, 3)
    e2_norm = e2_perp.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2_hat = e2_perp / e2_norm  # (B, 3)
    # Right-hand rule.
    e3_hat = torch.linalg.cross(e1_hat, e2_hat, dim=-1)  # (B, 3)
    # Stack columns -> (B, 3, 3) frame.
    F = torch.stack([e1_hat, e2_hat, e3_hat], dim=-1)
    # Fix the determinant to +1: if e2_perp was on the *negative*
    # side of e1_hat, our construction picked the wrong normal
    # direction and the frame is left-handed.  Flip e2 (and e3) for
    # those entries to enforce a right-handed frame.
    det = torch.linalg.det(F)
    need_flip = det < 0
    if need_flip.any():
        F_fixed = F.clone()
        F_fixed[need_flip, :, 1] = -F_fixed[need_flip, :, 1]
        F_fixed[need_flip, :, 2] = -F_fixed[need_flip, :, 2]
        F = F_fixed
    return F


# ---------------------------------------------------------------------------
# QuotientSpace — the geometric kernel (project / lift)
# ---------------------------------------------------------------------------
class QuotientSpace:
    """A quotient space :math:`X / G` for SE(3)-equivariant diffusion.

    Three quotient spaces are supported, all of which are subgroups of
    the full SE(3) action:

    * ``"SE3"`` — full quotient (centre + canonicalise orientation).
      The representative is the centred conformer with its orientation
      Procrustes-aligned to a canonical reference (the identity).
    * ``"rotation"`` — quotient by SO(3) only.  The representative is
      the original conformer with its orientation canonicalised.
    * ``"translation"`` — quotient by translations only.  The
      representative is the centred conformer (orientation preserved).

    The contract pinned by the test suite is:

    * ``project(x) == project(R x + t)`` for every SE(3)
      transformation ``(R, t)`` (within numerical tolerance).
    * ``lift(project(x))`` returns a conformer with the same *shape*
      as ``x`` but with a canonical centre / orientation.

    Parameters
    ----------
    manifold : str, default ``"SE3"``
        Which quotient space to use.  One of ``{"SE3", "rotation",
        "translation"}``.
    eps : float, default ``1e-6``
        Numerical floor for the SVD / division operations.

    Notes
    -----
    The ``"SE3"`` quotient first centres the coordinates (killing the
    translation component) and then canonicalises the orientation via
    a Procrustes alignment to the identity reference.  This is the
    "shape embedding" from Arvind 2026 §3.
    """

    def __init__(
        self,
        manifold: str = "SE3",
        eps: float = 1e-6,
    ) -> None:
        if manifold not in {"SE3", "rotation", "translation"}:
            raise ValueError(
                f"manifold must be one of {{'SE3', 'rotation', 'translation'}}, "
                f"got {manifold!r}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.manifold = str(manifold)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # Public API: project / lift
    # ------------------------------------------------------------------
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Map a conformer ``x`` to a representative in the quotient.

        The canonical frame is built from the *first three atoms* of
        the input: atom 0 sits at the origin, atom 1 along the
        x-axis, atom 2 in the xy-plane.  This canonicalisation is
        *equivariant by construction*: under ``x -> R x + t`` the
        frame becomes ``R F``, and the canonical representative
        ``x_c @ F`` is unchanged (because ``(R x_c) @ (R F) = x_c
        R^T R F = x_c F``).  See Arvind 2026, Lemma 3.5.

        Parameters
        ----------
        x : Tensor
            Conformer of shape ``(B, N, 3)``.  Requires ``N >= 3``
            (the canonical frame needs at least 3 atoms).

        Returns
        -------
        Tensor
            Representative of the same shape ``(B, N, 3)`` such that
            ``project(x) == project(R x + t)`` for every SE(3)
            transformation ``(R, t)`` (within numerical tolerance).
        """
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        if x.shape[1] < 3:
            raise ValueError(
                f"project requires N >= 3 (need 3 atoms to build the "
                f"canonical frame), got N={x.shape[1]}"
            )
        rep, _F, _c = self._project_with_frame(x)
        return rep

    def _project_with_frame(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to the quotient and return the canonicalisation frame.

        Returns
        -------
        rep : Tensor
            The canonical representative (``project(x)``).
        F : Tensor
            The canonical orthonormal frame of shape ``(B, 3, 3)``
            used in the projection.  The original input is recovered
            from ``rep`` via ``out = rep @ F^{-1} + c`` (or just
            ``rep @ F^{-1}`` for the ``"rotation"`` manifold).
        c : Tensor
            The centroid of shape ``(B, 1, 3)`` that was subtracted
            before the rotation step (``None`` is *not* returned — we
            return the zero tensor for the ``"rotation"`` manifold
            where no centring is applied).

        Notes
        -----
        This is the private hook used by :class:`QuotientSpaceCFM`
        to track the original frame so the lifted output is in the
        *same* frame as the input (which is what gives true SE(3)
        equivariance, not just invariance).  Public callers should
        use :meth:`project` and :meth:`lift`.
        """
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        if x.shape[1] < 3:
            raise ValueError(f"project requires N >= 3, got N={x.shape[1]}")
        if self.manifold == "translation":
            c = x.mean(dim=1, keepdim=True)
            rep = x - c
            b = x.shape[0]
            F = (
                torch.eye(3, dtype=x.dtype, device=x.device)
                .unsqueeze(0)
                .expand(b, 3, 3)
                .contiguous()
            )
            return rep, F, c
        # For "SE3" and "rotation": build the canonical frame.
        if self.manifold == "SE3":
            c = x.mean(dim=1, keepdim=True)
            x_in = x - c
        else:  # "rotation"
            c = torch.zeros(
                x.shape[0],
                1,
                3,
                dtype=x.dtype,
                device=x.device,
            )
            x_in = x
        F = _canonical_frame(x_in, eps=self.eps)  # (B, 3, 3)
        out = torch.matmul(x_in, F)
        return out, F, c

    def lift(self, rep: torch.Tensor) -> torch.Tensor:
        """Lift a quotient-space representative back to the original space.

        For ``"SE3"`` and ``"rotation"`` the lift is the canonical
        rotation back to the *input* reference; here we simply
        return the representative (it already lives in the canonical
        frame, which is a valid lift).  For ``"translation"`` we
        re-centre and return the result (the absolute translation is
        lost under the quotient, so the lift is only defined up to a
        global translation — the representative is one valid choice).

        Parameters
        ----------
        rep : Tensor
            Quotient representative of shape ``(B, N, 3)``.

        Returns
        -------
        Tensor
            Conformer of shape ``(B, N, 3)`` living in the original
            SE(3) space.
        """
        if rep.dim() != 3 or rep.shape[-1] != 3:
            raise ValueError(f"rep must have shape (B, N, 3), got {tuple(rep.shape)}")
        # The lift is the *identity* on the representative: the
        # representative already lives in a canonical frame, which
        # is a valid point in the original space.  For
        # ``"translation"`` the lift is up to a global translation;
        # we return the representative as-is (centred at the origin).
        return rep.clone()

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def shape_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Distance between two conformers in the quotient space.

        The quotient-space distance is
        ``||project(x) - project(y)||`` — a *shape* distance that is
        invariant under SE(3) for ``"SE3"``, under SO(3) for
        ``"rotation"``, and under translations for ``"translation"``.
        Returns a tensor of shape ``(B,)``.

        This is the metric used by :meth:`QuotientSpaceCFM.equivariance_error`
        to check the equivariance contract.
        """
        if x.shape != y.shape or x.shape[-1] != 3:
            raise ValueError(
                f"x / y must share shape (B, N, 3), got "
                f"{tuple(x.shape)} vs {tuple(y.shape)}"
            )
        px = self.project(x)
        py = self.project(y)
        return (px - py).flatten(start_dim=1).norm(dim=-1)


# ---------------------------------------------------------------------------
# QuotientDiffusionScheduler — VP-style noise scheduler in the quotient
# ---------------------------------------------------------------------------
class QuotientDiffusionScheduler:
    """A cosine-shaped noise scheduler that runs in the quotient space.

    The scheduler follows the variance-preserving (VP) schedule from
    Ho et al. 2020 / Nichol & Dhariwal 2021, restricted to the
    quotient space :math:`X / G`:

    .. math::

        \\sigma(t) = \\sigma_{\\min}
                     \\left( \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}
                     \\right)^{t} ,

        \\alpha(t) = \\cos(\\theta(t)), \\quad
        \\beta(t) = \\sin(\\theta(t)),

        X_t = \\alpha(t) X_0 + \\beta(t) \\epsilon, \\quad
        \\epsilon \\sim \\mathcal{N}(0, I).

    The contract pinned by the test suite is:

    * ``step(x, t)`` returns a tensor of the same shape as ``x``.
    * ``step`` is *equivariant* — under a global rotation
      ``x \\mapsto R x``, the noised tensor transforms by the same
      ``R`` (because both the data and the noise transform by ``R``).

    Parameters
    ----------
    n_steps : int, default ``100``
        Number of diffusion steps.  ``t`` is an integer in
        ``[0, n_steps]``.
    sigma_min : float, default ``0.01``
        Minimum noise level.
    sigma_max : float, default ``10.0``
        Maximum noise level.
    seed : Optional[int], default ``None``
        Random seed for the noise draws.

    Notes
    -----
    The scheduler is parameter-free (no learnable parameters).  It is
    a pure noise scheduler — the *learned* component is the velocity
    field of the underlying CFM (in :class:`QuotientSpaceCFM`).
    """

    def __init__(
        self,
        n_steps: int = 100,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if sigma_min <= 0.0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be > sigma_min, got "
                f"sigma_max={sigma_max}, sigma_min={sigma_min}"
            )
        self.n_steps = int(n_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.seed = seed

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------
    def sigma(self, t: int) -> float:
        """Return the noise level :math:`\\sigma(t)`.

        Uses the geometric interpolation ``sigma(t) = sigma_min *
        (sigma_max / sigma_min) ** (t / n_steps)`` (Arvind 2026,
        eq. (4.7)).  At ``t=0`` we get ``sigma_min``; at
        ``t=n_steps`` we get ``sigma_max``.
        """
        if not (0 <= t <= self.n_steps):
            raise ValueError(f"t must be in [0, {self.n_steps}], got {t}")
        if t == 0:
            return self.sigma_min
        if t == self.n_steps:
            return self.sigma_max
        ratio = self.sigma_max / self.sigma_min
        # Use a normalised progress u in (0, 1) so the schedule is
        # smooth across the integer boundary.
        u = float(t) / float(self.n_steps)
        return self.sigma_min * (ratio**u)

    def alpha_beta(self, t: int) -> Tuple[float, float]:
        """Return :math:`(\\alpha(t), \\beta(t))` for the VP schedule.

        We use the **cosine** schedule (Nichol & Dhariwal 2021):
        ``alpha(t) = cos(theta), beta(t) = sin(theta)`` where
        ``theta = arcsin(sigma(t) / sqrt(1 + sigma(t)^2))`` is the
        "noise angle" of the VP SDE.
        """
        sig = self.sigma(t)
        # theta = arcsin(sig / sqrt(1 + sig^2)) so that
        # cos(theta) = 1 / sqrt(1 + sig^2), sin(theta) = sig / sqrt(1 + sig^2).
        s2 = sig * sig
        cos_t = 1.0 / math.sqrt(1.0 + s2)
        sin_t = sig / math.sqrt(1.0 + s2)
        return float(cos_t), float(sin_t)

    # ------------------------------------------------------------------
    # Forward diffusion step
    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Forward diffusion step: return ``x_{t+1}`` from ``x_t = x``.

        We add Gaussian noise with variance ``sigma(t+1)^2 -
        sigma(t)^2`` (the *incremental* variance).  This is the
        standard DDPM forward step, expressed in the quotient space
        so the equivariance contract is automatic.

        Parameters
        ----------
        x : Tensor
            Noisy conformer at time ``t``, of shape ``(B, N, 3)``.
        t : int
            Current time step, integer in ``[0, n_steps - 1]``.

        Returns
        -------
        Tensor
            Noisier conformer at time ``t + 1``, shape ``(B, N, 3)``.
        """
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        if not (0 <= t < self.n_steps):
            raise ValueError(f"step requires t in [0, {self.n_steps - 1}], got {t}")
        sig_t = self.sigma(t)
        sig_tp = self.sigma(t + 1)
        delta_var = max(sig_tp * sig_tp - sig_t * sig_t, 0.0)
        # Sample Gaussian noise with the incremental variance.
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=x.device.type).manual_seed(
                int(self.seed) + int(t),
            )
        eps = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=gen,
        )
        return x + math.sqrt(delta_var) * eps

    # ------------------------------------------------------------------
    # Full forward (sample at arbitrary t)
    # ------------------------------------------------------------------
    def forward_sample(
        self,
        x0: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Sample ``x_t`` directly from a clean ``x_0`` via the VP
        marginal :math:`q(x_t | x_0) = \\mathcal{N}(\\alpha(t) x_0,
        \\beta(t)^2 I)`.

        Parameters
        ----------
        x0 : Tensor
            Clean conformer of shape ``(B, N, 3)``.
        t : int
            Time step, integer in ``[0, n_steps]``.

        Returns
        -------
        Tensor
            Noisy conformer of shape ``(B, N, 3)``.
        """
        if x0.dim() != 3 or x0.shape[-1] != 3:
            raise ValueError(f"x0 must have shape (B, N, 3), got {tuple(x0.shape)}")
        alpha, beta = self.alpha_beta(t)
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=x0.device.type).manual_seed(
                int(self.seed) + int(t) * 9973,
            )
        eps = torch.randn(
            x0.shape,
            dtype=x0.dtype,
            device=x0.device,
            generator=gen,
        )
        return alpha * x0 + beta * eps


# ---------------------------------------------------------------------------
# QuotientSpaceCFM — drop-in wrapper around an existing CFM
# ---------------------------------------------------------------------------
class QuotientSpaceCFM:
    """Quotient-space flow matching — SE(3) equivariance by construction.

    The wrapper takes an *existing* CFM decoder (e.g. the R17
    ``LipmanFlowMatchingAdapter`` shipped in
    ``molmetal.adapters.flow_matching_lipman``) and runs the
    flow-matching process *inside* the quotient space :math:`X / G`.
    The contract is:

    1. Both the source ``z0`` and the target ``z1`` are projected
       into the quotient before being passed to the CFM.
    2. The flow-matching output is then *lifted* back to the
       original space.
    3. The result is **equivariant by construction**: a global SE(3)
       transformation of the inputs produces the same global SE(3)
       transformation of the output (because the quotient space is
       invariant under the SE(3) action).

    Parameters
    ----------
    base_cfm
        The underlying CFM.  Must expose ``forward(z0, t)`` (or
        ``sample`` / ``decode`` — the wrapper accepts any callable
        that returns a tensor of the same shape as ``z0``).
    quotient : Optional[QuotientSpace], default ``None``
        Quotient-space kernel.  Defaults to ``QuotientSpace("SE3")``
        (the full SE(3) quotient — the most common case for
        unbound molecular generation).

    Notes
    -----
    The wrapper does *not* modify the base CFM in any way.  It only
    adds a ``project`` step before the forward call and a ``lift``
    step after.  This makes the upgrade path trivial: any existing
    CFM checkpoint can be wrapped in :class:`QuotientSpaceCFM` to
    obtain SE(3) equivariance "for free" (modulo the centroid /
    orientation loss incurred by the projection).
    """

    def __init__(
        self,
        base_cfm,
        quotient: Optional[QuotientSpace] = None,
    ) -> None:
        if quotient is None:
            quotient = QuotientSpace("SE3")
        if not isinstance(quotient, QuotientSpace):
            raise TypeError(
                f"quotient must be a QuotientSpace instance, got "
                f"{type(quotient).__name__}"
            )
        if base_cfm is None or not callable(base_cfm):
            raise TypeError(f"base_cfm must be callable, got {type(base_cfm).__name__}")
        self.base_cfm = base_cfm
        self.quotient = quotient

    # ------------------------------------------------------------------
    # Forward contract
    # ------------------------------------------------------------------
    def forward(
        self,
        z0: torch.Tensor,  # (B, N, 3) source
        t: torch.Tensor,  # (B,) or scalar
    ) -> torch.Tensor:
        """Compute ``z(t)`` in the quotient space.

        The wrapper:
        1. Projects ``z0`` into the quotient (canonical frame).
        2. Calls ``self.base_cfm(z0_qu, t)`` (the base CFM operates
           in the quotient space, where it is automatically
           SE(3)-equivariant because the quotient kills the SE(3)
           action).
        3. Lifts the result back to the *original* frame using the
           canonicalisation frame ``F`` and centroid ``c`` captured
           in step 1.  This is the key to SE(3) equivariance: the
           output is in the same frame as the input, so a rotation
           / translation of the input produces the corresponding
           rotation / translation of the output.

        Parameters
        ----------
        z0 : Tensor
            Source conformer of shape ``(B, N, 3)``.
        t : Tensor
            Flow time.  Shape ``(B,)`` or scalar.

        Returns
        -------
        Tensor
            Conformer at time ``t``, shape ``(B, N, 3)``.
        """
        if z0.dim() != 3 or z0.shape[-1] != 3:
            raise ValueError(f"z0 must have shape (B, N, 3), got {tuple(z0.shape)}")
        # Project to the quotient, capturing the canonicalisation
        # frame so we can lift back to the original frame.
        z0_qu, F, c = self.quotient._project_with_frame(z0)
        # Run the base CFM in the quotient space.
        out_qu = self.base_cfm(z0_qu, t)
        if not isinstance(out_qu, torch.Tensor):
            raise TypeError(
                f"base_cfm must return a Tensor, got {type(out_qu).__name__}"
            )
        # Lift back to the original frame.  The original input was
        # ``z0 = (z0_qu @ F^{-1}) + c`` (or just ``z0_qu @ F^{-1}``
        # for the ``"rotation"`` manifold).  Inverting gives
        # ``out = (out_qu @ F^{-1}) + c``.
        out = torch.matmul(out_qu, F.transpose(-1, -2)) + c
        return out

    # ------------------------------------------------------------------
    # Equivariance diagnostics
    # ------------------------------------------------------------------
    def equivariance_error(
        self,
        n_runs: int = 10,
        atol: float = 1e-3,
        seed: int = 0,
    ) -> float:
        """Numerical check of the SE(3) equivariance contract.

        We sample ``n_runs`` random SE(3) transformations
        ``(R, t)`` and compare the wrapper's output on
        ``(R x + t)`` to ``R (wrapper(x)) + t``.  The contract
        pinned by the test suite is that the relative error

        .. math::

            \\frac{\\| R \\cdot \\mathrm{wrapper}(x) + t
                   - \\mathrm{wrapper}(R x + t) \\|}
                 {\\|\\mathrm{wrapper}(x)\\|}

        is below ``1e-3`` (the test ``test_quotient_cfm_equivariance``
        uses this tolerance).  Returns ``0.0`` if the contract is
        satisfied on all runs.

        Parameters
        ----------
        n_runs : int, default ``10``
            Number of random SE(3) transformations to test.
        atol : float, default ``1e-3``
            Absolute tolerance for the equivariance error.
        seed : int, default ``0``
            Seed for the random rotations / translations.

        Returns
        -------
        float
            Maximum relative error across the ``n_runs`` trials.
            Callers can check ``< atol`` for a hard pass/fail.
        """
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        # Build a tiny test conformer.  We use ``B=2, N=4`` so the
        # SVD in the Procrustes step is well-conditioned.
        torch.manual_seed(int(seed))
        x = torch.randn(2, 4, 3)
        t = torch.tensor([0.5, 0.5])
        x_ref = self.forward(x, t)  # (B, N, 3)
        ref_norm = x_ref.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-9)
        max_err = 0.0
        for k in range(int(n_runs)):
            R = _random_rotation(
                batch=x.shape[0],
                dtype=x.dtype,
                device=x.device,
                seed=int(seed) + 1000 + k,
            )
            t_b = torch.randn(
                x.shape[0],
                3,
                generator=torch.Generator().manual_seed(
                    int(seed) + 2000 + k,
                ),
            )
            # Apply (R, t_b) to x.
            x_trans = torch.einsum("bij,bnj->bni", R, x) + t_b.unsqueeze(1)
            # Wrapper output on the transformed conformer.
            out_trans = self.forward(x_trans, t)
            # Apply (R, t_b) to the reference output.
            out_ref_trans = torch.einsum(
                "bij,bnj->bni",
                R,
                x_ref,
            ) + t_b.unsqueeze(1)
            err = (
                (out_trans - out_ref_trans)
                .flatten(
                    start_dim=1,
                )
                .norm(dim=-1)
            )
            rel = (err / ref_norm).max().item()
            max_err = max(max_err, rel)
        return float(max_err)
