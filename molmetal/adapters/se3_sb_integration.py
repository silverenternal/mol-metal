"""SE(3) Schrödinger-Bridge adapter — wraps the existing CFM stack with Lie-group
geodesics (T1 + W2.1, Mol-Metal R18 upgrade path).

Reference
---------
Mahmood, F., Halder, S. & Akhtar, R. (2026).  *Schrödinger Bridges on
Compact Connected Lie Groups with Application to Molecular
Generation*.  arXiv:2603.14049 [stat.ML].  Cited in ``paper/refs.bib``
as ``mahmood2026sb_lie``.

The round-17 (R17) close-out shipped
``molmetal.adapters.flow_matching_lipman`` — a Euclidean CFM
(Lipman 2023) + SE(3)-equivariant EGNN velocity net.  W2.1 is the
first step of the round-18 (R18) Lie-group migration: replace the
Euclidean linear interpolation

    X_t = (1 - t) X_0 + t X_1

used by the CFM decoder head with the SE(3) geodesic from
:meth:`molmetal.math_skills.se3_schrodinger_bridge.SE3Geodesic`,
keeping the rest of the CFM stack (ODE solver, EGNN velocity net,
pocket conditioning, atom-head) untouched.

Design constraints (per W2.1 brief)
----------------------------------
1. **Do NOT modify** ``molmetal/adapters/flow_matching_lipman/`` —
   the R17 close-out is locked.  This file is the *only* new
   adapter added under W2.1.
2. **CPU-only.**  All operations use ``torch.linalg`` (CPU backend by
   default).  No CUDA / Triton kernels required for the math layer.
3. **Standalone.**  No internal imports from
   ``molmetal.molmetal_lam.*``.  Only ``torch`` + the pure
   ``molmetal.math_skills`` package + the EGNN adapter for the
   velocity field.
4. **Batched.**  Every public function accepts a leading batch
   dimension and broadcasts over it.

Module map
----------
* :class:`SE3SBVelocityField` — Lie-group velocity field that wraps an
  existing EGNN velocity net and returns a *tangent vector in
  se(3)* (twist of shape ``(..., 6)`` = linear + angular).
* :class:`SE3SBDecoder` — decoder head that walks the SE(3) geodesic
  between source and target poses at a sequence of time points,
  with optional Schrödinger-Bridge Sinkhorn coupling for the
  cross-pair assignment (replaces the Euclidean ``lerp`` decoder).
* :func:`integrate_with_cfm` — drop-in wrapper that integrates
  :class:`SE3SBVelocityField` over ``n_steps`` using an Euler / midpoint
  ODE scheme on the SE(3) manifold, mirroring the *shape contract* of
  the existing CFM ODE solver (``molmetal.adapters.flow_matching_lipman``
  uses ``flow_matching.solver.ode_solver.ODESolver`` — we keep the
  signature compatible so a future refactor can swap them in).

The velocity field is the *only* object that touches the trained EGNN
net; the decoder + integrator are pure math (no learnable parameters).
This keeps the W2.1 surface area small and the test contract clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# T1 — pure-math SE(3) layer (CPU-only, batched).  See
# ``molmetal/math_skills/se3_schrodinger_bridge.py`` for the
# mathematical reference (Mahmood 2026, arXiv:2603.14049).
from molmetal.math_skills.se3_schrodinger_bridge import (
    LogSinkhornLoss,
    SchrodingerBridgeSE3,
    SE3Geodesic,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)

__all__ = [
    "SE3SBVelocityField",
    "SE3SBDecoder",
    "integrate_with_cfm",
]


# ---------------------------------------------------------------------------
# Numerical guards — mirror the constants from the math layer so the
# adapter is self-contained and never reaches into the module's
# private namespace.
# ---------------------------------------------------------------------------
_EPS_TRIG: float = 1.0e-6
_EPS_ROT: float = 1.0e-8


# ---------------------------------------------------------------------------
# SE(3) velocity field — wraps an EGNN velocity net
# ---------------------------------------------------------------------------
class SE3SBVelocityField(nn.Module):
    """SE(3) Lie-group velocity field wrapping an EGNN velocity net.

    The Lipman 2023 CFM velocity field returns per-atom velocity in
    :math:`\\mathbb{R}^3` (the ambient Euclidean tangent space).  On
    SE(3) the tangent space is the Lie algebra ``se(3) \\cong
    \\mathbb{R}^6`` (linear + angular twist).  This wrapper converts
    the EGNN's per-atom :math:`\\mathbb{R}^3` velocity into the SE(3)
    twist space by:

    1. Computing the centroid of the input pose at time ``t`` — this
       is the SE(3) **base point** (the Lie-group identity is at the
       origin in the tangent space).
    2. Reducing the per-atom :math:`\\mathbb{R}^3` velocity to a single
       6-D twist per batch element: the *linear* part is the
       centroid velocity, the *angular* part is the antisymmetric
       component of the moment-arm cross product (the textbook
       identification :math:`\\mathfrak{se}(3) \\cong \\mathbb{R}^6`).

    The output twist is **equivariant** in the sense of Definition 3.4
    of :cite:`mahmood2026sb_lie`: under a global rotation ``R``, the
    *angular* part of the twist transforms by ``R`` and the *linear*
    part transforms by ``R`` (the cross product is equivariant).  The
    :func:`_twist_equivariance_check` test in ``test_se3_sb_integration``
    pins this property.

    Parameters
    ----------
    base_velocity_net : nn.Module
        The underlying EGNN velocity field.  We **do not** require it
        to be a specific class — any ``nn.Module`` whose ``forward``
        returns a tensor of shape ``(B, N, 3)`` (per-atom Euclidean
        velocity) works.  This keeps the adapter composable with both
        :class:`molmetal.adapters.flow_matching_lipman.EGNNVelocityField`
        and any future R18 replacement.
    manifold : str, default ``"SE3"``
        Which Lie group to project onto.  Currently only ``"SE3"`` is
        supported; ``"SO3"`` (rotations only, no translation) is on
        the roadmap but not implemented here — passing it raises
        :class:`NotImplementedError`.
    eps : float, default ``1e-6``
        Numerical floor for the moment-arm division (kept > ``1e-8``
        so we never mis-classify a near-collinear pair as coplanar).

    Notes
    -----
    The wrapper does **not** call ``self.training``-dependent code in
    the EGNN (no CFG, no context dropout — the wrapper is an inference-
    time component).  Callers that need CFG should apply it on the
    underlying EGNN before passing the result here.
    """

    def __init__(
        self,
        base_velocity_net: nn.Module,
        manifold: str = "SE3",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if manifold not in {"SE3", "SO3"}:
            raise ValueError(
                f"manifold must be one of {{'SE3', 'SO3'}}, got {manifold!r}"
            )
        if manifold == "SO3":
            raise NotImplementedError(
                "SO3 manifold support is on the R18 roadmap; only SE3 "
                "is implemented in W2.1."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.base_velocity_net = base_velocity_net
        self.manifold = str(manifold)
        self.eps = float(eps)

    def forward(
        self,
        x_T: torch.Tensor,        # (B, N, 3) per-atom coords at time t
        t: torch.Tensor,          # (B,) or (B, 1) flow time
        pocket: Optional[torch.Tensor] = None,  # (B, P, 3) pocket coords, optional
    ) -> torch.Tensor:
        """Return the SE(3) tangent (twist) of shape ``(B, 6)``.

        The output is in the Lie algebra ``se(3)``: ``[..., :3]`` is
        the linear (translational) component and ``[..., 3:]`` is the
        angular (rotational) component.  Both are computed from the
        per-atom :math:`\\mathbb{R}^3` velocity of the underlying EGNN
        by reducing over atoms (centroid translation + antisymmetric
        moment-arm cross product for rotation).
        """
        if x_T.dim() != 3 or x_T.shape[-1] != 3:
            raise ValueError(
                f"x_T must have shape (B, N, 3), got {tuple(x_T.shape)}"
            )
        b, n = x_T.shape[:2]
        # Query the underlying EGNN.  We treat ``pocket`` as an optional
        # context vector — the wrapper is *agnostic* to how the EGNN
        # consumes it.  If the EGNN signature doesn't accept pocket,
        # we fall back to no-pocket (callers should pre-compute the
        # pocket bias and add it themselves in that case).
        try:
            v_euclidean = self.base_velocity_net(x_T, t=t, pocket=pocket)
        except TypeError:
            # Legacy EGNN signature: ``forward(x, t)``.
            v_euclidean = self.base_velocity_net(x_T, t)
        # Reduce the (B, N, 3) Euclidean velocity to a single (B, 3)
        # centroid velocity + per-atom (B, N, 3) residual.
        v_centroid = v_euclidean.mean(dim=1)        # (B, 3)
        residual = v_euclidean - v_centroid.unsqueeze(1)  # (B, N, 3)
        # Linear component of the SE(3) twist = centroid velocity.
        upsilon = v_centroid  # (B, 3)
        # Angular component = antisymmetric part of (moment-arm ×
        # residual) reduced over atoms.  Concretely, the per-atom
        # moment arm is (x_T[i] - centroid_x_T); the per-atom angular
        # velocity is r_i × v_i (where v_i is the per-atom residual).
        # The total angular velocity is the per-atom mean of r_i × v_i
        # — this is the standard rigid-body identification of
        # so(3) with skew-symmetric 3x3 matrices.
        centroid_x = x_T.mean(dim=1, keepdim=True)        # (B, 1, 3)
        r = x_T - centroid_x                              # (B, N, 3)
        # Numerically safe cross product — torch.linalg.cross handles
        # the broadcasting cleanly.
        angular_per_atom = torch.linalg.cross(r, residual, dim=-1)  # (B, N, 3)
        omega = angular_per_atom.mean(dim=1)              # (B, 3)
        # Stack into the (B, 6) SE(3) twist (linear, angular).
        xi = torch.cat([upsilon, omega], dim=-1)          # (B, 6)
        return xi


# ---------------------------------------------------------------------------
# SE(3) Schrödinger-Bridge decoder — replaces the Euclidean lerp head
# ---------------------------------------------------------------------------
class SE3SBDecoder(nn.Module):
    """Decoder head that walks the SE(3) geodesic + Sinkhorn SB coupling.

    The Euclidean CFM decoder is

        X_t = (1 - t) X_0 + t X_1

    (linear interpolation in :math:`\\mathbb{R}^3`).  On the SE(3)
    manifold the geodesic is

        T(t) = T_0 \\exp(t \\, \\log(T_0^{-1} T_1))      (SE(3) geodesic)

    (Proposition 3.5 of :cite:`mahmood2026sb_lie`).  This decoder
    wraps :meth:`SE3Geodesic.interpolate_se3` and adds an optional
    *Schrödinger-bridge* Sinkhorn coupling (Theorem 4.1 of
    :cite:`mahmood2026sb_lie`) so the cross-pair assignment between
    source and target poses is *optimal-transport-aware* rather than
    naive ``i <-> i`` self-bridge.

    Parameters
    ----------
    eps : float, default ``0.1``
        Sinkhorn entropy regularisation (matches the default of
        :class:`LogSinkhornLoss`).
    n_iters : int, default ``50``
        Number of Sinkhorn iterations.
    use_sinkhorn : bool, default ``True``
        When ``True``, the decoder computes a Sinkhorn coupling
        between ``z0`` and ``z1`` before walking the geodesic.  When
        ``False``, the decoder is a pure SE(3) geodesic with
        identity coupling ``(z0[i], z1[i])`` — useful for tests that
        want bit-exact decoder output without Sinkhorn overhead.
    seed : Optional[int], default ``None``
        Random seed for the SB sample draws (kept consistent with
        :class:`SchrodingerBridgeSE3`).

    Notes
    -----
    The decoder is parameter-free (no learnable parameters beyond
    ``eps`` / ``n_iters`` / ``use_sinkhorn`` / ``seed``).  This is
    intentional: the SB coupling is *inference-time geometry*, not a
    learned model.  The trained EGNN velocity field is the learned
    component; the decoder is the deterministic geometry kernel.
    """

    def __init__(
        self,
        eps: float = 0.1,
        n_iters: int = 50,
        use_sinkhorn: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.use_sinkhorn = bool(use_sinkhorn)
        self.seed = seed
        # The Sinkhorn loss is stateless; we instantiate it lazily so
        # the decoder doesn't pre-allocate a Sinkhorn object the
        # caller never uses (e.g. when ``use_sinkhorn=False``).
        self._sinkhorn: Optional[LogSinkhornLoss] = None
        self._bridge: Optional[SchrodingerBridgeSE3] = None
        self._geo = SE3Geodesic()

    # ------------------------------------------------------------------
    # Lazy builders
    # ------------------------------------------------------------------
    def _ensure_sinkhorn(self) -> LogSinkhornLoss:
        if self._sinkhorn is None:
            self._sinkhorn = LogSinkhornLoss(
                eps=self.eps, n_iters=self.n_iters,
            )
        return self._sinkhorn

    def _ensure_bridge(self) -> SchrodingerBridgeSE3:
        if self._bridge is None:
            self._bridge = SchrodingerBridgeSE3(
                eps=self.eps, n_iters=self.n_iters, seed=self.seed,
            )
        return self._bridge

    # ------------------------------------------------------------------
    # Forward contract
    # ------------------------------------------------------------------
    def decode(
        self,
        z0: torch.Tensor,        # (B, 4, 4) source SE(3) pose
        z1: torch.Tensor,        # (B, 4, 4) target SE(3) pose
        pocket: Optional[torch.Tensor] = None,  # (B, P, 3) pocket, optional
        t_values: Optional[torch.Tensor] = None,  # (n_t,) eval points
    ) -> torch.Tensor:
        """Walk the SE(3) geodesic (with SB coupling) and return poses.

        Parameters
        ----------
        z0, z1
            Source / target SE(3) homogeneous matrices of shape
            ``(B, 4, 4)``.  Must have matching shapes and last two
            dims ``(4, 4)``.
        pocket
            Optional pocket context — accepted for signature parity
            with :class:`SE3SBVelocityField` and the existing CFM
            decoder, but the *decoding* is currently pocket-agnostic
            (the pocket signal enters through the velocity field, not
            the geodesic interpolation).  Callers should pre-thread
            the pocket into the velocity field if needed.
        t_values
            Optional 1-D tensor of evaluation points along ``[0, 1]``.
            Defaults to ``torch.tensor([0.5])`` (single midpoint —
            matches :class:`SchrodingerBridgeSE3.sample`).

        Returns
        -------
        Tensor
            Decoded poses of shape ``(B, n_t, 4, 4)``.
        """
        if z0.shape != z1.shape or z0.shape[-2:] != (4, 4):
            raise ValueError(
                f"z0 / z1 must share shape (B, 4, 4), got "
                f"{tuple(z0.shape)} vs {tuple(z1.shape)}"
            )
        if t_values is None:
            t_values = torch.tensor([0.5], dtype=z0.dtype, device=z0.device)
        if t_values.ndim != 1:
            raise ValueError(
                f"t_values must be 1-D, got shape {tuple(t_values.shape)}"
            )

        b = z0.shape[0]
        if self.use_sinkhorn:
            # Build the SB coupling in twist coordinates (the math
            # layer's :class:`SchrodingerBridgeSE3` operates on (n, 6)
            # twists).  This costs ``B^2`` pairwise cost entries — fine
            # for inference (B is typically <= 64), expensive for
            # training (B is typically 128+).  Callers that need
            # training-time Sinkhorn should pre-compute the coupling
            # once and reuse it across ``t_values``.
            xi0 = se3_log(z0)  # (B, 6)
            xi1 = se3_log(z1)  # (B, 6)
            sinkhorn = self._ensure_sinkhorn()
            # Use the log-domain Sinkhorn loss directly (no sample
            # draw — we want a deterministic decoder).  Pairwise cost
            # = squared Euclidean distance in twist space (the
            # common proxy for the SE(3) geodesic distance, eq. (12)
            # of Mahmood 2026).
            diff = xi0.unsqueeze(1) - xi1.unsqueeze(0)  # (B, B, 6)
            C = (diff * diff).sum(dim=-1)              # (B, B) squared L2
            a = torch.full((b,), 1.0 / b, dtype=z0.dtype, device=z0.device)
            _loss, P = sinkhorn.forward(xi0, xi1, C=C, a=a, b=a)
            # Greedy deterministic assignment: for each source i pick
            # the target argmax_j P[i, j].  This is the standard
            # hard-assignment recovery of the SB plan (Theorem 4.1
            # of :cite:`mahmood2026sb_lie` — the SB coupling
            # converges to the un-regularised OT plan as ``eps -> 0``).
            assign = P.argmax(dim=-1)  # (B,)
            z1_matched = z1[assign]    # (B, 4, 4)
        else:
            z1_matched = z1

        # Walk the SE(3) geodesic at each t in t_values.
        out: List[torch.Tensor] = []
        for t in t_values:
            t_t = self._geo.interpolate_se3(z0, z1_matched, float(t))  # (B, 4, 4)
            out.append(t_t)
        return torch.stack(out, dim=-3)  # (B, n_t, 4, 4)

    # ------------------------------------------------------------------
    # Convenience: at-zero check (decode(z, z, pocket) ≈ z)
    # ------------------------------------------------------------------
    def decode_at(
        self,
        z: torch.Tensor,
        t: float = 0.0,
    ) -> torch.Tensor:
        """Single-t decoder with self-coupling: ``decode(z, z) at t``.

        When ``t == 0`` this returns ``z`` exactly (the SE(3) geodesic
        starts at ``T_0``); when ``t == 1`` it also returns ``z``
        (because the source == target).  This is the contract pinned
        by :func:`test_se3_sb_decoder_at_zero` in the test module.
        """
        if z.shape[-2:] != (4, 4):
            raise ValueError(
                f"decode_at expects (..., 4, 4) input, got {tuple(z.shape)}"
            )
        # Use the SB decoder with self-pair; the geodesic is the
        # identity at t=0 and t=1, so the output is exactly z.
        t_tensor = torch.tensor([float(t)], dtype=z.dtype, device=z.device)
        return self.decode(z, z, t_values=t_tensor).squeeze(-3)  # (B, 4, 4)


# ---------------------------------------------------------------------------
# ODE integrator — drop-in wrapper around the CFM ODE solver
# ---------------------------------------------------------------------------
def integrate_with_cfm(
    samples: torch.Tensor,             # (B, N, 3) per-atom initial noise
    velocity_field: SE3SBVelocityField,
    n_steps: int = 100,
    manifold: str = "SE3",
    method: str = "midpoint",
    eps: float = 1e-6,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Integrate an SE(3) ODE flow from ``t0`` to ``t1`` in ``n_steps``.

    Mirrors the *shape contract* of the existing CFM ODE solver
    (``flow_matching.solver.ode_solver.ODESolver.sample``) without
    importing it — the R17 ``flow_matching_lipman`` adapter's solver
    is locked, and we are forbidden from modifying it.  This wrapper
    is a *pure math* integrator: the velocity field is queried at
    each step, the per-atom ``(B, N, 3)`` velocity is wrapped through
    :class:`SE3SBVelocityField` into a ``(B, 6)`` twist, and the twist
    is integrated via :func:`_euler_se3` / :func:`_midpoint_se3` on
    the SE(3) manifold.  The *per-atom* trajectory is recovered by
    walking the SE(3) geodesic from each atom's initial coordinate.

    Parameters
    ----------
    samples
        Initial per-atom positions of shape ``(B, N, 3)``.  These are
        the CFM-style noise samples from which the SE(3) flow is
        integrated.
    velocity_field
        :class:`SE3SBVelocityField` instance wrapping the trained EGNN
        velocity net.
    n_steps
        Number of integration steps.  ``n_steps=100`` matches the
        default of the R17 CFM ODE solver.
    manifold
        Currently ``"SE3"`` only; ``"SO3"`` is on the R18 roadmap.
    method
        ``"euler"`` for first-order O(h) or ``"midpoint"`` for
        second-order O(h^2) (Heun's method).  Matches the
        :class:`molmetal.ports.GenerationConfig.method` enum.
    eps
        Numerical floor for the twist-magnitude division.
    t0, t1
        Start / end of the integration window.  Defaults to
        ``(0.0, 1.0)`` (the standard CFM convention).

    Returns
    -------
    Tensor
        Final per-atom positions of shape ``(B, N, 3)``.

    Notes
    -----
    The integrator walks the SE(3) flow *on the centroid* (single
    ``(B, 6)`` twist per batch element) and *on each atom* via a
    small-angle geodesic walk.  This is the standard SE(3)-CFM
    trick: the rigid-body centroid twist drives the global
    translation + rotation, and the per-atom positions are then
    *rotated by the centroid rotation* at each step.  The contract
    pinned by :func:`test_integrate_with_cfm_runs` is that the output
    shape matches the input shape and the integration is numerically
    stable for ``n_steps >= 1``.
    """
    if manifold not in {"SE3", "SO3"}:
        raise ValueError(
            f"manifold must be one of {{'SE3', 'SO3'}}, got {manifold!r}"
        )
    if manifold == "SO3":
        raise NotImplementedError(
            "SO3 integration is on the R18 roadmap; only SE3 is "
            "implemented in W2.1."
        )
    if method not in {"euler", "midpoint"}:
        raise ValueError(
            f"method must be one of {{'euler', 'midpoint'}}, got {method!r}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if samples.dim() != 3 or samples.shape[-1] != 3:
        raise ValueError(
            f"samples must have shape (B, N, 3), got {tuple(samples.shape)}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if not (0.0 <= t0 < t1 <= 1.0 + 1e-9):
        raise ValueError(
            f"require 0 <= t0 < t1 <= 1, got t0={t0}, t1={t1}"
        )

    b, n = samples.shape[:2]
    device, dtype = samples.device, samples.dtype
    # Initial SE(3) state: identity rotation + centroid translation
    # (the SE(3) base point is the identity).  The SE(3) flow
    # integrates over the batch centroid; per-atom positions are
    # updated by the centroid rotation (see below).
    T = torch.eye(4, dtype=dtype, device=device).expand(b, 4, 4).clone()
    T[..., :3, 3] = samples.mean(dim=1)  # centroid translation

    # Per-atom offsets (relative to the centroid) — these are
    # *rotated* by the centroid rotation at each step.  This is the
    # standard SE(3)-CFM rigid-body identification (atom positions
    # live in the body frame of the centroid pose).
    offsets = samples - samples.mean(dim=1, keepdim=True)  # (B, N, 3)

    dt = float(t1 - t0) / float(n_steps)
    integrator = _midpoint_se3 if method == "midpoint" else _euler_se3

    for k in range(n_steps):
        t_k = t0 + float(k) * dt
        # Evaluate the velocity field at time ``t_k``.  We pass the
        # per-atom centroid-rotated coordinates so the EGNN sees a
        # consistent pose at time ``t_k``.
        T_k = T
        # Per-atom positions at time t_k = centroid translation +
        # centroid rotation * offsets.
        R_k = T_k[..., :3, :3]                       # (B, 3, 3)
        x_k = T_k[..., :3, 3].unsqueeze(1) + torch.einsum(
            "bij,bnj->bni", R_k, offsets,
        )                                            # (B, N, 3)
        # Normalise t to (B,) for the velocity field.
        t_tensor = torch.full((b,), float(t_k), dtype=dtype, device=device)
        xi_k = velocity_field(x_k, t_tensor)         # (B, 6)

        if method == "midpoint":
            # Midpoint (Heun) step: probe the velocity at t_k + dt/2
            # with the Euler-updated pose.
            T_half = _euler_se3(T_k, xi_k, 0.5 * dt, eps=eps)
            R_half = T_half[..., :3, :3]
            x_half = T_half[..., :3, 3].unsqueeze(1) + torch.einsum(
                "bij,bnj->bni", R_half, offsets,
            )
            t_half = t_tensor + 0.5 * dt
            xi_half = velocity_field(x_half, t_half)  # (B, 6)
            T = integrator(T_k, xi_half, dt, eps=eps)
        else:
            T = integrator(T_k, xi_k, dt, eps=eps)

    # Final per-atom positions = centroid translation + centroid
    # rotation * offsets.
    R_final = T[..., :3, :3]                         # (B, 3, 3)
    x_final = T[..., :3, 3].unsqueeze(1) + torch.einsum(
        "bij,bnj->bni", R_final, offsets,
    )                                                # (B, N, 3)
    return x_final


# ---------------------------------------------------------------------------
# Internal SE(3) integrators (pure math — no learnable parameters)
# ---------------------------------------------------------------------------
def _euler_se3(
    T: torch.Tensor,        # (B, 4, 4) SE(3) pose at time t
    xi: torch.Tensor,       # (B, 6) twist at time t
    dt: float,
    eps: float,
) -> torch.Tensor:
    """Single Euler step on SE(3): ``T <- T @ exp(dt * xi)``.

    This is the standard first-order SE(3) integrator (Hall 2015,
    eq. (2.58)).  Returns the pose at time ``t + dt``.
    """
    T_out = T @ se3_exp(dt * xi)
    return T_out


def _midpoint_se3(
    T: torch.Tensor,        # (B, 4, 4) SE(3) pose at time t
    xi: torch.Tensor,       # (B, 6) midpoint twist
    dt: float,
    eps: float,
) -> torch.Tensor:
    """Single midpoint (Heun) step on SE(3).

    For Heun's method on SE(3) we use the *group operation* (not
    the tangent-space arithmetic): the step is

        T_new = T @ exp(dt * xi_midpoint)

    where ``xi_midpoint`` is the velocity field evaluated at the
    Euler-updated midpoint pose.  See :func:`integrate_with_cfm`
    for the full loop.
    """
    return _euler_se3(T, xi, dt, eps=eps)
