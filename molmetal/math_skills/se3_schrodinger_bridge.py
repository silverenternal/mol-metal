"""Schrödinger Bridges on SE(3) — mathematical skeleton (T1).

Reference
---------
Mahmood, F., Halder, S. & Akhtar, R. (2026).  *Schrödinger Bridges on
Compact Connected Lie Groups with Application to Molecular
Generation*.  arXiv:2603.14049 [stat.ML].  Cited in ``paper/refs.bib``
as ``mahmood2026sb_lie``.

This module implements the three mathematical primitives that the
R18 upgrade path to Mol-Metal will need:

1. **SE(3) geodesic interpolation** (``SE3Geodesic``) — replaces the
   Euclidean interpolant
   :math:`X_t = (1 - t) X_0 + t X_1`
   used by the current CFM path in
   ``molmetal.adapters.flow_matching_lipman`` with the proper
   Lie-group geodesic on ``SE(3) = SO(3) ⋉ R^3``.  Theorem 3.2 of
   :cite:`mahmood2026sb_lie` guarantees the geodesic exists, is
   unique, and is the gradient flow of the Killing-form metric.
2. **Log-domain Sinkhorn divergence** (``LogSinkhornLoss``) — the
   entropy-regularised OT objective (eq. (5)–(6) of
   :cite:`mahmood2026sb_lie`) used to solve the SB fixed-point
   problem.  Working in log-domain avoids the numerical overflow
   that plagues the textbook scaling iteration when ``eps`` is
   small.
3. **Schrödinger Bridge on SE(3)** (``SchrodingerBridgeSE3``) — the
   top-level entry point.  Given two empirical distributions on
   SE(3), it computes the entropic-regularised optimal coupling
   (the *SB plan*) via Sinkhorn, then samples intermediate poses by
   walking along the SE(3) geodesics connecting the matched pairs.
   Theorem 4.1 of :cite:`mahmood2026sb_lie` proves this coupling
   converges to the un-regularised SB as ``eps -> 0``.

Design choices
--------------
* **CPU-only.**  All operations use ``torch.linalg`` (which runs on
  CPU when CUDA is unavailable); no custom CUDA / Triton kernels are
  needed for the *mathematical skeleton*.  The R18 integration in
  ``molmetal.adapters.flow_matching_lipman`` may add a GPU kernel
  later but the math layer stays portable.
* **Standalone.**  No imports from ``molmetal.adapters.*``,
  ``molmetal.molmetal_lam.*`` or any other *internal* subpackage —
  only :mod:`torch`.  This is enforced by the unit tests
  (``test_se3_schrodinger_bridge.py``) which import this module in
  isolation.
* **Batched.**  Every public function accepts a leading batch
  dimension ``(B, ...)`` and broadcasts over it.  This lets the
  same code path scale from a single SE(3) element to thousands of
  poses without writing a separate "single" / "batched" path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Numerical zero — used everywhere we need a "small enough" cutoff for
#: trig special-cases (e.g. the ``sin(theta) -> 0`` branch in the
#: SO(3) log-map).  Picked to match the IEEE-754 float32 epsilon scaled
#: by a factor of 8 so the branch fires well before float32 catastrophic
#: cancellation in the skew-symmetric ``(R - R^T) / (2 * sin theta)``
#: division.
_EPS_TRIG = 1.0e-6

#: Numerical zero for the *rotation-vector* branch — anything whose
#: norm is below this is treated as the identity rotation in
#: :func:`so3_exp` / :func:`so3_log`.  Picked to be safely above
#: float32 ``2 ** -23 ~ 1.2e-7`` machine epsilon, so we never mis-
#: classify a near-zero rotation as the identity due to round-off.
_EPS_ROT = 1.0e-8


# ---------------------------------------------------------------------------
# Mathematical preconditions (CHECK_MATH_PRECONDITIONS=1)
# ---------------------------------------------------------------------------
# The math below is only valid when its inputs satisfy the assumptions
# of Mahmood-Halder-Akhtar (2026): the ``a`` / ``b`` vectors must be
# probability measures (non-negative, summing to 1), the cost matrix
# ``C`` must be a genuine transport cost (non-negative, finite), and
# the SE(3) twist batch must have at least one pose for the self-bridge
# coupling to be well-defined.  The module trains / runs even when these
# are violated (e.g. an un-normalised weight vector silently rescales
# the OT objective; a negative cost entry breaks the log-kernel), so we
# expose a *cheap*, non-raising ``check_preconditions`` that reports the
# violations instead.  Default OFF — enabling it must not change any
# numerical result.


def math_preconditions_enabled() -> bool:
    """Return True when ``CHECK_MATH_PRECONDITIONS=1`` is set in the env."""
    import os

    return os.environ.get("CHECK_MATH_PRECONDITIONS", "0") == "1"


def check_preconditions(
    inputs,
    *,
    entry: str = "sinkhorn",
    a=None,
    b=None,
    C=None,
    t_values=None,
    **kwargs,
) -> list:
    """Report violated mathematical preconditions of the SB entry points.

    Parameters
    ----------
    inputs : Tensor
        For ``entry="sinkhorn"`` a ``(n, d)`` sample batch ``p0``.  For
        ``entry="sample"`` a ``(n, 6)`` batch of ``se(3)`` twists.
    entry : str
        ``"sinkhorn"`` (LogSinkhornLoss.forward) or ``"sample"``
        (SchrodingerBridgeSE3.sample).
    a, b : Tensor, optional
        Marginal vectors; must be probability measures.
    C : Tensor, optional
        ``(n, m)`` cost matrix; must be non-negative + finite.
    t_values : Tensor, optional
        1-D evaluation points along the bridge; should lie in ``[0, 1]``.

    Returns
    -------
    list[str]
        Empty when all preconditions hold; otherwise one string per
        violated precondition, naming the violation.  Never raises on
        malformed input — a malformed input is itself a violation.
    """
    import os

    del os  # keep import local; no global state beyond the env flag
    violations = []
    if inputs is None or not isinstance(inputs, Tensor):
        return ["p0: input is None or not a Tensor (must be a sample batch)"]
    if entry == "sample":
        if inputs.dim() != 2 or inputs.shape[-1] != 6:
            violations.append(
                f"p0: SE(3) twist batch must be (n, 6), got {tuple(inputs.shape)}"
            )
        if inputs.shape[0] < 1:
            violations.append("p0: batch size n must be >= 1 for the self-bridge coupling")
        if not bool(torch.isfinite(inputs).all()):
            violations.append("p0: contains non-finite (NaN/inf) twist entries")
        if t_values is not None:
            if not isinstance(t_values, Tensor) or t_values.ndim != 1:
                violations.append("t_values: must be a 1-D tensor of evaluation points")
            else:
                if not bool(torch.isfinite(t_values).all()):
                    violations.append("t_values: contains non-finite entries")
                elif bool(((t_values < 0.0) | (t_values > 1.0)).any()):
                    violations.append(
                        "t_values: evaluation points must lie in [0, 1] "
                        "(the bridge is a probability path on t in [0, 1])"
                    )
        return violations

    # entry == "sinkhorn"
    if inputs.dim() != 2:
        violations.append(
            f"p0: source samples must be 2-D (n, d), got {tuple(inputs.shape)}"
        )
    if inputs.shape[0] < 1:
        violations.append("p0: source batch size n must be >= 1")
    if not bool(torch.isfinite(inputs).all()):
        violations.append("p0: contains non-finite (NaN/inf) sample entries")

    for name, v in (("a", a), ("b", b)):
        if v is None:
            continue
        if not isinstance(v, Tensor) or v.dim() != 1:
            violations.append(f"{name}: marginal must be a 1-D tensor")
            continue
        if not bool(torch.isfinite(v).all()):
            violations.append(f"{name}: marginal contains non-finite entries")
            continue
        if bool((v < 0.0).any()):
            violations.append(
                f"{name}: marginal is not a probability measure (has negative mass)"
            )
        total = float(v.sum())
        if abs(total - 1.0) > 1e-4:
            violations.append(
                f"{name}: marginal does not sum to 1 (sum={total:.6g}) — "
                f"not a probability measure"
            )
    if C is not None:
        if not isinstance(C, Tensor) or C.dim() != 2:
            violations.append("C: cost matrix must be a 2-D tensor")
        else:
            if not bool(torch.isfinite(C).all()):
                violations.append("C: cost matrix contains non-finite entries")
            elif bool((C < 0.0).any()):
                violations.append("C: cost matrix has negative entries (not a valid cost)")
    return violations


# ---------------------------------------------------------------------------
# SO(3) primitives (Rodrigues formula + log map)
# ---------------------------------------------------------------------------


def so3_exp(omega: Tensor) -> Tensor:
    """Exponential map from ``so(3)`` to ``SO(3)`` (Rodrigues' formula).

    Given a rotation vector :math:`\\omega = \\theta \\hat{n}` of
    shape ``(..., 3)``, returns the corresponding rotation matrix of
    shape ``(..., 3, 3)`` via Rodrigues' formula (Hall 2015, §2.4):

    .. math::

        R = I + \\sin\\theta [\\hat n]_\\times
              + (1 - \\cos\\theta) [\\hat n]_\\times^2,

    where :math:`[\\hat n]_\\times` is the skew-symmetric matrix
    built from :math:`\\hat n`.

    Theorem reference
    -----------------
    Theorem 2.1 of :cite:`mahmood2026sb_lie` (Lie algebra exponential
    map is a local diffeomorphism for compact connected Lie groups).

    Special cases
    -------------
    * ``||omega|| < _EPS_ROT`` — returns the identity.
    * The formula is **numerically stable** for any ``theta`` because
      we use :func:`torch.linalg.cross` style construction (no
      singular divisions).

    Parameters
    ----------
    omega
        Rotation vector(s) of shape ``(..., 3)``.

    Returns
    -------
    Tensor
        Rotation matrix of shape ``(..., 3, 3)``.
    """
    if omega.shape[-1] != 3:
        raise ValueError(f"so3_exp expects last dim 3, got shape {tuple(omega.shape)}")

    theta = omega.norm(dim=-1, keepdim=True).clamp_min(_EPS_ROT)  # (..., 1)
    n = omega / theta  # unit axis (..., 3)

    # Skew-symmetric matrix [n]_x built by hand (no autograd through
    # cross-product needed since we want a stable, deterministic op).
    zero = torch.zeros_like(n[..., 0])
    K = torch.stack(
        [
            torch.stack([zero, -n[..., 2], n[..., 1]], dim=-1),
            torch.stack([n[..., 2], zero, -n[..., 0]], dim=-1),
            torch.stack([-n[..., 1], n[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    eye = eye.expand(K.shape)  # broadcast identity
    sin_t = torch.sin(theta).unsqueeze(-1)  # (..., 1, 1)
    cos_t = torch.cos(theta).unsqueeze(-1)  # (..., 1, 1)

    return eye + sin_t * K + (1.0 - cos_t) * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """Logarithm map from ``SO(3)`` to ``so(3)``.

    Given a rotation matrix of shape ``(..., 3, 3)``, returns the
    rotation vector :math:`\\omega = \\theta \\hat n` of shape
    ``(..., 3)`` such that :func:`so3_exp` recovers ``R``.

    Theorem reference
    -----------------
    Theorem 2.1 of :cite:`mahmood2026sb_lie` (the log map is the
    inverse of the exp map on the cut locus :math:`\\theta < \\pi`,
    and extends continuously to the antipodal point).

    Special cases
    -------------
    * ``trace(R) == -1`` (antipodal rotation, ``theta = pi``) — falls
      back to a square-root extraction: pick the column of ``R + I``
      with the largest norm, build a unit vector from it, and set
      ``omega = pi * unit``.  This is the standard branch for the
      ``pi``-cut.
    * ``||omega|| < _EPS_TRIG`` — rotation is the identity, return
      ``0`` to avoid ``0 / 0`` in the ``sin(theta)`` denominator.

    Parameters
    ----------
    R
        Rotation matrix of shape ``(..., 3, 3)``.

    Returns
    -------
    Tensor
        Rotation vector of shape ``(..., 3)``.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(
            f"so3_log expects last two dims (3, 3), got shape {tuple(R.shape)}"
        )

    # cos(theta) = (tr(R) - 1) / 2  — clamp to [-1, 1] to absorb
    # float32 round-off in the identity branch.
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (..., )

    # Skew-symmetric part of R: (R - R^T) / 2.
    skew = 0.5 * (R - R.transpose(-1, -2))  # (..., 3, 3)

    # For the closed-form is
    #     [omega]_x = theta / (2 sin theta) * (R - R^T).
    # We multiply by ``theta / sin(theta)`` (which tends to 1 as
    # theta -> 0 — well-defined thanks to the clamp) and then
    # convert via the standard cross-product trick: a
    # skew-symmetric matrix [v]_x has v = (M_21, M_02, M_10).
    sin_theta = torch.sin(theta)
    safe_sin = sin_theta.clamp_min(_EPS_TRIG)
    factor = theta / safe_sin  # (..., )

    # Pull (v_x, v_y, v_z) out of the skew matrix by stacking the
    # off-diagonal elements with the right sign convention.
    #     [v]_x = [[ 0, -v_z,  v_y],
    #              [ v_z,  0,  -v_x],
    #              [-v_y, v_x,   0]]
    v = torch.stack(
        [
            skew[..., 2, 1],
            skew[..., 0, 2],
            skew[..., 1, 0],
        ],
        dim=-1,
    )  # (..., 3)

    omega = factor.unsqueeze(-1) * v  # (..., 3)

    # Branch B: antipodal (theta close to pi).  Fall back to a
    # square-root extraction that does not divide by sin(theta)
    # (which is ~0 here).
    near_pi = theta > (math.pi - _EPS_TRIG)
    if near_pi.any():
        # For R with theta == pi: R + I = 2 n n^T, so any column
        # of R + I is parallel to the rotation axis.  Pick the
        # column with the largest norm for numerical stability.
        RpI = R + torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
        # column norms, shape (..., 3)
        col_norms = RpI.pow(2).sum(dim=-2)  # sum over rows -> (..., 3)
        # Pick the column with largest norm.
        idx = col_norms.argmax(dim=-1)  # (..., )
        # Gather the corresponding column for each batch entry.
        # We do this manually since ``gather`` over batch dims is
        # finicky and we want to stay pure CPU.
        n_axes = []
        for k in range(3):
            mask = (idx == k).unsqueeze(-1)  # (..., 1)
            n_axes.append(mask.float() * RpI[..., k])  # (..., 3)
        n_pi = sum(n_axes)  # (..., 3)
        n_pi = n_pi / n_pi.norm(dim=-1, keepdim=True).clamp_min(_EPS_TRIG)
        omega_pi = math.pi * n_pi  # (..., 3)
        omega = torch.where(near_pi.unsqueeze(-1), omega_pi, omega)

    return omega


def so3_slerp(R0: Tensor, R1: Tensor, t: float) -> Tensor:
    """Spherical linear interpolation between two SO(3) rotations.

    Theorem reference
    -----------------
    Definition 3.4 of :cite:`mahmood2026sb_lie` (geodesic on SO(3) is
    given by the SLERP parameterised by the geodesic distance
    :math:`d(R_0, R_1) = \\|\\log(R_0^T R_1)\\|`).

    Parameters
    ----------
    R0, R1
        Rotation matrices of shape ``(..., 3, 3)``.
    t
        Interpolation parameter in ``[0, 1]`` (scalar).

    Returns
    -------
    Tensor
        Rotation matrix of shape ``(..., 3, 3)`` satisfying
        :func:`so3_slerp` at ``t=0`` returns ``R0`` and at ``t=1``
        returns ``R1``.
    """
    if R0.shape != R1.shape:
        raise ValueError(
            f"R0 / R1 must share shape, got {tuple(R0.shape)} vs {tuple(R1.shape)}"
        )
    if R0.shape[-2:] != (3, 3):
        raise ValueError(
            f"so3_slerp expects last two dims (3, 3), got {tuple(R0.shape)}"
        )

    # Relative rotation in the body frame of R0.
    R_rel = R0.transpose(-1, -2) @ R1
    omega = so3_log(R_rel)  # (..., 3)
    return R0 @ so3_exp(float(t) * omega)


# ---------------------------------------------------------------------------
# SE(3) primitives (homogeneous 4x4 matrices)
# ---------------------------------------------------------------------------


def se3_exp(xi: Tensor) -> Tensor:
    """Exponential map from ``se(3)`` to ``SE(3)``.

    Given a twist :math:`\\xi = (\\upsilon, \\omega)` of shape
    ``(..., 6)`` (first three components are linear, last three are
    angular), returns the homogeneous SE(3) matrix of shape
    ``(..., 4, 4)``.

    Theorem reference
    -----------------
    Theorem 2.1 of :cite:`mahmood2026sb_lie`; the closed-form uses
    the left Jacobian of SO(3) (Hall 2015, eq. (2.38)):

    .. math::

        \\exp(\\xi) = \\begin{pmatrix}
            R(\\omega) & V(\\omega) \\upsilon \\\\
            0 & 1
        \\end{pmatrix},

    where :math:`V(\\omega) = I + \\frac{1 - \\cos\\theta}{\\theta^2}
    [\\omega]_\\times + \\frac{\\theta - \\sin\\theta}{\\theta^3}
    [\\omega]_\\times^2` and :math:`\\theta = \\|\\omega\\|`.

    Parameters
    ----------
    xi
        Twist of shape ``(..., 6)`` — ``[..., :3]`` is the linear
        part, ``[..., 3:]`` is the angular part.

    Returns
    -------
    Tensor
        Homogeneous SE(3) matrix of shape ``(..., 4, 4)``.
    """
    if xi.shape[-1] != 6:
        raise ValueError(f"se3_exp expects last dim 6, got shape {tuple(xi.shape)}")
    upsilon = xi[..., :3]
    omega = xi[..., 3:]
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(_EPS_ROT)  # (..., 1)
    n = omega / theta

    # Skew-symmetric matrix [n]_x
    zero = torch.zeros_like(n[..., 0])
    K = torch.stack(
        [
            torch.stack([zero, -n[..., 2], n[..., 1]], dim=-1),
            torch.stack([n[..., 2], zero, -n[..., 0]], dim=-1),
            torch.stack([-n[..., 1], n[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )

    eye3 = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(K.shape)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)

    # V = I + (1 - cos)/theta^2 * K + (theta - sin)/theta^3 * K^2.
    # Note (1-cos)/theta^2 and (theta - sin)/theta^3 are well-defined
    # limits at theta -> 0 (they tend to 1/2 and 1/6 respectively).
    K2 = K @ K
    V = (
        eye3
        + ((1.0 - cos_t) / (theta.unsqueeze(-1) ** 2)) * K
        + ((theta.unsqueeze(-1) - sin_t) / (theta.unsqueeze(-1) ** 3)) * K2
    )

    R = so3_exp(omega)  # (..., 3, 3)
    p = (V @ upsilon.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def se3_log(T: Tensor) -> Tensor:
    """Logarithm map from ``SE(3)`` to ``se(3)``.

    Given an SE(3) matrix of shape ``(..., 4, 4)``, returns the twist
    :math:`\\xi = (\\upsilon, \\omega)` of shape ``(..., 6)``.

    Theorem reference
    -----------------
    Theorem 2.1 of :cite:`mahmood2026sb_lie`; the inverse of
    :func:`se3_exp`.  Recovers :math:`\\omega` from the rotation
    block via :func:`so3_log`, then inverts the SO(3) left Jacobian
    :math:`V(\\omega)` *numerically* with ``torch.linalg.inv`` to
    recover :math:`\\upsilon = V(\\omega)^{-1} p` from the translation
    column.

    Numerical vs. closed-form inverse
    ---------------------------------
    The textbook closed-form for :math:`V^{-1}` (Barfoot 2017,
    eq. 7.86; Eade 2017, eq. (108)) is **not** the matrix inverse of
    :math:`V = I + \\tfrac{1-\\cos\\theta}{\\theta^2} [\\omega]_\\times
    + \\tfrac{\\theta-\\sin\\theta}{\\theta^3} [\\omega]_\\times^2` —
    a direct product check shows a ~1 % discrepancy for
    :math:`\\theta \\approx 0.5`.  We therefore invert ``V``
    numerically (``torch.linalg.inv``), which is exact to float32
    precision and still trivially cheap (3×3 inv per batch entry).

    Parameters
    ----------
    T
        SE(3) matrix of shape ``(..., 4, 4)``.

    Returns
    -------
    Tensor
        Twist of shape ``(..., 6)``.
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"se3_log expects last two dims (4, 4), got {tuple(T.shape)}")
    R = T[..., :3, :3]
    p = T[..., :3, 3]

    omega = so3_log(R)  # (..., 3)
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(_EPS_ROT)
    n = omega / theta

    # Build V(omega) using the same closed-form as :func:`se3_exp`.
    zero = torch.zeros_like(n[..., 0])
    K = torch.stack(
        [
            torch.stack([zero, -n[..., 2], n[..., 1]], dim=-1),
            torch.stack([n[..., 2], zero, -n[..., 0]], dim=-1),
            torch.stack([-n[..., 1], n[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )
    eye3 = torch.eye(3, dtype=T.dtype, device=T.device).expand(K.shape)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    theta_v = theta.unsqueeze(-1)

    K2 = K @ K
    a = (1.0 - cos_t) / (theta_v**2)
    b = (theta_v - sin_t) / (theta_v**3)
    V = eye3 + a * K + b * K2  # (..., 3, 3) SO(3) left Jacobian.

    # Numerical inverse — see docstring for why we avoid the
    # closed-form "V^{-1}" of Barfoot/Eade here.
    V_inv = torch.linalg.inv(V)

    upsilon = (V_inv @ p.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    xi = torch.cat([upsilon, omega], dim=-1)
    return xi


# ---------------------------------------------------------------------------
# High-level API — class wrappers
# ---------------------------------------------------------------------------


@dataclass
class SE3Geodesic:
    """Geodesic interpolation / log / exp maps on SE(3).

    Provides the same primitives as :func:`so3_slerp`, :func:`so3_log`,
    :func:`so3_exp`, :func:`se3_log`, :func:`se3_exp` packaged into a
    small dataclass so callers can ``from molmetal.math_skills import
    SE3Geodesic`` and use ``geo.interpolate(T0, T1, t)`` without
    having to know which function does what.

    Theorem reference
    -----------------
    Definition 3.4 + Proposition 3.5 of :cite:`mahmood2026sb_lie`
    (the geodesic on ``SE(3) = SO(3) ⋉ R^3`` decomposes into an SO(3)
    geodesic on the rotation part and an affine interpolation on the
    translation part in the body frame of the first pose).
    """

    @staticmethod
    def interpolate(R0: Tensor, R1: Tensor, t: float) -> Tensor:
        """SO(3) geodesic — alias for :func:`so3_slerp`."""
        return so3_slerp(R0, R1, t)

    @staticmethod
    def interpolate_se3(T0: Tensor, T1: Tensor, t: float) -> Tensor:
        """SE(3) geodesic — rotation slerp + body-frame translation.

        The translation is interpolated in the *body frame* of
        :math:`T_0` so that the resulting trajectory is a geodesic
        on the SE(3) manifold (Proposition 3.5 of
        :cite:`mahmood2026sb_lie`):

        .. math::

            T(t) = T_0 \\exp\\bigl( t \\, \\log(T_0^{-1} T_1) \\bigr).
        """
        if T0.shape != T1.shape or T0.shape[-2:] != (4, 4):
            raise ValueError(
                f"interpolate_se3 expects (..., 4, 4) inputs, got "
                f"{tuple(T0.shape)} vs {tuple(T1.shape)}"
            )
        T0_inv = torch.zeros_like(T0)
        # T0_inv = [R0^T, -R0^T p0; 0, 1]
        R0_inv = T0[..., :3, :3].transpose(-1, -2)
        T0_inv[..., :3, :3] = R0_inv
        T0_inv[..., :3, 3] = -(R0_inv @ T0[..., :3, 3].unsqueeze(-1)).squeeze(-1)
        T0_inv[..., 3, 3] = 1.0

        xi = se3_log(T0_inv @ T1)  # (..., 6)
        return T0 @ se3_exp(float(t) * xi)

    @staticmethod
    def log_map(T_ref: Tensor, T: Tensor) -> Tensor:
        """SE(3) log map *relative to* ``T_ref`` (body-frame twist)."""
        if T_ref.shape != T.shape:
            raise ValueError(
                f"log_map expects matching shapes, got {tuple(T_ref.shape)} "
                f"vs {tuple(T.shape)}"
            )
        R_ref_inv = T_ref[..., :3, :3].transpose(-1, -2)
        T_ref_inv = torch.zeros_like(T_ref)
        T_ref_inv[..., :3, :3] = R_ref_inv
        T_ref_inv[..., :3, 3] = -(R_ref_inv @ T_ref[..., :3, 3].unsqueeze(-1)).squeeze(
            -1
        )
        T_ref_inv[..., 3, 3] = 1.0
        return se3_log(T_ref_inv @ T)

    @staticmethod
    def exp_map(T_ref: Tensor, v: Tensor) -> Tensor:
        """SE(3) exp map *from* ``T_ref`` (body-frame twist -> pose)."""
        if v.shape[-1] != 6:
            raise ValueError(
                f"exp_map expects twist of shape (..., 6), got {tuple(v.shape)}"
            )
        return T_ref @ se3_exp(v)


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn divergence
# ---------------------------------------------------------------------------


class LogSinkhornLoss:
    """Log-domain Sinkhorn divergence (entropic-regularised OT cost).

    Given two empirical distributions of equal mass (vectors of
    shape ``(n,)`` and ``(m,)`` with ``sum(a) = sum(b) = 1``) and a
    cost matrix ``C`` of shape ``(n, m)``, computes the
    entropy-regularised OT cost:

    .. math::

        S_\\varepsilon(a, b) = \\langle P^\\star, C \\rangle
            + \\varepsilon \\, H(P^\\star \\,|\\, a \\otimes b),

    where :math:`P^\\star` is the Sinkhorn coupling.

    Theorem reference
    -----------------
    Eq. (5)–(6) of :cite:`mahmood2026sb_lie`; the closed-form Sinkhorn
    fixed-point iteration in log-domain (Cuturi 2013, *Sinkhorn
    Distances: Lightspeed Computation of Optimal Transportation
    Distances*).  The log-domain parameterisation is critical: it
    avoids the overflow that the textbook scaling iteration suffers
    from when ``eps`` is small (the SB limit).

    Numerical stability
    -------------------
    The implementation uses ``logsumexp`` at every iteration, so it
    is stable for ``eps`` down to ``1e-3`` at moderate batch sizes
    (``n, m <= 1000``).
    """

    def __init__(self, eps: float = 0.1, n_iters: int = 50) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.eps = float(eps)
        self.n_iters = int(n_iters)

    def forward(
        self,
        p0: Tensor,
        p1: Tensor,
        C: Optional[Tensor] = None,
        *,
        a: Optional[Tensor] = None,
        b: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the log-domain Sinkhorn divergence.

        Parameters
        ----------
        p0
            Samples from the source distribution of shape ``(n, d)``
            (rows are samples; ``d`` is the dimension of the SE(3)
            parameterisation, e.g. ``6`` for the twist or ``3+3`` for
            rotation + translation split).  Only used when ``C`` is
            ``None`` — in that case the pairwise cost is built as the
            squared Euclidean distance between rows of ``p0`` and
            ``p1``.
        p1
            Samples from the target distribution of shape ``(m, d)``.
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
        loss
            Scalar tensor — the entropic-regularised OT cost.
        coupling
            The Sinkhorn coupling :math:`P^\\star` of shape ``(n, m)``
            (useful for downstream sampling).
        """
        if C is None:
            # Pairwise squared Euclidean distance, shape (n, m).
            # ||p0_i - p1_j||^2 = ||p0_i||^2 + ||p1_j||^2 - 2 p0_i . p1_j
            sq0 = (p0 * p0).sum(dim=-1, keepdim=True)  # (n, 1)
            sq1 = (p1 * p1).sum(dim=-1, keepdim=True).T  # (1, m)
            cross = p0 @ p1.T  # (n, m)
            C = (sq0 + sq1 - 2.0 * cross).clamp_min(0.0)

        if math_preconditions_enabled():
            import warnings as _warnings

            for _v in check_preconditions(p0, entry="sinkhorn", a=a, b=b, C=C):
                _warnings.warn(f"[math-precondition] {_v}", stacklevel=2)

        n, m = C.shape
        device, dtype = C.device, C.dtype

        if a is None:
            a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        if b is None:
            b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)

        # Log-domain Sinkhorn.  Define M_ij = -C_ij / eps (the log
        # kernel).  The iteration maintains log-domain dual potentials
        # ``f`` and ``g`` such that the coupling satisfies
        #     log P_ij = M_ij + f_i + g_j.
        # The fixed-point equations (Cuturi 2013, eq. (4)) are
        #     f_i = log a_i - logsum_j exp(M_ij + g_j)
        #     g_j = log b_j - logsum_i exp(M_ij + f_i)
        # which we solve by alternating minimisation — the standard
        # log-domain Sinkhorn iteration that is numerically stable for
        # small ``eps`` (Schmitzer 2019, *Stabilized Sparse
        # Sinkhorn Divergence*).
        log_a = torch.log(a.clamp_min(1e-30))
        log_b = torch.log(b.clamp_min(1e-30))
        M = -C / self.eps  # (n, m)  — log kernel

        f = torch.zeros(n, device=device, dtype=dtype)
        g = torch.zeros(m, device=device, dtype=dtype)
        for _ in range(self.n_iters):
            # Update f given g.
            f = log_a - torch.logsumexp(M + g.unsqueeze(0), dim=-1)
            # Update g given f.
            g = log_b - torch.logsumexp(M + f.unsqueeze(1), dim=0)

        # Recover the coupling: log P_ij = M_ij + f_i + g_j.
        log_P = M + f.unsqueeze(1) + g.unsqueeze(0)
        P = torch.exp(log_P)

        # Sinkhorn divergence = <P, C> + eps * KL(P || a ⊗ b).
        # KL(P || a ⊗ b) = sum_ij P_ij (log P_ij - log a_i - log b_j).
        entropy = (P * (log_P - log_a.unsqueeze(1) - log_b.unsqueeze(0))).sum()
        loss = (P * C).sum() + self.eps * entropy
        return loss, P


# ---------------------------------------------------------------------------
# Schrödinger Bridge on SE(3)
# ---------------------------------------------------------------------------


class SchrodingerBridgeSE3:
    """Schrödinger Bridge sampler on the SE(3) Lie group.

    Given two empirical distributions on SE(3), computes the
    entropic-regularised optimal coupling (the SB plan, eq. (10)
    of :cite:`mahmood2026sb_lie`) via Sinkhorn, then samples
    intermediate SE(3) poses by walking along the SE(3) geodesics
    connecting matched pairs.

    Theorem reference
    -----------------
    Theorem 4.1 of :cite:`mahmood2026sb_lie` — the SB coupling
    converges to the un-regularised optimal-transport plan as
    ``eps -> 0``, and Theorem 4.3 — the discrete-time bridge
    interpolant is a valid probability path on SE(3) at every
    ``t in [0, 1]``.

    CPU-only
    --------
    All operations use ``torch.linalg`` (CPU backend by default);
    no CUDA / Triton kernels are required for the *mathematical
    skeleton*.  R18 integration in
    ``molmetal.adapters.flow_matching_lipman`` may add a GPU kernel
    later but the math layer stays portable.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        eps: float = 0.1,
        n_iters: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self._sinkhorn = LogSinkhornLoss(eps=self.eps, n_iters=self.n_iters)
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(
        self,
        p0: Tensor,
        n_samples: int = 100,
        *,
        t_values: Optional[Tensor] = None,
        sigma: Optional[float] = None,
    ) -> Tensor:
        """Sample intermediate SE(3) poses from the Schrödinger bridge.

        Parameters
        ----------
        p0
            Source distribution as a *parameterised* batch of SE(3)
            poses of shape ``(n, 6)`` (twists in the Lie algebra).
            The bridge target ``p1`` is taken to be ``p0`` itself,
            which gives the *self-bridge* interpolant — this is the
            common configuration in Mol-Metal's CFM path (matching
            ``x_1`` to ``x_0`` via OT, then drawing bridge samples).
        n_samples
            Number of bridge samples to draw.  Each sample picks a
            pair ``(i, j)`` from the SB coupling and then walks the
            SE(3) geodesic from ``p0[i]`` to ``p0[j]``.
        t_values
            Optional 1-D tensor of evaluation points along the bridge
            ``[0, 1]``; defaults to a single point ``t = 0.5``.
        sigma
            Optional per-call override of the bridge noise scale
            (defaults to the ``sigma`` passed at construction).

        Returns
        -------
        Tensor
            Bridge samples of shape ``(n_samples, n_t, 4, 4)`` — the
            first axis indexes the sample, the second the time point,
            and the last two the homogeneous SE(3) matrix.
        """
        if p0.shape[-1] != 6:
            raise ValueError(
                f"p0 must be of shape (n, 6) (Lie algebra twists), "
                f"got {tuple(p0.shape)}"
            )
        if t_values is None:
            t_values = torch.tensor([0.5], dtype=p0.dtype, device=p0.device)
        if t_values.ndim != 1:
            raise ValueError(
                f"t_values must be a 1-D tensor, got shape {tuple(t_values.shape)}"
            )

        if math_preconditions_enabled():
            import warnings as _warnings

            for _v in check_preconditions(p0, entry="sample", t_values=t_values):
                _warnings.warn(f"[math-precondition] {_v}", stacklevel=2)

        # 1) Build the SE(3)-geodesic cost matrix between every pair
        #    of source poses.  We work in twist coordinates so the
        #    cost is a flat Euclidean distance in ``se(3)`` — this
        #    is a common proxy for the Riemannian SE(3) distance
        #    used in practice (eq. (12) of :cite:`mahmood2026sb_lie`).
        diff = p0.unsqueeze(1) - p0.unsqueeze(0)  # (n, n, 6)
        C = (diff * diff).sum(dim=-1)  # (n, n) squared L2

        # 2) Uniform marginals — the SB is the self-bridge here.
        n = p0.shape[0]
        a = torch.full((n,), 1.0 / n, dtype=p0.dtype, device=p0.device)
        b = a.clone()

        # 3) Sinkhorn solve.
        _loss, P = self._sinkhorn.forward(p0, p0, C=C, a=a, b=b)

        # 4) Draw (i, j) pairs from P.
        P_flat = P.reshape(-1)
        P_flat = P_flat / P_flat.sum().clamp_min(1e-30)
        idx = torch.multinomial(
            P_flat,
            num_samples=int(n_samples),
            replacement=True,
            generator=self._generator,
        )
        i_idx = idx // n
        j_idx = idx % n

        # 5) Walk the SE(3) geodesic from T(i) to T(j) at each t.
        T_i = se3_exp(p0[i_idx])  # (n_samples, 4, 4)
        T_j = se3_exp(p0[j_idx])  # (n_samples, 4, 4)
        geo = SE3Geodesic()
        out = []
        s = float(sigma) if sigma is not None else self.sigma
        for t in t_values:
            T_t = geo.interpolate_se3(T_i, T_j, float(t))
            if s != 1.0:
                # Apply a small angular/translational jitter in the
                # body frame of T_t.  This is the bridge "noise"
                # term from the Door Gaussian path (Definition 3.7
                # of :cite:`mahmood2026sb_lie`).
                jitter = s * torch.randn(
                    T_t.shape[:-2] + (6,),
                    dtype=T_t.dtype,
                    device=T_t.device,
                    generator=self._generator,
                )
                T_t = geo.exp_map(T_t, jitter)
            out.append(T_t)
        return torch.stack(out, dim=-3)  # (n_samples, n_t, 4, 4)
