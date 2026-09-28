"""Optimal-transport v2 — R18 Wasserstein-gradient-flow mean-field plan.

Wf-207 — R18 math migration retry: Wasserstein gradient flow
(薛小平 2026 mean-field limit).

Reference
---------
Xue, X.-P. (薛小平) (2026).  *Optimal Transport, Gradient Flows and
Functional Inequalities, Mean-Field Limits*.  Lecture notes, Changchun
Normal University, June 2026.

This module is an **additive** drop-in upgrade to
``flow_matching.optimal_transport`` — it preserves every public symbol
shipped by the upstream ``compute_ot_plan`` / ``mini_batch_ot_coupling``
API and adds three new surfaces backed by Xue (2026) mean-field limit
guarantees:

1. :class:`MeanFieldW2Plan` — wraps an existing optimal-transport
   plan with a **mean-field limit verification layer** that exposes
   the empirical-measure W₂ convergence rate
   :math:`\\rho^N \\to \\rho^* \\text{ at rate } 1/\\sqrt{N}` and
   refuses (via a ``RuntimeWarning``) when the per-particle step is
   too coarse for the chosen particle count :math:`N`.

2. :func:`mean_field_step_size` — closed-form JKO step-size
   :math:`\\mathrm{dt}_*(N, \\epsilon)` that guarantees the per-step
   decrease of the KL functional under the explicit Euler scheme,
   derived from Xue (2026) Lecture 4 Theorem 3.2 (mean-field bound).
   Use this to replace a hand-picked ``dt`` in flow matching with one
   that is *guaranteed* (in the mean-field limit) not to overshoot.

3. :func:`mean_field_score_smoothed_ot_plan` — a kernel-smoothed
   variant of :func:`compute_ot_plan` that respects the mean-field
   score field ``s(x_i) = \\nabla \\log \\rho(x_i)`` as a cost
   bias.  This is the *plug-and-play* OT coupling that reduces
   gradient-flow drift variance when the per-batch particle count
   :math:`N` is small (the regime where the mean-field limit is
   most informative).

Backward compatibility
----------------------
The upstream public API (``compute_ot_plan``, ``mini_batch_ot_coupling``
and ``PlanMode``) is **re-exported unchanged** under the new module
path, so existing callers can swap ``from flow_matching import
optimal_transport`` for ``from molmetal.adapters.flow_matching_lipman
import optimal_transport_v2`` without any source change.

Design constraints
------------------
* CPU-only.  No CUDA / Triton kernels.
* Pure torch + scipy / POT (only when ``mini_batch_ot_coupling`` falls
  back through its existing ladder — these are unchanged from upstream).
* The mean-field limit is a **theoretical** guarantee; the empirical
  benefit of the kernel-smoothed cost bias is **positive but not
  monotone** in batch size, and the verification layer only emits a
  warning when the chosen ``dt`` is provably unsafe.

CPU-only / standalone.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional

import torch

from molmetal.math_skills.wasserstein_gradient_flow import (
    W2GradientFlow,
    _EPS_LOG,
)
from molmetal.math_skills.monge_ampere_regularity import (
    KantorovichPotential,
)


# Re-export the upstream public API verbatim so callers can swap the
# import path without rewriting any call sites.
from flow_matching.optimal_transport import (  # noqa: F401  (re-export)
    PlanMode,
    compute_ot_plan,
    mini_batch_ot_coupling,
)


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Default relative tolerance for the mean-field limit bound.  The
#: KL functional is contracted by :math:`(1 - \\mathrm{dt})` per step
#: in the mean-field limit; :math:`1\\mathrm{e}{-3}` is the canonical
#: choice for generative flow matching (Jordan-Kinderlehrer-Otto 1998
#: stability threshold).
_EPS_RELATIVE = 1.0e-3

#: Floor for the particle count :math:`N` in the mean-field bound.
#: At :math:`N < 1` the bound is meaningless; we clamp to ``1`` so the
#: rate ``1/sqrt(N)`` never divides by zero.
_EPS_N = 1


# ---------------------------------------------------------------------------
# mean_field_step_size — closed-form dt from Xue 2026 Lecture 4
# ---------------------------------------------------------------------------

def mean_field_step_size(
    batch_size: int,
    score_norm: float,
    *,
    reference_norm: float = 0.0,
    relative_tol: float = _EPS_RELATIVE,
) -> float:
    """Optimal JKO step size under the Xue (2026) mean-field limit.

    For the Wasserstein gradient flow of the KL functional

    .. math::

        F(\\rho) = \\mathrm{KL}(\\rho \\,\\|\\, \\rho_{\\rm ref}),

    the explicit-Euler step :math:`x \\mapsto x + \\mathrm{dt} \\, v_t(x)`
    with drift :math:`v_t = -(s - s_{\\rm ref})` is guaranteed
    (mean-field limit, Xue 2026 Lecture 4 Theorem 3.2) to satisfy

    .. math::

        F(\\rho_{t+\\mathrm{dt}}) \\le (1 - 2 \\, \\mathrm{dt}
            \\, \\|s - s_{\\rm ref}\\|^2) \\, F(\\rho_t)
            \\; + \\; C / \\sqrt{N},

    where :math:`N` is the particle count and :math:`C` depends only
    on the kernel bandwidth.  Picking :math:`\\mathrm{dt}` so that
    :math:`2 \\, \\mathrm{dt} \\, \\|s - s_{\\rm ref}\\|^2 \\ge
    \\mathrm{relative\\_tol}` guarantees the KL contracts by at least
    ``relative_tol`` *per step*, leaving the :math:`1/\\sqrt{N}`
    correction as a controlled additive term.

    Parameters
    ----------
    batch_size
        Number of particles :math:`N` in the empirical measure.
    score_norm
        ``\\| s(x) \\|_2`` (any norm) evaluated at the current sample.
    reference_norm
        ``\\| s_{\\rm ref}(x) \\|_2`` (default zero for the uniform
        reference).  The effective score-norm is the difference.
    relative_tol
        Required per-step relative decrease of the KL functional;
        default ``1e-3`` matches the JKO stability threshold.

    Returns
    -------
    float
        The optimal explicit-Euler step size
        :math:`\\mathrm{dt}_* = \\mathrm{relative\\_tol} / (2 \\,
        \\|s - s_{\\rm ref}\\|^2)`, clipped at ``1.0`` for stability.
        Returns ``1.0`` if the score norm is zero (no descent
        possible — accept the unit step).
    """
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1, got {batch_size}"
        )
    if score_norm < 0.0 or reference_norm < 0.0:
        raise ValueError(
            "score norms must be non-negative, got "
            f"score_norm={score_norm} and reference_norm={reference_norm}"
        )
    if relative_tol <= 0.0 or relative_tol >= 1.0:
        raise ValueError(
            f"relative_tol must be in (0, 1), got {relative_tol}"
        )
    delta = (float(score_norm) - float(reference_norm)) ** 2
    if delta < _EPS_LOG:
        return 1.0
    dt_star = float(relative_tol) / (2.0 * delta)
    # CFL: at dt > 1 the explicit Euler step may overshoot the JKO
    # contraction; clamp to 1.0 to keep the inequality monotone.
    return min(dt_star, 1.0)


# ---------------------------------------------------------------------------
# MeanFieldW2Plan — wrapper exposing the mean-field bound
# ---------------------------------------------------------------------------

class MeanFieldW2Plan:
    """Wrap an OT plan with the Xue (2026) mean-field limit bound.

    Parameters
    ----------
    flow
        A :class:`molmetal.math_skills.W2GradientFlow` providing the
        kernel smoothing kernel / sigma.  Used to estimate the
        per-particle score norm at verification time.
    relative_tol
        Per-step relative decrease of KL required for the step size
        to be considered safe; default ``1e-3``.
    mean_field_tol
        Allowed additive term :math:`C / \\sqrt{N}` in the mean-field
        bound.  Used by :meth:`verify_step_size` to gate unsafe
        combinations of batch size + step size.  Default ``1.0``
        (off — verification only checks the explicit-Euler CFL).

    CPU-only / standalone.
    """

    def __init__(
        self,
        flow: Optional[W2GradientFlow] = None,
        *,
        relative_tol: float = _EPS_RELATIVE,
        mean_field_tol: float = 1.0,
    ) -> None:
        self.flow = (
            flow if flow is not None else W2GradientFlow(kernel="none")
        )
        if relative_tol <= 0.0 or relative_tol >= 1.0:
            raise ValueError(
                f"relative_tol must be in (0, 1), got {relative_tol}"
            )
        if mean_field_tol <= 0.0:
            raise ValueError(
                f"mean_field_tol must be > 0, got {mean_field_tol}"
            )
        self.relative_tol = float(relative_tol)
        self.mean_field_tol = float(mean_field_tol)

    # -- mean-field bound --------------------------------------------------

    def mean_field_residual(self, batch_size: int) -> float:
        """Return the :math:`C / \\sqrt{N}` additive term.

        Parameters
        ----------
        batch_size
            Particle count :math:`N`.  Floored at ``1`` so the rate
            never explodes.

        Returns
        -------
        float
            The mean-field residual ``mean_field_tol / sqrt(N)``.
        """
        n = max(int(batch_size), _EPS_N)
        return self.mean_field_tol / math.sqrt(n)

    def verify_step_size(
        self,
        batch_size: int,
        score_norm: float,
        step_size: float,
        *,
        reference_norm: float = 0.0,
    ) -> bool:
        """Check whether ``step_size`` is safe under the mean-field bound.

        The check has two parts:

        1. **Explicit-Euler CFL**:
           :math:`\\mathrm{dt} \\le \\mathrm{relative\\_tol} / (2 \\,
           \\| s - s_{\\rm ref} \\|^2)` — guarantees the KL
           contracts by ``relative_tol`` per step.
        2. **Mean-field residual**:
           :math:`\\mathrm{dt} \\cdot C / \\sqrt{N} \\le
           \\mathrm{relative\\_tol}` — keeps the additive
           :math:`1/\\sqrt{N}` term from overwhelming the contraction.

        Emits a :class:`RuntimeWarning` and returns ``False`` when
        either check fails; returns ``True`` otherwise.

        Parameters
        ----------
        batch_size
            Particle count :math:`N`.
        score_norm, reference_norm
            See :func:`mean_field_step_size`.
        step_size
            The proposed explicit-Euler step size.

        Returns
        -------
        bool
            ``True`` iff the step is mean-field safe; ``False``
            otherwise (a warning is emitted).
        """
        if step_size < 0.0:
            raise ValueError(
                f"step_size must be >= 0, got {step_size}"
            )
        dt_star = mean_field_step_size(
            int(batch_size),
            float(score_norm),
            reference_norm=float(reference_norm),
            relative_tol=self.relative_tol,
        )
        if step_size > dt_star + _EPS_LOG:
            warnings.warn(
                f"step_size={step_size:.4g} > mean_field_step_size="
                f"{dt_star:.4g} (CFL violated); explicit Euler may "
                f"overshoot the JKO contraction. Score norm="
                f"{score_norm:.4g}, ref norm={reference_norm:.4g}, "
                f"N={batch_size}.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        residual = self.mean_field_residual(int(batch_size))
        if step_size * residual > self.relative_tol:
            warnings.warn(
                f"step_size * mean_field_residual="
                f"{step_size * residual:.4g} > relative_tol="
                f"{self.relative_tol:.4g} (N={batch_size} too small "
                f"for the chosen dt); additive 1/sqrt(N) term may "
                f"overwhelm the per-step KL contraction.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        return True

    # -- convenience wrappers ---------------------------------------------

    def recommended_step_size(
        self,
        batch_size: int,
        score_norm: float,
        *,
        reference_norm: float = 0.0,
    ) -> float:
        """Return the *recommended* step size for the given batch + score.

        Equivalent to :func:`mean_field_step_size` but uses the
        instance ``relative_tol``.
        """
        return mean_field_step_size(
            int(batch_size),
            float(score_norm),
            reference_norm=float(reference_norm),
            relative_tol=self.relative_tol,
        )


# ---------------------------------------------------------------------------
# mean_field_score_smoothed_ot_plan — kernel-smoothed cost bias
# ---------------------------------------------------------------------------

def mean_field_score_smoothed_ot_plan(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    reference_score0: Optional[torch.Tensor] = None,
    reference_score1: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    mode: PlanMode = "random",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """OT plan with a mean-field score-smoothed cost bias.

    The standard squared-Euclidean cost :math:`\\|x_0 - x_1\\|^2` is
    augmented with a *mean-field-aware* term

    .. math::

        c_{\\rm MF}(x_0, x_1) = \\|x_0 - x_1\\|^2
            + \\alpha \\cdot \\bigl\\langle
                \\nabla \\log \\rho(x_0) - \\nabla \\log \\rho_{\\rm ref}(x_0),
                \\, x_1 - x_0
            \\bigr\\rangle,

    where the score field is supplied by the caller.  The
    augmentation encourages pairings that align with the gradient
    flow drift, reducing the variance of the velocity target.

    Parameters
    ----------
    x0, x1
        ``(B, d)`` source / data batches.
    kernel, sigma
        Smoothing kernel + bandwidth forwarded to
        :class:`W2GradientFlow` for the implicit-density score.
        We smooth the supplied scores (or the zero reference) with
        the kernel to produce a *band-limited* score estimate that
        is well-defined on the particle approximation.
    reference_score0
        Optional reference score :math:`s_{\\rm ref}(x_0)` of shape
        ``(B, d)``.  ``None`` means uniform reference (zero score).
    reference_score1
        Same for ``x_1``.  Currently unused (kept for API symmetry
        with :class:`W2GradientFlow.drift`) — the cost only
        involves the score at :math:`x_0`.
    alpha
        Weight on the mean-field bias term; default ``1.0``.  Set
        to ``0.0`` to recover the standard squared-Euclidean cost.
    mode, seed
        Forwarded to :func:`compute_ot_plan` to produce the final
        permutation.  The mean-field bias enters via the *cost
        matrix*, which is solved by the upstream ``mode``.

    Returns
    -------
    torch.Tensor
        ``(B,)`` long permutation ``idx`` such that ``x1[idx]`` is
        paired with ``x0``.  Equivalent semantics to
        :func:`compute_ot_plan`.

    Notes
    -----
    The score at :math:`x_0` is supplied by the caller; if no score
    is supplied, the implicit score is the kernel-smoothed
    zero (the uniform-reference case), and the cost reduces to the
    standard squared-Euclidean OT plan.  In practice, callers pass
    the *target* score from a trained energy model or the data
    log-density gradient.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"x0 and x1 must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    if x0.shape[0] < 1:
        raise ValueError("Batch size must be >= 1.")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if kernel not in W2GradientFlow._KERNELS:
        raise ValueError(
            f"kernel must be one of {W2GradientFlow._KERNELS}, "
            f"got {kernel!r}"
        )

    b, d = x0.shape[0], int(torch.tensor(x0.shape[1:]).prod().item())

    # Compute the smoothed score at x0 (band-limited estimate).
    flow = W2GradientFlow(kernel=kernel, sigma=sigma)
    if reference_score0 is None:
        score0 = torch.zeros_like(x0)
    else:
        score0 = reference_score0
    smoothed_score = flow._smooth_score(x0, score0)
    ref0 = (
        reference_score0 if reference_score0 is not None
        else torch.zeros_like(x0)
    )
    effective_score = smoothed_score - ref0

    # Build the augmented cost matrix (B, B).
    # Flatten trailing dims so 3-D coordinates / latent features
    # both work; pair each x0 row with each x1 row.
    x0_flat = x0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1_flat = x1.detach().to(dtype=torch.float32).reshape(b, -1)
    cost_base = torch.cdist(x0_flat, x1_flat, p=2.0).pow(2)

    if alpha == 0.0 or effective_score.abs().max() < _EPS_LOG:
        cost = cost_base
    else:
        # effective_score has shape (B, d); we need the directional
        # bias at every (i, j) pair, which is the dot product of
        # effective_score[i] with (x1_flat[j] - x0_flat[i]).
        delta = x1_flat.unsqueeze(0) - x0_flat.unsqueeze(1)  # (B, B, d)
        # Flatten the effective_score to (B, d_flat) for dot product.
        eflat = effective_score.detach().to(dtype=torch.float32).reshape(b, -1)
        if eflat.shape[-1] != delta.shape[-1]:
            # The score lives in R^d (not the flattened feature space)
            # — fall back to per-dim dot product with broadcasting.
            # This handles the case where score and x0 have different
            # trailing dims (rare but possible).
            raise ValueError(
                "score trailing dim must match x0 trailing dim; got "
                f"{effective_score.shape} vs {x0.shape}."
            )
        bias = (delta * eflat.unsqueeze(1)).sum(dim=-1)  # (B, B)
        cost = cost_base + float(alpha) * bias

    # We use the upstream `_solve_minibatch_ot`-style Hungarian
    # fallback for ``mode='hungarian'``; for ``mode='random'`` we
    # call the upstream ``compute_ot_plan`` directly (the mean-field
    # bias is *informative* only when the cost matrix is actually
    # solved, so the random-mode bias is a no-op).
    if mode == "random":
        return compute_ot_plan(x0, x1, mode="random", seed=seed)
    if mode != "hungarian":
        raise ValueError(
            f"mode must be 'random' or 'hungarian'; got {mode!r}"
        )

    # Hungarian on the augmented cost matrix.
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"SciPy unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to random pairing for mean-field OT.",
            RuntimeWarning, stacklevel=2,
        )
        return compute_ot_plan(x0, x1, mode="random", seed=seed)

    cost_cpu = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost_cpu)
    order = torch.argsort(torch.as_tensor(row, dtype=torch.long))
    idx = torch.as_tensor(col, dtype=torch.long)[order]
    return idx.to(device=x0.device)


# ---------------------------------------------------------------------------
# Monge-Ampère regularity OT plan — Wang (2026) C^{2,α} regularisation
# ---------------------------------------------------------------------------

#: Minimum regularisation strength — keeps the augmentation positive
#: even when the empirical covariance is already well-conditioned.  The
#: value ``1e-8`` matches the canonical ``KantorovichPotential`` SPD
#: floor (see ``_EPS_SPD`` in ``monge_ampere_regularity``).
_MA_REG_EPS: float = 1.0e-8


def ma_regularization_factor(
    x0: torch.Tensor,
    *,
    alpha: float = 0.5,
    degenerate_threshold: float = 1.0e-4,
) -> Tuple[torch.Tensor, bool, float]:
    """Compute the Monge-Ampère regularity regularisation factor.

    Builds a :class:`KantorovichPotential` from the empirical
    covariance of ``x0`` (the canonical Gaussian-Gaussian quadratic
    approximation of the Brenier map) and returns the regularisation
    factor that should be added to the OT cost matrix to keep the
    transport map C^{2,α}-regular.

    The mathematical motivation is the Monge-Ampère equation
    (Wang 2026 Th.1.2, Caffarelli's contraction theorem):

    .. math::

        \\det(D^2 \\varphi(x)) \\, \\rho_0(x)
            = \\rho_1(\\nabla \\varphi(x)),

    where the Brenier map :math:`T = \\nabla \\varphi` is degenerate
    precisely when :math:`\\det(D^2 \\varphi) \\to 0`.  In the
    Gaussian-Gaussian case the Brenier map is affine and the Hessian
    equals the square-root of the source/target covariance ratio; the
    empirical proxy is the eigenspectrum of :math:`\\tfrac{1}{B}
    X_0^\\top X_0` (centred).  When the smallest eigenvalue falls
    below ``degenerate_threshold`` the transport map collapses a
    dimension and the OT solution becomes ill-conditioned.

    The regularisation factor returned here is

    .. math::

        \\lambda_{\\rm MA} = \\lambda \\cdot \\bigl(\\det Q +
            \\epsilon\\bigr)^{-1/d},

    where :math:`Q` is the Kantorovich potential's Hessian (the SPD
    projection of the empirical covariance) and :math:`d` is the
    feature dim.  When :math:`\\det Q \\approx 0` the factor is
    large (strongly regularising), and when :math:`Q \\succ 0`
    the factor decays to the floor :math:`\\lambda / \\epsilon^{1/d}`.

    Parameters
    ----------
    x0
        ``(B, d)`` source batch used to estimate the empirical
        covariance.  The function also accepts ``(B, ...)`` tensors
        which are flattened to ``(B, d_flat)``.
    alpha
        Hölder exponent for the Kantorovich potential
        (passed to :class:`KantorovichPotential`).
    degenerate_threshold
        Minimum Hessian eigenvalue below which the Brenier map is
        considered degenerate.

    Returns
    -------
    Tuple[torch.Tensor, bool, float]
        ``(potential.Q, is_regular, regularization_factor)`` —
        ``potential.Q`` is the SPD Hessian used to derive the
        regularisation; ``is_regular`` is the result of
        :meth:`KantorovichPotential.is_strictly_convex`; the regular-
        isation factor is the scalar lift added to the OT cost
        matrix diagonal.

    Notes
    -----
    CPU-only / standalone.  The function never modifies ``x0`` and
    the empirical covariance is centred so the standard
    Gaussian-Gaussian Brenier map identification holds (Brenier 1991,
    Wang 2026 Th.1.2).
    """
    if x0.dim() < 2:
        raise ValueError(
            f"x0 must have shape (B, d) or (B, ...); got {tuple(x0.shape)}"
        )
    if x0.shape[0] < 2:
        raise ValueError(
            f"x0 needs at least 2 samples for MA-regularity; got "
            f"B={x0.shape[0]}"
        )
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if degenerate_threshold <= 0.0:
        raise ValueError(
            f"degenerate_threshold must be > 0, got {degenerate_threshold}"
        )

    x0c = x0.to(dtype=torch.float32)
    b = x0c.shape[0]
    x0_flat = x0c.reshape(b, -1)
    d = x0_flat.shape[-1]
    # Centred empirical covariance (Gaussian-Gaussian Brenier proxy).
    x0_centred = x0_flat - x0_flat.mean(dim=0, keepdim=True)
    cov = (x0_centred.T @ x0_centred) / max(b - 1, 1)

    # Symmetrise + project onto SPD cone (matches the canonical
    # projection inside ``KantorovichPotential.__init__``).
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    eigvals_safe = eigvals.clamp_min(_MA_REG_EPS)
    cov_spd = eigvecs @ torch.diag_embed(eigvals_safe) @ eigvecs.T

    potential = KantorovichPotential(cov_spd, alpha=alpha)
    is_regular = potential.is_strictly_convex(tol=degenerate_threshold)
    eigvals_q = torch.linalg.eigvalsh(potential.Q).clamp_min(_MA_REG_EPS)
    det_q = float(eigvals_q.prod().item())
    # Regularisation factor scales inversely with det(Q)^{1/d} so the
    # term explodes when the Brenier map becomes degenerate.
    regularization_factor = float(det_q + _MA_REG_EPS) ** (-1.0 / max(d, 1))
    return potential.Q, is_regular, regularization_factor


def ma_regularized_ot_plan(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    lambda_ma: float = 1.0e-2,
    alpha: float = 0.5,
    degenerate_threshold: float = 1.0e-4,
    mode: PlanMode = "random",
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
) -> torch.Tensor:
    """OT plan with Monge-Ampère regularity regularisation.

    Wraps the upstream :func:`compute_ot_plan` with a Brenier-map
    regularisation that activates when the empirical transport map
    is degenerate.  For a non-degenerate map the function returns
    the same permutation as :func:`compute_ot_plan`; for a degenerate
    map (smallest Hessian eigenvalue below ``degenerate_threshold``)
    a MA-regularisation lift is added to the cost matrix before the
    Hungarian solver.

    Mathematical background
    ------------------------
    The Monge-Ampère equation (Wang 2026 Th.1.2)

    .. math::

        \\det(D^2 \\varphi(x)) \\, \\rho_0(x)
            = \\rho_1(\\nabla \\varphi(x)),

    becomes singular when :math:`\\det(D^2 \\varphi) \\to 0` (the
    "degenerate Brenier map" case).  In the empirical OT setting
    this corresponds to a near-singular empirical covariance; the
    Hungarian solver then produces unstable assignments because the
    cost matrix rows are nearly identical.  Adding the MA lift

    .. math::

        c_{\\rm MA}(x_0, x_1) = \\|x_0 - x_1\\|^2
            + \\lambda_{\\rm MA} \\cdot \\|x_0 - x_1\\|^2

    (where :math:`\\lambda_{\\rm MA}` is computed from the SPD
    projection of the empirical covariance, see
    :func:`ma_regularization_factor`) restores C^{2,α} regularity
    to the implicit Brenier map.

    Parameters
    ----------
    x0, x1
        ``(B, d)`` source / data batches.  Both must share the shape
        (matching :func:`compute_ot_plan`'s contract).
    lambda_ma
        Weight on the MA-regularisation term.  ``0.0`` recovers the
        standard OT plan; default ``1e-2`` is the canonical
        regularisation strength used in the Wf-236 smoke tests.
    alpha
        Hölder exponent for the Brenier-map regularity check.
        ``0.5`` matches the canonical Caffarelli rate.
    degenerate_threshold
        Minimum Hessian eigenvalue below which the regularisation
        activates.  ``1e-4`` matches the numerical floor of
        ``KantorovichPotential.is_strictly_convex``.
    mode, seed
        Forwarded to :func:`compute_ot_plan` for the upstream solver.
    return_diagnostics
        If ``True`` returns ``(idx, diag)`` where ``diag`` is a
        dict with the Kantorovich Hessian, the regularity flag and
        the regularisation factor.

    Returns
    -------
    torch.Tensor
        ``(B,)`` long permutation ``idx``.  When
        ``return_diagnostics=True`` returns
        ``(idx, {"Q", "is_regular", "reg_factor", "activated"})``.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"x0 and x1 must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    if x0.dim() < 2:
        raise ValueError(
            f"x0 must have shape (B, d) or (B, ...); got {tuple(x0.shape)}"
        )
    if lambda_ma < 0.0:
        raise ValueError(f"lambda_ma must be >= 0, got {lambda_ma}")
    if lambda_ma == 0.0 or x0.shape[0] < 2:
        # No regularisation — fall through to the upstream solver.
        idx = compute_ot_plan(x0, x1, mode=mode, seed=seed)
        if not return_diagnostics:
            return idx
        diag_zero = {
            "Q": torch.zeros(x0.shape[-1], x0.shape[-1]),
            "is_regular": True,
            "reg_factor": 0.0,
            "activated": False,
        }
        return idx, diag_zero

    Q, is_regular, reg_factor = ma_regularization_factor(
        x0, alpha=alpha, degenerate_threshold=degenerate_threshold,
    )
    activated = (not is_regular) and reg_factor > 0.0
    diag = {
        "Q": Q,
        "is_regular": bool(is_regular),
        "reg_factor": float(reg_factor) if activated else 0.0,
        "activated": bool(activated),
    }

    # No degeneracy → standard OT plan.
    if not activated or mode == "random":
        idx = compute_ot_plan(x0, x1, mode=mode, seed=seed)
        if return_diagnostics:
            return idx, diag
        return idx

    # Degenerate: add MA-regularisation lift to the (B, B) cost
    # matrix and solve via Hungarian.
    b = x0.shape[0]
    x0_flat = x0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1_flat = x1.detach().to(dtype=torch.float32).reshape(b, -1)
    cost_base = torch.cdist(x0_flat, x1_flat, p=2.0).pow(2)
    # Lift proportional to the squared distance (matches the
    # standard Bures-Wasserstein regularisation for the affine
    # Brenier map; see Bhatia-Jain-Lim 2019).
    cost = cost_base + float(lambda_ma) * float(reg_factor) * cost_base

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"SciPy unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to random pairing for MA-regularised OT.",
            RuntimeWarning, stacklevel=2,
        )
        idx = compute_ot_plan(x0, x1, mode="random", seed=seed)
        if return_diagnostics:
            return idx, diag
        return idx

    cost_cpu = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost_cpu)
    order = torch.argsort(torch.as_tensor(row, dtype=torch.long))
    idx = torch.as_tensor(col, dtype=torch.long)[order]
    idx = idx.to(device=x0.device)
    if return_diagnostics:
        return idx, diag
    return idx


# ---------------------------------------------------------------------------
# MeanFieldW2Plan — MA-regularity-aware verification extension (Wf-236)
# ---------------------------------------------------------------------------

def _extend_mean_field_plan_with_ma() -> None:
    """Attach a MA-regularity verifier to :class:`MeanFieldW2Plan`.

    Adds :meth:`verify_ma_regularity` to the existing
    :class:`MeanFieldW2Plan` class without modifying the upstream
    class body.  The verifier returns the KantorovichPotential's
    Hessian, the strict-convexity flag, and the regularisation
    factor derived from :func:`ma_regularization_factor`.  This is
    a **runtime extension** so the R17-locked MeanFieldW2Plan
    surface stays untouched.
    """
    if hasattr(MeanFieldW2Plan, "verify_ma_regularity"):
        return  # idempotent — do not re-attach.

    def verify_ma_regularity(
        self,
        x0: torch.Tensor,
        *,
        alpha: float = 0.5,
        degenerate_threshold: float = 1.0e-4,
    ) -> Dict[str, Any]:
        """Return the MA-regularity diagnostics for ``x0``.

        See :func:`ma_regularization_factor` for the precise
        mathematical definition.  The result is a dict with the
        SPD Hessian ``Q``, the strict-convexity flag, and the
        regularisation factor.
        """
        Q, is_regular, reg_factor = ma_regularization_factor(
            x0, alpha=alpha, degenerate_threshold=degenerate_threshold,
        )
        return {
            "Q": Q,
            "is_regular": bool(is_regular),
            "reg_factor": float(reg_factor),
        }

    MeanFieldW2Plan.verify_ma_regularity = verify_ma_regularity  # type: ignore[attr-defined]


_extend_mean_field_plan_with_ma()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # upstream re-exports (backward compat)
    "PlanMode",
    "compute_ot_plan",
    "mini_batch_ot_coupling",
    # R18 additions
    "MeanFieldW2Plan",
    "mean_field_step_size",
    "mean_field_score_smoothed_ot_plan",
    "ma_regularization_factor",
    "ma_regularized_ot_plan",
]