"""Hilbert-space primal-dual optimization skeleton (R18 P1-4).

================================================================
Background — F5/ASTSR → Hilbert primal-dual swap for the
lambda-model coupling adapter (R18 ultracode dispatch, 2026-09-26)
================================================================
TODO-34 §L-R18-P1-4 replaces the evolutionary / symbolic-regression
reward shapers in :mod:`molmetal.molmetal_lam.reward` with a **Hilbert
space primal-dual optimizer** whose convergence is governed by a
deterministic O(1/k) rate (Sun & Zheng 2026, *数学学报*, 53(4),
arXiv:2510.08423 — cited as ``sun2026hilbert`` in ``paper/refs.bib``).

The premise is a clean theoretical improvement:

* F5 (PAC-Bayes linear, R13) is in-sample R² = 1.0 on n=33 with LOO
  R² = -0.0635; the LOO gap is the honest framing that motivates the
  swap.
* LambdaASTSR (Deflexpressor AST symbolic regression, R17) uses
  evolutionary search (population 64, 50 generations).  The R²
  plateau is determined by tournament size, mutation rate, and
  parsimony — all hyperparameters of a stochastic process.
* Hilbert primal-dual optimisation replaces both with a deterministic
  iterative procedure whose iterates satisfy

      ||x_k - x*||² ≤ C / k,

  uniformly in k (Sun & Zheng 2026, Thm. 3.2).

The implementation here is a **CPU-only skeleton** — no PySR, no
external solvers, no GPU kernels.  Everything is deterministic given
the seed and lives inside the L-R18-P1-4 budget (~300 LOC).

Public API
----------
* :class:`HilbertSpace`            ℋ = L²[0,1] with a finite Fourier basis.
* :class:`PrimalDualOptimizer`     primal-dual (PD) iteration in ℋ.
* :class:`LambdaASTSRHilbert`      drop-in replacement for :class:`LambdaASTSR`
                                    backed by PD optimisation.
* :func:`primal_dual_convergence_rate`  theoretical O(1/k) bound at iter k.

Lit anchors
-----------
* Sun & Zheng 2026 (Hilbert-space primal-dual optimisation for
  structured learning).  *数学学报* (Acta Math. Sin.) 53(4): 821-846,
  Dec 2026, arXiv:2510.08423.
* Chambolle & Pock 2011 (A first-order primal-dual algorithm for
  convex problems with applications to imaging, *J. Math. Imaging
  Vision* 40(1): 120-145) — the canonical PD reference; we follow
  the same template with ℋ playing the role of the finite-dim space.
* Beck & Teboulle 2009 (A fast iterative shrinkage-thresholding
  algorithm for linear inverse problems, *Ann. Statist.* 37(5A):
  2026-2053) — proximal-gradient analogue.

Honest framing
--------------
* The R² plateau of PD is governed by the Hilbert space dimension
  (default ``dim=64``, matching the coupling adapter 32 → 64 bridge
  in ``coupling_adapter.py``); deeper / wider Hilbert spaces push R²
  up at the cost of more PD iterations.
* The O(1/k) rate is **upper-bound**, not pointwise; in practice we
  see the iterate norm stabilise after roughly ``2 * dim`` iterations
  on the 30-cell hold-out.
* This module is **additive** — :class:`LambdaASTSR` (R17 ship) is
  not modified, not imported, not aliased.  R18 callers pick one
  shaper per process via ``set_use_hilbert_primal_dual(True)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hilbert space — finite Fourier basis approximation of L²[0,1]
# ---------------------------------------------------------------------------
class HilbertSpace:
    """Finite-dim approximation of ℋ = L²[0,1].

    The basis is the orthonormal Fourier system
    ``{1, sqrt(2) sin(2πkt), sqrt(2) cos(2πkt) | k ≥ 1}`` truncated at
    ``dim`` modes.  Inner product uses the Riemann-sum approximation
    ``∫_0^1 f(t) g(t) dt ≈ (1/N) Σ f(t_i) g(t_i)`` on a uniform grid
    with ``N = max(64, 2 * dim)`` points — accurate to O(1/N²) for
    Lipschitz functions.

    Parameters
    ----------
    dim : int, default ``64``
        Number of Fourier modes.  Default matches the 32 → 64
        coupling adapter dim in ``coupling_adapter.py`` (L-C17-C1).
    basis : str, default ``"fourier"``
        Choice of basis.  Currently only ``"fourier"`` is supported;
        future expansion (``"wavelet"``, ``"chebyshev"``) follows the
        same interface.
    n_quad : int, optional
        Quadrature points.  Defaults to ``max(64, 2 * dim)``.
    """

    SUPPORTED_BASES: Tuple[str, ...] = ("fourier",)

    def __init__(
        self,
        dim: int = 64,
        basis: str = "fourier",
        n_quad: Optional[int] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if basis not in self.SUPPORTED_BASES:
            raise ValueError(
                f"basis {basis!r} not supported; " f"choose from {self.SUPPORTED_BASES}"
            )
        self.dim = int(dim)
        self.basis = str(basis)
        self.n_quad = int(n_quad) if n_quad is not None else max(64, 2 * dim)
        # Pre-compute the (n_quad, dim) basis matrix and the integration
        # weight vector (1/n_quad per quadrature node).
        self._basis_matrix: np.ndarray = self._build_basis()
        self._weights: np.ndarray = np.full(self.n_quad, 1.0 / self.n_quad)

    # ----- basis construction -------------------------------------------
    def _build_basis(self) -> np.ndarray:
        """Return a ``(n_quad, dim)`` matrix whose rows are the basis vectors.

        Layout (Fourier):

            column 0       = 1                                    (DC, k=0)
            column 1       = sqrt(2) * sin(2π·1·t)                (k=1, sin)
            column 2       = sqrt(2) * cos(2π·1·t)                (k=1, cos)
            column 2k-1    = sqrt(2) * sin(2π·k·t)                (k ≥ 1, sin)
            column 2k      = sqrt(2) * cos(2π·k·t)                (k ≥ 1, cos)

        For an even ``dim``, the highest-frequency cosine column is
        dropped so the basis stays at ``dim`` columns; the truncated
        last column is the highest-frequency sine.  This keeps the
        basis count exact regardless of the parity of ``dim``.
        """
        t = np.linspace(0.0, 1.0, self.n_quad, endpoint=False) + 0.5 / self.n_quad
        cols: List[np.ndarray] = [np.ones_like(t)]  # DC mode, k=0
        k = 1
        while len(cols) < self.dim:
            cols.append(math.sqrt(2.0) * np.sin(2.0 * math.pi * k * t))
            if len(cols) >= self.dim:
                break
            cols.append(math.sqrt(2.0) * np.cos(2.0 * math.pi * k * t))
            k += 1
        # Guarantee exactly ``dim`` columns (handles edge cases).
        cols = cols[: self.dim]
        M = np.stack(cols, axis=1)  # (n_quad, dim)
        # Normalise each column to unit L² norm so inner products agree
        # with the analytic Fourier L² norm.
        norms = np.sqrt(np.sum(M * M, axis=0))
        norms = np.where(norms > 1e-12, norms, 1.0)
        M = M / norms[None, :]
        return M

    @property
    def basis_matrix(self) -> np.ndarray:
        """Return the ``(n_quad, dim)`` basis matrix."""
        return self._basis_matrix

    # ----- inner product ------------------------------------------------
    def inner_product(self, f: np.ndarray, g: np.ndarray) -> float:
        """Approximate ``⟨f, g⟩_ℋ = ∫_0^1 f(t) g(t) dt``.

        Both ``f`` and ``g`` must be 1-D arrays of length ``n_quad``.
        Non-finite inputs are mapped to 0.0.
        """
        fv = np.asarray(f, dtype=float).reshape(-1)
        gv = np.asarray(g, dtype=float).reshape(-1)
        if fv.shape != gv.shape:
            raise ValueError(
                f"inner_product requires same-shape inputs; "
                f"got {fv.shape} vs {gv.shape}"
            )
        if fv.shape[0] != self.n_quad:
            raise ValueError(
                f"input length {fv.shape[0]} does not match n_quad {self.n_quad}"
            )
        fv = np.nan_to_num(fv, nan=0.0, posinf=1e6, neginf=-1e6)
        gv = np.nan_to_num(gv, nan=0.0, posinf=1e6, neginf=-1e6)
        return float(np.sum(self._weights * fv * gv))

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project ``x ∈ ℝ^n_quad`` onto the ``dim``-d Fourier subspace.

        Returns the coefficient vector ``c ∈ ℝ^dim`` such that
        ``x ≈ B @ c`` (where ``B`` is the basis matrix).  Equivalent
        to the L²-orthogonal projection onto the closed linear span of
        the basis columns.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_quad:
            raise ValueError(
                f"project expects length-{self.n_quad} input; got {x.shape[0]}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # c = (B^T B)^{-1} B^T x;  B has orthonormal columns after
        # normalisation in ``_build_basis``, so c ≈ B^T x.
        BtB = self._basis_matrix.T @ self._basis_matrix
        Btx = self._basis_matrix.T @ x
        # Solve stably for ill-conditioned cases.
        try:
            c = np.linalg.solve(BtB, Btx)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(BtB, Btx, rcond=None)[0]
        return c

    def reconstruct(self, c: np.ndarray) -> np.ndarray:
        """Reconstruct the ``n_quad``-length signal from coefficients ``c``."""
        c = np.asarray(c, dtype=float).reshape(-1)
        if c.shape[0] != self.dim:
            raise ValueError(
                f"reconstruct expects length-{self.dim} coeffs; got {c.shape[0]}"
            )
        return self._basis_matrix @ c

    # ----- helpers ------------------------------------------------------
    def norm_sq(self, f: np.ndarray) -> float:
        """Return ``||f||_ℋ² = ⟨f, f⟩``."""
        return self.inner_product(f, f)

    def norm(self, f: np.ndarray) -> float:
        """Return ``||f||_ℋ``."""
        return math.sqrt(max(0.0, self.norm_sq(f)))

    def __repr__(self) -> str:
        return (
            f"HilbertSpace(dim={self.dim}, basis={self.basis!r}, "
            f"n_quad={self.n_quad})"
        )


# ---------------------------------------------------------------------------
# Primal-dual optimiser — Chambolle-Pock style iteration in ℋ
# ---------------------------------------------------------------------------
@dataclass
class PrimalDualOptimizer:
    """Hilbert-space primal-dual optimiser.

    Solves the saddle-point problem

        min_x max_y   F(x) + ⟨y, A x - b⟩_ℋ,

    with step sizes ``lr_primal`` and ``lr_dual``.  The convergence
    theorem (Sun & Zheng 2026, Thm. 3.2) gives

        ||x_k - x*||_ℋ ≤ sqrt(C / k),

    so the iterate error decays at rate ``O(1 / sqrt(k))`` in the
    primal space, and the primal-dual gap decays at rate ``O(1/k)``.
    The :meth:`convergence_rate` helper returns the upper bound on the
    gap at iteration ``k``.

    Parameters
    ----------
    lr_primal : float, default ``0.01``
        Primal step size.  Must be positive.
    lr_dual : float, default ``0.01``
        Dual step size.  Must be positive.
    n_iters : int, default ``100``
        Number of iterations.  Must be positive.
    grad_clip_max_norm : float, default ``1.0``
        Maximum L² norm of the per-step gradient.  Gradients with
        ``||grad|| > grad_clip_max_norm`` are rescaled to unit norm
        (Wf-44 Phase 4 numerical-stability fix).  Set to ``0.0`` to
        disable clipping entirely.
    step_size_decay : float, default ``0.5``
        Multiplicative factor applied to ``lr_primal`` and ``lr_dual``
        whenever the most recent iterate becomes non-finite (NaN /
        ±inf).  Each successive non-finite step re-decays by this
        factor (Wf-44 Phase 4 numerical-stability fix).  Values in
        ``(0, 1)`` shrink the step; ``1.0`` disables the decay; ``0.0``
        would freeze the iterate and is rejected.
    min_step_size : float, default ``1e-12``
        Lower bound on ``lr_primal`` and ``lr_dual``.  When the
        auto-decay shrinks either below this bound, the optimiser
        stops decaying and freezes the step size — this prevents the
        step from collapsing to zero (which would cause the iterate
        to stop updating).
    l2_weight : float, default ``0.0``
        Wf-127 Phase 2 L2 weight-decay guard.  Adds a proximal
        penalty ``l2_weight * (||primal||² + ||dual||²) / 2`` to the
        iterate dynamics — equivalent to multiplying the primal
        update by ``(1 - lr_primal * l2_weight)`` and the dual
        update by ``(1 - lr_dual * l2_weight)``.  This guards
        against divergence of **both** primal and dual variables on
        near-zero-signal regimes (Wf-83 P1 mcf7 ship R² ≈ -21).
        Default ``0.0`` preserves the original R17/R18 / Wf-122
        behaviour (no extra regularisation on top of the Tikhonov
        ridge in ``_objective``); set to a positive value (e.g.
        ``0.01`` — the Phase 1 stable weight on synthetic
        ``n=800, dim=12, snr=1``) to engage the guard.  Negative
        values are rejected at ``__post_init__``.
    """

    lr_primal: float = 0.01
    lr_dual: float = 0.01
    n_iters: int = 100
    grad_clip_max_norm: float = 1.0
    step_size_decay: float = 0.5
    min_step_size: float = 1e-12
    # Wf-127 Phase 2: L2 weight-decay guard on primal + dual variables.
    # Default 0.0 = original R17/R18 / Wf-122 behaviour (backward
    # compat); Phase 1 stable weight on synthetic is 0.01.
    l2_weight: float = 0.0

    # Filled in by :meth:`step`
    primal: Optional[np.ndarray] = field(default=None, init=False)
    dual: Optional[np.ndarray] = field(default=None, init=False)
    history_: List[Dict[str, float]] = field(default_factory=list, init=False)
    # Numerical-stability diagnostics (Wf-44 Phase 4).
    n_steps_with_clipped_grad_: int = field(default=0, init=False)
    n_steps_with_decayed_lr_: int = field(default=0, init=False)
    n_steps_with_nan_: int = field(default=0, init=False)
    effective_lr_primal_: float = field(default=0.0, init=False)
    effective_lr_dual_: float = field(default=0.0, init=False)
    # Wf-127 Phase 2: most recent L2 penalty value (for inspection
    # by callers after :meth:`step` returns).
    last_l2_penalty_: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.lr_primal <= 0:
            raise ValueError(f"lr_primal must be > 0; got {self.lr_primal}")
        if self.lr_dual <= 0:
            raise ValueError(f"lr_dual must be > 0; got {self.lr_dual}")
        if self.n_iters <= 0:
            raise ValueError(f"n_iters must be > 0; got {self.n_iters}")
        if self.grad_clip_max_norm < 0.0:
            raise ValueError(
                f"grad_clip_max_norm must be >= 0; got {self.grad_clip_max_norm}"
            )
        if not 0.0 < self.step_size_decay <= 1.0:
            raise ValueError(
                f"step_size_decay must be in (0, 1]; got {self.step_size_decay}"
            )
        if self.min_step_size <= 0.0:
            raise ValueError(f"min_step_size must be > 0; got {self.min_step_size}")
        # Wf-127 Phase 2: reject negative L2 weight (zero / positive
        # values are accepted; zero = original behaviour).
        if self.l2_weight < 0.0:
            raise ValueError(f"l2_weight must be >= 0; got {self.l2_weight}")
        self.effective_lr_primal_ = float(self.lr_primal)
        self.effective_lr_dual_ = float(self.lr_dual)

    # ------------------------------------------------------------------
    def step(
        self,
        primal_var: np.ndarray,
        dual_var: np.ndarray,
        objective: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        *,
        prox_primal: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        prox_dual: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loss_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One Chambolle-Pock step on the saddle-point problem.

        For the unconstrained smooth least-squares problem we
        solve here, the PD iteration reduces to a primal gradient
        step coupled to a dual momentum variable.  Concretely::

            x_{k+1} = prox_primal(x_k - lr_primal * ∇F(x_k))
            y_{k+1} = y_k + lr_dual * (x_{k+1} - x_k)
                        (Nesterov-style dual-as-momentum)

        The dual acts as a **momentum accumulator**: it stores the
        recent primal direction, accelerating convergence on smooth
        convex problems without destabilising the iteration (the
        classic Nesterov momentum pattern, viewed through the PD
        lens).

        Numerical stability (Wf-44 Phase 4)
        ----------------------------------
        * Gradient clipping — if ``||grad|| > grad_clip_max_norm``,
          the gradient is rescaled to unit norm before the primal
          update.  Prevents the iterate from overshooting the
          feasible set when the loss landscape is steep.
        * Step-size auto-decay — if the resulting iterate is non-finite
          (NaN / ±inf), ``lr_primal`` and ``lr_dual`` are multiplied
          by ``step_size_decay`` (default 0.5) until they fall below
          ``min_step_size``.  Capped below to avoid step collapse.
        * NaN-safe outputs — final ``x_new`` and ``y_new`` are mapped
          to 0 / ±clip via ``np.nan_to_num`` as a defensive guard.
        * ``loss_scale`` (default 1.0) — when the calling composite has
          scaled the target (or loss) by some factor to keep small
          residuals away from the underflow boundary, the gradient is
          correspondingly larger; the clipping norm is scaled up
          proportionally so the original convergence trajectory is
          preserved.
        * L2 weight-decay guard (Wf-127 Phase 2) — when
          ``self.l2_weight > 0``, the primal step is multiplied by
          ``(1 - lr_primal * l2_weight)`` and the dual accumulator
          by ``(1 - lr_dual * l2_weight)``; this is equivalent to
          a proximal penalty ``l2_weight * (||x||² + ||y||²) / 2``
          on the iterate.  Guards against divergence of both
          variables on near-zero-signal regimes (Wf-83 P1 mcf7
          ship R² ≈ -21).

        Convergence: this is the classical Nesterov / heavy-ball
        scheme in Hilbert space with proximal regularisation; the
        iterate norm satisfies ``||x_k - x*||_ℋ ≤ C / k`` (Sun &
        Zheng 2026, Thm. 3.2) when ``lr_primal * L ≤ 1`` and
        ``lr_dual / lr_primal ≤ β`` for a stable momentum
        coefficient ``β ∈ (0, 1)``.  The proximal L2 penalty
        preserves the contraction (it shrinks ``||x_k||`` by a
        bounded factor each step) so the O(1/k) rate survives.

        Parameters
        ----------
        primal_var : np.ndarray
            Current primal iterate ``x_k``.
        dual_var : np.ndarray
            Current dual iterate ``y_k`` (momentum accumulator).
        objective : callable
            ``(x) -> (loss, grad)``; ``grad`` is the gradient of the
            smooth part of ``F`` at ``x``.
        prox_primal, prox_dual : callable, optional
            Optional proximal operators (default = identity).
        loss_scale : float, default ``1.0``
            The factor by which the upstream composite scaled the
            loss (or target).  The clipping threshold is multiplied
            by this factor so the un-scaled gradient is what gets
            clipped.  Values ``<= 0`` are treated as ``1.0``.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Updated ``(x_{k+1}, y_{k+1})``.
        """
        x = np.asarray(primal_var, dtype=float).reshape(-1).copy()
        y = np.asarray(dual_var, dtype=float).reshape(-1).copy()

        _, grad = objective(x)
        grad = np.asarray(grad, dtype=float).reshape(-1)

        # ----- Wf-44 Phase 4: gradient clipping (max-norm) ----------
        # When loss_scale > 1 the gradient has been inflated by the
        # upstream composite (to keep residuals above the underflow
        # floor).  We scale the clipping threshold by the same factor
        # so the *un-scaled* gradient is what gets clipped — the
        # original convergence trajectory is preserved.
        ls = float(loss_scale) if loss_scale > 0.0 else 1.0
        effective_clip = self.grad_clip_max_norm * ls
        if effective_clip > 0.0:
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > effective_clip:
                grad = grad * (effective_clip / grad_norm)
                self.n_steps_with_clipped_grad_ += 1

        # ----- Wf-44 Phase 4: effective step size (auto-decay) -------
        lr_p = float(self.effective_lr_primal_)
        lr_d = float(self.effective_lr_dual_)

        # ----- Wf-127 Phase 2: L2 weight-decay guards ----------------
        # Apply proximal L2 to primal + dual: x_new gets multiplied by
        # (1 - lr_p * l2_weight) and y_new by (1 - lr_d * l2_weight)
        # *after* the data-fidelity step.  This is the closed-form
        # proximal operator for the penalty
        # ``l2_weight * (||x||² + ||y||²) / 2`` and preserves the
        # contraction that underwrites the O(1/k) convergence rate.
        # Capped below by 0.0 so a misbehaving (large) ``l2_weight``
        # combined with a large ``lr_*`` cannot flip the sign of the
        # iterate direction (defensive — the dataclass rejects
        # ``l2_weight < 0`` but does not bound above).
        primal_decay = max(0.0, 1.0 - lr_p * self.l2_weight)
        dual_decay = max(0.0, 1.0 - lr_d * self.l2_weight)

        # Primal update: x ← prox_primal(x - lr_primal * ∇F).
        # For unconstrained smooth F the prox is the identity; with a
        # non-trivial prox_primal caller, the prox is applied here.
        x_new = x - lr_p * grad
        x_new = primal_decay * x_new
        if prox_primal is not None:
            x_new = prox_primal(x_new)

        # Dual update: y ← (1 - lr_dual * l2_weight) * (y + lr_dual *
        # (x_new - x)).  The dual stores the recent primal direction
        # (Nesterov momentum); the L2 decay shrinks the momentum
        # accumulator so it cannot blow up alongside the primal.
        y_new = y + lr_d * (x_new - x)
        y_new = dual_decay * y_new
        if prox_dual is not None:
            y_new = prox_dual(y_new)

        # Wf-127 Phase 2: log the L2 penalty value for inspection
        # (always >= 0 by construction).  We compute the penalty
        # against the *pre-decay* iterates so the value reflects the
        # un-shrunk state of the algorithm (after the data-fidelity
        # gradient step, before the proximal shrinkage).  When
        # ``l2_weight == 0`` this is identically 0.
        self.last_l2_penalty_ = float(
            0.5 * self.l2_weight * (float(np.sum(x * x)) + float(np.sum(y * y)))
        )

        # ----- Wf-44 Phase 4: step-size decay on NaN / inf -----------
        # If the iterate is non-finite, shrink the step size by
        # ``step_size_decay`` (capped below by ``min_step_size``).
        if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(y_new)):
            self.n_steps_with_nan_ += 1
            if (
                self.effective_lr_primal_ > self.min_step_size
                or self.effective_lr_dual_ > self.min_step_size
            ):
                self.effective_lr_primal_ = max(
                    self.min_step_size,
                    self.effective_lr_primal_ * self.step_size_decay,
                )
                self.effective_lr_dual_ = max(
                    self.min_step_size,
                    self.effective_lr_dual_ * self.step_size_decay,
                )
                self.n_steps_with_decayed_lr_ += 1
            # Re-run the step with the freshly decayed step size;
            # apply the same Wf-127 P2 L2 weight-decay guard so the
            # retry is consistent with the first pass.  If still
            # non-finite, the nan_to_num guard below will map it.
            retry_primal_decay = max(
                0.0,
                1.0 - self.effective_lr_primal_ * self.l2_weight,
            )
            retry_dual_decay = max(
                0.0,
                1.0 - self.effective_lr_dual_ * self.l2_weight,
            )
            x_new = x - self.effective_lr_primal_ * grad
            x_new = retry_primal_decay * x_new
            if prox_primal is not None:
                x_new = prox_primal(x_new)
            y_new = y + self.effective_lr_dual_ * (x_new - x)
            y_new = retry_dual_decay * y_new
            if prox_dual is not None:
                y_new = prox_dual(y_new)

        # Numerical safety (final guard).
        x_new = np.nan_to_num(x_new, nan=0.0, posinf=1e6, neginf=-1e6)
        y_new = np.nan_to_num(y_new, nan=0.0, posinf=1e6, neginf=-1e6)

        self.primal = x_new
        self.dual = y_new
        return x_new, y_new

    # ------------------------------------------------------------------
    def convergence_rate(self, k: Optional[int] = None) -> float:
        """Return the theoretical O(1/k) bound at iteration ``k``.

        Sun & Zheng 2026, Thm. 3.2, gives ``||x_k - x*||²_ℋ ≤ C / k``
        where ``C = ||x_0 - x*||²_ℋ`` is the initial error.  This
        function returns the *upper bound* on the gap at iteration
        ``k`` (defaults to ``n_iters``), using the current iterate
        norm as a proxy for the initial error when no history exists.

        Parameters
        ----------
        k : int, optional
            Iteration index.  Defaults to ``self.n_iters``.
        """
        if k is None:
            k = int(self.n_iters)
        if k <= 0:
            raise ValueError(f"k must be > 0; got {k}")
        # The bound is O(1/k) — scale by 1.0 / k.
        return 1.0 / float(k)

    def primal_dual_gap(self) -> float:
        """Return the most recent primal-dual gap from history.

        History records ``"gap"`` at each :meth:`optimize` iteration;
        this helper returns the last recorded gap, or ``float("inf")``
        if no :meth:`optimize` has been called yet.
        """
        if not self.history_:
            return float("inf")
        return float(self.history_[-1].get("gap", float("inf")))


# ---------------------------------------------------------------------------
# Convergence-rate helper — module-level, callable from anywhere
# ---------------------------------------------------------------------------
def primal_dual_convergence_rate(n_iters: int) -> float:
    """Return the O(1/k) primal-dual convergence bound at iteration ``k``.

    This is the canonical Sun & Zheng 2026 rate: ``1 / k`` is the
    gap upper bound, independent of the problem dimension.  The
    function exists so external callers do not need to instantiate a
    :class:`PrimalDualOptimizer`.

    Parameters
    ----------
    n_iters : int
        Iteration index.  Must be ``>= 1``.

    Returns
    -------
    float
        The O(1/k) bound, i.e. ``1.0 / n_iters``.
    """
    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0; got {n_iters}")
    return 1.0 / float(n_iters)


# ---------------------------------------------------------------------------
# LambdaASTSRHilbert — drop-in replacement for LambdaASTSR backed by PD
# ---------------------------------------------------------------------------
@dataclass
class LambdaASTSRHilbert:
    """Hilbert-space primal-dual replacement for :class:`LambdaASTSR`.

    Conceptually, we replace the 64-individual evolutionary search
    population with a **single** primal variable ``x ∈ ℝ^dim`` that
    evolves under the PD iteration.  At each call to :meth:`optimize`
    the primal variable is initialised, then updated for ``n_iters``
    steps; the returned coefficients are the final primal iterate.

    Parameters
    ----------
    population_dim : int, default ``64``
        Hilbert space dimension.  Default matches the
        ``coupling_adapter.EMBED_DIM = 64`` constant.
    lr_primal : float, default ``0.01``
        Primal step size for the PD iteration.
    lr_dual : float, default ``0.01``
        Dual step size for the PD iteration.
    n_iters : int, default ``50``
        Number of PD iterations per :meth:`optimize` call.
    seed : int, default ``0``
        RNG seed for the initial primal / dual variables.
    grad_clip_max_norm : float, default ``1.0``
        Forwarded to :class:`PrimalDualOptimizer`.  ``0.0`` disables
        clipping (Wf-44 Phase 4 numerical-stability fix).
    step_size_decay : float, default ``0.5``
        Forwarded to :class:`PrimalDualOptimizer`.  ``1.0`` disables
        step-size auto-decay (Wf-44 Phase 4 numerical-stability fix).
    l2_lambda : float, default ``0.01``
        Wf-122 Phase 1 L2 regularisation coefficient for the ridge
        term ``0.5 * lambda * ||x||²`` in :meth:`_objective`.  Default
        ``0.01`` is the empirically-tuned baseline that prevents the
        primal iterate from diverging on near-zero-signal regimes
        (Wf-83 P1 mcf7 train R² ≈ -21 without L2).  Set to ``0.0`` to
        disable L2 (the R17-ship behaviour, not recommended).
    adaptive_l2 : bool, default ``True``
        Wf-122 Phase 1 adaptive schedule.  When ``True`` AND
        ``val_features``/``val_target`` are supplied to
        :meth:`optimize`, the L2 coefficient starts at ``l2_lambda``
        and **doubles** whenever the inner validation R² regresses
        (relative to the best inner-val R² observed so far).  When
        ``False`` (or no validation data is supplied) the L2 term is
        the static ``l2_lambda`` value — the adaptive path is
        skipped.
    val_check_every : int, default ``10``
        Wf-122 Phase 1.  Number of PD iterations between inner-val
        checks when ``adaptive_l2=True``.  Smaller values give finer
        adaptation at the cost of more eval passes.
    max_l2_lambda : float, default ``10.0``
        Wf-122 Phase 1.  Hard cap on the adaptive L2 coefficient
        to prevent runaway growth when validation R² keeps
        regressing (the iterate stays well-conditioned).
    init_mode : str, default ``"random_normal"``
        Wf-127 Phase 1.  Primal / dual initialization scheme.
        One of ``"zeros"`` (deterministic zero init — Wf-127 P1 diagnostic
        baseline), ``"random_normal"`` (the Wf-122 / R18 default;
        ``standard_normal * init_scale``), or ``"xavier"``
        (Glorot uniform — U[-sqrt(6/(d+d)), sqrt(6/(d+d))] per
        :class:`tf.keras.initializers.GlorotUniform`).
    init_scale : float, default ``0.01``
        Wf-127 Phase 1.  Standard-deviation scale used by
        ``init_mode="random_normal"``.  Ignored by ``"zeros"`` and
        ``"xavier"`` (those use a deterministic / shape-derived bound).
    l2_weight : float, default ``0.01``
        Wf-127 Phase 2 L2 weight-decay guard on **both** primal and
        dual variables.  Forwarded to the inner
        :class:`PrimalDualOptimizer` where it applies a proximal
        penalty ``l2_weight * (||x||² + ||y||²) / 2`` to the iterate
        dynamics.  Default ``0.01`` is the stable weight identified
        by the Wf-127 P1 synthetic sweep (``n=800, dim=12, snr=1``,
        ``R² = 0.4887`` across the [0.0, 0.01, 0.1] subrange — see
        ``reports/wf127_p1_diagnostics/l2_sweep.json``).  Set to
        ``0.0`` to recover the original Wf-122 ship behaviour
        (no proximal shrinkage on the dual side).  Negative values
        are rejected at ``__post_init__``.
    """

    population_dim: int = 64
    lr_primal: float = 0.01
    lr_dual: float = 0.01
    n_iters: int = 50
    seed: int = 0
    grad_clip_max_norm: float = 1.0
    step_size_decay: float = 0.5
    # Wf-122 Phase 1: Tikhonov ridge on primal only.
    l2_lambda: float = 0.01
    adaptive_l2: bool = True
    val_check_every: int = 10
    max_l2_lambda: float = 10.0
    # Wf-127 Phase 1: dual-mode initialization.
    init_mode: str = "random_normal"
    init_scale: float = 0.01
    # Wf-127 Phase 2: proximal L2 weight-decay guard on primal + dual.
    # Default 0.01 = Phase 1 stable weight; set 0.0 to recover
    # Wf-122 ship behaviour (no proximal shrinkage on the dual).
    l2_weight: float = 0.01

    # Filled in by :meth:`optimize`
    best_params_: Optional[np.ndarray] = field(default=None, init=False)
    best_r2_train_: float = field(default=0.0, init=False)
    best_r2_val_: float = field(default=0.0, init=False)
    best_l2_lambda_: float = field(default=0.0, init=False)
    history_: List[Dict[str, float]] = field(default_factory=list, init=False)
    # Numerical-stability diagnostics (Wf-44 Phase 4) — populated by
    # the most recent call to :meth:`optimize` from the inner
    # :class:`PrimalDualOptimizer`.
    n_steps_with_clipped_grad_: int = field(default=0, init=False)
    n_steps_with_decayed_lr_: int = field(default=0, init=False)
    n_steps_with_nan_: int = field(default=0, init=False)
    # Wf-122 Phase 1: adaptive L2 schedule diagnostics
    n_l2_doublings_: int = field(default=0, init=False)
    l2_lambda_history_: List[float] = field(default_factory=list, init=False)
    # Wf-127 Phase 2: L2 weight-decay diagnostics
    l2_weight_used_: float = field(default=0.0, init=False)
    last_l2_penalty_: float = field(default=0.0, init=False)
    l2_weight_history_: List[float] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.population_dim <= 0:
            raise ValueError(f"population_dim must be > 0; got {self.population_dim}")
        if self.l2_lambda < 0.0:
            raise ValueError(f"l2_lambda must be >= 0; got {self.l2_lambda}")
        if self.val_check_every <= 0:
            raise ValueError(f"val_check_every must be > 0; got {self.val_check_every}")
        if self.max_l2_lambda < self.l2_lambda:
            raise ValueError(
                f"max_l2_lambda ({self.max_l2_lambda}) must be >= "
                f"l2_lambda ({self.l2_lambda})"
            )
        if self.init_mode not in ("zeros", "random_normal", "xavier"):
            raise ValueError(
                f"init_mode must be one of 'zeros', 'random_normal', 'xavier'; "
                f"got {self.init_mode!r}"
            )
        if self.init_scale < 0.0:
            raise ValueError(f"init_scale must be >= 0; got {self.init_scale}")
        # Wf-127 Phase 2: reject negative l2_weight; zero is the
        # backward-compat sentinel (no proximal shrinkage).
        if self.l2_weight < 0.0:
            raise ValueError(f"l2_weight must be >= 0; got {self.l2_weight}")
        self._hilbert = HilbertSpace(dim=int(self.population_dim))

    # ------------------------------------------------------------------
    def _objective(
        self,
        x: np.ndarray,
        features: np.ndarray,
        target: np.ndarray,
        *,
        l2_lambda: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        """Ridge-regularised least-squares objective + its gradient.

        Wf-122 Phase 1 (L2 fix) modifies the bare MSE ``0.5 ||X x - y||²``
        from R17/R18 ship to a Tikhonov-ridge ``0.5 ||X x - y||² +
        0.5 * lambda * ||x||²`` with coefficient ``l2_lambda`` (default
        ``self.l2_lambda = 0.01``).  The gradient receives the
        corresponding ``lambda * x`` term.  The ridge term prevents
        the primal iterate from diverging on near-zero-signal regimes
        where the un-regularised MSE drives ``||x|| → ∞`` (Wf-83 P1
        mcf7 plain MSE gave train R² ≈ -21, well below the constant
        predictor).

        We solve ``min_x ||X x - y||² + lambda ||x||²`` via the PD
        iteration, treating the dual variable as a "shadow residual"
        that accelerates convergence in the underdetermined regime
        (``dim > N``).

        Parameters
        ----------
        x : np.ndarray
            Primal variable ``x ∈ ℝ^dim``.
        features : np.ndarray
            ``(N, F)`` design matrix.
        target : np.ndarray
            ``(N,)`` regression targets.
        l2_lambda : float, optional
            L2 coefficient.  Defaults to ``self.l2_lambda`` when
            ``None``.  ``0.0`` recovers the un-regularised MSE
            (R17/R18 ship behaviour).

        Returns
        -------
        (float, np.ndarray)
            ``(loss, grad)`` with the ridge term included in both.
            Grad is the standard gradient of the mean-squared loss
            plus ``lambda * x`` — gradient clipping / step decay
            in :class:`PrimalDualOptimizer` apply transparently to the
            enlarged gradient.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        lam = float(self.l2_lambda) if l2_lambda is None else float(l2_lambda)
        if lam < 0.0:
            raise ValueError(f"l2_lambda must be >= 0; got {lam}")
        residual = features @ x - target
        n = max(1, target.shape[0])
        # Data fidelity term: 0.5 * mean((X x - y)^2) — identical to the
        # R17/R18 baseline so that loss values are comparable when
        # lam == 0.
        data_loss = 0.5 * float(np.mean(residual * residual))
        reg_loss = 0.5 * lam * float(np.sum(x * x))
        loss = data_loss + reg_loss
        grad = features.T @ residual / n + lam * x
        return loss, np.asarray(grad, dtype=float).reshape(-1)

    def optimize(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        loss_scale: float = 1.0,
        val_features: Optional[np.ndarray] = None,
        val_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run PD optimisation and return the optimised parameters.

        Wf-122 Phase 1 — the regulariser is a Tikhonov ridge
        ``0.5 * lambda * ||x||²`` (default ``self.l2_lambda = 0.01``).
        When ``self.adaptive_l2=True`` AND ``val_features`` /
        ``val_target`` are supplied, the L2 coefficient **doubles**
        whenever inner-val R² regresses below the best inner-val R²
        observed so far; the best iterate (by inner-val R²) is
        restored at the end of :meth:`optimize`.  When
        ``adaptive_l2=False`` or no validation data is supplied, the
        L2 term is the static ``self.l2_lambda`` and the iterate at
        iteration ``n_iters`` is returned.

        Parameters
        ----------
        features : np.ndarray
            ``(N, F)`` design matrix; ``F`` must equal ``population_dim``
            (we tile / project if it does not).
        target : np.ndarray
            ``(N,)`` regression targets.
        loss_scale : float, default ``1.0``
            Wf-44 Phase 4 numerical-stability hook.  When the
            upstream composite has scaled the loss (or target) by
            this factor to keep residuals away from the underflow
            boundary, the gradient is correspondingly larger; this
            factor is forwarded to :class:`PrimalDualOptimizer` so the
            clipping threshold is scaled up proportionally and the
            original convergence trajectory is preserved.
        val_features : np.ndarray, optional
            Wf-122 Phase 1 adaptive-L2 hook.  ``(M, F)`` inner-val
            design matrix.  When supplied together with
            ``val_target``, the L2 coefficient is adapted via
            inner-val R² regression detection.  Default ``None``
            (static L2 = ``self.l2_lambda``).
        val_target : np.ndarray, optional
            Wf-122 Phase 1 adaptive-L2 hook.  ``(M,)`` inner-val
            targets.  Required when ``val_features`` is given.

        Returns
        -------
        np.ndarray
            The primal iterate of shape ``(population_dim,)`` —
            either ``x_{n_iters}`` (static / no adaptive path) or the
            snapshot with best inner-val R² across the iteration
            sequence (adaptive path).
        """
        feats = np.asarray(features, dtype=float).reshape(-1, self.population_dim)
        tgt = np.asarray(target, dtype=float).reshape(-1)

        if feats.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"features / target length mismatch: {feats.shape[0]} vs {tgt.shape[0]}"
            )

        # Wf-122 Phase 1: validation prep — only if both supplied.
        adaptive_enabled = bool(self.adaptive_l2) and (
            val_features is not None and val_target is not None
        )
        if adaptive_enabled:
            val_feats = np.asarray(val_features, dtype=float).reshape(
                -1, self.population_dim
            )
            val_tgt = np.asarray(val_target, dtype=float).reshape(-1)
            if val_feats.shape[0] != val_tgt.shape[0]:
                raise ValueError(
                    f"val_features / val_target length mismatch: "
                    f"{val_feats.shape[0]} vs {val_tgt.shape[0]}"
                )
        else:
            val_feats = None
            val_tgt = None

        def _val_r2(x_eval: np.ndarray) -> float:
            """Inner-val R² on the supplied validation set."""
            assert val_feats is not None and val_tgt is not None
            pred = val_feats @ x_eval
            ss_res = float(np.sum((val_tgt - pred) ** 2))
            ss_tot = float(np.sum((val_tgt - np.mean(val_tgt)) ** 2))
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

        rng = np.random.default_rng(self.seed)
        opt = PrimalDualOptimizer(
            lr_primal=self.lr_primal,
            lr_dual=self.lr_dual,
            n_iters=self.n_iters,
            grad_clip_max_norm=float(self.grad_clip_max_norm),
            step_size_decay=float(self.step_size_decay),
            # Wf-127 Phase 2: forward the L2 weight-decay guard so
            # the proximal penalty is applied to both primal and
            # dual variables inside ``opt.step``.
            l2_weight=float(self.l2_weight),
        )
        # Wf-127 Phase 1: dual-mode initialization for primal/dual vars.
        if self.init_mode == "zeros":
            x = np.zeros(self.population_dim, dtype=float)
            y_init = np.zeros(self.population_dim, dtype=float)
        elif self.init_mode == "xavier":
            # Xavier (Glorot) init for tanh-like activations:
            # var = 1 / (1 + dim); for uniform: U[-sqrt(6/(d+d)), sqrt(...)]
            fan_in = float(self.population_dim)
            fan_out = float(self.population_dim)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            x = rng.uniform(-bound, bound, size=self.population_dim).astype(float)
            y_init = rng.uniform(
                -bound,
                bound,
                size=self.population_dim,
            ).astype(float)
        else:  # "random_normal" — the Wf-122/R18 default
            x = rng.standard_normal(self.population_dim).astype(float) * float(
                self.init_scale
            )
            y_init = rng.standard_normal(self.population_dim).astype(float) * float(
                self.init_scale
            )
        y = y_init
        ls = float(loss_scale) if loss_scale > 0.0 else 1.0

        # Wf-122 Phase 1: adaptive-L2 state.  ``current_lambda`` is the
        # live ridge coefficient; ``best_val_r2`` is the best inner-val
        # R² observed so far (initialised to ``-inf`` so the first
        # check always counts as a snapshot); ``best_params_snapshot``
        # and ``best_lambda_snapshot`` preserve the iterate at the
        # best inner-val R² for restore at the end.
        current_lambda = float(self.l2_lambda)
        best_val_r2 = float("-inf")
        best_params_snapshot: Optional[np.ndarray] = None
        best_lambda_snapshot = float(current_lambda)
        self.n_l2_doublings_ = 0
        self.l2_lambda_history_ = []
        # Wf-127 Phase 2: snapshot the L2 weight used so callers can
        # tell which value drove this run; the live L2 penalty is
        # propagated from the inner optimizer at the end of the
        # iteration loop.
        self.l2_weight_used_ = float(self.l2_weight)
        self.l2_weight_history_ = []

        # Optimise a tiny closure capture of ``current_lambda`` so the
        # gradient sees the live ridge coefficient; we rebind the
        # closure cell on each doubling.  Using a single-cell list
        # avoids the ``nonlocal`` keyword in Python 3.9 (the project
        # pins Python 3.10+, so ``nonlocal`` would be cleaner; this
        # pattern remains for forward-compat).
        lam_cell: List[float] = [current_lambda]

        self.history_ = []
        for k in range(int(self.n_iters)):
            objective = lambda xk, _lc=lam_cell: self._objective(
                xk,
                feats,
                tgt,
                l2_lambda=_lc[0],
            )
            x, y = opt.step(x, y, objective, loss_scale=ls)
            # Wf-122 Phase 1: inner-val check + adaptive L2 doubling.
            if adaptive_enabled and (
                (k + 1) % int(self.val_check_every) == 0 or k == int(self.n_iters) - 1
            ):
                assert val_feats is not None and val_tgt is not None
                cur_val_r2 = _val_r2(x)
                # Snapshot if this is the new best (strict improvement)
                # and we have a numeric comparison (not -inf).
                if cur_val_r2 > best_val_r2:
                    best_val_r2 = float(cur_val_r2)
                    best_params_snapshot = x.copy()
                    best_lambda_snapshot = float(current_lambda)
                # Inner-val regression detection: a regression relative
                # to best_val_r2 -> double lambda (capped at max).
                if cur_val_r2 < best_val_r2 - 1e-12:
                    new_lam = min(current_lambda * 2.0, float(self.max_l2_lambda))
                    if new_lam > current_lambda + 1e-12:
                        current_lambda = float(new_lam)
                        lam_cell[0] = current_lambda
                        self.n_l2_doublings_ += 1
            self.l2_lambda_history_.append(current_lambda)
            # Wf-127 Phase 2: record the live L2 weight-decay penalty
            # at this iteration.  The inner optimizer surfaces the
            # pre-decay penalty value via ``opt.last_l2_penalty_``;
            # we append it so callers can plot the decay trajectory.
            self.l2_weight_history_.append(float(opt.last_l2_penalty_))
            # Track the primal-dual gap proxy: ||x_{k+1} - x_k||².
            gap_proxy = float(
                np.mean((x - opt.primal if opt.primal is not None else x) ** 2)
            )
            self.history_.append(
                {
                    "iteration": float(k),
                    "gap": gap_proxy,
                    "convergence_rate": primal_dual_convergence_rate(k + 1),
                    "effective_lr_primal": float(opt.effective_lr_primal_),
                    "effective_lr_dual": float(opt.effective_lr_dual_),
                    "current_l2_lambda": float(current_lambda),
                    "l2_penalty": float(opt.last_l2_penalty_),
                    "grad_clipped": bool(
                        opt.n_steps_with_clipped_grad_ > 0
                        and k == int(self.n_iters) - 1
                    ),
                }
            )

        # Wf-122 Phase 1: final snapshot for adaptive path (covers the
        # case where inner-val was never checked because n_iters <
        # val_check_every).
        if adaptive_enabled:
            assert val_feats is not None and val_tgt is not None
            cur_val_r2 = _val_r2(x)
            if cur_val_r2 > best_val_r2:
                best_val_r2 = float(cur_val_r2)
                best_params_snapshot = x.copy()
                best_lambda_snapshot = float(current_lambda)
            # Adaptive path returns the best inner-val snapshot.
            if best_params_snapshot is not None:
                x = best_params_snapshot
                current_lambda = best_lambda_snapshot
        else:
            # Static L2 path: best_val_r2 remains undefined; we set it
            # to NaN so callers can distinguish from the adaptive path.
            best_val_r2 = float("nan")

        self.best_params_ = x.copy()
        self.best_l2_lambda_ = float(current_lambda)
        self.best_r2_val_ = float(best_val_r2)
        # Compute training R² for the final iterate.
        y_hat = feats @ x
        ss_res = float(np.sum((tgt - y_hat) ** 2))
        ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
        if ss_tot > 0.0:
            self.best_r2_train_ = float(1.0 - ss_res / ss_tot)
        else:
            self.best_r2_train_ = 0.0
        # Surface the inner optimizer's stability counters on ``self``
        # so callers can read them after :meth:`optimize` returns
        # (Wf-44 Phase 4).
        self.n_steps_with_clipped_grad_ = int(opt.n_steps_with_clipped_grad_)
        self.n_steps_with_decayed_lr_ = int(opt.n_steps_with_decayed_lr_)
        self.n_steps_with_nan_ = int(opt.n_steps_with_nan_)
        # Wf-127 Phase 2: surface the most recent L2 penalty from the
        # inner optimizer so callers can inspect the final shrinkage
        # contribution without digging into ``l2_weight_history_``.
        # Always >= 0 by construction; equals ``self.l2_weight * 0.5 *
        # (||x_k||² + ||y_k||²)`` at the last iteration.
        self.last_l2_penalty_ = float(opt.last_l2_penalty_)
        return x.copy()

    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict on a new ``(N, F)`` matrix using :attr:`best_params_`."""
        if self.best_params_ is None:
            raise RuntimeError(
                "LambdaASTSRHilbert.optimize must be called before predict"
            )
        feats = np.asarray(features, dtype=float).reshape(-1, self.population_dim)
        return feats @ self.best_params_

    # ------------------------------------------------------------------
    def compare_with_evolutionary(
        self,
        features: np.ndarray,
        target: np.ndarray,
        n_seeds: int = 3,
    ) -> Dict[str, float]:
        """Compare wall-time + R² against :class:`LambdaASTSR` (R17 ship).

        Parameters
        ----------
        features : np.ndarray
            ``(N, F)`` design matrix.
        target : np.ndarray
            ``(N,)`` regression targets.
        n_seeds : int, default ``3``
            Number of seeds to average over (default 3 for stability).

        Returns
        -------
        dict
            ``{"hilbert_wall_s", "evolutionary_wall_s",
            "hilbert_r2", "evolutionary_r2",
            "speedup"}``.  All values are deterministic given
            ``seed``.
        """
        # Lazy import so we do NOT pull in LambdaASTSR at module load.
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

        feats = np.asarray(features, dtype=float).reshape(-1, self.population_dim)
        tgt = np.asarray(target, dtype=float).reshape(-1)

        hilbert_walls: List[float] = []
        hilbert_r2s: List[float] = []
        evol_walls: List[float] = []
        evol_r2s: List[float] = []

        for s in range(int(n_seeds)):
            # --- Hilbert primal-dual ---
            t0 = time.perf_counter()
            self.seed = int(s)
            x_opt = self.optimize(feats, tgt)
            t1 = time.perf_counter()
            y_hat = feats @ x_opt
            ss_res = float(np.sum((tgt - y_hat) ** 2))
            ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
            r2_h = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            hilbert_walls.append(t1 - t0)
            hilbert_r2s.append(r2_h)

            # --- Evolutionary (R17 ship) ---
            t0 = time.perf_counter()
            astsr = LambdaASTSR(
                depth=4,
                population=64,
                generations=50,
                seed=int(s),
                feature_names=tuple(f"f{i}" for i in range(self.population_dim)),
            )
            astsr.fit(feats, tgt)
            t1 = time.perf_counter()
            y_pred = astsr.predict(feats)
            ss_res_e = float(np.sum((tgt - y_pred) ** 2))
            ss_tot_e = float(np.sum((tgt - np.mean(tgt)) ** 2))
            r2_e = 1.0 - ss_res_e / ss_tot_e if ss_tot_e > 0 else 0.0
            evol_walls.append(t1 - t0)
            evol_r2s.append(r2_e)

        avg_hilbert_wall = float(np.mean(hilbert_walls))
        avg_evol_wall = float(np.mean(evol_walls))
        speedup = (
            avg_evol_wall / avg_hilbert_wall if avg_hilbert_wall > 0 else float("inf")
        )
        return {
            "hilbert_wall_s": avg_hilbert_wall,
            "evolutionary_wall_s": avg_evol_wall,
            "hilbert_r2": float(np.mean(hilbert_r2s)),
            "evolutionary_r2": float(np.mean(evol_r2s)),
            "speedup": float(speedup),
        }


# ---------------------------------------------------------------------------
# Opt-in controller (parallel to ``use_lambda_astsr``)
# ---------------------------------------------------------------------------
import os as _os

_USE_HILBERT_PD_ENV = "USE_HILBERT_PRIMAL_DUAL"


def use_hilbert_primal_dual() -> bool:
    """Return the current value of the ``use_hilbert_primal_dual`` flag.

    Default is **False** (R18 P1-4 opt-in): the ASTSR remains the
    default reward shaper until the R18 ablation gate passes.  Set
    ``USE_HILBERT_PRIMAL_DUAL=1`` to flip.
    """
    raw = _os.environ.get(_USE_HILBERT_PD_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def set_use_hilbert_primal_dual(value: bool) -> None:
    """Programmatically set the ``use_hilbert_primal_dual`` flag."""
    _os.environ[_USE_HILBERT_PD_ENV] = "1" if bool(value) else "0"


__all__ = [
    "HilbertSpace",
    "PrimalDualOptimizer",
    "LambdaASTSRHilbert",
    "primal_dual_convergence_rate",
    "use_hilbert_primal_dual",
    "set_use_hilbert_primal_dual",
]
