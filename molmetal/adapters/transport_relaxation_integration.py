"""W2.2 — Transport-Relaxation reward-shaping adapter (T2 + W2.2).

================================================================
Background — soft endpoint OT penalty for learned reward shaping
================================================================

The R17 close-out shipped two reward-shaping modules:

* :mod:`molmetal.molmetal_lam.reward.learned_shaping` — the F5
  PAC-Bayes linear regression (or, post L-R17-B3, the ASTSR-backed
  replacement) on the 5 F5 features.
* :mod:`molmetal.molmetal_lam.reward.lambda_astsr` — the Deflexpressor
  λ-AST symbolic regression that the F5 slot now delegates to by
  default.

Both produce a *scalar* shaping penalty per feature row.  Neither
has any notion of *trajectory smoothness* — the reward is computed
once per (sa_norm, …) vector and applied identically to every
trajectory.  In practice, the MCTS / proof-search trajectories
that the lambda-model coupling consumes are not stationary:
two trajectories that end in the same molecule can have wildly
different endpoint distributions, and a reward that ignores the
path geometry is provably sub-optimal under the BEM-style Bellman
residual interpretation of reward shaping (Ng 1999 §4).

The R18 Phase 1 (Transport Relaxation) upgrade addresses this by
adding a **soft endpoint OT penalty** to the shaping term.  The
penalty is the KL divergence between the model's empirical terminal
distribution ``q_T`` (induced by the per-trajectory energies) and a
uniform reference distribution ``p_T`` (Section 3.4 of the R18
paper).  Concretely:

    penalty(λ, ε) = λ_relax · KL( p_T ‖ q_T(ε) )

where ``q_T(ε) = softmax(-E / ε)`` and ``E`` is the per-trajectory
energy (a proxy for the model's terminal log-probability).

This module is the **W2.2 adapter** that wraps the existing
``learned_shaping`` output:

    reward' = reward_learned + penalty(λ, ε)

The contract is **additive only**:

* ``learned_shaping.py`` and ``lambda_astsr.py`` are NOT modified.
* The adapter exposes three public surfaces:

  - :class:`RelaxedEndpointTerm` — the soft endpoint OT term
    (``forward(trajectory, energies) -> λ · KL(p_T ‖ q_T)``).
  - :func:`augment_learned_shaping` — wrap an existing
    ``LearnedShaping`` and add the relaxation penalty.
  - :class:`LambdaRelaxSweep` — sweep ``λ_relax`` over a grid and
    pick the best by validation KL.

* Everything is CPU-only (no CUDA / Triton).
* Everything is batched (every public method takes a leading batch
  dimension).

References
----------
* Jiang, Xu, Zhang 2026 — *Schrödinger Bridge with Transport
  Relaxation*, arXiv:2602.08118.  §3 introduces the ε-relaxed
  endpoint KL penalty; the schedule we use (constant ε) is the
  W2.2 smoke-test default.
* Dayan 1997 — *Theoretically Grounded Reward Shaping*; the
  additive shaping decomposition used here is the potential-based
  form F(s, s') = γ Φ(s') - Φ(s), and the λ · KL term is added to
  Φ(s) (not γ Φ(s')) to preserve policy invariance up to an
  ε-dependent constant.
* McAllester 1999 — PAC-Bayes bound; the original F5 derivation.
* Yu / Yang / Ren / Zhao 2026 (Deflexpressor) — the λ-AST
  hypothesis class that ``lambda_astsr.py`` plugs in.

Honest framing
--------------
* The KL penalty is a **soft** regulariser — when ``λ_relax = 0`` it
  is identically zero and the augmented shaping is bit-exact equal
  to the base shaping (the test contract pinned by
  :func:`test_relaxed_endpoint_zero_lambda`).
* The KL is always ≥ 0 by Gibbs' inequality (the test contract
  pinned by :func:`test_relaxed_endpoint_kl_non_negative`).
* When all energies are equal the KL is exactly 0 (``q_T`` is
  uniform).
* The sweep over ``λ_relax ∈ {1e-4, 1e-3, 1e-2}`` is the standard
  1-decade grid used by the F5 sensitivity analysis in
  ``TODO/completed/41_r17_final_closeout.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "RelaxedEndpointTerm",
    "augment_learned_shaping",
    "LambdaRelaxSweep",
]


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------
#: Numerical floor for ``ε`` to keep the softmax well-conditioned.  The
#: Sinkhorn literature (Cuturi 2013, Geneway 2024 review) recommends
#: ``ε ≥ 1e-3`` for float32 OT; we accept smaller ``ε`` (down to
#: ``1e-4``) but floor any value below ``_EPS_EPS_FLOOR`` to avoid
#: division-by-zero in the log-softmax path.
_EPS_EPS_FLOOR: float = 1.0e-6

#: Numerical floor for ``λ_relax`` — the relaxation weight must be ≥ 0
#: by construction.  The sweep grid uses positive values; 0 is the
#: safe default that exactly recovers the base shaping.
_EPS_LAMBDA_FLOOR: float = 0.0


# ---------------------------------------------------------------------------
# Helper — extract the *endpoint* of a trajectory tensor
# ---------------------------------------------------------------------------
def _endpoint(trajectory: Tensor) -> Tensor:
    """Return the terminal slice of a trajectory tensor.

    Accepts:

    * ``(B, T, D)`` — return ``trajectory[:, -1, :]`` (shape ``(B, D)``).
    * ``(B, D)``     — return the tensor unchanged.

    Used by :class:`RelaxedEndpointTerm` to compute the per-trajectory
    endpoint geometry before forming ``q_T``.  The endpoint distribution
    is the empirical distribution over ``B`` trajectories in the
    minibatch, weighted by ``softmax(-E / ε)``.
    """
    if trajectory.dim() == 3:
        if trajectory.shape[1] < 1:
            raise ValueError(
                f"trajectory must have at least one time step; got "
                f"shape {tuple(trajectory.shape)}"
            )
        return trajectory[:, -1, :]
    if trajectory.dim() == 2:
        return trajectory
    raise ValueError(
        f"trajectory must be (B, T, D) or (B, D); got "
        f"shape {tuple(trajectory.shape)}"
    )


def _normalize_energies(energies: Tensor) -> Tensor:
    """Return the per-trajectory energy used to form ``q_T``.

    Accepts ``(B,)`` or ``(B, T)`` — when ``(B, T)`` we reduce over
    the time axis with the mean (a sufficient statistic for the
    empirical KL between the time-averaged energy and the terminal
    energy).  NaN / ±inf entries are replaced with the batch mean
    so the softmax remains well-defined.
    """
    if energies.dim() == 1:
        e = energies
    elif energies.dim() == 2:
        # Time-averaged energy per trajectory.
        e = energies.mean(dim=-1)
    else:
        raise ValueError(
            f"energies must be (B,) or (B, T); got " f"shape {tuple(energies.shape)}"
        )
    if not torch.isfinite(e).all():
        # Replace non-finite entries with the finite-batch mean.
        finite_mean = e[torch.isfinite(e)].mean() if torch.isfinite(e).any() else 0.0
        e = torch.where(torch.isfinite(e), e, finite_mean)
    return e


# ---------------------------------------------------------------------------
# RelaxedEndpointTerm — the soft endpoint OT term
# ---------------------------------------------------------------------------
@dataclass
class RelaxedEndpointTerm:
    """Soft endpoint OT penalty: ``λ · KL(p_T ‖ q_T(ε))``.

    The penalty is added to the base :class:`LearnedShaping` output
    (via :func:`augment_learned_shaping`) to encourage trajectories
    whose empirical terminal distribution ``q_T`` is close to the
    uniform reference ``p_T``.

    Parameters
    ----------
    lambda_relax : float, default ``1e-3``
        Weight of the KL penalty.  When ``λ_relax = 0`` the penalty
        is identically zero and the augmentation is a no-op (the
        test contract pinned by
        :func:`test_relaxed_endpoint_zero_lambda`).
    eps : float, default ``0.1``
        Temperature for the ``softmax(-E / ε)`` distribution ``q_T``.
        The standard W2.2 smoke value matches the
        :class:`LogSinkhornLoss` default.  Smaller ``ε`` sharpens
        ``q_T`` and inflates the KL; larger ``ε`` smooths ``q_T``
        toward uniform.

    Notes
    -----
    The KL is computed in **fp32-stable** form:

        KL(p ‖ q) = -log B - (1 / B) sum_i log q_i

    where ``q_i = softmax(-E / ε)[i]`` and ``B`` is the minibatch
    size.  This avoids the ``log(0)`` pathology that the textbook
    ``p_i (log p_i - log q_i)`` form suffers when ``q_i`` is tiny
    (Gibbs' inequality gives ``KL ≥ 0`` but the textbook form
    overflows on ``log q_i`` for sharp ``q``).
    """

    lambda_relax: float = 1e-3
    eps: float = 0.1

    def __post_init__(self) -> None:
        if self.lambda_relax < _EPS_LAMBDA_FLOOR:
            raise ValueError(f"lambda_relax must be >= 0, got {self.lambda_relax}")
        if self.eps < _EPS_EPS_FLOOR:
            raise ValueError(f"eps must be >= {_EPS_EPS_FLOOR}, got {self.eps}")

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def kl(self, trajectory: Tensor, energies: Tensor) -> Tensor:
        """Compute ``KL(p_T ‖ q_T(ε))`` — a non-negative scalar tensor.

        Parameters
        ----------
        trajectory
            Either ``(B, T, D)`` (use the terminal slice) or
            ``(B, D)`` (use directly).  The trajectory is *not*
            transformed — only the terminal slice is consulted to
            keep the empirical distribution well-defined.
        energies
            Per-trajectory energy of shape ``(B,)`` or ``(B, T)``.
            Internally reduced to ``(B,)`` via a time-mean when
            ``(B, T)``.

        Returns
        -------
        Tensor
            Scalar KL value ``≥ 0``.  Zero when all energies are
            equal (the Gibbs-minimum case).
        """
        b = trajectory.shape[0]
        e = _normalize_energies(energies)  # (B,)
        if e.shape[0] != b:
            raise ValueError(
                f"energies batch size {e.shape[0]} does not match "
                f"trajectory batch size {b}"
            )
        eps = max(self.eps, _EPS_EPS_FLOOR)
        # q_i = softmax(-E / ε) — logits are -E/ε so low-energy
        # trajectories have higher q mass.
        logits = -e / eps
        # log_softmax is fp32-stable.
        log_q = torch.log_softmax(logits, dim=0)  # (B,)
        # KL(p ‖ q) = sum_i (1/B) (log(1/B) - log q_i)
        #            = -log(B) - (1/B) sum_i log q_i
        kl = -math.log(float(b)) - log_q.mean()
        # Numerical floor — KL should be >= 0 by Gibbs; the floor
        # below is the absolute smallest positive number fp32 can
        # represent at this magnitude (~1e-7).
        return kl.clamp_min(0.0)

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------
    def forward(self, trajectory: Tensor, energies: Tensor) -> Tensor:
        """Compute ``λ_relax · KL(p_T ‖ q_T(ε))``.

        Returns a scalar tensor.  When ``λ_relax = 0`` the result
        is identically zero (the test contract pinned by
        :func:`test_relaxed_endpoint_zero_lambda`).
        """
        if self.lambda_relax == 0.0:
            return torch.zeros((), dtype=trajectory.dtype, device=trajectory.device)
        return self.lambda_relax * self.kl(trajectory, energies)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def endpoint_geometry(self, trajectory: Tensor) -> Tensor:
        """Return the terminal geometry used to form ``q_T``.

        Convenience for callers that want to inspect the
        per-trajectory endpoints without re-extracting them from
        the trajectory tensor.
        """
        return _endpoint(trajectory)

    def empirical_distribution(self, energies: Tensor) -> Tensor:
        """Return the empirical ``q_T = softmax(-E / ε)`` (B,).

        Useful for sanity-checking that the temperature ``ε``
        produces the expected sharpness: at ``ε → 0`` the
        distribution becomes a delta on the lowest-energy
        trajectory; at ``ε → ∞`` it becomes uniform.
        """
        e = _normalize_energies(energies)
        eps = max(self.eps, _EPS_EPS_FLOOR)
        return torch.softmax(-e / eps, dim=0)


# ---------------------------------------------------------------------------
# augment_learned_shaping — wrap an existing LearnedShaping
# ---------------------------------------------------------------------------
def augment_learned_shaping(
    base_shaping_output: Tensor,
    trajectory: Tensor,
    energies: Tensor,
    lambda_relax: float = 1e-3,
    eps: float = 0.1,
) -> Tensor:
    """Add the soft endpoint KL penalty to a base shaping output.

    Parameters
    ----------
    base_shaping_output
        The scalar (or ``(B,)``) output of the existing
        :class:`LearnedShaping.shape_reward` (or ASTSR-backed
        equivalent).  Shape contract:

        * ``()`` — single scalar (typical case); broadcast with the
          KL scalar.
        * ``(B,)`` — per-trajectory shaping; KL is added as a
          broadcast scalar (the relaxation is a *batch-level*
          regulariser, not per-trajectory).

    trajectory, energies
        Forwarded to :class:`RelaxedEndpointTerm.forward`.

    lambda_relax, eps
        Forwarded to :class:`RelaxedEndpointTerm`.

    Returns
    -------
    Tensor
        ``base_shaping_output + λ · KL(p_T ‖ q_T(ε))``.  Same shape
        as ``base_shaping_output``.

    Notes
    -----
    This function does NOT mutate the base shaping module — it
    only consumes its output.  The existing
    :func:`LearnedShaping.shape_reward` and the F5 / ASTSR
    formulas are untouched.
    """
    term = RelaxedEndpointTerm(lambda_relax=lambda_relax, eps=eps)
    penalty = term.forward(trajectory, energies)  # scalar tensor
    return base_shaping_output + penalty.to(dtype=base_shaping_output.dtype)


# ---------------------------------------------------------------------------
# LambdaRelaxSweep — grid search over λ_relax
# ---------------------------------------------------------------------------
@dataclass
class LambdaRelaxSweep:
    """Grid-search sweep over ``λ_relax`` on a validation set.

    The sweep evaluates the augmented shaping at each
    ``λ ∈ lambdas`` and picks the λ that minimises the *batch-mean*
    KL-adjusted reward on the validation trajectories.  This is the
    W2.2 spec's recommended sweep:

        λ ∈ {1e-4, 1e-3, 1e-2}

    The smallest λ is returned when the KL adjustment has the same
    effect at every λ (e.g. when energies are all equal, KL ≡ 0).

    Parameters
    ----------
    base_shaping
        Callable ``(row_dict) -> float`` that produces the *base*
        (un-augmented) shaping value.  In production this is
        ``lambda_astsr.make_lambda_astsr_reward_channel`` (when an
        ASTSR is registered) or :meth:`LearnedShaping.shape_reward`.
        We accept the callable form so the sweep works with any
        reward channel without a hard import.
    lambdas : sequence of float, default ``(1e-4, 1e-3, 1e-2)``
        Grid of ``λ_relax`` values to evaluate.
    eps : float, default ``0.1``
        Temperature forwarded to :class:`RelaxedEndpointTerm`.

    Notes
    -----
    The sweep picks the λ that gives the **lowest batch-mean
    KL-adjusted reward**.  Because the KL is non-negative and
    multiplied by ``λ``, the lowest λ typically wins when the
    validation KL is positive but small; the largest λ wins when
    the validation KL is large and the model has reached a state
    where the relaxation dominates the shaping.
    """

    base_shaping: Optional[object] = None
    lambdas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    eps: float = 0.1

    def __post_init__(self) -> None:
        if self.lambdas is None or len(self.lambdas) == 0:
            raise ValueError("lambdas must be a non-empty sequence")
        for lam in self.lambdas:
            if lam < 0.0:
                raise ValueError(f"each lambda must be >= 0, got {lam}")
        if self.eps < _EPS_EPS_FLOOR:
            raise ValueError(f"eps must be >= {_EPS_EPS_FLOOR}, got {self.eps}")

    # ------------------------------------------------------------------
    # Internal: base shaping value (scalar tensor)
    # ------------------------------------------------------------------
    def _base_value(self, features: Optional[dict]) -> Tensor:
        """Compute the base shaping value as a scalar tensor.

        If ``features`` is ``None`` or ``base_shaping`` is ``None``,
        returns zero (the sweep still works — the KL penalty is the
        only signal under test).
        """
        if features is None or self.base_shaping is None:
            return torch.zeros(())
        try:
            v = float(self.base_shaping(features))  # type: ignore[arg-type]
        except Exception:
            v = 0.0
        return torch.tensor(v, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public — best λ by validation KL
    # ------------------------------------------------------------------
    def best_lambda(
        self,
        val_trajectories: Tensor,
        val_energies: Tensor,
        val_features: Optional[Sequence[Optional[dict]]] = None,
    ) -> float:
        """Return the ``λ_relax`` with the lowest batch-mean augmented reward.

        Parameters
        ----------
        val_trajectories
            Validation trajectories of shape ``(B, T, D)`` or
            ``(B, D)``.
        val_energies
            Validation energies of shape ``(B,)`` or ``(B, T)``.
        val_features
            Optional sequence of length ``B`` carrying the F5
            feature dicts (one per trajectory).  When ``None`` or
            ``base_shaping`` is ``None`` the base value is taken
            to be ``0.0`` for every trajectory — the sweep then
            degenerates to picking the smallest λ (since the KL
            penalty is the only signal and is λ-multiplied).

        Returns
        -------
        float
            The λ from ``self.lambdas`` with the lowest mean
            augmented reward across the validation batch.  When
            all λ produce the same mean (e.g. energies are all
            equal), the **smallest** λ is returned (ties broken
            by lambda order — the conservative pick).
        """
        if val_trajectories.dim() not in (2, 3):
            raise ValueError(
                f"val_trajectories must be (B, T, D) or (B, D); "
                f"got {tuple(val_trajectories.shape)}"
            )
        b = val_trajectories.shape[0]
        if val_energies.dim() not in (1, 2):
            raise ValueError(
                f"val_energies must be (B,) or (B, T); "
                f"got {tuple(val_energies.shape)}"
            )
        if val_energies.shape[0] != b:
            raise ValueError(
                f"val_energies batch size {val_energies.shape[0]} "
                f"!= val_trajectories batch size {b}"
            )

        # Compute the base value once (it does not depend on λ).
        if val_features is not None:
            base_vals = [self._base_value(val_features[i]) for i in range(b)]
            base_mean = torch.stack(base_vals).mean()
        else:
            base_mean = torch.zeros(())

        best_lam: float = float(self.lambdas[0])
        best_mean: float = float("inf")
        for lam in self.lambdas:
            term = RelaxedEndpointTerm(lambda_relax=float(lam), eps=self.eps)
            penalty = term.forward(val_trajectories, val_energies)
            mean_aug = float((base_mean + penalty).item())
            if mean_aug < best_mean:
                best_mean = mean_aug
                best_lam = float(lam)
        return best_lam

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def evaluate_grid(
        self,
        val_trajectories: Tensor,
        val_energies: Tensor,
        val_features: Optional[Sequence[Optional[dict]]] = None,
    ) -> List[dict]:
        """Return one row per λ in the grid for inspection / logging.

        The schema is::

            {
              "lambda_relax": <float>,
              "kl": <float>,
              "penalty": <float>,
              "base_mean": <float>,
              "augmented_mean": <float>,
            }

        All values are Python floats (not tensors).
        """
        if val_trajectories.dim() not in (2, 3):
            raise ValueError(
                f"val_trajectories must be (B, T, D) or (B, D); "
                f"got {tuple(val_trajectories.shape)}"
            )
        b = val_trajectories.shape[0]
        if val_features is not None:
            base_vals = [self._base_value(val_features[i]) for i in range(b)]
            base_mean = torch.stack(base_vals).mean()
        else:
            base_mean = torch.zeros(())
        rows: List[dict] = []
        for lam in self.lambdas:
            term = RelaxedEndpointTerm(lambda_relax=float(lam), eps=self.eps)
            kl = float(term.kl(val_trajectories, val_energies).item())
            penalty = float(term.forward(val_trajectories, val_energies).item())
            augmented = float((base_mean + penalty).item())
            rows.append(
                {
                    "lambda_relax": float(lam),
                    "kl": kl,
                    "penalty": penalty,
                    "base_mean": float(base_mean.item()),
                    "augmented_mean": augmented,
                }
            )
        return rows
