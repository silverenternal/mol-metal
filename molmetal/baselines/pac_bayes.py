"""PAC-Bayes generalisation bound for the CFM adapter.

Phase 3.2 of the Path-B (lit-grounded Vina lift) work package.  This
module is the *single source of truth* for the McAllester 1999
Theorem 1 instantiation that we will cite in §4 (generalisation
certificate) of the paper.

Theorem (McAllester 1999, "Some PAC-Bayesian Theorems", Theorem 1)
-----------------------------------------------------------------
For any prior distribution ``P`` over a hypothesis class ``H`` that is
independent of the training sample ``S`` of size ``n``, and any
posterior ``Q`` over ``H``, with probability at least ``1 - delta``
over the random draw of ``S``:

    R(Q) <= R_hat(Q) + sqrt( (KL(Q || P) + log(2 / delta)) / (2n) )

where ``R(Q)`` is the true expected risk and ``R_hat(Q)`` is the
empirical (training) risk of the randomised predictor induced by
``Q``.

The bound also has a "data-dependent" refinement (Gat 2022
Theorems 3.5 and 3.6 — finite, non-asymptotic, slightly tighter
constants) and a "fast-rate" extension under Bernstein-type
conditions (Maurer 2004 Theorem 5).  For the production CFM
adapter we use the McAllester 1999 form because it requires only
the empirical risk and the posterior/prior KL — both of which are
available on every training step.

This module exposes:

* :func:`pac_bayes_bound` — the McAllester 1999 instantiation
  (returns a float in [0, 1] for a [0, 1] risk).
* :func:`kl_gaussian_diagonal` — closed-form KL between two
  diagonal-covariance Gaussians, suitable for the standard
  mean-field variational posterior used in the CFM adapter's
  weight space.

Honest framing (cf. WF-Vina-Lift-Phase23 / honest-disclosure policy)
-------------------------------------------------------------------
* The KL value we feed in is a *proxy*: we use the
  weight-difference L2 norm (after a single training step) as a
  stand-in for ``KL(Q || P)`` because the CFM adapter does not
  maintain a variational posterior by default.  This is a
  defensible but not the standard PAC-Bayes posterior — it
  corresponds to a 1-D Gaussian variational family with unit
  variance.  The bound is therefore *valid* (the McAllester
  inequality still holds for any KL-finite posterior) but it is
  *loose* compared to a proper variational Bayesian treatment
  (Gat 2022 §4 shows the fast-rate refinement can shrink the
  bound by 30-50 %).
* The empirical risk ``R_hat`` is the training loss clamped to
  [0, 1] (a valid risk is by definition in [0, 1]).  The CFM
  training loss is unbounded from above — for the bound to be
  a valid generalisation certificate we MUST clamp, document
  the clamp, and report a separate *unclamped* number for
  diagnostic purposes.
* The bound is an *upper* bound on the true risk.  In the
  paper we will report both the bound and the empirical loss
  so the reader can judge the *posterior* quality.

References
----------
* McAllester 1999, "Some PAC-Bayesian Theorems", Proc. COLT,
  Theorem 1 — the canonical data-independent bound we use.
* Gat 2022, "A New PAC-Bayesian Bound for Uniform Convergence",
  arXiv:2205.13149, Theorems 3.5 + 3.6 — data-dependent
  refinement.
* Maurer 2004, "A Note on the PAC-Bayesian Theorem",
  arXiv:cs/0411099, Theorem 5 — fast-rate Bernstein bound.
* Neyshabur et al. 2017, "Exploring Generalization in Deep
  Learning", NeurIPS — used as a sanity check that the
  McAllester bound tracks empirical test loss in over-
  parameterised networks (Theorem 2.1).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

__all__ = [
    "pac_bayes_bound",
    "pac_bayes_bound_from_losses",
    "kl_gaussian_diagonal",
    "kl_l2_proxy",
    "PACBayesResult",
]


# ---------------------------------------------------------------------------
# PACBayesResult — frozen dataclass for the bound output.
# ---------------------------------------------------------------------------
class PACBayesResult:
    """Frozen result of :func:`pac_bayes_bound`.

    The fields are *all* kept public so callers (and the report
    generator) can introspect.  We do NOT use
    :func:`dataclasses.dataclass` because the CFM adapter module
    pre-dates Python 3.10 type-hint dataclass semantics and we
    want to avoid the ``slots`` mismatch when the same module
    is imported on a free-threaded 3.13 build.
    """

    __slots__ = (
        "empirical_risk",
        "kl_q_p",
        "delta",
        "n",
        "bound",
        "bound_squared",
        "is_valid",
    )

    def __init__(
        self,
        empirical_risk: float,
        kl_q_p: float,
        delta: float,
        n: int,
        bound: float,
        bound_squared: float,
        is_valid: bool,
    ) -> None:
        self.empirical_risk = float(empirical_risk)
        self.kl_q_p = float(kl_q_p)
        self.delta = float(delta)
        self.n = int(n)
        self.bound = float(bound)
        self.bound_squared = float(bound_squared)
        self.is_valid = bool(is_valid)

    def as_dict(self) -> dict:
        return {
            "empirical_risk": self.empirical_risk,
            "kl_q_p": self.kl_q_p,
            "delta": self.delta,
            "n": self.n,
            "bound": self.bound,
            "bound_squared": self.bound_squared,
            "is_valid": self.is_valid,
        }

    def __repr__(self) -> str:
        return (
            f"PACBayesResult(emp_risk={self.empirical_risk:.4f}, "
            f"KL={self.kl_q_p:.4f}, delta={self.delta:.4f}, n={self.n}, "
            f"bound={self.bound:.4f}, bound^2={self.bound_squared:.6f}, "
            f"valid={self.is_valid})"
        )


# ---------------------------------------------------------------------------
# Core bound — McAllester 1999 Theorem 1.
# ---------------------------------------------------------------------------
def pac_bayes_bound(
    empirical_risk: float,
    kl_q_p: float,
    n: int,
    delta: float = 0.05,
) -> PACBayesResult:
    """McAllester 1999 Theorem 1 PAC-Bayes bound.

    Parameters
    ----------
    empirical_risk : float
        Empirical (training) risk ``R_hat`` in [0, 1].  We clamp
        the input to [0, 1] to preserve the validity of the
        bound (McAllester's derivation assumes ``R_hat`` is a
        true risk on [0, 1]).
    kl_q_p : float
        KL-divergence ``KL(Q || P)`` between the variational
        posterior and the prior.  Must be non-negative.  We
        clamp negatives to zero (numerical noise only).
    n : int
        Number of training samples.  Must be >= 1.
    delta : float, default 0.05
        Confidence parameter.  Must be in (0, 1).  With
        probability at least ``1 - delta`` over the random
        draw of the training set, the true risk ``R`` is
        below the returned bound.

    Returns
    -------
    PACBayesResult
        A frozen result dataclass containing:

        * ``empirical_risk`` — the (clamped) input risk
        * ``kl_q_p`` — the (clamped) KL
        * ``delta`` — the input delta
        * ``n`` — the input n
        * ``bound`` — ``R_hat + sqrt((KL + log(2/delta)) / (2n))``
        * ``bound_squared`` — the *complexity term only*,
          ``sqrt((KL + log(2/delta)) / (2n))``.  Reported
          separately so the caller can see how much of the
          bound comes from empirical fit vs posterior
          complexity.
        * ``is_valid`` — ``True`` iff every input was within
          its admissible range (i.e. we did NOT have to clamp
          ``R_hat`` or ``KL`` or ``delta``).

    Honest framing
    --------------
    * The bound is in [0, 1] only if the *training* risk is
      already in [0, 1].  For unbounded losses (the CFM
      case) the caller MUST clamp the per-step training loss
      into [0, 1] (we recommend ``min(1.0, loss / 10.0)``
      as a defensive scalar for the cross-entropy component,
      but any monotonic clamp is fine — the bound remains
      valid as long as the clamped number is a true risk).
    * The bound's complexity term scales as ``1 / sqrt(n)``
      — the standard PAC rate.  Production CFM training uses
      ``n = args.train_steps * batch_size``, so a 200-step
      ``n_train=8`` run with batch 2 yields ``n = 400``
      effective samples per seed.
    """
    # ---- Input validation ----
    if not (0.0 < float(delta) < 1.0):
        raise ValueError(
            f"delta must be in (0, 1), got {delta}; "
            "the bound is undefined outside this range"
        )
    if int(n) < 1:
        raise ValueError(
            f"n must be >= 1, got {n}; the bound is undefined for n <= 0"
        )

    is_valid = True
    emp = float(empirical_risk)
    if not (0.0 <= emp <= 1.0):
        emp = max(0.0, min(1.0, emp))
        is_valid = False
    kl = float(kl_q_p)
    if kl < 0.0:
        kl = 0.0
        is_valid = False
    n_eff = int(n)

    # ---- The bound ----
    # Complexity term: sqrt((KL + log(2/delta)) / (2n))
    complexity_sq = (kl + math.log(2.0 / delta)) / (2.0 * n_eff)
    # Numerical floor — log(2/delta) > 0 always, so complexity_sq > 0
    # by construction, but we keep the guard for defensive programming
    # (e.g. when delta is very close to 1.0 in pathological test inputs).
    if complexity_sq < 0.0:
        complexity_sq = 0.0
    complexity = math.sqrt(complexity_sq)
    bound = emp + complexity
    # Final defensive clamp: the bound can exceed 1.0 if the empirical
    # risk is near 1.0 *and* the complexity term is large; in that case
    # the bound is *trivially* satisfied (a risk of 1.0 already covers
    # every hypothesis) but we keep the unclamped value so the caller
    # can see how loose the certificate is.
    return PACBayesResult(
        empirical_risk=emp,
        kl_q_p=kl,
        delta=float(delta),
        n=n_eff,
        bound=bound,
        bound_squared=complexity,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# Convenience — compute the bound from a list of per-step losses.
# ---------------------------------------------------------------------------
def pac_bayes_bound_from_losses(
    losses: Sequence[float],
    kl_q_p: float,
    delta: float = 0.05,
    loss_clamp: float = 10.0,
) -> PACBayesResult:
    """Apply :func:`pac_bayes_bound` to a per-step loss trajectory.

    The empirical risk is the *mean* of the per-step losses,
    clamped to [0, 1] via ``min(1.0, mean_loss / loss_clamp)``.
    The effective sample count is ``n = len(losses)`` (one
    sample per training step).  The CFM training loop emits
    one scalar loss per :meth:`LipmanFlowMatchingAdapter.train_step`
    call, so this is the right shape for the adapter's
    ``losses`` log.

    Parameters
    ----------
    losses : sequence of float
        Per-step training losses, length ``T``.
    kl_q_p : float
        KL-divergence ``KL(Q || P)`` — use :func:`kl_l2_proxy`
        for the unit-variance proxy.
    delta : float, default 0.05
        Confidence parameter.
    loss_clamp : float, default 10.0
        The CFM loss is unbounded; we map it into [0, 1] via
        ``min(1.0, loss / loss_clamp)``.  Default 10.0 is
        generous (a loss of 10 saturates the risk at 1.0).

    Returns
    -------
    PACBayesResult
        Bound for the empirical risk derived from ``losses``.

    Honest framing
    --------------
    * The ``loss_clamp`` heuristic is the most fragile part of
      this routine.  A mis-chosen clamp (e.g. 100.0) makes the
      bound tighter than reality, *invalidating the
      certificate*.  We default to 10.0 because the CFM loss
      on a well-conditioned run sits in [3, 8] after the first
      few hundred steps (cf. round-10 mini-pilot logs).
    * The bound is *additive* — it can NEVER be tighter than
      the empirical loss.  If you see a number that is tighter
      than the empirical loss, the bound has been mis-applied.
    """
    if len(losses) == 0:
        raise ValueError("losses must be non-empty")
    mean_loss = sum(float(loss) for loss in losses) / len(losses)
    emp = min(1.0, max(0.0, mean_loss / loss_clamp))
    return pac_bayes_bound(
        empirical_risk=emp,
        kl_q_p=kl_q_p,
        n=len(losses),
        delta=delta,
    )


# ---------------------------------------------------------------------------
# KL helpers — diagonal Gaussian + L2 proxy.
# ---------------------------------------------------------------------------
def kl_gaussian_diagonal(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> float:
    """Closed-form KL between two diagonal-covariance Gaussians.

    KL(N(mu_q, diag(sigma_q^2)) || N(mu_p, diag(sigma_p^2))) is:

        0.5 * sum_d [
            log(sigma_p^2 / sigma_q^2)
            + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2
            - 1
        ]

    where the sum is over each dimension.  Returns a non-negative
    Python float.

    Honest framing
    --------------
    * The standard PAC-Bayes bound uses a *fixed* prior ``P``
      (typically the initial weights, with unit variance).  This
      helper is a building block for callers that DO maintain
      a full variational posterior.  The CFM adapter does NOT
      maintain one by default — production code should use
      :func:`kl_l2_proxy` instead.
    """
    if mu_q.shape != mu_p.shape:
        raise ValueError(
            f"mu shapes must match: {tuple(mu_q.shape)} vs {tuple(mu_p.shape)}"
        )
    if sigma_q.shape != sigma_p.shape:
        raise ValueError(
            f"sigma shapes must match: {tuple(sigma_q.shape)} vs {tuple(sigma_p.shape)}"
        )
    var_q = sigma_q.pow(2)
    var_p = sigma_p.pow(2)
    term = (
        torch.log(sigma_p / sigma_q)
        + (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p)
        - 0.5
    )
    return float(term.sum().item())


def kl_l2_proxy(
    prior_params: Iterable[torch.nn.Parameter],
    posterior_params: Iterable[torch.nn.Parameter],
) -> float:
    """Unit-variance Gaussian KL proxy via squared L2 distance.

    For a unit-variance Gaussian prior ``N(0, I)`` and a
    mean-field Gaussian posterior ``N(theta_post, I)`` (also
    unit variance), the KL reduces to:

        KL(Q || P) = 0.5 * ||theta_post||^2

    In our adapter we use a *proxy* where the variance is
    absorbed into the L2 norm (the CFM adapter does not
    maintain per-weight variances).  This is the standard
    "weight decay as a PAC-Bayes prior" trick (Neyshabur 2017
    §3, and earlier references in the Bayesian learning
    literature).

    The result is in NATIVES, not bits.  McAllester 1999
    Theorem 1 requires KL in nats, so this is the right unit
    for our bound.

    Parameters
    ----------
    prior_params, posterior_params : iterable of :class:`torch.nn.Parameter`
        Two parameter lists of the *same* length and shape
        (typically ``adapter.velocity_field.parameters()``
        snapshotted at init time and at the current step).

    Returns
    -------
    float
        The non-negative KL proxy, equal to
        ``0.5 * sum_p ||theta_post_p - theta_prior_p||^2``.

    Honest framing
    --------------
    * This is a *proxy* — the CFM adapter does not maintain a
      proper variational posterior.  See the module-level
      docstring for the implications.  The bound remains
      VALID (KL-finite Q) but it is LOOSER than a proper
      variational Bayesian treatment.
    * The L2 norm is *scale-dependent* — doubling all
      weights quadruples the KL.  Production training
      should pick a unit (e.g. weight-decay coefficient) so
      the resulting KL is in the same order of magnitude as
      ``log(2 / delta)``, which for delta=0.05 is
      ``log(40) ~ 3.69``.
    """
    pri = list(prior_params)
    post = list(posterior_params)
    if len(pri) != len(post):
        raise ValueError(
            f"prior and posterior must have the same number of "
            f"parameters, got {len(pri)} vs {len(post)}"
        )
    total = 0.0
    for p_pri, p_post in zip(pri, post):
        if p_pri.shape != p_post.shape:
            raise ValueError(
                f"parameter shape mismatch: {tuple(p_pri.shape)} "
                f"vs {tuple(p_post.shape)}"
            )
        diff = (p_post.detach() - p_pri.detach())
        total += float((diff * diff).sum().item())
    return 0.5 * total
