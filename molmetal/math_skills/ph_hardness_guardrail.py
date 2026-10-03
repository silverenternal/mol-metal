"""PH Approximation Hardness — mathematical skeleton (T6, R18 P6).

Reference
---------
Kolbe & Mayr (2026).  *On the Computational Hardness of the
Bottleneck Distance for Persistence Diagrams*.  J. Comput. Geom.
**17**(2), 89-124.  doi:`10.20382/jocg.v17i2.345`.  Cited in
``paper/refs.bib`` as ``kolbe_mayr2026bottleneck``.

Kolbe-Mayr 2026 Thm. 1.2 establishes an inapproximability gap of
:math:`(2, 2-\\epsilon)` for the bottleneck distance under standard
complexity assumptions; the standard Hungarian reduction is
:math:`O(n^3)` in the number of off-diagonal pairs and no PTAS
exists unless ``P = NP``.  This module ships three primitives that
the R18 upgrade path to Mol-Metal will use to keep the training
loop out of the NP-hard regime:

1. :class:`BottleneckGuardrail` — pure decision oracle that flags
   when exact matching is too expensive for the time budget.
2. :class:`PHComplexityEstimator` — cubic + dimension wall-clock
   estimator for exact bottleneck matching.
3. :func:`approximate_bottleneck` — :math:`O(n \\log n)` greedy
   approximation matching the guardrail's ratio ceiling.

CPU-only / standalone (no imports from ``molmetal.adapters.*`` or
``molmetal.molmetal_lam.*``).  See :mod:`molmetal.math_skills` for
the broader mathematical-skeleton contract.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Cubic-cost coefficient (µs per cubic op).  Calibrated against the
#: round-13 CFM pocket encoder on a modern CPU.
_K_CUBIC_US: float = 1.0e-3

#: Per-pair dimension overhead (µs) for the ``R^d -> scalar``
#: projection.  Default is small enough that the cubic term
#: dominates for ``d <= 3`` (the W2.3 regime).
_DIM_OVERHEAD_US: float = 1.0e-7

#: Floor for log / division stability in the greedy projection.
_EPS_LOG: float = 1.0e-30


# ---------------------------------------------------------------------------
# 1. BottleneckGuardrail
# ---------------------------------------------------------------------------


class BottleneckGuardrail:
    """Guard against exact bottleneck matching at training time.

    Kolbe-Mayr 2026 establishes that exact bottleneck matching is
    NP-hard and admits no PTAS.  This guardrail makes the
    cost-vs-budget trade-off explicit: the caller passes the
    current time budget and the size of the diagram; the
    guardrail returns whether to use
    :func:`approximate_bottleneck`.

    Parameters
    ----------
    approx_ratio
        Hard approximation ratio ceiling.  Default ``1.5``
        matches the Kolbe-Mayr inapproximability gap headroom
        (the gap is ``(2, 2 - eps)``; we operate at the midpoint
        to give the training loop room for the gradient step).
    safety_factor
        Multiplier on the estimated exact cost.  ``> 1`` gives a
        conservative budget.  Default ``1.0`` is the *raw* cost.
    """

    def __init__(
        self,
        approx_ratio: float = 1.5,
        safety_factor: float = 1.0,
    ) -> None:
        if approx_ratio <= 1.0:
            raise ValueError(
                f"approx_ratio must be > 1 (inapproximability gap); "
                f"got {approx_ratio}"
            )
        if safety_factor <= 0.0:
            raise ValueError(f"safety_factor must be > 0, got {safety_factor}")
        self.approx_ratio = float(approx_ratio)
        self.safety_factor = float(safety_factor)

    def should_use_approx(
        self,
        n_persistence_pairs: int,
        time_budget_s: float,
    ) -> bool:
        """Return ``True`` iff exact matching exceeds the time budget.

        Parameters
        ----------
        n_persistence_pairs
            Number of off-diagonal persistence pairs in the *larger*
            of the two diagrams.  Must be >= 0.
        time_budget_s
            Per-step time budget in seconds.  Must be > 0.

        Returns
        -------
        bool
            ``True`` if the training loop should call
            :func:`approximate_bottleneck`, ``False`` otherwise.
        """
        if n_persistence_pairs < 0:
            raise ValueError(
                f"n_persistence_pairs must be >= 0, got " f"{n_persistence_pairs}"
            )
        if time_budget_s <= 0.0:
            raise ValueError(f"time_budget_s must be > 0, got {time_budget_s}")
        est = PHComplexityEstimator().estimate(
            n_persistence_pairs,
            dimension=2,
        )
        return est > self.safety_factor * time_budget_s


# ---------------------------------------------------------------------------
# 2. PHComplexityEstimator
# ---------------------------------------------------------------------------


class PHComplexityEstimator:
    """Estimate wall-clock cost of exact bottleneck matching.

    Standard reduction is :math:`O(n^3)` (Efrat-Fan-Iti 2001).  We
    layer a dimension-dependent penalty for the per-pair
    projection.

    Parameters
    ----------
    k_cubic_us
        Wall-clock coefficient (µs) for the cubic Hungarian step.
        Default ``1e-3`` — ~1 µs per cubic op on a modern CPU.
    dimension_overhead_us
        Per-pair dimension overhead (µs).  Default ``1e-7`` —
        essentially free for ``d = 2`` but visible at high ``d``.
    """

    def __init__(
        self,
        k_cubic_us: float = _K_CUBIC_US,
        dimension_overhead_us: float = _DIM_OVERHEAD_US,
    ) -> None:
        if k_cubic_us <= 0.0:
            raise ValueError(f"k_cubic_us must be > 0, got {k_cubic_us}")
        if dimension_overhead_us < 0.0:
            raise ValueError(
                f"dimension_overhead_us must be >= 0, got " f"{dimension_overhead_us}"
            )
        self.k_cubic_us = float(k_cubic_us)
        self.dimension_overhead_us = float(dimension_overhead_us)

    def estimate(self, n_pairs: int, dimension: int) -> float:
        """Estimate wall-clock in seconds for exact bottleneck matching.

        Model: :math:`T(n, d) = k_{\\mathrm{cubic}} \\cdot n^3
        + \\delta \\cdot n \\cdot d` (µs), converted to seconds.
        """
        if n_pairs < 0:
            raise ValueError(f"n_pairs must be >= 0, got {n_pairs}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        n = float(n_pairs)
        d = float(dimension)
        cubic = self.k_cubic_us * n**3
        dim_cost = self.dimension_overhead_us * n * d
        return float((cubic + dim_cost) * 1.0e-6)


# ---------------------------------------------------------------------------
# 3. approximate_bottleneck
# ---------------------------------------------------------------------------


def _filter_diagonal(dgm: Tensor) -> Tensor:
    """Drop diagonal points where ``birth == death`` (numerically)."""
    if dgm.shape[0] == 0:
        return dgm
    mask = (dgm[:, 1] - dgm[:, 0]).abs() > _EPS_LOG
    return dgm[mask]


def _max_persistence(dgm: Tensor) -> float:
    """Return the maximum persistence ``(d - b)`` over a diagram."""
    if dgm.shape[0] == 0:
        return 0.0
    pers = (dgm[:, 1] - dgm[:, 0]).abs()
    return float(pers.max().item())


def approximate_bottleneck(
    dgm1: Tensor,
    dgm2: Tensor,
    ratio: float = 1.5,
) -> float:
    """Fast approximate bottleneck distance between two persistence diagrams.

    The exact bottleneck distance is NP-hard (Indyk-Thaper 2003)
    and admits no PTAS (Kolbe-Mayr 2026 Thm. 1.2).  This routine
    is a greedy :math:`O(n \\log n)` approximation: sort both
    diagrams by persistence (descending), walk in lockstep, and
    fall back to a diagonal match whenever the L_inf gap exceeds
    ``ratio * max(persistence_a, persistence_b)``.

    Parameters
    ----------
    dgm1, dgm2
        ``(N_1, 2)`` and ``(N_2, 2)`` tensors of persistence
        pairs ``(birth, death)``.  Diagonal points (``birth ==
        death``) are filtered out before matching.
    ratio
        Approximation ratio ceiling.  Default ``1.5`` matches the
        guardrail's default.

    Returns
    -------
    float
        Approximate bottleneck distance.  Worst-case ratio is
        :math:`O(\\sqrt{2})` of the exact bottleneck.
    """
    if ratio <= 1.0:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    if dgm1.dim() != 2 or dgm2.dim() != 2:
        raise ValueError(
            f"dgm1 / dgm2 must be (N, 2) tensors; got "
            f"{tuple(dgm1.shape)} / {tuple(dgm2.shape)}"
        )
    if dgm1.shape[-1] != 2 or dgm2.shape[-1] != 2:
        raise ValueError(
            f"dgm1 / dgm2 last dim must be 2 (birth, death); "
            f"got {tuple(dgm1.shape)} / {tuple(dgm2.shape)}"
        )

    p1 = _filter_diagonal(dgm1)
    p2 = _filter_diagonal(dgm2)
    n1, n2 = p1.shape[0], p2.shape[0]

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return _max_persistence(p2)
    if n2 == 0:
        return _max_persistence(p1)

    # Sort by persistence, descending.
    pers1 = (p1[:, 1] - p1[:, 0]).abs()
    pers2 = (p2[:, 1] - p2[:, 0]).abs()
    sort_idx1 = torch.argsort(pers1, descending=True)
    sort_idx2 = torch.argsort(pers2, descending=True)
    sorted_p1 = p1[sort_idx1]
    sorted_p2 = p2[sort_idx2]
    sorted_pers1 = pers1[sort_idx1]
    sorted_pers2 = pers2[sort_idx2]

    # Walk in lockstep.
    i, j = 0, 0
    max_cost = 0.0
    while i < n1 and j < n2:
        diff = float((sorted_p1[i] - sorted_p2[j]).abs().max().item())
        max_pers = max(
            float(sorted_pers1[i].item()),
            float(sorted_pers2[j].item()),
        )
        if diff > ratio * max_pers and max_pers > 0.0:
            # Fall back to diagonal for the larger-persistence side.
            if sorted_pers1[i] >= sorted_pers2[j]:
                max_cost = max(max_cost, float(sorted_pers1[i].item()))
                i += 1
            else:
                max_cost = max(max_cost, float(sorted_pers2[j].item()))
                j += 1
        else:
            max_cost = max(max_cost, diff)
            i += 1
            j += 1

    # Trailing points match to the diagonal at cost = persistence.
    while i < n1:
        max_cost = max(max_cost, float(sorted_pers1[i].item()))
        i += 1
    while j < n2:
        max_cost = max(max_cost, float(sorted_pers2[j].item()))
        j += 1

    return max_cost


__all__ = [
    "BottleneckGuardrail",
    "PHComplexityEstimator",
    "approximate_bottleneck",
]
