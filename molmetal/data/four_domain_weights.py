"""Per-domain :class:`torch.utils.data.WeightedRandomSampler` for Mol-Metal.

Formula (TODO-143 §2.5 Day 4)
-----------------------------
Given per-domain row counts ``N_d`` for d ∈ {0, 1, 2, 3}, define::

    N_target = max(N_d)
    w_d      = (N_target / N_d) ** (1 - alpha)

Each row in domain d is then assigned sample weight ``w_d``.  With:

* ``alpha = 0``    →  ``w_d = N_target / N_d``  → **inverse-frequency**.
                     Each row in any domain has the same *per-row* sampling
                     probability (because ``w_d * N_d = N_target``), so
                     per-domain share is **uniform** across the 4 domains
                     — every domain contributes the same number of effective
                     picks regardless of its size.
* ``alpha = 1``    →  ``w_d = 1``                → **per-row uniform**.
                     Every row has equal weight, so per-domain share is
                     proportional to ``N_d`` — DrugOOD (1.58M) dominates
                     and MetalCytoToxDB (5K) is barely represented.
* ``alpha = 0.5``  →  ``w_d = sqrt(N_target / N_d)``  → **soft balance**
                     (TODO-143 recommended default).  Per-row probability
                     for domain d is ``∝ sqrt(N_d)`` (sub-linear), giving
                     a partial upweighting of the small domains without
                     full inverse-frequency collapse.

Why (1 - alpha) and not alpha?
------------------------------
This matches the spec in the TODO: "alpha=0 → inverse-freq, alpha=1 → uniform".
The exponent is therefore (1 - alpha) so that alpha=1 collapses to uniform
weights (=1 for every row) and alpha=0 yields the canonical inverse-frequency
formula.

Honest caveat (paper §6 must ship)
----------------------------------
For very imbalanced domains (DrugOOD 1.58M vs MetalCytoToxDB 5K), even
``alpha = 0.5`` does **not** give equal domain representation per epoch —
DrugOOD still gets roughly ``sqrt(1.58M) / sqrt(5K) ≈ 18×`` more samples per
epoch than MetalCytoToxDB.  Equal mass per *domain* (not per row) is only
achieved at ``alpha = 0``.  We pick ``alpha = 0.5`` here because (a) it
matches the TODO-143 plan and (b) it is the standard soft-balancing default
in MTL literature; users who need exact per-domain balance should pass
``alpha=0``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from molmetal.data.four_domain_combined import (
    DOMAIN_BINDINGDB,
    DOMAIN_DRUGOOD,
    DOMAIN_METALCYTOTOX,
    DOMAIN_NCI60,
    FourDomainDataset,
)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------
def compute_domain_weights(
    domain_lengths: Dict[int, int],
    alpha: float = 0.5,
) -> Dict[int, float]:
    """Return ``w_d ∝ (N_target / N_d) ** (1 - alpha)`` for each domain.

    Parameters
    ----------
    domain_lengths : dict[int, int]
        Per-domain row counts (must include 0..3).
    alpha : float
        0 → inverse-freq, 1 → uniform, 0.5 → soft balancing.

    Returns
    -------
    dict[int, float]
        Per-domain weight ``w_d``.  All weights are positive.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not domain_lengths:
        raise ValueError("domain_lengths is empty")
    # Filter to domains with > 0 rows (avoids div-by-zero)
    pos = {d: n for d, n in domain_lengths.items() if int(n) > 0}
    if not pos:
        raise ValueError("All domain_lengths are zero")
    n_target = float(max(pos.values()))
    out: Dict[int, float] = {}
    for d, n in pos.items():
        n_d = float(n)
        # exponent (1 - alpha) so alpha=1 → uniform (w_d = 1)
        w_d = (n_target / n_d) ** (1.0 - alpha)
        out[d] = float(w_d)
    # Domains with zero rows get weight 0 (sampler will simply skip them)
    for d in domain_lengths:
        if d not in out:
            out[d] = 0.0
    return out


def expand_weights_to_per_row(
    dataset: FourDomainDataset,
    per_domain_weights: Dict[int, float],
) -> List[float]:
    """Build a flat per-row weight list for use with ``WeightedRandomSampler``.

    The returned list has length ``len(dataset)``; row ``i`` carries weight
    ``per_domain_weights[dataset._resolve_global_index(i)[0]]``.
    """
    weights: List[float] = []
    # Use the resolution machinery for consistency with __getitem__
    for d in range(4):
        w = float(per_domain_weights.get(d, 0.0))
        if w <= 0.0:
            # skip entirely — sampler cannot draw zero-weight rows
            d_len = dataset.domain_lengths()[d]
            continue
        d_len = dataset.domain_lengths()[d]
        weights.extend([w] * d_len)
    return weights


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------
def make_four_domain_sampler(
    dataset: FourDomainDataset,
    alpha: float = 0.5,
    num_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` over :class:`FourDomainDataset`.

    Parameters
    ----------
    dataset : FourDomainDataset
        The combined dataset to sample from.
    alpha : float
        0 → inverse-freq, 1 → uniform, 0.5 → soft balancing.
    num_samples : int | None
        Number of samples per epoch.  Default = ``len(dataset)``.
    generator : torch.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    WeightedRandomSampler
        Replacement-sampling sampler (PyTorch's ``WeightedRandomSampler``
        requires ``replacement=True``).  Already verified in
        :file:`molmetal/tests/test_four_domain_sampler.py`.
    """
    domain_lengths = dataset.domain_lengths()
    per_domain = compute_domain_weights(domain_lengths, alpha=alpha)
    per_row = expand_weights_to_per_row(dataset, per_domain)
    if not per_row:
        raise ValueError(
            "Cannot build sampler: all domains have weight 0.  "
            "Check that at least one domain has positive length."
        )
    weights_tensor = torch.as_tensor(per_row, dtype=torch.double)
    if num_samples is None:
        num_samples = len(per_row)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Diagnostics — useful in tests and notebooks
# ---------------------------------------------------------------------------
def expected_domain_mix(
    domain_lengths: Dict[int, int],
    alpha: float,
) -> Dict[int, float]:
    """Return expected sampling probabilities per domain (as a fraction).

    Note
    ----
    Expected per-row probability ``p_d ∝ w_d * N_d ∝ N_target**(1-alpha) *
    N_d**alpha``.  This is normalized so the entries sum to 1.
    """
    per_domain = compute_domain_weights(domain_lengths, alpha=alpha)
    # Per-row probability ∝ w_d * N_d
    raw: Dict[int, float] = {}
    for d, w in per_domain.items():
        n = float(domain_lengths.get(d, 0))
        raw[d] = float(w) * n
    total = sum(raw.values())
    if total <= 0:
        return {d: 0.0 for d in per_domain}
    return {d: v / total for d, v in raw.items()}


__all__ = [
    "compute_domain_weights",
    "expand_weights_to_per_row",
    "make_four_domain_sampler",
    "expected_domain_mix",
]
