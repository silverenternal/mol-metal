#!/usr/bin/env python
"""OPT-5 / Wf-A78 — training-time ring-template prior.

WHY THIS EXISTS
---------------
The OPT-2 ring audit (docs/CFM_SIX_ROUND_CONCLUSION_2026_10_13.md §6)
measured the defect as ring *size*, not ring *count*:

    | metric                  | generated | real   |
    |-------------------------|-----------|--------|
    | 5/6-membered rings      |    1.85%  | 96.92% |
    | 3-membered rings        |   89.8%   |  0.5%  |

A better bond *classifier* cannot fix this: it can relabel an edge, it
cannot make five atoms close a five-ring.  This module adds the missing
supervision — a prior on the ring-size profile of the predicted bond
graph, matched against the molecule's own truth profile.

HOW THE COUNT IS MADE DIFFERENTIABLE
------------------------------------
Exact ring perception is discrete and has no gradient.  The prior uses
the non-backtracking (Hashimoto) operator B on directed edges:

    B[(i->j), (j->l)] = p_jl      for l != i,      0 otherwise
    #cycles of size k = trace(B ** k) / (2k)

Every closed non-backtracking walk of length k IS a k-cycle, so unlike
trace(A ** k) there is no back-and-forth contamination at even k: a
6-ring counts as a 6-ring, not as an inflated 4- and 6-walk soup.  (A
closed 6-walk can still go around a TRIANGLE twice; that contribution is
subtracted exactly, since each triangle adds 0.5 to trace(B^6)/12.)

Value and gradient come from different graphs, on purpose:

    value    = cycle counts on the HARD graph (p > 0.5)
    gradient = d(cycle counts on the SOFT graph)/d(logits)

MEASURED 2026-10-18 (50-step CPU smoke): counting on the soft graph
alone gives ring_prior_term = 7.99e7 against bond_ce = 1.0 and
grad_norm = 6e9 -- every 5%-probability noise edge participates in the
high-order walks and the count explodes combinatorially.  Counting on
the hard graph alone gives an exact zero gradient almost everywhere
(cycle counts are piecewise constant).  Hard-value + soft-gradient is
the only combination that is both bounded and trainable.

DEFAULT-OFF
-----------
weight <= 0 returns an exact torch.zeros(()) without touching the inputs,
so a run that does not set flow.ring_template_weight is bit-exact with
the pre-Wf-A78 trainer (AGENTS.md §5.11).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RingTemplatePriorConfig:
    """Static knobs for RingTemplatePrior.

    sizes: ring sizes whose count is matched.  3-6 covers the measured
        defect (3-rings over-produced, 5/6-rings missing).
    size_weights: per-size weight on the squared count error.  Uniform
        by default: the target is the molecule's own profile, so no size
        needs to be privileged a priori.
    """

    sizes: Tuple[int, ...] = (3, 4, 5, 6)
    size_weights: Optional[Dict[int, float]] = None
    eps: float = 1e-6

    def weight_for(self, k: int) -> float:
        if self.size_weights is None:
            return 1.0
        return float(self.size_weights.get(k, 1.0))


def _cycle_counts(
    bond_prob: torch.Tensor,
    pair_index: torch.Tensor,
    sizes: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Cycle counts per size via the non-backtracking operator.

    Returns one entry per requested size (zeros included, so the caller
    can combine a hard-value pass with a soft-gradient pass).
    """
    src = pair_index[0].to(bond_prob.device)
    dst = pair_index[1].to(bond_prob.device)
    # Directed edge list: e = (src -> dst), e + E = (dst -> src).
    d_src = torch.cat([src, dst])
    d_dst = torch.cat([dst, src])
    p_dir = torch.cat([bond_prob, bond_prob])

    # B[a, b] is non-zero iff b starts where a ends (d_dst[a] == d_src[b])
    # and b does not immediately undo a (d_dst[b] != d_src[a]).
    continues = d_dst.unsqueeze(1) == d_src.unsqueeze(0)  # (2E, 2E)
    backtracks = d_dst.unsqueeze(0) == d_src.unsqueeze(1)  # (2E, 2E)
    mask = continues & ~backtracks
    b_mat = mask.to(bond_prob.dtype) * p_dir.unsqueeze(0)

    raw: Dict[int, torch.Tensor] = {}
    power = b_mat
    for k in range(2, max(sizes) + 1):
        power = power @ b_mat
        raw[k] = torch.diagonal(power).sum() / (2.0 * k)
    if 6 in raw and 3 in raw:
        # A closed non-backtracking 6-walk can go around a TRIANGLE twice;
        # that is not a 6-cycle.  Each triangle contributes exactly 0.5 to
        # trace(B^6)/12, and no other repetition fits in 6 steps (2-cycles
        # do not exist, 4+2 is not non-backtracking), so this subtraction
        # is exact for simple graphs.
        raw[6] = (raw[6] - 0.5 * raw[3]).clamp(min=0.0)
    return {k: raw[k] for k in sizes}


def _as_pair_index(pair_index) -> torch.Tensor:
    """Accept (2, E) tensors and (i_idx, j_idx) tuples alike."""
    if isinstance(pair_index, (tuple, list)) and len(pair_index) == 2:
        return torch.stack([pair_index[0], pair_index[1]], dim=0)
    if torch.is_tensor(pair_index):
        if pair_index.dim() != 2 or pair_index.shape[0] != 2:
            raise ValueError(
                "pair_index must be (2, E) or a 2-tuple of (E,) tensors, "
                f"got shape {tuple(pair_index.shape)}"
            )
        return pair_index
    raise TypeError(f"unsupported pair_index type: {type(pair_index)!r}")


class RingTemplatePrior:
    """Differentiable ring-size prior on the bond head's soft graph."""

    def __init__(
        self,
        weight: float = 0.0,
        config: Optional[RingTemplatePriorConfig] = None,
    ) -> None:
        self.weight = float(weight)
        self.config = config or RingTemplatePriorConfig()

    def compute(
        self,
        bond_logits: Optional[torch.Tensor],
        pair_index,
        n_atoms: int,
        truth_bonds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return weight * sum_k w_k (c_pred_k - c_truth_k)^2 for one molecule.

        Returns an exact zero tensor when:
          * weight <= 0 (default-off, bit-exact),
          * any input is missing,
          * the truth graph has no cycles (no profile to match; inventing
            one would penalise a correct output on an acyclic molecule).
        """
        if device is None:
            device = bond_logits.device if bond_logits is not None else torch.device("cpu")
        zero = torch.zeros((), device=device)
        if self.weight <= 0.0:
            return zero
        if bond_logits is None or bond_logits.numel() == 0:
            return zero
        if truth_bonds is None or truth_bonds.numel() == 0:
            return zero
        if int(n_atoms) < min(self.config.sizes):
            return zero

        pair_index = _as_pair_index(pair_index).to(bond_logits.device)
        if pair_index.shape[1] != bond_logits.shape[0]:
            return zero

        # Bond probability = 1 - P(NO_BOND).  Class 0 is NO_BOND in the
        # 5-class YuelBond head; this is the only assumption about the
        # head's label order and it matches BOND_LABELS.
        probs = F.softmax(bond_logits.float(), dim=-1)
        if probs.shape[-1] < 2:
            return zero
        soft = 1.0 - probs[:, 0]
        hard = (soft > 0.5).to(soft.dtype)

        soft_counts = _cycle_counts(soft, pair_index, self.config.sizes)
        hard_counts = _cycle_counts(hard, pair_index, self.config.sizes)

        truth = truth_bonds.to(bond_logits.device)
        if truth.dim() != 2 or truth.shape[0] != 2:
            return zero
        truth_counts = _cycle_counts(
            torch.ones(truth.shape[1], device=bond_logits.device, dtype=soft.dtype),
            truth,
            self.config.sizes,
        )
        if float(sum(v.detach() for v in truth_counts.values())) <= self.config.eps:
            return zero

        import os as _os
        if _os.environ.get("WFA78_DEBUG"):
            print("[wfa78] n_atoms", int(n_atoms), "E", pair_index.shape[1],
                  "max_idx", int(pair_index.max()), "bond_p>0.5", int(hard.sum()),
                  "soft_counts", {k: round(float(v), 2) for k, v in soft_counts.items()},
                  "hard_counts", {k: round(float(v), 2) for k, v in hard_counts.items()},
                  "truth_counts", {k: round(float(v), 2) for k, v in truth_counts.items()})
        # Soft counts are normalised by the candidate-edge count before
        # they enter the gradient path.  They are NOT normalised in the
        # value (the value is the hard count), so this only sets the
        # gradient scale: without it, a 53-atom cloud with 874 candidates
        # produced grad_norm 6.9e5 against a bond_ce-only grad_norm of
        # ~1.5 (MEASURED 2026-10-18), i.e. the prior would dominate the
        # step direction even at a small weight.
        denom = max(int(pair_index.shape[1]), 1)

        loss = zero
        for k in self.config.sizes:
            # Value from the hard graph, gradient from the soft graph.
            soft_scaled = soft_counts[k] / denom
            c = hard_counts[k] + soft_scaled - soft_scaled.detach()
            # log1p keeps the term bounded when the predicted soft graph is
            # dense (MEASURED 6.3e4 six-walks at epoch 0 on a 53-atom
            # cloud) and self-limits the gradient: d log1p(c)/dp =
            # (dc/dp)/(1+c), which shrinks as the count grows.
            loss = loss + self.config.weight_for(k) * (
                torch.log1p(c) - torch.log1p(truth_counts[k])
            ) ** 2
        return self.weight * loss

    def compute_batch(
        self,
        items: Iterable[Tuple[Optional[torch.Tensor], object, int, Optional[torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        """Mean prior over (logits, pair_index, n_atoms, truth_bonds) items."""
        zero = torch.zeros((), device=device)
        if self.weight <= 0.0:
            return zero
        terms = [
            self.compute(logits, pair_index, n_atoms, truth, device=device)
            for logits, pair_index, n_atoms, truth in items
        ]
        if not terms:
            return zero
        return torch.stack(terms).mean()
