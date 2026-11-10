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
the pre-Wf-A78 trainer (~~AGENTS.md §5.11~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31).

Wf-A107 (2026-10-21) — TWO GRADIENT DEFECTS IN THE COUNT FORM
-------------------------------------------------------------
Wf-A96 ran this prior at weight 0.01 for 5 epochs and measured
rings/mol 0.0 -> 7.05 (real 4.19) with the 3-ring share stuck at 0.44
(real 0.005) and the 5/6 share at 0.19 (real 0.97).  Two independent
causes were found, both in the gradient path:

1. ``log1p(count)`` plus a ``/E`` denominator make d(loss)/dp shrink as
   the count grows.  On a 20-atom cloud with a complete candidate graph
   (190 pairs), sweeping the bond probability:

       bond p   hard triangles   count grad   per_atom grad
       0.50     0                0.782        0.536
       0.90     1140             1.04e-4      226.0

   i.e. the count form's gradient falls ~7,500x exactly when the
   over-produced-3-ring regime is reached, while per_atom rises ~420x.
   The prior was loudest when it had nothing to correct.
   ``formulation="per_atom"`` normalises by n_atoms and drops log1p,
   keeping O(1/n_atoms) gradient in both regimes.
2. The triangle double-count correction subtracted ``0.5 * raw[3]``,
   which is exact only for a 0/1 graph.  On a soft graph a triangle
   contributes ``0.5 * (p1 p2 p3)**2``; the old form over-subtracted and
   the clamp turned the 6-count (and its whole gradient) into an exact
   zero whenever partial triangles existed.  ``_cycle_counts`` now uses
   the squared product, which reduces to the old value on hard graphs.

Both are reproduced by ``molmetal/scripts/wfa107_ring_gradient_probe.py``
-> ``metrics/by_metric/wfa107_ring_gradient_probe.json``.

Both changes are gradient-path only for the shipped ``count``
formulation: the reported ``ring_prior_term`` is the hard count, so its
VALUE is unchanged.  ``per_atom`` is opt-in.
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
    formulation: ``"count"`` (default, Wf-A78 behaviour, bit-exact),
        ``"per_atom"`` (Wf-A107) or ``"edge_ring"`` (Wf-A107b).  See
        :meth:`RingTemplatePrior.compute` for why ``per_atom`` exists and
        :meth:`RingTemplatePrior._edge_ring_loss` for why the two count
        forms cannot be fixed by reweighting.
    deficit_sizes: in ``per_atom`` mode, sizes whose error is one-sided
        *deficit* (promote rings).  Every other size is one-sided
        *excess* (suppress rings).  Default (5, 6) matches the chemistry
        the corpus shows: 3/4-rings are strained and near-absent, 5/6
        rings carry 97% of the real ring population.
    """

    sizes: Tuple[int, ...] = (3, 4, 5, 6)
    size_weights: Optional[Dict[int, float]] = None
    eps: float = 1e-6
    formulation: str = "count"
    deficit_sizes: Tuple[int, ...] = (5, 6)
    #: ``edge_ring`` only.  Weight on the positive (ring-edge) class of the
    #: per-edge BCE.  ``None`` derives ``neg/pos`` from the molecule's own
    #: candidate list, so a molecule with 6 ring edges out of 190 candidates
    #: does not drown them.
    edge_ring_positive_weight: Optional[float] = None

    def __post_init__(self) -> None:
        if self.formulation not in ("count", "per_atom", "edge_ring"):
            raise ValueError(
                "formulation must be 'count', 'per_atom' or 'edge_ring', got "
                f"{self.formulation!r}"
            )

    def weight_for(self, k: int) -> float:
        if self.size_weights is None:
            return 1.0
        return float(self.size_weights.get(k, 1.0))


def _cycle_counts(
    bond_prob: torch.Tensor,
    pair_index: torch.Tensor,
    sizes: Sequence[int],
    n_atoms: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Cycle counts per size via the non-backtracking operator.

    Returns one entry per requested size (zeros included, so the caller
    can combine a hard-value pass with a soft-gradient pass).

    Wf-A107 — the triangle correction is now taken on the SAME graph the
    count is taken on.  The pre-A107 code subtracted ``0.5 * raw[3]``,
    which is exact only when every edge probability is 0 or 1.  On a
    soft graph a triangle with edge probabilities ``(p1, p2, p3)``
    contributes ``0.5 * (p1 p2 p3)**2`` to ``trace(B**6)/12`` (the walk
    traverses each edge twice, so the probability is squared), while the
    old form subtracted ``0.5 * p1 p2 p3`` — far too much.  MEASURED on
    a 6-atom cloud with a soft hexagon (p=0.35) plus one chord at the
    same probability (triangle 0-1-2): raw 6-count 0.002757, old
    correction 0.021437, old 6-count 0.0 and d(count6)/d(hex-edge
    logit) = 0.0 -> the 6-ring term had NO gradient at all.  Corrected:
    6-count 0.001838 (= the hexagon's own 0.35**6) with gradient
    -0.001195.  That is the second reason the prior moved rings/mol
    without moving the 5/6 share (Wf-A96).  The corrected term is
    ``0.5 * trace((W o W)**3) / 6``, which reduces to the old
    ``0.5 * #triangles`` for a 0/1 graph.
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
        # that is not a 6-cycle.  Each triangle contributes
        # 0.5 * (product of its three edge probabilities)**2 to
        # trace(B^6)/12, so subtract exactly that.
        if n_atoms is None:
            n_atoms = int(torch.maximum(d_src.max(), d_dst.max()).item()) + 1
        n = max(int(n_atoms), int(torch.maximum(d_src.max(), d_dst.max()).item()) + 1)
        w = torch.zeros((n, n), dtype=bond_prob.dtype, device=bond_prob.device)
        w[d_src, d_dst] = p_dir
        w2 = w * w
        tri_sq = torch.trace(w2 @ w2 @ w2) / 6.0
        raw[6] = (raw[6] - 0.5 * tri_sq).clamp(min=0.0)
    return {k: raw[k] for k in sizes}


def _truth_ring_edges(
    truth_bonds: torch.Tensor,
    sizes: Sequence[int],
) -> Dict[Tuple[int, int], int]:
    """Map each truth edge that lies in a small ring to its smallest ring size.

    Wf-A107b — the per-edge supervision needs to know WHICH edges close a
    3-6 ring in the truth molecule.  This is computed exactly from the
    truth adjacency with the same non-backtracking operator used for the
    counts: an edge is in a k-cycle iff ``(B**k)[e, e] > 0``.  No RDKit
    call, no SSSR ambiguity, and the smallest k wins so a triangle edge is
    not relabelled as a 6-ring by the double traversal.

    Returns ``{(i, j): k}`` with ``i < j``; empty when the molecule has no
    ring of the requested sizes.
    """
    src = truth_bonds[0].to(torch.long)
    dst = truth_bonds[1].to(torch.long)
    if src.numel() == 0 or not sizes:
        return {}
    d_src = torch.cat([src, dst])
    d_dst = torch.cat([dst, src])
    continues = d_dst.unsqueeze(1) == d_src.unsqueeze(0)
    backtracks = d_dst.unsqueeze(0) == d_src.unsqueeze(1)
    b_mat = (continues & ~backtracks).to(torch.float64)

    ring: Dict[Tuple[int, int], int] = {}
    found = torch.zeros(d_src.numel(), dtype=torch.bool, device=b_mat.device)
    power = b_mat
    for k in range(2, max(sizes) + 1):
        power = power @ b_mat
        if k not in sizes:
            continue
        in_ring = torch.diagonal(power) > 0.5
        for e in torch.nonzero(in_ring & ~found).flatten().tolist():
            i, j = int(d_src[e]), int(d_dst[e])
            ring[(min(i, j), max(i, j))] = k
        found = found | in_ring
    return ring


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

        truth = truth_bonds.to(bond_logits.device)
        if truth.dim() != 2 or truth.shape[0] != 2:
            return zero

        # Wf-A107b — the per-edge form needs no cycle counts: it labels
        # each candidate edge from the truth graph directly.
        if self.config.formulation == "edge_ring":
            return self.weight * self._edge_ring_loss(soft, pair_index, truth)

        soft_counts = _cycle_counts(soft, pair_index, self.config.sizes, n_atoms)
        hard_counts = _cycle_counts(hard, pair_index, self.config.sizes, n_atoms)

        truth_counts = _cycle_counts(
            torch.ones(truth.shape[1], device=bond_logits.device, dtype=soft.dtype),
            truth,
            self.config.sizes,
            n_atoms,
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

        if self.config.formulation == "per_atom":
            if not self.config.sizes:
                return zero
            return self.weight * self._per_atom_loss(
                hard_counts, soft_counts, truth_counts, int(n_atoms)
            )

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

    def _edge_ring_loss(
        self,
        soft: torch.Tensor,
        pair_index: torch.Tensor,
        truth_bonds: torch.Tensor,
    ) -> torch.Tensor:
        """Wf-A107b — per-edge ring-membership BCE.

        WHY THIS FORM EXISTS (MEASURED 2026-10-21, Wf-A107)
        ----------------------------------------------------
        Both count forms have an EXTENSIVE gradient: the cycle count grows
        with graph density, so any bounded function of it has a derivative
        that shrinks exactly when the graph is densest.  Measured on the
        real runs, the count form's pre-clip gradient norm falls
        53.8 -> 86.2 -> 280.1 -> 25.1 -> 1.7 across five epochs *while the
        graph densifies*, and the bond AP only starts moving on the epoch
        where the prior has already gone quiet.  The share form does not
        fix it (probe: 0.011 at p=0.50 -> 1.4e-4 at p=0.90, the same
        collapse).  No reweighting of an extensive statistic can.

        This form is INTENSIVE: one Bernoulli per candidate edge, so the
        loss and its gradient are O(1) per edge regardless of how many
        rings the graph has, and it says which edges to close rather than
        how many rings to have.  The target is the truth molecule's own
        ring-edge set (smallest ring size in ``sizes``).

        Returns the mean weighted BCE over candidate edges; exact zero
        when the truth molecule has no ring of the requested sizes (there
        is nothing to prefer) — inventing a target would penalise a
        correct acyclic output.
        """
        ring = _truth_ring_edges(truth_bonds, self.config.sizes)
        if not ring:
            return torch.zeros((), device=soft.device, dtype=soft.dtype)

        src = pair_index[0].tolist()
        dst = pair_index[1].tolist()
        target = torch.tensor(
            [1.0 if (min(i, j), max(i, j)) in ring else 0.0
             for i, j in zip(src, dst)],
            device=soft.device,
            dtype=soft.dtype,
        )
        n_pos = float(target.sum())
        if n_pos <= 0.0:
            return torch.zeros((), device=soft.device, dtype=soft.dtype)

        if self.config.edge_ring_positive_weight is not None:
            pos_w = float(self.config.edge_ring_positive_weight)
        else:
            n_neg = float(target.numel()) - n_pos
            pos_w = max(n_neg / n_pos, 1.0)
        weights = torch.where(
            target > 0.5,
            torch.full_like(target, pos_w),
            torch.ones_like(target),
        )
        p = soft.clamp(1e-4, 1.0 - 1e-4)
        return F.binary_cross_entropy(p, target, weight=weights)

    def _per_atom_loss(
        self,
        hard_counts: Dict[int, torch.Tensor],
        soft_counts: Dict[int, torch.Tensor],
        truth_counts: Dict[int, torch.Tensor],
        n_atoms: int,
    ) -> torch.Tensor:
        """Wf-A107 — per-atom one-sided ring-count error.

        WHY THIS EXISTS (MEASURED 2026-10-21, Wf-A107)
        -----------------------------------------------
        The ``count`` formulation wraps the count in ``log1p`` and
        divides the soft count by the candidate-edge count E.  Both
        choices make the gradient vanish exactly where the prior is
        needed.  On a synthetic 20-atom cloud whose candidate graph is
        complete (190 pairs), sweeping the bond probability:

            bond p   hard triangles   count grad   per_atom grad
            0.50     0                0.782        0.536
            0.90     1140             1.04e-4      226.0

        At the over-produced-3-ring state the shipped prior's gradient
        is ~7,500x weaker than at the cold-start state where it had
        nothing to correct, while ``per_atom`` is ~420x STRONGER.  That
        is why weight 0.01 moved rings/mol from 0.0 to 7.05 while
        leaving the 3-ring share at 0.44: the prior fired hardest when
        the graph was ring-free.

        Normalising by ``n_atoms`` (not by E) and dropping ``log1p``
        keeps the gradient O(1/n_atoms) per unit of soft count in BOTH
        regimes, because d(c_k/n_atoms)/dp does not shrink as more
        cycles appear.

        ⚠️ The same measurement is also the warning: at p=0.90 the
        per_atom loss is 5,314 and its gradient 226, because an excess
        term grows with the over-production it is meant to remove.
        Wf-A107's GPU arm measured grad_norm 3,605 on a real batch.  The
        form fixes vanishing, not magnitude; clipping or a saturating
        excess term is still OPEN.

        Asymmetry: sizes in ``deficit_sizes`` (default 5, 6) are
        penalised only when the prediction has FEWER rings than the
        truth molecule — they are promoted.  Every other size is
        penalised only when the prediction has MORE — 3- and 4-rings
        are suppressed.  The asymmetry is what makes the term bounded:
        a deficit term can never exceed ``t_k**2``, while an excess
        term grows with the over-production it is meant to remove.
        """
        n = max(int(n_atoms), 1)
        loss = torch.zeros(
            (), device=next(iter(soft_counts.values())).device,
            dtype=next(iter(soft_counts.values())).dtype,
        )
        for k in self.config.sizes:
            soft = soft_counts[k] / float(n)
            # Value from the hard graph, gradient from the soft graph
            # (same straight-through contract as the count form).
            u = hard_counts[k] / float(n) + soft - soft.detach()
            t = truth_counts[k] / float(n)
            if k in self.config.deficit_sizes:
                err = torch.relu(t - u)
            else:
                err = torch.relu(u - t)
            loss = loss + self.config.weight_for(k) * err ** 2
        return loss

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
