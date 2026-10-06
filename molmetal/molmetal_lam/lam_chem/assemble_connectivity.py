# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Assemble-aware connectivity loss (Round-3 D1).

Why this module exists
----------------------
MEASURED on the tmQM all6 retrain (2026-10-04, ``results/cfm_tmqm_all6``,
20 epochs, seed 0): ``bond_ce`` fell monotonically 0.058 -> 0.043 while
``decode_ratio`` ALSO fell 0.531 -> 0.418, and ``single_fragment_ratio``
climbed only 0.000 -> 0.167.  The per-bond cross-entropy is a *local,
per-edge* objective; the metric that is actually gated is *global* -- the
decoded graph must assemble into ONE connected heavy-atom component.  A loss
that never looks across edges cannot express "these two predictions belong to
the same molecule".

So this module supplies the missing global term: a differentiable surrogate
for "the predicted bond graph is one fragment, not several".

Why the surrogate is a DIFFUSION term and not union-find
-------------------------------------------------------
The obvious implementation -- threshold at 0.5, run union-find, return
``(n_frags - 1) / n_heavy`` -- is **not differentiable**.  ``argmax``/``>``
have zero gradient almost everywhere, so the term would enter the loss sum,
print a number, and train nothing.  That is exactly the FLAG_ONLY failure
class this project has been bitten by repeatedly (see AGENTS.md 5.7 and
``molmetal/tests/test_cfm_loss_gradient_liveness.py``), and it is the failure
this whole Round-3 wave exists to fix.  A hard fragment count is therefore
kept here as a **detached diagnostic only**; the value that reaches the
gradient is the soft one below.

The soft term
-------------
Let ``p_ij = P(bond between i and j) = 1 - softmax(logits)[NO_BOND]`` -- the
same existence rule the decoder uses, so training and evaluation agree.
Build the **lazy** (self-looped) row-stochastic soft walk

    P = (A + w*I) / rowsum(A + w*I)

and ask, for each atom ``i``, the probability that a walk starting at ``i``
**reaches some other atom at some point** -- its "escape" probability::

    escape_i = 1 - prod_{k=1..K} ( (P^k)[i, i] )
             = 1 - prod_{k=1..K} (1 - s_k),   s_k = 1 - (P^k)[i, i]

which separates the two cases exactly:

* an atom **stranded in its own fragment** has ``A[i, :] ~ 0``, so the
  self-loop dominates, ``(P^k)[i, i] -> 1``, ``s_k -> 0`` and
  ``escape_i -> 0``;
* an atom in a **connected** graph leaves on the first step, ``s_1 = O(1)``
  and ``escape_i -> 1``.

The penalty is ``1 - escape_i`` averaged over heavy atoms: ~0 when the graph
is one component, ~1 when the heavy atoms are isolated.  That is a smooth,
absolute surrogate for ``1 - single_fragment_ratio`` -- precisely the Round-3
insight #2 signal (``largest_fragment_frac`` 0.21 -> 0.47 while
``single_fragment_ratio`` barely moved).

Why lazy (self-looped) and not the plain walk
---------------------------------------------
MEASURED, and it is not a detail.  The undamped walk on a chain is
**periodic**: ``(P^k)[i,i]`` alternates 0.50, ~0, 0.50, ~0, ... so any
truncated product of ``(1 - diag)`` is parity-dependent -- ``K=12`` and
``K=11`` disagree, and the value assigned to a perfectly connected atom
swings with an arbitrary truncation choice.  That formulation also inverted
the gradient sign (raising a NO_BOND logit, i.e. *disconnecting* an atom,
*lowered* the loss).  Damping makes the walk aperiodic, so the product
converges monotonically and the sign is correct.

Two further properties this formulation is careful about, both MEASURED on
isolated test graphs (``molmetal/tests/test_assemble_connectivity.py``):

* **It must be absolute, not relative.**  Normalising retained mass by its
  own maximum scores a graph where *every* atom is stranded as perfectly
  consistent (loss 0.0) while scoring a connected chain 0.16 -- exactly
  backwards.
* **A stranded atom must still receive gradient.**  Its row is near zero, so
  a term reading absolute mass off that row would give it none -- training
  nothing for the one case that matters.  The row-normalisation keeps the
  ratio alive at arbitrarily small ``p``.

Both quantities are returned so the caller can log the honest hard count next
to the soft value instead of presenting the surrogate as if it were a
fragment count.

A case this term CANNOT fix, stated plainly
--------------------------------------------
The head only emits candidate pairs within ``cutoff`` (3.5 Å).  An atom with
**no candidate edge at all** -- geometrically isolated in the input cloud --
has an all-zero adjacency row, so its penalty is a constant and its gradient
is exactly zero.  That is not a bug in the gradient path, it is the head
having no authority over that atom: there is no logit to move.  MEASURED on
a real ``build_synthetic_pt_click_clouds`` cloud, one of six atoms sits
beyond cutoff from every other and contributes ``loss=0.167`` with
``logit.grad.abs().sum() == 0.0``; the same graph WITH a candidate edge on
that atom gives ``3.7e-2``.

So ``assemble_loss`` can report a fragmentation the gradient cannot fix.  A
caller reading that number should treat it as "these atoms are unreachable
from the bond head", not as "the head needs to try harder".  Making such
atoms reachable is an input-geometry / cutoff question, not a loss-weight
question.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

#: Hydrogen.  Excluded from the connectivity graph: an H that fails to bond
#: is a valence detail, not a separate molecule.
_Z_HYDROGEN = 1


def _soft_bond_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """P(bonded) per candidate pair.

    Uses the SAME existence rule as
    :func:`molmetal.molmetal_lam.lam_chem.yuelbond_decoder.bond_decision`
    (``threshold`` mode): ``1 - P(NO_BOND)``.  Class 0 is ``BOND_NO_BOND``;
    training against a different rule than the decoder decodes with is the
    exact "metric and decoder disagreed" wiring gap the Round-1 fix closed for
    evaluation, and it would simply reopen here in the loss.
    """
    prob = torch.softmax(logits.float(), dim=-1)
    return 1.0 - prob[:, 0]


def assemble_fragment_loss(
    logits: torch.Tensor,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    atomic_numbers: torch.Tensor,
    *,
    n_diffusion_steps: int = 12,
    self_loop_weight: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float, float]:
    """Differentiable assemble-aware penalty for one cloud.

    Parameters
    ----------
    logits
        ``(E, NUM_BOND_CLASSES)`` bond logits for the candidate pairs.
    edge_index
        ``(i_idx, j_idx)``, each ``(E,)`` long -- the atom indices each row of
        ``logits`` refers to.  **Required.**  A flat ``(E, 5)`` tensor does not
        encode which atoms its rows connect, so the graph cannot be rebuilt
        from the logits alone; the caller must pass the same ``(i_idx, j_idx)``
        the head produced.
    atomic_numbers
        ``(N,)`` long.  Hydrogens are dropped from the connectivity graph.
    n_diffusion_steps
        Truncation ``K`` of the escape product.  The lazy walk makes the
        product converge geometrically, so this is a tail cut, not a
        parity coin-flip (see the module docstring).
    self_loop_weight
        Self-loop weight ``w``.  Larger ``w`` makes the walk stickier, so a
        weak-but-real bond counts less as an escape.  Must be > 0 for the
        escape probability to be defined; the default damps the periodic
        oscillation an undamped walk shows on chains.
    eps
        Numerical floor for divisions.

    Returns
    -------
    (loss, n_fragments, largest_fragment_frac)
        * ``loss`` -- scalar tensor WITH gradient, in ``[0, 1]``.  ~0 when the
          soft graph is one component, ~1 when heavy atoms are isolated.
        * ``n_fragments`` -- exact hard fragment count on the heavy-atom
          subgraph, ``float``.  Detached: for logging only.
        * ``largest_fragment_frac`` -- hard largest-component mass fraction,
          ``float``.  Detached: for logging only.
    """
    device = logits.device
    i_idx, j_idx = edge_index
    i_idx = i_idx.to(device=device, dtype=torch.long)
    j_idx = j_idx.to(device=device, dtype=torch.long)
    z = atomic_numbers.to(device=device, dtype=torch.long)
    dtype = logits.dtype

    n_atoms = int(z.shape[0])
    if n_atoms == 0:
        return torch.zeros((), device=device, dtype=dtype), 0.0, 0.0

    heavy = (z != _Z_HYDROGEN).nonzero(as_tuple=True)[0]
    n_heavy = int(heavy.shape[0])
    if n_heavy < 2:
        # Nothing to connect: a lone heavy atom is one fragment by
        # definition, so there is no fragmentation to penalise.
        return torch.zeros((), device=device, dtype=dtype), 1.0, 1.0

    # Remap global atom index -> heavy-atom index (-1 for hydrogen).
    remap = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
    remap[heavy] = torch.arange(n_heavy, device=device)

    if int(logits.shape[0]) > 0:
        ei = remap[i_idx]
        ej = remap[j_idx]
        keep = (ei >= 0) & (ej >= 0)
        ei = ei[keep]
        ej = ej[keep]
    else:
        keep = torch.zeros(0, dtype=torch.bool, device=device)
        ei = torch.zeros(0, dtype=torch.long, device=device)
        ej = torch.zeros(0, dtype=torch.long, device=device)

    adj = torch.zeros((n_heavy, n_heavy), device=device, dtype=dtype)
    if int(ei.shape[0]) > 0:
        p = _soft_bond_probabilities(logits)[keep]
        # Symmetric: a bond is a property of the unordered pair and the head
        # only emits the upper triangle.  ``accumulate=True`` so overlapping
        # writes cannot silently drop an edge.
        adj = adj.index_put((ei, ej), p, accumulate=True).index_put(
            (ej, ei), p, accumulate=True
        )

    # ---- Soft, differentiable term ------------------------------------
    # Lazy (self-looped) walk.  The self-loop is what makes the walk
    # aperiodic -- see the module docstring for the parity bug an undamped
    # walk produces -- and it is what makes "stuck" well defined: with
    # A[i,:] ~ 0 the self-loop dominates, (P^k)[i,i] -> 1, escape -> 0.
    w = float(self_loop_weight)
    walk = adj + w * torch.eye(n_heavy, device=device, dtype=dtype)
    row = walk.sum(dim=1, keepdim=True)
    trans = walk / row.clamp_min(eps)
    k = int(max(1, n_diffusion_steps))
    power = torch.eye(n_heavy, device=device, dtype=dtype)
    stay = torch.ones(n_heavy, device=device, dtype=dtype)
    for _ in range(k):
        power = power @ trans
        stay = stay * power.diagonal().clamp(min=0.0, max=1.0)
    escape = 1.0 - stay
    # 1 - escape: 0 for a connected atom, ~1 for a stranded one.  The upper
    # clamp is deliberately loose -- an exact clamp(max=1.0) saturates
    # precisely ON the stranded atoms, where gradient is most needed, and
    # MEASURED as grad == 0 there.
    loss = (1.0 - escape).clamp(min=0.0).mean()

    # ---- Hard diagnostic (detached, never reaches the gradient) -------
    with torch.no_grad():
        hard = (adj.detach() > 0.5).to(torch.int64)
        hard = torch.minimum(hard, torch.ones_like(hard))
        n_fragments, largest_frac = _count_components(hard)
    return loss, float(n_fragments), float(largest_frac)


def _count_components(adj: torch.Tensor) -> Tuple[float, float]:
    """Exact connected components on a 0/1 adjacency, via label propagation.

    ``float`` (component count, largest-component size fraction).  Iterative
    rather than union-find because it stays on-device and the graphs here are
    N ~ 20-40; the loop runs until the labels stop moving, so the result is
    the true component partition and not a fixed-iteration approximation.
    """
    n = int(adj.shape[0])
    if n == 0:
        return 0.0, 0.0
    labels = torch.arange(n, device=adj.device, dtype=torch.long)
    for _ in range(n + 1):
        # Every atom takes the smallest label among itself and its neighbours;
        # this is monotone, so the loop converges in <= n rounds.
        nbr = torch.where(
            adj > 0, labels.unsqueeze(0), torch.full_like(labels.unsqueeze(0), n)
        )
        new = torch.minimum(labels, nbr.min(dim=1).values)
        if bool(torch.equal(new, labels)):
            break
        labels = new
    uniq, counts = torch.unique(labels, return_counts=True)
    largest = float(counts.max().item()) / float(n)
    return float(uniq.numel()), largest


def batch_assemble_loss(
    per_cloud: Optional[list],
    device: torch.device,
) -> Tuple[torch.Tensor, float, float, int]:
    """Mean assemble loss over a batch of ``(logits, edge_index, z)`` triples.

    Returns ``(loss, mean_n_fragments, mean_largest_fragment_frac, n_used)``.
    An empty batch yields a hard zero and ``n_used == 0`` so the caller can
    skip the term without special-casing a ``None``.
    """
    used: list = []
    for item in per_cloud or []:
        if item is None:
            continue
        logits, edge_index, z = item
        if logits is None or int(logits.shape[0]) == 0:
            continue
        # No pair indices means the soft graph cannot be rebuilt, and
        # `assemble_fragment_loss` unpacks this unconditionally.  Skipping is
        # the honest outcome: the term reports itself unused (`n_used == 0`)
        # instead of failing the run or, worse, being given a graph the bond
        # head never emitted logits for.
        if edge_index is None:
            continue
        used.append((logits, edge_index, z))
    if not used:
        return (
            torch.zeros((), device=device),
            0.0,
            0.0,
            0,
        )
    losses = []
    frags: list = []
    fracs: list = []
    for logits, edge_index, z in used:
        term, n_frag, largest = assemble_fragment_loss(
            logits, edge_index, z
        )
        losses.append(term)
        frags.append(n_frag)
        fracs.append(largest)
    mean_loss = torch.stack(losses).mean()
    return (
        mean_loss,
        sum(frags) / len(frags),
        sum(fracs) / len(fracs),
        len(used),
    )
