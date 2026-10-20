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

import math
from typing import Optional, Tuple

import torch

#: Hydrogen.  Excluded from the connectivity graph: an H that fails to bond
#: is a valence detail, not a separate molecule.
_Z_HYDROGEN = 1

#: How often :func:`_count_components` pays for its host-side convergence test.
#: The test is a device comparison read on the host, i.e. a full queue drain;
#: amortising it costs at most this many minus one extra (no-op) propagation
#: rounds and cannot change the partition — see the proof note in that
#: function. 8 sits above the propagation diameter of the graphs this sees
#: (tmQM complexes, N ~ 20-40 heavy atoms), so the common case still exits on
#: the first test that follows convergence.
_COMPONENT_CHECK_EVERY = 8


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
    out = 1.0 - prob[:, 0]
    # The softmax stays in fp32 on purpose — under bf16/fp16 autocast a
    # half-precision softmax loses the small probabilities this term is
    # reading. But the CALLER builds `adj` with `dtype = logits.dtype`, and
    # returning fp32 from a float64 input made `index_put` raise
    # "source and destination dtypes match" (MEASURED 2026-10-10, which also
    # blocked double-precision gradcheck of the escape-time formulation).
    # Cast back so the dtype contract is the caller's, not this helper's.
    return out.to(logits.dtype)


def mean_stay_probability(
    logits: torch.Tensor,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    atomic_numbers: torch.Tensor,
    *,
    n_diffusion_steps: int = 12,
    self_loop_weight: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Detached ``stay_mean`` for one cloud — a DIAGNOSTIC, never a loss.

    This is the number :func:`assemble_fragment_loss` returns under
    ``formulation="escape_time"``, computed WITHOUT the hinge, so it can be
    read on its own. Two uses, and the second is why it exists as a separate
    function rather than a return value:

    1. reporting — it is the only way to see where a model currently sits on
       the disconnection axis without turning the term on and perturbing it;
    2. THRESHOLD DERIVATION — ``metrics/by_metric/cfm_assemble_hinge_tau.json``
       sets the hinge threshold from tmQM's ground-truth bond graphs (max
       stay_mean 0.2649 over 1500 complexes). The decision-relevant number is
       the same statistic on the model's OWN predictions, which soft bond
       probabilities push upwards. Reading this per epoch is how that gap is
       closed before an experiment is designed against the wrong threshold.

    Detached on purpose: a diagnostic that carries a gradient is a second,
    invisible path into the loss sum.
    """
    with torch.no_grad():
        i_idx, j_idx = edge_index
        i_idx = i_idx.to(device=logits.device, dtype=torch.long)
        j_idx = j_idx.to(device=logits.device, dtype=torch.long)
        z = atomic_numbers.to(device=logits.device, dtype=torch.long)
        n_atoms = int(z.shape[0])
        if n_atoms == 0 or int(i_idx.shape[0]) == 0:
            return 0.0
        keep = z > 1  # hydrogens leave the connectivity graph
        remap = torch.full((n_atoms,), -1, dtype=torch.long, device=logits.device)
        remap[keep] = torch.arange(int(keep.sum()), device=logits.device)
        ei, ej = remap[i_idx], remap[j_idx]
        ok = (ei >= 0) & (ej >= 0)
        if int(ok.sum()) == 0:
            return 0.0
        n_heavy = int(keep.sum())
        if n_heavy < 2:
            return 0.0
        dtype = logits.dtype
        adj = torch.zeros((n_heavy, n_heavy), device=logits.device, dtype=dtype)
        prob = _soft_bond_probabilities(logits)[ok]
        adj = adj.index_put((ei[ok], ej[ok]), prob, accumulate=True).index_put(
            (ej[ok], ei[ok]), prob, accumulate=True
        )
        w = float(self_loop_weight)
        walk = adj + w * torch.eye(n_heavy, device=logits.device, dtype=dtype)
        trans = walk / walk.sum(dim=1, keepdim=True).clamp_min(eps)
        k = int(max(1, n_diffusion_steps))
        cur = trans
        diags = []
        for _ in range(k):
            diags.append(torch.diagonal(cur))
            cur = cur @ trans
        return float(torch.stack(diags).mean(dim=0).mean())


def hinge_tau_from_values(
    values,
    *,
    headroom: float = 1.13,
    round_to: float = 0.01,
) -> float:
    """The ONE definition of the assemble dead-zone threshold rule.

    Wf-A99 (2026-10-20).  The rule "ceil(max * headroom / round_to) *
    round_to" existed twice: once inside
    :func:molmetal.scripts.train_cfm_yuelbond_wasserstein.derive_assemble_hinge_tau
    and once, spelled out as "math.ceil(max(truth) * 1.13 * 100) / 100",
    inside :mod:molmetal.scripts.wf_a98_assemble_discrimination.  Two copies
    of a calibration rule is how a probe ends up "reproducing" a trainer
    value that has since moved -- the same drift the donor-list unification
    (Wf-A94) and the bond-metric definition stamp (Wf-A99) exist to prevent.

    The rule reproduces the published tmQM value from its published input:
    0.2649 * 1.13 = 0.2993 -> 0.30.

    Parameters
    ----------
    values : iterable of float
        Per-cloud "what does a correct complex score" statistics.  Only the
        maximum matters; the full list is accepted so callers can hand over
        the same collection they already computed.
    headroom : float
        Multiplicative margin above the maximum.  1.13 is the published
        value and is what makes the rule reproduce 0.30.
    round_to : float
        Grid the result is rounded UP to.  0.01 keeps tau readable in a
        config without changing the hinge's behaviour materially.

    Raises
    ------
    ValueError
        When "values" is empty.  Silently returning a default here is the
        defect the caller exists to remove: a threshold from a different
        corpus reads ON and contributes zero gradient.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError(
            "hinge_tau_from_values needs at least one statistic; an empty "
            "corpus must fail loudly rather than fall back to a foreign tau"
        )
    step = float(round_to)
    if step <= 0.0:
        raise ValueError(f"round_to must be positive, got {round_to!r}")
    vmax = max(vals)
    return math.ceil(vmax * float(headroom) / step) * step


def _spectral_component_penalty(
    adj: torch.Tensor,
    *,
    sharpness: float = 0.05,
    heat_scale: float = 8.0,
) -> torch.Tensor:
    """Soft connected-component count -> a [0, 1) connectivity penalty.

    ``c = trace(exp(-t L))`` with ``L = D - W`` the combinatorial Laplacian
    of the DECODE-RULE graph (weights = sigmoid((p - 0.5) / sharpness)).  As
    ``t -> inf`` this equals the number of connected components exactly: each
    component contributes one zero eigenvalue, every other eigenvalue is
    damped.  The penalty ``1 - 1/c`` is 0 for one component, 0.5 for two, and
    approaches 1 as the graph shatters.

    Why the heat kernel and not ``eigvalsh(L)[1]``: a disconnected graph has
    a REPEATED zero eigenvalue, and the eigen-decomposition backward pass
    divides by eigenvalue differences, so it returns NaN exactly on the
    inputs this term is meant to penalise.  ``matrix_exp`` has no such
    degeneracy.  MEASURED on n=30: chain 1.0003, three fragments 1.74,
    isolated atoms 11.7 -- monotone in fragmentation, finite gradient at the
    bridge probability.

    ``t = heat_scale / lambda2(P_n)`` because the path is the sparsest
    connected graph, so its Fiedler value is the smallest any connected
    graph can have; scaling by it makes ``t`` resolve "connected" for every
    real molecule instead of only for dense ones.
    """
    n = int(adj.shape[0])
    if n < 2:
        return torch.zeros((), device=adj.device, dtype=adj.dtype)
    dtype = adj.dtype
    s = max(float(sharpness), 1e-6)
    # SHIFTED, RENORMALISED sigmoid.  A plain sigmoid((p-0.5)/s) never reaches
    # zero: at p=0 it still returns sigmoid(-0.5/s) = 4.5e-05 for s=0.05, and
    # a molecule has O(100) such pairs per atom, so the leakage alone sums to
    # ~5e-03 per atom -- enough to bridge every fragment at the heat-kernel
    # scale.  MEASURED before this shift: five 16-atom fragments on n=80
    # scored c=1.00 (loss 0.0) instead of c=5.  The shift makes a p=0 pair
    # contribute exactly nothing while leaving the gradient at p~0.5 intact.
    _lo = torch.sigmoid(torch.tensor(-0.5 / s, dtype=dtype))
    w = ((torch.sigmoid((adj - 0.5) / s) - _lo) / (1.0 - _lo)).clamp(min=0.0)
    # No self-loops: they are edges to nowhere and would inflate every degree.
    w = w - torch.diag_embed(torch.diagonal(w))
    w = (w + w.t()) * 0.5
    lap = torch.diag(w.sum(dim=1)) - w
    lam2_path = 2.0 * (1.0 - math.cos(math.pi / n))
    t = float(heat_scale) / max(lam2_path, 1e-9)
    c = torch.matrix_exp(-t * lap).diagonal().sum().clamp(min=1.0)
    return 1.0 - 1.0 / c

def assemble_fragment_loss(
    logits: torch.Tensor,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    atomic_numbers: torch.Tensor,
    *,
    n_diffusion_steps: int = 12,
    self_loop_weight: float = 1.0,
    eps: float = 1e-12,
    formulation: str = "product",
    hinge_tau: float | None = None,
    sharpness: float = 0.05,
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
    hinge_tau
        Optional dead-zone. When set, the penalty becomes
        ``clamp(penalty - hinge_tau, min=0)``, so nothing is charged below the
        threshold. ``None`` (the default) applies no dead zone. Derive the
        value from a corpus, not by eye — see the block in the function body.
    formulation
        How the ``K`` per-step stay probabilities ``(P^k)[i,i]`` are combined
        into the penalty.  ``"escape_time"`` (default) takes their arithmetic
        mean; ``"product"`` takes their geometric mean (the original Round-3 D1
        form).  Both order correctly — a stranded atom scores worse than a
        connected one — but the product's gradient scales with the product and
        so vanishes across the near-connected regime, which is exactly where a
        prevention objective needs to act.  See the block above for the
        measured gradient magnitudes on both.
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

    # ---- Spectral branch (Wf-A98) --------------------------------------
    #
    # WHY THIS EXISTS.  MEASURED 2026-10-19 on 64 real tmQM val clouds
    # (metrics/by_metric/wfa98_assemble_discrimination.json):
    #
    #   stay_mean   truth graph              0.175  (max 0.212)
    #               DECODED graph, fragmented 0.239  <- correctly higher
    #               DECODED graph, connected  0.123  <- correctly lower
    #               MODEL SOFT graph          0.145  <- below every truth
    #
    # The statistic is directionally RIGHT on the decoded graph: fragments
    # score 0.239 against 0.123 for connected molecules.  It is the SOFT
    # relaxation that breaks it.  A softmax head spreads probability over
    # every candidate pair inside the 3.5 A cutoff, so the soft graph is
    # dense, the walk leaves every atom quickly, and stay_mean lands BELOW
    # the p50 of the truth graphs it is supposed to reproduce.  The hinge
    # then cannot fire: any tau low enough to fire on the soft graph would
    # also fire on correct truth graphs (0.175-0.212).  That is not a
    # tuning problem, and it is why assemble_loss read exactly 0.0 at every
    # epoch on every cohort that ever ran this term.
    #
    # So the fix is not a different tau, it is a statistic that is evaluated
    # on the graph the DECODER will see.  MEASURED on the same 64 clouds, the
    # size-normalised algebraic connectivity of the NORMALISED Laplacian
    # separates by five orders of magnitude where stay_mean separates by 2x:
    #
    #   lambda2 ratio   truth graph              6.92 (min 1.90)
    #                   DECODED, fragmented      7.9e-05
    #                   DECODED, connected      27.49
    #
    # and 0/64 truth graphs are disconnected against 33/64 decoded ones.
    #
    # The spectral form measures the thing that actually defines
    # disconnection: a graph is disconnected iff its Laplacian has a second
    # zero eigenvalue.  Counting zero eigenvalues directly is not
    # differentiable, so this uses the heat-kernel trace
    #
    #     c = trace(exp(-t * L)),  L = D - W the COMBINATORIAL Laplacian
    #
    # which equals the number of connected components as t -> infinity
    # (each component contributes exactly one eigenvalue 0; every other
    # eigenvalue is damped away).  The penalty is 1 - 1/c, so it is 0 for one
    # component, 0.5 for two, and approaches 1 as the graph shatters -- the
    # same [0, 1] scale as the escape-time term it replaces, without a
    # threshold to calibrate.
    #
    # Two design choices that are load-bearing:
    #
    #   * COMBINATORIAL, not normalised.  Row-normalisation is scale-free,
    #     so a single sub-threshold edge (p=0.3 -> weight 0.018) counts as
    #     much as a real bond and every soft graph looks connected.  The
    #     combinatorial form keeps small weights small.
    #   * t = heat_scale / lambda2(P_n).  The path is the sparsest connected
    #     graph, so its Fiedler value is the smallest a connected graph can
    #     have; scaling by it makes the damping resolve "connected" for any
    #     real molecule.  MEASURED on n=30: chain c=1.0003, three fragments
    #     c=1.74, isolated atoms c=11.7.
    #
    # The sharpening sigmoid is what lets the term see the DECODE rule: it
    # is near 1 above the 0.5 decision boundary and near 0 below it, while
    # staying differentiable through it.  ``sharpness`` is its width; 0.05
    # is the default and is NOT tuned (recorded as such in the catalog).
    if formulation == "spectral":
        loss = _spectral_component_penalty(adj, sharpness=float(sharpness))
        if hinge_tau is not None:
            loss = (loss - float(hinge_tau)).clamp(min=0.0)
        with torch.no_grad():
            hard = (adj.detach() > 0.5).to(torch.int64)
            hard = torch.minimum(hard, torch.ones_like(hard))
            n_fragments, largest_frac = _count_components(hard)
        return loss, float(n_fragments), float(largest_frac)

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
    diags = []
    for _ in range(k):
        power = power @ trans
        d_k = power.diagonal().clamp(min=0.0, max=1.0)
        diags.append(d_k)
        stay = stay * d_k
    diags = torch.stack(diags)  # (K, N_heavy) — the per-step stay probabilities
    escape = 1.0 - stay

    # TWO AGGREGATIONS OF THE SAME PER-STEP STAY PROBABILITIES.
    #
    # "product" (the original Round-3 D1 form) takes the GEOMETRIC mean:
    #     loss_i = prod_{k=1..K} (P^k)[i,i]
    # It orders correctly — connected ~2e-07, metal-isolated ~2.9e-01 — but its
    # gradient is proportional to the product itself, so it VANISHES exactly
    # where the term is needed most.
    #
    # MEASURED 2026-10-10 on a 7-atom chain, AT THE LOGITS (what the optimiser
    # actually consumes — not the adjacency matrix, whose numbers look ~7
    # orders healthier and are the more flattering mistake to quote):
    #
    #     case              product val   product |g|   escape val   escape |g|
    #     connected p=.9    1.70e-07       4.62e-09       0.2629       3.46e-04
    #     metal isolated    2.86e-01       6.17e-09       0.4888       4.45e-04
    #     weak bonds p=.3   2.41e-07       3.98e-08       0.2722       2.60e-03
    #
    # The product's gradient is ~5e-09 in EVERY case — including where its own
    # value is 0.286. So it is a DIAGNOSTIC, not an objective: ~8 orders of
    # magnitude below the bond_ce gradient it would share a loss sum with.
    # On the real tmQM cohort it reads exactly 0.0 (the product underflows),
    # which is why `assemble_term` has never contributed anything.
    #
    # "escape_time" takes the ARITHMETIC mean of the same per-step values:
    #     loss_i = mean_{k=1..K} (P^k)[i,i]
    # Same zero-crossing, same ordering, bounded in [0, 1], and the gradient no
    # longer carries the product's scale. MEASURED, same graphs:
    #
    #     case                escape_time     |dL/dA|max
    #     connected chain     2.67e-01        1.14e-01
    #     metal isolated      4.92e-01        9.29e-01
    #     weak bonds (p=.3)   3.59e-01        2.66e-01
    #
    # All four invariants this module was written for still hold under it
    # (molmetal/tests/test_assemble_connectivity.py::test_escape_time_*):
    # stranded atoms get gradient; an all-isolated graph scores worst (1.000)
    # rather than perfect; cutting a bond RAISES the penalty; and the value is
    # monotone decreasing in bond probability.
    #
    # DEFAULT: "product" — UNCHANGED. A default flip to "escape_time" was tried
    # on 2026-10-10 and broke 10 of the 21 tests in test_assemble_connectivity.py,
    # including test_fully_connected_graph_has_near_zero_gradient, which pins
    # "a connected graph gets no gradient" as a DELIBERATE property: do not
    # spend gradient on graphs that are already fine. That is a real design
    # choice and it is in direct tension with this formulation's selling point
    # (liveness near the connected regime). The trade-off is recorded in
    # todo.json BLK-C11 rather than resolved by flipping a default here.
    #
    # Changing the default is in any case safe for existing runs only because
    # `assemble_loss_weight` is 0.0 in every config on disk, so no
    # shipped checkpoint or metric was ever produced with this term active.
    # "product" remains available to reproduce the Round-3 D1 numbers.
    formulation = str(formulation).lower()
    if formulation == "product":
        loss = (1.0 - escape).clamp(min=0.0).mean()
    elif formulation == "escape_time":
        loss = diags.mean(dim=0).mean()
    else:
        raise ValueError(
            f"formulation must be 'product' or 'escape_time', got {formulation!r}"
        )
    # HINGE (2026-10-10, user-selected).
    #
    # Neither aggregation alone satisfies both things this term needs:
    #   * "a graph that is already one component should not be charged for
    #      gradient" — pinned deliberately by
    #      test_fully_connected_graph_has_near_zero_gradient; and
    #   * "a graph that is drifting toward disconnection must feel a pull" —
    #      which the product cannot provide, its gradient being ~5e-09.
    # A hinge reconciles them by spending gradient only once the penalty
    # crosses a threshold that the DATA says a correct complex never reaches.
    #
    # MEASURED on a 7-atom chain, hinge_tau=0.35:
    #     case            value      |dL/dlogits|max
    #     connected p=.9   0.0000     0.000e+00   <- pinned property preserved
    #     metal isolated   0.1388     4.451e-04   <- alive, ~7e4x the product
    #     weak bonds p=.3   0.0000     0.000e+00
    #     all isolated      0.6500     -
    #
    # ``hinge_tau`` is deliberately NOT defaulted. A threshold is a statement
    # about a corpus; picking one by eye is how the term ends up either inert or
    # dominating. It is derived from tmQM in
    # ``metrics/by_metric/cfm_assemble_hinge_tau.json`` and passed in.
    if hinge_tau is not None:
        loss = (loss - float(hinge_tau)).clamp(min=0.0)

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
    # Both operands are loop-invariant and were being rebuilt every round: an
    # (N, N) comparison and an (N, N) allocation, per round, for a predicate
    # and a constant that cannot change while the loop runs.
    adj_pos = adj > 0
    absent = torch.full((1, n), n, device=adj.device, dtype=torch.long)
    for it in range(n + 1):
        # Every atom takes the smallest label among itself and its neighbours;
        # this is monotone, so the loop converges in <= n rounds.
        nbr = torch.where(adj_pos, labels.unsqueeze(0), absent)
        new = torch.minimum(labels, nbr.min(dim=1).values)
        # `bool(torch.equal(...))` is a device comparison read on the host, so
        # it drains the whole CUDA queue — once per round, up to N+1 times,
        # for a call that runs once per cloud per optimizer step. Testing only
        # every _COMPONENT_CHECK_EVERY rounds returns the IDENTICAL partition:
        # the update is idempotent at its fixed point (if `new == labels` then
        # the next round recomputes from an unchanged `labels` and returns the
        # same thing), so any round past convergence is a no-op and skipping
        # the intermediate tests cannot skip a real update. The cost of the
        # amortisation is at most _COMPONENT_CHECK_EVERY-1 extra rounds.
        if it % _COMPONENT_CHECK_EVERY == 0 and bool(torch.equal(new, labels)):
            break
        labels = new
    uniq, counts = torch.unique(labels, return_counts=True)
    largest = float(counts.max().item()) / float(n)
    return float(uniq.numel()), largest


def batch_assemble_loss(
    per_cloud: Optional[list],
    device: torch.device,
    *,
    formulation: str = "product",
    hinge_tau: Optional[float] = None,
    sharpness: float = 0.05,
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
            logits, edge_index, z,
            formulation=formulation,
            hinge_tau=hinge_tau,
            sharpness=sharpness,
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
