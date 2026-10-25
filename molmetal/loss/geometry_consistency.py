# SPDX-License-Identifier: Apache-2.0
"""Geometry consistency of the model's OWN prediction (cfm_todo MOD2 + MOD3).

Why this is not ``clash_penalty_weight``
---------------------------------------
The trainer already has a clash penalty (Wf-A11/A27), but it is computed on
``cloud.positions`` — the ground-truth geometry of the training batch — and its
per-cloud scalar is ``.detach()``-ed before it reaches the loss.  A term
evaluated on data, then detached, contributes a constant: it can be logged and
it cannot be trained.  That is the "live but inert" shape.

This module scores ``x1_hat``: the endpoint the model itself predicts, which is
a differentiable function of every parameter that produced it.  Three parts,
all hinge-squared so a satisfied pair contributes exactly zero:

``clash``
    non-bonded pairs closer than ``clash_frac * (r_i + r_j)``
``bond``
    bonded pairs outside ``[bond_lo, bond_hi] * (r_i + r_j)``
``metal``
    metal-ligand pairs outside the per-metal band

The metal band is MEASURED, not chosen: ``metrics/by_metric/cfm_metal_geometry.json``
(REAL_DFT cohort, 18,518 M-L pairs across the six train metals) gives the
p10/p90 of each metal's bond length.  Using that band as the target is the
whole of MOD3 — a metal-specific prior rather than one constant.

What this term can and cannot fix
---------------------------------
It can move geometry: clashes, stretched bonds, and metal centres at 1.23 A
when the real p10 is 1.95 A.  It cannot by itself connect two fragments that
the sampler placed far apart — the hinge is zero once the pair is at a sane
distance, and "these two atoms should be bonded" is a decoder decision, not a
distance.  ``single_fragment_rate`` is therefore the weakest of its three
targets and the one to check first for an honest negative.

CPU cost
--------
MEASURED 2026-10-25 on this host at B=64, A=57: 51 ms with one torch thread,
3.0 s with the default thread count.  The tensors are small, so 90+ threads
spend their time synchronising.  CPU smokes should set ``OMP_NUM_THREADS=1``
(the same rule the repo already uses for CPU per-molecule forwards); on GPU the
call is a few milliseconds and the thread count does not matter.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

#: Atomic numbers of the six training metals.
METAL_Z = (78, 44, 77, 45, 76, 75)

#: Per-metal metal-ligand band, in Angstrom, as (p10, p90) of the REAL_DFT
#: cohort in ``metrics/by_metric/cfm_metal_geometry.json``.  Keyed by atomic
#: number.  Measured 2026-10-12 over 18,518 M-L pairs.
METAL_ML_BAND_A: Dict[int, Tuple[float, float]] = {
    78: (1.979, 2.466),  # Pt, n=3571
    44: (1.939, 2.371),  # Ru, n=5723
    77: (2.005, 2.360),  # Ir, n=3549
    45: (1.995, 2.351),  # Rh, n=2374
    76: (1.870, 2.439),  # Os, n=844
    75: (1.938, 2.427),  # Re, n=2457
}

#: Fallback band for a metal outside the six (used only if a caller widens the
#: cohort).  It is the union of the measured bands above, not a guess.
DEFAULT_ML_BAND_A: Tuple[float, float] = (1.870, 2.466)


def geometry_consistency(
    x: torch.Tensor,
    z: torch.Tensor,
    batch_idx: torch.Tensor,
    bonds: Optional[torch.Tensor],
    radius: torch.Tensor,
    n_clouds: int,
    *,
    clash_frac: float = 0.7,
    bond_lo: float = 0.85,
    bond_hi: float = 1.25,
    metal_bands: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Dict[str, torch.Tensor]:
    """Score one predicted geometry (``x1_hat``) per cloud.

    Parameters
    ----------
    x
        ``(N, 3)`` predicted positions, differentiable.
    z
        ``(N,)`` atomic numbers.
    batch_idx
        ``(N,)`` cloud index per atom.
    bonds
        ``(2, E)`` global bonded pairs, or ``None``.
    radius
        ``(N,)`` per-atom covalent radius (the batch's ``atomic_radius``).
    n_clouds
        number of clouds in the batch.

    Returns
    -------
    dict with ``clash``, ``bond``, ``metal`` (each a scalar mean over the
    pairs that exist, or an exact zero when there are none) plus
    ``n_clash_pairs`` / ``n_bond_pairs`` / ``n_metal_pairs`` as floats so the
    caller can log what was actually scored.

    A cloud with no bonded pairs and no metal contributes zero to every part;
    the means are taken over pairs, not over clouds, so a batch of one large
    complex is not outvoted by a batch of hydrides.
    """
    bands = METAL_ML_BAND_A if metal_bands is None else metal_bands
    n_total = int(x.shape[0])
    n_clouds = int(n_clouds)
    zero = x.sum() * 0.0
    if n_total == 0 or n_clouds <= 0:
        return {
            "clash": zero,
            "bond": zero,
            "metal": zero,
            "n_clash_pairs": 0.0,
            "n_bond_pairs": 0.0,
            "n_metal_pairs": 0.0,
        }

    # Pad to (B, A_max, ...) and do every comparison as one batched op.
    #
    # The first version looped over clouds with a cdist and a handful of
    # (A, A) masks each.  MEASURED 2026-10-25 on this host: 6.6 s per call at
    # B=64, A=57 with torch's default thread count, because each cloud's
    # tensors are tiny and 90+ threads spend more time synchronising than
    # computing.  The same call takes 70 ms at one thread.  One batched call is
    # both fast and thread-count independent.
    counts = torch.bincount(batch_idx, minlength=n_clouds)
    offsets = torch.cumsum(counts, 0) - counts
    A = int(counts.max().item())
    pos_local = torch.arange(n_total, device=x.device) - offsets[batch_idx]

    pad = torch.zeros((n_clouds, A), dtype=torch.long, device=x.device)
    pad[batch_idx, pos_local] = torch.arange(n_total, device=x.device)
    mask = torch.arange(A, device=x.device)[None, :] < counts[:, None]

    xp = x[pad]
    zp = z[pad]
    rp = radius[pad].float().clamp_min(0.3)
    d = torch.cdist(xp, xp)
    eq = rp[:, :, None] + rp[:, None, :]
    pair_ok = mask[:, :, None] & mask[:, None, :]

    adj = torch.zeros((n_clouds, A, A), dtype=torch.bool, device=x.device)
    if bonds is not None and bonds.numel():
        src, dst = bonds[0].long(), bonds[1].long()
        ok = (src >= 0) & (src < n_total) & (dst >= 0) & (dst < n_total)
        src, dst = src[ok], dst[ok]
        adj[batch_idx[src], pos_local[src], pos_local[dst]] = True
        adj[batch_idx[dst], pos_local[dst], pos_local[src]] = True

    eye = torch.eye(A, dtype=torch.bool, device=x.device)[None]
    nonbonded = pair_ok & ~adj & ~eye

    # ---- clash: non-bonded pairs inside clash_frac x (r_i + r_j) -----------
    gap = (clash_frac * eq - d).clamp_min(0.0).masked_fill(~nonbonded, 0.0)
    n_clash = int((gap > 0).sum().item()) // 2
    # Mean over violating pairs only: dividing by every pair would make the
    # term shrink as the molecule grows, which is a batch-size effect, not a
    # geometry one.
    clash = gap.pow(2).sum() / float(n_clash) if n_clash else zero

    # ---- bond: bonded pairs outside [lo, hi] x equilibrium -----------------
    rel = d / eq.clamp_min(1e-6)
    dev_b = (bond_lo - rel).clamp_min(0.0) + (rel - bond_hi).clamp_min(0.0)
    dev_b = dev_b.masked_fill(~adj, 0.0)
    n_bond = int(adj.sum().item()) // 2
    bond = dev_b.pow(2).sum() / float(2 * n_bond) if n_bond else zero

    # ---- metal: M-L pairs outside the measured per-metal band --------------
    is_m = torch.zeros((n_clouds, A), dtype=torch.bool, device=x.device)
    for mz in METAL_Z:
        is_m |= (zp == int(mz)) & mask
    ml = pair_ok & (is_m[:, :, None] ^ is_m[:, None, :])
    n_metal = int(ml.sum().item()) // 2
    if n_metal:
        lo = torch.zeros_like(d)
        hi = torch.zeros_like(d)
        for mz in METAL_Z:
            band = bands.get(int(mz), DEFAULT_ML_BAND_A)
            mm = pair_ok & ((zp == int(mz))[:, :, None] ^ (zp == int(mz))[:, None, :])
            lo = torch.where(mm, d.new_full((), float(band[0])), lo)
            hi = torch.where(mm, d.new_full((), float(band[1])), hi)
        dev_m = (lo - d).clamp_min(0.0) + (d - hi).clamp_min(0.0)
        dev_m = dev_m.masked_fill(~ml, 0.0)
        metal = dev_m.pow(2).sum() / float(n_metal)
    else:
        metal = zero

    return {
        "clash": clash,
        "bond": bond,
        "metal": metal,
        "n_clash_pairs": float(n_clash),
        "n_bond_pairs": float(n_bond),
        "n_metal_pairs": float(n_metal),
    }
