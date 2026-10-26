# SPDX-License-Identifier: Apache-2.0
"""Coordination-number conditioning for the CFM velocity field (MOD5, D6).

THE DEFECT
----------
The velocity field knows WHICH metal it is building -- ``metal_conditioning:
embedding`` reads the metal out of ``atom_types`` -- but it knows nothing about
how many donors that metal should have.  MEASURED (c7 pool, 12,669 molecules):
the decoded metal has the wrong degree in a large share of samples, and the
field has no channel that could carry a request even if one were made.

Two consequences follow, and only the second one is about quality:

  * A sample cannot be ASKED for a coordination number.  "Give me an
    octahedral Pt(II)" is not expressible.
  * There is no unconditional/conditional split, so classifier-free guidance
    on a structural property is impossible: guidance needs a null token that
    the model was trained to recognise.

WHAT THIS DOES
--------------
One embedding table over ``cn = 0..cn_max``, added to every atom's input
hidden state, exactly where the existing metal bias is added.  Row 0 is the
NULL token and is initialised to ZERO, so:

  * ``cn_req=None`` (or an all-zero request) contributes exactly nothing and
    the forward is bit-identical to the un-conditioned field;
  * the unconditional branch of classifier-free guidance is the null token,
    which is what ``p(v | cn = ∅)`` means in the CFG formula.

The table is initialised at ``1/sqrt(H)`` rather than at zero for the same
reason the metal table is: a conditioning channel that emits zero until the
optimiser pays for it step by step is the inert-knob failure this repo keeps
finding.  The NULL row is the only zero row, and it is zero on purpose.

The table is drawn from a LOCAL generator (see ``_INIT_SEED``), so building
this module does not advance the process-wide RNG.  Without that, an on/off
A/B of the channel would also differ in every later draw -- the remaining
initialisation, the first dropout mask, the first noise sample -- and the
difference would be read as the channel's effect.

WHY CN IS A LABEL AND NOT AN INPUT
----------------------------------
At training time the request is the cloud's OWN coordination number, computed
from its truth bond graph.  That is a supervision target, in the same sense the
bond-type labels are: it is available for every tmQM complex and never
available for a sample being generated.  Passing it at generation is the
user's request, not a fact about the geometry -- which is exactly what makes
the conditional-success metric meaningful.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not make the field obey the request.  That is the training signal's
job, and whether it obeys is measured on decoded molecules
(``conditional_success``), not asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

#: Largest coordination number the table covers.  tmQM's complexes run 2-8;
#: 12 leaves room for the high-CN lanthanide-like cases without a table big
#: enough to be mostly unreachable rows.
DEFAULT_CN_MAX = 12

#: Seed for the conditioner's own generator.  Fixed, and never the global
#: RNG: constructing a conditioning channel must not change what the rest of
#: the model is initialised to, or every ablation of that channel compares
#: two models rather than one.
_INIT_SEED = 0x5EED


def coordination_number_from_bonds(
    bonds: torch.Tensor,
    pair_cloud_idx: torch.Tensor,
    metal_atom_index: torch.Tensor,
    *,
    n_clouds: int,
) -> torch.Tensor:
    """Coordination number of each cloud's metal atom, from its truth bonds.

    Parameters
    ----------
    bonds : (2, E) int64
        Per-cloud LOCAL atom indices.  This is the frame
        :meth:`molmetal.data.batch.Batch` stores them in: the collate
        concatenates each cloud's ``(2, E_c)`` block verbatim, without adding
        the cloud offset, so pair ``(i, j)`` of cloud ``c`` indexes that
        cloud's own atom axis.
    pair_cloud_idx : (E,) int64
        Which cloud each pair belongs to.
    metal_atom_index : (B,) int64
        Per-cloud LOCAL index of the metal atom (``NO_METAL_INDEX`` = -1 when
        the cloud has none) -- the same frame as ``bonds``.
    n_clouds : int

    Returns
    -------
    (B,) int64, the number of truth bonds incident to the metal.  ``0`` for a
    cloud with no metal: a ligand-only cloud has no coordination number, and
    writing a fabricated one would let the conditioner learn a value that
    never occurs.

    Why count bonds rather than read ``coord_neighbors``: the neighbour slot is
    a padded (N, 12) view with its own conventions about zero entries, and a
    coordination number derived from two different sources is the kind of
    two-definitions defect this repo has already paid for.  The bond list is
    the same object the bond head is trained on, so the label and the thing
    being decoded cannot disagree.
    """
    out = torch.zeros(int(n_clouds), dtype=torch.long, device=metal_atom_index.device)
    if bonds.numel() == 0 or int(metal_atom_index.numel()) == 0:
        return out
    src = bonds[0].to(torch.long)
    dst = bonds[1].to(torch.long)
    cloud = pair_cloud_idx.to(torch.long)
    # One gather instead of a per-cloud mask: this runs inside the training
    # step, and an O(B x E) loop over 64 clouds and ~10k pairs is 640k Python
    # iterations per step.
    metal_of_pair = metal_atom_index.to(torch.long)[cloud]  # (E,)
    # ``metal_of_pair == -1`` (no metal) matches no atom index, since indices
    # are non-negative, so the "no metal" cloud needs no special case.
    hit = (src == metal_of_pair) | (dst == metal_of_pair)
    return out.scatter_add_(0, cloud, hit.to(torch.long))


@dataclass
class ConditionRequest:
    """A per-sample request, kept as plain tensors so it can be serialised.

    ``cn`` is ``0`` for "no request" and that is the null token, not a
    coordination number: no real complex in the corpus has CN 0, and the
    embedding row is the zero row.
    """

    cn: torch.Tensor  # (B,) int64

    @classmethod
    def null(cls, n: int, device: Optional[torch.device] = None) -> "ConditionRequest":
        return cls(cn=torch.zeros(int(n), dtype=torch.long, device=device))

    def dropped(self, p: float, generator: Optional[torch.Generator] = None) -> "ConditionRequest":
        """Replace a Bernoulli ``p`` share of the rows with the null token.

        This is what teaches ``p(v | cn = ∅)``; without it, classifier-free
        guidance at inference is extrapolating from a branch the model never
        saw, which is the failure mode that makes guidance produce garbage at
        high scale.
        """
        if p <= 0.0:
            return self
        keep = (
            torch.rand(self.cn.shape, device=self.cn.device, generator=generator) >= float(p)
        )
        return ConditionRequest(cn=self.cn * keep.to(self.cn.dtype))


class CNConditioner(nn.Module):
    """``(B,)`` coordination numbers -> ``(B, H)`` additive conditioning.

    The output is added to the per-atom input hidden state and broadcast over
    atoms, the same way the metal bias is.  Row 0 is zero, so a null request
    emits an exact zero tensor and the caller's forward is unchanged.
    """

    def __init__(self, hidden_dim: int, cn_max: int = DEFAULT_CN_MAX) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cn_max = int(cn_max)
        # A LOCAL generator, not the global one, and the whole construction is
        # wrapped in fork_rng.  Two separate leaks: ``nn.Embedding``'s own
        # default init draws from the process-wide RNG, and so would
        # ``nn.init.normal_``.  The trainer seeds that RNG once, before
        # building the model -- so merely CONSTRUCTING this module would shift
        # every later draw (the remaining init, the first dropout mask, the
        # first noise sample) and an on/off A/B would compare two different
        # initialisations instead of one channel.  MEASURED consequence of the
        # naive version: the "null request is bit-exact" arm could never be
        # bit-exact, and the difference would have been attributed to the
        # channel.
        _g = torch.Generator(device="cpu").manual_seed(_INIT_SEED)
        with torch.random.fork_rng(devices=[]):
            self.embed = nn.Embedding(self.cn_max + 1, self.hidden_dim)
            with torch.no_grad():
                self.embed.weight.normal_(
                    0.0, float(self.hidden_dim) ** -0.5, generator=_g
                )
        # The null row is zero: this is the bit-exactness device and the CFG
        # unconditional branch, both at once.
        nn.init.zeros_(self.embed.weight[0])
        #: Rows clamped into range, so a corpus with a CN above ``cn_max``
        #: reports a number instead of silently conditioning on the wrong one.
        self.n_clamped = 0

    def forward(self, cn: Optional[torch.Tensor], b: int) -> torch.Tensor:
        """``(B,)`` request -> ``(B, 1, H)`` bias.  ``None`` is the null."""
        if cn is None:
            return torch.zeros(
                (int(b), 1, self.hidden_dim),
                device=self.embed.weight.device,
                dtype=self.embed.weight.dtype,
            )
        cn = cn.to(device=self.embed.weight.device, dtype=torch.long).reshape(-1)
        if int(cn.numel()) != int(b):
            raise ValueError(
                f"cn request has {int(cn.numel())} entries for a batch of {int(b)}"
            )
        over = cn > self.cn_max
        if bool(over.any()):
            self.n_clamped += int(over.sum().item())
            cn = torch.where(over, torch.full_like(cn, self.cn_max), cn)
        neg = cn < 0
        if bool(neg.any()):
            cn = torch.where(neg, torch.zeros_like(cn), cn)
        return self.embed(cn).unsqueeze(1)


def cn_from_cloud(cloud) -> int:
    """Coordination number of ONE cloud's metal, from its own truth bonds.

    The per-cloud counterpart of :func:`coordination_number_from_bonds`, for
    the sampler: a template carries ``bonds`` as ``(2, E)`` LOCAL indices and
    ``metal_atom_index`` as a scalar, so no batch plumbing is needed.

    Returns ``0`` when the cloud has no metal or no truth bonds.  That is the
    null token, and it is the honest value: a composition with no metal has no
    coordination number, and a cloud with no bonds has nothing to count.
    """
    m = getattr(cloud, "metal_atom_index", None)
    if m is None:
        return 0
    m = int(m.item()) if hasattr(m, "item") else int(m)
    if m < 0:
        return 0
    bonds = getattr(cloud, "bonds", None)
    if bonds is None or int(bonds.numel()) == 0:
        return 0
    src = bonds[0].to(torch.long)
    dst = bonds[1].to(torch.long)
    return int(((src == m) | (dst == m)).sum().item())
