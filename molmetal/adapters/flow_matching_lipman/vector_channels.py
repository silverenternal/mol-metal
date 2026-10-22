# SPDX-License-Identifier: Apache-2.0
"""Vector-channel equivariant coordinate head for the CFM velocity field.

WHY THIS MODULE EXISTS
----------------------
``EGNNVelocityField`` composes its coordinate output as

    vel = tanh(vel_head(h)) * vel_scale * (x - centroid) + last_v

and ``last_v`` is ``(1 / max(N-1, 1)) * sum_j tanh(phi_ij) * (x_j - x_i)``
(``models/velocity_net.py:287-300``).  Both terms are bounded:

* the radial term is the only one with a free magnitude, and it can only
  move an atom along the ray from the cloud centroid;
* the edge term's coefficients are ``tanh``-bounded *and* divided by
  ``N-1``, so it is a mean of neighbour directions — it cannot exceed the
  mean pair distance whatever the weights are.

Wf-A107h MEASURED the consequence: 200 AdamW steps at 10x the training
learning rate on ONE 8-cloud batch reached MSE 10.529 against a 10.652
``v = 0`` baseline and a 5.783 radial-only ceiling, with
``|v_pred| = 0.298`` against ``|target| = 5.159`` and cosine 0.128.
That is not slow convergence; the head cannot express the target.

WHAT THIS MODULE DOES
---------------------
It replaces the single scalar-times-ray term with ``C`` independent
vector channels per atom, following the construction that the 2024-2026
3D-generation literature converged on:

* GVP (Jing et al., ICML 2021, arXiv:2009.03092) — scalar gating of
  vector features, which is how a network gets more than one vector
  channel without a tensor product;
* MiDi (Vignac et al., ICLR 2023, arXiv:2303.04791) — per-atom RMS
  normalisation of the channel stack, which is what keeps a deep stack
  of vector channels numerically stable;
* SemlaFlow (Irwin et al., AISTATS 2025, arXiv:2406.07266, Eq. 4/10-13)
  — per-channel scalar times relative displacement, aggregated per
  destination, with the coordinate treated as one more channel rather
  than as a special object;
* FlowMol3 (Dunn & Koes, arXiv:2508.12629, Eq. 18) — the coordinate
  update is a learned projection of the vector features, not a gate on
  a fixed ray.

Per edge ``i -> j`` the message is

    m_ij^c = gate_ij^c * V_j^c + phi_ij^c * (x_j - x_i),    c = 0..C-1

which is ``O(E * C)`` rather than Semla's ``O(E * C^2)`` channel
mixing.  After aggregation the layer applies a shared channel-mixing
linear (zero-init, so a fresh layer is exactly the gated sum) and an
RMS normalisation with a learnable per-channel gain.  The output head is

    u_i = sum_c w_c(h_i) * V_i^c

with ``w(h)`` zero-init, so a freshly built field emits a zero velocity
exactly like the legacy ``vel_head`` does.

EQUIVARIANCE
------------
Every operation is either a scalar gate on a vector, a scalar times a
relative displacement, a linear combination of vectors, or a scalar
statistic of vector norms.  All of them commute with a rotation ``R``
and are invariant to a translation ``b``, so
``u(Rx + b) = R u(x)`` holds to float32 precision (tested in
``molmetal/tests/test_wfa108_vector_channels.py``).

BIT-EXACT DEFAULT
-----------------
Nothing here is constructed unless ``velocity_head="vector_channels"``
is passed.  The default ``"radial"`` path keeps ``state_dict`` keys and
forward numerics identical to the pre-Wf-A108 code.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models._scatter import scatter_sum

# Metals the corpus actually contains (Pt Ru Ir Rh Os Re) plus the rest of
# the d-block / coinage set, so a cloud with an Fe or Cu centre still gets a
# metal-relative channel.  This is a *feature* set, not a chemistry claim.
METAL_ATOMIC_NUMBERS = frozenset(
    {
        25, 26, 27, 28, 29, 30,          # Mn Fe Co Ni Cu Zn
        44, 45, 46, 47, 48,              # Ru Rh Pd Ag Cd
        75, 76, 77, 78, 79, 80,          # Re Os Ir Pt Au Hg
    }
)


class VectorChannelLayer(nn.Module):
    """One equivariant message-passing layer with ``C`` vector channels.

    Parameters
    ----------
    hidden_dim : int
        Width of the invariant (scalar) node state.
    n_channels : int, default 8
        Number of independent 3-vectors carried per atom.  ``C = 8`` keeps
        the edge message at ``O(E * 8)``; SemlaFlow uses 64 because it also
        runs attention over them, which we do not.
    edge_hidden : int, optional
        Width of the edge MLP.  Defaults to ``hidden_dim``.
    eps : float
        RMS-normalisation floor.  Also keeps padded atoms (all-zero
        channels) at exactly zero instead of ``0 / 0``.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_channels: int = 8,
        edge_hidden: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be >= 1")
        if n_channels < 1:
            raise ValueError("`n_channels` must be >= 1")
        edge_hidden = int(edge_hidden or hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_channels = int(n_channels)
        self.eps = float(eps)

        # Edge input: h_src, h_dst, dist, dist^2, is_metal_edge.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 3, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.SiLU(),
        )
        self.gate_head = nn.Linear(edge_hidden, self.n_channels)
        self.phi_head = nn.Linear(edge_hidden, self.n_channels)
        self.msg_scalar_head = nn.Linear(edge_hidden, self.hidden_dim)

        # Node update: [h, agg_scalar, rms] + cond -> 3H + 1.
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Shared channel mixing, zero-init so a fresh layer is exactly the
        # gated sum.  A zero-init *residual* is what keeps the block from
        # changing the initial function.
        self.channel_mix = nn.Linear(self.n_channels, self.n_channels, bias=False)
        nn.init.zeros_(self.channel_mix.weight)

        # Per-channel gain after RMS normalisation (MiDi's `set_weights`).
        self.norm_gain = nn.Parameter(torch.ones(1, 1, self.n_channels, 1))

        # Small phi init: the radial injection must be non-zero so channels
        # 1..C-1 can fill (they start at zero), but large enough values at
        # init would make the first epochs a scale search.
        nn.init.normal_(self.phi_head.weight, std=0.02)
        nn.init.zeros_(self.phi_head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_node: torch.Tensor,
        positions: torch.Tensor,
        vec: torch.Tensor,
        edge_index: torch.Tensor,
        cond_per_node: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        metal_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one vector-channel layer.

        Parameters
        ----------
        h_node : ``(B, N, H)``
        positions : ``(B, N, 3)``
        vec : ``(B, N, C, 3)``
        edge_index : ``(B, 2, E)``
        cond_per_node : ``(B, N, H)``, optional
        edge_mask : ``(B, E)`` bool, optional
        metal_mask : ``(B, N)`` bool, optional
            True at metal atoms.  Only used to build the ``is_metal_edge``
            scalar, so dative edges are distinguishable from covalent ones
            without threading a new input tensor through every caller.

        Returns
        -------
        ``(h_next, vec_next)`` with the same shapes as the inputs.
        """
        b, n, h = h_node.shape
        e = edge_index.shape[2]
        c = self.n_channels

        src = edge_index[:, 0]  # (B, E)
        dst = edge_index[:, 1]  # (B, E)

        h_src = torch.gather(h_node, 1, src.unsqueeze(-1).expand(b, e, h))
        h_dst = torch.gather(h_node, 1, dst.unsqueeze(-1).expand(b, e, h))

        pos_src = torch.gather(positions, 1, src.unsqueeze(-1).expand(b, e, 3))
        pos_dst = torch.gather(positions, 1, dst.unsqueeze(-1).expand(b, e, 3))
        rel = pos_src - pos_dst  # (B, E, 3)
        dist = rel.norm(dim=-1, keepdim=True)  # (B, E, 1)

        if metal_mask is None:
            is_metal_edge = torch.zeros(b, e, 1, device=h_node.device, dtype=h_node.dtype)
        else:
            m_src = torch.gather(metal_mask, 1, src).to(h_node.dtype)
            m_dst = torch.gather(metal_mask, 1, dst).to(h_node.dtype)
            is_metal_edge = (m_src + m_dst).clamp(max=1.0).unsqueeze(-1)

        edge_in = torch.cat([h_src, h_dst, dist, dist * dist, is_metal_edge], dim=-1)
        edge_emb = self.edge_mlp(edge_in)  # (B, E, edge_hidden)

        gate = torch.sigmoid(self.gate_head(edge_emb))  # (B, E, C) in (0, 1)
        phi = torch.tanh(self.phi_head(edge_emb))  # (B, E, C) in (-1, 1)
        m_scalar = self.msg_scalar_head(edge_emb)  # (B, E, H)

        # Gather the destination atom's vector channels.
        vec_dst = torch.gather(
            vec, 1, dst.unsqueeze(-1).unsqueeze(-1).expand(b, e, c, 3)
        )  # (B, E, C, 3)
        vec_msg = gate.unsqueeze(-1) * vec_dst + phi.unsqueeze(-1) * rel.unsqueeze(2)

        src_flat = src.reshape(b * e)
        dst_flat = dst.reshape(b * e)
        mask_flat = edge_mask.reshape(b * e) if edge_mask is not None else None

        agg_vec = scatter_sum(
            src=src_flat,
            dst=dst_flat,
            features=vec_msg.reshape(b * e, c * 3),
            n_atoms=n,
            mask=mask_flat,
            batch_size=b,
        ).reshape(b, n, c, 3)

        # Degree normalisation (sqrt, not mean): keeps a dense neighbour
        # shell from outvoting a sparse one while still bounding growth.
        ones = torch.ones(b * e, 1, device=h_node.device, dtype=h_node.dtype)
        deg = scatter_sum(
            src=src_flat,
            dst=dst_flat,
            features=ones,
            n_atoms=n,
            mask=mask_flat,
            batch_size=b,
        ).reshape(b, n, 1)
        vec = vec + agg_vec / deg.clamp_min(1.0).sqrt().unsqueeze(-1)

        # Shared channel mixing (zero-init residual).
        # ``channel_mix.weight`` is ``(C_out, C_in)``; the channel axis is
        # dim 2 of ``vec`` and dim 3 is xyz.
        vec = vec + torch.einsum("bncd,ec->bned", vec, self.channel_mix.weight)

        # Per-atom, per-channel unit normalisation with a learnable gain.
        #
        # Why per channel and not MiDi's joint RMS: channel 0 starts at
        # ``x - centroid`` (norm ~ cloud radius) while channels 1..C-1 start
        # at zero, so a joint RMS is dominated by channel 0 and suppresses
        # the others — measured on the Wf-A108 gate, that made the head
        # effectively radial again (a tangential target stayed at the radial
        # ceiling after 200 steps).  Unit-normalising each channel makes the
        # head a learned combination of C direction fields; the magnitude
        # lives entirely in ``VectorChannelHead``'s gate, which is
        # unconstrained.  Padded atoms have all-zero channels, so
        # ``0 / eps == 0`` and the mask needs no special case.
        norms = vec.norm(dim=-1, keepdim=True)  # (B, N, C, 1)
        vec = vec * self.norm_gain / (norms + self.eps)
        rms = norms.mean(dim=2)  # (B, N, 1) — the magnitude feature for h

        agg_scalar = scatter_sum(
            src=src_flat,
            dst=dst_flat,
            features=m_scalar.reshape(b * e, h),
            n_atoms=n,
            mask=mask_flat,
            batch_size=b,
        ).reshape(b, n, h)

        node_in = [h_node, agg_scalar, rms]
        if cond_per_node is None:
            # The update MLP has a fixed 3H+1 input width; an absent
            # condition is a zero condition, not a narrower matmul.
            node_in.append(torch.zeros_like(h_node))
        else:
            if cond_per_node.shape != h_node.shape:
                raise ValueError(
                    "`cond_per_node` must have the same shape as `h_node`; "
                    f"got {tuple(cond_per_node.shape)} vs {tuple(h_node.shape)}."
                )
            node_in.append(cond_per_node)
        h_next = h_node + self.update_mlp(torch.cat(node_in, dim=-1))
        return h_next, vec


class VectorChannelHead(nn.Module):
    """``u_i = sum_c w_c(h_i) * V_i^c`` with a zero-init per-atom gate.

    Zero-init means a freshly built field emits exactly zero velocity, which
    is the same contract the legacy ``vel_head`` has (``nn.init.zeros_`` at
    ``flow_matching_lipman/__init__.py:1642``).  It does NOT starve the
    vector channels of gradient: ``dL/dw_c = -2 * <target, V_c>`` is
    non-zero because ``V_c`` is non-zero, so the gate learns first and the
    channels refine afterwards.
    """

    def __init__(self, hidden_dim: int, n_channels: int = 8) -> None:
        super().__init__()
        self.channel_gate = nn.Linear(hidden_dim, n_channels, bias=False)
        nn.init.zeros_(self.channel_gate.weight)

    def forward(self, h_node: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        w = self.channel_gate(h_node)  # (B, N, C)
        return (w.unsqueeze(-1) * vec).sum(dim=2)  # (B, N, 3)


def initial_vector_channels(
    positions: torch.Tensor,
    n_channels: int,
    metal_mask: Optional[torch.Tensor] = None,
    metal_frame_channels: bool = False,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the ``(B, N, C, 3)`` channel stack the layers start from.

    Channel 0 is the old radial basis ``x - centroid``, so the head can
    *recover* the previous behaviour but is not limited to it.  When
    ``metal_frame_channels`` is on:

    * channel 1 is ``x_i - x_M`` for the nearest metal ``M`` in the same
      cloud (zero when the cloud has no metal);
    * channel 2 is ``x_i - centroid(donors)``, where a donor is any atom
      within 2.8 A of that metal.

    Those are the two bases TMCgen (arXiv:2606.00666) argues carry the
    informative degrees of freedom of a transition-metal complex: the
    metal-ligand ray and the coordination shell.
    """
    b, n = positions.shape[:2]
    dtype, device = positions.dtype, positions.device
    vec = torch.zeros(b, n, n_channels, 3, dtype=dtype, device=device)
    if node_mask is None:
        node_mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask_f = node_mask.to(dtype)

    # Centroid over REAL atoms only.  The trainer pads clouds up to a bucket
    # size with zero positions, so a plain mean over dim=1 shifts the centroid
    # toward the origin by (n_pad / n) — the same defect the radial head has
    # had since the padding path was added, and one that would silently
    # corrupt channel 0 here.
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
    centroid = (
        (positions * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True)
        / n_valid.unsqueeze(-1)
    )
    vec[:, :, 0, :] = (positions - centroid) * mask_f.unsqueeze(-1)

    if not metal_frame_channels or n_channels < 3:
        return vec

    if metal_mask is None:
        metal_mask = torch.zeros(b, n, dtype=torch.bool, device=device)

    # (B, N, 1, 3) metal positions; a cloud without a metal gets a zero
    # placeholder that is masked out below.
    metal_pos = positions * metal_mask.unsqueeze(-1).to(dtype)
    n_metal = metal_mask.sum(dim=1).clamp_min(1)
    metal_centroid = metal_pos.sum(dim=1, keepdim=True) / n_metal.view(b, 1, 1)

    has_metal = metal_mask.any(dim=1).view(b, 1, 1).to(dtype)
    vec[:, :, 1, :] = (positions - metal_centroid) * has_metal

    # Donor shell: distance to the metal centroid, not to a specific metal.
    # With one metal per cloud (the corpus case) they are the same thing;
    # with several the shell is the union, which is still a useful basis.
    d_metal = (positions - metal_centroid).norm(dim=-1)  # (B, N)
    # The metal itself is at distance 0 from the centroid; it is not a donor.
    #
    # ``node_mask`` is load-bearing here and was missing until Wf-A108g.
    # Padded atoms sit at the origin, so their distance to the metal centroid
    # is |metal_centroid| — which is inside the 2.8 A shell whenever the metal
    # is near the origin.  They then entered ``donor_f`` and dragged
    # ``donor_centroid`` toward zero by a factor ``n_valid / n_max``.
    # MEASURED: on a 4-cloud synthetic batch the channel stack moved by
    # 7.1e-1 (5% of its 1.47e1 scale) when the pad width grew from 40 to 50,
    # i.e. the "metal frame" channel was a function of the BATCH's largest
    # molecule rather than of the molecule.  Excluding padded atoms makes the
    # channel padding-invariant, which is what lets the trainer split a batch
    # into size-homogeneous groups without changing what is computed.
    donor = (
        (d_metal < 2.8)
        & ~metal_mask
        & metal_mask.any(dim=1, keepdim=True)
        & node_mask
    )
    donor_f = donor.to(dtype).unsqueeze(-1)
    donor_pos = positions * donor_f
    n_donor = donor_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    donor_centroid = donor_pos.sum(dim=1, keepdim=True) / n_donor
    # Every atom gets the vector to the coordination-shell centroid, not only
    # the donors: the direction "away from the donor shell" is informative for
    # a ligand atom too, and masking it to donors only made channel 2 zero on
    # 90% of the cloud.
    vec[:, :, 2, :] = (positions - donor_centroid) * has_metal

    # Padded atoms carry no signal: zero every channel so a unit-normalising
    # layer cannot amplify a placeholder into a direction.
    return vec * mask_f.unsqueeze(-1).unsqueeze(-1)


def build_metal_mask(atom_types: torch.Tensor) -> torch.Tensor:
    """``(B, N)`` bool, True at atoms in :data:`METAL_ATOMIC_NUMBERS`."""
    z = atom_types.long()
    mask = torch.zeros_like(z, dtype=torch.bool)
    for z_value in METAL_ATOMIC_NUMBERS:
        mask |= z == z_value
    return mask
