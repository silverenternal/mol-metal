"""Metal-aware D-MPNN (I1 + I3), with the metal-at-readout control (C2) built in.

Positioning
===========
chemprop v2 and every prior model here are **metal-blind** (the metal is absent
from the input; G2 MEASURED 0/19135 Ru rows carry a metal).  This class makes the
metal a first-class input, and — crucially — lets us place it at ONE of three
sites so the injection position is a single controlled variable:

``metal_injection``
    * ``"none"``    — metal-blind baseline (ignores ``metal_feats``).
    * ``"readout"`` — C2: concat the projected metal vector to the **pooled
                      readout** (exactly the PA15 harness family, incl. FiLM-like
                      gating).  This is the NEGATIVE CONTROL.
    * ``"message"`` — I3: inject the projected metal vector into the **GRU edge
                      update** so the metal modulates message passing itself.

``use_metal_node`` is a *data-side* flag (see ``molmetal.data.metal_graph``): when
True the graph carries an appended metal node with coordination bonds (I1), so the
metal also participates as a node.  The model is agnostic to it — it only sees a
graph of ``V`` atoms.

EX1 (the paper's core causal contrast) is therefore:
``message`` vs ``readout`` at identical backbone / protocol / split / budget.

Reuse (H5)
==========
Subclasses ``molmetal.baselines.dmpnn.DMPNNModel`` (atom_emb / bond_emb /
edge_init / _GRUUpdate / readout_mlp) and mirrors its flat-batch forward so the
two share the exact parameter names where possible.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from molmetal.baselines.dmpnn import (
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    DMPNN_HIDDEN,
    DMPNNModel,
    _GRUUpdate,
)
from molmetal.data.metal_graph import METAL_FEAT_DIM

_VALID_INJECTION = ("none", "readout", "message", "film")


class _FiLM(nn.Module):
    """Per-layer FiLM: ``h -> h * (1 + gamma(m)) + beta(m)``.

    Zero-initialised so the layer starts as an exact identity (an un-adapted
    ``film`` model equals the metal-blind ``none`` model at init — the METAL_GATE
    property).  ``cond`` is the per-edge metal context ``(E, cond_dim)``.
    """

    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, cond: torch.Tensor):
        g, b = self.to_gb(cond).chunk(2, dim=-1)
        return g, b


class _MetalAwareGRU(nn.Module):
    """GRU edge update whose input is ``[neighbor_sum, h_v_src, metal_ctx]`` (3H)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.W_z = nn.Linear(hidden * 3, hidden, bias=False)
        self.W_r = nn.Linear(hidden * 3, hidden, bias=False)
        self.W_h = nn.Linear(hidden * 3, hidden, bias=False)
        self.b_z = nn.Parameter(torch.zeros(hidden))
        self.b_r = nn.Parameter(torch.zeros(hidden))
        self.b_h = nn.Parameter(torch.zeros(hidden))

    def forward(self, h_e, h_v_src, neighbor_h_e_sum, metal_ctx):
        combined = torch.cat([neighbor_h_e_sum, h_v_src, metal_ctx], dim=-1)
        z = torch.sigmoid(self.W_z(combined) + self.b_z)
        r = torch.sigmoid(self.W_r(combined) + self.b_r)
        h_tilde = torch.tanh(
            self.W_h(torch.cat([r * neighbor_h_e_sum, h_v_src, metal_ctx], dim=-1))
            + self.b_h
        )
        return (1 - z) * h_e + z * h_tilde


def _readout_mlp(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


class _LowRankAdapter(nn.Module):
    """I4 — residual low-rank bottleneck on the pooled metal-aware representation.

    ``h -> h + W_up(relu(W_down(h)))``.  The down-projection is zero-initialised
    so the adapter starts as an exact identity: an un-adapted model is unchanged,
    and FSMA adaptation can move the metal-condition through this bottleneck
    instead of only the final head (the WM1 low-K failure mode).
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(torch.relu(self.down(h)))


class MetalAwareDMPNN(DMPNNModel):
    """D-MPNN that can consume the metal at the message site (I3) or readout (C2)."""

    def __init__(
        self,
        metal_feat_dim: int = METAL_FEAT_DIM,
        metal_injection: str = "message",
        hidden: int = DMPNN_HIDDEN,
        depth: int = DMPNN_DEPTH,
        dropout: float = DMPNN_DROPOUT,
        use_adapter: bool = False,
        adapter_rank: int = 8,
    ):
        if metal_injection not in _VALID_INJECTION:
            raise ValueError(
                f"metal_injection={metal_injection!r} not in {_VALID_INJECTION}"
            )
        super().__init__(hidden=hidden, depth=depth, dropout=dropout)
        self.metal_injection = metal_injection
        self.metal_proj = nn.Linear(metal_feat_dim, hidden)

        # I4: residual low-rank adapter on the pooled representation.
        self.use_adapter = bool(use_adapter)
        self.adapter = _LowRankAdapter(hidden * 2, adapter_rank) if use_adapter else None

        if metal_injection == "message":
            # Replace the 2H-input GRU with the 3H metal-aware one.
            self.gru_update = _MetalAwareGRU(hidden)
        elif metal_injection == "readout":
            # Widen the readout to accept [pooled(2H) || metal_ctx(H)].
            self.readout_mlp = _readout_mlp(hidden * 3, hidden, dropout)

        # W6 (#99): per-layer FiLM — a separate scale/shift per message-passing
        # round, conditioned on the metal context.  Zero-init => identity start.
        self.film = (
            nn.ModuleList([_FiLM(hidden, hidden) for _ in range(depth)])
            if metal_injection == "film" else None
        )

    # ------------------------------------------------------------------
    # Flat disjoint-union forward (molecules have different V/E).
    # ------------------------------------------------------------------
    def forward_graphs(
        self, graphs: Sequence[Optional[Tuple]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Run over a list of ``(af, bf, es, ed, mf)`` tuples (``None`` skipped).

        Returns ``(B,)`` logits for the valid molecules, in order.
        """
        valid = [g for g in graphs if g is not None]
        B = len(valid)
        if B == 0:
            return torch.zeros(0)
        dev = device or torch.from_numpy(valid[0][0]).device

        atom_list, bond_list, src_list, dst_list, atom_batch, edge_batch, mf_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        atom_off = 0
        for b, (af, bf, es, ed, mf) in enumerate(valid):
            nV, nE = af.shape[0], bf.shape[0]
            atom_list.append(torch.from_numpy(np.asarray(af, dtype=np.float32)))
            bond_list.append(torch.from_numpy(np.asarray(bf, dtype=np.float32)))
            src_list.append(torch.from_numpy(np.asarray(es, dtype=np.int64)) + atom_off)
            dst_list.append(torch.from_numpy(np.asarray(ed, dtype=np.int64)) + atom_off)
            atom_batch.append(torch.full((nV,), b, dtype=torch.long))
            edge_batch.append(torch.full((nE,), b, dtype=torch.long))
            mf_list.append(torch.from_numpy(np.asarray(mf, dtype=np.float32)))
            atom_off += nV

        V = torch.cat(atom_list).to(dev)
        E = torch.cat(bond_list).to(dev)
        src = torch.cat(src_list).to(dev)
        dst = torch.cat(dst_list).to(dev)
        atom_batch_t = torch.cat(atom_batch).to(dev)
        edge_batch_t = torch.cat(edge_batch).to(dev)
        mf = torch.stack(mf_list).to(dev)  # (B, metal_feat_dim)
        nV_total = V.size(0)

        h_v = self.dropout(torch.relu(self.atom_emb(V)))  # (N, H)
        h_e = self.dropout(torch.relu(self.bond_emb(E)))  # (E, H)
        metal_ctx = self.metal_proj(mf)  # (B, H)
        metal_edge = metal_ctx[edge_batch_t]  # (E, H)

        for layer in range(self.depth):
            atom_incoming = torch.zeros(nV_total, self.hidden, device=dev)
            atom_incoming.index_add_(0, dst, h_e)
            neighbor_sum = atom_incoming[src]
            if self.metal_injection == "message":
                h_e = self.gru_update(h_e, h_v[src], neighbor_sum, metal_edge)
            else:
                h_e = self.gru_update(h_e, h_v[src], neighbor_sum)
            if self.film is not None:
                gamma, beta = self.film[layer](metal_edge)
                h_e = h_e * (1.0 + gamma) + beta

        outgoing = torch.zeros(nV_total, self.hidden, device=dev)
        outgoing.index_add_(0, src, h_e)
        h_comb = torch.cat([h_v, outgoing], dim=-1)  # (N, 2H)
        pooled = torch.zeros(B, h_comb.size(-1), device=dev)
        pooled.index_add_(0, atom_batch_t, h_comb)  # (B, 2H)

        # I4: adapter on the 2H pooled metal-aware representation, BEFORE the
        # readout-mode metal concat (so its dim is always 2H).
        if self.adapter is not None:
            pooled = self.adapter(pooled)

        if self.metal_injection == "readout":
            pooled = torch.cat([pooled, metal_ctx], dim=-1)  # (B, 3H)

        return self.readout_mlp(pooled).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # Padded forward (parity with the base class signature) — only valid
    # for metal_injection != "message" without metal_feats; kept minimal.
    # ------------------------------------------------------------------
    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None,
        metal_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Padded per-molecule forward.  Requires ``metal_feats`` unless
        ``metal_injection == 'none'``."""
        if self.metal_injection != "none" and metal_feats is None:
            raise ValueError("metal_feats required when metal_injection != 'none'")
        B = atom_feats.size(0)
        device = atom_feats.device
        logits = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = bond_feats[b].abs().sum(dim=-1) > 0
            E = int(valid_edges.sum().item())
            if E == 0 or V == 0:
                logits.append(torch.tensor(0.0, device=device))
                continue
            mf = metal_feats[b : b + 1] if metal_feats is not None else None
            graphs = [
                (
                    atom_feats[b, :V].cpu().numpy(),
                    bond_feats[b, :E].cpu().numpy(),
                    edge_src[b, :E].cpu().numpy(),
                    edge_dst[b, :E].cpu().numpy(),
                    (mf[0].cpu().numpy() if mf is not None else np.zeros(METAL_FEAT_DIM, np.float32)),
                )
            ]
            logits.append(self.forward_graphs(graphs, device=device)[0])
        return torch.stack(logits, dim=0)


__all__ = ["MetalAwareDMPNN", "_MetalAwareGRU", "_FiLM"]
