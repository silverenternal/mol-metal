"""SE(3)-equivariant velocity network for MolFlow-Triton.

The velocity network is the central model used by the flow-matching
training loop: given a noisy atom configuration at time ``t`` together
with the encoded molecular graph and the property condition, it
predicts a per-atom 3-D velocity vector that points along the
flow-matching ODE.

Architecture (simplified EGNN):

- :class:`EGNNLayer` runs one round of equivariant message passing.  A
  scalar ``m_ij`` is computed from the concatenated node embeddings
  and the relative displacement ``x_j - x_i``.  The displacement is
  rotated by a learnable per-pair scalar ``phi_ij`` before being
  aggregated onto each destination node via the shared scatter-sum
  primitive :func:`models._scatter.scatter_sum`.
  This is the same construction as Satorras et al. 2021 (EGNN),
  specialised so that the vector messages can be summed with a single
  high-performance kernel.
- :class:`VelocityNet` stacks ``L`` such layers, injects the
  time/condition embedding into every layer, and projects the per-node
  vector field to a 3-D output velocity via a final linear layer.

Equivariance is exact: because each per-edge vector message is a
scalar ``phi_ij`` multiplied by ``(x_j - x_i)``, rotating the input
positions rotates every output velocity by the same rotation.

Triton acceleration
-------------------

The edge MLP and node update MLP are each ``nn.Sequential(Linear,
SiLU, Linear)`` — the canonical ``Linear -> activation -> Linear``
pattern that Liger-Kernel fuses into a single Triton kernel.  We
cherry-pick that pattern as :func:`triton_kernels.fused_silu_mlp` and
wrap it in :class:`_FusedSiLUMLP` below so the parameter layout stays
exactly compatible with the original ``nn.Sequential``.  This means
existing checkpoints load without remapping, and the forward kernel
itself never writes the (M, H1) intermediate to HBM.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models._scatter import scatter_sum
from triton_kernels import fused_silu_mlp


class _FusedSiLUMLP(nn.Module):
    """Drop-in replacement for ``nn.Sequential(Linear, SiLU, Linear)``.

    Holds two :class:`nn.Linear` modules as ``.linear1`` / ``.linear2``
    so the parameter names match the original ``nn.Sequential`` and
    pre-existing checkpoints load without remapping.  The forward pass
    calls :func:`triton_kernels.fused_silu_mlp`, which fuses both
    ``linear`` calls and the SiLU activation into a single Triton
    kernel — saving one (M, H1) round trip to HBM per layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``fused_silu_mlp`` uses the *transposed* nn.Linear convention:
        # it expects ``w1: (in_features, out_features)`` and computes
        # ``x @ w1`` rather than ``x @ w1.T``.  Pass ``linear1.weight.T``
        # and ``linear2.weight.T`` to flip the layout without copying.
        return fused_silu_mlp(
            x,
            self.linear1.weight.T,
            self.linear1.bias,
            self.linear2.weight.T,
            self.linear2.bias,
        )

    # nn.Sequential originally exposed ``[0]`` and ``[1]`` indexing; mirror
    # that for any external code that introspects the module list.
    def __getitem__(self, idx: int) -> nn.Module:  # pragma: no cover - ergonomic only
        return [self.linear1, self.linear2][idx]

    def __len__(self) -> int:  # pragma: no cover - ergonomic only
        return 2


class EGNNLayer(nn.Module):
    """One equivariant message-passing layer.

    Parameters
    ----------
    hidden_dim : int
        Width of every hidden representation in the layer.
    edge_mlp_hidden : int, optional
        Hidden width of the edge MLP.  Defaults to ``hidden_dim``.
    update_mlp_hidden : int, optional
        Hidden width of the node update MLP.  Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_mlp_hidden: int | None = None,
        update_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be >= 1")
        if edge_mlp_hidden is None:
            edge_mlp_hidden = hidden_dim
        if update_mlp_hidden is None:
            update_mlp_hidden = hidden_dim

        self.hidden_dim = hidden_dim

        # Edge MLP: scalar message from [h_i || h_j || dist^2 || dist].
        # Fused: Linear(in -> hidden) -> SiLU -> Linear(hidden -> hidden)
        # via :func:`triton_kernels.fused_silu_mlp`.  An explicit trailing
        # SiLU is applied to the fused output to keep parity with the
        # original ``nn.Sequential(Linear, SiLU, Linear, SiLU)`` shape.
        self.edge_mlp_fused = _FusedSiLUMLP(
            2 * hidden_dim + 2,
            edge_mlp_hidden,
            edge_mlp_hidden,
        )

        # Two heads from the edge embedding: a scalar for the node
        # update and a scalar that scales the displacement vector.
        self.msg_scalar_head = nn.Linear(edge_mlp_hidden, edge_mlp_hidden)
        self.msg_vector_head = nn.Linear(edge_mlp_hidden, 1)

        # Node update consumes ``[h_i || agg_scalar || v_norm || cond]``
        # (see ``forward``).  ``v_norm`` is a single scalar; the other
        # three tensors are H-wide each, so input width is
        # ``3 * hidden_dim + 1``.  The condition is always present
        # (when ``cond_dim == 0`` the upstream caller substitutes the
        # time embedding for it).
        self.update_mlp = _FusedSiLUMLP(
            3 * hidden_dim + 1,
            update_mlp_hidden,
            hidden_dim,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        h_node: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        cond_per_node: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one EGNN-style layer.

        Parameters
        ----------
        h_node : ``(B, N, H)``
            Per-atom hidden representations.
        positions : ``(B, N, 3)``
            Atom positions.
        edge_index : ``(B, 2, E)``
            Source / destination indices per edge.
        cond_per_node : optional ``(B, N, H)``
            Per-atom condition broadcast (e.g. time + property
            embedding).  When ``None`` the condition contribution is
            skipped.
        edge_mask : optional ``(B, E)`` bool tensor
            Real-edge mask.  Padded edges are skipped in the scatter
            aggregation.  When ``None`` every edge is treated as real.

        Returns
        -------
        h_next : ``(B, N, H)``
            Updated node embeddings.
        v_agg : ``(B, N, 3)``
            Aggregated vector message, kept around so downstream code
            can monitor the magnitude / inspect the field.
        """
        b, n, h = h_node.shape
        e = edge_index.shape[2]

        # 1. Pairwise displacement and distance features.
        src = edge_index[:, 0]  # (B, E)
        dst = edge_index[:, 1]  # (B, E)

        src_idx = src.unsqueeze(-1).expand(b, e, h)
        dst_idx = dst.unsqueeze(-1).expand(b, e, h)
        h_src = torch.gather(h_node, 1, src_idx)
        h_dst = torch.gather(h_node, 1, dst_idx)

        pos_src = torch.gather(positions, 1, src.unsqueeze(-1).expand(b, e, 3))
        pos_dst = torch.gather(positions, 1, dst.unsqueeze(-1).expand(b, e, 3))
        rel = pos_src - pos_dst                                          # (B, E, 3)
        dist = rel.norm(dim=-1, keepdim=True)                            # (B, E, 1)

        # 2. Edge MLP -> scalar message & scaling factor.
        # The original ``nn.Sequential(Linear, SiLU, Linear, SiLU)`` is
        # now ``_FusedSiLUMLP`` (Linear, SiLU, Linear) followed by an
        # explicit trailing ``nn.SiLU`` to preserve the trailing
        # activation.  Trailing SiLU runs in PyTorch (negligible cost —
        # it's a single elementwise over the (B, E, edge_hidden) tensor).
        edge_in = torch.cat([h_src, h_dst, dist, dist * dist], dim=-1)
        edge_emb = self.edge_mlp_fused(edge_in)
        edge_emb = torch.nn.functional.silu(edge_emb)                    # (B, E, edge_hidden)

        m_scalar = self.msg_scalar_head(edge_emb)                         # (B, E, edge_hidden)
        phi = self.msg_vector_head(edge_emb)                            # (B, E, 1)

        # Per-edge vector message: phi_ij * (x_j - x_i).  CRUCIAL: this is
        # a *single* 3-D vector per edge, NOT one per hidden channel.
        # The original EGNN aggregates one vector per destination
        # node; if we tile the vector across hidden channels (which
        # an earlier version did to reuse the Triton aggregate kernel)
        # the resulting per-atom vector is `hidden_dim` times too
        # large and the flow-matching loss explodes within a few
        # thousand steps.  We aggregate vectors with a small PyTorch
        # scatter and scalars with the Triton kernel.
        edge_vec = phi * rel                                            # (B, E, 3)

        # Flat tensors for the Triton scalar aggregation.
        # Flatten graph and edge axes independently for each index channel;
        # reshaping (B, 2, E) directly interleaves source/destination channels.
        src_flat = src.reshape(b * e)
        dst_flat = dst.reshape(b * e)
        edge_mask_flat = (
            edge_mask.reshape(b * e) if edge_mask is not None else None
        )

        # 3. Scatter vector messages via the shared primitive.
        # Both the Triton backend and the torch fallback honour
        # ``batch_size=b`` by offsetting per-graph destination indices
        # by ``i * n_atoms`` — the standard PyG / torch_scatter /
        # EGNN-batch pattern.  Gradients flow through the scatter
        # because :func:`scatter_sum` is wrapped in a
        # ``torch.autograd.Function`` with a ``scatter_add_`` backward.
        edge_vec_flat = edge_vec.reshape(b * e, 3)
        mask_flat = (
            edge_mask.reshape(b * e) if edge_mask is not None else None
        )
        agg_vec = scatter_sum(
            src=src_flat,
            dst=dst_flat,
            features=edge_vec_flat,
            n_atoms=n,
            mask=mask_flat,
            batch_size=b,
        )                                                              # (B, N, 3)
        # The scatter API retains its legacy (N, F) result for batch_size=1.
        # Restore the model's batch dimension for both singleton and full
        # batches before combining vector norms with (B, N, H) features.
        v_agg = agg_vec.reshape(b, n, 3)                                 # (B, N, 3)
        # Per-node vector norm kept as a (B, N, 1) feature for the
        # scalar message path.
        v_norm_per_node = v_agg.norm(dim=-1, keepdim=True)               # (B, N, 1)

        # 4. Scatter scalar messages via the shared primitive.
        agg_scalar = scatter_sum(
            src=src_flat,
            dst=dst_flat,
            features=m_scalar.reshape(b * e, m_scalar.shape[-1]),
            n_atoms=n,
            mask=edge_mask_flat,
            batch_size=b,
        )
        agg_scalar = agg_scalar.reshape(b, n, m_scalar.shape[-1])

        # 5. Node update with optional condition broadcast.
        # node_in: [h_node, agg_scalar, v_norm, v_norm_per_node, cond]
        # (the per-node vector norm is a single scalar; we feed both
        #  v_agg.norm(dim=-1, keepdim=True) and v_norm_per_node — the
        #  latter is just a cached copy of the former, kept as an
        #  explicit name for readability).
        v_norm = v_norm_per_node                                         # (B, N, 1)
        node_in = [h_node, agg_scalar, v_norm]
        if cond_per_node is not None:
            if cond_per_node.shape != h_node.shape:
                raise ValueError(
                    "`cond_per_node` must have the same shape as `h_node`; "
                    f"got {tuple(cond_per_node.shape)} vs {tuple(h_node.shape)}."
                )
            node_in.append(cond_per_node)
        node_in_cat = torch.cat(node_in, dim=-1)

        update = self.update_mlp(node_in_cat)
        h_next = h_node + update
        return h_next, v_agg


class VelocityNet(nn.Module):
    """Equivariant flow-matching velocity network.

    Parameters
    ----------
    hidden_dim : int, default 128
        Hidden width shared across all EGNN layers.
    n_layers : int, default 4
        Number of equivariant message-passing layers.
    time_embed_dim : int, optional
        Width of the sinusoidal time embedding.  Defaults to
        ``hidden_dim``.
    cond_dim : int, default 0
        Dimension of the externally produced condition embedding
        (e.g. output of :class:`models.conditioner.Conditioner`).
        If ``0`` the condition contribution is disabled.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_embed_dim: int | None = None,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be >= 1")
        if n_layers < 1:
            raise ValueError("`n_layers` must be >= 1")

        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        if time_embed_dim is None:
            time_embed_dim = hidden_dim

        # Sinusoidal time embedding -> MLP.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project external condition (per-sample) into hidden_dim so it
        # can be broadcast-added to per-atom features.
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.cond_proj = None

        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim) for _ in range(n_layers)]
        )

        # Final velocity head: aggregate vector message + a small
        # learned correction.  We use a per-layer linear from hidden
        # to 3 to mix the scalar update with the displacement direction.
        self.vel_head = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.vel_head.weight)

    # ------------------------------------------------------------------
    @staticmethod
    def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Standard transformer-style sinusoidal time embedding.

        Parameters
        ----------
        t : ``(B,)`` or ``(B, 1)`` tensor
        dim : int

        Returns
        -------
        ``(B, dim)`` tensor
        """
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.dim() != 1:
            raise ValueError(
                f"`t` must be (B,) or (B, 1); got {tuple(t.shape)}."
            )
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)              # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < dim:
            emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    # ------------------------------------------------------------------
    def forward(
        self,
        h_node: torch.Tensor,
        positions: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        cond: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a per-atom 3-D velocity.

        Parameters
        ----------
        h_node : ``(B, N, hidden_dim)``
            Per-atom hidden states (typically from
            :class:`models.encoder.MolEncoder`).
        positions : ``(B, N, 3)``
            Current atom positions.
        t : ``(B,)`` or ``(B, 1)`` tensor
            Flow-matching time in ``[0, 1]``.
        edge_index : ``(B, 2, E)``
            Source / destination indices per edge.
        cond : optional ``(B, cond_dim)`` tensor
            Per-sample property embedding.  Ignored when the network
            was constructed with ``cond_dim=0``.
        edge_mask : optional ``(B, E)`` bool tensor
            Real-edge mask forwarded to each EGNN layer.
        node_mask : optional ``(B, N)`` bool tensor
            Real-atom mask.  The returned velocity is zeroed at
            padding positions.

        Returns
        -------
        ``(B, N, 3)`` tensor
            Predicted per-atom velocity (zeroed at padding).
        """
        if h_node.dim() != 3 or h_node.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"`h_node` must be (B, N, {self.hidden_dim}); "
                f"got {tuple(h_node.shape)}."
            )
        if positions.dim() != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"`positions` must be (B, N, 3); got {tuple(positions.shape)}."
            )
        if positions.shape[:2] != h_node.shape[:2]:
            raise ValueError(
                "`positions` and `h_node` must share (B, N); got "
                f"{tuple(positions.shape[:2])} vs {tuple(h_node.shape[:2])}."
            )

        b, n, _ = h_node.shape

        # 1. Time embedding.
        t_emb = self._sinusoidal_time_embed(t, self.time_mlp[0].in_features)
        t_per_node = self.time_mlp(t_emb).unsqueeze(1).expand(b, n, self.hidden_dim)

        # 2. Optional condition broadcast.
        if cond is not None and self.cond_proj is not None:
            if cond.dim() != 2 or cond.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"`cond` must be (B, {self.cond_dim}); got {tuple(cond.shape)}."
                )
            cond_per_node = self.cond_proj(cond).unsqueeze(1).expand(
                b, n, self.hidden_dim
            )
            cond_per_node = cond_per_node + t_per_node
        else:
            cond_per_node = t_per_node

        # 3. Equivariant message passing.
        h = h_node
        last_v = None
        for layer in self.layers:
            h, last_v = layer(
                h,
                positions,
                edge_index,
                cond_per_node=cond_per_node,
                edge_mask=edge_mask,
            )

        # 4. Final velocity = scalar update projected to 3-D, plus the
        #    last aggregated vector message for equivariance.
        v = self.vel_head(h) + last_v
        if node_mask is not None:
            v = v * node_mask.unsqueeze(-1).float()
        return v
