"""Molecular graph encoder for MolFlow-Triton.

The encoder takes a batch of molecular graphs and produces:

- per-node hidden representations ``h_node`` of shape ``(B, N, hidden)``
- per-edge hidden representations ``h_edge`` of shape ``(B, E, hidden)``

The per-node features are obtained by:

1. Embedding the atomic numbers into ``hidden`` dimensions.
2. Running ``K`` rounds of message passing: each round computes a
   per-edge scalar feature from the concatenated source/destination
   embeddings, aggregates neighbour contributions onto the
   destination atom via the shared scatter-sum primitive
   :func:`models._scatter.scatter_sum`, and updates the node
   embedding with an MLP + residual connection.

Per-edge features are produced in parallel: a 2-layer MLP over
``[h_src || h_dst || dist]`` where ``dist`` is the Euclidean distance
between the two atoms.

The encoder is invariant to atom-index ordering (aggregation only
depends on edge topology) but, in this minimal form, does **not** yet
exploit 3-D equivariance — that is the responsibility of
:class:`models.velocity_net.VelocityNet`, which consumes the
``h_node`` features here to predict an equivariant vector field.
"""

from __future__ import annotations

import torch
from torch import nn

from models._scatter import scatter_sum
from triton_kernels import fused_layer_norm
from triton_kernels.config import triton_config


class _FusedLayerNormFn(torch.autograd.Function):
    """autograd bridge for :func:`triton_kernels.fused_layer_norm`.

    The kernel itself is forward-only with custom backward hand-written
    in PyTorch (mean / var / scale-shift chain rule), keeping the
    saved-tensor list minimal: we only need ``x`` and the
    normalization statistics derived from it.

    On CPU, or when the caller disables Triton via ``triton_config``,
    we delegate to :func:`torch.nn.functional.layer_norm` — the pure
    PyTorch reference — so the same module is safe to ship without a
    GPU.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.eps = float(eps)
        # Always save x for the backward path.  We avoid saving the
        # output because the chain rule for LayerNorm only needs x,
        # weight, and the in-block reduction statistics (which we
        # recompute cheaply in backward).
        ctx.save_for_backward(x, weight, bias)

        # CPU or shape gate: fall back to F.layer_norm so the module is
        # usable without a GPU (eval-mode, unit tests, etc.).
        if not triton_config.should_use_fused(x, op="layer_norm"):
            return torch.nn.functional.layer_norm(x, weight.shape, weight, bias, eps)

        out = fused_layer_norm(x, weight, bias, eps=eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        # Standard LayerNorm backward (matches F.layer_norm exactly).
        # Implemented in PyTorch because the kernel above is purely a
        # forward accelerator — the backward is cheap (one mean / one
        # variance pass over the last axis).
        x_f = x.to(torch.float32)
        mean = x_f.mean(dim=-1, keepdim=True)
        var = x_f.var(dim=-1, keepdim=True, unbiased=False)
        rstd = (var + eps).rsqrt()
        x_hat = (x_f - mean) * rstd

        grad_out_f = grad_out.to(torch.float32)
        dx_hat = grad_out_f * weight.to(torch.float32)
        dx_f = (1.0 / x.shape[-1]) * rstd * (
            dx_hat * x.shape[-1]
            - dx_hat.sum(dim=-1, keepdim=True)
            - x_hat * (dx_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        dx = dx_f.to(x.dtype)

        dw = (grad_out_f * x_hat).sum(dim=tuple(range(x.dim() - 1)))
        db = grad_out_f.sum(dim=tuple(range(x.dim() - 1)))
        return dx, dw.to(weight.dtype), db.to(bias.dtype), None


class _MaybeFusedLayerNorm(nn.Module):
    """Drop-in replacement for ``nn.LayerNorm(hidden_dim)``.

    Dispatches to :func:`triton_kernels.fused_layer_norm` when
    :class:`TritonConfig` says so and the last dim fits the 4096-wide
    in-block budget; otherwise falls back to ``nn.LayerNorm``'s
    forward (i.e. :func:`F.layer_norm`).  Parameter names (``weight``,
    ``bias``) and shapes match ``nn.LayerNorm`` exactly so existing
    checkpoints load without remapping.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback path covers CPU, eval mode (TRITON_USE_FUSED_EVAL=0),
        # very wide last axes, and tiny tensors.
        if not triton_config.should_use_fused(x, op="layer_norm"):
            return torch.nn.functional.layer_norm(
                x, (self.normalized_shape,), self.weight, self.bias, self.eps
            )
        return _FusedLayerNormFn.apply(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class MolEncoder(nn.Module):
    """Simple MLP-based molecular graph encoder.

    Parameters
    ----------
    max_atomic_number : int
        Size of the atomic-number embedding table.  Anything larger than
        ``max_atomic_number`` will be clipped before embedding.
    hidden_dim : int
        Width of every hidden representation produced by this module.
    n_layers : int
        Number of message-passing rounds applied to the node features.
    n_atom_types : int, optional
        Alias for ``max_atomic_number``; if both are given the explicit
        ``max_atomic_number`` wins.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_atom_types: int | None = None,
    ) -> None:
        super().__init__()
        if n_atom_types is not None:
            max_atomic_number = n_atom_types
        if max_atomic_number < 1:
            raise ValueError("`max_atomic_number` must be >= 1")
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be >= 1")
        if n_layers < 1:
            raise ValueError("`n_layers` must be >= 1")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Atomic-number embedding: integer -> hidden_dim.
        self.atom_embed = nn.Embedding(max_atomic_number, hidden_dim)

        # Edge MLP: [h_src || h_dst || dist] -> hidden_dim.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-message-passing-layer MLPs that consume the aggregated
        # neighbour vector (hidden_dim) together with the current node
        # embedding to produce the next node embedding.
        self.node_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        self._out_norm = _MaybeFusedLayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_features(
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[h_src || h_dst]`` for every edge.

        Parameters
        ----------
        node_features : ``(B, N, H)``
        edge_index : ``(B, 2, E)``

        Returns
        -------
        ``(B, E, 2*H)``
        """
        if node_features.dim() != 3:
            raise ValueError(
                f"`node_features` must be (B, N, H); got {tuple(node_features.shape)}"
            )
        if edge_index.dim() != 3 or edge_index.shape[1] != 2:
            raise ValueError(
                f"`edge_index` must be (B, 2, E); got {tuple(edge_index.shape)}"
            )
        if node_features.shape[0] != edge_index.shape[0]:
            raise ValueError(
                "Batch size mismatch between `node_features` "
                f"({node_features.shape[0]}) and `edge_index` "
                f"({edge_index.shape[0]})."
            )

        b, n, h = node_features.shape
        e = edge_index.shape[2]

        src = edge_index[:, 0]  # (B, E)
        dst = edge_index[:, 1]  # (B, E)

        # Gather: (B, E, H)
        src_idx = src.unsqueeze(-1).expand(b, e, h)
        dst_idx = dst.unsqueeze(-1).expand(b, e, h)
        h_src = torch.gather(node_features, 1, src_idx)
        h_dst = torch.gather(node_features, 1, dst_idx)
        return torch.cat([h_src, h_dst], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        atomic_numbers : ``(B, N)`` int64 tensor
            Atomic numbers (or any small-integer atom-type id).  Values
            ``>= self.atom_embed.num_embeddings`` are clamped; padding
            atoms should carry ``0`` here.
        positions : ``(B, N, 3)`` float tensor
            Atom positions (caller is responsible for centering +
            scaling).  Used only for the per-edge distance feature.
        edge_index : ``(B, 2, E)`` int64 tensor
            ``edge_index[:, 0]`` are source atom indices,
            ``edge_index[:, 1]`` are destination atom indices.
        edge_mask : optional ``(B, E)`` bool tensor
            Real-edge mask.  Padding edges are skipped in the scatter
            aggregation.
        node_mask : optional ``(B, N)`` bool tensor
            Real-atom mask.  The returned ``h_node`` is zeroed at
            padding positions so downstream consumers can simply
            multiply by ``node_mask`` (or ignore it if the loss does
            the masking).

        Returns
        -------
        h_node : ``(B, N, hidden_dim)``
            Per-atom hidden representations (zeroed at padding).
        h_edge : ``(B, E, hidden_dim)``
            Per-edge hidden representations (zeroed at padding edges).
        """
        if atomic_numbers.dim() != 2:
            raise ValueError(
                f"`atomic_numbers` must be (B, N); got {tuple(atomic_numbers.shape)}"
            )
        if positions.dim() != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"`positions` must be (B, N, 3); got {tuple(positions.shape)}"
            )
        if atomic_numbers.shape != positions.shape[:2]:
            raise ValueError(
                "`atomic_numbers` and `positions` must share the same "
                f"(B, N); got {tuple(atomic_numbers.shape)} vs {tuple(positions.shape[:2])}."
            )

        b, n = atomic_numbers.shape
        e = edge_index.shape[2]

        # 1. Embed atomic numbers.
        atom_clamped = atomic_numbers.clamp(min=0, max=self.atom_embed.num_embeddings - 1)
        h_node = self.atom_embed(atom_clamped)  # (B, N, H)

        # 2. Per-edge scalar distance feature.
        src = edge_index[:, 0]  # (B, E)
        dst = edge_index[:, 1]  # (B, E)
        src_pos = torch.gather(positions, 1, src.unsqueeze(-1).expand(b, e, 3))
        dst_pos = torch.gather(positions, 1, dst.unsqueeze(-1).expand(b, e, 3))
        dist = (src_pos - dst_pos).norm(dim=-1, keepdim=True)  # (B, E, 1)

        # 3. Message passing.
        for layer_idx in range(self.n_layers):
            pair = self._pair_features(h_node, edge_index)        # (B, E, 2H)
            msg_input = torch.cat([pair, dist], dim=-1)           # (B, E, 2H+1)
            msg = self.edge_mlp(msg_input)                        # (B, E, H) — scalar msgs

            # Flatten batch and edge dims so the Triton scatter-sum can
            # aggregate over all edges at once.
            msg_flat = msg.reshape(b * e, self.hidden_dim)
            edge_mask_flat = (
                edge_mask.reshape(b * e) if edge_mask is not None else None
            )

            agg = scatter_sum(
                src=src.reshape(b * e),
                dst=dst.reshape(b * e),
                features=msg_flat,
                n_atoms=n,
                mask=edge_mask_flat,
                batch_size=b,
            )
            agg = agg.reshape(b, n, self.hidden_dim)

            # Update node embedding with residual + MLP.
            update = self.node_mlps[layer_idx](torch.cat([h_node, agg], dim=-1))
            h_node = h_node + update

        h_node = self._out_norm(h_node)

        # Final per-edge features (recomputed using the updated node
        # embeddings so they carry the message-passed information).
        final_pair = self._pair_features(h_node, edge_index)
        h_edge = self.edge_mlp(torch.cat([final_pair, dist], dim=-1))

        # Mask out padding edges so callers can ignore them safely.
        if edge_mask is not None:
            h_edge = h_edge * edge_mask.unsqueeze(-1).float()
        if node_mask is not None:
            h_node = h_node * node_mask.unsqueeze(-1).float()

        return h_node, h_edge
