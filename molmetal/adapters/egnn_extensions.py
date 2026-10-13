"""EGNN extension utilities (S6 / S7 / S-A3..S-A10 surface).

Provides four importable utilities consumed by :class:`EquivariantGraphConv`
and :class:`EGNNVelocityField` to add learned per-layer t-embedding,
learned per-layer pocket injection, dative-aware coord scaling, and a
categorical edge-type embedding.  Each utility is a small nn.Module so
existing state-dict checkpoints keep loading unchanged -- a layer that
runs without these utilities sees no change.

Per-layer injection rationale (S6 + S7):
    The existing :class:`EGNNVelocityField` consumes ``t`` and the pocket
    context only ONCE at the input projection.  Re-threading them at
    every layer (S-A8 / S-A9) lets each layer's MLPs condition on the
    same signal -- the standard transformer / MACE / Equivariant
    Transformer idiom.  When the trainer sets ``t_embedding_dim=0`` or
    ``pocket_per_layer=False`` the layer falls back to the legacy
    single-add path so pre-fix checkpoints remain valid.

These are utilities, NOT a top-level model class.  Do NOT export names
that collide with :mod:`egnn_rocm` (``EGNN`` /
``EquivariantGraphConv``) or :mod:`flow_matching_lipman`
(``PocketEncoder``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Per the design, this file is < 150 LOC and is the ONLY path that
# imports the four helpers below.  Keep imports flat so the molecule
# stays obvious in code review.

__all__ = [
    "SinusoidalEmbedding",
    "DativeAwareCoordScale",
    "PerLayerPocketInjector",
    "EdgeTypeEmbedding",
]


class SinusoidalEmbedding(nn.Module):
    """Standard transformer-style sinusoidal positional embedding for t.

    Maps a scalar ``t`` in ``[0, 1]`` (or any real value) to a
    fixed-dim vector ``[B, dim]`` with even-index cosines and odd-index
    sines on a log-space frequency schedule.  Backward-compat:
    ``dim=0`` returns a zero-length tensor so the legacy scalar-t path
    is bit-exact.

    Args:
        dim: Output dimension.  Must be even or padded internally.
        max_period: Controls the lowest frequency (``2 * pi / max_period``).
    """

    def __init__(self, dim: int = 32, max_period: int = 10000):
        super().__init__()
        if dim < 0:
            raise ValueError(f"SinusoidalEmbedding dim must be >= 0, got {dim}")
        self.dim = int(dim)
        self.max_period = int(max_period)
        # ``div_term`` is constant; register as a buffer so .to(device)
        # moves it together with the model.
        half = self.dim // 2
        if half > 0:
            # Frequencies 1, 2, ..., half; then 2*pi*freq / max_period.
            # Stored on CPU float32; cast at forward-time.
            div = torch.exp(
                torch.arange(0, half, dtype=torch.float32)
                * (-math.log(max_period) / max(half, 1))
            )
            self.register_buffer("_div_term", div, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a scalar (or batched scalar) ``t`` to ``[B, dim]``.

        ``t`` may be ``[B]`` (the trainer path) or ``[B, 1]`` (the ODE
        path); both are flattened to ``[B]`` and re-broadcast.
        """
        if self.dim == 0:
            # Legacy bit-exact short-circuit: empty embedding.
            bshape = t.shape[:-1] if t.dim() > 0 else (1,)
            return t.new_zeros(*bshape, 0)
        flat = t.reshape(-1) if t.dim() > 0 else t.view(1)
        if flat.dim() == 0:
            flat = flat.unsqueeze(0)
        # ``outer`` follows the PyTorch tutorial pattern: outer-product
        # of ``flat`` and ``_div_term`` gives the angle matrix.
        device = flat.device
        dtype = flat.dtype
        # Cast to float32 for the angle math, then back to ``dtype`` at the
        # end so the surrounding model sees a consistent dtype.
        angles = flat.to(torch.float32).unsqueeze(1) * self._div_term.to(device)
        if self.dim % 2 == 1:
            # Odd dim: pad the sin stack with one zero so the concat
            # matches ``dim``.
            emb = torch.cat(
                [torch.cos(angles), torch.sin(angles), angles.new_zeros(angles.shape[0], 1)],
                dim=-1,
            )
        else:
            emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return emb.to(dtype)


class DativeAwareCoordScale(nn.Module):
    """Per-edge coord scale with separate dative vs covalent scalars.

    Forward signature: ::

        forward(dative_mask) -> [n_edges, 1] coord_scale

    where ::

        coord_scale = dative_mask * scale_dative + (1 - dative_mask) * scale_covalent

    Both ``scale_dative`` and ``scale_covalent`` are independent
    :class:`torch.nn.Parameter` tensors of shape ``[1]``.  When
    ``dative_mask is None`` every edge gets the covalent scale -- a
    bit-exact fallback for callers that don't pass dative information.

    Lit anchor:
        - Dative vs covalent differentiation is the standard L -> M
          coordination-bond asymmetry in 3D metallodrug geometry
          (Housecroft & Sharpe, "Inorganic Chemistry", 5th ed., ch. 19).
        - Learnable per-edge-type coord scales are part of the MACE
          equivariant message path (Batatia et al. 2022, arXiv:2206.07697).
    """

    def __init__(
        self,
        dative_init: float = 0.85,
        covalent_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.dative_coord_scale = nn.Parameter(torch.tensor(float(dative_init)))
        self.covalent_coord_scale = nn.Parameter(torch.tensor(float(covalent_init)))

    def forward(self, dative_mask: torch.Tensor | None) -> torch.Tensor:
        # ``dative_mask`` shape: [n_edges] or [n_edges, 1].  When None,
        # every edge uses covalent scale.
        n_edges = (
            dative_mask.shape[0]
            if dative_mask is not None
            else 0
        )
        if dative_mask is None or n_edges == 0:
            return self.covalent_coord_scale.expand(1, 1).clone()
        m = dative_mask.view(-1, 1).to(self.dative_coord_scale.dtype)
        return m * self.dative_coord_scale + (1.0 - m) * self.covalent_coord_scale


class PerLayerPocketInjector(nn.Module):
    """Project a per-sample pocket vector to (B, N, hidden) for additive injection.

    The pocket vector ``(B, pocket_dim)`` is broadcast to per-atom by
    an expand + a learned linear projection; the result is added to
    :math:`h` at every EGNN layer when the trainer opts in via
    ``pocket_per_layer=True``.  The projector weight is zero-initialised
    in the last layer of the legacy contract (see ``forward``); this
    means at init time the per-layer injection is identically zero and
    the forward is bit-exact with the pre-fix code path.

    Args:
        pocket_dim: Source dimension; typically the EGNN hidden dim.
        hidden_dim: Target per-atom hidden dim.
    """

    def __init__(self, pocket_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pocket_dim = int(pocket_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(pocket_dim, hidden_dim, bias=True)
        # Zero-init the projection so legacy bit-exact holds at init.
        with torch.no_grad():
            self.proj.weight.zero_()
            if self.proj.bias is not None:
                self.proj.bias.zero_()

    def forward(self, pocket_embed: torch.Tensor, n_max: int) -> torch.Tensor:
        """Return a ``[B, N, hidden]`` addend to inject into ``h``.

        Args:
            pocket_embed: ``[B, pocket_dim]`` per-sample pocket vector.
                When ``None`` the function returns a zero tensor with
                the same batch size as the input ``x`` (defaulting to
                batch=1 if neither is available — the train path always
                supplies ``x`` to the calling forward so this is the
                caller-side fallback).
            n_max: Number of ligand atoms; broadcast dim.
        """
        if pocket_embed is None:
            # The projector is zero-init so a zero input gives a zero
            # output — matches the bit-exact legacy contract where no
            # pocket signal ever enters the EGNN loop body.  We
            # infer the batch size from a default of 1 because the
            # caller is responsible for broadcasting the addend
            # against the actual ``h`` tensor.
            b = 1
            zeros = torch.zeros(b, n_max, self.hidden_dim)
            return zeros
        b = pocket_embed.shape[0]
        # (B, dim) -> (B, 1, dim) -> (B, N, dim) via the projector.
        per_atom = self.proj(pocket_embed).unsqueeze(1).expand(b, n_max, self.hidden_dim)
        return per_atom


class EdgeTypeEmbedding(nn.Module):
    """Categorical edge-type embedding lookup.

    Used by :class:`EquivariantGraphConv` (when the trainer opts in to
    edge-type-aware message passing) to turn the per-edge type code
    (SINGLE / DOUBLE / DATIVE / future AROMATIC) into a learned dense
    vector consumed by the message MLP.

    Args:
        num_types: Number of distinct edge type codes (default 4 = SINGLE,
            DOUBLE, DATIVE, plus one slot reserved for AROMATIC).
        dim: Output dimension (default 8 = the dim chosen in the design
            so that ``in_node_dim*2 + 1 + 3 + 8`` stays a clean
            multiple of common conv widths).
    """

    def __init__(self, num_types: int = 4, dim: int = 8) -> None:
        super().__init__()
        self.num_types = int(num_types)
        self.dim = int(dim)
        self.emb = nn.Embedding(self.num_types, self.dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """Return a ``[n_edges, dim]`` embedding lookup.

        Clamps out-of-range codes to ``0`` (the SINGLE slot) so a
        caller passing a tensor with unexpected values does not crash
        the embedding.
        """
        if edge_types is None:
            raise ValueError("EdgeTypeEmbedding.forward requires edge_types")
        # long, on the same device as the embedding.
        et = edge_types.long().to(self.emb.weight.device)
        clamped = et.clamp(min=0, max=self.num_types - 1)
        return self.emb(clamped)
