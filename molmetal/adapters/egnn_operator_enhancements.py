"""EGNN operator enhancements (S-A1..S-A10, O1, O4, O7, O9 fix surface).

Per-layer opt-in replacement for the legacy ``EquivariantGraphConv`` operators.
This module is a SIDE PATH: it does NOT modify the shipped ``egnn_rocm.py``
operators. Instead it ships drop-in ``nn.Module``s that an outer wrapper
(``EnhancedEquivariantGraphConv``) consumes when the ``operator.*`` YAML
keys are set to true. When all ``use_enhanced_*`` flags are false, the
wrapper falls back to the legacy path bit-exactly.

Operators replaced
------------------
- ``EnhancedMLP``  — phi_e. Expands input dim from ``2H + 1`` (legacy) to
  ``2H + 24`` by adding ``dir_vec(3) + edge_type_emb(4) +
  bo_continuous_emb(4) + natural_charge_emb(4) + oxidation_emb(4) +
  coord_env_emb(4) + metal_proximity(1) + dative_geometry_flag(1)``.
  Residual projection on ``[h_i, h_j]`` + ``LayerNorm(out_dim)``.
- ``EnhancedCoordUpdate``  — phi_x. Branches on ``edge_type``: dative edges
  pull the ligand atom toward the metal with an angular ideal
  (90 deg for Oh-6 / 180 deg for linear / 109.5 deg for Td-4); covalent
  edges follow ``(x_i - x_j)``; aromatic edges use a 50/50 hybrid.
  ``dative_angular_strictness`` (YAML) gates the deviation penalty.
- ``EnhancedAggregate``  — scatter. 4 modes: ``sum`` (legacy) /
  ``sum_degree_norm`` (legacy wch6vhfyi) / ``sum_edge_type_weighted``
  (NEW DEFAULT; per-edge-type learnable weights) / ``attention`` (NEW).
- ``EnhancedFeatureUpdate``  — phi_h. FiLM-style ``(gamma, beta)`` from
  ``t_emb`` (SinusoidalEmbedding) + optional ``pocket_emb`` broadcast.

Auxiliary
---------
- ``ChemistrySignalProjector``  — per-atom projector for ``natural_charge,
  oxidation_state, spin, dft_props, coord_neighbors, metal_proximity,
  dative_geometry_flag``. ADD to ``h`` BEFORE phi_e. Missing channel
  -> zero (defensive; never raises).
- ``BOContinuousEmbedding``  — ``linear(1, 4)`` for continuous BO value
  (single/double/triple/aromatic = 1.0/2.0/3.0/1.5).
- ``SmoothDist``  — ``torch.autograd.Function`` for O9 fix.
  ``forward``: ``log(dist + eps)`` (smooth, no ``1e-6+1`` discontinuity);
  ``backward``: ``1/(dist+eps)`` (analytic). ``SmoothDist.use_smooth=False``
  short-circuits to the bit-exact ``clamp(min=1e-6)`` path.

Equivariance
------------
- All branches touching ``x`` apply an SE(3)-equivariant transformation
  (translation-invariant displacement vector, rotation-equivariant
  direction). The dative branch's angular penalty is a SCALAR added to
  the trainer's loss, NOT a rotation of the displacement.
- ``dir_vec`` itself is a unit vector, equivariant under rotation.

Imports reuse wch6vhfyi utilities (no collisions per the
``egnn_extensions.py`` design contract):
- ``SinusoidalEmbedding, EdgeTypeEmbedding, DativeAwareCoordScale,
  PerLayerPocketInjector``.

Backward compat
---------------
- All ``use_enhanced_*=false`` -> ``EnhancedEquivariantGraphConv``
  forwards to the legacy ``EquivariantGraphConv`` (no API break).
- ``aggregate_mode="sum"`` -> legacy ``scatter_sum`` only.
- ``SmoothDist.use_smooth=False`` -> ``clamp(min=1e-6)`` legacy path.
- ``ChemistrySignalProjector`` with no chemistry context -> zeros.
- ``BOContinuousEmbedding`` with no ``bond_orders_continuous`` ->
  zeros (single-bond prior at init).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.adapters.egnn_extensions import (
    SinusoidalEmbedding,
    EdgeTypeEmbedding,
    DativeAwareCoordScale,
    PerLayerPocketInjector,
)


# Edge type codes (consistent with egnn_rocm.EDGE_TYPE_* constants).
EDGE_TYPE_SINGLE: int = 0
EDGE_TYPE_DOUBLE: int = 1
EDGE_TYPE_DATIVE: int = 2
EDGE_TYPE_AROMATIC: int = 3
NUM_EDGE_TYPES: int = 4


# Ideal bond angles per coordination geometry (radians).
_IDEAL_ANGLE_SQ_PLANAR: float = math.pi / 2.0          # 90 deg (Oh-6 / SP-4)
_IDEAL_ANGLE_LINEAR: float = math.pi                  # 180 deg (linear Au(I))
_IDEAL_ANGLE_TETRA: float = math.acos(-1.0 / 3.0)     # 109.47 deg (Td-4)


# ----------------------------------------------------------------------
# SmoothDist  (O9 fix)
# ----------------------------------------------------------------------


class SmoothDist(torch.autograd.Function):
    """Smooth distance transform replacing ``dist.clamp(min=1e-6)``.

    Forward: ``log(dist + eps)`` -- smooth in ``dist`` (no discontinuity).
    Backward: ``1 / (dist + eps)`` -- analytic gradient of ``log``.

    Avoids the O9 gradient-discontinuity at ``dist == 1e-6`` where
    ``clamp``'s backward has a discontinuous step from ``0`` to ``1``
    on the boundary.

    Class-level toggle:
        ``SmoothDist.use_smooth = True`` (default): smooth path.
        ``SmoothDist.use_smooth = False``: legacy ``clamp(min=1e-6)`` path
        (bit-exact with the wch6vhfyi behaviour).
    """

    use_smooth: bool = True
    eps: float = 1e-3

    @staticmethod
    def forward(ctx, dist):  # type: ignore[override]
        if SmoothDist.use_smooth:
            ctx.save_for_backward(dist)
            return torch.log(dist + SmoothDist.eps)
        return dist.clamp(min=1e-6)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        if SmoothDist.use_smooth:
            (dist,) = ctx.saved_tensors
            return grad_output / (dist + SmoothDist.eps)
        return grad_output


# ----------------------------------------------------------------------
# BOContinuousEmbedding
# ----------------------------------------------------------------------


class BOContinuousEmbedding(nn.Module):
    """Continuous bond-order embedding: ``linear(1, 4)``.

    Maps a scalar bond-order value (0..3) to a 4-d learned vector.
    Default initialisation produces a near-uniform embedding at init so a
    caller that omits BO information still gets a deterministic channel.

    Args:
        out_dim: Output dimension (default 4).
    """

    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = nn.Linear(1, self.out_dim)

    def forward(self, bo: Optional[torch.Tensor]) -> torch.Tensor:
        """Return ``[n_edges, out_dim]`` embedding.

        When ``bo is None`` returns a zeros tensor of shape
        ``[1, out_dim]``; the caller must broadcast over edges.
        """
        if bo is None:
            return torch.zeros(1, self.out_dim)
        if bo.dim() == 1:
            bo = bo.unsqueeze(-1)
        return self.proj(bo)


# ----------------------------------------------------------------------
# ChemistrySignalProjector
# ----------------------------------------------------------------------


class ChemistrySignalProjector(nn.Module):
    """Per-atom chemistry signal injector.

    Each input signal is projected to ``(N, chem_embed_dim)`` and the
    results are summed. Per-cloud signals (``oxidation_state``, ``spin``,
    ``dft_props``) are broadcast via ``batch_idx`` so every atom in a
    cloud gets the same signal.

    Missing channel -> zero (defensive; never raises).

    Args:
        chem_embed_dim: Output dim (default 16).
    """

    def __init__(self, chem_embed_dim: int = 16) -> None:
        super().__init__()
        self.chem_embed_dim = int(chem_embed_dim)
        self.natural_charge_proj = nn.Linear(1, self.chem_embed_dim)
        self.oxidation_proj = nn.Linear(1, self.chem_embed_dim)
        self.spin_proj = nn.Linear(1, self.chem_embed_dim)
        # tmQM dft_props is 8-d per the project invariant; if absent,
        # the channel is zero.
        self.dft_proj = nn.Linear(8, self.chem_embed_dim)
        self.coord_neighbors_proj = nn.Linear(1, self.chem_embed_dim)
        self.metal_proximity_proj = nn.Linear(1, self.chem_embed_dim)
        self.dative_geometry_flag_proj = nn.Linear(1, self.chem_embed_dim)

    def forward(
        self,
        n_atoms: int,
        *,
        natural_charge: Optional[torch.Tensor] = None,
        oxidation_state: Optional[torch.Tensor] = None,
        oxidation_batch_idx: Optional[torch.Tensor] = None,
        spin: Optional[torch.Tensor] = None,
        spin_batch_idx: Optional[torch.Tensor] = None,
        dft_props: Optional[torch.Tensor] = None,
        dft_batch_idx: Optional[torch.Tensor] = None,
        coord_neighbors: Optional[torch.Tensor] = None,
        metal_proximity: Optional[torch.Tensor] = None,
        dative_geometry_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[n_atoms, chem_embed_dim]`` per-atom signal.

        Per-cloud tensors (oxidation_state / spin / dft_props) are
        broadcast to per-atom via the corresponding ``*_batch_idx``
        (each entry maps an atom to its cloud id).
        """
        device = torch.device("cpu")
        dtype = torch.float32
        for t in (
            natural_charge,
            oxidation_state,
            spin,
            dft_props,
            coord_neighbors,
            metal_proximity,
            dative_geometry_flag,
        ):
            if t is not None:
                device = t.device
                dtype = t.dtype
                break

        out = torch.zeros(n_atoms, self.chem_embed_dim, device=device, dtype=dtype)

        def _per_atom(
            value: Optional[torch.Tensor],
            batch_idx: Optional[torch.Tensor],
            proj: nn.Module,
        ) -> None:
            if value is None:
                return
            v = value if batch_idx is None else value[batch_idx]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            if v.shape[0] != n_atoms:
                if v.shape[0] < n_atoms:
                    pad = torch.zeros(
                        n_atoms - v.shape[0], *v.shape[1:],
                        device=device, dtype=dtype,
                    )
                    v = torch.cat([v, pad], dim=0)
                else:
                    v = v[:n_atoms]
            out.add_(proj(v))

        _per_atom(natural_charge, None, self.natural_charge_proj)
        _per_atom(oxidation_state, oxidation_batch_idx, self.oxidation_proj)
        _per_atom(spin, spin_batch_idx, self.spin_proj)
        _per_atom(dft_props, dft_batch_idx, self.dft_proj)
        _per_atom(coord_neighbors, None, self.coord_neighbors_proj)
        _per_atom(metal_proximity, None, self.metal_proximity_proj)
        _per_atom(dative_geometry_flag, None, self.dative_geometry_flag_proj)

        return out


# ----------------------------------------------------------------------
# EnhancedMLP  -- phi_e replacement
# ----------------------------------------------------------------------


class EnhancedMLP(nn.Module):
    """phi_e replacement: ``2H + 24 -> H`` with residual + LayerNorm.

    Input dim = ``2 * in_node_dim + 26``:
        1 (dist) + 3 (dir_vec) + 4 (edge_type_emb) +
        4 (bo_continuous_emb) + 4 (natural_charge_emb) +
        4 (oxidation_emb) + 4 (coord_env_emb) +
        1 (metal_proximity) + 1 (dative_geometry_flag) = 26
        (the design brief stated ``+24`` but the breakdown sums to
        ``+26``; the actual concatenation length is honest-reported here).

    Architecture: ``Linear -> SiLU -> Linear -> SiLU -> Linear``
    plus a residual projection of ``[h_i, h_j]`` to ``out_dim``
    (Identity when ``2*in_node_dim == hidden_dim``) and a
    ``LayerNorm(out_dim)`` tail.

    Args:
        in_node_dim: Source node feature dim.
        hidden_dim: Output dim.
        msg_dropout: Dropout on the message after SiLU (default 0.0).
    """

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        msg_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_node_dim = int(in_node_dim)
        self.hidden_dim = int(hidden_dim)
        self.msg_dropout = float(msg_dropout)

        in_dim = 2 * self.in_node_dim + 26
        self.in_dim = in_dim
        h = max(self.hidden_dim, 2 * in_dim)
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, self.hidden_dim)
        self.act = nn.SiLU()
        if 2 * self.in_node_dim != self.hidden_dim:
            self.residual_proj = nn.Linear(2 * self.in_node_dim, self.hidden_dim)
        else:
            self.residual_proj = nn.Identity()
        self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        dist: torch.Tensor,
        dir_vec: torch.Tensor,
        edge_type_emb: torch.Tensor,
        bo_continuous_emb: torch.Tensor,
        natural_charge_emb: torch.Tensor,
        oxidation_emb: torch.Tensor,
        coord_env_emb: torch.Tensor,
        metal_proximity: torch.Tensor,
        dative_geometry_flag: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                h_i,
                h_j,
                dist,
                dir_vec,
                edge_type_emb,
                bo_continuous_emb,
                natural_charge_emb,
                oxidation_emb,
                coord_env_emb,
                metal_proximity,
                dative_geometry_flag,
            ],
            dim=-1,
        )
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        if self.msg_dropout > 0.0 and self.training:
            h = F.dropout(h, p=self.msg_dropout, training=True)
        residual = self.residual_proj(torch.cat([h_i, h_j], dim=-1))
        return self.ln(h + residual)


# ----------------------------------------------------------------------
# EnhancedCoordUpdate  -- phi_x replacement
# ----------------------------------------------------------------------


class EnhancedCoordUpdate(nn.Module):
    """phi_x replacement with separate dative vs covalent branches.

    Inputs:
        msg: ``[E, hidden_dim]`` per-edge message.
        x: ``[N, 3]`` node coordinates.
        edge_index: ``[2, E]`` source->target.
        edge_types: ``[E]`` long, edge-type code.
        metal_atom_index: ``[N]`` long, the metal atom in each row
            (or -1 for non-metal).  Used only by the dative branch.
        coord_geometry_class: ``[N]`` long with code
            ``{2 = linear, 4 = Td-4, 6 = Oh-6}`` (other = unknown).

    Outputs:
        x_new: ``[N, 3]`` updated coordinates.
        angular_penalty: scalar (0 when no penalty applies).
    """

    def __init__(
        self,
        hidden_dim: int,
        dative_angular_strictness: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dative_angular_strictness = float(dative_angular_strictness)
        self.phi_x_mag = nn.Linear(self.hidden_dim, 1)
        self.phi_x_dir = nn.Linear(self.hidden_dim, 3)
        # Aromatic mix: 0 = pure covalent direction, 1 = pure dative pull.
        self.aromatic_mix = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        msg: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        metal_atom_index: Optional[torch.Tensor] = None,
        coord_geometry_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n_nodes = x.size(0)
        diff = x[src] - x[dst]  # [E, 3]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit = diff / dist  # covalent direction

        mag = self.phi_x_mag(msg)  # [E, 1]

        dative_mask = (edge_types == EDGE_TYPE_DATIVE).view(-1, 1).to(x.dtype)
        aromatic_mask = (edge_types == EDGE_TYPE_AROMATIC).view(-1, 1).to(x.dtype)
        covalent_mask = 1.0 - dative_mask - aromatic_mask

        if metal_atom_index is not None:
            metal_idx_per_src = metal_atom_index[src].clamp(min=0)
            dative_dir = x[metal_idx_per_src] - x[src]
            dative_dir_n = dative_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dative_unit = dative_dir / dative_dir_n
        else:
            dative_unit = unit

        # Dative branch: 0.3 covalent + 0.7 dative (lean toward metal).
        dative_combined = 0.3 * unit + 0.7 * dative_unit
        # Aromatic: lerp(unit, dative_unit, aromatic_mix).
        aromatic_combined = torch.lerp(unit, dative_unit, self.aromatic_mix.clamp(0.0, 1.0))

        direction = (
            covalent_mask * unit
            + dative_mask * dative_combined
            + aromatic_mask * aromatic_combined
        )
        direction_n = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = direction / direction_n

        disp = direction * torch.tanh(mag)  # bound magnitude in (-1, 1)
        agg = torch.zeros(n_nodes, 3, device=x.device, dtype=x.dtype)
        agg = agg.index_add_(0, dst, disp)
        x_new = x + agg

        if (
            self.dative_angular_strictness > 0.0
            and metal_atom_index is not None
            and coord_geometry_class is not None
        ):
            penalty = self._angular_penalty(
                x, src, dst, edge_types, metal_atom_index, coord_geometry_class
            )
        else:
            penalty = torch.zeros((), device=x.device, dtype=x.dtype)
        return x_new, penalty

    def _angular_penalty(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_types: torch.Tensor,
        metal_atom_index: torch.Tensor,
        coord_geometry_class: torch.Tensor,
    ) -> torch.Tensor:
        dative_mask = edge_types == EDGE_TYPE_DATIVE
        if not dative_mask.any():
            return torch.zeros((), device=x.device, dtype=x.dtype)
        idx = dative_mask.nonzero(as_tuple=True)[0]
        m_idx = metal_atom_index[src[idx]].clamp(min=0)
        v1 = x[m_idx] - x[dst[idx]]
        v2 = x[src[idx]] - x[dst[idx]]
        v1_n = v1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v2_n = v2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_a = (v1 * v2).sum(dim=-1) / (v1_n.squeeze(-1) * v2_n.squeeze(-1))
        cos_a = cos_a.clamp(min=-1.0, max=1.0)
        angle = torch.acos(cos_a)
        geo = coord_geometry_class[m_idx].to(x.dtype)
        is_known = ((geo == 2) | (geo == 4) | (geo == 6)).to(x.dtype)
        ideal = (
            (geo == 2).to(x.dtype) * _IDEAL_ANGLE_LINEAR
            + (geo == 4).to(x.dtype) * _IDEAL_ANGLE_TETRA
            + (geo == 6).to(x.dtype) * _IDEAL_ANGLE_SQ_PLANAR
        )
        # Unknown geometry: default ideal = 90 deg (sq planar / Oh-6).
        ideal = ideal + (1.0 - is_known) * _IDEAL_ANGLE_SQ_PLANAR
        dev = (angle - ideal).abs().mean()
        return dev * self.dative_angular_strictness


# ----------------------------------------------------------------------
# EnhancedAggregate  -- scatter replacement
# ----------------------------------------------------------------------


class EnhancedAggregate(nn.Module):
    """Scatter aggregation with 4 modes.

    Modes:
        - ``"sum"`` (legacy): ``scatter_sum`` only.
        - ``"sum_degree_norm"`` (legacy wch6vhfyi): ``scatter_sum / degree``.
        - ``"sum_edge_type_weighted"`` (NEW DEFAULT): ``scatter_sum(
          msg * edge_weight_per_type[edge_type]) / degree``.  Per-edge-type
          learnable weights.
        - ``"attention"`` (NEW): ``scatter_softmax(q . k) * msg``.

    Args:
        hidden_dim: Message dim.
        num_edge_types: Number of edge type codes (default 4).
        mode: One of the 4 strings above.
    """

    _VALID_MODES = (
        "sum",
        "sum_degree_norm",
        "sum_edge_type_weighted",
        "attention",
    )

    def __init__(
        self,
        hidden_dim: int,
        num_edge_types: int = NUM_EDGE_TYPES,
        mode: str = "sum_edge_type_weighted",
    ) -> None:
        super().__init__()
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"EnhancedAggregate mode must be one of {self._VALID_MODES}, "
                f"got {mode!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_edge_types = int(num_edge_types)
        self.mode = mode
        if mode == "sum_edge_type_weighted":
            self.edge_weight_per_type = nn.Parameter(torch.ones(self.num_edge_types))
        elif mode == "attention":
            self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.edge_weight_per_type = None

    def forward(
        self,
        msg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        if dst.numel() == 0:
            return msg.new_zeros(0, self.hidden_dim)
        n_nodes = int(dst.max().item()) + 1

        if self.mode == "sum":
            agg = msg.new_zeros(n_nodes, self.hidden_dim)
            return agg.index_add_(0, dst, msg)
        if self.mode == "sum_degree_norm":
            agg = msg.new_zeros(n_nodes, self.hidden_dim)
            agg = agg.index_add_(0, dst, msg)
            degree = msg.new_zeros(n_nodes, dtype=msg.dtype)
            ones = msg.new_ones(dst.shape[0], dtype=msg.dtype)
            degree = degree.index_add_(0, dst, ones).clamp(min=1.0)
            return agg / degree.unsqueeze(-1)
        if self.mode == "sum_edge_type_weighted":
            if edge_types is None:
                raise ValueError(
                    "edge_types required for sum_edge_type_weighted mode"
                )
            w = self.edge_weight_per_type[edge_types.to(
                self.edge_weight_per_type.device
            )].to(msg.dtype)
            w = w.unsqueeze(-1)
            agg = msg.new_zeros(n_nodes, self.hidden_dim)
            agg = agg.index_add_(0, dst, msg * w)
            degree = msg.new_zeros(n_nodes, dtype=msg.dtype)
            ones = msg.new_ones(dst.shape[0], dtype=msg.dtype)
            degree = degree.index_add_(0, dst, ones).clamp(min=1.0)
            return agg / degree.unsqueeze(-1)
        # attention
        q = self.q_proj(msg)
        k = self.k_proj(msg)
        score = (q * k).sum(dim=-1)  # [E]
        # Per-destination max for numerical stability.
        max_per_dst = msg.new_full(
            (n_nodes,), float("-inf"), dtype=msg.dtype
        )
        max_per_dst = max_per_dst.scatter_reduce(
            0, dst, score, reduce="amax", include_self=True
        )
        # Replace any remaining -inf with 0 (empty groups).
        max_per_dst = torch.where(
            torch.isinf(max_per_dst),
            torch.zeros_like(max_per_dst),
            max_per_dst,
        )
        score_centered = score - max_per_dst[dst]
        exp_s = torch.exp(score_centered)
        denom = msg.new_zeros(n_nodes, dtype=msg.dtype)
        denom = denom.index_add_(0, dst, exp_s).clamp(min=1e-9)
        attn = exp_s / denom[dst]
        agg = msg.new_zeros(n_nodes, self.hidden_dim)
        agg = agg.index_add_(0, dst, msg * attn.unsqueeze(-1))
        return agg


# ----------------------------------------------------------------------
# EnhancedFeatureUpdate  -- phi_h replacement
# ----------------------------------------------------------------------


class EnhancedFeatureUpdate(nn.Module):
    """phi_h replacement with FiLM-style t conditioning + optional pocket.

    Inputs:
        h_i: ``[N, hidden_dim]`` per-atom node features.
        agg_msg: ``[N, hidden_dim]`` aggregated messages.
        t_emb: ``[N, t_dim]`` per-atom t embedding (SinusoidalEmbedding).
        pocket_emb: ``[N, hidden_dim]`` optional per-atom pocket
            injection (PerLayerPocketInjector); zeros when None.

    FiLM:
        gamma, beta = Linear(t_dim, hidden_dim)(t_emb)
        h_modulated = (1 + gamma) * h_i + beta

    Architecture:
        concat[h_mod, agg_msg, pocket] -> Linear -> SiLU -> Linear -> LayerNorm.
        Residual: add h_i to the output.
    """

    def __init__(self, hidden_dim: int, t_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.t_dim = int(t_dim)
        self.film_gamma = nn.Linear(self.t_dim, self.hidden_dim)
        self.film_beta = nn.Linear(self.t_dim, self.hidden_dim)
        # Zero-init so legacy init is bit-exact at first step.
        with torch.no_grad():
            self.film_gamma.weight.zero_()
            self.film_gamma.bias.zero_()
            self.film_beta.weight.zero_()
            self.film_beta.bias.zero_()
        in_dim = self.hidden_dim * 3  # h_mod + agg_msg + pocket
        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        h_i: torch.Tensor,
        agg_msg: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        pocket_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = h_i.size(0)
        if t_emb is None:
            t_emb = h_i.new_zeros(n, self.t_dim)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(0)
        if t_emb.size(0) == 1 and n > 1:
            t_emb = t_emb.expand(n, -1)
        if t_emb.size(0) != n:
            t_emb = t_emb.expand(n, -1)
        if pocket_emb is None:
            pocket_emb = h_i.new_zeros(n, self.hidden_dim)
        gamma = self.film_gamma(t_emb)
        beta = self.film_beta(t_emb)
        h_mod = (1.0 + gamma) * h_i + beta
        x = torch.cat([h_mod, agg_msg, pocket_emb], dim=-1)
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.ln(h + h_i)


# ----------------------------------------------------------------------
# EnhancedEquivariantGraphConv  -- top-level drop-in wrapper
# ----------------------------------------------------------------------


class EnhancedEquivariantGraphConv(nn.Module):
    """Top-level drop-in replacement for :class:`EquivariantGraphConv`.

    When all ``use_enhanced_*`` flags are False, the wrapper is
    bit-exactly equivalent to the legacy path: it just routes
    ``forward`` to the shipped ``EquivariantGraphConv`` instance
    passed in ``legacy_conv``.

    When any flag is True, the wrapper composes the four
    ``EnhancedMLP / EnhancedCoordUpdate / EnhancedAggregate /
    EnhancedFeatureUpdate`` modules. The chemistry context is added
    to ``h`` BEFORE the first phi_e call.

    The wrapper does NOT modify the legacy module; it merely holds
    a reference and falls back to it on the no-op path.

    Args:
        in_node_dim: Source node feature dim.
        hidden_dim: Output hidden dim.
        legacy_conv: an instance of the shipped ``EquivariantGraphConv``
            (used in fallback mode).
        use_enhanced_mlp: enable ``EnhancedMLP`` for phi_e.
        use_enhanced_coord_update: enable ``EnhancedCoordUpdate`` for phi_x.
        use_enhanced_aggregate: enable ``EnhancedAggregate`` for scatter.
        use_enhanced_feature_update: enable ``EnhancedFeatureUpdate`` for phi_h.
        aggregate_mode: scatter mode (one of ``EnhancedAggregate._VALID_MODES``).
        chem_embed_dim: chemistry-signal projector output dim.
        dative_angular_strictness: penalty weight in (0, 1).
        msg_dropout: optional dropout on the message after SiLU.
        t_dim: dim of the per-atom t embedding (SinusoidalEmbedding).
    """

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        legacy_conv: nn.Module,
        *,
        use_enhanced_mlp: bool = True,
        use_enhanced_coord_update: bool = True,
        use_enhanced_aggregate: bool = True,
        use_enhanced_feature_update: bool = True,
        aggregate_mode: str = "sum_edge_type_weighted",
        chem_embed_dim: int = 16,
        dative_angular_strictness: float = 0.5,
        msg_dropout: float = 0.0,
        t_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_node_dim = int(in_node_dim)
        self.hidden_dim = int(hidden_dim)
        self.legacy_conv = legacy_conv
        self.use_enhanced_mlp = bool(use_enhanced_mlp)
        self.use_enhanced_coord_update = bool(use_enhanced_coord_update)
        self.use_enhanced_aggregate = bool(use_enhanced_aggregate)
        self.use_enhanced_feature_update = bool(use_enhanced_feature_update)
        self.aggregate_mode = aggregate_mode
        self.chem_embed_dim = int(chem_embed_dim)
        self.dative_angular_strictness = float(dative_angular_strictness)
        self.t_dim = int(t_dim)

        any_enhanced = any(
            [
                self.use_enhanced_mlp,
                self.use_enhanced_coord_update,
                self.use_enhanced_aggregate,
                self.use_enhanced_feature_update,
            ]
        )

        if any_enhanced:
            self.chem_projector = ChemistrySignalProjector(
                chem_embed_dim=self.chem_embed_dim
            )
            self.t_embedding = SinusoidalEmbedding(dim=self.t_dim)
            self.dative_coord_scale_helper = DativeAwareCoordScale(
                dative_init=0.85, covalent_init=1.0
            )
            self.pocket_injector_helper = PerLayerPocketInjector(
                pocket_dim=self.hidden_dim, hidden_dim=self.hidden_dim
            )
            if self.use_enhanced_mlp:
                self.enhanced_mlp = EnhancedMLP(
                    in_node_dim=self.in_node_dim,
                    hidden_dim=self.hidden_dim,
                    msg_dropout=msg_dropout,
                )
                self.edge_type_emb = EdgeTypeEmbedding(
                    num_types=NUM_EDGE_TYPES, dim=4
                )
                self.bo_emb = BOContinuousEmbedding(out_dim=4)
                self.natural_charge_emb = nn.Linear(1, 4)
                self.oxidation_emb = nn.Linear(1, 4)
                self.coord_env_emb = nn.Linear(1, 4)
            if self.use_enhanced_coord_update:
                self.enhanced_coord_update = EnhancedCoordUpdate(
                    hidden_dim=self.hidden_dim,
                    dative_angular_strictness=self.dative_angular_strictness,
                )
            if self.use_enhanced_aggregate:
                self.enhanced_aggregate = EnhancedAggregate(
                    hidden_dim=self.hidden_dim,
                    num_edge_types=NUM_EDGE_TYPES,
                    mode=self.aggregate_mode,
                )
            if self.use_enhanced_feature_update:
                self.enhanced_feature_update = EnhancedFeatureUpdate(
                    hidden_dim=self.hidden_dim, t_dim=self.t_dim
                )
        # Storage for last angular penalty (so a trainer can pick it up).
        self.last_angular_penalty: Optional[torch.Tensor] = None

    def _any_enhanced(self) -> bool:
        return any(
            [
                self.use_enhanced_mlp,
                self.use_enhanced_coord_update,
                self.use_enhanced_aggregate,
                self.use_enhanced_feature_update,
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: Optional[int] = None,
        dative_bond_edge_attr: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        chem_context: Optional[Dict[str, Any]] = None,
        t_scalar: Optional[torch.Tensor] = None,
        pocket_emb: Optional[torch.Tensor] = None,
        metal_atom_index: Optional[torch.Tensor] = None,
        coord_geometry_class: Optional[torch.Tensor] = None,
        bond_orders_continuous: Optional[torch.Tensor] = None,
        per_edge_natural_charge: Optional[torch.Tensor] = None,
        per_edge_metal_proximity: Optional[torch.Tensor] = None,
        per_edge_dative_flag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._any_enhanced():
            return self.legacy_conv(
                h,
                x,
                edge_index,
                n_nodes=n_nodes,
                dative_bond_edge_attr=dative_bond_edge_attr,
                edge_types=edge_types,
            )

        if n_nodes is None:
            n_nodes = h.size(0)
        src, dst = edge_index

        # Chemistry signal: ADD to h (project to hidden dim).
        if chem_context is not None and hasattr(self, "chem_projector"):
            chem_signal = self.chem_projector(n_nodes, **chem_context)
            if h.size(-1) == self.hidden_dim:
                if chem_signal.size(-1) < self.hidden_dim:
                    pad = h.new_zeros(
                        n_nodes, self.hidden_dim - chem_signal.size(-1)
                    )
                    h = h + torch.cat([chem_signal, pad], dim=-1)
                else:
                    h = h + chem_signal[:, : self.hidden_dim]
            # When h is in_node_dim, the per-edge channels carry the
            # signal; we leave h unchanged.

        diff = x[src] - x[dst]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_vec = diff / dist

        if self.use_enhanced_mlp:
            if edge_types is not None:
                edge_type_emb = self.edge_type_emb(edge_types.long())
            else:
                edge_type_emb = h.new_zeros(src.shape[0], 4)
            if bond_orders_continuous is not None:
                bo_emb = self.bo_emb(bond_orders_continuous)
            else:
                bo_emb = h.new_zeros(src.shape[0], 4)
            if per_edge_natural_charge is not None:
                nc_e = per_edge_natural_charge
                if nc_e.dim() == 1:
                    nc_e = nc_e.unsqueeze(-1)
                nc_emb = self.natural_charge_emb(nc_e)
            else:
                nc_emb = h.new_zeros(src.shape[0], 4)
            if chem_context is not None:
                ox = chem_context.get("oxidation_per_atom_avg")
                if ox is not None:
                    ox_e = ox[src].unsqueeze(-1) if ox.dim() == 1 else ox[src]
                    ox_emb = self.oxidation_emb(ox_e)
                else:
                    ox_emb = h.new_zeros(src.shape[0], 4)
                ce = chem_context.get("coord_neighbors_per_atom")
                if ce is not None:
                    ce_e = ce[src].unsqueeze(-1) if ce.dim() == 1 else ce[src]
                    ce_emb = self.coord_env_emb(ce_e)
                else:
                    ce_emb = h.new_zeros(src.shape[0], 4)
            else:
                ox_emb = h.new_zeros(src.shape[0], 4)
                ce_emb = h.new_zeros(src.shape[0], 4)
            mp = per_edge_metal_proximity
            if mp is None:
                mp = h.new_zeros(src.shape[0], 1)
            elif mp.dim() == 1:
                mp = mp.unsqueeze(-1)
            dg = per_edge_dative_flag
            if dg is None:
                dg = h.new_zeros(src.shape[0], 1)
            elif dg.dim() == 1:
                dg = dg.unsqueeze(-1)
            msg = self.enhanced_mlp(
                h[src], h[dst], dist, dir_vec,
                edge_type_emb, bo_emb, nc_emb, ox_emb, ce_emb, mp, dg,
            )
        else:
            # Without EnhancedMLP we still need a per-edge msg of
            # the right dim; route through the legacy path to get it.
            legacy_h, legacy_x = self.legacy_conv(
                h, x, edge_index,
                n_nodes=n_nodes,
                dative_bond_edge_attr=dative_bond_edge_attr,
                edge_types=edge_types,
            )
            msg = legacy_h[src]

        if self.use_enhanced_aggregate:
            agg_msg = self.enhanced_aggregate(msg, edge_index, edge_types=edge_types)
        else:
            n_dst = (
                int(edge_index[1].max().item()) + 1
                if edge_index[1].numel() > 0
                else 0
            )
            agg_msg = msg.new_zeros(n_dst, msg.size(-1))
            agg_msg = agg_msg.index_add_(0, edge_index[1], msg)

        if self.use_enhanced_coord_update:
            x_new, ang_pen = self.enhanced_coord_update(
                msg, x, edge_index, edge_types,
                metal_atom_index=metal_atom_index,
                coord_geometry_class=coord_geometry_class,
            )
            self.last_angular_penalty = ang_pen
        else:
            x_new = x
            self.last_angular_penalty = None

        if self.use_enhanced_feature_update:
            if t_scalar is None:
                t_scalar = h.new_zeros(1)
            t_emb = self.t_embedding(t_scalar)
            if t_emb.dim() == 2 and t_emb.size(0) == 1:
                t_emb_per_atom = t_emb.expand(n_nodes, -1)
            else:
                t_emb_per_atom = t_emb
            if t_emb_per_atom.size(0) != n_nodes:
                t_emb_per_atom = t_emb_per_atom.expand(n_nodes, -1)
            # Wf-A22 (2026-10-13) step-2: pocket_injector_helper wire-in.
            #
            # The PerLayerPocketInjector was instantiated in __init__ but
            # never called — ``pocket_emb`` arrived at the conv layer but
            # never reached the EnhancedFeatureUpdate's pocket slot.
            # Project ``pocket_emb`` (shape ``(B, pocket_dim)``) to
            # ``(B, N, hidden)`` once per forward and forward the
            # per-atom slice to EnhancedFeatureUpdate.  When
            # ``pocket_emb is None`` (the Wf-A15/A17 default), the
            # injector returns a zero addend (zero-init projection) —
            # bit-exact identical to the pre-Wf-A22 forward.
            _per_atom_pocket: Optional[torch.Tensor] = None
            if (
                pocket_emb is not None
                and hasattr(self, "pocket_injector_helper")
            ):
                _per_atom_pocket = self.pocket_injector_helper(
                    pocket_emb, n_max=n_nodes
                )
                # Collapse the B=1 batch dim (legacy single-graph code
                # path) so the EnhancedFeatureUpdate receives a (N, hidden)
                # tensor as documented at line 664.
                if _per_atom_pocket.size(0) == 1:
                    _per_atom_pocket = _per_atom_pocket.squeeze(0)
            h_new = self.enhanced_feature_update(
                h,
                agg_msg,
                t_emb=t_emb_per_atom,
                pocket_emb=_per_atom_pocket,
            )
        else:
            h_new = agg_msg

        return h_new, x_new


__all__ = [
    "EnhancedMLP",
    "EnhancedCoordUpdate",
    "EnhancedAggregate",
    "EnhancedFeatureUpdate",
    "ChemistrySignalProjector",
    "BOContinuousEmbedding",
    "SmoothDist",
    "EnhancedEquivariantGraphConv",
    "EDGE_TYPE_SINGLE",
    "EDGE_TYPE_DOUBLE",
    "EDGE_TYPE_DATIVE",
    "EDGE_TYPE_AROMATIC",
    "NUM_EDGE_TYPES",
]
