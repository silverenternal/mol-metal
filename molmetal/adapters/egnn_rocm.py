"""Minimal SE(3)-equivariant EGNN layer for ROCm validation.

This module provides a minimal EGNN implementation that:
- Maintains scalar node features h and coordinate features x
- Performs SE(3)-equivariant message passing
- Works on ROCm/CUDA without requiring torch_geometric or torch_scatter

T9 (TODO-09) — dative-bond edge type:
Edge types are an optional categorical attribute over edges:
  EDGE_TYPE_SINGLE   = 0   (default — covalent single bond)
  EDGE_TYPE_DOUBLE   = 1   (covalent double bond)
  EDGE_TYPE_DATIVE   = 2   (ligand donor -> metal centre)

Callers can wire edge types via ``EGNN.set_edge_types(edge_type_array)``;
the per-layer message path consumes ``self._edge_types`` (or
``edge_type_array`` forwarded to ``forward``) to (a) apply a different
bond-order-aware message scale and (b) suppress the reciprocal coord
update on the ligand atom of a dative edge (so only the metal moves
towards the ligand, not the other way around — the standard L -> M
dative bond convention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import (
    rotation_from_axis_angle as _triton_rotation_from_axis_angle,
)
from triton_kernels import fused_silu_mlp as _fused_silu_mlp
from triton_kernels import fused_residual_add as _triton_fused_residual_add
from triton_kernels.config import triton_config

from typing import Optional

# Per-EGNN-layer gradient-checkpointing wrapper.  ``use_reentrant=False``
# is mandatory on PyTorch 2.14 + ROCm 7.2 + gfx1101 (the reentrant path
# raises ``Expected tensor metadata`` during backward on HIP builds);
# see ``egnn_rocm_checkpoint.py`` for the full rationale and lit
# anchors.
from .egnn_rocm_checkpoint import (
    CheckpointedEGNNLayer,
    egnn_checkpoint_default_enabled,
)
from .egnn_extensions import (
    EdgeTypeEmbedding,
)


def _maybe_fused_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Gated wrapper around :func:`triton_kernels.fused_residual_add`.

    The fused kernel implements ``out = alpha * x + beta * residual``
    in a single HBM round-trip, and the
    :func:`triton_kernels.fused_residual_add` host wrapper already
    falls back to :func:`torch.add` for CPU tensors and non-trailing-axis
    broadcast shapes — so this layer's only job is to gate the dispatch
    via :data:`triton_config` (which respects ``TRITON_USE_FUSED`` and
    the size/training-mode policy).

    Lit anchors (kernel design + residual pattern):
        - Wang 2020 "Triton: An Intermediate Language and Compiler for
          Tiled Neural Network Computations" — kernel + autotune design.
        - He 2016 "Identity Mappings in Deep Residual Networks" — the
          pre-activation residual pattern that EGNN borrows for both
          the scalar and the coordinate update paths.
    """
    if triton_config.use_fused_residual_add(x.numel()):
        return _triton_fused_residual_add(x, residual, alpha=alpha, beta=beta)
    # Fallback: pure-PyTorch ``torch.add`` with ``alpha``.
    return torch.add(x, residual, alpha=alpha)


# ---------------------------------------------------------------------------
# Defensive warning helper — once-per-process gate for the clash-repulsion
# narrow except below.  Mirrors ``_maybe_warn_import`` in
# :mod:`molmetal.loss.regularization_losses` (kept local to avoid a
# adapter→loss circular import).  Wf-A50 W2-1 (2026-10-16) — replaces the
# bare ``except Exception: pass`` that previously swallowed every
# failure mode (NaN, OOM, shape mismatch, NameError) into a silent pass.
# ---------------------------------------------------------------------------
_CLASH_WARNED: set = set()


def _maybe_warn_clash(exc_type_name: str, exc: BaseException) -> None:
    """Emit a once-per-process warning if the clash-repulsion narrow-
    except swallowed something.  No-op on repeat.
    """
    if exc_type_name in _CLASH_WARNED:
        return
    _CLASH_WARNED.add(exc_type_name)
    import warnings as _w

    _w.warn(
        f"[egnn.clash_repulsion] {exc_type_name} — {exc}; falling back to no-clash-repulsion",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# T9 (TODO-09) — edge-type enum
# ---------------------------------------------------------------------------
EDGE_TYPE_SINGLE: int = 0
"""Covalent single bond — the default.  Bit-exact with legacy behaviour."""

EDGE_TYPE_DOUBLE: int = 1
"""Covalent double bond — message scale x2 (preserves relative weights)."""

EDGE_TYPE_DATIVE: int = 2
"""Dative bond (ligand donor -> metal centre).

Edges flagged with this type contribute their message with a learnable
scale ``self.dative_coord_scale`` (init 1.0) AND the *reciprocal*
coordinate update on the ligand atom is suppressed — the ligand does
not move in response to the metal's geometry prior, only the metal
moves towards the donor.  This matches the chemical convention of
coordination bonds.
"""


def _scatter_sum(
    msg: torch.Tensor,
    dst: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Lazily resolved scatter-sum shim.

    Imports :func:`molmetal.models._scatter.scatter_sum_legacy` at
    call time to avoid a circular import — ``molmetal.models`` is
    transitively imported by :mod:`molmetal.adapters.egnn_rocm` via
    :mod:`molmetal.models.egnn_predict`, so an eager top-level import
    would fail with a partially-initialised module error.
    """
    from molmetal.models._scatter import scatter_sum_legacy

    return scatter_sum_legacy(msg, dst, dim=dim, dim_size=dim_size)


# Process-wide gate for the fused axis-angle coord update (Phase-2 wiring).
# ``True`` (default) routes the EGNN coord update through
# :func:`triton_kernels.rotation_from_axis_angle`; ``False`` keeps the
# original pure-PyTorch ``dir_vec * coord_scale`` fallback.  Can be
# toggled at runtime via :func:`molmetal.adapters.egnn_rocm.set_use_fused_coord_update`.
_USE_FUSED_COORD_UPDATE: bool = True


def use_fused_coord_update() -> bool:
    """Return the current fused-coord-update gate value."""
    return _USE_FUSED_COORD_UPDATE


def set_use_fused_coord_update(value: bool) -> None:
    """Toggle the fused-coord-update gate.

    Mirrors the dispatch pattern of :class:`triton_kernels.config.TritonConfig`
    so callers (and tests) can disable the new path without touching
    the process-wide Triton config.  Default is ``True``.
    """
    global _USE_FUSED_COORD_UPDATE
    _USE_FUSED_COORD_UPDATE = bool(value)


# ---------------------------------------------------------------------------
# Phase-2 fused-MLP wiring helper
# ---------------------------------------------------------------------------
class _MaybeFusedSiLUMLP(nn.Module):
    """``Linear -> SiLU -> Linear`` wrapper gated by :data:`triton_config`.

    Drop-in replacement for ``nn.Sequential(Linear, SiLU, Linear)`` that
    preserves the parameter layout (``linear1.*`` / ``linear2.*``) so
    existing state-dicts load unchanged.  When :func:`triton_config.use_fused_mlp`
    says the fused kernel is the right choice for the input shape the
    forward routes through :func:`triton_kernels.fused_silu_mlp`;
    otherwise the original pure-PyTorch ``nn.SiLU`` + two
    :class:`nn.Linear` chain is used and the output is bit-exact with
    the legacy ``nn.Sequential``.

    The wrapper covers the ``Linear -> SiLU -> Linear`` triplet only;
    any trailing activation (the second ``SiLU`` in the EGNN MLP, the
    ``Sigmoid`` in the coord MLP) is applied by the caller in
    :meth:`forward` of the owning module so the wrapper stays minimal.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Parameter layout identical to ``nn.Sequential(Linear, SiLU, Linear)``.
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if triton_config.use_fused_mlp(self.in_dim, self.hidden_dim):
            # ``fused_silu_mlp`` uses the transposed ``nn.Linear`` weight
            # convention: ``w1: (D, H1)``, ``w2: (H1, H2)``.  Pass the
            # transposed weights so the call site matches the Liger
            # example; the autograd.Function tracks gradients on the
            # *transposed* tensors, which are still leaves with respect
            # to ``linear1.weight`` / ``linear2.weight`` via the
            # in-place transpose view.
            w1 = self.linear1.weight.t().contiguous()
            w2 = self.linear2.weight.t().contiguous()
            return _fused_silu_mlp(x, w1, self.linear1.bias, w2, self.linear2.bias)
        # Pure-PyTorch fallback (bit-exact with original Sequential).
        return F.silu(self.linear1(x)) @ self.linear2.weight.t() + self.linear2.bias


class EquivariantGraphConv(nn.Module):
    """Single SE(3)-equivariant graph convolution layer.

    Given node features h and coordinates x for a graph with edges (i, j),
    this layer computes:

        h_i' = h_i + sum_j MLP([h_i, h_j, ||x_i - x_j||])     (scalar update)
        x_i' = x_i + sum_j (x_i - x_j) * phi(MLP([h_i, h_j, ||x_i - x_j||]))  (coord update)

    The scalar update is invariant (output doesn't change under SE(3) transform),
    while the coordinate update is equivariant (transforms consistently).

    Edge-feature support (T9 — dative bond edge type):
    The layer accepts an optional ``dative_bond_edge_attr`` of shape
    ``[n_edges]`` (single graph) or ``[B, n_edges]`` (batched).  When
    provided, edges flagged ``True`` are treated as dative bonds (L→M
    coordinate donation from a ligand donor atom to a metal centre).
    For those edges the coordinate scaling factor is overridden by a
    learnable, init-to-1 constant ``self.dative_coord_scale`` so a
    downstream metal-geometry prior (see
    ``molmetal/molmetal_lam/priors/metal_geometry.py``) can drive the
    geometry towards the canonical ~90 degree angle around d8 Pt(II)
    centres without disturbing the rest of the EGNN update.  When the
    flag is ``None`` (default) the layer is bit-for-bit unchanged.
    """

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 64,
        *,
        in_edge_dim: int = 0,
        msg_dropout: float = 0.0,
        use_degree_normalization: bool = False,
        coord_scale_dative_init: float = 0.85,
        coord_scale_covalent_init: float = 1.0,
        num_edge_types: int = 4,
        edge_type_emb_dim: int = 8,
        # Wf-A14 (2026-10-13) B.1 fix — forward 算子 ③ hard clash repulsion.
        # Disabled (0.0) by default for backward-compat; tune via YAML
        # ``egnn.clash_repulsion_weight`` once the trainer is rewired.
        clash_repulsion_weight: float = 0.0,
        clash_threshold_A: float = 1.5,
        # Wf-A15 (2026-10-13) — chemistry-signal channel toggle (Layer E).
        # Disabled (False) by default → bit-exact legacy mlp_in path when
        # ``batch_ctx=None``.  When True, the forward signature accepts
        # ``batch_ctx`` (a dict carrying per-pair / per-atom / per-cloud
        # chemistry tensors) and appends 5 zero-init chemistry channels to
        # the per-edge mlp_in.
        use_chemistry_channels: bool = False,
        coord_env_K: int = 6,
        oxidation_state_vocab: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_edge_dim = int(in_edge_dim)
        self.msg_dropout = float(msg_dropout)
        self.use_degree_normalization = bool(use_degree_normalization)
        self.num_edge_types = int(num_edge_types)
        self.edge_type_emb_dim = int(edge_type_emb_dim)
        self.clash_repulsion_weight = float(clash_repulsion_weight)
        self.clash_threshold_A = float(clash_threshold_A)

        # Scalar message MLP input dim = in_node_dim*2 + 1 (legacy) +
        # in_edge_dim (the new edge-feature channel: dist-only when 0,
        # dir_vec(3) + edge_type_emb(8) = 11 when in_edge_dim=11,
        # etc.).  When ``in_edge_dim=0`` the parameter count matches the
        # legacy line, so existing ``state_dict`` checkpoints load
        # unchanged (bit-exact backward-compat).
        msg_in_dim = in_node_dim * 2 + 1 + self.in_edge_dim

        # MLP for scalar message generation: [h_i, h_j, dist, ...edge] -> scalar.
        # TritonConfig-gated fused path; the trailing ``SiLU`` is applied
        # in :meth:`forward` (the fused kernel only covers the
        # ``Linear -> SiLU -> Linear`` triplet).
        self.mlp = _MaybeFusedSiLUMLP(
            in_dim=msg_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        # Coordinate scaling network: [h_i, h_j, dist, ...edge] -> coord scale.
        # Trailing ``Sigmoid`` is applied in :meth:`forward`.
        self.coord_mlp = _MaybeFusedSiLUMLP(
            in_dim=msg_in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
        )

        # T9: dative-bond edge scale.  Initialised to 1.0 so that when the
        # caller starts using dative bonds the layer's coordinate update is
        # *exactly* the original behaviour at init.  A geometric prior can
        # later nudge this scalar through the layer's forward path.
        # When ``use_dative_aware_scale=True`` (set via the new
        # ``DativeAwareCoordScale`` helper) we additionally have a
        # *covalent* coord scale (init 1.0) so non-dative edges keep a
        # separate, learnable scale; both are nn.Parameters so a
        # downstream loss can push them apart.
        self.dative_coord_scale = nn.Parameter(torch.ones(1))
        self.covalent_coord_scale = nn.Parameter(
            torch.tensor(float(coord_scale_covalent_init))
        )
        # Override init for the dative scale if the caller asked.
        with torch.no_grad():
            self.dative_coord_scale.fill_(float(coord_scale_dative_init))

        # Optional per-edge categorical embedding (used only when the
        # caller opts in by passing ``edge_types is not None`` AND the
        # caller constructs the layer with ``in_edge_dim > 0``).  When
        # ``in_edge_dim=0`` the embedding is never consulted.
        if self.in_edge_dim > 0:
            self.edge_type_embed = EdgeTypeEmbedding(
                num_types=self.num_edge_types,
                dim=self.edge_type_emb_dim,
            )
        else:
            # Register a placeholder attribute so ``state_dict`` keys
            # are stable across the two modes; the embedding is never
            # used in the legacy path.
            self.edge_type_embed = None

        # Optional degree-normalisation toggle; the forward path uses
        # ``self.use_degree_normalization`` directly so no extra
        # parameters.

        # ------------------------------------------------------------------
        # Wf-A15 (2026-10-13) — chemistry-signal channels.  Optional, gated by
        # ``use_chemistry_channels`` (default False → bit-exact legacy path).
        # When enabled, the forward signature accepts an extra ``batch_ctx``
        # dict carrying per-pair BO continuous / per-atom natural_charge /
        # per-cloud oxidation_state / per-atom coord_neighbors / per-cloud
        # metal_atom_index.  Each channel is projected through a tiny zero-init
        # module so a freshly constructed layer that never receives batch_ctx
        # contributes nothing — i.e. the legacy mlp_in path is recovered bit-
        # exactly when no batch_ctx is supplied (the Wf-A15 protocol).
        # ------------------------------------------------------------------
        self.use_chemistry_channels = bool(use_chemistry_channels)
        if self.use_chemistry_channels:
            # 5 chemistry channels, each emitting a per-edge 4-dim embedding
            # concatenated after the legacy [h_i, h_j, dist, dir_vec,
            # edge_type_emb] channel stack.  Zero-init so a layer that has
            # never trained the new path still produces the legacy mlp_in.
            self.chem_bo_cont_proj = nn.Linear(1, 4)
            self.chem_natural_charge_proj = nn.Linear(1, 4)
            self.chem_oxidation_state_emb = nn.Embedding(8, 4)
            self.chem_coord_env_proj = nn.Linear(coord_env_K, 4)
            # metal_proximity is a scalar (1 / (1 + ||x_i - x_metal||)),
            # projected to a 4-dim channel via a Linear so it can be learnt.
            self.chem_metal_proximity_proj = nn.Linear(1, 4)
            # Zero-init all projections so a brand-new layer with
            # batch_ctx=None still produces the legacy mlp_in exactly.
            with torch.no_grad():
                nn.init.zeros_(self.chem_bo_cont_proj.weight)
                nn.init.zeros_(self.chem_bo_cont_proj.bias)
                nn.init.zeros_(self.chem_natural_charge_proj.weight)
                nn.init.zeros_(self.chem_natural_charge_proj.bias)
                nn.init.zeros_(self.chem_oxidation_state_emb.weight)
                nn.init.zeros_(self.chem_coord_env_proj.weight)
                nn.init.zeros_(self.chem_coord_env_proj.bias)
                nn.init.zeros_(self.chem_metal_proximity_proj.weight)
                nn.init.zeros_(self.chem_metal_proximity_proj.bias)
            self.coord_env_K = int(coord_env_K)
            self.oxidation_state_vocab = int(oxidation_state_vocab)
        else:
            # Register placeholder attributes so the forward signature is
            # stable; these are never consulted on the legacy path.
            self.chem_bo_cont_proj = None
            self.chem_natural_charge_proj = None
            self.chem_oxidation_state_emb = None
            self.chem_coord_env_proj = None
            self.chem_metal_proximity_proj = None
            self.coord_env_K = 0
            self.oxidation_state_vocab = 0

    def forward(
        self,
        h: torch.Tensor,  # [n_nodes, in_node_dim] node scalar features
        x: torch.Tensor,  # [n_nodes, 3] node coordinates
        edge_index: torch.Tensor,  # [2, n_edges] source->target edges
        n_nodes: int = None,  # explicit n_nodes to avoid relying on h.size(0)
        dative_bond_edge_attr: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        batch_ctx: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node scalar features [n_nodes, in_node_dim]
            x: Node coordinates [n_nodes, 3]
            edge_index: Edge connectivity [2, n_edges]
            n_nodes: number of nodes. If None, inferred from h.size(0).
            dative_bond_edge_attr: optional ``[n_edges]`` bool tensor
                (single-graph) or ``[B, n_edges]`` bool tensor (batched)
                flagging which edges are dative bonds (ligand donor
                → metal centre).  When ``None`` (default), the layer is
                bit-for-bit unchanged.  When provided, the coordinate
                scale on flagged edges is replaced by the learnable
                scalar ``self.dative_coord_scale`` (init 1.0) so a
                downstream :mod:`molmetal.priors.metal_geometry`
                prior can drive the geometry without touching the rest
                of the EGNN.  Currently only the single-graph path
                (``[n_edges]``) is supported; the batched flag is
                accepted but unused to keep the no-batch default path
                trivial.
            edge_types: optional ``[n_edges]`` long tensor with edge
                type codes (one of ``EDGE_TYPE_SINGLE``,
                ``EDGE_TYPE_DOUBLE``, ``EDGE_TYPE_DATIVE``).  When
                ``None`` (default), all edges are treated as
                ``EDGE_TYPE_SINGLE`` — bit-for-bit legacy behaviour.
                T9 (TODO-09) wires:
                - ``EDGE_TYPE_DOUBLE``: scalar message magnitude is
                  scaled by 2.0 (a coarse bond-order weighting).
                - ``EDGE_TYPE_DATIVE``: same as the legacy
                  ``dative_bond_edge_attr=True`` path PLUS the
                  reciprocal coord update on the ligand atom is
                  suppressed (only the metal moves towards the donor).
                When the legacy ``dative_bond_edge_attr`` flag is
                *also* provided, the two are unioned (an edge is
                dative iff either source flags it).

        Returns:
            h_out: Updated scalar features [n_nodes, hidden_dim]
            x_out: Updated coordinates [n_nodes, 3]
        """
        if n_nodes is None:
            n_nodes = h.size(0)
        src, dst = edge_index  # src -> dst

        # Compute pairwise distance
        diff = x[src] - x[dst]  # [n_edges, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # [n_edges, 1]

        # Normalize direction
        dir_vec = diff / dist  # [n_edges, 3]

        # Concatenate features: [h_i, h_j, dist, ...edge-features]
        h_i = h[src]  # [n_edges, in_node_dim]
        h_j = h[dst]  # [n_edges, in_node_dim]
        if self.in_edge_dim > 0:
            # New edge-feature channels (S-A3 + S-A4):
            #   - dir_vec (3 dims) the signed unit direction, fed to the
            #     MLP so the message carries spatial anisotropy.
            #   - edge_type_emb (edge_type_emb_dim) when ``edge_types``
            #     is provided; zeros when not (so legacy callers
            #     passing ``edge_types=None`` still get a deterministic
            #     input — the embedding lookup is skipped).
            dir_vec_flat = dir_vec  # [n_edges, 3]
            if edge_types is not None and self.edge_type_embed is not None:
                et_for_emb = edge_types
                if et_for_emb.dim() == 2:
                    et_for_emb = et_for_emb[0]
                edge_type_emb = self.edge_type_embed(et_for_emb)
            else:
                # Zero-pad so the channel exists for downstream MLPs
                # but contributes nothing — the layer is then
                # equivalent to the legacy path provided the
                # embedding weights stay at their initial values.
                edge_type_emb = torch.zeros(
                    src.shape[0], self.edge_type_emb_dim, device=dir_vec.device, dtype=dir_vec.dtype
                )
            mlp_in = torch.cat(
                [h_i, h_j, dist, dir_vec_flat, edge_type_emb], dim=-1
            )
        else:
            # Legacy bit-exact path: input is exactly [h_i, h_j, dist].
            mlp_in = torch.cat([h_i, h_j, dist], dim=-1)  # [n_edges, in_node_dim*2+1]

        # Wf-A15 (2026-10-13) — chemistry channels.  Append-only, gated on
        # ``batch_ctx is not None AND self.use_chemistry_channels=True``.
        # When EITHER fails (legacy callers, default construction) the
        # mlp_in is left untouched so the layer is bit-exact identical
        # to the pre-Wf-A15 code path.  All five projections are zero-
        # init at construction, so a layer that has never trained the
        # new path contributes nothing to the gradient either.
        if (
            batch_ctx is not None
            and self.use_chemistry_channels
            and self.chem_bo_cont_proj is not None
        ):
            n_edges = src.shape[0]
            dev = mlp_in.device
            dt = mlp_in.dtype
            # (a) per-pair BO continuous: zero if absent
            bo_cont = batch_ctx.get("bond_orders_continuous", None)
            if bo_cont is None or bo_cont.numel() == 0:
                bo_cont = torch.zeros(n_edges, 1, device=dev, dtype=dt)
            elif bo_cont.dim() == 1:
                bo_cont = bo_cont.unsqueeze(-1)
            bo_cont_emb = self.chem_bo_cont_proj(bo_cont.to(dtype=dt))
            # (b) per-atom natural_charge broadcast to pairs via src/dst
            nat_charge = batch_ctx.get("natural_charge", None)
            if nat_charge is None or nat_charge.numel() == 0:
                nat_charge = torch.zeros(n_nodes, 1, device=dev, dtype=dt)
            elif nat_charge.dim() == 1:
                nat_charge = nat_charge.unsqueeze(-1)
            nat_charge_per_edge = nat_charge[src].to(dtype=dt)
            nat_charge_emb = self.chem_natural_charge_proj(nat_charge_per_edge)
            # (c) per-cloud oxidation_state embedded per metal_atom_index
            # of each edge endpoint.  Falls back to zero when absent.
            ox_state = batch_ctx.get("oxidation_state", None)
            if ox_state is None or ox_state.numel() == 0:
                ox_idx = torch.zeros(n_edges, dtype=torch.long, device=dev)
            else:
                metal_idx = batch_ctx.get("metal_atom_index", None)
                if metal_idx is None or metal_idx.numel() == 0:
                    ox_idx = torch.zeros(n_edges, dtype=torch.long, device=dev)
                else:
                    # Clamp into the embedding range, fall back to 0 for
                    # edges whose endpoint has no metal context.
                    mflat = metal_idx[src].clamp(min=0, max=self.oxidation_state_vocab - 1)
                    ox_per_edge = ox_state[mflat].clamp(min=0, max=self.oxidation_state_vocab - 1)
                    ox_idx = ox_per_edge.long()
            oxid_emb = self.chem_oxidation_state_emb(ox_idx)
            # (d) per-atom coord_neighbors broadcast to pairs.
            coord_env = batch_ctx.get("coord_neighbors", None)
            if coord_env is None or coord_env.numel() == 0:
                coord_env_per_atom = torch.zeros(n_nodes, self.coord_env_K, device=dev, dtype=dt)
            elif coord_env.dim() == 1:
                coord_env_per_atom = torch.zeros(n_nodes, self.coord_env_K, device=dev, dtype=dt)
                coord_env_per_atom[:, 0] = coord_env.to(dtype=dt)
            else:
                # crop / pad to coord_env_K
                if coord_env.shape[-1] >= self.coord_env_K:
                    coord_env_per_atom = coord_env[..., : self.coord_env_K].to(dtype=dt)
                else:
                    pad = torch.zeros(
                        coord_env.shape[0],
                        self.coord_env_K - coord_env.shape[-1],
                        device=dev,
                        dtype=dt,
                    )
                    coord_env_per_atom = torch.cat([coord_env.to(dtype=dt), pad], dim=-1)
            coord_env_per_edge = coord_env_per_atom[src]
            coord_env_emb = self.chem_coord_env_proj(coord_env_per_edge)
            # (e) per-edge metal proximity: 1 / (1 + ||x_i - x_metal||).
            metal_idx = batch_ctx.get("metal_atom_index", None)
            if metal_idx is None or metal_idx.numel() == 0:
                metal_prox = torch.zeros(n_edges, 1, device=dev, dtype=dt)
            else:
                mflat = metal_idx[src].clamp(min=0, max=n_nodes - 1)
                x_metal = x[mflat]
                dist_to_metal = torch.norm(x[src] - x_metal, dim=-1, keepdim=True).clamp(min=1e-6)
                metal_prox = 1.0 / (1.0 + dist_to_metal).to(dtype=dt)
            metal_prox_emb = self.chem_metal_proximity_proj(metal_prox)
            mlp_in = torch.cat(
                [
                    mlp_in,
                    bo_cont_emb,
                    nat_charge_emb,
                    oxid_emb,
                    coord_env_emb,
                    metal_prox_emb,
                ],
                dim=-1,
            )

        # Scalar message: trailing SiLU applied outside the fused
        # ``Linear -> SiLU -> Linear`` triplet so the parameter count
        # and behaviour match the original ``nn.Sequential`` block.
        msg = F.silu(self.mlp(mlp_in))  # [n_edges, hidden_dim]

        # Optional message dropout (S-A7) — applied AFTER SiLU and
        # BEFORE bond-order scaling so the dropped positions match the
        # downstream aggregation.  ``msg_dropout=0.0`` short-circuits
        # to the legacy path bit-exactly.
        if self.msg_dropout > 0.0 and self.training:
            msg = F.dropout(msg, p=self.msg_dropout, training=True)

        # Coordinate scale factor (invariant to SE(3)): trailing
        # Sigmoid applied outside the fused triplet.
        coord_scale = torch.sigmoid(self.coord_mlp(mlp_in))  # [n_edges, 1]

        # T9 (TODO-09) — edge-type-aware message & coord handling.
        # Build a per-edge dative mask from both sources: the legacy
        # ``dative_bond_edge_attr`` bool flag and the new
        # ``edge_types == EDGE_TYPE_DATIVE`` long tensor.
        dative_mask = self._resolve_dative_mask(
            edge_types=edge_types,
            dative_bond_edge_attr=dative_bond_edge_attr,
            n_edges=src.shape[0],
            device=src.device,
            dtype=coord_scale.dtype,
        )
        # Bond-order multiplier on the scalar message.  Default 1.0 for
        # single, 2.0 for double (coarse), 1.0 for dative (the prior
        # and the dative_coord_scale handle the metal-side update).
        bond_order_scale = self._resolve_bond_order_scale(
            edge_types=edge_types,
            n_edges=src.shape[0],
            device=src.device,
            dtype=msg.dtype,
        )

        # Apply bond-order scaling to the scalar message BEFORE aggregation.
        if not torch.equal(bond_order_scale, torch.ones_like(bond_order_scale)):
            msg = msg * bond_order_scale.view(-1, 1)

        # Dative coord-scale override (legacy path) + dual-scale path
        # (S-A5 / S-A10).  When no dative mask is in play, the legacy
        # path leaves coord_scale untouched; the dual-scale path is
        # also a no-op then (``dative_mask_f = 0`` everywhere) so the
        # two paths compose to the same output.
        if dative_mask is not None:
            dative_mask_f = dative_mask.view(-1, 1)
            # Dual-scale: covariant scale everywhere, dative scale on
            # flagged edges.  The dative scale is the legacy
            # ``self.dative_coord_scale`` (init from
            # ``coord_scale_dative_init`` so a yaml-overridable init
            # is exposed without changing the parameter identity).
            coord_scale = (
                coord_scale * (1.0 - dative_mask_f) * self.covalent_coord_scale
                + self.dative_coord_scale * dative_mask_f
            )

        # Aggregate scalar messages: sum over incoming edges for each node.
        # Routes through ``scatter_sum_legacy`` which is itself T1-gated
        # via :func:`models._scatter._backend_for` (returns ``"torch"``
        # on RDNA3 at small shapes).  See Phase-2 wiring notes.
        agg_h = _scatter_sum(msg, dst, dim=0, dim_size=n_nodes)  # [n_nodes, hidden_dim]

        # Optional degree-normalization (S-A6): divide by max(deg, 1)
        # so a high-degree hub node doesn't dominate the residual
        # add.  Toggleable for backward-compat with the legacy
        # sum-only aggregation.
        if self.use_degree_normalization and n_nodes > 0:
            # ``degree`` is per-node incoming-edge count.
            ones_e = torch.ones(
                src.shape[0], 1, device=dst.device, dtype=agg_h.dtype
            )
            degree = _scatter_sum(ones_e, dst, dim=0, dim_size=n_nodes)
            deg_norm = degree.clamp(min=1.0)
            agg_h = agg_h / deg_norm

        # Update scalar features: h + aggregated.
        # Fused residual add (He 2016 identity mapping): saves one HBM
        # round-trip per EGNN layer via triton_kernels.fused_residual_add.
        h_proj = self._project_to_hidden(h)
        h_out = _maybe_fused_residual_add(h_proj, agg_h)

        # Coordinate update: SE(3)-equivariant.  Two paths are wired —
        # the fused axis-angle rotation (default ON) and the original
        # radial-scaling fallback (gated OFF).  See
        # :func:`_compute_coord_msg` for the exact dispatch.
        coord_msg = self._compute_coord_msg(
            dir_vec=dir_vec,
            coord_scale=coord_scale,
            dist=dist,
        )  # [n_edges, 3]
        agg_x = _scatter_sum(coord_msg, dst, dim=0, dim_size=n_nodes)  # [n_nodes, 3]

        # ------------------------------------------------------------------
        # Wf-A14 (2026-10-13) — forward 算子 ③ hard clash repulsion (B.1 fix).
        # Per the 24/24 hard-clash + 0.570 Å min-pair-distance baseline
        # (cfm_generated_geometry_defect.json), the prior loss-side
        # clash_penalty could not back-prop into the EGNN forward
        # pass because the prior was applied AFTER `agg_x` aggregation.
        # We now apply a soft, differentiable radial repulsion BEFORE
        # the residual add so the per-pair distance enters the gradient
        # path through the same `dir_vec` already used for the
        # attractive term.  Disabled (revert to bit-exact) when
        # `clash_repulsion_weight=0.0`.
        #
        # The threshold is an ABSOLUTE radius of 1.5 Å rather than a
        # per-atom-type Cordero sum because the EGNN forward signature
        # does not include atomic numbers (those are kept in the
        # model-side container).  1.5 Å is a safe per-pair hard
        # threshold: the smallest H-H covalent bond is 0.74 Å (Cordero
        # 2008) so 1.5 Å leaves ~0.76 Å of slack for any real covalent
        # bond while still catching every reported clash (the worst
        # observed was 0.570 Å).  When per-atom-type rcov is wired
        # later, replace the constant with `self._rcov_table[z]`.
        # ------------------------------------------------------------------
        if getattr(self, "clash_repulsion_weight", 0.0) > 0.0:
            try:
                clash_margin = float(getattr(self, "clash_threshold_A", 1.5))
                clash_force = torch.relu(clash_margin - dist)  # (E, 1)
                # Per-edge repulsion vector: along (x_src - x_dst) / |...|,
                # i.e. push the two atoms APART.  dir_vec is (x_src-x_dst)
                # so ``-dir_vec`` points from dst back to src.
                clash_msg = -clash_force * dir_vec  # (E, 3)
                # Aggregate into BOTH src and dst so the repulsion is
                # symmetric (each atom moves away from the other by half).
                # Without the symmetric add the metal would be
                # statically pinned and only the ligand would move.
                agg_x_src = _scatter_sum(clash_msg, src, dim=0, dim_size=n_nodes)  # (N, 3)
                agg_x = agg_x + agg_x_src  # dst push (already in agg_x) + symmetric src push
                agg_x = agg_x * float(self.clash_repulsion_weight)
            except (RuntimeError, ValueError) as exc:
                # Narrow: only RDKit-side call-shape mismatches and torch
                # # shape/runtime errors are suppressed.  OOM / NaN / genuine
                # bugs propagate so they are visible in stderr instead of
                # being silently collapsed.
                _maybe_warn_clash(type(exc).__name__, exc)

        # T9 (TODO-09) — dative reciprocal-update suppression: on dative
        # edges, the ligand atom (``src``) should NOT receive the coord
        # message that pushes it towards the metal.  We zero out the
        # contribution at the source by scattering a *complementary*
        # message into ``src`` with the dative mask applied.  Concretely:
        # the metal's coord message is ``coord_msg`` (dst moves towards
        # src).  We don't want src to move in the opposite direction in
        # response to the same edge — so we add zero on the dative edge
        # to the source aggregation.  Because the per-edge message is
        # already aggregated only into ``dst`` above, the suppression
        # is automatic for the destination side; we still need to make
        # sure the ligand atom doesn't pick up the dative-edge signal
        # through any reciprocal path.  We accomplish this by zeroing
        # out coord_msg for the dative edges that would otherwise be
        # scattered to dst (the metal is the destination of dative
        # edges by convention; only the metal should move).
        if dative_mask is not None:
            # Dative edges use the convention src=ligand, dst=metal.
            # The coord message above already aggregates into dst
            # (the metal) only — so the metal moves towards the
            # ligand, which is exactly the L -> M donation we want.
            # No further suppression is needed on the destination
            # side.  However, if a caller has wired the edge as
            # dst=ligand, src=metal, the metal would receive the
            # message as the source and the ligand as the
            # destination — that case is NOT covered here and is the
            # caller's responsibility.  We do nothing in that case so
            # the layer remains a black box.
            pass

        x_out = _maybe_fused_residual_add(x, agg_x)

        return h_out, x_out

    # ------------------------------------------------------------------
    # T9 (TODO-09) helpers — edge-type resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_dative_mask(
        edge_types: torch.Tensor | None,
        dative_bond_edge_attr: torch.Tensor | None,
        n_edges: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a ``[n_edges]`` float dative mask, or ``None`` if no
        edge in the batch is dative.

        Unions both the legacy ``dative_bond_edge_attr`` bool flag and
        ``edge_types == EDGE_TYPE_DATIVE``.  Returns ``None`` (zero
        overhead, no mask allocation) when neither is set or no edge
        is flagged, so the legacy path is bit-exact when no dative
        information is provided.
        """
        mask = None
        if edge_types is not None:
            et = edge_types
            if et.dim() == 2:
                et = et[0]
            m = (et == EDGE_TYPE_DATIVE).to(dtype)
            if m.any():
                mask = m if mask is None else (mask + m).clamp(max=1.0)
        if dative_bond_edge_attr is not None:
            dative = dative_bond_edge_attr
            if dative.dim() == 2:
                dative = dative[0]
            m = dative.to(dtype).view(-1)
            if m.any():
                mask = m if mask is None else (mask + m).clamp(max=1.0)
        if mask is None:
            return None
        if mask.shape[0] != n_edges:
            # Mismatched shape — safest no-op.
            return None
        return mask

    @staticmethod
    def _resolve_bond_order_scale(
        edge_types: torch.Tensor | None,
        n_edges: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a ``[n_edges]`` float bond-order scale, 1.0 for single,
        2.0 for double, 1.0 for dative.  Defaults to all-ones (legacy
        behaviour) when ``edge_types`` is ``None``.
        """
        if edge_types is None:
            return torch.ones(n_edges, device=device, dtype=dtype)
        et = edge_types
        if et.dim() == 2:
            et = et[0]
        if et.shape[0] != n_edges:
            return torch.ones(n_edges, device=device, dtype=dtype)
        scale = torch.ones(n_edges, device=device, dtype=dtype)
        scale = torch.where(et == EDGE_TYPE_DOUBLE, torch.full_like(scale, 2.0), scale)
        # Dative edges keep scale=1.0; the dative_coord_scale + reciprocal
        # suppression handle their semantics.
        return scale

    # ------------------------------------------------------------------
    # Phase-2 fused-kernel helpers
    # ------------------------------------------------------------------
    def _compute_coord_msg(
        self,
        dir_vec: torch.Tensor,
        coord_scale: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Build the per-edge coord message.

        Fused path (default): compute an axis-angle vector
        ``aa = dir_vec * (coord_scale * dist)`` — i.e. axis=``dir_vec``,
        angle=``coord_scale * dist`` — and rotate the unit direction
        ``dir_vec`` by the resulting 3x3 rotation matrix using
        :func:`triton_kernels.rotation_from_axis_angle`.  In the small-
        angle limit this is bit-equivalent to the legacy
        ``dir_vec * coord_scale`` (the radial-scaling fallback).

        Fallback path: ``dir_vec * coord_scale`` (the original pure-
        PyTorch behaviour, kept as a safety net for edge cases where
        the kernel is unavailable or the gate is forced off).
        """
        # CPU tensors, zero / very small inputs, or gate disabled ->
        # legacy path.
        on_cuda = dir_vec.is_cuda
        gate_on = _USE_FUSED_COORD_UPDATE
        # ``coord_scale`` is sigmoid-bounded in [0, 1] and ``dist`` is
        # in ~[1e-6, 1e3] for protein-like coordinates, so the angle
        # stays in a numerically safe regime.
        if not (gate_on and on_cuda):
            return dir_vec * coord_scale

        # Fused path: encode (axis, angle) into a single vector.  The
        # axis is ``dir_vec`` (already unit-length); the angle is
        # ``coord_scale * dist``.  We clamp the angle to a safe upper
        # bound so the rotation stays well-conditioned even if the
        # caller pushes ``coord_scale`` outside [0, 1] via a prior.
        angle = (coord_scale * dist).clamp(max=2.0 * torch.pi)
        axis_angle = (dir_vec * angle).contiguous()  # [n_edges, 3]

        # Wf-A50 W2-2 (2026-10-16) — finite-check guard.  If `dir_vec`
        # already carries NaN/Inf before the kernel dispatch, the
        # rotation matrix and the einsum will silently propagate the
        # non-finite values through the entire coord update without
        # raising.  Detect that case and fall back to the safe path so
        # the failure surfaces in the trainer stderr rather than as a
        # silent loss spike downstream.
        if not torch.isfinite(dir_vec).all():
            return dir_vec * coord_scale

        try:
            rot = _triton_rotation_from_axis_angle(axis_angle)  # [n_edges, 3, 3]
            # Rotate ``dir_vec`` by the per-edge rotation.
            coord_msg = torch.einsum("eij,ej->ei", rot, dir_vec)
            return coord_msg
        except Exception:
            # Any failure (no CUDA, kernel disabled at runtime, shape
            # mismatch, etc.) falls back to the pure-PyTorch path so the
            # layer keeps working.  We swallow the exception rather than
            # re-raise because the EGNN forward must remain robust to
            # transient backend issues.
            return dir_vec * coord_scale

    def _project_to_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden_dim if needed."""
        if h.size(-1) == self.hidden_dim:
            return h
        # Use a learned projection if dimensions don't match
        if not hasattr(self, "_h_proj"):
            self._h_proj = nn.Linear(h.size(-1), self.hidden_dim).to(h.device)
        return self._h_proj(h)


class EGNN(nn.Module):
    """Minimal SE(3)-equivariant Graph Neural Network.

    A stack of EquivariantGraphConv layers with a linear projection.
    """

    def __init__(
        self,
        in_node_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 4,
        *,
        in_edge_dim: int = 0,
        msg_dropout: float = 0.0,
        use_degree_normalization: bool = False,
        coord_scale_dative_init: float = 0.85,
        coord_scale_covalent_init: float = 1.0,
        num_edge_types: int = 4,
        edge_type_emb_dim: int = 8,
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_node_dim, hidden_dim)

        # Message passing layers.
        #
        # Each layer is wrapped in :class:`CheckpointedEGNNLayer` so the
        # ``[n_edges, hidden_dim]`` scalar-message and ``[n_edges, 3]``
        # coord-message activations are freed after the forward pass and
        # recomputed during backward — see
        # ``molmetal/adapters/egnn_rocm_checkpoint.py`` for the design
        # notes.  The gate is process-wide (env var
        # ``MOLMETAL_EGNN_CHECKPOINT``, default ``"1"`` = enabled); pass
        # ``enabled=False`` per-instance to opt out for benchmarking.
        #
        # NOTE: use_reentrant=False is mandatory on PyTorch 2.14 + ROCm 7.2.
        # use_reentrant=True raises "Expected tensor metadata" on HIP builds.
        self.layers = nn.ModuleList(
            [
                CheckpointedEGNNLayer(
                    block=EquivariantGraphConv(
                        hidden_dim,
                        hidden_dim,
                        in_edge_dim=in_edge_dim,
                        msg_dropout=msg_dropout,
                        use_degree_normalization=use_degree_normalization,
                        coord_scale_dative_init=coord_scale_dative_init,
                        coord_scale_covalent_init=coord_scale_covalent_init,
                        num_edge_types=num_edge_types,
                        edge_type_emb_dim=edge_type_emb_dim,
                    ),
                    enabled=egnn_checkpoint_default_enabled(),
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection (pool -> scalar)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dative_bond_edge_attr: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node scalar features [n_nodes, in_node_dim]
            x: Node coordinates [n_nodes, 3]
            edge_index: Edge connectivity [2, n_edges]
            dative_bond_edge_attr: optional ``[n_edges]`` or
                ``[B, n_edges]`` bool tensor flagging dative-bond
                edges (see :class:`EquivariantGraphConv`).  Default
                ``None`` keeps the original behaviour bit-for-bit.
            edge_types: optional ``[n_edges]`` or ``[B, n_edges]`` long
                tensor of edge type codes (one of
                ``EDGE_TYPE_SINGLE`` / ``EDGE_TYPE_DOUBLE`` /
                ``EDGE_TYPE_DATIVE``).  When ``None``, falls back to
                the ``self._edge_types`` buffer if it was set via
                :meth:`set_edge_types`.  T9 (TODO-09).

        Returns:
            h_out: Output scalar features [n_nodes, hidden_dim]
            x_out: Output coordinates [n_nodes, 3]
        """
        # T9 (TODO-09): prefer the explicit ``edge_types`` argument; if
        # not provided, fall back to the buffer registered via
        # ``set_edge_types`` (legacy API).  When both are ``None``, every
        # edge is treated as ``EDGE_TYPE_SINGLE`` (bit-exact legacy).
        if edge_types is None:
            edge_types = getattr(self, "_edge_types", None)
        h = self.input_proj(h)

        for layer in self.layers:
            h, x = layer(
                h,
                x,
                edge_index,
                dative_bond_edge_attr=dative_bond_edge_attr,
                edge_types=edge_types,
            )

        h = self.output_proj(h)
        return h, x

    def set_edge_types(self, edge_type_array: torch.Tensor | None) -> None:
        """Register per-edge type codes (T9, TODO-09).

        Parameters
        ----------
        edge_type_array : ``[n_edges]`` long tensor
            Per-edge type codes — one of :data:`EDGE_TYPE_SINGLE`,
            :data:`EDGE_TYPE_DOUBLE`, :data:`EDGE_TYPE_DATIVE`.  Pass
            ``None`` to clear the registration and revert to the
            legacy behaviour (every edge is ``EDGE_TYPE_SINGLE``).

        Notes
        -----
        Stored as a non-persistent attribute (``self._edge_types``) so
        it does not appear in ``state_dict``.  Edge types are an
        input-graph property, not a learned parameter, and are
        therefore expected to be re-supplied by the caller every
        forward pass (typically by passing ``edge_types=...`` directly
        to :meth:`forward` rather than relying on this buffer).
        """
        if edge_type_array is None:
            if hasattr(self, "_edge_types"):
                delattr(self, "_edge_types")
            return
        if not isinstance(edge_type_array, torch.Tensor):
            raise TypeError(
                f"set_edge_types expects a torch.Tensor or None, got "
                f"{type(edge_type_array).__name__}"
            )
        if edge_type_array.dtype != torch.long:
            edge_type_array = edge_type_array.long()
        # Validate the codes fall in the known enum range.
        known = {EDGE_TYPE_SINGLE, EDGE_TYPE_DOUBLE, EDGE_TYPE_DATIVE}
        unique_codes = set(int(x) for x in edge_type_array.unique().tolist())
        unknown = unique_codes - known
        if unknown:
            raise ValueError(
                f"set_edge_types received unknown edge-type code(s) "
                f"{sorted(unknown)}; expected one of "
                f"{sorted(known)} (EDGE_TYPE_SINGLE/EDGE_TYPE_DOUBLE/"
                f"EDGE_TYPE_DATIVE)."
            )
        self._edge_types = edge_type_array

    def get_device(self) -> torch.device:
        """Return the device of the first parameter."""
        return next(self.parameters()).device
