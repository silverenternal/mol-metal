"""Coordination-Aware GNN — CPU skeleton for R18 P2-3 (ChemRxiv 2026).

Coordination-aware GNN is a metal-coordination-chemistry-aware message
passing layer that explicitly conditions message computation on each
node's *coordination class* (one-hot over coordination numbers 0..N).
This is the standard remedy for the long-standing failure mode of
generic EGNNs on metalloproteins / coordination complexes (Huheey 1993
§4-5; Reedijk 1987): vanilla message passing treats every neighbor
edge the same, so it cannot distinguish e.g. the four equatorial
ligands of a square-planar d8 Pt(II) centre from the six vertices of
an octahedral Ru(II) complex. By gating the message MLP on a
coordination-class one-hot, the layer learns a separate set of
filters for each coordination regime (0 = isolated, 2 = linear,
4 = tetrahedral / square-planar, 6 = octahedral, …).

Design constraints (R18 P2-3):

* CPU-only skeleton — no ``triton_kernels``, no torch_scatter, no
  ROCm-specific paths. The goal is a portable, dependency-light
  building block that matches the ChemRxiv-2026 spec.
* SE(3)-**invariance** of the scalar output: rotating / translating
  the optional ``coords`` argument must NOT change the returned
  features. We use *distances* (rotationally + translationally
  invariant scalars) as the only coord-derived feature.
* Does NOT modify ``molmetal/adapters/egnn_rocm.py`` — this module
  is independent. The R17 EGNN ship is read-only here.

API
---
``CoordinationAwareGNNLayer(hidden_dim, coord_channels)``
    Scalar graph-conv layer. ``forward(x, edge_index, coordination,
    coords=None) -> updated_features``.

``CoordinationNumberHead(n_metals, max_coordination)``
    Per-metal coordination-number classifier. Output shape
    ``(batch, n_metals, max_coordination)``.

``CoordinationGeometryHead(n_geometries)``
    Per-batch coordination-geometry classifier. Output shape
    ``(batch, n_geometries)``.

``CoordinationAwareGNNComposite(layer, num_head, geom_head)``
    Bundles the layer + the two heads. Forward returns
    ``(features, num_logits, geom_logits)``.

Tests live in ``molmetal/tests/test_coordination_gnn.py`` (~25 unit
tests).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Coordination-class defaults
# ---------------------------------------------------------------------------

#: Default number of coordination classes.  Covers the five canonical
#: d-block geometries (coord 0/2/4/6) plus a few off-numbers for
#: low-coordination main-group and exotic cases.  Mirrors the
#: ``n_coordination_classes=12`` default in the R18 spec.
N_COORDINATION_CLASSES_DEFAULT: int = 12

#: Default feature dimensionality for hidden state.
HIDDEN_DIM_DEFAULT: int = 128

#: Default number of metals (matches the 30-entry tmQM + 6-entry
#: legacy subset; we round to 12 here to keep the head lightweight).
N_METALS_DEFAULT: int = 12

#: Default maximum coordination number per metal.  Coordination
#: numbers in known tmQM complexes lie in [1, 9] with 6 dominating;
#: 12 is a safe upper bound for one-hot logits.
MAX_COORDINATION_DEFAULT: int = 12

#: Default number of distinct coordination geometries (covers
#: octahedral / tetrahedral / square-planar / linear / trigonal-planar
#: / trigonal-bipyramidal / square-pyramidal / unknown = 8).
N_GEOMETRIES_DEFAULT: int = 8


# ---------------------------------------------------------------------------
# Pure-torch scatter-sum (no torch_scatter dependency)
# ---------------------------------------------------------------------------

def _scatter_sum(
    msg: torch.Tensor,
    dst: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Pure-torch segment-sum aggregator.

    ``msg`` is ``(n_edges, ...)``, ``dst`` is ``(n_edges,)``.  Returns
    a tensor of shape ``(dim_size, ...)`` whose ``i``-th row is the
    sum of ``msg`` over edges ``e`` with ``dst[e] == i``.

    Uses ``index_add_`` which is bit-equivalent to torch_scatter's
    ``scatter_add`` for non-negative indices — fine for our edge-index
    conventions (which are always non-negative ints).
    """
    if msg.numel() == 0:
        shape = (dim_size,) + tuple(msg.shape[1:])
        return msg.new_zeros(shape)
    out = msg.new_zeros((dim_size,) + tuple(msg.shape[1:]))
    out.index_add_(0, dst, msg)
    return out


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class CoordinationAwareGNNLayer(nn.Module):
    """Coordination-aware graph convolution layer.

    Each node's hidden state ``x`` is updated by aggregating messages
    from its graph neighbors, where the message MLP is *gated* by
    the destination node's coordination-class one-hot.  This is the
    key inductive bias that lets a coordination-aware GNN distinguish
    e.g. a square-planar d8 Pt(II) centre (coord 4, geometry =
    square-planar) from an octahedral Ru(II) centre (coord 6,
    geometry = octahedral) — a distinction a vanilla MPNN cannot
    make.

    Args:
        hidden_dim: feature dimensionality ``D``.  Defaults to 128.
        n_coordination_classes: number of coordination-class channels
            in the one-hot / soft-max vector passed in via
            ``coordination``.  Defaults to 12 (the R18 spec value).

    Forward
    -------
    ``forward(x, edge_index, coordination, coords=None)``

    * ``x``: ``(n_atoms, hidden_dim)`` node features.
    * ``edge_index``: ``(2, n_edges)`` with the conventional
      ``(src, dst)`` orientation.  Each edge contributes a single
      message from ``src`` to ``dst``; the layer then aggregates into
      ``dst``.
    * ``coordination``: ``(n_atoms, n_coordination_classes)`` per-node
      coordination class (one-hot or soft).  Used to gate both the
      message MLP and the self-update MLP.
    * ``coords``: optional ``(n_atoms, 3)`` 3-D coordinates.  When
      provided, the layer additionally consumes the pairwise
      **distance** ``||x_i - x_j||`` (an SE(3)-invariant scalar) as
      a fourth input to the message MLP.  When ``None``, the
      distance feature is zero — the layer is still a valid
      coordination-aware MPNN but loses any geometric inductive
      bias.

    Returns ``(n_atoms, hidden_dim)`` updated features with a
    residual connection (``out = x + update_mlp(...)``).

    SE(3) invariance
    ----------------
    The returned scalar features are *invariant* under any rigid
    ``SE(3)`` transformation of the ``coords`` argument: rotating
    and/or translating ``coords`` does not change the output.  This
    is because the only coord-derived feature is the pairwise
    distance, which is rotation- and translation-invariant.  See
    ``test_coordination_gnn_equivariance`` in the test suite for
    the formal check.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM_DEFAULT,
        n_coordination_classes: int = N_COORDINATION_CLASSES_DEFAULT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_coordination_classes = n_coordination_classes

        # Message MLP input: [h_i || h_j || dist || coord_class_dst].
        # The destination's coordination class gates the message
        # computation — this is the "coordination-aware" inductive
        # bias that distinguishes this layer from a vanilla MPNN.
        msg_in_dim = 2 * hidden_dim + 1 + n_coordination_classes
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Self-update MLP input: [aggregated_msg || own_coord_class].
        # A residual connection (``+ x``) keeps the layer a faithful
        # perturbation of the identity — convenient for deep stacks.
        upd_in_dim = hidden_dim + n_coordination_classes
        self.update_mlp = nn.Sequential(
            nn.Linear(upd_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coordination: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one coordination-aware GNN layer.

        See the class docstring for arg / return semantics.
        """
        if x.dim() != 2:
            raise ValueError(
                f"x must be (n_atoms, hidden_dim); got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.hidden_dim:
            raise ValueError(
                f"x last dim must be hidden_dim={self.hidden_dim}; "
                f"got {x.size(-1)}"
            )
        if coordination.size(-1) != self.n_coordination_classes:
            raise ValueError(
                f"coordination last dim must be "
                f"n_coordination_classes={self.n_coordination_classes}; "
                f"got {coordination.size(-1)}"
            )

        n_nodes = x.size(0)
        n_edges = edge_index.size(1) if edge_index.numel() > 0 else 0

        if n_edges > 0:
            src = edge_index[0]
            dst = edge_index[1]
        else:
            # Empty graph: skip message aggregation entirely.
            # The update reduces to a coord-class-conditioned MLP
            # on the zero message, plus the residual connection.
            update_in = torch.cat(
                [x.new_zeros((n_nodes, self.hidden_dim)), coordination],
                dim=-1,
            )
            return x + self.update_mlp(update_in)

        # Pairwise distance — SE(3)-invariant scalar feature.
        if coords is not None:
            diff = coords[src] - coords[dst]
            dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            dist = x.new_zeros((n_edges, 1))

        # Build per-edge input.
        h_i = x[src]
        h_j = x[dst]
        coord_class_dst = coordination[dst]
        msg_in = torch.cat([h_i, h_j, dist, coord_class_dst], dim=-1)
        msg = self.msg_mlp(msg_in)  # (n_edges, hidden_dim)

        # Aggregate into destination nodes.
        agg = _scatter_sum(msg, dst, dim_size=n_nodes)  # (n_nodes, hidden_dim)

        # Self-update with residual.
        update_in = torch.cat([agg, coordination], dim=-1)
        out = x + self.update_mlp(update_in)
        return out


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class CoordinationNumberHead(nn.Module):
    """Per-metal coordination-number classifier.

    Given graph-level features of shape ``(batch, hidden_dim)``,
    outputs logits of shape ``(batch, n_metals, max_coordination)``.
    Each row ``(i, m, :)`` is the un-normalised log-probability that
    the ``m``-th metal centre in the ``i``-th graph has coordination
    number ``0 .. max_coordination-1`` (or, equivalently,
    ``1 .. max_coordination`` depending on indexing convention).

    Args:
        n_metals: number of metal centres to predict per graph.
            Defaults to 12.
        max_coordination: number of coordination-number classes per
            metal.  Defaults to 12 (covers [0, 12) or, equivalently,
            coordination numbers 1..12 under a +1 shift).
        hidden_dim: input feature dim.  Defaults to 128 to match
            ``CoordinationAwareGNNLayer``.
    """

    def __init__(
        self,
        n_metals: int = N_METALS_DEFAULT,
        max_coordination: int = MAX_COORDINATION_DEFAULT,
        hidden_dim: int = HIDDEN_DIM_DEFAULT,
    ) -> None:
        super().__init__()
        self.n_metals = n_metals
        self.max_coordination = max_coordination
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, n_metals * max_coordination)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to per-metal coordination-number logits.

        Args:
            features: ``(batch, hidden_dim)`` graph-level features.

        Returns:
            ``(batch, n_metals, max_coordination)`` logits.
        """
        if features.dim() != 2:
            raise ValueError(
                f"features must be (batch, hidden_dim); got {tuple(features.shape)}"
            )
        if features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"features last dim must be hidden_dim={self.hidden_dim}; "
                f"got {features.size(-1)}"
            )
        batch = features.size(0)
        out = self.proj(features)  # (batch, n_metals * max_coordination)
        return out.view(batch, self.n_metals, self.max_coordination)


class CoordinationGeometryHead(nn.Module):
    """Coordination-geometry classifier.

    Given graph-level features of shape ``(batch, hidden_dim)``,
    outputs logits of shape ``(batch, n_geometries)``.  The
    ``n_geometries`` rows cover the eight canonical coordination
    geometries we track: octahedral, tetrahedral, square-planar,
    linear, trigonal-planar, trigonal-bipyramidal, square-pyramidal,
    unknown.

    Args:
        n_geometries: number of geometry classes.  Defaults to 8.
        hidden_dim: input feature dim.  Defaults to 128 to match
            ``CoordinationAwareGNNLayer``.
    """

    def __init__(
        self,
        n_geometries: int = N_GEOMETRIES_DEFAULT,
        hidden_dim: int = HIDDEN_DIM_DEFAULT,
    ) -> None:
        super().__init__()
        self.n_geometries = n_geometries
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, n_geometries)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to geometry logits.

        Args:
            features: ``(batch, hidden_dim)`` graph-level features.

        Returns:
            ``(batch, n_geometries)`` logits.
        """
        if features.dim() != 2:
            raise ValueError(
                f"features must be (batch, hidden_dim); got {tuple(features.shape)}"
            )
        if features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"features last dim must be hidden_dim={self.hidden_dim}; "
                f"got {features.size(-1)}"
            )
        return self.proj(features)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class CoordinationAwareGNNComposite(nn.Module):
    """Layer + heads bundle for end-to-end coordination reasoning.

    Wraps a :class:`CoordinationAwareGNNLayer`, a
    :class:`CoordinationNumberHead`, and a
    :class:`CoordinationGeometryHead`.  The forward path is:

    1. Pool node features → graph feature (mean over atoms).
    2. Run the layer (if any per-node update is wanted — currently we
       expose ``run_layer=True`` for callers that want the layer in
       the forward path; default ``False`` keeps the heads usable
       even when the layer is decoupled).
    3. Project pooled features through the two heads.

    Args:
        layer: a :class:`CoordinationAwareGNNLayer` (used when
            ``run_layer=True``).
        num_head: a :class:`CoordinationNumberHead`.
        geom_head: a :class:`CoordinationGeometryHead`.
        run_layer: when ``True`` (default), the layer is applied to
            the node features before pooling; when ``False``, the
            heads receive the input ``x`` directly (after pooling).
    """

    def __init__(
        self,
        layer: CoordinationAwareGNNLayer,
        num_head: CoordinationNumberHead,
        geom_head: CoordinationGeometryHead,
        run_layer: bool = True,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_head = num_head
        self.geom_head = geom_head
        self.run_layer = run_layer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coordination: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(n_atoms, hidden_dim)`` node features.
            edge_index: ``(2, n_edges)`` graph connectivity.
            coordination: ``(n_atoms, n_coordination_classes)`` per-
                node coordination class.
            coords: optional ``(n_atoms, 3)`` coordinates — passed
                through to the layer for distance computation.
            batch: optional ``(n_atoms,)`` long tensor mapping each
                atom to its graph in the batch.  When ``None``, we
                treat the input as a single graph.

        Returns:
            ``(features, num_logits, geom_logits)``:

            * ``features``: ``(n_atoms, hidden_dim)`` node-level
              features (after the layer, if ``run_layer=True``;
              otherwise the input ``x``).
            * ``num_logits``: ``(batch, n_metals, max_coordination)``
              per-metal coordination-number logits.
            * ``geom_logits``: ``(batch, n_geometries)`` geometry
              logits.
        """
        # Apply the coordination-aware GNN layer (optional).
        if self.run_layer:
            features = self.layer(x, edge_index, coordination, coords=coords)
        else:
            features = x

        # Pool to graph-level features via batched mean.
        if batch is None:
            pooled = features.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        else:
            n_graphs = int(batch.max().item()) + 1
            pooled = features.new_zeros((n_graphs, features.size(-1)))
            counts = features.new_zeros((n_graphs,))
            pooled.index_add_(0, batch, features)
            counts.index_add_(0, batch, features.new_ones(features.size(0)))
            pooled = pooled / counts.clamp(min=1).unsqueeze(-1)

        # Heads.
        num_logits = self.num_head(pooled)
        geom_logits = self.geom_head(pooled)
        return features, num_logits, geom_logits


__all__ = [
    "CoordinationAwareGNNLayer",
    "CoordinationNumberHead",
    "CoordinationGeometryHead",
    "CoordinationAwareGNNComposite",
    "_scatter_sum",
    "N_COORDINATION_CLASSES_DEFAULT",
    "HIDDEN_DIM_DEFAULT",
    "N_METALS_DEFAULT",
    "MAX_COORDINATION_DEFAULT",
    "N_GEOMETRIES_DEFAULT",
]
