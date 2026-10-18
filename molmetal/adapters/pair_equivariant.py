"""Allegro-style pairwise equivariant latent layer — SCAFFOLD ONLY.

SCAFFOLD — naive Linear baseline; future wave (Wf-A77+) will swap in
e3nn.o3.EquivariantTensorProduct.  Do NOT use as the production path.

Per Musaelian et al. 2023 ("Allegro: Equivariant Geometry-Enhanced
Deep Generative Model", Nat. Commun., DOI 10.1038/s41467-023-36714-5,
arXiv:2204.09490), the central architectural choice in Allegro is a
**per-pair** latent (vs EGNN's per-atom): each edge ``(i, j)`` carries
its own hidden state ``h_e`` that is updated by messages on the line
graph ``L(G)``, not on the original atom graph.  The motivation is the
same one Chemprop D-MPNN cites for **directed** pre-activation
messages: per-pair state keeps the directional ``(i, j)`` signal
through every layer, which per-atom pooling would conflate.

This module ships the SCAFFOLD:

* the **L=0 (invariant) path is LIVE and unit-tested** — the scalar
  per-edge update, which is what survives the e3nn swap;
* the **L≥1 (equivariant) path is a ``NotImplementedError`` stub**
  gated on ``use_equivariant=True``, with a TODO marker naming e3nn.

The naive-Linear baseline that an earlier draft considered (raw
direction via ``Linear(3, H)``) is NOT rotation-equivariant and was
rejected per the Wf-A72 plan Section 3: shipping a non-equivariant
equivariant-named layer would commit the codebase to dead code that
the future wire-in has to delete.  A pure-PyTorch spherical-harmonics
``Y_1(d̂)`` stub was also rejected: it is mathematically correct
(``Y_1 = d̂`` for real spherical harmonics) but 3-dim, gives no
tensor-product structure to test, and is itself dead code on the way
to e3nn.  The honest SCAFFOLD therefore leaves the L=0 path live and
the L≥1 path stubbed.

Design constraints (mirroring ``molmetal/adapters/coordination_gnn.py``,
the canonical CPU-skeleton template):

* **CPU-only** — no ``triton_kernels``, no torch_scatter, no
  ROCm-specific paths.  The goal is a portable, dependency-light
  building block.
* **Does NOT modify** any existing module.  ``YuelBondEdgeMPNNHead``
  consumes a separate ``edge_init`` + ``edge_mp`` stack; this
  primitive is independently callable.
* **SCAFFOLD contract** — the L=0 output is ``(E, H)`` and is
  SE(3)-invariant under rigid transformations of ``edge_directions``
  and ``edge_distances``.  The L≥1 output is a stub that raises
  ``NotImplementedError`` when ``use_equivariant=True`` is requested.

API
---
``AllegroPairwiseLayer(hidden_dim, edge_feat_dim, use_equivariant)``
    Per-pair (Allegro-style) latent layer.  ``forward(edge_index,
    edge_features, edge_distances, edge_directions) -> (h_e_scalar,
    h_e_equivariant_or_none)``.

``HIDDEN_DIM_DEFAULT``
    Default ``hidden_dim``.  Matches the project-wide Wf-A70 W3 / Wf-A75
    ``hidden_dim=128`` baseline.

``EDGE_FEAT_DIM_DEFAULT``
    Default ``edge_feat_dim``.  Matches the 4-dim per-edge feature
    vector built by ``YuelBondEdgeMPNNHead`` (``is_short``, ``is_long``,
    ``is_dative``, plus distance — the SCAFFOLD consumes only the
    pre-built feature tensor; callers build the flags themselves).

Tests live in ``molmetal/tests/test_pair_equivariant.py`` (5 tests,
all CPU-only).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

#: Default feature dimensionality for the per-pair hidden state.
#: Matches the project-wide Wf-A70 W3 / Wf-A75 ``hidden_dim=128``
#: baseline; the per-pair latent is the same width as the per-atom
#: hidden state in EGNN so that a future wire-in into
#: :class:`YuelBondEdgeMPNNHead` is a no-op for the head's
#: downstream code.
HIDDEN_DIM_DEFAULT: int = 128

#: Default dimensionality of the pre-built per-edge feature vector.
#: Matches the 4-dim per-edge feature vector built by
#: ``YuelBondEdgeMPNNHead.forward`` (yuelbond_decoder.py:1035-1043):
#: ``(d, is_short, is_long, is_dative)`` — the SCAFFOLD consumes
#: only the pre-built feature tensor; callers (YuelBond head or
#: tests) build the flags themselves using the same 1.7/3.0 Å
#: thresholds and N/O/P/S donor mask as the head.
EDGE_FEAT_DIM_DEFAULT: int = 4


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class AllegroPairwiseLayer(nn.Module):
    """Allegro-style per-pair latent layer — SCAFFOLD (naive Linear baseline).

    SCAFFOLD — naive Linear baseline; future wave (Wf-A77+) will swap
    in e3nn.o3.EquivariantTensorProduct.  Do NOT use as the production
    path.  See the module docstring for the deferred work queue.

    Per-pair (Allegro) vs per-atom (EGNN): this layer updates an
    edge-level hidden state ``h_e`` for each edge ``(i, j)``, NOT a
    per-atom state.  The message function is therefore over EDGE
    neighbourhoods on the line graph ``L(G)`` (edges sharing a
    vertex), not over atom neighbourhoods.

    Args:
        hidden_dim: feature dimensionality ``H``.  Defaults to
            :data:`HIDDEN_DIM_DEFAULT` (128).
        edge_feat_dim: dimensionality of the pre-built per-edge
            feature vector.  Defaults to :data:`EDGE_FEAT_DIM_DEFAULT`
            (4) — matches ``YuelBondEdgeMPNNHead``'s
            ``(d, is_short, is_long, is_dative)`` schema.  The
            SCAFFOLD does NOT compute these flags; the caller builds
            them (the head, the unit tests, or a future wire-in).
        use_equivariant: must be ``False`` for this SCAFFOLD.  Setting
            ``True`` raises ``NotImplementedError`` pointing at the
            e3nn wire-in TODO marker.  The flag exists so a future
            contributor who flips it cannot accidentally ship a dead
            path; the SCAFFOLD contract test
            ``test_layer_use_equivariant_true_raises_not_implemented``
            in :mod:`molmetal.tests.test_pair_equivariant` is the
            load-bearing check.

    Forward
    -------
    ``forward(edge_index, edge_features, edge_distances,
    edge_directions) -> (h_e_scalar, h_e_equivariant_or_none)``

    * ``edge_index``: ``(2, E)`` long tensor with the conventional
      ``(src, dst)`` orientation.  Each edge contributes one row.
      Currently the SCAFFOLD does NOT actually use ``edge_index``
      for the L=0 path (no message passing on the line graph) — the
      forward signature accepts it for future-compat with the wire-in.
    * ``edge_features``: ``(E, edge_feat_dim)`` float tensor.  The
      pre-built per-edge feature vector (``d``, ``is_short``,
      ``is_long``, ``is_dative``).
    * ``edge_distances``: ``(E,)`` float tensor of pairwise distances
      in Å.  Consumed by the L=0 path (it is rotation- and
      translation-invariant by construction).
    * ``edge_directions``: ``(E, 3)`` float tensor of unit direction
      vectors.  Consumed ONLY when ``use_equivariant=True`` (which
      raises); the L=0 path accepts but ignores it.  The signature
      keeps the parameter so a future wire-in can compute the
      equivariant features without re-shaping callers.

    Returns
    -------
    ``(h_e_scalar, h_e_equivariant)`` where ``h_e_scalar`` is
    ``(E, hidden_dim)`` (the live L=0 path) and ``h_e_equivariant``
    is ``None`` when ``use_equivariant=False`` (the SCAFFOLD default).
    When ``use_equivariant=True`` is requested, the forward raises
    ``NotImplementedError`` immediately (before any compute), so the
    returned tuple is never materialised.

    SE(3) invariance
    ----------------
    The L=0 ``h_e_scalar`` output is **invariant** under any rigid
    ``SE(3)`` transformation of the input geometry: rotating and/or
    translating ``edge_directions`` and ``edge_distances`` does NOT
    change the output, because the layer consumes only the invariant
    scalars ``edge_distances`` + ``edge_features`` (the latter being
    pre-built by the caller from invariant flags).  The formal check
    lives in ``test_layer_scalar_path_se3_invariant`` in
    :mod:`molmetal.tests.test_pair_equivariant`.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM_DEFAULT,
        edge_feat_dim: int = EDGE_FEAT_DIM_DEFAULT,
        use_equivariant: bool = False,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(
                f"hidden_dim must be a positive int; got {hidden_dim!r}"
            )
        if int(edge_feat_dim) <= 0:
            raise ValueError(
                f"edge_feat_dim must be a positive int; got {edge_feat_dim!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.use_equivariant = bool(use_equivariant)
        # L=0 (invariant) path: project the 4-dim per-edge feature
        # vector up to hidden_dim, then a 2-layer MLP with ReLU.  The
        # SCAFFOLD's equivariant path is the NotImplementedError stub
        # below; the L=0 path is what survives the future e3nn
        # wire-in.
        #
        # We keep a residual connection around the inner MLP so the
        # layer can recover the identity at init (and so the gradient
        # signal is non-vanishing when the future wire-in replaces
        # this with a tensor product).  The projection Linear
        # (``edge_proj``) maps edge_feat_dim -> hidden_dim; the MLP
        # reads/writes hidden_dim; the residual adds ``edge_proj(x)``
        # to the MLP output.
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # TODO (Wf-A77+, deferred): replace the L=0 path's
        # ``edge_proj`` + ``edge_mlp`` with an e3nn ``Irreps``
        # decomposition: ``Irreps("{H}x0e + Hx1o + Hx2e")`` for
        # scalars + vectors + l=2 tensors, an
        # ``e3nn.o3.EquivariantTensorProduct`` for the message
        # function, and a learnable per-pair ``h_e`` initialised
        # from a small MLP.  See the Wf-A77 catalog entry
        # (HONEST_NEGATIVE_CATALOG.md §6.104) for the deferred
        # work queue.
        #
        # The ``equivariant_proj`` Linear below is intentionally a
        # placeholder so that state_dict serialisation is
        # well-defined when ``use_equivariant=True`` is later wired
        # in; the current SCAFFOLD never enters this branch.
        if self.use_equivariant:
            # SCAFFOLD contract: the forward raises BEFORE this
            # placeholder is reached (see ``forward`` below).  This
            # Linear exists only so the module can be instantiated
            # with ``use_equivariant=True`` for the
            # ``test_layer_use_equivariant_true_raises_not_implemented``
            # test — that test calls ``__init__`` then ``forward``,
            # and the forward is the load-bearing check.
            self.equivariant_proj: Optional[nn.Linear] = nn.Linear(3, hidden_dim)
        else:
            self.equivariant_proj = None

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_distances: torch.Tensor,
        edge_directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(h_e_scalar, h_e_equivariant_or_none)``.

        See the class docstring for the per-arg shape contract.
        """
        if self.use_equivariant:
            raise NotImplementedError(
                "Wf-A77 SCAFFOLD: the equivariant path is not "
                "implemented.  The L=0 (invariant) path is live; "
                "the L≥1 (equivariant) path will be wired in by the "
                "future Wf-A77+ e3nn.o3.EquivariantTensorProduct "
                "wire-in.  See the module docstring + "
                "HONEST_NEGATIVE_CATALOG.md §6.104 + the TODO marker "
                "in __init__ for the deferred work queue."
            )
        # Shape validation: every input must be a 2-D / 1-D tensor
        # with the right leading dim.  Empty-input short-circuit
        # matches the convention in coordination_gnn.py:106-108.
        if edge_features.dim() != 2:
            raise ValueError(
                f"edge_features must be (E, edge_feat_dim); got "
                f"shape {tuple(edge_features.shape)}"
            )
        if edge_distances.dim() != 1:
            raise ValueError(
                f"edge_distances must be (E,); got shape "
                f"{tuple(edge_distances.shape)}"
            )
        if edge_directions.dim() != 2:
            raise ValueError(
                f"edge_directions must be (E, 3); got shape "
                f"{tuple(edge_directions.shape)}"
            )
        e = edge_features.shape[0]
        if edge_distances.shape[0] != e:
            raise ValueError(
                f"edge_distances leading dim must match edge_features "
                f"(E={e}); got shape {tuple(edge_distances.shape)}"
            )
        if edge_directions.shape[0] != e:
            raise ValueError(
                f"edge_directions leading dim must match edge_features "
                f"(E={e}); got shape {tuple(edge_directions.shape)}"
            )
        if edge_features.shape[1] != self.edge_feat_dim:
            raise ValueError(
                f"edge_features last dim must equal edge_feat_dim="
                f"{self.edge_feat_dim}; got {edge_features.shape[1]}"
            )
        if edge_directions.shape[1] != 3:
            raise ValueError(
                f"edge_directions last dim must be 3 (unit vectors); "
                f"got {edge_directions.shape[1]}"
            )
        # TODO (Wf-A77+, deferred): validate edge_index shape
        # (2, E) here too, once the wire-in actually consumes it
        # (the SCAFFOLD's L=0 path does not).
        if e == 0:
            # Empty-input short-circuit.  Matches the convention in
            # coordination_gnn.py:106-108 and YuelBondEdgeMPNNHead
            # (yuelbond_decoder.py:471-494).
            empty = edge_features.new_zeros((0, self.hidden_dim))
            return empty, None
        # L=0 (invariant) path: project edge_features -> hidden_dim,
        # apply the MLP, residual-add the projection.  The
        # ``edge_directions`` argument is accepted but ignored
        # (the SCAFFOLD contract).
        proj = self.edge_proj(edge_features)  # (E, H)
        update = self.edge_mlp(proj)          # (E, H)
        h_e_scalar = proj + update            # (E, H)
        return h_e_scalar, None

    def _tensor_product(
        self,
        scalar_features: torch.Tensor,
        equivariant_features: torch.Tensor,
    ) -> torch.Tensor:
        """SCAFFOLD: stub for the future L=0 ⊗ L≥1 → L≥1 tensor product.

        TODO (Wf-A77+, deferred): replace this stub with
        ``e3nn.o3.EquivariantTensorProduct`` once e3nn is on the
        dependency list.  The replacement must satisfy:

        * **SE(3)-equivariance** of the output: rotating the input
          directions rotates the output features by the same ``R``;
        * **Wigner-Eckart selection rules** (``L_out ∈ {|L1-L2|,
          ..., L1+L2}``) — these are what e3nn enforces
          automatically;
        * **HIDDEN_DIM_DEFAULT-equal channel count** on the
          L=0 and L≥1 irreps so the future
          ``YuelBondEdgeMPNNHead`` wire-in is a no-op for the
          head's downstream code.

        This stub is intentionally a function (not a method of the
        layer state) so a future wire-in can replace it with a
        learned ``nn.Module`` without breaking callers.
        """
        raise NotImplementedError(
            "Wf-A77 SCAFFOLD: _tensor_product is a stub.  See the "
            "module docstring + HONEST_NEGATIVE_CATALOG.md §6.104 "
            "+ the TODO in __init__ for the deferred work queue."
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "AllegroPairwiseLayer",
    "HIDDEN_DIM_DEFAULT",
    "EDGE_FEAT_DIM_DEFAULT",
]
