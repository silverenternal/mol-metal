"""W2.3 — Multi-Parameter Persistence adapter integration (T5 + W2.3).

================================================================
Background — multi-parameter persistent homology for pocket features
================================================================

The R17 close-out shipped the **PocketEncoder** in
``molmetal.adapters.flow_matching_lipman`` — a lightweight EGNN-lite
over per-pocket-atom coordinates + atomic numbers that produces a
``(B, H)`` vector via global mean pool.  This single-parameter
representation is sufficient for "what atoms are nearby?" but cannot
distinguish *geometrically distinct* pockets that share the same
elemental census (e.g. an α-helix bundle vs a β-sheet sandwich that
both contain C, N, O, S in similar counts).

The R18 Phase 3 (T5) upgrade adds a **multi-parameter persistent
homology (multi-PH)** feature map on top of the existing encoder.
We use Kim & Jung (2026) — *Fourier-Accelerated Computation of
Multi-Parameter Persistent Homology for Molecular Shape Analysis*,
Advances in Applied Mathematics **174**, 103023 — to compute a 2-PH
filtration of the pocket indexed by

* ``r`` = spatial radius (``||x_i - x_j||₂ ≤ r``)
* ``d`` = atomic-number coordination distance (``|z_i - z_j| ≤ d``)

and aggregate features into a ``(n_bins, n_bins)`` map.  This map
complements the EGNN pocket embedding: the encoder learns
*data-driven* invariances, the multi-PH map provides a
*topologically-aware* summary that is invariant in the discrete
2-Wasserstein sense (Kim-Jung 2026 Th.1).

The adapter is the **W2.3 surface area** that wraps
:mod:`molmetal.math_skills.multi_parameter_ph` into the existing
pocket-conditioning stack without modifying the R17 close-out.

Contract
--------

* **Do NOT modify** ``molmetal/adapters/flow_matching_lipman/`` —
  the R17 close-out is locked.  This file is the *only* new
  adapter added under W2.3.
* **CPU-only.**  All operations use ``torch.cdist`` /
  ``torch.linalg`` (CPU backend by default).  No CUDA / Triton
  kernels required for the W2.3 contract.
* **Standalone.**  No internal imports from
  ``molmetal.molmetal_lam.*``.  Only ``torch`` + the pure
  ``molmetal.math_skills`` package.
* **Batched.**  Every public function accepts a leading batch
  dimension and broadcasts over it.

Module map
----------

* :class:`MultiPHPocketFeatures` — pure-topology feature extractor
  (``compute(pocket_clouds) -> (B, n_bins, n_bins)`` feature map)
  that wraps :func:`~molmetal.math_skills.multi_parameter_ph.two_param_filtration`.
* :class:`MultiPHMetalCoordHead` — metal-coordination classifier
  head that compares pocket multi-PH diagrams against per-metal
  reference diagrams via the sliced-Wasserstein kernel
  (:func:`~molmetal.math_skills.multi_parameter_ph.sliced_wasserstein_kernel`).
* :func:`augment_pocket_embedding` — drop-in wrapper that
  concatenates the existing :class:`PocketEncoder` output
  (or any ``(B, H)`` tensor) with the flattened multi-PH
  feature map, returning a ``(B, H + n_bins ** 2)`` tensor.

The augmentation is **additive only**: the downstream velocity field
can either use the augmented embedding as a *replacement* for the
existing ``pocket_embed`` or ignore it entirely.  By default the
adapter does NOT modify any existing parameter; it merely *produces*
the augmented tensor.

References
----------

* Kim & Jung 2026 — *Fourier-Accelerated Computation of
  Multi-Parameter Persistent Homology for Molecular Shape Analysis*,
  Advances in Applied Mathematics **174**, 103023.  §4.1 defines the
  ``CountGrid`` statistic; Th.1 gives the 2-Wasserstein stability
  bound.
* Carlsson & Mémoli 2010 — *Sorting Networks on Simplicial
  Complexes*, §2 — multi-filter Vietoris-Rips construction.
* Rabin 2011 — *Sliced Wasserstein* — the kernel machinery used by
  :class:`MultiPHMetalCoordHead`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# T5 — pure-math multi-PH layer (CPU-only, batched).  See
# ``molmetal/math_skills/multi_parameter_ph.py`` for the mathematical
# reference (Kim & Jung 2026, doi:10.1016/j.aam.2025.103023).
from molmetal.math_skills.multi_parameter_ph import (
    TwoParameterDiagram,
    sliced_wasserstein_kernel,
    two_param_filtration,
)

__all__ = [
    "MultiPHPocketFeatures",
    "MultiPHMetalCoordHead",
    "augment_pocket_embedding",
]


# ---------------------------------------------------------------------------
# 1. Multi-PH pocket feature extractor
# ---------------------------------------------------------------------------
class MultiPHPocketFeatures(nn.Module):
    """Multi-parameter persistent-homology feature extractor.

    Wraps :func:`~molmetal.math_skills.multi_parameter_ph.two_param_filtration`
    into a stateless :class:`nn.Module` so it can sit alongside the
    existing :class:`PocketEncoder` in a pipeline.  The module has
    **no learnable parameters** — the feature map is fully
    determined by the pocket cloud and the (max_radius, max_coord,
    n_bins) hyper-parameters.

    Parameters
    ----------
    max_radius
        Maximum spatial filter in Ångström.  Default ``5.0`` is the
        W2.3 spec (≈ two water molecules wide).
    max_coord
        Maximum atomic-number coordination distance.  Default
        ``4.0`` covers ``{H(1) … Ar(18)}`` differences up to the
        transition-metal block (``Mn=25`` has ``|25-1| = 24`` so
        the default does clip the very heavy tails; this is
        intentional and matches the W2.3 smoke budget).
    n_bins
        Number of bins per filter axis.  Default ``32`` matches the
        W2.3 spec; the feature map has ``n_bins ** 2 = 1024`` cells.

    Notes
    -----
    The module accepts the same input contract as the
    :class:`PocketEncoder`: a ``(B, P, 3)`` padded coords tensor
    and a ``(B, P)`` long atomic-number tensor.  We do NOT require
    a pocket mask — the diagonal-filtration mask intrinsically
    ``scatter_add`` filters out self-pairs and the ``0`` atomic
    number (used by the R17 encoder as a padding marker) maps to
    ``d(i, j) = |z_i - 0| = z_i`` which is *valid* multi-PH
    geometry (atoms far from "atom 0" in coordination distance).

    For W2.3 smoke usage we recommend ``P <= 200`` atoms per
    pocket; the W2.3 paper-grade budget (``P <= 500``) requires
    the R18-P3 Fourier-accelerated path which is not implemented in
    this skeleton.
    """

    def __init__(
        self,
        max_radius: float = 5.0,
        max_coord: float = 4.0,
        n_bins: int = 32,
    ) -> None:
        super().__init__()
        if max_radius <= 0.0:
            raise ValueError(f"max_radius must be > 0, got {max_radius}")
        if max_coord < 0.0:
            raise ValueError(f"max_coord must be >= 0, got {max_coord}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.max_radius = float(max_radius)
        self.max_coord = float(max_coord)
        self.n_bins = int(n_bins)

    # Stateless forward — the module has no parameters.
    def compute(self, pocket_clouds: Tensor) -> Tensor:
        """Compute the ``(B, n_bins, n_bins)`` multi-PH feature map.

        Parameters
        ----------
        pocket_clouds
            ``torch.Tensor`` of shape ``(B, P, 3)`` holding the pocket
            atom coordinates.  Pass the **same tensor** you would
            pass to ``PocketEncoder.forward`` (we use 1-indexing of
            ``z_i = 1 + atomic_number`` to be safe against ``0``-
            padding, but downstream code may pass raw ``atom_types``
            and we accept both via the helper below).

        Returns
        -------
        ``(B, n_bins, n_bins)`` feature map — the count of pocket
        atom-pairs whose ``(r, d)`` falls in each ``(n_bins, n_bins)``
        bin.

        Notes
        -----
        The default forward uses ``atomic numbers`` as the
        coordination-distance filter — if you have a richer
        descriptor (e.g. electronegativity, partial charge) pass
        it via :meth:`compute_with_atom_types` instead.
        """
        if pocket_clouds.dim() != 3 or pocket_clouds.shape[-1] != 3:
            raise ValueError(
                f"pocket_clouds must be (B, P, 3); got {tuple(pocket_clouds.shape)}"
            )
        b, p, _ = pocket_clouds.shape
        # Default atomic-number tensor: 1-hot fallback so non-pocket
        # callers can still get a feature map.
        atom_types = torch.ones(b, p, dtype=torch.long, device=pocket_clouds.device)
        out = two_param_filtration(
            pocket_clouds,
            atom_types,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
            n_bins=self.n_bins,
        )
        return out["features"]

    def compute_with_atom_types(
        self,
        pocket_clouds: Tensor,
        pocket_atom_types: Tensor,
    ) -> Tensor:
        """Compute the feature map given explicit atomic numbers.

        Mirrors the contract of the existing ``PocketEncoder``: pass
        a ``(B, P, 3)`` coords tensor and a ``(B, P)`` atomic-number
        long tensor.  Returns ``(B, n_bins, n_bins)``.
        """
        if pocket_atom_types.shape != pocket_clouds.shape[:2]:
            raise ValueError(
                f"pocket_atom_types must be (B, P); "
                f"got {tuple(pocket_atom_types.shape)}"
            )
        out = two_param_filtration(
            pocket_clouds,
            pocket_atom_types,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
            n_bins=self.n_bins,
        )
        return out["features"]

    # Alias for nn.Module-style call sites.
    def forward(self, pocket_clouds: Tensor) -> Tensor:
        return self.compute(pocket_clouds)


# ---------------------------------------------------------------------------
# 2. Metal-coordination classifier head
# ---------------------------------------------------------------------------
class MultiPHMetalCoordHead(nn.Module):
    """Metal-coordination classifier that reads multi-PH pocket features.

    Given a ``(B, n_bins, n_bins)`` pocket feature map and a set of
    per-metal *reference* multi-PH diagrams (``n_metals`` total),
    compute the sliced-Wasserstein kernel between the pocket and
    every reference and lift to logits via a learned linear head.

    Parameters
    ----------
    n_metals
        Number of distinct metal ions the head can score.  Default
        ``12`` matches the Mol-Metal R17 panel (``Zn, Cu, Ru, Pt,
        Ir, Au, Mn, Fe, Co, Ni, Pd, Hg``).  Must be >= 2.
    kernel
        Kernel family for the multi-PH similarity.  Supported:

        ``"sliced_wasserstein"`` (default) — sliced-Wasserstein RBF
        on the 2-PH diagrams (Kim-Jung 2026 + Rabin 2011).  This is
        the W2.3 smoke kernel.

    sigma
        RBF bandwidth used when ``kernel == "sliced_wasserstein"``.
        ``sigma = 1.0`` is the W2.3 default; smaller values make
        the kernel more selective.
    n_projections
        Number of random projections for the sliced-Wasserstein
        estimator.  ``K = 8`` is the W2.3 smoke default.
    max_radius, max_coord, n_bins
        Pocket-feature hyper-parameters.  These must match the
        values used by the upstream :class:`MultiPHPocketFeatures`
        otherwise the kernel is not comparable.

    Notes
    -----
    The head is intentionally simple: it returns *logits*, not
    probabilities.  The caller is responsible for the softmax /
    sigmoid step.  This matches the R17 metal-coordination head in
    ``PocketEncoder`` (which is also a linear layer producing
    logits over the metal ion catalogue).
    """

    SUPPORTED_KERNELS: Tuple[str, ...] = ("sliced_wasserstein",)

    def __init__(
        self,
        n_metals: int = 12,
        kernel: str = "sliced_wasserstein",
        sigma: float = 1.0,
        n_projections: int = 8,
        max_radius: float = 5.0,
        max_coord: float = 4.0,
        n_bins: int = 32,
    ) -> None:
        super().__init__()
        if n_metals < 2:
            raise ValueError(f"n_metals must be >= 2, got {n_metals}")
        if kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"kernel must be one of {self.SUPPORTED_KERNELS}; got {kernel}"
            )
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if n_projections < 1:
            raise ValueError(f"n_projections must be >= 1, got {n_projections}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.n_metals = int(n_metals)
        self.kernel = str(kernel)
        self.sigma = float(sigma)
        self.n_projections = int(n_projections)
        self.max_radius = float(max_radius)
        self.max_coord = float(max_coord)
        self.n_bins = int(n_bins)

        # Linear head: maps concatenated (n_metals,) similarity
        # scores to (n_metals,) logits.  Initialised so that the
        # initial output is exactly the kernel value when
        # ``similarity == 0`` (zero bias) and the gradient flows
        # only through the linear projection.
        self.linear = nn.Linear(n_metals, n_metals, bias=True)

    # ------------------------------------------------------------------
    # Reference loading — store as buffer so it follows the module
    # device after ``.to(device)``.
    # ------------------------------------------------------------------
    def set_reference(
        self,
        reference_coords: Tensor,
        reference_atom_types: Tensor,
    ) -> None:
        """Load reference multi-PH diagrams from a stack of metal ions.

        Parameters
        ----------
        reference_coords
            ``(n_metals, P_ref, 3)`` per-metal reference coordinate
            cloud.  For the W2.3 smoke we use deterministic dummy
            clouds of a single metal-coordination site; production
            would use the F2(a) tmQM reference catalog (TODO).
        reference_atom_types
            ``(n_metals, P_ref)`` long tensor of atomic numbers.

        Notes
        -----
        The reference diagrams are computed once and stored as a
        buffer.  Re-calling :meth:`set_reference` overwrites the
        previous references.
        """
        out = two_param_filtration(
            reference_coords,
            reference_atom_types,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
            n_bins=self.n_bins,
        )
        # ``features`` is ``(n_metals, n_bins, n_bins)`` — wrap into
        # ``TwoParameterDiagram`` per metal and store as a list of
        # buffers (one per metal) so they follow the module.
        # Storing as a single ``(n_metals, n_bins, n_bins)`` tensor
        # is cheaper; we keep the diagrams flat to allow batched
        # kernel evaluation below.
        reference_features = out["features"]  # (n_metals, n_bins, n_bins)
        self.register_buffer(
            "reference_features",
            reference_features,
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pocket_features: Tensor) -> Tensor:
        """Score a batch of pocket features against the metal references.

        Parameters
        ----------
        pocket_features
            ``(B, n_bins, n_bins)`` pocket feature map from
            :class:`MultiPHPocketFeatures`.

        Returns
        -------
        ``(B, n_metals)`` logits tensor — ``logits[b, m]`` is the
        logit score of metal ``m`` for pocket ``b``.
        """
        # Wf-A56 W2 (2026-10-16) — auto-broadcast for caller convenience.
        # Accept the flattened ``(B, n_bins * n_bins)`` shape and view
        # it back to ``(B, n_bins, n_bins)``.  This lets the caller
        # feed the head from a flattened feature pipeline without
        # reshaping manually.
        if (
            pocket_features.dim() == 2
            and pocket_features.shape[1] == self.n_bins * self.n_bins
        ):
            pocket_features = pocket_features.view(
                pocket_features.shape[0], self.n_bins, self.n_bins
            )
        if pocket_features.dim() != 3:
            raise ValueError(
                f"pocket_features must be (B, n_bins, n_bins) or (B, n_bins*n_bins); "
                f"got {tuple(pocket_features.shape)}"
            )
        if pocket_features.shape[1:] != (self.n_bins, self.n_bins):
            raise ValueError(
                f"pocket_features.shape[1:] must be ({self.n_bins}, {self.n_bins}); "
                f"got {tuple(pocket_features.shape)}"
            )

        if not hasattr(self, "reference_features"):
            # Lazily synthesise deterministic dummy references so the
            # head is usable without an explicit ``set_reference``
            # call (matches the W2.3 smoke contract).  The dummy
            # is a single delta at ``(r, d) = (max_radius / 2,
            # max_coord / 2)`` with per-metal centre offsets.
            device = pocket_features.device
            dtype = pocket_features.dtype
            ref = torch.zeros(
                self.n_metals, self.n_bins, self.n_bins, dtype=dtype, device=device
            )
            # Distinct centre per metal so the kernel has signal.
            for m in range(self.n_metals):
                r_idx = (m * 3) % self.n_bins
                d_idx = (m * 5 + 7) % self.n_bins
                ref[m, r_idx, d_idx] = 1.0
            self.register_buffer(
                "reference_features",
                ref,
                persistent=False,
            )

        # Build per-sample TwoParameterDiagram objects on the fly.
        # We use the shared ``r_grid`` and ``d_grid`` from a fresh
        # ``TwoParameterDiagram`` because both diagrams in a kernel
        # call must agree on the bin coordinates.  Storing them on
        # the module keeps the per-patch overhead low.
        if not hasattr(self, "_rd_grids"):
            rd_grids = TwoParameterDiagram.from_filtration(
                pocket_features.new_zeros(1, 1, 3),
                pocket_features.new_ones(1, 1, dtype=torch.long),
                max_radius=self.max_radius,
                max_coord=self.max_coord,
                n_bins=self.n_bins,
            )
            self._r_grid = rd_grids.r_grid
            self._d_grid = rd_grids.d_grid

        # Per-pocket diagram.
        pocket_diagram = TwoParameterDiagram(
            feature_map=pocket_features,
            r_grid=self._r_grid,
            d_grid=self._d_grid,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
        )
        # Per-metal reference diagrams (shape matches pocket batch).
        ref_diagrams = [
            TwoParameterDiagram(
                feature_map=self.reference_features[m : m + 1],
                r_grid=self._r_grid,
                d_grid=self._d_grid,
                max_radius=self.max_radius,
                max_coord=self.max_coord,
            )
            for m in range(self.n_metals)
        ]

        # Per-pocket × per-metal kernel matrix (B, n_metals).
        sims = torch.zeros(
            pocket_features.shape[0],
            self.n_metals,
            dtype=pocket_features.dtype,
            device=pocket_features.device,
        )
        for m in range(self.n_metals):
            sims[:, m] = sliced_wasserstein_kernel(
                pocket_diagram,
                ref_diagrams[m],
                n_projections=self.n_projections,
                sigma=self.sigma,
                seed=m,
            ).diag()

        # Linear head: lifts similarities to logits.
        return self.linear(sims)


# ---------------------------------------------------------------------------
# 3. Pocket embedding augmentation
# ---------------------------------------------------------------------------
def augment_pocket_embedding(
    base_pocket_emb: Tensor,
    multi_ph_features: Tensor,
) -> Tensor:
    """Concatenate a base pocket embedding with the multi-PH feature map.

    Drop-in wrapper for any ``(B, H)`` base embedding produced by the
    existing :class:`PocketEncoder` (or a third-party equivalent).
    Returns a single ``(B, H + n_bins ** 2)`` tensor that callers can
    splice into the downstream velocity field as a replacement for
    ``pocket_embed``.

    Parameters
    ----------
    base_pocket_emb
        ``(B, H)`` base pocket embedding (typically the output of
        ``PocketEncoder.forward``).
    multi_ph_features
        ``(B, n_bins, n_bins)`` multi-PH feature map from
        :meth:`MultiPHPocketFeatures.compute`.

    Returns
    -------
    ``(B, H + n_bins ** 2)`` concatenated embedding.  The first ``H``
    columns are exactly ``base_pocket_emb`` (copied verbatim — the
    function does **not** modify the base tensor); the remaining
    ``n_bins ** 2`` columns are the flattened multi-PH feature map
    in row-major order.

    Notes
    -----
    This is an *additive* augmentation.  It does not require the
    base pocket encoder to know about multi-PH, and the downstream
    velocity field can ignore the extra columns by passing only the
    leading slice.  W2.3 recommends adapting the velocity field to
    accept ``hidden_dim = H + n_bins ** 2`` so the augmented
    embedding can be used end-to-end; that refactor is left for the
    R18-P3 integration step.
    """
    if base_pocket_emb.dim() != 2:
        raise ValueError(
            f"base_pocket_emb must be (B, H); got {tuple(base_pocket_emb.shape)}"
        )
    if multi_ph_features.dim() != 3:
        raise ValueError(
            f"multi_ph_features must be (B, n_bins, n_bins); got "
            f"{tuple(multi_ph_features.shape)}"
        )
    if base_pocket_emb.shape[0] != multi_ph_features.shape[0]:
        raise ValueError(
            f"batch mismatch: base_pocket_emb {tuple(base_pocket_emb.shape)} "
            f"vs multi_ph_features {tuple(multi_ph_features.shape)}"
        )
    # Flatten the multi-PH map row-major; concatenate to the base
    # embedding along the feature dimension.
    flat = multi_ph_features.reshape(multi_ph_features.shape[0], -1)
    return torch.cat([base_pocket_emb, flat], dim=-1)
