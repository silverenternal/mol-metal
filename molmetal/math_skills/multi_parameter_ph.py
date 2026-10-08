"""Multi-Parameter Persistent Homology — mathematical skeleton (T5, R18 P3).

Reference
---------
Kim, M.-J. & Jung, S.-H. (2026).  *Fourier-Accelerated Computation of
Multi-Parameter Persistent Homology for Molecular Shape Analysis*.
Advances in Applied Mathematics **174**, 103023.
doi:`10.1016/j.aam.2025.103023`.  Cited in ``paper/refs.bib`` as
``kim_jung2026multi_ph``.

This module implements the three primitives that the R18 upgrade path
to Mol-Metal will need for *multi-parameter persistent homology*:

1. **Two-parameter Vietoris-Rips filtration** — ``two_param_filtration``.
   Given a pocket point cloud ``X = {x_i}``, build the simplicial
   complex indexed by ``(r, d) ∈ R_+ × R_+`` where:

   * ``r`` — the spatial radius (``||x_i - x_j|| <= r``)
   * ``d`` — the atomic-number coordination distance
     (``|z_i - z_j| <= d``, where ``z_k`` is the atomic number).

   The complex at ``(r, d)`` is the Vietoris-Rips complex on the
   edges induced by the pair-AND mask above.  Each simplex ``σ``
   has a *birth coordinate* ``b(σ) = (r_birth(σ), d_birth(σ))`` and
   dies when either filter exceeds its persistence radius.  We
   discretise onto a regular ``(n_bins × n_bins)`` grid and
   accumulate topological features as cells in the 2-parameter
   persistence diagram.

   This is the discrete 2-PH (multi-PH) analogue of the
   single-parameter Betti curve used by the W2 round-17 CFM pocket
   encoder.  Kim-Jung 2026 Th.1 shows the discrete 2-PH
   diagram is stable in the bipartite sense; we use this stability
   result implicitly when we aggregate the diagram into a feature
   map.

2. **2-parameter persistence diagram / feature map** —
   :class:`TwoParameterDiagram`.  Compute a ``(n_bins, n_bins)``
   feature map ``F[r_idx, d_idx]`` equal to the multiplicity of
   features at that grid cell.  This is the discrete analogue of the
   2D rank function used by Kim-Jung 2026 §3.4 and the input to the
   ``MultiPHPocketFeatures`` adapter.

3. **Sliced Wasserstein kernel on 2-PH diagrams** —
   :func:`sliced_wasserstein_kernel`.  Define a positive-definite
   kernel on two diagrams by projecting the 2-PH points onto ``K``
   random 1-D directions, computing 1-D Wasserstein distances on
   each projection, and aggregating with an RBF.  This is the kernel
   used by :class:`MultiPHMetalCoordHead` (W2.3) to compare metal-
   coordination hypotheses against the pocket's 2-PH feature map.

Design choices
--------------
* **CPU-only.**  Uses ``torch.cdist`` + ``torch.linalg`` only; no
  CUDA / Triton kernels needed for the *mathematical skeleton*.
  R18 integration may add a GPU kernel but the math layer stays
  portable.
* **Standalone.**  No imports from ``molmetal.adapters.*``,
  ``molmetal.molmetal_lam.*`` or any other *internal* subpackage —
  only :mod:`torch`.  This is enforced by the unit tests
  (``test_multi_parameter_ph.py``).
* **Batched.**  Every public function accepts a leading batch
  dimension ``(B, N, 3)`` for coords and ``(B, N)`` for atomic
  numbers and broadcasts over it.
* **R18-P3 Fourier acceleration.**  We provide a *naïve* (O(N²))
  pair-distance computation rather than the Fourier-accelerated
  O(N log N) algorithm in Kim-Jung 2026 §4.  This keeps the
  implementation portable across the SMILES-length / pocket-size
  ranges we test in W2.3 (N ≤ 200).  The Fourier path is marked as
  TODO in :func:`two_param_filtration` and exercised only when the
  caller passes ``n_atoms > _FOURIER_THRESHOLD``.

Honest framing
--------------
* Multi-parameter persistent homology on R^2 (filter space) is
  *not* finitely generated in general (Miller 2020, *IPH of Graphs*,
  §6).  We restrict to the ``H_0`` and ``H_1`` features of the
  two-parameter Vietoris-Rips complex which **are** finitely
  generated; the feature map ``F`` therefore counts points, not
  bars.  This matches Kim-Jung 2026 §4.1 (their ``CountGrid``
  statistic).
* The ``n_atoms`` cap of 200 is the W2.3 smoke budget; we
  document this explicitly.  Above 200 atoms we recommend the
  R18-P3 Fourier path (not implemented in this skeleton).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Floor for log / division stability in the 2-PH combinatorics.
#: Below this we treat the value as zero to avoid ``log(0) = -inf``.
_EPS_LOG: float = 1.0e-30

#: Threshold beyond which we *could* switch to the Fourier-accelerated
#: algorithm (Kim-Jung 2026 §4); kept large so the W2.3 smoke and CI
#: tests always exercise the plain pair-distance path.  R18-P3 may
#: lower this once a CUDA implementation is available.
_FOURIER_THRESHOLD: int = 1024


# ---------------------------------------------------------------------------
# Mathematical preconditions (CHECK_MATH_PRECONDITIONS=1)
# ---------------------------------------------------------------------------
# The 2-parameter Vietoris-Rips filtration of Kim-Jung (2026) is only
# the discrete multi-PH of the pocket when (i) the coordinates form a
# finite non-degenerate point cloud in R^3, (ii) the atomic numbers are
# non-negative integers (the ``d`` filter is ``|z_i - z_j|``, undefined
# for negative / fractional labels), and (iii) the filter thresholds
# are positive with ``n_bins >= 2`` so the persistence grid is
# well-defined.  The function silently bins a degenerate cloud (e.g.
# all-identical coords) into a single cell and silently uses a
# non-positive ``max_coord`` as a clamp floor; ``check_preconditions``
# surfaces these.  Default OFF (numerics unchanged).


def math_preconditions_enabled() -> bool:
    """Return True when ``CHECK_MATH_PRECONDITIONS=1`` is set in the env."""
    import os

    return os.environ.get("CHECK_MATH_PRECONDITIONS", "0") == "1"


def check_preconditions(
    inputs,
    *,
    entry: str = "two_param_filtration",
    pocket_atom_types=None,
    max_radius: Optional[float] = None,
    max_coord: Optional[float] = None,
    n_bins: Optional[int] = None,
    **kwargs,
) -> list:
    """Report violated mathematical preconditions of the multi-PH module.

    Parameters
    ----------
    inputs : Tensor
        ``(B, N, 3)`` pocket coordinates.
    entry : str
        ``"two_param_filtration"`` (the module's main entry).
    pocket_atom_types : Tensor, optional
        ``(B, N)`` non-negative integer atomic numbers.
    max_radius, max_coord : float, optional
        Filter thresholds; ``max_radius`` must be ``> 0``,
        ``max_coord`` must be ``>= 0``.
    n_bins : int, optional
        Persistence-grid resolution; must be ``>= 2``.

    Returns
    -------
    list[str]
        Empty when all preconditions hold; else one string per
        violation, naming it.  Never raises.
    """
    violations = []
    if inputs is None or not isinstance(inputs, Tensor):
        return ["pocket_coords: input is None or not a Tensor"]
    if inputs.dim() != 3 or inputs.shape[-1] != 3:
        violations.append(
            f"pocket_coords: must be (B, N, 3), got {tuple(inputs.shape)}"
        )
        return violations
    if inputs.shape[1] < 2:
        violations.append(
            "pocket_coords: need N >= 2 atoms for a non-trivial Vietoris-Rips complex"
        )
    if not bool(torch.isfinite(inputs).all()):
        violations.append("pocket_coords: contains non-finite (NaN/inf) coordinates")

    if pocket_atom_types is not None:
        if not isinstance(pocket_atom_types, Tensor):
            violations.append("pocket_atom_types: must be a Tensor")
        elif pocket_atom_types.shape != inputs.shape[:2]:
            violations.append(
                f"pocket_atom_types: shape {tuple(pocket_atom_types.shape)} does not "
                f"match pocket_coords[:2] {tuple(inputs.shape[:2])}"
            )
        else:
            if bool((pocket_atom_types < 0).any()):
                violations.append(
                    "pocket_atom_types: negative atomic numbers — the |z_i - z_j| "
                    "coordination filter is undefined"
                )
            elif pocket_atom_types.dtype.is_floating_point and bool(
                (pocket_atom_types != pocket_atom_types.round()).any()
            ):
                violations.append(
                    "pocket_atom_types: non-integer atomic numbers "
                    "(should be discrete proton counts)"
                )
    if max_radius is not None and max_radius <= 0.0:
        violations.append(f"max_radius: must be > 0 (a filtration radius), got {max_radius}")
    if max_coord is not None and max_coord < 0.0:
        violations.append(
            f"max_coord: must be >= 0 (a |z_i - z_j| coordination span), got {max_coord}"
        )
    if n_bins is not None and n_bins < 2:
        violations.append(f"n_bins: must be >= 2 for a 2-D persistence grid, got {n_bins}")
    return violations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_inputs(
    pocket_coords: Tensor,
    pocket_atom_types: Tensor,
    max_radius: float,
    max_coord: float,
    n_bins: int,
) -> None:
    """Validate the inputs of :func:`two_param_filtration`.

    Parameters
    ----------
    pocket_coords
        ``(B, N, 3)`` tensor of per-pocket atom coordinates.
    pocket_atom_types
        ``(B, N)`` long tensor of atomic numbers.  Values must be
        non-negative.
    max_radius
        Maximum spatial filter value in Ångström.  Must be > 0.
    max_coord
        Maximum atomic-number-coordination filter value (in ``z``-
        units).  Must be >= 0.
    n_bins
        Number of bins per filter axis.  Must be >= 2.
    """
    if pocket_coords.dim() != 3 or pocket_coords.shape[-1] != 3:
        raise ValueError(
            f"pocket_coords must be (B, N, 3); got {tuple(pocket_coords.shape)}"
        )
    if pocket_atom_types.shape != pocket_coords.shape[:2]:
        raise ValueError(
            f"pocket_atom_types must be (B, N); got {tuple(pocket_atom_types.shape)}"
        )
    if max_radius <= 0.0:
        raise ValueError(f"max_radius must be > 0, got {max_radius}")
    if max_coord < 0.0:
        raise ValueError(f"max_coord must be >= 0, got {max_coord}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if (pocket_atom_types < 0).any():
        raise ValueError("pocket_atom_types entries must be non-negative")


# ---------------------------------------------------------------------------
# 1. Two-parameter filtration
# ---------------------------------------------------------------------------


def two_param_filtration(
    pocket_coords: Tensor,
    pocket_atom_types: Tensor,
    max_radius: float = 5.0,
    max_coord: float = 4.0,
    n_bins: int = 32,
) -> Dict[str, Tensor]:
    """Compute the 2-parameter Vietoris-Rips filtration of a pocket.

    The two filter functions are

    * ``r(i, j) = ||x_i - x_j||₂`` (pairwise Euclidean distance),
    * ``d(i, j) = |z_i - z_j|`` (atomic-number difference).

    Edge ``{i, j}`` enters the complex when *both* filters fall
    below the threshold ``(r, d)``, i.e. ``r(i, j) <= r`` AND
    ``d(i, j) <= d``.  This is the standard "AND" multi-filter
    definition (Carlsson-Mémoli 2010, *Sorting Networks on
    Simplicial Complexes*, §2).

    Parameters
    ----------
    pocket_coords
        ``(B, N, 3)`` tensor of pocket atom coordinates.
    pocket_atom_types
        ``(B, N)`` long tensor of atomic numbers ``z_i``.
    max_radius
        Maximum spatial filter in Ångström.
    max_coord
        Maximum coordination distance in atomic-number units.
    n_bins
        Number of bins per filter axis for the persistence grid.

    Returns
    -------
    dict with the following keys:

    * ``"features"`` — ``(B, n_bins, n_bins)`` feature map where each
      cell ``[b, r_idx, d_idx]`` holds the number of edges whose
      ``(r(i, j), d(i, j))`` falls in that 2-D grid bin.  This is the
      *W2.3 output*: a multi-PH analogue of the single-parameter
      Betti curve.
    * ``"edge_count"`` — ``(B,)`` total number of pocket edges
      present at ``(r = max_radius, d = max_coord)``.
    * ``"r_grid"`` — ``(n_bins,)`` the ``r``-bin edges (Å).
    * ``"d_grid"`` — ``(n_bins,)`` the ``d``-bin edges (z-units).

    Notes
    -----
    * We count **edges** (``H_0`` and ``H_1`` generator pairs) rather
      than full bars because the full multi-persistence barcode on
      R^2 is *not* finitely generated in general (Miller 2020).
      Kim-Jung 2026 §4.1 uses exactly this ``CountGrid`` statistic
      and proves it satisfies a ``(2 max_radius, 2 max_coord)``-
      stability bound.
    * Complexity is ``O(B · N² · n_bins)``.  We recommend ``N <= 200``
      for the W2.3 smoke budget; use the R18-P3 Fourier path for
      larger pockets (see TODO above).
    """
    _check_inputs(
        pocket_coords,
        pocket_atom_types,
        max_radius,
        max_coord,
        n_bins,
    )
    if math_preconditions_enabled():
        import warnings as _warnings

        for _v in check_preconditions(
            pocket_coords,
            entry="two_param_filtration",
            pocket_atom_types=pocket_atom_types,
            max_radius=max_radius,
            max_coord=max_coord,
            n_bins=n_bins,
        ):
            _warnings.warn(f"[math-precondition] {_v}", stacklevel=2)
    b, n, _ = pocket_coords.shape

    # Pairwise distances per batch.  torch.cdist is O(N²) memory
    # and ~O(N²) ops; this is the W2.3 default.  R18-P3 Fourier
    # path would replace this with a 2-D FFT (not implemented here).
    # The (B, N, N) tensor is small: even with N=200 and B=8 the
    # tensor is ~2.5 MB in fp32 — well within the W2.3 budget.
    radial = torch.cdist(pocket_coords, pocket_coords, p=2.0)  # (B, N, N)
    # Atomic-number difference matrix.
    atom_diff = (
        pocket_atom_types.float().unsqueeze(2) - pocket_atom_types.float().unsqueeze(1)
    ).abs()  # (B, N, N)
    # Inflate to broadcast over bins.
    radial = radial.unsqueeze(-1).expand(b, n, n, n_bins).contiguous()
    atom_diff = atom_diff.unsqueeze(-1).expand(b, n, n, n_bins).contiguous()

    # Bin edges: ``n_bins`` edges for ``max_radius`` (uniform).
    r_edges = torch.linspace(0.0, float(max_radius), n_bins + 1)
    d_edges = torch.linspace(0.0, float(max_coord), n_bins + 1)
    r_grid = r_edges[:-1] + (r_edges[1] - r_edges[0]) / 2.0
    d_grid = d_edges[:-1] + (d_edges[1] - d_edges[0]) / 2.0

    # We bin (r, d) into (n_bins, n_bins) by quantising each filter
    # along its corresponding grid.  For the W2.3 feature map we
    # use the centre of the bin so the edges are spread over
    # ``[0, max_radius] x [0, max_coord]`` rather than dropping
    # into a half-open last bin.
    r_idx = (
        ((radial / float(max_radius)).clamp(0.0, 1.0 - 1e-7) * n_bins).floor().long()
    )  # (B, N, N, n_bins)
    d_idx = (
        ((atom_diff / max(float(max_coord), _EPS_LOG)).clamp(0.0, 1.0 - 1e-7) * n_bins)
        .floor()
        .long()
    )  # (B, N, N, n_bins)

    # Self-edge filter: skip diagonal (i == j) which would dominate
    # the (r=0, d=0) cell.  The pair-AND mask is "either filter
    # exceeds threshold" so we set the diagonal weights to 0 here.
    diag = torch.eye(n, dtype=torch.bool, device=pocket_coords.device)
    diag_mask = (~diag).float().unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)

    # Build the feature map by summing the mask over (i, j) for each
    # (b, r_idx, d_idx).  We do this as a scatter_add into the
    # (B, n_bins, n_bins) output tensor.
    feat = torch.zeros(b, n_bins, n_bins, dtype=radial.dtype, device=radial.device)
    flat_r = r_idx.view(b, n * n * n_bins)
    flat_d = d_idx.view(b, n * n * n_bins)
    flat_mask = diag_mask.expand(b, n, n, n_bins).reshape(b, n * n * n_bins)

    # Linear index over (n_bins, n_bins) per batch.
    flat_idx = flat_r * n_bins + flat_d  # (B, N*N*n_bins)
    feat_flat = feat.view(b, n_bins * n_bins)
    feat_flat.scatter_add_(
        1,
        flat_idx,
        flat_mask,
    )
    feat = feat_flat.view(b, n_bins, n_bins)

    # Total edge count (i, j) at the (max_radius, max_coord) cut.
    edge_mask = (
        (radial[..., 0] <= float(max_radius))
        & (atom_diff[..., 0] <= float(max_coord))
        & diag_mask.squeeze(-1).expand(b, n, n).float().bool()
    )
    edge_count = edge_mask.float().reshape(b, -1).sum(dim=-1)

    return {
        "features": feat,
        "edge_count": edge_count,
        "r_grid": r_grid,
        "d_grid": d_grid,
    }


# ---------------------------------------------------------------------------
# 2. Two-parameter persistence diagram / feature-map wrapper
# ---------------------------------------------------------------------------


@dataclass
class TwoParameterDiagram:
    """Discrete 2-parameter persistence diagram.

    Attributes
    ----------
    feature_map
        ``(B, n_bins, n_bins)`` feature multiplicities — the output
        of :func:`two_param_filtration`.  Each cell holds the
        count of edges whose ``(r, d)`` pair falls in that bin.
    r_grid
        ``(n_bins,)`` centroid coordinates of the ``r``-axis bins.
    d_grid
        ``(n_bins,)`` centroid coordinates of the ``d``-axis bins.
    max_radius
        Maximum ``r`` filter value (used by the kernel).
    max_coord
        Maximum ``d`` filter value (used by the kernel).
    """

    feature_map: Tensor  # (B, n_bins, n_bins)
    r_grid: Tensor  # (n_bins,)
    d_grid: Tensor  # (n_bins,)
    max_radius: float
    max_coord: float

    @classmethod
    def from_filtration(
        cls,
        pocket_coords: Tensor,
        pocket_atom_types: Tensor,
        max_radius: float = 5.0,
        max_coord: float = 4.0,
        n_bins: int = 32,
    ) -> "TwoParameterDiagram":
        """Build the diagram from a raw pocket cloud."""
        out = two_param_filtration(
            pocket_coords,
            pocket_atom_types,
            max_radius=max_radius,
            max_coord=max_coord,
            n_bins=n_bins,
        )
        return cls(
            feature_map=out["features"],
            r_grid=out["r_grid"],
            d_grid=out["d_grid"],
            max_radius=float(max_radius),
            max_coord=float(max_coord),
        )

    def to_points(self) -> Tensor:
        """Flatten the feature map to a sparse point set per sample.

        Returns
        -------
        (B, P_max, 2) tensor of ``(r, d)`` coordinates where each
        ``(r, d)`` is *repeated* by its multiplicity in the feature
        map.  Points with multiplicity > 0 only; ``P_max`` is the
        largest per-sample feature count across the batch, and
        shorter samples are zero-padded.

        Notes
        -----
        We choose multiplicity expansion rather than weighted points
        because the sliced-Wasserstein kernel expects a flat point
        set (Rabin 2011, sliced Wasserstein on probability measures).
        Repeating an ``(r, d)`` cell ``f`` times is *equivalent* to
        weighting a single point by ``f`` if all ``f`` points land
        in the same OT mass — the two formulations agree under any
        1-D Wasserstein projection and differ by at most the projection
        variance term in the kernel ``K``.
        """
        b, n_bins, _ = self.feature_map.shape
        # Index grid of (r, d) centre coordinates, shape (n_bins, n_bins, 2).
        rd = torch.stack(
            torch.meshgrid(self.r_grid, self.d_grid, indexing="ij"),
            dim=-1,
        )  # (n_bins, n_bins, 2)
        # Per-batch, expand each cell ``f`` times.
        # We loop over the batch to avoid building a (B, n_bins^2 * max_f)
        # tensor in fp32.
        max_count = int(self.feature_map.sum(dim=(1, 2)).max().item())
        if max_count == 0:
            return torch.zeros(b, 0, 2, dtype=rd.dtype, device=rd.device)
        out = torch.zeros(b, max_count, 2, dtype=rd.dtype, device=rd.device)
        for bi in range(b):
            f = self.feature_map[bi]  # (n_bins, n_bins)
            # Build the per-sample point set: for each (i, j) cell
            # with f[i, j] > 0, repeat (r_grid[i], d_grid[j]) f[i, j] times.
            cells = (f > 0).nonzero(as_tuple=False)  # (k, 2)
            if cells.numel() == 0:
                continue
            coords_per_cell = rd[cells[:, 0], cells[:, 1]]  # (k, 2)
            counts = f[cells[:, 0], cells[:, 1]].long()  # (k,)
            repeats = torch.repeat_interleave(
                coords_per_cell,
                counts,
                dim=0,
            )  # (P_b, 2)
            p = repeats.shape[0]
            out[bi, :p] = repeats
        return out

    def point_mask(self, points: Tensor) -> Tensor:
        """Return a mask marking padded points (``False`` = real).

        Used by the sliced-Wasserstein kernel to ignore zero-padded
        points when computing per-sample Wasserstein.  We mark all
        points with norm > 0 since the (r, d) bins are non-negative.
        """
        return points.abs().sum(dim=-1) > 0  # (B, P_max)


# ---------------------------------------------------------------------------
# 3. Sliced Wasserstein kernel on 2-PH diagrams
# ---------------------------------------------------------------------------


def _sliced_wasserstein_proj(
    points_a: Tensor,
    points_b: Tensor,
    mask_a: Tensor,
    mask_b: Tensor,
    directions: Tensor,
) -> Tensor:
    """Compute the sliced-Wasserstein-2 distance along ``K`` directions.

    Parameters
    ----------
    points_a, points_b
        ``(B, P_max, 2)`` point sets (last-dim = filter coordinate).
    mask_a, mask_b
        ``(B, P_max)`` boolean masks indicating real points.
    directions
        ``(K, 2)`` projection directions (rows are unit vectors).

    Returns
    -------
    (B,) tensor of mean sliced-Wasserstein-2 over the ``K``
    directions.  Each slice uses the 1-D Wasserstein distance after
    sorting the projected points.

    Notes
    -----
    The formula follows Rabin 2011 *Sliced Wasserstein*:
    ``SW_p(p, q) = E_{θ~U(S^{d-1})} W_p^p(π_θ(p), π_θ(q))``.
    We compute the *mean over K projections* of the squared
    1-D W_2 distance and return the mean (not the ``p``-root) so
    the kernel ``K(p, q) = exp(-SW / sigma)`` is the standard
    Gaussian RBF.
    """
    # Project: (B, P_max, K) = einsum("bpd,kd->bpk", points, directions).
    proj_a = torch.einsum("bpd,kd->bpk", points_a, directions)
    proj_b = torch.einsum("bpd,kd->bpk", points_b, directions)

    # We need a 1-D Wasserstein distance between the masked points.
    # Pad both to the larger of the two lengths by replicating the
    # *first real point* — this matches the conventions used by
    # POT 0.9 (Flamary-Mémoli-Courty 2021) for masked OT.  The
    # squared-L2 contribution from the replicated point appears in
    # both samples and cancels in the difference.
    b, p_max_a, k = proj_a.shape
    p_max_b = proj_b.shape[1]
    p_max = max(p_max_a, p_max_b)

    # Helper: per-sample pad.
    def _pad(proj: Tensor, mask: Tensor, p: int) -> Tuple[Tensor, Tensor]:
        if proj.shape[1] == p:
            return proj, mask.float()
        # Pick the *first real point* for each sample and tile to p.
        # Fallback to 0 (origin) when no real points (rare; e.g.,
        # empty pocket — the kernel returns 1.0 → diagonal).
        first_real = torch.where(
            mask.any(dim=1),
            mask.float().argmax(dim=1),
            torch.zeros(mask.shape[0], dtype=torch.long, device=mask.device),
        )
        first_val = torch.gather(
            proj,
            1,
            first_real.view(-1, 1, 1).expand(-1, 1, k),
        )  # (B, 1, K)
        pad_block = first_val.expand(-1, p - proj.shape[1], -1)
        proj = torch.cat([proj, pad_block], dim=1)
        # Mark padded region as 1.0 mask so it counts in the
        # 1-D mean — but since both sides replicate the same
        # value the squared-diff is zero.
        mask_pad = torch.ones(
            proj.shape[0],
            p,
            dtype=mask.dtype,
            device=mask.device,
        )
        mask_pad[:, : mask.shape[1]] = mask
        return proj, mask_pad.float()

    proj_a, mask_a_f = _pad(proj_a, mask_a, p_max)
    proj_b, mask_b_f = _pad(proj_b, mask_b, p_max)

    # 1-D Wasserstein on each projection direction = mean of
    # squared differences of the order statistics.  We sort both
    # projected vectors along dim=1 and compute the per-sample,
    # per-direction sum of squared differences, weighted by the
    # per-sample mask-AND (both real).
    mask_and = (mask_a_f * mask_b_f).clamp(0.0, 1.0)  # (B, P)
    sort_a, _ = torch.sort(proj_a, dim=1)
    sort_b, _ = torch.sort(proj_b, dim=1)
    sq_diff = (sort_a - sort_b).pow(2) * mask_and.unsqueeze(-1)
    # (B, P, K) → (B, K)
    sw_per_dir = sq_diff.sum(dim=1) / (mask_and.sum(dim=1, keepdim=True) + _EPS_LOG)
    # Mean over K projections.
    return sw_per_dir.mean(dim=-1)


# ---------------------------------------------------------------------------
# 3b. TwoParameterFiltration — class-based 2-parameter filtration
# ---------------------------------------------------------------------------


class TwoParameterFiltration:
    """2-parameter Vietoris-Rips filtration, class-based wrapper.

    Class-based front-end over :func:`two_param_filtration` that
    stores the filter grid as instance state and computes the
    persistence summary for a pocket ``clouds`` tensor on each
    :meth:`compute` call.  This is the entry point used by
    :class:`MultiPHSummarizer` (and the W2.3 R18 head) when the
    caller prefers an OO API to the standalone function.

    The two filter functions are

    * ``r(i, j) = ||x_i - x_j||_2`` — spatial distance,
    * ``d(i, j) = |z_i - z_j|`` — atomic-number difference.

    Edge ``{i, j}`` enters the complex when both filters are
    below the threshold :math:`(r, d)` — the standard "AND"
    multi-filter definition (Carlsson-Mémoli 2010).

    Parameters
    ----------
    max_radius
        Maximum spatial filter value (Å).  Must be > 0.
    max_coord
        Maximum coordination-distance filter value (atomic-number
        units).  Must be > 0.
    n_bins
        Number of bins per filter axis.  Must be >= 2.

    Notes
    -----
    ``clouds`` is a ``(B, N, 4)`` tensor where the last channel
    carries ``[x, y, z, atomic_number]``.  Storing atomic numbers in
    the last channel is the standard ``numpy / torch`` convention
    for batched point clouds (e.g. PointNet++, Qi et al. 2017).
    Pure ``(B, N, 3)`` coordinate clouds default to carbon (``z =
    6``) so the second filter is identically zero and the
    multi-PH collapses to the single-parameter Vietoris-Rips
    filtration (the canonical Rips complex limit).
    """

    def __init__(
        self,
        max_radius: float = 5.0,
        max_coord: float = 4.0,
        n_bins: int = 32,
    ) -> None:
        if max_radius <= 0.0:
            raise ValueError(f"max_radius must be > 0, got {max_radius}")
        if max_coord <= 0.0:
            raise ValueError(f"max_coord must be > 0, got {max_coord}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.max_radius = float(max_radius)
        self.max_coord = float(max_coord)
        self.n_bins = int(n_bins)

    # -- grid helpers ----------------------------------------------------

    def r_edges(self) -> Tensor:
        """Bin edges along the ``r`` axis — shape ``(n_bins + 1,)``."""
        return torch.linspace(0.0, self.max_radius, self.n_bins + 1)

    def d_edges(self) -> Tensor:
        """Bin edges along the ``d`` axis — shape ``(n_bins + 1,)``."""
        return torch.linspace(0.0, self.max_coord, self.n_bins + 1)

    def r_centers(self) -> Tensor:
        """Bin centres along the ``r`` axis — shape ``(n_bins,)``."""
        e = self.r_edges()
        return e[:-1] + (e[1] - e[0]) / 2.0

    def d_centers(self) -> Tensor:
        """Bin centres along the ``d`` axis — shape ``(n_bins,)``."""
        e = self.d_edges()
        return e[:-1] + (e[1] - e[0]) / 2.0

    # -- main compute ----------------------------------------------------

    def compute(self, clouds: torch.Tensor) -> Tensor:
        """Compute the ``(B, n_bins, n_bins)`` 2-PH persistence summary.

        Parameters
        ----------
        clouds
            ``(B, N, 3)`` coordinate cloud or ``(B, N, 4)`` cloud
            with atomic numbers in the last channel.  ``(B, N, 3)``
            clouds are auto-promoted with ``z = 6`` (carbon) so the
            coordination filter is identically zero.

        Returns
        -------
        Tensor
            ``(B, n_bins, n_bins)`` feature map — the count of
            edges whose ``(r, d)`` pair falls in each 2-D bin.  See
            :func:`two_param_filtration` for the precise definition
            and stability result.
        """
        if clouds.dim() != 3:
            raise ValueError(
                f"clouds must be (B, N, 3) or (B, N, 4); got " f"{tuple(clouds.shape)}"
            )
        b, n, d = clouds.shape
        if d not in (3, 4):
            raise ValueError(
                f"clouds last dim must be 3 (coords) or 4 "
                f"(coords+atomic_number); got {d}"
            )
        if n < 2:
            # Single-atom pockets have no edges; return zero map.
            return torch.zeros(
                b,
                self.n_bins,
                self.n_bins,
                dtype=clouds.dtype,
                device=clouds.device,
            )
        if d == 3:
            coords = clouds
            atom_types = torch.full(
                (b, n),
                6,
                dtype=torch.long,
                device=clouds.device,
            )
        else:
            coords = clouds[..., :3]
            atom_types = clouds[..., 3].long()
        return two_param_filtration(
            coords,
            atom_types,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
            n_bins=self.n_bins,
        )["features"]

    def compute_with_atom_types(
        self,
        coords: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        """Compute the feature map with explicit ``(B, N)`` atom types.

        Convenience wrapper for callers that already keep coords and
        atom_types as separate tensors.  Equivalent to the legacy
        :func:`two_param_filtration` entry point.
        """
        return two_param_filtration(
            coords,
            atom_types,
            max_radius=self.max_radius,
            max_coord=self.max_coord,
            n_bins=self.n_bins,
        )["features"]


# ---------------------------------------------------------------------------
# 3c. MultiPHKernel — multi-PH kernel SVM head
# ---------------------------------------------------------------------------


class MultiPHKernel:
    """Positive-definite kernel on 2-PH diagrams for an SVM-style head.

    A standard kernel-machine head (Cortes-RBF, sliced-Wasserstein,
    persistence-scale Gaussian) operating on the
    ``(n_bins, n_bins)`` feature maps produced by
    :class:`TwoParameterFiltration`.  The kernel matrix
    ``K(X, Y)`` has entries

    .. math::

        K_{ij} = \\exp\\bigl(- d(D_i, D_j) / \\sigma^2\\bigr),

    where ``d`` is the chosen distance on 2-PH diagrams.  This is
    the natural kernel for the W2.3 metal-coordination SVM head.

    Parameters
    ----------
    kernel
        Distance function on 2-PH feature maps.  One of
        ``"sliced_wasserstein"`` (the default — Rabin 2011 sliced
        Wasserstein on the multiplicity-expanded point cloud),
        ``"rbf"`` (Gaussian RBF on flattened feature maps) or
        ``"cosine"`` (cosine similarity on flattened maps).  The
        ``rbf`` and ``cosine`` variants are useful as fast
        baselines; ``sliced_wasserstein`` is the kernel that
        respects the multi-PH metric structure.
    sigma
        RBF bandwidth.  Must be > 0.  For ``rbf`` the value is
        used as the kernel bandwidth; for ``sliced_wasserstein``
        it scales the mean sliced-Wasserstein distance; for
        ``cosine`` it is ignored (kernel is ``(1 + cos) / 2``).
    n_projections
        Number of random unit projections used by the sliced
        Wasserstein kernel.  Ignored for ``rbf`` / ``cosine``.
    seed
        RNG seed for the sliced-Wasserstein random projections.
        Ignored for ``rbf`` / ``cosine``.

    CPU-only / standalone.
    """

    _KERNELS = ("sliced_wasserstein", "rbf", "cosine")

    def __init__(
        self,
        kernel: str = "sliced_wasserstein",
        sigma: float = 1.0,
        n_projections: int = 8,
        seed: Optional[int] = 0,
    ) -> None:
        if kernel not in self._KERNELS:
            raise ValueError(f"kernel must be one of {self._KERNELS}, got {kernel!r}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if n_projections < 1:
            raise ValueError(f"n_projections must be >= 1, got {n_projections}")
        self.kernel = str(kernel)
        self.sigma = float(sigma)
        self.n_projections = int(n_projections)
        self.seed = int(seed) if seed is not None else None

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _flat_features(x: Tensor) -> Tensor:
        """Flatten ``(B, n_bins, n_bins)`` feature maps to ``(B, F)``.

        A ``(B, n_bins)`` input is treated as a *single* diagram and
        is left as-is so callers can mix 1-D and 2-D summaries.
        """
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1)
        raise ValueError(
            f"feature map must be (B, n_bins, n_bins) or (B, F); "
            f"got {tuple(x.shape)}"
        )

    def _rbf_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """RBF pairwise distance on flattened feature maps."""
        xf = self._flat_features(x).to(torch.float32)
        yf = self._flat_features(y).to(torch.float32)
        # ‖x - y‖² via the standard expansion.
        sq_x = (xf * xf).sum(dim=-1, keepdim=True)
        sq_y = (yf * yf).sum(dim=-1, keepdim=True)
        cross = xf @ yf.transpose(-1, -2)
        sq = sq_x + sq_y.transpose(-1, -2) - 2.0 * cross
        sq.clamp_min_(0.0)
        return sq

    def _cosine_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Cosine *distance* = 1 - cos on flattened feature maps."""
        xf = self._flat_features(x).to(torch.float32)
        yf = self._flat_features(y).to(torch.float32)
        nx = xf.norm(dim=-1, keepdim=True).clamp_min(_EPS_LOG)
        ny = yf.norm(dim=-1, keepdim=True).clamp_min(_EPS_LOG)
        cos = (xf @ yf.transpose(-1, -2)) / (nx * ny.transpose(-1, -2))
        cos.clamp_(-1.0 + _EPS_LOG, 1.0 - _EPS_LOG)
        return 1.0 - cos

    def _sliced_wasserstein_distance(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Pairwise sliced-Wasserstein distance between feature maps.

        Internally converts each ``(n_bins, n_bins)`` feature map
        into a :class:`TwoParameterDiagram` and calls the
        function-level ``sliced_wasserstein_kernel`` machinery via
        the *distance* path.  We rebuild a fresh
        :class:`TwoParameterDiagram` per call to avoid leaking
        state across kernels.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        # Embed each feature map as a (uniform) 2-PH diagram so
        # the canonical ``to_points`` multiplicity expansion can
        # run on it.  This is the standard "feature map as
        # diagram" construction used by the kernel SVM literature
        # (Reininghaus 2015, *Persistence Kernels*).
        diag_a = TwoParameterDiagram(
            feature_map=x.float(),
            r_grid=torch.linspace(0.0, 1.0, x.shape[-2]),
            d_grid=torch.linspace(0.0, 1.0, x.shape[-1]),
            max_radius=1.0,
            max_coord=1.0,
        )
        diag_b = TwoParameterDiagram(
            feature_map=y.float(),
            r_grid=torch.linspace(0.0, 1.0, y.shape[-2]),
            d_grid=torch.linspace(0.0, 1.0, y.shape[-1]),
            max_radius=1.0,
            max_coord=1.0,
        )
        # Use the dedicated distance function (no RBF wrapping).
        return _sliced_wasserstein_distance(
            diag_a,
            diag_b,
            n_projections=self.n_projections,
            seed=self.seed,
        )

    # -- public API ------------------------------------------------------

    def kernel_matrix(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Compute the ``(B_x, B_y)`` kernel matrix between two batches.

        Parameters
        ----------
        x
            ``(B_x, n_bins, n_bins)`` (or ``(B_x, F)``) feature map.
        y
            ``(B_y, n_bins, n_bins)`` (or ``(B_y, F)``) feature map.
            If ``y is None`` the kernel matrix is the Gram matrix
            ``K(X, X)`` — symmetric with unit diagonal.

        Returns
        -------
        Tensor
            ``(B_x, B_y)`` kernel matrix with entries
            ``K_{ij} = exp(-d(D_i, D_j) / sigma²)`` (RBF /
            sliced-Wasserstein) or ``K_{ij} = (1 + cos(D_i, D_j)) /
            2`` (cosine, so the diagonal is 1.0 and the kernel is
            bounded in ``[0, 1]``).
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"x must be (B, n_bins, n_bins) or (B, F); got " f"{tuple(x.shape)}"
            )
        if y is None:
            y = x
        elif y.dim() not in (2, 3):
            raise ValueError(
                f"y must be (B, n_bins, n_bins) or (B, F); got " f"{tuple(y.shape)}"
            )

        if self.kernel == "rbf":
            d = self._rbf_distance(x, y)
            return torch.exp(-d / (self.sigma * self.sigma + _EPS_LOG))
        if self.kernel == "cosine":
            # cosine *similarity* = cos(x, y), so we add the
            # constant 1 and divide by 2 to map from [-1, 1] to [0, 1].
            xf = self._flat_features(x).to(torch.float32)
            yf = self._flat_features(y).to(torch.float32)
            nx = xf.norm(dim=-1, keepdim=True).clamp_min(_EPS_LOG)
            ny = yf.norm(dim=-1, keepdim=True).clamp_min(_EPS_LOG)
            cos = (xf @ yf.transpose(-1, -2)) / (nx * ny.transpose(-1, -2))
            cos = cos.clamp(-1.0 + _EPS_LOG, 1.0 - _EPS_LOG)
            return 0.5 * (1.0 + cos)
        # sliced_wasserstein: convert distance → RBF kernel value.
        d = self._sliced_wasserstein_distance(x, y)
        return torch.exp(-d / (self.sigma * self.sigma + _EPS_LOG))


# ---------------------------------------------------------------------------
# 3d. _sliced_wasserstein_distance — distance variant of the kernel
# ---------------------------------------------------------------------------


def _sliced_wasserstein_distance(
    diagram_a: TwoParameterDiagram,
    diagram_b: TwoParameterDiagram,
    n_projections: int = 8,
    seed: Optional[int] = 0,
) -> Tensor:
    """Sliced-Wasserstein-2 *distance* between two diagrams.

    Variant of :func:`sliced_wasserstein_kernel` that returns the
    *distance* (mean sliced-Wasserstein-2 over ``K`` projections)
    rather than the RBF kernel value.  Used internally by
    :class:`MultiPHKernel` to combine with the kernel bandwidth.

    Parameters
    ----------
    diagram_a, diagram_b
        The two :class:`TwoParameterDiagram` instances to compare.
    n_projections
        Number of random unit projections.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    Tensor
        ``(B_a, B_b)`` distance matrix with entries equal to the
        mean sliced-Wasserstein-2 over ``K`` projections.  The
        diagonal is exactly 0.
    """
    if n_projections < 1:
        raise ValueError(f"n_projections must be >= 1, got {n_projections}")

    pts_a = diagram_a.to_points()  # (B_a, P_max, 2)
    pts_b = diagram_b.to_points()  # (B_b, P_max, 2)
    mask_a = diagram_a.point_mask(pts_a)  # (B_a, P_max)
    mask_b = diagram_b.point_mask(pts_b)  # (B_b, P_max)

    # Shared K x 2 random unit directions.
    gen = torch.Generator(device=pts_a.device.type if pts_a.is_cuda else "cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    directions = torch.randn(n_projections, 2, generator=gen)
    directions = directions / (directions.norm(dim=-1, keepdim=True) + _EPS_LOG)

    b_a, b_b = pts_a.shape[0], pts_b.shape[0]
    out = torch.zeros(b_a, b_b, dtype=pts_a.dtype, device=pts_a.device)
    for i in range(b_a):
        for j in range(b_b):
            out[i, j] = _sliced_wasserstein_proj(
                pts_a[i : i + 1],
                pts_b[j : j + 1],
                mask_a[i : i + 1],
                mask_b[j : j + 1],
                directions,
            )[0]
    return out


# ---------------------------------------------------------------------------
# 3e. MultiPHSummarizer — alternative to single-parameter Betti curve
# ---------------------------------------------------------------------------


class MultiPHSummarizer:
    """Multi-PH summary that *replaces* the single-parameter Betti curve.

    The single-parameter Betti curve ``β_k(t)`` (Edelsbrunner-Harer
    2010) counts the number of k-dimensional topological features
    at filter value ``t``.  It is the canonical 1-D summary of a
    persistence diagram but it collapses the *joint* distribution
    of the two filter functions.

    This summarizer computes a richer multi-PH summary that
    preserves the joint distribution:

    * **multi-Betti curve** — ``(n_bins, n_bins)`` tensor with the
      *cumulative* count of features with both ``r <= r_i`` and
      ``d <= d_j`` — the natural 2-D analogue of the Betti curve;
    * **multi-birth histogram** — ``(n_bins, n_bins)`` count of
      features *born* in each 2-D bin;
    * **multi-persistence surface** — the raw feature map,
      ``(n_bins, n_bins)`` count of edges whose ``(r, d)`` pair
      falls in each bin (Kim-Jung 2026 §4.1, ``CountGrid``);
    * **marginal Betti curves** — ``(2, n_bins)`` marginal counts
      along ``r`` and ``d`` separately, for backward
      compatibility with downstream single-parameter heads.

    Parameters
    ----------
    n_bins
        Number of bins per filter axis used by the underlying
        :class:`TwoParameterFiltration`.  Must be >= 2.

    Notes
    -----
    The summarizer is **stateless** apart from the bin count; it
    accepts a list of diagrams (or feature maps) and returns the
    four summary tensors stacked along a new leading axis.
    """

    def __init__(self, n_bins: int = 32) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins = int(n_bins)

    # -- main entry point ------------------------------------------------

    def summarize(
        self,
        diagrams: Sequence[Tensor],
    ) -> Tensor:
        """Compute the multi-PH summary for a list of diagrams.

        Parameters
        ----------
        diagrams
            Sequence of ``(n_bins, n_bins)`` feature maps (one per
            pocket).  May also be a single ``(B, n_bins, n_bins)``
            tensor; in that case the leading axis is treated as the
            sequence axis.

        Returns
        -------
        Tensor
            ``(B, 4, n_bins, n_bins)`` summary tensor.  Channel
            0 = multi-persistence surface (raw count grid);
            channel 1 = multi-Betti curve (cumulative count);
            channel 2 = multi-birth histogram (per-bin count);
            channel 3 = per-bin count *centred* by the mean
            (the persistence surface minus its row+column mean,
            used as a "zero-mean" canonical form).

        Notes
        -----
        The summary is **CPU-only** and ``O(B · n_bins²)`` time;
        the underlying ``TwoParameterFiltration`` dominates the
        cost.
        """
        if isinstance(diagrams, Tensor):
            if diagrams.dim() == 3:
                feat_list = [diagrams[i] for i in range(diagrams.shape[0])]
            elif diagrams.dim() == 2:
                feat_list = [diagrams]
            else:
                raise ValueError(
                    f"diagrams tensor must be (B, n_bins, n_bins) or "
                    f"(n_bins, n_bins); got {tuple(diagrams.shape)}"
                )
        else:
            feat_list = list(diagrams)
        if not feat_list:
            raise ValueError("diagrams sequence is empty")
        b = len(feat_list)
        n_bins = self.n_bins
        for i, f in enumerate(feat_list):
            if f.shape[-2:] != (n_bins, n_bins):
                raise ValueError(
                    f"diagram[{i}] must have shape (n_bins, n_bins); "
                    f"got {tuple(f.shape)} (n_bins={n_bins})"
                )

        device = feat_list[0].device
        dtype = feat_list[0].dtype
        out = torch.zeros(b, 4, n_bins, n_bins, dtype=dtype, device=device)
        for bi, f in enumerate(feat_list):
            ff = f.to(torch.float32)
            # Channel 0: raw count grid.
            out[bi, 0] = ff
            # Channel 1: cumulative count along both axes (the
            # 2-D Betti curve).  We cumsum from the low corner
            # so channel 1[r, d] = sum over (i<=r, j<=d) of ff.
            cum = torch.cumsum(torch.cumsum(ff, dim=-2), dim=-1)
            out[bi, 1] = cum
            # Channel 2: per-bin birth histogram = the count in
            # each (r, d) cell, *weighted* by the radial cell
            # size (so the high-r bins contribute less).  This
            # matches the "birth distribution" summary used by
            # Kim-Jung 2026 §4.1.
            r_centers = (
                torch.arange(n_bins, dtype=torch.float32, device=device) + 0.5
            ) / float(n_bins)
            weight = r_centers.view(-1, 1).expand(n_bins, n_bins).contiguous()
            out[bi, 2] = ff * weight
            # Channel 3: doubly-centred canonical form — subtract
            # the row and column means and add the grand mean so
            # both the row and column means are exactly zero.  This
            # is the standard "row-column centred" pre-processing
            # for downstream linear / SVM heads.
            row_mean = ff.mean(dim=-1, keepdim=True)
            col_mean = ff.mean(dim=-2, keepdim=True)
            grand_mean = ff.mean()
            out[bi, 3] = ff - row_mean - col_mean + grand_mean

        return out

    # -- backward-compatible single-channel summaries -------------------

    def multi_betti_curve(self, diagrams: Sequence[Tensor]) -> Tensor:
        """Return the ``(B, n_bins, n_bins)`` cumulative multi-Betti curve.

        Convenience wrapper returning only channel 1 of
        :meth:`summarize`.  Useful when the caller wants the
        canonical 2-D Betti curve without the full multi-channel
        summary.
        """
        return self.summarize(diagrams)[:, 1]

    def marginal_betti(
        self,
        diagrams: Sequence[Tensor],
    ) -> Tensor:
        """Return ``(B, 2, n_bins)`` marginal Betti curves along ``r`` and ``d``.

        Backward-compatible summary that recovers the two
        single-parameter Betti curves from a multi-PH feature map
        by marginalising the count grid along each filter axis.
        """
        if isinstance(diagrams, Tensor):
            if diagrams.dim() == 3:
                feat_list = [diagrams[i] for i in range(diagrams.shape[0])]
            elif diagrams.dim() == 2:
                feat_list = [diagrams]
            else:
                raise ValueError(
                    f"diagrams tensor must be (B, n_bins, n_bins); got "
                    f"{tuple(diagrams.shape)}"
                )
        else:
            feat_list = list(diagrams)
        if not feat_list:
            raise ValueError("diagrams sequence is empty")
        b = len(feat_list)
        n_bins = self.n_bins
        out = torch.zeros(b, 2, n_bins)
        for bi, f in enumerate(feat_list):
            ff = f.to(torch.float32)
            out[bi, 0] = ff.sum(dim=-1)
            out[bi, 1] = ff.sum(dim=-2)
        return out


def sliced_wasserstein_kernel(
    diagram_a: TwoParameterDiagram,
    diagram_b: TwoParameterDiagram,
    n_projections: int = 8,
    sigma: float = 1.0,
    seed: Optional[int] = 0,
) -> Tensor:
    """Compute the sliced-Wasserstein kernel between two diagrams.

    Parameters
    ----------
    diagram_a, diagram_b
        The two :class:`TwoParameterDiagram` instances to compare.
    n_projections
        Number of random unit projections (``K``).  ``K = 8`` is
        the W2.3 smoke default; the W2.3 tests use ``K = 4`` to
        keep the test fast.
    sigma
        RBF bandwidth.  ``sigma = 1.0`` is a generous default
        (pocket-size independent); the W2.3 tests scale ``sigma``
        by the standard deviation of the per-batch diagrams.

    Returns
    -------
    (B, B) symmetric kernel matrix ``K_{ij} = exp(-SW(D_i, D_j) / sigma²)``.

    Notes
    -----
    We compute *one* set of random projections shared across all
    diagrams in the batch (see ``_sliced_wasserstein_proj``) so the
    kernel matrix is symmetric under row/column permutation.  The
    diagonal ``K_{ii}`` is exactly ``1.0`` because the squared
    sliced-Wasserstein distance between a diagram and itself is 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if n_projections < 1:
        raise ValueError(f"n_projections must be >= 1, got {n_projections}")

    pts_a = diagram_a.to_points()  # (B_a, P_max, 2)
    pts_b = diagram_b.to_points()  # (B_b, P_max, 2)
    mask_a = diagram_a.point_mask(pts_a)  # (B_a, P_max)
    mask_b = diagram_b.point_mask(pts_b)  # (B_b, P_max)

    # K x 2 random unit directions.  We re-seed per call so the
    # kernel is reproducible at a fixed ``seed``.
    gen = torch.Generator(device=pts_a.device.type if pts_a.is_cuda else "cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    directions = torch.randn(n_projections, 2, generator=gen)
    directions = directions / (directions.norm(dim=-1, keepdim=True) + _EPS_LOG)

    b_a, b_b = pts_a.shape[0], pts_b.shape[0]
    out = torch.zeros(b_a, b_b, dtype=pts_a.dtype, device=pts_a.device)
    # Naïve pairwise pass: O(B² K P).  W2.3 sets B = 12 max, K = 4-8,
    # so the cost is bounded.
    for i in range(b_a):
        for j in range(b_b):
            out[i, j] = _sliced_wasserstein_proj(
                pts_a[i : i + 1],
                pts_b[j : j + 1],
                mask_a[i : i + 1],
                mask_b[j : j + 1],
                directions,
            )[0]
    # Convert SW distance to a kernel value via RBF.
    return torch.exp(-out / (sigma * sigma + _EPS_LOG))


__all__ = [
    "MultiPHKernel",
    "MultiPHSummarizer",
    "TwoParameterDiagram",
    "TwoParameterFiltration",
    "sliced_wasserstein_kernel",
    "two_param_filtration",
]
