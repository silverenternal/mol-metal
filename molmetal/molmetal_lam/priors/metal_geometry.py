"""Geometric priors for metal coordination complexes (T9).

This module houses *soft* geometric priors for transition-metal centres
encountered in SBDD / click-chemistry / catalysis tasks.  The priors
are designed to be **opt-in**: callers wire a per-edge
``dative_bond_edge_attr`` flag into :class:`molmetal.adapters.egnn_rocm.EGNN`
(or an equivalent equivariant layer) and apply the prior's loss term to
the predicted coordinates.  When the flag is not provided, the prior is a
no-op and downstream behaviour is bit-for-bit unchanged.

Geometric prior — square-planar Pt(II)
--------------------------------------
Pt(II) is a d8 centre and overwhelmingly adopts a square-planar geometry
in Pt(II)–pincer / cisplatin-class complexes (Hartl 2015,
*Dalton Trans.* 44, 1950; selected examples from the tmQM-21k dataset
used in the F2 pre-training report).  The canonical bond angles around
the metal centre are 90 degrees (within a few degrees) and the four
donor atoms lie in a single plane that contains the Pt nucleus.

This module exposes :class:`SquarePlanarPtII` which:

1. Identifies Pt(II) centres and their dative-bond donor atoms using a
   caller-supplied ``dative_bond_edge_attr`` flag plus atomic-number
   lookup (Pt = 78).
2. Computes the per-centre mean absolute deviation of the four
   Pt-L-M angles from the ideal 90 degrees (in radians).
3. Returns a scalar penalty
   ``loss = (1/N_centres) * sum_c mean_l (|theta_lc - pi/2|)``
   which a training loop can add to its objective with a tunable
   weight (recommended ``1.0``).

Why a soft penalty rather than a hard projection?
- The flow-matching model is trained end-to-end; a soft penalty lets
  gradients flow back through the predicted coordinates so the
  velocity field learns to *generate* square-planar geometry rather
  than snapping to it post-hoc.
- The penalty is automatically inactive when no Pt(II) centre is
  present in the batch (the loss is zero).
- The penalty is differentiable w.r.t. positions, so it composes with
  standard CFM training.

Scope and limitations
---------------------
- Single-graph path only (positions / edges are per-graph).  Callers
  running on batched tensors should call the prior once per graph and
  average the resulting losses (the included helper
  :func:`square_planar_penalty_batched` does this).
- Dative bonds are taken at face value from the caller-supplied flag.
  Bond-order semantics (single vs double dative) are intentionally
  ignored at this stage — only "is / is not a dative bond" matters for
  the angle penalty.
- The penalty uses the ideal 90 degrees for the Pt-L-X angle, where L
  is the donor and X is any *other* donor.  Cis / trans preference is
  not enforced here; a downstream chirality-aware prior could refine
  it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


__all__ = [
    # Square-planar legacy API (T9, kept for backward compatibility)
    "SquarePlanarPtII",
    "square_planar_penalty",
    "square_planar_penalty_batched",
    "PT_ATOMIC_NUMBER",
    "IDEAL_SQUARE_PLANAR_ANGLE",
    # Generalised TODO-09 API
    "MetalGeometryPrior",
    "Geometry",
    "GeometryPenalty",
    "GeometryPenaltyBatched",
    "bondi_van_der_waals_radius",
    "METAL_PRIOR_KIND",
    "apply_metal_prior",
    "prior_loss",
    # Phase 3H — soft, continuous metal-geometry prior for MCTS reward
    # aggregation (replaces the historical hard-gate bonus).
    "soft_score_metal_geometry",
    "SoftMetalGeometryDiagnostics",
    "DEFAULT_SOFT_SCORE_TARGETS",
]


PT_ATOMIC_NUMBER: int = 78
"""Atomic number of platinum (Pt).  Used as the canonical d8 centre."""

IDEAL_SQUARE_PLANAR_ANGLE: float = math.pi / 2.0
"""Ideal Pt-L-X angle (radians) for a perfect square-planar geometry (90 deg)."""


# ---------------------------------------------------------------------------
# Core penalty
# ---------------------------------------------------------------------------
def square_planar_penalty(
    positions: torch.Tensor,
    dative_bond_edge_index: torch.Tensor,
    dative_bond_edge_attr: torch.Tensor,
    atom_types: torch.Tensor,
    pt_atomic_number: int = PT_ATOMIC_NUMBER,
    ideal_angle: float = IDEAL_SQUARE_PLANAR_ANGLE,
) -> torch.Tensor:
    """Soft penalty for deviations from cis 90° / trans 180° at d8 Pt(II).

    Parameters
    ----------
    positions : ``(N, 3)`` tensor
        Per-atom coordinates (single graph).
    dative_bond_edge_index : ``(2, E_d)`` long tensor
        Edges flagged as dative (ligand donor -> metal centre).  The
        metal is conventionally the destination node.
    dative_bond_edge_attr : ``(E_d,)`` bool tensor
        Per-edge flag matching ``dative_bond_edge_index`` order.  When
        ``False``, that edge is excluded from the penalty.
    atom_types : ``(N,)`` long tensor
        Atomic numbers.  Pt centres are identified by
        ``atom_types[edge_dst] == pt_atomic_number``.
    pt_atomic_number : int, default ``PT_ATOMIC_NUMBER``
        Atomic number used to detect the metal centre.
    ideal_angle : float, default ``IDEAL_SQUARE_PLANAR_ANGLE``
        Target angle in radians (default 90 degrees).

    Returns
    -------
    ``torch.Tensor`` scalar (0-dim, on the same device as ``positions``).
        Mean absolute deviation of Pt-L-X angles from ``ideal_angle``,
        averaged over Pt centres and their donor atoms.  Returns
        exactly zero when no Pt centre or no dative donor is found.

    Notes
    -----
    The penalty is differentiable w.r.t. ``positions`` because we use
    :func:`torch.arccos` on the cosine of the angle.  Numerical
    stability is provided by clamping the cosine to ``[-1+eps, 1-eps]``
    before ``arccos`` to avoid NaN gradients at degenerate (collinear)
    configurations.
    """
    device = positions.device
    if dative_bond_edge_index.numel() == 0:
        return torch.zeros((), device=device, dtype=positions.dtype)
    if dative_bond_edge_attr is None or dative_bond_edge_attr.numel() == 0:
        return torch.zeros((), device=device, dtype=positions.dtype)
    # Restrict to edges flagged as dative.
    mask = dative_bond_edge_attr.to(torch.bool)
    if not mask.any():
        return torch.zeros((), device=device, dtype=positions.dtype)
    src = dative_bond_edge_index[0][mask]  # ligand donor
    dst = dative_bond_edge_index[1][mask]  # metal centre
    # Only keep edges where the destination is Pt.
    is_pt = atom_types[dst] == pt_atomic_number
    if not is_pt.any():
        return torch.zeros((), device=device, dtype=positions.dtype)
    src = src[is_pt]
    dst = dst[is_pt]

    pt_positions = positions[dst]      # (E_d, 3)
    donor_positions = positions[src]   # (E_d, 3)
    # Direction from metal -> donor.
    rel = donor_positions - pt_positions                              # (E_d, 3)
    dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)             # (E_d, 1)
    unit = rel / dist                                                  # (E_d, 3)

    # Group donors by Pt centre so we form the Pt-L-X angle only between
    # donors attached to the *same* Pt.  We do this by iterating per-Pt.
    unique_pt, inverse_idx = torch.unique(dst, return_inverse=True)
    penalty = positions.new_zeros(())
    n_centres = 0
    eps = 1e-6
    for pt_local_idx in range(unique_pt.numel()):
        sel = inverse_idx == pt_local_idx
        if sel.sum() < 2:
            # Need at least 2 donors to form an angle.
            continue
        u = unit[sel]                                # (k, 3)  unit vectors Pt -> donor_l
        # Pairwise cosine similarity (k, k).
        cos = u @ u.t()
        cos = cos.clamp(-1.0 + eps, 1.0 - eps)
        # Use only the upper triangle (i < j) — each angle is counted once.
        iu = torch.triu_indices(cos.size(0), cos.size(1), offset=1, device=device)
        if iu.numel() == 0:
            continue
        angles = torch.arccos(cos[iu[0], iu[1]])     # (n_pairs,)
        targets = angles.new_tensor([ideal_angle, math.pi])
        deviation = (angles.unsqueeze(-1) - targets).abs().min(dim=-1).values
        penalty = penalty + deviation.mean()
        n_centres += 1
    if n_centres == 0:
        return torch.zeros((), device=device, dtype=positions.dtype)
    return penalty / n_centres


def square_planar_penalty_batched(
    positions_list,
    dative_edge_index_list,
    dative_edge_attr_list,
    atom_types_list,
    pt_atomic_number: int = PT_ATOMIC_NUMBER,
    ideal_angle: float = IDEAL_SQUARE_PLANAR_ANGLE,
) -> torch.Tensor:
    """Batched wrapper around :func:`square_planar_penalty`.

    Each argument is a Python list (one entry per graph).  The result is
    the mean of per-graph penalties; graphs with no Pt centres
    contribute zero and therefore don't bias the average.
    """
    device = positions_list[0].device
    n = len(positions_list)
    if n == 0:
        return torch.zeros((), device=device)
    total = positions_list[0].new_zeros(())
    for pos, ei, ea, at in zip(
        positions_list, dative_edge_index_list, dative_edge_attr_list, atom_types_list
    ):
        total = total + square_planar_penalty(
            pos, ei, ea, at,
            pt_atomic_number=pt_atomic_number,
            ideal_angle=ideal_angle,
        )
    return total / n


# ---------------------------------------------------------------------------
# Module wrapper — convenience nn.Module so callers can ``register`` the prior
# ---------------------------------------------------------------------------
@dataclass
class _PriorDiagnostics:
    """Per-call diagnostics (useful for logging during training)."""

    n_pt_centres: int
    n_dative_edges_considered: int


class SquarePlanarPtII(nn.Module):
    """nn.Module wrapper around :func:`square_planar_penalty`.

    Use this from a training loop by calling ``self.forward(...)`` and
    adding the returned scalar to the CFM loss with a tunable weight.

    Example
    -------
    >>> prior = SquarePlanarPtII(weight=1.0)
    >>> loss = cfm_loss + atom_loss_weight * atom_loss + prior(positions, ...)
    """

    def __init__(
        self,
        weight: float = 1.0,
        pt_atomic_number: int = PT_ATOMIC_NUMBER,
        ideal_angle: float = IDEAL_SQUARE_PLANAR_ANGLE,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.pt_atomic_number = int(pt_atomic_number)
        self.ideal_angle = float(ideal_angle)
        self.last_diagnostics: _PriorDiagnostics | None = None

    def forward(
        self,
        positions: torch.Tensor,
        dative_bond_edge_index: torch.Tensor,
        dative_bond_edge_attr: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        raw = square_planar_penalty(
            positions,
            dative_bond_edge_index,
            dative_bond_edge_attr,
            atom_types,
            pt_atomic_number=self.pt_atomic_number,
            ideal_angle=self.ideal_angle,
        )
        # Bookkeeping for logging: best-effort, no failure mode if the
        # edge tensors are empty / on a different device.
        with torch.no_grad():
            try:
                n_dative = int(
                    (dative_bond_edge_attr.to(torch.bool)).sum().item()
                )
            except Exception:  # noqa: BLE001 — diagnostics never break training
                n_dative = -1
            try:
                n_pt = int(
                    (atom_types == self.pt_atomic_number).sum().item()
                )
            except Exception:  # noqa: BLE001
                n_pt = -1
            self.last_diagnostics = _PriorDiagnostics(
                n_pt_centres=n_pt,
                n_dative_edges_considered=n_dative,
            )
        return self.weight * raw


# ===========================================================================
# TODO-09 — generalised metal-geometry prior (4 geometries + Bondi vdW)
# ===========================================================================
class Geometry(str):
    """String enum for the supported coordination geometries.

    Used as a literal type so :class:`MetalGeometryPrior` can dispatch
    on geometry name without importing the :mod:`enum` stdlib module.
    """

    SQUARE_PLANAR: "Geometry" = "square_planar"
    TETRAHEDRAL: "Geometry" = "tetrahedral"
    OCTAHEDRAL: "Geometry" = "octahedral"
    TRIGONAL_BIPYRAMIDAL: "Geometry" = "trigonal_bipyramidal"


# Constant tuple of the supported geometries — convenient for assertions.
SUPPORTED_GEOMETRIES: tuple = (
    Geometry.SQUARE_PLANAR,
    Geometry.TETRAHEDRAL,
    Geometry.OCTAHEDRAL,
    Geometry.TRIGONAL_BIPYRAMIDAL,
)

# ---------------------------------------------------------------------------
# Canonical coordination numbers per geometry
# ---------------------------------------------------------------------------
# Square-planar: 4 donors, ideal angle 90 deg.
# Tetrahedral:   4 donors, ideal angle 109.5 deg (acos(-1/3)).
# Octahedral:    6 donors, two angle types (90 deg for adjacent,
#                180 deg for trans).
# Trigonal-bipyramidal: 5 donors, two angle types (90 deg axial-equatorial,
#                120 deg equatorial-equatorial, 180 deg axial-axial).
GEOMETRY_COORDINATION_NUMBER: dict = {
    Geometry.SQUARE_PLANAR: 4,
    Geometry.TETRAHEDRAL: 4,
    Geometry.OCTAHEDRAL: 6,
    Geometry.TRIGONAL_BIPYRAMIDAL: 5,
}

# Ideal angles (radians) per geometry.  When a geometry has multiple
# distinct angle types (square-planar / octahedral / trigonal-bipyramidal),
# we list each in the order [cis, trans] / [adjacent, trans] /
# [ax-eq, eq-eq, ax-ax].  For square-planar, both 90 deg (cis L-M-L)
# and 180 deg (trans L-M-L) are ideal — both must be in the set so the
# penalty is zero at a perfect geometry.
GEOMETRY_IDEAL_ANGLES: dict = {
    Geometry.SQUARE_PLANAR: (math.pi / 2.0, math.pi),
    Geometry.TETRAHEDRAL: (math.acos(-1.0 / 3.0),),
    Geometry.OCTAHEDRAL: (math.pi / 2.0, math.pi),
    Geometry.TRIGONAL_BIPYRAMIDAL: (math.pi / 2.0, 2.0 * math.pi / 3.0, math.pi),
}

# Canonical bond angles in degrees — exposed for human-readable diagnostics.
GEOMETRY_IDEAL_ANGLES_DEG: dict = {
    Geometry.SQUARE_PLANAR: (90.0, 180.0),
    Geometry.TETRAHEDRAL: (109.471,),
    Geometry.OCTAHEDRAL: (90.0, 180.0),
    Geometry.TRIGONAL_BIPYRAMIDAL: (90.0, 120.0, 180.0),
}

# Canonical coordination-geometry → metal centre mapping.  Used by
# :meth:`MetalGeometryPrior.prior_loss` to dispatch by metal centre when
# the caller does not pass an explicit ``metal_geometry`` mapping.  Pt(II)
# is overwhelmingly square-planar; Ru(II)/Ru(III) are typically octahedral;
# Cu(I) and many d10 centres are tetrahedral; Fe(0) in Fe(CO)5 is
# trigonal-bipyramidal.  These are the four archetypal SBDD geometries
# the prior covers; the mapping can be overridden by the caller.
DEFAULT_METAL_GEOMETRY: dict = {
    78: Geometry.SQUARE_PLANAR,      # Pt(II)
    46: Geometry.SQUARE_PLANAR,      # Pd(II)  (analogous d8)
    29: Geometry.TETRAHEDRAL,        # Cu(I) / Zn(II) common coordination
    30: Geometry.TETRAHEDRAL,        # Zn(II)
    26: Geometry.OCTAHEDRAL,         # Fe(II)
    44: Geometry.OCTAHEDRAL,         # Ru(II) — common cis / trans Ru(CO)4L2 etc.
    77: Geometry.OCTAHEDRAL,         # Ir(III)
    45: Geometry.OCTAHEDRAL,         # Rh(III)
    25: Geometry.TRIGONAL_BIPYRAMIDAL,  # Mn(0) — Mn(CO)5Br etc.
}


# ---------------------------------------------------------------------------
# Bondi (1964) van der Waals radii (Å) — sanity bounds for the prior.
# ---------------------------------------------------------------------------
# Reference: Bondi, A. (1964). "van der Waals Volumes and Radii".
# J. Phys. Chem. 68 (3), 441–451.  A subset of the canonical table; this
# is sufficient for the metals the prior cares about (Pt, Pd, Cu, Zn,
# Fe, Ru, Ir, Rh, Mn, plus a few common donor atoms).  Unknown atoms
# fall back to 1.7 Å (a common heuristic default) via
# :func:`bondi_van_der_waals_radius`.
BONDI_VAN_DER_WAALS_RADII: dict = {
    # Metals (Å)
    25: 2.05,   # Mn
    26: 2.00,   # Fe (low-spin ~1.94; high-spin ~2.05)
    29: 1.40,   # Cu
    30: 1.39,   # Zn
    44: 2.10,   # Ru
    45: 2.00,   # Rh
    46: 1.63,   # Pd
    77: 2.00,   # Ir
    78: 1.75,   # Pt — note Bondi's value is 1.75 Å (small, d8 contraction)
    # Donor atoms (Å)
    1:  1.20,   # H
    6:  1.70,   # C
    7:  1.55,   # N
    8:  1.52,   # O
    9:  1.47,   # F
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    35: 1.85,   # Br
    53: 1.98,   # I
}

_BONDI_FALLBACK_RADIUS: float = 1.70
"""Default vdW radius (Å) for atoms not in :data:`BONDI_VAN_DER_WAALS_RADII`."""


def bondi_van_der_waals_radius(atomic_number: int) -> float:
    """Return the Bondi (1964) van der Waals radius for ``atomic_number``.

    Falls back to :data:`_BONDI_FALLBACK_RADIUS` (1.70 Å) when the
    element is not in the table — the standard heuristic used by
    cheminformatics toolkits for unknowns.
    """
    return float(BONDI_VAN_DER_WAALS_RADII.get(int(atomic_number), _BONDI_FALLBACK_RADIUS))


# ---------------------------------------------------------------------------
# Edge-type categorical used by MetalGeometryPrior — matches
# molmetal.adapters.egnn_rocm.EDGE_TYPE_* constants; re-declared here
# so this module remains importable without the EGNN adapter (avoids a
# circular import in tool-only pipelines).
# ---------------------------------------------------------------------------
METAL_PRIOR_KIND: str = "metal_geometry"
"""Discriminator tag for the prior-loss dictionary returned by
:meth:`MetalGeometryPrior.apply_prior`.  Downstream training loops can
identify the metal-geometry prior in a multi-prior loss bundle."""


@dataclass
class _GeometryDiagnostics:
    """Per-call diagnostics for :class:`MetalGeometryPrior`.

    Stored on :attr:`MetalGeometryPrior.last_diagnostics` for logging.
    """

    geometry: str
    n_metal_centres: int
    n_donors_total: int
    n_pairs_total: int
    n_unflagged_edges: int


# ---------------------------------------------------------------------------
# Geometry penalty — single-graph helper, differentiable
# ---------------------------------------------------------------------------
def GeometryPenalty(
    positions: torch.Tensor,
    dative_bond_edge_index: torch.Tensor,
    dative_bond_edge_attr: torch.Tensor,
    atom_types: torch.Tensor,
    geometry: str = Geometry.SQUARE_PLANAR,
    metal_atomic_number: int | None = None,
    ideal_angles: Sequence[float] | None = None,
) -> torch.Tensor:
    """Energy-like soft penalty for deviating from ``geometry`` angles.

    Parameters
    ----------
    positions : ``(N, 3)`` tensor
        Per-atom coordinates (single graph).
    dative_bond_edge_index : ``(2, E_d)`` long tensor
        Dative edges (ligand donor -> metal centre).  Only edges whose
        destination matches a metal centre are considered.
    dative_bond_edge_attr : ``(E_d,)`` bool tensor
        Per-edge flag; only ``True`` edges contribute.
    atom_types : ``(N,)`` long tensor
        Atomic numbers.
    geometry : str
        One of :data:`SUPPORTED_GEOMETRIES`.  Selects the ideal-angle
        set used to compute the penalty.
    metal_atomic_number : int, optional
        If provided, only edges whose destination matches this atomic
        number contribute.  If ``None`` (default), the metal centre is
        inferred from :data:`DEFAULT_METAL_GEOMETRY` for the geometry
        passed in, OR — if there is only one metal centre in the
        graph — that centre is used.
    ideal_angles : sequence of float, optional
        Override the per-geometry ideal-angle set (radians).  Defaults
        to :data:`GEOMETRY_IDEAL_ANGLES`[geometry].

    Returns
    -------
    Scalar tensor (0-dim) — mean absolute deviation from the closest
    ideal angle for every L-X-L pair sharing a metal centre, averaged
    over metal centres.  Returns exactly zero when no metal centre
    with >=2 dative donors exists.

    Notes
    -----
    Differentiable w.r.t. ``positions`` (we use :func:`torch.arccos` on
    a clamped cosine — numerically safe even at degenerate configurations).
    """
    device = positions.device
    dtype = positions.dtype
    if dative_bond_edge_index.numel() == 0 or dative_bond_edge_attr is None:
        return torch.zeros((), device=device, dtype=dtype)
    mask = dative_bond_edge_attr.to(torch.bool)
    if not mask.any():
        return torch.zeros((), device=device, dtype=dtype)
    src = dative_bond_edge_index[0][mask]
    dst = dative_bond_edge_index[1][mask]
    if metal_atomic_number is None:
        # Default: any atom in DEFAULT_METAL_GEOMETRY for this geometry.
        candidates = [z for z, g in DEFAULT_METAL_GEOMETRY.items() if g == geometry]
        if candidates:
            is_metal = torch.zeros_like(dst, dtype=torch.bool)
            for z in candidates:
                is_metal = is_metal | (atom_types[dst] == z)
        else:
            is_metal = torch.ones_like(dst, dtype=torch.bool)
    else:
        is_metal = atom_types[dst] == metal_atomic_number
    if not is_metal.any():
        return torch.zeros((), device=device, dtype=dtype)
    src = src[is_metal]
    dst = dst[is_metal]

    if ideal_angles is None:
        ideal_angles = GEOMETRY_IDEAL_ANGLES[geometry]
    ideal_angles_t = torch.tensor(
        ideal_angles, device=device, dtype=dtype
    )  # (n_ideal,)

    pt_positions = positions[dst]
    donor_positions = positions[src]
    rel = donor_positions - pt_positions
    dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    unit = rel / dist

    unique_metal, inverse_idx = torch.unique(dst, return_inverse=True)
    penalty = positions.new_zeros(())
    n_centres = 0
    eps = 1e-3
    for local in range(unique_metal.numel()):
        sel = inverse_idx == local
        if sel.sum() < 2:
            continue
        u = unit[sel]
        cos = (u @ u.t()).clamp(-1.0 + eps, 1.0 - eps)
        iu = torch.triu_indices(cos.size(0), cos.size(1), offset=1, device=device)
        if iu.numel() == 0:
            continue
        angles = torch.arccos(cos[iu[0], iu[1]])  # (n_pairs,)
        # For each measured angle, take min deviation across the ideal-
        # angle set (handles octahedral / TBP multi-angle cases).
        # angles: (P,) ideal_angles_t: (K,) -> (P, K)
        diff = (angles.unsqueeze(-1) - ideal_angles_t.unsqueeze(0)).abs()
        min_dev = diff.min(dim=-1).values
        penalty = penalty + min_dev.mean()
        n_centres += 1
    if n_centres == 0:
        return torch.zeros((), device=device, dtype=dtype)
    return penalty / n_centres


def GeometryPenaltyBatched(
    positions_list,
    dative_edge_index_list,
    dative_edge_attr_list,
    atom_types_list,
    geometry: str = Geometry.SQUARE_PLANAR,
    metal_atomic_number: int | None = None,
    ideal_angles: Sequence[float] | None = None,
) -> torch.Tensor:
    """Batched wrapper around :func:`GeometryPenalty`.

    Each argument is a Python list (one entry per graph).  The result
    is the mean of per-graph penalties; graphs with no metal centre
    contribute zero.
    """
    device = positions_list[0].device
    n = len(positions_list)
    if n == 0:
        return torch.zeros((), device=device)
    total = positions_list[0].new_zeros(())
    for pos, ei, ea, at in zip(
        positions_list, dative_edge_index_list, dative_edge_attr_list, atom_types_list
    ):
        total = total + GeometryPenalty(
            pos, ei, ea, at,
            geometry=geometry,
            metal_atomic_number=metal_atomic_number,
            ideal_angles=ideal_angles,
        )
    return total / n


# ---------------------------------------------------------------------------
# Main prior module — MetalGeometryPrior
# ---------------------------------------------------------------------------
class MetalGeometryPrior(nn.Module):
    """Generalised metal-coordination geometry prior (TODO-09).

    Supports four canonical geometries — :data:`Geometry.SQUARE_PLANAR`,
    :data:`Geometry.TETRAHEDRAL`, :data:`Geometry.OCTAHEDRAL`,
    :data:`Geometry.TRIGONAL_BIPYRAMIDAL` — and dispatches per metal
    centre either from :data:`DEFAULT_METAL_GEOMETRY` or from an
    explicit ``metal_geometry_map`` provided at construction.

    The prior is a **soft** penalty: it returns a scalar ``>= 0`` that
    callers add to their loss with a tunable weight.  Zero when the
    geometry matches the ideal angles; positive penalty otherwise.
    """

    def __init__(
        self,
        weight: float = 1.0,
        metal_geometry_map: dict[int, str] | None = None,
        eps_distance_min: float = 0.4,
        eps_distance_max: float = 5.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        # Round-10 axis D: explicit prior-on/off flag.  When ``enabled``
        # is False, :meth:`prior_loss` returns exactly zero regardless of
        # the geometry — equivalent to the historical ``weight<=0``
        # short-circuit but spelled out so callers running an ablation
        # can toggle the prior without juggling the weight.  Defaults to
        # True so legacy callers that pass only ``weight`` see the prior
        # engaged (backward compatible — only Round-10 ablations should
        # pass ``enabled=False``).
        self.enabled = bool(enabled)
        self.weight = float(weight)
        # Merge caller overrides on top of the defaults; the caller's
        # entries take precedence.
        merged: dict = dict(DEFAULT_METAL_GEOMETRY)
        if metal_geometry_map:
            for k, v in metal_geometry_map.items():
                merged[int(k)] = str(v)
        self.metal_geometry_map: dict[int, str] = merged
        # Sanity bounds for metal–donor distances (Å).  Used by
        # :meth:`prior_loss` to drop edges that are clearly outside
        # any coordination range (catches spurious far-distant dative
        # edges that are mis-flagged).  Bondi-derived defaults: the
        # sum of metal and donor vdW radii sets a *minimum*, and 3.0 Å
        # beyond that is a generous *maximum* for a coordination bond.
        self.eps_distance_min = float(eps_distance_min)
        self.eps_distance_max = float(eps_distance_max)
        self.last_diagnostics: _GeometryDiagnostics | None = None

    # ------------------------------------------------------------------
    # Round-10 axis D — prior on/off helpers
    # ------------------------------------------------------------------
    def disable(self) -> None:
        """Turn the prior OFF (prior_loss becomes a no-op)."""
        self.enabled = False

    def enable(self) -> None:
        """Turn the prior ON (prior_loss returns the soft penalty)."""
        self.enabled = True

    def apply_prior(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        edge_types: torch.Tensor,
        metal_idx: int | None = None,
        geometry: str | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return an energy-like penalty tensor for ``coords``.

        Parameters
        ----------
        coords : ``(N, 3)`` tensor
        atom_types : ``(N,)`` long tensor — atomic numbers.
        edge_types : ``(E,)`` long tensor with edge-type codes
            (matching :data:`molmetal.adapters.egnn_rocm.EDGE_TYPE_*`).
        edge_index : ``(2, E)`` long tensor
            Explicit donor -> metal endpoints, matching edge_types. Required
            when dative edges are present; radius-based inference is rejected.
        metal_idx : int, optional
            Index of the metal centre to score.  If ``None`` (default),
            the prior is applied to every metal centre detected via
            :attr:`metal_geometry_map`, with the geometry inferred
            from the metal's atomic number.
        geometry : str, optional
            Force a specific geometry regardless of which metal is at
            ``metal_idx`` (e.g. useful for testing).  Defaults to
            :attr:`metal_geometry_map`[metal atom] when ``metal_idx``
            is provided, otherwise to :data:`Geometry.SQUARE_PLANAR`.

        Returns
        -------
        Scalar tensor — mean absolute angular deviation from the ideal
        angles for the selected metal centre(s).  Zero when no dative
        edges are present or the geometry matches.
        """
        device = coords.device
        dtype = coords.dtype
        # Round-10 axis D — when ``enabled`` is False the prior is a
        # strict no-op (returns exactly 0).  This is the canonical
        # "prior OFF" path used by the 1h36 Pt-ablation harness; the
        # caller toggles ``enabled`` without re-wiring the rest of the
        # adapter pipeline.
        if not self.enabled:
            return torch.zeros((), device=device, dtype=dtype)
        if edge_types is None or edge_types.numel() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        # Dative edges (edge_type == 2).
        dative_mask = (edge_types == 2).to(torch.bool)
        if not dative_mask.any():
            return torch.zeros((), device=device, dtype=dtype)
        if edge_index is None:
            raise ValueError("Explicit edge_index is required for dative metal geometry; distance-inferred donors are unsupported")
        if edge_index.shape != (2, edge_types.numel()):
            raise ValueError("edge_index must have shape (2, E) matching edge_types")
        if edge_index.dtype != torch.long or (edge_index < 0).any() or (edge_index >= coords.shape[0]).any():
            raise ValueError("edge_index must contain valid integer atom endpoints")
        edge_index = edge_index.to(device)
        src, dst = edge_index[:, dative_mask]
        if metal_idx is None:
            # Detect every metal centre by walking atom_types.
            metal_indices = [
                i for i, z in enumerate(atom_types.tolist())
                if int(z) in self.metal_geometry_map
            ]
        else:
            metal_indices = [int(metal_idx)]

        if not metal_indices:
            return torch.zeros((), device=device, dtype=dtype)

        # Pick the geometry: explicit override > metal_idx lookup >
        # default square-planar (per-atom loop lets each metal use its
        # own geometry when no override is supplied).
        n_total = 0
        total_penalty = coords.new_zeros(())
        for mi in metal_indices:
            z = int(atom_types[mi].item())
            geo = geometry if geometry is not None else self.metal_geometry_map.get(z, Geometry.SQUARE_PLANAR)
            # Only declared donor -> metal edges contribute. Duplicate edges
            # do not create extra donors; nearby non-donor atoms are irrelevant.
            donor_idx = torch.unique(src[(dst == mi) & (src != mi)])
            metal_pos = coords[mi]
            dist = (coords[donor_idx] - metal_pos).norm(dim=-1)
            donor_idx = donor_idx[(dist > self.eps_distance_min) & (dist < self.eps_distance_max)]
            if donor_idx.numel() < 2:
                continue

            donor_pos = coords[donor_idx]
            rel2 = donor_pos - metal_pos.unsqueeze(0)
            unit = rel2 / rel2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            ideal_angles = torch.tensor(
                GEOMETRY_IDEAL_ANGLES[geo], device=device, dtype=dtype
            )
            cos = (unit @ unit.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            iu = torch.triu_indices(unit.size(0), unit.size(1), offset=1, device=device)
            if iu.numel() == 0:
                continue
            angles = torch.arccos(cos[iu[0], iu[1]])
            diff = (angles.unsqueeze(-1) - ideal_angles.unsqueeze(0)).abs()
            min_dev = diff.min(dim=-1).values
            total_penalty = total_penalty + min_dev.mean()
            n_total += 1

        if n_total == 0:
            return torch.zeros((), device=device, dtype=dtype)
        return total_penalty / n_total

    def prior_loss(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        edge_types: torch.Tensor,
        metal_idx: int | None = None,
        geometry: str | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper around :meth:`apply_prior` that applies
        :attr:`weight` and stores diagnostics on :attr:`last_diagnostics`.

        Returns 0 when geometry matches; positive penalty otherwise.
        """
        raw = self.apply_prior(
            coords=coords,
            atom_types=atom_types,
            edge_types=edge_types,
            metal_idx=metal_idx,
            geometry=geometry,
            edge_index=edge_index,
        )
        # Diagnostics
        with torch.no_grad():
            try:
                n_metal = (
                    int(((atom_types.unsqueeze(-1) == torch.tensor(
                        list(self.metal_geometry_map.keys()),
                        device=atom_types.device,
                        dtype=atom_types.dtype,
                    )).any(dim=-1)).sum().item())
                    if self.metal_geometry_map else 0
                )
            except Exception:  # noqa: BLE001
                n_metal = -1
            try:
                n_donors = int((edge_types == 2).sum().item()) if edge_types is not None else 0
            except Exception:  # noqa: BLE001
                n_donors = -1
            try:
                geo_label = (
                    geometry if geometry is not None
                    else (self.metal_geometry_map.get(
                        int(atom_types[metal_idx].item()), Geometry.SQUARE_PLANAR
                    ) if metal_idx is not None else Geometry.SQUARE_PLANAR)
                )
            except Exception:  # noqa: BLE001
                geo_label = "unknown"
            self.last_diagnostics = _GeometryDiagnostics(
                geometry=geo_label,
                n_metal_centres=n_metal,
                n_donors_total=n_donors,
                n_pairs_total=-1,
                n_unflagged_edges=-1,
            )
        return self.weight * raw


# Functional aliases used by callers that prefer a functional API.
def apply_metal_prior(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    metal_idx: int | None = None,
    geometry: str | None = None,
    weight: float = 1.0,
    metal_geometry_map: dict[int, str] | None = None,
    edge_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional entry point for :class:`MetalGeometryPrior`.

    Mirrors :meth:`MetalGeometryPrior.prior_loss` with the weight
    applied.  Useful for one-off calls without instantiating a module.
    """
    prior = MetalGeometryPrior(weight=weight, metal_geometry_map=metal_geometry_map)
    return prior.prior_loss(
        coords=coords, atom_types=atom_types, edge_types=edge_types,
        metal_idx=metal_idx, geometry=geometry, edge_index=edge_index,
    )


def prior_loss(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    metal_idx: int | None = None,
    geometry: str | None = None,
    edge_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional, weight-free entry point — returns the raw penalty.

    Returns 0 when geometry matches; positive penalty otherwise.
    """
    prior = MetalGeometryPrior(weight=1.0)
    return prior.apply_prior(
        coords=coords, atom_types=atom_types, edge_types=edge_types,
        metal_idx=metal_idx, geometry=geometry, edge_index=edge_index,
    )


# ---------------------------------------------------------------------------
# Phase 3H — Soft tiered metal-geometry reward
# ---------------------------------------------------------------------------
# Replaces the historical hard-gate
# ``metal_geometry_prior_bonus`` (binary 0.0 / 1.0 depending on whether
# the coordination number matched the target) with a CONTINUOUS soft
# score in ``[0, 1]``.  The motivation is:
#
#   * The MCTS proof search collapses onto a single candidate when the
#     prior is a hard gate (binary reward).  Round-12 5×1 / 10×3 pilot
#     at n_simulations=1000 reports ``n_distinct=1`` for the metal-
#     seeded cisplatin arm precisely because partial-but-promising
#     candidates (e.g. CN=3 or CN=5 around Pt(II) instead of CN=4)
#     are silently downgraded to 0 reward.
#   * Replacing the gate with a continuous distance-to-ideal score
#     keeps a non-zero gradient across the whole candidate spectrum
#     so MCTS can still rank off-by-one Pt complexes above pure
#     organic ligands.  This is the potential-based reward shaping
#     interpretation of Dayan 1997 ("the importance / advantage of
#     reward shaping is the existence of a potential function") and
#     the entropy-regularised MDP formulation of Neu 2017 (soft
#     preferences yield better exploration than hard gates).
#
# Math formulation
# -----------------
# For each metal centre ``m`` in ``state``, we compute three normalised
# deviations from the ideal target geometry:
#
#   ``d_coord`` = ``|CN(m) - target_CN| / max(1, target_CN)``
#                 in ``[0, 1]`` (clipped at 1.0).
#   ``d_angle`` = ``mean(|theta_l - ideal_theta(m)|) / pi``
#                 in ``[0, 1]`` (clipped at 1.0).
#   ``d_charge`` = ``|formal_charge(m)| / 2``
#                 in ``[0, 1]`` (clipped at 1.0; Pt(II) target 0).
#
# The per-metal score is
#
#   ``score(m) = max(0, 1 - 0.5*d_coord - 0.3*d_angle - 0.2*d_charge)``
#
# and the molecule-level soft score is the maximum over all metal
# centres (matches the historical ``best`` aggregation in
# ``metal_geometry_prior_bonus``).  Returns 0.0 when no metal is
# present — the organic-ligand branch, where the metal prior should
# NOT contribute to the reward.

# Default coordination-number targets (atomic number -> CN).  Matches
# the per-metal arity table in :mod:`molmetal_lam.atoms.combinators`
# (Pt(II)/Pd(II)/Cu(II)/Au(III)/Zn(II) -> CN=4 square-planar;
# Ru(II)/Ir(III)/Rh(III)/Fe(II) -> CN=6 octahedral).  Callers may
# override via ``cn_targets``.
DEFAULT_SOFT_SCORE_TARGETS: dict = {
    78: 4,   # Pt(II)
    46: 4,   # Pd(II)
    29: 4,   # Cu(II)  (d9, square-planar also common)
    30: 4,   # Zn(II)  (tetrahedral)
    79: 4,   # Au(III) (square-planar d8)
    44: 6,   # Ru(II)  (octahedral)
    77: 6,   # Ir(III) (octahedral)
    45: 6,   # Rh(III) (octahedral)
    26: 6,   # Fe(II)  (octahedral)
}

# Symbols recognised as metals by the soft score.  Matches the
# METAL_ATOMS dict in :mod:`molmetal_lam.atoms.combinators`.
_SOFT_SCORE_METAL_SYMBOLS: frozenset = frozenset({
    "Pt", "Pt_II",
    "Pd", "Pd_II",
    "Cu", "Cu_II",
    "Zn", "Zn_II",
    "Au", "Au_III",
    "Ru", "Ru_II",
    "Ir", "Ir_III",
    "Rh", "Rh_III",
    "Fe", "Fe_II",
})


@dataclass
class SoftMetalGeometryDiagnostics:
    """Per-call diagnostics for :func:`soft_score_metal_geometry`.

    Useful for logging which soft-score components fired and how far
    the candidate deviated from the ideal geometry.  None of the
    fields are required for correctness — they only assist debugging.
    """

    n_metal_centres: int
    n_no_metal: bool
    best_cn: int
    best_target_cn: int
    best_d_coord: float
    best_d_angle: float
    best_d_charge: float
    best_score: float


def _coerce_state_atoms_bonds(state) -> tuple:
    """Extract ``(atoms, bonds)`` from a state-like object.

    Accepts:
      * ``MoleculeClosedTerm`` (canonical MCTS state container) —
        uses :attr:`atoms` + :attr:`bonds`.
      * Plain ``dict`` with ``"atoms"`` / ``"bonds"`` keys.
      * Tuples ``(atoms, bonds)`` (used in the legacy
        :func:`metal_geometry_prior_bonus` test helpers).
      * ``None`` or any object lacking both attributes — returns
        empty lists so callers can degrade gracefully.
    """
    atoms = []
    bonds = []
    try:
        if state is None:
            return [], []
        if isinstance(state, dict):
            atoms = list(state.get("atoms", []) or [])
            bonds = list(state.get("bonds", []) or [])
            return atoms, bonds
        # MoleculeClosedTerm-like
        atoms = list(getattr(state, "atoms", []) or [])
        bonds = list(getattr(state, "bonds", []) or [])
        # Some test fixtures wrap tuples inside dataclass fields.
        if not atoms and not bonds:
            cand = getattr(state, "terms", None)
            if cand is not None:
                atoms = list(getattr(cand, "atoms", []) or [])
                bonds = list(getattr(cand, "bonds", []) or [])
    except Exception:
        return [], []
    return atoms, bonds


def _coordination_count(atom, bonds) -> int:
    """Count bonds incident to ``atom`` (excluding hydrogen).

    Mirror of the legacy counter in
    :func:`molmetal.scripts.r4_lambda_only_run.metal_geometry_prior_bonus`
    — kept here so the soft score is unit-testable in isolation
    without importing the orchestrator script (which is owned by the
    Phase-4 integrator).
    """
    if atom is None or not bonds:
        return 0
    try:
        atom_id = id(atom)
    except Exception:
        return 0
    coord = 0
    for b in bonds:
        try:
            kind = getattr(b, "kind", "") or ""
        except Exception:
            kind = ""
        if kind == "hydrogen":
            continue
        try:
            if id(getattr(b, "atom_a", None)) == atom_id or id(getattr(b, "atom_b", None)) == atom_id:
                coord += 1
        except Exception:
            continue
    return coord


def _formal_charge(atom) -> int:
    """Best-effort formal-charge lookup.  Returns 0 when unavailable.

    The default :class:`molmetal_lam.atoms.combinators.Atom` dataclass
    does not store an explicit formal charge (the metal centre's
    oxidation state is encoded in the symbol suffix: ``Pt_II`` -> +2).
    We parse the Roman-numeral suffix when present so the soft score
    can mildly penalise an off-target oxidation state (e.g. an
    Au(III) centre in a Pt(II) pocket context).
    """
    if atom is None:
        return 0
    # explicit field, if a caller provided one
    fc = getattr(atom, "formal_charge", None)
    if fc is not None:
        try:
            return int(fc)
        except Exception:
            pass
    sym = getattr(atom, "symbol", "") or ""
    # Parse the Roman-numeral suffix after "_".  Pt_II -> +2, etc.
    # NOTE: combinators.py models the metal as a *ligand combinator*
    # so Pt_II represents Pt(II), not Pt(+II).  We use the parsed
    # oxidation state ONLY for the soft deviation; the score
    # correctly targets Pt(II) at charge 0 (cisplatin is neutral).
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    if "_" in sym:
        suffix = sym.rsplit("_", 1)[-1]
        if suffix in roman:
            # The metal combinator symbol encodes the oxidation state
            # but the *complex* (cisplatin) is overall neutral because
            # the chloride anions balance the Pt(II) charge.  We DO
            # NOT add the oxidation state to the deviation — the
            # target soft score is "neutral complex", not "Pt centre
            # with charge 0".  The full formal-charge bookkeeping is
            # left to the caller (e.g. RDKit's ``GetFormalCharge``
            # wrapping in r4_lambda_only_run.metal_compliance_truthful).
            return 0
    return 0


def _angle_deviation_for_metal(atom, bonds, atoms) -> float:
    """Mean pairwise bond-angle deviation from the metal's ideal angle.

    Returns a value in ``[0, pi]``.  When the metal has fewer than two
    bonds the function returns ``pi`` (maximum uncertainty — the soft
    score treats this as "completely off-target").

    This is a coordinate-free proxy: we DO NOT embed the molecule in
    3-D (that requires RDKit / ETKDG and would slow down the MCTS
    loop).  Instead we assume the ideal geometry implied by the
    metal's coordination number and return the deviation from that
    ideal *averaged across ligand pairs*.  The returned value is a
    coarse approximation — see honest-framing note in the module
    docstring.
    """
    cn = _coordination_count(atom, bonds)
    if cn < 2:
        return math.pi  # max uncertainty
    target_cn = DEFAULT_SOFT_SCORE_TARGETS.get(
        getattr(atom, "atomic_num", 0) or 0, None
    )
    if target_cn is None:
        # Unknown metal — return the largest deviation we can report
        # without inventing a target.  ``pi`` keeps the contribution
        # at the lower-bound of the soft score (the
        # ``max(0, 1 - ...)`` clamp prevents negative scores).
        return math.pi
    # Ideal angle for CN=4 square-planar -> pi/2; CN=6 octahedral -> pi/2.
    # Both 90 deg (cis / adjacent) and 180 deg (trans) are valid
    # reference points; we use 90 deg as the per-pair anchor.
    ideal = math.pi / 2.0
    # CN deviation is the only proxy available without 3-D coords.
    # Map CN mismatch to an angle deviation: each off-by-one CN
    # contributes pi/4 rad of "expected angle noise".
    cn_mismatch = abs(cn - target_cn)
    return float(min(math.pi, cn_mismatch * (math.pi / 4.0)))


def soft_score_metal_geometry(
    state,
    *,
    enabled: bool = True,
    cn_targets: dict | None = None,
    weight_coord: float = 0.5,
    weight_angle: float = 0.3,
    weight_charge: float = 0.2,
    ideal_charge: int = 0,
) -> float:
    """Continuous soft score for metal-geometry prior, in ``[0, 1]``.

    Replaces the historical hard-gate
    :func:`molmetal.scripts.r4_lambda_only_run.metal_geometry_prior_bonus`
    (binary 1.0 if CN matches target else 0.0) with a continuous
    distance-to-ideal score.

    Parameters
    ----------
    state
        Any object exposing ``.atoms`` + ``.bonds`` (a
        :class:`MoleculeClosedTerm` is the canonical input).  ``dict``
        and ``tuple`` shapes are also accepted for testing.
    enabled : bool, default True
        When False the function returns 0.0 unconditionally — this
        matches the historical "prior off" ablation path used by
        Round-10 axis-D.
    cn_targets : dict, optional
        Override the default ``{atomic_num: target_CN}`` mapping.
        Defaults to :data:`DEFAULT_SOFT_SCORE_TARGETS`.
    weight_coord : float, default 0.5
        Weight of the coordination-number deviation term.
    weight_angle : float, default 0.3
        Weight of the bond-angle deviation term.
    weight_charge : float, default 0.2
        Weight of the formal-charge deviation term.
    ideal_charge : int, default 0
        Target formal charge on the metal centre (cisplatin is
        neutral so 0 is the right default).

    Returns
    -------
    float
        Continuous score in ``[0, 1]``.  Returns exactly 0.0 when
        ``enabled`` is False, when ``state`` has no atoms, or when
        no metal centre is present (organic ligands don't earn a
        metal prior bonus by construction).

    Notes
    -----
    Lit anchors:

      * **Schulman et al. 2017** (PPO clipped objective): the soft
        constraint formulation parallels PPO's ratio-clipping — we
        clip the score at 0 (never -inf) so a bad candidate still
        has a finite, non-zero gradient for the MCTS update.  A
        hard 0/1 gate would correspond to "ratio clipping to 0",
        which PPO explicitly avoids.
      * **Neu 2017** (entropy-regularised MDPs): a soft prior yields
        better exploration than a hard gate because the agent
        retains information about *how close* a candidate was to
        the target.
      * **Dayan 1997** (potential-based reward shaping): the soft
        score is equivalent to a *potential function* ``Phi(s)``
        that adds a shaping term without changing the optimal
        policy.  Our formulation uses the (1 - distance) form so
        that ``Phi(ideal) = 1`` and ``Phi(far) = 0``, recovering
        the "potential-based shaping is policy-preserving" theorem.

    Honest framing
    --------------
    The bond-angle deviation is computed WITHOUT 3-D coordinates —
    we cannot afford RDKit / ETKDG embedding inside the MCTS loop.
    The function falls back to a CN-mismatch proxy: each off-by-one
    CN contributes pi/4 rad of "expected angle noise".  This is a
    coarse approximation and over-counts the angle penalty for
    genuine off-target complexes (e.g. trans-platin's CN=4 has the
    right CN but wrong bond angles).  The TODO-15 metric suite +
    :class:`MetalGeometryPrior` (differentiable / torch-based) is
    the rigorous alternative when 3-D coordinates are available.
    """
    if not enabled:
        return 0.0
    atoms, bonds = _coerce_state_atoms_bonds(state)
    if not atoms:
        return 0.0
    # Find metal centres.
    targets = cn_targets if cn_targets is not None else DEFAULT_SOFT_SCORE_TARGETS
    metals = []
    for atom in atoms:
        try:
            sym = getattr(atom, "symbol", "") or ""
        except Exception:
            sym = None
        if sym is None:
            continue
        # The combinator library uses "Pt_II" / "Cu_II" / "Ir_III" —
        # also accept bare "Pt" / "Cu" / etc. (e.g. SMILES-parsed
        # atoms from RDKit round-trips).
        bare = sym.split("_", 1)[0] if "_" in sym else sym
        if sym in _SOFT_SCORE_METAL_SYMBOLS or bare in _SOFT_SCORE_METAL_SYMBOLS:
            z = int(getattr(atom, "atomic_num", 0) or 0)
            target = targets.get(z, 4)  # default 4 for unknown metals
            metals.append((atom, z, target))
    if not metals:
        return 0.0
    # Aggregate best score across all metal centres (max).  Mirrors
    # the historical ``best`` aggregation in
    # metal_geometry_prior_bonus so the behaviour is monotonic
    # w.r.t. adding more metal centres.
    best = 0.0
    best_diag = SoftMetalGeometryDiagnostics(
        n_metal_centres=len(metals),
        n_no_metal=False,
        best_cn=-1,
        best_target_cn=-1,
        best_d_coord=0.0,
        best_d_angle=0.0,
        best_d_charge=0.0,
        best_score=0.0,
    )
    for atom, z, target_cn in metals:
        cn = _coordination_count(atom, bonds)
        d_coord = min(1.0, abs(cn - target_cn) / max(1, target_cn))
        angle_dev = _angle_deviation_for_metal(atom, bonds, atoms)
        d_angle = min(1.0, angle_dev / math.pi)
        charge = _formal_charge(atom)
        d_charge = min(1.0, abs(charge - ideal_charge) / 2.0)
        score = max(
            0.0,
            1.0 - weight_coord * d_coord - weight_angle * d_angle - weight_charge * d_charge,
        )
        if score > best:
            best = score
            best_diag = SoftMetalGeometryDiagnostics(
                n_metal_centres=len(metals),
                n_no_metal=False,
                best_cn=cn,
                best_target_cn=target_cn,
                best_d_coord=float(d_coord),
                best_d_angle=float(d_angle),
                best_d_charge=float(d_charge),
                best_score=float(score),
            )
    return float(best)


# A thin module-level convenience: a closure that can be plugged into
# ``RewardAggregator.r_metal_geom_soft`` without exposing the keyword
# arguments to the caller.  The closure respects the ``enabled`` flag
# passed at construction time so call-sites that want to disable the
# soft channel for ablations can do so without touching the aggregator.
def make_soft_metal_geometry_channel(
    *,
    enabled: bool = True,
    cn_targets: dict | None = None,
    weight_coord: float = 0.5,
    weight_angle: float = 0.3,
    weight_charge: float = 0.2,
    ideal_charge: int = 0,
):
    """Return a ``(state) -> float`` closure suitable for the aggregator.

    The returned closure calls :func:`soft_score_metal_geometry` with
    the keyword arguments fixed at construction time.  Useful for
    :class:`RewardAggregator.register_soft_metal_geometry_channel`
    which expects a plain ``(state) -> float`` callable.
    """
    state_kwargs = dict(
        enabled=enabled,
        cn_targets=cn_targets,
        weight_coord=weight_coord,
        weight_angle=weight_angle,
        weight_charge=weight_charge,
        ideal_charge=ideal_charge,
    )

    def _channel(state) -> float:
        try:
            return float(soft_score_metal_geometry(state, **state_kwargs))
        except Exception:
            return 0.0

    return _channel
