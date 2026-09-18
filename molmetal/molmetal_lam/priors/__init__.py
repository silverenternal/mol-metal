"""Geometric and chemical priors for SBDD / lambda-elevation tasks.

Public API
----------
- :mod:`.metal_geometry` — square-planar Pt(II), tetrahedral, octahedral,
  trigonal-bipyramidal priors, plus the original :class:`SquarePlanarPtII`
  retained for backward compatibility.
"""

from __future__ import annotations

from .metal_geometry import (
    # Square-planar legacy API (kept for backward compatibility)
    SquarePlanarPtII,
    square_planar_penalty,
    square_planar_penalty_batched,
    PT_ATOMIC_NUMBER,
    IDEAL_SQUARE_PLANAR_ANGLE,
    # Generalised TODO-09 API
    MetalGeometryPrior,
    Geometry,
    GeometryPenalty,
    GeometryPenaltyBatched,
    bondi_van_der_waals_radius,
    METAL_PRIOR_KIND,
    apply_metal_prior,
    prior_loss,
)

__all__ = [
    "SquarePlanarPtII",
    "square_planar_penalty",
    "square_planar_penalty_batched",
    "PT_ATOMIC_NUMBER",
    "IDEAL_SQUARE_PLANAR_ANGLE",
    "MetalGeometryPrior",
    "Geometry",
    "GeometryPenalty",
    "GeometryPenaltyBatched",
    "bondi_van_der_waals_radius",
    "METAL_PRIOR_KIND",
    "apply_metal_prior",
    "prior_loss",
]
from .metal_hydration import MetalHydrationAnalyzer
__all__.append("MetalHydrationAnalyzer")
from .anticancer_metric_suite import AnticancerMetricSuite
__all__.append("AnticancerMetricSuite")
