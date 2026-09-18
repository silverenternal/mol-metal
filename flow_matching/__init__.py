"""Flow-matching core for MolFlow-Triton.

This package collects the (conditional) flow-matching logic that sits
on top of the :mod:`models` and :mod:`triton_kernels` layers:

* :mod:`flow_matching.interpolation` - the small functional helpers
  :func:`interpolate` (linear ``x_t = (1 - t) x0 + t x1``) and
  :func:`target_velocity` (``x1 - x0``).
* :mod:`flow_matching.loss` - the :class:`LossOutput` dataclass and
  the :class:`ConditionalFlowMatchingLoss` that trains a
  :class:`models.velocity_net.VelocityNet`.
* :mod:`flow_matching.vector_field` - thin backward-compatible
  re-export shim that still surfaces ``interpolate``,
  ``target_velocity``, :class:`LossOutput` and
  :class:`ConditionalFlowMatchingLoss` for legacy imports.
* :mod:`flow_matching.optimal_transport` - the (optional) OT-CFM
  pairing helper :func:`compute_ot_plan` that returns the
  permutation ``idx`` such that ``x1[idx]`` is paired with ``x0``.
  Defaults to a random pairing and supports an exact Hungarian
  solver on a squared-distance cost when ``scipy`` is available.
* :mod:`flow_matching.sampler` - :class:`FlowMatchingSampler`,
  which drives :mod:`triton_kernels.ode_solver` (Euler or RK4) to
  integrate the learned ODE from noise to data.

The package is intentionally framework-light: training loops live in
:mod:`scripts`, models in :mod:`models`, kernels in
:mod:`triton_kernels`.  Nothing here owns an optimizer or holds
gradient state across calls.
"""

from __future__ import annotations

from flow_matching.interpolation import interpolate, target_velocity
from flow_matching.loss import ConditionalFlowMatchingLoss, LossOutput
from .vector_field import (
    ConditionalFlowMatchingLoss as _VectorFieldConditionalFlowMatchingLoss,
    LossOutput as _VectorFieldLossOutput,
    interpolate as _VectorFieldInterpolate,
    target_velocity as _VectorFieldTargetVelocity,
)
from .optimal_transport import PlanMode, compute_ot_plan
from .sampler import FlowMatchingSampler
from .sampler_config import SamplerConfig

__all__ = [
    # interpolation
    "interpolate",
    "target_velocity",
    # loss
    "LossOutput",
    "ConditionalFlowMatchingLoss",
    # optimal transport
    "compute_ot_plan",
    "PlanMode",
    # sampler
    "FlowMatchingSampler",
    "SamplerConfig",
]

# Re-export aliases from the legacy ``vector_field`` module so existing
# imports like ``from flow_matching.vector_field import interpolate``
# keep working.  Keep the names private to avoid polluting ``__all__``.
__vector_field_reexports__ = {
    "_VectorFieldInterpolate": _VectorFieldInterpolate,
    "_VectorFieldTargetVelocity": _VectorFieldTargetVelocity,
    "_VectorFieldLossOutput": _VectorFieldLossOutput,
    "_VectorFieldConditionalFlowMatchingLoss": _VectorFieldConditionalFlowMatchingLoss,
}

__version__ = "0.1.0"
