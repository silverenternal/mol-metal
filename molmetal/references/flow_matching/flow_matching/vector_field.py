"""Backward-compatible re-export shim for the flow-matching vector field.

Historically this module hosted both the functional helpers
(:func:`interpolate`, :func:`target_velocity`) and the
:class:`ConditionalFlowMatchingLoss` training objective.  Those have
been split into:

* :mod:`flow_matching.interpolation` - the small functional helpers.
* :mod:`flow_matching.loss` - the :class:`LossOutput` dataclass and
  :class:`ConditionalFlowMatchingLoss`.

This module is kept as a thin re-export shim so existing imports such
as ::

    from flow_matching import (
        ConditionalFlowMatchingLoss, LossOutput, interpolate, target_velocity,
    )

keep working without change.
"""

from __future__ import annotations

from flow_matching.interpolation import interpolate, target_velocity
from flow_matching.loss import ConditionalFlowMatchingLoss, LossOutput

__all__ = [
    "interpolate",
    "target_velocity",
    "LossOutput",
    "ConditionalFlowMatchingLoss",
]
