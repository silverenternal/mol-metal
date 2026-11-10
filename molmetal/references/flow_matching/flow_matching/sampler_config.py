"""Configuration container for the flow-matching sampler.

This module defines :class:`SamplerConfig`, the lightweight
dataclass that captures the integration hyperparameters used by
:class:`flow_matching.sampler.FlowMatchingSampler`.  It is kept in
its own module so that callers can construct a configuration without
triggering the (relatively heavy) imports required by the sampler
itself (``torch``, :mod:`models.velocity_net`,
:mod:`triton_kernels`).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------
@dataclass
class SamplerConfig:
    """Convenience container for sampler hyperparameters.

    Attributes
    ----------
    n_steps : int
        Number of integration steps between ``t=0`` and ``t=1``.
    method : ``"euler"`` | ``"rk4"``
        Integration scheme.
    t_start : float, default 0.0
        Start of the integration interval.
    t_end : float, default 1.0
        End of the integration interval.
    """

    n_steps: int = 100
    method: str = "euler"
    t_start: float = 0.0
    t_end: float = 1.0

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1.")
        if self.method not in ("euler", "rk4"):
            raise ValueError(
                f"`method` must be 'euler' or 'rk4'; got {self.method!r}."
            )
        if self.t_end <= self.t_start:
            raise ValueError(
                f"`t_end` ({self.t_end}) must be strictly greater than "
                f"`t_start` ({self.t_start})."
            )


__all__ = ["SamplerConfig"]