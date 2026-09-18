"""Linear interpolation and conditional target velocity for flow matching.

This module hosts the small functional helpers that underpin the
flow-matching training objective of :mod:`MolFlow-Triton`:

* :func:`interpolate` - linear interpolation
  ``x_t = (1 - t) * x0 + t * x1`` between a noise sample ``x0`` and a
  data sample ``x1``.
* :func:`target_velocity` - the conditional flow-matching target
  ``v_t = x1 - x0``.

These are deliberately split from :class:`ConditionalFlowMatchingLoss`
so they can be reused (e.g. by visualization tools or by alternative
loss formulations) without pulling in :class:`VelocityNet`.
"""

from __future__ import annotations

import torch


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation ``x_t = (1 - t) * x0 + t * x1``.

    Parameters
    ----------
    x0 : ``(B, ...)`` tensor
        Source distribution (typically noise).
    x1 : ``(B, ...)`` tensor
        Target distribution (data).  Must share shape with ``x0``.
    t : ``(B,)`` or ``(B, 1)`` tensor
        Flow-matching time in ``[0, 1]``.

    Returns
    -------
    ``(B, ...)`` tensor with the same shape and dtype as ``x0`` / ``x1``.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"`x0` and `x1` must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    if t.dim() not in (1, 2):
        raise ValueError(f"`t` must be (B,) or (B, 1); got {tuple(t.shape)}")
    if t.shape[0] != x0.shape[0]:
        raise ValueError(
            f"`t` batch size ({t.shape[0]}) must match `x0` batch size "
            f"({x0.shape[0]})."
        )

    t_b = t.view(-1, *([1] * (x0.dim() - 1))).to(dtype=x0.dtype)
    return (1.0 - t_b) * x0 + t_b * x1


def target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Return the conditional flow-matching target velocity ``x1 - x0``.

    Parameters
    ----------
    x0, x1 : ``(B, ...)`` tensors
        Source and target.  Must share shape and dtype.

    Returns
    -------
    ``(B, ...)`` tensor.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"`x0` and `x1` must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    return x1 - x0


__all__ = ["interpolate", "target_velocity"]
