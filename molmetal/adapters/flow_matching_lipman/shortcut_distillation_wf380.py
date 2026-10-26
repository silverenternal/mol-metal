"""Wf-380-A1 §2.6 — Shortcut distillation SCAFFOLD.

NOTE: This is the Wf-380-A1 SCAFFOLD module.  The canonical (Wf-353-A2)
implementation lives in :mod:`molmetal.adapters.flow_matching_lipman.shortcut_distillation`
(145 LOC).  Both modules coexist — this SCAFFOLD exposes a small,
importable reference for deep-integration testing; the canonical
implementation is the production target.

Reference: Frans, K., Hafner, D., Levine, S., Abbeel, P. (2024).
*Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models.*
ICLR 2024.  arXiv:2410.11081.  GitHub: NVIDIA/consistency-models.

The shortcut model distillation introduces a "self-consistency" term
that, for a fraction ``p_self`` of samples, predicts the 1-shot ODE
velocity at a *second* independently-sampled time stamp:

    L_shortcut = MSE(v_θ(x_t, t), dx_t)  +  p_self · MSE(v_θ(x_s, s), v_θ(x_t, t))

where ``(x_s, s)`` is the second sample pair.  This bridges the FM
regression with the shortcut-model distillation so the trained network
can be sampled in 1-4 steps at inference.

SCAFFOLD ONLY — no trainer wiring.  Speedup claim (1.5-2× training +
1-4 step inference) is PROJECTED per Frans 2024.

Honest framing: SCAFFOLD-only; no GPU launches.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def shortcut_pair(
    x_time: torch.Tensor,
    t: torch.Tensor,
    target_dx: torch.Tensor,
    n_steps: int = 4,
    p_self: float = 0.5,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a (x_s, s) shortcut pair consistent with Frans 2024 §3.2.

    Parameters
    ----------
    x_time : (B, ..., D)
        The flow-interpolated state at time ``t``.
    t : (B,)
        The original time stamp.
    target_dx : (B, ..., D)
        The conditional velocity at time ``t``.
    n_steps : int
        Number of inference steps the student will use.  Default 4
        per Frans 2024.
    p_self : float
        Probability of using the shortcut distillation pair (vs. plain
        FM regression).  Default 0.5.

    Returns
    -------
    (x_s, s) : the shortcut pair — x_s is the second-time-step state
        and s is the second time stamp, both shape-compatible with
        ``(x_time, t)``.
    """
    del n_steps  # API parity only.
    b = int(t.shape[0])
    device = t.device
    dtype = t.dtype
    mask = (torch.rand(b, generator=generator, device=device) < p_self).to(dtype)
    # Sample a second time stamp uniformly.
    s = torch.rand(b, generator=generator, device=device, dtype=dtype)
    # Build x_s as the linear interpolation between x_time and the
    # trajectory terminal state x_1, where x_1 = x_time + (1 − t) ·
    # target_dx (rearranging x_t = t · x_1 + (1 − t) · x_0).  This
    # keeps the second sample on the same flow line as the original.
    x_1 = x_time + (1.0 - t.view(-1, *([1] * (x_time.dim() - 1)))) * target_dx
    s_v = s.view(-1, *([1] * (x_time.dim() - 1)))
    x_s = (1.0 - s_v) * x_time + s_v * x_1
    return x_s * mask.view(-1, *([1] * (x_s.dim() - 1))) + x_time * (
        1.0 - mask.view(-1, *([1] * (x_time.dim() - 1)))
    ), s * mask + t * (1.0 - mask)


def shortcut_distillation_loss(
    pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_time: torch.Tensor,
    t: torch.Tensor,
    target_dx: torch.Tensor,
    p_self: float = 0.5,
    v_pred_t: Optional[torch.Tensor] = None,
    include_fm_term: bool = True,
) -> torch.Tensor:
    """Shortcut distillation loss (Frans 2024 §3.2).

    Parameters
    ----------
    pred_fn : callable
        Velocity field ``pred_fn(x, t) -> (B, ..., D)``.
    x_time : (B, ..., D)
        The flow-interpolated state at time ``t``.
    t : (B,)
        The original time stamp.
    target_dx : (B, ..., D)
        The conditional velocity ``x_1 − x_0``.
    p_self : float
        Probability of using the shortcut distillation term.  Default 0.5.
    v_pred_t : (B, ..., D), optional
        ``pred_fn(x_time, t)`` already evaluated by the caller.  The trainer
        computes this as its main FM term, so passing it here avoids a second
        forward — and, because the field carries dropout, avoids fitting the
        two terms to two different noise draws.
    include_fm_term : bool
        Whether to return ``fm_loss + p_self * shortcut_mse`` (the paper's
        standalone objective) or only ``p_self * shortcut_mse`` (the
        add-on form).  A caller that already optimises the FM regression as
        its main term must pass ``False``: otherwise the same MSE enters the
        total twice and ``fm_weight`` stops meaning what it says.

    Returns
    -------
    Scalar tensor — combined FM MSE + shortcut MSE, or the shortcut MSE alone.
    """
    # First term: standard FM regression at the original (x_t, t).
    if v_pred_t is None:
        v_pred_t = pred_fn(x_time, t)
    fm_loss = ((v_pred_t - target_dx) ** 2).mean()
    # Second term: shortcut consistency at the second sample (x_s, s).
    x_s, s = shortcut_pair(x_time, t, target_dx, p_self=p_self)
    v_pred_s = pred_fn(x_s, s)
    shortcut_mse = ((v_pred_s - v_pred_t) ** 2).mean()
    if not include_fm_term:
        return p_self * shortcut_mse
    return fm_loss + p_self * shortcut_mse


__all__ = ["shortcut_pair", "shortcut_distillation_loss"]
