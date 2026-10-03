"""Wf-380-A1 §2.7 — Consistency regularization SCAFFOLD.

NOTE: This is the Wf-380-A1 SCAFFOLD module.  The canonical (Wf-353-A3)
implementation lives in :mod:`molmetal.adapters.flow_matching_lipman.consistency_regularization`
(164 LOC).  Both modules coexist — this SCAFFOLD exposes a small,
importable reference for deep-integration testing; the canonical
implementation is the production target.

Reference: Karras, T., Aittala, M., Aila, T., Laine, S. (2022).
*Elucidating the Design Space of Diffusion-Based Generative Models.*
NeurIPS 2022.  arXiv:2206.00364.  NVlabs/edm (CC BY-NC-SA 4.0).

The EDM consistency regulariser penalises deviations between the
velocity field evaluated at the noise-perturbed state and the
deterministic trajectory target:

    L_consistency = (1/B) · Σ_i  ‖v_θ(x_t + σ_i · n_i, t) − v_θ(x_t, t)‖²

where ``σ_i`` is sampled from the log-normal schedule and ``n_i`` is
standard Gaussian noise.  This is a softer form of the EDM denoiser
training that doesn't require the denoiser output but does help
stabilise training under noise.

SCAFFOLD ONLY — no trainer wiring.  Lift claim (+5-15% on m14 CFM
decode) is PROJECTED per Karras 2022 + Wf-340-A3.

Honest framing: SCAFFOLD-only; no GPU launches.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def sample_edm_sigma(
    batch_size: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample ``batch_size`` σ values from the EDM log-normal schedule.

    Reference: Karras 2022 §3.4 + EDM reference implementation.  The
    log-normal distribution with mean ``P_mean`` and std ``P_std`` is
    clamped to ``[sigma_min, sigma_max]``.  Defaults match the EDM
    reference.
    """
    p_mean = -1.2
    p_std = 1.2
    normal = torch.randn(batch_size, generator=generator, device=device)
    log_sigma = p_mean + p_std * normal
    sigma = log_sigma.exp()
    return sigma.clamp(min=sigma_min, max=sigma_max)


def consistency_loss_fn(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    weight: float = 1.0,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """EDM consistency regulariser.

    Parameters
    ----------
    pred_fn : callable
        Velocity field ``pred_fn(x) -> velocity``.
    x_t : (B, ..., D)
        The flow-interpolated state.
    weight : float
        Multiplier on the consistency loss.  Default 1.0.
    sigma_min, sigma_max : float
        EDM σ range.

    Returns
    -------
    Scalar tensor — the consistency regulariser.
    """
    b = int(x_t.shape[0])
    sigma = sample_edm_sigma(b, sigma_min=sigma_min, sigma_max=sigma_max)
    noise = torch.randn_like(x_t)
    sigma_b = sigma.view(-1, *([1] * (x_t.dim() - 1)))
    x_perturbed = x_t + sigma_b * noise
    v_clean = pred_fn(x_t)
    v_perturbed = pred_fn(x_perturbed)
    return weight * ((v_perturbed - v_clean) ** 2).mean()


__all__ = ["sample_edm_sigma", "consistency_loss_fn"]
