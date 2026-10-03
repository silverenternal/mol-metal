"""Wf-380-A1 §2.9 — Log-normal t-sampling SCAFFOLD.

Reference: Karras, T., Aittala, M., Aila, T., Laine, S. (2022).
*Elucidating the Design Space of Diffusion-Based Generative Models.*
NeurIPS 2022.  arXiv:2206.00364.  NVlabs/edm (CC BY-NC-SA 4.0).

The log-normal t-sampling distribution re-weights the uniform-in-[0, 1]
CFM t-sample toward intermediate times (the typical "interesting"
region of the flow trajectory):

    ln(t) ~ N(mu, sigma²)
    t = sigmoid(ln(t)) ∈ (0, 1)

For CFM, defaults ``mu = -0.5`` and ``sigma = 1.0`` per Wf-340-A3.
The Karras EDM reference uses ``P_mean = -1.2, P_std = 1.2`` for the
*σ* schedule; for the CFM t-schedule the values are tuned to match
the empirical t-distribution under flow matching.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.acceleration.log_normal_t_sampling`
(Wf-340-I1).  This module exposes a tiny importable reference.

Honest framing: SCAFFOLD-only; no GPU launches; speedup claim
(+5-10% convergence) is PROJECTED per Karras 2022.
"""

from __future__ import annotations

from typing import Optional

import torch


def log_normal_t_sampling(
    batch_size: int,
    mu: float = -0.5,
    sigma: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample ``batch_size`` flow-time values from the log-normal schedule.

    Parameters
    ----------
    batch_size : int
    mu, sigma : float
        Log-normal hyperparameters.  Defaults per Wf-340-A3.
    generator : optional torch.Generator for reproducibility.
    device : torch.device

    Returns
    -------
    t : (batch_size,) — flow times in (0, 1).
    """
    normal = torch.randn(batch_size, generator=generator, device=device)
    log_t = mu + sigma * normal
    return torch.sigmoid(log_t)


__all__ = ["log_normal_t_sampling"]
