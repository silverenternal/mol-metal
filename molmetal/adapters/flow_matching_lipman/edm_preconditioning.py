"""Wf-380-A1 §2.1 — EDM σ-preconditioning SCAFFOLD.

Reference: Karras, T., Aittala, M., Aila, T., Laine, S. (2022).
*Elucidating the Design Space of Diffusion-Based Generative Models.*
NeurIPS 2022.  arXiv:2206.00364.  NVlabs/edm (CC BY-NC-SA 4.0).

The Karras formulation targets the denoiser prediction
``D_θ(x; σ) ≈ x_0``.  For CFM we adapt it to the conditional velocity
``v_t = x_1 − x_0``:

    c_skip(σ) = σ_data² / (σ² + σ_data²)
    c_out(σ) = σ · σ_data / sqrt(σ² + σ_data²)
    c_in(σ)  = 1 / sqrt(σ² + σ_data²)
    c_noise  = 0.5 · ln(σ)

The pre-conditioned velocity is

    v_pred = c_skip · x_t + c_out · F_θ(c_in · x_t; c_noise · σ)

and the training loss is

    λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

so the weighted MSE matches the EDM denoiser loss bit-exactly.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.acceleration.EDMPreconditioner`
(Wf-340-I1).  This module exposes a tiny CPU-only reference so the
deep-integration scaffold surface can be smoke-tested in isolation.

Honest framing (party-twenty-spirit §3.5): SCAFFOLD-only, NOT trainer-
integrated.  No GPU launches.  All speedup claims are PROJECTED per
Wf-340-A1 + Karras 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EDMConfig:
    """Configuration for EDM σ-preconditioning (Karras 2022 §3.4).

    Attributes
    ----------
    sigma_data : float
        Empirical standard deviation of the target — Karras default 0.5
        for natural images; for our 3D coord data we use 1.5 Å (the
        empirical std of bond lengths around equilibrium).
    sigma_min, sigma_max : float
        Sampling range for the log-normal σ schedule.
    """

    sigma_data: float = 1.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0


def c_skip(sigma: torch.Tensor, cfg: EDMConfig) -> torch.Tensor:
    """c_skip(σ) = σ_data² / (σ² + σ_data²).  Karras eq. 2 component."""
    return cfg.sigma_data ** 2 / (sigma ** 2 + cfg.sigma_data ** 2)


def c_out(sigma: torch.Tensor, cfg: EDMConfig) -> torch.Tensor:
    """c_out(σ) = σ · σ_data / sqrt(σ² + σ_data²).  Karras eq. 2."""
    return sigma * cfg.sigma_data / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)


def c_in(sigma: torch.Tensor, cfg: EDMConfig) -> torch.Tensor:
    """c_in(σ) = 1 / sqrt(σ² + σ_data²).  Karras eq. 2."""
    return 1.0 / torch.sqrt(sigma ** 2 + cfg.sigma_data ** 2)


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    """c_noise(σ) = 0.5 · ln(σ).  Karras eq. 2 time embedding."""
    return 0.5 * torch.log(sigma)


def preconditioned_loss(
    pred_raw: torch.Tensor,
    x_t: torch.Tensor,
    target_dx: torch.Tensor,
    sigma: torch.Tensor,
    cfg: EDMConfig = EDMConfig(),
) -> torch.Tensor:
    """EDM-preconditioned weighted MSE loss for CFM velocity fields.

    Parameters
    ----------
    pred_raw : (B, ..., D)
        The "raw" velocity prediction ``F_θ(c_in · x_t; c_noise · σ)``.
        Caller is responsible for applying the ``c_in`` scale + ``c_noise``
        conditioning before passing here.
    x_t : (B, ..., D)
        The interpolated state at flow time t.
    target_dx : (B, ..., D)
        The conditional velocity ``v_t = x_1 − x_0``.
    sigma : (B,) or scalar
        The per-sample noise scale σ.

    Returns
    -------
    Scalar tensor — the preconditioned weighted MSE.
    """
    sigma_b = sigma.view(-1, *([1] * (x_t.dim() - 1))) if sigma.dim() == 1 else sigma
    v_pred = c_skip(sigma_b, cfg) * x_t + c_out(sigma_b, cfg) * pred_raw
    weight = (sigma_b ** 2 + cfg.sigma_data ** 2) / (sigma_b * cfg.sigma_data) ** 2
    err2 = (v_pred - target_dx) ** 2
    return (weight * err2).mean()


__all__ = [
    "EDMConfig",
    "c_skip",
    "c_out",
    "c_in",
    "c_noise",
    "preconditioned_loss",
]
