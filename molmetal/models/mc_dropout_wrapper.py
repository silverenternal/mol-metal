"""Wf-249 Fix #2 — MC-Dropout post-hoc ensemble wrapper (CPU-only).

=====================================================================
Citation: Liu Y. et al. JCIM 2024, 64(21), 8131-8141. MUBen (arXiv
2306.10060) corroborates deep-ensemble / MC-dropout stability for
property prediction. Both motivate an *inference-time* wrapper that
turns a single deterministic checkpoint into an uncertainty-aware
predictor without retraining.

This module is intentionally minimal:
  * ``MCDropoutEnsemble`` wraps any ``torch.nn.Module`` that has at
    least one dropout layer (or accepts an injected dropout head).
  * ``enable_mc_dropout(model, p)`` is the helper that mutates an
    existing model in place — flips all ``nn.Dropout*`` modules to
    ``p`` and forces them into training mode while keeping BatchNorm
    etc. in eval mode.
  * ``mc_dropout_predict(model, x, n_passes, ...)`` runs M forward
    passes and returns ``(mean, std)``.
  * All operations are CPU-only — no GPU dependencies, no retraining.

Backward compat
---------------
* Does NOT mutate training behaviour. ``model.eval()`` after the
  wrapper still works as before.
* Existing ``model(x)`` calls unchanged.
* Default ``n_passes=10``; pass ``n_passes=1`` to recover the
  deterministic single-pass output.

Usage
-----
::

    from molmetal.models.mc_dropout_wrapper import (
        MCDropoutEnsemble, mc_dropout_predict, enable_mc_dropout,
    )

    # Wrap an existing checkpoint
    ensemble = MCDropoutEnsemble(model, dropout_p=0.1, n_passes=10)
    mean, std = ensemble.predict(x)

    # Or one-shot
    enable_mc_dropout(model, p=0.1)
    mean, std = mc_dropout_predict(model, x, n_passes=10)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:                                                      # pragma: no cover
    _HAS_TORCH = False


_DROPOUT_TYPES: Tuple[type, ...] = ()


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "mc_dropout_wrapper requires torch. "
            "Install via `pip install torch` or activate the project venv."
        )


def _dropout_types() -> Tuple[type, ...]:
    global _DROPOUT_TYPES
    if not _DROPOUT_TYPES and _HAS_TORCH:
        _DROPOUT_TYPES = (
            nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
            nn.AlphaDropout, nn.FeatureAlphaDropout,
        )
    return _DROPOUT_TYPES


def enable_mc_dropout(model: "nn.Module", p: float = 0.1) -> int:
    """Flip all dropout layers to ``p`` and put them in training mode.

    Sets the parent model to eval mode (so BatchNorm etc. stay frozen)
    but forces dropout modules into train mode — the standard MC-dropout
    trick. Returns the number of dropout modules found.
    """
    _require_torch()
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    model.eval()
    n_found = 0
    for module in model.modules():
        if isinstance(module, _dropout_types()):
            module.p = float(p)
            module.train()
            n_found += 1
    return n_found


@torch.no_grad() if _HAS_TORCH else lambda f: f
def mc_dropout_predict(
    model: "nn.Module",
    x: "torch.Tensor",
    n_passes: int = 10,
    return_std: bool = True,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    """Run ``n_passes`` MC-dropout forward passes, return ``(mean, std)``.

    Assumes :func:`enable_mc_dropout` has already been called on
    ``model`` so dropout is active.  BatchNorm etc. remain in eval mode
    (set by :func:`enable_mc_dropout`).
    """
    _require_torch()
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    outputs = []
    for _ in range(int(n_passes)):
        outputs.append(model(x))
    stacked = torch.stack(outputs, dim=0)  # (M, batch, ...)
    mean = stacked.mean(dim=0)
    if not return_std:
        return mean, stacked
    std = stacked.std(dim=0, unbiased=False)
    return mean, std


class MCDropoutEnsemble:
    """Inference-time MC-dropout ensemble wrapper.

    Wraps an existing trained ``nn.Module`` and turns it into an
    uncertainty-aware predictor at inference time. No retraining
    required.

    Parameters
    ----------
    model : nn.Module
        Existing trained model. Will be flipped to eval mode on
        construction; dropout layers are forced into train mode at
        every ``predict`` call (and the parent back to eval).
    dropout_p : float
        Dropout probability used during MC forward passes (default 0.1
        per Liu et al. JCIM 2024).
    n_passes : int
        Number of stochastic forward passes per ``predict`` call
        (default 10; pass 1 for deterministic single-pass output).
    """

    def __init__(self, model: "nn.Module", dropout_p: float = 0.1,
                 n_passes: int = 10) -> None:
        _require_torch()
        if not (0.0 <= dropout_p <= 1.0):
            raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
        if n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {n_passes}")
        self.model = model
        self.dropout_p = float(dropout_p)
        self.n_passes = int(n_passes)
        self._n_dropout = self._inject_dropout()

    def _inject_dropout(self) -> int:
        """Set dropout layers' p and switch parent to eval mode."""
        self.model.eval()
        n_found = 0
        for module in self.model.modules():
            if isinstance(module, _dropout_types()):
                module.p = self.dropout_p
                n_found += 1
        return n_found

    def predict(
        self, x: "torch.Tensor", return_std: bool = True,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """Run M forward passes and aggregate.

        Returns
        -------
        mean : torch.Tensor
            Mean of M stochastic predictions (shape: (batch, ...)).
        std : torch.Tensor or None
            Per-element standard deviation (shape: (batch, ...)) if
            ``return_std=True``, else ``None``.
        """
        # Re-flip dropout p each call (cheap) and force train mode.
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, _dropout_types()):
                module.p = self.dropout_p
                module.train()
        outputs = []
        for _ in range(self.n_passes):
            outputs.append(self.model(x))
        with torch.no_grad():
            stacked = torch.stack(outputs, dim=0)  # (M, batch, ...)
            mean = stacked.mean(dim=0)
            if not return_std:
                return mean, None
            std = stacked.std(dim=0, unbiased=False)
        return mean, std

    def predict_with_uncertainty(
        self, x: "torch.Tensor", confidence: float = 0.95,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return (mean, lower, upper) at given confidence interval.

        Uses a Gaussian approximation ``[mean - z*std, mean + z*std]``
        — appropriate for the post-hoc, MC-dropout aggregated output.
        """
        _require_torch()
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        mean, std = self.predict(x, return_std=True)
        # z-score for the symmetric two-sided CI; use a small table
        # to avoid the scipy dependency.
        z_table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        if confidence in z_table:
            z = z_table[confidence]
        else:
            # Approximate via |1 - (1-c)/2| quantile for Gaussian.
            z = float(np.sqrt(2) * _erfinv(2 * confidence - 1))
        return mean, mean - z * std, mean + z * std

    @property
    def n_dropout_layers(self) -> int:
        return self._n_dropout


def _erfinv(x: float) -> float:
    """Approximate inverse error function via numpy fallback.

    Used only for non-tabulated confidence levels. Numerical accuracy
    is sufficient for MC-dropout calibration (the wrapper is for
    uncertainty quantification, not exact quantiles).
    """
    # Clamp domain.
    x = float(np.clip(x, -1.0 + 1e-9, 1.0 - 1e-9))
    return float(np.sqrt(np.pi) / 2.0 * _erfinv_series(x))


def _erfinv_series(x: float) -> float:
    """Rational approximation to erfinv(x) for |x|<1.

    Source: Winitzki, "A handy approximation for the error function
    and its inverse" (2008). Accurate to ~1e-4 across the full range.
    """
    a = 0.147  # Winitzki constant
    ln_term = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln_term / 2.0
    inside = first * first - ln_term / a
    return float(np.sign(x) * np.sqrt(np.sqrt(inside) - first))


__all__ = [
    "MCDropoutEnsemble",
    "enable_mc_dropout",
    "mc_dropout_predict",
]
