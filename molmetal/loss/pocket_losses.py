"""Wf-A16 (2026-10-13) — pocket-fit additive loss.

Pure additive loss; the trainer invokes it ONLY when the YAML key
``loss.pocket_fit_weight > 0`` AND ``eval.compute_pocket_fit_loss=True``.
When both default values are in force (backward-compat), the trainer
never calls :class:`PocketFitLoss` and the loss path is byte-identical
to Wf-A15.

Why this exists
---------------
The c-flow (pocket signal) end-to-end propagation gives the EGNN
velocity field + BondOrderHead a per-pocket embedding.  That embedding
has no learning signal beyond the W₂ anchor + the bond-CE term — both
of which are pocket-invariant.  A pocket-fit loss closes the loop:

    L_pocket = mean over pairs of (decoded_bo - pocket_target_bo)^2
                                  * pocket_mask

``pocket_target_bos`` is a pre-computed per-pocket median Wiberg BO
sourced from the training corpus (typically the cross-pocket median
across the entire pocket set).  The mask zeros out padded slots so the
gradient is the per-real-pair mean.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PocketFitLoss(nn.Module):
    """Pocket-fit bond-order regression loss (Wf-A16).

    Parameters
    ----------
    weight : float
        Multiplier on the MSE term.  0.0 disables the loss (the trainer
        can short-circuit before calling :meth:`forward`).
    enabled : bool
        Master switch.  When False, :meth:`forward` returns ``0``.
    name : str
        Sink key for the loss aggregator.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "pocket_fit_loss",
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def forward(
        self,
        decoded_bond_orders: Optional[torch.Tensor],
        pocket_target_bos: Optional[torch.Tensor],
        pocket_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return a scalar MSE masked by ``pocket_mask``.

        Args
        ----
        decoded_bond_orders
            (E,) or (E, 1) float tensor of predicted Wiberg-like BOs.
        pocket_target_bos
            (E,) or (E, 1) float tensor of per-pocket target BOs.
        pocket_mask
            (E,) bool tensor — True for real pairs, False for padded.
            ``None`` is treated as ``all True``.
        device
            Optional device for the zero scalar.

        Returns
        -------
        torch.Tensor
            Scalar (0-d) tensor.
        """
        if not self.enabled or self.weight <= 0.0:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if decoded_bond_orders is None or pocket_target_bos is None:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if decoded_bond_orders.shape != pocket_target_bos.shape:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        if pocket_mask is None:
            mask = torch.ones_like(decoded_bond_orders, dtype=torch.bool)
        else:
            mask = pocket_mask.to(decoded_bond_orders.device).bool()
        if int(mask.sum().item()) == 0:
            return (
                torch.zeros((), device=device)
                if device is not None
                else torch.zeros(())
            )
        diff = (decoded_bond_orders - pocket_target_bos) ** 2
        masked = diff[mask]
        loss = float(self.weight) * masked.mean()
        return (
            torch.tensor(loss, device=device)
            if device is not None
            else torch.tensor(loss)
        )


__all__ = ["PocketFitLoss"]