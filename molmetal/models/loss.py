"""Multi-task loss for MetalCytoToxDB: pIC50 regression + activity classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import fused_cross_entropy
from triton_kernels.config import triton_config


class MetalCytotoxLoss(nn.Module):
    """Dual-head loss: alpha * MSE(pIC50) + (1-alpha) * BCE(logit, active).

    The classification head is a learned logit derived from the predicted pIC50
    (not a separate head), following the paper design of multi-task learning.

    Args:
        alpha: weighting factor for regression loss (default 0.5).
               Total = alpha * MSE + (1-alpha) * BCE.
        pic50_min/max: for converting pIC50 prediction to a logit.
                       logit = (pic50_pred - pic50_min) / (pic50_max - pic50_min) * 2 - 1
                       This maps pIC50 into [-1, +1] range for BCE with labels in {0,1}.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pic50_min: float = 4.0,
        pic50_max: float = 9.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.pic50_min = float(pic50_min)
        self.pic50_max = float(pic50_max)

    def _to_logit(self, pic50_pred: torch.Tensor) -> torch.Tensor:
        """Map predicted pIC50 to a [-1, +1] score, then to a logit."""
        # Normalise to [0, 1]
        t = (pic50_pred - self.pic50_min) / (self.pic50_max - self.pic50_min + 1e-8)
        t = t.clamp(0.0, 1.0)
        # Map to [-1, +1]
        score = t * 2.0 - 1.0
        # BCEWithLogits expects raw logits; we use softplus on score as pseudo-logit
        # For direct BCE, we use score as the logit (positive = active)
        return score

    def forward(
        self,
        pic50_pred: torch.Tensor,
        true_pic50: torch.Tensor,
        active_label: torch.Tensor,
        mask: torch.Tensor | None = None,
        active_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pic50_pred:    (B,) predicted pIC50 values
            true_pic50:    (B,) ground-truth pIC50 values
            active_label:  (B,) binary labels (1=active, 0=inactive)
            mask:          (B,) optional, 1=real sample, 0=padding.
                           If None, all samples are real.
            active_logits: (B, 2) optional — explicit 2-class logits from the dual head.
                           If provided, BCE is computed over (inactive, active) classes
                           instead of deriving a logit from pIC50_pred.

        Returns:
            scalar loss
        """
        if pic50_pred.shape != true_pic50.shape:
            raise ValueError(
                f"Shape mismatch: pic50_pred={pic50_pred.shape}, "
                f"true_pic50={true_pic50.shape}"
            )

        # Regression loss
        reg_loss = F.mse_loss(pic50_pred, true_pic50, reduction="none")  # (B,)

        # Classification loss
        if active_logits is not None:
            # Use explicit dual-head logits (B, 2) — index 1 is the "active" class
            if active_logits.dim() != 2 or active_logits.size(1) < 2:
                raise ValueError(
                    f"active_logits must be (B, 2), got {tuple(active_logits.shape)}"
                )
            targets = active_label.long()
            # Route through the Triton fused cross-entropy when the
            # dispatch gate approves (training-mode default ON; CPU,
            # eval-mode, or very small batches fall back to PyTorch).
            if triton_config.should_use_fused(active_logits, op="cross_entropy"):
                cls_loss = fused_cross_entropy(
                    active_logits, targets, reduction="none"
                )  # (B,)
            else:
                cls_loss = F.cross_entropy(
                    active_logits, targets, reduction="none"
                )  # (B,)
        else:
            # Fallback: derive a scalar logit from predicted pIC50
            logit = self._to_logit(pic50_pred)  # (B,)
            cls_loss = F.binary_cross_entropy_with_logits(
                logit, active_label.float(), reduction="none"
            )  # (B,)

        per_sample = self.alpha * reg_loss + (1.0 - self.alpha) * cls_loss  # (B,)

        if mask is not None:
            per_sample = per_sample * mask
            denom = mask.sum().clamp(min=1.0)
        else:
            denom = float(per_sample.numel())

        return (per_sample.sum() / denom).clamp(min=1e-8)

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, "
            f"pic50_min={self.pic50_min}, pic50_max={self.pic50_max}"
        )
