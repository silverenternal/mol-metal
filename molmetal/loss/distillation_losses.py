"""Distillation losses — L-A9 placeholder.

``ShortcutDistillLoss`` is a placeholder until a pretrained teacher head
lands. Returns ``0`` when ``teacher_logits is None`` — the trainer's
default call site has no teacher, so this is a no-op in the shipped
config path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class ShortcutDistillLoss:
    """Placeholder for KL(student || teacher) distillation.

    YAML keys
    ---------
    loss.shortcut_distill_weight: float = 0.0   # default OFF
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "shortcut_distill_loss",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        student_logits: Optional[torch.Tensor],
        teacher_logits: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or teacher_logits is None
            or student_logits is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if student_logits.shape != teacher_logits.shape:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        T = float(student_logits.shape[-1])
        log_p = torch.log_softmax(student_logits, dim=-1)
        log_q = torch.log_softmax(teacher_logits.detach(), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1).mean()
        return kl * self.weight

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
