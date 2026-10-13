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
        x_prime: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or student_logits is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # Branch (a): teacher logits present → KL(student || teacher).
        # Frans 2024 distillation contract.
        if teacher_logits is not None:
            if student_logits.shape != teacher_logits.shape:
                return torch.zeros((), device=device) if device is not None else torch.zeros(())
            log_p = torch.log_softmax(student_logits, dim=-1)
            log_q = torch.log_softmax(teacher_logits.detach(), dim=-1)
            kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1).mean()
            return kl * self.weight
        # Branch (b): Wf-A23 step 5 — no teacher → dropout self-consistency.
        # Half MSE between ``s_θ(x)`` and ``sg(s_θ(x'))`` where ``x'`` is an
        # input dropout of ``x`` and ``sg`` is stop-gradient.  This is the
        # Frans 2024 self-distillation special case — same family as the
        # teacher KL but with the second view coming from the model itself.
        if x_prime is None or student_logits.shape != x_prime.shape:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            target = student_logits.detach()  # stop-grad
            return 0.5 * float(self.weight) * (student_logits - target).pow(2).mean()
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
