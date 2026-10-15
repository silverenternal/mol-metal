"""Distillation losses — L-A9 placeholder.

``ShortcutDistillLoss`` is a placeholder until a pretrained teacher head
lands. Returns ``0`` when ``teacher_logits is None`` — the trainer's
default call site has no teacher, so this is a no-op in the shipped
config path.

Wf-A34-1 B4 (2026-10-14) — the single ``except Exception: return zeros``
swallow in ``ShortcutDistillLoss.compute`` now calls
``_maybe_warn_import`` (defined in ``regularization_losses``) so a
silent collapse is surfaced once per process.  Mirrors the same B4
refactor over 6 sibling classes in :mod:`molmetal.loss.regularization_losses`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from molmetal.loss.regularization_losses import _maybe_warn_import


class ShortcutDistillLoss:
    """Placeholder for KL(student || teacher) distillation.

    YAML keys
    ---------
    loss.shortcut_distill_weight: float = 0.05   # default ON (Wf-A43 C shortcut_distill default flip)
    """

    def __init__(
        self,
        weight: float = 0.05,
        enabled: bool = True,
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
        #
        # Wf-A43 (2026-10-15) — pre-Wf-A43 the inner block compared
        # ``student_logits`` against ``student_logits.detach()`` which is
        # always zero (same tensor).  The intent is to compare against the
        # SECOND view's logits (which the trainer produces via a separate
        # forward pass on a dropout-perturbed ``x'``); the argument is
        # passed in as ``x_prime`` here (semantic: "logits on x'").  When
        # ``x_prime`` is None we still short-circuit (the trainer's
        # pre-Wf-A43 inline path keeps using `_sd.shortcut_distillation_loss`
        # directly with no second-view input).
        if x_prime is None or student_logits.shape != x_prime.shape:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            target = x_prime.detach()  # stop-grad on the second view
            return 0.5 * float(self.weight) * (student_logits - target).pow(2).mean()
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
