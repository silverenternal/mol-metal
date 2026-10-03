"""Wf-353-A2: Teacher-student shortcut distillation (Frans 2024).

Why this module exists
======================
Wf-340-I1 / Wf-341-C3 already shipped a *self-consistency* shortcut
regulariser in :mod:`molmetal.adapters.flow_matching_lipman.acceleration`
(Same model as both teacher and student).  Frans et al. 2024 (arXiv
2410.12557) additionally describe a *teacher-student distillation* path
where the teacher is a frozen EMA copy of the student.  This module
ships that complementary wire-up behind a single
``--use-shortcut-distillation`` CLI flag.

  * Teacher: ``deepcopy`` of the base velocity field, frozen via
    ``requires_grad_(False)`` + ``eval()``, updated as an EMA of the
    student every training step.
  * Student: the *same* base velocity field (no parameter duplication).
  * Loss: for a fraction ``p_distill`` of the batch, sample a fresh
    (t, x_t) and compute
        L_distill = MSE( F_student(x_t, t), sg( F_teacher(x_t, t) ) )
    By training the student to match the teacher at *random* time
    stamps, the student implicitly learns the shortcut direction
    ``Δv = F(x_{t'}, t') - F(x_t, t)`` Frans 2024 §3.2 exploits for
    few-step inference.

CLI (train_cfm_yuelbond_wasserstein.py):
  --use-shortcut-distillation (default False)
  --shortcut-distillation-weight (default 1.0)
  --shortcut-distillation-p-distill (default 0.5)
  --shortcut-distillation-ema-decay (default 0.999)

Honest framing (party-twenty-spirit §3.5):
  * LOC + tests are MEASURED on disk.
  * Speedup numbers (1-step vs 50-100 NFE inference) are PROJECTED
    from upstream paper benchmarks and NOT measured on Mol-Metal.
  * Memory cost: ~1× model size for the frozen teacher copy.

References: Frans et al. 2024 (arXiv 2410.12557); Karras 2022 EMA
halflife convention; Wf-340-I1 / Wf-341-C3 (prior self-consistency).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class ShortcutDistillationConfig:
    """Teacher-student shortcut distillation configuration.

    Args:
      p_distill: Branch probability (rest fall back to plain CFM MSE).
      weight: Distillation loss weight.
      teacher_ema_decay: EMA decay rate for the teacher copy.
      t_min / t_max: Sampling range for the random time stamp; avoids
        the trivial t=0 (pure noise) / t=1 (clean data) endpoints.
    """

    p_distill: float = 0.5
    weight: float = 1.0
    teacher_ema_decay: float = 0.999
    t_min: float = 0.05
    t_max: float = 0.95


class ShortcutDistiller(nn.Module):
    """Teacher-student shortcut distillation wrapper.

    The wrapper owns the teacher copy and exposes the EMA update +
    distillation loss; the student model is shared with the caller so
    gradients flow into the original parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[ShortcutDistillationConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or ShortcutDistillationConfig()
        self.student = model
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA update of teacher weights from the student."""
        decay = self.cfg.teacher_ema_decay
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(decay).add_(s_p.data, alpha=1.0 - decay)
        for t_b, s_b in zip(self.teacher.buffers(), self.student.buffers()):
            t_b.data.copy_(s_b.data)

    @torch.no_grad()
    def teacher_forward(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        out = self.teacher(x_t, t, **kwargs)
        return out["vel"] if isinstance(out, dict) else out

    def student_forward(
        self, x_t: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        out = self.student(x_t, t, **kwargs)
        return out["vel"] if isinstance(out, dict) else out

    def distillation_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        path_sampler: Callable,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Teacher-student distillation loss (Frans 2024).

        For the ``p_distill`` fraction of the batch, sample a fresh
        ``t ~ U(t_min, t_max)`` and compute ``x_t = path_sampler(...)``;
        the student is trained to match the teacher's velocity under
        MSE.  Returns a zero scalar when the mask selects nothing.
        """
        b = x_0.shape[0]
        device = x_0.device
        mask = torch.rand(b, device=device) < self.cfg.p_distill
        if not mask.any():
            return torch.zeros((), device=device, dtype=x_0.dtype)
        t = (
            torch.rand(b, device=device) * (self.cfg.t_max - self.cfg.t_min)
            + self.cfg.t_min
        )
        ps = path_sampler(x_0=x_0[mask], x_1=x_1[mask], t=t[mask])
        x_t = ps.x_t
        with torch.no_grad():
            v_teacher = self.teacher_forward(x_t, t[mask], **extra_kwargs)
        v_student = self.student_forward(x_t, t[mask], **extra_kwargs)
        diff = (v_student - v_teacher.detach()) ** 2
        per_atom = diff.flatten(1).sum(dim=1)
        return per_atom.mean() * self.cfg.weight


__all__ = ["ShortcutDistillationConfig", "ShortcutDistiller"]
