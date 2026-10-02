"""Wf-380-A1 §2.10 — Linear lr-warmup scheduler SCAFFOLD.

Reference: Goyal, P., Dollár, P., Girshick, R., Noordhuis, P.,
Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K. (2017).
*Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
arXiv:1706.02677.

The linear lr-warmup scheduler ramps the learning rate from 0 to the
target ``base_lr`` over ``warmup_steps`` gradient steps, then keeps it
constant at ``base_lr`` (caller is responsible for the post-warmup
schedule).  This is the canonical pre-warmup for large-batch SGD and
the EDM/CFM training convention.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.acceleration.LinearWarmup`
(Wf-340-I1).  This module exposes the tiny importable reference.

Honest framing: SCAFFOLD-only; no GPU launches; lift claim
(+5-10% training stability) is PROJECTED per Goyal 2017.
"""

from __future__ import annotations

from typing import Iterator

import torch


class LinearWarmupLR:
    """Linear lr-warmup scheduler (Goyal 2017).

    The scheduler yields the current learning rate at each gradient step.
    ``lr(step) = base_lr * min(step / warmup_steps, 1.0)`` for
    ``0 <= step <= warmup_steps``; afterwards ``lr == base_lr``.

    Parameters
    ----------
    base_lr : float
        Target learning rate after warmup.
    warmup_steps : int
        Number of warmup steps.  Default 1000 per Goyal 2017 + EDM.
    """

    def __init__(self, base_lr: float = 1e-4, warmup_steps: int = 1000):
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)

    def lr_at(self, step: int) -> float:
        """Return the learning rate at gradient step ``step``."""
        if step < 0:
            return 0.0
        scale = min(float(step + 1) / self.warmup_steps, 1.0)
        return self.base_lr * scale

    def __call__(self, step: int) -> float:
        return self.lr_at(step)

    def iter_steps(self, n_steps: int) -> Iterator[float]:
        """Iterate ``n_steps`` learning rates starting at step 0."""
        return iter(self.lr_at(s) for s in range(n_steps))


__all__ = ["LinearWarmupLR"]