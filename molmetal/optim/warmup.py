"""Linear-warmup learning-rate scheduler (Wf-A11 / optimizer wave).

Ramps the learning rate linearly from 0 to ``optimizer.defaults['lr']`` over
``warmup_steps`` calls to :meth:`step`.  After warmup completes, the
schedule either:

  * Delegates to ``base_scheduler.step()`` when ``base_scheduler`` is given
    (e.g. a :class:`torch.optim.lr_scheduler.CosineAnnealingLR` or
    ReduceLROnPlateau).
  * Holds LR constant when ``base_scheduler`` is None.

Backward-compat
---------------
* ``warmup_steps=0`` (default) keeps the first :meth:`step` from changing
  the LR.  This matches the legacy Wf-13 trainer (no warmup, Adam fixed
  at ``args.lr``).
* The wrapped optimizer's ``.defaults['lr']`` is read once at construction
  time.  If a caller mutates the LR via ``param_group['lr'] = ...`` AFTER
  constructing the scheduler, the warmup target does not auto-update;
  re-build the scheduler or set :attr:`target_lr` manually.

Lit anchors
-----------
* Goyal et al. 2017 ("Accurate, Large Minibatch SGD: Training ImageNet in
  1 Hour", arXiv:1706.02677) — the linear warmup rationale.
* Smith 2017 ("Cyclical Learning Rates for Training Neural Networks",
  arXiv:1506.01186) — warmup-then-cycle as the canonical schedule.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
    """LR ramps 0 -> target lr over ``warmup_steps``, then delegates to *base_scheduler*.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.  Its ``param_groups[i].default_lr_set`` or
        ``defaults['lr']`` provides the target LR.
    warmup_steps : int, default 100
        Number of ``step()`` calls over which to ramp from 0 to target.
        ``0`` is a no-op (legacy Wf-13 behaviour).
    base_scheduler : _LRScheduler, optional
        Scheduler invoked *after* warmup completes.  ``None`` (default)
        holds LR constant after warmup.
    last_epoch : int, default -1
        Standard :class:`_LRScheduler` argument; ``-1`` starts from step 0.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ) -> None:
        if int(warmup_steps) < 0:
            raise ValueError(
                f"warmup_steps must be >= 0; got {warmup_steps}"
            )
        self.warmup_steps = int(warmup_steps)
        self.base_scheduler = base_scheduler
        # Capture target lr from optimizer.defaults.  If optimizer has
        # multiple param groups with different lr, use the first.
        try:
            target = float(optimizer.defaults["lr"])
        except (KeyError, AttributeError):
            target = 0.0
        self.target_lr = target
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Return the warmup-interpolated LR for the current step."""
        step = max(0, int(self.last_epoch) + 1)
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
        scale = float(step) / float(self.warmup_steps)
        return [
            float(self.target_lr) * scale
            for _ in self.optimizer.param_groups
        ]

    def step(self, epoch: Optional[int] = None) -> None:  # type: ignore[override]
        """Advance the schedule by one step.

        During warmup, scales param-group LRs linearly.
        After warmup, defers to base_scheduler.step (or no-op).
        """
        super().step(epoch=epoch)
        if (
            self.base_scheduler is not None
            and int(self.last_epoch) >= int(self.warmup_steps)
        ):
            # When we just handed off to the base scheduler, advance it too.
            # The base scheduler has its own last_epoch independent of ours, so
            # it should always see "one new step" since our last delegation.
            self.base_scheduler.step()


__all__ = ["LinearWarmupScheduler"]