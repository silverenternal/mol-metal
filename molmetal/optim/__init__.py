"""Mol-Metal optimizer package (Wf-A11 + AGENTS.md §5.11 cross-wave).

Re-exports the four primitives the CFM trainer (and any future trainer that
adopts yaml-driven hyperparameters per AGENTS.md §5.11) needs:

  * :func:`build_differential_optimizer` — 4-group param split (head_decay,
    head_no_decay, backbone_decay, backbone_no_decay) so the head lr can
    run Nx backbone lr without either decay part eating the no-decay learning
    rate or vice-versa.
  * :class:`EMAModel` — EMA shadow weights with state_dict round-trip and
    a default decay of ``0.9999`` (was 0.999 in the legacy 1.x trainer).
  * :class:`LinearWarmupScheduler` — LinearWarmup from 0 -> base_lr over
    ``warmup_steps`` calls to :meth:`step`, then delegates to an optional
    base scheduler (CosineAnnealing, ReduceLROnPlateau, None = constant).
  * :func:`check_bf16_compatibility` — detect SM >= 8.0 / 8.6 / 12.0 so the
    trainer can refuse BF16 autocast on incompatible GPUs.

Backward-compat
---------------
* None of these primitives is invoked from the trainer by default.  When a
  config key (e.g. ``optimizer.use_differential_lr``) is at its default,
  the trainer preserves the existing single-group Adam path (legacy Wf-13
  behaviour, ``lr=1e-4``, no warmup, no EMA, full BF16 trust).
* :func:`check_bf16_compatibility` accepts a ``min_compute_capability`` so
  the legacy default of 8.0 (Ampere+) is overridable per-config.

Lit anchors
-----------
* Loshchilov & Hutter 2017 ("Fixing Weight Decay Regularization in Adam",
  arXiv:1711.05101) — the canonical no_decay / decay split this module
  uses for ``head_no_decay`` vs ``head_decay`` / backbone pairs.
* Smith 2017 ("Cyclical Learning Rates for Training Neural Networks",
  arXiv:1506.01186) — warmup-then-cycle as the schedule policy.
* PyTorch 2.0+ AMP docs — BF16 autocast requires ``sm_80+``; mixed FP16/BF16
  hardware availability table.
"""

from __future__ import annotations

from molmetal.optim.bf16_guard import check_bf16_compatibility
from molmetal.optim.differential_lr import build_differential_optimizer
from molmetal.optim.ema import EMAModel
from molmetal.optim.warmup import LinearWarmupScheduler

__all__ = [
    "build_differential_optimizer",
    "EMAModel",
    "LinearWarmupScheduler",
    "check_bf16_compatibility",
]