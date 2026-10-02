"""Wf-380-A1 §2.10 — Test for lr_warmup SCAFFOLD.

Verifies the linear lr-warmup scheduler structure (Goyal 2017).
"""

from __future__ import annotations

import pytest

from molmetal.adapters.flow_matching_lipman.lr_warmup import LinearWarmupLR


def test_lr_warmup_at_step_zero_is_small_fraction():
    """At step 0, lr = base_lr * (1/warmup_steps) — the warmup ramp."""
    sched = LinearWarmupLR(base_lr=1.0, warmup_steps=100)
    assert abs(sched.lr_at(0) - 1.0 / 100.0) < 1e-6


def test_lr_warmup_at_step_warmup_is_base_lr():
    """At step warmup_steps, lr = base_lr (ramp complete)."""
    sched = LinearWarmupLR(base_lr=1.0, warmup_steps=100)
    assert abs(sched.lr_at(100) - 1.0) < 1e-6
    assert abs(sched.lr_at(101) - 1.0) < 1e-6
    assert abs(sched.lr_at(1000) - 1.0) < 1e-6


def test_lr_warmup_monotonic_increase():
    """The learning rate is non-decreasing up to base_lr."""
    sched = LinearWarmupLR(base_lr=1e-3, warmup_steps=50)
    lrs = [sched.lr_at(s) for s in range(60)]
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i - 1] - 1e-12


def test_lr_warmup_validates_warmup_steps():
    """warmup_steps < 1 raises ValueError."""
    with pytest.raises(ValueError):
        LinearWarmupLR(base_lr=1.0, warmup_steps=0)


def test_module_importable():
    """Sanity check: the module is importable from its package."""
    import molmetal.adapters.flow_matching_lipman.lr_warmup as mod

    assert hasattr(mod, "LinearWarmupLR")