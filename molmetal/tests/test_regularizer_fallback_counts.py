"""Regression guard — a regulariser that swallows an exception must be counted.

WHY THIS EXISTS (2026-10-21, FIX-C12).  Every term in
``molmetal/loss/regularization_losses.py`` wraps its body in
``try/except Exception`` and returns ``torch.zeros(())`` on failure, warning
once per ``(class_name, exc_type_name)`` pair.  That warning gate is correct
for the case it was built for (a missing optional import on CPU-only CI), but
it also covers the case that actually happened: memory pressure.

MEASURED on the 2026-10-20 full-corpus C7 run (``n_train=26715``), which
crashed in epoch 4 with the allocator at 97,331 MiB:

    [EnergyConservation] OutOfMemoryError — CUDA out of memory ...; returning 0
    [SBOO3Term] RuntimeError — cusolver error ...; returning 0

Both terms were configured at weight 0.05 and both were silently zero for the
rest of the run, with no trace in ``events.jsonl``.  A run whose terms are
being zeroed is indistinguishable, from its own artifacts, from a run whose
terms are live — which is exactly the "live but inert" family this repo keeps
re-discovering.

``FALLBACK_COUNTS`` increments on every swallow (not just the first), so the
trainer can report how many steps each term actually lost.  The warning
behaviour is deliberately unchanged: still once per pair.

The tests need neither CUDA nor a real model — they drive one term with a
``pred_fn`` that raises.
"""

from __future__ import annotations

import warnings

import torch

from molmetal.loss import regularization_losses as rl


def _boom(_x):
    raise RuntimeError("simulated failure inside pred_fn")


def _reset() -> None:
    rl.FALLBACK_COUNTS.clear()
    rl._IMPORT_WARNED.clear()


def test_swallowed_exception_is_counted_every_time_not_just_once():
    """The warning is once-per-process; the count must be once-per-call."""
    _reset()
    term = rl.EnergyConservation(weight=0.05, enabled=True)
    x = torch.randn(4, 3)

    out1 = term.compute(pred_fn=_boom, x_t=x)
    out2 = term.compute(pred_fn=_boom, x_t=x)
    out3 = term.compute(pred_fn=_boom, x_t=x)

    for out in (out1, out2, out3):
        assert float(out) == 0.0, "a swallowed term must still return a zero scalar"
    assert rl.FALLBACK_COUNTS["EnergyConservation"] == 3


def test_the_warning_still_fires_only_once_per_pair():
    """Counting must not turn the once-per-process warning into a flood."""
    _reset()
    term = rl.EnergyConservation(weight=0.05, enabled=True)
    x = torch.randn(4, 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            term.compute(pred_fn=_boom, x_t=x)

    hits = [w for w in caught if "EnergyConservation" in str(w.message)]
    assert len(hits) == 1, f"expected 1 warning, got {len(hits)}"
    assert rl.FALLBACK_COUNTS["EnergyConservation"] == 4


def test_a_term_that_does_not_raise_leaves_the_counter_empty():
    """The counter must not fire on the healthy path, including weight=0."""
    _reset()
    x = torch.randn(4, 3)

    disabled = rl.EnergyConservation(weight=0.0, enabled=False)
    assert float(disabled.compute(pred_fn=_boom, x_t=x)) == 0.0
    assert rl.FALLBACK_COUNTS == {}, (
        "weight<=0 short-circuits before the try block, so nothing was swallowed"
    )

    healthy = rl.EnergyConservation(weight=0.05, enabled=True)
    out = healthy.compute(pred_fn=lambda v: v * 0.5, x_t=x)
    assert torch.isfinite(out).all()
    assert rl.FALLBACK_COUNTS == {}
