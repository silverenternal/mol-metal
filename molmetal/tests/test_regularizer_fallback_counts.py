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
    rl.FALLBACK_REASONS.clear()
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


def test_the_reason_is_recorded_per_exception_type():
    """A lost term is either a memory problem or a maths problem.

    MEASURED 2026-10-21: ``EnergyConservation`` died of CUDA OOM and
    ``SBOO3Term`` of a cusolver error in the same run.  The class-level count
    cannot tell them apart, and they need different repairs.
    """
    _reset()
    x = torch.randn(4, 3)
    oom = rl.EnergyConservation(weight=0.05, enabled=True)
    solver = rl.SBOO3Term(weight=0.05, enabled=True)

    def _raise_oom(_v):
        raise MemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def _raise_solver(_v):
        raise RuntimeError("cusolver error: 6")

    oom.compute(pred_fn=_raise_oom, x_t=x)
    oom.compute(pred_fn=_raise_oom, x_t=x)
    solver.compute(
        per_sample_loss=torch.randn(4, 1), pred_fn=_raise_solver, x_t=x
    )

    assert rl.FALLBACK_COUNTS == {"EnergyConservation": 2, "SBOO3Term": 1}
    assert rl.FALLBACK_REASONS["EnergyConservation"] == {"MemoryError": 2}
    assert rl.FALLBACK_REASONS["SBOO3Term"] == {"RuntimeError": 1}


def test_two_exception_types_on_one_class_are_counted_separately():
    _reset()
    x = torch.randn(4, 3)
    term = rl.EnergyConservation(weight=0.05, enabled=True)
    calls = [RuntimeError, MemoryError, RuntimeError]

    def _raise(kind):
        def _fn(_v):
            raise kind("boom")
        return _fn

    for kind in calls:
        term.compute(pred_fn=_raise(kind), x_t=x)

    assert rl.FALLBACK_COUNTS["EnergyConservation"] == 3
    assert rl.FALLBACK_REASONS["EnergyConservation"] == {
        "RuntimeError": 2,
        "MemoryError": 1,
    }
