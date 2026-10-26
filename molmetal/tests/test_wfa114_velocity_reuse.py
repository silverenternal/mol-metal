"""Wf-A114 — the regularisers must not recompute the main velocity pass.

MEASURED 2026-10-26: one optimiser step ran the velocity field **11 times**
(``fm_velocity_calls/step``), of which only one was the main FM term.  Three
of the others were byte-identical recomputations of ``pred_fn(x_t)``:

* ``EnergyConservation``: ``v0 = pred_fn(x_t)``;
* ``SBOO3Term``: ``pred_fn(x_t)`` **inside** the rotation loop, so ``n_rot``
  copies of a quantity that does not depend on the rotation;
* ``shortcut_distillation_loss``: its first term IS the main FM regression.

On top of the wasted forward, the field carries dropout, so each
recomputation returned a different sample than the one the main term was
being fitted to — the regulariser and the objective were optimising two
different functions of the same weights.

These tests pin the reuse: the caller hands over its own ``pred_fn(x_t)``,
the number of forwards drops, and the value does not change.
"""

from __future__ import annotations

import torch

from molmetal.loss.regularization_losses import EnergyConservation, SBOO3Term


def _counting_field(scale: float = 0.3):
    calls: list = []

    def pred_fn(x: torch.Tensor) -> torch.Tensor:
        calls.append(int(x.shape[0]))
        return x * scale

    return pred_fn, calls


def test_energy_conservation_reuses_the_main_pass():
    torch.manual_seed(0)
    x_t = torch.randn(12, 3)
    pred_fn, calls = _counting_field()
    v_x_t = pred_fn(x_t)
    calls.clear()

    term = EnergyConservation(weight=1.0, enabled=True)
    out = term.compute(pred_fn=pred_fn, x_t=x_t, v_x_t=v_x_t)

    assert calls == [12], f"expected 1 forward (v1 only), got {len(calls)}"
    assert out.item() >= 0.0


def test_energy_conservation_value_is_unchanged_by_the_reuse():
    torch.manual_seed(0)
    x_t = torch.randn(12, 3)
    pred_fn, _ = _counting_field()
    term = EnergyConservation(weight=1.0, enabled=True)

    baseline = term.compute(pred_fn=pred_fn, x_t=x_t)
    reused = term.compute(pred_fn=pred_fn, x_t=x_t, v_x_t=pred_fn(x_t))

    assert torch.allclose(baseline, reused), (
        f"reuse changed the term: {baseline.item():.6f} vs {reused.item():.6f}"
    )


def test_sboo3_evaluates_v_x_t_once_outside_the_rotation_loop():
    """n_rot rotations must cost n_rot + 1 forwards, not 2 * n_rot."""
    torch.manual_seed(0)
    x_t = torch.randn(16, 3)
    per_sample = torch.rand(16)
    pred_fn, calls = _counting_field()

    term = SBOO3Term(weight=1.0, enabled=True, alpha=0.0, n_perm=2, n_rot=3)
    term.compute(per_sample_loss=per_sample, pred_fn=pred_fn, x_t=x_t)

    assert len(calls) == 4, (
        f"expected 3 rotated + 1 shared forward, got {len(calls)}"
    )


def test_sboo3_reuses_the_main_pass():
    torch.manual_seed(0)
    x_t = torch.randn(16, 3)
    per_sample = torch.rand(16)
    pred_fn, calls = _counting_field()
    v_x_t = pred_fn(x_t)
    calls.clear()

    term = SBOO3Term(weight=1.0, enabled=True, alpha=0.0, n_perm=2, n_rot=3)
    term.compute(
        per_sample_loss=per_sample, pred_fn=pred_fn, x_t=x_t, v_x_t=v_x_t,
    )

    assert len(calls) == 3, f"expected only the 3 rotated forwards, got {len(calls)}"


def test_sboo3_value_is_unchanged_by_the_reuse():
    x_t = torch.randn(16, 3)
    per_sample = torch.rand(16)
    pred_fn, _ = _counting_field()
    term = SBOO3Term(weight=1.0, enabled=True, alpha=0.0, n_perm=2, n_rot=3)

    torch.manual_seed(11)
    baseline = term.compute(
        per_sample_loss=per_sample, pred_fn=pred_fn, x_t=x_t,
    )
    v_x_t = pred_fn(x_t)
    torch.manual_seed(11)
    reused = term.compute(
        per_sample_loss=per_sample, pred_fn=pred_fn, x_t=x_t, v_x_t=v_x_t,
    )

    assert torch.allclose(baseline, reused), (
        f"reuse changed the term: {baseline.item():.6f} vs {reused.item():.6f}"
    )


def test_disabled_terms_still_cost_nothing():
    """The reuse must not make a disabled term reach the field at all."""
    x_t = torch.randn(8, 3)
    pred_fn, calls = _counting_field()

    EnergyConservation(weight=0.0, enabled=False).compute(
        pred_fn=pred_fn, x_t=x_t,
    )
    SBOO3Term(weight=0.0, enabled=False, alpha=0.5, n_perm=2, n_rot=2).compute(
        per_sample_loss=torch.rand(8), pred_fn=pred_fn, x_t=x_t,
    )

    assert calls == [], f"disabled terms ran {len(calls)} forwards"
