"""Wf-A110 (D11/T2) — checkpointing ``FmOdeTerm`` must not change its maths.

The term holds 16 velocity-field graphs per step by construction.  MEASURED
(``metrics/by_metric/wfa109_fmode_graph_retention_long.json``): ~294 MB each
at 405 atoms, i.e. ~4.7 GB per step, which is the 42.5 GB peak at the
full-corpus batch.  Over 160 steps the live-tensor count is constant at 382,
so this is a budget, not a leak.

The fix recomputes the stage forwards in the backward.  It is only a fix if
the recomputation is exact, which is the property these tests pin — including
for a field with dropout, because ``use_reentrant=False`` saves and restores
the RNG state and the reentrant path does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from molmetal.loss.regularization_losses import FmOdeTerm  # noqa: E402


class _Field(torch.nn.Module):
    """A stand-in velocity field with an optional dropout.

    The real one is an EGNN, but the property under test is the checkpoint
    boundary, not the network: any differentiable map with a random op is
    enough to tell the two code paths apart.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.drop = torch.nn.Dropout(dropout)
        torch.manual_seed(0)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(3) * 0.7)
            self.lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.lin(x)) - 0.3 * x


def _run(use_checkpoint: bool, dropout: float = 0.0, seed: int = 0):
    torch.manual_seed(seed)
    field = _Field(dropout=dropout).train()
    term = FmOdeTerm(weight=1.0, enabled=True, use_checkpoint=use_checkpoint)
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(16, 3, generator=g, requires_grad=True)
    x1 = torch.randn(16, 3, generator=g)
    value = term.compute(pred_fn=field, x0=x0, x1=x1, n_steps=4)
    value.backward()
    return float(value), field.lin.weight.grad.clone(), x0.grad.clone()


def test_checkpointing_preserves_the_value():
    v_plain, _, _ = _run(use_checkpoint=False)
    v_ckpt, _, _ = _run(use_checkpoint=True)
    assert abs(v_plain - v_ckpt) < 1e-6, (v_plain, v_ckpt)


def test_checkpointing_preserves_the_parameter_gradient():
    _, g_plain, _ = _run(use_checkpoint=False)
    _, g_ckpt, _ = _run(use_checkpoint=True)
    assert torch.allclose(g_plain, g_ckpt, atol=1e-7)


def test_checkpointing_preserves_the_input_gradient():
    _, _, x_plain = _run(use_checkpoint=False)
    _, _, x_ckpt = _run(use_checkpoint=True)
    assert torch.allclose(x_plain, x_ckpt, atol=1e-7)


def test_checkpointing_preserves_dropout_masks_across_the_recompute():
    """The reentrant path re-draws masks on recompute and this one must not."""
    v_plain, g_plain, _ = _run(use_checkpoint=False, dropout=0.3, seed=7)
    v_ckpt, g_ckpt, _ = _run(use_checkpoint=True, dropout=0.3, seed=7)
    assert abs(v_plain - v_ckpt) < 1e-6
    assert torch.allclose(g_plain, g_ckpt, atol=1e-7)


def test_the_term_still_works_without_a_grad_requiring_input():
    """The sampler and the eval path call the field under ``no_grad``."""
    field = _Field()
    term = FmOdeTerm(weight=1.0, enabled=True, use_checkpoint=True)
    x0 = torch.randn(8, 3)
    x1 = torch.randn(8, 3)
    with torch.no_grad():
        v = term.compute(pred_fn=field, x0=x0, x1=x1, n_steps=4)
    assert torch.isfinite(v).all()


def test_a_disabled_term_is_still_an_exact_zero():
    term = FmOdeTerm(weight=0.0, enabled=False)
    v = term.compute(pred_fn=_Field(), x0=torch.randn(4, 3), x1=torch.randn(4, 3))
    assert float(v) == 0.0
