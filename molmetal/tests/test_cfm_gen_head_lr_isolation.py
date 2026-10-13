"""Regression test — the gen-head LR isolation is a no-op under OneCycle.

=====================================================================
Why this file exists
=====================================================================
``gen_head_lr_scale`` builds three param groups (atom / gen_flow / bond) with
the gen-flow group at ``args.lr * scale``. On the CFM trainer the scale is
``0.1``, i.e. the generation head is meant to peak at 1/10 of the rest.

``build_lr_scheduler``'s onecycle branch calls::

    OneCycleLR(optimizer, max_lr=base_lr * 10, ...)

``max_lr`` is a SCALAR. PyTorch broadcasts it to every param group, so the
per-group ``lr`` written at construction is overwritten and all three groups
train at the same rate. MEASURED on this repo's torch: a group built at
``1e-4 / 1e-5 / 1e-4`` comes out of the scheduler constructor as
``4e-5 / 4e-5 / 4e-5`` — the isolation is gone before the first step, and the
generation head trains at 100x its configured LR.

This matters because ``gen_head_lr_scale=0.1`` was one of the primary levers
proposed to stop the CFM generation collapse, and the ``master.log`` line
``gen_flow_lr=1e-05`` prints the PRE-scheduler value — so it was never evidence
that isolation was in effect.

The test builds a scheduler the same way the trainer does and asserts the
gen group actually stays at ``scale`` of the others. It FAILS today.

Hardware: CPU-only, no GPU, seconds to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402


BASE_LR = 1e-4
GEN_HEAD_LR_SCALE = 0.1
TOTAL_STEPS = 100


def _build_groups() -> AdamW:
    """Mirror the trainer's three-group construction (trainer ~line 7406-7413)."""
    return AdamW(
        [
            {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": BASE_LR},
            {
                "params": [torch.nn.Parameter(torch.zeros(2))],
                "lr": BASE_LR * GEN_HEAD_LR_SCALE,
            },
            {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": BASE_LR},
        ]
    )


def _run_scheduler(opt: AdamW, **max_lr_kwargs) -> list[float]:
    """Drive the scheduler to completion and return each group's peak LR."""
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        total_steps=TOTAL_STEPS,
        pct_start=0.3,
        **max_lr_kwargs,
    )
    peaks = [0.0, 0.0, 0.0]
    for _ in range(TOTAL_STEPS):
        # the real trainer steps the optimiser first; do the same so PyTorch
        # does not emit its "step() before optimizer.step()" warning
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.zeros_like(p)
        opt.step()
        sch.step()
        for i, group in enumerate(opt.param_groups):
            peaks[i] = max(peaks[i], group["lr"])
    return peaks


class TestScalarMax_lrDestroysIsolation:
    def test_scalar_max_lr_collapses_the_groups(self) -> None:
        """The defect, pinned: a scalar max_lr flattens all three groups."""
        opt = _build_groups()
        peaks = _run_scheduler(opt, max_lr=BASE_LR * 10)
        assert peaks[1] == pytest.approx(peaks[0], rel=1e-6), (
            f"scalar max_lr was expected to collapse the groups "
            f"(peaks={peaks}); if this now fails the trainer's fix has landed "
            "and this control needs rewriting"
        )
        assert peaks[1] / peaks[0] == pytest.approx(1.0, rel=1e-6)


class TestPerGroupMax_lrRestoresIsolation:
    def test_per_group_max_lr_preserves_the_scale(self) -> None:
        """What the trainer must do: max_lr as a list, one entry per group."""
        opt = _build_groups()
        peaks = _run_scheduler(
            opt,
            max_lr=[
                BASE_LR * 10,
                BASE_LR * 10 * GEN_HEAD_LR_SCALE,
                BASE_LR * 10,
            ],
        )
        assert peaks[1] / peaks[0] == pytest.approx(GEN_HEAD_LR_SCALE, rel=1e-6), (
            f"per-group max_lr must keep gen_flow at {GEN_HEAD_LR_SCALE}x the "
            f"atom group; measured peaks={peaks}"
        )
        assert peaks[2] / peaks[0] == pytest.approx(1.0, rel=1e-6)

    def test_atom_and_bond_stay_tied_to_each_other(self) -> None:
        opt = _build_groups()
        peaks = _run_scheduler(
            opt,
            max_lr=[
                BASE_LR * 10,
                BASE_LR * 10 * GEN_HEAD_LR_SCALE,
                BASE_LR * 10,
            ],
        )
        assert peaks[0] == pytest.approx(peaks[2], rel=1e-6)


class TestTrainerIsolatesGenHead:
    """The real assertion: the shipped trainer must build the right thing.

    This is the test that fails on the current trainer and passes once
    ``build_lr_scheduler`` passes a per-group ``max_lr`` list. It calls the
    trainer's own helper rather than re-deriving it, so it cannot drift from
    what the trainer actually does.
    """

    def test_trainer_scheduler_preserves_group_ratios(self) -> None:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            build_lr_scheduler,
        )

        opt = _build_groups()
        sched = build_lr_scheduler(
            optimizer=opt,
            total_steps=TOTAL_STEPS,
            scheduler_type="onecycle",
            warmup_steps=0,
            min_lr=BASE_LR / 100,
            pct_start=0.3,
        )
        assert sched is not None, "onecycle scheduler must be built"

        peaks = [0.0, 0.0, 0.0]
        for _ in range(TOTAL_STEPS):
            for group in opt.param_groups:
                for p in group["params"]:
                    p.grad = torch.zeros_like(p)
            opt.step()
            sched.step()
            for i, group in enumerate(opt.param_groups):
                peaks[i] = max(peaks[i], group["lr"])

        ratio = peaks[1] / peaks[0]
        assert ratio == pytest.approx(GEN_HEAD_LR_SCALE, rel=1e-3), (
            f"trainer's onecycle scheduler collapsed the gen-head LR group: "
            f"measured ratio {ratio:.4f}, expected {GEN_HEAD_LR_SCALE}. "
            f"peaks={peaks}. The scalar max_lr=base_lr*10 is broadcast to all "
            f"groups, so gen_flow_lr_scale has no effect at all."
        )


class TestHonestLogging:
    """``events.jsonl`` must carry the LIVE per-group lr, not just the config.

    ``"lr": args.lr`` is the configured base — constant by construction, so a
    scheduler's ramp is invisible in every run log. That is how a scalar
    ``max_lr`` (which flattened all three groups) stayed invisible for a whole
    run while ``master.log`` cheerfully printed ``gen_flow_lr=1e-05`` from the
    pre-scheduler value. Keeping ``lr`` for back-compat is fine; what matters is
    that ``lr_live`` exists.
    """

    def test_event_payload_carries_the_live_per_group_lr(self) -> None:
        from molmetal.scripts import train_cfm_yuelbond_wasserstein as t

        src = Path(t.__file__).read_text(encoding="utf-8")
        assert '"lr_live"' in src, (
            "the events.jsonl payload has no live per-group lr, so no scheduler "
            "can be verified after the fact. Add `lr_live` read from "
            "optimiser.param_groups alongside the existing configured `lr`."
        )