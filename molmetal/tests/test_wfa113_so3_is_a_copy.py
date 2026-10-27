# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T40 — the SO(3)-averaged FM term is a scaled copy of the FM MSE.

THE CLAIM
---------
``so3_average_loss`` averages ``||v(R x_t) - R target||^2`` over K random
rotations R.  For a field that is equivariant by construction — which the EGNN
velocity field is — ``v(R x) = R v(x)``, so every term of that average is the
UNROTATED MSE and the result is the FM MSE itself.  The module's own docstring
says so ("For a velocity field that is equivariant by construction this loss is
identical to the single-sample FM loss").  What was never checked is whether
that is what the trainer's run actually produced.

MEASURED on the only run that enabled it
(molmetal/reports/cfm_c8_fullgen_2026_10_24/events.jsonl, 3 epochs):
``so3_term / fm_term`` = 0.1000000026, 0.1000000020, 0.1000000020.  That is
``so3_average_weight`` (0.1) to 9 significant figures, at every epoch.  So the
term adds no gradient direction of its own; it multiplies the FM gradient by
(1 + 0.1).

WHY THE CODE PATH IS KEPT
-------------------------
Deleting it would change what an archived config means, and the kill clause in
T40 ("does the fm trajectory move > 2% without it") cannot be answered without
a 288-step run.  So the path stays, and the inflation is printed at launch.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
C8_EVENTS = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c8_fullgen_2026_10_24"
    / "events.jsonl"
)


def _epoch_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "train_epoch":
            continue
        lt = e.get("loss_terms")
        if isinstance(lt, str):
            lt = ast.literal_eval(lt)
        if lt:
            rows.append((int(e["epoch"]), float(lt["fm_term"]), float(lt["so3_term"])))
    return rows


@pytest.mark.skipif(not C8_EVENTS.is_file(), reason="c8 events file not on disk")
def test_c8_ratio_is_the_weight_at_every_epoch():
    rows = _epoch_rows(C8_EVENTS)
    assert rows, "no train_epoch rows in the c8 events file"
    ratios = [so / fm for _, fm, so in rows if fm]
    assert ratios, "no epoch recorded a non-zero fm_term"
    for (ep, fm, so), r in zip(rows, ratios):
        assert r == pytest.approx(0.1, abs=1e-6), (
            f"epoch {ep}: so3_term/fm_term = {r!r}, not so3_average_weight 0.1; "
            "the term is NOT a pure copy and T40's premise needs re-checking"
        )
    spread = max(ratios) - min(ratios)
    assert spread < 1e-6, f"ratio spread {spread!r} exceeds the 1e-6 copy threshold"


def test_equivariant_prediction_makes_the_average_a_no_op():
    """The mechanism, independent of the run: if v(Rx) = R v(x), the K-rotation
    average equals the unrotated MSE."""
    from molmetal.adapters.flow_matching_lipman import so3_average as so3

    g = torch.Generator().manual_seed(0)
    x_t = torch.randn(4, 5, 3, generator=g)
    target = torch.randn(4, 5, 3, generator=g)

    def equivariant(x):
        # A linear, rotation-equivariant field: v(Rx) = R v(x).
        return 1.5 * x

    plain = ((equivariant(x_t) - target) ** 2).mean()
    avg = so3.so3_average_loss(
        pred_fn=equivariant, x_t=x_t, target=target, n_rotations=4, generator=g
    )
    assert torch.allclose(avg, plain, atol=1e-6), (
        f"SO(3)-averaged loss {avg.item()} != plain MSE {plain.item()} for an "
        "equivariant field; the copy claim does not hold in general"
    )


def test_a_non_equivariant_field_does_break_the_equality():
    """The control: the equality is a property of equivariance, not of the
    averaging code.  Without this, the test above would also pass on a
    function that ignored the rotations."""
    from molmetal.adapters.flow_matching_lipman import so3_average as so3

    g = torch.Generator().manual_seed(1)
    x_t = torch.randn(4, 5, 3, generator=g)
    target = torch.randn(4, 5, 3, generator=g)

    def non_equivariant(x):
        # Absolute-position dependence: v(Rx) != R v(x).
        return 1.5 * x + 0.5

    plain = ((non_equivariant(x_t) - target) ** 2).mean()
    avg = so3.so3_average_loss(
        pred_fn=non_equivariant, x_t=x_t, target=target, n_rotations=8, generator=g
    )
    assert not torch.allclose(avg, plain, atol=1e-6)


def test_trainer_states_the_duplication_at_launch():
    src = (
        REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    assert "scaled copy, not a second objective" in src
    assert "effective FM coefficient" in src
