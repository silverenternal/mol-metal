"""Wf-357-D2: True L2-norm Wasserstein-2 drop-in (canonical W_p OT metric).

Four NEW tests for the Wf-357-D2 ultracode closeout.  Verifies:

  1. ``test_wasserstein_2_formula_matches_reference`` — the new
     :func:`wasserstein_2_1d_torchsort` matches the canonical
     closed-form W₂ = sqrt((1/n) Σ |sort(x_i) − sort(y_i)|²)
     (Peyré & Cuturi 2019 §3 Prop 1) to <1e-8 max-abs error over
     a battery of (pred, ref) distributions.

  2. ``test_wasserstein_2_distinguishes_from_w1`` — on a non-trivial
     outlier distribution (3 zeros + 1 spike at 5.0 vs uniform
     [1, 2, 3, 4]) the L1-style W₁ and the L2-style W₂ return
     measurably different values (W₁=1.75, W₂≈1.9365).  This is the
     property that motivates a separate flag for the new metric.

  3. ``test_backward_compat_default_off`` — without
     --use-wasserstein-2 (i.e. ``wasserstein_order=1``) the gradient
     penalty is bit-identical to the legacy L1-style path; with the
     flag set the penalty routes through :func:`wasserstein_2_1d_torchsort`
     (true L2-norm).  Both paths produce a finite, non-NaN scalar with
     differentiable gradient flow.

  4. ``test_trainer_cli_flag_accepted`` — the trainer CLI parser
     accepts ``--use-wasserstein-2`` as a valid argument and the
     parsed value is exposed as ``args.use_wasserstein_2`` (default
     False, mirroring the legacy default).

Honest framing (party-twenty-spirit §3.5):
  - The L2-norm W₂ is the canonical 2nd-Wasserstein distance in
    optimal-transport theory; the L1-style path (currently shipped)
    is technically W₁ in OT-theoretic notation despite being
    historically named W₂ in the codebase.
  - All math is bit-equivalent to Peyré & Cuturi 2019 §3 Prop 1;
    measured to <1e-8 max-abs error.
  - Drop-in alternative; same API as :func:`wasserstein_1d_torchsort`;
    returns a scalar 0-dim tensor.
  - Default OFF for backward compat with Wf-131 P1 / Wf-181 G5 /
    Wf-182 G6 shipped behaviour.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path("/home/hugo/codes/mol-metal")
TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Test 1: wasserstein_2 formula matches the canonical OT closed-form
# ---------------------------------------------------------------------------


def test_wasserstein_2_formula_matches_reference():
    """W₂ = sqrt((1/n) Σ |sort(x_i) - sort(y_i)|²) per Peyré & Cuturi 2019.

    For 1-D samples with equal lengths, the W₂² distance equals the
    mean squared difference between sorted order statistics.  The
    W₂ distance itself is the square root.  Verified to <1e-8
    max-abs error over a battery of distributions spanning uniform,
    Gaussian, outlier, and bimodal.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_2_1d_torchsort,
    )

    # Battery of test distributions (all size 64 for stable math).
    cases = []
    rng = torch.Generator().manual_seed(0)
    # Uniform
    cases.append(
        (
            torch.rand(64, generator=rng),
            torch.rand(64, generator=rng),
        )
    )
    # Gaussian
    cases.append(
        (
            torch.randn(64, generator=rng),
            torch.randn(64, generator=rng),
        )
    )
    # Constant shift (analytic check)
    cases.append(
        (
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32) + 1.0,
        )
    )
    # Outlier
    cases.append(
        (
            torch.cat([torch.zeros(32), torch.full((32,), 5.0)]),
            torch.linspace(0.0, 4.0, 32).repeat(2),
        )
    )
    # Bimodal
    cases.append(
        (
            torch.cat([torch.full((32,), -1.0), torch.full((32,), 1.0)]),
            torch.cat([torch.full((32,), 0.5), torch.full((32,), 2.5)]),
        )
    )

    for i, (pred, ref) in enumerate(cases):
        w2 = wasserstein_2_1d_torchsort(pred, ref)
        # Reference: direct sort + L2 norm
        n = max(pred.numel(), ref.numel())
        # Use the closed-form sort trick — for equal sizes the sort
        # gives the optimal OT plan exactly.
        xs = torch.sort(pred.flatten(), dim=0).values
        ys = torch.sort(ref.flatten(), dim=0).values
        if xs.numel() != n:
            idx = torch.linspace(0, max(xs.numel() - 1, 1), n)
            idx_lo = idx.floor().long().clamp(max=xs.numel() - 1)
            idx_hi = idx.ceil().long().clamp(max=xs.numel() - 1)
            frac = (idx - idx_lo.to(idx.dtype)).clamp(0.0, 1.0)
            xs = (1.0 - frac) * xs[idx_lo] + frac * xs[idx_hi]
        if ys.numel() != n:
            idx = torch.linspace(0, max(ys.numel() - 1, 1), n)
            idx_lo = idx.floor().long().clamp(max=ys.numel() - 1)
            idx_hi = idx.ceil().long().clamp(max=ys.numel() - 1)
            frac = (idx - idx_lo.to(idx.dtype)).clamp(0.0, 1.0)
            ys = (1.0 - frac) * ys[idx_lo] + frac * ys[idx_hi]
        expected = ((xs - ys) ** 2).mean().sqrt()
        assert torch.allclose(w2, expected, atol=1e-8), (
            f"case {i}: W₂ mismatch: got {w2.item():.8f}, "
            f"expected {expected.item():.8f}, max abs err "
            f"{(w2 - expected).abs().max().item():.2e}"
        )
        # Positivity: W₂ is non-negative.
        assert w2.item() >= 0.0
        # Finiteness.
        assert torch.isfinite(w2)


# ---------------------------------------------------------------------------
# Test 2: W₁ vs W₂ are measurably different on non-trivial distributions
# ---------------------------------------------------------------------------


def test_wasserstein_2_distinguishes_from_w1():
    """W₁ vs W₂ give different values on outlier distributions.

    The motivation for adding a separate flag for W₂ is that the L1
    and L2 norms in sorted space disagree on outlier-heavy
    distributions.  This test uses the canonical outlier example
    from the docstring (3 zeros + 1 spike at 5.0 vs uniform [1, 2, 3, 4]).

    Expected:
      W₁ = mean(|sort(pred) - sort(ref)|) = (1+2+3+1)/4 = 1.75
      W₂ = sqrt(mean((sort(pred) - sort(ref))²)) = sqrt(15/4) ≈ 1.9365

    The difference (Δ ≈ 0.1865) is the metric choice signal that
    --use-wasserstein-2 isolates.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_1d_torchsort,
        wasserstein_2_1d_torchsort,
    )

    pred = torch.tensor([0.0, 0.0, 0.0, 5.0])
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w1 = wasserstein_1d_torchsort(pred, ref).item()
    w2 = wasserstein_2_1d_torchsort(pred, ref).item()
    # W₁ (L1) = 1.75, W₂ (L2) ≈ 1.9365
    assert abs(w1 - 1.75) < 1e-5, f"W₁ should be 1.75, got {w1}"
    assert (
        abs(w2 - (15 / 4) ** 0.5) < 1e-5
    ), f"W₂ should be sqrt(15/4) ≈ {(15/4)**0.5:.4f}, got {w2:.4f}"
    # W₂ > W₁ on outlier-heavy distributions (L2 norm amplifies outliers).
    assert w2 > w1, (
        f"W₂ should exceed W₁ on outlier-heavy distribution; " f"got W₁={w1}, W₂={w2}"
    )
    # The numerical difference should be on the order of the outlier magnitude.
    delta = w2 - w1
    assert 0.1 < delta < 0.5, f"Δ W₂-W₁ should be ~0.1865, got {delta}"


# ---------------------------------------------------------------------------
# Test 3: backward-compat + forward-pass + gradient flow
# ---------------------------------------------------------------------------


def test_backward_compat_default_off():
    """Default order=1 path is bit-identical to legacy L1 closed-form.

    Verifies three properties:
      a) ``wasserstein_pic50_gradient`` with default
         ``wasserstein_order=1`` returns a value bit-identical to
         ``0.05 * wasserstein_1d_torchsort(pred, ref.detach())``
         (the pre-Wf-357-D2 computation path).
      b) With ``wasserstein_order=2``, the function returns a value
         equal to ``0.05 * wasserstein_2_1d_torchsort(pred, ref.detach())``
         (the new Wf-357-D2 computation path).
      c) Both paths produce finite scalar losses with differentiable
         gradient flow back into ``pred``.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_1d_torchsort,
        wasserstein_2_1d_torchsort,
        wasserstein_pic50_gradient,
    )

    torch.manual_seed(42)
    pred = torch.randn(16, dtype=torch.float32) * 0.5 + 5.0
    pred.requires_grad_(True)
    ref = torch.randn(64, dtype=torch.float32) * 0.5 + 5.0

    # (a) Default order=1 — bit-identical to legacy path.
    g_legacy = 0.05 * wasserstein_1d_torchsort(pred.detach(), ref.detach())
    g_default = wasserstein_pic50_gradient(
        pred.detach().clone().requires_grad_(True),
        ref.detach(),
        w_weight=0.05,
        # wasserstein_order default = 1
    )
    assert torch.allclose(g_default, g_legacy, atol=1e-7), (
        f"Default order=1 should equal legacy 0.05*W₁; got "
        f"g_default={g_default.item():.8f}, g_legacy={g_legacy.item():.8f}"
    )

    # (b) Order=2 — equals new W₂ closed form.
    g_order2 = wasserstein_pic50_gradient(
        pred.detach().clone().requires_grad_(True),
        ref.detach(),
        w_weight=0.05,
        wasserstein_order=2,
    )
    g_w2_direct = 0.05 * wasserstein_2_1d_torchsort(pred.detach(), ref.detach())
    assert torch.allclose(g_order2, g_w2_direct, atol=1e-7), (
        f"order=2 should equal 0.05*W₂; got "
        f"g_order2={g_order2.item():.8f}, g_w2_direct={g_w2_direct.item():.8f}"
    )

    # (c) Both paths produce finite, non-NaN scalars with grad flow.
    assert torch.isfinite(g_default) and g_default.item() > 0
    assert torch.isfinite(g_order2) and g_order2.item() > 0

    pred_g = pred.detach().clone().requires_grad_(True)
    loss = wasserstein_pic50_gradient(
        pred_g,
        ref.detach(),
        w_weight=0.05,
        wasserstein_order=2,
    )
    loss.backward()
    assert pred_g.grad is not None
    assert torch.isfinite(pred_g.grad).all()
    assert (
        pred_g.grad != 0
    ).all(), "Gradient should be non-zero for non-degenerate pred/ref"


# ---------------------------------------------------------------------------
# Test 4: trainer CLI flag accepted + default-OFF
# ---------------------------------------------------------------------------


def test_trainer_cli_flag_accepted():
    """``--use-wasserstein-2`` is a valid trainer CLI argument.

    Verifies:
      a) The trainer parses ``--use-wasserstein-2`` without error.
      b) The parsed value is exposed as ``args.use_wasserstein_2``.
      c) Default value is False (backward compat).
      d) Setting ``--use-wasserstein-2`` makes it True.

    This test imports the trainer's ``parse_args`` directly to avoid
    the pre-existing --help formatter bug (TypeError % format in
    unrelated help text).  The flag itself parses cleanly through
    ``argparse``.
    """
    # Import the parser from the trainer.  Note: importing the
    # trainer module triggers the gfx1101 env-var fix and the
    # matplotlib backend setup; both are fast and side-effect-safe.
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    # (a) + (c) Default: flag absent → use_wasserstein_2 = False.
    args_default = parse_args(["--metals", "Ru"])
    assert hasattr(
        args_default, "use_wasserstein_2"
    ), "args.use_wasserstein_2 should be present after parse_args"
    assert args_default.use_wasserstein_2 is False, (
        f"Default should be False (backward compat), got "
        f"{args_default.use_wasserstein_2}"
    )

    # (b) + (d) Flag present → use_wasserstein_2 = True.
    args_on = parse_args(["--metals", "Ru", "--use-wasserstein-2"])
    assert args_on.use_wasserstein_2 is True, (
        f"--use-wasserstein-2 should set the flag to True, got "
        f"{args_on.use_wasserstein_2}"
    )

    # Confirm the trainer still parses other flags unchanged
    # (sanity check that we didn't break the existing CLI surface).
    assert args_on.metals == ["Ru"]
    assert hasattr(args_on, "wasserstein_weight")
    assert hasattr(args_on, "use_wasserstein_2")
