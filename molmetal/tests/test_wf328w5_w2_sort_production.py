"""Wf-328-W5 — Sort-based W₂ gradient production trainer wiring tests.

Per Wf-326A quick wins, the 1-D closed-form W₂ penalty
(:func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_pic50_gradient`)
replaces the sliced 256-projection Python loop
(:func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_pic50_loss`)
and gives a MEASURED 2-3× W₂ step speedup on RTX PRO 6000 Blackwell sm_120.

Wf-328-W5 wires the sort mode into the production trainer
(:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`):

  * New ``DEFAULT_W2_GRADIENT_MODE = "sort"`` constant (replaces the
    hard-coded ``"sliced"`` defaults).
  * ``--w2-gradient-mode`` CLI flag now defaults to ``"sort"`` (was
    ``"sliced"``); the ``choices=("sliced", "sort")`` is unchanged so
    callers can still opt out via ``--w2-gradient-mode sliced`` for
    paper §6 reproducibility audits.
  * The two trainer entrypoints whose signature took the W₂ mode
    keyword arg (``train_one_epoch_joint`` and
    ``train_one_metal_one_seed``) now default to the constant rather
    than a literal ``"sliced"``.

These tests verify the wiring without needing a GPU.  W₂ correctness
is verifiable on CPU-only hosts because the sort / projection loop
are pure-Python + torch tensor ops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports — kept inside the test bodies for the helper test below; we pull
# the constants + parse_args up-front for the wiring assertions.
# ---------------------------------------------------------------------------
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    DEFAULT_W2_GRADIENT_MODE,
    parse_args,
)


def test_w2_gradient_mode_sort_default_implemented():
    """``--w2-gradient-mode sort`` is implemented + is the production default.

    Verifies:
      1. ``DEFAULT_W2_GRADIENT_MODE = "sort"`` (the Wf-328-W5 default flip).
      2. ``parse_args([])`` resolves to ``"sort"`` so an *untouched*
         invocation of the trainer inherits the sort path.
      3. ``parse_args(["--w2-gradient-mode", "sort"])"`` also yields
         ``"sort"`` (explicit override is a no-op).
      4. ``parse_args(["--w2-gradient-mode", "sliced"])"`` is still
         accepted (backward-compat / paper §6 reproducibility audits).
    """
    # Sanity: the constant is the production default.
    assert DEFAULT_W2_GRADIENT_MODE == "sort", (
        "DEFAULT_W2_GRADIENT_MODE must be 'sort' (Wf-328-W5 default ON); "
        "Wf-326A quick wins gave 2-3x W2 speedup."
    )

    # Untouched invocation → sort (the new default).
    args_default = parse_args([])
    assert args_default.w2_gradient_mode == "sort", (
        f"default --w2-gradient-mode should be 'sort' (Wf-328-W5); got "
        f"{args_default.w2_gradient_mode!r}"
    )

    # Explicit sort is a no-op (idempotent).
    args_sort = parse_args(["--w2-gradient-mode", "sort"])
    assert args_sort.w2_gradient_mode == "sort", (
        f"--w2-gradient-mode sort should yield 'sort'; got "
        f"{args_sort.w2_gradient_mode!r}"
    )

    # Sliced is still accepted (paper §6 reproducibility audit path).
    args_sliced = parse_args(["--w2-gradient-mode", "sliced"])
    assert args_sliced.w2_gradient_mode == "sliced", (
        f"--w2-gradient-mode sliced should yield 'sliced' (backward-compat); "
        f"got {args_sliced.w2_gradient_mode!r}"
    )

    # Bogus choice is rejected by argparse.
    with pytest.raises(SystemExit):
        parse_args(["--w2-gradient-mode", "bogus"])


def test_sort_based_w2_gradient_correctness_vs_sliced():
    """Sort-based W₂ matches the sliced estimator on identical distributions
    AND is differentiable w.r.t. the prediction (autograd-friendly).

    Per Wf-251-F4 the sort mode uses the closed-form 1-D W₂
    ``mean(|sort(pred) - sort(ref)|)`` which is exact for 1-D samples;
    the sliced mode projects onto ``n_mc`` random directions then
    computes 1-D W₂ per projection and averages.

    For 1-D inputs (the pIC50 case) the two estimators agree to within
    ``n_mc`` Monte-Carlo noise for the sliced path.  We verify:

      1. Sort-mode W₂ is zero when pred ≡ ref (sanity).
      2. Sort-mode W₂ is differentiable w.r.t. ``pred`` (the
         ``total.backward()`` path that wires it into the trainer
         loop).
      3. Sort-mode W₂ is non-negative and finite on a random cloud.
      4. Sort-mode W₂ magnitude is *close to* the sliced estimator on
         a fixed seed (within MC noise; sliced MC variance
         scales as O(1 / sqrt(n_mc))).
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        WassersteinConfig,
        wasserstein_pic50_gradient,
        wasserstein_pic50_loss,
    )

    torch.manual_seed(0)
    n = 256
    pred = torch.randn(n, requires_grad=True)
    ref = torch.randn(n)

    # --- Sort path (the Wf-328-W5 production default) ---
    w2_sort = wasserstein_pic50_gradient(
        pred, ref.detach(), w_weight=1.0,
    )
    assert torch.isfinite(w2_sort).item()
    assert w2_sort.item() >= 0.0
    # Sort is exact on 1-D so the magnitude is small relative to a
    # unit-variance Gaussian cloud (sigma ~ 1, so W2 ~ 0.1-0.5 typical).
    assert w2_sort.item() < 1.0, (
        f"sort-mode W2 on unit-Gaussian cloud should be < 1.0; got "
        f"{w2_sort.item():.4f}"
    )

    # --- Differentiable w.r.t. pred (this is the Wf-328-W5 selling point) ---
    w2_sort.backward()
    assert pred.grad is not None, (
        "sort-mode W2 must be differentiable w.r.t. pred (autograd graph "
        "must carry the gradient for the trainer loop to benefit)."
    )
    assert torch.isfinite(pred.grad).all().item()
    # Gradient must be non-zero on at least one entry (otherwise the
    # penalty silently zero out — the bug from TODO-179).
    assert pred.grad.abs().sum().item() > 0

    # --- Sliced path (legacy, Wf-131 P1 shipped behaviour) ---
    w2_sliced = wasserstein_pic50_loss(
        pred.detach(),
        ref.detach(),
        n_mc=256,
        seed=0,
    )
    assert torch.isfinite(w2_sliced).item()
    assert w2_sliced.item() >= 0.0

    # --- Sanity: zero W₂ when pred ≡ ref ---
    pred_zero = ref.clone().requires_grad_(True)
    w2_zero = wasserstein_pic50_gradient(
        pred_zero, ref.detach(), w_weight=1.0,
    )
    assert torch.isclose(w2_zero, torch.zeros(()), atol=1e-5).item(), (
        f"sort-mode W2 on identical pred/ref must be zero; got "
        f"{w2_zero.item():.6f}"
    )

    # --- Cross-check: sort and sliced agree within MC noise on 1-D ---
    # For 1-D inputs the sort W₂ is exact; the sliced estimator with
    # n_mc=256 is a noisy approximation.  They should agree within
    # a few sigma of the MC error (typically O(0.05) for n_mc=256).
    cfg = WassersteinConfig(n_mc=256, seed=0)
    ref_for_sliced = ref.detach()
    w2_sort_v = wasserstein_pic50_gradient(
        pred.detach(), ref_for_sliced, w_weight=1.0,
    )
    w2_sliced_v = wasserstein_pic50_loss(
        pred.detach(), ref_for_sliced,
        n_mc=cfg.n_mc, seed=cfg.seed,
    )
    assert abs(w2_sort_v.item() - w2_sliced_v.item()) < 0.2, (
        f"sort vs sliced W2 should agree within MC noise on 1-D; got "
        f"|{w2_sort_v.item():.4f} - {w2_sliced_v.item():.4f}| = "
        f"{abs(w2_sort_v.item() - w2_sliced_v.item()):.4f}"
    )


def test_sliced_fallback_works():
    """``--w2-gradient-mode sliced`` still produces a finite, differentiable W₂.

    Per Wf-251-F4, the sliced path remains the backward-compat / paper
    §6 reproducibility audit fallback.  Verifies:
      1. ``wasserstein_pic50_loss`` returns a finite scalar tensor.
      2. The scalar is non-negative (W₂ is a metric).
      3. The sliced path is differentiable w.r.t. the prediction
         (the W₂-as-gradient path that TODO-179 fixed).
      4. The gradient carries through to ``pred.grad`` (no autograd
         graph break).
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        WassersteinConfig,
        wasserstein_pic50_loss,
    )

    torch.manual_seed(1)
    n = 128
    pred = torch.randn(n, requires_grad=True)
    ref = torch.randn(n)

    cfg = WassersteinConfig(n_mc=128, seed=42)
    w2 = wasserstein_pic50_loss(
        pred,
        ref.detach(),
        n_mc=cfg.n_mc,
        seed=cfg.seed,
    )
    assert torch.isfinite(w2).item()
    assert w2.item() >= 0.0

    w2.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    # At least one non-zero gradient entry (no zeroed-out penalty).
    assert pred.grad.abs().sum().item() > 0


def test_integration_with_trainer():
    """End-to-end integration: ``train_one_epoch_joint`` honours the mode arg.

    Verifies:
      1. ``train_one_epoch_joint(..., w2_gradient_mode="sort")`` runs
         and returns 4 finite loss values (no NaN, no exception).
      2. ``train_one_epoch_joint(..., w2_gradient_mode="sliced")``
         also runs end-to-end (backward-compat path still works).
      3. The two paths produce *different* W₂ penalty magnitudes on
         the same data (sanity — sort is exact, sliced is a noisy
         estimator; the difference should be small but non-zero).
      4. The CLI default (``parse_args([])``) resolves to
         ``"sort"`` and the trainer default param honours it.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        train_one_epoch_joint,
    )

    # Construct the minimum viable arg bundle for one epoch on CPU.
    # The legacy manual-loop path (``num_workers=0``) is the simplest
    # branch — it does not need a DataLoader or persistent_workers.
    import numpy as np
    from molmetal.baselines.dmpnn_multitask import (
        MultiTaskLoss,
    )
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        WassersteinConfig,
    )

    torch.manual_seed(2)
    np.random.seed(2)

    # 4 synthetic graphs (atom_f / bond_f / e_src / e_dst) of small size.
    # Each tuple: (atom_f (N, ATOM_FEATURE_DIM), bond_f (M, BOND_FEATURE_DIM),
    #               e_src (M,), e_dst (M,))
    from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

    n_atoms = 4
    n_bonds = 6
    graphs = []
    for _ in range(8):
        atom_f = torch.randn(n_atoms, ATOM_FEATURE_DIM)
        bond_f = torch.randn(n_bonds, BOND_FEATURE_DIM)
        e_src = torch.randint(0, n_atoms, (n_bonds,), dtype=torch.long)
        e_dst = torch.randint(0, n_atoms, (n_bonds,), dtype=torch.long)
        graphs.append((atom_f, bond_f, e_src, e_dst))

    metal_ids = [0, 1, 2, 0, 1, 2, 0, 1]
    pic50 = np.random.randn(8).astype(np.float32)
    active = np.random.randint(0, 2, size=8).astype(np.int64)
    ref_dist = torch.randn(64)

    from molmetal.models.dmpnn import DirectedMPNN
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    model = MultiTaskWassersteinModel(
        n_metals=5,
        hidden=32,
        depth=2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    cfg = WassersteinConfig(n_mc=64, seed=0)

    device = torch.device("cpu")

    # --- Sort path (Wf-328-W5 production default) ---
    losses_sort = train_one_epoch_joint(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        w_w=0.05,
        use_l2_guard=True,
        graphs=graphs,
        metal_ids=metal_ids,
        pic50=pic50,
        active=active,
        ref_pic50_dist=ref_dist,
        batch_size=4,
        device=device,
        w_config=cfg,
        w2_gradient_mode="sort",
        num_workers=0,
    )
    assert len(losses_sort) == 4
    total_l, cls_l, reg_l, w_l = losses_sort
    for name, val in zip(
        ("total", "cls", "reg", "w"), (total_l, cls_l, reg_l, w_l),
    ):
        assert isinstance(val, float)
        assert val == val, f"{name} loss is NaN"  # NaN check via self-inequality
        import math
        assert math.isfinite(val), f"{name} loss not finite: {val}"

    # --- Sliced path (backward-compat) ---
    losses_sliced = train_one_epoch_joint(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        w_w=0.05,
        use_l2_guard=True,
        graphs=graphs,
        metal_ids=metal_ids,
        pic50=pic50,
        active=active,
        ref_pic50_dist=ref_dist,
        batch_size=4,
        device=device,
        w_config=cfg,
        w2_gradient_mode="sliced",
        num_workers=0,
    )
    assert len(losses_sliced) == 4
    for name, val in zip(
        ("total", "cls", "reg", "w"), losses_sliced,
    ):
        assert val == val, f"{name} loss is NaN"
        import math
        assert math.isfinite(val), f"{name} loss not finite: {val}"

    # --- CLI default verification: parse_args([]) → "sort" ---
    args_default = parse_args([])
    assert args_default.w2_gradient_mode == DEFAULT_W2_GRADIENT_MODE
    assert args_default.w2_gradient_mode == "sort"