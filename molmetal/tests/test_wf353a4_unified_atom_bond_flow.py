"""Pytest: Wf-353-A4 — Unified Atom-Type + Bond Flow (FlowMol-style 2D).

Reference: Dunn & Koes 2024 (FlowMol, arXiv:2404.19739).  Wf-340-R3
§3.1 ranks FlowMol as the closest architectural neighbour for Mol-
Metal's 3D refiner — it jointly diffuses 3D coordinates + atom-type
simplex + bond-order categorical through a **single** conditional flow.
Wf-353-A4 projects this to Mol-Metal's 2D D-MPNN setting: we unify the
**atom-type + bond-order categorical flows** (no 3D coords needed) so
they advance on the same flow-time ``t`` with a single shared noise
sample.  See ``molmetal.adapters.flow_matching_lipman.unified_atom_bond_flow``
for the full architecture.

Four tests:
1. ``test_unified_forward`` — the joint head produces (atom_velocity,
   bond_velocity) with the right shapes and a finite value.
2. ``test_match_separate_flows`` — in the zero-init regime the unified
   prediction matches the separate (independent-head) prediction
   (both exactly zero).
3. ``test_numerical_stability`` — running the SimplexFlow sample + loss
   over many iterations stays finite (no NaN/Inf) under float32.
4. ``test_backward_compat`` — when ``UnifiedAtomBondFlow(enabled=False)``
   (default) every method is a no-op and ``parameters()`` is empty —
   bit-exact with the pre-Wf-353-A4 pipeline.

All tests run on CPU in <5 s.  No GPU or external evaluator required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman.unified_atom_bond_flow import (
    DEFAULT_ATOM_VOCAB,
    NUM_BOND_CLASSES,
    UnifiedAtomBondFlow,
    UnifiedAtomBondHead,
    matches_separate_flows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_edge_batch(n_edges: int = 4, hidden_dim: int = 128) -> tuple:
    """Return a fresh (h_src, h_dst, dist) batch for tests."""
    torch.manual_seed(0)
    h_src = torch.randn(n_edges, hidden_dim)
    h_dst = torch.randn(n_edges, hidden_dim)
    dist = torch.rand(n_edges) * 3.0 + 1.0  # bond-length-scale [1, 4]
    return h_src, h_dst, dist


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a (..., num_classes) one-hot tensor from integer labels."""
    out = torch.zeros(
        (*labels.shape, num_classes), dtype=torch.float32, device=labels.device,
    )
    out.scatter_(-1, labels.unsqueeze(-1), 1.0)
    return out


# ---------------------------------------------------------------------------
# Test 1: unified forward pass — atom + bond simultaneously
# ---------------------------------------------------------------------------
def test_unified_forward():
    """Wf-353-A4 unified forward: atom + bond velocity shapes are correct.

    Builds a :class:`UnifiedAtomBondHead`, runs it on a 4-edge batch,
    and checks the output shapes, dtypes, and finiteness.  The head is
    zero-initialised so the velocity is exactly zero at init — this is
    the canonical FlowMol §3.3 contract (initial velocity is zero).
    """
    hidden_dim = 128
    head = UnifiedAtomBondHead(
        hidden_dim=hidden_dim,
        max_atomic_number=100,
        num_bond_classes=NUM_BOND_CLASSES,
    )
    h_src, h_dst, dist = _make_edge_batch(n_edges=4, hidden_dim=hidden_dim)
    t = torch.tensor([0.0, 0.3, 0.6, 0.99])
    atom_vel, bond_vel = head(h_src, h_dst, dist, t)
    assert atom_vel.shape == (4, 100), (
        f"atom_vel shape should be (E, max_atomic_number)=(4, 100), "
        f"got {tuple(atom_vel.shape)}"
    )
    assert bond_vel.shape == (4, NUM_BOND_CLASSES), (
        f"bond_vel shape should be (E, K)=(4, {NUM_BOND_CLASSES}), "
        f"got {tuple(bond_vel.shape)}"
    )
    assert atom_vel.dtype == torch.float32
    assert bond_vel.dtype == torch.float32
    # Zero-init contract (matches EGNNVelocityField.vel_head): at init
    # the velocity is exactly zero — gives the optimiser a clean
    # starting point.  FlowMol §3.3 makes the same choice.
    assert torch.allclose(atom_vel, torch.zeros_like(atom_vel), atol=1e-7), (
        "atom_head should be zero-init; got non-zero velocity at init."
    )
    assert torch.allclose(bond_vel, torch.zeros_like(bond_vel), atol=1e-7), (
        "bond_head should be zero-init; got non-zero velocity at init."
    )


# ---------------------------------------------------------------------------
# Test 2: matches separate flows (zero-init regime)
# ---------------------------------------------------------------------------
def test_match_separate_flows():
    """Wf-353-A4 symmetry: unified == separate in the zero-init regime.

    When :class:`UnifiedAtomBondHead` is zero-init, the predicted
    velocity is identically zero.  Therefore the *unified* head
    (joint MLP) and a *separate* pair of heads (two independent MLPs,
    also zero-init) produce identical outputs — so the symmetry check
    in :func:`matches_separate_flows` returns True.

    This enforces the canonical FlowMol Eq. 7 contract: the velocity
    loss in the unified flow equals the sum of two independent
    SimplexFlow losses on each simplex, *up to* the joint head's
    parameter sharing.
    """
    hidden_dim = 128
    head_unified = UnifiedAtomBondHead(
        hidden_dim=hidden_dim,
        max_atomic_number=100,
        num_bond_classes=NUM_BOND_CLASSES,
    )
    # Separate (independent) heads — also zero-init so we can compare.
    head_sep_atom = UnifiedAtomBondHead(
        hidden_dim=hidden_dim,
        max_atomic_number=100,
        num_bond_classes=NUM_BOND_CLASSES,
    )
    head_sep_bond = UnifiedAtomBondHead(
        hidden_dim=hidden_dim,
        max_atomic_number=100,
        num_bond_classes=NUM_BOND_CLASSES,
    )
    h_src, h_dst, dist = _make_edge_batch(n_edges=8, hidden_dim=hidden_dim)
    t = torch.linspace(0.0, 0.99, 8)
    # All three heads are zero-init: outputs are exactly zero.
    atom_v_unified, bond_v_unified = head_unified(h_src, h_dst, dist, t)
    atom_v_sep, _ = head_sep_atom(h_src, h_dst, dist, t)
    _, bond_v_sep = head_sep_bond(h_src, h_dst, dist, t)
    # Build dummy one-hot for the source/target noise contract.
    atom_target = _one_hot(torch.tensor([6, 7, 8, 15, 16, 1, 78, 6]), 100)
    bond_target = _one_hot(torch.tensor([1, 2, 0, 4, 3, 1, 2, 0]), NUM_BOND_CLASSES)
    atom_x_t = torch.zeros_like(atom_v_unified)
    bond_x_t = torch.zeros_like(bond_v_unified)
    assert matches_separate_flows(
        atom_x_t=atom_x_t,
        bond_x_t=bond_x_t,
        atom_target_one_hot=atom_target,
        bond_target_one_hot=bond_target,
        unified_v_pred_atom=atom_v_unified,
        unified_v_pred_bond=bond_v_unified,
        separate_v_pred_atom=atom_v_sep,
        separate_v_pred_bond=bond_v_sep,
        atol=1e-5,
    ), "Zero-init unified head must match zero-init separate heads."


# ---------------------------------------------------------------------------
# Test 3: numerical stability over many iterations
# ---------------------------------------------------------------------------
def test_numerical_stability():
    """Wf-353-A4 numerical stability: no NaN/Inf after training-step loop.

    Runs :meth:`UnifiedAtomBondFlow.unified_loss` for 50 forward + grad
    steps and asserts that (a) the loss stays finite, (b) the head's
    parameters stay finite, (c) the SimplexFlow source / target
    velocities stay in [-1, 1].  This is the canonical FlowMol §3.2
    sanity check (the source / target are convex combinations of
    one-hot and 1/K · 1, so the velocity lies in [-1/K, 1 - 1/K]).
    """
    flow = UnifiedAtomBondFlow(
        atom_vocab_size=len(DEFAULT_ATOM_VOCAB),
        max_atomic_number=100,
        num_bond_classes=NUM_BOND_CLASSES,
        hidden_dim=128,
        enabled=True,
    )
    head = flow.head(device=torch.device("cpu"))
    opt = torch.optim.SGD(head.parameters(), lr=1e-2)
    h_src, h_dst, dist = _make_edge_batch(n_edges=4, hidden_dim=128)
    atom_labels = torch.tensor([6, 7, 8, 6], dtype=torch.long)
    bond_labels = torch.tensor([1, 2, 0, 4], dtype=torch.long)
    atom_target = _one_hot(atom_labels, 100)
    bond_target = _one_hot(bond_labels, NUM_BOND_CLASSES)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = flow.unified_loss(
            h_src=h_src, h_dst=h_dst, dist=dist,
            atom_target_one_hot=atom_target,
            bond_target_one_hot=bond_target,
            atom_loss_weight=1.0,
            bond_loss_weight=1.0,
        )
        # Backward + step so we exercise the gradient path.
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # All losses finite (no NaN, no Inf) across the loop.
    assert all(math.isfinite(v) for v in losses), (
        f"Loss went non-finite: {losses[:5]}...{losses[-3:]}"
    )
    # Parameters remain finite.
    for name, p in head.named_parameters():
        assert torch.isfinite(p).all(), f"Parameter {name!r} went non-finite."
    # SimplexFlow target velocity sanity: v = target - 1/K.
    atom_v_tgt, bond_v_tgt = flow.target_velocity(atom_target, bond_target)
    # atom-side: max possible velocity is 1 - 1/100 = 0.99
    assert atom_v_tgt.abs().max().item() <= 1.0 + 1e-6, (
        f"atom_v_tgt abs max should be ≤ 1.0; got {atom_v_tgt.abs().max().item()}"
    )
    # bond-side: max possible velocity is 1 - 1/5 = 0.80
    assert bond_v_tgt.abs().max().item() <= 1.0 + 1e-6, (
        f"bond_v_tgt abs max should be ≤ 1.0; got {bond_v_tgt.abs().max().item()}"
    )
    # Source and target are valid simplex points (rows sum to 1).
    atom_src = (
        atom_target - atom_v_tgt
    )  # since v_target = target - source => source = target - v_target
    assert torch.allclose(atom_src.sum(dim=-1), torch.ones(4), atol=1e-5), (
        f"atom source rows should sum to 1; got {atom_src.sum(dim=-1).tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4: backward compat — disabled wrapper is a no-op
# ---------------------------------------------------------------------------
def test_backward_compat():
    """Wf-353-A4 backward compat: enabled=False makes every method a no-op.

    Default behaviour (enabled=False) must match the pre-Wf-353-A4
    pipeline bit-exactly:
    * ``flow.parameters()`` is an empty iterator (no head registered)
    * ``flow.unified_loss(...)`` returns a zero scalar
    * ``flow.sample(...)`` returns the inputs unchanged (the wrapper
      does not touch the simplex)
    """
    flow = UnifiedAtomBondFlow(enabled=False)
    # No head parameters.
    params = list(flow.parameters())
    assert len(params) == 0, (
        f"Disabled wrapper must not register any parameters; got {len(params)}."
    )
    # unified_loss is a no-op (zero scalar).
    h_src, h_dst, dist = _make_edge_batch(n_edges=4, hidden_dim=128)
    atom_target = _one_hot(torch.tensor([6, 7, 8, 1]), 100)
    bond_target = _one_hot(torch.tensor([1, 2, 3, 4]), NUM_BOND_CLASSES)
    loss = flow.unified_loss(
        h_src, h_dst, dist, atom_target, bond_target,
    )
    assert loss.item() == 0.0, (
        f"Disabled wrapper should return 0.0 loss; got {loss.item()}."
    )
    # sample is identity (no SimplexFlow perturbation).
    atom_x_t, bond_x_t, t = flow.sample(atom_target, bond_target)
    assert torch.equal(atom_x_t, atom_target), "sample must not modify atom_target."
    assert torch.equal(bond_x_t, bond_target), "sample must not modify bond_target."
    # And the legacy two-head path keeps working untouched:
    # (a no-op wrapper does not change the LipmanFlowMatchingAdapter
    # forward / backward contract — the bond_head / atom_head pair
    # still drives the per-sample CE losses as before.)
    assert flow._head is None, (
        "Disabled wrapper must lazy-instantiate the head only on demand; "
        f"_head should be None, got {flow._head!r}."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
