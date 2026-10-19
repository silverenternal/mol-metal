# SPDX-License-Identifier: Apache-2.0
"""Wf-A98 (2026-10-19) -- the spectral connectivity surrogate.

WHY: MEASURED on 64 real tmQM val clouds, stay_mean separates a decoded
fragmented graph (0.239) from a decoded connected one (0.123), but the MODEL
SOFT graph reads 0.145 -- below every truth graph (p50 0.175, max 0.212).
The soft relaxation densifies the graph, so the hinge cannot fire on the soft
statistic without also firing on correct complexes.  The size-normalised
algebraic connectivity separates by five orders of magnitude on the same
data: decoded fragmented 7.9e-05 vs decoded connected 27.5.

These tests pin the replacement: connected -> 0, fragmented -> large, and a
FINITE NEGATIVE gradient on the bridge probability (a term that cannot move
the logits is the FLAG_ONLY failure this project keeps re-finding).
"""
from __future__ import annotations

import pytest
import torch


def _m():
    import molmetal.molmetal_lam.lam_chem.assemble_connectivity as A

    return A


def _chain_adj(n: int, weight: float = 1.0) -> torch.Tensor:
    adj = torch.zeros(n, n)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = weight
    return adj


def _fragments_adj(n: int, k: int) -> torch.Tensor:
    adj = torch.zeros(n, n)
    size = n // k
    for f in range(k):
        lo, hi = f * size, (f + 1) * size
        for i in range(lo, hi - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def test_connected_chain_scores_zero():
    A = _m()
    loss = A._spectral_component_penalty(_chain_adj(30))
    assert float(loss) < 1e-2


def test_fragments_score_high():
    A = _m()
    two = float(A._spectral_component_penalty(_fragments_adj(30, 2)))
    three = float(A._spectral_component_penalty(_fragments_adj(30, 3)))
    assert two > 0.2
    assert three >= two


def test_isolated_atoms_are_the_worst_case():
    A = _m()
    loss = float(A._spectral_component_penalty(torch.zeros(30, 30)))
    assert loss > 0.5


def test_bridge_gradient_is_finite_and_negative():
    """Raising a bridge probability must LOWER the penalty."""
    A = _m()
    adj = torch.zeros(30, 30)
    for i in list(range(0, 14)) + list(range(15, 29)):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    adj[14, 15] = adj[15, 14] = 0.30
    adj.requires_grad_(True)
    loss = A._spectral_component_penalty(adj)
    loss.backward()
    g = adj.grad[14, 15]
    assert torch.isfinite(g)
    assert float(g) < 0.0


def test_fragment_loss_dispatch_is_finite():
    A = _m()
    # 6 heavy atoms, two disconnected triangles, p_bond ~ 0.88 inside each.
    i_idx = torch.tensor([0, 1, 3, 4])
    j_idx = torch.tensor([1, 2, 4, 5])
    logits = torch.full((4, 5), -20.0)
    logits[:, 1] = 0.0
    logits[:, 0] = -2.0
    z = torch.full((6,), 6, dtype=torch.long)
    loss, n_frag, largest = A.assemble_fragment_loss(
        logits, (i_idx, j_idx), z, formulation="spectral"
    )
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.5, abs=0.02)  # 1 - 1/2
    assert n_frag == 2.0
    assert 0.0 < largest < 1.0


def test_saturated_graph_has_no_gradient_and_that_is_documented():
    """A hard-fragmented graph sits at a plateau: c is locally constant.

    This is a REAL limitation of the statistic, pinned here so nobody
    "fixes" the test by loosening it: the heat-kernel trace is an integer
    in the saturated regime, so its gradient is exactly zero there.  The
    term can only move logits that sit in the transition band around the
    decision boundary -- see the next test.
    """
    A = _m()
    adj = torch.zeros(30, 30)
    for i in range(0, 14):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    for i in range(15, 29):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    adj.requires_grad_(True)
    A._spectral_component_penalty(adj).backward()
    # 4e-08 in float32: numerically zero, not exactly zero.
    assert float(adj.grad.abs().sum()) < 1e-5


def test_gradient_is_live_near_the_decision_boundary():
    """The band where the term can act: a bridge just under 0.5."""
    A = _m()
    adj = torch.zeros(30, 30)
    for i in range(0, 14):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    for i in range(15, 29):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    adj[14, 15] = adj[15, 14] = 0.40
    adj.requires_grad_(True)
    loss = A._spectral_component_penalty(adj)
    loss.backward()
    assert float(loss.detach()) > 1e-4
    assert torch.isfinite(adj.grad).all()
    assert float(adj.grad[14, 15]) < 0.0


def test_unknown_formulation_still_raises():
    A = _m()
    i_idx = torch.tensor([0])
    j_idx = torch.tensor([1])
    logits = torch.zeros(1, 5)
    z = torch.full((2,), 6, dtype=torch.long)
    with pytest.raises(ValueError, match="formulation must be"):
        A.assemble_fragment_loss(logits, (i_idx, j_idx), z, formulation="nope")
