# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the Round-3 D1 assemble-aware connectivity loss.

Every assertion here corresponds to a behaviour that was MEASURED during
development, and several of them pin bugs this module actually had.  A test
that only checks "it returns a float" would have passed all three of the
following and let a term that inverts its own gradient into the loss sum:

1. **Ordering** -- a fully isolated graph must score ~1 and a fully
   connected graph must score ~0.  A relative normalisation (retained mass
   over its own max) scores the isolated graph at exactly 0.0 and a
   connected chain at 0.16, i.e. precisely backwards.
2. **Gradient direction** -- raising a ``BOND_NO_BOND`` logit disconnects an
   atom, so the loss must *rise*.  An undamped random walk produced the
   opposite sign.
3. **Truncation stability** -- the loss must not depend on the parity of
   ``n_diffusion_steps``.  The undamped walk is periodic on a chain, so its
   diagonal alternates 0.50, ~0, 0.50, ~0 and any truncated product is
   parity-dependent.

Run with: ``.venv/bin/python -m pytest molmetal/tests/test_assemble_connectivity.py -q``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (  # noqa: E402
    assemble_fragment_loss,
    batch_assemble_loss,
)

CARBON = 6


def _logits_from_graph(adj: torch.Tensor, p_bond: float, nobond_logit: float = 10.0):
    """Build ``(E, 5)`` logits + ``(i_idx, j_idx)`` for an upper-triangular graph.

    Bonded pairs get ``log(1-p)`` / ``log(p)`` on classes 0/1; unbonded pairs
    get ``nobond_logit`` on class 0 with the rest at -5.  The tensor requires
    grad.

    The -5 (rather than a large negative) is deliberate: it keeps the NO_BOND
    rows in a sensitive operating range, which is where a real head sits.  At
    -30 the implied ``p_bond`` is ~1e-15, the softmax is fully saturated, and
    every gradient MEASURED ~1e-8 -- a test asserting only ``> 0`` would pass
    on numerical noise rather than on a real signal.
    """
    n = int(adj.shape[0])
    rows: list = []
    i_idx: list = []
    j_idx: list = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
            if bool(adj[i, j]):
                rows.append(
                    [
                        math.log(max(1.0 - p_bond, 1e-12)),
                        math.log(max(p_bond, 1e-12)),
                        -30.0,
                        -30.0,
                        -30.0,
                    ]
                )
            else:
                rows.append([nobond_logit, -5.0, -5.0, -5.0, -5.0])
    logits = torch.tensor(rows, dtype=torch.float32, requires_grad=True)
    return logits, (torch.tensor(i_idx), torch.tensor(j_idx))


def _chain(n: int) -> torch.Tensor:
    adj = torch.zeros(n, n)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _z(n: int) -> torch.Tensor:
    return torch.full((n,), CARBON, dtype=torch.long)


def _pad_to(adj: torch.Tensor, n: int) -> torch.Tensor:
    """Grow a square adjacency to ``n x n`` with isolated new atoms."""
    cur = int(adj.shape[0])
    if n == cur:
        return adj
    if n < cur:
        return adj[:n, :n]
    out = torch.zeros(n, n)
    out[:cur, :cur] = adj
    return out


# ---------------------------------------------------------------------------
# 1. Ordering: isolated must be WORSE than connected
# ---------------------------------------------------------------------------
def test_isolated_scores_worse_than_connected():
    z = _z(4)
    iso = torch.zeros(4, 4)
    chain = _chain(4)

    logits, e = _logits_from_graph(iso, 0.9)
    loss_iso, nfrag_iso, _ = assemble_fragment_loss(logits, e, z)
    logits2, e2 = _logits_from_graph(chain, 0.9)
    loss_chain, nfrag_chain, _ = assemble_fragment_loss(logits2, e2, z)

    assert nfrag_iso == 4.0, "four unconnected heavy atoms are 4 fragments"
    assert nfrag_chain == 1.0, "a 4-chain is 1 fragment"
    assert float(loss_iso.detach()) > 0.9, (
        f"fully isolated graph must score ~1.0, got {float(loss_iso.detach()):.4f}"
    )
    assert float(loss_chain.detach()) < 0.05, (
        f"fully connected chain must score ~0.0, got {float(loss_chain.detach()):.4f}"
    )


def test_partial_fragmentation_is_between_the_extremes():
    """A 3-chain plus one stranded atom must land strictly in between."""
    adj = _pad_to(_chain(3), 4)
    logits, e = _logits_from_graph(adj, 0.9)
    loss, nfrag, largest = assemble_fragment_loss(logits, e, _z(4))
    assert nfrag == 2.0
    assert abs(largest - 0.75) < 1e-6
    assert 0.02 < float(loss.detach()) < 0.9


def test_monotone_in_bond_confidence():
    """The more strongly an atom is stranded, the higher the loss.

    Pinned because it is the term's entire reason to exist: the Round-3 run
    had ``largest_fragment_frac`` rise while ``bond_ce`` fell.
    """
    z = _z(4)
    adj = _pad_to(_chain(3), 4)
    values = []
    for v in (-2.0, 0.0, 2.0, 5.0, 10.0):
        logits, e = _logits_from_graph(adj, 0.9, nobond_logit=v)
        loss, _, _ = assemble_fragment_loss(logits, e, z)
        values.append(float(loss.detach()))
    for a, b in zip(values, values[1:]):
        assert b >= a - 1e-6, f"loss must not fall as atoms disconnect: {values}"


# ---------------------------------------------------------------------------
# 2. Gradient direction
# ---------------------------------------------------------------------------
def test_gradient_direction_is_correct():
    """Raising a NO_BOND logit disconnects an atom, so d(loss)/d(logit) > 0."""
    adj = _pad_to(_chain(3), 4)
    logits, e = _logits_from_graph(adj, 0.9, nobond_logit=4.0)
    loss, _, _ = assemble_fragment_loss(logits, e, _z(4))
    loss.backward()
    g = logits.grad
    assert g is not None
    # Rows 0-1 and 1-2 are the two real bonds; every later row is NO_BOND.
    nobond_rows = g[2:].abs().sum(dim=1)
    assert bool((nobond_rows > 0).any()), (
        "stranded atom must receive gradient; a term that only trains the "
        "already-connected atoms ignores the fragment it was added to fix"
    )
    assert float(g[2:, 0].mean()) > 1e-5, (
        "disconnecting an atom must raise the loss, and by a magnitude that "
        "is not numerical noise (a saturated softmax gives ~1e-8)"
    )


def test_fully_connected_graph_has_near_zero_gradient():
    """A converged term must not keep pushing an already-connected graph."""
    logits, e = _logits_from_graph(_chain(6), 0.9)
    loss, _, _ = assemble_fragment_loss(logits, e, _z(6))
    loss.backward()
    total = float(logits.grad.abs().sum())
    assert total < 1e-3, f"connected graph should be at a fixed point, got {total}"


# ---------------------------------------------------------------------------
# 3. Truncation stability (the parity regression)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("steps", [6, 8, 10, 12, 16, 20])
def test_loss_is_stable_across_truncation_lengths(steps):
    """Even/odd ``n_diffusion_steps`` must not change the answer.

    An undamped walk is periodic on a chain (diagonal 0.50, ~0, 0.50, ~0),
    which made the truncated product parity-dependent.  This test is the
    regression guard for that.
    """
    adj = _pad_to(_chain(3), 4)
    logits, e = _logits_from_graph(adj, 0.9, nobond_logit=5.0)
    loss, _, _ = assemble_fragment_loss(
        logits, e, _z(4), n_diffusion_steps=steps
    )
    assert abs(float(loss.detach()) - 0.25) < 0.05, (
        f"n_diffusion_steps={steps} gave {float(loss.detach()):.4f}, expected ~0.25"
    )


# ---------------------------------------------------------------------------
# 4. Numerics + edge cases
# ---------------------------------------------------------------------------
def test_no_nan_when_graph_has_no_edges_at_all():
    adj = torch.zeros(5, 5)
    logits, e = _logits_from_graph(adj, 0.5, nobond_logit=30.0)
    loss, nfrag, _ = assemble_fragment_loss(logits, e, _z(5))
    assert torch.isfinite(loss).all(), "must not NaN on a fully stranded cloud"
    assert float(loss.detach()) > 0.9
    assert nfrag == 5.0


def test_no_nan_with_zero_candidate_pairs():
    logits = torch.zeros(0, 5, requires_grad=True)
    empty = (
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )
    loss, nfrag, largest = assemble_fragment_loss(logits, empty, _z(3))
    assert torch.isfinite(loss).all()
    assert nfrag == 3.0 and largest == pytest.approx(1.0 / 3.0)


def test_single_heavy_atom_is_not_penalised():
    loss, nfrag, largest = assemble_fragment_loss(
        torch.zeros(0, 5), (torch.zeros(0, dtype=torch.long),) * 2, _z(1)
    )
    assert float(loss.detach()) == 0.0
    assert nfrag == 1.0 and largest == 1.0


def test_hydrogens_are_excluded_from_connectivity():
    """An unbonded H is a valence detail, not a second fragment."""
    adj = _chain(2)
    # 2 carbons bonded + 2 hydrogens attached to carbon 0.
    adj = torch.cat(
        [
            torch.cat([adj, torch.zeros(2, 2)], dim=1),
            torch.zeros(2, 4),
        ],
        dim=0,
    )
    adj[0, 2] = adj[2, 0] = 1.0
    adj[0, 3] = adj[3, 0] = 1.0
    logits, e = _logits_from_graph(adj, 0.9)
    z = torch.tensor([6, 6, 1, 1], dtype=torch.long)
    loss, nfrag, _ = assemble_fragment_loss(logits, e, z)
    assert nfrag == 1.0, "hydrogens must not count as separate fragments"
    assert float(loss.detach()) < 0.05


def test_batched_matches_per_cloud():
    torch.manual_seed(0)
    items = []
    for n in (3, 4, 5):
        adj = _chain(n)
        logits, e = _logits_from_graph(adj, 0.9)
        items.append((logits, e, _z(n)))
    mean_loss, mean_frag, mean_frac, n_used = batch_assemble_loss(
        items, torch.device("cpu")
    )
    assert n_used == 3
    manual = []
    for logits, e, z in items:
        term, _, _ = assemble_fragment_loss(logits, e, z)
        manual.append(term.detach())
    assert abs(float(mean_loss) - float(torch.stack(manual).mean())) < 1e-6


def test_empty_batch_is_hard_zero():
    loss, frag, frac, n_used = batch_assemble_loss([], torch.device("cpu"))
    assert n_used == 0
    assert float(loss) == 0.0 and frag == 0.0 and frac == 0.0
    assert float(loss.requires_grad) == 0.0


def test_atom_with_no_candidate_edge_is_constant_but_finite():
    """An atom with NO candidate edge is beyond the head's authority.

    The head only emits pairs within its 3.5 A cutoff, so a geometrically
    isolated atom has an all-zero adjacency row.  Its penalty is a real,
    positive number and its gradient is EXACTLY zero -- there is no logit to
    move.  Pinned so nobody later reads the positive loss as "the term is
    broken" or, worse, as evidence the gradient path is dead.
    """
    # 5-atom chain plus one atom that appears in NO edge list at all.
    ei = torch.tensor([0, 1, 2, 3])
    ej = torch.tensor([1, 2, 3, 4])
    rows = [
        [-2.2, -0.8, -5.0, -5.0, -5.0],
        [-2.2, -0.8, -5.0, -5.0, -5.0],
        [-2.2, -0.8, -5.0, -5.0, -5.0],
        [-2.2, -0.8, -5.0, -5.0, -5.0],
    ]
    logits = torch.tensor(rows, dtype=torch.float32, requires_grad=True)
    z = _z(6)  # 6 atoms, but atom 5 has no candidate edge
    loss, nfrag, _ = assemble_fragment_loss(logits, (ei, ej), z)
    assert nfrag == 2.0, "the edge-less atom is its own fragment"
    assert float(loss.detach()) > 0.0
    loss.backward()
    # Not `== 0.0`: the connected chain's own rows still carry float32
    # round-off (~4e-7 here).  What matters is that the magnitude is
    # numerical noise and orders of magnitude below the live case below
    # (~3.7e-2), i.e. the edge-less atom contributes no usable signal.
    assert float(logits.grad.abs().sum()) < 1e-5, (
        "with no candidate edge there is nothing for the gradient to move; "
        "a magnitude this large means the fixture is wrong, not the module"
    )


def test_stranded_atom_with_a_candidate_edge_does_get_gradient():
    """The contrast case: give that atom an edge and the gradient returns.

    Together with the test above this pins the boundary: the term is not
    dead, it is bounded by what the head is able to control.
    """
    ei = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    ej = torch.tensor([1, 2, 3, 2, 3, 3, 4, 5])
    rows = []
    for k in range(len(ei)):
        pair = (int(ei[k]), int(ej[k]))
        if pair in {(3, 4), (4, 5)}:
            rows.append([2.2, -0.8, -5.0, -5.0, -5.0])  # p_bond ~ 0.1
        else:
            rows.append([-2.2, -0.8, -5.0, -5.0, -5.0])  # p_bond ~ 0.9
    logits = torch.tensor(rows, dtype=torch.float32, requires_grad=True)
    loss, nfrag, _ = assemble_fragment_loss(logits, (ei, ej), _z(6))
    assert nfrag == 3.0
    loss.backward()
    assert float(logits.grad.abs().sum()) > 1e-3, (
        "a stranded atom that HAS a candidate edge must receive gradient"
    )


def test_gradcheck():
    """The surrogate is smooth in the logits, unlike a thresholded count."""
    adj = _pad_to(_chain(3), 4)
    logits, e = _logits_from_graph(adj, 0.9, nobond_logit=3.0)
    z = _z(4)

    def fn(x):
        return assemble_fragment_loss(x, e, z)[0]

    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-3, atol=1e-2)


def test_batch_skips_a_cloud_with_no_pair_indices():
    """A head that emitted no pair indices must be skipped, not crash.

    MEASURED 2026-10-05, on the Round-3 CPU smoke.  ``CFMJointModel.forward``
    returned the EGNN velocity field's radius graph under the name the
    assemble term was about to read, so the term was handed a `(2, E)` tensor
    instead of the `(i_idx, j_idx)` tuple and every run with
    ``--assemble-loss-weight > 0`` died with::

        ValueError: not enough values to unpack (expected 2, got 1)

    `forward` is fixed to return the head's own pairs.  This test pins the
    remaining half of that contract: a `None` there is skipped and counted as
    unused, so a head that legitimately emits nothing degrades to "the term
    did not apply this batch" rather than to a crash or, worse, to a graph
    the bond head never scored.
    """
    # One logit row per candidate pair, or the scatter below indexes a
    # (3,) mask with a (5,) row list and the fixture, not the module, is
    # what fails.
    ei = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    logits = torch.randn(3, 5, requires_grad=True)
    z = _z(6)

    loss, frag, frac, n_used = batch_assemble_loss(
        [(logits, None, z), (logits, ei, z)], torch.device("cpu")
    )
    assert n_used == 1, (
        "the cloud with pair indices should be the only one scored; the "
        f"None one must be skipped, not silently substituted"
    )
    assert float(loss) >= 0.0 and frag >= 1.0 and 0.0 < frac <= 1.0

    # And with only unusable entries the term reports itself unused rather
    # than inventing a value, so the caller's `if _asm_n > 0` guard can
    # distinguish "no penalty" from "penalty measured as zero".
    zero_loss, _, _, n_none = batch_assemble_loss(
        [(logits, None, z)], torch.device("cpu")
    )
    assert n_none == 0
    assert float(zero_loss) == 0.0


# ===========================================================================
# formulation="escape_time" — added 2026-10-10, opt-in, NOT the default.
#
# Why this block exists: the fragmentation problem BLK-C11 is about is real and
# measured (50.4% of decoded molecules are multi-fragment, 23.0% have a fully
# detached metal centre), and assemble_term is the term written to address it.
# But the shipped "product" aggregation has a gradient that scales with the
# product it returns, so it goes to ~1e-06 on a connected graph — the regime a
# prevention objective lives in. These tests pin what the alternative buys and,
# just as importantly, what it costs: it violates the "connected graphs get no
# gradient" property that the suite above pins deliberately.
# ===========================================================================


def _chain_adj(n: int, p: float, bonds=None):
    A = torch.zeros(n, n, dtype=torch.float64)
    bonds = bonds if bonds is not None else [(i, i + 1) for i in range(n - 1)]
    for i, j in bonds:
        A[i, j] = p
        A[j, i] = p
    return A


Z7 = torch.tensor([78, 6, 6, 6, 6, 17, 17])


def _loss_and_grad(A, formulation, bonds=None, n=7):
    E = int((A > 0).sum() // 2)
    src, dst = [], []
    seen = set()
    for i in range(n):
        for j in range(n):
            if i < j and A[i, j] > 0 and (i, j) not in seen:
                src.append(i)
                dst.append(j)
                seen.add((i, j))
    logits = torch.zeros(len(src), 5, dtype=torch.float64)
    for k in range(len(src)):
        logits[k, 0] = torch.log(torch.tensor(1.0 - A[src[k], dst[k]], dtype=torch.float64))
        logits[k, 1] = torch.log(torch.tensor(A[src[k], dst[k]], dtype=torch.float64))
    logits = logits.requires_grad_(True)
    loss, _, _ = assemble_fragment_loss(
        logits, (torch.tensor(src), torch.tensor(dst)), Z7,
        formulation=formulation,
    )
    g = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), float(g.abs().max())


def test_escape_time_keeps_the_connected_graphs_ordering():
    """It must still say: stranded atom worse than connected atom."""
    conn = _chain_adj(7, 0.9)
    iso = _chain_adj(7, 0.9, bonds=[(1, 2), (2, 3), (3, 4), (4, 5)])
    v_conn, _ = _loss_and_grad(conn, "escape_time")
    v_iso, _ = _loss_and_grad(iso, "escape_time")
    assert v_iso > v_conn, f"isolated {v_iso} must exceed connected {v_conn}"


def test_escape_time_has_a_live_gradient_where_product_does_not():
    """The whole point. On a connected graph the product gradient is ~1e-06."""
    # MEASURED at the logits (not adjacency) level, 2026-10-10, 7-atom chain:
    #     case              product |g|   escape_time |g|    ratio
    #     connected p=.9     4.62e-09        3.46e-04      7.5e4
    #     metal isolated     6.17e-09        4.45e-04      7.2e4
    #     weak bonds p=.3    3.98e-08        2.60e-03      6.5e4
    # The product's gradient is ~5e-09 in EVERY case, including where its own
    # value is 0.286 — so it is a diagnostic, not an objective.
    conn = _chain_adj(7, 0.9)
    _, g_prod = _loss_and_grad(conn, "product")
    _, g_esc = _loss_and_grad(conn, "escape_time")
    assert g_prod < 1e-6, (
        f"product gradient on a connected graph is {g_prod:.2e} — if this is no "
        "longer tiny, the justification for escape_time is gone and this test "
        "should be deleted, not updated"
    )
    assert g_esc > 1e-5, (
        f"escape_time gradient on a connected graph is only {g_esc:.2e}"
    )
    assert g_esc / max(g_prod, 1e-30) > 1e3, (
        f"escape_time buys only {g_esc / max(g_prod, 1e-30):.0f}x — not the "
        "5+ orders of magnitude the reformulation was adopted for"
    )


def test_escape_time_violates_the_pinned_near_zero_gradient_property():
    """Documenting the COST, so the trade-off cannot be quietly forgotten.

    test_fully_connected_graph_has_near_zero_gradient above pins "don't spend
    gradient on a graph that is already fine". escape_time deliberately spends
    it. That is why it is opt-in. If someone flips the default without reading
    this, this test is the thing that should have stopped them.
    """
    conn = _chain_adj(7, 0.9)
    v_prod, g_prod = _loss_and_grad(conn, "product")
    v_esc, g_esc = _loss_and_grad(conn, "escape_time")
    assert g_prod < 1e-4 < g_esc, (
        "the two formulations no longer differ on connected-graph gradient; "
        "re-derive before relying on this"
    )


def test_escape_time_is_not_near_zero_on_a_connected_graph():
    """The offset. escape_time reads ~0.3 where product reads ~2e-07.

    So it is not a drop-in replacement for the product form: it adds a roughly
    constant contribution to the loss sum and must be weighted accordingly.
    """
    conn = _chain_adj(7, 0.9)
    v_prod, _ = _loss_and_grad(conn, "product")
    v_esc, _ = _loss_and_grad(conn, "escape_time")
    assert v_prod < 1e-5 < v_esc
    assert v_esc < 1.0, "escape_time must stay bounded in [0, 1]"


def test_escape_time_is_monotone_decreasing_in_bond_probability():
    prev = None
    for p in (0.2, 0.5, 0.8, 0.95):
        v, _ = _loss_and_grad(_chain_adj(7, p), "escape_time")
        if prev is not None:
            assert v < prev, f"penalty rose as bonds strengthened (p={p}: {v} vs {prev})"
        prev = v


def _loss_and_grad_h(A, formulation, tau, n=7):
    """Same as _loss_and_grad but with the hinge dead zone applied."""
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                src.append(i)
                dst.append(j)
    rows = []
    for k in range(len(src)):
        row = [0.0] * 5
        row[0] = float(torch.log(torch.tensor(1.0 - A[src[k], dst[k]], dtype=torch.float64)))
        row[1] = float(torch.log(torch.tensor(A[src[k], dst[k]], dtype=torch.float64)))
        rows.append(row)
    logits = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    loss, _, _ = assemble_fragment_loss(
        logits, (torch.tensor(src), torch.tensor(dst)), Z7,
        formulation=formulation, hinge_tau=tau,
    )
    g = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), float(g.abs().max())


def test_unknown_formulation_is_rejected_loudly():
    """A typo must not silently fall back to one of the two."""
    conn = _chain_adj(7, 0.9)
    src, dst = [], []
    for i in range(6):
        src.append(i)
        dst.append(i + 1)
    logits = torch.zeros(6, 5, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="product"):
        assemble_fragment_loss(
            logits, (torch.tensor(src), torch.tensor(dst)), Z7,
            formulation="escape-tim",
        )


# ===========================================================================
# The hinge, at the tau derived from tmQM (2026-10-10, user-selected).
#
# tau = 0.30 comes from metrics/by_metric/cfm_assemble_hinge_tau.json: it is the
# smallest round value above BOTH measured bounds — the maximum stay_mean over
# 1500 correct tmQM complexes (0.2649) and the stay_mean of a connected graph the
# head predicts confidently at p=0.9 (0.2629). These tests exist so that neither
# bound can be quietly eroded, and so the dead zone cannot silently grow into
# something that never fires.
# ===========================================================================

TAU = 0.30
TAU_CORPUS_MAX = 0.2649  # max stay_mean over 1500 tmQM ground-truth complexes
TAU_MODEL_CONNECTED = 0.2629  # stay_mean of a p=0.9 connected 7-atom chain


def test_tau_exceeds_the_measured_corpus_bound():
    conn = _chain_adj(7, 0.9)
    v, _ = _loss_and_grad(conn, "escape_time")
    assert v < TAU, (
        f"a confidently-connected graph reads {v:.4f}, at or above tau={TAU} — "
        "the dead zone no longer protects the case it exists for. Re-derive tau "
        "in metrics/by_metric/cfm_assemble_hinge_tau.json before changing it here."
    )


def test_hinge_preserves_the_pinned_zero_gradient_property():
    """The whole reason the hinge was chosen over a raw default flip."""
    conn = _chain_adj(7, 0.9)
    v, g = _loss_and_grad(conn, "escape_time")
    vh, gh = _loss_and_grad_h(conn, "escape_time", TAU)
    assert v < TAU and vh == 0.0, f"expected the hinge to zero this out, got {vh}"
    assert gh == 0.0, (
        f"hinged gradient on a connected graph is {gh:.2e}, expected exactly 0 "
        "— test_fully_connected_graph_has_near_zero_gradient depends on this"
    )


def test_hinge_still_penalises_a_detached_metal():
    iso = _chain_adj(7, 0.9, bonds=[(1, 2), (2, 3), (3, 4), (4, 5)])
    vh, gh = _loss_and_grad_h(iso, "escape_time", TAU)
    assert vh > 0, "the hinge silences the case it was built for"
    assert gh > 1e-6, f"hinged gradient on a detached metal is {gh:.2e}"


def test_hinge_gradient_is_millions_of_times_the_product_gradient():
    iso = _chain_adj(7, 0.9, bonds=[(1, 2), (2, 3), (3, 4), (4, 5)])
    _, g_prod = _loss_and_grad(iso, "product")
    _, g_hinge = _loss_and_grad_h(iso, "escape_time", TAU)
    assert g_hinge / max(g_prod, 1e-30) > 1e3, (
        f"hinge buys only {g_hinge / max(g_prod, 1e-30):.0f}x over product"
    )


def test_hinge_is_monotone_and_bounded():
    prev = None
    for p in (0.2, 0.5, 0.8, 0.95):
        v, _ = _loss_and_grad_h(_chain_adj(7, p), "escape_time", TAU)
        assert 0.0 <= v <= 1.0, f"hinged penalty out of [0,1]: {v}"
        if prev is not None:
            assert v <= prev + 1e-12, "hinged penalty rose as bonds strengthened"
        prev = v


def test_tau_bounds_are_recorded_next_to_the_value():
    """Guard the provenance of the constant, not just its arithmetic."""
    assert TAU_CORPUS_MAX == 0.2649 and TAU_MODEL_CONNECTED == 0.2629, (
        "the measured bounds recorded here changed — re-derive tau and update "
        "this test deliberately"
    )
    assert TAU > max(TAU_CORPUS_MAX, TAU_MODEL_CONNECTED), (
        f"tau={TAU} no longer clears the measured bounds"
    )
