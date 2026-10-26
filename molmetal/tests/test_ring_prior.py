# SPDX-License-Identifier: Apache-2.0
"""Wf-A111 (MOD4) — the ring-size prior refuses triangles and nothing else.

The defect (cfm_todo D3): 66.8% of the c7 pool's rings are three-membered and
2.9% six-membered, against 0.01% and 87.5% in real tmQM.  A three-membered
ring is what a greedy selector produces when three atoms happen to sit within
bonding distance, so the prior tests the one thing the selector can know
exactly: the shortest path between a candidate pair's endpoints in the bonds
already accepted, plus one, is the only ring that pair can close.

These tests pin:

* a triangle closure is refused, a four-ring closure is not;
* a closure that would leave an atom with no bonds is ALLOWED, because trading
  a wrong ring for an unparseable molecule is not a fix;
* refusal never disconnects anything — the path that made the ring possible
  is still there;
* on a synthetic three-atom clump the prior removes the triangle and the
  molecule stays one connected component.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from molmetal.eval.ring_profile import ring_sizes_from_edges  # noqa: E402
from molmetal.molmetal_lam.lam_chem.ring_prior import (  # noqa: E402
    RingSizePrior,
    adjacency_from_bonds,
)


def _adj(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


# ---------------------------------------------------------------------------
# the primitive
# ---------------------------------------------------------------------------


def test_shortest_path_length_counts_edges():
    adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
    assert RingSizePrior.shortest_path_length(adj, 0, 0) == 0
    assert RingSizePrior.shortest_path_length(adj, 0, 1) == 1
    assert RingSizePrior.shortest_path_length(adj, 0, 3) == 3
    assert RingSizePrior.shortest_path_length(adj, 0, 9) == -1


def test_a_pair_with_no_path_closes_no_ring():
    prior = RingSizePrior()
    adj = _adj(4, [(0, 1)])
    accept, ring = prior.should_accept(adj, 2, 3)
    assert accept and ring == 0


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------


def test_closing_a_triangle_is_refused_when_both_ends_are_already_bonded():
    """0-1-2 is a path; committing 0-2 would make a three-membered ring."""
    prior = RingSizePrior(min_ring_size=4)
    adj = _adj(3, [(0, 1), (1, 2)])
    accept, ring = prior.should_accept(adj, 0, 2)
    assert ring == 3
    assert accept is False
    assert prior.stats.n_refused_too_small == 1
    assert prior.stats.refused_ring_sizes == {3: 1}


def test_closing_a_four_membered_ring_is_allowed():
    prior = RingSizePrior(min_ring_size=4)
    adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
    accept, ring = prior.should_accept(adj, 0, 3)
    assert ring == 4
    assert accept is True


def test_an_atom_with_no_bonds_is_never_refused():
    """It has no path to anything, so it closes no ring — by construction.

    This is why there is no "protect isolated atoms" flag: the protection is a
    property of the test, not a knob.  A flag that can never fire is the
    inert-knob defect this project keeps finding.
    """
    prior = RingSizePrior(min_ring_size=4)
    adj = _adj(3, [(0, 1)])
    accept, ring = prior.should_accept(adj, 0, 2)
    assert ring == 0 and accept
    accept2, ring2 = prior.should_accept(adj, 1, 2)
    assert ring2 == 0 and accept2
    assert prior.stats.n_refused_too_small == 0


def test_max_ring_size_refuses_a_large_closure():
    prior = RingSizePrior(min_ring_size=4, max_ring_size=5)
    adj = _adj(8, [(i, i + 1) for i in range(7)])
    accept, ring = prior.should_accept(adj, 0, 7)
    assert ring == 8
    assert accept is False
    assert prior.stats.n_refused_too_large == 1


def test_min_ring_size_below_three_is_rejected():
    try:
        RingSizePrior(min_ring_size=2)
    except ValueError:
        return
    raise AssertionError("min_ring_size=2 should be refused: no such ring")


# ---------------------------------------------------------------------------
# the pathology, end to end
# ---------------------------------------------------------------------------


def _greedy(n_atoms, candidates, prior=None):
    """The decoder's accept loop, reduced to what the prior touches.

    ``candidates`` are ``(score, i, j)`` in descending score order, exactly the
    order ``_select_bonds`` walks.  Every pair is allowed by valence so the
    only thing under test is the ring gate.
    """
    adj = [set() for _ in range(n_atoms)]
    kept = []
    for _score, i, j in candidates:
        if prior is not None:
            ok, _ring = prior.should_accept(adj, i, j)
            if not ok:
                continue
        adj[i].add(j)
        adj[j].add(i)
        kept.append((i, j))
    return kept, adj


def test_a_clump_of_three_atoms_becomes_a_triangle_without_the_prior():
    """The baseline: this is what the decoder does today."""
    n = 4
    # 0,1,2 mutually close (a clump); 2-3 is a real chain bond.
    cands = [(0.9, 0, 1), (0.8, 1, 2), (0.7, 0, 2), (0.6, 2, 3)]
    kept, _ = _greedy(n, cands)
    assert (0, 2) in kept
    assert ring_sizes_from_edges(n, kept) == [3]


def test_the_prior_removes_the_triangle_and_keeps_the_molecule_connected():
    n = 4
    cands = [(0.9, 0, 1), (0.8, 1, 2), (0.7, 0, 2), (0.6, 2, 3)]
    prior = RingSizePrior(min_ring_size=4)
    kept, adj = _greedy(n, cands, prior=prior)
    assert (0, 2) not in kept
    assert ring_sizes_from_edges(n, kept) == []
    # Connectivity is untouched: every atom still reaches every other.
    seen = {0}
    stack = [0]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert seen == set(range(n))


def test_the_prior_leaves_a_six_membered_ring_alone():
    n = 6
    cands = [(1.0 - 0.01 * k, k, (k + 1) % 6) for k in range(6)]
    prior = RingSizePrior(min_ring_size=4)
    kept, _ = _greedy(n, cands, prior=prior)
    assert len(kept) == 6
    assert ring_sizes_from_edges(n, kept) == [6]
    assert prior.stats.n_refused_too_small == 0


def test_adjacency_from_bonds_ignores_self_loops_and_out_of_range():
    adj = adjacency_from_bonds(3, [(0, 0, 1), (0, 1, 1), (1, 5, 1)])
    assert adj[0] == {1} and adj[1] == {0}
    assert adj[2] == set()
