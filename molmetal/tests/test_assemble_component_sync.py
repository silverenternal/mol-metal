# SPDX-License-Identifier: Apache-2.0
"""The component counter must return the SAME partition after the sync fix.

WHAT CHANGED (2026-10-10)
-------------------------
`_count_components` propagated labels in a loop whose every round rebuilt two
loop-invariants (`adj > 0`, an (N, N) `full_like` fill) and ended with
`bool(torch.equal(new, labels))` — a device comparison read on the host, so a
full CUDA queue drain, once per round, up to N+1 times. It runs once per cloud
per optimizer step, so that drain was the dominant cost of the call.

The fix hoists the invariants and tests convergence only every
`_COMPONENT_CHECK_EVERY` rounds.

WHY EQUIVALENCE IS THE WHOLE CLAIM
----------------------------------
Nothing else about the function changed, so "faster" is only worth anything if
the partition is bit-identical. The argument is that the update is idempotent
at its fixed point: if `new == labels` then the following round recomputes
from an unchanged `labels` and returns the same value, so every round past
convergence is a no-op. Skipping intermediate tests therefore cannot skip a
real update — it only runs no-ops that would not have moved anything.

These tests check that argument empirically rather than trusting it: the
amortised counter is compared against a reference implementation that checks
every round, over graphs built to converge at very different depths.
"""

from __future__ import annotations

import torch

from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (
    _COMPONENT_CHECK_EVERY,
    _count_components,
)


def _reference(adj: torch.Tensor) -> tuple[float, float]:
    """The pre-fix loop, verbatim: host-side convergence test EVERY round."""
    n = int(adj.shape[0])
    if n == 0:
        return 0.0, 0.0
    labels = torch.arange(n, device=adj.device, dtype=torch.long)
    for _ in range(n + 1):
        nbr = torch.where(
            adj > 0, labels.unsqueeze(0), torch.full_like(labels.unsqueeze(0), n)
        )
        new = torch.minimum(labels, nbr.min(dim=1).values)
        if bool(torch.equal(new, labels)):
            break
        labels = new
    uniq, counts = torch.unique(labels, return_counts=True)
    largest = float(counts.max().item()) / float(n)
    return float(uniq.numel()), largest


def _from_edges(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    a = torch.zeros((n, n), dtype=torch.long)
    for i, j in edges:
        a[i, j] = 1
        a[j, i] = 1
    return a


def test_matches_reference_on_a_long_path():
    """A path graph propagates ONE label per round, so it converges at the
    maximum depth the amortised check could overshoot — the worst case."""
    n = 40
    adj = _from_edges(n, [(i, i + 1) for i in range(n - 1)])
    assert _count_components(adj) == _reference(adj)
    assert _count_components(adj)[0] == 1.0


def test_matches_reference_on_many_isolated_atoms():
    """No edges at all: converges on round 1, and exercises the `n` sentinel
    the hoisted `absent` row carries."""
    adj = _from_edges(12, [])
    assert _count_components(adj) == _reference(adj)
    assert _count_components(adj) == (12.0, 1.0 / 12.0)


def test_matches_reference_across_random_graphs():
    """Random graphs give a spread of propagation depths, including depths
    that straddle the check interval — the case where an off-by-one in the
    amortisation would show up as a different component count."""
    g = torch.Generator().manual_seed(0)
    for trial in range(40):
        n = int(torch.randint(3, 45, (1,), generator=g))
        p = float(torch.rand(1, generator=g)) * 0.3
        adj = (torch.rand((n, n), generator=g) < p).to(torch.long)
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.T
        assert _count_components(adj) == _reference(adj), f"trial {trial}"


def test_matches_reference_on_a_long_path_longer_than_the_check_interval():
    """The overshoot argument is only exercised if the chain is LONGER than
    _COMPONENT_CHECK_EVERY: the counter will run extra rounds past the round
    the reference would have exited on, and must still land identically."""
    n = _COMPONENT_CHECK_EVERY * 5
    adj = _from_edges(n, [(i, i + 1) for i in range(n - 1)])
    assert _count_components(adj) == _reference(adj)
    assert _count_components(adj)[0] == 1.0


def test_empty_graph_still_reports_zero():
    assert _count_components(torch.zeros((0, 0), dtype=torch.long)) == (0.0, 0.0)


def test_two_components_are_counted_and_the_largest_is_reported():
    # {0,1,2} {3} {4} {5..9} — four components, largest 5 of 10. The two
    # isolated atoms are the reason this is not the two-component case the
    # name suggests; they are here on purpose, since a stranded atom is the
    # case the assemble term exists to detect.
    adj = _from_edges(10, [(0, 1), (1, 2), (5, 6), (6, 7), (7, 8), (8, 9)])
    got = _count_components(adj)
    assert got == (4.0, 0.5)
    assert got == _reference(adj)