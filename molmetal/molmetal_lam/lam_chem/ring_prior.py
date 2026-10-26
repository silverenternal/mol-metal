# SPDX-License-Identifier: Apache-2.0
"""Ring-SIZE prior at decode time (MOD4, addresses D3).

THE DEFECT
----------
MEASURED on the c7 pool (``metrics/by_metric/c7_fullgen_pool_quality.json``,
12,669 molecules / 79,945 rings): 66.8% of generated rings are three-membered
and 2.9% are six-membered.  The c8 pool is worse in the same direction, and a
32-sample pool from the T3 smoke gives 0.76-0.77.

The real-corpus reference is the same artifact's ``reference`` block -- 18,137
molecules of ``metalcytotox_5metal_multitask_v4.csv``, 113,558 rings -- where
the three-ring share is 0.01% and the six-ring share 87.5%.  OPT-2's separate
tmQM sample (``metrics/by_metric/cfm_opt2_ring_content.json``, 2,000
molecules, 8,386 rings) reads 0.49% and 80.4%.  The two references disagree by
a factor of 40 on a number that is near zero in both, and that is worth
knowing before either is quoted: the gap to generated output is three orders
of magnitude either way, so the defect does not depend on which one is used.

A three-membered ring is what a greedy selector produces when three atoms
happen to sit within bonding distance of each other and each pair clears the
bond threshold.  It is not a learned preference; nothing in the objective or
the decode rule has an opinion about ring size.

WHAT THIS DOES
--------------
The decoder's selector is already greedy and sequential: it sorts candidate
pairs by confidence and walks them, keeping each pair whose valence still
fits.  That order makes a ring test cheap and exact: before committing a pair
``(i, j)``, take the shortest path between ``i`` and ``j`` in the graph of
bonds ALREADY accepted.  If that path has ``d`` edges, committing the pair
closes a ring of ``d + 1`` members, and no other ring is created.

So the prior is one BFS per candidate, on a graph with at most a few hundred
edges, and it can refuse exactly the closures that produce a ring size the
data does not have.

WHY REFUSING IS SAFE HERE
-------------------------
The MOD4 risk note says a cycle prior can only remove edges, so if the
geometry already placed a triangle, refusal lowers connectivity.  Refusing an
edge never disconnects a graph: it declines to add a CYCLE, and the path that
made the ring possible is still there.  It can also never isolate an atom,
because an atom with no accepted bond has no path to the other endpoint and
therefore closes no ring in the first place.

WHAT IT MEASURABLY DOES (and which min_ring_size to use)
--------------------------------------------------------
MEASURED (``metrics/by_metric/wfa111_ring_prior_ab.json``, 3 pools x 256
clouds, one 20-epoch smoke checkpoint, prior-on vs prior-off on the same
generated geometry):

    arm              ring_3  ring_4  ring_5  ring_6   5+6 share   rings/mol
    prior off           325     369      74      91       0.148        1.45
    min_ring_size=4       0     418      95     100       0.217        1.17
    min_ring_size=5       0       0     142     169       0.476        0.85

``min_ring_size=4`` removes every triangle and the freed valence lands on
FOUR-membered rings (369 -> 418).  The OPT-2 tmQM reference has four-rings at
0.50% against 0.49% for three-rings, so that is the same pathology in a
different size.  ``min_ring_size=5`` routes the valence into five- and
six-rings instead, which is the direction the data has (5+6 is 96.9% of that
reference's rings).  So when this prior is enabled, 5 is the defensible value
and 4 is the minimal one.

The default stays 4: it is the minimal intervention, and a module default is
not the place to encode a result measured on one checkpoint.  A config that
enables the prior for generation should set ``ring_prior_min_ring_size: 5``
explicitly.

WHAT IT CANNOT DO
-----------------
It cannot put a six-membered ring where the geometry has none.  Raising
``six_ring_share`` requires the coordinate field to place atoms on a hexagon,
which is MOD2/MOD3's job and, before that, the sampler's.  A prior that
rejects triangles moves the three-ring share down; whether the freed valence
goes into six-rings or into nothing is an empirical question this module
cannot answer on its own.

A RING SHARE IS NOT A QUALITY METRIC
------------------------------------
MEASURED (1-epoch trainer smoke, 32 samples, a model whose geometry is still a
blob): with ``min_ring_size=5`` the 5+6 share reads 0.986 while
``single_fragment_ratio`` is 0.000 and the pool holds 6.47 rings per molecule.
The triangles became pentagons; the share reached the right number for the
wrong reason.  Any report of this prior has to carry ``rings_per_mol``,
``single_fragment_rate`` and ``decode_ratio`` beside the share, which is why
``wfa111_ring_prior_ab.py`` reports all of them and why the gate in
``cfm_todo.json`` T7 is not met by the share alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class RingPriorStats:
    """What the prior did, so the effect is a number and not an opinion."""

    n_considered: int = 0
    n_refused_too_small: int = 0
    n_refused_too_large: int = 0
    refused_ring_sizes: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_considered": int(self.n_considered),
            "n_refused_too_small": int(self.n_refused_too_small),
            "n_refused_too_large": int(self.n_refused_too_large),
            "refused_ring_sizes": {
                str(k): int(v) for k, v in sorted(self.refused_ring_sizes.items())
            },
        }


class RingSizePrior:
    """Refuse bond closures that would create an implausible ring size.

    Parameters
    ----------
    min_ring_size : int, default 4
        Smallest ring the prior will allow a closure to create.  ``4`` refuses
        three-membered rings, which is the measured defect: 0.01% of real
        tmQM rings against 67-77% of generated ones.
    max_ring_size : int, optional
        Largest ring a closure may create.  ``None`` (default) leaves it
        unbounded: macrocycles are rare but real, and the measured defect is
        at the small end.  Setting it refuses closures that would create a
        ring larger than this, which is a different claim.
    Notes
    -----
    There is deliberately no "protect atoms that would be left isolated" flag.
    An atom with no accepted bond has no path to anything, so it closes no
    ring and is never refused — the protection is a property of the test, not
    a knob.  Refusing a closure also never disconnects a graph: it declines to
    add a cycle, and the path that made the ring possible stays.
    """

    def __init__(
        self,
        min_ring_size: int = 4,
        max_ring_size: Optional[int] = None,
    ) -> None:
        if int(min_ring_size) < 3:
            raise ValueError(
                "min_ring_size must be >= 3: a ring cannot be smaller than a "
                f"triangle, got {min_ring_size!r}"
            )
        if max_ring_size is not None and int(max_ring_size) < int(min_ring_size):
            raise ValueError(
                f"max_ring_size {max_ring_size!r} is below min_ring_size "
                f"{min_ring_size!r}"
            )
        self.min_ring_size = int(min_ring_size)
        self.max_ring_size = None if max_ring_size is None else int(max_ring_size)
        self.stats = RingPriorStats()

    # ------------------------------------------------------------------
    @staticmethod
    def shortest_path_length(
        adjacency: Sequence[Set[int]], i: int, j: int, limit: int = 64
    ) -> int:
        """Edge count on the shortest ``i -> j`` path, or ``-1`` if none.

        ``limit`` bounds the search so a pathological graph cannot turn one
        candidate into a full traversal.  A path longer than ``limit`` is
        reported as no path, which allows the closure: an unbounded ring is
        not the defect this prior exists for.
        """
        if i == j:
            return 0
        if i >= len(adjacency) or j >= len(adjacency):
            return -1
        seen = {i}
        queue = deque([(i, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= limit:
                continue
            for nxt in adjacency[node]:
                if nxt == j:
                    return dist + 1
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return -1

    def would_close_ring_size(
        self, adjacency: Sequence[Set[int]], i: int, j: int
    ) -> int:
        """Ring size a new ``(i, j)`` bond would create; ``0`` if it closes none."""
        d = self.shortest_path_length(adjacency, i, j)
        return 0 if d < 0 else int(d) + 1

    def should_accept(
        self, adjacency: Sequence[Set[int]], i: int, j: int
    ) -> Tuple[bool, int]:
        """``(accept, ring_size)`` for committing the bond ``(i, j)``.

        ``ring_size`` is the size the closure would create, ``0`` when it
        closes nothing.  The decision is recorded in :attr:`stats`.
        """
        self.stats.n_considered += 1
        ring = self.would_close_ring_size(adjacency, i, j)
        if ring == 0:
            return True, 0
        too_small = ring < self.min_ring_size
        too_large = self.max_ring_size is not None and ring > self.max_ring_size
        if not (too_small or too_large):
            return True, ring
        if too_small:
            self.stats.n_refused_too_small += 1
        else:
            self.stats.n_refused_too_large += 1
        self.stats.refused_ring_sizes[ring] = (
            self.stats.refused_ring_sizes.get(ring, 0) + 1
        )
        return False, ring


def adjacency_from_bonds(
    n_atoms: int, bonds: Sequence[Tuple[int, int, int]]
) -> List[Set[int]]:
    """Adjacency list over ``n_atoms`` from ``(i, j, order)`` triples."""
    adj: List[Set[int]] = [set() for _ in range(int(n_atoms))]
    for b in bonds:
        i, j = int(b[0]), int(b[1])
        if i == j or not (0 <= i < n_atoms and 0 <= j < n_atoms):
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj
