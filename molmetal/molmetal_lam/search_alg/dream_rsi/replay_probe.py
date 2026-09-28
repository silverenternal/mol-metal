"""Replay probe — deterministic reveal of the frozen discovery tree.

Pure-Python port of ``extensions/pi-dream-rsi/engine/replay.ts``
(``makeReplayProbe``).

Replay never generates candidates: it deterministically reveals what was
recorded in the frozen discovery tree.  For a batch of cells from the
policy, :func:`make_replay_probe` applies two distinct rules:

*   **Unopened-root-slot rule** — for any selected cell that is an
    unopened root slot (``parent_id == ROOT_ID``, ``attempt == 0``,
    ``not is_revealed(cell)``), reveal the earliest-created (lowest
    ``seq``) recorded child of ``ROOT`` that is *both* unrevealed
    *and* ``is_opened`` (i.e. executed: ``evaluated`` or
    ``workspace`` not null).  Only executed branch roots are eligible,
    so revealing a never-attempted slot cannot charge a probe and
    leave it selectable.  The slot selector is consumed first;
    multiple slot selectors in one batch reveal the first N earliest
    unrevealed opened-root children in ``seq`` order.
*   **Opened-frontier rule** — for any other selected cell (a revealed
    frontier), reveal that node's unique recorded child if one exists
    in the world (``world.childrenOf(cell)[0]``) and that child is not
    yet revealed; ``Child(v) = ∅`` when no recorded continuation
    remains.

The probe is paired with an ``is_revealed`` predicate over the host's
observed prefix so the simulator and the host cannot disagree about
what is revealed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .discovery_tree import ROOT_ID, DiscoveryTree, is_opened
from .policy_host import CellId, ProbeFn, RevealedNode


@dataclass
class ProbeResult:
    """Result of one replay probe call.

    Equivalent to a ``List[RevealedNode]`` returned by the TS probe, but
    shaped as a dataclass for clean test introspection.
    """

    revealed: List[RevealedNode]
    #: Number of slot-selector reveals in this batch.
    slot_reveals: int = 0
    #: Number of frontier-selector reveals in this batch.
    frontier_reveals: int = 0

    def __len__(self) -> int:
        return len(self.revealed)

    def __bool__(self) -> bool:
        return bool(self.revealed)


def make_replay_probe(
    world: DiscoveryTree,
    is_revealed: Callable[[CellId], bool],
) -> ProbeFn:
    """Return a probe that deterministically reveals recorded children of ``world``.

    The probe is async-mirroring — the reference implementation awaits
    ``probeFn`` even in replay, so the returned callable may be either
    an async or a sync function (the host awaits regardless).
    """

    def _probe(cells: Sequence[CellId], _round: int) -> List[RevealedNode]:
        return _reveal(list(cells), world, is_revealed)

    return _probe


def _reveal(
    cells: List[CellId],
    world: DiscoveryTree,
    is_revealed: Callable[[CellId], bool],
) -> List[RevealedNode]:
    """Apply the two replay rules in order.

    Public to enable unit testing without instantiating the closure.
    """
    out: List[RevealedNode] = []

    def is_slot_probe(cell: CellId) -> bool:
        node = world.get(cell)
        if node is None:
            return False
        return (
            node.meta.parent_id == ROOT_ID
            and node.meta.attempt == 0
            and not is_revealed(cell)
        )

    # ---- root rule ----------------------------------------------------
    # Earliest-created recorded branch outside the observed tree.  Only
    # *executed* branch roots are eligible: a live episode may have
    # opened fewer branches than the grid planned, and revealing an
    # unused slot would charge a probe and leave it selectable,
    # distorting the sweep.
    slots = [c for c in cells if is_slot_probe(c)]
    if slots:
        pending = sorted(
            [
                n
                for n in world.list()
                if n.meta.parent_id == ROOT_ID
                and not is_revealed(n.meta.cell_id)
                and is_opened(n)
            ],
            key=lambda n: n.meta.seq,
        )
        for i, slot_cell in enumerate(slots):
            if i >= len(pending):
                break
            out.append(
                RevealedNode(
                    cell=pending[i].meta.cell_id, node=pending[i].to_dict()
                )
            )

    # ---- opened frontier rule ----------------------------------------
    for cell in cells:
        if is_slot_probe(cell):
            continue
        children = world.children_of(cell)
        if not children:
            continue
        child = children[0]
        if is_revealed(child.meta.cell_id):
            continue
        out.append(RevealedNode(cell=child.meta.cell_id, node=child.to_dict()))

    return out
