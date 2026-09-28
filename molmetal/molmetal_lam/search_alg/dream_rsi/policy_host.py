"""Policy host — the *question* object a policy talks to.

Pure-Python port of ``extensions/pi-dream-rsi/policy/host.ts``.

Two modes share one implementation:

*   ``live``   — probing executes real generation-evaluation attempts
    (host callback).
*   ``replay`` — probing deterministically reveals recorded children of a
    frozen world (paper §3).

The host owns the *observed* tree; in replay the full recorded world
stays private, which is what makes prefix-only policies enforceable
rather than merely requested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from .discovery_tree import (
    ROOT_ID,
    CellMeta,
    DiscoveryTree,
    Observation,
    TreeNode,
    cell_id,
    is_opened,
)


CellId = str


@dataclass
class RevealedNode:
    """One revealed cell returned by a probe: the cell id plus its outcome record."""

    cell: CellId
    node: Dict[str, Any]


# A probe returns revealed-node records for one batch.  The function is
# async-mirroring — the reference implementation awaits ``probeFn`` even
# in replay, so the host accepts either an async or a sync callable.
ProbeFn = Callable[[List[CellId], int], "Awaitable[List[RevealedNode]] | List[RevealedNode]"]


@dataclass
class Decision:
    """One decision round recorded by the host."""

    round: int
    observed: List[CellId]
    batch: List[CellId]
    revealed: Dict[CellId, Observation]
    best_after: Optional[float]


@dataclass
class HostOptions:
    """Constructor options for :class:`PolicyHost`."""

    mode: str  # "live" | "replay"
    max_parallelism: int
    probe: ProbeFn
    slots: Optional[Sequence[int]] = None
    grid: Optional[Dict[str, int]] = None
    max_rounds: Optional[int] = None
    baseline_score: Optional[float] = None
    refine_count: int = 0


_CELL_RE = re.compile(r"^b(\d+)a(\d+)$")


class PolicyHost:
    """Replay-safe observed-tree owner.

    Mirrors the TypeScript reference's surface.  In *replay* mode the host
    rejects any batch whose selected cells aren't part of the observed
    prefix, so a policy cannot outrun the round limit by ignoring its
    budget (K1/K2 enforcement).
    """

    def __init__(self, opts: HostOptions) -> None:
        if opts.mode not in {"live", "replay"}:
            raise ValueError(f"unknown host mode: {opts.mode!r}")
        self.mode = opts.mode
        self.probe_fn: ProbeFn = opts.probe
        self.grid: Optional[Dict[str, int]] = opts.grid
        self.max_parallelism: int = max(1, int(opts.max_parallelism))
        self.baseline_score: Optional[float] = (
            float(opts.baseline_score) if opts.baseline_score is not None else None
        )
        self.slot_branches: List[int] = list(opts.slots or [])
        self.refine_count_hint: int = int(opts.refine_count)
        self.max_rounds: int = max(1, int(opts.max_rounds) if opts.max_rounds is not None else 2**31 - 1)
        self.observed_tree = DiscoveryTree(
            baseline_score=self.baseline_score,
        )
        self.decisions: List[Decision] = []
        self._batch_count: int = 0
        self._probe_count: int = 0
        self._revealed: set = set()
        # Public alias for replay-probe pairing (the probe reads from here).
        self.is_revealed = self._make_is_revealed()
        self.reset()

    # ------------------------------------------------------------------
    # per-episode state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-episode state and re-seed branch slots."""
        self.observed_tree = DiscoveryTree(
            baseline_score=self.baseline_score,
            branch_count=(self.grid or {}).get("branch_count", len(self.slot_branches)),
            refine_count=(self.grid or {}).get("refine_count", self.refine_count_hint),
        )
        self._batch_count = 0
        self._probe_count = 0
        self._revealed.clear()
        self.decisions = []
        if self.slot_branches:
            branches = list(self.slot_branches)
        else:
            default = max(1, (self.grid or {}).get("branch_count", 1))
            branches = list(range(default))
        for branch in branches:
            self.observed_tree.add_branch_slot(branch)

    def is_revealed_fn(self, cell: CellId) -> bool:
        """Public predicate over the host's current revealed set.

        Bound method (not closure) so it survives ``reset()`` reassigning
        the underlying set.
        """
        return cell in self._revealed

    def _make_is_revealed(self) -> Callable[[CellId], bool]:
        """Build a callable that reads the host's revealed set.

        The replay probe pairs with this predicate so the simulator and
        the host cannot disagree about what is revealed.
        """
        host = self

        def _is_revealed(cell: CellId) -> bool:
            return host.is_revealed_fn(cell)

        return _is_revealed

    # ------------------------------------------------------------------
    # observed / legal surfaces
    # ------------------------------------------------------------------

    def observed(self) -> Dict[CellId, Observation]:
        """Observations only for cells currently in the revealed set (prefix-only)."""
        out: Dict[CellId, Observation] = {}
        for node in self.observed_tree.list():
            if node.meta.cell_id not in self._revealed:
                continue
            out[node.meta.cell_id] = self.observed_tree.observation(node.meta.cell_id)
        return out

    def legal_actions(self) -> List[CellId]:
        """Legal actions; empty when the round limit is exhausted."""
        if self.rounds_exhausted:
            return []
        return self.observed_tree.legal_actions()

    def legal_roots(self) -> List[CellId]:
        """Unopened roots; empty when the round limit is exhausted."""
        if self.rounds_exhausted:
            return []
        return self.observed_tree.unopened_roots()

    def opened_branches(self) -> List[int]:
        return self.observed_tree.opened_branches()

    def meta(self, cell_id: CellId) -> CellMeta:
        """Return structural meta for a cell; :data:`ROOT_ID` returns the root meta."""
        if cell_id == ROOT_ID:
            return CellMeta(
                cell_id=ROOT_ID,
                branch=-1,
                attempt=-1,
                parent_id=None,
                seq=0,
                tags=["root"],
            )
        node = self.observed_tree.get(cell_id)
        if node is None:
            raise ValueError(f"illegal cell id: {cell_id}")
        return CellMeta(
            cell_id=node.meta.cell_id,
            branch=node.meta.branch,
            attempt=node.meta.attempt,
            parent_id=node.meta.parent_id,
            seq=node.meta.seq,
            tags=list(node.meta.tags),
        )

    def node(self, cell_id: CellId) -> Optional[TreeNode]:
        """Host-side accessor for the raw :class:`TreeNode` (not part of policy API)."""
        return self.observed_tree.get(cell_id)

    # ------------------------------------------------------------------
    # probing
    # ------------------------------------------------------------------

    async def probe_batch(
        self,
        cells: Sequence[CellId],
        on_reveal: Optional[Callable[[Observation], None]] = None,
    ) -> List[Observation]:
        """Validate and execute a probe batch."""
        batch = self._validate_batch(list(cells))
        round_no = self._batch_count + 1
        before = [n.meta.cell_id for n in self.observed_tree.list()]
        # The round is in flight while the batch runs, so counters read correctly inside ``on_reveal``.
        self._batch_count = round_no
        records = await _call_probe(self.probe_fn, batch, round_no)
        revealed: Dict[CellId, Observation] = {}
        produced: List[Observation] = []
        for record in records:
            self._materialize(record.cell)
            self.observed_tree.set_outcome(record.cell, record.node)
            self._revealed.add(record.cell)
            observation = self.observed_tree.observation(record.cell)
            revealed[record.cell] = observation
            produced.append(observation)
            self._probe_count += 1
            if on_reveal is not None:
                on_reveal(observation)
        self.decisions.append(
            Decision(
                round=round_no,
                observed=before,
                batch=list(batch),
                revealed=revealed,
                best_after=self.best_score,
            )
        )
        return produced

    # ------------------------------------------------------------------
    # counters / state queries
    # ------------------------------------------------------------------

    @property
    def rounds_used(self) -> int:
        return self._batch_count

    @property
    def probes_used(self) -> int:
        return self._probe_count

    @property
    def best_score(self) -> Optional[float]:
        return self.observed_tree.best_score()

    @property
    def best_so_far(self) -> Optional[float]:
        return self.best_score

    @property
    def budget_spent(self) -> int:
        return self._probe_count

    def budget_done(self, budget: Optional[Dict[str, int]] = None) -> bool:
        """True when ``legal_actions`` is empty or the budget is exceeded."""
        if not self.legal_actions():
            return True
        if not budget:
            return False
        if "attempts" in budget and self.budget_spent >= int(budget["attempts"]):
            return True
        if "rounds" in budget and self.rounds_used >= int(budget["rounds"]):
            return True
        return False

    @property
    def rounds_exhausted(self) -> bool:
        return self._batch_count >= self.max_rounds

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _materialize(self, cell: CellId) -> None:
        """Attach a revealed cell that the observed tree has not seen yet.

        Replay-only invariant: an ``attempt > 0`` revealed cell must have a
        revealed parent, otherwise an error is raised (matches the TS reference).
        """
        if self.observed_tree.has(cell):
            return
        match = _CELL_RE.match(cell)
        if not match:
            raise ValueError(f"malformed cell id: {cell}")
        branch = int(match.group(1))
        attempt = int(match.group(2))
        if attempt == 0:
            self.observed_tree.add_branch_slot(branch)
            return
        parent_id = cell_id(branch, attempt - 1)
        if not self.observed_tree.has(parent_id):
            raise ValueError(f"revealed cell {cell} has no revealed parent")
        self.observed_tree.add_child(parent_id, branch, attempt)

    def _validate_batch(self, cells: List[CellId]) -> List[CellId]:
        """Enforce batch invariants: size ≤ max_parallelism, no duplicates, etc."""
        if self.rounds_exhausted:
            raise RuntimeError(
                f"round limit reached: {self._batch_count} of {self.max_rounds} decision rounds used"
            )
        if not isinstance(cells, list):
            raise ValueError("batch must be a list of cell ids")
        if len(cells) == 0:
            raise ValueError("empty batch: a policy must stop by returning no batches")
        if len(cells) > self.max_parallelism:
            raise ValueError(
                f"batch of {len(cells)} exceeds max_parallelism {self.max_parallelism}"
            )
        legal = set(self.observed_tree.legal_actions())
        seen: set = set()
        for cid in cells:
            if not isinstance(cid, str) or not cid:
                raise ValueError("batch contains a non-string cell id")
            if cid in seen:
                raise ValueError(f"duplicate cell id in batch: {cid}")
            seen.add(cid)
            if cid == ROOT_ID:
                raise ValueError("the root cell is not itself an action; select a root slot instead")
            if cid not in legal:
                raise ValueError(f"illegal cell id for the current prefix: {cid}")
        # A parent and its child must never travel in the same batch.
        for cid in seen:
            node = self.observed_tree.get(cid)
            if node is not None and node.meta.parent_id and node.meta.parent_id in seen:
                raise ValueError(
                    f"batch contains both a parent and its child: {node.meta.parent_id} + {cid}"
                )
        return list(seen)


async def _call_probe(probe_fn: ProbeFn, batch: List[CellId], round_no: int) -> List[RevealedNode]:
    """Call ``probe_fn`` accepting either an async or sync callable."""
    result = probe_fn(batch, round_no)
    if hasattr(result, "__await__"):
        return await result  # type: ignore[return-value]
    return result  # type: ignore[return-value]
