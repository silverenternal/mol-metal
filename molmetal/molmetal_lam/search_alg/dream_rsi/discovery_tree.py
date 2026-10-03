"""Discovery tree — the structure Dream-RSI explores and replays.

Pure-Python port of ``extensions/pi-dream-rsi/engine/tree.ts``.

Paper (arXiv 2609.14858 §3): a discovery tree is rooted at ``r``, the
initial workspace state.  Every non-root node has exactly one primary
parent whose saved workspace it resumes.  A node records the outcome
of one generation-evaluation attempt.  Eligible nodes form
``A(T) = {r} ∪ leaves``.

Concrete shape: the root ``r`` holds the task baseline; each *branch*
is a chain of refinement attempts ``b{branch}a{attempt}`` whose primary
parent is the previous attempt in the branch (the first attempt's
parent is ``r``).  Opening a branch = one independent workspace
(paper: "several roots" in a batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


ROOT_ID = "root"


def cell_id(branch: int, attempt: int) -> str:
    """``b3a2`` = branch 3, refinement attempt 2."""
    return f"b{branch}a{attempt}"


@dataclass
class CellMeta:
    """Structural metadata for one cell (paper Appendix B)."""

    cell_id: str
    #: Branch index; -1 for the root.
    branch: int
    #: Refinement attempt index within the branch; -1 for the root.
    attempt: int
    parent_id: Optional[str]
    #: Creation order — the replay root rule reveals the earliest-created unrevealed branch.
    seq: int
    tags: List[str] = field(default_factory=list)


def _make_cell_meta(
    cell_id: str,
    branch: int,
    attempt: int,
    parent_id: Optional[str],
    seq: int,
    tags: Sequence[str],
) -> Dict[str, Any]:
    """Build the canonical meta dict used by the on-disk JSON shape."""
    return {
        "cell_id": cell_id,
        "branch": branch,
        "attempt": attempt,
        "parent_id": parent_id,
        "seq": seq,
        "tags": list(tags),
    }


@dataclass
class TreeNode:
    """One completed (or revealed) generation-evaluation attempt.

    Mirrors the TypeScript ``TreeNode`` shape — mutability matches the
    reference implementation, where :meth:`DiscoveryTree.setOutcome`
    writes back into the same object via ``Object.assign``.
    """

    meta: CellMeta
    evaluated: bool = False
    valid: bool = False
    fail_class: str = "ok"
    error: Optional[str] = None
    #: ``s_v`` — present only for successful evaluations that reported a number.
    score: Optional[float] = None
    #: Raw evaluator value before ``higher_is_better`` normalization (diagnostics only).
    raw_score: Optional[float] = None
    #: Workspace snapshot path, relative to the Dream-RSI root.
    workspace: Optional[str] = None
    #: Path to ``proposal.md``, relative to the Dream-RSI root.
    proposal: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    #: Wall-clock ms of the attempt, for status reporting only (never used by replay).
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the canonical TreeJSON v1 shape."""
        return {
            "meta": {
                "cell_id": self.meta.cell_id,
                "branch": self.meta.branch,
                "attempt": self.meta.attempt,
                "parent_id": self.meta.parent_id,
                "seq": self.meta.seq,
                "tags": list(self.meta.tags),
            },
            "evaluated": self.evaluated,
            "valid": self.valid,
            "fail_class": self.fail_class,
            "error": self.error,
            "score": self.score,
            "raw_score": self.raw_score,
            "workspace": self.workspace,
            "proposal": self.proposal,
            "artifacts": list(self.artifacts),
            "duration_ms": self.duration_ms,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "TreeNode":
        meta_payload = payload.get("meta", {})
        meta = CellMeta(
            cell_id=meta_payload.get("cell_id", ROOT_ID),
            branch=int(meta_payload.get("branch", -1)),
            attempt=int(meta_payload.get("attempt", -1)),
            parent_id=meta_payload.get("parent_id"),
            seq=int(meta_payload.get("seq", 0)),
            tags=list(meta_payload.get("tags", []) or []),
        )
        return TreeNode(
            meta=meta,
            evaluated=bool(payload.get("evaluated", False)),
            valid=bool(payload.get("valid", False)),
            fail_class=str(payload.get("fail_class", "ok")),
            error=payload.get("error"),
            score=payload.get("score"),
            raw_score=payload.get("raw_score"),
            workspace=payload.get("workspace"),
            proposal=payload.get("proposal"),
            artifacts=list(payload.get("artifacts", []) or []),
            duration_ms=payload.get("duration_ms"),
        )


def is_success(node: Any) -> bool:
    """A *successful evaluation* = evaluated, no error, ``fail_class == "ok"``.

    Paper Appendix B: an evaluated observation with ``error == null`` and
    ``fail_class == "ok"`` is a *successful evaluation* — even when the
    evaluator's own ``valid`` flag is false or the counts are unavailable.
    """
    evaluated = getattr(node, "evaluated", False)
    fail_class = getattr(node, "fail_class", None)
    error = getattr(node, "error", None)
    return bool(evaluated) and error is None and fail_class == "ok"


def is_opened(node: Any) -> bool:
    """An unopened branch slot has no outcome yet and is therefore not an opened branch."""
    evaluated = getattr(node, "evaluated", False)
    workspace = getattr(node, "workspace", None)
    return bool(evaluated) or workspace is not None


@dataclass
class Observation:
    """What the policy sees for one revealed cell (paper Appendix B)."""

    branch: int
    attempt: int
    score: Optional[float]
    evaluated: bool
    #: The evaluator's task-level validity flag (not the same thing as a successful evaluation).
    valid: bool
    fail_class: str
    error: Optional[str]
    delta_vs_baseline: Optional[float]
    delta_vs_parent: Optional[float]
    n_valid: int
    n_total: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "attempt": self.attempt,
            "score": self.score,
            "evaluated": self.evaluated,
            "valid": self.valid,
            "fail_class": self.fail_class,
            "error": self.error,
            "delta_vs_baseline": self.delta_vs_baseline,
            "delta_vs_parent": self.delta_vs_parent,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
        }


@dataclass
class TreeJSON:
    """TreeJSON v1 — the on-disk shape of a discovery tree."""

    version: int = 1
    branch_count: int = 0
    refine_count: int = 0
    seq: int = 1
    baseline_score: Optional[float] = None
    cells: List[TreeNode] = field(default_factory=list)


class DiscoveryTree:
    """Mutable discovery tree with the observation/reveal operations the policy API needs.

    Behaviour is identical to the TypeScript reference:
        *   the root holds the task baseline;
        *   :meth:`addBranchSlot` materializes a not-yet-opened root slot;
        *   :meth:`addChild` registers a refinement child;
        *   :meth:`observation` aggregates deltas over the branch trajectory
            revealed so far;
        *   :meth:`bestScore` walks the successful evaluations.
    """

    def __init__(
        self,
        baseline_score: Optional[float] = None,
        branch_count: int = 0,
        refine_count: int = 0,
    ) -> None:
        self.branch_count = int(branch_count)
        self.refine_count = int(refine_count)
        self.seq = 1

        # Use mutable defaults; the dataclass will not deep-copy them because
        # TreeNode is a dataclass with default_factory for the list fields.
        root_meta = CellMeta(
            cell_id=ROOT_ID,
            branch=-1,
            attempt=-1,
            parent_id=None,
            seq=0,
            tags=["root"],
        )
        self.root = TreeNode(
            meta=root_meta,
            evaluated=True,
            valid=isinstance(baseline_score, (int, float)),
            fail_class="ok",
            error=None,
            score=_opt_float(baseline_score),
            raw_score=_opt_float(baseline_score),
            workspace=None,
            proposal=None,
            artifacts=[],
            duration_ms=None,
        )
        self.cells: Dict[str, TreeNode] = {}

    # ------------------------------------------------------------------
    # baseline / accessors
    # ------------------------------------------------------------------

    @property
    def baseline_score(self) -> Optional[float]:
        """Returns the root's score as the baseline."""
        return self.root.score

    def get(self, cell: str) -> Optional[TreeNode]:
        """Retrieve node by cell id; special-cases :data:`ROOT_ID`."""
        if cell == ROOT_ID:
            return self.root
        return self.cells.get(cell)

    def has(self, cell: str) -> bool:
        return cell == ROOT_ID or cell in self.cells

    def list(self) -> List[TreeNode]:
        """All revealed non-root cells, in creation order."""
        return sorted(self.cells.values(), key=lambda n: n.meta.seq)

    @property
    def n_non_root(self) -> int:
        """Number of revealed non-root nodes = ``N`` in the replay objective."""
        return len(self.cells)

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def add_branch_slot(self, branch: int) -> TreeNode:
        """Materialize a not-yet-opened root slot (a root child with no outcome yet).

        Idempotent on existing cell — re-calling returns the existing node
        instead of allocating a second one (matches the TS reference).
        """
        cid = cell_id(branch, 0)
        existing = self.cells.get(cid)
        if existing is not None:
            return existing
        meta = CellMeta(
            cell_id=cid,
            branch=int(branch),
            attempt=0,
            parent_id=ROOT_ID,
            seq=self.seq,
            tags=["root-slot"],
        )
        node = TreeNode(meta=meta)
        self.seq += 1
        self.cells[cid] = node
        return node

    def add_child(
        self,
        parent_id: str,
        branch: int,
        attempt: int,
        tags: Optional[Sequence[str]] = None,
    ) -> TreeNode:
        """Register a new refinement child of ``parent_id`` (used by the live phase)."""
        cid = cell_id(branch, attempt)
        if cid in self.cells:
            return self.cells[cid]
        meta = CellMeta(
            cell_id=cid,
            branch=int(branch),
            attempt=int(attempt),
            parent_id=parent_id,
            seq=self.seq,
            tags=list(tags or []),
        )
        node = TreeNode(meta=meta)
        self.seq += 1
        self.cells[cid] = node
        return node

    def set_outcome(self, cell: str, outcome: Dict[str, Any]) -> TreeNode:
        """Record the outcome of an attempt.

        The evaluator's own ``valid`` flag wins when provided; otherwise
        falls back to :func:`is_success` (matches the TS reference's
        ``Object.assign`` semantics).
        """
        node = self.get(cell)
        if node is None:
            raise KeyError(f"unknown cell: {cell}")
        for key, value in outcome.items():
            if key == "meta":
                if value is None:
                    continue
                # Merge meta dict into existing meta, prefer incoming values.
                for mk, mv in value.items():
                    setattr(node.meta, mk, mv)
                continue
            setattr(node, key, value)
        if "valid" not in outcome:
            node.valid = is_success(node)
        return node

    # ------------------------------------------------------------------
    # structural queries
    # ------------------------------------------------------------------

    def children_of(self, cell: str) -> List[TreeNode]:
        """All nodes whose primary parent is the given cell."""
        return [n for n in self.list() if n.meta.parent_id == cell]

    def leaves(self) -> List[TreeNode]:
        """Non-root nodes without revealed children."""
        parents = {n.meta.parent_id for n in self.cells.values() if n.meta.parent_id}
        return [n for n in self.list() if n.meta.cell_id not in parents]

    def eligible(self) -> List[str]:
        """``A(T) = {r} ∪ leaves`` — ids of eligible cells, creation order with the root last."""
        leaf_ids = [n.meta.cell_id for n in self.leaves()]
        return [*leaf_ids, ROOT_ID]

    def opened_branches(self) -> List[int]:
        """Branches that have been opened (their slot has an outcome)."""
        out: set = set()
        for n in self.cells.values():
            if is_opened(n):
                out.add(n.meta.branch)
        return sorted(out)

    def unopened_roots(self) -> List[str]:
        """Root slots that have not been opened yet, in branch order."""
        return [
            n.meta.cell_id
            for n in sorted(self.cells.values(), key=lambda x: x.meta.branch)
            if n.meta.parent_id == ROOT_ID and not is_opened(n)
        ]

    def frontier_of(self, branch: int) -> Optional[TreeNode]:
        """Deepest revealed cell of a branch (the branch frontier), or null."""
        chain = self.branch_chain(branch)
        return chain[-1] if chain else None

    def branch_chain(self, branch: int) -> List[TreeNode]:
        return [n for n in self.list() if n.meta.branch == branch]

    def legal_actions(self) -> List[str]:
        """Legal action set: unopened roots + one frontier per opened branch.

        Frontiers at the grid's refinement limit are not selectable: the grid
        is ``branch x attempt``.
        """
        actions: List[str] = list(self.unopened_roots())
        seen: set = set()
        # Newest frontier first: prefer progress on branches already investigated.
        for node in reversed(self.list()):
            if node.meta.branch in seen:
                continue
            if not is_opened(node):
                continue
            frontier = self.frontier_of(node.meta.branch)
            if frontier is None:
                continue
            seen.add(node.meta.branch)
            if self.refine_count > 0 and frontier.meta.attempt + 1 >= self.refine_count:
                continue
            actions.append(frontier.meta.cell_id)
        return actions

    # ------------------------------------------------------------------
    # observations / metrics
    # ------------------------------------------------------------------

    def observation(self, cell: str) -> Observation:
        """Observation for a revealed cell.

        ``n_valid`` / ``n_total`` count the branch trajectory revealed so far
        (a branch with zero valid attempts is evidence, not an automatic
        closure).
        """
        node = self.get(cell)
        if node is None:
            raise KeyError(f"unknown cell: {cell}")
        parent = self.get(node.meta.parent_id) if node.meta.parent_id else None
        if node.meta.branch >= 0:
            chain = [
                n
                for n in self.branch_chain(node.meta.branch)
                if n.meta.seq <= node.meta.seq
            ]
        else:
            chain = [node]
        evaluated = [n for n in chain if n.evaluated]
        baseline = self.baseline_score
        score = node.score

        def _delta(ref: Optional[float]) -> Optional[float]:
            if isinstance(ref, (int, float)) and isinstance(score, (int, float)):
                return round(float(score) - float(ref), 10)
            return None

        parent_score = (
            parent.score
            if parent is not None and isinstance(parent.score, (int, float))
            else baseline
        )
        return Observation(
            branch=node.meta.branch,
            attempt=node.meta.attempt,
            score=score,
            evaluated=bool(node.evaluated),
            valid=bool(node.valid),
            fail_class=str(node.fail_class),
            error=node.error,
            delta_vs_baseline=_delta(baseline),
            delta_vs_parent=_delta(parent_score),
            n_valid=sum(1 for n in evaluated if n.valid),
            n_total=len(evaluated),
        )

    def best_score(self) -> Optional[float]:
        """Best score over successful evaluations across the revealed tree.

        Includes the root baseline when the root itself is a successful
        evaluation (matches the TS reference).
        """
        best: Optional[float] = None
        root_ok = self.root.error is None and self.root.fail_class == "ok"
        if root_ok:
            best = self.root.score
        for node in self.cells.values():
            if not is_success(node):
                continue
            if not isinstance(node.score, (int, float)):
                continue
            if best is None or float(node.score) > float(best):
                best = float(node.score)
        return best

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        """Serialize to TreeJSON v1."""
        return {
            "version": 1,
            "branch_count": self.branch_count,
            "refine_count": self.refine_count,
            "seq": self.seq,
            "baseline_score": self.root.score,
            "cells": [n.to_dict() for n in self.list()],
        }

    @staticmethod
    def from_json(payload: Dict[str, Any]) -> "DiscoveryTree":
        """Deserialize from TreeJSON v1."""
        if not isinstance(payload, dict):
            raise ValueError("tree json must be a dict")
        baseline = payload.get("baseline_score")
        tree = DiscoveryTree(
            baseline_score=_opt_float(baseline),
            branch_count=int(payload.get("branch_count", 0) or 0),
            refine_count=int(payload.get("refine_count", 0) or 0),
        )
        tree.seq = int(payload.get("seq", 1) or 1)
        for raw in payload.get("cells", []) or []:
            if not isinstance(raw, dict):
                continue
            meta = raw.get("meta") or {}
            cid = meta.get("cell_id")
            if not cid or cid == ROOT_ID:
                continue
            node = TreeNode.from_dict(raw)
            tree.cells[cid] = node
            tree.seq = max(tree.seq, node.meta.seq + 1)
        return tree


def _opt_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to ``float`` when possible, else return ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None
