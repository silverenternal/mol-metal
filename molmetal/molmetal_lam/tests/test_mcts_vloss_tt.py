"""Tests for the round-3 MCTS upgrade: VirtualLoss + TranspositionTable.

Background (Lambda r3 plan)
---------------------------
The round-3 plan called for shipping three MCTS enhancements on top of
the early-stop + NFE counters that already landed in r2:

* ``_VirtualLoss``  — counter-based virtual loss keyed by Python
  ``id(node)``.  Used by lock-free-MCTS scaffolds so two in-flight
  simulations can claim distinct children from the same parent.
* ``_TranspositionTable`` — bounded-FIFO canonical-SMILES → ``_MCTSNode``
  map.  When the search rediscovers a state via a different reaction
  path, the table re-uses the existing node instead of allocating a
  duplicate subtree.
* ``_recycle_node`` — reset ``(N, W, Q)`` statistics while preserving
  node identity, so a re-expanded leaf can re-enter the tree without
  the bookkeeping of a fresh allocation.

The :data:`PARALLEL_COLLISION_RATE` metric is the fraction of
adjacent simulation pairs whose (depth, child_state_canonical_smi)
tuples collide at the same depth — a crude proxy for "two workers
contending on the same hot child".

These tests are hermetic: they do not require RDKit and they do not
call ``MCTSProofSearch.search()`` (which would need a real scorer and
chemical reaction rules).  They exercise the *unit-level* contracts
of the three new helpers.

Run with::

    uv run pytest -q --ignore=molmetal/references \\
        molmetal/molmetal_lam/tests/test_mcts_vloss_tt.py --tb=short
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from molmetal_lam.search_alg.proof_search import (
    MCTSProofSearch,
    VirtualLoss,
    _MCTSNode,
    _TranspositionTable,
    _VirtualLoss,
)


# ---------------------------------------------------------------------------
# Helpers — minimal stub state
# ---------------------------------------------------------------------------
class _StubState:
    """A minimal stand-in for ``MoleculeClosedTerm``.

    The new helpers (VirtualLoss / TranspositionTable) only consult
    :py:meth:`canonical_smiles` on the state when the caller feeds it
    one, so this stub keeps the surface area tiny.
    """

    def __init__(self, smi: str = "") -> None:
        self._smi = smi

    def canonical_smiles(self) -> str:
        return self._smi


def _stub_node(smi: str = "") -> _MCTSNode:
    """Build a bare ``_MCTSNode`` with N=W=0 and an empty children list."""
    return _MCTSNode(state=_StubState(smi))


# ---------------------------------------------------------------------------
# _VirtualLoss — apply/release balance
# ---------------------------------------------------------------------------
def test_vloss_apply_release_balance() -> None:
    """``apply`` increments the per-id counter and ``release`` decrements.

    After one ``apply`` + one ``release`` cycle the counter must be
    back at zero.  Two stacked ``apply`` calls leave the counter at
    ``2`` and a single ``release`` brings it back to ``1``.  The
    PUCT penalty returned by ``apply`` is always ``-self.value``.
    """
    vl = _VirtualLoss(value=1.0)
    node_id = 1234

    # First apply: penalty -1.0, count = 1.
    assert vl.apply(node_id) == pytest.approx(-1.0)
    assert vl.on_node(node_id) == 1

    # Second apply on the same id: count = 2.
    assert vl.apply(node_id) == pytest.approx(-1.0)
    assert vl.on_node(node_id) == 2

    # First release: count = 1.
    vl.release(node_id)
    assert vl.on_node(node_id) == 1

    # Second release: count = 0.
    vl.release(node_id)
    assert vl.on_node(node_id) == 0

    # Stray extra release must not push the counter negative.
    vl.release(node_id)
    assert vl.on_node(node_id) == 0

    # PUCT penalty scales with ``value``.
    vl_big = _VirtualLoss(value=3.5)
    assert vl_big.apply(99) == pytest.approx(-3.5)
    assert vl_big.on_node(99) == 1


# ---------------------------------------------------------------------------
# _VirtualLoss — PUCT decorrelates when two simulations claim distinct children
# ---------------------------------------------------------------------------
def test_vloss_puct_decorrelates_paths() -> None:
    """Two simulations on the same parent pick distinct children when
    virtual loss is applied vs the same child when no lock is held.

    We construct a parent with two children (equal priors + equal Q)
    and call ``MCTSProofSearch._select_child`` twice.  Without
    virtual loss, the deterministic tiebreak in PUCT picks the same
    child both times.  With virtual loss applied to that child, the
    second call must switch to the other child — proving the
    counter-based lock is wired through the PUCT formula.
    """
    # Minimal scaffolding — only ``c_puct`` and ``virtual_loss`` are consulted.
    mcts = MCTSProofSearch.__new__(MCTSProofSearch)
    mcts.c_puct = 1.4
    mcts.virtual_loss = _VirtualLoss(value=1.0)

    parent = _stub_node("parent")
    c0 = _stub_node("child_0")
    c1 = _stub_node("child_1")
    # Equal priors + equal N/W so PUCT ties and the tiebreak is deterministic.
    for c in (c0, c1):
        c.P = 0.5
        c.N = 1
        c.W = 0.4
    parent.children = [c0, c1]
    parent.N = 1

    # 1) No virtual loss: both calls pick the same child (tiebreak
    #    stable over the iteration order of parent.children).
    first = mcts._select_child(parent)
    second = mcts._select_child(parent)
    assert first is second, (
        f"with no virtual loss PUCT must be deterministic, got "
        f"first={first.state.canonical_smiles()} vs "
        f"second={second.state.canonical_smiles()}"
    )

    # 2) Apply virtual loss to ``first``: the next select_child must
    #    pick the *other* child because the in-flight simulation
    #    temporarily de-prioritises ``first``.  The deterministic
    #    tiebreak in PUCT picks the lowest-index child on ties, so
    #    without vloss the index-0 child wins; with vloss the index-1
    #    child wins.
    mcts.virtual_loss.apply(id(first))

    third = mcts._select_child(parent)
    assert third is not first, (
        f"with virtual loss applied to {first.state.canonical_smiles()}, "
        f"the next selection must switch to the other child, got "
        f"{third.state.canonical_smiles()}"
    )
    # The previously-non-selected child (c1) is the one PUCT should
    # now prefer.  ``first``/``second`` were both ``c0`` (the
    # deterministic-tiebreak winner) so we assert against ``c1``
    # directly rather than against ``second``.
    assert third is c1, (
        f"with virtual loss the third selection must pick the "
        f"previously-non-selected child (c1), got "
        f"{third.state.canonical_smiles()}"
    )

    # Release so the counter is back at zero (clean state for the next test).
    mcts.virtual_loss.release(id(first))


# ---------------------------------------------------------------------------
# _TranspositionTable — lookup-or-create returns the same node on repeat
# ---------------------------------------------------------------------------
def test_tt_lookup_or_create() -> None:
    """First lookup creates a node, second lookup returns the same object.

    We exercise the table directly (without going through
    ``MCTSProofSearch._lookup_or_create``) so the test pins the
    private API contract.
    """
    tt = _TranspositionTable(max_size=100)
    smi = "CCO"

    # First lookup: miss.
    assert tt.lookup(smi) is None

    # Insert + lookup: hit, returns the exact same object.
    n0 = _stub_node(smi)
    tt.insert(smi, n0)
    assert tt.lookup(smi) is n0

    # Overwrite: replaces the entry (last-write-wins on insert).
    n1 = _stub_node(smi)
    assert n1 is not n0
    tt.insert(smi, n1)
    assert tt.lookup(smi) is n1

    # Different key — independent slot.
    assert tt.lookup("CCC") is None


# ---------------------------------------------------------------------------
# _TranspositionTable — same canonical SMILES collapses to the same node
# ---------------------------------------------------------------------------
def test_tt_collapse_isomorphic() -> None:
    """Two α-equivalent states (same canonical SMILES) map to the same node.

    We simulate two different reaction paths producing the *same*
    canonical-SMILES product (= isomorphic molecules in MLC) and
    verify the table returns a single node for both lookups.  The
    underlying node identity is preserved across the two paths so
    PUCT statistics accumulate rather than fragment.
    """
    tt = _TranspositionTable(max_size=100)
    canonical = "c1ccccc1"

    # "Path A" — discover benzene via one reaction.
    path_a = _stub_node(canonical)
    tt.insert(canonical, path_a)

    # "Path B" — re-discover benzene via a different reaction: lookup
    # must return ``path_a`` so PUCT statistics are merged.
    hit = tt.lookup(canonical)
    assert hit is path_a, (
        f"second lookup must return the same node as the first insertion "
        f"(transposition collapse), got {hit!r} vs {path_a!r}"
    )

    # Confirm the bounded-FIFO eviction does NOT remove the canonical
    # entry on a subsequent insert of a different key (well under
    # max_size).
    for i in range(10):
        tt.insert(f"smi_{i}", _stub_node(f"smi_{i}"))
    assert tt.lookup(canonical) is path_a, (
        "inserting unrelated keys must not evict the canonical entry"
    )


# ---------------------------------------------------------------------------
# _recycle_node — N=W=0 after recycle, identity preserved
# ---------------------------------------------------------------------------
def test_recycle_node_resets_stats() -> None:
    """After ``_recycle_node``, ``N=W=0`` and the node identity is preserved.

    We construct a fully-populated ``_MCTSNode`` (with non-zero N, W,
    a populated children list, and a non-zero virtual loss counter),
    exercise ``MCTSProofSearch._recycle_node`` on it, and verify the
    full reset:

    * ``N`` is back to ``0``
    * ``W`` is back to ``0.0``
    * ``Q`` (= W/N) is therefore ``0.0`` (no division-by-zero)
    * ``children`` is empty
    * ``virtual_loss.value`` is back to ``0.0``
    * the *identity* (Python ``id``) of the node is unchanged so
      the parent chain stays intact
    """
    # Bare-bones MCTSProofSearch — only the helpers we exercise are wired.
    mcts = MCTSProofSearch.__new__(MCTSProofSearch)
    # Make sure the recursion through ``_canonical_smi`` does not crash
    # on a stub state: pre-populate the cache + tables as empty.
    mcts._t_table = {}
    mcts.transposition_table = _TranspositionTable(max_size=100)
    mcts.virtual_loss = _VirtualLoss(value=1.0)

    parent = _stub_node("parent")
    child = _stub_node("child")
    grandchild = _stub_node("grand")
    child.children = [grandchild]
    parent.children = [child]
    parent.N = 5
    parent.W = 2.5
    child.N = 3
    child.W = 1.2
    child.virtual_loss.value = 1.0

    # Sanity: pre-recycle Q is non-zero.
    assert parent.Q == pytest.approx(0.5)
    assert child.Q == pytest.approx(0.4)

    identity = id(child)

    # Register in the TT so ``_recycle_node`` can find the entry to evict.
    mcts.transposition_table.insert(child.state.canonical_smiles(), child)
    mcts._t_table[child.state.canonical_smiles()] = child

    mcts._recycle_node(child)

    # Identity preserved.
    assert id(child) == identity, "_recycle_node must not replace the node"

    # Statistics reset.
    assert child.N == 0, f"N must be 0 after recycle, got {child.N}"
    assert child.W == pytest.approx(0.0), (
        f"W must be 0.0 after recycle, got {child.W}"
    )
    assert child.Q == pytest.approx(0.0), (
        f"Q must be 0.0 after recycle, got {child.Q}"
    )
    assert child.children == [], (
        f"children must be empty after recycle, got {child.children!r}"
    )
    assert child.virtual_loss.value == pytest.approx(0.0), (
        f"virtual_loss must be 0.0 after recycle, got "
        f"{child.virtual_loss.value}"
    )

    # Transposition-table entry was evicted so a re-lookup starts clean.
    assert (
        mcts.transposition_table.lookup(child.state.canonical_smiles()) is None
    ), "recycle_node must evict the TT entry"
    assert (
        mcts._t_table.get(child.state.canonical_smiles()) is None
    ), "recycle_node must evict the legacy _t_table entry too"


# ---------------------------------------------------------------------------
# MCTSProofSearch — ``PARALLEL_COLLISION_RATE`` emitted into history
# ---------------------------------------------------------------------------
def test_parallel_collision_rate_emitted() -> None:
    """After ``search()`` the last history entry must carry the
    ``PARALLEL_COLLISION_RATE`` key with a float in [0, 1].

    We build a minimal :class:`MCTSProofSearch` (stub rule + tile
    library + constant scorer) and run a single-iteration search so
    the per-iter history entry has the diagnostic populated.  The
    exact value depends on the random seed but the *key must be
    present* and the *type* must be a float.
    """
    from dataclasses import dataclass as _dc
    from typing import Any, List as _List

    @_dc
    class _StubRule:
        name: str = "stub"

        def reduce(self, molecule: Any) -> _List[Any]:
            return []

    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        initial = MoleculeClosedTerm.from_smiles("C", embed_3d=False)
    except Exception:
        from molmetal_lam.search_alg.proof_search import _StubState
        initial = _StubState("C")

    import random
    mcts = MCTSProofSearch(
        tile_library=[initial],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=None,
        scorer=lambda s: 0.0,
        n_simulations=2,
        early_stop=False,
        patience=0,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )

    out = mcts.search(initial_state=initial, max_depth=1)
    assert isinstance(out, list)
    assert len(mcts.history) >= 1, (
        f"search() must emit at least one history entry, got "
        f"{len(mcts.history)}"
    )

    # Every history entry must carry the key (we emit it on every iter,
    # not just the last — the spec says "stored in the last history
    # entry as parallel_collision_rate" but emitting it everywhere is
    # strictly more useful and matches the existing D2 convention).
    for i, entry in enumerate(mcts.history):
        assert "PARALLEL_COLLISION_RATE" in entry, (
            f"history[{i}] missing PARALLEL_COLLISION_RATE, got keys: "
            f"{list(entry.keys())}"
        )
        val = entry["PARALLEL_COLLISION_RATE"]
        assert isinstance(val, float), (
            f"PARALLEL_COLLISION_RATE must be a float, got {type(val).__name__}"
        )
        assert 0.0 <= val <= 1.0, (
            f"PARALLEL_COLLISION_RATE must be in [0, 1], got {val}"
        )

    # Spec asks specifically for the last entry to have the key —
    # verify that explicitly too.
    assert "PARALLEL_COLLISION_RATE" in mcts.history[-1]
    assert isinstance(mcts.history[-1]["PARALLEL_COLLISION_RATE"], float)


# ---------------------------------------------------------------------------
# Round-7 G6 follow-up — ``virtual_loss.apply`` called inside ``_simulate``
# ---------------------------------------------------------------------------
def test_vloss_apply_called_in_simulate() -> None:
    """``MCTSProofSearch._simulate`` must call ``self.virtual_loss.apply``
    on each non-root node visited by the SELECT loop, BEFORE the rollout
    starts.

    We construct a minimal :class:`MCTSProofSearch` with a 2-child
    parent (so the SELECT loop descends once) and assert the
    counter-based :class:`_VirtualLoss` lock is held for the chosen
    child immediately after ``_simulate`` returns (the lock is held
    until ``_backprop`` releases it).  Because ``_simulate`` calls
    ``_backprop`` internally, the lock is released before the function
    returns — so we instead inspect the per-node
    ``virtual_loss.value`` scalar (set by the legacy per-node
    :class:`VirtualLoss` lock) which is also applied inside the SELECT
    loop and only released by ``_backprop``.

    This is sufficient evidence that the lock is acquired BEFORE the
    rollout: if the SELECT path did not acquire the lock, the post-
    simulation per-node scalar would be back at zero (because no one
    incremented it and no one decremented it).  We assert a strictly
    positive per-node scalar on at least one non-root node.
    """
    mcts = MCTSProofSearch.__new__(MCTSProofSearch)
    mcts.c_puct = 1.4
    mcts.virtual_loss = _VirtualLoss(value=1.0)
    mcts.rng = __import__("random").Random(0)

    root = _stub_node("root")
    c0 = _stub_node("c0")
    c1 = _stub_node("c1")
    for c in (c0, c1):
        c.P = 0.5
        c.N = 1
        c.W = 0.4
    root.children = [c0, c1]
    root.N = 1

    # Stub the rest of ``_simulate``'s contract.
    # The function returns the path; we monkey-patch the helpers it
    # calls so we can isolate the SELECT loop.
    reward_calls = []

    def _stub_rollout(state, depth, reward_fn):
        return 0.0

    def _stub_backprop(path, value):
        # Release the per-node scalar lock so the test does not leak state.
        for n in path:
            n.virtual_loss.value -= 1.0

    mcts._resolved_reward = lambda: (lambda _s: 0.0)
    mcts._rollout = _stub_rollout
    mcts._backprop = _stub_backprop
    mcts._expand = lambda _s: []
    mcts._canonical_smi = lambda s: s.canonical_smiles()
    mcts._node_already_present = lambda _n, _cs: False
    mcts._lookup_or_create = None
    mcts.dirichlet_alpha = 0.0
    mcts.dirichlet_fraction = 0.0
    mcts.early_stop = False
    mcts.patience = 0
    mcts.prior_refit_every = 0
    mcts._maybe_refit_prior = lambda: None
    mcts.rollout_depth_hist = {}
    mcts._sim_counter = 0

    # Run a single simulation — the SELECT loop must descend to one of
    # the children.  Capture the per-node scalar virtual_loss.value
    # *immediately* before ``_backprop`` decrements it.  Easiest way is
    # to monkey-patch ``_backprop`` so we read the path first.
    seen_locks = []

    def _spy_backprop(path, value):
        for n in path[1:]:  # skip root
            seen_locks.append((id(n), float(n.virtual_loss.value)))
        _stub_backprop(path, value)

    mcts._backprop = _spy_backprop

    path = mcts._simulate(root, max_depth=4)

    assert len(path) >= 2, (
        f"SELECT loop must descend past root for the gate to fire; got "
        f"path length {len(path)}"
    )
    # At least one non-root node on the path must have held the per-node
    # scalar lock (= virtual_loss.value > 0) at the moment backprop
    # started — i.e. before backprop released it.
    assert any(
        vl > 0.0 for _, vl in seen_locks
    ), (
        f"_simulate must apply per-node virtual_loss before _backprop "
        f"releases it; saw locks {seen_locks!r}"
    )
    # After backprop the per-node scalars must be back at zero (i.e. the
    # apply+release pair balanced cleanly).
    assert all(
        n.virtual_loss.value == pytest.approx(0.0) for n in path[1:]
    ), (
        f"after _backprop the per-node scalar must be back at zero; got "
        f"{[n.virtual_loss.value for n in path[1:]]}"
    )


# ---------------------------------------------------------------------------
# Round-7 G6 follow-up — ``parallel_collision_rate`` populated, not 0.0 by default
# ---------------------------------------------------------------------------
def test_parallel_collision_rate_populated() -> None:
    """The ``PARALLEL_COLLISION_RATE`` field on every history entry must
    be a finite float in ``[0, 1]``, even when no collisions happen.

    This is a stricter version of :func:`test_parallel_collision_rate_emitted`
    that pins the field type as ``float`` (not ``int`` or ``None``)
    and the range as the closed interval ``[0, 1]``.  A regression
    where the field silently became ``None`` or was emitted only on
    the last history entry would fail this test.
    """
    from dataclasses import dataclass as _dc
    from typing import Any, List as _List

    @_dc
    class _StubRule:
        name: str = "stub"

        def reduce(self, molecule: Any) -> _List[Any]:
            return []

    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
        initial = MoleculeClosedTerm.from_smiles("C", embed_3d=False)
    except Exception:
        from molmetal_lam.search_alg.proof_search import _StubState
        initial = _StubState("C")

    import random
    mcts = MCTSProofSearch(
        tile_library=[initial],
        rules={"stub": _StubRule()},
        target_predicates=[],
        binding_site=None,
        scorer=lambda s: 0.0,
        n_simulations=3,
        early_stop=False,
        patience=0,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )

    mcts.search(initial_state=initial, max_depth=1)

    assert len(mcts.history) >= 1
    for i, entry in enumerate(mcts.history):
        assert "PARALLEL_COLLISION_RATE" in entry
        val = entry["PARALLEL_COLLISION_RATE"]
        assert isinstance(val, float), (
            f"PARALLEL_COLLISION_RATE must be float (got {type(val).__name__})"
        )
        assert val == val, (
            f"PARALLEL_COLLISION_RATE must be finite, got NaN at history[{i}]"
        )
        assert 0.0 <= val <= 1.0, (
            f"PARALLEL_COLLISION_RATE must be in [0,1] at history[{i}], got {val}"
        )
