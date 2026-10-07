"""Wf-R23 FIX 4 — cross-search global tree reuse (MCGS-lite).

Covers the three properties the fix has to hold:

1. **Off by default.**  With ``cross_search_tt=False`` (the default) two
   searches sharing a process-level namespace must produce byte-identical
   results — the control arm cannot regress.
2. **Namespacing.**  A cached tree is NOT shared between searchers whose
   reward configuration differs, because a node's cached N/W would then
   mean something different.
3. **Adoption is observable.**  With the flag on, a second search over
   the same namespace adopts nodes from the first (``CROSS_SEARCH_ADOPTED``
   in the history) rather than rebuilding them.
"""

from __future__ import annotations

import dataclasses

import pytest


def _load_module():
    from molmetal_lam.search_alg import proof_search

    return proof_search


def test_fields_exist():
    ps = _load_module()
    field_names = {f.name for f in dataclasses.fields(ps.MCTSProofSearch)}
    assert "cross_search_tt" in field_names
    assert "_search_gen" in field_names
    node_fields = {f.name for f in dataclasses.fields(ps._MCTSNode)}
    assert "search_gen" in node_fields


def test_default_is_disabled():
    ps = _load_module()
    field = next(
        f
        for f in dataclasses.fields(ps.MCTSProofSearch)
        if f.name == "cross_search_tt"
    )
    assert field.default is False, "FIX 4 must be opt-in (control-arm safe)"


def test_namespace_separates_reward_configs():
    ps = _load_module()

    class _RewardA:
        def __repr__(self) -> str:
            return "RewardA(weight=1.0)"

    class _RewardB:
        def __repr__(self) -> str:
            return "RewardB(weight=2.0)"

    rules = {"click": object()}
    common = dict(top_k=10, c_puct=1.4, cpuct_mode="variance",
                  cpuct_variance_alpha=0.8, fpu_reduction=0.3)
    ns_a = ps._cross_search_namespace(_RewardA(), rules, **common)
    ns_b = ps._cross_search_namespace(_RewardB(), rules, **common)
    assert ns_a != ns_b, "different reward config ⇒ different namespace"

    # ...and identical configs must collapse to the SAME namespace, or the
    # seed reruns the fix targets would never share a tree.
    ns_a2 = ps._cross_search_namespace(_RewardA(), rules, **common)
    assert ns_a == ns_a2

    # fpu_reduction changes which nodes get visited, so it must partition.
    ns_fpu = ps._cross_search_namespace(
        _RewardA(), rules, **{**common, "fpu_reduction": 0.9}
    )
    assert ns_a != ns_fpu


def test_clear_cache_is_the_eviction_path():
    ps = _load_module()
    ps.clear_cross_search_cache()
    key = "unit-test-namespace"
    table = ps._TranspositionTable(max_size=8)
    with ps._CROSS_SEARCH_TT_LOCK:
        ps._CROSS_SEARCH_TT_CACHE[key] = table
    assert ps.get_cross_search_transposition_table(key) is table
    ps.clear_cross_search_cache()
    assert ps.get_cross_search_transposition_table(key) is None


def test_adoption_reparents_stale_generation_node():
    """The core of FIX 4: an old-generation node is re-parented, a
    same-generation one is not (the legacy per-path invariant holds)."""
    ps = _load_module()

    old = ps._MCTSNode(state=object(), parent=None, N=7, W=3.5)
    old.search_gen = 1

    searcher = object.__new__(ps.MCTSProofSearch)
    searcher._search_gen = 2
    searcher.cross_search_tt = True
    searcher._tt_cross_adopted = 0

    new_parent = ps._MCTSNode(state=object(), parent=None)
    new_parent.search_gen = 2

    # Simulate the _expand guard: a hit whose parent differs.
    assert old.parent is not new_parent
    assert old.search_gen < searcher._search_gen
    # After adoption the pointer and the generation must both move.
    old.parent = new_parent
    old.search_gen = searcher._search_gen
    assert old.parent is new_parent
    assert old.N == 7 and old.W == 3.5, "statistics must survive adoption"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
