"""Tests for TODO-30 Rank-9 P5.3 ``register_rule`` + ``vet_rule`` public API.

Background
----------
P5.3 in ``TODO/pending/30_pitfall_reinforce_plan.md`` calls for a
small public API so future rules (and tests) can introspect the
rule registry without re-importing the click / metal-coordination
modules.  These tests pin:

* ``register_rule`` + ``list_rules`` round-trip.
* ``unregister_rule`` removes the rule and the ``vet_rule`` lookup
  raises ``KeyError`` afterwards.
* ``vet_rule`` returns a populated :class:`RuleRecord` even when
  the rule has never fired.
* Multiple scopes (``click``, ``Pt``, ``Ru``, ``Au``, ``custom``)
  coexist and ``list_rules(scope=...)`` filters correctly.
* Duplicate registration raises an informative :class:`ValueError`.
* ``vet_rule`` returns the correct ``yield_range`` from the
  declared metadata.
* ``record_test_result`` updates ``last_test_result`` and
  preserves other fields.
* The integration helpers (``wire_click_rules``,
  ``wire_metal_coordination_rules``) populate the registry from
  the existing modules without modifying them.

Each test uses a unique rule name (prefixed ``t30_``) so it
coexists with the integration helpers that register the canonical
``CuAAC`` / ``SPAAC`` / etc.  We ``clear_registry()`` in setup
where isolation matters; otherwise the tests run against the
shared process-local state.

Honest framing
--------------
* The metadata is *declared* — ``yield_range`` is the
  literature-cited bracket, not a measurement.  Tests verify the
  *shape* of the dataclass, not the chemistry.
* The integration helpers are tested by counting records + name
  containment, not by firing rules.  The plumbing must not
  change chemistry.
"""

from __future__ import annotations

import pytest

from molmetal_lam.reactions.rule_registry import (
    RuleRecord,
    clear_registry,
    list_rules,
    record_test_result,
    register_rule,
    registry_snapshot,
    unregister_rule,
    vet_rule,
    wire_all_default_rules,
    wire_click_rules,
    wire_metal_coordination_rules,
)


# ---------------------------------------------------------------------------
# Test isolation helper.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear the registry around every test so a test that
    registers ``t30_cuaac`` cannot leak into a sibling test that
    expects the registry to be empty."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# 1. register then list returns the rule
# ---------------------------------------------------------------------------
def test_register_then_list_returns_the_rule():
    """``register_rule`` puts the rule in the registry; ``list_rules``
    returns it.  The returned record is a :class:`RuleRecord`."""
    rec = register_rule(
        "t30_demo_cuaac",
        smarts="[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[c:2][n:3][n:4][n:5]1",
        metadata={"yield_range": (0.40, 0.95),
                  "fg_compat": ["azide", "alkyne"],
                  "partner_tiles": 60},
        scope="click",
    )
    assert isinstance(rec, RuleRecord)
    assert rec.name == "t30_demo_cuaac"
    assert rec.scope == "click"
    assert rec.partner_tiles == 60
    rules = list_rules()
    assert len(rules) == 1
    assert rules[0].name == "t30_demo_cuaac"
    assert rules[0].yield_range == (0.40, 0.95)
    assert "azide" in rules[0].fg_compat
    assert "alkyne" in rules[0].fg_compat


# ---------------------------------------------------------------------------
# 2. unregister removes it
# ---------------------------------------------------------------------------
def test_unregister_removes_the_rule():
    """``unregister_rule`` returns the removed record and removes it
    from ``list_rules``.  ``vet_rule`` then raises ``KeyError``."""
    register_rule("t30_to_remove", smarts="X", scope="custom")
    assert len(list_rules()) == 1
    removed = unregister_rule("t30_to_remove")
    assert removed is not None
    assert removed.name == "t30_to_remove"
    assert list_rules() == []
    with pytest.raises(KeyError):
        vet_rule("t30_to_remove")


def test_unregister_missing_returns_none():
    """Unregistering a name that was never registered returns ``None``;
    does not raise."""
    assert unregister_rule("t30_never_existed") is None


# ---------------------------------------------------------------------------
# 3. vet_rule returns metadata even before any usage
# ---------------------------------------------------------------------------
def test_vet_rule_returns_metadata_before_any_usage():
    """Vetting a rule immediately after registration returns the
    declared metadata without forcing the caller to ``reduce`` first.
    ``last_test_result`` is the empty dict (no test record yet)."""
    register_rule(
        "t30_pre_use",
        smarts="[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]",
        metadata={"yield_range": (0.50, 0.95),
                  "fg_compat": ["leaving_group", "neutral_2e_donor"],
                  "partner_tiles": 12},
        scope="Pt",
    )
    rec = vet_rule("t30_pre_use")
    assert rec.name == "t30_pre_use"
    assert rec.smarts.startswith("[Pt:1]")
    assert rec.scope == "Pt"
    assert rec.yield_range == (0.50, 0.95)
    assert rec.partner_tiles == 12
    assert rec.last_test_result == {}
    assert rec.registered_at > 0.0


# ---------------------------------------------------------------------------
# 4. multiple scopes coexist
# ---------------------------------------------------------------------------
def test_multiple_scopes_coexist():
    """Registering rules with different scope tags keeps them all in
    the registry; ``list_rules(scope=...)`` filters by exact match."""
    register_rule("t30_cuaac", smarts="X", scope="click")
    register_rule("t30_metal_lig_exch", smarts="Y", scope="Pt")
    register_rule("t30_ru_demo", smarts="Z", scope="Ru")
    register_rule("t30_au_demo", smarts="W", scope="Au")
    register_rule("t30_custom_demo", smarts="V", scope="custom")

    # All five visible in the unfiltered list.
    all_rules = list_rules()
    names = sorted(r.name for r in all_rules)
    assert names == sorted([
        "t30_cuaac", "t30_metal_lig_exch", "t30_ru_demo",
        "t30_au_demo", "t30_custom_demo",
    ])

    # Each scope filter returns exactly one record.
    assert [r.name for r in list_rules(scope="click")] == ["t30_cuaac"]
    assert [r.name for r in list_rules(scope="Pt")]   == ["t30_metal_lig_exch"]
    assert [r.name for r in list_rules(scope="Ru")]   == ["t30_ru_demo"]
    assert [r.name for r in list_rules(scope="Au")]   == ["t30_au_demo"]
    assert [r.name for r in list_rules(scope="custom")] == ["t30_custom_demo"]

    # Unknown scope returns empty.
    assert list_rules(scope="nonexistent") == []


# ---------------------------------------------------------------------------
# 5. duplicate registration raises informative error
# ---------------------------------------------------------------------------
def test_duplicate_registration_raises_informative_error():
    """Registering the same name twice without ``overwrite=True``
    raises :class:`ValueError` whose message mentions the rule
    name and the existing record's scope + timestamp."""
    register_rule("t30_dup", smarts="X", scope="click")
    with pytest.raises(ValueError) as excinfo:
        register_rule("t30_dup", smarts="Y", scope="Pt")
    msg = str(excinfo.value)
    assert "t30_dup" in msg
    assert "already registered" in msg
    assert "overwrite=True" in msg

    # ``overwrite=True`` succeeds and replaces the record.
    new = register_rule(
        "t30_dup", smarts="Y", scope="Pt", overwrite=True,
    )
    assert new.scope == "Pt"
    assert vet_rule("t30_dup").scope == "Pt"


# ---------------------------------------------------------------------------
# 6. vet_rule returns correct yield_range from metadata
# ---------------------------------------------------------------------------
def test_vet_rule_yield_range_is_returned_verbatim():
    """``vet_rule`` returns the exact ``(low, high)`` tuple passed to
    ``register_rule``.  Cast to float on the way in; preserved
    bit-for-bit on the way out."""
    register_rule(
        "t30_yield_test",
        smarts="X",
        metadata={"yield_range": (0.37, 0.92)},
        scope="custom",
    )
    rec = vet_rule("t30_yield_test")
    assert rec.yield_range is not None
    low, high = rec.yield_range
    assert isinstance(low, float)
    assert isinstance(high, float)
    assert abs(low - 0.37) < 1e-9
    assert abs(high - 0.92) < 1e-9

    # No yield_range declared -> None, not (0.0, 0.0).
    register_rule("t30_no_yield", smarts="X", scope="custom")
    assert vet_rule("t30_no_yield").yield_range is None


# ---------------------------------------------------------------------------
# 7. record_test_result preserves other fields
# ---------------------------------------------------------------------------
def test_record_test_result_preserves_other_fields():
    """Recording a test result only changes ``last_test_result``;
    name, smarts, scope, yield_range, fg_compat, partner_tiles,
    registered_at are bit-for-bit preserved."""
    register_rule(
        "t30_test_record",
        smarts="[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]",
        metadata={"yield_range": (0.50, 0.95),
                  "fg_compat": ["leaving_group"],
                  "partner_tiles": 7},
        scope="Pt",
    )
    before = vet_rule("t30_test_record")
    record_test_result(
        "t30_test_record",
        test_id="test_demo.py::test_pass",
        outcome="PASS",
    )
    after = vet_rule("t30_test_record")
    assert after.name == before.name
    assert after.smarts == before.smarts
    assert after.scope == before.scope
    assert after.yield_range == before.yield_range
    assert after.fg_compat == before.fg_compat
    assert after.partner_tiles == before.partner_tiles
    assert after.registered_at == before.registered_at
    assert "test_demo.py::test_pass" in after.last_test_result
    assert after.last_test_result["test_demo.py::test_pass"] == "PASS"
    # Original record is still unchanged (frozen dataclass).
    assert before.last_test_result == {}

    # Recording on an unknown name raises KeyError.
    with pytest.raises(KeyError):
        record_test_result("t30_unknown", "test_x", "PASS")


# ---------------------------------------------------------------------------
# 8. integration helpers wire the canonical rules
# ---------------------------------------------------------------------------
def test_wire_click_rules_registers_5_canonical_rules():
    """``wire_click_rules`` registers the 5 canonical click rules
    (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling) under the
    ``click`` scope.  Aliases (camelCase + lowercase) are
    deduped — the registry holds 5, not 12."""
    registered = wire_click_rules()
    canonical = {"CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"}
    assert {r.name for r in registered} == canonical
    by_name = {r.name: r for r in list_rules()}
    assert set(by_name.keys()) == canonical
    for r in by_name.values():
        assert r.scope == "click"
        assert r.yield_range is not None
        low, high = r.yield_range
        assert 0.0 <= low <= high <= 1.0
        assert len(r.fg_compat) >= 1


def test_wire_metal_coordination_rules_registers_2_rules():
    """``wire_metal_coordination_rules`` registers MetalLigandExchange
    + AquaExchange under the ``Pt`` scope."""
    registered = wire_metal_coordination_rules()
    names = {r.name for r in registered}
    assert names == {"MetalLigandExchange", "AquaExchange"}
    for r in list_rules():
        assert r.scope == "Pt"


def test_wire_all_default_rules_registers_both_families():
    """``wire_all_default_rules`` returns the union of click + Pt
    rules (5 + 2 = 7 records) without overlap."""
    out = wire_all_default_rules()
    assert len(out) == 7
    scopes = {r.scope for r in out}
    assert scopes == {"click", "Pt"}


# ---------------------------------------------------------------------------
# 9. registry_snapshot is JSON-friendly
# ---------------------------------------------------------------------------
def test_registry_snapshot_is_json_friendly():
    """``registry_snapshot`` returns a flat ``{name: dict}`` mapping
    whose values are :class:`RuleRecord`-shaped dicts."""
    register_rule(
        "t30_snap",
        smarts="X",
        metadata={"yield_range": (0.1, 0.2), "partner_tiles": 3},
        scope="custom",
    )
    snap = registry_snapshot()
    assert "t30_snap" in snap
    rec = snap["t30_snap"]
    assert rec["name"] == "t30_snap"
    assert rec["scope"] == "custom"
    assert rec["yield_range"] == (0.1, 0.2)  # asdict preserves the tuple
    assert rec["partner_tiles"] == 3
    assert rec["last_test_result"] == {}
    assert rec["registered_at"] > 0.0


# ---------------------------------------------------------------------------
# 10. validation guards
# ---------------------------------------------------------------------------
def test_register_rule_rejects_empty_name():
    """``name`` must be a non-empty string."""
    with pytest.raises(ValueError):
        register_rule("", smarts="X", scope="click")
    with pytest.raises(ValueError):
        register_rule("   ", smarts="X", scope="click")


def test_register_rule_rejects_bad_yield_range():
    """``yield_range`` must be a 2-tuple of floats."""
    with pytest.raises(ValueError):
        register_rule("t30_bad_yield", smarts="X",
                       metadata={"yield_range": (0.5,)}, scope="custom")
    with pytest.raises(ValueError):
        register_rule("t30_bad_yield2", smarts="X",
                       metadata={"yield_range": "0.5"}, scope="custom")


def test_register_rule_rejects_negative_partner_tiles():
    """``partner_tiles`` must be a non-negative int."""
    with pytest.raises(ValueError):
        register_rule("t30_bad_pt", smarts="X",
                       metadata={"partner_tiles": -1}, scope="custom")
