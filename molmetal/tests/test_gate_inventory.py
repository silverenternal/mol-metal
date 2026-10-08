"""Tests for the gate-inventory audit surface (molmetal/scripts/gate_inventory.py).

Verifies that the pre-flight gate surface is auditable:
  * the inventory builds for the ``all`` pipeline (introspection only);
  * every registered gate has a non-empty name + callable_name;
  * the zombie audit returns a list (opt-in gates whose env var is unset);
  * the inventory count matches ``len(build_gates("all"))`` exactly.

CPU-only, no gate executed (the inventory introspects callables, never calls
them), so this module runs in well under a second.
"""

from __future__ import annotations


def test_inventory_builds() -> None:
    """build_inventory("all") returns one entry per registered gate."""
    from molmetal.scripts.gate_inventory import build_inventory

    inventory = build_inventory("all")
    assert isinstance(inventory, list)
    assert inventory, "gate inventory must not be empty"


def test_every_gate_has_a_name() -> None:
    """Every registered gate has a non-empty name + callable_name."""
    from molmetal.scripts.gate_inventory import build_inventory

    inventory = build_inventory("all")
    for entry in inventory:
        assert entry.get("name"), f"gate missing name: {entry!r}"
        assert entry.get("callable_name"), f"gate missing callable_name: {entry!r}"


def test_zombie_audit_returns_a_list() -> None:
    """audit_zombie_gates returns a list of opt-in-unset gates."""
    from molmetal.scripts.gate_inventory import (
        audit_zombie_gates,
        build_inventory,
    )

    zombies = audit_zombie_gates(build_inventory("all"))
    assert isinstance(zombies, list)
    for z in zombies:
        assert z["name"] and z["opt_in_env"], f"bad zombie entry: {z!r}"


def test_inventory_count_matches_build_gates() -> None:
    """The inventory length equals len(build_gates("all"))."""
    from molmetal.scripts.gate_inventory import build_inventory
    from molmetal.scripts.pre_flight_check import build_gates

    assert len(build_inventory("all")) == len(build_gates("all"))


def test_gate_registered_in_all_pipeline() -> None:
    """p0_gate_inventory_audit is registered in build_gates("all")."""
    from molmetal.scripts.pre_flight_check import build_gates

    names = [name for name, _ in build_gates("all")]
    assert "p0_gate_inventory_audit" in names


def test_gate_inventory_audit_passes() -> None:
    """The report-only gate always PASSes and reports the zombie count."""
    from molmetal.scripts.pre_flight_check import gate_gate_inventory_audit

    result = gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all")
    assert result.status == "PASS"
    assert "zombie_gates=" in result.evidence
