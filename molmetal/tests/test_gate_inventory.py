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
    """The audit gate PASSes on the declared state and reports the count.

    (2026-11-04, Wf-A269 dead-gates sweep — the original docstring said
    "The report-only gate always PASSes".  That is SUPERSEDED but preserved
    here for the record: the gate is now a P0 BLOCKER that FAILs on an
    UNDECLARED opt-in gate.  On the DECLARED state it still PASSes, and the
    assertion below still holds.)
    """
    from molmetal.scripts.pre_flight_check import gate_gate_inventory_audit

    result = gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all")
    assert result.status == "PASS"
    assert "zombie_gates=" in result.evidence


# ---------------------------------------------------------------------------
# Wf-A269 dead-gates sweep (2026-11-04) — the two repairs, each with an
# in-test MUTATION PROOF (AGENTS.md §5.5 rule #6 discipline: a repair that
# adds a test must mutate the thing it guards and show the test catches the
# mutation).
# ---------------------------------------------------------------------------

#: The five opt-in env vars the audit must see as UNSET for a gate to be a
#: zombie.  Cleared from the ambient environment so a shell that happens to
#: export one of them cannot make the result non-deterministic.
_OPT_IN_ENV_VARS = (
    "WF43_M16_PT_CKPT",
    "WF43_M1_PAPER_COHORT",
    "WF43_M2_COHORT_INDEPENDENCE",
    "WF43_M7_SUBPOCKET_UNIFORM",
    "WF43_M19_LOMO_HEADLINE",
)


def _clear_opt_in_env(monkeypatch) -> None:
    for var in _OPT_IN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_delegate_hop_reaches_the_opt_in_read(monkeypatch) -> None:
    """The thin-wrapper -> delegate hop is what makes the opt-in scan work.

    MEASURED on this working tree: the pre-flight registers 133 gates; 5 of
    them read their opt-in env var ONE HOP DOWN in a delegated module
    (``gate_wf_r43_*`` in ``pre_flight_check.py`` imports its implementation
    from ``molmetal/scripts/wf_r43a*_*.py``).  Before the repair the scan
    read only the wrapper and found 0.
    """
    from molmetal.scripts import gate_inventory

    expected = {
        "p3_wf_r43_m16_pt_ckpt_gap": "WF43_M16_PT_CKPT",
        "p3_wf_r43_m1_paper_cohort": "WF43_M1_PAPER_COHORT",
        "p3_wf_r43_m2_cohort_independence": "WF43_M2_COHORT_INDEPENDENCE",
        "p3_wf_r43_m7_subpocket_uniform": "WF43_M7_SUBPOCKET_UNIFORM",
        "p3_wf_r43_m19_lomo_headline": "WF43_M19_LOMO_HEADLINE",
    }
    inv = gate_inventory.build_inventory("all")
    optin = {e["name"]: e["opt_in_env"] for e in inv if e.get("opt_in_env")}
    assert optin == expected, f"opt-in detection changed: {sorted(optin)}"
    for entry in inv:
        if entry.get("opt_in_env"):
            assert str(entry.get("opt_in_env_source") or "").startswith(
                "delegate:"
            ), f"opt-in gate not found via the delegate hop: {entry!r}"

    # ---- MUTATION (pre-repair behaviour: the delegate hop removed) ----
    monkeypatch.setattr(gate_inventory, "_delegate_functions", lambda src: [])
    mutated = [
        e for e in gate_inventory.build_inventory("all") if e.get("opt_in_env")
    ]
    assert mutated == [], (
        "MUTATION NOT CAUGHT: removing the delegate hop still finds opt-in gates"
    )


def test_declared_baseline_matches_the_detected_zombies() -> None:
    """The DECLARED_OPT_IN_GATES table is EXACTLY the detected zombie set.

    Deterministic: the audit is run against an EMPTY environment so a
    WF43_* var set in the ambient shell cannot change the answer.
    """
    from molmetal.scripts.gate_inventory import (
        DECLARED_OPT_IN_GATES,
        audit_zombie_gates,
        build_inventory,
        undisclosed_zombies,
    )

    inv = build_inventory("all")
    zombies = audit_zombie_gates(inv, env={})
    names = {z["name"] for z in zombies}
    assert names == set(DECLARED_OPT_IN_GATES), (
        "the declared table has drifted from the live detection: "
        f"detected={sorted(names)} declared={sorted(DECLARED_OPT_IN_GATES)}"
    )
    assert undisclosed_zombies(inv, env={}) == []
    for gate_name, reason in DECLARED_OPT_IN_GATES.items():
        assert reason.strip(), f"a declared opt-in gate has no reason: {gate_name}"


def test_audit_gate_fails_on_an_undeclared_zombie(monkeypatch) -> None:
    """An opt-in gate that is NOT declared must make the P0 gate FAIL."""
    from molmetal.scripts import gate_inventory
    from molmetal.scripts.pre_flight_check import gate_gate_inventory_audit

    _clear_opt_in_env(monkeypatch)
    # (1) the DECLARED state PASSes
    assert (
        gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all").status
        == "PASS"
    )

    # (2) MUTATION: drop one declaration == "a gate was registered and never
    #     enforced, and nobody said so".
    monkeypatch.delitem(
        gate_inventory.DECLARED_OPT_IN_GATES, "p3_wf_r43_m16_pt_ckpt_gap"
    )
    result = gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all")
    assert result.status == "FAIL", "the undeclared zombie did not FAIL the gate"
    assert "p3_wf_r43_m16_pt_ckpt_gap" in result.evidence
    assert "undisclosed_zombie_gates=1" in result.evidence

    # (3) MUTATION PROOF: with the PRE-REPAIR semantics restored (the
    #     blocking set is empty because "the zombie count is report-only"),
    #     the very same undeclared zombie PASSes -> the assertion in (2)
    #     would NOT hold.  This is what the repair changed.
    monkeypatch.setattr(
        gate_inventory, "undisclosed_zombies", lambda *a, **k: []
    )
    reverted = gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all")
    assert reverted.status == "PASS", (
        "the pre-repair semantics no longer hide the finding — the "
        "mutation proof needs updating"
    )


def test_audit_gate_fails_when_the_inventory_build_raises(monkeypatch) -> None:
    """A broken audit cannot certify the gate surface (fail-closed)."""
    from molmetal.scripts import gate_inventory
    from molmetal.scripts.pre_flight_check import gate_gate_inventory_audit

    def boom(*_a, **_k):
        raise RuntimeError("inventory build exploded")

    monkeypatch.setattr(gate_inventory, "build_inventory", boom)
    result = gate_gate_inventory_audit("p0_gate_inventory_audit", "P0", "all")
    assert result.status == "FAIL"
    assert "inventory build error" in result.evidence


def test_cli_exit_code_reflects_the_audit(monkeypatch, tmp_path) -> None:
    """The CLI exit code is no longer unconditional 0."""
    from molmetal.scripts import gate_inventory

    _clear_opt_in_env(monkeypatch)
    out = tmp_path / "inv.json"
    monkeypatch.delitem(
        gate_inventory.DECLARED_OPT_IN_GATES, "p3_wf_r43_m19_lomo_headline"
    )
    rc = gate_inventory.main(["--json-out", str(out)])
    assert rc == 1, "the CLI did not report the undisclosed zombie"
    assert out.is_file(), "the CLI did not honour --json-out"


def test_zombie_audit_is_static_not_executing_gates() -> None:
    """The audit must NOT run gates — an earlier revision executed all 121,
    costing ~27s per pre-flight (and ~150s inside the audit gate). Static
    detection must stay under 2s."""
    import time

    from molmetal.scripts.gate_inventory import audit_zombie_gates, build_inventory

    inv = build_inventory("all")
    t0 = time.perf_counter()
    audit_zombie_gates(inv)
    assert time.perf_counter() - t0 < 2.0, "zombie audit is executing gates again"


def test_opt_in_shape_distinguishes_opt_in_from_downgrade() -> None:
    """The static shape test flags `not in` (default-SKIP = opt-in) but not
    `in` (default-enforce = downgrade hatch). Guards the fix for the false
    positives p0_csv_hash_verify / p0_nci60_split_no_leak."""
    from molmetal.scripts.gate_inventory import _OPT_IN_SHAPE_RE

    opt_in = (
        '    if os.environ.get("MY_GATE", "").strip() not in ("1", "true"):\n'
        '        return SKIP\n'
    )
    downgrade = (
        '    if os.environ.get("MY_HATCH", "").strip() in ("0", "false"):\n'
        '        return SKIP\n'
    )
    assert _OPT_IN_SHAPE_RE.search(opt_in)
    assert not _OPT_IN_SHAPE_RE.search(downgrade)
