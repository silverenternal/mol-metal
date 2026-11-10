# Licensed under the Apache License, Version 2.0.
"""Wf-A269 frontier sweep — the config-binding gate checks argparse's `choices`.

THE DEFECT CLASS (a new one, in the §6.269.17 DEAD-GATE family — a check that
cannot fail because the check does not exist)
----------------------------------------------------------------------
``argparse``'s ``choices=`` is a **DECLARED CONTRACT**: it enumerates the
values the trainer claims to accept.  MEASURED 2026-11-04 (AST census over the
32 trainers a config's ``entry.script`` names): **98** actions declare it.

But the YAML path **BYPASSES argparse entirely** — ``wfA2_config_loader``'s
``apply_yaml_overrides`` writes the value straight onto the namespace with
``setattr`` — so a config could select a value the trainer's own parser
FORBIDS and **every §5.11 gate in the repo passed**.  MEASURED 2026-11-04 on
the real CFM trainer: with ``choices=("argmax", "threshold")`` declared, a
config's ``flow.bond_decision: per_class`` reached ``args.bond_decision``
unchanged — the CLI could not express what the YAML could.

⇒ the class is not "the value is wrong"; it is **"a declared contract that no
gate checked"**.

WHAT THIS TEST DOES (it is the MECHANISM, not a restatement)
-------------------------------------------------------------
No torch, no model, no GPU: the contract lives in an ``argparse`` declaration,
and ``argparse`` is a pure standard-library object.

* a value outside the declared contract IS reported;
* a value inside it is NOT;
* the value tested is the one the loader **DELIVERS** (converted for a
  scalar), which is what argparse itself validates;
* each element of a **sequence** dest is tested, which is what argparse
  itself validates.

MUTATION-PROVABLE: neutering ``gate._choices_violation`` (**M1**) or stripping
the ``choices`` key out of the shape map the gate builds (**M2**) makes the
detection tests FAIL — so they are not a restatement of the implementation.

APPEND-ONLY (§5.5 rule #6): the gate's PRE-REPAIR rendering — it modelled
``nargs`` and ``type`` only — is preserved VERBATIM in
``scripts/_config_binding_gate.py``'s dated comments.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_config_binding_gate.py"

_spec = importlib.util.spec_from_file_location("_config_binding_gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_config_binding_gate"] = gate
_spec.loader.exec_module(gate)


class _Ns:
    """Stand-in for an argparse namespace (same idiom as the sibling test)."""

    def __init__(self, **dests):
        self.__dict__.update(dests)


def _write(tmp_path: Path, name: str, body: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return p


def _shapes(**flags) -> dict[str, dict]:
    """Build the gate's dest->shape map from real argparse declarations.

    Built from a LIVE parser (never a hand-written dict) so the fixture cannot
    drift from what ``_capture_parser`` hands ``audit_config`` — a fixture that
    invents its own shape map tests the fixture, not the gate.
    """
    ap = argparse.ArgumentParser()
    for flag, kwargs in flags.items():
        ap.add_argument(flag, **kwargs)
    return gate._actions_by_dest(ap)


def _choices_only(found: list[dict]) -> list[dict]:
    return [f for f in found if f.get("reason") == "choices"]


# --------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------


def test_a_value_outside_the_declared_contract_is_reported(tmp_path):
    cfg = _write(
        tmp_path,
        "c.yaml",
        {"flow": {"bond_decision": "per_class"}},
    )
    found, _ = gate.audit_config(
        cfg,
        _Ns(bond_decision="per_class"),
        _shapes(**{"--bond-decision": {"choices": ("argmax", "threshold")}}),
    )
    bad = _choices_only(found)
    assert len(bad) == 1, found
    assert bad[0]["key"] == "flow.bond_decision"
    assert bad[0]["dest"] == "bond_decision"
    # The reason must be its OWN value: the printer and the per-config tag
    # counter branch on it, and a "choices" violation labelled "dead" would
    # send the reader after a different defect.
    assert bad[0]["reason"] == "choices"
    assert "choices" in bad[0]["detail"]


def test_a_value_inside_the_declared_contract_is_clean(tmp_path):
    cfg = _write(tmp_path, "c.yaml", {"flow": {"bond_decision": "threshold"}})
    found, _ = gate.audit_config(
        cfg,
        _Ns(bond_decision="threshold"),
        _shapes(**{"--bond-decision": {"choices": ("argmax", "threshold")}}),
    )
    assert found == [], found


def test_a_dest_with_no_choices_is_never_flagged(tmp_path):
    """``choices=None`` means "no contract" — argparse skips the test too."""
    cfg = _write(tmp_path, "c.yaml", {"flow": {"bond_decision": "anything_at_all"}})
    found, _ = gate.audit_config(
        cfg,
        _Ns(bond_decision="anything_at_all"),
        _shapes(**{"--bond-decision": {}}),
    )
    assert found == [], found


def test_the_converted_value_is_the_one_tested(tmp_path):
    """argparse converts FIRST and tests membership SECOND.

    ``apply_yaml_overrides`` runs the action's own ``type=`` converter for a
    scalar, so the value that lands on the dest is the CONVERTED one — and
    that is what argparse would have validated.  A YAML ``"1"`` against
    ``type=int, choices=(0, 1)`` is therefore IN contract; a YAML ``2`` is not.
    """
    shapes = _shapes(**{"--flag": {"type": int, "choices": (0, 1)}})

    ok = _write(tmp_path, "ok.yaml", {"flow": {"flag": "1"}})
    found, _ = gate.audit_config(ok, _Ns(flag=1), shapes)
    assert _choices_only(found) == [], found

    bad = _write(tmp_path, "bad.yaml", {"flow": {"flag": 2}})
    found, _ = gate.audit_config(bad, _Ns(flag=2), shapes)
    assert len(_choices_only(found)) == 1, found


def test_a_sequence_dest_tests_each_element(tmp_path):
    """argparse checks membership on EACH element of a sequence nargs."""
    shapes = _shapes(**{"--arms": {"nargs": "+", "choices": ("a", "b")}})

    ok = _write(tmp_path, "ok.yaml", {"flow": {"arms": ["a", "b"]}})
    found, _ = gate.audit_config(ok, _Ns(arms=["a", "b"]), shapes)
    assert _choices_only(found) == [], found

    bad = _write(tmp_path, "bad.yaml", {"flow": {"arms": ["a", "z"]}})
    found, _ = gate.audit_config(bad, _Ns(arms=["a", "z"]), shapes)
    assert len(_choices_only(found)) == 1, found
    assert "z" in _choices_only(found)[0]["detail"]


# --------------------------------------------------------------------------
# MUTATION PROOFS — the checks above must be able to FAIL
# --------------------------------------------------------------------------


def test_m1_neutering_the_check_makes_the_detection_fail(tmp_path, monkeypatch):
    """M1: replace the check with a no-op — the detection must disappear.

    This is the proof that the tests above are load-bearing rather than a
    restatement of the implementation: a gate whose check cannot fail is the
    §6.269.17 defect class.
    """
    cfg = _write(tmp_path, "c.yaml", {"flow": {"bond_decision": "per_class"}})
    shapes = _shapes(**{"--bond-decision": {"choices": ("argmax", "threshold")}})

    assert len(
        _choices_only(gate.audit_config(cfg, _Ns(bond_decision="per_class"), shapes)[0])
    ) == 1

    monkeypatch.setattr(gate, "_choices_violation", lambda *a, **k: None)
    assert _choices_only(gate.audit_config(cfg, _Ns(bond_decision="per_class"), shapes)[0]) == []


def test_m2_stripping_the_choices_key_makes_the_detection_fail(tmp_path):
    """M2: the shape map must CARRY the contract, not the checker alone.

    Without the ``choices`` key the checker has nothing to test against — so
    the check would silently pass everything.  That is the other half of the
    same defect class: a check whose INPUT is missing.
    """
    cfg = _write(tmp_path, "c.yaml", {"flow": {"bond_decision": "per_class"}})
    shapes = _shapes(**{"--bond-decision": {"choices": ("argmax", "threshold")}})
    stripped = {
        dest: {k: v for k, v in shape.items() if k != "choices"}
        for dest, shape in shapes.items()
    }
    assert "choices" in shapes["bond_decision"]
    found, _ = gate.audit_config(cfg, _Ns(bond_decision="per_class"), stripped)
    assert _choices_only(found) == [], found


def test_m3_the_real_trainers_declare_the_contract_the_gate_reads():
    """M3: the gate reads the LIVE action, so the repo must actually declare.

    MEASURED 2026-11-04: 98 actions across the 32 trainers a config's
    ``entry.script`` names declare ``choices``.  If a future edit removed them
    all, the tier would still PASS — vacuously.  Pin the count's existence,
    not its exact value.
    """
    script = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    if not script.is_file():
        pytest.skip("trainer not on disk")
    ns, actions, err = gate._capture_parser(script)
    assert err is None, err
    declared = {d: s["choices"] for d, s in actions.items() if s.get("choices")}
    assert declared, "the CFM trainer declares no choices at all — the tier would be vacuous"
    assert "bond_decision" in declared
    # The Wf-A269 wave-20 repair widened this tuple to cover the decoder's
    # THIRD rule ("per_class"); the gate must SEE the widened tuple.
    assert "per_class" in declared["bond_decision"]


def test_the_repo_wide_audit_is_green_and_the_tier_actually_ran():
    """The repo's own configs must not violate any declared contract.

    MEASURED 2026-11-04 (this tree): 534 configs checked, verdict PASS,
    ``n_choices_violations == 0``.  The counter is asserted to EXIST as well,
    because a tier that never fires is indistinguishable from a tier that does
    not exist.
    """
    report = gate.audit()
    assert "n_choices_violations" in report
    assert report["n_choices_violations"] == 0, report["violations"]
    assert report["verdict"] == "PASS", report["violations"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
