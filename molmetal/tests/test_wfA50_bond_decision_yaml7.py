"""Wf-A50 §6.55g — Exercise the bond-decision wire-in across shipped CFM YAMLs.

The Wf-A35-1 wire-in lives at
``molmetal/eval/bond_decision.py:97`` and is read by the trainer
``train_cfm_yuelbond_wasserstein.py:10088-10118``.  Per §6.55g
MEASURED rule, a wire-in is only MEASURED-end-to-end when a shipped
YAML sets the flag to a non-default value AND that value reaches
``bond_decision_settings`` unchanged.

This test loops over the first 7 shipped CFM YAMLs (deterministic
order via :func:`glob`), loads each via the canonical
:func:`load_config` helper, applies them through
:func:`apply_yaml_overrides` against the real trainer's
``ArgumentParser``, and asserts the
``bond_decision_settings(state)`` triplet matches the YAML-declared
values for both ``bond_decision`` and ``bond_decision_threshold``.

What this test pins
-------------------

* ``flow.bond_decision`` ("argmax" | "threshold") lands in
  ``args.bond_decision``.
* ``flow.bond_decision_threshold`` lands in
  ``args.bond_decision_threshold`` as a ``float`` (per §5.11 type
  check — see AGENTS.md memory note "LIST-valued flags must be
  YAML lists").
* The helper's view of the namespace matches what the decoder
  would see at construction time.

It does NOT check ``class_logit_bias`` for every YAML — only the
shipped yamls that declare ``bond_aromatic_logit_bias`` carry one,
and the helper returns ``None`` for the rest.  Per-shipment checks
of ``class_logit_bias`` are out of scope for the §6.55g smoke.

Determinism
-----------

The YAML iteration order is the order :func:`sorted` produces on the
globbed paths, so a green run is reproducible across machines.  The
test does NOT introduce an `argparse` option-parsing dependency —
it uses the trainer's own ``parse_args`` so any future rename of
``--bond-decision`` / ``--bond-decision-threshold`` will be caught
here when the namespace no longer carries the dest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Fixtures — 7 shipped CFM YAMLs in deterministic order
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shipped_cfm_yamls() -> list[Path]:
    """First 7 shipped CFM YAMLs (deterministic via sorted)."""
    yamls = sorted(CONFIGS_DIR.glob("cfm_*.yaml"))
    # The Wf-A50 §6.55g test surface (this file itself is in configs/)
    # MUST NOT be in the loop — it is one of the 7 but the test loads
    # it via trainer.parse_args below; including it in the iteration
    # is harmless but we exclude it for clarity.
    yamls = [
        p
        for p in yamls
        if p.name != "cfm_5metal_default_log_normal_2026_10_16.yaml"
    ]
    return yamls[:7]


@pytest.fixture(scope="module")
def trainer_namespace_factory():
    """A factory that parses an argv against the trainer's full parser
    and returns the post-YAML-override namespace.

    Cached at module scope so each YAML is parsed once across the
    suite.  ``parse_args()`` consumes ``sys.argv`` so we save/restore
    the caller's argv around every invocation.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args
    import argparse

    saved_argv = list(sys.argv)

    def _factory(yaml_path: Path) -> argparse.Namespace:
        try:
            sys.argv = ["trainer", "--config", str(yaml_path)]
            return parse_args()
        finally:
            sys.argv = saved_argv

    return _factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yaml_declares_bond_decision(yaml_path: Path) -> dict:
    """Return the raw YAML leaf pair ``{decision, threshold}`` IF the
    yaml declares them, else ``None``.  Used to assert the
    round-trip preserved the YAML-declared value verbatim.
    """
    from molmetal.scripts.wfA2_config_loader import load_config

    flat = load_config(str(yaml_path))
    if "flow_bond_decision" not in flat and "flow_bond_decision_threshold" not in flat:
        return None
    return {
        "decision": flat.get("flow_bond_decision"),
        "threshold": flat.get("flow_bond_decision_threshold"),
    }


# ---------------------------------------------------------------------------
# Test 1: 7 shipped CFM YAMLs loadable end-to-end
# ---------------------------------------------------------------------------
def test_yaml7_loadable(shipped_cfm_yamls, trainer_namespace_factory):
    """All 7 shipped CFM YAMLs parse cleanly via the trainer's
    full ``parse_args`` + ``apply_yaml_overrides`` path.

    A parse failure here would mean the YAML is broken at the §5.11
    surface (e.g. a key that fails argparse coercion, a missing
    entry-point field).  We assert the namespace exists and is
    non-empty, but DO NOT assert any specific bond-decision value —
    most shipped yamls do not declare ``flow.bond_decision`` and
    should fall back to the canonical defaults.
    """
    assert len(shipped_cfm_yamls) == 7, (
        f"expected 7 shipped CFM YAMLs in configs/cfm_*.yaml; got "
        f"{len(shipped_cfm_yamls)} — the test is stale if this number "
        f"changes; update the fixture + the brief accordingly"
    )
    for yaml_path in shipped_cfm_yamls:
        ns = trainer_namespace_factory(yaml_path)
        # Defaults are present even on yamls that don't declare them.
        assert hasattr(ns, "bond_decision"), (
            f"{yaml_path.name}: trainer parse_args did not produce "
            f"a `bond_decision` dest — the YAML train-step loop is "
            f"broken for this config"
        )
        assert hasattr(ns, "bond_decision_threshold"), (
            f"{yaml_path.name}: trainer parse_args did not produce "
            f"a `bond_decision_threshold` dest"
        )
        # Threshold MUST be a float (§5.11 round-trip type check).
        assert isinstance(ns.bond_decision_threshold, float), (
            f"{yaml_path.name}: bond_decision_threshold is "
            f"{type(ns.bond_decision_threshold).__name__}, expected float — "
            f"YAML coercion failed"
        )
        # Decision MUST be a string in {"argmax", "threshold"} — anything
        # else would be a YAML typo that survives into training.
        assert ns.bond_decision in {"argmax", "threshold"}, (
            f"{yaml_path.name}: bond_decision={ns.bond_decision!r} is "
            f"outside the canonical {{argmax, threshold}} set; the "
            f"YuelBondBondDecoder would refuse this at construction"
        )


# ---------------------------------------------------------------------------
# Test 2: bond_decision_settings matches the YAML-declared triplet
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaml_idx", list(range(7)))
def test_bond_decision_yaml7(shipped_cfm_yamls, trainer_namespace_factory, yaml_idx):
    """For each shipped YAML that declares ``flow.bond_decision`` /
    ``flow.bond_decision_threshold``, the helper
    :func:`bond_decision_settings` returns the YAML-declared values.

    This is the §6.55g MEASURED rule end-to-end: the YAML says
    ``bond_decision: threshold + threshold: 0.3``, the trainer's
    argparse sees the YAML, and the helper the decoder consults at
    construction time returns exactly that pair.

    For yamls that do NOT declare the triplet, the helper returns
    the canonical defaults ``("argmax", 0.5, None)`` — also
    asserted here so a future YAML drift is caught.
    """
    from molmetal.eval.bond_decision import (
        DEFAULT_DECISION,
        DEFAULT_THRESHOLD,
        bond_decision_settings,
    )

    yaml_path = shipped_cfm_yamls[yaml_idx]
    declared = _yaml_declares_bond_decision(yaml_path)
    ns = trainer_namespace_factory(yaml_path)

    decision, threshold, bias = bond_decision_settings(vars(ns))

    if declared is None:
        # YAML did NOT declare the triplet → helper returns canonical
        # defaults.  This is the most common case in shipped yamls.
        assert decision == DEFAULT_DECISION, (
            f"{yaml_path.name}: YAML did not declare flow.bond_decision, "
            f"but helper returned decision={decision!r} (expected "
            f"canonical default {DEFAULT_DECISION!r})"
        )
        assert threshold == DEFAULT_THRESHOLD, (
            f"{yaml_path.name}: YAML did not declare "
            f"flow.bond_decision_threshold, but helper returned "
            f"threshold={threshold} (expected canonical default "
            f"{DEFAULT_THRESHOLD})"
        )
        return

    # YAML DID declare the triplet → helper MUST return those values.
    expected_decision = declared["decision"]
    expected_threshold = declared["threshold"]
    # Some yamls declare threshold as a non-default float (e.g. 0.5).
    # The canonical helper cast is `float(...)`, so the equality is
    # exact — no atol.
    assert decision == expected_decision, (
        f"{yaml_path.name}: YAML declared flow.bond_decision="
        f"{expected_decision!r}, but bond_decision_settings returned "
        f"decision={decision!r} — the §6.55g wire-in is broken for this YAML"
    )
    assert threshold == expected_threshold, (
        f"{yaml_path.name}: YAML declared flow.bond_decision_threshold="
        f"{expected_threshold}, but bond_decision_settings returned "
        f"threshold={threshold} — the §6.55g wire-in is broken for this YAML"
    )


# ---------------------------------------------------------------------------
# Test 3: the new §6.55g config's triplet is non-default
# ---------------------------------------------------------------------------
def test_new_config_declares_non_default_triplet():
    """The Wf-A50 §6.55g config
    ``configs/cfm_5metal_default_log_normal_2026_10_16.yaml`` MUST
    declare a NON-DEFAULT triplet, otherwise the §6.55g MEASURED
    rule is not satisfied (the wire-in would still be
    DEFAULT-pass-through).

    The config declares ``flow.bond_decision: threshold`` and
    ``flow.bond_decision_threshold: 0.3`` — both distinct from the
    canonical defaults ``("argmax", 0.5, None)``.
    """
    from molmetal.eval.bond_decision import (
        DEFAULT_DECISION,
        DEFAULT_THRESHOLD,
        bond_decision_settings,
    )

    yaml_path = (
        CONFIGS_DIR / "cfm_5metal_default_log_normal_2026_10_16.yaml"
    )
    assert yaml_path.is_file(), (
        f"Wf-A50 §6.55g config {yaml_path} not found on disk — "
        f"the W1 deliverable was not actually shipped"
    )
    declared = _yaml_declares_bond_decision(yaml_path)
    assert declared is not None, (
        f"{yaml_path.name}: does not declare flow.bond_decision/"
        f"flow.bond_decision_threshold — §6.55g MEASURED rule "
        f"NOT satisfied (wire-in would be DEFAULT-pass-through only)"
    )
    assert declared["decision"] != DEFAULT_DECISION, (
        f"{yaml_path.name}: flow.bond_decision={declared['decision']!r} "
        f"equals the canonical default {DEFAULT_DECISION!r}; "
        f"the §6.55g test surface must set a NON-DEFAULT value"
    )
    assert declared["threshold"] != DEFAULT_THRESHOLD, (
        f"{yaml_path.name}: flow.bond_decision_threshold="
        f"{declared['threshold']} equals the canonical default "
        f"{DEFAULT_THRESHOLD}; the §6.55g test surface must set a "
        f"NON-DEFAULT value"
    )

    # End-to-end through the trainer's parse_args + helper:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["trainer", "--config", str(yaml_path)]
        ns = parse_args()
    finally:
        sys.argv = saved_argv

    decision, threshold, _bias = bond_decision_settings(vars(ns))
    assert decision == declared["decision"]
    assert threshold == declared["threshold"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
