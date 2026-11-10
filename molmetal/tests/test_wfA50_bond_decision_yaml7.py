"""Wf-A50 §6.55g — Exercise the bond-decision wire-in across shipped CFM YAMLs.

The Wf-A35-1 wire-in lives at
``molmetal/eval/bond_decision.py:97`` and is read by the trainer
``train_cfm_yuelbond_wasserstein.py:10088-10118``.  Per §6.55g
MEASURED rule, a wire-in is only MEASURED-end-to-end when a shipped
YAML sets the flag to a non-default value AND that value reaches
``bond_decision_settings`` unchanged.

This test loops over ALL shipped CFM YAMLs (deterministic
order via :func:`glob`), loads each via the canonical

.. note::

   2026-11-04 (Wf-A269 wave-20) — the sentence below is the PRE-REPAIR
   rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6
   (append-only, dated, never a silent edit):

       This test loops over the first 7 shipped CFM YAMLs (deterministic
       order via :func:`glob`), loads each via the canonical

   The population was a POSITIONAL slice (``sorted(glob(...))[:7]``),
   which covered 7 of 80 shipped CFM configs (8.9 %) and could not
   detect a config swapping in and out of the slice.  MEASURED: all 80
   parse cleanly through the same path, so the cap hid nothing and
   cost nothing to remove (0.86 s for the whole population).
:func:`load_config` helper, applies them through
:func:`apply_yaml_overrides` against the real trainer's
``ArgumentParser``, and asserts the
``bond_decision_settings(state)`` triplet matches the YAML-declared
values for both ``bond_decision`` and ``bond_decision_threshold``.

What this test pins
-------------------

* ``flow.bond_decision`` ("argmax" | "threshold") lands in
  ``args.bond_decision``.

  2026-11-04 (Wf-A269 wave-20) — the bullet above is the PRE-REPAIR
  rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6; it is
  now INCOMPLETE, because the trainer declares a THIRD rule
  ("per_class", the Wf-A80 per-class calibrated thresholds) and TWO
  shipped configs select it.  The assertion itself is NOT this
  docstring's: it is derived from the trainer's own live declaration by
  ``molmetal/tests/test_wfa269_bond_decision_contract.py``.
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
    """ALL shipped CFM YAMLs (deterministic via sorted).

    WHY IT WAS REPAIRED (all MEASURED read-only, CPU-only, 2026-11-04):

    * COVERAGE:  ``configs/cfm_*.yaml`` = **80** files; the population
      after the exclusion = **79**; the slice covered **7** ⇒ **8.9 %**,
      leaving **72** shipped CFM configs unparsed by this guard.
    * THE SLICE WAS POSITIONAL:  ``[:7]`` is an alphabetical cut, so a
      new ``cfm_*.yaml`` sorting before the current 7th would silently
      enter the guard and push the current 7th out — while
      ``assert len(...) == 7`` still passed.
    * THE EXCLUSION HID NOTHING:  the excluded config
      (``cfm_5metal_default_log_normal_2026_10_16.yaml``) parses
      cleanly through the same path, so "for clarity" was not a reason.
    * THE COST OF THE FULL POPULATION IS NIL:  80 configs parse in
      **0.86 s** (11 ms/config).

    The whole live population is therefore returned, and
    :func:`test_guard_population_is_the_whole_population` pins the
    property (with an anti-vacuity proof) so the cap cannot come back.
    """
    # 2026-11-04 (Wf-A269 wave-20) — the block below is the PRE-REPAIR
    # rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6
    # (append-only, dated, never a silent edit):
    #
    #     # The Wf-A50 §6.55g test surface (this file itself is in configs/)
    #     # MUST NOT be in the loop — it is one of the 7 but the test loads
    #     # it via trainer.parse_args below; including it in the iteration
    #     # is harmless but we exclude it for clarity.
    #     yamls = [
    #         p
    #         for p in yamls
    #         if p.name != "cfm_5metal_default_log_normal_2026_10_16.yaml"
    #     ]
    #     return yamls[:7]
    #
    # with the PRE-REPAIR docstring:
    #
    #     """First 7 shipped CFM YAMLs (deterministic via sorted)."""
    return sorted(CONFIGS_DIR.glob("cfm_*.yaml"))


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
    # 2026-11-04 (Wf-A269 wave-20) — the block below is the PRE-REPAIR
    # rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6:
    #
    #     assert len(shipped_cfm_yamls) == 7, (
    #         f"expected 7 shipped CFM YAMLs in configs/cfm_*.yaml; got "
    #         f"{len(shipped_cfm_yamls)} — the test is stale if this number "
    #         f"changes; update the fixture + the brief accordingly"
    #     )
    #
    # That pin was VACUOUS: it pinned the *truncated* count, so the very
    # truncation it should have reported satisfied it.  It is replaced by
    # a RE-DERIVED quantity — the fixture must BE the live population.
    _live_population = sorted(CONFIGS_DIR.glob("cfm_*.yaml"))
    assert list(shipped_cfm_yamls) == list(_live_population), (
        f"this guard covers {len(shipped_cfm_yamls)} of "
        f"{len(_live_population)} configs/cfm_*.yaml — the population must "
        f"be the WHOLE live set (no cap, no exclusion); the missing ones "
        f"are: "
        f"{sorted(p.name for p in set(_live_population) - set(shipped_cfm_yamls))}"
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
        # 2026-11-04 (Wf-A269 wave-20) — the block below is the PRE-REPAIR
        # rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6
        # (append-only, dated, never a silent edit):
        #
        #     # Decision MUST be a string in {"argmax", "threshold"} — anything
        #     # else would be a YAML typo that survives into training.
        #     assert ns.bond_decision in {"argmax", "threshold"}, (
        #         f"{yaml_path.name}: bond_decision={ns.bond_decision!r} is "
        #         f"outside the canonical {{argmax, threshold}} set; the "
        #         f"YuelBondBondDecoder would refuse this at construction"
        #     )
        #
        # That pin was STALE on two counts, both MEASURED read-only:
        #
        #  (1) THE SET.  The trainer declares a THIRD rule, "per_class"
        #      (the Wf-A80 per-class calibrated thresholds):
        #          molmetal/scripts/train_cfm_yuelbond_wasserstein.py
        #              choices=("argmax", "threshold", "per_class")
        #      and TWO shipped configs select it
        #      (configs/cfm_5metal_wfa80_per_class_cpu_smoke.yaml,
        #       configs/cfm_5metal_wfa80_per_class_decision.yaml).  A
        #      hard-coded pair is a FROZEN PIN that goes stale the moment
        #      the vocabulary grows — the class this repo is hunting
        #      (docs/HONEST_NEGATIVE_CATALOG.md §6.269.17).
        #
        #  (2) THE REASON.  "the YuelBondBondDecoder would refuse this at
        #      construction" was FALSE: the decoder IMPLEMENTS per_class
        #      and accepts it —
        #          molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py:742
        #              if decision == "per_class":
        #      and its own error message enumerates all three rules
        #      (":783-784  expected 'argmax', 'threshold' or 'per_class'").
        #
        # The replacement set is DERIVED from the trainer's own live
        # declaration (never written down here), so this file cannot drift
        # from the trainer again; the trainer-vs-decoder divergence itself
        # is enforced by molmetal/tests/test_wfa269_bond_decision_contract.py.
        from molmetal.tests.test_wfa269_bond_decision_contract import (
            _bond_decision_choices,
        )

        _declared_bond_decision_choices = set(_bond_decision_choices())
        assert ns.bond_decision in _declared_bond_decision_choices, (
            f"{yaml_path.name}: bond_decision={ns.bond_decision!r} is "
            f"outside the trainer's declared set "
            f"{sorted(_declared_bond_decision_choices)} — the "
            f"YuelBondBondDecoder refuses any rule outside "
            f"{{argmax, threshold, per_class}} "
            f"(yuelbond_decoder.py:783-784)"
        )


# ---------------------------------------------------------------------------
# Test 2: bond_decision_settings matches the YAML-declared triplet
# ---------------------------------------------------------------------------
# 2026-11-04 (Wf-A269 wave-20) — the line below is the PRE-REPAIR
# rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6:
#
#     @pytest.mark.parametrize("yaml_idx", list(range(7)))
#
# `range(7)` was the SAME cap as the fixture's `[:7]`: it hard-coded the
# population size instead of deriving it, so a config that sorted into
# the first 7 was tested and a config that sorted out of it was not.
# The parametrisation is now over the WHOLE live population, and the
# parametrised ids carry the config's own name so a failure names it.
@pytest.mark.parametrize(
    "yaml_path",
    sorted(CONFIGS_DIR.glob("cfm_*.yaml")),
    ids=lambda p: p.name,
)
def test_bond_decision_yaml7(trainer_namespace_factory, yaml_path):
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

    # 2026-11-04 (Wf-A269 wave-20) — the line below is the PRE-REPAIR
    # rendering and is preserved VERBATIM per AGENTS.md §5.5 rule #6:
    #
    #     yaml_path = shipped_cfm_yamls[yaml_idx]
    #
    # The parameter is now the config path itself (the parametrisation
    # covers the whole live population), so the index lookup is gone.
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


# ---------------------------------------------------------------------------
# Test 4 (Wf-A269 wave-20): the guard's POPULATION is the whole population
# ---------------------------------------------------------------------------
def _population_is_complete(selected, live) -> bool:
    """The coverage predicate.

    True iff ``selected`` IS the whole live population — no cap, no
    exclusion, no re-ordering.  Kept as a separate function so the
    anti-vacuity check below can MUTATE it against a broken population.
    """
    return list(selected) == list(live)


def test_guard_population_is_the_whole_population(shipped_cfm_yamls):
    """The §6.55g bond-decision wire-in must be exercised against EVERY
    shipped CFM config, not a positional slice of them.

    MEASURED 2026-11-04: the pre-repair fixture covered 7 of 80
    ``configs/cfm_*.yaml`` (**8.9 %**), by an alphabetical ``[:7]`` cut
    whose length was then pinned as ``== 7`` — a pin that was satisfied
    by the truncation it should have reported, and that could not detect
    a config swapping into or out of the slice.
    """
    live = sorted(CONFIGS_DIR.glob("cfm_*.yaml"))
    assert live, (
        "no configs/cfm_*.yaml found — the guard would vacuously pass"
    )
    assert _population_is_complete(shipped_cfm_yamls, live), (
        f"this guard covers {len(shipped_cfm_yamls)} of {len(live)} "
        f"configs/cfm_*.yaml ({len(shipped_cfm_yamls)/len(live)*100:.1f} %) "
        f"— a §6.55g wire-in that is not exercised on every shipped "
        f"config is a wire-in that can silently break on the ones it "
        f"does not cover"
    )

    # ANTI-VACUITY: the predicate must REJECT the two broken populations
    # this guard was written against.  Without this, the assertion above
    # could be satisfied by a predicate that always returns True.
    assert not _population_is_complete(live[:7], live), (
        "the coverage predicate accepts a positional slice — it is "
        "vacuous"
    )
    assert not _population_is_complete(
        [p for p in live if p.name != "cfm_5metal_default_log_normal_2026_10_16.yaml"],
        live,
    ), (
        "the coverage predicate accepts an excluded population — it is "
        "vacuous"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
