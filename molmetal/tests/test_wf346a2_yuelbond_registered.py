"""Wf-346-A2 — yuelbond_decoder gate registration verification (2 NEW tests).

================================================================
Per the 2026-09-29 ultracode user directive ("基于刚才的洞察，可以启动
ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"),
Wf-346-A2 audits the registration of
``p3_wf343c4_yuelbond_decoder_env`` in the ``build_gates('all')``
dispatcher (Wf-343C4 pre-retrain internal check; Wf-345A2 env-aware SKIP
contract renaming ``_registered`` → ``_env``; TODO/completed/343c4_integration.md
+ TODO/completed/345a2_yuelbond_decoder_env.md).

This file is the verification surface: 2 tests confirming (a) the gate
is registered in the 'all' pipeline only (NOT 'frontier' /
'chemprop_v2_v3' / 'cfm'; pre-retrain internal checks are not
paper-grade), and (b) ``build_gates('all')`` matches
``EXPECTED_GATE_COUNTS['all']`` (the Wf-321D equality invariant; the
literal count drifts as gates are added) with the anchor-freshness
sibling launch-only per Wf-A16.

Tests
-----

1. ``test_wf343c4_yuelbond_decoder_gate_registered_in_dispatcher`` —
   asserts the gate string ``p3_wf343c4_yuelbond_decoder_env`` appears
   in :func:`molmetal.scripts.pre_flight_check.build_gates` for pipeline
   ``'all'``. Also asserts it does NOT appear in ``'frontier'`` (the
   pre-retrain internal check contract: only in the 'all' aggregate).
   The Wf-345A2 renaming from ``_registered`` → ``_env`` reflects the
   gate's primary concern (env-aware SKIP on torch-missing host); the
   dispatcher string literal at line 7911 is canonical.

2. ``test_build_gates_all_count_invariant_with_wf343c4_gates`` —
   asserts ``len(build_gates('all')) == EXPECTED_GATE_COUNTS['all']``
   AND that the 3 commit-safe Wf-343C4 gates are present in the 'all'
   pipeline (hyperparams + yuelbond + wave_budget), while the
   anchor_freshness gate is launch-only (Wf-A16 split). This locks the
   Wf-321D count invariant so future registration drift (a gate
   silently lost) cannot regress the audit count.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (registry membership +
  Wf-321D invariant).
* No GPU / no pytest fixtures — pure introspection.
* No Edit to ``pre_flight_check.py`` is needed: per
  ``TODO/completed/343c4_integration.md``, all 4 Wf-343C4 gates
  (including ``p3_wf343c4_yuelbond_decoder_env``) were registered at
  the Wf-343C4 ship; the Wf-346-A2 task verifies this MEASURED state,
  not a hypothetical drift. The Wf-345A2 renaming
  (``_registered`` → ``_env``) is the canonical env-aware SKIP contract
  per TODO/completed/345a2_yuelbond_decoder_env.md; the dispatcher
  string ``p3_wf343c4_yuelbond_decoder_env`` is what the runner emits
  in the pre-flight JSON.
* The gate runner is the Wf-272I / Wf-269C structural pre-retrain
  check (verifies ``cfm_decoder_yuelbond.py`` exports
  ``YuelBondDecoder``; closes the Wf-291D ``decode_ratio=0`` regression
  class); the gate's PASS / SKIP / FAIL contract is asserted in the
  Wf-343C4 integration suite
  (``molmetal/tests/test_wf343c4_integration.py::test_gate_wf343c4_yuelbond_decoder_env_on_canonical``);
  this file is the registration-only slice.
"""

from __future__ import annotations

import pytest


def _build_gates_all():
    """Import build_gates lazily to avoid importing gfx1101 env at module load."""
    from molmetal.scripts.pre_flight_check import build_gates

    return build_gates("all")


def _build_gates_frontier():
    """Import build_gates lazily; returns frontier pipeline runners."""
    from molmetal.scripts.pre_flight_check import build_gates

    return build_gates("frontier")


def test_wf343c4_yuelbond_decoder_gate_registered_in_dispatcher():
    """Wf-346-A2 — yuelbond gate IS in build_gates('all') dispatcher.

    MEASURED verification: per ``TODO/completed/343c4_integration.md`` +
    ``TODO/completed/345a2_yuelbond_decoder_env.md``, the 4 Wf-343C4
    gates were registered as a batch at the Wf-343C4 ship; the
    Wf-345A2 renaming to ``_env`` reflects the env-aware SKIP contract
    on torch-missing host. This test pins the registration of the
    yuelbond gate (the 2nd of the 4) so any future drift (gate silently
    removed) fails fast here, NOT in production pre-flight runs.
    """
    runners = _build_gates_all()
    gate_names = [name for name, _ in runners]
    assert "p3_wf343c4_yuelbond_decoder_env" in gate_names, (
        f"Wf-346-A2: p3_wf343c4_yuelbond_decoder_env NOT registered in "
        f"build_gates('all'); expected to be present per Wf-343C4 ship "
        f"(renamed from _registered by Wf-345A2 env-aware SKIP contract). "
        f"All gate names: {gate_names}"
    )

    # The pre-retrain internal check contract (Wf-343C4 §3.4): the
    # 4 NEW gates are 'all'-only, NOT in 'frontier' / 'chemprop_v2_v3' /
    # 'cfm'. Pre-retrain internal checks are not paper-grade claims.
    frontier_runners = _build_gates_frontier()
    frontier_names = [name for name, _ in frontier_runners]
    assert "p3_wf343c4_yuelbond_decoder_env" not in frontier_names, (
        f"Wf-346-A2: p3_wf343c4_yuelbond_decoder_env should NOT be in "
        f"'frontier' pipeline (pre-retrain internal check is 'all'-only by "
        f"Wf-343C4 §3.4 design contract). Frontier gate names: {frontier_names}"
    )


def test_build_gates_all_count_invariant_with_wf343c4_gates():
    """Wf-346-A2 — 'all' count invariant + Wf-343C4 registration split.

    The Wf-321D gate-count-consistency audit asserts
    ``len(build_gates('all')) == EXPECTED_GATE_COUNTS['all']`` (the literal
    count drifts over time as gates are added; the invariant is the equality
    itself). Per Wf-A16 (2026-10-03) the Wf-343C4 gates split across
    pipelines: the 3 commit-safe gates stay on 'all', while the anchor
    freshness gate is LAUNCH-only — a commit must not fail purely on
    wall-clock anchor age. This test pins the count invariant AND the split
    registration (the Wf-343C4 integration suite at
    ``test_wf343c4_integration.py`` already asserts the gates'
    runner-level PASS / FAIL / SKIP behaviour; this test is the registry +
    count invariant slice).
    """
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS, build_gates

    runners = _build_gates_all()
    assert len(runners) >= 63, (
        f"Wf-346-A2: build_gates('all') returns {len(runners)} gates, "
        f"expected >= 63 (62 baseline + 1 NEW minimum). "
        f"This is a silent gate-loss regression — update "
        f"EXPECTED_GATE_COUNTS['all'] + TODO/completed/321d_preflight_gate_audit.md together."
    )
    assert len(runners) == EXPECTED_GATE_COUNTS["all"], (
        f"Wf-346-A2: build_gates('all') returns {len(runners)} gates, "
        f"EXPECTED_GATE_COUNTS['all'] = {EXPECTED_GATE_COUNTS['all']}. "
        f"Wf-321D audit invariant broken — update EXPECTED_GATE_COUNTS + "
        f"TODO/completed/321d_preflight_gate_audit.md together."
    )

    gate_names = [name for name, _ in runners]
    expected_wf343c4_gates = [
        "p3_wf343c4_hyperparams_config_complete",
        "p3_wf343c4_yuelbond_decoder_env",
        "p3_wf343c4_wave_budget_remaining_nonneg",
    ]
    for expected_gate in expected_wf343c4_gates:
        assert expected_gate in gate_names, (
            f"Wf-346-A2: Wf-343C4 gate '{expected_gate}' NOT in build_gates('all'). "
            f"This is the silent gate-loss regression that triggered Wf-346-A2. "
            f"All gate names: {gate_names}"
        )

    # Wf-A16: the anchor-freshness gate is launch-only — present in
    # build_gates('launch'), absent from the commit path.
    launch_names = [name for name, _ in build_gates("launch")]
    assert "p3_wf343c4_anchor_freshness_24h" in launch_names, (
        "Wf-346-A2: p3_wf343c4_anchor_freshness_24h NOT in "
        "build_gates('launch') — gate registration drifted from the "
        "Wf-343C4 + Wf-A16 contract."
    )
    assert "p3_wf343c4_anchor_freshness_24h" not in gate_names, (
        "Wf-346-A2: p3_wf343c4_anchor_freshness_24h was re-added to "
        "build_gates('all') — Wf-A16 moved it to the launch-only block on "
        "purpose (commit != launch; wall-clock FAILs on frozen anchors made "
        "the repo uncommittable)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
