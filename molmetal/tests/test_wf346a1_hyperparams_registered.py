"""Wf-346-A1 — registration-regression guard for the 4 Wf-343C4 'all'-only gates.

================================================================================
Per the 2026-09-29 ultracode wave (user directive "基于刚才的洞察，可以启动
ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"),
Wf-343D1 surfaced a registration drift in
:func:`molmetal.scripts.pre_flight_check.build_gates`:

    * ``EXPECTED_GATE_COUNTS["all"] == 66`` (bumped by 4 to make room for
      the 4 NEW Wf-343C4 pre-retrain internal-check gates)
    * ``build_gates("all")`` returned only 62 runners (the 4 NEW gates
      were added to :data:`EXPECTED_GATE_COUNTS` but NOT registered in
      the ``all`` pipeline's runner list)

The Wf-345-A1 follow-up work then modified the gate LOGIC at lines
2940-3069 (and 3072-3214 / 3215-3336 / 3337-3504 for the other 3 gates)
but, because the 4 gates were never registered in the dispatcher, the
entire Wf-345-A1 logic-fix effort was moot — none of those 4 functions
were ever CALLED.

Wf-346-A1 registers the 4 NEW Wf-343C4 gates in the ``all`` pipeline
dispatcher and writes the ADMEASURABLE regression-guard surface here:

Tests
-----

1. **test_build_gates_all_returns_63_or_more_after_registration** —
   pin the post-registration floor: ``len(build_gates("all")) >= 63``.
   Pre-registration baseline (Wf-343D1) was 62; the 4 NEW Wf-343C4
   gates push the count to 66 (the ``EXPECTED_GATE_COUNTS["all"]``
   baseline).  A future regression that REMOVES any of the 4 NEW
   gates from the dispatcher (or any other future gate-removal
   regression) trips this test.  We use ``>= 63`` rather than
   ``== 66`` so that future additive gates do not break the test;
   the equality invariant is enforced separately by
   ``TestExpectedGateCountsSync`` in
   :mod:`molmetal.tests.test_wf327f11_gate_count_consistency`.

2. **test_gate_wf343c4_hyperparams_config_complete_in_all_pipeline** —
   pin the SPECIFIC gate name in the ``all`` pipeline dispatcher.
   Wf-346-A1's primary concern (per the user's task brief) was the
   hyperparams gate specifically; this test catches a future rename / re-
   location / accidental deletion of just that one gate (the broader
   4-gate drift is covered by test 1 above).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Pre-registration state: ``build_gates("all") == 62`` MEASURED at
  Wf-343D1 timestamp (2026-09-29 07:51:24 UTC, wall 112.35s).
* Post-registration state: ``build_gates("all") == 66`` MEASURED at
  Wf-346-A1 timestamp (2026-09-29; see TODO/completed/346a1_register
  _hyperparams_gate.md for the verification log).
* The 2 NEW tests are ADDITIVE only — they do NOT modify any pre-
  existing test / gate / runner / path.
* Wf-345-A1 logic fixes remain moot for downstream coverage until
  the gate LOGIC is itself fixed (3 of 4 Wf-343C4 gates currently
  FAIL on logic per Wf-346-A1 verification; that is Wf-345 Phase 3
  follow-up, not in scope for Wf-346-A1).
* CPU-only; runs in < 200 ms.

Standing-rule compliance
------------------------
- AGENTS.md §1 MEASURED-vs-PROJECTED: every numeric is build_gates-
  derived.
- AGENTS.md §5.2 honest framing: this is a registration fix + reverify
  guard, not a metric regression.
- party-twenty-spirit §3.5: every test is additive (no Wf-343C4 /
  Wf-321D test removed).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Path constants — pin the dispatcher file on disk so the registration
# guard has a fixed target.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_FLIGHT_SCRIPT = (
    REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
)


# ---------------------------------------------------------------------------
# Constants — the Wf-343C4 gate-name contract.
# ---------------------------------------------------------------------------
# The 4 NEW Wf-343C4 gates added to EXPECTED_GATE_COUNTS["all"] baseline
# (62 → 66).  These names MUST appear in the ``all`` pipeline's runner
# list; the broader count invariant (== 66) is enforced by the Wf-327F11
# test_pre_flight_check.py TestExpectedGateCountsSync test.
WF343C4_GATE_NAMES = (
    "p3_wf343c4_hyperparams_config_complete",
    "p3_wf343c4_yuelbond_decoder_env",
    "p3_wf343c4_anchor_freshness_24h",
    "p3_wf343c4_wave_budget_remaining_nonneg",
)
# Primary target gate per the Wf-346-A1 task brief.
PRIMARY_GATE_NAME = "p3_wf343c4_hyperparams_config_complete"
# Pre-registration baseline (Wf-343D1 MEASURED).
PRE_REGISTRATION_ALL_BASELINE = 62
# Post-registration floor (>= 63, == 66 at Wf-346-A1 timestamp).
POST_REGISTRATION_ALL_FLOOR = 63


def _build_gates_all() -> List[Tuple[str, object]]:
    """Return the (name, runner) tuple list for the ``all`` pipeline.

    Wraps :func:`molmetal.scripts.pre_flight_check.build_gates` so the
    test file can be collected even when the dispatcher surface
    changes (a future refactor that moves build_gates into a separate
    module only requires updating this one call site).
    """
    from molmetal.scripts.pre_flight_check import build_gates

    return list(build_gates("all"))


# ---------------------------------------------------------------------------
# Test 1 — build_gates('all') >= POST_REGISTRATION_ALL_FLOOR after registration
# ---------------------------------------------------------------------------
class TestBuildGatesAllRegistered:
    """Post-registration floor: build_gates('all') >= 63 runners.

    Pre-registration baseline (Wf-343D1 MEASURED): ``build_gates('all')``
    returned 62 runners.  The 4 NEW Wf-343C4 gates push the count to 66
    (the ``EXPECTED_GATE_COUNTS["all"]`` constant).

    This test pins the POST-registration floor at 63 (one above the
    pre-registration count) without coupling to the exact 66 count,
    so future ADDITIVE gates (e.g. Wf-347+ workflow additions) do
    not break the test.  The exact-equality invariant (== 66) is
    covered separately by the Wf-327F11
    TestExpectedGateCountsSync::test_expected_gate_counts_match_actual
    test.
    """

    def test_build_gates_all_returns_63_or_more_after_registration(
        self,
    ) -> None:
        """``len(build_gates("all")) >= 63`` post-registration floor.

        MEASURED drift invariant: a future regression that REMOVES
        the Wf-343C4 gates (or any other gate) from the dispatcher
        silently drops build_gates("all") below the floor, which
        this test catches.  At Wf-346-A1 timestamp the actual count
        is 66 (matching ``EXPECTED_GATE_COUNTS["all"]``); the test
        asserts ``>= 63`` (one above the 62 pre-registration
        baseline) to leave headroom for future additive gates.

        Pre-registration failure mode: Wf-343D1 reported 62 runners
        (4 short of the 66 expected).  Post-registration success
        mode: 66 runners.  The 4-runner lift between these states
        is exactly the 4 NEW Wf-343C4 gates that Wf-346-A1 was
        opened to register.
        """
        runners = _build_gates_all()
        actual_count = len(runners)
        names = [r[0] for r in runners]
        assert actual_count >= POST_REGISTRATION_ALL_FLOOR, (
            f"build_gates('all') returned only {actual_count} runners "
            f"(expected >= {POST_REGISTRATION_ALL_FLOOR}); the "
            f"Wf-343C4 registration regressed.  At Wf-343D1 timestamp "
            f"the count was {PRE_REGISTRATION_ALL_BASELINE}; post-"
            f"registration baseline is "
            f"EXPECTED_GATE_COUNTS['all'] = 66.  Failing names: "
            f"{[n for n in WF343C4_GATE_NAMES if n not in names]}"
        )
        # Sanity: every Wf-343C4 gate-name contract holder is present.
        missing = [n for n in WF343C4_GATE_NAMES if n not in names]
        assert not missing, (
            f"Wf-343C4 gate-name contract drift: {missing} not "
            f"registered in build_gates('all'); Wf-346-A1 was opened "
            f"to register exactly these 4 gates.  Full registered "
            f"set: {sorted(n for n in names if 'wf343c4' in n)}"
        )


# ---------------------------------------------------------------------------
# Test 2 — primary gate name (hyperparams) is in the all pipeline
# ---------------------------------------------------------------------------
class TestWf343C4GateInAllPipeline:
    """Per-gate registration check for the Wf-346-A1 primary target.

    The Wf-346-A1 task brief named
    ``gate_wf343c4_hyperparams_config_complete`` as the SPECIFIC gate
    that needed registration; this test catches a future regression
    that removes just that one gate (e.g. a careless Edit to the
    dispatcher that drops one of the 4 Wf-343C4 entries).
    """

    def test_gate_wf343c4_hyperparams_config_complete_in_all_pipeline(
        self,
    ) -> None:
        """``p3_wf343c4_hyperparams_config_complete`` is in build_gates('all').

        Per the Wf-346-A1 task brief: the gate must appear in the
        ``all`` pipeline's runner list AND the runner must be a
        callable (the pre-registration state had no such entry at
        all).  This test pins the SPECIFIC gate name so a future
        Edit that accidentally renames / removes just this one
        gate surfaces immediately.
        """
        runners = _build_gates_all()
        names_to_runners = {name: runner for name, runner in runners}
        assert PRIMARY_GATE_NAME in names_to_runners, (
            f"Wf-343C4 hyperparams gate '{PRIMARY_GATE_NAME}' not "
            f"registered in build_gates('all'); the Wf-346-A1 task "
            f"brief was to register this specific gate.  Current "
            f"Wf-343C4 gates in pipeline: "
            f"{sorted(n for n in names_to_runners if 'wf343c4' in n)}"
        )
        # The runner must be a callable (lambda or function); assert
        # it can be invoked without raising ImportError at definition
        # time.  We do NOT assert the return-status here because
        # Wf-345-A1 logic-fix is a downstream responsibility (3 of 4
        # Wf-343C4 gates currently FAIL on logic per Wf-346-A1
        # verification; this is not Wf-346-A1's scope).
        runner = names_to_runners[PRIMARY_GATE_NAME]
        assert callable(runner), (
            f"runner for '{PRIMARY_GATE_NAME}' is not callable: "
            f"{runner!r}; the dispatcher registration contract is "
            f"(name, Callable[[], GateResult])."
        )
        # Optional smoke: invoke the runner and assert it returns a
        # GateResult-shaped object (status / evidence / wall_sec).
        # Wrap in a try/except so a transient Wf-345-A1 logic-fix
        # FAIL does not mask the registration contract — we only
        # care that the runner EXISTS + RUNS, not that it returns
        # status=Pass.
        try:
            result = runner()
        except Exception as exc:
            pytest.fail(
                f"runner for '{PRIMARY_GATE_NAME}' raised at "
                f"invocation: {type(exc).__name__}: {exc}.  The "
                f"registration is in place but the gate LOGIC is "
                f"broken - this is Wf-345-A1 logic-fix scope, NOT "
                f"Wf-346-A1 registration scope."
            )
        # GateResult shape check (per pre_flight_check.py GateResult
        # dataclass: status / priority / pipeline / evidence / wall_sec).
        for attr in ("status", "priority", "pipeline", "evidence",
                     "wall_sec"):
            assert hasattr(result, attr), (
                f"GateResult for '{PRIMARY_GATE_NAME}' missing "
                f"attribute {attr!r}: {result!r}.  The runner "
                f"returned an unexpected shape — verify it routes "
                f"through gate_wf343c4_hyperparams_config_complete "
                f"at pre_flight_check.py:2940."
            )
        assert result.status in {"PASS", "FAIL", "SKIP"}, (
            f"unexpected status {result.status!r} from "
            f"'{PRIMARY_GATE_NAME}' runner; pre_flight GateResult "
            f"contract is one of PASS / FAIL / SKIP."
        )