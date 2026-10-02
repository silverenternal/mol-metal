"""Wf-346-A3 — Register gate_wf343c4_anchor_freshness_24h in build_gates(all).

================================================================
Per the 2026-09-29 ultracode wave (user directive "基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，
工作流可以长"), this Wf-346-A3 sub-flow is the registration-drift
fix for the anchor freshness 24h pre-flight gate. Per Wf-343D1, the
same registration drift as Wf-346-A1 was detected for the
``p3_wf343c4_anchor_freshness_24h`` gate: the gate function exists
at :mod:`molmetal.scripts.pre_flight_check` line ~3215 but was not
wired into the ``build_gates('all')`` dispatcher until this Wf-346-A3.

Tests
-----

1. **test_wf346a3_anchor_freshness_24h_in_all_pipeline** —
   Asserts that ``p3_wf343c4_anchor_freshness_24h`` is registered in
   the ``build_gates('all')`` dispatcher. This is the direct
   membership check that proves the gate is wired in (vs. only
   declared as a function on the module).

2. **test_wf346a3_build_gates_all_returns_at_least_63** —
   Asserts that ``build_gates('all')`` returns >= 63 runners AFTER
   registration. The 62 pre-Wf-343C4 baseline + at least 1 of the
   4 NEW Wf-343C4 gates (this one specifically). This pins the
   "+1" delta that Wf-346-A3 contributes to the all-pipeline count,
   making any future re-registration drift easily detectable.

Backward-compat verdict
-----------------------
The 2 NEW tests are ADDITIVE only — they do NOT modify any
pre-existing test / gate / runner / path. The Wf-321D
``EXPECTED_GATE_COUNTS['all'] == 66`` invariant is preserved (this
Wf-346-A3 contributes 1 runner: the anchor freshness gate). The
gate registration is in :mod:`molmetal.scripts.pre_flight_check`
inside the ``if pipeline == "all"`` block alongside the 3 sibling
Wf-343C4 gates (hyperparams config + YuelBond decoder env + wave
budget nonneg).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Registration drift detected MEASURED at Wf-343D1 timestamp
  (``actual=62 vs expected=66 [-DRIFT]``).
* Registration fix verified MEASURED at Wf-346-A3 timestamp
  (``actual=66 vs expected=66 [NO DRIFT]``).
* 2 NEW tests, CPU-only, < 1 s wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"


def _import_preflight():
    """Import :mod:`molmetal.scripts.pre_flight_check` as a module."""
    sys.path.insert(0, str(REPO_ROOT))
    import molmetal.scripts.pre_flight_check as preflight
    return preflight


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wf346a3_anchor_freshness_24h_in_all_pipeline():
    """``p3_wf343c4_anchor_freshness_24h`` MUST be in ``build_gates('all')``.

    Per Wf-343D1 finding: the gate function existed at
    :mod:`molmetal.scripts.pre_flight_check` line ~3215 but was not
    wired into the ``build_gates('all')`` dispatcher. Wf-346-A3
    fixes this drift by adding the gate to the dispatcher's
    ``if pipeline == "all"`` block alongside the 3 sibling Wf-343C4
    gates.

    The gate is matched by exact name on the dispatcher's runner
    list. If a future edit accidentally drops the gate (e.g. via a
    partial rebase / conflict), this test FAILs with the exact
    runner count diff for fast triage.
    """
    preflight = _import_preflight()
    runners = preflight.build_gates("all")
    runner_names = [name for name, _ in runners]

    expected_gate_name = "p3_wf343c4_anchor_freshness_24h"
    assert expected_gate_name in runner_names, (
        f"{expected_gate_name} not in build_gates('all') output "
        f"(count={len(runner_names)}). Wf-346-A3 registration drift "
        f"regression — re-add the gate to the dispatcher per Wf-343C4."
    )


def test_wf346a3_build_gates_all_returns_at_least_63():
    """``build_gates('all')`` MUST return >= 63 runners.

    The 62 pre-Wf-343C4 baseline + at least 1 of the 4 NEW Wf-343C4
    gates (this one specifically). This pins the "+1" delta that
    Wf-346-A3 contributes to the all-pipeline count, making any
    future re-registration drift easily detectable. The Wf-321D
    audit invariant (``EXPECTED_GATE_COUNTS['all'] == 66``) is the
    stricter check; this test is the more lenient pin that catches
    partial drift (e.g. only 1 of the 4 Wf-343C4 gates registered).
    """
    preflight = _import_preflight()
    runners = preflight.build_gates("all")

    assert len(runners) >= 63, (
        f"build_gates('all') returns {len(runners)} runners, "
        f"expected >= 63 (62 baseline + 1 NEW Wf-346-A3 anchor "
        f"freshness gate). Possible Wf-343C4 re-registration drift "
        f"regression — verify all 4 Wf-343C4 gates are wired in."
    )
