"""Wf-345-A4 — pre-flight wave-budget gate 3-way verdict (PASS / SKIP / FAIL).

================================================================
Per the 2026-09-29 user request ("基于刚才的洞察，可以启动ultracode，覆盖所有
发现的问题把残余的问题解决了，一定要全，工作流可以长") + AGENTS.md §5.1
standing GPU grant, the Wf-343C4 wave-budget gate is upgraded from
a binary PASS / FAIL to a 3-way verdict:

  * **PASS** — remaining_hr >= 0 AND non-deferred wave cost fits
    within budget_hr (the normal healthy case).
  * **SKIP** — remaining_hr < 0 BUT every over-budget cost is
    attributable to ``DEFERRED`` waves (the user-gated class);
    non-deferred cost fits. This is BY-DESIGN per Wf-262 §7.1
    budget discipline — the deferred waves need ``--no-enforce-budget``
    or schedule outside the 10h session, but the gate is honest
    about the contract: the budget is being honoured for what
    WILL run, and the DEFERRED portion is transparently flagged.
  * **FAIL** — non-deferred wave cost (excluding COMPLETE
    already-incurred) exceeds budget_hr, OR the report is
    structurally broken (missing JSON, missing fields, etc.).

Why SKIP instead of FAIL on DEFERRED-only exhaustion: per
``party-twenty-spirit`` §3.5 honest framing, "SKIP" preserves the
no-fabrication invariant — we are NOT claiming the budget is healthy
(that would be a lie); we are ALSO not claiming the retrain is
blocked (that would be a false-positive — the deferred waves are
user-gated and excluded from the current session's launch plan).

Tests (3 NEW tests, one per verdict path)
----------------------------------------

1. **test_gate_wf345a4_skip_when_only_deferred_exhaust_budget** —
   writes a 10h-budget / 16h-DEFERRED-wave / 4h-non-deferred-wave /
   0h-PLANNED fixture, asserts the gate emits ``SKIP`` (not FAIL,
   not PASS). Verifies the Wf-345-A4 honest-framing revision is
   wired correctly.

2. **test_gate_wf345a4_fail_when_non_deferred_overrun** —
   writes a 10h-budget / 14h-non-deferred-wave (no DEFERRED)
   / 5h-COMPLETE fixture where non-deferred cost exceeds
   budget_hr, asserts the gate emits ``FAIL`` (real overrun —
   dropping the deferred waves from the equation doesn't fix it).

3. **test_gate_wf345a4_pass_when_budget_healthy** —
   writes a 10h-budget / 4h-non-deferred-wave / 5h-COMPLETE
   / 0-DEFERRED fixture where everything fits, asserts the gate
   emits ``PASS``.

Backward-compat verdict
-----------------------
The 3 NEW tests are ADDITIVE only — they cover the new SKIP path
that the Wf-345-A4 ship surface introduces; the existing FAIL and
PASS paths are already covered by the Wf-343C4 integration test
(``test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script``)
which accepts FAIL-or-SKIP on its deferred-wave fixture. No gate
count changes (the wave-budget gate function is the same; the
verdict matrix is the only change).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (verdict status + key
  evidence substring) so failures localise cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
  to the on-disk wave_budget_report.json fixture.
* The 10h budget vs 193.04h cumulative mismatch is HONEST-NEGATIVE
  and the SKIP verdict is the correct contract for the deferred-only
  portion (per AGENTS.md §5.1 + Wf-262 §7.1 budget discipline).

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py`` — the
  ``gate_wf343c4_wave_budget_remaining_nonneg`` 3-way verdict
  function updated by Wf-345-A4.
* ``molmetal/scripts/wave_budget_tracker.py`` — the source of the
  wave_budget_report.json consumed by the gate.
* ``TODO/completed/345a4_wave_budget_skip.md`` — the Wf-345-A4 ship
  summary + the 3-way verdict design context.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    function directly. Mirrors the helper used in
    ``test_wf343c4_integration.py``.
    """
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_wf345a4",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_wf345a4"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-level fixture exposing the pre_flight_check module."""
    return _load_preflight_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_wave_budget_report(
    path: Path,
    *,
    budget_hr: float,
    remaining_hr: float,
    deferred_waves: list,
    waves: list,
    honest_framing: str = "fixture",
) -> Path:
    """Write a Wf-262 §7.1 schema wave_budget_report.json fixture.

    Mirrors the schema produced by
    :func:`molmetal.scripts.wave_budget_tracker.summarise`. The
    ``cumulative_wave_hr`` field is derived as ``budget_hr -
    remaining_hr`` for consistency with the report contract — tests
    should care about ``remaining_hr`` + ``deferred_waves`` + the
    per-wave status, not the raw cumulative sum.
    """
    report = {
        "schema_version": "262.v1",
        "ts": "2026-09-29T15:30:00+00:00",
        "budget_hr": budget_hr,
        "cumulative_wave_hr": budget_hr - remaining_hr,
        "remaining_hr": remaining_hr,
        "deferred_waves": deferred_waves,
        "honest_framing": honest_framing,
        "waves": waves,
        "wave_source": "from --waves CLI (Wf-345-A4 fixture)",
        "auto_discovered": False,
        "n_configs_scanned": len(waves),
        "n_actual_hr_loaded": sum(1 for w in waves if w.get("actual_hr") is not None),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1 — SKIP path: only DEFERRED waves exhaust the budget
# ---------------------------------------------------------------------------


def test_gate_wf345a4_skip_when_only_deferred_exhaust_budget(
    preflight,
    tmp_path,
) -> None:
    """Wf-345-A4 (1) — gate emits SKIP (not FAIL, not PASS) when
    the over-budget figure is wholly attributable to DEFERRED waves.

    Per Wf-262 §7.1 + Wf-345-A4 3-way verdict: a 10h session budget
    where the non-deferred wave cost (PLANNED + RUNNING) is 4.0h but
    cumulative_wave_hr (including a 16.0h DEFERRED CFM wave) is
    20.0h → ``remaining_hr = -10.0h`` is negative BUT the only wave
    pushing the cumulative over budget is the DEFERRED one (which
    is user-gated and will not run this session). The gate must emit
    SKIP, not FAIL (real budget overrun) and not PASS (honest
    framing — the deferred waves are explicitly named in the evidence).
    """
    skip_report = _write_wave_budget_report(
        tmp_path / "wave_budget_skip.json",
        budget_hr=10.0,
        remaining_hr=-10.0,
        deferred_waves=["Wave-cfm:cfm_full_retrain"],
        waves=[
            # COMPLETE — already done, excluded from non-deferred cost.
            {
                "wave_name": "Wave-1:chemprop_smoke",
                "estimated_hr": 0.5,
                "actual_hr": 0.5,
                "status": "COMPLETE",
            },
            # PLANNED — WILL run this session, costs 4.0h, fits budget.
            {
                "wave_name": "Wave-2:chemprop_v2_full_sweep",
                "estimated_hr": 4.0,
                "actual_hr": None,
                "status": "PLANNED",
            },
            # DEFERRED — user-gated CFM wave (would be 16h, but DEFERRED
            # so excluded from non-deferred cost).
            {
                "wave_name": "Wave-cfm:cfm_full_retrain",
                "estimated_hr": 16.0,
                "actual_hr": None,
                "status": "DEFERRED",
            },
        ],
        honest_framing=(
            "10.0h budget exceeded by 10.0h; 1 wave(s) DEFERRED to next "
            "session (Wave-cfm:cfm_full_retrain); non-deferred fits"
        ),
    )
    res = preflight.gate_wf343c4_wave_budget_remaining_nonneg(
        "wf345a4_skip_test",
        "P3",
        "wf345a4",
        skip_report,
    )
    assert res.status == "SKIP", (
        f"deferred-only budget exhaustion should emit SKIP; got "
        f"{res.status}: {res.evidence}"
    )
    # Evidence must surface both the deferred wave name + the BY-DESIGN
    # framing so downstream consumers know it's an honest negative,
    # not a launch blocker.
    assert (
        "DEFERRED" in res.evidence
    ), f"SKIP evidence should name DEFERRED waves; got {res.evidence!r}"
    assert (
        "cfm_full_retrain" in res.evidence
    ), f"SKIP evidence should name the deferred wave; got {res.evidence!r}"
    assert (
        "BY-DESIGN" in res.evidence or "non-deferred" in res.evidence.lower()
    ), f"SKIP evidence should explain the BY-DESIGN framing; got {res.evidence!r}"
    # Status must not be FAIL (no real overrun) or PASS (would lie about
    # healthy budget).
    assert (
        res.status != "FAIL"
    ), f"deferred-only exhaustion must NOT be FAIL; got {res.evidence!r}"
    assert res.status != "PASS", (
        f"deferred-only exhaustion must NOT be PASS (honest negative); "
        f"got {res.evidence!r}"
    )


# ---------------------------------------------------------------------------
# 2 — FAIL path: non-DEFERRED waves themselves exceed the budget
# ---------------------------------------------------------------------------


def test_gate_wf345a4_fail_when_non_deferred_overrun(
    preflight,
    tmp_path,
) -> None:
    """Wf-345-A4 (2) — gate emits FAIL when no DEFERRED waves exist
    AND remaining_hr < 0 (real budget overrun — every wave in the
    report contributes; the user has not deferred anything).

    A 10h session budget with NO DEFERRED waves but a 14.0h PLANNED
    wave (alone exceeds budget) → remaining_hr = -4.0h → real
    overrun. The gate must emit FAIL because there are no DEFERRED
    waves to attribute the over-budget to (per Wf-345-A4 SKIP-takes-
    precedence logic: SKIP only fires when DEFERRED waves exist).
    """
    fail_report = _write_wave_budget_report(
        tmp_path / "wave_budget_fail.json",
        budget_hr=10.0,
        remaining_hr=-4.0,
        deferred_waves=[],  # NO DEFERRED — every wave contributes
        waves=[
            # COMPLETE — already incurred.
            {
                "wave_name": "Wave-1:chemprop_smoke",
                "estimated_hr": 0.5,
                "actual_hr": 0.5,
                "status": "COMPLETE",
            },
            # PLANNED — 14h alone exceeds budget.
            {
                "wave_name": "Wave-2:chemprop_v2_full_sweep",
                "estimated_hr": 14.0,
                "actual_hr": None,
                "status": "PLANNED",
            },
        ],
        honest_framing=(
            "10.0h budget exceeded by 4.0h; 0 wave(s) DEFERRED; "
            "real budget overrun (every wave in the report contributes)"
        ),
    )
    res = preflight.gate_wf343c4_wave_budget_remaining_nonneg(
        "wf345a4_fail_test",
        "P3",
        "wf345a4",
        fail_report,
    )
    assert res.status == "FAIL", (
        f"real overrun with no DEFERRED waves should emit FAIL; got "
        f"{res.status}: {res.evidence}"
    )
    # Evidence must name the no-DEFERRED + negative remaining_hr so the
    # downstream consumer can diagnose the real problem.
    assert (
        "no DEFERRED" in res.evidence or "0 DEFERRED" in res.evidence
    ), f"FAIL evidence should clarify no DEFERRED waves; got {res.evidence!r}"
    assert (
        "remaining_hr=-4.00h" in res.evidence or "remaining_hr=-4.0h" in res.evidence
    ), f"FAIL evidence should name the negative remaining_hr; got {res.evidence!r}"
    assert (
        "real overrun" in res.evidence.lower()
    ), f"FAIL evidence should describe the real overrun; got {res.evidence!r}"


# ---------------------------------------------------------------------------
# 3 — PASS path: budget is healthy
# ---------------------------------------------------------------------------


def test_gate_wf345a4_pass_when_budget_healthy(
    preflight,
    tmp_path,
) -> None:
    """Wf-345-A4 (3) — gate emits PASS when budget is healthy
    (remaining_hr >= 0 AND non-deferred cost fits within budget_hr).

    A 10h session budget with 4.0h non-deferred cost and 0 DEFERRED
    waves → fits comfortably with 6.0h buffer. Gate must emit PASS
    (the normal healthy case).
    """
    pass_report = _write_wave_budget_report(
        tmp_path / "wave_budget_pass.json",
        budget_hr=10.0,
        remaining_hr=6.0,
        deferred_waves=[],
        waves=[
            # COMPLETE — already incurred.
            {
                "wave_name": "Wave-1:chemprop_smoke",
                "estimated_hr": 0.5,
                "actual_hr": 0.5,
                "status": "COMPLETE",
            },
            # PLANNED — WILL run, 4.0h total, fits within budget.
            {
                "wave_name": "Wave-2:chemprop_v2_full_sweep",
                "estimated_hr": 4.0,
                "actual_hr": None,
                "status": "PLANNED",
            },
        ],
        honest_framing=(
            "10.0h budget sufficient; 6.0h buffer remaining after " "projected waves."
        ),
    )
    res = preflight.gate_wf343c4_wave_budget_remaining_nonneg(
        "wf345a4_pass_test",
        "P3",
        "wf345a4",
        pass_report,
    )
    assert res.status == "PASS", (
        f"healthy 10h-budget / 4h-non-deferred should PASS; got "
        f"{res.status}: {res.evidence}"
    )
    # Evidence must surface the budget + remaining + non-deferred cost
    # so downstream consumers can audit the contract.
    assert (
        "budget_hr=10" in res.evidence
    ), f"PASS evidence should name budget_hr; got {res.evidence!r}"
    assert (
        "remaining_hr=6.0" in res.evidence
    ), f"PASS evidence should name remaining_hr; got {res.evidence!r}"
    assert (
        "non_deferred_cost=4.00h" in res.evidence
    ), f"PASS evidence should name non_deferred_cost; got {res.evidence!r}"
