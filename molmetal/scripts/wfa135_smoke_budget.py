#!/usr/bin/env python3
"""Wf-A135 (2026-10-31) — the ONE source of truth for the pre-flight's
AGENTS.md §5.7 wire-in SMOKE BUDGETS.

WHY THIS MODULE EXISTS
======================

MEASURED 2026-10-31 on this host::

    .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all --strict
    -> {"pass": 113, "fail": 3, "skip": 11, "warn": 0, "total": 127,
        "total_wall_sec": 411.0377}

All THREE FAILs were wall-clock TIMEOUTS and not defects:

=========================================  =========================
gate                                       budget / observed
=========================================  =========================
``p1_chemprop_smoke``                      90 s budget, TIMED OUT @ 94.8 s
``p3_wf385_scaffold_loss_wire_in_verified`` 150 s budget (inner smoke)
``p3_dmpnn_loss_wire_in_smoke``             300 s budget, rc=124
=========================================  =========================

That is a defect in the GATE, not a fact about the code.  For these three
gates the FAIL verdict LITERALLY means *"the trainer flag is NOT wired
in"* — and a timeout means the MEASUREMENT WAS NEVER OBTAINED.  The two
were indistinguishable in the report, so the red state could not be
read, and a reader who trusted "FAIL" would chase a wire-in bug that
does not exist.  Per the Wf-A128 principle — *a gate that FAILs on
missing data is worse than no gate* — Wf-A135 gives a timeout its own
verdict (``TIMEOUT``) in ``pre_flight_check.py``.

WHAT THIS MODULE HOLDS
======================

For each of the three smokes:

* a **NAMED constant** holding the wall budget in seconds,
* the **environment variable** that overrides it,
* the **MEASURED standalone baseline** the budget was derived from,
* the **MEASURED contended wall** the same smoke showed inside a real
  pre-flight run, and
* the **headroom factor** that was applied.

The measured baseline, the contended wall and the headroom factor are
recorded in the comment next to each constant, and mirrored
machine-readably in
``molmetal/tests/fixtures/wfa135_smoke_budget.json``.

This module is the ONE source of truth.  ``pre_flight_check.py`` imports
it; nothing else may hard-code a smoke budget.

THE DERIVATION (how a budget becomes a number)
===============================================

Three MEASURED numbers per smoke, all CPU-only
(``CUDA_VISIBLE_DEVICES=""``) on this host:

``S``  the **standalone baseline** — the wall time of that smoke run
       STANDALONE, as the LARGEST of the standalone repeats
       (conservative by construction: a smoke that once took this long
       on this host must still fit in the budget);

``Cw`` the **contended wall** — the wall time the SAME smoke showed
       inside a real ``--pipeline all --strict`` pre-flight run, where
       ``--jobs`` defaults to ``min(8, cpu_count)`` so the smokes share
       the host with seven other gates;

``Hw`` the **worst observed wall** — the largest of {``S``, ``Cw``, and
       the wall at which the pre-flight itself CUT the smoke off in the
       Wf-A133 baseline run.  A cut-off is a LOWER BOUND (the smoke did
       not finish within it), so using it as ``Hw`` is conservative.

From those, one GLOBAL safety factor and one budget::

    R      = ceil_0.05( max over the three smokes of ( Hw / Cw ) )
    budget = ceil_0.1( Hw * R )

``R`` is not a number picked by habit: it is the MEASURED run-to-run
inflation — how much slower the worst observation of a smoke was than a
completed observation of the same smoke on the same host, taken as the
worst case over the three smokes.  It is >= 1.0 by construction.

``headroom_factor`` (recorded per smoke) is the TOTAL multiplier that
the budget applies to the standalone baseline::

    headroom_factor = budget / S

so that ``budget == S * headroom_factor`` can be re-derived by hand
from the numbers printed beside the constant.

THE ENVIRONMENT OVERRIDE
=========================

Each budget can be overridden by its own NAMED environment variable
(``MOLMETAL_SMOKE_BUDGET_<NAME>_SEC``).  This exists because a budget
derived on one host is a *measurement*, not a law: a slower host needs a
different number, and an operator must be able to supply it without
editing code.  A malformed or non-positive override is REPORTED on
stderr and then IGNORED (the default is used) — a typo in an env var
must never turn into a silently-instant timeout.

See also: ``docs/HONEST_NEGATIVE_CATALOG.md`` (the honest-negative
catalog) and ``TODO/INDEX.md`` for the wave ledger.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Machine-readable mirror of the MEASURED baselines.  The module below is
#: authoritative; the fixture is what a machine (the regression test) reads
#: to prove the two have not drifted apart.
FIXTURE_PATH = (
    REPO_ROOT / "molmetal" / "tests" / "fixtures" / "wfa135_smoke_budget.json"
)


# ---------------------------------------------------------------------------
# Gate names — the three AGENTS.md §5.7 wire-in smokes that carry a budget.
# ---------------------------------------------------------------------------

GATE_CHEMPROP_SMOKE = "p1_chemprop_smoke"
GATE_DMPNN_SMOKE = "p3_dmpnn_loss_wire_in_smoke"
GATE_WF385_SMOKE = "p3_wf385_scaffold_loss_wire_in_verified"


# ---------------------------------------------------------------------------
# Environment variable names — one per smoke, so one can be retuned
# without touching the others.
# ---------------------------------------------------------------------------

ENV_CHEMPROP_SMOKE_BUDGET_SEC = "MOLMETAL_SMOKE_BUDGET_CHEMPROP_SEC"
ENV_DMPNN_SMOKE_BUDGET_SEC = "MOLMETAL_SMOKE_BUDGET_DMPNN_SEC"
ENV_WF385_SMOKE_BUDGET_SEC = "MOLMETAL_SMOKE_BUDGET_WF385_SEC"


# ---------------------------------------------------------------------------
# The budget record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeBudget:
    """One smoke's wall budget, with the provenance that justifies it."""

    gate_name: str
    budget_sec: float
    measured_baseline_sec: float
    headroom_factor: float
    env_var: str
    measured_at: str
    device: str
    measured_command: str
    standalone_walls_sec: tuple
    contended_observed_sec: float = 0.0
    headroom_allowance: float = 1.0
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "gate_name": self.gate_name,
            "budget_sec": self.budget_sec,
            "measured_baseline_sec": self.measured_baseline_sec,
            "headroom_factor": self.headroom_factor,
            "env_var": self.env_var,
            "measured_at": self.measured_at,
            "device": self.device,
            "measured_command": self.measured_command,
            "standalone_walls_sec": list(self.standalone_walls_sec),
            "contended_observed_sec": self.contended_observed_sec,
            "headroom_allowance": self.headroom_allowance,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# THE MEASURED SAFETY FACTOR.
#
# MEASURED 2026-10-31, CPU-only (CUDA_VISIBLE_DEVICES=""), this host:
#
#   smoke      Hw (worst observed)  Cw (contended, inside the pre-flight)
#   chemprop   94.8  (cut-off)      83.7284   -> Hw/Cw = 1.132232
#   dmpnn     300.0  (cut-off)     215.5782   -> Hw/Cw = 1.391606
#   wf385     223.832 (standalone)  159.6427   -> Hw/Cw = 1.402084
#
#   R = ceil_0.05( max(1.132232, 1.391606, 1.402084) ) = ceil_0.05(1.402084) = 1.45
#
# 1.45 is therefore the MEASURED worst-case run-to-run inflation of a
# §5.7 smoke's wall on this host (the worst observation of a smoke was
# up to 40 % slower than a completed observation of the same smoke),
# not a round number chosen by habit.
# ---------------------------------------------------------------------------

HEADROOM_ALLOWANCE = 1.45


# ---------------------------------------------------------------------------
# THE BUDGETS.
#
# Every constant below is `ceil_0.1(Hw * R)` with Hw, R and the
# standalone baseline S written out, so the number can be re-derived by
# hand from the numbers printed beside it.
#
# MEASURED 2026-10-31 on this host, CPU-only (CUDA_VISIBLE_DEVICES=""),
# full-subprocess wall clock.
# ---------------------------------------------------------------------------

#: ``p1_chemprop_smoke`` — ``train_chemprop_v2_v3.py --quick``
#:
#: S  (standalone, MAX over repeats) = 81.516323 s
#:    repeats: (49.170, 6.919, 81.516, 78.936, 33.136)
#: Cw (contended, inside the pre-flight) = 83.7284 s
#: Hw (worst observed) = 94.8 s
#:    — the Wf-A133 baseline run cut this smoke off at its 90 s budget
#:      after 94.8 s, so 94.8 s is a LOWER BOUND on its wall.
#: R  = 1.45 (the global MEASURED inflation, see HEADROOM_ALLOWANCE)
#: budget = ceil_0.1(94.8 x 1.45) = 137.5 s
#: headroom_factor = 137.5 / 81.516323 = 1.686779
CHEMPROP_SMOKE_BUDGET_SEC = 137.5

#: ``p3_dmpnn_loss_wire_in_smoke`` — ``train_dmpnn_multitask_wasserstein.py``
#:
#: S  (standalone, MAX over repeats) = 127.284939 s
#:    repeats: (94.750, 123.375, 127.285, 116.105)
#: Cw (contended, inside the pre-flight) = 215.5782 s
#: Hw (worst observed) = 300.0 s
#:    — the Wf-A133 baseline run cut this smoke off at its 300 s budget
#:      (rc=124), so 300.0 s is a LOWER BOUND on its wall.
#: R  = 1.45 (the global MEASURED inflation, see HEADROOM_ALLOWANCE)
#: budget = ceil_0.1(300.0 x 1.45) = 435.0 s
#: headroom_factor = 435.0 / 127.284939 = 3.417529
DMPNN_SMOKE_BUDGET_SEC = 435.0

#: ``p3_wf385_scaffold_loss_wire_in_verified`` —
#: ``wf385a3_scaffold_wire_in_gate.py`` (which itself shells out to a
#: 50-step CFM smoke).
#:
#: S  (standalone, MAX over repeats) = 223.832467 s
#:    repeats: (223.832, 142.879)
#: Cw (contended, inside the pre-flight) = 159.6427 s
#: Hw (worst observed) = 223.832467 s
#:    — the largest standalone repeat; the Wf-A133 baseline run cut the
#:      INNER smoke off at its 150 s budget, and that cut-off is below
#:      the standalone maximum, so the standalone maximum binds.
#: R  = 1.45 (the global MEASURED inflation, see HEADROOM_ALLOWANCE)
#: budget = ceil_0.1(223.832467 x 1.45) = 324.6 s
#: headroom_factor = 324.6 / 223.832467 = 1.450192
#: (this is the OUTER budget; the inner gate script gets budget - 30 s so
#:  it can still write its JSON before the outer subprocess is killed)
WF385_SMOKE_BUDGET_SEC = 324.6


CHEMPROP_SMOKE_BUDGET = SmokeBudget(
    gate_name=GATE_CHEMPROP_SMOKE,
    budget_sec=CHEMPROP_SMOKE_BUDGET_SEC,
    measured_baseline_sec=81.516323,
    headroom_factor=1.686779,
    env_var=ENV_CHEMPROP_SMOKE_BUDGET_SEC,
    measured_at="2026-10-31",
    device="cpu",
    measured_command=(
        ".venv/bin/python molmetal/scripts/train_chemprop_v2_v3.py --quick "
        "--allow-single-seed --device cpu --out-dir <tmp> --metals Ru "
        "--seeds 0"
    ),
    standalone_walls_sec=(49.170, 6.919, 81.516, 78.936, 33.136),
    contended_observed_sec=83.7284,
    headroom_allowance=HEADROOM_ALLOWANCE,
    note=(
        "Hw = 94.8 s is a LOWER BOUND (the Wf-A133 baseline run cut the "
        "smoke off at its 90 s budget after 94.8 s). Standalone repeats "
        "1-2 were measured while a concurrent Wf-A135 D-MPNN measurement "
        "shared the host; repeats 3-5 were measured with no Wf-A135 job "
        "of ours in flight. The MAX over repeats (81.516 s) is used as "
        "the baseline, so the budget is conservative by construction."
    ),
)

DMPNN_SMOKE_BUDGET = SmokeBudget(
    gate_name=GATE_DMPNN_SMOKE,
    budget_sec=DMPNN_SMOKE_BUDGET_SEC,
    measured_baseline_sec=127.284939,
    headroom_factor=3.417529,
    env_var=ENV_DMPNN_SMOKE_BUDGET_SEC,
    measured_at="2026-10-31",
    device="cpu",
    measured_command=(
        ".venv/bin/python molmetal/scripts/"
        "train_dmpnn_multitask_wasserstein.py --device cpu --epochs 2 "
        "--csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv "
        "--data-csv .data/molmetal/MetalCytoToxDB.csv --metals Re --seeds 0 "
        "--batch-size 32 --cpu-num-workers 2 --hidden 64 --depth 2 "
        "--out-dir <tmp> --ckpt-dir <tmp>/ckpt"
    ),
    standalone_walls_sec=(94.750, 123.375, 127.285, 116.105),
    contended_observed_sec=215.5782,
    headroom_allowance=HEADROOM_ALLOWANCE,
    note=(
        "Hw = 300.0 s is a LOWER BOUND (the Wf-A133 baseline run cut the "
        "smoke off at its 300 s budget, rc=124). This smoke showed the "
        "LARGEST run-to-run spread of the three (215.6 s completed vs "
        ">300 s cut off), which is what sets the global R = 1.45."
    ),
)

WF385_SMOKE_BUDGET = SmokeBudget(
    gate_name=GATE_WF385_SMOKE,
    budget_sec=WF385_SMOKE_BUDGET_SEC,
    measured_baseline_sec=223.832467,
    headroom_factor=1.450192,
    env_var=ENV_WF385_SMOKE_BUDGET_SEC,
    measured_at="2026-10-31",
    device="cpu",
    measured_command=(
        ".venv/bin/python molmetal/scripts/wf385a3_scaffold_wire_in_gate.py "
        "--out-dir <tmp> --python-bin .venv/bin/python --timeout 900"
    ),
    standalone_walls_sec=(223.832, 142.879),
    contended_observed_sec=159.6427,
    headroom_allowance=HEADROOM_ALLOWANCE,
    note=(
        "The standalone repeat 1 (223.832 s) is the worst observation of "
        "this smoke on this host and binds over both the contended wall "
        "(159.6 s) and the Wf-A133 baseline run's 150 s INNER cut-off. "
        "The budget is the OUTER subprocess budget; the inner gate script "
        "is given budget - 30 s so it can still write gate_result.json."
    ),
)


#: Every Wf-A135 budget, keyed by gate name.
SMOKE_BUDGETS: Dict[str, SmokeBudget] = {
    GATE_CHEMPROP_SMOKE: CHEMPROP_SMOKE_BUDGET,
    GATE_DMPNN_SMOKE: DMPNN_SMOKE_BUDGET,
    GATE_WF385_SMOKE: WF385_SMOKE_BUDGET,
}


def get_budget(gate_name: str) -> SmokeBudget:
    """Return the budget record for ``gate_name``.

    Raises ``KeyError`` for an unknown gate — a typo must not silently
    fall back to some other smoke's budget.
    """
    try:
        return SMOKE_BUDGETS[gate_name]
    except KeyError:
        raise KeyError(
            f"unknown Wf-A135 smoke gate {gate_name!r}; known gates: "
            f"{sorted(SMOKE_BUDGETS)}"
        ) from None


def smoke_budget_sec(
    gate_name: str,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> float:
    """Resolve the EFFECTIVE wall budget for ``gate_name``, in seconds.

    Resolution order (highest priority first):

    1. the NAMED environment variable for this gate — e.g.
       ``MOLMETAL_SMOKE_BUDGET_CHEMPROP_SEC``;
    2. the NAMED default constant in this module.

    A malformed or non-positive override is REPORTED on stderr and then
    IGNORED (the default budget is used).  Silently turning a typo'd
    env var into a 0-second budget would make every smoke report an
    instant timeout — a false verdict dressed as a measurement.
    """
    budget = get_budget(gate_name)
    environ = os.environ if env is None else env
    raw = environ.get(budget.env_var)
    if raw is None or not str(raw).strip():
        return budget.budget_sec
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        print(
            f"[wfa135] WARN: ignoring malformed {budget.env_var}={raw!r} "
            f"(not a number) — using the default budget "
            f"{budget.budget_sec:.0f}s",
            file=sys.stderr,
        )
        return budget.budget_sec
    if not (value > 0.0):
        print(
            f"[wfa135] WARN: ignoring non-positive {budget.env_var}={raw!r} "
            f"— using the default budget {budget.budget_sec:.0f}s",
            file=sys.stderr,
        )
        return budget.budget_sec
    return value


def load_measured_fixture() -> Dict[str, object]:
    """Load the machine-readable mirror of the MEASURED baselines.

    Raises ``FileNotFoundError`` / ``ValueError`` — a missing or corrupt
    fixture is a hard error, not a silently-empty table.
    """
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "smokes" not in payload:
        raise ValueError(
            f"{FIXTURE_PATH} does not carry a 'smokes' object; it cannot "
            f"mirror the Wf-A135 budget table"
        )
    return payload


def fixture_matches_budgets() -> Dict[str, bool]:
    """Compare the fixture mirror against this module's budget table.

    Returns ``{gate_name: agrees}``.  The module is authoritative; the
    fixture is the machine-readable mirror, and a mismatch means one of
    them was edited without the other.
    """
    payload = load_measured_fixture()
    smokes = payload.get("smokes") or {}
    out: Dict[str, bool] = {}
    for gate_name, budget in SMOKE_BUDGETS.items():
        row = smokes.get(gate_name)
        if not isinstance(row, dict):
            out[gate_name] = False
            continue
        out[gate_name] = (
            abs(float(row.get("budget_sec", -1)) - budget.budget_sec) < 1e-9
            and abs(
                float(row.get("measured_baseline_sec", -1))
                - budget.measured_baseline_sec
            )
            < 1e-9
            and abs(float(row.get("headroom_factor", -1)) - budget.headroom_factor)
            < 1e-9
        )
    return out


def derived_budget_sec(worst_observed_sec: float) -> float:
    """The derivation itself, in code: ``ceil_0.1(Hw * R)``.

    Exposed so that the arithmetic behind every constant can be
    re-computed rather than taken on trust.  ``R`` is
    ``HEADROOM_ALLOWANCE``.
    """
    return math.ceil(worst_observed_sec * HEADROOM_ALLOWANCE * 10 - 1e-9) / 10


if __name__ == "__main__":  # pragma: no cover — manual inspection helper
    print(json.dumps({k: v.to_dict() for k, v in SMOKE_BUDGETS.items()}, indent=2))
    print(f"fixture={FIXTURE_PATH}")
    print(f"fixture_matches_budgets={fixture_matches_budgets()}")
