"""Wf-363-B2 — Budget remaining indicator tests for GPU launch dashboard.

============================================================================
Owner:       Hugo Lee (autonomous Wf-363-B2 ultracode unit)
Date:        2026-09-29
Parent:      Wf-363 ultracode residual wave + AGENTS.md §5.1 standing GPU auth +
             Wf-262 §7.1 budget discipline.

Per Wf-262 §7.1 (TODO-208 budget discipline): the GPU launch dashboard
must surface a "budget remaining" indicator so operators can see at-a-glance
how much of the 10h session cap each launcher consumes, plus the
DEFERRED_TEMP wave aggregate from wave_budget_tracker.py (per Wf-360-A1
rebalance).

This file delivers 2 NEW tests:

  - test_budget_status_schema
        Validates the budget_status JSON schema emitted by
        `molmetal/scripts/gpu_launch_aggregator.py`. Required fields:
          session_cap_hr : 10.0
          per_launcher   : list with one entry per known launcher
                           ({launcher, budget_hr, remaining_hr,
                            status, status_evidence, wall_projected_hr,
                            pair_progress_share, launcher_status})
          total_projected_hr : float
          total_remaining_hr : float
          deferred_temp_total_hr : float
          deferred_temp_waves   : list[dict]
          deferred_temp_remaining_hr : float
          honest_framing : str

  - test_per_launcher_remaining_hr_computation
        For every launcher, asserts:
          remaining_hr == session_cap_hr - budget_hr
          status == "WITHIN" iff remaining_hr >= 0 else "OVER"
        Also verifies that the SHELL script's --budget-json output emits
        the same schema parity (per Wf-363-B2 cross-language parity
        contract).

All 2 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU
launches; aggregator + shell dashboard are read-only). All numeric values
trace to live-run output; no fabricated claims.

Cross-references
----------------
- `molmetal/scripts/gpu_launch_aggregator.py`  — Python aggregator (budget_status)
- `scripts/gpu_launch_dashboard.sh`           — Bash dashboard (--budget-json)
- `molmetal/scripts/wave_budget_tracker.py`    — DEFERRED_TEMP source of truth
- `TODO/completed/363b2_budget_indicator.md`   — Wf-363-B2 ship doc.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGGREGATOR_PY = PROJECT_ROOT / "molmetal" / "scripts" / "gpu_launch_aggregator.py"
DASHBOARD_SH = PROJECT_ROOT / "scripts" / "gpu_launch_dashboard.sh"
DASHBOARD_JSON = PROJECT_ROOT / "molmetal" / "reports" / "gpu_launch_dashboard.json"
WAVE_BUDGET_REPORT = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf262_wave_budget"
    / "wave_budget_report.json"
)

EXPECTED_LAUNCHERS = {
    "wf355_a1",
    "wf355_a2",
    "wf355_a3",
    "wf355_b1",
    "wf355_b2",
    "g5_retrain",
    "g6_recover",
}

# Per-launcher registered wall_projected_hr (Wf-355; PROJECTED).
# Source of truth: LAUNCHERS list in gpu_launch_aggregator.py.
EXPECTED_BUDGETS_HR = {
    "wf355_a1": 10.0,
    "wf355_a2": 7.0,
    "wf355_a3": 12.0,
    "wf355_b1": 5.0,
    "wf355_b2": 3.0,
    "g5_retrain": 5.0,
    "g6_recover": 11.4,
}

SESSION_CAP_HR = 10.0  # Per Wf-262 §7.1 budget discipline.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_aggregator():
    """Import the aggregator module via importlib (path-based, no install)."""
    import importlib.util as importlib_util

    spec = importlib_util.spec_from_file_location(
        "gpu_launch_aggregator_under_test", str(AGGREGATOR_PY)
    )
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_aggregator(extra_args: list[str] | None = None) -> dict:
    """Invoke the aggregator's CLI and parse its JSON stdout."""
    cmd = [sys.executable, str(AGGREGATOR_PY), "--stdout"]
    if extra_args:
        cmd += extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"aggregator failed: {proc.stderr}"
    return json.loads(proc.stdout)


def _run_shell_budget_json() -> dict:
    """Invoke the shell dashboard --budget-json flag and parse stdout."""
    proc = subprocess.run(
        ["bash", str(DASHBOARD_SH), "--budget-json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"shell dashboard failed: {proc.stderr}"
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Test 1 — budget_status JSON schema
# ---------------------------------------------------------------------------
def test_budget_status_schema():
    """Verify the budget_status JSON schema emitted by the aggregator.

    The aggregator must emit a top-level `budget_status` field with the
    following structure (Wf-363-B2 contract):

      budget_status: {
        session_cap_hr               : 10.0
        per_launcher                 : list[{launcher, budget_hr,
                                             remaining_hr, status,
                                             status_evidence,
                                             wall_projected_hr,
                                             pair_progress_share,
                                             launcher_status}, ...]
        total_projected_hr           : float
        total_remaining_hr           : float
        deferred_temp_total_hr       : float
        deferred_temp_waves          : list[{wave_name, deferred_class,
                                              hr, evidence}, ...]
        deferred_temp_remaining_hr   : float
        honest_framing               : str
      }

    This test imports the aggregator module, calls aggregate(), and
    validates every required field. It is CPU-only and runs in < 1 s.
    """
    # Import the aggregator module and call aggregate() in-process.
    mod = _import_aggregator()
    result = mod.aggregate()

    # Top-level budget_status field is present
    assert "budget_status" in result, (
        f"aggregator output missing 'budget_status' field; "
        f"top-level keys: {list(result.keys())}"
    )
    bs = result["budget_status"]

    # Required top-level fields
    required_top = {
        "session_cap_hr",
        "per_launcher",
        "total_projected_hr",
        "total_remaining_hr",
        "deferred_temp_total_hr",
        "deferred_temp_waves",
        "deferred_temp_remaining_hr",
        "honest_framing",
    }
    missing = required_top - set(bs.keys())
    assert not missing, f"budget_status missing fields: {missing}"

    # session_cap_hr = 10.0 (per Wf-262 §7.1)
    assert (
        bs["session_cap_hr"] == SESSION_CAP_HR
    ), f"session_cap_hr must equal {SESSION_CAP_HR}, got {bs['session_cap_hr']!r}"

    # per_launcher list has one entry per known launcher
    assert isinstance(
        bs["per_launcher"], list
    ), f"per_launcher must be a list, got {type(bs['per_launcher'])}"
    pl_names = {entry["launcher"] for entry in bs["per_launcher"]}
    assert pl_names == EXPECTED_LAUNCHERS, (
        f"per_launcher name set mismatch: got {pl_names}, "
        f"expected {EXPECTED_LAUNCHERS}"
    )

    # Per-launcher entry required fields
    required_pl = {
        "launcher",
        "budget_hr",
        "remaining_hr",
        "status",
        "status_evidence",
        "wall_projected_hr",
        "pair_progress_share",
        "launcher_status",
    }
    for pl in bs["per_launcher"]:
        miss = required_pl - set(pl.keys())
        assert (
            not miss
        ), f"per_launcher entry {pl.get('launcher')!r} missing fields: {miss}"
        # Status enum check
        assert pl["status"] in {"WITHIN", "OVER"}, (
            f"{pl['launcher']!r}: invalid status {pl['status']!r}; "
            "must be WITHIN or OVER"
        )

    # Numeric field types
    for k in (
        "total_projected_hr",
        "total_remaining_hr",
        "deferred_temp_total_hr",
        "deferred_temp_remaining_hr",
    ):
        assert isinstance(
            bs[k], (int, float)
        ), f"budget_status.{k} must be numeric, got {type(bs[k])}"
        assert (
            -1e9 <= bs[k] <= 1e9
        ), f"budget_status.{k} out of plausible range: {bs[k]}"

    # honest_framing is a non-empty string
    assert isinstance(
        bs["honest_framing"], str
    ), f"honest_framing must be a string, got {type(bs['honest_framing'])}"
    assert len(bs["honest_framing"]) > 0, "honest_framing must be non-empty"

    # deferred_temp_waves is a list of dicts with required fields
    assert isinstance(
        bs["deferred_temp_waves"], list
    ), f"deferred_temp_waves must be a list, got {type(bs['deferred_temp_waves'])}"
    required_w = {"wave_name", "deferred_class", "hr", "evidence"}
    for w in bs["deferred_temp_waves"]:
        miss = required_w - set(w.keys())
        assert not miss, f"deferred_temp_waves entry missing fields: {miss}"
        assert w["deferred_class"] == "DEFERRED_TEMP", (
            f"deferred_temp_waves entry class must be DEFERRED_TEMP, "
            f"got {w['deferred_class']!r}"
        )
        assert w["evidence"] in {
            "MEASURED_actual_hr",
            "PROJECTED_estimated_hr",
        }, f"evidence must be MEASURED or PROJECTED, got {w['evidence']!r}"

    # Total math invariant: sum of per-launcher budgets == total_projected_hr
    sum_budgets = sum(float(pl["budget_hr"]) for pl in bs["per_launcher"])
    assert abs(sum_budgets - bs["total_projected_hr"]) < 1e-3, (
        f"sum of per_launcher budgets ({sum_budgets}) must equal "
        f"total_projected_hr ({bs['total_projected_hr']})"
    )

    # Cross-language parity: --stdout --budget-summary produces the same
    # budget_status shape (only slightly truncated). Run CLI to verify.
    cli_budget = _run_aggregator(["--budget-summary"])
    assert cli_budget["session_cap_hr"] == SESSION_CAP_HR, (
        f"CLI --budget-summary session_cap_hr mismatch: "
        f"{cli_budget['session_cap_hr']!r}"
    )
    assert {
        pl["launcher"] for pl in cli_budget["per_launcher"]
    } == EXPECTED_LAUNCHERS, "CLI --budget-summary per_launcher mismatch"

    # On-disk dashboard JSON also carries the budget_status field.
    proc = subprocess.run(
        [sys.executable, str(AGGREGATOR_PY)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"aggregator on-disk run failed: {proc.stderr}"
    assert DASHBOARD_JSON.exists(), f"dashboard JSON not written: {DASHBOARD_JSON}"
    on_disk = json.loads(DASHBOARD_JSON.read_text())
    assert (
        "budget_status" in on_disk
    ), f"on-disk dashboard missing budget_status; keys: {list(on_disk.keys())}"
    assert on_disk["budget_status"]["session_cap_hr"] == SESSION_CAP_HR


# ---------------------------------------------------------------------------
# Test 2 — per-launcher remaining_hr computation
# ---------------------------------------------------------------------------
def test_per_launcher_remaining_hr_computation():
    """Verify remaining_hr = session_cap_hr - budget_hr per launcher.

    For each of the 7 known launchers:
      remaining_hr == session_cap_hr - budget_hr   (within 1e-6 tolerance)
      status      == "WITHIN"  iff remaining_hr >= 0
                  == "OVER"    iff remaining_hr < 0
      status_evidence is non-empty and human-readable

    The 7 launchers collectively PROJECTED = 53.4h vs the 10h cap, so
    `total_projected_hr` = 53.4 and `total_remaining_hr` = -43.4
    (HONEST-NEGATIVE — operators must schedule across sessions).

    Cross-language parity: the bash dashboard --budget-json output
    matches the python aggregator's budget_status field structure.
    """
    mod = _import_aggregator()
    result = mod.aggregate()
    bs = result["budget_status"]

    # Per-launcher math
    for pl in bs["per_launcher"]:
        name = pl["launcher"]
        assert name in EXPECTED_BUDGETS_HR, f"unknown launcher {name!r}"
        expected_budget = EXPECTED_BUDGETS_HR[name]
        expected_remaining = round(SESSION_CAP_HR - expected_budget, 6)
        expected_status = "WITHIN" if expected_remaining >= -1e-9 else "OVER"

        # remaining_hr math
        assert abs(pl["remaining_hr"] - expected_remaining) < 1e-3, (
            f"{name}: remaining_hr={pl['remaining_hr']} != "
            f"{SESSION_CAP_HR} - {expected_budget} = {expected_remaining}"
        )

        # budget_hr matches registry
        assert (
            abs(pl["budget_hr"] - expected_budget) < 1e-6
        ), f"{name}: budget_hr={pl['budget_hr']} != expected {expected_budget}"

        # wall_projected_hr mirrors budget_hr (alias for compatibility)
        assert abs(pl["wall_projected_hr"] - expected_budget) < 1e-6, (
            f"{name}: wall_projected_hr={pl['wall_projected_hr']} != "
            f"expected {expected_budget}"
        )

        # status enum matches math
        assert pl["status"] == expected_status, (
            f"{name}: status={pl['status']!r} != {expected_status!r} "
            f"(remaining_hr={pl['remaining_hr']})"
        )

        # status_evidence is informative
        ev = pl["status_evidence"]
        assert (
            isinstance(ev, str) and len(ev) > 0
        ), f"{name}: status_evidence must be a non-empty string, got {ev!r}"
        assert (
            str(int(expected_budget)) in ev or f"{expected_budget}" in ev
        ), f"{name}: status_evidence {ev!r} does not reference budget {expected_budget}"

    # Cross-launcher total math
    expected_total = round(sum(EXPECTED_BUDGETS_HR.values()), 6)
    assert (
        abs(bs["total_projected_hr"] - expected_total) < 1e-3
    ), f"total_projected_hr={bs['total_projected_hr']} != expected {expected_total}"
    expected_total_remaining = round(SESSION_CAP_HR - expected_total, 6)
    assert abs(bs["total_remaining_hr"] - expected_total_remaining) < 1e-3, (
        f"total_remaining_hr={bs['total_remaining_hr']} != "
        f"expected {expected_total_remaining}"
    )

    # Known OVER launchers: wf355_a3 (12.0h) + g6_recover (11.4h)
    over_launchers = {
        pl["launcher"] for pl in bs["per_launcher"] if pl["status"] == "OVER"
    }
    assert over_launchers == {
        "wf355_a3",
        "g6_recover",
    }, f"OVER launchers should be {{wf355_a3, g6_recover}}; got {over_launchers}"

    # Cross-language parity: shell --budget-json emits equivalent schema
    shell_budget = _run_shell_budget_json()
    assert shell_budget["session_cap_hr"] == SESSION_CAP_HR
    shell_over = {
        pl["launcher"] for pl in shell_budget["per_launcher"] if pl["status"] == "OVER"
    }
    assert shell_over == {
        "wf355_a3",
        "g6_recover",
    }, f"shell OVER launchers mismatch: {shell_over}"
    assert {
        pl["launcher"] for pl in shell_budget["per_launcher"]
    } == EXPECTED_LAUNCHERS, (
        f"shell per_launcher mismatch: got "
        f"{[pl['launcher'] for pl in shell_budget['per_launcher']]}"
    )
    # Per-launcher budget_hr matches
    shell_pl_by_name = {pl["launcher"]: pl for pl in shell_budget["per_launcher"]}
    for name, expected_budget in EXPECTED_BUDGETS_HR.items():
        assert shell_pl_by_name[name]["budget_hr"] == pytest.approx(
            expected_budget, abs=1e-3
        ), (
            f"shell budget_hr mismatch for {name}: "
            f"{shell_pl_by_name[name]['budget_hr']} != {expected_budget}"
        )

    # honest_framing mentions the OVER status (HONEST-NEGATIVE surfacing)
    assert (
        "OVER" in bs["honest_framing"] or "HONEST-NEGATIVE" in bs["honest_framing"]
    ), (
        f"honest_framing must surface OVER status / HONEST-NEGATIVE; got "
        f"{bs['honest_framing']!r}"
    )
    assert str(int(SESSION_CAP_HR)) in bs["honest_framing"], (
        f"honest_framing must reference session cap {SESSION_CAP_HR}h; "
        f"got {bs['honest_framing']!r}"
    )
