"""Pytest: Wf-358-A2 — Cascade-fix tests for the STALE pre-flight JSON
``p3_wf343c2_per_wave_runtime_check`` regression.

Root cause (per Wf-357-A3 finding)
----------------------------------
``molmetal/reports/wf259_run/pre_flight.json`` was last refreshed 24.04 h
ago (88581.5 s) which exceeded the 24 h cap (86400 s) enforced by
``Wf-302A AnchorGuard`` (``--max-age-sec`` default). The STALE flag
cascaded into the Wf-343C2 per-wave runtime check (one of the 7
sub-checks is ``preflight_freshness_24h``), which itself is wired as
the pre-flight gate ``p3_wf343c2_per_wave_runtime_check``. The cascade
also touched ``preflight_no_fail_gates`` + ``wave_budget_remaining_nonneg``
via downstream coupling.

Wf-358-A2 closes the cascade by:
  1. Refreshing the pre-flight JSON via
     ``python molmetal/scripts/pre_flight_check.py --pipeline all``
     (24 h cap re-armed).
  2. Verifying ``preflight_freshness_24h`` returns PASS (cascade-fix
     step 1 — the STALE → fresh transition).
  3. Verifying the wf343c2 per-wave check invokes cleanly with the
     gate's default args (``--pairs-metals Ru,Ir,Os,Rh,Re`` ×
     ``--pairs-seeds 0,1,2`` × ``--budget-hr 10.0`` × ``--skip-live
     --strict``) and emits ``preflight_freshness_24h: PASS``.

This module ships two regression tests:

1. ``test_preflight_json_age_hr_below_24h`` — the refreshed pre-flight
   JSON has ``mtime - now`` ≤ 86400 s (i.e. < 24 h since refresh).
2. ``test_wf343c2_cascade_fix_preflight_freshness_24h_pass`` — invoking
   ``wf343c2_per_wave_check.py`` with the gate's exact args returns
   ``preflight_freshness_24h: PASS`` evidence (the cascade-fix
   contract).

Both tests are CPU-only and finish in <10 s on RTX 5090 / Blackwell.

Source: Wf-358-A2 closeout + Wf-357-A3 finding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREFLIGHT_JSON = PROJECT_ROOT / "molmetal" / "reports" / "wf259_run" / "pre_flight.json"
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
WF343C2_SCRIPT = SCRIPTS_DIR / "wf343c2_per_wave_check.py"
VENV_PY = Path("/home/hugo/codes/mol-metal/.venv/bin/python")
PREFLIGHT_FRESHNESS_CAP_SEC = 86400  # 24 h per Wf-302A AnchorGuard


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_preflight_json_age_hr_below_24h() -> None:
    """The pre-flight JSON has ``mtime - now`` ≤ 24 h (86400 s).

    This is the primary Wf-358-A2 contract: the Wf-357-A3 cascade
    regression was caused by the JSON being STALE (> 24 h old) so the
    Wf-302A AnchorGuard refused PASS. Wf-358-A2 re-runs
    ``pre_flight_check.py --pipeline all`` so the mtime is refreshed
    and the cap is re-armed.
    """
    assert PREFLIGHT_JSON.is_file(), f"pre_flight.json missing: {PREFLIGHT_JSON}"
    mtime = PREFLIGHT_JSON.stat().st_mtime
    now = os.path.getmtime(PREFLIGHT_JSON.parent)  # fallback if needed
    import time

    age_sec = time.time() - mtime
    assert age_sec <= PREFLIGHT_FRESHNESS_CAP_SEC, (
        f"pre_flight.json is STALE: age={age_sec:.1f}s "
        f"(cap={PREFLIGHT_FRESHNESS_CAP_SEC}s = 24h)"
    )
    # Sanity: also verify JSON parses and has the expected top-level keys
    with PREFLIGHT_JSON.open() as f:
        data = json.load(f)
    for key in ("timestamp", "gates", "summary", "schema_version"):
        assert key in data, f"pre_flight.json missing key: {key}"


@pytest.mark.skipif(not VENV_PY.exists(), reason="vlm venv not on disk")
def test_wf343c2_cascade_fix_preflight_freshness_24h_pass() -> None:
    """The wf343c2 per-wave check emits ``preflight_freshness_24h: PASS``.

    This is the cascade-fix contract: after Wf-358-A2 refreshes the
    pre-flight JSON, the wf343c2 per-wave check (invoked with the gate's
    exact default args: ``--pairs-metals Ru,Ir,Os,Rh,Re`` × ``--pairs-seeds
    0,1,2`` × ``--budget-hr 10.0`` × ``--skip-live --strict``) MUST emit
    ``preflight_freshness_24h`` as PASS.

    The output JSON is written to a temp path and re-read here so we can
    assert on the structured ``checks[]`` array.
    """
    assert WF343C2_SCRIPT.is_file(), f"wf343c2 script missing: {WF343C2_SCRIPT}"
    out_path = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf343c2_per_wave_check"
        / ("wf358a2_cascade_fix_test.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(VENV_PY),
        str(WF343C2_SCRIPT),
        "--wave",
        "Wf-358A2 cascade-fix smoke",
        "--pairs-metals",
        "Ru,Ir,Os,Rh,Re",
        "--pairs-seeds",
        "0,1,2",
        "--budget-hr",
        "10.0",
        "--skip-live",
        "--strict",
        "--preflight",
        str(PREFLIGHT_JSON),
        "--out",
        str(out_path),
    ]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    # The script may return non-zero (rc=1 if any FAIL, rc=2 if --strict
    # + any SKIP); what we care about is the structured JSON output.
    assert out_path.is_file(), (
        f"wf343c2 did not write {out_path}: "
        f"stdout={result.stdout[:200]} stderr={result.stderr[:200]}"
    )
    with out_path.open() as f:
        report = json.load(f)
    checks_by_id = {c["check_id"]: c for c in report["checks"]}
    assert "preflight_freshness_24h" in checks_by_id, (
        f"preflight_freshness_24h missing from report checks: "
        f"{list(checks_by_id.keys())}"
    )
    freshness = checks_by_id["preflight_freshness_24h"]
    assert freshness["status"] == "PASS", (
        f"CASCADE FIX FAILED: preflight_freshness_24h status="
        f"{freshness['status']!r} evidence={freshness.get('evidence', '')!r}"
    )
    # Cascade-fix evidence string includes "preflight age ...s ≤ 86400s"
    assert "86400" in freshness.get(
        "evidence", ""
    ), f"unexpected evidence format: {freshness.get('evidence')!r}"
