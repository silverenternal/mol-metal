"""Wf-356-A2 — anchor refresh verification (2 NEW tests).

================================================================
Per the 2026-09-29 ultracode wave (user directive "基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，
工作流可以长"), the Wf-345-C1..C4 closeouts refreshed the 4 stale
anchors catalogued by the Wf-345-A3 audit
(``molmetal/reports/wf345a3_anchor_audit/audit.json``). This file
pins the post-refresh invariant: every refresh-target anchor has
``age_hr < 24`` (mtime-based, self-contained — does NOT depend on
the audit JSON) AND the pre-flight gate ``p3_wf343c4_anchor_freshness_24h``
returns PASS in a fresh-run.

Tests
-----

1. **test_wf356a2_4_anchors_age_under_24h** — assert each of the 4
   refresh-target anchors has ``age_hr < 24`` using mtime inspection
   (real filesystem mtime, not the audit JSON). If any anchor drifts
   back to ``age_hr >= 24``, this test FAILs with a clear drift
   report naming the stale path(s) and their exact age.

2. **test_wf356a2_preflight_anchor_freshness_gate_passes** — invoke
   the pre-flight check script as a subprocess (the canonical gate
   consumer path; no import-time side effects on the gate registry)
   and assert the JSON output carries ``"name": "p3_wf343c4_anchor_freshness_24h"``
   with ``"status": "PASS"``. If the gate FAILs, this test FAILs with
   the gate's full evidence string so investigators can debug.

Backward-compat verdict
-----------------------
The 2 NEW tests are ADDITIVE only — they do NOT modify any
pre-existing test / gate / runner / path. The
``EXPECTED_GATE_COUNTS["all"] == 62`` invariant is unchanged because
these tests live in a new file (do NOT register any gate).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* All 4 anchors age < 24h MEASURED (mtime inspection at Wf-356-A2
  measurement window).
* Drift vs predecessor: 2 anchors bit-identical (NO_DRIFT), 1 anchor
  HONEST-NEGATIVE ~7e-3 (Wf-170, CUDA non-determinism — documented
  in Wf-345-C3), 1 anchor expected 5→100 pocket expansion (Wf-67).
* 0 GPU launches performed.
"""

import json
import pathlib
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants — keep aligned with Wf-345-A3 audit + Wf-345-C1..C4 closeouts
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The 4 refresh-target anchors from the Wf-345-A3 audit (relative to REPO_ROOT).
# Order matches the Wf-345-C1..C4 closeout numbering.
REFRESH_TARGET_ANCHORS = (
    "molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json",
    "molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json",
    "molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json",
    "molmetal/reports/wf170_permetal_singletask/sweep_summary.json",
)

MAX_AGE_SEC = 24 * 3600.0
MAX_AGE_HR = 24

PYTHONPATH = "/home/hugo/codes/mol-metal"
PYTHON_BIN = "/home/hugo/codes/mol-metal/.venv/bin/python"
PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
ANCHOR_FRESHNESS_GATE_NAME = "p3_wf343c4_anchor_freshness_24h"


def _classify_age_hr(path_rel):
    """Return ``age_hr`` for an anchor file (resolved relative to REPO_ROOT)."""
    abs_path = REPO_ROOT / path_rel
    mtime = abs_path.stat().st_mtime
    age_sec = time.time() - mtime
    return age_sec / 3600.0


# ---------------------------------------------------------------------------
# Test 1 — all 4 refresh-target anchors age < 24h
# ---------------------------------------------------------------------------


def test_wf356a2_4_anchors_age_under_24h():
    """All 4 refresh-target anchors have age_hr < 24 at Wf-356-A2 measurement window.

    If this test FAILs, one of the 4 anchors has drifted back over the
    24h freshness budget. The failure message names the stale path(s)
    and the exact age in hours so investigators can decide whether to
    re-run the sweep script (per Wf-345-C1..C4 closeouts) or accept
    the staleness as expected (e.g. external-dependency blocked).
    """
    stale = []
    for path_rel in REFRESH_TARGET_ANCHORS:
        abs_path = REPO_ROOT / path_rel
        if not abs_path.exists():
            stale.append((path_rel, "MISSING"))
            continue
        age_hr = _classify_age_hr(path_rel)
        if age_hr >= MAX_AGE_HR:
            stale.append((path_rel, age_hr))
    assert not stale, (
        f"anchor freshness drift: {len(stale)} of {len(REFRESH_TARGET_ANCHORS)} "
        f"refresh-target anchors are now stale (age >= 24h); "
        f"re-run the Wf-345-C1..C4 sweep scripts to refresh; "
        f"stale paths: {stale}"
    )
    # Sanity: print ages for the convenience of investigators.
    for path_rel in REFRESH_TARGET_ANCHORS:
        age_hr = _classify_age_hr(path_rel)
        print(f"  {path_rel} age={age_hr:.4f}h")


# ---------------------------------------------------------------------------
# Test 2 — pre-flight anchor_freshness gate returns PASS
# ---------------------------------------------------------------------------


def test_wf356a2_preflight_anchor_freshness_gate_passes():
    """Pre-flight v3 ``p3_wf343c4_anchor_freshness_24h`` gate returns PASS.

    Invokes ``pre_flight_check.py --pipeline all`` as a subprocess and
    parses the JSON output. The gate must be present with
    ``status == "PASS"``. If the gate is FAIL, the test FAILs with the
    full evidence string so investigators can see why (which anchors
    are stale, etc.).
    """
    cmd = [
        PYTHON_BIN,
        str(PRE_FLIGHT_SCRIPT.relative_to(REPO_ROOT)),
        "--pipeline",
        "all",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env={"PYTHONPATH": PYTHONPATH, "PATH": "/usr/bin:/bin"},
    )
    # The pre-flight may exit non-zero when OTHER gates FAIL (e.g.
    # p2_tests_pass when some pytest tests regress). We only care
    # about the anchor_freshness_24h gate, so we parse the JSON
    # regardless of overall rc.
    out = proc.stdout
    json_start = out.find("{")
    assert json_start >= 0, (
        f"pre_flight_check.py stdout has no JSON object start; " f"stdout={out[:500]}"
    )
    payload = json.loads(out[json_start:])
    # Locate the gate by name.
    if isinstance(payload, dict) and "gates" in payload:
        gate_list = payload["gates"]
    elif isinstance(payload, list):
        gate_list = payload
    else:
        gate_list = []
    found = [g for g in gate_list if g.get("name") == ANCHOR_FRESHNESS_GATE_NAME]
    assert found, (
        f"gate {ANCHOR_FRESHNESS_GATE_NAME} not found in pre-flight JSON; "
        f"gate_names={[g.get('name') for g in gate_list][:20]}; "
        f"stdout_tail={out[-500:]}"
    )
    gate = found[0]
    assert gate.get("status") == "PASS", (
        f"gate {ANCHOR_FRESHNESS_GATE_NAME} returned {gate.get('status')}; "
        f"evidence={gate.get('evidence', '<none>')}; full_gate={gate}"
    )
