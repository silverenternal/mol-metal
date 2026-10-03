"""Wf-356-A2 — anchor refresh verification (2 NEW tests; frozen-aware since 2026-10-03).

================================================================
Per the 2026-09-29 ultracode wave (user directive "基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，
工作流可以长"), the Wf-345-C1..C4 closeouts refreshed the 4 stale
anchors catalogued by the Wf-345-A3 audit
(``molmetal/reports/wf345a3_anchor_audit/audit.json``).  The 2026-10-03
anchor-scope fix (defect 2) then moved all 11 historical anchors —
including these 4 — onto ``ANCHOR_FROZEN_PATHS``: they are shipped
historical reports with no producer in today's pipeline, so a wall-clock
``age < 24h`` floor is the wrong contract (it would re-fail within 24 h
of any refresh and could never be legitimately repaired — touching an
anchor is fabricated freshness, AGENTS.md §5.2).  This file now pins the
post-fix invariant: the 4 refresh-target anchors exist AND are listed as
frozen-exempt, and the pre-flight gate
``p3_wf343c4_anchor_freshness_24h`` (launch pipeline; Wf-A16) returns a
non-FAIL verdict — PASS on a fresh live anchor set, or a loud SKIP when
the mtime scope contains only frozen historical anchors.

Tests
-----

1. **test_wf356a2_4_refresh_target_anchors_frozen_exempt** — assert
   each of the 4 refresh-target anchors exists on disk AND appears in
   ``ANCHOR_FROZEN_PATHS`` (staleness is documented history, not an
   unnoticed regression).  Live anchors (not on the frozen list) keep
   the 24 h mtime contract — pinned in test_wf343c4_integration.py.

2. **test_wf356a2_preflight_anchor_freshness_gate_passes** — invoke
   the pre-flight check script as a subprocess (``--pipeline launch``,
   the pipeline the gate is registered on per Wf-A16) and assert the
   JSON output carries ``"name": "p3_wf343c4_anchor_freshness_24h"``
   with status in {PASS, SKIP} and NEVER a wall-clock FAIL on frozen
   history.  A FAIL fails the test with the gate's full evidence string.

Backward-compat verdict
-----------------------
The 2 NEW tests are ADDITIVE only — they do NOT modify any
pre-existing test / gate / runner / path. The
``EXPECTED_GATE_COUNTS`` invariants are unchanged because
these tests live in a new file (do NOT register any gate).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* All 4 anchors age < 24h MEASURED at the Wf-356-A2 measurement window
  (mtime inspection); they are frozen-exempt since 2026-10-03 because
  no producer in today's pipeline regenerates them.
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
# Test 1 — the 4 refresh-target anchors exist AND are frozen-exempt
# ---------------------------------------------------------------------------


def test_wf356a2_4_refresh_target_anchors_frozen_exempt():
    """The 4 refresh-target anchors exist AND are listed in ANCHOR_FROZEN_PATHS.

    Post-2026-10-03 contract (defect-2 fix): these shipped historical
    reports have no producer in today's pipeline, so they are exempt
    from the 24 h mtime budget — their staleness is documented history,
    NOT an unnoticed regression. If one disappears from disk, or a
    future edit drops it from ``ANCHOR_FROZEN_PATHS`` without adding a
    live producer, this test FAILs with the exact path + defect class.
    Live anchors (not on the frozen list) keep the 24 h contract; that
    surface is pinned in test_wf343c4_integration.py.
    """
    sys.path.insert(0, str(REPO_ROOT))
    import molmetal.scripts.pre_flight_check as preflight

    frozen = {rel for rel, _reason in preflight.ANCHOR_FROZEN_PATHS}
    problems = []
    for path_rel in REFRESH_TARGET_ANCHORS:
        abs_path = REPO_ROOT / path_rel
        if not abs_path.exists():
            problems.append((path_rel, "MISSING"))
        elif path_rel not in frozen:
            problems.append((path_rel, "NOT_FROZEN"))
    assert not problems, (
        f"refresh-target anchor contract drift: {problems}; expected every "
        f"path to exist on disk AND to be listed in ANCHOR_FROZEN_PATHS "
        f"(otherwise the live 24h mtime budget applies and can never be "
        f"legitimately satisfied — no producer regenerates these files)."
    )
    # Informational: ages at this run (deliberately NOT a freshness assert).
    for path_rel in REFRESH_TARGET_ANCHORS:
        age_hr = _classify_age_hr(path_rel)
        print(f"  {path_rel} age={age_hr:.4f}h (frozen-exempt)")


# ---------------------------------------------------------------------------
# Test 2 — pre-flight anchor_freshness gate returns a non-FAIL verdict
# ---------------------------------------------------------------------------


def test_wf356a2_preflight_anchor_freshness_gate_passes():
    """Pre-flight ``p3_wf343c4_anchor_freshness_24h`` gate is PASS or SKIP.

    Invokes ``pre_flight_check.py --pipeline launch`` as a subprocess
    (Wf-A16: the gate is launch-only — a commit is not a launch) and
    parses the JSON output. Expected verdicts: PASS (the live anchor set
    is fresh) or SKIP (the mtime scope contains only frozen historical
    anchors — the documented 2026-10-03 remedy). A wall-clock FAIL on
    frozen history is exactly the defect this test pins shut; the FAIL
    branch prints the gate's full evidence string for triage.
    """
    cmd = [
        PYTHON_BIN,
        str(PRE_FLIGHT_SCRIPT.relative_to(REPO_ROOT)),
        "--pipeline",
        "launch",
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
    status = gate.get("status")
    assert status in {"PASS", "SKIP"}, (
        f"gate {ANCHOR_FRESHNESS_GATE_NAME} returned {status}; expected PASS "
        f"(live anchors fresh) or SKIP (frozen-only mtime scope — "
        f"2026-10-03 scope fix), never a wall-clock FAIL on frozen history. "
        f"evidence={gate.get('evidence', '<none>')}; full_gate={gate}"
    )
    evidence = str(gate.get("evidence", ""))
    assert "frozen" in evidence, (
        f"gate {ANCHOR_FRESHNESS_GATE_NAME} evidence must name the "
        f"frozen-vs-live scope explicitly so the SKIP is loud, not silent; "
        f"evidence={evidence!r}"
    )
