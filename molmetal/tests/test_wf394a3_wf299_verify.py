"""Wf-394-A3 — tests for Wf-299 pre-launch verification gate.

Per TODO-207 + AGENTS.md §5.1 + Wf-378-A2 lesson + Wf-385-A2 wrapper review:
the Wf-299 CFM 3-wave launcher must satisfy 4 structural contracts
(USER_OK gate present, pre-flight chain wired, 3-wave split, Wf-388
flags pass-through).

Tests:
  1. test_user_ok_gate — USER_OK gate present + before --dry-run block.
  2. test_3_wave_split — smoke / full / ablation waves present in case stmt.
  3. test_w388_flags — Wf-388 diversity + SA flags pass through CFM_NEW_FLAGS.

Each test exercises ONE verify_* function in isolation + asserts the
structural correctness + emits a JSON dump for downstream consumers
(paper §6 reproducibility audits + pre-flight dashboard).

CPU-only by design. 0 GPU launches. ~3-5 s wall on a 4-core host.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


# Repo root — derived from this file's own location (molmetal/tests/X.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Add scripts dir to path so we can import the gate module.
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import wf394a3_wf299_pre_launch_verify as gate  # noqa: E402

# The TRACKED Wf-299 launcher lives in the repo-root scripts/ dir (added
# 2026-10-02, commit bc29aa90); the legacy molmetal/scripts/ location it
# replaced is gone from disk and from git history.  The gate module's own
# canonical-path constant still points at that legacy location, so redirect
# it here (path-only fix — the assertions below are unchanged).
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_wf299_cfm_main.sh"
gate.LAUNCHER = LAUNCHER_PATH


# --- Test 1: USER_OK gate ----------------------------------------------------
def test_user_ok_gate():
    """USER_OK gate present + fires BEFORE --dry-run block.

    Per Wf-378-A2 G6 lesson: the gate must fire BEFORE the --dry-run
    exit block, otherwise --dry-run bypasses the gate and the launcher
    can be silently auto-launched without per-launch user OK.
    """
    result = gate.verify_wf299_user_ok_gate(LAUNCHER_PATH)

    # After Wf-394-A3 wrapper edit, the gate MUST be present.
    assert result["gate_present"], (
        f"USER_OK gate missing in {LAUNCHER_PATH}; "
        f"AGENTS.md §5.1 + Wf-378-A2 G6 lesson require it. "
        f"evidence: {result['evidence']}"
    )
    assert result["gate_fires_before_dry_run"], (
        f"USER_OK gate must fire BEFORE --dry-run block; "
        f"Wf-378-A2 G6 lesson: --dry-run was bypassing the gate. "
        f"evidence: {result['evidence']}"
    )
    assert result["gate_exit_code_64_present"], (
        f"USER_OK gate must exit 64 (per Wf-366-A1 + Wf-378-A2 family); "
        f"evidence: {result['evidence']}"
    )

    # Persist for downstream consumers.
    out_dir = REPO_ROOT / "molmetal" / "reports" / "wf394a3_wf299_pre_launch"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_user_ok_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


# --- Test 2: 3-wave split ----------------------------------------------------
def test_3_wave_split():
    """3 waves (smoke / full / ablation) present in case statement + smoke path coverage.

    Per Wf-299 brief + TODO-207 §3.2..§3.4: Wave A = P1.4 smoke,
    Wave B = G6 no_wass smoke, Wave C = G6 full sweep.
    """
    result = gate.verify_wf299_3_wave_separation(LAUNCHER_PATH)

    # All 3 wave functions must be defined.
    assert sorted(result["waves_present"]) == [
        "A",
        "B",
        "C",
    ], f"Expected all 3 waves present (A, B, C), got: {result['waves_present']}"

    # Wave A must include CFM_NEW_FLAGS expansion (smoke path coverage).
    assert result["wave_a_has_cfm_flags"], (
        f"Wave A must include CFM_NEW_FLAGS expansion; "
        f"per TODO-207 §2.5 smoke 50-step training coverage. "
        f"evidence: {result['evidence']}"
    )

    # Wave B must use --wasserstein-weight 0.0 (no_wass smoke).
    assert result["wave_b_no_wass"], (
        f"Wave B must use --wasserstein-weight 0.0 (no_wass smoke); "
        f"evidence: {result['evidence']}"
    )

    # Wave C must reference 60-run full sweep enumeration.
    assert result["wave_c_full_sweep"], (
        f"Wave C must reference 60-run full sweep (per g6 yaml §runs.matrix); "
        f"evidence: {result['evidence']}"
    )

    # Case statement must include all 3 waves + 'all'.
    assert result["case_statement_complete"], (
        f"Case statement must dispatch on A, B, C, and 'all'; "
        f"evidence: {result['evidence']}"
    )

    # Persist for downstream consumers.
    out_dir = REPO_ROOT / "molmetal" / "reports" / "wf394a3_wf299_pre_launch"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_3_wave_split.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


# --- Test 3: Wf-388 flags pass-through ---------------------------------------
def test_w388_flags():
    """Wf-388 diversity + SA flags pass through CFM_NEW_FLAGS to trainer.

    Per Wf-392-A3: Wave A smoke invocation must include the 16 NEW
    trainer flags (13 Wf-380 + 3 Wf-388). --flowmol-unified is
    intentionally NOT in the pass-through list (LICENSE-GATED CC-BY-NC).
    """
    result = gate.verify_wf299_w388_flags(LAUNCHER_PATH)

    # All 16 expected flags must be in CFM_NEW_FLAGS array.
    assert result["all_present"], (
        f"Expected all {result['expected_flags_count']} CFM flags in "
        f"CFM_NEW_FLAGS, found {result['cfm_new_flags_count']}; "
        f"missing: {result['missing_flags']}"
    )

    # --flowmol-unified must be intentionally skipped (LICENSE-GATED).
    assert result["flowmol_intentionally_skipped"], (
        f"--flowmol-unified must be intentionally skipped (CC-BY-NC LICENSE-GATED); "
        f"evidence: {result['evidence']}"
    )

    # Wave A invocations must expand ${CFM_NEW_FLAGS[@]}.
    assert result["wave_a_expansion_present"], (
        f"Wave A must expand ${{CFM_NEW_FLAGS[@]}}; " f"evidence: {result['evidence']}"
    )

    # Verify the 3 NEW Wf-388 flags are explicitly in the array.
    src = LAUNCHER_PATH.read_text(encoding="utf-8")
    for flag in gate.CFM_WF388_FLAGS:
        # The flag name should appear in the array literal.
        assert flag in src, f"Wf-388 flag {flag} missing from launcher source"

    # Persist for downstream consumers.
    out_dir = REPO_ROOT / "molmetal" / "reports" / "wf394a3_wf299_pre_launch"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_w388_flags.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


# --- Optional: end-to-end main() smoke ---------------------------------------
def test_main_entrypoint_runs():
    """End-to-end main() smoke: returns 0 or 1, never crashes, emits result.json.

    Wf-A269 wave-1 item C — two holes are closed here:

    1. **A crash used to be indistinguishable from a legitimate FAIL.**
       ``main()`` returns 0 on PASS and 1 on FAIL, but an UNHANDLED
       exception also exits 1.  The old assertion
       ``proc.returncode in (0, 1)`` therefore passed for a crashed
       subprocess.  The contract the docstring claims ("never crash") was
       not the contract the assertion tested.  The fix asserts the absence
       of a Python traceback in stderr, which is what actually separates
       the two outcomes.

    2. **A STALE result.json used to satisfy the existence check.**
       The artifact lives at a fixed path in the repo, so a file left by
       any earlier run made ``result_json.exists()`` true even when
       ``main()`` wrote nothing at all.  The fix records the pre-run
       mtime and requires the file to be NEWER, so the assertion can only
       be satisfied by THIS run's output.
    """
    result_json = (
        REPO_ROOT / "molmetal" / "reports" / "wf394a3_wf299_pre_launch" / "result.json"
    )
    # Pre-run stamp of the artifact (ns since the epoch).  A missing file
    # is stamped 0 so "the file did not exist before" is also covered.
    pre_mtime_ns = result_json.stat().st_mtime_ns if result_json.exists() else 0

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "wf394a3_wf299_pre_launch_verify.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # main() returns 0 on PASS, 1 on FAIL — both are valid outcomes.
    assert proc.returncode in (0, 1), (
        f"main() returned unexpected exit code {proc.returncode}: "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    # ... but a crash ALSO exits 1.  The contract is "never crash", so the
    # traceback is what must be absent.
    assert "Traceback (most recent call last)" not in proc.stderr, (
        f"main() CRASHED (unhandled exception) — exit code {proc.returncode} "
        f"alone cannot tell a crash from a legitimate FAIL: "
        f"stderr={proc.stderr!r}"
    )
    assert result_json.exists(), "result.json not emitted by main()"
    assert result_json.stat().st_mtime_ns > pre_mtime_ns, (
        "result.json was NOT (re)written by this run — the file on disk is a "
        f"stale leftover (mtime_ns={result_json.stat().st_mtime_ns}, "
        f"pre-run={pre_mtime_ns})"
    )
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert "verdict" in payload
    assert "checks" in payload
    assert "details" in payload
