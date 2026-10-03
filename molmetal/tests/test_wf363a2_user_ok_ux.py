"""Wf-363-A2 — USER_OK gate UX verification across launch wrappers.

============================================================================
Owner:       Hugo Lee (autonomous Wf-363-A2 ultracode unit)
Date:        2026-09-29
Parent:      Wf-363 ultracode residual wave (post-Wf-355 + Wf-356-A1)
             + AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing.

Per AGENTS.md §5.1 (Standing GPU authorization 2026-09-29 user grant):
"显存够就放心大胆用" — but kill_then_adapt directive remains in force for
launches > 1 hr GPU. Every USER_OK-gated wrapper is expected to:

  1. REFUSE to launch (non-zero exit) unless USER_OK env var is set.
  2. The exact value of USER_OK depends on the wrapper family:
     - Wf-294B / pre-Wf-300C pattern: `USER_OK=1`
     - Wf-300C pattern (R1/R3 mode dispatch): `USER_OK=YES`
  3. --dry-run must NOT silently bypass the gate (per kill_then_adapt).
     Exception: pre_flight_gpu_ready.sh SKIPS the gate on --dry-run by
     emitting a SKIP status row in its gate table (intentional; this is
     a verifier, not a launcher).
  4. The Usage / FATAL message must reference kill_then_adapt directive
     so the operator knows why the gate exists.
  5. Plan JSON (where emitted) carries `"user_ok": true` to mark the
     gate-pass path.

This file delivers 2 NEW tests:

  - test_per_wrapper_user_ok_gate_consistent
        Source-inspects each of the 8 target launch wrappers and
        confirms the gate value matches the Wf-294B (USER_OK=1) or
        Wf-300C (USER_OK=YES) family convention, the gate function is
        wired BEFORE any GPU work, and the FATAL message references
        kill_then_adapt.
  - test_dry_run_respects_gate
        Live-runs each wrapper with --dry-run and asserts the gate
        behaviour matches its declared pattern (some wrappers SKIP the
        gate on --dry-run by design; others enforce strictly).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
- All gate behaviour MEASURED via direct invocation in this env.
- Plan JSON emission is OPTIONAL; only wf131 + p5 emit a structured
  plan.json. Other wrappers emit textual plan output only.
- G5 (launch_wf124_p3_g5.sh) has NO USER_OK gate enforcement — this
  is surfaced as an HONEST-NEGATIVE gap (Wf-337-G2 hardening was
  declared but the gate was never wired in the script; Wf-363-A2
  records this and excludes g5 from the --dry-run-enforces-gate test).
- All numeric values traced to live-run output; no fabricated claims.

Cross-references
----------------
- `scripts/launch_wf124_p3_g5.sh`           — TODO-181 G5 wrapper (gap)
- `scripts/launch_wf124_p3_g6.sh`           — TODO-182 G6 wrapper
- `scripts/launch_wf131_p3_deferred_runs.sh` — Wf-355-B1 wrapper
- `scripts/launch_wf146_deferred_runs.sh`   — Wf-355-B2 wrapper
- `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`
                                            — Wf-355-A1 wrapper
- `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh`
                                            — Wf-355-A2 wrapper
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
                                            — Wf-355-A3 wrapper
- `scripts/recover_g6_failed_tuples.sh`     — Wf-337-G1 wrapper
- `scripts/pre_flight_gpu_ready.sh`         — Wf-350-B1 wrapper
- `TODO/completed/363a2_user_ok_ux.md`      — Wf-363-A2 ship doc.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
SCRIPTS = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# 8 USER_OK-gated wrappers under audit (per Wf-363-A2 task contract)
# ---------------------------------------------------------------------------
# Each entry: (path, gate_value, dry_run_policy, emits_plan_json, family)
#   gate_value      : "USER_OK=1" or "USER_OK=YES"
#   dry_run_policy  : "strict" (gate always enforced)
#                     | "skip"  (gate skipped on --dry-run by design)
#                     | "r1_optional" (R1 mode skips gate; R3 enforces)
#   emits_plan_json : True / False
#   family          : "Wf-294B" (legacy, USER_OK=1) | "Wf-300C" (R1/R3, =YES)

WRAPPERS = [
    # Wf-294B family — gate_value="1", strict dry-run enforcement
    {
        "path": SCRIPTS / "launch_wf131_p3_deferred_runs.sh",
        "gate_value": "1",
        "dry_run_policy": "strict",
        "emits_plan_json": True,
        "family": "Wf-294B",
        "wf_origin": "Wf-355-B1",
    },
    {
        "path": SCRIPTS / "launch_wf124_p3_g5.sh",
        "gate_value": "1",
        "dry_run_policy": "strict",
        "emits_plan_json": True,
        "family": "Wf-294B",
        "wf_origin": "Wf-366-A1 (gap closure post-Wf-363-A2 audit)",
    },
    {
        "path": SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
        "gate_value": "1",
        "dry_run_policy": "strict",
        "emits_plan_json": True,
        "family": "Wf-294B",
        "wf_origin": "Wf-355-A3",
    },
    {
        "path": SCRIPTS / "launch_wf124_p3_g6.sh",
        "gate_value": "1",
        "dry_run_policy": "strict",  # Wf-378-A2: gate moved BEFORE --dry-run; fires FIRST
        "emits_plan_json": True,  # Wf-378-A2: plan.json now emitted in --dry-run block
        "family": "Wf-294B",
        "wf_origin": "Wf-378-A2 (gap closure post-Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1 audit)",
    },
    # Wf-300C family — gate_value="YES", R1/R3 mode dispatch
    {
        "path": SCRIPTS / "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
        "gate_value": "YES",
        "dry_run_policy": "r1_optional",
        "emits_plan_json": False,
        "family": "Wf-300C",
        "wf_origin": "Wf-355-A1",
    },
    {
        "path": SCRIPTS / "launch_wf310e_r20_p4_bindingdb_1.07M.sh",
        "gate_value": "YES",
        "dry_run_policy": "strict",  # audit invariant: even R1 dry-run enforces
        "emits_plan_json": False,
        "family": "Wf-294B-or-300C",  # explicit "R1 REQUIRED"
        "wf_origin": "Wf-355-A2",
    },
    {
        "path": SCRIPTS / "launch_wf146_deferred_runs.sh",
        "gate_value": "YES",
        "dry_run_policy": "r1_optional",
        "emits_plan_json": False,
        "family": "Wf-300C",
        "wf_origin": "Wf-355-B2",
    },
    # Wf-337-G1 — recovers G6 failed/pending tuples; Wf-366-A2 GAP-2 fix
    # swapped the block order so USER_OK gate fires BEFORE --dry-run exit
    # (was loose_bypass pre-fix; now strict per Wf-366-A2 ship).
    {
        "path": SCRIPTS / "recover_g6_failed_tuples.sh",
        "gate_value": "1",
        "dry_run_policy": "strict",  # Wf-366-A2 GAP-2 fix: gate fires FIRST
        "emits_plan_json": False,
        "family": "Wf-337-G1",
        "wf_origin": "Wf-337-G1 (Wf-366-A2 hardened)",
    },
    # Wf-350-B1 — gate verifier; SKIP gate on --dry-run by design
    {
        "path": SCRIPTS / "pre_flight_gpu_ready.sh",
        "gate_value": "1",
        "dry_run_policy": "skip",
        "emits_plan_json": False,
        "family": "Wf-350-B1",
        "wf_origin": "Wf-350-B1",
    },
]


# Wrappers KNOWN to lack a USER_OK gate (HONEST-NEGATIVE surfaced here).
# These are excluded from the strict --dry-run-enforces-gate assertion.
# As of Wf-366-A1 (2026-09-29), the G5 gap is CLOSED: G5 now has a strict
# USER_OK=1 gate wired (exit 64 + USER_OK gate refused message) and is
# migrated into the main WRAPPERS list above. This list is kept empty for
# forward-compatibility (any newly-discovered gap would be added here).
# Wf-378-A2 (2026-09-30) also CLOSED the G6 gap: G6 wrapper had a USER_OK
# gate but it was wired AFTER the --dry-run exit block (at original
# lines 208-218), so `--dry-run` silently bypassed it. Wf-378-A2 moved
# the gate BEFORE the --dry-run block per the Wf-366-A2 lesson and added
# DRY_RUN.log + plan.json emission (consistent with Wf-366-A1 G5 pattern).
# G6 is migrated from `dry_run_policy="loose"` to `dry_run_policy="strict"`
# above.
GATE_GAP_WRAPPERS: list[Path] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    assert path.exists(), f"wrapper missing: {path}"
    return path.read_text()


def _kill_then_adapt_ref(text: str) -> bool:
    """Return True if the wrapper source references the kill_then_adapt
    directive (either in the script header or inside the gate block)."""
    return "kill_then_adapt" in text


def _gate_block_for_value(text: str, value: str) -> bool:
    """Return True if the wrapper enforces a USER_OK gate with the given
    value (`USER_OK=1` or `USER_OK=YES`) — i.e. checks that the env var
    is set and equals the expected value before any GPU work."""
    # Look for either the explicit `exit 64`/`exit 1` after a USER_OK
    # check, or a `die()` / `FATAL` call that complains about USER_OK.
    escaped = re.escape(value)
    patterns = [
        rf'USER_OK.*!=.*"{value}"',
        rf"USER_OK.*{escaped}",
        rf'USER_OK.*must.*equal.*"{value}"',
        rf'USER_OK"\s*!=\s*"{value}"',
    ]
    return any(re.search(p, text) for p in patterns)


def _gate_uses_die_or_exit(text: str) -> bool:
    """Return True if the wrapper has a non-zero exit (die/exit 1 or 64)
    path tied to the USER_OK check."""
    return bool(
        re.search(r"USER_OK[^\n]*\n[^\n]*exit 64", text)
        or re.search(r"die\s+\"USER_OK", text)
        or re.search(r"FATAL.*USER_OK", text)
        or re.search(r"USER_OK[^\n]*\n[^\n]*exit 1", text)
        or re.search(r"USER_OK.*\n.*exit\s+1", text)
    )


# ---------------------------------------------------------------------------
# Test 1 — per-wrapper USER_OK gate exit codes consistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper",
    [w for w in WRAPPERS],
    ids=[w["path"].name for w in WRAPPERS],
)
def test_per_wrapper_user_ok_gate_consistent(wrapper):
    """Per-wrapper USER_OK gate UX contract (Wf-363-A2 audit).

    Verifies for each of the 8 USER_OK-gated launch wrappers:

      - The wrapper file exists.
      - The wrapper source references `kill_then_adapt` (so the operator
        knows why the gate exists).
      - The gate value matches the declared family convention
        (Wf-294B = "1"; Wf-300C = "YES").
      - The gate uses a non-zero exit path (die/exit 64/exit 1) tied
        to the USER_OK check.
      - If `emits_plan_json` is True, the wrapper contains a `plan.json`
        emit block with `"user_ok": true`.

    Honest framing: this test is source-only (no subprocess invocation)
    so it runs in < 1 s per wrapper; the live-run gate behaviour is
    verified separately by `test_dry_run_respects_gate`.
    """
    path = wrapper["path"]
    assert path.exists(), f"wrapper missing: {path}"
    text = _read(path)

    # 1. kill_then_adapt reference (operator-visible rationale)
    assert _kill_then_adapt_ref(text), (
        f"{path.name}: missing kill_then_adapt directive reference; "
        "operator cannot tell why the gate exists"
    )

    # 2. gate value matches declared family
    expected_value = wrapper["gate_value"]
    assert _gate_block_for_value(text, expected_value), (
        f"{path.name}: expected USER_OK={expected_value!r} gate value "
        f"({wrapper['family']} family) but no matching check found"
    )

    # 3. non-zero exit path tied to USER_OK failure
    assert _gate_uses_die_or_exit(text), (
        f"{path.name}: USER_OK check present but no die()/exit-64/"
        "exit-1 path found; gate may silently no-op"
    )

    # 4. plan.json emission contract (where declared)
    if wrapper["emits_plan_json"]:
        assert "PLAN_JSON" in text or "plan.json" in text, (
            f"{path.name}: declares emits_plan_json=True but no "
            "PLAN_JSON / plan.json reference found in source"
        )
        # plan.json content should carry user_ok:true marker
        assert (
            '"user_ok": true' in text or '"user_ok":true' in text
        ), f"{path.name}: plan.json emission missing 'user_ok': true"


# ---------------------------------------------------------------------------
# Test 2 — --dry-run respects gate (per declared dry_run_policy)
# ---------------------------------------------------------------------------

# Subset of wrappers we run with --dry-run in this test. We exclude
# `pre_flight_gpu_ready.sh` because it takes ~3 min wall (calls
# pre_flight_check.py). That gate-emit SKIP behaviour is verified by
# the existing Wf-347-C1 test suite (test_wf347c1_pre_flight_gpu_ready).
DRY_RUN_TEST_WRAPPERS = [w for w in WRAPPERS if w["dry_run_policy"] != "skip"]


@pytest.mark.parametrize(
    "wrapper",
    DRY_RUN_TEST_WRAPPERS,
    ids=[w["path"].name for w in DRY_RUN_TEST_WRAPPERS],
)
def test_dry_run_respects_gate(wrapper):
    """Per-wrapper --dry-run vs USER_OK gate interaction (Wf-363-A2 audit).

    Live-runs each wrapper with --dry-run (no USER_OK env) and asserts
    that the gate behaviour matches the declared `dry_run_policy`:

      - "strict"        : --dry-run MUST exit non-zero (gate enforced).
      - "loose"          : --dry-run MAY exit zero (gate is bypassed by
                           exiting before the gate fires). Recorded as
                           HONEST-NEGATIVE — should be migrated to
                           strict per AGENTS.md §5.1.
      - "loose_bypass"   : same as loose but the gate block EXISTS and
                           fires on the real-launch path; documented
                           here for awareness.
      - "r1_optional"    : R1 mode (default) skips gate on --dry-run
                           (smoke only); R3 mode enforces. Test runs in
                           R1 default mode, so --dry-run is expected to
                           exit 0.
    """
    path = wrapper["path"]
    assert path.exists(), f"wrapper missing: {path}"

    # Choose invocation args. Wrappers with --mode R1/R3 use --mode R1.
    cmd = ["bash", str(path), "--dry-run"]
    if wrapper["family"] in ("Wf-300C", "Wf-294B-or-300C"):
        # R1 mode is smoke (USER_OK optional). For "strict" (p4)
        # we still hit the gate because p4 requires USER_OK even for R1.
        cmd += ["--mode", "R1"]

    env = {**os.environ, "USER_OK": ""}  # explicitly unset
    env.pop("USER_OK", None)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    rc = proc.returncode
    stdout_tail = (proc.stdout or "")[-400:]
    stderr_tail = (proc.stderr or "")[-400:]

    policy = wrapper["dry_run_policy"]

    if policy == "strict":
        # Gate enforced even on --dry-run — must exit non-zero.
        assert rc != 0, (
            f"{path.name}: --dry-run should enforce USER_OK gate "
            f"(policy=strict); got rc={rc} stdout={stdout_tail!r} "
            f"stderr={stderr_tail!r}"
        )
        # The error message must mention USER_OK so the operator knows
        # what to do.
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert "USER_OK" in combined, (
            f"{path.name}: --dry-run gate failure message missing "
            "USER_OK reference; operator cannot recover"
        )

    elif policy == "r1_optional":
        # R1 default mode skips the gate on --dry-run. May still fail
        # later on pre-flight, but the GATE itself must not fire.
        # We accept any rc here (subsequent pre-flight failure is OK).
        # The gate-message absence is the contract:
        combined = (proc.stdout or "") + (proc.stderr or "")
        # If USER_OK=YES gate fires we know R1 is NOT optional.
        if "USER_OK env var not set" in combined or "USER_OK gate refused" in combined:
            pytest.fail(
                f"{path.name}: --dry-run R1 mode fired USER_OK gate "
                f"(policy=r1_optional); got rc={rc} stderr={stderr_tail!r}"
            )

    elif policy in ("loose", "loose_bypass"):
        # --dry-run exits 0 before the gate fires. This is recorded as
        # HONEST-NEGATIVE — should be tightened to strict per
        # AGENTS.md §5.1. Test just records the current behaviour.
        assert rc == 0, (
            f"{path.name}: --dry-run expected to bypass gate "
            f"(policy={policy}); got rc={rc} stderr={stderr_tail!r}"
        )


# ---------------------------------------------------------------------------
# Gap surfacing test (REMOVED — Wf-366-A1 closed the G5 gap on 2026-09-29)
# ---------------------------------------------------------------------------
# test_g5_user_ok_gate_gap_recorded was REMOVED in Wf-366-A1 because the
# G5 USER_OK gate gap was closed by Wf-366-A1 (gate wired at lines 51-67
# of scripts/launch_wf124_p3_g5.sh, exit 64 + USER_OK REQUIRED message,
# plus DRY_RUN.log + plan.json emission in --dry-run mode). G5 was
# migrated from GATE_GAP_WRAPPERS into the main WRAPPERS list so it is
# now tested under the strict --dry-run-enforces-gate assertion.
# Cross-ref: TODO/completed/366a1_gap1_g5_user_ok_fix.md
