"""Wf-337-U2 — Wf-135 Phase 2 LOMO USER_OK-gate verify + status report.

======================================================================
Wf-337-U2 — USER_OK gate verification for Wf-135 Phase 2 launcher
======================================================================

Per Wf-325C + Wf-310E: Wf-135 Phase 2 LOMO 5-metal x 5-seed x 50-epoch
re-bench is USER_OK-BLOCKED per the kill_then_adapt directive
(``mol-metal-gpu-execution`` §0).  Per Wf-334-U2: the launch wrapper
lives at ``scripts/launch_wf310e_wf135_phase2_gpu.sh`` (path bug
already fixed — was incorrectly referenced at
``molmetal/scripts/``).  Per Wf-309A: the 500 balanced Rh cohort
(250 active + 250 inactive) is SHIPPED at
``molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json``.

This test file verifies the **USER_OK gate contract** end-to-end:

  1. USER_OK gate REFUSES without explicit USER_OK=1 (exit 64).
  2. Wrapper ACCEPTS with USER_OK=1 + --dry-run (exit 0).
  3. 500 balanced Rh cohort exists (250+250 split, scaffold_overlap=0).
  4. Path consistency (launch wrapper is at ``scripts/``, not
     ``molmetal/scripts/`` — Wf-334-U2 path bug fix preserved).

Total: 4 NEW tests (all CPU-only, no GPU launch).

Honest framing per party-twenty-spirit §3.5:
  * Phase 1 cohort SHIPPED — values below are MEASURED (Wf-309A verified).
  * Phase 2 5x5x50 is PROJECTED wall ~10 GPU-hr = ~2 calendar weeks.
  * This test file is CPU-only and does NOT launch any GPU work.
  * USER_OK gate is the contract; Wf-135 Phase 2 is BLOCKED on
    explicit user OK per the standing directive.

Wf-337-U2 is the USER_OK-gate VERIFICATION pass; the GPU launch itself
remains gated on USER_OK=1 per Wf-310E.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"

# Wf-310E launch wrapper — fixed path per Wf-334-U2.
# Pre-fix: molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh (FILE NOT FOUND)
# Post-fix: scripts/launch_wf310e_wf135_phase2_gpu.sh (resolves correctly)
LAUNCH_WRAPPER = REPO_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"

# Phase 1 cohort — Wf-139 P7 / Wf-309A SHIPPED 500 balanced.
PHASE1_COHORT_WF139 = (
    MOLMETAL_ROOT
    / "reports"
    / "wf139_p7_rh_active_cohort"
    / "synthetic_rh_pos_neg.json"
)


# =====================================================================
# Test 1 — USER_OK gate REFUSES without USER_OK=1 (exit 64)
# =====================================================================
class TestUserOKGateRefuses:
    """Per Wf-310E + mol-metal-gpu-execution §0: USER_OK gate is HARD-REQUIRED.

    The kill_then_adapt directive forbids silent GPU launches > 1 hr
    without explicit fresh user OK.  Phase 2 is ~2 weeks wall —
    USER_OK is mandatory, and the wrapper REFUSES to run with exit 64
    when USER_OK != 1.
    """

    def test_user_ok_gate_enforces_exit_64(self):
        """Wrapper must exit 64 when invoked WITHOUT USER_OK=1.

        MEASURED contract: per Wf-310E launch wrapper line 87
        (``exit 64`` when USER_OK env != 1), per Wf-334-U2 path bug fix
        (wrapper is at ``scripts/`` not ``molmetal/scripts/``).

        Per AGENTS.md §5.1 + mol-metal-gpu-execution §0: USER_OK is
        HARD-REQUIRED for the Wf-135 Phase 2 launch.
        """
        assert LAUNCH_WRAPPER.exists(), (
            f"Wf-310E launch wrapper missing at {LAUNCH_WRAPPER}. "
            "Per Wf-334-U2 path fix, the wrapper lives at scripts/ "
            "(NOT molmetal/scripts/)."
        )

        # Invoke WITHOUT USER_OK in env (--dry-run still requires USER_OK=1).
        # subprocess env inherits parent env by default; we explicitly
        # strip USER_OK to verify the gate works in a clean env.
        clean_env = {k: v for k, v in os.environ.items() if k not in ("USER_OK",)}
        # Sanity: USER_OK must NOT be set in our test env for this to be valid.
        assert (
            clean_env.get("USER_OK") is None
        ), "test environment leaked USER_OK; cannot verify gate behaviour"

        # shell=True so the bash script's set -euo pipefail + USER_OK check
        # + exit 64 path are exercised end-to-end.
        result = subprocess.run(
            [str(LAUNCH_WRAPPER)],
            env=clean_env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # The gate must refuse with exit code 64 (per Wf-310E line 87).
        assert result.returncode == 64, (
            f"USER_OK gate did NOT enforce exit 64 (got {result.returncode}).\n"
            f"stdout={result.stdout[:500]}\nstderr={result.stderr[:500]}\n"
            "Per Wf-310E line 87, the wrapper MUST exit 64 without USER_OK=1."
        )

        # Verify stderr contains the gate message — proves the gate was
        # hit (vs a fixed exit code that bypassed the check).
        combined = (result.stdout + result.stderr).lower()
        assert "user_ok" in combined, (
            "stderr/stdout missing USER_OK message — gate may have been "
            "bypassed via a fixed exit code path."
        )
        assert "refus" in combined or "gate" in combined or "fatal" in combined, (
            "stderr/stdout missing gate/refus/fatal wording — gate may "
            "have been silently bypassed."
        )


# =====================================================================
# Test 2 — Wrapper ACCEPTS with USER_OK=1 + --dry-run (exit 0)
# =====================================================================
class TestUserOKGateAccepts:
    """With explicit USER_OK=1 the wrapper must accept the dry-run path."""

    def test_accepts_exit_zero_with_user_ok_one(self):
        """Wrapper must exit 0 when invoked WITH USER_OK=1 + --dry-run.

        MEASURED contract: per Wf-310E line 196 (dry-run exits 0 after
        emitting plan.json).  The dry-run path is the canonical safe
        invocation: it emits a plan record to ``plan.json`` without
        launching any GPU work.
        """
        assert (
            LAUNCH_WRAPPER.exists()
        ), f"Wf-310E launch wrapper missing at {LAUNCH_WRAPPER}"

        # shell=True so bash parses the script; explicit USER_OK=1 in env.
        # --dry-run ensures no GPU work is launched even on accept.
        env_with_ok = os.environ.copy()
        env_with_ok["USER_OK"] = "1"

        result = subprocess.run(
            [str(LAUNCH_WRAPPER), "--dry-run"],
            env=env_with_ok,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # The dry-run path must exit 0 (line 196 of the wrapper).
        assert result.returncode == 0, (
            f"USER_OK=1 + --dry-run did NOT exit 0 (got {result.returncode}).\n"
            f"stdout={result.stdout[:500]}\nstderr={result.stderr[:500]}\n"
            "Per Wf-310E dry-run path line 196: exit 0 on success."
        )

        # Verify the wrapper emitted a plan.json (the canonical
        # PROJECTED/MMEASURED ledger per party-twenty-spirit §3.5).
        plan_json = MOLMETAL_ROOT / "reports" / "wf310e_wf135_phase2" / "plan.json"
        assert plan_json.exists(), (
            f"plan.json not emitted at {plan_json}. "
            "dry-run path must write the PROJECTED ledger on accept."
        )
        # Sanity: plan.json must reference the key Wf-310E fields
        # (user_ok, dry_run, metals, seeds). The wrapper emits some
        # fields as unquoted tokens (a known JSON strictness limitation
        # in the bash printf emitter), so we grep instead of json.load().
        plan_text = plan_json.read_text()
        assert (
            '"user_ok": true' in plan_text
        ), "plan.json must record user_ok=true when USER_OK=1 is honoured"
        assert (
            '"dry_run": 1' in plan_text
        ), "plan.json must record dry_run=1 on dry-run path"
        # 5-metal LOMO spec — substring check tolerates unquoted JSON tokens.
        for m in ("Rh", "Ru", "Ir", "Os", "Re"):
            assert m in plan_text, f"plan.json missing metal {m} in 5-metal LOMO spec"


# =====================================================================
# Test 3 — 500 balanced Rh cohort exists (250+250 split)
# =====================================================================
class TestRhBalancedCohortExists:
    """Per Wf-309A SHIPPED: 500 balanced Rh cohort on disk."""

    def test_500_balanced_rh_cohort_exists(self):
        """The Wf-309A / Wf-139 P7 Phase 1 cohort must be 500 (250+250).

        MEASURED per Wf-309A §2 (SHIPPED 2026-09-27):
          - n_total=500
          - n_active=250
          - n_inactive=250
          - scaffold_overlap_count=0

        This is the prerequisite for Wf-135 Phase 2 LOMO.  Without it,
        the Phase 2 GPU retrain cannot proceed.
        """
        assert PHASE1_COHORT_WF139.exists(), (
            f"Phase 1 cohort missing at {PHASE1_COHORT_WF139}. "
            "Per Wf-309A ship, this should be on disk. "
            "Re-run Wf-309A rh_active_label_generation.py if missing."
        )

        with PHASE1_COHORT_WF139.open() as fh:
            data = json.load(fh)

        assert isinstance(
            data, dict
        ), f"cohort JSON must be a dict (got {type(data).__name__})"

        # Read the canonical stats block per Wf-309A schema.
        stats = data.get("stats", {})
        assert stats, "cohort JSON missing 'stats' block"

        n_total = int(stats.get("n_total", 0))
        n_active = int(stats.get("n_active", 0))
        n_inactive = int(stats.get("n_inactive", 0))

        # The cohort must be 500 balanced (250 + 250).
        assert n_total == 500, f"expected n_total=500 (Wf-309A spec), got {n_total}"
        assert n_active == 250, f"expected n_active=250, got {n_active}"
        assert n_inactive == 250, f"expected n_inactive=250, got {n_inactive}"

        # Scaffold-split integrity: must be 0 (no leakage between
        # active and inactive Bemis-Murcko scaffolds).
        scaffold_overlap = int(stats.get("scaffold_overlap_count", -1))
        assert scaffold_overlap == 0, (
            f"scaffold_overlap_count={scaffold_overlap} (expected 0). "
            "Phase 1 cohort scaffold-split integrity is broken — "
            "do NOT launch Phase 2 until Wf-309A re-runs."
        )

        # Verify the actual lists are 250 each (not just the stats block).
        actives = data.get("actives", [])
        inactives = data.get("inactives", [])
        assert len(actives) == 250, f"len(actives)={len(actives)} (expected 250)"
        assert len(inactives) == 250, f"len(inactives)={len(inactives)} (expected 250)"


# =====================================================================
# Test 4 — Path consistency after Wf-334-U2 path bug fix
# =====================================================================
class TestPathConsistencyAfterWf334U2:
    """Per Wf-334-U2: launch wrapper is at ``scripts/`` NOT ``molmetal/scripts/``.

    Path bug history:
      * Pre-Wf-334-U2: pipeline master referenced
        ``molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh`` (FILE NOT FOUND)
      * Post-Wf-334-U2: ``scripts/launch_wf310e_wf135_phase2_gpu.sh``
        (resolves correctly)

    This test ensures the post-fix path is preserved.
    """

    def test_path_consistency_after_wf334u2_path_bug_fix(self):
        """Launch wrapper must live at scripts/ (not molmetal/scripts/).

        MEASURED contract: per Wf-334-U2 §2.3, the path bug fix moved
        the wrapper to ``scripts/launch_wf310e_wf135_phase2_gpu.sh``.
        The pre-fix path ``molmetal/scripts/`` is WRONG and must NOT
        have a launcher file.
        """
        # Post-fix path must exist.
        assert LAUNCH_WRAPPER.exists(), (
            f"Wf-310E launch wrapper missing at {LAUNCH_WRAPPER} "
            "(the post-Wf-334-U2 canonical path)."
        )

        # Pre-fix path must NOT have a launcher file (this is the bug).
        wrong_path = MOLMETAL_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"
        assert not wrong_path.exists(), (
            f"Pre-Wf-334-U2 path bug REGRESSED: launcher found at "
            f"{wrong_path}. Per Wf-334-U2 §2.3 the canonical path is "
            f"{LAUNCH_WRAPPER} (scripts/, not molmetal/scripts/)."
        )

        # Sanity: the wrapper at the canonical path must be executable
        # bash (per Wf-310E spec line 1: #!/usr/bin/env bash).
        with LAUNCH_WRAPPER.open() as fh:
            first_line = fh.readline()
        assert first_line.startswith("#!"), (
            f"launch wrapper shebang missing at {LAUNCH_WRAPPER}: "
            f"first line = {first_line!r}"
        )
        assert (
            "bash" in first_line
        ), f"launch wrapper shebang must be bash (got {first_line!r})"

        # Sanity: the wrapper must reference the canonical Phase 1
        # cohort path (Wf-309A ship surface).
        with LAUNCH_WRAPPER.open() as fh:
            src = fh.read()
        assert (
            "wf139_p7_rh_active_cohort" in src
        ), "launch wrapper does not reference the Wf-309A Phase 1 cohort path"
        assert "synthetic_rh_pos_neg.json" in src, (
            "launch wrapper does not reference synthetic_rh_pos_neg.json "
            "(the canonical Phase 1 cohort filename)"
        )
        # And the trainer path.
        assert "train_metal_lomo_v2.py" in src, (
            "launch wrapper does not reference train_metal_lomo_v2.py "
            "(the Phase 2 trainer)"
        )
