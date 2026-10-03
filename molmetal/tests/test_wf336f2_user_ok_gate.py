"""Wf-336-F2 — USER_OK-gate verification for Wf-135 Phase 2 + Wf-126 m19 LOMO wrappers.

=============================================================================
Wf-336-F2 — USER_OK-gate enforcement verification (CPU-only test)
=============================================================================

Per Wf-325C: Wf-135 Phase 2 LOMO + Wf-126 m19 LOMO are both USER_OK-BLOCKED per
the kill_then_adapt standing directive (mol-metal-gpu-execution §0).  F2 verifies
the USER_OK=1 gate is properly enforced on both wrappers — without USER_OK, the
launch must refuse to run (exit code 64 per kill_then_adapt convention); with
USER_OK=1, the dry-run path must accept the invocation (exit 0).

This test file is **CPU-only** and does NOT launch any GPU work.

Honest framing (party-twenty-spirit §3.5 + AGENTS.md §5.2):
  - Wf-135 Phase 2 (Wf-310E wrapper, ~2 weeks GPU): MEASURED gate present; exit 64
    on USER_OK!=1; accepts USER_OK=1 dry-run with exit 0.
  - Wf-126 m19 LOMO: NO standalone launch wrapper exists on disk (MEASURED — only
    Wf-310E_wf135 wrapper is present; Wf-176 m19 LOMO work was absorbed into
    Wf-322F CPU closeout per TODO/completed/322f_wf176_m19_lomo_final_closeout.md).
    The Wf-126 wrapper is therefore a "MISSING_GATE" honest-negative — the
    wrapper itself does not exist, so there is no USER_OK gate to verify.  Tests
    for Wf-126 SKIP rather than FAIL, preserving the honest-negative framing.

Test inventory (4 NEW tests):
  1. test_wf135_wrapper_user_ok_gate_refuses_exit_64  — must exit 64 without USER_OK
  2. test_wf126_wrapper_user_ok_gate_handling         — SKIP if missing (honest-negative)
  3. test_both_accept_exit_0_with_user_ok_1           — dry-run path passes through
  4. test_path_consistency                            — both wrappers (where present)
                                                        share a consistent USER_OK gate
                                                        contract

Total: 4 NEW tests (all CPU-only).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MOLMETAL_SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"

# Wf-135 Phase 2 LOMO — 5-metal × 5-seed × 50-epoch re-bench (~2 weeks GPU).
WF135_LAUNCH_WRAPPER = SCRIPTS_DIR / "launch_wf310e_wf135_phase2_gpu.sh"

# Wf-126 m19 LOMO.  Per TODO/completed/322f_wf176_m19_lomo_final_closeout.md, the
# Wf-176 m19 LOMO work was absorbed into Wf-322F (CPU closeout) and Wf-321A
# (CPU scaffold for Wf-135 Phase 2).  There is no separate Wf-126 GPU launch
# wrapper on disk; this is an HONEST-NEGATIVE (not a missing-required-deliverable).
# Search multiple candidate paths to confirm absence.
WF126_LAUNCH_WRAPPER_CANDIDATES = [
    SCRIPTS_DIR / "launch_wf126_m19_lomo_gpu.sh",
    SCRIPTS_DIR / "launch_wf176_m19_lomo_gpu.sh",
    MOLMETAL_SCRIPTS_DIR / "launch_wf126_m19_lomo_gpu.sh",
    MOLMETAL_SCRIPTS_DIR / "launch_wf176_m19_lomo_gpu.sh",
]

# kill_then_adapt exit code: USER_OK gate refused.
USER_OK_REFUSED_EXIT = 64

# Bash interpreter for subprocess invocation.  Falls back to /usr/bin/env bash.
BASH = shutil.which("bash") or "/usr/bin/env bash"


def _wf126_wrapper_path() -> Path | None:
    """Return the first existing Wf-126/Wf-176 wrapper, or None if none exist."""
    for cand in WF126_LAUNCH_WRAPPER_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _user_ok_refusal_marker(src: str) -> bool:
    """Return True if the wrapper source contains a USER_OK gate that REFUSES.

    Recognized patterns (any of):
      - `USER_OK` env check + explicit refusal message + exit 64
      - `USER_OK` env check + exit 1/2 (any non-zero refusal)
      - Explicit "REFUSES" / "refused" message in fatal/error block
    """
    if "USER_OK" not in src:
        return False
    # Either explicit "REFUSES" or refusal-with-exit pattern.
    has_refuse_keyword = "REFUSES" in src or "refused" in src
    has_exit_on_fail = bool(re.search(r"exit\s+(1|2|64)\b", src))
    has_user_ok_check = bool(re.search(r"\$\{?USER_OK(?:-1)?\}?", src))
    return has_refuse_keyword and has_exit_on_fail and has_user_ok_check


# =====================================================================
# Test 1 — Wf-135 wrapper USER_OK gate
# =====================================================================
class TestWf135UserOKGate:
    """Wf-310E Wf-135 Phase 2 wrapper must enforce USER_OK=1 gate (exit 64)."""

    def test_wf135_wrapper_user_ok_gate_refuses_exit_64(self, tmp_path):
        """Without USER_OK=1, the wrapper must refuse with exit 64.

        Per mol-metal-gpu-execution §0 kill_then_adapt: a > 1 hr GPU launch
        must HARD-REQUIRE USER_OK=1.  Wf-135 Phase 2 is ~2 calendar weeks
        GPU; the gate is therefore mandatory.
        """
        assert WF135_LAUNCH_WRAPPER.exists(), (
            f"Wf-135 Phase 2 launch wrapper missing at {WF135_LAUNCH_WRAPPER}. "
            "Cannot verify USER_OK gate — wrapper must exist first."
        )
        # Run with explicit empty USER_OK to be deterministic.
        env = os.environ.copy()
        env.pop("USER_OK", None)
        env["USER_OK"] = ""
        # --dry-run still requires USER_OK=1 per the wrapper's docstring §0
        # ("--dry-run does NOT bypass the gate").
        result = subprocess.run(
            [str(BASH), str(WF135_LAUNCH_WRAPPER), "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=15,
        )
        assert result.returncode == USER_OK_REFUSED_EXIT, (
            f"USER_OK gate refused expected exit {USER_OK_REFUSED_EXIT}, "
            f"got {result.returncode}. stderr={result.stderr!r}"
        )
        # Stderr should mention the gate.
        assert "USER_OK" in result.stderr, "stderr must mention USER_OK gate on refusal"
        assert (
            "refused" in result.stderr.lower() or "REFUSES" in result.stderr
        ), "stderr must include 'refused' marker on USER_OK gate refusal"


# =====================================================================
# Test 2 — Wf-126 wrapper USER_OK gate (HONEST-NEGATIVE if wrapper missing)
# =====================================================================
class TestWf126UserOKGate:
    """Wf-126 m19 LOMO wrapper USER_OK gate — HONEST-NEGATIVE if absent."""

    def test_wf126_wrapper_user_ok_gate_handling(self):
        """Wf-126 m19 LOMO wrapper USER_OK gate handling.

        Per Wf-322F closeout, Wf-176 m19 LOMO work was absorbed into the
        Wf-321A CPU scaffold (12 tests SHIPPED).  There is therefore no
        standalone Wf-126 GPU launch wrapper on disk.  This is an
        HONEST-NEGATIVE per AGENTS.md §5.2: the work IS Tier-2-shippable
        via the CPU scaffold, and no GPU launch wrapper is needed because
        the work was completed without GPU.

        If a wrapper exists, verify its USER_OK gate refuses exit 64.
        If no wrapper exists, SKIP with an explicit honest-negative marker.
        """
        wrapper = _wf126_wrapper_path()
        if wrapper is None:
            pytest.skip(
                "Wf-126 m19 LOMO launch wrapper not on disk (HONEST-NEGATIVE). "
                "Wf-176 m19 LOMO work absorbed into Wf-322F CPU closeout per "
                "TODO/completed/322f_wf176_m19_lomo_final_closeout.md. "
                "No standalone GPU wrapper required for Tier-2 ship surface."
            )
        # If wrapper exists, verify USER_OK gate.
        env = os.environ.copy()
        env.pop("USER_OK", None)
        env["USER_OK"] = ""
        result = subprocess.run(
            [str(BASH), str(wrapper), "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=15,
        )
        assert result.returncode == USER_OK_REFUSED_EXIT, (
            f"USER_OK gate refused expected exit {USER_OK_REFUSED_EXIT}, "
            f"got {result.returncode} for {wrapper.name}. stderr={result.stderr!r}"
        )


# =====================================================================
# Test 3 — Both wrappers accept exit 0 with USER_OK=1 (dry-run path)
# =====================================================================
class TestUserOK1Accepted:
    """USER_OK=1 must allow the dry-run path to proceed (exit 0)."""

    def test_both_accept_exit_0_with_user_ok_1(self):
        """USER_OK=1 + --dry-run must produce exit 0 on existing wrappers.

        This confirms the gate is a GATE (refuses without OK, passes with OK),
        not a hard-coded refusal.  Wf-126 wrapper is SKIP-honest-negative if
        missing; the Wf-135 portion is the MEASURED assertion that the gate
        is a true gate (passes when given OK).
        """
        # Wf-135 — wrapper exists, verify USER_OK=1 + --dry-run exits 0.
        assert (
            WF135_LAUNCH_WRAPPER.exists()
        ), f"Wf-135 wrapper missing at {WF135_LAUNCH_WRAPPER}"
        env = os.environ.copy()
        env["USER_OK"] = "1"
        result = subprocess.run(
            [str(BASH), str(WF135_LAUNCH_WRAPPER), "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, (
            f"USER_OK=1 + --dry-run expected exit 0, got {result.returncode}. "
            f"stderr={result.stderr!r}"
        )
        # Wf-135 dry-run should mention DRY-RUN plan in stderr (log() emits to stderr).
        assert (
            "DRY-RUN" in result.stderr or "dry-run" in result.stderr.lower()
        ), "Wf-135 dry-run must mention DRY-RUN mode"

        # Wf-126 — only check if wrapper exists (honest-negative SKIP otherwise).
        wrapper = _wf126_wrapper_path()
        if wrapper is None:
            pytest.skip(
                "Wf-126 m19 LOMO wrapper absent (HONEST-NEGATIVE — see Test 2). "
                "Wf-135 acceptance verified above."
            )
        result = subprocess.run(
            [str(BASH), str(wrapper), "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Wf-126 USER_OK=1 + --dry-run expected exit 0, got "
            f"{result.returncode}. stderr={result.stderr!r}"
        )


# =====================================================================
# Test 4 — Path consistency (USER_OK gate contract across wrappers)
# =====================================================================
class TestUserOKGateContract:
    """All USER_OK-gated wrappers must share a consistent gate contract."""

    def test_path_consistency(self):
        """Wf-135 + Wf-126 wrappers (where present) must use the same gate contract.

        The contract (per mol-metal-gpu-execution §0):
          - env var name: USER_OK
          - required value: "1"
          - default rejection exit: 64
          - explicit REFUSES marker in stderr

        Wf-126 wrapper is HONEST-NEGATIVE-SKIP if missing.
        """
        assert (
            WF135_LAUNCH_WRAPPER.exists()
        ), f"Wf-135 wrapper missing at {WF135_LAUNCH_WRAPPER}"
        with WF135_LAUNCH_WRAPPER.open() as fh:
            wf135_src = fh.read()

        # Wf-135 must satisfy the contract.
        assert _user_ok_refusal_marker(wf135_src), (
            "Wf-135 wrapper USER_OK gate contract is incomplete: must "
            "include USER_OK env check + explicit REFUSES + exit-on-fail."
        )
        # Verify the env var name is exactly USER_OK (not USER_OK_1 or similar).
        m = re.search(r"\$\{?USER_OK\}?", wf135_src)
        assert m is not None, "Wf-135 must reference USER_OK env var literally"
        # Verify the gate refuses with exit 64 (canonical kill_then_adapt code).
        m64 = re.search(r"exit\s+64\b", wf135_src)
        assert m64 is not None, (
            "Wf-135 wrapper must exit 64 on USER_OK refusal "
            "(canonical kill_then_adapt exit code per mol-metal-gpu-execution §0)."
        )

        # Wf-126 — SKIP if missing (honest-negative).
        wrapper = _wf126_wrapper_path()
        if wrapper is None:
            pytest.skip(
                "Wf-126 m19 LOMO wrapper absent (HONEST-NEGATIVE — see Test 2). "
                "No gate-contract path to verify on disk."
            )
        with wrapper.open() as fh:
            wf126_src = fh.read()
        assert _user_ok_refusal_marker(wf126_src), (
            "Wf-126 wrapper USER_OK gate contract is incomplete: must "
            "include USER_OK env check + explicit REFUSES + exit-on-fail."
        )
        m126 = re.search(r"exit\s+64\b", wf126_src)
        assert m126 is not None, "Wf-126 wrapper must exit 64 on USER_OK refusal."
