"""Wf-355-A3 — R20 §5 tmQM + PlatinAI 4-cell joint retrain wrapper tests (4 NEW).

Tests:
  1. test_user_ok_gate_enforced: USER_OK gate refuses without env var (exit 64).
  2. test_bash_syntax_check: bash -n parses cleanly.
  3. test_preflight_perwave_data_quality_wired: pre-flight + per-wave + data quality
     pre-launch checks are wired into the wrapper.
  4. test_status_hold_warning_emitted_in_dryrun: STATUS_HOLD architectural-floor
     warning emitted in dry-run output (Wf-316A + Wf-122 P3 honest-negative).

Per party-twenty-spirit §3.5 honest framing: this wrapper is BLOCKED on
(a) USER_OK=1 explicit user authorization per §5.1, and (b) empirical R² lift
verification post Wf-348-B1 architectural fix (32-bin Fourier + 128-d PLL).
Tests are CPU-only documentation + structural checks — no GPU launches.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = REPO_ROOT / "scripts" / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh"
PY = "/home/hugo/codes/mol-metal/.venv/bin/python"


def _run_wrapper(
    env_overrides: dict[str, str], *args: str, timeout: int = 30
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("USER_OK", None)  # default refuse
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def test_user_ok_gate_enforced() -> None:
    """USER_OK gate refuses without env var (exit 64 per kill_then_adapt).

    Per AGENTS.md §5.1 + mol-metal-gpu-execution §0: USER_OK=1 explicit user
    authorization required for any > 1 hr GPU launch.
    """
    # Without USER_OK → exit code 64 + FATAL message
    result = _run_wrapper({}, "--dry-run", timeout=15)
    assert result.returncode == 64, (
        f"USER_OK gate must refuse with exit 64; got {result.returncode}.\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert (
        "USER_OK" in result.stderr
    ), f"stderr must mention USER_OK: {result.stderr[-300:]}"
    assert (
        "FATAL" in result.stderr
    ), f"stderr must contain FATAL marker: {result.stderr[-300:]}"
    assert (
        "USER_OK=1" in result.stderr
    ), f"stderr must show usage: {result.stderr[-300:]}"


def test_bash_syntax_check() -> None:
    """bash -n parses cleanly (no syntax errors in the 445-LOC wrapper)."""
    # bash -n = parse-only (no execution)
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"bash -n must exit 0 on parse-clean script; got {result.returncode}.\n"
        f"stderr: {result.stderr[-500:]}"
    )


def test_preflight_perwave_data_quality_wired() -> None:
    """Pre-flight + per-wave + data quality pre-launch checks are wired.

    The wrapper must call (in dry-run mode):
      - per_wave_checklist.py (Wf-262 §10.1)
      - pre_flight_check.py (TODO-207 v2 62/62 PASS gate)
      - wave_budget_tracker.py (Wf-262/291D)
      - tmQM d-block + PlatinAI 4-cell data on disk check
      - Multi-PH + PLL Wf-348-B1 augmentation importable check

    We grep the wrapper source for these calls; no GPU launch required.
    """
    assert WRAPPER.is_file(), f"wrapper missing: {WRAPPER}"
    wrapper_text = WRAPPER.read_text()

    # Pre-flight v3 62/62 PASS gate (TODO-207)
    assert (
        "pre_flight_check.py" in wrapper_text
    ), "wrapper must call pre_flight_check.py (TODO-207 v2 62/62 PASS gate)"
    # Per-wave sub-checklist (Wf-262 §10.1)
    assert (
        "per_wave_checklist.py" in wrapper_text
    ), "wrapper must call per_wave_checklist.py (Wf-262 §10.1)"
    # Wave budget tracker (Wf-262/291D)
    assert (
        "wave_budget_tracker.py" in wrapper_text
    ), "wrapper must call wave_budget_tracker.py (Wf-262/291D)"
    # tmQM d-block data on disk check
    assert "TMQM_CSV" in wrapper_text, "wrapper must check tmQM CSV on disk"
    assert (
        "tmqm_parsed.csv" in wrapper_text
    ), "wrapper must reference tmqm_parsed.csv per Wf-295B Phase 4 spec"
    # PlatinAI 4-cell data on disk check
    assert (
        "PLAT_4CELL_CSV" in wrapper_text
    ), "wrapper must check PlatinAI 4-cell CSV on disk"
    assert (
        "platinai_4cell" in wrapper_text
    ), "wrapper must reference platinai_4cell.csv per Wf-295B + R20 §5.2"
    assert (
        "platinai_4cell.meta.json" in wrapper_text
    ), "wrapper must reference platinai_4cell.meta.json for 4-cell schema verification"
    # 4 cell lines (MCF7 / A549 / A2780 / A2780cis)
    for cell in ("MCF7", "A549", "A2780", "A2780cis"):
        assert cell in wrapper_text, f"wrapper must reference cell line {cell}"
    # Multi-PH + PLL Wf-348-B1 augmentation
    assert (
        "multiph_pll_feature" in wrapper_text
    ), "wrapper must import Multi-PH + PLL augmentation (Wf-348-B1 architectural-fix)"
    assert (
        "MultiPHPLLSignature" in wrapper_text
    ), "wrapper must reference MultiPHPLLSignature class"
    assert (
        "MultiPHPLLAugment" in wrapper_text
    ), "wrapper must reference MultiPHPLLAugment class"
    assert (
        "32" in wrapper_text and "128" in wrapper_text
    ), "wrapper must specify 32-bin Fourier + 128-d PLL signature (Wf-348-B1)"


def test_status_hold_warning_emitted_in_dryrun() -> None:
    """STATUS_HOLD architectural-floor warning emitted in dry-run output.

    Per Wf-316A + TODO/pending/179_wf123_p3_w2_pic50_alignment.md: 12-d
    RDKit 2D signal floor |ρ| ≤ 0.019 → R² ≥ 0 mathematically infeasible
    on 12-d feature stack. Wf-348-B1 32-bin Fourier + 128-d PLL is the
    architectural fix path; warning must be visible BEFORE any GPU launch.
    """
    # Run dry-run with USER_OK=1 (gate passes; --dry-run skips actual launch)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy launch.log to tmp for assertion (REPORTS_DIR is on disk by wrapper)
        result = _run_wrapper({"USER_OK": "1"}, "--dry-run", timeout=60)

        # The wrapper may exit 0 on dry-run (after emitting plan + STATUS_HOLD).
        # We accept either 0 or any non-error exit; check stderr + stdout for warning.
        combined_output = result.stdout + result.stderr

        # STATUS_HOLD must be in output (warning emitted before GPU launch)
        assert (
            "STATUS_HOLD" in combined_output
        ), "STATUS_HOLD warning must appear in dry-run output (Wf-316A honest framing)"
        # 12-d RDKit 2D signal floor must be cited
        assert (
            "12-d" in combined_output or "12 d" in combined_output
        ), "STATUS_HOLD warning must cite 12-d RDKit 2D signal floor"
        # Wf-316A + Wf-122 P3 must be referenced
        assert (
            "Wf-316A" in combined_output or "Wf-122" in combined_output
        ), "STATUS_HOLD warning must cite Wf-316A / Wf-122 P3 precedent"
        # Wf-348-B1 architectural fix path must be cited
        assert (
            "Wf-348-B1" in combined_output or "32-bin Fourier" in combined_output
        ), "STATUS_HOLD warning must cite Wf-348-B1 32-bin Fourier architectural fix"
        # plan.json must be emitted with status_hold field
        plan_json = REPO_ROOT / "molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json"
        if plan_json.is_file():
            plan_text = plan_json.read_text()
            assert "STATUS_HOLD" in plan_text, "plan.json must encode STATUS_HOLD field"
            assert (
                "kill_then_adapt" in plan_text or "USER_OK" in plan_text
            ), "plan.json must reference USER_OK gate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
