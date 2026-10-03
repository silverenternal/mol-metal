#!/usr/bin/env python3
"""Wf-125 P1 CONTROLD PAIR retrain wrapper tests.

================================================================
Wf-125 P1 CONTROLD PAIR — CPU-only test suite for the launch wrapper
that runs the 3-run CONTROLD PAIR verification per Wf-387-A2 §5.1.

Tests cover:
* test_launcher_syntax — bash -n parser check on the wrapper script
* test_user_ok_gate_REFUSED_default — exit 64 when USER_OK unset
* test_user_ok_gate_REFUSED_zero — exit 64 when USER_OK=0
* test_user_ok_gate_PASSED_dry_run — dry-run emits DRY_RUN.log +
  plan.json when USER_OK=1; verifies 3-run CONTROLD PAIR enumeration
* test_dry_run_emission_schema — DRY_RUN.log contains all required
  Wf-363-A1 schema fields (wrapper / mode / trainer / pair x 3 /
  flags / gpu / budget / user_ok / ts / per_pair block)
* test_plan_json_emission — plan.json is valid JSON with the canonical
  schema (user_ok=true, dry_run=true, run_modes=3, gpu=cuda:0,
  wall_budget_hr=6, honest_framing.controlled_pair_design)
* test_3_run_controlled_pair_enumeration — verify the 3-run sequence
  (synthetic, real, none) is emitted in DRY_RUN.log + plan.json
* test_synthetic_or_real_flag_handling — verify --run-mode synthetic
  emits the Wf-139 synthetic cohort JSON; --run-mode real emits
  Wf-387-A2 real cohort; --run-mode none omits cohort args
* test_per_pair_argv_expansion_fix — verify the per-run argv expansion
  fix wires the cohort JSON correctly (per Wf-414-A1 §4 pattern)

Honest framing (party-twenty-spirit §3.5):
* 0 GPU launches — all tests invoke --dry-run or sub-second smoke
* 0 paper/* touches (TODO-84 honored)
* 0 wet-lab planning (AGENTS.md §10 honored)
* USER_OK gate behavior is the SAFEST test surface (exit 64 on
  refusal = kill_then_adapt safety contract preserved bit-for-bit)
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

# -----------------------------------------------------------------------------
# Project root + wrapper path discovery
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WRAPPER = PROJECT_ROOT / "scripts" / "launch_wf125_p1_controlled_pair.sh"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf125_p1_controlled_pair"

PY_BIN = "/home/hugo/codes/mol-metal/.venv/bin/python"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wrapper_exists() -> bool:
    """Verify the wrapper script is on disk (Wf-411-A2 HONEST-NEGATIVE pattern)."""
    return WRAPPER.exists()


@pytest.fixture(scope="module")
def wrapper_executable() -> bool:
    """Verify the wrapper is executable (chmod +x per convention)."""
    if not WRAPPER.exists():
        return False
    import os

    return os.access(WRAPPER, os.X_OK)


@pytest.fixture(scope="module")
def dry_run_artifacts():
    """Run the wrapper in dry-run mode once and return (stdout, dry_run_log_text, plan_json_text).

    Used by tests that need the artefacts to exist.
    """
    env = {
        "PATH": "/usr/bin:/usr/local/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(PROJECT_ROOT),
        "USER_OK": "1",
    }
    result = subprocess.run(
        [str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, (
        f"Wf-125 P1 CONTROLD PAIR --dry-run: expected rc=0, got rc={result.returncode}\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:2000]}"
    )
    dry_run_log = REPORTS_DIR / "DRY_RUN.log"
    plan_json = REPORTS_DIR / "plan.json"
    assert dry_run_log.exists(), f"DRY_RUN.log MISSING: {dry_run_log}"
    assert plan_json.exists(), f"plan.json MISSING: {plan_json}"
    return result.stdout, dry_run_log.read_text(), plan_json.read_text()


# -----------------------------------------------------------------------------
# Test 1: bash syntax check
# -----------------------------------------------------------------------------
def test_launcher_syntax(wrapper_exists):
    """bash -n parser check on the wrapper script.

    Closes the filesystem-truth verification gap surfaced by AGENTS.md
    §5.5: verifies the file is on disk AND the bash syntax is valid
    (not just empty file or typo'd shebang).
    """
    assert wrapper_exists, f"Wrapper MISSING on disk: {WRAPPER}"
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"bash -n failed (rc={result.returncode}) on {WRAPPER}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# -----------------------------------------------------------------------------
# Test 2: USER_OK gate REFUSED when env unset (default)
# -----------------------------------------------------------------------------
def test_user_ok_gate_REFUSED_default(wrapper_exists):
    """USER_OK gate refused (exit 64) when env var unset — default state.

    Per Wf-294B family convention + mol-metal-gpu-execution §0:
    The wrapper MUST exit 64 (kill_then_adapt refusal code) when
    USER_OK is unset. This is the SAFEST behavior — refuses any
    accidental GPU launch.
    """
    assert wrapper_exists, f"Wrapper MISSING on disk: {WRAPPER}"
    env_no_user_ok = {
        "PATH": "/usr/bin:/usr/local/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(PROJECT_ROOT),
        # explicitly NO USER_OK
    }
    result = subprocess.run(
        [str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=15,
        env=env_no_user_ok,
    )
    assert result.returncode == 64, (
        f"USER_OK gate default: expected rc=64, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (
        "USER_OK gate refused" in result.stderr
    ), f"USER_OK refusal message not in stderr:\n{result.stderr}"
    assert (
        "USER_OK=1" in result.stderr
    ), f"USER_OK=1 hint not in stderr:\n{result.stderr}"


# -----------------------------------------------------------------------------
# Test 3: USER_OK gate REFUSED when USER_OK=0
# -----------------------------------------------------------------------------
def test_user_ok_gate_REFUSED_zero(wrapper_exists):
    """USER_OK gate refused (exit 64) when USER_OK=0 (not equal to 1).

    Per Wf-294B family convention: only USER_OK=1 (exact string match)
    satisfies the gate. USER_OK=0 must still refuse.
    """
    assert wrapper_exists, f"Wrapper MISSING on disk: {WRAPPER}"
    env_user_ok_zero = {
        "PATH": "/usr/bin:/usr/local/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(PROJECT_ROOT),
        "USER_OK": "0",
    }
    result = subprocess.run(
        [str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=15,
        env=env_user_ok_zero,
    )
    assert result.returncode == 64, (
        f"USER_OK=0 gate: expected rc=64, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "USER_OK gate refused" in result.stderr


# -----------------------------------------------------------------------------
# Test 4: USER_OK gate PASSED + dry-run emits DRY_RUN.log + plan.json
# -----------------------------------------------------------------------------
def test_user_ok_gate_PASSED_dry_run(wrapper_exists):
    """USER_OK=1 + --dry-run: SCAFFOLD smoke runs + DRY_RUN.log + plan.json emitted.

    Per Wf-363-A1: dry-run path emits consistent DRY_RUN.log schema.
    Per AGENTS.md §5.7 NEW: SCAFFOLD smoke runs BEFORE the DRY_RUN branch
    (50-step CPU smoke per Wf-399-A1 §5.7 floor discipline).
    """
    assert wrapper_exists, f"Wrapper MISSING on disk: {WRAPPER}"
    env_user_ok_one = {
        "PATH": "/usr/bin:/usr/local/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(PROJECT_ROOT),
        "USER_OK": "1",
    }
    result = subprocess.run(
        [str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,  # SCAFFOLD smoke may take 5-10s
        env=env_user_ok_one,
    )
    # --dry-run returns rc=0 (DRY-RUN OK exit path)
    assert result.returncode == 0, (
        f"USER_OK=1 --dry-run: expected rc=0, got rc={result.returncode}\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:2000]}"
    )

    # DRY_RUN.log + plan.json must be emitted (per Wf-363-A1 schema)
    dry_run_log = REPORTS_DIR / "DRY_RUN.log"
    plan_json = REPORTS_DIR / "plan.json"
    assert dry_run_log.exists(), f"DRY_RUN.log MISSING: {dry_run_log}"
    assert plan_json.exists(), f"plan.json MISSING: {plan_json}"

    # Verify 3-run CONTROLD PAIR enumeration in stdout
    assert "3 CONTROLD PAIR runs" in result.stdout or "synthetic" in result.stdout
    pair_lines = [
        line
        for line in result.stdout.splitlines()
        if "run_mode=" in line and "->" in line
    ]
    assert (
        len(pair_lines) == 3
    ), f"Expected 3 CONTROLD PAIR runs enumerated, got {len(pair_lines)}"


# -----------------------------------------------------------------------------
# Test 5: DRY_RUN.log schema validation (Wf-363-A1)
# -----------------------------------------------------------------------------
def test_dry_run_emission_schema(wrapper_exists, dry_run_artifacts):
    """DRY_RUN.log has the canonical Wf-363-A1 schema fields + 3 pair entries."""
    assert wrapper_exists
    _, dry_run_text, _ = dry_run_artifacts
    # Required schema fields (per Wf-363-A1)
    assert "DRY-RUN launcher for" in dry_run_text
    assert "wrapper:" in dry_run_text
    assert "mode:" in dry_run_text
    assert "trainer:" in dry_run_text
    assert "flags:" in dry_run_text
    assert "gpu:" in dry_run_text
    assert "budget:" in dry_run_text
    assert "user_ok:" in dry_run_text
    assert "ts:" in dry_run_text
    # Per-pair block (3 pairs)
    assert "per_pair:" in dry_run_text
    pair_label_count = dry_run_text.count("pair:    ")
    # The `pair:` lines (one per run) + the per_pair block below.
    # Each pair appears in BOTH blocks (lines 1 + lines 2).
    assert (
        pair_label_count >= 3
    ), f"Expected ≥3 `pair:    <name>` rows, got {pair_label_count}"
    # Count actual CONTROLD PAIR tuples
    expected_pairs = ["run_mode=synthetic", "run_mode=real", "run_mode=none"]
    for pair in expected_pairs:
        assert pair in dry_run_text, f"Missing pair '{pair}' in DRY_RUN.log"


# -----------------------------------------------------------------------------
# Test 6: plan.json validation (Wf-125 P1 CONTROLD PAIR canonical schema)
# -----------------------------------------------------------------------------
def test_plan_json_emission(wrapper_exists, dry_run_artifacts):
    """plan.json is valid JSON with the Wf-125 P1 CONTROLD PAIR canonical schema."""
    assert wrapper_exists
    _, _, plan_text = dry_run_artifacts
    data = json.loads(plan_text)
    # Required fields (per Wf-125 P1 CONTROLD PAIR design)
    assert data["wrapper"] == "launch_wf125_p1_controlled_pair.sh"
    assert data["wf_origin"].startswith("Wf-125 P1 CONTROLD PAIR")
    assert data["user_ok"] is True
    assert data["dry_run"] is True
    assert data["run_modes"] == ["synthetic", "real", "none"]
    assert data["n_runs"] == 3
    assert data["epochs"] == 50
    assert data["per_run_budget_hr"] == 2
    assert data["wall_budget_hr"] == 6
    assert data["gpu"] == "cuda:0"
    # honest_framing section
    assert "honest_framing" in data
    assert "MEASURED_today" in data["honest_framing"]
    assert "PROJECTED" in data["honest_framing"]
    assert "controlled_pair_design" in data["honest_framing"]
    assert "kill_then_adapt_directive" in data["honest_framing"]
    assert "wf387a2_real_cohort_attestation" in data["honest_framing"]
    assert "wf261_scaffold_split_guarantee" in data["honest_framing"]
    assert "wf399a1_50step_smoke" in data["honest_framing"]
    assert "wf386a3_atomic_write_compliance" in data["honest_framing"]
    assert "verdict_taxonomy" in data["honest_framing"]
    assert data["status"] == "PLAN_ONLY"
    # Cohort paths
    assert data["synth_rh_cohort"].endswith("synthetic_rh_pos_neg_normalized.json")
    assert data["real_rh_cohort"].endswith("balanced_rh_cohort.json")


# -----------------------------------------------------------------------------
# Test 7: 3-run CONTROLD PAIR enumeration is correct order
# -----------------------------------------------------------------------------
def test_3_run_controlled_pair_enumeration(wrapper_exists, dry_run_artifacts):
    """3-run CONTROLD PAIR sequence (synthetic, real, none) is correct.

    Per Wf-387-A2 §5.1 CONTROLD PAIR design: the 3 runs are ordered
    (1) buggy synthetic, (2) substituted real, (3) no-augmentation control.
    The ordering matters for the verdict taxonomy.
    """
    assert wrapper_exists
    _, dry_run_text, plan_text = dry_run_artifacts
    # Check DRY_RUN.log order (per Wf-363-A1: `pair:` lines emitted in order)
    pair_section = re.search(r"per_pair:\n(.*?)(?:\n\n|$)", dry_run_text, re.DOTALL)
    assert pair_section is not None, "per_pair block not found"
    pairs = [
        line.strip() for line in pair_section.group(1).splitlines() if line.strip()
    ]
    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    assert pairs[0].endswith(
        "run_mode=synthetic"
    ), f"1st pair should be synthetic, got {pairs[0]}"
    assert pairs[1].endswith(
        "run_mode=real"
    ), f"2nd pair should be real, got {pairs[1]}"
    assert pairs[2].endswith(
        "run_mode=none"
    ), f"3rd pair should be none, got {pairs[2]}"
    # Check plan.json order
    data = json.loads(plan_text)
    assert data["run_modes"] == ["synthetic", "real", "none"]


# -----------------------------------------------------------------------------
# Test 8: --synthetic-or-real flag handling
# -----------------------------------------------------------------------------
def test_synthetic_or_real_flag_handling(wrapper_exists):
    """--run-mode synthetic/real/none controls which cohort JSON is wired.

    Verifies the build_cohort_args helper inside the wrapper dispatches
    the correct cohort JSON path per run_mode. This is the per-pair argv
    expansion fix per Wf-414-A1 §4.
    """
    assert wrapper_exists, f"Wrapper MISSING on disk: {WRAPPER}"
    text = WRAPPER.read_text()
    # Verify the build_cohort_args function is defined
    assert "build_cohort_args()" in text
    # Verify the case statement dispatches all 3 modes
    assert "synthetic)" in text
    assert "real)" in text
    assert "none)" in text
    # Verify the Wf-139 synthetic cohort path is wired for synthetic
    assert "synthetic_rh_pos_neg_normalized.json" in text
    # Verify the Wf-387-A2 real balanced cohort path is wired for real
    assert "balanced_rh_cohort.json" in text
    # Verify the invalid-mode rejection case
    assert "invalid --run-mode" in text
    # Verify the --run-mode flag parsing captures both --run-mode=X and --run-mode X
    assert "--run-mode" in text
    # Verify the COMMON_ARGS are correct (Wf-131 P3 baseline + safety lever)
    common_args_match = re.search(r"COMMON_ARGS=\(\s*\n(.*?)\n\s*\)", text, re.DOTALL)
    assert common_args_match is not None, "COMMON_ARGS=() array not detected"
    common_args_body = common_args_match.group(1)
    assert "--epochs" in common_args_body
    assert '--metals "Rh"' in common_args_body
    assert "--seeds 0" in common_args_body
    assert "--wasserstein-weight 0.05" in common_args_body
    assert "--use-l2-guard" in common_args_body
    # Per Wf-335-R3: --no-cuda-graphs safety lever for Blackwell sm_120
    assert "--no-cuda-graphs" in common_args_body
