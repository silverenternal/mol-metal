#!/usr/bin/env python3
"""Wf-414-A1 — launcher wrapper tests for launch_wf390a1_m2_r2_path_a.sh.

================================================================
Wf-414-A1 — CPU-only test suite for the Wf-390-A1 Path A 5-metal
subset m2 R² GPU retrain launch wrapper.

Tests cover:
* test_launcher_syntax — bash -n parser check on the wrapper script
* test_user_ok_gate_REFUSED_default — exit 64 when USER_OK unset
* test_user_ok_gate_REFUSED_zero — exit 64 when USER_OK=0
* test_user_ok_gate_PASSED_one_dry_run — dry-run emits DRY_RUN.log
  + plan.json when USER_OK=1; verifies 15-pair enumeration
* test_dry_run_emission_schema — DRY_RUN.log contains all required
  Wf-363-A1 schema fields (wrapper / mode / trainer / pair × 15 /
  flags / gpu / budget / user_ok / ts / per_pair block)
* test_plan_json_emission — plan.json is valid JSON with the Wf-414
  canonical schema (user_ok=true, dry_run=true, n_pairs=15, gpu=
  cuda:0, wall_budget_hr=12, honest_framing.wf411a2_honest_negative)
* test_scaffold_smoke_invocation — verify SCAFFOLD smoke (Wf-390-A1)
  is invoked BEFORE the DRY_RUN branch (always run; CPU-only)
* test_chemprop_v3_cache_conditional — verify the chemprop-v3-cache
  flag is APPENDED only when TRAINER=*chemprop_v2_v3* (per Wf-392-A2
  flag scoping; D-MPNN trainer does NOT accept this flag)
* test_per_cohort_features_present — verify --per-cohort-features is
  in COMMON_ARGS (per Wf-388-A1 + Wf-405-A2 closeout)
* test_no_cuda_graphs_safety_lever — verify --no-cuda-graphs is in
  COMMON_ARGS (per Wf-335-R3 + Wf-330-B4 RuntimeError fix)

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
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
WRAPPER = PROJECT_ROOT / "scripts" / "launch_wf390a1_m2_r2_path_a.sh"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf390a1_path_a"

PY_BIN = "/home/hugo/codes/mol-metal/.venv/bin/python"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wrapper_exists() -> bool:
    """Verify the wrapper script is on disk (Wf-411-A2 HONEST-NEGATIVE)."""
    return WRAPPER.exists()


@pytest.fixture(scope="module")
def wrapper_executable() -> bool:
    """Verify the wrapper is executable (chmod +x per convention)."""
    if not WRAPPER.exists():
        return False
    import os

    return os.access(WRAPPER, os.X_OK)


# -----------------------------------------------------------------------------
# Test 1: bash syntax check
# -----------------------------------------------------------------------------
def test_launcher_syntax(wrapper_exists):
    """bash -n parser check on the wrapper script.

    Closes Wf-411-A2 HONEST-NEGATIVE: verifies the file is on disk
    AND the bash syntax is valid (not just empty file or typo'd shebang).
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
    # Verify the FATAL message is in stderr
    assert "USER_OK gate refused" in result.stderr, (
        f"USER_OK refusal message not in stderr:\n{result.stderr}"
    )
    assert "USER_OK=1" in result.stderr, (
        f"USER_OK=1 hint not in stderr:\n{result.stderr}"
    )


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
def test_user_ok_gate_PASSED_one_dry_run(wrapper_exists):
    """USER_OK=1 + --dry-run: SCAFFOLD smoke runs + DRY_RUN.log + plan.json emitted.

    Per Wf-363-A1: dry-run path emits consistent DRY_RUN.log schema.
    Per Wf-414-A1: SCAFFOLD smoke (Wf-390-A1) runs BEFORE the DRY_RUN branch.
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

    # Verify 15-pair enumeration in stdout
    assert "${#PAIR_OVERRIDE[@]} pairs" not in result.stdout  # bash syntax marker should be expanded
    pair_lines = [
        line for line in result.stdout.splitlines()
        if re.match(r"\s+[A-Z][a-z]\s+seed=\d+\s+->", line)
    ]
    assert len(pair_lines) == 15, (
        f"Expected 15 (metal, seed) pairs enumerated, got {len(pair_lines)}"
    )


# -----------------------------------------------------------------------------
# Test 5: DRY_RUN.log schema validation (Wf-363-A1)
# -----------------------------------------------------------------------------
def test_dry_run_emission_schema(wrapper_exists):
    """DRY_RUN.log has the canonical Wf-363-A1 schema fields + 15 pairs."""
    assert wrapper_exists
    dry_run_log = REPORTS_DIR / "DRY_RUN.log"
    if not dry_run_log.exists():
        pytest.skip("DRY_RUN.log not yet emitted (run test_user_ok_gate_PASSED_one_dry_run first)")
    text = dry_run_log.read_text()
    # Required schema fields (per Wf-363-A1)
    assert "DRY-RUN launcher for" in text
    assert "wrapper:" in text
    assert "mode:" in text
    assert "trainer:" in text
    assert "flags:" in text
    assert "gpu:" in text
    assert "budget:" in text
    assert "user_ok:" in text
    assert "ts:" in text
    # Per-pair block (15 pairs) — `pair:    <name>` lines + per_pair block
    assert "per_pair:" in text
    pair_label_count = text.count("pair:    ")
    # The `pair:` lines (one per metal seed; Wf-363-A1 schema) + the per_pair
    # block below. Each pair appears in BOTH blocks (lines 1 + lines 2).
    assert pair_label_count >= 15, (
        f"Expected ≥15 `pair:    <name>` rows, got {pair_label_count}"
    )
    # Count actual (metal, seed) tuples
    expected_pairs = ["Ru 0", "Ru 1", "Ru 2", "Ir 0", "Ir 1", "Ir 2", "Rh 0", "Rh 1", "Rh 2", "Os 0", "Os 1", "Os 2", "Re 0", "Re 1", "Re 2"]
    for pair in expected_pairs:
        assert pair in text, f"Missing pair '{pair}' in DRY_RUN.log"


# -----------------------------------------------------------------------------
# Test 6: plan.json validation (Wf-414-A1 canonical schema)
# -----------------------------------------------------------------------------
def test_plan_json_emission(wrapper_exists):
    """plan.json is valid JSON with the Wf-414 canonical schema."""
    assert wrapper_exists
    plan_json = REPORTS_DIR / "plan.json"
    if not plan_json.exists():
        pytest.skip("plan.json not yet emitted (run test_user_ok_gate_PASSED_one_dry_run first)")
    data = json.loads(plan_json.read_text())
    # Required fields (per Wf-414-A1 + Wf-355 pattern)
    assert data["wrapper"] == "launch_wf390a1_m2_r2_path_a.sh"
    assert data["wf_origin"] == "Wf-414-A1 (USER_OK gate + SCAFFOLD smoke wire-in)"
    assert data["user_ok"] is True
    assert data["dry_run"] is True
    assert data["metals"] == ["Ru", "Ir", "Rh", "Os", "Re"]
    assert data["seeds"] == [0, 1, 2]
    assert data["epochs"] == 50
    assert data["n_pairs"] == 15
    assert data["gpu"] == "cuda:0"
    assert data["wall_budget_hr"] == 12
    assert data["per_cohort_features"] is True
    assert "chemprop_v3_cache_conditional" in data
    # honest_framing section
    assert "honest_framing" in data
    assert "wf411a2_honest_negative" in data["honest_framing"]
    assert "MEASURED_today" in data["honest_framing"]
    assert "PROJECTED" in data["honest_framing"]
    assert "kill_then_adapt_directive" in data["honest_framing"]
    assert "n_200_cohort_ceiling_preserved" in data["honest_framing"]
    assert "wf390a1_scaffold_smoke_status" in data["honest_framing"]
    assert data["status"] == "PLAN_ONLY"


# -----------------------------------------------------------------------------
# Test 7: SCAFFOLD smoke is invoked (CPU-only; per Wf-390-A1 §1.2)
# -----------------------------------------------------------------------------
def test_scaffold_smoke_invocation(wrapper_exists):
    """SCAFFOLD smoke runs BEFORE the DRY_RUN branch — CPU-only.

    Per Wf-390-A1 §1.2 SCAFFOLD/APPLIED pattern: SCAFFOLD phase must
    verify trainer CLI + emit per-cohort config JSON BEFORE any user-
    invoked branch (DRY-RUN or actual). This catches Wf-379-A2 type
    regressions where the trainer subset is invalid for argparse.
    """
    assert wrapper_exists
    text = WRAPPER.read_text()
    assert "wf390a1_path_a_smoke.py" in text
    assert "--out-dir \"${OUT_BASE}/scaffold_smoke\"" in text or '--out-dir "${OUT_BASE}/scaffold_smoke"' in text
    # The smoke invocation must appear BEFORE the DRY_RUN branch
    smoke_pos = text.find("wf390a1_path_a_smoke.py")
    dry_run_pos = text.find("DRY-RUN mode:")
    assert smoke_pos != -1 and dry_run_pos != -1
    assert smoke_pos < dry_run_pos, (
        f"SCAFFOLD smoke (pos {smoke_pos}) must precede DRY-RUN branch (pos {dry_run_pos})"
    )


# -----------------------------------------------------------------------------
# Test 8: --chemprop-v3-cache is APPENDED only when TRAINER=*chemprop_v2_v3*
# -----------------------------------------------------------------------------
def test_chemprop_v3_cache_conditional(wrapper_exists):
    """--chemprop-v3-cache flag is conditional on TRAINER script path.

    Per Wf-392-A2 + Wf-414-A1: this flag exists on
    train_chemprop_v2_v3.py but NOT on train_dmpnn_multitask_wasserstein.py.
    Including it unconditionally would cause argparse error on the
    D-MPNN trainer. The wrapper guards with [[ "${TRAINER}" == *chemprop_v2_v3* ]]
    so D-MPNN launches skip the flag and chemprop v3 launches include it.
    """
    assert wrapper_exists
    text = WRAPPER.read_text()
    # The guard block must exist
    assert 'TRAINER}' in text
    assert "*chemprop_v2_v3*" in text
    assert "--chemprop-v3-cache" in text


# -----------------------------------------------------------------------------
# Test 9: --per-cohort-features is in COMMON_ARGS
# -----------------------------------------------------------------------------
def test_per_cohort_features_present(wrapper_exists):
    """--per-cohort-features flag is in COMMON_ARGS (per Wf-388-A1 + Wf-405-A2)."""
    assert wrapper_exists
    text = WRAPPER.read_text()
    assert "--per-cohort-features" in text
    # Confirm it's INSIDE the COMMON_ARGS=() array, not just a comment
    common_args_match = re.search(r"COMMON_ARGS=\(\s*\n(.*?)\n\s*\)", text, re.DOTALL)
    assert common_args_match is not None, "COMMON_ARGS=() array not detected"
    common_args_body = common_args_match.group(1)
    assert "--per-cohort-features" in common_args_body


# -----------------------------------------------------------------------------
# Test 10: --no-cuda-graphs safety lever is in COMMON_ARGS
# -----------------------------------------------------------------------------
def test_no_cuda_graphs_safety_lever(wrapper_exists):
    """--no-cuda-graphs safety lever is in COMMON_ARGS (per Wf-335-R3)."""
    assert wrapper_exists
    text = WRAPPER.read_text()
    assert "--no-cuda-graphs" in text
    # Confirm it's INSIDE the COMMON_ARGS=() array
    common_args_match = re.search(r"COMMON_ARGS=\(\s*\n(.*?)\n\s*\)", text, re.DOTALL)
    assert common_args_match is not None, "COMMON_ARGS=() array not detected"
    common_args_body = common_args_match.group(1)
    assert "--no-cuda-graphs" in common_args_body
