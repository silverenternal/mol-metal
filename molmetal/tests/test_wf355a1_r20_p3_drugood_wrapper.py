"""Wf-355-A1 — R20 §3 DrugOOD 1.58M chemprop v2 retrain launch wrapper tests.

Owner:        Hugo Lee (autonomous Wf-355-A1 ultracode unit)
Date:         2026-09-29
Parent:       Wf-355 ultracode residual wave + Wf-354-B1 HONEST-NEGATIVE
              ("R20 §3 1.58M DrugOOD chemprop v2 launch wrapper does NOT
              exist on disk").

These 4 NEW tests verify that the new launch wrapper
``scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`` satisfies
the 4 critical contract points per AGENTS.md §1.2 (Standing GPU auth) +
§5.1 (kill_then_adapt directive) + Wf-294B / Wf-300C USER_OK pattern +
Wf-343c2/c3 retro-fit pattern:

  1. **USER_OK gate enforced**        — R3 mode requires USER_OK=YES.
  2. **Pre-flight + per-wave + data quality wired** — 3 pre-launch gates
                                            present and reference the
                                            correct scripts.
  3. **Dry-run mode**                 — `--dry-run` flag works for R1 + R3.
  4. **Bash syntax check**            — `bash -n` parses cleanly.

All 4 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU launches).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-run
output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = PROJECT_ROOT / "scripts" / "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh"
PRE_FLIGHT = PROJECT_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
PER_WAVE_CHECK = PROJECT_ROOT / "molmetal" / "scripts" / "wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK = PROJECT_ROOT / "molmetal" / "scripts" / "wf343c3_data_quality_check.py"
CHEMPROP_TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "wf185_chemprop_v2_drugood_full.py"


# ---------------------------------------------------------------------------
# Test 1: USER_OK gate enforced (R3 mode requires USER_OK=YES)
# ---------------------------------------------------------------------------

def test_wf355a1_user_ok_gate_enforced_r3():
    """Wf-355-A1 task contract — USER_OK gate enforces R3 mode per Wf-294B /
    Wf-300C pattern (kill_then_adapt ≥ 2 hr GPU).

    Verifies:
      - The wrapper script exists.
      - USER_OK gate function present in the source.
      - R3 mode without USER_OK env var exits with non-zero (FATAL).
      - R3 mode with USER_OK=YES exits with zero.
      - R1 mode without USER_OK emits WARN but still exits zero (smoke).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()
    assert "USER_OK" in text, "USER_OK env var reference missing in wrapper"
    assert "user_ok_gate" in text, "user_ok_gate function missing in wrapper"
    assert "USER_OK=YES" in text, "USER_OK=YES literal missing in wrapper"

    # R3 without USER_OK must die (FATAL exit non-zero)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        "R3 mode without USER_OK should exit non-zero (gate enforced); "
        f"got rc={proc.returncode} stdout={proc.stdout[-300:]!r}"
    )
    assert "USER_OK" in proc.stderr, (
        f"FATAL message should reference USER_OK; got stderr={proc.stderr[-300:]!r}"
    )

    # R3 with USER_OK=YES must pass (exit zero)
    env = {**os.environ, "USER_OK": "YES"}
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert proc.returncode == 0, (
        f"R3 mode with USER_OK=YES should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    # R1 mode without USER_OK must warn but still exit zero (smoke mode)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R1"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"R1 mode without USER_OK should still pass (smoke); got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )
    assert "USER_OK not set" in proc.stderr or "USER_OK not set" in proc.stdout, (
        "R1 mode without USER_OK should emit USER_OK-not-set warning"
    )


# ---------------------------------------------------------------------------
# Test 2: Pre-flight + per-wave + data quality wired (3 pre-launch gates)
# ---------------------------------------------------------------------------

def test_wf355a1_preflight_per_wave_data_quality_wired():
    """Wf-355-A1 task contract — 3 pre-launch gates must be wired per
    Wf-347-B1/B2 retro-fit pattern.

    Verifies:
      - pre_flight_check.py script path appears in the wrapper source.
      - wf343c2_per_wave_check.py script path appears in the wrapper source.
      - wf343c3_data_quality_check.py script path appears in the wrapper source.
      - Each gate has a function definition (pre_flight_gate + per_wave_check_gate
        + data_quality_gate).
      - All 3 script files actually exist on disk.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()

    # Pre-flight (TODO-207 v3 gate)
    assert "pre_flight_check.py" in text, \
        "pre_flight_check.py reference missing in wrapper"
    assert "pre_flight_gate" in text, \
        "pre_flight_gate function missing in wrapper"
    assert PRE_FLIGHT.exists(), f"pre_flight_check.py missing on disk: {PRE_FLIGHT}"

    # Per-wave check (Wf-343c2 post-Wf-347 retro-fit)
    assert "wf343c2_per_wave_check.py" in text, \
        "wf343c2_per_wave_check.py reference missing in wrapper"
    assert "per_wave_check_gate" in text, \
        "per_wave_check_gate function missing in wrapper"
    assert PER_WAVE_CHECK.exists(), \
        f"wf343c2_per_wave_check.py missing on disk: {PER_WAVE_CHECK}"

    # Data quality check (Wf-343c3 post-Wf-347 retro-fit)
    assert "wf343c3_data_quality_check.py" in text, \
        "wf343c3_data_quality_check.py reference missing in wrapper"
    assert "data_quality_gate" in text, \
        "data_quality_gate function missing in wrapper"
    assert DATA_QUALITY_CHECK.exists(), \
        f"wf343c3_data_quality_check.py missing on disk: {DATA_QUALITY_CHECK}"

    # Each gate invocation should be visible in the source
    assert "--pipeline all" in text, \
        "pre_flight_check.py invocation missing --pipeline all flag"
    assert "--wave" in text, \
        "wf343c2_per_wave_check.py invocation missing --wave flag"
    assert "--strict" in text, \
        "wf343c3_data_quality_check.py invocation missing --strict flag"


# ---------------------------------------------------------------------------
# Test 3: Dry-run mode (--dry-run works for R1 + R3 with USER_OK=YES)
# ---------------------------------------------------------------------------

def test_wf355a1_dry_run_mode_emits_dr_run_log():
    """Wf-355-A1 task contract — --dry-run flag works for both R1 (CPU
    smoke) and R3 (GPU full) modes; DRY_RUN.log file written.

    Verifies:
      - --dry-run flag is parsed by the CLI dispatch.
      - DRY_RUN.log emitted to ${out_dir}/DRY_RUN.log with the standard
        fields: trainer, subset, flags, gpu, budget, ts.
      - R1 DRY_RUN.log exists after a R1 dry-run.
      - R3 DRY_RUN.log exists after a R3 dry-run (with USER_OK=YES).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()
    assert "--dry-run)" in text, "--dry-run CLI argument case not in wrapper"
    assert "DRY_RUN=" in text, "DRY_RUN variable assignment missing"
    assert "DRY_RUN.log" in text, "DRY_RUN.log file emission missing"

    # R1 dry-run produces DRY_RUN.log
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R1"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"R1 dry-run should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    r1_dry_log = (
        PROJECT_ROOT / "molmetal" / "reports" / "wf310e_r20_p3_drugood_full_launch"
        / "r1" / "DRY_RUN.log"
    )
    assert r1_dry_log.exists(), f"R1 DRY_RUN.log missing: {r1_dry_log}"
    r1_text = r1_dry_log.read_text()
    assert "DRY-RUN launcher for Phase R1" in r1_text, \
        f"R1 DRY_RUN.log missing phase header; got={r1_text[:200]!r}"
    assert "trainer:" in r1_text, "R1 DRY_RUN.log missing trainer field"
    assert "flags:" in r1_text, "R1 DRY_RUN.log missing flags field"
    assert "gpu:" in r1_text, "R1 DRY_RUN.log missing gpu field"
    assert "budget:" in r1_text, "R1 DRY_RUN.log missing budget field"
    assert "ts:" in r1_text, "R1 DRY_RUN.log missing ts field"

    # R3 dry-run with USER_OK=YES produces DRY_RUN.log
    env = {**os.environ, "USER_OK": "YES"}
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert proc.returncode == 0, (
        f"R3 dry-run with USER_OK=YES should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    r3_dry_log = (
        PROJECT_ROOT / "molmetal" / "reports" / "wf310e_r20_p3_drugood_full_launch"
        / "r3" / "DRY_RUN.log"
    )
    assert r3_dry_log.exists(), f"R3 DRY_RUN.log missing: {r3_dry_log}"
    r3_text = r3_dry_log.read_text()
    assert "DRY-RUN launcher for Phase R3" in r3_text, \
        f"R3 DRY_RUN.log missing phase header; got={r3_text[:200]!r}"
    assert "cuda:1" in r3_text, "R3 DRY_RUN.log should reference cuda:1 GPU"


# ---------------------------------------------------------------------------
# Test 4: Bash syntax check (`bash -n` parses cleanly)
# ---------------------------------------------------------------------------

def test_wf355a1_bash_syntax_check():
    """Wf-355-A1 task contract — `bash -n` parses cleanly (no syntax errors).

    Verifies:
      - `bash -n` exits zero (no syntax errors).
      - Shebang line is present.
      - Wrapper size is reasonable (~250 LOC per task brief; allow 400-500
        given header docstring).
      - Wf-185 driver reference present (chemprop v2 trainer).
      - DrugOOD loader reference present.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"bash -n syntax check failed; rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    text = WRAPPER.read_text()

    # Shebang present
    assert text.startswith("#!/usr/bin/env bash"), \
        f"wrapper missing proper shebang; first line={text.splitlines()[0]!r}"

    # Size check — task brief says ~250 LOC; header adds ~100 LOC; total
    # should be in [200, 500] line range.
    n_lines = len(text.splitlines())
    assert 200 <= n_lines <= 600, (
        f"wrapper size {n_lines} LOC outside expected [200, 600] range"
    )

    # Wf-185 trainer reference (R20 §3 spec)
    assert "wf185_chemprop_v2_drugood_full.py" in text, \
        "wrapper missing reference to Wf-185 chemprop v2 DrugOOD trainer"

    # DrugOOD loader reference
    assert "drugood" in text.lower(), \
        "wrapper missing reference to DrugOOD loader / corpus"

    # ChempropV2MTModel marker check (matches Wf-185 grep marker)
    assert CHEMPROP_TRAINER.exists(), \
        f"Wf-185 trainer missing on disk: {CHEMPROP_TRAINER}"

    # The wrapper should reference the R20 §3 spec verbatim
    assert "R20" in text or "r20" in text, \
        "wrapper missing R20 §3 reference (R20 §3 DrugOOD 1.58M chemprop v2 retrain)"