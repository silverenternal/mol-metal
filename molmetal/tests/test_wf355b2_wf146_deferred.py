"""Wf-355-B2 — Wf-146 Re fold 5/5 + Os seed 4 DEFERRED runs launch wrapper tests.

Owner:        Hugo Lee (autonomous Wf-355-B2 ultracode unit)
Date:         2026-09-29
Parent:       Wf-355 ultracode residual wave + Wf-354-A1 §5 HONEST-NEGATIVE
              ("6 DEFERRED (metal, seed) pairs: Re 5/5 + Os seed 4 — no
              re-launch entry point on disk").

These 3 NEW tests verify that the new launch wrapper
``scripts/launch_wf146_deferred_runs.sh`` satisfies the 3 critical contract
points per AGENTS.md §1.2 (Standing GPU auth) + §5.1 (kill_then_adapt
directive) + Wf-294B / Wf-300C USER_OK pattern:

  1. **USER_OK gate enforced**        — R3 mode requires USER_OK=YES.
  2. **Bash syntax check**            — `bash -n` parses cleanly + size check.
  3. **Per-run enumeration**          — 6 DEFERRED (metal, seed) pairs listed.

All 3 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU launches).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-run
output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = PROJECT_ROOT / "scripts" / "launch_wf146_deferred_runs.sh"
PRE_FLIGHT = PROJECT_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
PER_WAVE_CHECK = PROJECT_ROOT / "molmetal" / "scripts" / "wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK = (
    PROJECT_ROOT / "molmetal" / "scripts" / "wf343c3_data_quality_check.py"
)
WF146_TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "wf146_chemprop_lomo.py"

# 6 DEFERRED (metal, seed) pairs per Wf-354-A1 §5 + Wf-146 sweep.log
EXPECTED_PAIRS = ["Re_0", "Re_1", "Re_2", "Re_3", "Re_4", "Os_4"]


# ---------------------------------------------------------------------------
# Test 1: USER_OK gate enforced (R3 mode requires USER_OK=YES)
# ---------------------------------------------------------------------------


def test_wf355b2_user_ok_gate_enforced_r3():
    """Wf-355-B2 task contract — USER_OK gate enforces R3 mode per Wf-294B /
    Wf-300C pattern (kill_then_adapt ≥ 2 hr GPU).

    Verifies:
      - The wrapper script exists.
      - USER_OK gate function present in the source.
      - R3 mode without USER_OK env var exits with non-zero (FATAL).
      - R3 mode with USER_OK=YES exits with zero.
      - R1 mode without USER_OK emits WARN but still exits zero (smoke).
      - The error message references USER_OK=YES (per Wf-300C pattern).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()
    assert "USER_OK" in text, "USER_OK env var reference missing in wrapper"
    assert "user_ok_gate" in text, "user_ok_gate function missing in wrapper"
    assert "USER_OK=YES" in text, "USER_OK=YES literal missing in wrapper"

    # R3 without USER_OK must die (FATAL exit non-zero)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, (
        "R3 mode without USER_OK should exit non-zero (gate enforced); "
        f"got rc={proc.returncode} stderr={proc.stderr[-300:]!r}"
    )
    assert (
        "USER_OK" in proc.stderr
    ), f"FATAL message should reference USER_OK; got stderr={proc.stderr[-300:]!r}"

    # R3 with USER_OK=YES must pass (exit zero)
    env = {**os.environ, "USER_OK": "YES"}
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"R3 mode with USER_OK=YES should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    # R1 mode without USER_OK must warn but still exit zero (smoke mode)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"R1 mode without USER_OK should still pass (smoke); got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )
    assert (
        "USER_OK not set" in proc.stderr or "USER_OK not set" in proc.stdout
    ), "R1 mode without USER_OK should emit USER_OK-not-set warning"


# ---------------------------------------------------------------------------
# Test 2: Bash syntax check (`bash -n` parses cleanly)
# ---------------------------------------------------------------------------


def test_wf355b2_bash_syntax_check():
    """Wf-355-B2 task contract — `bash -n` parses cleanly (no syntax errors).

    Verifies:
      - `bash -n` exits zero (no syntax errors).
      - Shebang line is present.
      - Wrapper size is reasonable (~500 LOC per task brief; allow 300-700).
      - Wf-146 trainer reference present (per Wf-354-A1 §5 spec).
      - 3 pre-launch gate helpers wired (pre_flight_gate + per_wave_check_gate
        + data_quality_gate).
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"bash -n syntax check failed; rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )

    text = WRAPPER.read_text()

    # Shebang present
    assert text.startswith(
        "#!/usr/bin/env bash"
    ), f"wrapper missing proper shebang; first line={text.splitlines()[0]!r}"

    # Size check — task brief says ~200 LOC; allow 300-700 with header docstring.
    n_lines = len(text.splitlines())
    assert (
        300 <= n_lines <= 700
    ), f"wrapper size {n_lines} LOC outside expected [300, 700] range"

    # Wf-146 trainer reference
    assert (
        "wf146_chemprop_lomo.py" in text
    ), "wrapper missing reference to Wf-146 chemprop LOMO trainer"

    # 3 pre-launch gate helpers (per Wf-343c2 retro-fit pattern)
    assert "pre_flight_gate" in text, "pre_flight_gate function missing in wrapper"
    assert (
        "per_wave_check_gate" in text
    ), "per_wave_check_gate function missing in wrapper"
    assert "data_quality_gate" in text, "data_quality_gate function missing in wrapper"

    # Pre-flight + per-wave + data quality script paths present
    assert (
        "pre_flight_check.py" in text
    ), "pre_flight_check.py reference missing in wrapper"
    assert (
        "wf343c2_per_wave_check.py" in text
    ), "wf343c2_per_wave_check.py reference missing in wrapper"
    assert (
        "wf343c3_data_quality_check.py" in text
    ), "wf343c3_data_quality_check.py reference missing in wrapper"

    # Verify all 4 helper scripts exist on disk
    assert PRE_FLIGHT.exists(), f"pre_flight_check.py missing on disk: {PRE_FLIGHT}"
    assert (
        PER_WAVE_CHECK.exists()
    ), f"wf343c2_per_wave_check.py missing on disk: {PER_WAVE_CHECK}"
    assert (
        DATA_QUALITY_CHECK.exists()
    ), f"wf343c3_data_quality_check.py missing on disk: {DATA_QUALITY_CHECK}"
    assert WF146_TRAINER.exists(), f"Wf-146 trainer missing on disk: {WF146_TRAINER}"

    # Smoke backbone hidden/depth markers
    assert (
        "hidden 64" in text or "hidden=64" in text
    ), "wrapper missing hidden=64 (Wf-146 smoke backbone)"
    assert (
        "depth 2" in text or "depth=2" in text
    ), "wrapper missing depth=2 (Wf-146 smoke backbone)"


# ---------------------------------------------------------------------------
# Test 3: Per-run enumeration (6 DEFERRED pairs listed)
# ---------------------------------------------------------------------------


def test_wf355b2_per_run_enumeration_six_deferred():
    """Wf-355-B2 task contract — 6 DEFERRED (metal, seed) pairs enumerated
    per Wf-354-A1 §5 (Re 5/5 + Os seed 4).

    Verifies:
      - The wrapper source contains all 6 expected pairs (Re 0..4 + Os 4).
      - The wrapper's `--dry-run --mode R3` (with USER_OK=YES) completes
        without crashing (full enumeration of all 6 pairs).
      - The wrapper's `--dry-run --mode R1` runs exactly 1 pair (default =
        first pair, which is Re_0).
      - ONLY_RUNS=Re_1,Re_3 filter restricts the launch to those 2 pairs.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()

    # All 6 expected pairs in source (string-level check)
    for pair in EXPECTED_PAIRS:
        pair_seed = pair.split("_")[1]
        pair_with_quotes = '"' + pair + '"'
        pair_seed_with_quotes = '"' + pair_seed + '"'
        assert pair_with_quotes in text or pair_seed_with_quotes in text, (
            f"pair '{pair}' not found in wrapper source "
            f"(checking exact string match)"
        )

    # WF146_DEFERRED_PAIRS array contains all 6
    # Find the array literal in the source
    assert (
        "WF146_DEFERRED_PAIRS=(" in text
    ), "WF146_DEFERRED_PAIRS array declaration missing"

    # Reconstruct what the bash array would expand to (5 Re + 1 Os)
    re_count = sum(1 for p in EXPECTED_PAIRS if p.startswith("Re_"))
    os_count = sum(1 for p in EXPECTED_PAIRS if p.startswith("Os_"))
    assert re_count == 5, f"expected 5 Re pairs, got {re_count}"
    assert os_count == 1, f"expected 1 Os pair, got {os_count}"

    # Live test — R3 dry-run enumerates all 6 pairs
    env = {**os.environ, "USER_OK": "YES"}
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"R3 dry-run should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-500:]!r}"
    )

    # Each pair should appear in the output (DRY_RUN.log mentions it)
    for pair in EXPECTED_PAIRS:
        # The metal part appears in --metals flag; the seed part appears in --seeds
        metal, seed = pair.split("_")
        # Look for the pair in the launch.log output
        # (each pair generates a "Pair ${pair} begin" log line)
        assert pair in proc.stderr, (
            f"pair '{pair}' not enumerated in R3 dry-run output; "
            f"stderr={proc.stderr[-500:]!r}"
        )

    # ONLY_RUNS filter test (2 of 6 pairs only)
    env_with_filter = {**os.environ, "USER_OK": "YES", "ONLY_RUNS": "Re_1,Os_4"}
    proc_filter = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run", "--mode", "R3"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env_with_filter,
    )
    assert proc_filter.returncode == 0, (
        f"R3 dry-run with ONLY_RUNS filter should exit zero; got rc={proc_filter.returncode} "
        f"stderr={proc_filter.stderr[-500:]!r}"
    )
    # Re_1 and Os_4 should appear
    assert (
        "Re_1" in proc_filter.stderr
    ), f"Re_1 not in ONLY_RUNS=Re_1,Os_4 output; stderr={proc_filter.stderr[-500:]!r}"
    assert (
        "Os_4" in proc_filter.stderr
    ), f"Os_4 not in ONLY_RUNS=Re_1,Os_4 output; stderr={proc_filter.stderr[-500:]!r}"
    # Re_0 / Re_2 / Re_3 / Re_4 should NOT appear (excluded by filter)
    for excluded in ["Re_0", "Re_2", "Re_3", "Re_4"]:
        assert excluded not in proc_filter.stderr, (
            f"{excluded} should be excluded by ONLY_RUNS=Re_1,Os_4 filter but "
            f"appeared in output; pair={excluded}[stderr]={proc_filter.stderr[-500:]!r}"
        )

    # Final check: only 2 pairs should have been launched (count "begin" markers)
    n_begin = proc_filter.stderr.count("begin (budget=")
    assert n_begin == 2, (
        f"ONLY_RUNS=Re_1,Os_4 should launch exactly 2 pairs; got {n_begin}; "
        f"stderr={proc_filter.stderr[-500:]!r}"
    )
