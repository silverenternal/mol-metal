"""Wf-355-B1 — Wf-131 P3 7 DEFERRED runs re-launch wrapper tests.

Owner:        Hugo Lee (autonomous Wf-355-B1 ultracode unit)
Date:         2026-09-29
Parent:       Wf-355 ultracode residual wave + Wf-354-A1 §5 HONEST-NEGATIVE
              ("Wf-131 P3 Re-launch wrapper needed (USER_OK-gated for >1hr
              GPU per AGENTS.md §5.1)").

These 3 NEW tests verify that the new launch wrapper
``scripts/launch_wf131_p3_deferred_runs.sh`` satisfies the 3 critical
contract points per AGENTS.md §1.2 (Standing GPU auth) + §5.1
(kill_then_adapt directive) + Wf-294B / Wf-300C USER_OK pattern:

  1. **USER_OK gate enforced** — wrapper REFUSES to run unless USER_OK=1.
  2. **Bash syntax check** — `bash -n` parses cleanly + reasonable size.
  3. **Per-run enumeration** — 7 specific (metal, seed) pairs are listed
     in the wrapper (Rh seeds 3+4 + Re seeds 0-4).

All 3 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU launches).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-run
output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = PROJECT_ROOT / "scripts" / "launch_wf131_p3_deferred_runs.sh"
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"

# 7 DEFERRED runs per Wf-354-A1 §5 HONEST-NEGATIVE
EXPECTED_RUNS = [
    ("Rh", 3),
    ("Rh", 4),
    ("Re", 0),
    ("Re", 1),
    ("Re", 2),
    ("Re", 3),
    ("Re", 4),
]


# ---------------------------------------------------------------------------
# Test 1: USER_OK gate enforced
# ---------------------------------------------------------------------------


def test_wf355b1_user_ok_gate_enforced():
    """Wf-355-B1 task contract — USER_OK gate enforced per Wf-294B /
    Wf-300C pattern (kill_then_adapt ≥ 1 hr GPU).

    Verifies:
      - The wrapper script exists.
      - USER_OK gate function present in the source.
      - Wrapper without USER_OK env var exits with non-zero (FATAL exit 64).
      - Wrapper with USER_OK=1 --dry-run exits with zero (plan only).
      - FATAL message references USER_OK.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()
    assert "USER_OK" in text, "USER_OK env var reference missing in wrapper"
    assert (
        "USER_OK gate refused" in text or "USER_OK=" in text
    ), "USER_OK gate logic missing in wrapper"

    # Without USER_OK must die (FATAL exit 64)
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, (
        "without USER_OK should exit non-zero (gate enforced); "
        f"got rc={proc.returncode} stdout={proc.stdout[-300:]!r}"
    )
    assert (
        "USER_OK" in proc.stderr
    ), f"FATAL message should reference USER_OK; got stderr={proc.stderr[-300:]!r}"

    # With USER_OK=1 + --dry-run must pass (exit zero)
    env = {**os.environ, "USER_OK": "1"}
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, (
        f"with USER_OK=1 --dry-run should exit zero; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Bash syntax check (`bash -n` parses cleanly)
# ---------------------------------------------------------------------------


def test_wf355b1_bash_syntax_check():
    """Wf-355-B1 task contract — `bash -n` parses cleanly (no syntax errors).

    Verifies:
      - `bash -n` exits zero (no syntax errors).
      - Shebang line is present.
      - Wrapper size is reasonable (~200 LOC per task brief; allow 200-350
        given header docstring).
      - Trainer reference present (train_metal_lomo_wasserstein.py — the
        canonical Wf-131 P3 entry per Wf-354-A1 §3.2).
      - Wf-131 P3 / Wf-355-B1 cross-references present in wrapper.
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

    # Size check — task brief says ~200 LOC; header adds ~80 LOC; total
    # should be in [200, 350] line range.
    n_lines = len(text.splitlines())
    assert (
        200 <= n_lines <= 350
    ), f"wrapper size {n_lines} LOC outside expected [200, 350] range"

    # Trainer reference (canonical Wf-131 P3 entry per Wf-354-A1 §3.2)
    assert "train_metal_lomo_wasserstein.py" in text, (
        "wrapper missing reference to canonical Wf-131 P3 trainer "
        "(train_metal_lomo_wasserstein.py)"
    )

    # Wf-131 P3 + Wf-355-B1 + Wf-354-A1 cross-references
    assert (
        "Wf-131 P3" in text or "wf131_p3" in text
    ), "wrapper missing Wf-131 P3 reference"
    assert (
        "Wf-355-B1" in text or "wf355b1" in text.lower()
    ), "wrapper missing Wf-355-B1 self-reference"
    assert (
        "354a1" in text or "Wf-354-A1" in text
    ), "wrapper missing Wf-354-A1 parent reference"

    # The canonical trainer file should exist on disk
    assert TRAINER.exists(), f"Wf-131 P3 canonical trainer missing on disk: {TRAINER}"


# ---------------------------------------------------------------------------
# Test 3: Per-run enumeration (7 specific runs listed)
# ---------------------------------------------------------------------------


def test_wf355b1_per_run_enumeration_7_runs():
    """Wf-355-B1 task contract — 7 specific DEFERRED runs are listed in the
    wrapper (Rh seeds 3+4 + Re seeds 0-4 per Wf-354-A1 §5).

    Verifies:
      - DEFAULT_RUNS array contains all 7 (metal, seed) tuples.
      - Each expected run appears in the wrapper as `Rh 3`, `Rh 4`, `Re 0`,
        `Re 1`, `Re 2`, `Re 3`, `Re 4` in the body.
      - --runs "metal seed" CLI override is wired.
      - The wrapper prints the per-run list at dry-run time.
    """
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"

    text = WRAPPER.read_text()

    # DEFAULT_RUNS array literal must contain all 7 pairs
    assert "DEFAULT_RUNS" in text, "DEFAULT_RUNS array literal missing in wrapper"

    # Each expected (metal, seed) pair appears somewhere in the wrapper.
    # The DEFAULT_RUNS array uses 14 separate tokens (7 metals + 7 seeds):
    #     DEFAULT_RUNS=(
    #         "Rh" 3
    #         "Rh" 4
    #         "Re" 0
    #         ...
    #     )
    # Each (metal, seed) pair occupies two consecutive lines.
    for metal, seed in EXPECTED_RUNS:
        # Look for "Rh" <whitespace> 3 with flexible whitespace
        # (DEFAULT_RUNS array is multi-line)
        pattern = rf'"{re.escape(metal)}"[^a-zA-Z0-9"]{{1,40}}{seed}\b'
        assert re.search(
            pattern, text, re.DOTALL
        ), f"run pair ({metal}, {seed}) missing from wrapper"

    # --runs CLI override should be wired
    assert "--runs" in text, "--runs CLI override flag missing in wrapper"

    # Cross-reference: Wf-131 P3 had 18/25 done; 7 remaining — wrapper
    # should advertise n_runs=7 by default
    assert "7" in text, "wrapper missing '7' (default n_runs)"
    assert (
        "DEFERRED" in text or "deferred" in text
    ), "wrapper missing 'DEFERRED' status marker"

    # Wf-354-A1 §5 says: "training stopped at Rh seed=3 epoch=23" +
    # "Re not started before training stopped" — both should be in the wrapper
    assert (
        "Rh seed=3" in text or "Rh seed=3 epoch" in text
    ), "wrapper missing Rh seed=3 epoch=23 stop-anchor reference"
    assert (
        "Re not started" in text or "Re 0/5" in text or "0/5 (DEFERRED" in text
    ), "wrapper missing Re '0/5 DEFERRED' reference"

    # Per-run sequencing log should reference all 7 pairs
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "USER_OK": "1"},
    )
    assert proc.returncode == 0, (
        f"dry-run should exit zero with USER_OK=1; got rc={proc.returncode} "
        f"stderr={proc.stderr[-300:]!r}"
    )
    log_output = proc.stdout + proc.stderr
    for metal, seed in EXPECTED_RUNS:
        # Should appear in either stdout or the DRY_RUN.log file
        # The dry-run log prints "Rh seed=3  ->  ..." etc.
        # Match "Rh seed=3" or "Rh 3" or "Rh/seed3"
        pattern = rf"{metal}\s+seed\s*=\s*{seed}|{metal}\s+{seed}\b"
        assert re.search(pattern, log_output), (
            f"run pair ({metal}, {seed}) missing from dry-run log output; "
            f"got first 500 chars={log_output[:500]!r}"
        )
