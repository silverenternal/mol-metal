#!/usr/bin/env python3
# Copyright (c) 2026 Mol-Metal authors. Licensed under the project's license.
"""
Tests for Wf-398-A2 m2 R² real-cohort verification.

Wave: Wf-398 (decode_ratio regression diagnosis + m2 R² recovery).
CPU-only tests. Validates:
  - per_cohort_r2 computed (returns non-zero for each cohort)
  - lift_vs_baseline (platinai_4cell R² > 0 if lift successful)
  - hilbert_lift (hilbert R² > 0 if lift successful)

Honest framing: tests do NOT fabricate Wf-398-A1 results. They verify
the verification logic + structural assertions against MEASURED data.
If the upstream retrain does not exist on disk, the test falls back to
Wf-388-A1 (the most recent MEASURED real-cohort m2 R² result).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf398a2_m2_r2_real_verify.py"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf398a2_m2_r2_real_verify"
RESULT_JSON = OUT_DIR / "result.json"


def _run_script() -> dict:
    """Run the verification script and return the parsed result.json."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    assert RESULT_JSON.exists(), f"result.json not produced at {RESULT_JSON}"
    with open(RESULT_JSON) as f:
        return json.load(f)


def test_per_cohort_r2_computed() -> None:
    """Per-cohort R² is computed for each tested cohort.

    Asserts that the verification script returns a non-empty per_cohort
    list with at least one entry per tested cohort. The R² value may be
    negative (HONEST_NEGATIVE_R2_DEEP) — that is the structural finding.
    """
    result = _run_script()
    assert result["wf"] == "Wf-398-A2"
    cohorts = result["cohorts_tested"]
    assert len(cohorts) >= 1, "no cohorts tested"
    per_cohort = result["per_cohort"]
    assert len(per_cohort) == len(cohorts), (
        f"per_cohort count {len(per_cohort)} != cohorts_tested {len(cohorts)}"
    )
    for entry in per_cohort:
        # R² may be negative — that's the honest-negative finding.
        assert isinstance(entry["r2_mean"], (int, float)), (
            f"r2_mean not numeric for {entry['cohort']}"
        )
        assert entry["r2_mean"] != 0.0, (
            f"r2_mean exactly 0.0 (suspicious — verify upstream)"
        )
        # Lift vs Wf-335-M2 baseline is computed (numerical field).
        assert "lift_vs_wf335_m2" in entry, "lift_vs_wf335_m2 missing"
        assert isinstance(entry["lift_vs_wf335_m2"], (int, float)), (
            f"lift_vs_wf335_m2 not numeric for {entry['cohort']}"
        )
        # Verdict must be a known classification.
        valid_verdicts = {
            "HONEST_NEGATIVE_R2_DEEP",
            "NO_LIFT_HONEST_NEGATIVE",
            "WEAK_RECOVERY",
            "CEILING_BREAK",
        }
        assert entry["verdict"] in valid_verdicts, (
            f"unknown verdict {entry['verdict']!r} for {entry['cohort']}"
        )


def test_lift_vs_baseline() -> None:
    """platinai_4cell R² lift vs Wf-335-M2 baseline is computed correctly.

    The test does NOT assert a positive lift (that would be aspirational
    per party-twenty-spirit §3.5 honest framing). It asserts that the
    lift computation matches the formula: r2_mean - baseline.
    """
    result = _run_script()
    per_cohort = {p["cohort"]: p for p in result["per_cohort"]}

    # If platinai_4cell is in the result, verify lift arithmetic.
    if "platinai_4cell" in per_cohort:
        entry = per_cohort["platinai_4cell"]
        baseline = entry["baseline_wf335_m2"]
        r2 = entry["r2_mean"]
        expected_lift = r2 - baseline
        actual_lift = entry["lift_vs_wf335_m2"]
        assert abs(actual_lift - expected_lift) < 1e-6, (
            f"lift arithmetic mismatch: {actual_lift} != {expected_lift}"
        )

        # Honest-negative test: if r2 is negative, lift is negative (or zero).
        # If r2 is positive, test that lift is consistent with above_zero flag.
        assert entry["above_zero"] == (r2 > 0.0), (
            f"above_zero flag inconsistent: r2={r2}, above_zero={entry['above_zero']}"
        )
    else:
        # platinai_4cell missing — accept this as HONEST_NEGATIVE_MISSING_UPSTREAM.
        assert result["aggregate_status"] in {
            "HONEST_NEGATIVE_MISSING_UPSTREAM",
            "EMPTY",
        }


def test_hilbert_lift() -> None:
    """hilbert R² lift computation matches arithmetic; verdict is recorded.

    The test does NOT assert a positive lift. It validates that:
      1. hilbert is in the per_cohort list when present.
      2. lift arithmetic matches r2_mean - baseline.
      3. verdict is a known classification string.
    """
    result = _run_script()
    per_cohort = {p["cohort"]: p for p in result["per_cohort"]}

    if "hilbert" in per_cohort:
        entry = per_cohort["hilbert"]
        baseline = entry["baseline_wf335_m2"]
        r2 = entry["r2_mean"]
        expected_lift = r2 - baseline
        actual_lift = entry["lift_vs_wf335_m2"]
        assert abs(actual_lift - expected_lift) < 1e-6, (
            f"hilbert lift arithmetic mismatch: {actual_lift} != {expected_lift}"
        )
        # Verdict must be recorded.
        assert entry["verdict"], "hilbert verdict empty"
        assert entry["honest_framing"], "hilbert honest_framing empty"
    else:
        # hilbert missing — accept HONEST_NEGATIVE_MISSING_UPSTREAM fallback.
        assert result["aggregate_status"] in {
            "HONEST_NEGATIVE_MISSING_UPSTREAM",
            "EMPTY",
        }


if __name__ == "__main__":
    test_per_cohort_r2_computed()
    print("PASS test_per_cohort_r2_computed")
    test_lift_vs_baseline()
    print("PASS test_lift_vs_baseline")
    test_hilbert_lift()
    print("PASS test_hilbert_lift")
