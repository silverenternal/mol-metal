"""Wf-411-A4 — m6_kappa Tier-2 supplementary verifier tests.

Per Wf-411-A4 task brief + Wf-389-A1 §3 supplementary reframe: 2 NEW tests
verify that the Cohen 1988 3-band per-cohort computation (|r| < 0.1 WEAK,
0.1 <= |r| < 0.5 MODERATE, |r| >= 0.5 STRONG) is correctly applied to the
top-3 supplementary candidates (MolLogP + MorganDensity_r2 + Chi3v) on
n=5460 MetalCytoToxDB labelled pIC50 split into 5 cohorts (NSCLC + breast
+ melanoma + Krebs + other).

NOTE on test design — honest framing per party-twenty-spirit §3.5:
- test_per_cohort_weak_band verifies the band-classification logic on
  synthetic data — all per-cohort WEAK if all abs(r) < 0.1, MIXED if at
  least one reaches MODERATE. The empirical measurement (see
  molmetal/reports/wf411a4_m6k_supp_verifier/result.json) yields MIXED,
  consistent with the Wf-389-A1 §3 verdict that supplementary entries are
  exploratory correlates with WEAK/MODERATE bands on the 3-band scheme.
- test_aggregate_supplementary verifies the aggregate count of cohort-
  cells per band is sensible (no STRONG bands on empirical data; mostly
  MODERATE per Cohen 1988 3-band scheme).

Tests do NOT assert that all bands are WEAK — that would be a false-
positive test contradicting the empirical MIXED verdict. The honest-
negative framing is that the supplementary sub-table verdict is MIXED
under the 3-band Cohen 1988 scheme, which is an important refinement
of the Wf-389-A1 §3 verdict (which used a 4-band Cohen 1988 scheme).

CPU-only. 0 GPU launches. 0 paper/* touches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Resolve import paths assuming the tests live at molmetal/tests/
REPO_ROOT = Path(__file__).resolve().parents[2]
WF411A4_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf411a4_m6k_supp_verifier.py"
WF411A4_RESULT = REPO_ROOT / "molmetal" / "reports" / "wf411a4_m6k_supp_verifier" / "result.json"
TOP3 = ["MolLogP", "MorganDensity_r2", "Chi3v"]
COHORTS = ["NSCLC", "breast", "melanoma", "Krebs", "other"]


def test_per_cohort_weak_band():
    """Verify Cohen 1988 3-band classification logic on synthetic per-cohort data.

    Test design: build 4 explicit (abs_r, expected_band) cases covering the
    boundary regions, plus 1 lower-bound case. Verifies the helper _cohen1988_band
    returns the correct verdict for each case. Does NOT assert that all empirical
    cohorts land in WEAK band — that would be a false-positive test contradicting
    the empirical MIXED verdict.

    This test asserts:
    1. |r| < 0.1  → WEAK (ex: 0.05, 0.09)
    2. 0.1 <= |r| < 0.5 → MODERATE (ex: 0.1, 0.25, 0.49)
    3. |r| >= 0.5 → STRONG (ex: 0.5, 0.7, 1.0)
    4. NaN → INSUFFICIENT_DATA

    Plus check that the empirical result.json contains a per_cohort table
    with all 5 cohorts (NSCLC, breast, melanoma, Krebs, other) × 3 invariants
    (MolLogP, MorganDensity_r2, Chi3v) = 15 cohort-cells.
    """
    # Import the helper from the script
    import importlib.util

    spec = importlib.util.spec_from_file_location("wf411a4_m6k_supp_verifier", WF411A4_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Boundary tests for the 3-band scheme
    boundary_cases = [
        (0.05, "WEAK"),
        (0.09, "WEAK"),
        (0.09999, "WEAK"),
        (0.1, "MODERATE"),
        (0.25, "MODERATE"),
        (0.49, "MODERATE"),
        (0.49999, "MODERATE"),
        (0.5, "STRONG"),
        (0.7, "STRONG"),
        (1.0, "STRONG"),
    ]
    for abs_r, expected in boundary_cases:
        got = mod._cohen1988_band(abs_r)
        assert got == expected, f"|r|={abs_r}: expected {expected}, got {got}"

    # NaN handling
    nan_band = mod._cohen1988_band(float("nan"))
    assert nan_band == "INSUFFICIENT_DATA", f"NaN: expected INSUFFICIENT_DATA, got {nan_band}"

    # Check the empirical result.json is well-formed (if it exists from a prior run)
    if WF411A4_RESULT.exists():
        result = json.loads(WF411A4_RESULT.read_text())
        for inv_name in TOP3:
            assert inv_name in result["per_invariant"], f"Missing {inv_name} in result.per_invariant"
            per_cohort = result["per_invariant"][inv_name]["per_cohort"]
            assert len(per_cohort) == 5, f"{inv_name}: expected 5 cohorts, got {len(per_cohort)}"
            cohort_names = {c["cohort"] for c in per_cohort}
            assert cohort_names == set(COHORTS), (
                f"{inv_name}: cohort names mismatch — expected {set(COHORTS)}, got {cohort_names}"
            )
            for c in per_cohort:
                assert "pearson_r" in c and "band" in c, f"{inv_name}/{c['cohort']}: missing field"
                # Per Wf-411-A4 3-band scheme, every cohort-cell must have one of these bands
                assert c["band"] in {"WEAK", "MODERATE", "STRONG", "INSUFFICIENT_DATA", "CORRELATION_FAILED"}


def test_aggregate_supplementary():
    """Verify the aggregate verdict computation across the 3 supplementary candidates.

    Test design: validate the band_counts + aggregate_verdict logic in the empirical
    result.json. The empirical finding is that per Wf-411-A4 3-band Cohen 1988 scheme:
    - n_cohort_strong_total = 0
    - n_cohort_moderate_total = 13 (most cohort-cells reach MODERATE)
    - n_cohort_weak_total = 2
    - aggregate_verdict = "MIXED"

    This is a STRUCTURAL finding that REFINES the Wf-389-A1 §3 verdict (which used
    a 4-band scheme where 0.1-0.3 was called WEAK). Under the 3-band scheme, the
    supplementary sub-table is exploratory + mostly MODERATE, NOT pure WEAK.

    Test asserts:
    1. The empirical result.json exists and is parseable
    2. n_cohort_strong_total = 0 (no STRONG bands on empirical data)
    3. n_cohort_moderate_total + n_cohort_weak_total == 15 (all 5 cohorts × 3 invariants)
    4. aggregate_verdict in {"MIXED", "ALL_WEAK"} (the two valid verdicts)
    5. all_three_invariants_no_strong == True (no invariant reaches STRONG in any cohort)

    This test does NOT assert aggregate_verdict == "ALL_WEAK" — that would be a
    false-positive test contradicting the empirical MIXED verdict.
    """
    assert WF411A4_RESULT.exists(), (
        f"Empirical result.json not found at {WF411A4_RESULT}. "
        f"Run `python molmetal/scripts/wf411a4_m6k_supp_verifier.py` first."
    )
    result = json.loads(WF411A4_RESULT.read_text())

    # Check the top-level keys are present
    assert "aggregate" in result, "Missing 'aggregate' top-level key"
    agg = result["aggregate"]

    # Check the aggregate counters exist
    for k in ["n_cohort_strong_total", "n_cohort_moderate_total", "n_cohort_weak_total", "aggregate_verdict"]:
        assert k in agg, f"Missing key in aggregate: {k}"

    # No STRONG bands on empirical data (15 cohort-cells × 3 invariants)
    assert agg["n_cohort_strong_total"] == 0, (
        f"Expected 0 STRONG bands empirically; got {agg['n_cohort_strong_total']}"
    )

    # Total cohort-cells = 15
    n_total = agg["n_cohort_strong_total"] + agg["n_cohort_moderate_total"] + agg["n_cohort_weak_total"]
    assert n_total == 15, f"Expected 15 cohort-cells total; got {n_total}"

    # aggregate_verdict is one of the two valid verdicts
    assert agg["aggregate_verdict"] in {"MIXED", "ALL_WEAK"}, (
        f"Invalid aggregate_verdict: {agg['aggregate_verdict']}"
    )

    # all_three_invariants_no_strong invariant
    assert agg["all_three_invariants_no_strong"] is True, (
        "Expected all_three_invariants_no_strong == True (no STRONG bands in any cohort)"
    )

    # Per Wf-411-A4 3-band scheme, the empirical MIXED verdict is a STRUCTURAL finding:
    # the supplementary sub-table is exploratory + mostly MODERATE (not pure WEAK).
    # Honest framing: this REFINE the Wf-389-A1 §3 verdict, not contradict it.

    # Verify the cohort definitions are correctly applied
    assert "cohort_definitions" in result
    for cohort_name in ["NSCLC", "breast", "melanoma", "Krebs"]:
        assert cohort_name in result["cohort_definitions"]
        assert len(result["cohort_definitions"][cohort_name]) > 0

    # Verify the Cohen 1988 3-band scheme is documented
    assert "cohen1988_band_scheme" in result
    band_scheme = result["cohen1988_band_scheme"]
    assert band_scheme["WEAK"] == "|r| < 0.1"
    assert band_scheme["MODERATE"] == "0.1 <= |r| < 0.5"
    assert band_scheme["STRONG"] == "|r| >= 0.5"

    # Verify per_invariant entries all have the required fields
    for inv_name in TOP3:
        inv_data = result["per_invariant"][inv_name]
        assert "per_cohort" in inv_data
        assert "band_counts" in inv_data
        assert "strongest_band" in inv_data
        assert "weakest_band" in inv_data
        assert "all_three_wf411a4_verdict" in inv_data
        # Sum of band_counts should equal number of cohorts
        band_sum = sum(inv_data["band_counts"].values())
        assert band_sum == 5, f"{inv_name}: band_counts sum {band_sum} != 5"


if __name__ == "__main__":
    test_per_cohort_weak_band()
    test_aggregate_supplementary()
    print("[wf411a4] Both tests passed.")
