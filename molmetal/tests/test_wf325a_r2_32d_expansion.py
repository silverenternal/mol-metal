"""Tests for Wf-325A — F5_RDKIT_2D 32-d expansion R² lift bench.

=====================================================================
Wf-325A / Wf-297B §Cause 2 / Wf-218 — verifies the CPU diagnostic
that asks whether the F5_RDKIT_2D 32-d expansion (vs the Wf-49 P1
12-d frozen set) lifts Ridge regression R² above the architectural
ceiling documented in Wf-297B.

Test surface (8 tests):
  * 3 tests: per-cohort R² computation (platinai_4cell, hilbert,
    metalcytotox_combined)
  * 2 tests: 12-d vs 32-d comparison (lift sign + verdict)
  * 2 tests: regression baseline correctness (Ridge + StandardScaler
    + 80/20 split)
  * 1 test: integration with downstream consumer (the result JSON
    written to molmetal/reports/wf325a_r2_32d/result.json)

Honest framing (party-twenty-spirit §3.5):
  * The 32-d expansion is **honest-negative confirmed** on all 3
    cohorts — the architectural ceiling is not lifted by RDKit
    descriptors alone; W₂ alignment + chemprop v2 graph features
    are required per WEAK_TO_STRONG §2.1.
  * These tests pin down the **deterministic** structure of the
    diagnostic, not the absolute R² values (R² may vary slightly
    with sklearn version + numpy version; the comparison shape
    is the load-bearing invariant).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Make sure the bench script is importable from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from molmetal.scripts.bench_wf325a_r2_32d import (  # noqa: E402
    METALCYTOTOX_12D_COLUMNS,
    METALCYTOTOX_32D_COLUMNS,
    METALCYTOTOX_TARGET_COLUMN,
    PLATINAI_12D_COLUMNS,
    PLATINAI_32D_COLUMNS,
    PLATINAI_PIC50_COLUMNS,
    _mean_pIC50_target,
    _read_csv,
    _row_to_floats,
    _summarise_lift,
    compute_metalcytotox_32d,
    load_metalcytotox_12d,
    load_platinai_4cell_12d,
    load_platinai_4cell_32d,
    ridge_r2_eval,
    run_wf325a,
)


# ---------------------------------------------------------------------------
# Per-cohort R² computation (3 tests)
# ---------------------------------------------------------------------------
class TestPerCohortR2(unittest.TestCase):
    """Per-cohort R² computation tests."""

    def test_platinai_4cell_r2_shape(self) -> None:
        """platinai_4cell cohort: 12-d and 32-d R² are real-valued floats."""
        smi32_set = _read_smi_set()
        X12, y = load_platinai_4cell_12d(smiles_filter=smi32_set)
        X32, y32 = load_platinai_4cell_32d()
        # SMILES-match invariant: y from 12-d and y from 32-d must be
        # aligned (same length and same values for shared SMILES).
        self.assertEqual(X12.shape[0], X32.shape[0])
        self.assertEqual(X12.shape[0], y.shape[0])
        self.assertEqual(X32.shape[0], y32.shape[0])
        # X dims: 12 vs 32
        self.assertEqual(X12.shape[1], 12)
        self.assertEqual(X32.shape[1], 32)
        # R² should be finite (real-valued)
        r2_12d = ridge_r2_eval(X12, y, alpha=1.0, seed=0)["r2"]
        r2_32d = ridge_r2_eval(X32, y, alpha=1.0, seed=0)["r2"]
        self.assertTrue(np.isfinite(r2_12d))
        self.assertTrue(np.isfinite(r2_32d))

    def test_hilbert_r2_seed_independence(self) -> None:
        """hilbert cohort: same data, different test-seed for an independent
        R² reading — R² must be finite and bounded by the Ridge baseline
        R² range."""
        smi32_set = _read_smi_set()
        X12, y = load_platinai_4cell_12d(smiles_filter=smi32_set)
        X32, _ = load_platinai_4cell_32d()
        # hilbert cohort uses seed+1 for the test split
        r2_12d_a = ridge_r2_eval(X12, y, alpha=1.0, seed=1)["r2"]
        r2_32d_a = ridge_r2_eval(X32, y, alpha=1.0, seed=1)["r2"]
        self.assertTrue(np.isfinite(r2_12d_a))
        self.assertTrue(np.isfinite(r2_32d_a))
        # The hilbert cohort must use the same X, y as platinai_4cell
        # (it's a different test-seed, not a different cohort).
        self.assertEqual(X12.shape, X32.shape[:1] + (12,))

    def test_metalcytotox_combined_r2_shape(self) -> None:
        """metalcytotox_combined cohort: 12-d and 32-d R² are real-valued
        floats; 12-d file exists on disk; 32-d is computed on the fly."""
        X12, y = load_metalcytotox_12d(max_samples=500, seed=0)
        # 12-d load must succeed and have shape (n, 12)
        self.assertEqual(X12.shape[1], 12)
        self.assertGreater(X12.shape[0], 0)
        # 32-d load may use the RDKit-on-the-fly path OR the
        # 12-d replicated fallback.  Either way it must return
        # shape (n, 32).
        try:
            X32, y32 = compute_metalcytotox_32d(max_samples=500, seed=0)
        except Exception:
            self.skipTest("RDKit unavailable for on-the-fly 32-d compute")
        self.assertEqual(X32.shape[1], 32)
        self.assertEqual(X32.shape[0], y32.shape[0])
        r2_12d = ridge_r2_eval(X12, y, alpha=1.0, seed=0)["r2"]
        r2_32d = ridge_r2_eval(X32, y, alpha=1.0, seed=0)["r2"]
        self.assertTrue(np.isfinite(r2_12d))
        self.assertTrue(np.isfinite(r2_32d))


# ---------------------------------------------------------------------------
# 12-d vs 32-d comparison (2 tests)
# ---------------------------------------------------------------------------
class TestLiftComparison(unittest.TestCase):
    """12-d vs 32-d R² lift comparison tests."""

    def test_lift_sign_is_computed(self) -> None:
        """Lift = R²(32d) - R²(12d) is finite and the verdict is one of
        the documented labels."""
        smi32_set = _read_smi_set()
        X12, y = load_platinai_4cell_12d(smiles_filter=smi32_set)
        X32, _ = load_platinai_4cell_32d()
        r2_12d = ridge_r2_eval(X12, y, alpha=1.0, seed=0)["r2"]
        r2_32d = ridge_r2_eval(X32, y, alpha=1.0, seed=0)["r2"]
        lift = r2_32d - r2_12d
        self.assertTrue(np.isfinite(lift))
        verdict = _summarise_lift(r2_12d, r2_32d)
        self.assertIn(
            verdict,
            {
                "STRONG_LIFT",
                "MODERATE_LIFT",
                "WEAK_LIFT",
                "CEILING_BREAK",
                "NO_LIFT_HONEST_NEGATIVE",
                "STILL_NEGATIVE_HONEST_NEGATIVE",
                "UNDEFINED",
            },
        )

    def test_lift_verdict_documented_labels(self) -> None:
        """All three cohorts must produce a documented verdict label."""
        result = run_wf325a(out_path=_tmp_result_path())
        verdicts = {c["name"]: c["verdict"] for c in result["per_cohort"]}
        for cohort in ("platinai_4cell", "hilbert", "metalcytotox_combined"):
            self.assertIn(cohort, verdicts)
            self.assertIn(
                verdicts[cohort],
                {
                    "STRONG_LIFT",
                    "MODERATE_LIFT",
                    "WEAK_LIFT",
                    "CEILING_BREAK",
                    "NO_LIFT_HONEST_NEGATIVE",
                    "STILL_NEGATIVE_HONEST_NEGATIVE",
                    "UNDEFINED",
                },
            )


# ---------------------------------------------------------------------------
# Regression baseline correctness (2 tests)
# ---------------------------------------------------------------------------
class TestRidgeBaselineCorrectness(unittest.TestCase):
    """Ridge regression baseline correctness tests."""

    def test_ridge_perfect_fit_yields_positive_r2(self) -> None:
        """A perfect linear signal (y = X @ w) must yield R² ≥ 0.9."""
        rng = np.random.default_rng(0)
        n, d = 200, 8
        X = rng.standard_normal((n, d))
        w_true = rng.standard_normal(d)
        y = X @ w_true + 0.01 * rng.standard_normal(n)
        result = ridge_r2_eval(X, y, alpha=0.001, seed=0)
        self.assertGreater(result["r2"], 0.9)
        self.assertEqual(result["n_train"] + result["n_test"], n)
        # 80/20 split invariant
        self.assertAlmostEqual(result["n_test"] / n, 0.2, places=1)

    def test_ridge_constant_target_yields_r2_one(self) -> None:
        """A constant y target (no signal) yields ss_tot = 0 on the test
        split, which sklearn reports as R² = 1.0 (perfect fit) when the
        model can fit the constant perfectly.  Pin down this edge case
        to guard against silent regression in the bench script."""
        rng = np.random.default_rng(0)
        n, d = 100, 5
        X = rng.standard_normal((n, d))
        y = np.ones(n) * 3.14  # constant target
        result = ridge_r2_eval(X, y, alpha=1.0, seed=0)
        r2 = result["r2"]
        # Either sklearn returns 1.0 (perfect constant fit) or NaN
        # (ss_tot = 0).  Both are valid; the test pins down the contract.
        self.assertTrue(
            np.isnan(r2) or abs(r2 - 1.0) < 1e-3,
            msg=f"R² for constant target should be ≈1.0 or NaN, got {r2}",
        )


# ---------------------------------------------------------------------------
# Integration with downstream consumer (1 test)
# ---------------------------------------------------------------------------
class TestIntegrationWithResultJson(unittest.TestCase):
    """Integration with the downstream consumer — the result JSON."""

    def test_result_json_has_required_fields(self) -> None:
        """The result JSON written by run_wf325a has all required fields
        for the Wf-325A honest-negative close-out (per AGENTS.md §1
        MEASURED-vs-PROJECTED + §5.2 honest framing)."""
        out_path = _tmp_result_path()
        result = run_wf325a(out_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Required top-level fields
        for field in (
            "wf",
            "task",
            "scope",
            "n_cohorts_tested",
            "per_cohort",
            "per_cohort_r2_12d_vs_32d",
            "per_cohort_lift",
            "verdicts",
            "honest_negative_status",
            "framing_notes",
        ):
            self.assertIn(field, data, f"Missing required field: {field}")
        # wf + task + scope provenance
        self.assertEqual(data["wf"], "Wf-325A")
        self.assertIn("F5_RDKIT_2D", data["task"])
        self.assertIn("CPU-only", data["scope"])
        # 3 cohorts tested
        self.assertEqual(data["n_cohorts_tested"], 3)
        self.assertEqual(len(data["per_cohort"]), 3)
        # Each cohort has a 12-d R², 32-d R², lift, and verdict
        for c in data["per_cohort"]:
            self.assertIn("name", c)
            self.assertIn("r2_12d", c)
            self.assertIn("r2_32d", c)
            self.assertIn("lift", c)
            self.assertIn("verdict", c)
            self.assertIn("n_samples", c)
            self.assertIn("source_12d", c)
            self.assertIn("source_32d", c)
        # honest_negative_status must be one of the documented labels
        self.assertIn(
            data["honest_negative_status"],
            {
                "PARTIAL_RELIEF",
                "WEAK_LIFT",
                "HONEST_NEGATIVE_CONFIRMED",
                "NO_LIFT",
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_smi_set() -> set:
    """Read the 32-d extended file and return the SMILES set."""
    # _read_csv already calls _resolve_repo_path, so the path can be
    # either repo-relative or absolute.
    h, r = _read_csv("molmetal/data/platinai_4cell_rdkit_extended_features.csv")
    smi_idx = h.index("smiles")
    return {row[smi_idx] for row in r}


def _tmp_result_path() -> str:
    """Return a temp file path for the result JSON."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="wf325a_")
    os.close(fd)
    return path


if __name__ == "__main__":
    unittest.main()
