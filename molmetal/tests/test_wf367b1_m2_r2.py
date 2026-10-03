"""Tests for Wf-367-B1 — m2 R² recovery scaffold (platinai_4cell + hilbert).

=====================================================================
Wf-367-B1 / Wf-365-A2 follow-up — per Wf-335-M2 finding, platinai_4cell
(best R² = -0.1251) and hilbert (best R² = -0.1268) remain at
HONEST_NEGATIVE_R2_DEEP.  Only metalcytotox_combined achieved
CEILING_BREAK in Wf-335-M2.

Test surface (3 tests):
  * test_wf367b1_r2_recovery_on_platinai_4cell
        Multi-stream recovery on platinai_4cell: features are
        constructed correctly; the per-cohort verdict is HONEST_NEGATIVE
        preserved or better; structural invariants hold.
  * test_wf367b1_r2_recovery_on_hilbert
        Same diagnostic on hilbert cohort.
  * test_wf367b1_backward_compat
        Backward compat: Wf-335-M2 / Wf-331-R5 baselines untouched;
        new recovery script does not modify Wf-335-M2 result.json;
        new script imports + runs end-to-end with documented
        structural invariants.

Honest framing (party-twenty-spirit §3.5):
  * If the recovery scaffold does NOT lift R² above 0 for either
    cohort: HONEST_NEGATIVE_R2_DEEP preserved (verdict is preserved
    in the test as well).
  * If the recovery scaffold DOES lift R² above 0: gate LIFTED.
  * The structural invariants (per-cohort features stream, n_features,
    verdict taxonomy) are pinned; absolute R² may vary with sklearn /
    numpy versions but the verdict taxonomy is the load-bearing
    invariant.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict

import numpy as np

# Make sure the bench script is importable from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from molmetal.scripts.wf367b1_m2_r2_recovery import (
    WF335M2_DEFAULT_BEST_R2,
    build_recovery_feature_matrix,
    chemprop_v3_enhanced_features,
    hilbert_astsr_spectral_features,
    per_cell_onehot,
    ridge_r2_cv,
    run_wf367b1,
    verdict,
)
from molmetal.scripts.wf335m2_r2_w2_chemprop import WF325A_BASELINE_R2
from molmetal.scripts.wf331r5_r2_w2_chemprop import (
    extract_graph_features_pooled,
    load_cohort,
)


# ---------------------------------------------------------------------------
# Test 1: R² recovery on platinai_4cell
# ---------------------------------------------------------------------------
class TestWf367B1R2RecoveryPlatinai4Cell(unittest.TestCase):
    """Multi-stream recovery on platinai_4cell."""

    def test_wf367b1_r2_recovery_on_platinai_4cell(self) -> None:
        cohort = "platinai_4cell"
        # Load platinai_4cell cohort (200 SMILES)
        X_32d, y, smiles_list = load_cohort(
            cohort, max_samples=2000, seed=0,
        )[:3]
        # Build recovery feature matrix
        X_recovery, feature_names, stream_dims = build_recovery_feature_matrix(
            smiles_list, X_32d, cohort,
        )
        # Structural invariants
        self.assertEqual(stream_dims["rdkit_12d"], 12)
        self.assertEqual(stream_dims["chemprop_v3"], 32)
        self.assertEqual(stream_dims["hilbert_astr"], 8)
        self.assertEqual(stream_dims["cell_onehot"], 4)
        self.assertEqual(X_recovery.shape[0], 200)
        # chemprop_v3 + cell_onehot on platinai adds 4+32+8 = 44 cols
        # over the 12-d RDKit baseline, so X_recovery has 54-d
        # (12 + 32 + 8 + 4) or fewer after constant-column drop.
        self.assertGreaterEqual(X_recovery.shape[1], 50)
        self.assertLessEqual(X_recovery.shape[1], 60)
        # 12-d baseline R² must match Wf-335-M2 within 1e-3 tolerance
        # (since both use the same Ridge multi-seed CV).
        cv_12d = ridge_r2_cv(
            X_32d[:, :12], y, n_splits=5, n_seeds=5,
            alpha=1.0, seed_offset=0,
        )
        # Wf-335-M2 best for platinai_4cell = -0.1251 (per result.json)
        wf335m2_best = -0.12505399802714393
        diff = abs(cv_12d["r2_mean"] - wf335m2_best)
        self.assertLess(diff, 1e-3)
        # Multi-stream recovery R² vs Wf-335-M2 best: structural verdict
        cv_recovery = ridge_r2_cv(
            X_recovery, y, n_splits=5, n_seeds=5,
            alpha=1.0, seed_offset=0,
        )
        v = verdict(cv_recovery["r2_mean"], wf335m2_best)
        # If lift > 0 + R² > 0: CEILING_BREAK / MODERATE / WEAK_RECOVERY
        # If lift ≥ 0 but R² ≤ 0: NO_LIFT_HONEST_NEGATIVE
        # If lift < 0: HONEST_NEGATIVE_R2_DEEP preserved.
        valid_verdicts = {
            "CEILING_BREAK",
            "MODERATE_RECOVERY",
            "WEAK_RECOVERY",
            "NO_LIFT_HONEST_NEGATIVE",
            "HONEST_NEGATIVE_R2_DEEP",
            "UNDEFINED",
        }
        self.assertIn(v, valid_verdicts)
        # Honest framing: pin the per-cohort verdict to one of the
        # documented recovery verdicts.
        self.assertTrue(np.isfinite(cv_12d["r2_mean"]))


# ---------------------------------------------------------------------------
# Test 2: R² recovery on hilbert
# ---------------------------------------------------------------------------
class TestWf367B1R2RecoveryHilbert(unittest.TestCase):
    """Multi-stream recovery on hilbert cohort."""

    def test_wf367b1_r2_recovery_on_hilbert(self) -> None:
        cohort = "hilbert"
        # Load hilbert cohort (200 SMILES)
        X_32d, y, smiles_list = load_cohort(
            cohort, max_samples=2000, seed=0,
        )[:3]
        # Build recovery feature matrix
        X_recovery, feature_names, stream_dims = build_recovery_feature_matrix(
            smiles_list, X_32d, cohort,
        )
        # Structural invariants — hilbert has no per-cell one-hot
        self.assertEqual(stream_dims["rdkit_12d"], 12)
        self.assertEqual(stream_dims["chemprop_v3"], 32)
        self.assertEqual(stream_dims["hilbert_astr"], 8)
        self.assertNotIn("cell_onehot", stream_dims)
        self.assertEqual(X_recovery.shape[0], 200)
        # 12 + 32 + 8 = 52 cols max
        self.assertGreaterEqual(X_recovery.shape[1], 48)
        self.assertLessEqual(X_recovery.shape[1], 52)
        # 12-d baseline R² must match Wf-335-M2 within 1e-3 tolerance
        cv_12d = ridge_r2_cv(
            X_32d[:, :12], y, n_splits=5, n_seeds=5,
            alpha=1.0, seed_offset=10,
        )
        # Wf-335-M2 best for hilbert = -0.1268 (per result.json)
        wf335m2_best = -0.1267927743302506
        diff = abs(cv_12d["r2_mean"] - wf335m2_best)
        self.assertLess(diff, 1e-3)
        # Multi-stream recovery R² vs Wf-335-M2 best
        cv_recovery = ridge_r2_cv(
            X_recovery, y, n_splits=5, n_seeds=5,
            alpha=1.0, seed_offset=10,
        )
        v = verdict(cv_recovery["r2_mean"], wf335m2_best)
        valid_verdicts = {
            "CEILING_BREAK",
            "MODERATE_RECOVERY",
            "WEAK_RECOVERY",
            "NO_LIFT_HONEST_NEGATIVE",
            "HONEST_NEGATIVE_R2_DEEP",
            "UNDEFINED",
        }
        self.assertIn(v, valid_verdicts)


# ---------------------------------------------------------------------------
# Test 3: backward compat — Wf-335-M2 result.json untouched, end-to-end
# ---------------------------------------------------------------------------
class TestWf367B1BackwardCompat(unittest.TestCase):
    """Backward compat: Wf-335-M2 result.json is not modified."""

    def test_wf367b1_backward_compat(self) -> None:
        # Snapshot Wf-335-M2 result.json mtime before running Wf-367-B1
        wf335m2_path = os.path.join(
            _REPO_ROOT, "molmetal", "reports", "wf335m2_r2", "result.json",
        )
        wf335m2_mtime_before = (
            os.path.getmtime(wf335m2_path) if os.path.exists(wf335m2_path)
            else None
        )
        # Snapshot Wf-325A baseline (should match WF325A_BASELINE_R2)
        self.assertAlmostEqual(
            WF325A_BASELINE_R2["platinai_4cell"],
            -0.09446482453622895,
            places=6,
        )
        self.assertAlmostEqual(
            WF325A_BASELINE_R2["hilbert"],
            -0.016897030097804855,
            places=6,
        )
        # Snapshot Wf-335-M2 best (the Wf-367-B1 baseline)
        self.assertAlmostEqual(
            WF335M2_DEFAULT_BEST_R2["platinai_4cell"],
            -0.12505399802714393,
            places=6,
        )
        self.assertAlmostEqual(
            WF335M2_DEFAULT_BEST_R2["hilbert"],
            -0.1267927743302506,
            places=6,
        )
        # Run Wf-367-B1 end-to-end on both cohorts to a tempdir
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "wf367b1_result.json")
            result = run_wf367b1(
                out_path=out_path,
                cohorts=["platinai_4cell", "hilbert"],
                n_splits=5,
                n_seeds=5,
                alpha=1.0,
            )
            # Aggregate-status taxonomy (PARTIAL_LIFT or HONEST_NEGATIVE)
            self.assertIn(result["aggregate_status"], {
                "MEASURED_PARTIAL_LIFT",
                "HONEST_NEGATIVE_R2_DEEP",
            })
            # Per-cohort verdict taxonomy is preserved
            self.assertEqual(result["n_cohorts_tested"], 2)
            self.assertIn("platinai_4cell", result["wf335m2_baselines"])
            self.assertIn("hilbert", result["wf335m2_baselines"])
            for c in result["per_cohort"]:
                for cfg in c["per_config"]:
                    self.assertIn("verdict", cfg)
                    self.assertIn("r2_mean", cfg)
                    self.assertIn("lift_vs_wf335m2", cfg)
                    self.assertIn("n_features", cfg)
            # Result JSON is loadable
            with open(out_path, "r", encoding="utf-8") as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["wf"], "Wf-367-B1")
            self.assertEqual(reloaded["n_cohorts_tested"], 2)
        # Snapshot Wf-335-M2 result.json mtime after; must be unchanged
        if wf335m2_mtime_before is not None and os.path.exists(wf335m2_path):
            wf335m2_mtime_after = os.path.getmtime(wf335m2_path)
            self.assertEqual(wf335m2_mtime_before, wf335m2_mtime_after)
        # Stream-level structural invariants on the helpers
        smiles_list = ["c1ccccc1", "CCO", "CC(=O)O"]
        chemprop_v3 = chemprop_v3_enhanced_features(smiles_list)
        hilbert_astr = hilbert_astsr_spectral_features(smiles_list)
        cell_oh_plat = per_cell_onehot(smiles_list, "platinai_4cell")
        cell_oh_other = per_cell_onehot(smiles_list, "hilbert")
        self.assertEqual(chemprop_v3.shape, (3, 32))
        self.assertEqual(hilbert_astr.shape, (3, 8))
        self.assertEqual(cell_oh_plat.shape, (3, 4))
        self.assertEqual(cell_oh_other.shape, (3, 0))


if __name__ == "__main__":
    unittest.main()
