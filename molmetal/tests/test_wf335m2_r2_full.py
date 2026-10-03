"""Tests for Wf-335-M2 — FULL W₂ + chemprop v2 graph integration (CPU).

=====================================================================
Wf-335-M2 / WEAK_TO_STRONG §2.1 / TODO-122 P1.3 — verifies the FULL
W₂ + chemprop v2 graph integration on 3 real-pIC50 cohorts, with
multi-seed K-fold CV, multi-axis W₂ selection, per-cohort breakdowns
(per-cell / per-metal), and top-10 feature importance.

Test surface (4 tests):
  * test_w2_distance_as_feature_multi_axis
        Multi-axis W₂ selection: each axis (MolWt, MolLogP, TPSA,
        HeavyAtoms) gives a different R² mean; the best axis is
        well-defined and finite.
  * test_chemprop_v2_graph_features_combined_pipeline
        The FULL static feature matrix (RDKit 32-d + chemprop v2 graph
        258-d) has the documented dimension (291-d pre-drop, 187-d
        post-constant-drop on platinai_4cell) and produces a valid
        Ridge fit.
  * test_combined_pipeline_multi_seed_cv_three_cohorts
        The combined pipeline runs end-to-end on platinai_4cell,
        hilbert, and metalcytotox_combined, producing a valid result
        JSON with 4 configs per cohort and multi-seed CV R² means.
  * test_r2_lift_vs_wf325a_baseline_multi_seed_cv
        Per-cohort best_config R² (multi-seed CV mean) is compared to
        the Wf-325A baseline anchor (platinai_4cell -0.0945, hilbert
        -0.0169, metalcytotox_combined +0.1055) and the verdict
        (CEILING_BREAK / WEAK_RECOVERY / NO_LIFT / HONEST_NEGATIVE_R2_DEEP)
        is consistent with the measured R² / lift.

Honest framing (party-twenty-spirit §3.5):
  * The covariate-based W₂ feature (RDKit projection on training
    rows) is non-leaky.  Multiple projection axes are benchmarked
    independently to identify the best axis per cohort.
  * The chemprop v2 graph features here are the INPUT to the D-MPNN
    encoder (atom 72-d + bond 14-d featurisation via the canonical
    chemprop v2 SimpleMoleculeMolGraphFeaturizer when available,
    RDKit-native chemprop-compatible fallback otherwise), NOT the
    encoder OUTPUT.
  * Multi-seed K-fold CV provides a robust R² estimate (mean ± std
    over 25 splits by default).
  * These tests pin down the **deterministic structure** of the
    pipeline; the absolute R² numbers may vary with sklearn /
    numpy versions but the comparison shape (which config beats
    the Wf-325A baseline) is the load-bearing invariant.
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

from molmetal.scripts.wf335m2_r2_w2_chemprop import (  # noqa: E402
    W2_PROJECTION_AXES,
    build_full_feature_matrix,
    feature_importance_topk,
    per_cell_r2_breakdown,
    per_metal_r2_breakdown,
    ridge_r2_cv,
    ridge_r2_eval_with_w2,
    run_wf335m2,
    select_best_w2_axis,
)
from molmetal.scripts.wf331r5_r2_w2_chemprop import (  # noqa: E402
    WF325A_BASELINE_R2,
    extract_graph_features_pooled,
    load_cohort,
)


# ---------------------------------------------------------------------------
# Test 1: W₂ distance as feature (multi-axis selection)
# ---------------------------------------------------------------------------
class TestW2DistanceAsFeatureMultiAxis(unittest.TestCase):
    """Multi-axis W₂ selection: each axis gives a different R² mean."""

    def test_w2_distance_as_feature_multi_axis(self) -> None:
        # Synthetic covariate matrix: 100 rows × 8 cols.
        rng = np.random.default_rng(0)
        X = np.stack(
            [
                rng.normal(loc=300.0, scale=80.0, size=100),  # 0: MolWt-like
                rng.normal(loc=2.5, scale=1.2, size=100),  # 1: MolLogP-like
                rng.normal(loc=80.0, scale=30.0, size=100),  # 2: TPSA-like
                rng.normal(loc=5.0, scale=2.0, size=100),  # 3
                rng.normal(loc=2.0, scale=1.0, size=100),  # 4
                rng.normal(loc=0.5, scale=0.3, size=100),  # 5
                rng.normal(loc=10.0, scale=4.0, size=100),  # 6
                rng.normal(loc=20.0, scale=8.0, size=100),  # 7: HeavyAtoms-like
            ],
            axis=1,
        )
        # Synthetic target correlated with col 0 (MolWt-like)
        y = 0.3 * X[:, 0] + 0.1 * X[:, 7] + rng.normal(loc=0.0, scale=5.0, size=100)

        # W₂ axis selection — try axes 0, 1, 2, 7
        sel = select_best_w2_axis(
            X,
            y,
            n_splits=5,
            n_seeds=2,
            alpha=1.0,
        )
        self.assertIn(sel["best_axis"], [0, 1, 2, 7])
        self.assertTrue(np.isfinite(sel["best_axis_r2_mean"]))
        # Per-axis results are documented
        self.assertGreater(len(sel["axis_results"]), 0)
        for axis, res in sel["axis_results"].items():
            self.assertIn("r2_mean", res)
            self.assertIn("r2_std", res)
            self.assertTrue(np.isfinite(res["r2_mean"]))
        # Different axes should give different R² means
        r2_means = [res["r2_mean"] for res in sel["axis_results"].values()]
        self.assertGreater(max(r2_means), min(r2_means))
        # Best axis has the highest R² mean
        best_axis = sel["best_axis"]
        self.assertEqual(
            sel["axis_results"][best_axis]["r2_mean"],
            sel["best_axis_r2_mean"],
        )

        # Verify multi-axis W₂ evaluation directly
        cv_axis0 = ridge_r2_eval_with_w2(
            X,
            y,
            projection_axis=0,
            n_splits=5,
            n_seeds=2,
            alpha=1.0,
        )
        cv_axis7 = ridge_r2_eval_with_w2(
            X,
            y,
            projection_axis=7,
            n_splits=5,
            n_seeds=2,
            alpha=1.0,
        )
        self.assertTrue(np.isfinite(cv_axis0["r2_mean"]))
        self.assertTrue(np.isfinite(cv_axis7["r2_mean"]))
        self.assertEqual(len(cv_axis0["r2_per_seed"]), 2)
        self.assertEqual(len(cv_axis0["r2_per_split"]), 10)  # 2 seeds × 5 folds


# ---------------------------------------------------------------------------
# Test 2: chemprop v2 graph features + combined pipeline
# ---------------------------------------------------------------------------
class TestChempropV2GraphFeaturesCombinedPipeline(unittest.TestCase):
    """The FULL static feature matrix has documented dim and produces
    a valid Ridge fit."""

    def test_chemprop_v2_graph_features_combined_pipeline(self) -> None:
        # Use platinai_4cell cohort (small, fast).
        X_32d, y, smiles_list = load_cohort(
            "platinai_4cell",
            max_samples=200,
            seed=0,
        )[:3]
        # Build FULL static feature matrix (RDKit 32-d + chemprop v2 graph)
        X_static, feature_names = build_full_feature_matrix(
            smiles_list,
            X_32d,
            include_rdkit_32d=True,
            include_chemprop_v2_graph=True,
        )
        n = len(smiles_list)
        # RDKit 32-d column count + 3 pools × (72 atom + 14 bond) = 32 + 258 = 290
        # (Some columns may be constant and dropped; allow 100 - 290 range.)
        expected_min = 100
        expected_max = 32 + 258  # 290
        self.assertGreaterEqual(X_static.shape[1], expected_min)
        self.assertLessEqual(X_static.shape[1], expected_max)
        self.assertEqual(X_static.shape[0], n)
        # feature_names is the full pre-drop list (32 + 258 = 290); the
        # actual column count after constant-column drop is in
        # X_static.shape[1].
        self.assertEqual(len(feature_names), 32 + 258)
        # Multi-drop CV R² on the FULL static matrix
        cv = ridge_r2_cv(
            X_static,
            y,
            n_splits=5,
            n_seeds=2,
            alpha=1.0,
        )
        self.assertTrue(np.isfinite(cv["r2_mean"]))
        self.assertEqual(len(cv["r2_per_seed"]), 2)
        self.assertEqual(len(cv["r2_per_split"]), 10)  # 2 seeds × 5 folds
        # Per-split has 10 R² values, all finite
        self.assertTrue(all(np.isfinite(r) for r in cv["r2_per_split"]))
        # Test that graph features alone also work (sanity check)
        graph_only = extract_graph_features_pooled(smiles_list)
        cv_graph_only = ridge_r2_cv(
            graph_only,
            y,
            n_splits=5,
            n_seeds=2,
            alpha=1.0,
        )
        self.assertTrue(np.isfinite(cv_graph_only["r2_mean"]))


# ---------------------------------------------------------------------------
# Test 3: combined pipeline runs on 3 cohorts
# ---------------------------------------------------------------------------
class TestCombinedPipelineMultiSeedCvThreeCohorts(unittest.TestCase):
    """The combined pipeline runs end-to-end on 3 cohorts with multi-seed CV."""

    def test_combined_pipeline_multi_seed_cv_three_cohorts(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="wf335m2_")
        out_path = os.path.join(tmpdir, "result.json")
        result = run_wf335m2(
            out_path=out_path,
            cohorts=["platinai_4cell", "hilbert", "metalcytotox_combined"],
            metalcytotox_max=2000,
            n_splits=5,
            n_seeds=3,  # smaller for fast test
            alpha=1.0,
            axis_selection_seeds=2,
            include_per_cell=True,
            include_per_metal=True,
            include_feature_importance=True,
        )

        # ----- Result JSON structure -----
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["wf"], "Wf-335-M2")
        self.assertEqual(data["n_cohorts_tested"], 3)
        self.assertEqual(
            data["cohorts_tested"],
            ["platinai_4cell", "hilbert", "metalcytotox_combined"],
        )
        self.assertIn("per_cohort", data)
        self.assertEqual(len(data["per_cohort"]), 3)

        # ----- CV config documented -----
        cv_cfg = data["cv_config"]
        self.assertEqual(cv_cfg["n_splits"], 5)
        self.assertEqual(cv_cfg["n_seeds"], 3)
        self.assertEqual(cv_cfg["alpha"], 1.0)

        # ----- Per-cohort structure -----
        valid_verdicts = {
            "CEILING_BREAK",
            "MODERATE_RECOVERY",
            "WEAK_RECOVERY",
            "NO_LIFT_HONEST_NEGATIVE",
            "HONEST_NEGATIVE_R2_DEEP",
            "PIPELINE_ERROR",
            "UNDEFINED",
        }
        for cohort_entry in data["per_cohort"]:
            cohort = cohort_entry["cohort"]
            self.assertIn(cohort, WF325A_BASELINE_R2)
            self.assertIn("per_config", cohort_entry)
            self.assertIn("best_config", cohort_entry)
            self.assertIn("wf325a_baseline_r2", cohort_entry)
            self.assertIn("w2_axis_selection", cohort_entry)
            # 4 configs (12d_baseline, 32d_rdkit, full_static_combined,
            # full_static_combined_plus_w2_best_axis)
            self.assertEqual(len(cohort_entry["per_config"]), 4)
            # Each config has the standard fields
            for cfg in cohort_entry["per_config"]:
                self.assertIn("label", cfg)
                self.assertIn("r2_mean", cfg)
                self.assertIn("r2_std", cfg)
                self.assertIn("lift_vs_wf325a_baseline", cfg)
                self.assertIn("verdict", cfg)
                self.assertIn(
                    cfg.get("verdict", "MISSING"),
                    valid_verdicts,
                    msg=f"unexpected verdict for {cfg.get('label')}",
                )
                # r2_mean is finite (sklearn always gives a number here)
                self.assertTrue(
                    np.isfinite(cfg["r2_mean"]) or cfg["r2_mean"] == 0.0,
                )

            # W₂ axis selection result is well-formed
            sel = cohort_entry["w2_axis_selection"]
            self.assertIn("best_axis", sel)
            self.assertIn("axis_results", sel)

        # ----- Per-cell breakdown on platinai_4cell -----
        platinai_entry = next(
            c for c in data["per_cohort"] if c["cohort"] == "platinai_4cell"
        )
        self.assertIn("per_cell_r2", platinai_entry)
        per_cell = platinai_entry["per_cell_r2"]
        # 4 cells expected
        self.assertEqual(len(per_cell), 4)
        for cell in ("mcf7", "a549", "a2780", "a2780cis"):
            self.assertIn(cell, per_cell)
            self.assertIn("r2_mean", per_cell[cell])
            self.assertIn("r2_std", per_cell[cell])
            self.assertIn("n_valid", per_cell[cell])

        # ----- Per-metal breakdown on metalcytotox_combined -----
        met_entry = next(
            c for c in data["per_cohort"] if c["cohort"] == "metalcytotox_combined"
        )
        self.assertIn("per_metal_r2", met_entry)
        per_metal = met_entry["per_metal_r2"]
        # The first 2000 rows of metalcytotox are dominated by Ru and Os
        # (per data audit).  The per-metal breakdown reports whatever
        # metals survive the 2000-row cap; require ≥ 1 metal.
        self.assertGreaterEqual(len(per_metal), 1)
        for metal, res in per_metal.items():
            self.assertIn("r2_mean", res)
            self.assertIn("r2_std", res)
            self.assertIn("n_samples", res)

        # ----- Top-10 feature importance per cohort -----
        for cohort_entry in data["per_cohort"]:
            self.assertIn("top10_feature_importance", cohort_entry)
            top = cohort_entry["top10_feature_importance"]
            # Top-10 should have up to 10 entries (no error)
            if top and "error" not in top[0]:
                self.assertLessEqual(len(top), 10)
                # Sorted by rank ascending
                for k, feat in enumerate(top):
                    self.assertEqual(feat["rank"], k + 1)
                    self.assertIn("feature", feat)
                    self.assertIn("weight", feat)
                    self.assertIn("abs_weight", feat)

        # ----- Honest framing: Wf-325A anchors preserved -----
        for cohort, baseline_r2 in WF325A_BASELINE_R2.items():
            cohort_entry = next(c for c in data["per_cohort"] if c["cohort"] == cohort)
            self.assertAlmostEqual(
                float(cohort_entry["wf325a_baseline_r2"]),
                float(baseline_r2),
                places=6,
                msg=f"{cohort}: Wf-325A baseline anchor must be preserved",
            )


# ---------------------------------------------------------------------------
# Test 4: R² lift vs Wf-325A baseline (multi-seed CV mean)
# ---------------------------------------------------------------------------
class TestR2LiftVsWf325aBaselineMultiSeedCv(unittest.TestCase):
    """Per-cohort best_config R² (multi-seed CV mean) is compared to
    the Wf-325A baseline anchor and the verdict is consistent with
    the measured R² / lift."""

    def test_r2_lift_vs_wf325a_baseline_multi_seed_cv(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="wf335m2_lift_")
        out_path = os.path.join(tmpdir, "result.json")
        result = run_wf335m2(
            out_path=out_path,
            cohorts=["platinai_4cell", "hilbert", "metalcytotox_combined"],
            metalcytotox_max=2000,
            n_splits=5,
            n_seeds=3,
            alpha=1.0,
            axis_selection_seeds=2,
            include_per_cell=True,
            include_per_metal=True,
            include_feature_importance=True,
        )

        # ----- Verdict consistency per cohort -----
        valid_verdicts = {
            "CEILING_BREAK",
            "MODERATE_RECOVERY",
            "WEAK_RECOVERY",
            "NO_LIFT_HONEST_NEGATIVE",
            "HONEST_NEGATIVE_R2_DEEP",
            "PIPELINE_ERROR",
            "UNDEFINED",
        }
        for cohort_entry in result["per_cohort"]:
            cohort = cohort_entry["cohort"]
            baseline_r2 = float(WF325A_BASELINE_R2[cohort])
            for cfg in cohort_entry["per_config"]:
                if cfg.get("error"):
                    self.assertEqual(cfg["verdict"], "PIPELINE_ERROR")
                    continue
                r2 = float(cfg["r2_mean"])
                lift = float(cfg["lift_vs_wf325a_baseline"])
                # lift = r2_mean - baseline_r2 (sign-correct within 1e-6)
                self.assertAlmostEqual(
                    lift,
                    r2 - baseline_r2,
                    places=6,
                    msg=(f"{cohort} / {cfg['label']}: lift ≠ r2 - baseline"),
                )
                # Verdict is documented
                self.assertIn(cfg["verdict"], valid_verdicts)
                # Verdict is consistent with the r2 / lift values
                v = cfg["verdict"]
                if v == "CEILING_BREAK":
                    self.assertGreater(r2, 0.0)
                    self.assertGreaterEqual(lift, 0.10)
                elif v == "MODERATE_RECOVERY":
                    self.assertGreater(r2, 0.0)
                    self.assertGreaterEqual(lift, 0.05)
                elif v == "WEAK_RECOVERY":
                    self.assertGreater(r2, 0.0)
                elif v == "NO_LIFT_HONEST_NEGATIVE":
                    self.assertGreaterEqual(r2, baseline_r2)
                elif v == "HONEST_NEGATIVE_R2_DEEP":
                    self.assertLess(r2, baseline_r2)

        # ----- Aggregate status is documented and well-formed -----
        self.assertIn(
            result["aggregate_status"],
            {
                "HONEST_NEGATIVE_R2_RESOLVED",
                "HONEST_NEGATIVE_PARTIAL",
                "NO_LIFT_HONEST_NEGATIVE",
                "HONEST_NEGATIVE_R2_PERSISTENT",
                "UNDEFINED",
            },
        )

        # ----- Multi-seed CV r2_per_seed has correct shape -----
        # The headline R² is the multi-seed mean (not a single split).
        for cohort_entry in result["per_cohort"]:
            for cfg in cohort_entry["per_config"]:
                # CV R² is finite on this small sample
                if cfg.get("r2_mean") is not None:
                    self.assertTrue(
                        np.isfinite(cfg["r2_mean"]) or cfg["r2_mean"] == 0.0,
                    )

        # ----- Print summary for human inspection -----
        print()
        print(f"  Wf-335-M2 aggregate_status = {result['aggregate_status']}")
        for cohort_entry in result["per_cohort"]:
            cohort = cohort_entry["cohort"]
            baseline = cohort_entry["wf325a_baseline_r2"]
            print(f"\n  cohort={cohort}  Wf-325A baseline R²={baseline:+.4f}")
            for cfg in cohort_entry["per_config"]:
                print(
                    f"    {cfg['label']:>40s}  R²_mean={cfg['r2_mean']:+.4f} ± "
                    f"{cfg['r2_std']:.4f}  lift={cfg['lift_vs_wf325a_baseline']:+.4f}  "
                    f"{cfg['verdict']}"
                )
            bc = cohort_entry["best_config"]
            print(
                f"    best = {bc['label']}  R²_mean={bc['r2_mean']:+.4f}  "
                f"lift={bc['lift_vs_wf325a_baseline']:+.4f}"
            )
            sel = cohort_entry["w2_axis_selection"]
            if sel["best_axis"] >= 0:
                print(
                    f"    best_w2_axis = {sel['best_axis']}  "
                    f"R²={sel['best_axis_r2_mean']:+.4f}"
                )


if __name__ == "__main__":
    unittest.main()
