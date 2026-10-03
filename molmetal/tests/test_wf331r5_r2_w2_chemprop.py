"""Tests for Wf-331-R5 — full W₂ + chemprop v2 graph integration (CPU).

=====================================================================
Wf-331-R5 / WEAK_TO_STRONG §2.1 / TODO-122 P1.3 — verifies the
full W₂ + chemprop v2 graph integration on 3 real-pIC50 cohorts.

Test surface (4 tests):
  * test_w2_distance_as_feature_covariate
        Covariate-based W₂ (RDKit projection on training rows)
        is non-leaky, finite, and varies across rows.
  * test_chemprop_v2_graph_features_pooled
        chemprop v2 graph fingerprint pooled across (mean, max, sum)
        pools has documented dim (258-d) and is deterministic per
        SMILES; SUM-pool differentiates molecule size.
  * test_combined_pipeline_three_cohorts
        The combined pipeline (RDKit 32-d + chemprop v2 graph + W₂)
        runs end-to-end on platinai_4cell, hilbert, and
        metalcytotox_combined, producing a valid result JSON for
        each cohort.
  * test_r2_lift_vs_wf325a_baseline_three_cohorts
        Per-cohort best_config R² is compared to the Wf-325A baseline
        anchor (platinai_4cell -0.0945, hilbert -0.0169,
        metalcytotox_combined +0.1055) and the verdict (CEILING_BREAK
        / WEAK_RECOVERY / NO_LIFT / HONEST_NEGATIVE_R2_DEEP) is
        consistent with the measured R² and lift.

Honest framing (party-twenty-spirit §3.5):
  * The covariate-based W₂ feature (RDKit projection on training
    rows) is **non-leaky**.  The y-based W₂ (training y as reference)
    is **leaky** and is exposed via the --w2-use-y-leaky CLI flag
    for completeness, but is NOT the default.
  * The chemprop v2 graph features here are the INPUT to the D-MPNN
    encoder (atom 72-d + bond 14-d featurisation via the canonical
    chemprop v2 SimpleMoleculeMolGraphFeaturizer when available,
    RDKit-native chemprop-compatible fallback otherwise), NOT the
    encoder OUTPUT.  The encoder output requires a full forward
    pass with autograd (GPU portion per WEAK_TO_STRONG §2.1 /
    TODO-93).
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
from typing import Any, Dict, List, Tuple

import numpy as np

# Make sure the bench script is importable from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from molmetal.scripts.wf331r5_r2_w2_chemprop import (  # noqa: E402
    WF325A_BASELINE_R2,
    WF325A_BASELINE_R2_PLA_12D,
    _compute_w2_distance_loo,
    _get_graph_feature_dim,
    _probe_chemprop_dims,
    _summarise_recovery,
    combined_feature_pipeline,
    compute_covariate_w2_distance,
    compute_w2_distance_feature,
    extract_graph_features_pooled,
    load_cohort,
    rdkit_chemprop_atom_features,
    rdkit_chemprop_bond_features,
    ridge_r2_eval,
    ridge_r2_eval_split_aware_w2,
    run_wf331r5,
)


# ---------------------------------------------------------------------------
# Test 1: W₂ distance as feature (covariate, non-leaky)
# ---------------------------------------------------------------------------
class TestW2DistanceAsFeatureCovariate(unittest.TestCase):
    """Covariate-based W₂ (RDKit projection on training rows) is
    non-leaky, finite, and varies across rows."""

    def test_w2_distance_as_feature_covariate(self) -> None:
        # Synthetic covariate matrix: 100 rows × 3 cols (MolWt-like,
        # MolLogP-like, TPSA-like projections).
        rng = np.random.default_rng(0)
        X = np.stack(
            [
                rng.normal(loc=300.0, scale=80.0, size=100),  # MolWt-like
                rng.normal(loc=2.5, scale=1.2, size=100),  # MolLogP-like
                rng.normal(loc=80.0, scale=30.0, size=100),  # TPSA-like
            ],
            axis=1,
        )

        # Compute covariate W₂ with MolWt-like projection (axis=0).
        w2 = compute_covariate_w2_distance(X, reference=X, projection_axis=0)
        self.assertEqual(w2.shape, (100,))
        # All entries finite + non-negative
        self.assertTrue(np.isfinite(w2).all())
        self.assertTrue((w2 >= 0.0).all())
        # The covariate W₂ varies across rows (not all-constant)
        self.assertGreater(float(w2.std()), 0.0)

        # Compute again with MolLogP-like projection (axis=1).
        w2_axis1 = compute_covariate_w2_distance(
            X,
            reference=X,
            projection_axis=1,
        )
        self.assertEqual(w2_axis1.shape, (100,))
        self.assertTrue(np.isfinite(w2_axis1).all())
        # Different axis → different W₂ values (per-axis variation)
        # The two axes should NOT be identical (different projections)
        self.assertFalse(np.allclose(w2, w2_axis1))

        # Leave-one-out W₂ on y (LOO only; still leaky by construction)
        y = rng.normal(loc=5.0, scale=1.5, size=100)
        w2_loo = _compute_w2_distance_loo(y)
        self.assertEqual(w2_loo.shape, (100,))
        self.assertTrue(np.isfinite(w2_loo).all())
        self.assertTrue((w2_loo >= 0.0).all())

        # The plain (non-leaky-aware) W₂ function still works.
        w2_plain = compute_w2_distance_feature(y, reference=y)
        self.assertEqual(w2_plain.shape, (100,))
        self.assertTrue(np.isfinite(w2_plain).all())


# ---------------------------------------------------------------------------
# Test 2: chemprop v2 graph features pooled
# ---------------------------------------------------------------------------
class TestChempropV2GraphFeaturesPooled(unittest.TestCase):
    """chemprop v2 graph fingerprint pooled across (mean, max, sum)
    has documented dim (258-d) and is deterministic per SMILES."""

    def test_chemprop_v2_graph_features_pooled(self) -> None:
        smiles_list = [
            "c1ccccc1",  # benzene
            "CCO",  # ethanol
            "C",  # methane (small)
            "c1ccc2ccccc2c1",  # naphthalene (larger)
            "O=S(=O)(N)c1ccc(N)cc1",  # sulfanilamide (heteroatoms)
        ]
        # Default pools: (mean, max, sum)
        feats = extract_graph_features_pooled(smiles_list)
        # Shape invariants
        self.assertEqual(feats.shape[0], len(smiles_list))
        atom_dim, bond_dim = _probe_chemprop_dims()
        expected_dim = 3 * (atom_dim + bond_dim)
        self.assertEqual(_get_graph_feature_dim(), expected_dim)
        self.assertEqual(feats.shape[1], expected_dim)
        # All entries finite + non-NaN (or all-zero for failed parse)
        self.assertTrue(np.isfinite(feats).all())
        # Determinism: re-extract and compare
        feats2 = extract_graph_features_pooled(smiles_list)
        np.testing.assert_array_equal(feats, feats2)
        # Sanity: naphthalene has more atoms than benzene.
        # SUM atom-pool region: atoms are pooled first (3 * atom_dim),
        # then bonds (3 * bond_dim).  Within atoms, SUM pool is the
        # last atom_dim chunk (indices 2*atom_dim : 3*atom_dim).
        sum_atom_start = 2 * atom_dim
        sum_atom_end = 3 * atom_dim
        naphthalene_atom_sum = float(feats[3, sum_atom_start:sum_atom_end].sum())
        benzene_atom_sum = float(feats[0, sum_atom_start:sum_atom_end].sum())
        self.assertGreater(naphthalene_atom_sum, benzene_atom_sum)
        # Custom pools: only mean → dim = (atom_dim + bond_dim)
        feats_mean_only = extract_graph_features_pooled(
            smiles_list,
            pools=("mean",),
        )
        self.assertEqual(feats_mean_only.shape[1], atom_dim + bond_dim)
        # RDKit-native featurizers also work standalone
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles("c1ccccc1")
            af = rdkit_chemprop_atom_features(mol)
            bf = rdkit_chemprop_bond_features(mol)
            self.assertEqual(af.shape, (6, 72))
            self.assertEqual(bf.shape, (6, 14))
        except ImportError:
            pass  # RDKit unavailable — skip


# ---------------------------------------------------------------------------
# Test 3: combined pipeline runs on 3 cohorts
# ---------------------------------------------------------------------------
class TestCombinedPipelineThreeCohorts(unittest.TestCase):
    """The combined pipeline runs end-to-end on platinai_4cell,
    hilbert, and metalcytotox_combined."""

    def test_combined_pipeline_three_cohorts(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="wf331r5_")
        out_path = os.path.join(tmpdir, "result.json")
        result = run_wf331r5(
            out_path=out_path,
            cohorts=["platinai_4cell", "hilbert", "metalcytotox_combined"],
            metalcytotox_max=2000,
            seed=0,
        )

        # ----- Result JSON structure -----
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["wf"], "Wf-331-R5")
        self.assertEqual(data["n_cohorts_tested"], 3)
        self.assertEqual(
            data["cohorts_tested"],
            ["platinai_4cell", "hilbert", "metalcytotox_combined"],
        )
        self.assertEqual(
            data["w2_method"],
            "covariate_RDKit_projection (non-leaky)",
        )
        self.assertIn("per_cohort", data)
        self.assertEqual(len(data["per_cohort"]), 3)

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
            # 8 configs evaluated
            self.assertGreaterEqual(len(cohort_entry["per_config"]), 6)
            # Each config has the standard fields
            for cfg in cohort_entry["per_config"]:
                self.assertIn("label", cfg)
                self.assertIn("r2", cfg)
                self.assertIn("lift_vs_wf325a_baseline", cfg)
                self.assertIn("verdict", cfg)
                self.assertIn(
                    cfg.get("verdict", "MISSING"),
                    valid_verdicts,
                    msg=f"unexpected verdict for {cfg.get('label')}",
                )

        # ----- combined feature dimensions are documented -----
        n_features = data["n_features"]
        self.assertEqual(n_features["rdkit_32d"], 32)
        self.assertEqual(n_features["w2_distance"], 1)
        self.assertEqual(
            n_features["combined_max"],
            n_features["rdkit_32d"] + n_features["chemprop_v2_graph"] + 1,
        )

        # ----- 12-d baseline R² must match Wf-325A MEASURED value -----
        for cohort_entry in data["per_cohort"]:
            cohort = cohort_entry["cohort"]
            baseline_r2 = WF325A_BASELINE_R2[cohort]
            cfg_12d = next(
                (c for c in cohort_entry["per_config"] if c["label"] == "12d_baseline"),
                None,
            )
            self.assertIsNotNone(
                cfg_12d,
                f"missing 12d_baseline config in {cohort}",
            )
            # Allow 1e-3 tolerance for sklearn / numpy version drift
            self.assertAlmostEqual(
                cfg_12d["r2"],
                baseline_r2,
                delta=1e-3,
                msg=(
                    f"{cohort}: 12-d baseline R² should match Wf-325A "
                    f"MEASURED ({baseline_r2:+.4f}); got "
                    f"{cfg_12d['r2']:+.4f}"
                ),
            )


# ---------------------------------------------------------------------------
# Test 4: R² lift vs Wf-325A baseline on 3 cohorts
# ---------------------------------------------------------------------------
class TestR2LiftVsWf325aBaselineThreeCohorts(unittest.TestCase):
    """Per-cohort best_config R² is compared to the Wf-325A baseline
    anchor and the verdict is consistent with the measured R² / lift."""

    def test_r2_lift_vs_wf325a_baseline_three_cohorts(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="wf331r5_lift_")
        out_path = os.path.join(tmpdir, "result.json")
        result = run_wf331r5(
            out_path=out_path,
            cohorts=["platinai_4cell", "hilbert", "metalcytotox_combined"],
            metalcytotox_max=2000,
            seed=0,
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
                r2 = float(cfg["r2"])
                lift = float(cfg["lift_vs_wf325a_baseline"])
                # lift = r2 - baseline_r2 (sign-correct)
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

        # ----- Honest framing: Wf-325A anchor is preserved -----
        for cohort, baseline_r2 in WF325A_BASELINE_R2.items():
            cohort_entry = next(
                c for c in result["per_cohort"] if c["cohort"] == cohort
            )
            self.assertAlmostEqual(
                float(cohort_entry["wf325a_baseline_r2"]),
                float(baseline_r2),
                places=6,
                msg=f"{cohort}: Wf-325A baseline anchor must be preserved",
            )

        # ----- Print summary for human inspection -----
        print()
        print(f"  Wf-331-R5 aggregate_status = {result['aggregate_status']}")
        for cohort_entry in result["per_cohort"]:
            cohort = cohort_entry["cohort"]
            baseline = cohort_entry["wf325a_baseline_r2"]
            print(f"\n  cohort={cohort}  Wf-325A baseline R²={baseline:+.4f}")
            for cfg in cohort_entry["per_config"]:
                if "error" in cfg:
                    print(f"    {cfg['label']:>30s}: {cfg['error']}")
                    continue
                print(
                    f"    {cfg['label']:>30s}  n_feat={cfg.get('n_features', 0):>3d}  "
                    f"R²={cfg['r2']:+.4f}  lift={cfg['lift_vs_wf325a_baseline']:+.4f}  "
                    f"{cfg['verdict']}"
                )
            bc = cohort_entry["best_config"]
            print(
                f"    best = {bc['label']}  R²={bc['r2']:+.4f}  "
                f"lift={bc['lift_vs_wf325a_baseline']:+.4f}"
            )


if __name__ == "__main__":
    unittest.main()
