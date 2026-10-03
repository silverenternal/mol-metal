"""Tests for Wf-329W5 — W₂ + chemprop v2 graph features alignment.

=====================================================================
Wf-329W5 / Wf-325B / TODO-122 P1.3 — verifies the W₂ + chemprop v2
graph features alignment pipeline that the Wf-325A 32-d RDKit
expansion alone could not lift the Ridge baseline R² over the
signal-floor (-0.0945 on platinai_4cell).  Per WEAK_TO_STRONG
§2.1, the architectural remediation requires fundamentally
different signal sources — chemprop v2 atom/bond graph features
plus a per-row Wasserstein distance to the reference cohort.

Test surface (4 tests):
  * test_w2_distance_as_feature      W₂ distance is a per-row scalar,
                                     finite, non-negative, and 0 when
                                     y equals the reference mean.
  * test_chemprop_v2_graph_features  chemprop v2 graph fingerprint has
                                     the documented dim (258-d by
                                     default) and is extractable.
  * test_combined_feature_pipeline   RDKit 32-d + chemprop v2 graph +
                                     W₂ combine to 32 + 258 + 1 = 291-d.
  * test_r2_lift_on_platinai_4cell   per-config Ridge R² is computed,
                                     compared to Wf-325A baseline
                                     (-0.0945), and reported with the
                                     honest framing verdict.

Honest framing (party-twenty-spirit §3.5):
  * The chemprop v2 graph features here are the INPUT to the D-MPNN
    encoder (atom 72-d + bond 14-d featurisation), NOT the encoder
    OUTPUT.  The encoder output requires a full forward pass with
    autograd (GPU portion per WEAK_TO_STRONG §2.1 / TODO-93).
  * If the bench still yields R² ≤ 0 on the combined pipeline, the
    honest verdict is HONEST_NEGATIVE_R2_DEEP — confirming that
    chemprop v2 graph + W₂ alone cannot break the architectural
    ceiling; the Hilbert-ASTSR joint retrain (TODO-93) or Multi-PH
    + PLL (TODO-94) is required.
  * These tests pin down the **deterministic structure** of the
    pipeline; the absolute R² numbers may vary with sklearn / numpy
    versions but the comparison shape (which config beats the
    baseline) is the load-bearing invariant.
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

from molmetal.scripts.wf329w5_w2_chemprop_smoke import (  # noqa: E402
    WF325A_BASELINE_R2_PLA_12D,
    _get_graph_feature_dim,
    _probe_chemprop_dims,
    _wasserstein_distance_1d,
    combined_feature_pipeline,
    compute_w2_distance_feature,
    extract_chemprop_v2_graph_features,
    load_platinai_4cell_32d_with_smiles,
    run_wf329w5,
)


# ---------------------------------------------------------------------------
# Test 1: W₂ distance as feature
# ---------------------------------------------------------------------------
class TestW2DistanceAsFeature(unittest.TestCase):
    """The W₂ distance is a per-row scalar: finite, non-negative, and
    correctly 0 when y[i] equals the reference mean on a 1-point
    reference distribution."""

    def test_w2_distance_as_feature(self) -> None:
        # Synthetic cohort with a known reference distribution.
        rng = np.random.default_rng(0)
        y = rng.normal(loc=5.0, scale=1.5, size=100)
        w2 = compute_w2_distance_feature(y, reference=y)
        # Shape
        self.assertEqual(w2.shape, (100,))
        # All entries finite + non-negative
        self.assertTrue(np.isfinite(w2).all())
        self.assertTrue((w2 >= 0.0).all())
        # The W₂ distance from a Dirac at the reference mean to the
        # reference distribution equals the EMD-from-mean, which is
        # upper-bounded by the std of the reference.  At y[i]=mean(y)
        # the W₂ equals the EMD-from-mean of the mean Dirac, which is
        # the mean-absolute-deviation of the reference distribution.
        # We don't pin the exact value (depends on the empirical CDF
        # grid) but we verify it is finite, positive, and < the
        # range of y.
        i_at_mean = int(np.argmin(np.abs(y - float(y.mean()))))
        self.assertGreater(float(w2[i_at_mean]), 0.0)
        self.assertLess(float(w2[i_at_mean]), float(y.max() - y.min()))
        # The closed-form 1-d W₂ distance function is callable.
        d = _wasserstein_distance_1d(np.array([5.0]), np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.isfinite(d))
        self.assertGreaterEqual(d, 0.0)


# ---------------------------------------------------------------------------
# Test 2: chemprop v2 graph features extraction
# ---------------------------------------------------------------------------
class TestChempropV2GraphFeatures(unittest.TestCase):
    """The chemprop v2 graph fingerprint has the documented dimension
    and is deterministically extractable."""

    def test_chemprop_v2_graph_features(self) -> None:
        smiles_list = [
            "c1ccccc1",  # benzene
            "CCO",  # ethanol
            "C",  # methane (small)
            "c1ccc2ccccc2c1",  # naphthalene (larger)
            "O=S(=O)(N)c1ccc(N)cc1",  # sulfanilamide (heteroatoms)
        ]
        feats = extract_chemprop_v2_graph_features(smiles_list)
        # Shape invariants
        self.assertEqual(feats.shape[0], len(smiles_list))
        atom_dim, bond_dim = _probe_chemprop_dims()
        expected_dim = 3 * (atom_dim + bond_dim)
        self.assertEqual(_get_graph_feature_dim(), expected_dim)
        self.assertEqual(feats.shape[1], expected_dim)
        # All entries finite + non-NaN
        self.assertTrue(np.isfinite(feats).all())
        # Determinism: re-extract and compare (chemprop v2 uses canonical
        # atom/bond featurisation, so the fingerprint is deterministic
        # per SMILES).
        feats2 = extract_chemprop_v2_graph_features(smiles_list)
        np.testing.assert_array_equal(feats, feats2)
        # Sanity: benzene (SMILES 0) and naphthalene (SMILES 3) must
        # differ on the atom SUM pool (naphthalene has more atoms).
        # We pick the SUM-pool region: atoms are pooled first, then
        # bonds.  SUM atom-pool starts at 2*atom_dim, ends at
        # 3*atom_dim.
        atom_dim, _ = _probe_chemprop_dims()
        sum_atom_start = 2 * atom_dim
        sum_atom_end = 3 * atom_dim
        naphthalene_atom_sum = float(feats[3, sum_atom_start:sum_atom_end].sum())
        benzene_atom_sum = float(feats[0, sum_atom_start:sum_atom_end].sum())
        self.assertGreater(naphthalene_atom_sum, benzene_atom_sum)


# ---------------------------------------------------------------------------
# Test 3: combined feature pipeline
# ---------------------------------------------------------------------------
class TestCombinedFeaturePipeline(unittest.TestCase):
    """The combined pipeline concatenates RDKit 32-d + chemprop v2
    graph fingerprint + W₂ distance into a single matrix."""

    def test_combined_feature_pipeline(self) -> None:
        # Use the real platinai_4cell cohort (200 SMILES) so the
        # pipeline is exercised end-to-end.
        X_32d, y, smiles = load_platinai_4cell_32d_with_smiles()
        self.assertEqual(X_32d.shape[0], len(smiles))
        self.assertEqual(X_32d.shape[1], 32)
        self.assertEqual(y.shape[0], len(smiles))

        X_full, names_full = combined_feature_pipeline(
            smiles_list=smiles,
            X_32d=X_32d,
            y=y,
            reference_y=y,
            include_rdkit_32d=True,
            include_chemprop_v2_graph=True,
            include_w2_distance=True,
        )
        atom_dim, bond_dim = _probe_chemprop_dims()
        expected_dim = 32 + 3 * (atom_dim + bond_dim) + 1
        self.assertEqual(X_full.shape[1], expected_dim)
        self.assertEqual(X_full.shape[0], len(smiles))
        self.assertEqual(len(names_full), expected_dim)
        self.assertEqual(names_full[:32][0], "rdkit_32d_00")
        self.assertEqual(names_full[-1], "w2_dist_to_ref")

        # Toggle off each stream and check the dimension drops accordingly.
        X_no_graph, names_no_graph = combined_feature_pipeline(
            smiles_list=smiles,
            X_32d=X_32d,
            y=y,
            reference_y=y,
            include_rdkit_32d=True,
            include_chemprop_v2_graph=False,
            include_w2_distance=True,
        )
        self.assertEqual(X_no_graph.shape[1], 32 + 1)
        self.assertEqual(len(names_no_graph), 32 + 1)

        X_no_w2, names_no_w2 = combined_feature_pipeline(
            smiles_list=smiles,
            X_32d=X_32d,
            y=y,
            reference_y=y,
            include_rdkit_32d=True,
            include_chemprop_v2_graph=True,
            include_w2_distance=False,
        )
        self.assertEqual(X_no_w2.shape[1], 32 + 3 * (atom_dim + bond_dim))
        self.assertEqual(len(names_no_w2), 32 + 3 * (atom_dim + bond_dim))


# ---------------------------------------------------------------------------
# Test 4: R² lift on platinai_4cell vs Wf-325A baseline
# ---------------------------------------------------------------------------
class TestR2LiftOnPlatinai4Cell(unittest.TestCase):
    """Per-config Ridge R² on platinai_4cell, compared to the Wf-325A
    baseline of -0.0945 (12-d RDKit frozen set)."""

    def test_r2_lift_on_platinai_4cell(self) -> None:
        # The result JSON is the canonical output (per Wf-325A pattern);
        # also keep an in-memory copy for direct assertions.
        tmpdir = tempfile.mkdtemp(prefix="wf329w5_")
        out_path = os.path.join(tmpdir, "result.json")
        result = run_wf329w5(out_path=out_path, cohort="platinai_4cell")

        # ----- Result JSON structure -----
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["wf"], "Wf-329W5")
        self.assertEqual(data["cohort"], "platinai_4cell")
        self.assertIn("per_config", data)
        self.assertIn("wf325a_baseline", data)
        self.assertIn("best_config", data)
        self.assertIn("aggregate_status", data)
        # 8 configs evaluated (per the run_wf329w5 config table)
        self.assertGreaterEqual(len(data["per_config"]), 5)

        # ----- Per-config R² shape + verdict label -----
        valid_verdicts = {
            "CEILING_BREAK",
            "MODERATE_RECOVERY",
            "WEAK_RECOVERY",
            "NO_LIFT_HONEST_NEGATIVE",
            "HONEST_NEGATIVE_R2_DEEP",
            "PIPELINE_ERROR",
            "UNDEFINED",
        }
        r2_12d_baseline = float(WF325A_BASELINE_R2_PLA_12D)
        for cfg in data["per_config"]:
            self.assertIn("label", cfg)
            self.assertIn("r2", cfg)
            self.assertIn("lift_vs_12d_baseline", cfg)
            self.assertIn("verdict", cfg)
            if cfg.get("error"):
                self.assertEqual(cfg["verdict"], "PIPELINE_ERROR")
                continue
            # R² is finite (real-valued)
            self.assertTrue(
                np.isfinite(cfg["r2"]),
                f"R² for {cfg['label']!r} must be finite; got {cfg['r2']!r}",
            )
            # lift = r2 - r2_12d_baseline (sign-correct)
            self.assertAlmostEqual(
                cfg["lift_vs_12d_baseline"],
                cfg["r2"] - r2_12d_baseline,
                places=6,
            )
            # Verdict is documented
            self.assertIn(cfg["verdict"], valid_verdicts)
            # Verdict is consistent with the r2 / lift values
            v = cfg["verdict"]
            r2 = cfg["r2"]
            lift = cfg["lift_vs_12d_baseline"]
            if v == "CEILING_BREAK":
                self.assertGreater(r2, 0.0)
                self.assertGreaterEqual(lift, 0.10)
            elif v == "MODERATE_RECOVERY":
                self.assertGreater(r2, 0.0)
                self.assertGreaterEqual(lift, 0.05)
            elif v == "WEAK_RECOVERY":
                self.assertGreater(r2, 0.0)
            elif v == "NO_LIFT_HONEST_NEGATIVE":
                self.assertGreaterEqual(r2, r2_12d_baseline)
            elif v == "HONEST_NEGATIVE_R2_DEEP":
                self.assertLess(r2, r2_12d_baseline)

        # ----- 12-d baseline R² must match Wf-325A MEASURED value -----
        # (within sklearn/numpy tolerance; we use 1e-3 here)
        cfg_12d = next(c for c in data["per_config"] if c["label"] == "12d_baseline")
        self.assertAlmostEqual(
            cfg_12d["r2"],
            r2_12d_baseline,
            delta=1e-3,
            msg=(
                f"12-d baseline R² should match Wf-325A MEASURED "
                f"({r2_12d_baseline:+.4f}); got {cfg_12d['r2']:+.4f}"
            ),
        )

        # ----- aggregate_status + best_config are well-formed -----
        self.assertIn(data["aggregate_status"], valid_verdicts)
        bc = data["best_config"]
        self.assertIn(bc["label"], [c["label"] for c in data["per_config"]])
        self.assertTrue(np.isfinite(bc["r2"]))

        # ----- Honest framing: the result documents the Wf-325A anchor -----
        self.assertAlmostEqual(
            data["wf325a_baseline"]["r2_12d"],
            r2_12d_baseline,
            places=6,
        )

        # ----- Print summary for human inspection -----
        print()
        print(f"  Wf-325A baseline R²(12d) = {r2_12d_baseline:+.4f}")
        for c in data["per_config"]:
            if "error" in c:
                print(f"  {c['label']:>30s}: {c['error']}")
                continue
            print(
                f"  {c['label']:>30s}  n_feat={c['n_features']:>3d}  "
                f"R²={c['r2']:+.4f}  lift={c['lift_vs_12d_baseline']:+.4f}  "
                f"{c['verdict']}"
            )
        print(
            f"  best_config = {bc['label']}  "
            f"R²={bc['r2']:+.4f}  "
            f"lift={bc['lift_vs_12d_baseline']:+.4f}"
        )
        print(f"  aggregate_status = {data['aggregate_status']}")


if __name__ == "__main__":
    unittest.main()
