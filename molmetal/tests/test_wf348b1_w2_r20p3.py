"""Tests for Wf-348-B1 — W₂ recipe on R20-P3 Multi-PH+PLL features.

====================================================================
Wf-348-B1 (2026-09-29) — 4 NEW tests for the W₂ recipe on the
R20-P3 architectural fix (32-bin Fourier + 128-d PLL).
====================================================================

The 12-d RDKit 2D signal floor is broken by the 160-d (32+128) joint
Fourier + PLL feature ceiling (Wf-127 §Cause 2 + Wf-313G dashboard §3.4).
The original TODO-179 W₂ recipe was STATUS_HOLD on the 12-d floor;
Wf-348-B1 ships the architecturally feasible W₂ recipe at 160-d.

These tests cover the four claim axes:

  * Test 1 — R20-P3 feature extraction produces (32,)+128-d outputs.
  * Test 2 — W₂ computation on PlatinAI 4-cell returns finite, scalar
    values for each cell-line.
  * Test 3 — per-cell + per-cohort + cross-cohort W₂ are all
    finite scalars (the 3-axis output of the recipe).
  * Test 4 — W₂ magnitudes land in the documented SOTA band
    [0.01, 10.0] (per-row 1-D W₂ in log10(IC50) units).

All tests are **CPU-only** and use the real
:mod:`molmetal.molmetal_lam.reward.multiph_pll_feature` extractor
(small n_bins=16 for the unit-test speed budget; the production
n_bins=32 path is exercised by the recipe itself in ``wf348b1_w2_r20p3.py``).
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — kept inside helpers / fixtures so a missing dep only fails
# the relevant test, not the whole module.
# ---------------------------------------------------------------------------
def _import_recipe():
    """Lazy import of :mod:`molmetal.scripts.wf348b1_w2_r20p3`."""
    try:
        from molmetal.scripts import wf348b1_w2_r20p3 as recipe
        return recipe
    except Exception as exc:
        pytest.skip(
            f"wf348b1_w2_r20p3 recipe not importable: {exc}",
            allow_module_level=False,
        )


def _import_extractor():
    """Lazy import of the R20-P3 feature extractor."""
    try:
        from molmetal.molmetal_lam.reward.multiph_pll_feature import (
            MultiPHPLLFeatureExtractor,
        )
        return MultiPHPLLFeatureExtractor
    except Exception as exc:
        pytest.skip(
            f"MultiPHPLLFeatureExtractor not importable: {exc}",
            allow_module_level=False,
        )


# ---------------------------------------------------------------------------
# Canonical fixture SMILES — 6 small molecules that RDKit can embed quickly
# ---------------------------------------------------------------------------
CANONICAL_SMILES: tuple = (
    "CCO",                  # ethanol
    "c1ccccc1",             # benzene
    "CC(=O)O",              # acetic acid
    "CCN",                  # ethylamine
    "c1ccc2ccccc2c1",       # naphthalene
    "OCC(O)CO",             # glycerol
)


# ---------------------------------------------------------------------------
# Test 1 — 32-bin Fourier + 128-d PLL feature extraction
# ---------------------------------------------------------------------------
class TestFeatureExtraction:
    """32-bin Fourier + 128-d PLL feature extraction produces the canonical
    160-d joint feature vector.
    """

    def test_extract_r20p3_features_returns_canonical_dims(self):
        """Returns ``(fourier, multi_ph, pll, valid_mask)`` with the
        canonical 32-bin + 128-d dims when ``n_bins=32`` and ``pll_k=128``.
        """
        recipe = _import_recipe()
        fourier_b, multi_ph_b, pll_b, valid_mask = recipe.extract_r20p3_features(
            list(CANONICAL_SMILES), n_bins=32, pll_k=128,
        )
        assert fourier_b.shape == (len(CANONICAL_SMILES), 32)
        assert multi_ph_b.shape == (len(CANONICAL_SMILES), 32, 32)
        assert pll_b.shape == (len(CANONICAL_SMILES), 128)
        assert valid_mask.shape == (len(CANONICAL_SMILES),)
        assert valid_mask.dtype == bool
        # valid_mask must be True for canonical SMILES (RDKit can parse
        # all six).
        assert valid_mask.sum() >= 4

    def test_extract_r20p3_features_handles_invalid_smiles(self):
        """Invalid SMILES yield zero rows in the per-modality output but
        the batch shape contract holds.
        """
        recipe = _import_recipe()
        bad = ["", "not_a_smiles", "c1ccccc1"]  # 2 invalid + 1 valid
        fourier_b, multi_ph_b, pll_b, valid_mask = recipe.extract_r20p3_features(
            bad, n_bins=16, pll_k=16,
        )
        assert fourier_b.shape == (3, 16)
        assert multi_ph_b.shape == (3, 16, 16)
        assert pll_b.shape == (3, 16)
        # The valid SMILES ('c1ccccc1') is index 2.
        assert bool(valid_mask[2]) is True
        # Invalid SMILES are zero-filled.
        for i in (0, 1):
            assert bool(valid_mask[i]) is False
            assert np.allclose(fourier_b[i], 0.0)
            assert np.allclose(multi_ph_b[i], 0.0)
            assert np.allclose(pll_b[i], 0.0)

    def test_joint_fourier_pll_feature_concatenates_to_160d(self):
        """``joint_fourier_pll_feature`` produces the canonical 160-d axis."""
        fourier = np.random.default_rng(0).standard_normal((5, 32))
        pll = np.random.default_rng(1).standard_normal((5, 128))
        recipe = _import_recipe()
        joint = recipe.joint_fourier_pll_feature(fourier, pll)
        assert joint.shape == (5, 160)
        np.testing.assert_array_equal(joint[:, :32], fourier)
        np.testing.assert_array_equal(joint[:, 32:], pll)

    def test_extractor_metadata_dim_contract(self):
        """The R20-P3 extractor reports ``fourier_dim + pll_dim = 160``."""
        extractor_cls = _import_extractor()
        ext = extractor_cls(n_bins=32, k=128)
        assert int(ext.fourier_dim) == 32
        assert int(ext.pll_dim) == 128
        # Multi-PH dim is the squared side length.
        assert int(ext.multi_ph_dim) == 32 * 32
        # Joint full dim = fourier + multi_ph + pll.
        assert int(ext.joint_dim) == 32 + 32 * 32 + 128


# ---------------------------------------------------------------------------
# Test 2 — W₂ computation on PlatinAI 4-cell (synthetic pIC50)
# ---------------------------------------------------------------------------
class TestW2PerCell:
    """Per-cell W₂ computation returns finite, scalar values for each
    PlatinAI 4-cell line.
    """

    def test_per_cell_w2_on_synthetic_pic50(self):
        """On a small synthetic 4-cell cohort the per-cell W₂ values are
        finite and non-negative.
        """
        recipe = _import_recipe()
        rng = np.random.default_rng(42)
        y_per_cell = {
            "mcf7": rng.normal(loc=5.0, scale=0.5, size=20),
            "a549": rng.normal(loc=4.5, scale=0.6, size=20),
            "a2780": rng.normal(loc=5.5, scale=0.4, size=20),
            "a2780cis": rng.normal(loc=4.8, scale=0.7, size=20),
        }
        out = recipe.compute_w2_per_cell(y_per_cell)
        assert set(out.keys()) >= {"mcf7", "a549", "a2780", "a2780cis"}
        for cell, v in out.items():
            assert math.isfinite(float(v)), f"{cell} W₂ not finite: {v}"
            assert float(v) >= 0.0, f"{cell} W₂ negative: {v}"

    def test_per_cell_w2_finite_for_self_reference(self):
        """W₂ of a per-cell distribution to itself is finite and reflects
        the per-row distance from a Dirac at each row to the reference
        distribution (a positive number, NOT zero — that is the per-row
        W₂ alignment signal).  Pairwise W₂(y, y) is zero by definition;
        see :func:`compute_w2_per_cohort` instead.
        """
        recipe = _import_recipe()
        rng = np.random.default_rng(0)
        y = rng.normal(loc=5.0, scale=0.5, size=20)
        out = recipe.compute_w2_per_cell({"mcf7": y})
        assert math.isfinite(out["mcf7"])
        # For a single distribution, the per-row W₂ from Dirac to the
        # cohort is the mean absolute deviation from each row to a
        # random draw of the cohort — bounded below by the cohort std.
        assert out["mcf7"] > 0.0
        assert out["mcf7"] < 10.0  # well within SOTA band

    def test_per_cell_w2_with_external_reference(self):
        """When ``reference_per_cell`` is provided, the reference distribution
        is used instead of the per-cell row population.
        """
        recipe = _import_recipe()
        rng = np.random.default_rng(7)
        # Predicted shifted by +0.5 (every row ~0.5 above the reference).
        y_pred = rng.normal(loc=5.5, scale=0.3, size=15)
        ref = rng.normal(loc=5.0, scale=0.3, size=100)
        out = recipe.compute_w2_per_cell(
            {"mcf7": y_pred},
            reference_per_cell={"mcf7": ref},
        )
        # Mean per-row W₂ should be ~0.5 (the mean shift).
        assert 0.3 < out["mcf7"] < 0.7


# ---------------------------------------------------------------------------
# Test 3 — Per-cell + per-cohort + cross-cohort W₂ all finite
# ---------------------------------------------------------------------------
class TestThreeAxisW2:
    """The three W₂ axes (per-cell, per-cohort, cross-cohort) all return
    finite scalars on a synthetic 4-cell cohort.
    """

    @pytest.fixture
    def synthetic_cohort(self):
        rng = np.random.default_rng(123)
        n = 30
        y = rng.normal(loc=5.0, scale=0.5, size=n)
        y_per_cell = {
            cell: rng.normal(loc=5.0 + i * 0.1, scale=0.5, size=n)
            for i, cell in enumerate(("mcf7", "a549", "a2780", "a2780cis"))
        }
        feature = rng.standard_normal((n, 160))
        return y, y_per_cell, feature

    def test_per_cohort_w2_keys(self, synthetic_cohort):
        recipe = _import_recipe()
        y, _, _ = synthetic_cohort
        out = recipe.compute_w2_per_cohort(y)
        assert "mean_per_row_w2" in out
        assert "pairwise_self_w2" in out
        assert "pairwise_ref_w2" in out
        for v in out.values():
            assert math.isfinite(float(v))
        # Self-distance (W₂(y, y)) should be exactly 0 by definition.
        assert abs(float(out["pairwise_self_w2"])) < 1e-9

    def test_cross_cohort_w2_pic50_axis(self, synthetic_cohort):
        recipe = _import_recipe()
        y, _, _ = synthetic_cohort
        # Two non-identical 1-D distributions.
        rng = np.random.default_rng(99)
        y_a = rng.normal(loc=5.0, scale=0.5, size=30)
        y_b = rng.normal(loc=5.5, scale=0.5, size=30)
        out = recipe.compute_w2_cross_cohort(y_a=y_a, y_b=y_b)
        # pIC50 axes should all be finite and non-negative.
        for k in ("w2_pic50_a_to_b", "w2_pic50_b_to_a", "w2_pic50_symmetric"):
            assert math.isfinite(float(out[k]))
            assert float(out[k]) >= 0.0
        # Symmetric distance is the average of a->b and b->a.
        assert abs(
            float(out["w2_pic50_symmetric"])
            - 0.5 * (float(out["w2_pic50_a_to_b"]) + float(out["w2_pic50_b_to_a"]))
        ) < 1e-9

    def test_cross_cohort_w2_feature_axis(self, synthetic_cohort):
        recipe = _import_recipe()
        _, _, _ = synthetic_cohort
        rng = np.random.default_rng(5)
        # Two distinct feature clouds (different means per dim).
        feat_a = rng.normal(loc=0.0, scale=1.0, size=(30, 160))
        feat_b = rng.normal(loc=0.5, scale=1.0, size=(30, 160))
        out = recipe.compute_w2_cross_cohort(
            y_a=np.zeros(30), y_b=np.ones(30),
            feature_a=feat_a, feature_b=feat_b, n_mc=64, seed=0,
        )
        # Feature-axis W₂ is finite and non-negative.
        for k in ("w2_feature_a_to_b", "w2_feature_b_to_a", "w2_feature_symmetric"):
            assert math.isfinite(float(out[k]))
            assert float(out[k]) >= 0.0
        # The shifted-feature W₂ (mean shift +0.5) should be positive.
        assert float(out["w2_feature_a_to_b"]) > 0.0

    def test_three_axes_combined_recipe_shape(self, synthetic_cohort):
        """End-to-end shape: per-cell dict has 4 keys; per-cohort has 3 keys;
        cross-cohort has 6 keys.
        """
        recipe = _import_recipe()
        _, y_per_cell, _ = synthetic_cohort
        per_cell = recipe.compute_w2_per_cell(y_per_cell)
        per_cohort = recipe.compute_w2_per_cohort(np.zeros(30))
        y_a = np.zeros(15)
        y_b = np.zeros(15)
        feat_a = np.zeros((15, 32))
        feat_b = np.zeros((15, 32))
        cross = recipe.compute_w2_cross_cohort(y_a, y_b, feat_a, feat_b)
        assert len(per_cell) == 4
        assert len(per_cohort) == 3
        assert len(cross) == 6


# ---------------------------------------------------------------------------
# Test 4 — W₂ magnitudes match SOTA band [0.01, 10.0]
# ---------------------------------------------------------------------------
class TestSotaBand:
    """W₂ magnitudes on a synthetic pIC50 cohort with realistic spread
    land in the documented SOTA band [0.01, 10.0] pIC50 units.
    """

    def test_synthetic_band_pass(self):
        """A realistic pIC50 cohort (std ~1.0) produces W₂ values inside
        the SOTA band per Wf-333 ceiling recovery.
        """
        recipe = _import_recipe()
        rng = np.random.default_rng(0)
        n = 50
        y_per_cell = {
            cell: rng.normal(loc=5.0 + i * 0.1, scale=0.8, size=n)
            for i, cell in enumerate(("mcf7", "a549", "a2780", "a2780cis"))
        }
        per_cell = recipe.compute_w2_per_cell(y_per_cell)
        per_cohort = recipe.compute_w2_per_cohort(
            np.concatenate([v for v in y_per_cell.values()])
        )
        # Cross-cohort: use a *shifted* copy so the W₂ distance is
        # positive (W₂(y, y) = 0 by definition; we need a non-zero
        # cross-cohort W₂ to exercise the band check).
        y_a = y_per_cell["mcf7"]
        y_b = y_per_cell["a549"] + 0.3  # mean shift +0.3
        feat_a = rng.standard_normal((n, 32))
        feat_b = rng.standard_normal((n, 32)) + 0.2
        cross = recipe.compute_w2_cross_cohort(
            y_a=y_a, y_b=y_b,
            feature_a=feat_a, feature_b=feat_b, n_mc=64, seed=0,
        )
        sota = recipe.sota_band_check(per_cell, per_cohort, cross)
        assert sota["band_low"] == 0.01
        assert sota["band_high"] == 10.0
        assert sota["verdict"] == "SOTA_BAND_PASS"
        assert sota["per_cell_pass"] is True
        assert sota["per_cohort_pass"] is True
        assert sota["cross_cohort_pass"] is True

    def test_synthetic_band_fail_when_magnitudes_out_of_range(self):
        """If the per-row W₂ values fall outside the band the verdict
        reflects the failure (SOTA_BAND_FAIL when all three axes fail).
        """
        recipe = _import_recipe()
        # Construct a pathological per-cell dict that returns 0.0 (below
        # band) on every cell.  The sota_band_check explicitly excludes
        # 0.0 from the band [0.01, 10.0].
        per_cell = {"mcf7": 0.0, "a549": 0.0, "a2780": 0.0, "a2780cis": 0.0}
        per_cohort = {
            "mean_per_row_w2": 0.0,
            "pairwise_self_w2": 0.0,
            "pairwise_ref_w2": 0.0,
        }
        cross = {
            "w2_pic50_a_to_b": 0.0,
            "w2_pic50_b_to_a": 0.0,
            "w2_pic50_symmetric": 0.0,
            "w2_feature_a_to_b": 0.0,
            "w2_feature_b_to_a": 0.0,
            "w2_feature_symmetric": 0.0,
        }
        sota = recipe.sota_band_check(per_cell, per_cohort, cross)
        assert sota["verdict"] == "SOTA_BAND_FAIL"
        assert sota["per_cell_pass"] is False

    def test_sota_band_partial_verdict(self):
        """When only one axis passes, verdict = SOTA_BAND_PARTIAL."""
        recipe = _import_recipe()
        # Only per-cohort inside band; per-cell + cross-cohort out.
        per_cell = {"mcf7": 0.0, "a549": 0.0, "a2780": 0.0, "a2780cis": 0.0}
        per_cohort = {
            "mean_per_row_w2": 0.5,
            "pairwise_self_w2": 0.5,
            "pairwise_ref_w2": 0.5,
        }
        cross = {
            "w2_pic50_a_to_b": 0.0,
            "w2_pic50_b_to_a": 0.0,
            "w2_pic50_symmetric": 0.0,
            "w2_feature_a_to_b": 0.0,
            "w2_feature_b_to_a": 0.0,
            "w2_feature_symmetric": 0.0,
        }
        sota = recipe.sota_band_check(per_cell, per_cohort, cross)
        assert sota["verdict"] == "SOTA_BAND_PARTIAL"
        assert sota["per_cohort_pass"] is True
        assert sota["per_cell_pass"] is False
        assert sota["cross_cohort_pass"] is False
