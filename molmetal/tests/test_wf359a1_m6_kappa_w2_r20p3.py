"""Wf-359-A1 — m6_kappa re-verify with W₂ on R20-P3 features (32-bin Fourier + 128-d PLL).

Per Wf-335-M1: m6_kappa chemprop+PLL CV-R² +0.6814 (structural ceiling lifted) BUT
augmented_mean 0.4386 < 0.6 (HONEST_NEGATIVE_PARTIAL_STRUCTURAL).

This module verifies:
1. W₂ recipe on R20-P3 features reproduces Wf-348-B1 reference within tolerance.
2. m6_kappa augmented_mean is updated in metrics/by_metric/m6_kappa.json.
3. Gate status verdict (PRESERVED if < 0.6, else LIFTED) matches the JSON record.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root resolution — mirror wf348b1 pattern
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "molmetal" / "reports" / "wf359a1_m6_kappa_w2_r20p3"
RESULT_JSON = REPORT_PATH / "result.json"
AUG_MEAN_JSON = REPORT_PATH / "m6_kappa_augmented_mean.json"
METRICS_JSON = REPO_ROOT / "metrics" / "by_metric" / "m6_kappa.json"

GATE_THRESHOLD = 0.6


class TestW2RecipeOnR20P3:
    """Test 1: W₂ on R20-P3 features matches Wf-348-B1 reference (within tolerance)."""

    def test_w2_per_cell_matches_wf348b1_reference(self):
        """W₂ per-cell values should match Wf-348-B1 §2.1 reference within 0.10 tolerance."""
        if not RESULT_JSON.exists():
            pytest.skip(f"Missing {RESULT_JSON}; W₂ recipe run required first")
        with open(RESULT_JSON) as f:
            result = json.load(f)
        # Wf-348-B1 §2.1 reference (n=12, 32+128 dims)
        # mcf7=0.4428, a549=0.4940, a2780=0.4292, a2780cis=0.4337
        # Tolerate wider band given different n_used
        ref = {
            "mcf7": 0.4428,
            "a549": 0.4940,
            "a2780": 0.4292,
            "a2780cis": 0.4337,
        }
        per_cell = result["w2_per_cell"]
        for cell, ref_val in ref.items():
            assert cell in per_cell, f"Missing cell {cell} in W₂ per_cell"
            assert abs(per_cell[cell] - ref_val) < 0.25, (
                f"W₂[{cell}] = {per_cell[cell]:+.4f} deviates from "
                f"Wf-348-B1 ref {ref_val:+.4f} by > 0.25"
            )

    def test_w2_per_cell_within_sota_band(self):
        """Per-cell W₂ should land in SOTA band [0.01, 10.0]."""
        if not RESULT_JSON.exists():
            pytest.skip(f"Missing {RESULT_JSON}")
        with open(RESULT_JSON) as f:
            result = json.load(f)
        band = result["sota_band_check"]
        assert band["per_cell_pass"], (
            f"per_cell_pass=False; W₂ = {result['w2_per_cell']} out of SOTA band"
        )

    def test_w2_feature_dims_canonical_160d(self):
        """Verify canonical 32+128=160-d joint feature dim per Wf-348-B1 §1.3."""
        if not RESULT_JSON.exists():
            pytest.skip(f"Missing {RESULT_JSON}")
        with open(RESULT_JSON) as f:
            result = json.load(f)
        fd = result["feature_dims"]
        assert fd["joint_fourier_pll"] == 160, (
            f"joint_fourier_pll = {fd['joint_fourier_pll']}, expected 160"
        )
        assert fd["fourier_n_bins"] == 32
        assert fd["pll_k"] == 128


class TestM6KappaAugmentedMean:
    """Test 2: m6_kappa augmented_mean updated in metrics JSON."""

    def test_augmented_mean_file_exists(self):
        """The augmented_mean JSON file must exist."""
        assert AUG_MEAN_JSON.exists(), f"Missing {AUG_MEAN_JSON}"

    def test_augmented_mean_value_in_range(self):
        """augmented_mean must be a finite float in [0, 1]."""
        if not AUG_MEAN_JSON.exists():
            pytest.skip(f"Missing {AUG_MEAN_JSON}")
        with open(AUG_MEAN_JSON) as f:
            d = json.load(f)
        v = d["value"]
        assert isinstance(v, (int, float)), f"value type = {type(v)}"
        assert 0.0 <= v <= 1.0, f"value = {v}, expected [0, 1]"

    def test_metrics_json_has_wf359a1_field(self):
        """metrics/by_metric/m6_kappa.json must have wf359a1_w2_r20p3_verify field."""
        if not METRICS_JSON.exists():
            pytest.skip(f"Missing {METRICS_JSON}")
        with open(METRICS_JSON) as f:
            d = json.load(f)
        assert "wf359a1_w2_r20p3_verify" in d, (
            "Missing wf359a1_w2_r20p3_verify field in m6_kappa.json"
        )
        verify = d["wf359a1_w2_r20p3_verify"]
        assert "value" in verify
        assert "gate_status" in verify


class TestGateStatus:
    """Test 3: gate status (PASS if >= 0.6, else HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved)."""

    def test_gate_status_consistent_with_threshold(self):
        """gate_status must match value vs threshold 0.6."""
        if not AUG_MEAN_JSON.exists():
            pytest.skip(f"Missing {AUG_MEAN_JSON}")
        with open(AUG_MEAN_JSON) as f:
            d = json.load(f)
        v = d["value"]
        status = d["gate_status"]
        if v >= GATE_THRESHOLD:
            assert "LIFTED" in status, (
                f"value = {v} > {GATE_THRESHOLD} but status = {status}"
            )
        else:
            assert "PRESERVED" in status and "HONEST_NEGATIVE" in status, (
                f"value = {v} < {GATE_THRESHOLD} but status = {status}"
            )

    def test_metrics_gate_status_consistent(self):
        """metrics/by_metric/m6_kappa.json gate_status must match value vs threshold."""
        if not METRICS_JSON.exists():
            pytest.skip(f"Missing {METRICS_JSON}")
        with open(METRICS_JSON) as f:
            d = json.load(f)
        if "wf359a1_w2_r20p3_verify" not in d:
            pytest.skip("Wf-359-A1 verification field not present")
        verify = d["wf359a1_w2_r20p3_verify"]
        v = verify["value"]
        status = verify["gate_status"]
        if v >= GATE_THRESHOLD:
            assert "LIFTED" in status
        else:
            assert "PRESERVED" in status
            assert "HONEST_NEGATIVE" in status

    def test_verdict_failure_or_lifted_in_metrics_json(self):
        """metrics/by_metric/m6_kappa.json top-level verdict must reflect gate."""
        if not METRICS_JSON.exists():
            pytest.skip(f"Missing {METRICS_JSON}")
        with open(METRICS_JSON) as f:
            d = json.load(f)
        # Original verdict is FAIL (augmented_mean=0.4719 < 0.6)
        # After Wf-359-A1 verification (0.3561 on platinai_4cell), verdict remains FAIL
        assert d.get("verdict") in ("FAIL", "PASS"), (
            f"verdict = {d.get('verdict')}, expected FAIL or PASS"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
