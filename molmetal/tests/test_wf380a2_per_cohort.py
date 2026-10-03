"""Tests for Wf-380-A2 — m2 R² per-cohort feature engineering (TODO-201).

=====================================================================
Wf-380-A2 / TODO-201 — Per-cohort-specific feature selection
matched to cohort size to prevent the Wf-367-B1 OVERFIT finding
(50+ features on n=200 cohorts produce negative R² via bias-variance
inflation).

Test surface (3 NEW tests):
  * test_per_cohort_config_match
        The select_features(cohort_size) function returns configs
        matching the YAML config matrix thresholds (n<300 -> 12-d,
        n<1000 -> 24-d, n>=1000 -> 50-d).
  * test_12d_ridge_no_overfit
        On a synthetic cohort of n=200 + 12-d features, the per-cohort
        pipeline gives positive R² (12-d prevents overfit vs the 50-d
        ensemble baseline).
  * test_24d_intermediate
        On a synthetic cohort of n=500 + 24-d features, the per-cohort
        pipeline gives positive R² (intermediate tier works).

Honest framing (party-twenty-spirit §3.5):
  * These tests are SYNTHETIC validation, NOT real-cohort retrain.
  * If the per-cohort config produces positive synthetic R² across
    all tiers, the METHODOLOGY is validated (real-cohort lift is
    PROJECTED, pending GPU retrain per AGENTS.md §5.1).
  * MEASURED-vs-PROJECTED tag discipline: synthetic R² is MEASURED;
    real-cohort lift is PROJECTED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wf380a2_m2_r2_per_cohort import (  # noqa: E402
    load_config,
    per_cohort_pipeline,
    select_features,
)


# ---------------------------------------------------------------------------
# Test 1: per-cohort config match (thresholds + features)
# ---------------------------------------------------------------------------
def test_per_cohort_config_match() -> None:
    """select_features(cohort_size) returns configs matching the YAML matrix.

    Per TODO-201 §2.2:
      * n < 300  -> tier n_lt_300, 12-d RDKit, Ridge alpha=1.0
      * n < 1000 -> tier n_lt_1000, 24-d, Ridge alpha=0.5
      * n >= 1000 -> tier n_gte_1000, 50-d, ensemble alpha=0.1

    The YAML config's per_cohort_targets table is the source-of-truth
    baseline + target R²; the select_features thresholds must agree.
    """
    # Tier 1: n=200 -> n_lt_300
    cfg_200 = select_features(200)
    assert cfg_200["tier"] == "n_lt_300"
    assert cfg_200["n_features_expected"] == 12
    assert cfg_200["alpha"] == 1.0
    assert "ridge" in cfg_200["model"]

    # Tier 1: n=299 -> still n_lt_300 (boundary)
    cfg_299 = select_features(299)
    assert cfg_299["tier"] == "n_lt_300"
    assert cfg_299["n_features_expected"] == 12

    # Tier 2: n=300 -> n_lt_1000 (boundary — n<300 fails; n=300 enters tier 2)
    cfg_300 = select_features(300)
    assert cfg_300["tier"] == "n_lt_1000"
    assert cfg_300["n_features_expected"] == 24
    assert cfg_300["alpha"] == 0.5

    # Tier 2: n=500 -> n_lt_1000
    cfg_500 = select_features(500)
    assert cfg_500["tier"] == "n_lt_1000"
    assert cfg_500["n_features_expected"] == 24
    assert cfg_500["alpha"] == 0.5

    # Tier 2: n=999 -> n_lt_1000 (boundary)
    cfg_999 = select_features(999)
    assert cfg_999["tier"] == "n_lt_1000"
    assert cfg_999["n_features_expected"] == 24

    # Tier 3: n=1000 -> n_gte_1000 (boundary — n>=1000 enters tier 3)
    cfg_1000 = select_features(1000)
    assert cfg_1000["tier"] == "n_gte_1000"
    assert cfg_1000["n_features_expected"] == 50
    assert cfg_1000["alpha"] == 0.1
    assert cfg_1000["model"] == "ensemble"

    # Tier 3: n=2000 -> n_gte_1000
    cfg_2000 = select_features(2000)
    assert cfg_2000["tier"] == "n_gte_1000"
    assert cfg_2000["n_features_expected"] == 50

    # YAML config consistency check
    cfg_yaml = load_config("molmetal/configs/m2_r2_per_cohort.yaml")
    assert cfg_yaml["version"] == "1.0"
    assert cfg_yaml["tiers"]["n_lt_300"]["alpha"] == 1.0
    assert cfg_yaml["tiers"]["n_lt_1000"]["alpha"] == 0.5
    assert cfg_yaml["tiers"]["n_gte_1000"]["alpha"] == 0.1
    assert cfg_yaml["per_cohort_targets"]["platinai_4cell"]["tier"] == "n_lt_300"
    assert cfg_yaml["per_cohort_targets"]["hilbert"]["tier"] == "n_lt_300"
    assert (
        cfg_yaml["per_cohort_targets"]["metalcytotox_combined"]["tier"] == "n_gte_1000"
    )
    assert cfg_yaml["per_cohort_targets"]["intermediate_n500"]["tier"] == "n_lt_1000"


# ---------------------------------------------------------------------------
# Test 2: 12-d + Ridge alpha=1.0 gives positive R² on n=200 synthetic
# ---------------------------------------------------------------------------
def test_12d_ridge_no_overfit() -> None:
    """n=200 + 12-d RDKit + Ridge alpha=1.0 gives positive synthetic R².

    Per TODO-201 §4.1 — this is the remediation for the Wf-367-B1
    OVERFIT finding on the n=200 cohorts (platinai_4cell + hilbert).
    The 50+ feature ensemble overfits on n=200; reducing to 12-d for
    n<300 cohorts is the per-cohort axis.
    """
    # n=200 + 12-d
    out = per_cohort_pipeline(
        cohort_size=200,
        seed=42,
        n_splits=5,
        n_seeds=5,
    )
    assert out["config"]["tier"] == "n_lt_300"
    assert out["config"]["n_features_expected"] == 12
    assert out["n_features_built"] == 12
    assert out["n_samples_built"] == 200
    # MEASURED synthetic R² > 0 (12-d Ridge prevents overfit)
    r2 = out["measured_synthetic_r2_mean"]
    assert r2 > 0, f"expected positive R² on n=200 + 12-d synthetic, got {r2:+.4f}"
    # Std should be finite (reproducible K-fold CV)
    assert out["measured_synthetic_r2_std"] >= 0
    assert out["alpha_used"] == 1.0


# ---------------------------------------------------------------------------
# Test 3: 24-d + Ridge alpha=0.5 works for n=500 (intermediate tier)
# ---------------------------------------------------------------------------
def test_24d_intermediate() -> None:
    """n=500 + 24-d RDKit + chemprop v3 + Ridge alpha=0.5 works.

    Per TODO-201 §2.1 — the intermediate cohort tier covers n=500
    (e.g. future cohorts between platinai_4cell n=200 and
    metalcytotox_combined n=2000). The 24-d features + Ridge alpha=0.5
    is the medium-regularization tier.
    """
    out = per_cohort_pipeline(
        cohort_size=500,
        seed=42,
        n_splits=5,
        n_seeds=5,
    )
    assert out["config"]["tier"] == "n_lt_1000"
    assert out["config"]["n_features_expected"] == 24
    assert out["n_features_built"] == 24
    assert out["n_samples_built"] == 500
    # MEASURED synthetic R² > 0 (24-d Ridge with medium regularization)
    r2 = out["measured_synthetic_r2_mean"]
    assert r2 > 0, f"expected positive R² on n=500 + 24-d synthetic, got {r2:+.4f}"
    assert out["alpha_used"] == 0.5
    # Also: n>=1000 tier (50-d) should give positive R² on n=2000
    out_2000 = per_cohort_pipeline(
        cohort_size=2000,
        seed=42,
        n_splits=5,
        n_seeds=5,
    )
    assert out_2000["config"]["tier"] == "n_gte_1000"
    assert out_2000["n_features_built"] == 50
    r2_2000 = out_2000["measured_synthetic_r2_mean"]
    assert (
        r2_2000 > 0
    ), f"expected positive R² on n=2000 + 50-d synthetic, got {r2_2000:+.4f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
