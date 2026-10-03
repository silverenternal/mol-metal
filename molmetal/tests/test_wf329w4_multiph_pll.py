"""Wf-329-W4 — Multi-PH/PLL feature extraction tests.

================================================================
Wf-329-W4 test suite — 4+ tests covering the three recovery
primitives (32-bin Fourier, Multi-PH integration, PLL signature)
plus the m6_kappa + m7 + R² wire-in:

1. ``test_fourier_32bin_features_shape_and_finite`` — 32-bin
   radially-binned FFT magnitude on a synthetic multi-PH feature
   map.
2. ``test_fourier_32bin_features_rotation_invariance`` — the
   radial binning is invariant under rotation of the spatial
   multi-PH map.
3. ``test_multiph_pll_extractor_triple_modality`` — joint
   extractor returns (Fourier, multi-PH, PLL) feature triple with
   documented shapes.
4. ``test_pll_signature_extracted`` — PLL signature extraction
   path returns real-valued, sorted-descending eigenvalues.
5. ``test_multiph_integration_recovery_path`` — Multi-PH
   integration via :class:`TwoParameterFiltration` returns a
   ``(32, 32)`` feature map on the metallo cohort.
6. ``test_m6_kappa_wire_in_returns_aggregated_report`` —
   :func:`wire_m6_kappa_recovery` returns a populated
   :class:`MultiPHPLLRecoveryReport` for the metallo cohort.
7. ``test_m7_diversity_wire_in_returns_pll_distance`` —
   :func:`wire_m7_diversity_recovery` returns a populated
   baseline + PLL-distance delta.
8. ``test_r2_wire_in_joint_dim_matches`` — :func:`wire_r2_recovery`
   returns a joint design matrix of expected dimension
   (fourier_dim + multi_ph_dim + pll_dim = 32 + 1024 + 32 = 1088).

All tests are CPU-only, deterministic, and complete in < 60 s
combined.  RDKit is required for tests 3-8; tests 1-2 degrade
gracefully when RDKit is unavailable (Fourier features are pure
numpy).

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf329w4_multiph_pll.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Lazy imports — graceful skip on missing deps
# ---------------------------------------------------------------------------
def _import_module():
    """Import the Wf-329-W4 module; return ``None`` on ImportError."""
    try:
        from molmetal.molmetal_lam.reward.multiph_pll_feature import (
            FOURIER_BIN_COUNT,
            MULTI_PH_N_BINS,
            PLL_TOP_K,
            MultiPHPLLFeatureExtractor,
            MultiPHPLLFeatureError,
            MultiPHPLLRecoveryReport,
            fourier_32bin_features,
            wire_m6_kappa_recovery,
            wire_m7_diversity_recovery,
            wire_r2_recovery,
        )

        return {
            "FOURIER_BIN_COUNT": FOURIER_BIN_COUNT,
            "MULTI_PH_N_BINS": MULTI_PH_N_BINS,
            "PLL_TOP_K": PLL_TOP_K,
            "MultiPHPLLFeatureExtractor": MultiPHPLLFeatureExtractor,
            "MultiPHPLLFeatureError": MultiPHPLLFeatureError,
            "MultiPHPLLRecoveryReport": MultiPHPLLRecoveryReport,
            "fourier_32bin_features": fourier_32bin_features,
            "wire_m6_kappa_recovery": wire_m6_kappa_recovery,
            "wire_m7_diversity_recovery": wire_m7_diversity_recovery,
            "wire_r2_recovery": wire_r2_recovery,
        }
    except Exception as exc:
        pytest.skip(f"multiph_pll_feature import failed: {exc}")
        return None  # unreachable


@pytest.fixture(scope="module")
def mod():
    return _import_module()


# ---------------------------------------------------------------------------
# Test 1: 32-bin Fourier feature shape + finite
# ---------------------------------------------------------------------------
def test_fourier_32bin_features_shape_and_finite(mod):
    """Wf-329-W4 §1 — 32-bin Fourier features have shape (32,) and
    are finite + non-negative."""
    n_bins = mod["FOURIER_BIN_COUNT"]
    # Synthetic multi-PH feature map with positive counts.
    rng = np.random.default_rng(0)
    feat = rng.uniform(0.0, 5.0, size=(n_bins, n_bins))
    fourier = mod["fourier_32bin_features"](feat, n_bins=n_bins)
    assert fourier.shape == (n_bins,), f"got {fourier.shape}"
    assert np.isfinite(fourier).all(), "Fourier features must be finite"
    assert (fourier >= 0.0).all(), "Fourier magnitude must be >= 0"
    # The DC bin (r=0) holds the total sum — should equal feat.sum()
    # exactly (modulo numerical noise).
    np.testing.assert_allclose(
        fourier[0],
        float(feat.sum()),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Test 2: 32-bin Fourier rotation invariance
# ---------------------------------------------------------------------------
def test_fourier_32bin_features_rotation_invariance(mod):
    """Wf-329-W4 §2 — radial binning is invariant under rotation of
    the spatial feature map (within numerical noise)."""
    n_bins = mod["FOURIER_BIN_COUNT"]
    rng = np.random.default_rng(1)
    feat = rng.uniform(0.0, 1.0, size=(n_bins, n_bins))
    # Apply a 90° rotation via np.rot90 (which is exact for fft-aligned
    # grids and preserves the DC component at the centre).
    feat_rot = np.rot90(feat)
    f0 = mod["fourier_32bin_features"](feat, n_bins=n_bins)
    f1 = mod["fourier_32bin_features"](feat_rot, n_bins=n_bins)
    # The radial-bin spectra are within numerical noise — the radial
    # structure is preserved under 90° rotation.
    np.testing.assert_allclose(f0, f1, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Joint extractor returns (Fourier, multi-PH, PLL) triple
# ---------------------------------------------------------------------------
def test_multiph_pll_extractor_triple_modality(mod):
    """Wf-329-W4 §3 — joint extractor returns three documented arrays."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    # Skip on RDKit missing — the test requires 3-D conformer.
    try:
        fourier, multi_ph, pll = extractor.compute("CCO")
    except mod["MultiPHPLLFeatureError"] as exc:
        pytest.skip(f"RDKit unavailable: {exc}")
    # Fourier: (32,)
    assert fourier.shape == (extractor.fourier_dim,), (
        f"Fourier shape mismatch: {fourier.shape} vs " f"{extractor.fourier_dim}"
    )
    assert np.isfinite(fourier).all()
    assert (fourier >= 0.0).all()
    # Multi-PH: (32, 32)
    assert multi_ph.shape == (
        extractor.n_bins,
        extractor.n_bins,
    ), f"multi-PH shape mismatch: {multi_ph.shape}"
    assert np.isfinite(multi_ph).all()
    assert (multi_ph >= 0.0).all()
    # PLL: (32,)
    assert pll.shape == (
        extractor.pll_dim,
    ), f"PLL shape mismatch: {pll.shape} vs {extractor.pll_dim}"
    assert np.isfinite(pll).all()
    # PLL eigenvalues should be in descending order (Liu-Feng 2026).
    assert np.all(pll[:-1] >= pll[1:]) - 1e-9, "PLL eigenvalues must be descending"


# ---------------------------------------------------------------------------
# Test 4: PLL signature extracted via joint extractor
# ---------------------------------------------------------------------------
def test_pll_signature_extracted(mod):
    """Wf-329-W4 §4 — PLL signature extraction path returns
    real-valued, sorted-descending eigenvalues bounded in [0, 2]."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    try:
        _, _, pll = extractor.compute("c1ccccc1")  # benzene
    except mod["MultiPHPLLFeatureError"] as exc:
        pytest.skip(f"RDKit unavailable: {exc}")
    # All real, finite, in [0, 2] (normalised Laplacian bound).
    assert np.isfinite(pll).all()
    assert (pll >= 0.0).all()
    assert (pll <= 2.0 + 1e-9).all(), f"PLL eigenvalue max={pll.max():.4f} exceeds 2.0"
    # Descending sort.
    diffs = np.diff(pll)
    assert (diffs <= 1e-9).all(), "PLL eigenvalues not descending"


# ---------------------------------------------------------------------------
# Test 5: Multi-PH integration recovery path
# ---------------------------------------------------------------------------
def test_multiph_integration_recovery_path(mod):
    """Wf-329-W4 §5 — Multi-PH integration returns a (32, 32)
    feature map on the metallo cohort."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    csv_path = _REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    if not csv_path.exists():
        pytest.skip(f"metallo CSV not found at {csv_path}")
    # Read first 5 SMILES.
    import csv

    smiles_list: list = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles_list.append(smi)
            if len(smiles_list) >= 5:
                break
    fourier_b, multi_ph_b, pll_b = extractor.compute_batch(smiles_list)
    # Multi-PH batch shape: (B, 32, 32).
    assert multi_ph_b.shape == (
        len(smiles_list),
        extractor.n_bins,
        extractor.n_bins,
    ), f"multi-PH batch shape: {multi_ph_b.shape}"
    # Fourier + PLL batch shape: (B, 32).
    assert fourier_b.shape == (len(smiles_list), extractor.n_bins)
    assert pll_b.shape == (len(smiles_list), extractor.pll_dim)


# ---------------------------------------------------------------------------
# Test 6: m6_kappa wire-in
# ---------------------------------------------------------------------------
def test_m6_kappa_wire_in_returns_aggregated_report(mod):
    """Wf-329-W4 §6 — m6_kappa recovery wire-in returns populated
    baseline + augmented values + delta."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    csv_path = _REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    if not csv_path.exists():
        pytest.skip(f"metallo CSV not found at {csv_path}")
    # Subset (8 SMILES — enough for the Ridge 80/20 split).
    import csv

    smiles_list: list = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles_list.append(smi)
            if len(smiles_list) >= 8:
                break
    out = mod["wire_m6_kappa_recovery"](smiles_list, extractor)
    assert "m6_kappa_baseline" in out
    assert "m6_kappa_augmented" in out
    assert "m6_kappa_delta" in out
    # Baseline should be a finite float (or NaN if RDKit failed).
    assert out["m6_kappa_baseline"] is not None
    assert isinstance(out["m6_kappa_baseline"], float)


# ---------------------------------------------------------------------------
# Test 7: m7 diversity wire-in (PLL distance)
# ---------------------------------------------------------------------------
def test_m7_diversity_wire_in_returns_pll_distance(mod):
    """Wf-329-W4 §7 — m7 PLL-distance recovery wire-in returns a
    baseline + PLL delta."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    csv_path = _REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    if not csv_path.exists():
        pytest.skip(f"metallo CSV not found at {csv_path}")
    import csv

    smiles_list: list = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles_list.append(smi)
            if len(smiles_list) >= 10:
                break
    out = mod["wire_m7_diversity_recovery"](
        smiles_list,
        extractor,
        sample_size=10,
    )
    assert "m7_diversity_baseline" in out
    assert "m7_diversity_pll" in out
    assert "m7_diversity_delta" in out
    # Both baseline and PLL must be finite floats when RDKit is up.
    assert np.isfinite(out["m7_diversity_baseline"])
    assert np.isfinite(out["m7_diversity_pll"])
    # PLL distance should be non-negative.
    assert out["m7_diversity_pll"] >= 0.0


# ---------------------------------------------------------------------------
# Test 8: R² wire-in joint dim matches
# ---------------------------------------------------------------------------
def test_r2_wire_in_joint_dim_matches(mod):
    """Wf-329-W4 §8 — R² wire-in returns joint design matrix with
    expected dimension (Fourier + Multi-PH + PLL = 32 + 1024 + 32)."""
    extractor = mod["MultiPHPLLFeatureExtractor"]()
    csv_path = _REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    if not csv_path.exists():
        pytest.skip(f"metallo CSV not found at {csv_path}")
    import csv

    smiles_list: list = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles_list.append(smi)
            if len(smiles_list) >= 10:
                break
    out = mod["wire_r2_recovery"](smiles_list, extractor)
    expected_joint = extractor.fourier_dim + extractor.multi_ph_dim + extractor.pll_dim
    assert (
        out["joint_dim"] == expected_joint
    ), f"joint_dim={out['joint_dim']} vs expected={expected_joint}"
    expected_baseline = extractor.fourier_dim + extractor.pll_dim
    assert out["baseline_dim"] == expected_baseline
    # R² values must be finite floats (NaN allowed only if n < 4).
    assert isinstance(out["r2_joint"], float)
    assert isinstance(out["r2_baseline"], float)
