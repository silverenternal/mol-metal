"""Unit tests for Wf-249 Fix #1 — 1024-bit Morgan / RDK fingerprint extractor.

Tests cover:
  1. Morgan FP shape, dtype, determinism.
  2. RDK FP shape, dtype, determinism.
  3. compute_features_v2 mode routing (3 modes).
  4. Backward-compat (existing 12-d path unchanged).
  5. Empty / invalid SMILES handling.
  6. CLI flag --features accepted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.extract_rdkit_2d_descriptors import (
    FEATURES_V2_MODES,
    compute_features_v2,
    compute_morgan_fp_1024,
    compute_rdk_fp_1024,
    compute_rdkit_descriptors,
)


CISPLATIN = "Cl[Pt](N)(N)Cl"
OXALIPLATIN = "O=C1OC(=O)C2(CC2)N1.[Pt]"  # simplified; may be parsed as 2 fragments


def test_morgan_fp_shape_and_dtype():
    fp = compute_morgan_fp_1024(CISPLATIN)
    assert fp is not None, "FP must not be None on parseable SMILES"
    assert fp.shape == (1024,), f"Expected (1024,), got {fp.shape}"
    assert fp.dtype == np.uint8, f"Expected uint8, got {fp.dtype}"
    assert 0 <= int(fp.sum()) <= 1024, "Bit count out of range"


def test_rdk_fp_shape_and_dtype():
    fp = compute_rdk_fp_1024(CISPLATIN)
    assert fp is not None
    assert fp.shape == (1024,)
    assert fp.dtype == np.uint8


def test_morgan_fp_deterministic():
    fp_a = compute_morgan_fp_1024(CISPLATIN)
    fp_b = compute_morgan_fp_1024(CISPLATIN)
    assert np.array_equal(fp_a, fp_b), "FP must be deterministic"


def test_features_v2_rdkit12_mode():
    v = compute_features_v2(CISPLATIN, mode="rdkit12")
    assert v is not None
    assert v.shape == (12,), f"rdkit12 mode should be 12-d, got {v.shape}"
    d12 = compute_rdkit_descriptors(CISPLATIN)
    # First 12 entries must match legacy path bit-for-bit (cast to float32).
    assert np.allclose(v, np.asarray(d12, dtype=np.float32))


def test_features_v2_fp1024_mode():
    v = compute_features_v2(CISPLATIN, mode="fp1024")
    assert v is not None
    assert v.shape == (1024,)


def test_features_v2_combined_mode():
    v = compute_features_v2(CISPLATIN, mode="rdkit12+fp1024")
    assert v is not None
    assert v.shape == (1036,), f"Combined mode should be 1036-d, got {v.shape}"


def test_features_v2_unknown_mode():
    try:
        compute_features_v2(CISPLATIN, mode="not_a_real_mode")
    except ValueError:
        return
    raise AssertionError("Unknown mode should raise ValueError")


def test_features_v2_all_modes_in_registry():
    assert "rdkit12" in FEATURES_V2_MODES
    assert "fp1024" in FEATURES_V2_MODES
    assert "rdkit12+fp1024" in FEATURES_V2_MODES
    assert len(FEATURES_V2_MODES) == 3


def test_empty_smiles_returns_none():
    fp = compute_morgan_fp_1024("")
    assert fp is None, "Empty SMILES should return None"


def test_invalid_smiles_returns_none():
    fp = compute_morgan_fp_1024("not_a_real_smiles@@@")
    assert fp is None, "Invalid SMILES should return None"


def test_empty_smiles_v2_combined():
    # rdkit12+fp1024 mode: rdkit12 is None so falls back to nan+median
    # imputation, fp1024 is None so falls back to zero vector.
    # Output should be 1036-d finite (after imputation in caller) or
    # 1036-d with NaN at start (this test asserts the shape contract).
    v = compute_features_v2("", mode="rdkit12+fp1024")
    assert v.shape == (1036,)


def test_cli_features_flag_accepted():
    """Verify --features flag is wired into the CLI."""
    proc = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "molmetal"
                / "scripts"
                / "extract_rdkit_2d_descriptors.py"
            ),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"--help exit code {proc.returncode}"
    assert "--features" in proc.stdout, "--features flag must be in --help output"
    assert "rdkit12+fp1024" in proc.stdout, "Combined mode description missing"


if __name__ == "__main__":
    test_morgan_fp_shape_and_dtype()
    test_rdk_fp_shape_and_dtype()
    test_morgan_fp_deterministic()
    test_features_v2_rdkit12_mode()
    test_features_v2_fp1024_mode()
    test_features_v2_combined_mode()
    test_features_v2_unknown_mode()
    test_features_v2_all_modes_in_registry()
    test_empty_smiles_returns_none()
    test_invalid_smiles_returns_none()
    test_empty_smiles_v2_combined()
    test_cli_features_flag_accepted()
    print("OK - all F1 (Morgan FP 1024) tests passed")
