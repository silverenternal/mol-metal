"""Unit tests for counterion feature extraction + ablation configs.

Run with::

    source .venv/bin/activate
    pytest molmetal/tests/test_counterion_utils.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.data.counterion_utils import (
    CONFIGS,
    DEFAULT_FEATURES,
    FEATURE_NAMES,
    apply_config,
    extract_counterion_features,
    merge_counterion_features,
)


# ---------------------------------------------------------------------------
# 1. test_extract_features_valid: '[Cl-].[Na+]' -> dict with charge, MW
# ---------------------------------------------------------------------------
def test_extract_features_valid():
    """A multi-component counter-ion SMILES must yield sensible numerics.

    RDKit requires bracket-atom notation for charged species, so we use
    ``[Cl-].[Na+]`` (equivalent to ``Cl-.Na+`` in the loose convention used
    by some text representations of ion pairs).
    """
    feats = extract_counterion_features("[Cl-].[Na+]")
    # All five named keys present
    for k in FEATURE_NAMES:
        assert k in feats, f"missing key {k!r} in features"
    # Net formal charge of the *primary* component (split('.')[0] = '[Cl-]').
    # Primary is Cl- with charge -1.
    assert feats["charge"] == pytest.approx(-1.0)
    # MW of Cl- (chlorine = 35.45 g/mol)
    assert feats["mw"] == pytest.approx(35.45, rel=1e-2)
    # 1 heavy atom (Cl)
    assert feats["num_atoms"] == pytest.approx(1.0)
    # No carbon → inorganic
    assert feats["is_inorganic"] == pytest.approx(1.0)
    # No metal symbol in primary fragment (Cl is a halogen, not a metal)
    assert feats["has_metal"] == pytest.approx(0.0)
    # logP proxy: very negative for a small ion
    assert feats["logp"] <= 0.5


# ---------------------------------------------------------------------------
# 2. test_extract_features_invalid: 'XX' -> returns defaults
# ---------------------------------------------------------------------------
def test_extract_features_invalid():
    """Unparseable SMILES must yield all-zero defaults, not crash."""
    feats = extract_counterion_features("XX")
    assert feats == DEFAULT_FEATURES
    assert all(feats[k] == 0.0 for k in FEATURE_NAMES)

    # Also empty string
    feats_empty = extract_counterion_features("")
    assert feats_empty == DEFAULT_FEATURES
    assert all(feats_empty[k] == 0.0 for k in FEATURE_NAMES)

    # Also None
    feats_none = extract_counterion_features(None)  # type: ignore[arg-type]
    assert feats_none == DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# 3. test_merge_counterion_features: row + counterion -> combined dict
# ---------------------------------------------------------------------------
def test_merge_counterion_features():
    """Merging must keep original keys and add 5 prefixed counterion_* keys."""
    row = {"smiles": "[Ru]Cl", "metal": "Ru", "Counterion": "[Cl-]"}
    out = merge_counterion_features(row)
    # Original keys preserved
    assert out["smiles"] == "[Ru]Cl"
    assert out["metal"] == "Ru"
    assert out["Counterion"] == "[Cl-]"
    # 5 new counterion_* keys added
    for k in FEATURE_NAMES:
        assert f"counterion_{k}" in out, f"missing counterion_{k}"
    # counterion_smiles preserved
    assert out["counterion_smiles"] == "[Cl-]"
    # Cl- numeric sanity
    assert out["counterion_charge"] == pytest.approx(-1.0)
    assert out["counterion_num_atoms"] == pytest.approx(1.0)
    assert out["counterion_is_inorganic"] == pytest.approx(1.0)

    # Row without Counterion field → all-zero features, no crash
    row_empty = {"smiles": "CCO"}
    out_empty = merge_counterion_features(row_empty)
    for k in FEATURE_NAMES:
        assert out_empty[f"counterion_{k}"] == 0.0
    assert out_empty["counterion_smiles"] == ""

    # NaN counterion → all-zero features
    import math
    row_nan = {"smiles": "CCO", "Counterion": float("nan")}
    out_nan = merge_counterion_features(row_nan)
    for k in FEATURE_NAMES:
        assert out_nan[f"counterion_{k}"] == 0.0


# ---------------------------------------------------------------------------
# 4. test_configs_three: all 3 configs runnable
# ---------------------------------------------------------------------------
def test_configs_three():
    """Each of the 3 ablation configs must produce a well-shaped matrix."""
    smiles = [
        "[Ru](Cl)(Cl)(Cl)Cl",
        "O=Cc1ccccc1",
        "c1ccccc1",
    ]
    counterions = ["[Cl-].[Na+]", "F[B-](F)(F)F", ""]

    # no_counterion: feature dim == morgan_nbits
    X_no = apply_config(smiles, counterions, "no_counterion", morgan_nbits=2048)
    assert X_no.shape == (3, 2048), X_no.shape
    assert X_no.dtype == np.uint8

    # counterion_features: feature dim == morgan_nbits + len(FEATURE_NAMES)
    X_feat = apply_config(smiles, counterions, "counterion_features", morgan_nbits=2048)
    assert X_feat.shape == (3, 2048 + len(FEATURE_NAMES)), X_feat.shape
    assert X_feat.dtype == np.float32
    # Counter-ion features for the 3rd row (empty counter-ion) should all be 0
    assert np.all(X_feat[2, 2048:] == 0.0)

    # counterion_concat_smiles: feature dim == morgan_nbits
    X_concat = apply_config(smiles, counterions, "counterion_concat_smiles", morgan_nbits=2048)
    assert X_concat.shape == (3, 2048), X_concat.shape
    assert X_concat.dtype == np.uint8

    # CONFIGS tuple has exactly the 3 names
    assert set(CONFIGS) == {
        "no_counterion",
        "counterion_features",
        "counterion_concat_smiles",
    }

    # Unknown config raises
    with pytest.raises(ValueError, match="Unknown counterion config"):
        apply_config(smiles, counterions, "bogus_config", morgan_nbits=512)
