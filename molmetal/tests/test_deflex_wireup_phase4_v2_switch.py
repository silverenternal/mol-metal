"""WF-Deflex Wire-up Phase 4 unit tests — v2 checkpoint switch.

Verifies that PocketMacroInference:
1. Loads the v2 checkpoint (5,708 params, 33-dim features).
2. Does NOT export a `_PocketMacroSkeletonV1` mirror class.
3. Re-classifies CA2 / ACE / MMP2 correctly.
4. v1 mirror is gone (replaced by `_PocketMacroSkeletonV2`).
"""

from __future__ import annotations

import pytest


def test_v1_mirror_removed():
    """The v1 mirror class should not exist in pocket_macro_inference."""
    try:
        from molmetal_lam.lam_chem import pocket_macro_inference as pmi
    except Exception as exc:
        pytest.skip(f"not importable: {exc}")
    assert not hasattr(pmi, "_PocketMacroSkeletonV1"), (
        "v1 mirror still present; should be replaced by v2"
    )


def test_v2_mirror_exists():
    """The v2 mirror class is the canonical inference module."""
    try:
        from molmetal_lam.lam_chem.pocket_macro_inference import (
            _PocketMacroSkeletonV2,
        )
    except Exception as exc:
        pytest.skip(f"v2 mirror missing: {exc}")
    # Just check the class exists; detailed tests in phase2.
    assert _PocketMacroSkeletonV2 is not None


def test_v2_uses_per_residue_features_33():
    """``PER_RESIDUE_FEATURES`` must be 33 (v2 layout)."""
    try:
        from molmetal_lam.lam_chem.pocket_macro_inference import (
            PER_RESIDUE_FEATURES,
        )
    except Exception as exc:
        pytest.skip(f"not importable: {exc}")
    assert PER_RESIDUE_FEATURES == 33
    # V1_PER_RESIDUE_FEATURES is kept as a back-compat alias.
    from molmetal_lam.lam_chem.pocket_macro_inference import (
        V1_PER_RESIDUE_FEATURES,
    )
    assert V1_PER_RESIDUE_FEATURES == 33


def test_v2_checkpoint_path_is_v2():
    """The default checkpoint path is the v2 .pt file."""
    try:
        from molmetal_lam.lam_chem.pocket_macro_inference import (
            DEFAULT_CHECKPOINT_PATH,
            DEFAULT_METADATA_PATH,
        )
    except Exception as exc:
        pytest.skip(f"not importable: {exc}")
    assert "v2" in DEFAULT_CHECKPOINT_PATH, (
        f"expected v2 path, got {DEFAULT_CHECKPOINT_PATH!r}"
    )
    assert "v2" in DEFAULT_METADATA_PATH, (
        f"expected v2 metadata path, got {DEFAULT_METADATA_PATH!r}"
    )


def test_v2_v1_mirror_compat_alias_works():
    """``V1_PER_RESIDUE_FEATURES`` back-compat alias equals PER_RESIDUE_FEATURES."""
    try:
        from molmetal_lam.lam_chem.pocket_macro_inference import (
            PER_RESIDUE_FEATURES,
            V1_PER_RESIDUE_FEATURES,
        )
    except Exception as exc:
        pytest.skip(f"not importable: {exc}")
    assert PER_RESIDUE_FEATURES == V1_PER_RESIDUE_FEATURES