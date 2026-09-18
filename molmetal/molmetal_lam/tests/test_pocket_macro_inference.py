"""Tests for :mod:`molmetal_lam.lam_chem.pocket_macro_inference`.

WF-Deflex Follow-up Phase 2 — covers the ``PocketMacroInference`` adapter
that wraps the trained ``molmetal/models/pocket_macro_skeleton.pt``
checkpoint.

Run from project root::

    uv run pytest molmetal/molmetal_lam/tests/test_pocket_macro_inference.py -x --tb=short -q
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pytest
import torch

from molmetal_lam.lam_chem.pocket_macro_inference import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METADATA_PATH,
    HIDDEN_DIM,
    PocketMacroInference,
)
from molmetal_lam.lam_chem.pocket_macro_skeleton import ScaffoldClass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inference() -> PocketMacroInference:
    """A single ``PocketMacroInference`` instance shared across the module."""
    return PocketMacroInference()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_checkpoint(inference: PocketMacroInference) -> None:
    """The trained ``.pt`` state-dict loads with strict=True (no shape drift)."""
    # The constructor calls load() automatically when both files exist.
    assert inference.is_available(), (
        f"Checkpoint missing: expected {DEFAULT_CHECKPOINT_PATH} and "
        f"{DEFAULT_METADATA_PATH} to exist on disk."
    )
    assert inference._loaded is True, (
        "load() did not succeed even though is_available() reported True."
    )
    # The internal v1 mirror must have 5,580 trainable parameters
    # (matches the .pt.json metadata "n_params": 5580).
    n_params = sum(p.numel() for p in inference._model.parameters())
    assert n_params == 5580, (
        f"Parameter count drift: expected 5580, got {n_params}. "
        "The checkpoint may have been overwritten with a different arch."
    )


def test_is_available(inference: PocketMacroInference) -> None:
    """``is_available()`` is True after a successful load."""
    assert inference.is_available() is True


def test_get_embedding_dim(inference: PocketMacroInference) -> None:
    """``get_embedding()`` returns a (32,) float32 vector for known targets."""
    for target in ["CA2", "MMP2", "PKA", "CYP3A4"]:
        emb = inference.get_embedding(target)
        assert isinstance(emb, np.ndarray), (
            f"get_embedding({target!r}) must return np.ndarray, "
            f"got {type(emb).__name__}"
        )
        assert emb.shape == (HIDDEN_DIM,), (
            f"get_embedding({target!r}) shape must be ({HIDDEN_DIM},), "
            f"got {emb.shape}"
        )
        assert emb.dtype == np.float32, (
            f"get_embedding({target!r}) dtype must be float32, "
            f"got {emb.dtype}"
        )
        assert np.all(np.isfinite(emb)), (
            f"get_embedding({target!r}) contains NaN/Inf"
        )


def test_predict_pka(inference: PocketMacroInference) -> None:
    """PKA target predicts MG_OCTA_KINASE with high confidence."""
    predicted, conf = inference.predict_scaffold_class_with_confidence("PKA")
    assert predicted == "MG_OCTA_KINASE", (
        f"PKA expected MG_OCTA_KINASE, got {predicted} (conf={conf:.3f})"
    )
    assert conf >= 0.80, (
        f"PKA confidence should be >= 0.80 (train acc 1.0 on MG_OCTA_KINASE), "
        f"got {conf:.3f}"
    )


def test_predict_mmp2(inference: PocketMacroInference) -> None:
    """MMP2 target predicts ZN_TETRA_HHE with high confidence."""
    predicted, conf = inference.predict_scaffold_class_with_confidence("MMP2")
    assert predicted == "ZN_TETRA_HHE", (
        f"MMP2 expected ZN_TETRA_HHE, got {predicted} (conf={conf:.3f})"
    )
    assert conf >= 0.80, (
        f"MMP2 confidence should be >= 0.80 (train acc 1.0 on ZN_TETRA_HHE), "
        f"got {conf:.3f}"
    )


def test_predict_cyp3a4(inference: PocketMacroInference) -> None:
    """CYP3A4 target predicts FE_HEME_CYS (train acc 1.0).

    Honest framing: the FE_HEME_CYS class has only 8 training PDBs
    and a small mean-pool input (5 residues), so the model's softmax
    is less peaked than the larger classes — we expect confidence
    >= 0.50 (above the floor) rather than >= 0.80.
    """
    predicted, conf = inference.predict_scaffold_class_with_confidence(
        "CYP3A4"
    )
    assert predicted == "FE_HEME_CYS", (
        f"CYP3A4 expected FE_HEME_CYS, got {predicted} (conf={conf:.3f})"
    )
    assert conf >= 0.50, (
        f"CYP3A4 confidence should be >= 0.50 (above floor), got {conf:.3f}"
    )


def test_predict_ca2_honest_collapse(inference: PocketMacroInference) -> None:
    """CA2 → ZN_TETRA_HHH is the documented Phase 2 §4.2 collapse (0/8 train).

    Honest framing: the v1 checkpoint predicts ZN_TETRA_HHE for CA2 with
    softmax mass ~0.68 (above the 0.50 floor).  The model has learned
    the *amino-acid pattern* (3×His triad) but conflated it with the
    MMP2 ZN_TETRA_HHE class because the v1 mean-pool aggregator cannot
    distinguish "His at anchor-position 0,1,2" from "His at all three
    positions with one Glu present" without the v2 anchor_position
    one-hot signal.

    We assert the failure mode HONESTLY:
    * Raw argmax is NOT ZN_TETRA_HHH (the model fails to learn it).
    * Source is "model" (weights loaded; not a no-op).
    * The confidence is in the documented collapse band (0.50-0.85).
    * Ground-truth class (from the metadata mapping) IS ZN_TETRA_HHH.
    """
    # Raw argmax (no floor) should NOT be ZN_TETRA_HHH.
    raw_predicted, probs, source = inference.predict_logits("CA2")
    assert raw_predicted != "ZN_TETRA_HHH", (
        "Documented Phase 2 §4.2 collapse: CA2 should NOT predict "
        "ZN_TETRA_HHH (train acc 0/8). If this assertion fires, the "
        "model has been retrained and the expected collapse is gone."
    )
    assert raw_predicted == "ZN_TETRA_HHE", (
        f"Expected CA2 collapse prediction to be ZN_TETRA_HHE "
        f"(mean-pool conflation), got {raw_predicted!r}"
    )
    assert source == "model", f"Expected source='model', got {source!r}"

    # Confidence is in the documented collapse band: above the 0.50 floor
    # but below the 1.0 high-confidence ceiling (so floored prediction
    # stays at the model's raw argmax, NOT UNKNOWN).
    floored, conf = inference.predict_scaffold_class_with_confidence("CA2")
    assert floored == "ZN_TETRA_HHE", (
        f"CA2 floored prediction should stay at ZN_TETRA_HHE (collapse "
        f"is confident, just wrong), got {floored!r} (conf={conf:.3f}). "
        f"Lower DEFAULT_CONFIDENCE_FLOOR if this fires."
    )
    assert 0.50 <= conf < 1.0, (
        f"CA2 confidence should be in the [0.50, 1.0) collapse band, "
        f"got {conf:.3f}"
    )

    # Ground truth (from the deterministic mapping) is ZN_TETRA_HHH.
    from molmetal_lam.lam_chem.pocket_macro_skeleton import (
        scaffold_class_from_target_name,
    )
    gt = scaffold_class_from_target_name("CA2")
    assert gt.name == "ZN_TETRA_HHH", (
        f"Ground-truth scaffold class for CA2 should be ZN_TETRA_HHH, "
        f"got {gt.name}"
    )


def test_handles_missing_checkpoint(tmp_path) -> None:
    """Returns graceful UNKNOWN when the .pt file is missing."""
    bogus_ckpt = tmp_path / "no_such_checkpoint.pt"
    bogus_meta = tmp_path / "no_such_checkpoint.pt.json"
    inf = PocketMacroInference(
        checkpoint_path=str(bogus_ckpt),
        metadata_path=str(bogus_meta),
    )
    assert inf.is_available() is False, (
        "is_available() should be False when files are missing"
    )
    assert inf._loaded is False, (
        "load() should not have been called automatically when files "
        "are missing."
    )
    # Predictions should return UNKNOWN without raising.
    predicted = inf.predict_scaffold_class("CA2")
    assert predicted == "UNKNOWN", (
        f"Missing-checkpoint fallback should return UNKNOWN, got {predicted!r}"
    )
    # get_embedding should return zeros, not raise.
    emb = inf.get_embedding("PKA")
    assert emb.shape == (HIDDEN_DIM,)
    assert np.allclose(emb, 0.0), (
        "Missing-checkpoint get_embedding should return zeros"
    )


def test_unknown_target_returns_unknown(
    inference: PocketMacroInference,
) -> None:
    """An unknown target name returns UNKNOWN gracefully (fixture miss)."""
    predicted, conf = inference.predict_scaffold_class_with_confidence(
        "NO_SUCH_TARGET_XYZ"
    )
    assert predicted == "UNKNOWN", (
        f"Unknown target should return UNKNOWN, got {predicted!r}"
    )
    # Source reports "fallback" because the fixture lookup failed (no
    # means to feed residues to the model).  This is by design: the
    # source string tracks *what path the answer came from*, not
    # whether the model is loaded.
    _raw, _probs, source = inference.predict_logits("NO_SUCH_TARGET_XYZ")
    assert source == "fallback", (
        f"Unknown target source should be 'fallback' (no fixture), "
        f"got {source!r}"
    )


def test_metadata_sidecar_loaded(inference: PocketMacroInference) -> None:
    """The ``.pt.json`` metadata is loaded and exposes the per-class accuracy."""
    meta = inference.get_metadata()
    assert "results" in meta, (
        f"Metadata should contain 'results' block, got keys {list(meta.keys())}"
    )
    per_class = inference.scaffold_class_distribution()
    assert "ZN_TETRA_HHH" in per_class, (
        f"Per-class accuracy should include ZN_TETRA_HHH, got "
        f"{list(per_class.keys())}"
    )
    # The honest collapse: ZN_TETRA_HHH has accuracy 0/8 = 0.0.
    zhh = per_class["ZN_TETRA_HHH"]
    assert zhh["accuracy"] == 0.0, (
        f"ZN_TETRA_HHH train accuracy should be 0.0 (documented collapse), "
        f"got {zhh['accuracy']}"
    )
    # MG_OCTA_KINASE has 16/16 = 1.0.
    mg_octa = per_class["MG_OCTA_KINASE"]
    assert mg_octa["accuracy"] == 1.0, (
        f"MG_OCTA_KINASE train accuracy should be 1.0, got {mg_octa['accuracy']}"
    )
