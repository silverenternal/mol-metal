"""Tests for the AttentiveDMPNN pIC50 predictor wrapper.

Two contract tests:

1. ``test_predictor_loads`` — the checkpoint exists and the wrapper can
   load it without error.

2. ``test_predictor_shape`` — ``predict_pic50`` returns a Python float
   (not NaN, not None) for a SMILES the model is expected to handle.
   SMILES that fail to parse must return ``float("nan")``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest


# Skip the whole module if torch / rdkit aren't available.
torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")


# Project root = 3 hops up from molmetal/tests/test_*.py (PROJECT_ROOT/molmetal/tests/...).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT = _PROJECT_ROOT / "molmetal" / "checkpoints" / "dmpnn_attn_ru_pic50.pt"


pytestmark = pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason=(
        "AttentiveDMPNN pIC50 checkpoint missing. "
        "Run: python -m molmetal.molmetal_lam.scripts.calibrate_pic50_predictor"
    ),
)


def _import_predictor():
    """Lazy import so the skip marker fires cleanly when torch is missing."""
    from molmetal.molmetal_lam.sbdd_env.pic50_predictor import (
        AttentiveDMPNNPredictor,
        predict_pic50,
    )
    return AttentiveDMPNNPredictor, predict_pic50


def test_predictor_loads():
    """Checkpoint exists and the wrapper instantiates without raising."""
    AttentiveDMPNNPredictor, _ = _import_predictor()
    predictor = AttentiveDMPNNPredictor(ckpt=str(_CHECKPOINT))
    assert predictor is not None
    assert predictor.model is not None
    # The checkpoint should carry training metadata (n_train, etc.).
    meta = getattr(predictor, "meta", None)
    assert isinstance(meta, dict), "predictor.meta missing or wrong type"
    assert meta.get("n_train", 0) > 0, f"unexpected n_train in meta: {meta}"


def test_predictor_shape():
    """predict_pic50 returns a Python float for valid SMILES, NaN for invalid."""
    AttentiveDMPNNPredictor, predict_pic50 = _import_predictor()

    # 1. valid SMILES — must return a finite float.
    valid = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    out = predict_pic50(valid)
    assert isinstance(out, float), f"predict_pic50 returned {type(out)}"
    assert out == out, "predict_pic50 returned NaN for aspirin"
    assert math.isfinite(out), f"predict_pic50 returned non-finite {out}"
    # Drug-like pIC50s cluster in [3, 9]; loosen to a generous sanity range.
    assert 0.0 <= out <= 12.0, f"predict_pic50 out of plausible range: {out}"

    # 2. invalid SMILES — must return NaN, not raise.
    out_bad = predict_pic50("not-a-smiles!!")
    assert isinstance(out_bad, float), f"predict_pic50 returned {type(out_bad)}"
    assert math.isnan(out_bad), f"expected NaN for invalid SMILES, got {out_bad}"

    # 3. empty SMILES — must also return NaN.
    out_empty = predict_pic50("")
    assert isinstance(out_empty, float)
    assert math.isnan(out_empty)
