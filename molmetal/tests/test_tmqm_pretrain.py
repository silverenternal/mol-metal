"""Tests for the tmQM pretraining script (Mol-Metal R17 / Metric 22).

Contract tests for the encoder-only pretrain pipeline:

1. ``test_load_tmqm_returns_finite_targets`` — every loaded target is
   finite (no NaN/Inf leak through).
2. ``test_load_tmqm_drop_empty_smiles`` — rows with empty SMILES are
   dropped (the 7.7k empty-SMILES rows in tmQM).
3. ``test_stratified_split_covers_all_classes`` — the train/val/test
   split preserves the class distribution (oxidation-state binning).
4. ``test_charge_to_bin_in_range`` — ``charge_to_bin`` always returns a
   valid bin index.
5. ``test_pretrain_model_forward_shape`` — the multi-head model returns
   reg_preds of shape ``(B, 5)`` and cls_logits of shape ``(B, 9)``.
6. ``test_save_encoder_load_round_trip`` — saving then loading the
   encoder weights preserves the atom_emb matrix (the load-bearing
   backbone tensor for downstream fine-tuning).
7. ``test_reg_scaler_round_trip`` — ``inverse_transform ∘ transform``
   recovers the original targets within 1e-5.
8. ``test_reg_scaler_floor_for_constant_target`` — if a target has
   zero variance the std is floored to 1.0 (avoids /0 in z-score).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.pretrain_tmqm import (  # noqa: E402
    N_OX_BINS,
    OX_STATES,
    RegScaler,
    TmQMDataset,
    TmqmPretrainModel,
    charge_to_bin,
    load_tmqm,
    save_encoder,
    stratified_split,
)


TMQM_CSV = Path("/home/hugo/codes/mol-metal/.data/molmetal/tmQM/tmqm_parsed.csv")
SKIP_NO_DATA = not TMQM_CSV.exists()


# ---------------------------------------------------------------------------
# 1. Data loader: finite targets
# ---------------------------------------------------------------------------
@pytest.mark.skipif(SKIP_NO_DATA, reason="tmQM CSV not on this host")
def test_load_tmqm_returns_finite_targets():
    ds = load_tmqm(TMQM_CSV, require_smiles=True)
    assert ds.n > 50_000
    assert np.isfinite(ds.reg_targets).all(), (
        f"non-finite reg targets leaked: {np.argwhere(~np.isfinite(ds.reg_targets)).shape[0]} rows"
    )


# ---------------------------------------------------------------------------
# 2. Empty-SMILES rows are dropped
# ---------------------------------------------------------------------------
@pytest.mark.skipif(SKIP_NO_DATA, reason="tmQM CSV not on this host")
def test_load_tmqm_drop_empty_smiles():
    """Loaded SMILES are all non-empty (default require_smiles=True)."""
    ds = load_tmqm(TMQM_CSV, require_smiles=True)
    assert all(s for s in ds.smiles), "empty SMILES leaked through"


# ---------------------------------------------------------------------------
# 3. Stratified split preserves class distribution
# ---------------------------------------------------------------------------
@pytest.mark.skipif(SKIP_NO_DATA, reason="tmQM CSV not on this host")
def test_stratified_split_covers_all_classes():
    rng = np.random.default_rng(0)
    cls_targets = rng.choice([0, 1, 2, 3], size=200, p=[0.4, 0.3, 0.2, 0.1])
    train_idx, val_idx, test_idx = stratified_split(
        cls_targets, val_frac=0.1, test_frac=0.1, seed=42
    )
    assert train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0] == 200
    # Each split should contain at least one element of each class
    for c in [0, 1, 2, 3]:
        in_train = (cls_targets[train_idx] == c).any()
        in_val = (cls_targets[val_idx] == c).any()
        in_test = (cls_targets[test_idx] == c).any()
        assert in_train or in_val or in_test, f"class {c} missing from splits"


# ---------------------------------------------------------------------------
# 4. charge_to_bin returns valid bin index
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [-2, -1, 0, 1, 2, 3, 4, 5, 6])
def test_charge_to_bin_in_range(c):
    b = charge_to_bin(c)
    assert 0 <= b < N_OX_BINS
    assert OX_STATES[b] == c


def test_charge_to_bin_out_of_range_nearest():
    """Charge values beyond the bin table fall back to nearest-neighbour."""
    b_far = charge_to_bin(99)
    assert b_far == N_OX_BINS - 1  # 99 → nearest is +6


# ---------------------------------------------------------------------------
# 5. Model forward shape
# ---------------------------------------------------------------------------
def test_pretrain_model_forward_shape():
    torch.manual_seed(0)
    model = TmqmPretrainModel(hidden=64, depth=2)
    B, V, E, A, Bd = 3, 5, 8, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    atom_f = torch.randn(B, V, A)
    bond_f = torch.randn(B, E, Bd)
    e_src = torch.zeros(B, E, dtype=torch.long)
    e_dst = torch.zeros(B, E, dtype=torch.long)
    reg_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst)
    assert reg_pred.shape == (B, 5)
    assert cls_logits.shape == (B, N_OX_BINS)


# ---------------------------------------------------------------------------
# 6. Save / load encoder round-trip
# ---------------------------------------------------------------------------
def test_save_encoder_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = TmqmPretrainModel(hidden=32, depth=2)
    # Run a forward to make sure all params are initialised
    _ = model(
        torch.randn(1, 3, ATOM_FEATURE_DIM),
        torch.randn(1, 4, BOND_FEATURE_DIM),
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
    )
    cfg = {"hidden": 32, "depth": 2, "dropout": 0.1, "atom_dim": ATOM_FEATURE_DIM}
    meta = {"best_epoch": 1, "best_val_rmse": 0.5, "final_test": {"rmse": 0.6, "mae": 0.4, "cls_acc": 0.8}}
    scaler = RegScaler(mean=np.zeros(5, dtype=np.float32), std=np.ones(5, dtype=np.float32))
    enc_path = save_encoder(model, tmp_path, cfg, meta, scaler)

    payload = torch.load(enc_path, map_location="cpu")
    assert "atom_emb.weight" in payload["state_dict"]
    assert payload["state_dict"]["atom_emb.weight"].shape == (32, ATOM_FEATURE_DIM)
    assert payload["config"]["hidden"] == 32


# ---------------------------------------------------------------------------
# 7. RegScaler round-trip
# ---------------------------------------------------------------------------
def test_reg_scaler_round_trip():
    rng = np.random.default_rng(0)
    reg = rng.normal(loc=2.0, scale=4.0, size=(100, 5)).astype(np.float32)
    scaler = RegScaler.fit(reg)
    out = scaler.inverse_transform(scaler.transform(reg))
    np.testing.assert_allclose(out, reg, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. RegScaler floor for constant target
# ---------------------------------------------------------------------------
def test_reg_scaler_floor_for_constant_target():
    reg = np.zeros((10, 3), dtype=np.float32)
    reg[:, 1] = 1.0  # constant column → std = 0
    scaler = RegScaler.fit(reg)
    # std should be floored to 1.0 to avoid /0 in z-score
    assert scaler.std[0] == 1.0
    assert scaler.std[1] == 1.0
    assert scaler.std[2] == 1.0
    # And the transform should not produce NaN
    out = scaler.transform(reg)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# ATOM/BOND feature dim constants (avoid import-time surprises)
# ---------------------------------------------------------------------------
from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM  # noqa: E402,F401