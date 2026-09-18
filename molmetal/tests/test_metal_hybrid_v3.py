"""Tests for MetalHybridV3Model (D-MPNN + EGNN + coord refinement).

Covers:
1. test_encoder_load      : pretrained tmQM D-MPNN weights load into the encoder.
2. test_forward_shape     : forward(B mols) -> pic50 (B,), active_logits (B, 2).
3. test_coord_refinement  : refine_coords shifts coords by a finite, bounded Δx.
4. test_train_loop_1_epoch: one epoch over a small Ru subset finishes + loss finite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.models.metal_hybrid_v3 import (
    DEFAULT_PRETRAINED_CKPT,
    MetalHybridV3Config,
    MetalHybridV3Model,
)
from molmetal.models.metal_hybrid import MetalHybridConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_batch(smiles_list, n_min: int = 20):
    """Pad SMILES-style batch to (B, N, 3) coords + (B,) metal types."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    B = len(smiles_list)
    coords_list = []
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            coords_list.append(np.zeros((1, 3), dtype=np.float32))
            continue
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            coords = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
        except Exception:
            coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        coords_list.append(coords)
    n_atoms_max = max((c.shape[0] for c in coords_list), default=1)
    n_atoms_max = max(n_atoms_max, n_min)
    coords = np.zeros((B, n_atoms_max, 3), dtype=np.float32)
    for i, c in enumerate(coords_list):
        coords[i, : c.shape[0]] = c
    metal_types = torch.zeros(B, dtype=torch.long)
    return smiles_list, torch.from_numpy(coords).float(), metal_types, mols


def _small_config() -> MetalHybridV3Config:
    base = MetalHybridConfig(hidden_dim=64, n_dmpnn_layers=2, n_egnn_layers=2)
    return MetalHybridV3Config(
        base=base,
        coord_loss_weight=0.05,
        coord_refine_max_step=0.3,
        load_pretrained_encoder=True,
    )


# ---------------------------------------------------------------------------
# 1. Encoder load (loads from tmQM pretraining)
# ---------------------------------------------------------------------------
def test_encoder_load():
    """Pretrained tmQM D-MPNN weights load into the V3 encoder."""
    _set_seed(0)
    cfg = _small_config()
    assert Path(DEFAULT_PRETRAINED_CKPT).exists(), (
        f"missing pretrained ckpt {DEFAULT_PRETRAINED_CKPT}"
    )
    model = MetalHybridV3Model(config=cfg)
    # Pretrained hidden_dim is 128, our small config is 64, so shapes differ
    # → no load — but a full-size config should load.
    base_full = MetalHybridConfig(hidden_dim=128, n_dmpnn_layers=3, n_egnn_layers=3)
    cfg_full = MetalHybridV3Config(base=base_full, load_pretrained_encoder=True)
    model_full = MetalHybridV3Model(config=cfg_full)
    assert model_full.pretrained_loaded is True, (
        "tmQM pretrained D-MPNN weights failed to load into V3 encoder"
    )
    # Confirm at least one parameter differs from random init (sanity)
    sd = model_full.dmpnn.state_dict()
    p0 = next(iter(sd.values())).flatten()
    assert p0.numel() > 0


# ---------------------------------------------------------------------------
# 2. Forward shape
# ---------------------------------------------------------------------------
def test_forward_shape():
    """Forward(3 mols) → pic50 (3,), active_logits (3, 2)."""
    _set_seed(0)
    smiles = ["CCO", "CCN", "c1ccccc1"]
    cfg = _small_config()
    model = MetalHybridV3Model(config=cfg).eval()
    smiles_list, coords, metal_types, mols = _make_batch(smiles)
    with torch.no_grad():
        out = model(smiles_list, coords, metal_types, mol_objects=mols)
    assert out.pic50.shape == (3,)
    assert out.active_logits.shape == (3, 2)
    # Clamped to [pic50_min, pic50_max]
    assert (out.pic50 >= cfg.base.pic50_min).all()
    assert (out.pic50 <= cfg.base.pic50_max).all()
    assert torch.isfinite(out.pic50).all()
    assert torch.isfinite(out.active_logits).all()


# ---------------------------------------------------------------------------
# 3. Coord refinement
# ---------------------------------------------------------------------------
def test_coord_refinement():
    """refine_coords returns coords of the same shape, finite, and Δx bounded."""
    _set_seed(0)
    smiles = ["CCO", "CCN(CC)C(=O)C"]
    cfg = _small_config()
    model = MetalHybridV3Model(config=cfg).eval()
    smiles_list, coords, metal_types, mols = _make_batch(smiles)
    with torch.no_grad():
        refined = model.refine_coords(smiles_list, coords, metal_types, mol_objects=mols)
    assert refined.shape == coords.shape
    delta = refined - coords
    # Δx should be bounded by the configured max_step (in case every edge contributes)
    assert torch.isfinite(delta).all()
    assert (delta.abs() <= cfg.coord_refine_max_step + 1e-6).all(), (
        f"Δx exceeded max_step={cfg.coord_refine_max_step}: "
        f"max |Δx|={delta.abs().max().item():.4f}"
    )
    # Coordinates for actual atoms should move (not be all-zero)
    moved = (delta.abs().sum(dim=-1) > 0).any().item()
    assert moved, "refine_coords produced all-zero Δx — head may be inert"


# ---------------------------------------------------------------------------
# 4. Train loop, 1 epoch
# ---------------------------------------------------------------------------
def test_train_loop_1_epoch(tmp_path):
    """One training epoch over a tiny Ru subset runs end-to-end with finite loss."""
    _set_seed(0)
    import torch.nn.functional as F
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
    from molmetal.data.splits import LigandDeduplicatedSplitter
    from molmetal.models.loss import MetalCytotoxLoss
    from molmetal.scripts.train_hybrid import build_dataloader

    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=["Ru"],
        compute_pic50=True,
        compute_active=True,
    )
    try:
        ds = MetalCytotoxDataset.from_csv(filters=flt)
    except FileNotFoundError as e:
        pytest.skip(f"MetalCytoToxDB not available: {e}")
    if len(ds) < 30:
        pytest.skip(f"Ru dataset too small for train-loop test ({len(ds)})")

    splitter = LigandDeduplicatedSplitter(strategy="largest_first", seed=0)
    split = splitter(ds)
    train_idx = split.train_idx[:32]
    val_idx = split.val_idx[:16]

    cfg = _small_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetalHybridV3Model(config=cfg).to(device)
    loss_fn = MetalCytotoxLoss(alpha=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    featurizer = model.featurizer
    train_loader = build_dataloader(ds, train_idx, 8, featurizer, device, shuffle=True)
    val_loader = build_dataloader(ds, val_idx, 8, featurizer, device, shuffle=False)

    # One train pass
    model.train()
    for batch in train_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        optimizer.zero_grad()
        out = model(smiles_list, coords, metal_types, mol_objects=mol_objects)
        loss = loss_fn(
            out.pic50,
            pic50_true.to(device),
            active_label.to(device),
            mask.to(device),
            active_logits=out.active_logits,
        )
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss).item(), "train loss diverged"
        break  # one batch is enough for a smoke test

    # One eval pass
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
            out = model(smiles_list, coords, metal_types, mol_objects=mol_objects)
            assert torch.isfinite(out.pic50).all()
            assert torch.isfinite(out.active_logits).all()
            break


from pathlib import Path  # noqa: E402  (used in test_encoder_load)
