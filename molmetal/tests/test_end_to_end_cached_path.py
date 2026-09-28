"""Pytest trace-verification for the pre-featurized + cached-forward path.

Companion to ``molmetal/scripts/verify_end_to_end_cached_path.py``.
The script does the actual MEASURED end-to-end smoke on a real GPU +
45 GB ``all_pretrain.pt``; this test file packages the same three
contracts as pytest assertions so they can be wired into CI:

  (i)   Pre-featurized loader is RDKit-free (no MolFromSmiles /
        MolToSmiles reached during data slicing).
  (ii)  Cached forward + backward produces finite loss and gradient
        flow into the shared encoder AND every per-metal head.
  (iii) Running the same ``run_steps`` helper used by the smoke
        completes successfully without raising.

A small synthetic ``all_pretrain.pt`` is written to a temp file rather
than loading the 45 GB production file.  The script-level smoke is the
authoritative MEASURED run on real data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.v4_multitask_dmpnn import (
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
    V4MultiTaskLoss,
)
from molmetal.scripts import verify_end_to_end_cached_path as vec


# ---------------------------------------------------------------------------
# Helpers — synthetic mini-pre_featurized pack
# ---------------------------------------------------------------------------
def _make_synthetic_pack(
    n_rows: int = 32,
    n_atoms: int = 6,
    n_edges: int = 10,
    atom_feat_dim: int = 39,
    edge_feat_dim: int = 6,
    n_metals: int = 5,
) -> dict:
    """Tiny all_pretrain.pt schema with metal_idx in [0, n_metals)."""
    rng = np.random.default_rng(0)
    af = torch.randn(n_rows, n_atoms, atom_feat_dim, dtype=torch.float32)
    ei = torch.zeros(n_rows, 2, n_edges, dtype=torch.int64)
    ea = torch.randn(n_rows, n_edges, edge_feat_dim, dtype=torch.float32)
    mask = torch.ones(n_rows, n_atoms, dtype=torch.bool)
    metal_idx = torch.from_numpy(
        rng.integers(0, n_metals, size=n_rows)
    ).long()
    y = torch.randn(n_rows, dtype=torch.float32)
    active = torch.from_numpy(rng.integers(0, 2, size=n_rows)).float()
    domain = torch.zeros(n_rows, dtype=torch.int64)
    smiles = ["C"] * n_rows
    return {
        "atom_features": af,
        "edge_index": ei,
        "edge_attr": ea,
        "atom_mask": mask,
        "metal_idx": metal_idx,
        "y": y,
        "active": active,
        "domain": domain,
        "smiles": smiles,
    }


@pytest.fixture()
def synthetic_pack() -> dict:
    return _make_synthetic_pack(n_rows=32, n_atoms=6, n_edges=10)


# ---------------------------------------------------------------------------
# (i) loader is RDKit-free
# ---------------------------------------------------------------------------
def test_load_pre_featurized_skips_rdkit(tmp_path: Path) -> None:
    """``load_pre_featurized`` must not call any RDKit symbol — it is a
    pure tensor slice over an mmap'd ``all_pretrain.pt`` file."""
    pack = _make_synthetic_pack(n_rows=16)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)

    with patch("rdkit.Chem.MolFromSmiles") as mock_mol, \
         patch("rdkit.Chem.MolToSmiles") as mock_to:
        loaded = vec.load_pre_featurized(p, max_rows=8, canonical_only=True)
        assert mock_mol.call_count == 0
        assert mock_to.call_count == 0

    # After slicing, only canonical-metal rows remain. With 16 rows
    # and metal_idx ∈ [0, 4] all rows pass canonical_only=True.
    assert loaded["atom_features"].size(0) == 8
    assert loaded["atom_features"].size(1) == 6
    assert loaded["atom_features"].size(2) == 39
    assert (loaded["metal_idx"] >= 0).all()
    assert (loaded["metal_idx"] < 5).all()


def test_load_pre_featurized_canonical_filter(tmp_path: Path) -> None:
    """Rows with metal_idx == -1 (non-canonical metals) are dropped."""
    pack = _make_synthetic_pack(n_rows=12)
    pack["metal_idx"] = torch.tensor(
        [-1, -1, 0, 1, 2, -1, 3, 4, 0, -1, 1, 0], dtype=torch.int64
    )
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)

    loaded = vec.load_pre_featurized(p, canonical_only=True)
    # 12 rows total, 8 have metal_idx >= 0, max_rows=None so all 8 kept.
    assert loaded["atom_features"].size(0) == 8
    assert (loaded["metal_idx"] >= 0).all()


def test_load_pre_featurized_no_canonical_raises_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """When canonical_only=True and no rows have metal_idx >= 0 the
    loader still returns an empty pack — the operator (not the loader)
    is responsible for catching this and skipping the run."""
    pack = _make_synthetic_pack(n_rows=4)
    pack["metal_idx"] = torch.tensor([-1, -1, -1, -1], dtype=torch.int64)
    p = tmp_path / "all_pretrain.pt"
    torch.save(pack, p)

    loaded = vec.load_pre_featurized(p, canonical_only=True)
    assert loaded["atom_features"].size(0) == 0


# ---------------------------------------------------------------------------
# (ii) cached forward + backward works
# ---------------------------------------------------------------------------
def test_run_steps_grad_flow(synthetic_pack: dict) -> None:
    """3 cached-forward steps with RDKit mocked complete successfully
    and gradients flow into the shared encoder + every metal head."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with patch("rdkit.Chem.MolFromSmiles") as mock_mol, \
         patch("rdkit.Chem.MolToSmiles") as mock_to:
        res = vec.run_steps(
            synthetic_pack,
            n_steps=3,
            batch_size=4,
            hidden_dim=32,
            n_layers=2,
            seed=0,
            device=device,
        )
        # Contract (i): no RDKit call in the inner loop.
        assert mock_mol.call_count == 0
        assert mock_to.call_count == 0
        # Contract (ii): gradients flowed everywhere.
        assert res["grad_flow_ok"] is True
        # Contract (iii): 3 steps completed, all losses finite.
        assert len(res["step_times_ms"]) == 3
        assert all(t > 0 for t in res["step_times_ms"])
        assert res["final_loss"] is not None
        assert np.isfinite(res["final_loss"])


def test_run_steps_rdkit_contract(synthetic_pack: dict) -> None:
    """``rdkit_called_in_inner_loop`` flag is False when no RDKit call
    is observed. The script-level smoke asserts the same property on
    real data; this test pins the same invariant on a synthetic pack."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with patch("rdkit.Chem.MolFromSmiles"), patch("rdkit.Chem.MolToSmiles"):
        res = vec.run_steps(
            synthetic_pack, n_steps=2, batch_size=4,
            hidden_dim=32, n_layers=2, device=device,
        )
    assert res["rdkit_called_in_inner_loop"] is False
    assert res["rdkit_mol_called"] == 0
    assert res["rdkit_mol_to_smiles_called"] == 0


def test_run_steps_step_time_under_budget(synthetic_pack: dict) -> None:
    """Mean step time on the synthetic 32-row pack is finite and
    positive — this guards against pathological regressions like a
    kernel hang."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with patch("rdkit.Chem.MolFromSmiles"), patch("rdkit.Chem.MolToSmiles"):
        res = vec.run_steps(
            synthetic_pack, n_steps=3, batch_size=4,
            hidden_dim=32, n_layers=2, device=device,
        )
    assert 0.0 < res["mean_step_time_ms"] < 30_000.0, (
        f"mean step time out of plausible range: {res['mean_step_time_ms']:.1f}ms"
    )


# ---------------------------------------------------------------------------
# (iii) integration: V4MultiTaskDMPNN(use_cached_encoding=True) roundtrip
# ---------------------------------------------------------------------------
def test_v4multitask_cached_full_roundtrip(synthetic_pack: dict) -> None:
    """End-to-end roundtrip:

        pre-featurized tensors → model(use_cached_encoding=True) →
        multi-task loss → backward → step

    mirrors what the production training loop does.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = V4MultiTaskDMPNN(
        hidden_dim=32, n_layers=2, n_metals=5, n_coord_classes=5,
        use_cached_encoding=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = V4MultiTaskLoss(n_coord_classes=5)

    af = synthetic_pack["atom_features"].to(device)
    ei = synthetic_pack["edge_index"].to(device)
    ea = synthetic_pack["edge_attr"].to(device)
    mask = synthetic_pack["atom_mask"].to(device)
    mi = synthetic_pack["metal_idx"].to(device)
    act = synthetic_pack["active"].to(device).long()

    with patch("rdkit.Chem.MolFromSmiles"), patch("rdkit.Chem.MolToSmiles"):
        for step in range(3):
            optimizer.zero_grad()
            out = model.forward_cached(
                af, ei, mi, atom_mask=mask, edge_attr=ea,
            )
            y_true = {m: act for m in SUPPORTED_METALS}
            coord_true = act
            loss, _ = loss_fn(out, y_true, coord_true)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()

    # Encoder + every metal head received gradient
    enc_grad = model.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0
    for m in SUPPORTED_METALS:
        g = model.metal_heads[m][0].weight.grad
        assert g is not None and g.abs().sum().item() > 0.0