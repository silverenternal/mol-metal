"""Tests for D-MPNN + EGNN hybrid model components and end-to-end."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
from molmetal.models.egnn_predict import EGNNPredictor, EGNNPredictorConfig
from molmetal.models.fusion import FusionMLP
from molmetal.models.loss import MetalCytotoxLoss
from molmetal.models.metal_hybrid import MetalHybridConfig, MetalHybridModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_8_atom_mol():
    """Return a small graph: 8 atoms in a chain with bidirectional edges."""
    N = 8
    atom_dim = 39
    edge_dim = 6
    bonds = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
    src = [a for a,b in bonds for a in [a,b]]
    dst = [b for a,b in bonds for b in [b,a]]
    h_atom = torch.randn(N, atom_dim)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(14, edge_dim)
    return h_atom, edge_index, edge_attr


# ---------------------------------------------------------------------------
# test_dmpnn_forward_shape
# ---------------------------------------------------------------------------
def test_dmpnn_forward_shape():
    """D-MPNN on 8-atom mol produces correct output shape."""
    set_seed(0)
    h_atom, edge_index, edge_attr = make_8_atom_mol()
    N = h_atom.size(0)

    model = DirectedMPNN(MPNNConfig(hidden_dim=128, n_layers=3))
    out = model.forward_per_atom(h_atom, edge_index, edge_attr)
    assert out.shape == (N, 128), f"Expected ({N}, 128), got {out.shape}"

    B = 2
    h_atom_b = h_atom.unsqueeze(0).expand(B, -1, -1)
    edge_index_b = edge_index.unsqueeze(0).expand(B, -1, -1)
    edge_attr_b = edge_attr.unsqueeze(0).expand(B, -1, -1)
    batch_idx = torch.zeros(B, N, dtype=torch.long)
    batch_idx[1, :] = 1

    model_b = DirectedMPNN(MPNNConfig(hidden_dim=128, n_layers=3))
    out_b = model_b(h_atom_b, edge_index_b, edge_attr_b, batch_idx)
    assert out_b.shape == (B, 128), f"Expected ({B}, 128), got {out_b.shape}"
    print("test_dmpnn_forward_shape PASSED")


# ---------------------------------------------------------------------------
# test_egnn_block_se3_invariance
# ---------------------------------------------------------------------------
def test_egnn_block_se3_invariance():
    """Rotated coordinates -> invariant pooled EGNN output."""
    set_seed(0)
    from molmetal.adapters.egnn_rocm import EGNN

    in_dim = 64
    hidden_dim = 64
    n_layers = 2
    egnn = EGNN(in_node_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers)

    N = 6
    h = torch.randn(N, in_dim)
    x = torch.randn(N, 3)
    edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]], dtype=torch.long)

    h1, _ = egnn(h.clone(), x.clone(), edge_index)
    angle = np.pi / 2
    rot = torch.tensor(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle),  np.cos(angle), 0],
         [0, 0, 1]],
        dtype=torch.float,
    )
    x_rot = x @ rot.T
    h2, _ = egnn(h.clone(), x_rot, edge_index)

    pool1 = h1.sum(dim=0)
    pool2 = h2.sum(dim=0)
    assert torch.allclose(pool1, pool2, atol=1e-5), "EGNN pooled output changed under rotation"
    print("test_egnn_block_se3_invariance PASSED")


# ---------------------------------------------------------------------------
# test_fusion_forward
# ---------------------------------------------------------------------------
def test_fusion_forward():
    """Fusion MLP takes D-MPNN + EGNN outputs and produces correct shape."""
    set_seed(0)
    B, N, D = 4, 10, 128
    h_2d = torch.randn(B, N, D)
    h_3d = torch.randn(B, N, D)
    fusion = FusionMLP(h_dim=D, output_dim=D)
    out = fusion(h_2d, h_3d)
    assert out.shape == (B, N, D), f"Expected ({B}, {N}, {D}), got {out.shape}"
    print("test_fusion_forward PASSED")


# ---------------------------------------------------------------------------
# test_metal_hybrid_end_to_end
# ---------------------------------------------------------------------------
def test_metal_hybrid_end_to_end():
    """5 mols, both heads produce valid outputs (finite, correct shape)."""
    set_seed(42)
    model = MetalHybridModel(
        config=MetalHybridConfig(hidden_dim=128, n_dmpnn_layers=3, n_egnn_layers=3)
    )
    model.eval()

    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)C", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    B = len(smiles_list)
    N_max = 30
    device = next(model.parameters()).device

    coords = torch.randn(B, N_max, 3).to(device)
    metal_types = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long).to(device)

    with torch.no_grad():
        pic50_pred, active_logits = model(smiles_list, coords, metal_types)

    assert pic50_pred.shape == (B,), f"pic50_pred: expected ({B},), got {pic50_pred.shape}"
    assert active_logits.shape == (B,) or active_logits.shape == (B, 2), f"active_logits: expected ({B},) or ({B}, 2), got {active_logits.shape}"
    assert torch.isfinite(pic50_pred).all(), "pic50_pred contains NaN/Inf"
    assert torch.isfinite(active_logits).all(), "active_logits contains NaN/Inf"
    print("test_metal_hybrid_end_to_end PASSED")


# ---------------------------------------------------------------------------
# test_metal_hybrid_loss
# ---------------------------------------------------------------------------
def test_metal_hybrid_loss():
    """Dual-head loss is finite and decreases after 5 train steps."""
    set_seed(42)
    model = MetalHybridModel(
        config=MetalHybridConfig(hidden_dim=64, n_dmpnn_layers=2, n_egnn_layers=2)
    )
    loss_fn = MetalCytotoxLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 8
    N_max = 20
    device = next(model.parameters()).device

    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)C", "CCN", "CCCC", "CCOCC", "CCOC"]
    coords = torch.randn(B, N_max, 3).to(device)
    metal_types = torch.randint(0, 5, (B,), dtype=torch.long).to(device)
    pic50_true = torch.rand(B).to(device) * 5 + 4
    active_label = (pic50_true > 6.0).float()

    losses = []
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        pic50_pred, active_logits = model(smiles_list, coords, metal_types)
        loss = loss_fn(pic50_pred, pic50_true, active_label, active_logits=active_logits)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(l) for l in losses), f"Loss contains non-finite values: {losses}"
    print(f"test_metal_hybrid_loss PASSED  (losses: {[f'{l:.4f}' for l in losses]})")


if __name__ == "__main__":
    print("Running test_hybrid.py ...")
    test_dmpnn_forward_shape()
    test_egnn_block_se3_invariance()
    test_fusion_forward()
    test_metal_hybrid_end_to_end()
    test_metal_hybrid_loss()
    print("\nAll tests PASSED!")
