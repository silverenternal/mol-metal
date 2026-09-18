"""Tests for MetalHybridV4Model (LLRD + real Satorras EGNN + Perceiver + Kendall).

Covers five properties derived from
``molmetal/reports/review_pretraining_init.md``,
``review_egnn.md``, ``review_fusion.md``, ``review_multitask_loss.md``:

  (i)   Real Satorras-2021 EGNN layer is SE(3)-equivariant:
        f(Rx) ≈ R f(x) on the *output* (x and h projections).
        For the pooled fused vector, the EGNN contributes the per-atom
        scalar features (invariant under SE(3)) — we test rotation
        equivariance on the *coordinate* output (x').

  (ii)  Perceiver gate initial bias is set so that σ(bias) ≈ 0.05.

  (iii) LLRD parameter groups: lr(encoder_bottom) / lr(head) ≈ 1e-5 / 1e-3.

  (iv)  γ annealing (LossV4.gamma_at) starts at gamma_start and ends at
        gamma_end at epoch T-1.

  (v)   Kendall weights (log_s) converge / move away from initial value
        after a handful of optimiser updates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from molmetal.models.metal_hybrid_v4 import (
    LossV4,
    LossV4MB1,
    MetalHybridV4Config,
    MetalHybridV4Model,
    _PerceiverLatentFusion,
    _SatorrasEGNNLayer,
)
from molmetal.models.metal_hybrid import MetalHybridConfig


def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_egnn_layer() -> _SatorrasEGNNLayer:
    _set_seed(0)
    return _SatorrasEGNNLayer(hidden_dim=32)


# ---------------------------------------------------------------------------
# (i) SE(3) equivariance of the real Satorras EGNN layer
# ---------------------------------------------------------------------------
def test_egnn_layer_rotation_equivariance():
    """Verify f(Rx) = R f(x) for a random SO(3) rotation R.

    The scalar output h is invariant under SE(3) (by design), and the
    coordinate output x' is **equivariant**: rotating the input coords
    rotates the output coords by the same matrix.
    """
    layer = _make_egnn_layer()
    layer.eval()

    # 6 atoms, 2D layout
    N = 6
    h = torch.randn(N, 32)
    x = torch.randn(N, 3)

    # Build a small chain edge index
    edges = []
    for i in range(N - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)

    with torch.no_grad():
        # Original forward
        h_out, x_out = layer(h, x, edge_index)

        # Build a random rotation matrix R ∈ SO(3) via QR
        A = torch.randn(3, 3)
        Q, R = torch.linalg.qr(A)
        # Correct for the sign so det(Q) = +1
        Q = Q * torch.sign(torch.det(Q))
        x_rot = x @ Q.t()

        h_out_r, x_out_r = layer(h, x_rot, edge_index)

    # h is invariant: should match exactly
    h_diff = (h_out - h_out_r).abs().max().item()
    assert h_diff < 1e-5, f"h should be invariant under rotation, got diff={h_diff}"

    # x_out is equivariant: R @ x_out should equal x_out_r
    equiv_diff = (Q @ x_out.t() - x_out_r.t()).abs().max().item()
    assert equiv_diff < 1e-4, (
        f"x' should be SE(3)-equivariant (R@x' = R@layer(x)), "
        f"got max |Q@x' − x'_rot|={equiv_diff}"
    )


# ---------------------------------------------------------------------------
# (ii) Perceiver-latent gate bias init ≈ 0.05
# ---------------------------------------------------------------------------
def test_perceiver_gate_init_005():
    """After init, sigmoid(gate.bias) ≈ 0.05 — i.e. EGNN contribution is
    a small perturbation at step 0 (review_fusion.md §2)."""
    _set_seed(0)
    fusion = _PerceiverLatentFusion(dmpnn_dim=32, egnn_dim=32, latent_dim=128)
    bias = fusion.gate.bias.detach()
    sig = torch.sigmoid(bias)
    # sigmoid(log(0.05/0.95)) = 0.05 by construction
    assert torch.allclose(sig, torch.full_like(sig, 0.05), atol=1e-6), (
        f"Expected σ(bias)≈0.05, got mean={sig.mean().item():.4f}"
    )


# ---------------------------------------------------------------------------
# (iii) LLRD — lr(encoder_bottom) / lr(head) ≈ 1e-5 / 1e-3 = 1/100
# ---------------------------------------------------------------------------
def test_llrd_param_group_lr_ratio():
    """The D-MPNN bottom-most layer (atom_embed) gets the lowest LR;
    the head layer gets the highest LR.  Default config: ratio 1e-5 / 1e-3."""
    _set_seed(0)
    cfg = MetalHybridV4Config(
        base=MetalHybridConfig(hidden_dim=64, n_dmpnn_layers=2, n_egnn_layers=2),
    )
    model = MetalHybridV4Model(config=cfg)
    groups = model.llrd_param_groups()
    by_name = {g["name"]: g["lr"] for g in groups}

    assert "dmpnn.atom_embed" in by_name, "atom_embed group missing"
    assert "head+metal" in by_name, "head+metal group missing"

    # Bottom LR ≥ cfg.llrd_lr_bottom (lower bound)
    assert by_name["dmpnn.atom_embed"] >= cfg.llrd_lr_bottom - 1e-12, (
        f"atom_embed LR {by_name['dmpnn.atom_embed']:.2e} < "
        f"cfg.llrd_lr_bottom={cfg.llrd_lr_bottom:.2e}"
    )
    # Head LR == cfg.llrd_lr_top exactly
    assert math.isclose(by_name["head+metal"], cfg.llrd_lr_top, rel_tol=1e-9), (
        f"head LR {by_name['head+metal']:.2e} != cfg.llrd_lr_top={cfg.llrd_lr_top:.2e}"
    )
    # Decay monotonicity (top > next > ... > bottom)
    layer_order = [
        "dmpnn.atom_embed",
        "dmpnn.edge_embed",
        "dmpnn.edge_mlp[*]",
        "dmpnn.gru_updates[*]",
        "dmpnn.atom_to_edge[*]",
        "dmpnn.readout_mlp",
        "egnn",
        "fusion",
        "coord_refine",
        "head+metal",
    ]
    lrs = [by_name[name] for name in layer_order]
    # Each layer below should have LR ≤ the layer above it
    for i in range(len(lrs) - 1):
        assert lrs[i] <= lrs[i + 1] + 1e-12, (
            f"LLRD monotonicity broken at idx={i}: {layer_order[i]}={lrs[i]:.2e} > "
            f"{layer_order[i+1]}={lrs[i+1]:.2e}"
        )
    # Ratio test
    ratio = by_name["dmpnn.atom_embed"] / by_name["head+metal"]
    assert ratio <= 1.0, f"atom_embed LR / head LR should be <= 1, got {ratio:.2e}"
    # Decay factor 0.95^9 ≈ 0.630, so the lowest group should sit between
    # lr_bottom and lr_top * 0.95^9.
    expected_bottom = cfg.llrd_lr_top * (cfg.llrd_decay ** 9)
    expected_bottom = max(expected_bottom, cfg.llrd_lr_bottom)
    assert math.isclose(by_name["dmpnn.atom_embed"], expected_bottom, rel_tol=1e-9), (
        f"atom_embed LR {by_name['dmpnn.atom_embed']:.2e} != "
        f"expected {expected_bottom:.2e}"
    )


# ---------------------------------------------------------------------------
# (iv) γ annealing (cosine 0.1 → 0.0 over T epochs)
# ---------------------------------------------------------------------------
def test_loss_v4_gamma_annealing():
    """gamma_at(0) = gamma_start; gamma_at(T-1) = gamma_end; monotone."""
    _set_seed(0)
    loss = LossV4(alpha=0.3, gamma_start=0.1, gamma_end=0.0, ls=0.05)
    T = 30
    g0 = loss.gamma_at(0, T)
    g_mid = loss.gamma_at(T // 2, T)
    g_end = loss.gamma_at(T - 1, T)

    assert math.isclose(g0, 0.1, rel_tol=1e-6), f"γ(0)={g0}, expected 0.1"
    assert math.isclose(g_end, 0.0, abs_tol=1e-6), f"γ(T-1)={g_end}, expected 0.0"
    # Monotone non-increasing
    assert g0 >= g_mid >= g_end, f"γ not monotone: {g0}, {g_mid}, {g_end}"
    # Mid-point ~average of endpoints in cosine
    expected_mid = 0.5 * (g0 + g_end)
    assert abs(g_mid - expected_mid) < 0.1, (
        f"γ(T/2)={g_mid} far from average {expected_mid}"
    )


# ---------------------------------------------------------------------------
# (v) Kendall weights (log_sigma) move from initial 0 after a few updates
# ---------------------------------------------------------------------------
def test_kendall_weights_converge():
    """After ~5 optimiser steps, at least one Kendall log_sigma must have
    moved away from 0 (i.e. the loss is actually modulating them via
    gradients)."""
    _set_seed(0)
    loss = LossV4(alpha=0.3, gamma_start=0.1, gamma_end=0.0, ls=0.05)
    initial = (
        loss.log_s_cls.detach().item(),
        loss.log_s_reg.detach().item(),
        loss.log_s_crd.detach().item(),
    )

    # Build a tiny batch + fake outputs
    B = 4
    out = {
        "active_logits": torch.randn(B, 2),
        "pic50_pred": torch.randn(B),
        "delta_pred": torch.zeros(B, 5, 3),
        "trunk": torch.randn(B, 128),
    }
    batch = {
        "active": torch.tensor([0, 1, 0, 1]),
        "pic50": torch.tensor([5.0, 6.5, 4.5, 7.0]),
        "target_delta": torch.zeros(B, 5, 3),
        "mask": torch.ones(B),
        "atom_mask": torch.ones(B, 5, dtype=torch.bool),
    }

    opt = torch.optim.Adam(loss.parameters(), lr=0.05)
    for _ in range(8):
        opt.zero_grad()
        total, _ = loss(out, batch, epoch=0, T=10)
        total.backward()
        opt.step()

    after = (
        loss.log_s_cls.detach().item(),
        loss.log_s_reg.detach().item(),
        loss.log_s_crd.detach().item(),
    )
    # At least one log_sigma must have moved by > 1e-4
    moved = max(
        abs(after[i] - initial[i]) for i in range(3)
    )
    assert moved > 1e-4, (
        f"Kendall log-sigmas did not move after 8 optimizer steps: "
        f"initial={initial}, after={after}"
    )


# ---------------------------------------------------------------------------
# (extra) Forward smoke + LLRD shapes — sanity for the full model
# ---------------------------------------------------------------------------
def test_v4_forward_smoke():
    """Forward pass returns the expected dict shapes."""
    _set_seed(0)
    cfg = MetalHybridV4Config(
        base=MetalHybridConfig(hidden_dim=128, n_dmpnn_layers=2, n_egnn_layers=2),
        egnn_in_node_dim=128,
        egnn_hidden_dim=128,
    )
    model = MetalHybridV4Model(config=cfg).eval()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    smiles = ["CCO", "CCN(CC)C(=O)C"]
    mols = []
    coords_list = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        mols.append(m)
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m)
        coords_list.append(np.array(m.GetConformer(0).GetPositions(), dtype=np.float32))
    N_max = max(c.shape[0] for c in coords_list)
    coords = np.zeros((2, N_max, 3), dtype=np.float32)
    for i, c in enumerate(coords_list):
        coords[i, : c.shape[0]] = c
    coords = torch.from_numpy(coords).float()
    metal_types = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        out = model(smiles, coords, metal_types, mol_objects=mols)
    assert out["pic50"].shape == (2,)
    assert out["active_logits"].shape == (2, 2)
    assert out["delta_pred"].shape == (2, N_max, 3)
    assert out["trunk"].shape == (2, 128)
    assert torch.isfinite(out["pic50"]).all()
    assert torch.isfinite(out["active_logits"]).all()


# ---------------------------------------------------------------------------
# (vi) M-B1: two-stage forward exposes pic50_raw / active_p
# ---------------------------------------------------------------------------
def test_v4_forward_two_stage_smoke():
    """``forward_two_stage`` returns ``pic50_raw`` (log-space) and
    ``active_p`` (sigmoid) — the keys ``LossV4MB1`` consumes."""
    _set_seed(0)
    cfg = MetalHybridV4Config(
        base=MetalHybridConfig(hidden_dim=128, n_dmpnn_layers=2, n_egnn_layers=2),
        egnn_in_node_dim=128,
        egnn_hidden_dim=128,
    )
    model = MetalHybridV4Model(config=cfg).eval()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    smiles = ["CCO", "CCN(CC)C(=O)C"]
    mols, coords_list = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        mols.append(m)
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m)
        coords_list.append(np.array(m.GetConformer(0).GetPositions(), dtype=np.float32))
    N_max = max(c.shape[0] for c in coords_list)
    coords = np.zeros((2, N_max, 3), dtype=np.float32)
    for i, c in enumerate(coords_list):
        coords[i, : c.shape[0]] = c
    coords = torch.from_numpy(coords).float()
    metal_types = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        out = model.forward_two_stage(smiles, coords, metal_types, mol_objects=mols)
    assert "pic50_raw" in out, "two-stage forward must expose pic50_raw"
    assert "active_p" in out, "two-stage forward must expose active_p"
    assert out["pic50_raw"].shape == (2,)
    assert out["active_p"].shape == (2,)
    assert (out["active_p"] >= 0.0).all() and (out["active_p"] <= 1.0).all()
    # pIC50_pred >= active_p (since 5*relu(...) >= 0)
    assert (out["pic50"] >= out["active_p"] - 1e-5).all()


# ---------------------------------------------------------------------------
# (vii) M-B1: LossV4MB1 — Kendall weights move + commit_epoch snapshots
# ---------------------------------------------------------------------------
def test_loss_v4_mb1_kendall_and_commit():
    """After 8 optimiser steps at least one ``log_s`` must have moved
    away from 0; ``commit_epoch()`` snapshots the snapshot for the
    ``sigma_*_log_change`` diff."""
    _set_seed(0)
    loss_fn = LossV4MB1(
        alpha=0.3, gamma_start=0.1, gamma_end=0.0, ls=0.05,
        warmup=4, active_weight=3.0,
    )

    B = 4
    out = {
        "active_logits": torch.tensor(
            [[0.5, 0.2], [0.3, 0.7], [0.6, 0.1], [0.4, 0.6]],
            requires_grad=True,
        ),
        "pic50_raw": torch.tensor([0.1, 0.5, -0.1, 0.3], requires_grad=True),
        "delta_pred": torch.zeros(B, 5, 3, requires_grad=True),
        "trunk": torch.randn(B, 128, requires_grad=True),
    }
    batch = {
        "active": torch.tensor([0, 1, 0, 1]),
        "pic50": torch.tensor([4.2, 6.5, 4.5, 7.0]),
        "target_delta": torch.zeros(B, 5, 3),
        "mask": torch.ones(B),
        "atom_mask": torch.ones(B, 5, dtype=torch.bool),
    }

    initial = (
        loss_fn.log_s_cls.detach().item(),
        loss_fn.log_s_reg.detach().item(),
    )

    opt = torch.optim.Adam(loss_fn.parameters(), lr=0.05)
    for _ in range(8):
        opt.zero_grad()
        total, info = loss_fn(out, batch, epoch=0, T=10)
        total.backward()
        opt.step()

    after = (
        loss_fn.log_s_cls.detach().item(),
        loss_fn.log_s_reg.detach().item(),
    )
    moved = max(abs(after[i] - initial[i]) for i in range(2))
    assert moved > 1e-4, (
        f"Kendall log-sigmas did not move: initial={initial}, after={after}"
    )

    # commit_epoch snapshots — next forward diff should be (near) zero
    loss_fn.commit_epoch()
    total2, info2 = loss_fn(out, batch, epoch=0, T=10)
    assert abs(info2["sigma_cls_log_change"]) < 1e-5
    assert abs(info2["sigma_reg_log_change"]) < 1e-5
    # info has the multi-fidelity diagnostics
    assert "active_frac" in info2
    assert "loss_reg_active" in info2
    assert "loss_reg_inactive" in info2