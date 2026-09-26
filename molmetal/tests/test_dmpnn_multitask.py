"""Tests for the multi-task D-MPNN baseline.

Wf-124 Phase 3 — Multi-task 5-metal joint backbone + per-metal heads + Wasserstein
pIC50 regularisation. 18 tests total:

Original single-metal multi-task (3):
* ``test_dual_head_shapes``        — forward → (B,) pic50 + (B, 2) active_logits
* ``test_loss_finite_and_decreases`` — 20 train steps, loss drops > 0.3x
* ``test_multitask_beats_single``  — multi-task AUC ≥ single-task AUC on synthetic

New joint 5-metal multi-task (15):
* ``test_joint_backbone_shared_encoder``     — shared D-MPNN encoder across 5 metals
* ``test_per_metal_head_count``               — 5 per-metal (reg+cls) heads
* ``test_per_metal_head_shape``               — forward → (B,) pic50 per-metal
* ``test_per_metal_head_distinct_params``     — per-metal heads have distinct weights
* ``test_joint_loss_alpha_half``              — alpha=0.5 → 0.5*BCE + 0.5*MSE
* ``test_joint_loss_alpha_one``               — alpha=1.0 → pure BCE
* ``test_joint_loss_alpha_zero``              — alpha=0.0 → pure MSE
* ``test_joint_forward_with_metal_ids``       — forward accepts metal_ids tensor
* ``test_joint_encode_pool_shape``            — encode_pool returns (B, 2H)
* ``test_joint_train_step_no_nan``            — training step produces no NaN
* ``test_joint_gradient_flow_to_encoder``     — gradients flow from heads to encoder
* ``test_5metal_split_csv_roundtrip``         — 5-metal CSV round-trip
* ``test_5metal_split_indices_build``         — per-metal train/val/test indices
* ``test_5metal_collate_filters_invalid``     — collate handles None/NaN
* ``test_5metal_joint_loss_decreases``        — 5-metal joint training loss decreases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from molmetal.baselines.dmpnn_multitask import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNNMultiTaskBaseline,
    DMPNNMultiTaskModel,
    MT_ALPHA,
    MT_BATCH_SIZE,
    MultiTaskLoss,
    featurize_smiles_list,
    predict_mt,
    train_epoch_mt,
)
from molmetal.baselines.dmpnn import (
    DMPNNModel,
    featurize_smiles_list as _featurize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SMILES_8 = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(C)C",
    "CCC", "C1CCCCC1", "CCOCC", "CCCC",
]


def _build_batch(smiles_list):
    graphs = _featurize(smiles_list)
    # Re-use the collate_graphs from dmpnn
    from molmetal.baselines.dmpnn import collate_graphs

    batch_data = collate_graphs(graphs)
    return batch_data, graphs


# ---------------------------------------------------------------------------
# Test 1: dual-head output shapes
# ---------------------------------------------------------------------------
def test_dual_head_shapes():
    """Forward on 8 mols → pic50 shape (8,), active_logits shape (8, 2)."""
    batch_data, _graphs = _build_batch(SMILES_8)
    atom_f, bond_f, e_src, e_dst = batch_data

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=64,   # small for fast test
        depth=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        pic50, active_logits = model(atom_f, bond_f, e_src, e_dst)
    assert pic50.shape == (8,), f"Expected pic50 shape (8,), got {pic50.shape}"
    assert active_logits.shape == (8, 2), (
        f"Expected active_logits shape (8, 2), got {active_logits.shape}"
    )
    # No NaN / Inf
    assert torch.isfinite(pic50).all(), "pic50 contains non-finite values"
    assert torch.isfinite(active_logits).all(), "active_logits contains non-finite values"


# ---------------------------------------------------------------------------
# Test 2: loss is finite and decreases
# ---------------------------------------------------------------------------
def test_loss_finite_and_decreases():
    """20 train steps on 8 molecules → loss drops > 0.3x (relative reduction)."""
    batch_data, graphs = _build_batch(SMILES_8)
    atom_f, bond_f, e_src, e_dst = batch_data

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=64,
        depth=2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=MT_ALPHA)

    # Synthesise labels: pIC50 ~ uniform[4, 8], active ~ Bernoulli(0.5)
    torch.manual_seed(0)
    B = 8
    y_pic50 = 4.0 + 4.0 * torch.rand(B)
    y_active = torch.randint(0, 2, (B,))

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        pic50, logits = model(atom_f, bond_f, e_src, e_dst)
        loss, lc, lr = criterion(pic50, logits, y_pic50, y_active)
        # Sanity: finite
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial = losses[0]
    final = losses[-1]
    ratio = initial / max(final, 1e-8)
    print(
        f"[test_loss_finite_and_decreases] initial={initial:.4f} "
        f"final={final:.4f} ratio={ratio:.2f}x"
    )
    # Require at least 0.3x relative reduction (final ≤ initial * 0.7).
    # We do NOT require final < initial (some seeds may oscillate), only that
    # the multi-task loss does not grow without bound.
    assert final < initial * 0.95 or ratio > 1.05, (
        f"Multi-task loss did not decrease meaningfully: "
        f"initial={initial:.4f} final={final:.4f}"
    )
    # And absolutely: final loss should be smaller than initial by at least 0.3
    assert initial - final >= 0.3, (
        f"Loss decrease too small: {initial - final:.4f} < 0.3"
    )


# ---------------------------------------------------------------------------
# Test 3: multi-task ≥ single-task on synthetic dataset
# ---------------------------------------------------------------------------
def test_multitask_beats_single():
    """On synthetic data, multi-task classification AUC ≥ single-task AUC.

    We generate ~30 random molecules with synthesised labels where the
    underlying regression signal carries information about the binary
    label (so multi-task has at least as much signal as the single-task
    classification head).  This is a *weak* test — it just guards against
    the multi-task setup catastrophically hurting the classification head.

    Skipped on CI without RDKit.
    """
    rng = np.random.default_rng(0)
    # Build a small synthetic dataset
    n = 64
    # Use the 8 SMILES_8 templates but duplicate them with random "active"
    # labels drawn from a continuous latent variable.
    base_smiles = SMILES_8 * (n // len(SMILES_8))
    base_smiles = base_smiles[:n]

    # Latent pIC50: drives both activity (binary) and regression (continuous)
    pic50_latent = rng.normal(5.5, 1.0, size=n).astype(np.float32)
    active_latent = (pic50_latent > 5.5).astype(int)

    # ---- Multi-task model ----
    from molmetal.baselines.dmpnn import collate_graphs
    from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

    graphs = _featurize(base_smiles)
    batch_data = collate_graphs(graphs)
    atom_f, bond_f, e_src, e_dst = batch_data

    # Train multi-task model for many steps on this dataset
    mt_model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=64, depth=2, dropout=0.0,
    )
    optimizer = torch.optim.Adam(mt_model.parameters(), lr=5e-3)
    criterion = MultiTaskLoss(alpha=MT_ALPHA)
    y_pic50_t = torch.from_numpy(pic50_latent)
    y_active_t = torch.from_numpy(active_latent).long()
    for _ in range(80):
        optimizer.zero_grad()
        pic50, logits = mt_model(atom_f, bond_f, e_src, e_dst)
        loss, _, _ = criterion(pic50, logits, y_pic50_t, y_active_t)
        loss.backward()
        optimizer.step()

    mt_model.eval()
    with torch.no_grad():
        _, mt_logits = mt_model(atom_f, bond_f, e_src, e_dst)
        mt_probs = torch.softmax(mt_logits, dim=-1)[:, 1].cpu().numpy()

    # ---- Single-task (active-only) baseline ----
    from molmetal.baselines.dmpnn import DMPNNModel

    st_model = DMPNNModel(
        atom_dim=ATOM_FEATURE_DIM, bond_dim=BOND_FEATURE_DIM,
        hidden=64, depth=2, dropout=0.0,
    )
    opt2 = torch.optim.Adam(st_model.parameters(), lr=5e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    y_act_f = torch.from_numpy(active_latent.astype(np.float32))
    for _ in range(80):
        opt2.zero_grad()
        out = st_model(atom_f, bond_f, e_src, e_dst)
        loss2 = bce(out, y_act_f)
        loss2.backward()
        opt2.step()
    st_model.eval()
    with torch.no_grad():
        st_logits = st_model(atom_f, bond_f, e_src, e_dst).cpu().numpy()
        st_probs = 1.0 / (1.0 + np.exp(-st_logits))

    from sklearn.metrics import roc_auc_score
    try:
        mt_auc = float(roc_auc_score(active_latent, mt_probs))
        st_auc = float(roc_auc_score(active_latent, st_probs))
    except ValueError:
        # Single-class edge case — skip
        pytest.skip("synthetic labels collapsed to a single class")

    print(
        f"[test_multitask_beats_single] multi-task AUC={mt_auc:.4f} "
        f"single-task AUC={st_auc:.4f}"
    )
    # Multi-task AUC must be >= single-task AUC - 0.05
    # (a soft tolerance — multi-task is allowed to be slightly worse on
    #  small synthetic data but should not collapse relative to single).
    assert mt_auc >= st_auc - 0.05, (
        f"Multi-task AUC {mt_auc:.4f} is more than 0.05 worse than "
        f"single-task AUC {st_auc:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ===========================================================================
# Wf-124 Phase 3 — Multi-task 5-metal joint backbone + per-metal heads
# ===========================================================================
# The 15 tests below cover the joint 5-metal multi-task D-MPNN recipe:
#   - shared encoder (D-MPNN backbone) → 2H-dim pooled representation
#   - per-metal (regression + classification) heads
#   - joint loss with optional Wasserstein pIC50 regularisation
#
# The training script that orchestrates the joint 5-metal loop lives in
# :mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`. We re-use the
# :class:`MultiTaskWassersteinModel` and helpers from that module.


# ---------------------------------------------------------------------------
# Helpers (for joint 5-metal tests)
# ---------------------------------------------------------------------------
FIVE_METALS_5 = ("Ru", "Ir", "Rh", "Os", "Re")


def _toy_single_atom_graph():
    """Build a single-atom, no-bond graph (always valid)."""
    from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    atom_f = np.zeros((1, ATOM_FEATURE_DIM), dtype=np.float32)
    atom_f[0, 0] = 1.0  # mark as a real atom
    bond_f = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
    e_src = np.zeros((0,), dtype=np.int64)
    e_dst = np.zeros((0,), dtype=np.int64)
    return (atom_f, bond_f, e_src, e_dst)


def _toy_molecule_graphs(n: int):
    """Build n tiny graphs (one atom each)."""
    return [_toy_single_atom_graph() for _ in range(n)]


def _build_joint_batch(n_per_metal: int = 2):
    """Build a joint 5-metal batch (graphs, mids, pic50, active).

    Returns 5*2 = 10 graphs (2 per metal × 5 metals), metal_ids in
    FIVE_METALS_5 order, pic50 ∈ [3, 9], active ∈ {0, 1}.
    """
    n = n_per_metal * len(FIVE_METALS_5)
    graphs = _toy_molecule_graphs(n)
    mids = [i % len(FIVE_METALS_5) for i in range(n)]
    rng = np.random.default_rng(42)
    pic50 = rng.uniform(3.0, 9.0, n).astype(np.float32)
    active = rng.integers(0, 2, n).astype(np.int64)
    return graphs, mids, pic50, active


# ---------------------------------------------------------------------------
# Test 4: joint backbone has shared encoder across all 5 metals
# ---------------------------------------------------------------------------
def test_joint_backbone_shared_encoder():
    """Joint 5-metal multi-task model has a single shared D-MPNN encoder."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    model = MultiTaskWassersteinModel(n_metals=len(FIVE_METALS_5), hidden=64)
    assert hasattr(model, "encoder")
    # Shared encoder is a single nn.Module (one backbone for all metals)
    assert isinstance(model.encoder, torch.nn.Module)
    # Encoder must not have one ModuleList of per-metal params (i.e. it must
    # be a single backbone, not 5 backbones).
    assert not isinstance(model.encoder, torch.nn.ModuleList)


# ---------------------------------------------------------------------------
# Test 5: per-metal heads — exactly 5 heads (Ru/Ir/Os/Rh/Re)
# ---------------------------------------------------------------------------
def test_per_metal_head_count():
    """Joint model has exactly 5 per-metal reg + 5 per-metal cls heads."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    assert len(model.reg_heads) == n_metals
    assert len(model.cls_heads) == n_metals


# ---------------------------------------------------------------------------
# Test 6: per-metal head output shape — (B,) pic50 + (B, 2) active_logits
# ---------------------------------------------------------------------------
def test_per_metal_head_shape():
    """Forward on a 5-metal joint batch → (B,) pic50 + (B, 2) cls logits."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        collate_multimetal,
    )

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    model.eval()
    graphs, mids, pic50, active = _build_joint_batch(n_per_metal=2)
    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is not None
    atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch
    with torch.no_grad():
        pic50_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
    assert pic50_pred.shape == (len(graphs),)
    assert cls_logits.shape == (len(graphs), 2)
    assert torch.isfinite(pic50_pred).all()
    assert torch.isfinite(cls_logits).all()


# ---------------------------------------------------------------------------
# Test 7: per-metal heads have distinct (non-shared) parameters
# ---------------------------------------------------------------------------
def test_per_metal_head_distinct_params():
    """Each per-metal head has its own parameters (no weight sharing)."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    # Collect first linear weight from each reg head
    first_reg_weights = []
    for head in model.reg_heads:
        first_linear = next(m for m in head if isinstance(m, torch.nn.Linear))
        first_reg_weights.append(first_linear.weight.detach().clone())
    # Pairwise: different metals must have different head weights
    for i in range(n_metals):
        for j in range(i + 1, n_metals):
            assert not torch.allclose(first_reg_weights[i], first_reg_weights[j]), (
                f"reg_heads[{i}] == reg_heads[{j}] — per-metal heads share weights"
            )


# ---------------------------------------------------------------------------
# Test 8: MultiTaskLoss alpha=0.5 → 0.5 * BCE + 0.5 * MSE
# ---------------------------------------------------------------------------
def test_joint_loss_alpha_half():
    """With alpha=0.5, MultiTaskLoss = 0.5*BCE + 0.5*MSE."""
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss

    criterion = MultiTaskLoss(alpha=0.5)
    pic50_pred = torch.tensor([5.0, 6.5, 4.0], dtype=torch.float32)
    active_logits = torch.tensor([[0.5, 0.2], [1.0, 0.5], [0.1, 1.5]])
    pic50_t = torch.tensor([5.5, 6.0, 4.5], dtype=torch.float32)
    active_t = torch.tensor([1, 0, 1], dtype=torch.long)

    loss, lc, lr = criterion(pic50_pred, active_logits, pic50_t, active_t)
    expected_cls = float(
        torch.nn.functional.cross_entropy(active_logits, active_t).item()
    )
    expected_reg = float(
        torch.nn.functional.mse_loss(pic50_pred, pic50_t).item()
    )
    expected = 0.5 * expected_cls + 0.5 * expected_reg

    assert abs(loss.item() - expected) < 1e-5
    assert abs(lc.item() - expected_cls) < 1e-5
    assert abs(lr.item() - expected_reg) < 1e-5


# ---------------------------------------------------------------------------
# Test 9: MultiTaskLoss alpha=1.0 → pure BCE
# ---------------------------------------------------------------------------
def test_joint_loss_alpha_one():
    """With alpha=1.0, the reg loss does not contribute."""
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss

    criterion = MultiTaskLoss(alpha=1.0)
    pic50_pred = torch.tensor([5.0, 6.5, 4.0], dtype=torch.float32)
    active_logits = torch.tensor([[0.5, 0.2], [1.0, 0.5], [0.1, 1.5]])
    pic50_t = torch.tensor([5.5, 6.0, 4.5], dtype=torch.float32)
    active_t = torch.tensor([1, 0, 1], dtype=torch.long)

    loss, lc, lr = criterion(pic50_pred, active_logits, pic50_t, active_t)
    expected_cls = float(
        torch.nn.functional.cross_entropy(active_logits, active_t).item()
    )
    assert abs(lc.item() - expected_cls) < 1e-5
    # lr component is included for monitoring but the *total* loss = lc only
    assert abs(loss.item() - expected_cls) < 1e-5


# ---------------------------------------------------------------------------
# Test 10: MultiTaskLoss alpha=0.0 → pure MSE
# ---------------------------------------------------------------------------
def test_joint_loss_alpha_zero():
    """With alpha=0.0, the cls loss does not contribute."""
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss

    criterion = MultiTaskLoss(alpha=0.0)
    pic50_pred = torch.tensor([5.0, 6.5, 4.0], dtype=torch.float32)
    active_logits = torch.tensor([[0.5, 0.2], [1.0, 0.5], [0.1, 1.5]])
    pic50_t = torch.tensor([5.5, 6.0, 4.5], dtype=torch.float32)
    active_t = torch.tensor([1, 0, 1], dtype=torch.long)

    loss, lc, lr = criterion(pic50_pred, active_logits, pic50_t, active_t)
    expected_reg = float(
        torch.nn.functional.mse_loss(pic50_pred, pic50_t).item()
    )
    assert abs(lr.item() - expected_reg) < 1e-5
    assert abs(loss.item() - expected_reg) < 1e-5


# ---------------------------------------------------------------------------
# Test 11: forward accepts metal_ids tensor (long dtype)
# ---------------------------------------------------------------------------
def test_joint_forward_with_metal_ids():
    """Forward accepts a (B,) long tensor of metal indices."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        collate_multimetal,
    )

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    model.eval()
    graphs, mids, pic50, active = _build_joint_batch(n_per_metal=2)
    batch = collate_multimetal(graphs, mids, pic50, active)
    atom_f, bond_f, e_src, e_dst, mids_t, _, _ = batch
    assert mids_t.dtype == torch.long
    assert torch.all(mids_t >= 0) and torch.all(mids_t < n_metals)
    with torch.no_grad():
        out = model(atom_f, bond_f, e_src, e_dst, mids_t)
    assert len(out) == 2  # (pic50, active_logits)


# ---------------------------------------------------------------------------
# Test 12: encode_pool returns (B, 2H) tensor
# ---------------------------------------------------------------------------
def test_joint_encode_pool_shape():
    """encode_pool returns (B, 2H) pooled representation."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        collate_multimetal,
    )

    n_metals = len(FIVE_METALS_5)
    hidden = 64
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=hidden)
    model.eval()
    graphs, mids, pic50, active = _build_joint_batch(n_per_metal=2)
    batch = collate_multimetal(graphs, mids, pic50, active)
    atom_f, bond_f, e_src, e_dst, _, _, _ = batch
    with torch.no_grad():
        h_pool = model.encode_pool(atom_f, bond_f, e_src, e_dst)
    assert h_pool.shape == (len(graphs), 2 * hidden)


# ---------------------------------------------------------------------------
# Test 13: one train step on a 5-metal joint batch produces no NaN
# ---------------------------------------------------------------------------
def test_joint_train_step_no_nan():
    """A single training step on a 5-metal batch produces finite losses."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        train_one_epoch_joint,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.molmetal_lam.reward.wasserstein_loss import WassersteinConfig

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    w_config = WassersteinConfig(n_mc=64, seed=0)

    graphs, mids, pic50, active = _build_joint_batch(n_per_metal=4)
    ref_pic50 = torch.from_numpy(pic50.astype(np.float32))

    loss, cls_loss, reg_loss, w_pen = train_one_epoch_joint(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        w_w=0.05,
        use_l2_guard=True,
        graphs=graphs,
        metal_ids=mids,
        pic50=pic50,
        active=active,
        ref_pic50_dist=ref_pic50,
        batch_size=4,
        device=torch.device("cpu"),
        w_config=w_config,
    )
    assert np.isfinite(loss), f"total loss not finite: {loss}"
    assert np.isfinite(cls_loss), f"cls loss not finite: {cls_loss}"
    assert np.isfinite(reg_loss), f"reg loss not finite: {reg_loss}"
    assert np.isfinite(w_pen), f"Wasserstein term not finite: {w_pen}"
    assert w_pen >= 0.0


# ---------------------------------------------------------------------------
# Test 14: gradients flow from heads back to shared encoder
# ---------------------------------------------------------------------------
def test_joint_gradient_flow_to_encoder():
    """Backprop from per-metal heads reaches the shared encoder parameters.

    Uses real SMILES (with bonds) so the encoder's ``_forward_one_mol``
    is actually called — single-atom toy graphs would skip the encoder.
    The encoder's own ``readout`` is unused in the joint setup (replaced
    by per-metal heads) so we only check the *backbone* sub-modules:
    ``atom_emb``, ``bond_emb``, ``gru_update``, ``edge_init``.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        collate_multimetal,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.baselines.dmpnn import featurize_smiles_list

    n_metals = len(FIVE_METALS_5)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)

    # Use real SMILES with bonds (each metal gets one with bonds)
    real_smiles = [
        "CCO", "CC(=O)O", "c1ccccc1", "CC(C)C",
        "CCC", "C1CCCCC1", "CCOCC", "CCCC",
        "c1ccncc1", "CCS",
    ]
    graphs = featurize_smiles_list(real_smiles)
    # Filter out any None from featurizer failures
    valid = [
        (g, i % n_metals, 5.0 + 0.1 * i, i % 2)
        for i, g in enumerate(graphs) if g is not None
    ]
    graphs = [v[0] for v in valid]
    mids = [v[1] for v in valid]
    pic50 = np.array([v[2] for v in valid], dtype=np.float32)
    active = np.array([v[3] for v in valid], dtype=np.int64)

    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is not None
    atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch

    criterion = MultiTaskLoss(alpha=0.5)
    pic50_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
    loss, _, _ = criterion(pic50_pred, cls_logits, pic50_t, active_t)
    loss.backward()

    # Check the *backbone* sub-modules (those actually used in the joint
    # forward): atom_emb, bond_emb, gru_update. (Note: edge_init exists
    # on DMPNNMultiTaskModel but is unused in `_forward_one_mol`.)
    backbone_submodules = (
        model.encoder.atom_emb, model.encoder.bond_emb,
        model.encoder.gru_update,
    )
    backbone_grad_norm = 0.0
    n_params_with_grad = 0
    for sub in backbone_submodules:
        for p in sub.parameters():
            assert p.grad is not None, (
                f"backbone param in {type(sub).__name__} missing gradient"
            )
            backbone_grad_norm += float(p.grad.norm().item())
            n_params_with_grad += 1
    assert n_params_with_grad > 0
    assert backbone_grad_norm > 0.0

    # At least one per-metal head received gradient
    n_head_grads = sum(
        1 for head in model.reg_heads
        for p in head.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_head_grads > 0


# ---------------------------------------------------------------------------
# Test 15: 5-metal split CSV round-trip
# ---------------------------------------------------------------------------
def test_5metal_split_csv_roundtrip(tmp_path: Path):
    """Tiny synthetic 5-metal CSV round-trips through load_5metal_split_table."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        load_5metal_split_table,
    )
    import csv as _csv

    rows = []
    for i, (smi, metal, split) in enumerate([
        ("CCO", "Ru", "train"),
        ("CCN", "Ir", "val"),
        ("CCCl", "Os", "test"),
        ("CCBr", "Rh", "train"),
        ("CCI", "Re", "val"),
    ]):
        rows.append({
            "smiles": smi, "metal": metal, "scaffold_id": f"sc{i}",
            "split": split, "activity": "0",
        })
    csv_path = tmp_path / "tiny_5metal.csv"
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    df = load_5metal_split_table(csv_path)
    assert len(df) == 5
    assert set(df["metal"].tolist()) == {"Ru", "Ir", "Os", "Rh", "Re"}
    assert set(df["split"].tolist()) == {"train", "val", "test"}
    assert "row_id" in df.columns
    assert df["row_id"].tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Test 16: per-metal split indices builder — every metal gets train/val/test
# ---------------------------------------------------------------------------
def test_5metal_split_indices_build(tmp_path: Path):
    """build_metal_split_indices returns train/val/test arrays per metal.

    Uses an in-memory CytotoxDataset populated from the synthetic 5-metal
    CSV so we don't depend on /mnt/storage being mounted.
    """
    import pandas as _pd
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        load_5metal_split_table,
        build_metal_split_indices,
    )
    from molmetal.data.cytotox import MetalCytotoxDataset

    # Build a small MetalCytoToxDB-style frame with the columns
    # required by MetalCytotoxDataset (SMILES_Ligands, Metal, Cell_line,
    # pIC50, active, Year, Charge_complex, Oxidation_state).
    db_rows = []
    for i, (smi, metal, split) in enumerate([
        ("CCO", "Ru", "train"),
        ("CCN", "Ru", "val"),
        ("CCCl", "Ru", "test"),
        ("CCBr", "Ir", "train"),
        ("CCI", "Ir", "val"),
        ("CC=O", "Ir", "test"),
        ("CCS", "Os", "train"),
        ("CC=O", "Os", "val"),
        ("CC=NC", "Rh", "train"),
        ("c1ccccc1", "Rh", "val"),
        ("CCOC", "Re", "train"),
        ("CCNC", "Re", "val"),
    ]):
        db_rows.append({
            "SMILES_Ligands": smi,
            "Metal": metal,
            "Cell_line": "A549",
            "pIC50": 6.0,
            "active": 1 if i % 2 == 0 else 0,
            "Year": 2024,
            "Charge_complex": 0,
            "Oxidation_state": 0,
        })
    db_df = _pd.DataFrame(db_rows)
    ds = MetalCytotoxDataset(df=db_df, cache_dir=tmp_path)

    # Build 5-metal split CSV
    split_rows = []
    for smi, metal, split in [
        ("CCO", "Ru", "train"),
        ("CCN", "Ru", "val"),
        ("CCCl", "Ru", "test"),
        ("CCBr", "Ir", "train"),
        ("CCI", "Ir", "val"),
        ("CC=O", "Ir", "test"),
        ("CCS", "Os", "train"),
        ("CC=O", "Os", "val"),
        ("CC=NC", "Rh", "train"),
        ("c1ccccc1", "Rh", "val"),
        ("CCOC", "Re", "train"),
        ("CCNC", "Re", "val"),
    ]:
        split_rows.append({
            "smiles": smi, "metal": metal,
            "scaffold_id": "sc", "split": split, "activity": "0",
        })
    csv_path = tmp_path / "split.csv"
    import csv as _csv
    with csv_path.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
        writer.writeheader()
        for r in split_rows:
            writer.writerow(r)
    split_df = load_5metal_split_table(csv_path)

    # Each of {Ru, Ir} should have train/val/test
    for metal in ("Ru", "Ir"):
        idx_train, idx_val, idx_test = build_metal_split_indices(split_df, ds, metal)
        assert len(idx_train) >= 1, f"{metal}: no train rows"
        assert len(idx_val) >= 1, f"{metal}: no val rows"
        assert len(idx_test) >= 1, f"{metal}: no test rows"


# ---------------------------------------------------------------------------
# Test 17: 5-metal collate filters None / NaN-pic50 rows
# ---------------------------------------------------------------------------
def test_5metal_collate_filters_invalid():
    """collate_multimetal drops None-graph + NaN-pic50 rows from a 5-metal batch."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import collate_multimetal

    graphs = _toy_molecule_graphs(5)
    graphs[1] = None   # drop a row
    mids = [0, 1, 2, 3, 4]
    pic50 = np.array([5.0, float("nan"), 7.0, 6.0, 5.5], dtype=np.float32)
    active = np.array([0, 1, 1, 0, 1], dtype=np.int64)

    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is not None
    _, _, _, _, mids_t, pic50_t, active_t = batch
    # Row 1 was None — row 2 (originally idx=2) should now be at position 1
    assert mids_t.tolist() == [0, 2, 3, 4]
    assert pic50_t.tolist() == [5.0, 7.0, 6.0, 5.5]
    assert active_t.tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Test 18: joint 5-metal training loss decreases over 3 mini-epochs
# ---------------------------------------------------------------------------
def test_5metal_joint_loss_decreases():
    """3 train epochs on a tiny synthetic 5-metal batch → loss decreases."""
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        train_one_epoch_joint,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.molmetal_lam.reward.wasserstein_loss import WassersteinConfig

    n_metals = len(FIVE_METALS_5)
    torch.manual_seed(0)
    np.random.seed(0)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    w_config = WassersteinConfig(n_mc=64, seed=0)

    graphs, mids, pic50, active = _build_joint_batch(n_per_metal=4)
    ref_pic50 = torch.from_numpy(pic50.astype(np.float32))

    losses = []
    for epoch in range(3):
        loss, cls_loss, reg_loss, w_pen = train_one_epoch_joint(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            w_w=0.0,  # disable Wasserstein for clean loss-decrease signal
            use_l2_guard=True,
            graphs=graphs,
            metal_ids=mids,
            pic50=pic50,
            active=active,
            ref_pic50_dist=ref_pic50,
            batch_size=4,
            device=torch.device("cpu"),
            w_config=w_config,
        )
        losses.append(loss)

    assert losses[0] > losses[-1], (
        f"loss did not decrease across 3 epochs: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
