"""Tests for :mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`.

Wf-131 Phase 1 — 5-metal multi-task D-MPNN + Wasserstein pIC50 reg
=====================================================================

Verifies the helper functions, model class, CLI, and loss components
for the Wf-131 multitask Wasserstein trainer.

**Key invariants verified**
---------------------------
1. ``MultiTaskWassersteinModel`` has shared encoder + per-metal heads.
2. ``MultiTaskWassersteinModel.forward`` returns correct shapes.
3. ``MultiTaskWassersteinModel.encode_pool`` returns (B, 2H) per mol.
4. ``collate_multimetal`` filters None / NaN-pic50 rows.
5. ``collate_multimetal`` rejects empty / all-invalid batches.
6. ``_safe_classification_metrics`` returns NaN AUC for one-class input.
7. ``_safe_classification_metrics`` returns finite AUC for balanced input.
8. ``train_one_epoch_joint`` finite loss for tiny synthetic batch.
9. ``train_one_epoch_joint`` with w_w=0 collapses to baseline MSE+BCE.
10. ``train_one_epoch_joint`` with w_w>0 produces a finite Wasserstein term.
11. ``L2 guard`` (grad-norm clip) keeps gradients bounded under synthetic blow-up.
12. ``parse_args`` defaults to 5 metals × 5 seeds × 50 epochs.
13. ``parse_args`` accepts ``--use-l2-guard`` and ``--no-l2-guard`` flags.
14. ``parse_args`` rejects invalid metal names via argparse choices.
15. ``load_5metal_split_table`` round-trips a tiny synthetic CSV.
16. ``MultiTaskLoss(alpha=0.5)`` reproduces the Wf-120 P1 weights end-to-end.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _toy_graphs(n: int = 4) -> List:
    """Build n tiny synthetic graphs (single atom, no bonds)."""
    from molmetal.baselines.dmpnn import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

    out = []
    for _ in range(n):
        atom_f = np.random.RandomState(0).randn(1, ATOM_FEATURE_DIM).astype(np.float32)
        bond_f = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
        e_src = np.zeros((0,), dtype=np.int64)
        e_dst = np.zeros((0,), dtype=np.int64)
        out.append((atom_f, bond_f, e_src, e_dst))
    return out


def _toy_batch(n: int = 4, n_metals: int = 5):
    """Build a synthetic training batch (graphs, mids, pic50, active)."""
    graphs = _toy_graphs(n)
    mids = [i % n_metals for i in range(n)]
    pic50 = np.random.RandomState(42).uniform(3.0, 9.0, n).astype(np.float32)
    active = np.random.RandomState(43).randint(0, 2, n).astype(np.int64)
    return graphs, mids, pic50, active


# ---------------------------------------------------------------------------
# Test 1: MultiTaskWassersteinModel has shared encoder + per-metal heads
# ---------------------------------------------------------------------------
def test_mtw_model_has_shared_encoder_and_per_metal_heads():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        FIVE_METALS,
    )

    n_metals = len(FIVE_METALS)
    model = MultiTaskWassersteinModel(n_metals=n_metals)
    # Shared encoder
    assert hasattr(model, "encoder")
    # Per-metal heads (one reg + one cls per metal)
    assert len(model.reg_heads) == n_metals
    assert len(model.cls_heads) == n_metals


# ---------------------------------------------------------------------------
# Test 2: forward() returns correct shapes
# ---------------------------------------------------------------------------
def test_mtw_model_forward_shapes():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    n_metals = 5
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    model.eval()
    graphs, mids, pic50, active = _toy_batch(n=4, n_metals=n_metals)
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import collate_multimetal

    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is not None
    atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch
    with torch.no_grad():
        pic50_pred, cls_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
    assert pic50_pred.shape == (4,)
    assert cls_logits.shape == (4, 2)


# ---------------------------------------------------------------------------
# Test 3: encode_pool returns (B, 2H) per mol
# ---------------------------------------------------------------------------
def test_mtw_encode_pool_shape():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
    )

    n_metals = 5
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    model.eval()
    graphs, mids, pic50, active = _toy_batch(n=4, n_metals=n_metals)
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import collate_multimetal

    batch = collate_multimetal(graphs, mids, pic50, active)
    atom_f, bond_f, e_src, e_dst, _, _, _ = batch
    with torch.no_grad():
        h_pool = model.encode_pool(atom_f, bond_f, e_src, e_dst)
    assert h_pool.shape == (4, 2 * 64)


# ---------------------------------------------------------------------------
# Test 4: collate_multimetal filters None / NaN-pic50 rows
# ---------------------------------------------------------------------------
def test_collate_multimetal_filters_invalid():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import collate_multimetal

    graphs = _toy_graphs(3)
    graphs[1] = None  # invalid mol
    mids = [0, 1, 2]
    pic50 = np.array([5.0, float("nan"), 7.0], dtype=np.float32)
    active = np.array([0, 1, 1], dtype=np.int64)
    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is not None
    _, _, _, _, mids_t, pic50_t, active_t = batch
    assert mids_t.tolist() == [0, 2]
    assert pic50_t.tolist() == [5.0, 7.0]
    assert active_t.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Test 5: collate_multimetal rejects empty / all-invalid batches
# ---------------------------------------------------------------------------
def test_collate_multimetal_rejects_all_invalid():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import collate_multimetal

    # All None
    graphs = [None, None, None]
    mids = [0, 1, 2]
    pic50 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    active = np.array([0, 1, 1], dtype=np.int64)
    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is None

    # All NaN pic50
    graphs = _toy_graphs(3)
    pic50 = np.array([float("nan")] * 3, dtype=np.float32)
    batch = collate_multimetal(graphs, mids, pic50, active)
    assert batch is None


# ---------------------------------------------------------------------------
# Test 6: _safe_classification_metrics returns NaN AUC for one-class input
# ---------------------------------------------------------------------------
def test_safe_classification_metrics_one_class():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        _safe_classification_metrics,
    )

    y_true = np.array([1, 1, 1, 1], dtype=int)
    y_score = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    m = _safe_classification_metrics(y_true, y_score)
    assert math.isnan(m["roc_auc"])
    assert m["n"] == 4
    assert m["n_pos"] == 4
    assert m["n_neg"] == 0


# ---------------------------------------------------------------------------
# Test 7: _safe_classification_metrics returns finite AUC for balanced input
# ---------------------------------------------------------------------------
def test_safe_classification_metrics_balanced():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        _safe_classification_metrics,
    )

    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_score = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)
    m = _safe_classification_metrics(y_true, y_score)
    assert m["roc_auc"] == 1.0
    assert m["n"] == 4
    assert m["n_pos"] == 2
    assert m["n_neg"] == 2


# ---------------------------------------------------------------------------
# Test 8: train_one_epoch_joint produces finite loss
# ---------------------------------------------------------------------------
def test_train_one_epoch_joint_finite():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        train_one_epoch_joint,
        FIVE_METALS,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.molmetal_lam.reward.wasserstein_loss import WassersteinConfig

    n_metals = len(FIVE_METALS)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    w_config = WassersteinConfig(n_mc=64, seed=0)

    graphs, mids, pic50, active = _toy_batch(n=8, n_metals=n_metals)
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
    assert np.isfinite(loss)
    assert np.isfinite(cls_loss)
    assert np.isfinite(reg_loss)
    assert np.isfinite(w_pen)
    assert w_pen >= 0.0  # W2 is non-negative


# ---------------------------------------------------------------------------
# Test 9: train_one_epoch_joint with w_w=0 collapses to baseline
# ---------------------------------------------------------------------------
def test_train_one_epoch_joint_w_w_zero():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        train_one_epoch_joint,
        FIVE_METALS,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.molmetal_lam.reward.wasserstein_loss import WassersteinConfig

    n_metals = len(FIVE_METALS)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    w_config = WassersteinConfig(n_mc=64, seed=0)

    graphs, mids, pic50, active = _toy_batch(n=8, n_metals=n_metals)
    ref_pic50 = torch.from_numpy(pic50.astype(np.float32))

    # w_w=0 → no Wasserstein penalty
    _, _, _, w_pen = train_one_epoch_joint(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        w_w=0.0,
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
    # When w_w=0 the wasserstein contribution should be zero (or near it)
    assert w_pen == 0.0


# ---------------------------------------------------------------------------
# Test 10: train_one_epoch_joint with w_w>0 produces a finite Wasserstein term
# ---------------------------------------------------------------------------
def test_train_one_epoch_joint_w_w_positive():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        train_one_epoch_joint,
        FIVE_METALS,
    )
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss
    from molmetal.molmetal_lam.reward.wasserstein_loss import WassersteinConfig

    n_metals = len(FIVE_METALS)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(alpha=0.5)
    w_config = WassersteinConfig(n_mc=64, seed=0)

    graphs, mids, pic50, active = _toy_batch(n=8, n_metals=n_metals)
    # Reference distribution is shifted → W2 > 0
    ref_pic50 = torch.from_numpy((pic50 + 3.0).astype(np.float32))

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
    assert w_pen > 0.0
    assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 11: L2 guard keeps gradients bounded under synthetic blow-up
# ---------------------------------------------------------------------------
def test_l2_guard_keeps_gradients_bounded():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        MultiTaskWassersteinModel,
        FIVE_METALS,
    )
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        L2_GUARD_MAX_NORM,
    )

    n_metals = len(FIVE_METALS)
    model = MultiTaskWassersteinModel(n_metals=n_metals, hidden=64)
    # Force a huge gradient
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e6)
    pre_norm = float(
        torch.norm(
            torch.stack(
                [p.grad.norm() for p in model.parameters() if p.grad is not None]
            )
        )
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), L2_GUARD_MAX_NORM)
    post_norm = float(
        torch.norm(
            torch.stack(
                [p.grad.norm() for p in model.parameters() if p.grad is not None]
            )
        )
    )
    # post_norm is the L2 norm of the *concatenated* gradient vector — clip
    # ensures its scalar norm is ≤ max_norm; the per-tensor norms may differ.
    assert pre_norm > L2_GUARD_MAX_NORM  # Sanity: pre-clip norm was huge
    # Allow numerical slack (≤ 1e-3) since each per-tensor norm is reported
    # independently and rounding can inflate the stacked L2 norm slightly.
    assert post_norm <= L2_GUARD_MAX_NORM + 1e-3


# ---------------------------------------------------------------------------
# Test 12: parse_args defaults to 5 metals × 5 seeds × 50 epochs
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        parse_args,
        FIVE_METALS,
        DEFAULT_SEEDS,
        DEFAULT_EPOCHS,
    )

    args = parse_args([])
    assert args.metals == list(FIVE_METALS)
    assert args.seeds == list(DEFAULT_SEEDS)
    assert args.epochs == DEFAULT_EPOCHS
    assert args.alpha == 0.5
    assert args.wasserstein_weight == 0.05


# ---------------------------------------------------------------------------
# Test 13: parse_args accepts --use-l2-guard and --no-l2-guard flags
# ---------------------------------------------------------------------------
def test_parse_args_l2_guard_flags():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    # default: guard ON
    args_on = parse_args([])
    assert args_on.use_l2_guard is True

    # --no-l2-guard: guard OFF
    args_off = parse_args(["--no-l2-guard"])
    assert args_off.use_l2_guard is False


# ---------------------------------------------------------------------------
# Test 14: parse_args rejects invalid metal names
# ---------------------------------------------------------------------------
def test_parse_args_invalid_metal():
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    with pytest.raises(SystemExit):
        # "Fe" is not in {Ru, Ir, Rh, Os, Re}
        parse_args(["--metals", "Fe"])


# ---------------------------------------------------------------------------
# Test 15: load_5metal_split_table round-trips a tiny synthetic CSV
# ---------------------------------------------------------------------------
def test_load_5metal_split_table_roundtrip(tmp_path: Path):
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
        load_5metal_split_table,
    )

    rows = [
        {
            "smiles": "CCO",
            "metal": "Ru",
            "scaffold_id": "scA",
            "split": "train",
            "activity": "0",
        },
        {
            "smiles": "CCN",
            "metal": "Ir",
            "scaffold_id": "scB",
            "split": "val",
            "activity": "1",
        },
        {
            "smiles": "CCCl",
            "metal": "Os",
            "scaffold_id": "scC",
            "split": "test",
            "activity": "0",
        },
    ]
    csv_path = tmp_path / "tiny.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    df = load_5metal_split_table(csv_path)
    assert len(df) == 3
    assert set(df.columns) >= {
        "smiles",
        "metal",
        "scaffold_id",
        "split",
        "activity",
        "row_id",
    }
    assert df["metal"].tolist() == ["Ru", "Ir", "Os"]
    assert df["row_id"].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Test 16: MultiTaskLoss(alpha=0.5) reproduces the Wf-120 P1 weights end-to-end
# ---------------------------------------------------------------------------
def test_multitask_loss_alpha_half_weights():
    from molmetal.baselines.dmpnn_multitask import MultiTaskLoss

    criterion = MultiTaskLoss(alpha=0.5)
    pic50_pred = torch.tensor([5.0, 7.0], dtype=torch.float32)
    active_logits = torch.tensor([[1.0, 0.5], [0.2, 1.5]])
    pic50_t = torch.tensor([5.5, 6.5], dtype=torch.float32)
    active_t = torch.tensor([0, 1], dtype=torch.long)

    loss, lc, lr_ = criterion(pic50_pred, active_logits, pic50_t, active_t)
    # Expected cls and reg losses separately
    expected_cls = float(
        torch.nn.functional.cross_entropy(active_logits, active_t).item()
    )
    expected_reg = float(torch.nn.functional.mse_loss(pic50_pred, pic50_t).item())
    expected_total = 0.5 * expected_cls + 0.5 * expected_reg

    assert abs(loss.item() - expected_total) < 1e-5
    assert abs(lc.item() - expected_cls) < 1e-5
    assert abs(lr_.item() - expected_reg) < 1e-5
