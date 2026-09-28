"""Tests for Wf-125 Phase 3 — 172-d feature stack + multi-task 4-cell backbone.

================================================================
Pins the Wf-125 Phase 3 contract for the **172-d feature stack +
multi-task 4-cell backbone + Wasserstein pIC50 regularization**
script.  Eighteen tests across six categories:

1. **Constants & shape** — INPUT_DIM = 172, PLATINAI cell-line keys,
   default cell lines.
2. **Backbone shapes** — shared trunk output dim, per-cell-line head
   output dim, head-to-trunk compatibility.
3. **Wasserstein loss** — sliced-W₂ pIC50 returns scalar + non-neg +
   decay check on fake history.
4. **172-d compute + cache** — `compute_172d_for_smiles` returns
   (172,) float32 vector for a known SMILES.
5. **PlatinAI load + cache** — `load_platinai_4cell_with_cache`
   returns X (N, 172), Y (N, 4), cell_y dict with 4 keys.
6. **K-fold splitting** — strict-scaffold disjoint + random K-fold
   invariants (no overlap, all rows covered, partition sizes).

All tests run on CPU in < 60 s.  No GPU dependencies required.

References
----------
* TODO/pending/122_platinai_4cell_r2_strengthen.md
* molmetal/scripts/build_172d_features.py (Wf-128 P1)
* molmetal/molmetal_lam/reward/wasserstein_loss.py (Wf-123 P1)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRIPTS = _REPO_ROOT / "molmetal/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import bench_hilbert_astr_172d_multitask as m  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Constants & shape (the Wf-125 P3 spec)
# ---------------------------------------------------------------------------
def test_input_dim_is_172() -> None:
    """The feature stack is **exactly** 172-d (12 RDKit + 32 Multi-PH + 128 PLL)."""
    assert m.INPUT_DIM == 172
    assert m.TRUNK_HIDDEN == 256
    assert m.TRUNK_LATENT == 128
    assert m.HEAD_HIDDEN == 64


def test_default_cell_lines_are_four_platinai_lines() -> None:
    """Default cell lines: mcf7, a549, a2780, a2780cis (4 lines)."""
    assert m.DEFAULT_CELL_LINES == ("mcf7", "a549", "a2780", "a2780cis")
    assert len(m.DEFAULT_CELL_LINES) == 4


def test_platinai_cell_columns_map_correctly() -> None:
    """Each PlatinAI cell line has a unique target column."""
    assert m.PLATINAI_CELL_COLS == {
        "mcf7": "ic50_mcf7_nm",
        "a549": "ic50_a549_nm",
        "a2780": "ic50_a2780_nm",
        "a2780cis": "ic50_a2780cis_nm",
    }
    assert len(set(m.PLATINAI_CELL_COLS.values())) == 4


def test_default_wasserstein_weight_matches_spec() -> None:
    """Wasserstein weight = 0.05 (Wf-123 P1 + Wf-125 P3 spec)."""
    assert abs(m.DEFAULT_WASSERSTEIN_WEIGHT - 0.05) < 1e-9


def test_cache_path_is_in_data_dir() -> None:
    """172-d cache lives under ``molmetal/data/``."""
    assert "molmetal/data/" in str(m.DEFAULT_CACHE_PATH)
    assert str(m.DEFAULT_CACHE_PATH).endswith(".npz")


# ---------------------------------------------------------------------------
# 2. Backbone shapes
# ---------------------------------------------------------------------------
def test_shared_trunk_output_shape() -> None:
    """Shared trunk output = (B, 128)."""
    trunk = m._SharedTrunk(
        input_dim=172, hidden_dim=256, latent_dim=128, dropout=0.2,
    )
    trunk.eval()
    x = torch.randn(8, 172)
    z = trunk(x)
    assert z.shape == (8, 128)


def test_per_cell_head_output_shape() -> None:
    """Per-cell-line head output = (B, 1)."""
    head = m._PerCellHead(in_dim=128, hidden_dim=64, dropout=0.2)
    head.eval()
    z = torch.randn(8, 128)
    y = head(z)
    assert y.shape == (8, 1)


def test_multitask_backbone_four_heads() -> None:
    """MultiTaskFourCellBackbone has exactly 4 per-cell-line heads."""
    model = m.MultiTaskFourCellBackbone()
    assert len(model.heads) == 4
    for cl in m.DEFAULT_CELL_LINES:
        assert cl in model.heads


def test_multitask_backbone_encode_then_head() -> None:
    """End-to-end backbone smoke: encode(8, 172) → 4 heads output (8, 1)."""
    model = m.MultiTaskFourCellBackbone()
    model.eval()
    x = torch.randn(8, 172)
    with torch.no_grad():
        z = model.encode(x)
        assert z.shape == (8, 128)
        for cl in m.DEFAULT_CELL_LINES:
            pred = model.head_forward(z, cl)
            assert pred.shape == (8, 1)


# ---------------------------------------------------------------------------
# 3. Wasserstein loss
# ---------------------------------------------------------------------------
def test_wasserstein_pic50_returns_scalar() -> None:
    """`_wasserstein_pic50_torch` returns a 0-dim tensor."""
    pred = torch.tensor([5.0, 5.5, 6.0])
    ref = torch.tensor([4.0, 5.0, 6.0, 7.0])
    out = m._wasserstein_pic50_torch(pred, ref, n_mc=128, seed=0)
    assert out.dim() == 0


def test_wasserstein_pic50_nonnegative() -> None:
    """W₂ distance is non-negative."""
    pred = torch.tensor([5.0, 5.5, 6.0])
    ref = torch.tensor([4.0, 5.0, 6.0, 7.0])
    out = m._wasserstein_pic50_torch(pred, ref, n_mc=64, seed=0)
    assert float(out.item()) >= 0.0


def test_wasserstein_decay_check_passes_on_decreasing_history() -> None:
    """A strictly decreasing history passes the decay check."""
    history = [10.0, 9.5, 9.1, 8.8, 8.5, 8.3, 8.1, 7.95, 7.82, 7.7]
    assert m.wasserstein_decay_check_helper(history, window=10, tol=1e-3) is True


def test_wasserstein_decay_check_fails_on_growing_history() -> None:
    """A growing history fails the decay check."""
    history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert m.wasserstein_decay_check_helper(history, window=10, tol=1e-3) is False


# ---------------------------------------------------------------------------
# 4. 172-d compute + cache
# ---------------------------------------------------------------------------
def test_compute_172d_returns_correct_shape() -> None:
    """`compute_172d_for_smiles` returns a (172,) float32 vector for CCO."""
    v = m.compute_172d_for_smiles("CCO")
    if v is None:
        pytest.skip("172-d compute failed on this host (RDKit/Multi-PH/PLL missing)")
    assert v.shape == (172,)
    assert v.dtype == np.float32


def test_compute_172d_handles_garbage_smiles() -> None:
    """Garbage SMILES → None (no exception)."""
    out = m.compute_172d_for_smiles("not_a_smiles_!!!")
    assert out is None
    out_empty = m.compute_172d_for_smiles("")
    assert out_empty is None


# ---------------------------------------------------------------------------
# 5. PlatinAI load + cache
# ---------------------------------------------------------------------------
def test_load_platinai_4cell_with_cache_shapes() -> None:
    """`load_platinai_4cell_with_cache` returns X (N, 172), Y (N, 4)."""
    cache_path = Path("molmetal/data/features_172d_platinai_4cell.npz")
    # Force rebuild to test the load path (it'll be cached after the first run).
    X, Y, smiles, cell_y = m.load_platinai_4cell_with_cache(
        m._REPO_ROOT,
        cache_path=cache_path,
        verbose=False,
    )
    assert X.ndim == 2 and X.shape[1] == 172
    assert Y.ndim == 2 and Y.shape[1] == 4
    assert X.shape[0] == Y.shape[0] == len(smiles)
    # cell_y has 4 keys.
    assert set(cell_y.keys()) == set(m.DEFAULT_CELL_LINES)
    for cl in m.DEFAULT_CELL_LINES:
        assert cell_y[cl].shape == (len(smiles),)


def test_load_platinai_4cell_cell_y_values_finite() -> None:
    """Per-cell-line pIC50 arrays have at least some finite values."""
    cache_path = Path("molmetal/data/features_172d_platinai_4cell.npz")
    X, Y, smiles, cell_y = m.load_platinai_4cell_with_cache(
        m._REPO_ROOT,
        cache_path=cache_path,
        verbose=False,
    )
    for cl in m.DEFAULT_CELL_LINES:
        n_finite = int(np.sum(np.isfinite(cell_y[cl])))
        assert n_finite > 1000, (
            f"Expected > 1000 finite pIC50 for {cl}; got {n_finite}"
        )


# ---------------------------------------------------------------------------
# 6. K-fold splitting
# ---------------------------------------------------------------------------
def test_random_kfold_no_overlap_and_complete() -> None:
    """Random K-fold: train/test are disjoint and partition [0, n)."""
    n = 100
    splits = m._random_kfold_indices(n, k=5, seed=0)
    assert len(splits) == 5
    seen_test: List[int] = []
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0  # disjoint
        assert len(tr) + len(te) == n  # complete
        seen_test.extend(te.tolist())
    # Every index appears in exactly one test fold.
    assert sorted(seen_test) == list(range(n))


def test_strict_scaffold_kfold_no_overlap_and_complete() -> None:
    """Strict-scaffold K-fold: each scaffold lands in exactly one test fold."""
    n = 50
    # 5 scaffolds × 10 rows each.
    scaffold_ids = [f"sc_{i // 10}" for i in range(n)]
    splits = m._strict_scaffold_kfold_indices(scaffold_ids, k=5, seed=0)
    assert len(splits) == 5
    seen_test: List[int] = []
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0  # disjoint
        assert len(tr) + len(te) == n  # complete
        seen_test.extend(te.tolist())
    assert sorted(seen_test) == list(range(n))
    # Each scaffold's rows should land in exactly one test fold —
    # collect, per scaffold, the set of folds that contain its rows.
    scaffold_to_folds: Dict[str, set] = {}
    for fold_idx, (_tr, te) in enumerate(splits):
        for idx in te.tolist():
            sc = scaffold_ids[idx]
            scaffold_to_folds.setdefault(sc, set()).add(fold_idx)
    unique_sc = set(scaffold_ids)
    assert set(scaffold_to_folds.keys()) == unique_sc
    # And each scaffold appears in exactly one test fold.
    for sc in unique_sc:
        assert len(scaffold_to_folds[sc]) == 1, (
            f"scaffold {sc} appears in {len(scaffold_to_folds[sc])} folds"
        )


def test_kfold_dispatcher_picks_strict_scaffold() -> None:
    """`_kfold_indices` selects strict-scaffold when `scaffold_ids` is supplied."""
    n = 30
    scaffold_ids = [f"sc_{i // 6}" for i in range(n)]
    splits = m._kfold_indices(
        n, k=3, seed=0,
        scaffold_ids=scaffold_ids, strict_scaffold=True,
    )
    assert len(splits) == 3


def test_bemis_murcko_scaffold_returns_string() -> None:
    """`_bemis_murcko_scaffold` returns a string (possibly empty)."""
    out = m._bemis_murcko_scaffold("CCO")
    assert isinstance(out, str)


def test_compute_r2_basic_invariant() -> None:
    """`_compute_r2` returns 1.0 on perfect prediction and 0.0 on mean predictor."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r2_perfect = m._compute_r2(y, y)
    assert abs(r2_perfect - 1.0) < 1e-9
    r2_mean = m._compute_r2(y, np.full_like(y, y.mean()))
    assert abs(r2_mean - 0.0) < 1e-9
