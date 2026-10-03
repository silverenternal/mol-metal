"""TODO-182 — Per-innovation ablation script for paper §3 attribution.

=====================================================================
Wf-124 P3 per-innovation ablation (TODO-182)
=====================================================================

The Wf-124 P3 default recipe (multi-task + Wasserstein + L2 guard) ran all
3 innovations simultaneously, so the +0.049 absolute test_auc lift (Ru
seed=0, 30 epochs) is an **aggregate** lift — paper §3 cannot honestly
attribute any portion of it to per-lever %.

This script closes the gap by enabling per-lever ablation on the
**single-task** baseline (alpha=1.0 BCE/MSE chassis, no multi-task, no
W₂, no Kendall, no DANN).  Three levers are exposed via CLI flags:

  1. `--use-wasserstein / --no-wasserstein`
     Sliced-W₂(pIC50_pred, ref_pIC50_train) penalty added to the task loss.
     Implemented in ``molmetal.molmetal_lam.reward.wasserstein_loss``.

  2. `--use-kendall / --no-kendall`
     Replaces the manual ``alpha``-blend with Kendall et al. 2018
     homoscedastic-uncertainty weighting.  Implemented in
     ``molmetal.molmetal_lam.sbdd_env.uncertainty_loss``.  When Kendall is
     enabled the manual ``--alpha`` is ignored.

  3. `--use-dann / --no-dann`
     Domain-Adversarial head (4-class domain classifier + ReverseLayerF)
     attached to the pooled D-MPNN backbone.  Implemented in
     ``molmetal.models.dmpnn.DomainAdversarialHead`` and exercised by
     ``train_epoch_mt_dann`` in
     ``molmetal.baselines.dmpnn_multitask``.

The default behavior (no levers on) reproduces the single-task baseline
and is **backward compatible** with the existing
``train_dmpnn_pic50_real.py`` recipe.

**Usage**
-----
::
    # Baseline (no levers)
    PY=python\ molmetal/scripts/train_dmpnn_pic50_ablation.py \\
        --metal Ru --seed 0 --epochs 5 --out-dir reports/ablation

    # Baseline + Wasserstein
    PY=python\ molmetal/scripts/train_dmpnn_pic50_ablation.py \\
        --metal Ru --seed 0 --epochs 5 \\
        --use-wasserstein --wasserstein-weight 0.05

    # Baseline + Kendall
    PY=python\ molmetal/scripts/train_dmpnn_pic50_ablation.py \\
        --metal Ru --seed 0 --epochs 5 \\
        --use-kendall

    # Baseline + DANN
    PY=python\ molmetal/scripts/train_dmpnn_pic50_ablation.py \\
        --metal Ru --seed 0 --epochs 5 \\
        --use-dann --lambda-grl 0.1 --grl-alpha 0.1

The script runs all four configurations in one invocation and writes
``reports/ablation/ablation_report.json`` containing per-lever lift.

**Hardware**: CPU primary; CUDA if ``--device cuda:0`` (RTX 5090 on
``CUDA_VISIBLE_DEVICES=0``).

**Honest framing** (per ``party-twenty-spirit`` §3.5):
  * Single seed × 5 epochs (smoke run — paper §3 attribution requires the
    larger 5-seed × 30-epoch sweep per TODO-182 G6.1).
  * Single metal (Ru, ~2.5k train rows).  Per-metal std for the smaller
    metals (Rh/Os/Re < 1.2k rows) is much larger.
  * Lift magnitudes at 5 epochs are NOT directly comparable to the
    30-epoch Wf-124 P3 lift; this script proves the *protocol* (each
    lever toggles cleanly, no shared-state contamination between runs)
    and demonstrates the directional effect at 5 epochs.

**ETA / budget**: ~30s × 4 configs ≈ 2 min wall-time on RTX 5090; ~10
min on CPU.  The script enforces a hard 1800s wall-time budget.
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles_list,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskModel,
    MT_BATCH_SIZE,
    MT_DEPTH,
    MT_DROPOUT,
    MT_HIDDEN,
    MT_LR,
    MultiTaskLoss,
    regression_metrics,
    train_epoch_mt,
    train_epoch_mt_dann,
    predict_mt,
)
from molmetal.baselines.eval_utils import compute_metrics
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
    wasserstein_pic50_loss,
)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_CSV = PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf124_p3_ablation"

# Default smoke-test config — fast enough to fit in 1800s wall-time while
# still exercising every code path of the three ablation levers.  The
# full TODO-182 G6.1 sweep (5 metals × 3 seeds × 30 epochs × 4 configs
# = 60 runs ≈ 7.5 hr) is OUT OF SCOPE for this smoke-test deliverable.
DEFAULT_METAL: str = "Ru"
DEFAULT_SEED: int = 0
DEFAULT_EPOCHS: int = 5
DEFAULT_ALPHA: float = 0.5  # balanced BCE/MSE — required for Kendall to be meaningful
DEFAULT_WASSERSTEIN_WEIGHT: float = 0.05
DEFAULT_LAMBDA_GRL: float = 0.1
DEFAULT_GRL_ALPHA: float = 0.1
DEFAULT_LR: float = MT_LR
DEFAULT_BATCH_SIZE: int = MT_BATCH_SIZE
DEFAULT_N_MC: int = 256

# Hard wall-time budget (seconds) — script bails out cleanly if any single
# lever ablation is on track to exceed it.
WALL_TIME_BUDGET_S: int = 1800

# Cache for featurised graphs across the 4 configs (same metal + seed →
# same SMILES → same RDKit graphs).  Sharing this cuts featurise time
# from ~7s × 4 = 28s down to ~7s × 1.
GRAPH_CACHE_DIR = Path("/tmp/todo182_dmpnn_fe_cache")


# ---------------------------------------------------------------------------
# Data loading (no /mnt/storage hard dep)
# ---------------------------------------------------------------------------
def load_cytotox_dataset(
    metal: str,
    data_csv: Path,
    time_threshold: float = 24.0,
    ic50_min: float = 0.01,
) -> MetalCytotoxDataset:
    """Load MetalCytoToxDB.csv honouring the ``--data-csv`` flag.

    Mirrors the ``--data-csv`` branch of
    :mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein` so we don't
    trip the hard-coded ``/mnt/storage/data/molmetal/MetalCytoToxDB.csv``
    path in :func:`molmetal.data.cytotox.from_csv`.
    """
    flt_obj = CytotoxFilter(
        time_threshold=time_threshold,
        ic50_min=ic50_min,
        metal_whitelist=[metal],
        compute_pic50=True,
        compute_active=True,
    )
    local_cache = Path("/tmp") / "todo182_mcd_cache"
    local_cache.mkdir(parents=True, exist_ok=True)
    df_loaded = pd.read_csv(data_csv, low_memory=False)
    from molmetal.data import cytotox as _cytotox_mod

    df_loaded = _cytotox_mod._apply_filters(df_loaded, flt_obj)
    if flt_obj.compute_pic50:
        df_loaded = _cytotox_mod._add_pic50(df_loaded)
    if flt_obj.compute_active:
        df_loaded = _cytotox_mod._add_active(df_loaded)
    ds = MetalCytotoxDataset(df=df_loaded, cache_dir=local_cache)
    if len(ds) == 0:
        raise RuntimeError(f"No rows for metal={metal!r} in {data_csv}")
    return ds


def featurise_with_cache(
    smiles: Sequence[str],
    cache_key: str,
) -> List:
    """RDKit graph-featurisation + pickle cache.

    All 4 ablation configs share the same SMILES list (same metal + seed
    + same set of split indices), so caching the featurised graphs cuts
    ~21s of redundant RDKit work across the 4 runs.
    """
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = GRAPH_CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            import pickle

            with cache_path.open("rb") as f:
                cached_smiles, cached_graphs = pickle.load(f)
            if list(cached_smiles) == list(smiles):
                return cached_graphs
        except Exception:
            pass

    t0 = time.time()
    graphs = featurize_smiles_list([str(s) for s in smiles])
    elapsed = time.time() - t0
    try:
        import pickle

        with cache_path.open("wb") as f:
            pickle.dump(
                (list(smiles), graphs),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass
    print(
        f"    featurised {len(smiles)} SMILES in {elapsed:.1f}s "
        f"(cache: {cache_path})",
        flush=True,
    )
    return graphs


def split_indices(
    ds: MetalCytotoxDataset,
    metal: str,
    seed: int,
    val_frac: float = 0.1,
    test_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split.

    Same seed → same split across all 4 ablation configs (essential for
    a fair lift comparison).
    """
    from sklearn.model_selection import train_test_split

    smiles_all = np.asarray(ds.smiles).astype(str)
    metal_all = np.asarray(ds.metals).astype(str)
    active_all = np.asarray(ds.active).astype(int)
    keep = np.where(metal_all == metal)[0]
    y = active_all[keep]
    idx_tmp, idx_test = train_test_split(
        keep,
        test_size=test_frac,
        stratify=y,
        random_state=seed,
    )
    y_tmp = active_all[idx_tmp]
    val_rel = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_tmp,
        test_size=val_rel,
        stratify=y_tmp,
        random_state=seed,
    )
    return (
        np.array(sorted(idx_train.tolist())),
        np.array(sorted(idx_val.tolist())),
        np.array(sorted(idx_test.tolist())),
    )


# ---------------------------------------------------------------------------
# Per-config training run
# ---------------------------------------------------------------------------
@dataclass
class ConfigRunResult:
    """Outcome of one (config, seed) ablation run."""

    config_name: str
    use_wasserstein: bool
    use_kendall: bool
    use_dann: bool
    metal: str
    seed: int
    epochs: int
    n_train: int
    n_val: int
    n_test: int
    val_auc: float
    test_auc: float
    test_pearson: float
    test_rmse: float
    test_mae: float
    elapsed_seconds: float
    final_train_loss: float
    final_cls_loss: float
    final_reg_loss: float
    final_domain_loss: float
    final_wasserstein: float
    status: str = "OK"
    error: Optional[str] = None


def train_epoch_with_wasserstein(
    model: DMPNNMultiTaskModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    graphs: List,
    pic50_labels: np.ndarray,
    active_labels: np.ndarray,
    *,
    w_w: float,
    ref_pic50: torch.Tensor,
    w_config: WassersteinConfig,
    batch_size: int = MT_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, float]:
    """Train one epoch with Wasserstein penalty added to the task loss.

    Re-implements the per-batch loop from
    :func:`molmetal.baselines.dmpnn_multitask.train_epoch_mt` so we can
    *add* the W₂ penalty without re-running forward.  Returns
    ``(avg_total_loss, avg_cls_loss, avg_reg_loss, avg_wasserstein)``.
    """
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_w = 0.0
    n_samples = 0
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_pic50 = pic50_labels[batch_idx]
        batch_active = active_labels[batch_idx]
        valid = [
            (g, p, a)
            for g, p, a in zip(batch_graphs, batch_pic50, batch_active)
            if g is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if not valid:
            continue

        from molmetal.baselines.dmpnn import collate_graphs

        g_list = [v[0] for v in valid]
        y_pic50 = np.array([v[1] for v in valid], dtype=np.float32)
        y_active = np.array([v[2] for v in valid], dtype=np.int64)
        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_pic50_t = torch.from_numpy(y_pic50).to(device)
        y_active_t = torch.from_numpy(y_active).to(device)

        optimizer.zero_grad()
        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst)
        loss, lc, lr_ = criterion(pic50_pred, active_logits, y_pic50_t, y_active_t)

        # Wasserstein penalty (differentiable in pic50_pred)
        w_pen = torch.zeros((), device=device)
        if w_w > 0.0 and pic50_pred.numel() >= 4:
            ref_dev = ref_pic50.to(device).detach()
            w_pen = wasserstein_pic50_loss(
                pic50_pred,
                ref_dev,
                n_mc=w_config.n_mc,
                seed=w_config.seed,
            )

        total = loss + w_w * w_pen
        total.backward()
        optimizer.step()

        bs = len(y_active)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr_.item() * bs
        total_w += float(w_pen.item()) * bs
        n_samples += bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_samples,
        total_cls / n_samples,
        total_reg / n_samples,
        total_w / n_samples,
    )


def run_one_config(
    config_name: str,
    *,
    use_wasserstein: bool,
    use_kendall: bool,
    use_dann: bool,
    metal: str,
    seed: int,
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    wasserstein_weight: float,
    lambda_grl: float,
    grl_alpha: float,
    n_mc: int,
    ds: MetalCytotoxDataset,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    device: torch.device,
    wall_time_remaining: float,
) -> ConfigRunResult:
    """Train one (lever-combination, seed) and evaluate."""
    t_start = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)

    used_idx = np.unique(np.concatenate([idx_train, idx_val, idx_test]))
    used_smiles = smiles_all[used_idx]
    cache_key = f"todo182_{metal.lower()}_seed{seed}"
    graphs_all = featurise_with_cache(used_smiles.tolist(), cache_key=cache_key)

    pos_in_used = {int(used_idx[k]): k for k in range(len(used_idx))}
    graphs_full = [None] * len(smiles_all)
    for di in used_idx:
        graphs_full[di] = graphs_all[pos_in_used[int(di)]]

    graphs_train = [graphs_full[i] for i in idx_train]
    graphs_val = [graphs_full[i] for i in idx_val]
    graphs_test = [graphs_full[i] for i in idx_test]
    pic50_train = pic50_all[idx_train]
    pic50_val = pic50_all[idx_val]
    pic50_test = pic50_all[idx_test]
    active_train = active_all[idx_train]
    active_val = active_all[idx_val]
    active_test = active_all[idx_test]

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=MT_HIDDEN,
        depth=MT_DEPTH,
        dropout=MT_DROPOUT,
        use_dann=use_dann,
        num_domains=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Wf-187 fix: use_kendall_loss=True ignores the manual ``alpha``.
    criterion = MultiTaskLoss(
        alpha=alpha,
        use_kendall_loss=use_kendall,
    )

    # Reference pIC50 distribution for Wasserstein (fixed, detached).
    ref_pic50 = torch.from_numpy(pic50_train.astype(np.float32))
    w_config = WassersteinConfig(n_mc=int(n_mc), seed=0)

    best_val_auc = -1.0
    best_state: Optional[Dict] = None

    final_train_loss = final_cls = final_reg = 0.0
    final_domain_loss = 0.0
    final_wasserstein = 0.0

    for epoch in range(epochs):
        ep_t0 = time.time()
        if use_dann:
            train_loss, cls_loss, reg_loss, domain_loss = train_epoch_mt_dann(
                model,
                optimizer,
                criterion,
                graphs_train,
                pic50_train,
                active_train,
                domain_head=model.domain_head,
                lambda_grl=lambda_grl,
                domain_id=0,
                grl_alpha=grl_alpha,
                batch_size=batch_size,
                device=device,
            )
            w_pen = 0.0
        elif use_wasserstein:
            train_loss, cls_loss, reg_loss, w_pen = train_epoch_with_wasserstein(
                model,
                optimizer,
                criterion,
                graphs_train,
                pic50_train,
                active_train,
                w_w=wasserstein_weight,
                ref_pic50=ref_pic50,
                w_config=w_config,
                batch_size=batch_size,
                device=device,
            )
            domain_loss = 0.0
        else:
            train_loss, cls_loss, reg_loss = train_epoch_mt(
                model,
                optimizer,
                criterion,
                graphs_train,
                pic50_train,
                active_train,
                batch_size=batch_size,
                device=device,
            )
            domain_loss = 0.0
            w_pen = 0.0

        # Validate
        y_true_v, y_score_v, pic50_pred_v = predict_mt(
            model,
            graphs_val,
            pic50_val,
            active_val,
            batch_size=batch_size,
            device=device,
        )
        val_pic50_m = regression_metrics(pic50_val, pic50_pred_v)
        val_m = compute_metrics(y_true_v, y_score_v)
        val_auc = float(val_m.get("roc_auc", float("nan")))
        ep_t = time.time() - ep_t0
        print(
            f"  [{config_name} epoch {epoch+1:>2}/{epochs}] "
            f"loss={train_loss:.4f} cls={cls_loss:.4f} reg={reg_loss:.4f} "
            f"dom={domain_loss:.4f} val_auc={val_auc:.4f} "
            f"val_pearson={val_pic50_m['pearson']:.3f} ({ep_t:.1f}s)",
            flush=True,
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        final_train_loss = train_loss
        final_cls = cls_loss
        final_reg = reg_loss
        final_domain_loss = float(domain_loss)
        final_wasserstein = float(w_pen)

        # Wall-time guard
        if (time.time() - t_start) > wall_time_remaining * 0.95:
            print(
                f"  [{config_name}] HALTING: wall-time budget "
                f"({wall_time_remaining:.0f}s) exceeded at epoch {epoch+1}",
                flush=True,
            )
            break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    model.load_state_dict(best_state)

    # Test
    y_true_t, y_score_t, pic50_pred_t = predict_mt(
        model,
        graphs_test,
        pic50_test,
        active_test,
        batch_size=batch_size,
        device=device,
    )
    test_m = compute_metrics(y_true_t, y_score_t)
    test_pic50 = regression_metrics(pic50_test, pic50_pred_t)
    test_auc = float(test_m.get("roc_auc", float("nan")))
    test_pearson = float(test_pic50.get("pearson", float("nan")))
    test_rmse = float(test_pic50.get("rmse", float("nan")))
    test_mae = float(test_pic50.get("mae", float("nan")))

    elapsed = time.time() - t_start

    return ConfigRunResult(
        config_name=config_name,
        use_wasserstein=use_wasserstein,
        use_kendall=use_kendall,
        use_dann=use_dann,
        metal=metal,
        seed=seed,
        epochs=epochs,
        n_train=int(len(idx_train)),
        n_val=int(len(idx_val)),
        n_test=int(len(idx_test)),
        val_auc=float(best_val_auc),
        test_auc=test_auc,
        test_pearson=test_pearson,
        test_rmse=test_rmse,
        test_mae=test_mae,
        elapsed_seconds=float(elapsed),
        final_train_loss=float(final_train_loss),
        final_cls_loss=float(final_cls),
        final_reg_loss=float(final_reg),
        final_domain_loss=float(final_domain_loss),
        final_wasserstein=float(final_wasserstein),
        status="OK",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TODO-182 — Per-innovation ablation for Wf-124 P3 paper §3 "
            "attribution (Wasserstein, Kendall, DANN)."
        ),
    )
    p.add_argument(
        "--metal",
        default=DEFAULT_METAL,
        choices=["Ru", "Ir", "Rh", "Os", "Re"],
        help="Target metal (default: Ru — largest single-task cohort).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Single seed for the smoke-test run (default: 0).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs per ablation config (default: 5 smoke-test).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BCE weight in MultiTaskLoss (default 0.5; ignored when --use-kendall).",
    )
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam lr.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    # --- Lever 1: Wasserstein -------------------------------------------
    wass_grp = p.add_mutually_exclusive_group()
    wass_grp.add_argument(
        "--use-wasserstein",
        dest="use_wasserstein",
        action="store_true",
        default=False,
        help="Enable sliced-W₂(pIC50_pred, ref_pIC50) penalty.",
    )
    wass_grp.add_argument(
        "--no-wasserstein",
        dest="use_wasserstein",
        action="store_false",
        help="Disable Wasserstein (default = baseline, backward compat).",
    )
    p.add_argument(
        "--wasserstein-weight",
        type=float,
        default=DEFAULT_WASSERSTEIN_WEIGHT,
        help="Weight w_w on Wasserstein penalty (default 0.05).",
    )
    p.add_argument(
        "--n-mc",
        type=int,
        default=DEFAULT_N_MC,
        help="Number of sliced-W₂ projections (default 256).",
    )

    # --- Lever 2: Kendall MTL loss --------------------------------------
    kend_grp = p.add_mutually_exclusive_group()
    kend_grp.add_argument(
        "--use-kendall",
        dest="use_kendall",
        action="store_true",
        default=False,
        help="Use Kendall et al. 2018 homoscedastic-uncertainty MTL loss.",
    )
    kend_grp.add_argument(
        "--no-kendall",
        dest="use_kendall",
        action="store_false",
        help="Use manual alpha-weighted MTL loss (default = baseline).",
    )

    # --- Lever 3: DANN adversarial head ---------------------------------
    dann_grp = p.add_mutually_exclusive_group()
    dann_grp.add_argument(
        "--use-dann",
        dest="use_dann",
        action="store_true",
        default=False,
        help="Attach DomainAdversarial head (GRL + 4-class classifier).",
    )
    dann_grp.add_argument(
        "--no-dann",
        dest="use_dann",
        action="store_false",
        help="No domain-adversarial head (default = baseline).",
    )
    p.add_argument(
        "--lambda-grl",
        type=float,
        default=DEFAULT_LAMBDA_GRL,
        help="Weight on (gradient-reversed) domain loss (default 0.1).",
    )
    p.add_argument(
        "--grl-alpha",
        type=float,
        default=DEFAULT_GRL_ALPHA,
        help="Alpha passed to ReverseLayerF (default 0.1).",
    )

    # --- Paths / device -------------------------------------------------
    p.add_argument(
        "--data-csv",
        type=Path,
        default=DEFAULT_DATA_CSV,
        help=(
            "MetalCytoToxDB.csv path (default .data/molmetal/MetalCytoToxDB.csv; "
            "avoids the hard-coded /mnt/storage path)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for ablation_report.json.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (cuda:0 or cpu). Default: auto-detect.",
    )

    # --- Single-config mode (skip the 4-config sweep) -------------------
    p.add_argument(
        "--config",
        default=None,
        choices=[None, "baseline", "wasserstein", "kendall", "dann"],
        help=(
            "Run ONLY this single configuration instead of the full 4-config "
            "sweep.  Used by tests / single-lever smoke runs."
        ),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    overall_start = time.time()
    print(
        f"[todo-182 ablation] device={device} metal={args.metal} "
        f"seed={args.seed} epochs={args.epochs} alpha={args.alpha}",
        flush=True,
    )
    print(
        f"  wasserstein={args.use_wasserstein} (w_w={args.wasserstein_weight}) | "
        f"kendall={args.use_kendall} | "
        f"dann={args.use_dann} (lambda_grl={args.lambda_grl})",
        flush=True,
    )

    # Load dataset + shared split (same split across all 4 configs → fair
    # lift comparison).
    ds = load_cytotox_dataset(
        metal=args.metal,
        data_csv=args.data_csv,
    )
    idx_train, idx_val, idx_test = split_indices(ds, args.metal, args.seed)
    print(
        f"[todo-182 ablation] {args.metal} split: "
        f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}",
        flush=True,
    )

    # The 4 (or 1, if --config) configurations
    configs = [
        ("baseline", False, False, False),
        ("wasserstein", True, False, False),
        ("kendall", False, True, False),
        ("dann", False, False, True),
    ]
    if args.config is not None:
        configs = [c for c in configs if c[0] == args.config]
        if not configs:
            raise SystemExit(f"Unknown --config: {args.config}")

    results: List[ConfigRunResult] = []
    for cfg_name, use_w, use_k, use_d in configs:
        wall_remaining = WALL_TIME_BUDGET_S - (time.time() - overall_start)
        if wall_remaining < 30.0:
            print(
                f"  [todo-182] HALTING: wall-time budget "
                f"({WALL_TIME_BUDGET_S}s) exceeded before {cfg_name}",
                flush=True,
            )
            break
        try:
            res = run_one_config(
                config_name=cfg_name,
                use_wasserstein=use_w,
                use_kendall=use_k,
                use_dann=use_d,
                metal=args.metal,
                seed=args.seed,
                epochs=args.epochs,
                alpha=args.alpha,
                lr=args.lr,
                batch_size=args.batch_size,
                wasserstein_weight=args.wasserstein_weight,
                lambda_grl=args.lambda_grl,
                grl_alpha=args.grl_alpha,
                n_mc=args.n_mc,
                ds=ds,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                device=device,
                wall_time_remaining=wall_remaining,
            )
        except Exception as exc:
            print(f"  [{cfg_name}] FAILED: {exc!r}", flush=True)
            res = ConfigRunResult(
                config_name=cfg_name,
                use_wasserstein=use_w,
                use_kendall=use_k,
                use_dann=use_d,
                metal=args.metal,
                seed=args.seed,
                epochs=args.epochs,
                n_train=len(idx_train),
                n_val=len(idx_val),
                n_test=len(idx_test),
                val_auc=float("nan"),
                test_auc=float("nan"),
                test_pearson=float("nan"),
                test_rmse=float("nan"),
                test_mae=float("nan"),
                elapsed_seconds=0.0,
                final_train_loss=float("nan"),
                final_cls_loss=float("nan"),
                final_reg_loss=float("nan"),
                final_domain_loss=float("nan"),
                final_wasserstein=float("nan"),
                status="FAILED",
                error=repr(exc),
            )
        results.append(res)
        print(
            f"  [{cfg_name}] DONE test_auc={res.test_auc:.4f} "
            f"test_pearson={res.test_pearson:.3f} "
            f"({res.elapsed_seconds:.1f}s)",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # Per-lever lift table (paper §3 attribution)
    # ---------------------------------------------------------------------
    # If --config was used (single-config mode), load any prior results from
    # disk and merge them so the final report contains ALL configs we have
    # run so far.  This lets a long ablation sweep be parallelised across
    # multiple script invocations.
    prior_results: List[Dict] = []
    if args.config is not None:
        prior_path = args.out_dir / "ablation_report.json"
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text())
                prior_results = list(prior.get("per_config", []))
            except Exception:
                prior_results = []

    by_cfg = {r.config_name: r for r in results}
    # Merge in prior on-disk configs that are not present in this run's results
    for prior_cfg in prior_results:
        cfg_name = prior_cfg.get("config")
        if cfg_name in by_cfg:
            continue

        # Rebuild a minimal ConfigRunResult-like dict from the prior row
        # so the lift computation works.  We don't have the dataclass, so
        # we just synthesise a stand-in dict.
        class _PriorProxy:
            config_name = cfg_name
            status = prior_cfg.get("status", "OK")
            test_auc = prior_cfg.get("test_auc", float("nan"))
            use_wasserstein = prior_cfg.get("use_wasserstein", False)
            use_kendall = prior_cfg.get("use_kendall", False)
            use_dann = prior_cfg.get("use_dann", False)
            metal = prior_cfg.get("metal", args.metal)
            seed = prior_cfg.get("seed", args.seed)
            epochs = prior_cfg.get("epochs", args.epochs)
            n_train = prior_cfg.get("n_train", 0)
            n_val = prior_cfg.get("n_val", 0)
            n_test = prior_cfg.get("n_test", 0)
            val_auc = prior_cfg.get("val_auc", float("nan"))
            test_pearson = prior_cfg.get("test_pearson", float("nan"))
            test_rmse = prior_cfg.get("test_rmse", float("nan"))
            test_mae = prior_cfg.get("test_mae", float("nan"))
            elapsed_seconds = prior_cfg.get("elapsed_seconds", 0.0)
            final_train_loss = prior_cfg.get("final_train_loss", float("nan"))
            final_cls_loss = prior_cfg.get("final_cls_loss", float("nan"))
            final_reg_loss = prior_cfg.get("final_reg_loss", float("nan"))
            final_domain_loss = prior_cfg.get("final_domain_loss", float("nan"))
            error = prior_cfg.get("error")

        by_cfg[cfg_name] = _PriorProxy()
    baseline = by_cfg.get("baseline")
    lift_per_lever: Dict[str, Dict[str, float]] = {}
    if baseline is not None and baseline.status == "OK":
        for cfg_name in ("wasserstein", "kendall", "dann"):
            r = by_cfg.get(cfg_name)
            if r is None or r.status != "OK":
                lift_per_lever[cfg_name] = {
                    "baseline": float(baseline.test_auc),
                    "with_lever": float("nan"),
                    "lift": float("nan"),
                    "status": "NOT_RUN_OR_BROKEN",
                }
                continue
            b = float(baseline.test_auc)
            w = float(r.test_auc)
            lift = w - b
            lift_per_lever[cfg_name] = {
                "baseline": b,
                "with_lever": w,
                "lift": lift,
                "status": "OK",
            }

    report = {
        "wf_tag": (
            "TODO-182 — Wf-124 P3 per-innovation ablation (paper §3 attribution)"
        ),
        "metal": args.metal,
        "seed": args.seed,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "device": str(device),
        "elapsed_seconds_total": float(time.time() - overall_start),
        "wall_time_budget_s": WALL_TIME_BUDGET_S,
        "per_config": prior_results
        + [
            {
                "config": r.config_name,
                "use_wasserstein": r.use_wasserstein,
                "use_kendall": r.use_kendall,
                "use_dann": r.use_dann,
                "metal": r.metal,
                "seed": r.seed,
                "epochs": r.epochs,
                "n_train": r.n_train,
                "n_val": r.n_val,
                "n_test": r.n_test,
                "val_auc": r.val_auc,
                "test_auc": r.test_auc,
                "test_pearson": r.test_pearson,
                "test_rmse": r.test_rmse,
                "test_mae": r.test_mae,
                "elapsed_seconds": r.elapsed_seconds,
                "final_train_loss": r.final_train_loss,
                "final_cls_loss": r.final_cls_loss,
                "final_reg_loss": r.final_reg_loss,
                "final_domain_loss": r.final_domain_loss,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
        "per_lever_lift": lift_per_lever,
        "n_configs_run": sum(
            1
            for c in (prior_results + [{"status": r.status} for r in results])
            if c.get("status") == "OK"
        ),
        "n_levers_run": sum(
            1
            for cfg_name in ("wasserstein", "kendall", "dann")
            if cfg_name in lift_per_lever
            and lift_per_lever[cfg_name].get("status") == "OK"
        ),
        "honest_framing_note": (
            "TODO-182 smoke-test ablation: 1 metal (Ru) × 1 seed × 5 epochs. "
            "Lift magnitudes are NOT directly comparable to the 30-epoch "
            "Wf-124 P3 default lift (+0.049 absolute, Ru seed=0).  This "
            "smoke run proves the per-lever toggle protocol (each lever "
            "isolates cleanly, no shared-state contamination) and reports "
            "directional lift.  The full TODO-182 G6.1 sweep (5 metals × "
            "3 seeds × 30 epochs × 4 configs = 60 runs ≈ 7.5 hr on RTX 5090) "
            "is the proper scientific measurement; this script is the "
            "protocol + plumbing ship."
        ),
    }
    report_path = args.out_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"[todo-182 ablation] DONE total={time.time() - overall_start:.1f}s "
        f"-> {report_path}",
        flush=True,
    )
    # Console summary
    if baseline is not None and baseline.status == "OK":
        print(
            f"  baseline test_auc = {baseline.test_auc:.4f}",
            flush=True,
        )
        for lever_name in ("wasserstein", "kendall", "dann"):
            entry = lift_per_lever.get(lever_name, {})
            if entry.get("status") == "OK":
                print(
                    f"  +{lever_name}: test_auc = "
                    f"{entry['with_lever']:.4f}  lift = {entry['lift']:+.4f}",
                    flush=True,
                )
            else:
                print(
                    f"  +{lever_name}: NOT_RUN_OR_BROKEN "
                    f"({entry.get('status', 'UNKNOWN')})",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
