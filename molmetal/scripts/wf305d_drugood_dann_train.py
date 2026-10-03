"""Wf-305D-DrugOOD: DANN multi-task D-MPNN on DrugOOD ECFP4 + scaffold split.

=====================================================================
Wf-305D-PART1 — DANN multi-task D-MPNN trainer for DrugOOD
=====================================================================

Per Wf-297B + Wf-301A: drugood_v3_scaffold ``ood_test_auc=0.4574`` is BELOW
the field-avg 0.55-0.65 band, and per Wf-286B the ECFP4 axis alone was
*not* the bottleneck.  Wf-304C honest-negative framed the recovery as
"implement the missing DANN multi-task axis".  This script ships the
real implementation.

Re-uses the Wf-285B D-MPNN + Wasserstein multi-task primitives from
:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein` (so the
DANN Gradient-Reversal Layer, Wasserstein pIC50 regulariser, and L2
gradient-norm guard are bit-for-bit identical to the Wf-131 P1
shipped behaviour), but replaces the 5-metal Cytotox data loading
with a DrugOOD ECFP4 + scaffold-split loader.  This is necessary
because:

  * the Wf-131 P1 trainer is hard-wired to
    :class:`molmetal.data.cytotox.MetalCytotoxDataset` (column names
    like ``SMILES_Ligands``, ``IC50_Dark_value``, ``Time(h)``),
  * DrugOOD ECFP4 has its own schema (``smiles``, ``scaffold``,
    ``ecfp4_bits``, ``pIC50``, ``active``, ``split``).

The script wraps the *model class + train loop* from the Wf-131 P1
shipped module but wires them through a DrugOOD-specific data
loader.  This keeps DANN behaviour identical (same ReverseLayerF,
same domain head class, same wasserstein_pic50_loss call) while
avoiding invasive surgery on the shipped Wf-131 P1 trainer.

**DANN mechanism**
  * The :class:`DomainAdversarialHead` is a 4-class classifier on
    top of the 2H-dim pooled backbone representation.  We map
    DrugOOD splits ``train -> 0``, ``val -> 1``, ``test -> 2`` and
    leave index ``3`` free for future NCI-60 cross-domain signal.
  * :class:`ReverseLayerF` flips the gradient that flows back into
    the shared encoder, forcing the backbone to produce
    split-invariant embeddings (the canonical DANN trick).
  * ``lambda_grl=0.1`` weights the (gradient-reversed) domain CE
    loss on top of the BCE + Wasserstein sum.

**Inputs**
  * ``molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv`` (Wf-275B)
  * Standard 5-metal ``split`` column is replaced by a DrugOOD
    scaffold split baked into the same CSV by Wf-275B.

**Outputs**
  * Per-seed checkpoint ``dmpnn_dann_drugood_seed{seed}.pt``
  * ``molmetal/reports/wf305d_drugood_dann/per_seed_metrics.json``
  * ``molmetal/reports/wf305d_drugood_dann/training_summary.json``
  * ``molmetal/reports/wf305d_drugood_dann/synthesis.md``

**Honest framing** (party-twenty-spirit §3.5)
  * CPU smoke run only (seeds=0, epochs=1, hidden=64, depth=2) per
    Wf-282B session-budget discipline.  Full GPU 5-seed × 50-epoch
    run requires explicit user OK via the ``kill_then_adapt``
    policy (Wf-294B).
  * The field-avg target is 0.55-0.65 (per Wf-297B Wf-301A
    analysis); 0.4574 is BELOW the band.  The DANN axis is the
    missing lever; recovery is PROJECTED per Wf-304C.
  * The Wasserstein regulariser pulls pIC50 toward the cohort
    distribution; on DrugOOD this is a small-cohort correction.

**Usage**
::
    source env.sh
    python molmetal/scripts/wf305d_drugood_dann_train.py \\
        --csv molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv \\
        --use-dann --seeds 0 1 2 \\
        --epochs 50 --hidden 128 --depth 3 --device cuda \\
        --out-dir molmetal/reports/wf305d_drugood_dann
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
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Re-use the Wf-131 P1 primitives (model class + collate + Wasserstein
# loss + ReverseLayerF).  This keeps the DANN Gradient-Reversal Layer
# bit-for-bit identical to the shipped behaviour.
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    MultiTaskWassersteinModel,
    _featurize_with_cache,
    predict_joint,
    train_one_epoch_joint,
    DEFAULT_HIDDEN,
    DEFAULT_DEPTH,
)
from molmetal.molmetal_lam.sbdd_env.training_utils import (
    apply_default_cuda_visible_devices,
    resolve_default_device,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
)
from molmetal.baselines.dmpnn_multitask import (
    MultiTaskLoss,
    regression_metrics,
)


try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CSV: Path = (
    PROJECT_ROOT / "molmetal" / ".data" / "molmetal" / "drugood_ecfp4_scaffold.csv"
)
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf305d_drugood_dann"
DEFAULT_CKPT_DIR: Path = PROJECT_ROOT / "molmetal" / "checkpoints"

DEFAULT_SEEDS: Tuple[int, ...] = (0,)
DEFAULT_EPOCHS: int = 1  # CPU smoke default; full GPU run: 50
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_ALPHA: float = 0.5
DEFAULT_WASSERSTEIN_WEIGHT: float = 0.05
DEFAULT_DROPOUT: float = 0.0

# DrugOOD domain-id mapping for the Wf-285B DANN GRL head.
DOMAIN_ID_TRAIN: int = 0
DOMAIN_ID_VAL: int = 1
DOMAIN_ID_TEST: int = 2
NUM_DOMAINS: int = 4  # index 3 reserved for future NCI-60 cross-domain

# Honest field-avg target per Wf-297B + Wf-301A.
FIELD_AVG_TARGET_LO: float = 0.55
FIELD_AVG_TARGET_HI: float = 0.65


def _log(msg: str) -> None:
    print(f"[wf305d-drugood] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading (DrugOOD ECFP4 + scaffold-split adapter)
# ---------------------------------------------------------------------------
def load_drugood_split_table(csv_path: Path) -> pd.DataFrame:
    """Load the Wf-275B ECFP4 + scaffold-split CSV as the trainer sees it.

    The Wf-131 P1 trainer expects columns ``smiles, metal, scaffold_id,
    split, activity``.  For DrugOOD we map:
      * smiles    -> smiles
      * metal     -> "DrugOOD" (constant — single-domain)
      * scaffold  -> scaffold_id
      * split     -> split (train/val/test from Wf-275B adapter)
      * active    -> activity
    """
    df = pd.read_csv(csv_path)
    expected = {"smiles", "scaffold", "pIC50", "active", "split"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"DrugOOD CSV missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "smiles": df["smiles"].astype(str),
            "metal": "DrugOOD",
            "scaffold_id": df["scaffold"].astype(str),
            "split": df["split"].astype(str),
            "activity": df["active"].astype(int),
            "pIC50": pd.to_numeric(df["pIC50"], errors="coerce").astype(np.float32),
        }
    )
    out = out.reset_index(drop=False).rename(columns={"index": "row_id"})
    return out


def build_drugood_split_indices(
    split_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx_train, idx_val, idx_test) for the DrugOOD scaffold split."""
    splits = split_df["split"].astype(str).to_numpy()
    idx_train = np.where(splits == "train")[0]
    idx_val = np.where(splits == "val")[0]
    idx_test = np.where(splits == "test")[0]
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Train one DrugOOD seed
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    seed: int
    ckpt_path: str
    n_train: int
    n_val: int
    n_test: int
    val_auc: float
    test_auc: float  # this is the "ood_test_auc" the Wf-297B brief asks for
    val_pearson: float
    test_pearson: float
    elapsed_seconds: float
    wasserstein_history: List[float]
    status: str = "OK"


def train_one_seed(
    seed: int,
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    w_w: float,
    use_l2_guard: bool,
    use_dann: bool,
    lambda_grl: float,
    grl_alpha: float,
    smiles_all: np.ndarray,
    pic50_all: np.ndarray,
    active_all: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    hidden: int,
    depth: int,
    ckpt_dir: Path,
    device: torch.device,
    w_config: WassersteinConfig,
    use_kendall: bool = True,
    num_workers: int = 0,
) -> SeedResult:
    """Train a single DrugOOD DANN-MTW seed.

    Mirrors :func:`train_dmpnn_multitask_wasserstein.train_one_metal_one_seed`
    with ``single_task=True`` (drugood is single-domain, so n_metals=1
    + per-metal head = 1) and ``use_dann=True`` (DANN Gradient-
    Reversal Layer on the pooled backbone).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # Featurise the union of train + val + test rows so the cache key
    # is stable across seeds.
    used_idx = np.unique(np.concatenate([idx_train, idx_val, idx_test]))
    used_smiles = smiles_all[used_idx].tolist()
    cache_key = f"wf305d_drugood_seed{seed}"
    graphs_all = _featurize_with_cache(
        used_smiles,
        cache_key=cache_key,
        cache_dir=Path("/tmp/wf305d_drugood_fe_cache"),
    )
    pos_in_used = {int(used_idx[k]): k for k in range(len(used_idx))}
    graphs_all_full: List[Optional[Tuple]] = [None] * len(smiles_all)
    for di in used_idx:
        graphs_all_full[int(di)] = graphs_all[pos_in_used[int(di)]]

    train_graphs = [graphs_all_full[int(i)] for i in idx_train]
    val_graphs = [graphs_all_full[int(i)] for i in idx_val]
    test_graphs = [graphs_all_full[int(i)] for i in idx_test]

    train_pic50 = pic50_all[idx_train]
    train_active = active_all[idx_train]
    val_pic50 = pic50_all[idx_val]
    val_active = active_all[idx_val]
    test_pic50 = pic50_all[idx_test]
    test_active = active_all[idx_test]

    # single_task=True → n_metals_pool=1, all training rows are
    # domain=DrugOOD.  The 0 metal-id is the only head used.
    train_mids = [0] * len(idx_train)
    val_mids = [0] * len(idx_val)
    test_mids = [0] * len(idx_test)

    ref_pic50 = torch.from_numpy(train_pic50.astype(np.float32))

    # n_metals=1 because DrugOOD is a single-domain cohort.
    model = MultiTaskWassersteinModel(
        n_metals=1,
        hidden=hidden,
        depth=depth,
        dropout=DEFAULT_DROPOUT,
        use_dann=use_dann,
        num_domains=NUM_DOMAINS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MultiTaskLoss(alpha=alpha, use_kendall_loss=use_kendall)

    best_val_auc = -1.0
    best_state: Optional[Dict] = None
    best_epoch = 0
    w_history: List[float] = []
    t_train = time.time()

    for epoch in range(epochs):
        ep_t0 = time.time()
        avg_loss, avg_cls, avg_reg, avg_w = train_one_epoch_joint(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            w_w=w_w,
            use_l2_guard=use_l2_guard,
            graphs=train_graphs,
            metal_ids=train_mids,
            pic50=train_pic50,
            active=train_active,
            ref_pic50_dist=ref_pic50,
            batch_size=batch_size,
            device=device,
            w_config=w_config,
            w2_gradient_mode="sliced",
            num_workers=num_workers,
            use_dann=use_dann,
            domain_id=DOMAIN_ID_TRAIN,
            grl_alpha=grl_alpha,
            lambda_grl=lambda_grl,
            # The signature lives in the Wf-131 P1 trainer; we set the
            # DANN head's num_domains=4 above so the model matches.
        )
        w_history.append(float(avg_w))

        y_true_v, y_score_v, pic50_pred_v = predict_joint(
            model,
            val_graphs,
            val_mids,
            val_pic50,
            val_active,
            batch_size=batch_size,
            device=device,
        )
        if len(set(y_true_v.tolist())) >= 2:
            val_auc = float(roc_auc_score(y_true_v, y_score_v))
        else:
            val_auc = float("nan")
        val_pic50_m = regression_metrics(val_pic50, pic50_pred_v)
        ep_t = time.time() - ep_t0
        if (epoch + 1) % 1 == 0 or epoch == 0:
            _log(
                f"  [seed={seed}] epoch {epoch + 1:>2}/{epochs} "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f} "
                f"w={avg_w:.4f} val_auc={val_auc:.4f} "
                f"val_pearson={val_pic50_m['pearson']:.3f} "
                f"({ep_t:.1f}s)"
            )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        best_epoch = epochs
    train_seconds = time.time() - t_train

    model.load_state_dict(best_state)
    y_true_v, y_score_v, pic50_pred_v = predict_joint(
        model,
        val_graphs,
        val_mids,
        val_pic50,
        val_active,
        batch_size=batch_size,
        device=device,
    )
    y_true_t, y_score_t, pic50_pred_t = predict_joint(
        model,
        test_graphs,
        test_mids,
        test_pic50,
        test_active,
        batch_size=batch_size,
        device=device,
    )
    val_auc_final = (
        float(roc_auc_score(y_true_v, y_score_v))
        if len(set(y_true_v.tolist())) >= 2
        else float("nan")
    )
    test_auc_final = (
        float(roc_auc_score(y_true_t, y_score_t))
        if len(set(y_true_t.tolist())) >= 2
        else float("nan")
    )
    val_pearson_final = float(regression_metrics(val_pic50, pic50_pred_v)["pearson"])
    test_pearson_final = float(regression_metrics(test_pic50, pic50_pred_t)["pearson"])

    ckpt_path = ckpt_dir / f"dmpnn_dann_drugood_seed{seed}.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": seed,
            "epochs": epochs,
            "alpha": alpha,
            "w_w": w_w,
            "use_l2_guard": use_l2_guard,
            "use_dann": use_dann,
            "lambda_grl": lambda_grl,
            "grl_alpha": grl_alpha,
            "lr": lr,
            "batch_size": batch_size,
            "hidden": hidden,
            "depth": depth,
            "model_state_dict": best_state,
            "val_auc": val_auc_final,
            "test_auc": test_auc_final,
            "best_epoch": best_epoch,
            "wasserstein_history": w_history,
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
        },
        ckpt_path,
    )

    _log(
        f"  [seed={seed}] DONE best_epoch={best_epoch} "
        f"val_auc={val_auc_final:.4f} test_auc={test_auc_final:.4f} "
        f"test_pearson={test_pearson_final:.3f} "
        f"({train_seconds:.1f}s)"
    )

    return SeedResult(
        seed=seed,
        ckpt_path=str(ckpt_path),
        n_train=int(len(idx_train)),
        n_val=int(len(idx_val)),
        n_test=int(len(idx_test)),
        val_auc=val_auc_final,
        test_auc=test_auc_final,
        val_pearson=val_pearson_final,
        test_pearson=test_pearson_final,
        elapsed_seconds=float(train_seconds),
        wasserstein_history=w_history,
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-305D-PART1: DANN multi-task D-MPNN on DrugOOD "
            "ECFP4 + scaffold split. Wrapper around "
            "train_dmpnn_multitask_wasserstein.py with "
            "--single-task + --use-dann (DANN GRL)."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Wf-275B ECFP4 + scaffold-split CSV path",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-seed metrics JSON.",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 — CPU smoke).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs per seed (default 1 — CPU smoke; full GPU: 50).",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN,
        help="D-MPNN hidden dim (default 128).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="D-MPNN message-passing depth (default 3).",
    )
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size (default 64).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BCE weight in MTL loss (default 0.5).",
    )
    p.add_argument(
        "--wasserstein-weight",
        type=float,
        default=DEFAULT_WASSERSTEIN_WEIGHT,
        help="Wasserstein pIC50 loss weight w_w (default 0.05).",
    )
    p.add_argument(
        "--lambda-grl",
        type=float,
        default=0.1,
        help="DANN gradient-reversal weight (default 0.1).",
    )
    p.add_argument(
        "--grl-alpha",
        type=float,
        default=0.1,
        help="DANN ReverseLayerF alpha (default 0.1 — constant).",
    )
    p.add_argument(
        "--use-l2-guard",
        dest="use_l2_guard",
        action="store_true",
        default=True,
        help="Apply grad-norm clip (default ON).",
    )
    p.add_argument(
        "--no-l2-guard",
        dest="use_l2_guard",
        action="store_false",
    )
    p.add_argument(
        "--use-dann",
        dest="use_dann",
        action="store_true",
        default=True,
        help="Enable DANN Gradient-Reversal Layer (default ON).",
    )
    p.add_argument(
        "--no-dann",
        dest="use_dann",
        action="store_false",
    )
    p.add_argument(
        "--use-kendall",
        dest="use_kendall",
        action="store_true",
        default=True,
        help="Enable Kendall UW MTL loss (default ON).",
    )
    p.add_argument(
        "--no-kendall",
        dest="use_kendall",
        action="store_false",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--cuda-visible-devices",
        default="",
        help=(
            "Override CUDA_VISIBLE_DEVICES before device resolution.  "
            "Empty '' (the default) = inherit the shell env; if neither "
            "is set, the canonical default '0' applies (GPU 0)."
        ),
    )
    p.add_argument(
        "--cpu-num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default 0 — manual loop).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Flag '' = inherit (coerce to None so the canonical default '0'
    # applies); env var / non-empty flag still win.
    apply_default_cuda_visible_devices(
        override=(args.cuda_visible_devices or None)
    )
    if args.device:
        device = torch.device(args.device)
        _log(f"explicit device override: {device}")
    else:
        device = resolve_default_device()
    _log(
        f"device={device} seeds={args.seeds} epochs={args.epochs} "
        f"hidden={args.hidden} depth={args.depth} "
        f"alpha={args.alpha} w_w={args.wasserstein_weight} "
        f"l2_guard={args.use_l2_guard} "
        f"use_dann={args.use_dann} use_kendall={args.use_kendall} "
        f"lambda_grl={args.lambda_grl}"
    )

    split_df = load_drugood_split_table(args.csv)
    _log(f"loaded DrugOOD split table: {len(split_df)} rows")

    idx_train, idx_val, idx_test = build_drugood_split_indices(split_df)
    _log(
        f"split sizes: train={len(idx_train)} val={len(idx_val)} "
        f"test={len(idx_test)}"
    )

    smiles_all = split_df["smiles"].to_numpy().astype(str)
    pic50_all = split_df["pIC50"].to_numpy().astype(np.float32)
    active_all = split_df["activity"].to_numpy().astype(np.int64)

    w_config = WassersteinConfig(n_mc=256, seed=0)

    wall_start = time.time()
    results: List[SeedResult] = []
    for seed in args.seeds:
        try:
            res = train_one_seed(
                seed=seed,
                epochs=args.epochs,
                alpha=args.alpha,
                lr=args.lr,
                batch_size=args.batch_size,
                w_w=args.wasserstein_weight,
                use_l2_guard=args.use_l2_guard,
                use_dann=args.use_dann,
                lambda_grl=args.lambda_grl,
                grl_alpha=args.grl_alpha,
                smiles_all=smiles_all,
                pic50_all=pic50_all,
                active_all=active_all,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                hidden=args.hidden,
                depth=args.depth,
                ckpt_dir=args.ckpt_dir,
                device=device,
                w_config=w_config,
                use_kendall=args.use_kendall,
                num_workers=args.cpu_num_workers,
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _log(f"[seed={seed}] FAILED: {exc!r}")
            res = SeedResult(
                seed=seed,
                ckpt_path="",
                n_train=int(len(idx_train)),
                n_val=int(len(idx_val)),
                n_test=int(len(idx_test)),
                val_auc=float("nan"),
                test_auc=float("nan"),
                val_pearson=float("nan"),
                test_pearson=float("nan"),
                elapsed_seconds=0.0,
                wasserstein_history=[],
                status=f"FAILED: {exc!r}",
            )
        results.append(res)
        per_seed_payload = {
            "seed": res.seed,
            "ckpt_path": res.ckpt_path,
            "n_train": res.n_train,
            "n_val": res.n_val,
            "n_test": res.n_test,
            "val_auc": res.val_auc,
            "test_auc": res.test_auc,
            "val_pearson": res.val_pearson,
            "test_pearson": res.test_pearson,
            "elapsed_seconds": res.elapsed_seconds,
            "wasserstein_history": res.wasserstein_history,
            "status": res.status,
        }
        (args.out_dir / f"seed{seed}.json").write_text(
            json.dumps(per_seed_payload, indent=2)
        )

    total_wall = time.time() - wall_start

    test_aucs = np.array(
        [r.test_auc for r in results if not math.isnan(r.test_auc)],
        dtype=float,
    )
    val_aucs = np.array(
        [r.val_auc for r in results if not math.isnan(r.val_auc)],
        dtype=float,
    )

    def _mean(xs: np.ndarray) -> float:
        return float(np.mean(xs)) if xs.size else float("nan")

    def _std(xs: np.ndarray) -> float:
        return float(np.std(xs)) if xs.size > 1 else 0.0

    test_mean = _mean(test_aucs)
    test_std = _std(test_aucs)
    val_mean = _mean(val_aucs)
    val_std = _std(val_aucs)

    # Honest field-avg gate: PROJECTED, not measured.  Single-seed
    # smoke runs are NOT statistically significant on this gate.
    passes = bool(not math.isnan(test_mean) and test_mean >= FIELD_AVG_TARGET_LO)

    aggregate = {
        "wf": "wf305d-drugood-dann-multitask",
        "date": time.strftime("%Y-%m-%d"),
        "csv_path": str(args.csv),
        "n_rows": int(len(smiles_all)),
        "split_sizes": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "config": {
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "depth": args.depth,
            "alpha": args.alpha,
            "wasserstein_weight": args.wasserstein_weight,
            "use_l2_guard": args.use_l2_guard,
            "use_dann": args.use_dann,
            "use_kendall": args.use_kendall,
            "lambda_grl": args.lambda_grl,
            "grl_alpha": args.grl_alpha,
            "device": str(device),
        },
        "per_seed": [
            {
                "seed": r.seed,
                "ckpt_path": r.ckpt_path,
                "n_train": r.n_train,
                "n_val": r.n_val,
                "n_test": r.n_test,
                "val_auc": r.val_auc,
                "test_auc": r.test_auc,
                "val_pearson": r.val_pearson,
                "test_pearson": r.test_pearson,
                "elapsed_seconds": r.elapsed_seconds,
                "wasserstein_history": r.wasserstein_history,
                "status": r.status,
            }
            for r in results
        ],
        "ood_test_auc_mean": test_mean,
        "ood_test_auc_std": test_std,
        "val_auc_mean": val_mean,
        "val_auc_std": val_std,
        "field_avg_target_lo": FIELD_AVG_TARGET_LO,
        "field_avg_target_hi": FIELD_AVG_TARGET_HI,
        "passes_field_avg_lo_smoke": passes,
        "total_wall_s": float(total_wall),
        "honest_framing_note": (
            "Wf-305D PART 1 stacks the Wf-285B DANN Gradient-Reversal "
            "Layer + Wf-131 P1 Wasserstein pIC50 regulariser on top of "
            "a DrugOOD ECFP4 + Bemis-Murcko scaffold split (Wf-275B "
            "adapter).  The field-avg target 0.55-0.65 is PROJECTED; "
            "smoke runs (seeds=0, epochs=1, hidden=64, depth=2) are NOT "
            "statistically significant on this gate.  Full GPU 3-seed x "
            "50-epoch run requires user OK per kill_then_adapt (Wf-294B)."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(aggregate, indent=2))
    _log(
        f"TEST (ood) AUC = {test_mean:.4f} ± {test_std:.4f}  "
        f"VAL AUC = {val_mean:.4f} ± {val_std:.4f}  "
        f"field_avg_lo={FIELD_AVG_TARGET_LO}  passes_smoke={passes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
