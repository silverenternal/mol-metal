"""Wf-84 Phase 3 — D-MPNN pIC50 training on MetalCytoToxDB 5-metal.

=====================================================================
Per-metal D-MPNN pIC50 (regression) baseline trainer
=====================================================================

Trains the existing ``DMPNNMultiTaskBaseline`` (with ``alpha=1.0`` → pure
regression on the pIC50 head, BCE term zeroed out — see
``TODO/04_architecture/model_design.md`` "MetalCytotoxLoss alpha=0.5"
documentation) on the MetalCytoToxDB **5-metal** subset (Ru/Ir/Rh/Os/Re).

Per the wf84 brief:
  * **dataset** = ``metalcytotox_5metal_multitask_v4.csv`` (split definitions)
  * **source**   = ``MetalCytoToxDB.csv`` (IC50 → pIC50 via ``_add_pic50``)
  * **splitting** = reuse the scaffold_id-derived train/val/test partitioning
                    encoded in the 5-metal CSV (so m16/m20/m22 downstream
                    users get exactly the same row indices)
  * **seeds** = ``0 1 2`` (default)
  * **epochs** = 30 (default)
  * **output** = per-metal per-seed ``.pt`` checkpoint + ``per_seed_metrics.json``

**Hardware**: CPU primary; CUDA if ``--device cuda:0`` (RTX 5090 GPU-1
when invoked with ``CUDA_VISIBLE_DEVICES=0``).

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * Smaller metals (Rh/Os/Re < 1.2 k rows each) give noisy pIC50 R²; we
    *report* the per-seed spread but do *not* paper over it.
  * Pure-regression pIC50 (alpha=1.0) means we drop the activity head
    contribution; the multi-task baseline was already shipped and this
    re-uses the same model class for parity.

**Usage**
-----
::
    source env.sh
    \
    python molmetal/scripts/train_dmpnn_pic50_real.py \\
        --dataset metalcytotox_5metal \\
        --seeds 0 1 2 --epochs 30 \\
        --out-dir molmetal/reports/wf84_dmpnn_ckpt
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
import pickle
import random
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
    predict_mt,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

# Wf-120 P3 — pure-stdlib SHA256 overwrite guards (no pandas/torch deps).
# See molmetal/scripts/_dmpnn_sha256_guard.py for the canonical helpers.
from molmetal.scripts._dmpnn_sha256_guard import (
    post_save_sha256_overwrite_guard,
    pre_train_sha256_overwrite_guard,
)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_5METAL_CSV = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf84_dmpnn_ckpt"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"

FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EPOCHS: int = 30
DEFAULT_ALPHA: float = 1.0  # 1.0 → pure pIC50 regression (BCE term = 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_5metal_split_table(csv_path: Path) -> pd.DataFrame:
    """Load the 5-metal multitask CSV with splits encoded per row.

    Returns a DataFrame with columns: ``smiles, metal, scaffold_id, split, activity``
    and a stable ``row_id`` (0-indexed in original CSV order).
    """
    df = pd.read_csv(csv_path)
    expected = {"smiles", "metal", "scaffold_id", "split", "activity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"5-metal CSV missing columns: {missing}")
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    return df


def _featurize_with_cache(
    smiles: List[str],
    cache_key: str,
    cache_dir: Path,
) -> List:
    """Featurise SMILES with on-disk pickle cache.

    The first call writes ``{cache_dir}/{cache_key}.pkl``; subsequent
    calls reuse it. This is critical for multi-seed training — without
    the cache, RDKit graph construction (~7s for 3.6k SMILES) is
    repeated every (metal, seed, epoch) tuple.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_smiles, cached_graphs = pickle.load(f)
            if cached_smiles == smiles:
                return cached_graphs
        except Exception:
            pass  # corrupt cache; rebuild

    t0 = time.time()
    graphs = featurize_smiles_list([str(s) for s in smiles])
    elapsed = time.time() - t0
    try:
        with cache_path.open("wb") as f:
            pickle.dump((smiles, graphs), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # cache write failure is non-fatal
    print(
        f"    featurised {len(smiles)} SMILES in {elapsed:.1f}s "
        f"(cache: {cache_path})",
        flush=True,
    )
    return graphs


def build_metal_split_indices(
    split_df: pd.DataFrame,
    ds: MetalCytotoxDataset,
    metal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect the 5-metal CSV rows for *metal* with the loaded dataset.

    The 5-metal CSV uses string SMILES (complex + counterion) which the
    Cytotox loader exposes as ``ds.smiles`` (canonical SMILES_Ligands).
    Because the 5-metal CSV was built from the *raw* MetalCytoToxDB
    (without the strict time/IC50 filters), every unique CSV SMILES
    appears in the dataset at least once. We:

    1. Subset the 5-metal CSV to ``metal`` rows
    2. Build a SMILES → split map (each unique SMILES belongs to exactly
       one split; verified during CSV build)
    3. Walk every dataset row whose metal matches and whose SMILES is in
       the CSV's smiles_set, emit ``(idx_train, idx_val, idx_test)`` of
       dataset indices.

    Returns ``(idx_train, idx_val, idx_test)`` aligned to the **dataset**
    row order. Duplicates per unique SMILES (multi-cell-line, multi-time
    measurements) are intentionally preserved — each measurement becomes
    a separate training sample.
    """
    metal_df = split_df[split_df["metal"] == metal].copy()
    if metal_df.empty:
        raise RuntimeError(f"No 5-metal rows for metal={metal!r}")

    smiles_set = set(metal_df["smiles"].astype(str).tolist())

    # SMILES → split (each unique SMILES belongs to exactly one split)
    smi_to_split: Dict[str, str] = {}
    for _, row in metal_df.iterrows():
        smi = str(row["smiles"])
        # Last-wins is fine because per-metal CSV has consistent splits per
        # unique SMILES; the assignment is deterministic in the CSV.
        smi_to_split[smi] = str(row["split"])

    ds_smiles = np.asarray(ds.smiles).astype(str)
    ds_metal = np.asarray(ds.metals).astype(str)
    in_metal = ds_metal == metal
    in_smiles = np.array([s in smiles_set for s in ds_smiles], dtype=bool)
    keep = in_metal & in_smiles
    keep_idx_all = np.where(keep)[0]

    splits = np.array(
        [smi_to_split.get(ds_smiles[i], "train") for i in keep_idx_all],
        dtype=object,
    )

    idx_train = keep_idx_all[splits == "train"]
    idx_val = keep_idx_all[splits == "val"]
    idx_test = keep_idx_all[splits == "test"]
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Per-metal, per-seed trainer
# ---------------------------------------------------------------------------
@dataclass
class MetalSeedResult:
    metal: str
    seed: int
    ckpt_path: str
    n_train: int
    n_val: int
    n_test: int
    val_pic50_metrics: Dict[str, float]
    test_pic50_metrics: Dict[str, float]
    elapsed_seconds: float


def _guard_ckpt_overwrite(
    metal: str,
    seed: int,
    ckpt_dir: Path,
) -> None:
    """Wf-120 P1.5 — refuse to overwrite a ckpt if its SHA256 already
    matches a different (metal, seed) ckpt (smoking gun for the silent
    copy-paste bug that originally produced identical
    ``dmpnn_pic50_rh_seed0.pt`` and ``dmpnn_pic50_rh_seed1.pt`` files).

    Backwards-compatible shim that delegates to the canonical helper in
    :mod:`molmetal.scripts._dmpnn_sha256_guard`. Kept as a top-level
    symbol so :mod:`molmetal.scripts.train_dmpnn_pic5_5seed` (which
    imports ``_guard_ckpt_overwrite`` from this module) keeps working.

    Walks every existing ``dmpnn_pic50_*.pt`` under ``ckpt_dir``, hashes
    them, and verifies that the **target** (metal, seed) ckpt — if it
    exists — has a unique SHA256 that does NOT match any other
    (metal, seed) ckpt. Raises ``RuntimeError`` with a loud message
    instead of silently overwriting.

    Note: this guard does NOT block first-time training — if the target
    ckpt does not yet exist, it returns silently. It only fires when a
    pre-existing ckpt would be replaced with a byte-identical file.
    """
    pre_train_sha256_overwrite_guard(metal=metal, seed=seed, ckpt_dir=ckpt_dir)


def train_one_metal_one_seed(
    metal: str,
    seed: int,
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    ds: MetalCytotoxDataset,
    split_df: pd.DataFrame,
    ckpt_dir: Path,
    device: Optional[torch.device] = None,
) -> MetalSeedResult:
    """Train a D-MPNN pIC50 model for one metal, one seed.

    Re-uses ``DMPNNMultiTaskModel`` with ``alpha=1.0`` (pure regression)
    so the existing ``train_epoch_mt`` / ``predict_mt`` plumbing works.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Wf-120 P1.5: refuse silent overwrites of a duplicate ckpt
    _guard_ckpt_overwrite(metal, seed, ckpt_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # SMILES + targets aligned to dataset row order
    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)
    metals_all = np.asarray(ds.metals).astype(str)

    idx_train, idx_val, idx_test = build_metal_split_indices(split_df, ds, metal)
    if len(idx_train) < 50:
        raise RuntimeError(
            f"metal={metal} seed={seed}: only {len(idx_train)} train rows (< 50) — skip"
        )

    # Featurise only the union of train+val+test rows (avoid OOM on full ds)
    used_idx = np.unique(np.concatenate([idx_train, idx_val, idx_test]))
    used_smiles = smiles_all[used_idx]

    t_fe = time.time()
    cache_key = f"dmpnn_pic50_{metal.lower()}_seed{seed}"
    graphs_all = _featurize_with_cache(
        [str(s) for s in used_smiles],
        cache_key=cache_key,
        cache_dir=Path("/tmp/wf84_dmpnn_fe_cache"),
    )
    featurise_seconds = time.time() - t_fe

    # Map dataset row → local row
    pos_in_used = {int(used_idx[k]): k for k in range(len(used_idx))}
    graphs_local = [None] * len(smiles_all)
    for di in used_idx:
        graphs_local[di] = graphs_all[pos_in_used[int(di)]]

    graphs_train = [graphs_local[i] for i in idx_train]
    graphs_val = [graphs_local[i] for i in idx_val]
    graphs_test = [graphs_local[i] for i in idx_test]
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MultiTaskLoss(alpha=alpha)

    best_val_rmse = float("inf")
    best_state: Optional[Dict] = None
    t_train = time.time()

    for epoch in range(epochs):
        ep_t0 = time.time()
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
        # Validate
        y_true_v, y_score_v, pic50_pred_v = predict_mt(
            model,
            graphs_val,
            pic50_val,
            active_val,
            batch_size=batch_size,
            device=device,
        )
        val_m = regression_metrics(pic50_val, pic50_pred_v)
        ep_t = time.time() - ep_t0
        print(
            f"  [{metal} seed={seed}] epoch {epoch+1:>2}/{epochs} "
            f"loss={train_loss:.4f} cls={cls_loss:.4f} reg={reg_loss:.4f} "
            f"val_rmse={val_m['rmse']:.3f} val_pearson={val_m['pearson']:.3f} "
            f"({ep_t:.1f}s)",
            flush=True,
        )
        # Select best by val RMSE (lower is better)
        if val_m["rmse"] < best_val_rmse and np.isfinite(val_m["rmse"]):
            best_val_rmse = val_m["rmse"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        # Fallback: use last epoch weights
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    train_seconds = time.time() - t_train

    # Reload best + test
    model.load_state_dict(best_state)
    y_true_t, y_score_t, pic50_pred_t = predict_mt(
        model,
        graphs_test,
        pic50_test,
        active_test,
        batch_size=batch_size,
        device=device,
    )
    test_m = regression_metrics(pic50_test, pic50_pred_t)
    y_true_v2, y_score_v2, pic50_pred_v2 = predict_mt(
        model,
        graphs_val,
        pic50_val,
        active_val,
        batch_size=batch_size,
        device=device,
    )
    val_m_final = regression_metrics(pic50_val, pic50_pred_v2)

    # Save checkpoint + metadata
    ckpt_path = ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metal": metal,
            "seed": seed,
            "epochs": epochs,
            "alpha": alpha,
            "lr": lr,
            "batch_size": batch_size,
            "hidden": MT_HIDDEN,
            "depth": MT_DEPTH,
            "model_state_dict": best_state,
            "val_rmse": float(val_m_final.get("rmse", float("nan"))),
            "test_rmse": float(test_m.get("rmse", float("nan"))),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
        },
        ckpt_path,
    )

    # Wf-120 P3 — POST-save SHA256 overwrite guard. Refuses to leave a
    # byte-identical file on disk if the new training accidentally
    # produced a ckpt that byte-matches any sibling. On collision the
    # just-written file is removed (rollback) and a RuntimeError is
    # raised. The new SHA256 is always logged to stdout for the audit
    # trail. See molmetal/scripts/_dmpnn_sha256_guard.py for the full
    # contract.
    try:
        new_ckpt_sha = post_save_sha256_overwrite_guard(
            metal=metal,
            seed=seed,
            ckpt_dir=ckpt_dir,
            remove_on_fail=True,
        )
    except RuntimeError as _exc:
        # Surface as a loud, structured error so the caller (CLI driver)
        # can record the failure in per_seed_metrics.json without
        # silently continuing with a corrupt on-disk state.
        raise RuntimeError(
            f"[{metal} seed={seed}] POST-SAVE SHA256 guard refused: {_exc}"
        ) from _exc

    print(
        f"  [{metal} seed={seed}] DONE "
        f"val_pearson={val_m_final['pearson']:.3f} test_pearson={test_m['pearson']:.3f} "
        f"test_rmse={test_m['rmse']:.3f} ({train_seconds:.1f}s, featurise={featurise_seconds:.1f}s)",
        flush=True,
    )

    return MetalSeedResult(
        metal=metal,
        seed=seed,
        ckpt_path=str(ckpt_path),
        n_train=int(len(idx_train)),
        n_val=int(len(idx_val)),
        n_test=int(len(idx_test)),
        val_pic50_metrics=val_m_final,
        test_pic50_metrics=test_m,
        elapsed_seconds=float(train_seconds + featurise_seconds),
    )


def parse_seeds(arg: str) -> int:
    """Convert a single CLI arg (string or int) into one int.

    Supports comma-separated values within a single token (``"0,1,2"``)
    by splitting and returning the first element — argparse with
    ``nargs='+'`` already handles the multi-token case.
    """
    if isinstance(arg, int):
        return int(arg)
    return int(str(arg).split(",")[0])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-84 Phase 3 — D-MPNN pIC50 per-metal training on 5-metal data.",
    )
    p.add_argument(
        "--dataset",
        default="metalcytotox_5metal",
        choices=["metalcytotox_5metal"],
        help="Dataset tag (only metalcytotox_5metal is supported).",
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(FIVE_METALS),
        choices=FIVE_METALS,
        help="Subset of {Ru,Ir,Rh,Os,Re} to train (default: all 5).",
    )
    p.add_argument(
        "--seeds",
        type=parse_seeds,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (one or more ints, e.g. `--seeds 0 1 2`).",
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs per seed."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BCE weight; 1.0 = pure pIC50 regression (default).",
    )
    p.add_argument(
        "--lr", type=float, default=MT_LR, help="Adam lr (default MT_LR=1e-3)."
    )
    p.add_argument("--batch-size", type=int, default=MT_BATCH_SIZE, help="Batch size.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-seed metrics JSON (default wf84_dmpnn_ckpt/).",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory (default molmetal/checkpoints/).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_5METAL_CSV,
        help="Path to the 5-metal multitask CSV (split definitions).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (cuda:0 or cpu). Default: auto-detect.",
    )
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on train rows per metal (for smoke tests).",
    )
    p.add_argument(
        "--loose-filter",
        action="store_true",
        help=(
            "Disable CytotoxFilter time_threshold/ic50_min (loads all rows; "
            "much slower at training, marginally more data per metal)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[wf84-dmpnn-pic50] device={device} metals={args.metals} "
        f"seeds={args.seeds} epochs={args.epochs} alpha={args.alpha}",
        flush=True,
    )

    split_df = load_5metal_split_table(args.csv)
    print(
        f"[wf84-dmpnn-pic50] loaded 5-metal split table: "
        f"{len(split_df)} rows; per-metal counts:",
        flush=True,
    )
    print(split_df.groupby(["metal", "split"]).size().to_string(), flush=True)

    # Load the parent MetalCytoToxDB. Default = the strict baseline filter
    # (time_threshold=24h, ic50_min=0.01) which keeps ~3.7k Ru rows and
    # matches the existing dmpnn_multitask baseline. Use --loose-filter
    # to disable (all 5-metal CSV rows survive; much slower).
    if args.loose_filter:
        flt = CytotoxFilter(
            time_threshold=None,
            ic50_min=None,
            metal_whitelist=list(args.metals),
            compute_pic50=True,
            compute_active=True,
        )
    else:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=list(args.metals),
            compute_pic50=True,
            compute_active=True,
        )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    print(
        f"[wf84-dmpnn-pic50] loaded MetalCytoToxDataset: {len(ds)} rows "
        f"across {sorted(set(ds.metals))} (loose_filter={args.loose_filter})",
        flush=True,
    )

    overall_start = time.time()
    all_results: Dict[str, Dict[str, MetalSeedResult]] = {}
    for metal in args.metals:
        all_results[metal] = {}
        metal_dir = args.out_dir / metal.lower()
        metal_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            try:
                res = train_one_metal_one_seed(
                    metal=metal,
                    seed=seed,
                    epochs=args.epochs,
                    alpha=args.alpha,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    ds=ds,
                    split_df=split_df,
                    ckpt_dir=args.ckpt_dir,
                    device=device,
                )
            except Exception as exc:
                print(
                    f"  [{metal} seed={seed}] FAILED: {exc!r}",
                    flush=True,
                )
                continue
            all_results[metal][str(seed)] = res
            # Per-seed metrics JSON
            seed_payload = {
                "metal": metal,
                "seed": seed,
                "epochs": args.epochs,
                "alpha": args.alpha,
                "ckpt_path": res.ckpt_path,
                "n_train": res.n_train,
                "n_val": res.n_val,
                "n_test": res.n_test,
                "elapsed_seconds": res.elapsed_seconds,
                "val_pic50_metrics": res.val_pic50_metrics,
                "test_pic50_metrics": res.test_pic50_metrics,
            }
            (metal_dir / f"per_seed_metrics_seed{seed}.json").write_text(
                json.dumps(seed_payload, indent=2)
            )

    overall_elapsed = time.time() - overall_start

    # Per-metal aggregate (mean ± std over seeds)
    summary_per_metal: Dict[str, Dict] = {}
    for metal, seed_results in all_results.items():
        if not seed_results:
            summary_per_metal[metal] = {
                "n_seeds": 0,
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
                "status": "ALL_SEEDS_FAILED",
            }
            continue
        pearsons = np.array(
            [
                r.test_pic50_metrics.get("pearson", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        rmses = np.array(
            [
                r.test_pic50_metrics.get("rmse", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        maes = np.array(
            [
                r.test_pic50_metrics.get("mae", float("nan"))
                for r in seed_results.values()
            ],
            dtype=float,
        )
        # r² = pearson**2 (sign-aware Pearson squared)
        r2 = pearsons**2
        first = next(iter(seed_results.values()))
        summary_per_metal[metal] = {
            "n_seeds": len(seed_results),
            "n_train": first.n_train,
            "n_val": first.n_val,
            "n_test": first.n_test,
            "test_pearson_mean": float(np.nanmean(pearsons)),
            "test_pearson_std": float(np.nanstd(pearsons)),
            "test_rmse_mean": float(np.nanmean(rmses)),
            "test_rmse_std": float(np.nanstd(rmses)),
            "test_mae_mean": float(np.nanmean(maes)),
            "test_mae_std": float(np.nanstd(maes)),
            "test_r2_mean": float(np.nanmean(r2)),
            "test_r2_std": float(np.nanstd(r2)),
            "ckpt_paths": [r.ckpt_path for r in seed_results.values()],
            "status": "OK",
        }

    summary = {
        "wf_tag": "Wf-84 Phase 3 — D-MPNN pIC50 per-metal training on 5-metal",
        "dataset": args.dataset,
        "metals": args.metals,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "device": str(device),
        "elapsed_seconds_total": overall_elapsed,
        "per_metal": summary_per_metal,
        "honest_framing_note": (
            "Smaller metals (Rh/Os/Re < 1.2k rows each) produce noisier per-seed "
            "pIC50 R²; we report mean ± std across seeds without smoothing. "
            "Pure pIC50 regression (alpha=1.0); BCE term is zero."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[wf84-dmpnn-pic50] DONE total elapsed={overall_elapsed:.1f}s",
        flush=True,
    )
    for metal, s in summary_per_metal.items():
        print(
            f"  {metal}: n_train={s.get('n_train', '?')} n_val={s.get('n_val', '?')} "
            f"n_test={s.get('n_test', '?')} test_pearson={s.get('test_pearson_mean', float('nan')):.3f}±{s.get('test_pearson_std', float('nan')):.3f} "
            f"test_rmse={s.get('test_rmse_mean', float('nan')):.3f}±{s.get('test_rmse_std', float('nan')):.3f} "
            f"test_r2={s.get('test_r2_mean', float('nan')):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
