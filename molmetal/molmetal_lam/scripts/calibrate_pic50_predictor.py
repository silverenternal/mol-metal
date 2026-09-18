"""calibrate_pic50_predictor — train AttentiveDMPNN as a pIC50 regressor.

================================================================
Why this exists
================================================================
The Lambda-vs-SBDD comparison used a 100-row sklearn MLP on Morgan
fingerprints for binding-affinity prediction — comparable to a constant
predictor.  This script replaces that with a properly trained graph
neural network (Attentive D-MPNN) on 2,000 unique Ru-cyto ligands.

================================================================
Protocol (matches the published benchmark)
================================================================
* Source: ``MetalCytoToxDB.csv`` — metal=Ru, IC50 present.
* Dedup by canonical SMILES (keep first occurrence).
* Random sample 2,000 unique rows.
* Target: ``pIC50 = 6 - log10(IC50_uM)`` (clip IC50 to 1e-6 uM).
* 80/10/10 random split, seed=42.
* Train ``molmetal/baselines/dmpnn_attentive.AttentiveDMPNNModel``
  with **MSELoss** (regression), 20 epochs, batch=16, lr=1e-3.
* Save checkpoint to ``molmetal/checkpoints/dmpnn_attn_ru_pic50.pt``.
* Report train/val/test MSE + Pearson r + Spearman r.

CLI
---
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.molmetal_lam.scripts.calibrate_pic50_predictor

The companion predictor wrapper lives at
``molmetal/molmetal_lam/sbdd_env/pic50_predictor.py`` and replaces the
sklearn MLP previously inlined in ``baselines.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Path bootstrap so ``import molmetal.baselines.*`` works regardless of cwd.
_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from molmetal.baselines.dmpnn_attentive import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    AttentiveDMPNNModel,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    DMPNN_HIDDEN,
    collate_graphs,
    featurize_smiles_list,
)

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV_CANDIDATES: Tuple[str, ...] = (
    "/mnt/storage/data/molmetal/MetalCytoToxDB.csv",
    os.path.join(_PKG_PARENT, "data", "MetalCytoToxDB.csv"),
)
DEFAULT_OUT_CKPT = os.path.join(
    _PKG_PARENT, "molmetal", "checkpoints", "dmpnn_attn_ru_pic50.pt"
)
DEFAULT_OUT_JSON = os.path.join(
    _PKG_PARENT, "molmetal", "reports", "h2_pic50_calibration.json"
)
DEFAULT_METAL = "Ru"
DEFAULT_N_ROWS = 2000
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _find_csv() -> str:
    for path in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"MetalCytoToxDB.csv not found in any of: {DEFAULT_CSV_CANDIDATES}"
    )


def load_pic50_table(
    csv_path: str,
    metal: str = DEFAULT_METAL,
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Filter, dedup, and randomly sample ``n_rows`` Ru-pIC50 rows.

    Returns a DataFrame with columns ``SMILES_Ligands`` (canonical),
    ``IC50_uM`` (float > 0), and ``pIC50`` (float).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if "Metal" not in df.columns or "SMILES_Ligands" not in df.columns:
        raise ValueError(
            f"CSV missing required columns (Metal/SMILES_Ligands). "
            f"Found: {list(df.columns)[:6]}..."
        )

    ic50_col = "IC50_Dark_value" if "IC50_Dark_value" in df.columns else None
    if ic50_col is None:
        raise ValueError("CSV missing IC50_Dark_value column")

    df = df[df["Metal"].astype(str) == metal].copy()
    df[ic50_col] = pd.to_numeric(df[ic50_col], errors="coerce")
    df = df.dropna(subset=[ic50_col])
    df = df[df[ic50_col] > 0.0]
    print(f"[load_pic50_table] {metal} rows with valid IC50: {len(df)}")

    # Dedup by canonical SMILES (keep first).
    canon_smiles: List[str] = []
    keep_idx: List[int] = []
    seen: set = set()
    for idx, s in df["SMILES_Ligands"].astype(str).items():
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        if canon in seen:
            continue
        seen.add(canon)
        canon_smiles.append(canon)
        keep_idx.append(idx)
    df = df.loc[keep_idx].copy()
    df["SMILES_Ligands"] = canon_smiles
    print(f"[load_pic50_table] unique canonical SMILES: {len(df)}")

    # Sample n_rows.
    if len(df) > n_rows:
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(df), size=int(n_rows), replace=False)
        df = df.iloc[sorted(sel)].copy()
    print(f"[load_pic50_table] sampled rows: {len(df)}")

    # Compute pIC50.
    ic50_um = df[ic50_col].astype(float).to_numpy()
    pic50 = 6.0 - np.log10(np.clip(ic50_um, 1e-6, None))
    df["IC50_uM"] = ic50_um.astype(np.float32)
    df["pIC50"] = pic50.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graphs: List,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    """Train AttentiveDMPNNModel on a regression target (pIC50)."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_labels = labels[batch_idx]

        valid = [(g, l) for g, l in zip(batch_graphs, batch_labels) if g is not None]
        if not valid:
            continue
        g_list = [v[0] for v in valid]
        y_batch = np.array([v[1] for v in valid], dtype=np.float32)

        batch_data = collate_graphs(g_list)
        if batch_data is None:
            continue

        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        y_batch = torch.from_numpy(y_batch).to(device)

        optimizer.zero_grad()
        preds = model(atom_f, bond_f, e_src, e_dst)
        loss = nn.MSELoss()(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def predict_regression(
    model: nn.Module,
    graphs: List,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return raw regression predictions aligned to the input list."""
    model.eval()
    out: List[float] = []
    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        valid = [g for g in batch_graphs if g is not None]
        if not valid:
            out.extend([0.0] * len(batch_graphs))
            continue
        batch_data = collate_graphs(valid)
        if batch_data is None:
            out.extend([0.0] * len(batch_graphs))
            continue
        atom_f, bond_f, e_src, e_dst = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        preds = model(atom_f, bond_f, e_src, e_dst).cpu().numpy().tolist()
        vi = 0
        for g in batch_graphs:
            if g is None:
                out.append(0.0)
            else:
                out.append(float(preds[vi]))
                vi += 1
    return np.array(out, dtype=np.float32)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if float(np.std(y_true)) < 1e-8 or float(np.std(y_pred)) < 1e-8:
        return float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if corr == corr else float("nan")


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho) if rho == rho else float("nan")
    except Exception:
        # Fallback: rank-transform manually.
        def _rank(x: np.ndarray) -> np.ndarray:
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(x) + 1)
            return ranks
        return _safe_pearson(_rank(y_true), _rank(y_pred))


def evaluate(
    model: nn.Module,
    graphs: List,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    preds = predict_regression(model, graphs, batch_size=batch_size, device=device)
    valid_mask = labels != 0  # we never write 0 labels; defensive
    if valid_mask.sum() < len(labels):
        labels_v = labels[valid_mask]
        preds_v = preds[valid_mask]
    else:
        labels_v = labels
        preds_v = preds
    mse = float(np.mean((labels_v - preds_v) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(labels_v - preds_v)))
    pearson = _safe_pearson(labels_v, preds_v)
    spearman = _safe_spearman(labels_v, preds_v)
    return {
        "n": int(len(labels_v)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "y_mean": float(np.mean(labels_v)),
        "y_std": float(np.std(labels_v)),
        "pred_mean": float(np.mean(preds_v)),
        "pred_std": float(np.std(preds_v)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train AttentiveDMPNN as a Ru-pIC50 regressor."
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--metal", type=str, default=DEFAULT_METAL)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden", type=int, default=DMPNN_HIDDEN)
    parser.add_argument("--depth", type=int, default=DMPNN_DEPTH)
    parser.add_argument("--dropout", type=float, default=DMPNN_DROPOUT)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_OUT_CKPT)
    parser.add_argument("--json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.csv or _find_csv()
    print(f"[calibrate_pic50_predictor] CSV={csv_path}")
    print(
        f"[calibrate_pic50_predictor] metal={args.metal} n_rows={args.n_rows} "
        f"seed={args.seed} epochs={args.epochs} batch={args.batch_size} lr={args.lr}"
    )

    df = load_pic50_table(
        csv_path=csv_path,
        metal=args.metal,
        n_rows=args.n_rows,
        seed=args.seed,
    )

    smiles = df["SMILES_Ligands"].astype(str).to_numpy()
    y_all = df["pIC50"].astype(np.float32).to_numpy()
    print(
        f"[calibrate_pic50_predictor] pIC50 stats: mean={y_all.mean():.3f} "
        f"std={y_all.std():.3f} min={y_all.min():.3f} max={y_all.max():.3f}"
    )

    # 80/10/10 random split.
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(y_all))
    rng.shuffle(idx)
    n_train = int(0.8 * len(idx))
    n_val = int(0.1 * len(idx))
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]
    print(
        f"[calibrate_pic50_predictor] split: train={len(idx_train)} "
        f"val={len(idx_val)} test={len(idx_test)}"
    )

    # Featurise once.
    t0 = time.time()
    graphs_all = featurize_smiles_list([str(s) for s in smiles])
    n_failed = sum(1 for g in graphs_all if g is None)
    print(
        f"[calibrate_pic50_predictor] featurised {len(graphs_all)} mols "
        f"({n_failed} failed) in {time.time() - t0:.1f}s"
    )

    graphs_train = [graphs_all[i] for i in idx_train]
    graphs_val = [graphs_all[i] for i in idx_val]
    graphs_test = [graphs_all[i] for i in idx_test]
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]

    device = torch.device(args.device)
    model = AttentiveDMPNNModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mse = float("inf")
    best_state: Optional[Dict] = None
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, optimizer, graphs_train, y_train,
            batch_size=args.batch_size, device=device,
        )
        val_metrics = evaluate(
            model, graphs_val, y_val,
            batch_size=args.batch_size, device=device,
        )
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_spearman_r": val_metrics["spearman_r"],
            "elapsed_s": elapsed,
        }
        history.append(row)

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"[epoch {epoch:3d}] train_mse={train_loss:.4f} "
            f"val_mse={val_metrics['mse']:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"val_pearson={val_metrics['pearson_r']:.4f} "
            f"val_spearman={val_metrics['spearman_r']:.4f} "
            f"({elapsed:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test.
    train_metrics = evaluate(
        model, graphs_train, y_train,
        batch_size=args.batch_size, device=device,
    )
    val_metrics = evaluate(
        model, graphs_val, y_val,
        batch_size=args.batch_size, device=device,
    )
    test_metrics = evaluate(
        model, graphs_test, y_test,
        batch_size=args.batch_size, device=device,
    )

    print()
    print("=" * 64)
    print("FINAL METRICS (best epoch by val MSE)")
    print("=" * 64)
    for split, m in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        print(
            f"{split:<6} n={m['n']:>4}  MSE={m['mse']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAE={m['mae']:.4f}  Pearson={m['pearson_r']:.4f}  "
            f"Spearman={m['spearman_r']:.4f}"
        )

    # Persist checkpoint.
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "atom_dim": ATOM_FEATURE_DIM,
                "bond_dim": BOND_FEATURE_DIM,
                "hidden": args.hidden,
                "depth": args.depth,
                "dropout": args.dropout,
            },
            "meta": {
                "metal": args.metal,
                "n_rows_total": int(len(df)),
                "n_train": int(len(idx_train)),
                "n_val": int(len(idx_val)),
                "n_test": int(len(idx_test)),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "history": history,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved -> {ckpt_path}")

    # Persist JSON summary.
    summary = {
        "ckpt_path": str(ckpt_path),
        "metal": args.metal,
        "csv": csv_path,
        "n_rows_total": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary JSON  -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
