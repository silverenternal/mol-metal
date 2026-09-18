"""T3 training script: D-MPNN + EGNN hybrid V3 on MetalCytoToxDB (Ru subset).

Loads the tmQM-pretrained D-MPNN encoder from
``molmetal/checkpoints/dmpnn_tmqm_pretrained.pt`` and trains
:class:`MetalHybridV3Model` on the Ru subset of MetalCytoToxDB
under multiple splits (random / scaffold / temporal / ligand_dedup).

Usage:
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_metal_hybrid --metal Ru --epochs 5 \\
        --seeds 1 2 3 --splits scaffold temporal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset  # noqa: E402
from molmetal.data.splits import (  # noqa: E402
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)
from molmetal.models.loss import MetalCytotoxLoss  # noqa: E402
from molmetal.models.metal_hybrid_v3 import MetalHybridV3Config, MetalHybridV3Model  # noqa: E402
from molmetal.utils.device import get_device  # noqa: E402

from molmetal.scripts.train_hybrid import (  # noqa: E402
    METAL_TO_IDX,
    build_dataloader,
    compute_metrics,
    train_epoch,
    evaluate,
)

CHECKPOINT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _split_for(name: str, seed: int):
    if name == "random":
        return RandomSplitter(seed=seed)
    if name == "temporal":
        # TemporalSplitter has no `seed` kwarg (deterministic on cutoff_year).
        # Seed the global RNG so any stochastic validation shuffle is reproducible.
        np.random.seed(seed)
        return TemporalSplitter()
    if name == "scaffold":
        return ScaffoldSplitter(seed=seed)
    if name == "ligand_dedup":
        return LigandDeduplicatedSplitter(strategy="largest_first", seed=seed)
    raise ValueError(f"Unknown split: {name}")


def _subset_indices_by_indices(dataset, all_idx, metal):
    """Pick only rows in ``all_idx`` that belong to ``metal`` (cheap filter)."""
    arr = np.asarray(all_idx, dtype=np.int64)
    metals = dataset.metals
    mask = np.array([str(metals[int(i)]) == str(metal) for i in arr])
    return arr[mask]


def _train_one(
    metal: str,
    split: str,
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
    alpha: float,
    coord_weight: float,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"\n[train_metal_hybrid] metal={metal} split={split} seed={seed} device={device}")

    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        compute_pic50=True,
        compute_active=True,
    )
    dataset = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[train_metal_hybrid] Dataset (full {metal}): {len(dataset)} rows")
    if len(dataset) == 0:
        return {"metal": metal, "split": split, "seed": seed, "error": "empty_dataset"}

    splitter = _split_for(split, seed)
    try:
        split_result = splitter(dataset)
    except Exception as e:
        print(f"[train_metal_hybrid] splitter failed: {e}")
        return {"metal": metal, "split": split, "seed": seed, "error": str(e)}

    # Filter split indices to the requested metal (splitters operate over the
    # unfiltered dataset, but here the dataset is already metal-scoped).
    train_idx = split_result.train_idx
    val_idx = split_result.val_idx
    test_idx = split_result.test_idx
    print(
        f"[train_metal_hybrid] split sizes: train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )

    cfg = MetalHybridV3Config(
        base=None,
        coord_loss_weight=coord_weight,
        load_pretrained_encoder=True,
    )
    cfg.base = None  # use default
    model = MetalHybridV3Model(config=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_metal_hybrid] model={n_params:,} params, pretrained={model.pretrained_loaded}")

    loss_fn = MetalCytotoxLoss(alpha=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    featurizer = model.featurizer
    train_loader = build_dataloader(dataset, train_idx, batch, featurizer, device, shuffle=True)
    val_loader = build_dataloader(dataset, val_idx, batch, featurizer, device, shuffle=False)
    test_loader = build_dataloader(dataset, test_idx, batch, featurizer, device, shuffle=False)

    t0 = time.time()
    history = []
    best_val_auc = -1.0
    for epoch in range(epochs):
        ep_t = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics.get("loss", float("nan"))),
            "val_auc": float(val_metrics.get("roc_auc", float("nan"))),
            "val_pr": float(val_metrics.get("pr_auc", float("nan"))),
            "val_pic50_mae": float(val_metrics.get("pic50_mae", float("nan"))),
            "seconds": float(time.time() - ep_t),
        })
        print(
            f"  epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = float(val_metrics["roc_auc"])

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(
        f"[train_metal_hybrid] TEST split={split} seed={seed} "
        f"AUC={test_metrics['roc_auc']:.4f} PR={test_metrics['pr_auc']:.4f} "
        f"pic50_mae={test_metrics['pic50_mae']:.4f}"
    )

    elapsed = time.time() - t0
    return {
        "metal": metal,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "n_params": n_params,
        "pretrained_loaded": bool(model.pretrained_loaded),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_val_auc": best_val_auc,
        "test_auc": float(test_metrics["roc_auc"]),
        "test_pr": float(test_metrics["pr_auc"]),
        "test_pic50_mae": float(test_metrics["pic50_mae"]),
        "test_pic50_rmse": float(test_metrics["pic50_rmse"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "seconds": float(elapsed),
        "history": history,
    }


def _aggregate(rows):
    """Aggregate per-seed rows: mean / std of AUC."""
    if not rows:
        return {}
    aucs = np.array([r["test_auc"] for r in rows if not np.isnan(r["test_auc"])], dtype=float)
    prs = np.array([r["test_pr"] for r in rows if not np.isnan(r["test_pr"])], dtype=float)
    pic50s = np.array(
        [r["test_pic50_mae"] for r in rows if not np.isnan(r["test_pic50_mae"])], dtype=float
    )
    out = {
        "n_seeds": len(aucs),
        "auc_mean": float(aucs.mean()) if aucs.size else float("nan"),
        "auc_std": float(aucs.std(ddof=0)) if aucs.size > 0 else float("nan"),
        "pr_mean": float(prs.mean()) if prs.size else float("nan"),
        "pr_std": float(prs.std(ddof=0)) if prs.size > 0 else float("nan"),
        "pic50_mae_mean": float(pic50s.mean()) if pic50s.size else float("nan"),
        "pic50_mae_std": float(pic50s.std(ddof=0)) if pic50s.size > 0 else float("nan"),
        "aucs": aucs.tolist(),
        "prs": prs.tolist(),
        "pic50_maes": pic50s.tolist(),
    }
    return out


def main():
    p = argparse.ArgumentParser(description="Train D-MPNN+EGNN hybrid V3 on Ru subset")
    p.add_argument("--metal", type=str, default="Ru")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--coord-weight", type=float, default=0.05)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3],
        help=">=3 seeds for variance estimate",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["scaffold", "temporal"],
        choices=["random", "temporal", "ligand_dedup", "scaffold"],
    )
    p.add_argument("--out", type=str, default=str(REPORTS_DIR / "metal_hybrid_ru_results.json"))
    args = p.parse_args()

    if len(args.seeds) < 3:
        print("[train_metal_hybrid] WARN: <3 seeds requested — variance estimate will be unreliable")

    all_results = {"metal": args.metal, "epochs": args.epochs, "by_split": {}}

    for split in args.splits:
        per_seed = []
        for seed in args.seeds:
            res = _train_one(
                metal=args.metal,
                split=split,
                seed=seed,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                alpha=args.alpha,
                coord_weight=args.coord_weight,
            )
            per_seed.append(res)
        agg = _aggregate(per_seed)
        all_results["by_split"][split] = {"per_seed": per_seed, "aggregate": agg}
        print(
            f"[train_metal_hybrid] split={split} AUC={agg['auc_mean']:.4f}±{agg['auc_std']:.4f} "
            f"(n={agg['n_seeds']})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"[train_metal_hybrid] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
