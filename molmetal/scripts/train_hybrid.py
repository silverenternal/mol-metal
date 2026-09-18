"""Training script for D-MPNN + EGNN hybrid on MetalCytoToxDB.

Usage:
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_hybrid --metal Ru --epochs 30 --batch 32

Outputs:
    molmetal/checkpoints/metal_hybrid_{metal}.pt
    Training curves printed to stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm  # removed for compatibility

from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.featurize import GraphFeaturizer
from molmetal.data.splits import LigandDeduplicatedSplitter
from molmetal.models.loss import MetalCytotoxLoss
from molmetal.models.metal_hybrid import MetalHybridConfig, MetalHybridModel
from molmetal.utils.device import get_device

# Metal name -> embedding index
METAL_TO_IDX = {"Ru": 0, "Ir": 1, "Rh": 2, "Os": 3, "Re": 4, "Pt": 5, "Pd": 6, "Au": 7, "Ag": 8, "Cu": 9}

CHECKPOINT_DIR = Path("molmetal/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def collate_with_3d(
    batch,
    featurizer: GraphFeaturizer,
    device: torch.device,
):
    """Collate a batch of MetalCytoToxDB rows with on-the-fly 3D embedding.

    Args:
        batch: list of dicts from MetalCytotoxDataset.__getitem__
        featurizer: GraphFeaturizer instance
        device: torch device

    Returns:
        smiles_list, coords_tensor, metal_types, pic50_tensor, active_tensor, mask_tensor
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    B = len(batch)
    smiles_list = [item["smiles"] for item in batch]
    pic50_true = np.array([item["pIC50"] for item in batch], dtype=np.float32)
    active_label = np.array([float(item["active"]) for item in batch], dtype=np.float32)

    # 2D features via featurizer
    atom_dim = featurizer.atom_feature_dim
    edge_dim = featurizer.edge_feature_dim

    # Embed conformers first so mol has 3D positions for the same atoms
    # that GraphFeaturizer will extract (avoids H-count mismatch)
    for i, item in enumerate(batch):
        try:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol is not None and mol.GetNumAtoms() > 0:
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass
                batch[i]["_mol"] = mol  # store for featurizer
        except Exception:
            pass

    # Generate 3D coords per molecule (collect first, pad later).
    # CRITICAL: coords_list[i] must always have mol.GetNumAtoms() rows so the
    # padded tensor matches the atom count that GraphFeaturizer will use.
    # If conformer embedding failed, fall back to zero coords of full size.
    coords_list = []
    valid_mask = np.zeros(B, dtype=np.float32)
    for i, item in enumerate(batch):
        mol = item.get("_mol", None)
        n_atoms_mol = mol.GetNumAtoms() if mol is not None else 0
        coords_arr = np.zeros((max(n_atoms_mol, 1), 3), dtype=np.float32)
        if mol is not None and mol.GetNumConformers() > 0:
            try:
                coords_arr = np.array(
                    mol.GetConformer(0).GetPositions(), dtype=np.float32
                )
                # Conformers may have a different (smaller) atom count if
                # RDKit stripped Hs; ensure full-length output.
                if coords_arr.shape[0] < n_atoms_mol:
                    full = np.zeros((n_atoms_mol, 3), dtype=np.float32)
                    full[: coords_arr.shape[0]] = coords_arr
                    coords_arr = full
            except Exception:
                coords_arr = np.zeros((max(n_atoms_mol, 1), 3), dtype=np.float32)
        coords_list.append(coords_arr)
        valid_mask[i] = 1.0 if n_atoms_mol > 0 else 0.0

    # Pad to max atoms across the batch (after conformer generation)
    n_atoms_max = max((c.shape[0] for c in coords_list), default=1)
    coords = np.zeros((B, n_atoms_max, 3), dtype=np.float32)
    for i, c in enumerate(coords_list):
        n = c.shape[0]
        if n > 0:
            coords[i, :n] = c

    metal_types = np.array(
        [METAL_TO_IDX.get(item["metal"], 0) for item in batch], dtype=np.int64
    )

    return (
        smiles_list,
        torch.from_numpy(coords).to(device),
        torch.from_numpy(metal_types).to(device),
        torch.from_numpy(pic50_true).to(device),
        torch.from_numpy(active_label).to(device),
        torch.from_numpy(valid_mask).to(device),
        [item.get("_mol", None) for item in batch],
    )


def _uncollate(batch: dict) -> list:
    """Convert a default-collate dict-of-lists back to a list of dicts."""
    keys = list(batch.keys())
    n = len(batch[keys[0]])
    return [{k: batch[k][i] for k in keys} for i in range(n)]


class _SimpleDataset(torch.utils.data.Dataset):
    """Wraps MetalCytotoxDataset for indexed access."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]


def build_dataloader(
    dataset: MetalCytotoxDataset,
    indices: np.ndarray,
    batch_size: int,
    featurizer: GraphFeaturizer,
    device: torch.device,
    shuffle: bool = True,
):
    """Returns a simple iterator that yields collated + featurized batches."""
    ds = _SimpleDataset(dataset, indices)
    n = len(ds)
    order = list(range(n))
    if shuffle:
        order = list(np.random.permutation(order))
    
    class _BatchLoader:
        def __init__(self, ds, order, batch_size, featurizer, device):
            self.ds = ds
            self.order = order
            self.batch_size = batch_size
            self.featurizer = featurizer
            self.device = device
        
        def __iter__(self):
            for start in range(0, len(self.order), self.batch_size):
                batch_idx = self.order[start:start+self.batch_size]
                items = [self.ds[i] for i in batch_idx]
                yield collate_with_3d(items, self.featurizer, self.device)
        
        def __len__(self):
            return (len(self.order) + self.batch_size - 1) // self.batch_size
    
    return _BatchLoader(ds, order, batch_size, featurizer, device)


def compute_metrics(scores: np.ndarray, labels: np.ndarray):
    """Compute ROC-AUC and PR-AUC."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, scores)
    except ValueError:
        pr_auc = float("nan")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)

        optimizer.zero_grad()
        out = model(smiles_list, coords, metal_types, mol_objects=mol_objects)
        loss = loss_fn(
            out.pic50,
            pic50_true.to(device),
            active_label.to(device),
            mask,
            active_logits=out.active_logits,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    loss_fn,
    device,
) -> dict:
    model.eval()
    all_pic50_pred = []
    all_pic50_true = []
    all_active_logit = []
    all_active_label = []
    all_mask = []
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)

        out = model(smiles_list, coords, metal_types, mol_objects=mol_objects)
        loss = loss_fn(
            out.pic50,
            pic50_true.to(device),
            active_label.to(device),
            mask,
            active_logits=out.active_logits,
        )
        total_loss += loss.item()
        n_batches += 1

        all_pic50_pred.append(out.pic50.cpu().numpy())
        all_pic50_true.append(pic50_true.cpu().numpy() if isinstance(pic50_true, torch.Tensor) else pic50_true)
        # Use logit over the "active" class (index 1) as the AUC score
        all_active_logit.append(out.active_logits[:, 1].cpu().numpy())
        all_active_label.append(active_label.cpu().numpy() if isinstance(active_label, torch.Tensor) else active_label)
        all_mask.append(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask)

    avg_loss = total_loss / max(n_batches, 1)
    pic50_pred_all = np.concatenate(all_pic50_pred)
    pic50_true_all = np.concatenate(all_pic50_true)
    logit_all = np.concatenate(all_active_logit)
    active_all = np.concatenate(all_active_label)
    mask_all = np.concatenate(all_mask)

    # Use logit as the "score" for AUC
    metrics = compute_metrics(logit_all, active_all)

    # pIC50 regression metrics (masked)
    valid = mask_all.astype(bool)
    if valid.sum() > 0:
        diff = pic50_pred_all[valid] - pic50_true_all[valid]
        metrics["pic50_mae"] = float(np.mean(np.abs(diff)))
        metrics["pic50_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    else:
        metrics["pic50_mae"] = float("nan")
        metrics["pic50_rmse"] = float("nan")

    # Classification accuracy at threshold 0.5 on softmax(logits)
    if valid.sum() > 0:
        from scipy.special import softmax

        logits_for_acc = logit_all[valid]
        # If the logits are 1-D (already extracted as a score), softmax on 1-D
        # gives a scalar array which can't be indexed with [:, 1]. Reshape to
        # 2-D (N, 1) so accuracy computation is consistent across shapes.
        if logits_for_acc.ndim == 1:
            probs = 1.0 / (1.0 + np.exp(-logits_for_acc))  # sigmoid
            pred_label = (probs >= 0.5).astype(np.int32)
        else:
            probs = softmax(logits_for_acc, axis=-1)
            pred_label = (probs[:, 1] >= 0.5).astype(np.int32)
        metrics["accuracy"] = float(np.mean(pred_label == active_all[valid].astype(np.int32)))
    else:
        metrics["accuracy"] = float("nan")

    metrics["loss"] = avg_loss
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train D-MPNN + EGNN hybrid model")
    parser.add_argument("--metal", type=str, default="Ru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="loss alpha weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="ligand_dedup",
        choices=["random", "temporal", "ligand_dedup", "scaffold"],
    )
    parser.add_argument("--checkpoint", type=str, default="hybrid_ru.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"[train_hybrid] device={device}")

    # ---- Load data ----
    print(f"[train_hybrid] Loading MetalCytoToxDB for metal={args.metal} ...")
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    dataset = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[train_hybrid] Dataset: {len(dataset)} rows")
    if len(dataset) == 0:
        print("ERROR: Empty dataset after filtering.")
        return

    # ---- Split ----
    if args.split == "ligand_dedup":
        splitter = LigandDeduplicatedSplitter(strategy="largest_first", seed=args.seed)
    elif args.split == "random":
        from molmetal.data.splits import RandomSplitter
        splitter = RandomSplitter(seed=args.seed)
    elif args.split == "temporal":
        from molmetal.data.splits import TemporalSplitter
        splitter = TemporalSplitter(seed=args.seed)
    elif args.split == "scaffold":
        from molmetal.data.splits import ScaffoldSplitter
        splitter = ScaffoldSplitter(seed=args.seed)
    else:
        splitter = LigandDeduplicatedSplitter(strategy="largest_first", seed=args.seed)
    split_result = splitter(dataset)
    train_idx = split_result.train_idx
    val_idx = split_result.val_idx
    test_idx = split_result.test_idx
    print(
        f"[train_hybrid] Train: {len(train_idx)}, "
        f"Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    # ---- Model ----
    model = MetalHybridModel(
        config=MetalHybridConfig(
            n_dmpnn_layers=3,
            n_egnn_layers=3,
            hidden_dim=128,
            dropout=0.1,
        )
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_hybrid] Model: {n_params:,} parameters")

    # ---- Loss + optimizer ----
    loss_fn = MetalCytotoxLoss(alpha=args.alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Featurizer ----
    featurizer = GraphFeaturizer()

    # ---- Dataloaders ----
    train_loader = build_dataloader(
        dataset, train_idx, args.batch, featurizer, device, shuffle=True
    )
    val_loader = build_dataloader(
        dataset, val_idx, args.batch, featurizer, device, shuffle=False
    )

    # ---- Training loop ----
    print(f"[train_hybrid] Starting {args.epochs}-epoch training ...")
    t0 = time.time()
    train_losses = []
    val_aucs = []
    val_prs = []

    best_val_auc = 0.0

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        epoch_dt = time.time() - epoch_t0

        train_losses.append(train_loss)
        val_aucs.append(val_metrics["roc_auc"])
        val_prs.append(val_metrics["pr_auc"])

        print(
            f"  epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_ROC-AUC={val_metrics['roc_auc']:.4f} | "
            f"val_PR-AUC={val_metrics['pr_auc']:.4f} | "
            f"dt={epoch_dt:.1f}s"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            ckpt_path = CHECKPOINT_DIR / f"metal_hybrid_{args.metal}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "val_pr": val_metrics["pr_auc"],
                    "train_loss": train_loss,
                },
                ckpt_path,
            )
            print(f"  -> New best val AUC: {best_val_auc:.4f} (saved to {ckpt_path})")

    total_dt = time.time() - t0
    print(f"\n[train_hybrid] Training complete in {total_dt:.1f}s wall-clock")
    print(f"[train_hybrid] Best val ROC-AUC: {best_val_auc:.4f}")
    print(f"[train_hybrid] Best val PR-AUC:  {val_prs[np.argmax(val_aucs)]:.4f}")

    # ---- Final test-set evaluation ----
    print("\n[train_hybrid] Evaluating on test set ...")
    test_loader = build_dataloader(
        dataset, test_idx, args.batch, featurizer, device, shuffle=False
    )
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(
        f"[train_hybrid] TEST ROC-AUC={test_metrics['roc_auc']:.4f} | "
        f"PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"pIC50 MAE={test_metrics.get('pic50_mae', float('nan')):.4f} | "
        f"RMSE={test_metrics.get('pic50_rmse', float('nan')):.4f} | "
        f"Acc={test_metrics.get('accuracy', float('nan')):.4f}"
    )

    # Persist a side-car metrics JSON next to the checkpoint
    import json
    metrics_path = CHECKPOINT_DIR / Path(args.checkpoint).stem / "metrics.json"
    # Save alongside the checkpoint under a sibling JSON file
    side_car = (CHECKPOINT_DIR / args.checkpoint).with_suffix(".json")
    side_car.write_text(
        json.dumps(
            {
                "metal": args.metal,
                "split": args.split,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "alpha": args.alpha,
                "seed": args.seed,
                "n_params": n_params,
                "wall_clock_seconds": total_dt,
                "best_val_auc": best_val_auc,
                "best_val_pr": val_prs[np.argmax(val_aucs)] if val_prs else float("nan"),
                "test": test_metrics,
                "train_losses": train_losses,
                "val_aucs": val_aucs,
                "val_prs": val_prs,
            },
            indent=2,
        )
    )
    print(f"[train_hybrid] Metrics side-car: {side_car}")

    # ---- Final summary ----
    print("\n=== Training Summary ===")
    print(f"Metal:        {args.metal}")
    print(f"Split:        {args.split}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Model params: {n_params:,}")
    print(f"Wall-clock:   {total_dt:.1f}s")
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Best val PR:  {val_prs[np.argmax(val_aucs)]:.4f}")
    print(f"TEST ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"TEST pIC50 MAE: {test_metrics.get('pic50_mae', float('nan')):.4f}")
    print(f"Train loss curve (last 5): {train_losses[-5:]}")
    print(f"Honest baseline (Morgan+XGB, ligand_dedup): ~0.80")
    if test_metrics["roc_auc"] > 0.80:
        print("RESULT: D-MPNN+EGNN HYBRID > Morgan+XGB baseline!")
    else:
        print(
            f"RESULT: D-MPNN+EGNN HYBRID (AUC={test_metrics['roc_auc']:.4f}) "
            f"<= Morgan+XGB baseline (0.80)"
        )


if __name__ == "__main__":
    main()
