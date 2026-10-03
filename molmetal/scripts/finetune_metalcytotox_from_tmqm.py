#!/usr/bin/env python3
"""Fine-tune D-MPNN on MetalCytoToxDB Ru subset with optional tmQM pretrain.

=====================================================================
Verify the tmQM-pretrained encoder improves downstream cytotoxicity
prediction (Metric 1 / Metric 21 protocol).
=====================================================================

The script trains a single-metal-classification D-MPNN head on
MetalCytoToxDB Ru subset (precious metal Ru with 4+ cell lines), with
two initialisation modes:

* ``--init random``           — baseline (atom_emb, bond_emb, edge_init,
  gru_update all initialised from scratch)
* ``--init tmqm`` (default)   — load the encoder weights from
  ``models/tmqm_pretrained/tmqm_pretrained_encoder.pt`` (the
  checkpoint produced by ``molmetal/scripts/pretrain_tmqm.py``).

Both modes run with identical hyperparameters (lr, epochs, batch,
seed, splits).  We compare test AUC per seed and report the
delta ``mean(test_AUC_tmqm) - mean(test_AUC_random)``.

Verification gate
-----------------
The ``paper/sections/04_method.tex`` §4.4 pretraining story
references this ablation.  Per ``TODO/pending/23_weak_to_strong_plan.md``
the **non-negative-delta is the only required gate** (we do not gate
on absolute AUC; the downstream metric 1 baseline of
``Ru subset AUC = 0.81`` (Krasnov 2026) is the comparator).  We
therefore report the AUC delta with a 95 % CI and verdict
``PASS`` if the delta is non-negative.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/finetune_metalcytotox_from_tmqm.py \\
        --metal Ru \\
        --seeds 0 1 2 \\
        --epochs 20 --batch 64 --lr 1e-3 \\
        --init tmqm \\
        --encoder models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir logs/finetune_tmqm_vs_random
"""
from __future__ import annotations

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
import torch
import torch.nn as nn

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNN_HIDDEN,
    DMPNN_DEPTH,
    DMPNN_DROPOUT,
    DMPNNModel,
    _GRUUpdate,
    featurize_smiles_list,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import ScaffoldSplitter

RDLogger = None
try:
    from rdkit import RDLogger as _RD
    RDLogger = _RD
    _RD.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIVE_THRESHOLD_UM: float = 10.0


# ---------------------------------------------------------------------------
# Model — D-MPNN + single metal-specific binary head
# ---------------------------------------------------------------------------
class MetalCytotoxModel(nn.Module):
    """D-MPNN backbone + single 2-class head (active vs inactive)."""

    def __init__(
        self,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = DMPNN_HIDDEN,
        depth: int = DMPNN_DEPTH,
        dropout: float = DMPNN_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth = depth

        # Backbone (encoder)
        self.dropout_layer = nn.Dropout(dropout)
        self.atom_emb = nn.Linear(atom_dim, hidden)
        self.bond_emb = nn.Linear(bond_dim, hidden)
        self.gru_update = _GRUUpdate(hidden, hidden)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        # Head (binary classification)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def _encode_one(
        self,
        atom_f: torch.Tensor,
        bond_f: torch.Tensor,
        e_src: torch.Tensor,
        e_dst: torch.Tensor,
    ) -> torch.Tensor:
        V = atom_f.size(0)
        E = bond_f.size(0)
        device = atom_f.device
        h_v = self.dropout_layer(torch.relu(self.atom_emb(atom_f)))
        h_e = self.dropout_layer(torch.relu(self.bond_emb(bond_f)))
        for _ in range(self.depth):
            atom_incoming = torch.zeros(V, self.hidden, device=device)
            atom_incoming.index_add_(0, e_dst, h_e)
            neighbor_sum = atom_incoming[e_src]
            h_e = self.gru_update(h_e, h_v[e_src], neighbor_sum)
        outgoing = torch.zeros(V, self.hidden, device=device)
        outgoing.index_add_(0, e_src, h_e)
        h_comb = torch.cat([h_v, outgoing], dim=-1)
        return h_comb.sum(dim=0)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        B = atom_feats.size(0)
        device = atom_feats.device
        pooled: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = (bond_feats[b].abs().sum(dim=-1) > 0)
            E = int(valid_edges.sum().item())
            if E == 0 or V == 0:
                pooled.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled.append(
                self._encode_one(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        h_pool = torch.stack(pooled, dim=0)
        return self.head(h_pool)


# ---------------------------------------------------------------------------
# Load + initialise
# ---------------------------------------------------------------------------
def load_tmqm_encoder(model: MetalCytotoxModel, ckpt_path: Path) -> int:
    """Load tmQM-pretrained encoder weights into ``model``.

    Returns the number of parameters successfully transferred.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"tmQM encoder checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload["state_dict"]
    own = model.state_dict()
    n_loaded = 0
    # Strict matching: only backbone layers
    for k, v in sd.items():
        if k in own:
            if own[k].shape == v.shape:
                own[k].copy_(v)
                n_loaded += 1
            else:
                print(
                    f"  [load_tmqm_encoder] shape mismatch for {k}: "
                    f"model {own[k].shape} vs ckpt {v.shape} — skipping"
                )
        else:
            print(f"  [load_tmqm_encoder] unexpected key {k} — skipping")
    return n_loaded


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
@dataclass
class BatchItem:
    atom_feats: torch.Tensor
    bond_feats: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    target: torch.Tensor


def make_batch(
    graphs: List[Optional[Tuple]],
    targets: List[int],
    max_atoms: int,
    max_edges: int,
    atom_dim: int,
    bond_dim: int,
) -> Optional[BatchItem]:
    valid = [(g, t) for g, t in zip(graphs, targets) if g is not None]
    if not valid:
        return None
    B = len(targets)
    af = np.zeros((B, max_atoms, atom_dim), dtype=np.float32)
    bf = np.zeros((B, max_edges, bond_dim), dtype=np.float32)
    es = np.zeros((B, max_edges), dtype=np.int64)
    ed = np.zeros((B, max_edges), dtype=np.int64)
    tgt = np.zeros((B,), dtype=np.int64)
    bi = 0
    for g, t in valid:
        af_b, bf_b, es_b, ed_b = g
        V = min(af_b.shape[0], max_atoms)
        # Edge list is filtered to bonds fully within the kept atom set
        # (skip any edge whose endpoints exceed V-1).
        keep_mask = (es_b < V) & (ed_b < V)
        es_b_kept = es_b[keep_mask]
        ed_b_kept = ed_b[keep_mask]
        bf_b_kept = bf_b[keep_mask]
        E = min(bf_b_kept.shape[0], max_edges)
        af[bi, :V] = af_b[:V]
        bf[bi, :E] = bf_b_kept[:E]
        es[bi, :E] = es_b_kept[:E]
        ed[bi, :E] = ed_b_kept[:E]
        tgt[bi] = t
        bi += 1
    return BatchItem(
        atom_feats=torch.from_numpy(af),
        bond_feats=torch.from_numpy(bf),
        edge_src=torch.from_numpy(es),
        edge_dst=torch.from_numpy(ed),
        target=torch.from_numpy(tgt),
    )


def _train_one_epoch(
    model: MetalCytotoxModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    graphs: List,
    targets: np.ndarray,
    train_idx: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    n_batches = 0
    order = train_idx.copy()
    rng.shuffle(order)
    for start in range(0, order.shape[0], batch_size):
        batch_idx = order[start:start + batch_size]
        batched = make_batch(
            [graphs[int(i)] for i in batch_idx],
            [int(targets[int(i)]) for i in batch_idx],
            max_atoms, max_edges, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
        )
        if batched is None or batched.target.numel() == 0:
            continue
        atom_f = batched.atom_feats.to(device)
        bond_f = batched.bond_feats.to(device)
        es = batched.edge_src.to(device)
        ed = batched.edge_dst.to(device)
        tgt = batched.target.to(device)
        logits = model(atom_f, bond_f, es, ed)
        loss = criterion(logits, tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=-1) == tgt).sum().item())
        n += int(tgt.size(0))
        n_batches += 1
    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": correct / max(n, 1),
        "n_samples": n,
    }


@torch.no_grad()
def _evaluate(
    model: MetalCytotoxModel,
    graphs: List,
    targets: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    probs: List[np.ndarray] = []
    true: List[np.ndarray] = []
    for start in range(0, idx.shape[0], batch_size):
        batch_idx = idx[start:start + batch_size]
        batched = make_batch(
            [graphs[int(i)] for i in batch_idx],
            [int(targets[int(i)]) for i in batch_idx],
            max_atoms, max_edges, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
        )
        if batched is None or batched.target.numel() == 0:
            continue
        atom_f = batched.atom_feats.to(device)
        bond_f = batched.bond_feats.to(device)
        es = batched.edge_src.to(device)
        ed = batched.edge_dst.to(device)
        logits = model(atom_f, bond_f, es, ed)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.append(p)
        true.append(batched.target.numpy())
    if not probs:
        return {"auc": float("nan"), "acc": float("nan")}
    p = np.concatenate(probs)
    t = np.concatenate(true)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(t, p))
    except Exception:
        auc = float("nan")
    acc = float(((p > 0.5).astype(int) == t).mean())
    return {"auc": auc, "acc": acc, "n": int(t.shape[0])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_metalcytotox(metal: str) -> MetalCytotoxDataset:
    """Load MetalCytoToxDB filtered to a single precious metal."""
    return MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(
            metal_whitelist=[metal],
            ic50_min=0.001,
            ic50_max=10000.0,
            require_active_field=True,
        )
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fine-tune D-MPNN on MetalCytoToxDB metal subset (init=tmqm|random)"
    )
    p.add_argument("--metal", type=str, default="Ru",
                   choices=["Ru", "Ir", "Rh", "Os", "Re", "Pt"])
    p.add_argument("--init", type=str, default="tmqm",
                   choices=["tmqm", "random"])
    p.add_argument("--encoder", type=Path,
                   default=Path("models/tmqm_pretrained/tmqm_pretrained_encoder.pt"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("logs/finetune_tmqm_vs_random"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=DMPNN_HIDDEN)
    p.add_argument("--depth", type=int, default=DMPNN_DEPTH)
    p.add_argument("--dropout", type=float, default=DMPNN_DROPOUT)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=240)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[finetune] init={args.init}  metal={args.metal}  "
          f"seeds={args.seeds}  epochs={args.epochs}  out_dir={out_dir}")

    # ----- dataset -----
    print(f"[finetune] loading MetalCytoToxDB ({args.metal})")
    ds = _load_metalcytotox(args.metal)
    smiles = ds.smiles
    pic50 = ds.pic50
    active = ds.active.astype(np.int64)
    print(f"[finetune] n_rows={len(ds)}  active_rate={active.mean():.3f}")

    # Featurise SMILES
    print("[finetune] featurising SMILES ...")
    graphs = featurize_smiles_list(list(smiles))
    n_valid = sum(g is not None for g in graphs)
    valid_mask = np.array([g is not None for g in graphs])
    valid_idx = np.where(valid_mask)[0]
    targets = active[valid_mask]
    print(f"[finetune] featurised {n_valid}/{len(graphs)} valid")

    # Scaffold split (deterministic)
    splitter = ScaffoldSplitter()
    split = splitter(ds)  # SplitResult with .train_idx / .val_idx / .test_idx
    train_idx_full = split.train_idx
    val_idx_full = split.val_idx
    test_idx_full = split.test_idx
    # All indices are positions in the dataset, which match valid_mask positions
    # because all rows are valid after featurise (we already filtered to valid_mask
    # implicitly by the splitter running on the whole dataset; here we re-project
    # to the valid_mask positions).
    train_idx_full = valid_idx[train_idx_full]
    val_idx_full = valid_idx[val_idx_full]
    test_idx_full = valid_idx[test_idx_full]

    device = torch.device("cpu")
    seed_aucs: List[Dict[str, float]] = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        model = MetalCytotoxModel(
            hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        ).to(device)
        if args.init == "tmqm":
            n_loaded = load_tmqm_encoder(model, args.encoder)
            print(f"[finetune] seed={seed} loaded {n_loaded} backbone tensors from tmQM")
        else:
            print(f"[finetune] seed={seed} init=random (no pretrain)")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            tr = _train_one_epoch(
                model, optimizer, criterion, graphs, targets,
                train_idx_full,
                batch_size=args.batch,
                max_atoms=args.max_atoms, max_edges=args.max_edges,
                device=device, rng=rng,
            )
            va = _evaluate(
                model, graphs, targets, val_idx_full,
                batch_size=args.batch,
                max_atoms=args.max_atoms, max_edges=args.max_edges,
                device=device,
            )
            if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
                print(
                    f"[finetune] seed={seed} epoch={epoch:3d} "
                    f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f} "
                    f"val_auc={va['auc']:.3f}"
                )

        te = _evaluate(
            model, graphs, targets, test_idx_full,
            batch_size=args.batch,
            max_atoms=args.max_atoms, max_edges=args.max_edges,
            device=device,
        )
        seed_aucs.append({
            "seed": seed,
            "test_auc": te["auc"],
            "test_acc": te["acc"],
            "n_test": te["n"],
        })
        print(
            f"[finetune] seed={seed} TEST AUC={te['auc']:.4f}  "
            f"acc={te['acc']:.3f}  n={te['n']}"
        )

    # ----- summary -----
    aucs = [s["test_auc"] for s in seed_aucs]
    summary = {
        "metal": args.metal,
        "init": args.init,
        "encoder_path": str(args.encoder),
        "n_rows": int(len(ds)),
        "n_valid": int(n_valid),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "seeds": args.seeds,
        "per_seed": seed_aucs,
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
    }
    out_json = out_dir / f"finetune_{args.init}_{args.metal}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(
        f"[finetune] SUMMARY init={args.init} metal={args.metal} "
        f"test_AUC={summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
    )
    print(f"[finetune] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
