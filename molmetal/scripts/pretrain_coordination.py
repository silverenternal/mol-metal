"""Pre-train a D-MPNN on tmQM coordination chemistry (TODO F2 / P1).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.pretrain_coordination --epochs 10

What it does
------------
Self-supervised-ish pre-training on the 21.6k Pt/Ru/Ir complexes of tmQM
(Balcells & Skjelstad, JCIM 2020; TPSSh-D3BJ/def2-SVP).  Two DFT-derived
regression targets are predicted jointly from the 2-D molecular graph:

  (a) **metal coordination number** — tmQM's MND (metal node degree)
  (b) **Wiberg bond order at the metal** — the metal atom's total BO

Both are properties *of the metal centre*, so the readout is deliberately
metal-aware: the head consumes ``[h_metal_atom ‖ mean-pooled h_graph]``
rather than a plain graph pooling.  This is the inductive bias we want to
transfer to MetalCytoToxDB, where the metal centre drives cytotoxicity.

Targets are z-scored for a balanced joint MSE; metrics are reported back in
raw units (MAE in coordination number / bond-order units).

Architecture reuse
------------------
``molmetal.models.dmpnn.DirectedMPNN`` (unchanged) + a 2-output MLP head.
Batching uses the standard block-diagonal (disjoint union) trick so a whole
batch is a single ``forward_per_atom`` call — index-based scatter makes this
exactly equivalent to per-molecule forwards, but ~20x faster on ROCm.

Checkpoint
----------
``molmetal/checkpoints/dmpnn_tmqm_pretrained.pt`` stores the encoder +
head state dicts, the MPNNConfig, the target normalisation constants and
the full loss curve, so ``load_pretrained_encoder()`` can be used for
fine-tuning without re-reading tmQM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: E402  (added for round-5 GELU-MLP helper)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.featurize import GraphFeaturizer  # noqa: E402
from molmetal.data.tmqm import PAPER_METALS, TRANSITION_METALS, load_tmqm  # noqa: E402
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig, _scatter_sum  # noqa: E402

# Round-5 Triton wiring.  The pretrain head is a 3-layer MLP
# ``Linear(2d, d) → ReLU → Dropout → Linear(d, d/2) → ReLU →
# Linear(d/2, 2)``.  We expose a GELU-MLP helper that can be called
# from per-task heads (coord_number / metal_bo) — currently a no-op
# fallback (the head uses ReLU).  When GELU is desired at a call site
# the helper selects batched-fused-GELU-MLP via the Triton gate.
from triton_kernels import (
    batched_fused_gelu_mlp as _batched_fused_gelu_mlp,
)
from triton_kernels.config import triton_config as _triton_config


def _maybe_batched_gelu_mlp(
    x: torch.Tensor,
    linear_layers: list[nn.Linear],
    *,
    training: bool,
) -> torch.Tensor:
    """Run K stacked ``Linear → GELU → Linear`` triplets via the batched
    fused GELU MLP kernel when the gate is open; else fall back to the
    sequential ``F.gelu`` + Linear chain.

    Parameters
    ----------
    x : ``(M, D_in)`` tensor
        Input shared across all K MLPs.
    linear_layers : list of ``2*K`` ``nn.Linear`` modules
        Layer weights arranged as ``[w_1a, w_1b, w_2a, w_2b, ...]``
        where triplet ``k`` is ``Linear_k_a → GELU → Linear_k_b``.

    Returns
    -------
    ``(M, D_out)`` tensor.

    Notes
    -----
    The pretrain head MLP uses ReLU; this helper is wired so a future
    GELU-based head can opt in by calling it.  When the model is in
    ReLU mode (the default), callers should use the original
    ``nn.Sequential`` chain for bit-exact parity with published results.
    """
    K = len(linear_layers) // 2
    if K == 0:
        return x
    if K == 1 or (not x.is_cuda) or (not _triton_config.should_use_fused(
        x, op="batched_gelu_mlp"
    )):
        # Sequential fallback: explicit ``F.gelu`` + linear chain.
        h = x
        for k in range(K):
            la = linear_layers[2 * k]
            lb = linear_layers[2 * k + 1]
            h = F.gelu(la(h))
            h = lb(h)
        return h

    D_in = linear_layers[0].in_features
    w1s = torch.stack(
        [la.weight.t().contiguous() for la in (linear_layers[0::2])], dim=0
    )
    b1s = torch.stack([la.bias for la in (linear_layers[0::2])], dim=0)
    w2s = torch.stack(
        [lb.weight.t().contiguous() for lb in (linear_layers[1::2])], dim=0
    )
    b2s = torch.stack([lb.bias for lb in (linear_layers[1::2])], dim=0)
    return _batched_fused_gelu_mlp(x, w1s, b1s, w2s, b2s)

CHECKPOINT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
DEFAULT_CKPT = CHECKPOINT_DIR / "dmpnn_tmqm_pretrained.pt"

_TM_SET = frozenset(TRANSITION_METALS)


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------
@dataclass
class GraphSample:
    x: np.ndarray            # (n_atoms, 39)
    edge_index: np.ndarray   # (2, n_edges)
    edge_attr: np.ndarray    # (n_edges, 6)
    metal_idx: int
    coord_number: float
    metal_bo: float


def featurize_rows(
    smiles: Sequence[str],
    coord_numbers: Sequence[float],
    metal_bos: Sequence[float],
    verbose: bool = True,
) -> Tuple[List[GraphSample], int]:
    """SMILES -> graph samples, skipping unparseable / metal-less entries.

    Returns ``(samples, n_failed)``.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    gf = GraphFeaturizer()
    out: List[GraphSample] = []
    n_failed = 0
    total = len(smiles)
    for i, smi in enumerate(smiles):
        if verbose and total > 5000 and i and i % 5000 == 0:
            print(f"  [featurize] {i:,}/{total:,} ({len(out):,} kept)", flush=True)
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                n_failed += 1
                continue
            metal_idx = next(
                (
                    a.GetIdx()
                    for a in mol.GetAtoms()
                    if a.GetSymbol() in _TM_SET
                ),
                None,
            )
            if metal_idx is None:
                n_failed += 1
                continue
            g = gf(mol)
            if g["edge_index"].shape[1] == 0:
                n_failed += 1
                continue
            out.append(
                GraphSample(
                    x=g["x"].astype(np.float32),
                    edge_index=g["edge_index"].astype(np.int64),
                    edge_attr=g["edge_attr"].astype(np.float32),
                    metal_idx=int(metal_idx),
                    coord_number=float(coord_numbers[i]),
                    metal_bo=float(metal_bos[i]),
                )
            )
        except Exception:
            n_failed += 1
            continue
    return out, n_failed


def collate(
    samples: Sequence[GraphSample], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Block-diagonal (disjoint-union) batching of variable-size graphs."""
    xs, eis, eas, batch, metal_idx = [], [], [], [], []
    offset = 0
    for b, s in enumerate(samples):
        n = s.x.shape[0]
        xs.append(s.x)
        eis.append(s.edge_index + offset)
        eas.append(s.edge_attr)
        batch.append(np.full(n, b, dtype=np.int64))
        metal_idx.append(s.metal_idx + offset)
        offset += n
    return {
        "x": torch.from_numpy(np.concatenate(xs, 0)).to(device),
        "edge_index": torch.from_numpy(np.concatenate(eis, 1)).to(device),
        "edge_attr": torch.from_numpy(np.concatenate(eas, 0)).to(device),
        "batch": torch.from_numpy(np.concatenate(batch, 0)).to(device),
        "metal_idx": torch.tensor(metal_idx, dtype=torch.long, device=device),
        "y": torch.tensor(
            [[s.coord_number, s.metal_bo] for s in samples],
            dtype=torch.float32,
            device=device,
        ),
        "n_nodes": torch.tensor(
            [s.x.shape[0] for s in samples], dtype=torch.float32, device=device
        ).unsqueeze(-1),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CoordinationPretrainer(nn.Module):
    """D-MPNN encoder + metal-aware 2-task regression head."""

    def __init__(
        self,
        config: Optional[MPNNConfig] = None,
        *,
        use_gelu_head: bool = False,
    ):
        super().__init__()
        self.config = config or MPNNConfig(atom_feat_dim=39, edge_feat_dim=6)
        self.encoder = DirectedMPNN(self.config)
        d = self.config.hidden_dim
        # Default: ReLU head (matches the published tmQM checkpoint).
        # When ``use_gelu_head=True`` we swap to GELU + wire the
        # batched-fused-GELU-MLP path for the first
        # ``Linear → GELU → Linear`` triplet (K=1 → sequential fused).
        # The trailing ``Linear(d/2, 2)`` is unchanged (no fused-MLP
        # benefit for the final down-projection).
        self.use_gelu_head = bool(use_gelu_head)
        act_cls = nn.GELU if self.use_gelu_head else nn.ReLU
        self.head = nn.Sequential(
            nn.Linear(2 * d, d),
            act_cls(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d, d // 2),
            act_cls(),
            nn.Linear(d // 2, 2),  # [coord_number_z, metal_bo_z]
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the (B, 2*hidden) metal-aware graph representation."""
        h = self.encoder.forward_per_atom(
            batch["x"], batch["edge_index"], batch["edge_attr"]
        )  # (N_total, D)
        n_graphs = int(batch["batch"].max().item()) + 1
        pooled = _scatter_sum(h, batch["batch"], dim=0, dim_size=n_graphs)
        pooled = pooled / batch["n_nodes"].clamp(min=1.0)  # mean pool
        h_metal = h[batch["metal_idx"]]  # (B, D)
        return torch.cat([h_metal, pooled], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode(batch)
        # Round-5 batched-fused-GELU-MLP wiring.  When the head uses
        # GELU (opt-in via ``use_gelu_head=True``) we route the first
        # ``Linear → GELU → Linear`` triplet through the fused kernel
        # when the gate is open; otherwise we use the sequential head.
        if self.use_gelu_head and self.training and h.is_cuda:
            try:
                h = _maybe_batched_gelu_mlp(
                    h,
                    [self.head[0], self.head[3]],
                    training=True,
                )
                # Apply the dropout + the rest of the head.
                h = self.head[2](h)  # dropout
                h = self.head[4](h)  # Linear(d/2, 2)
                return h
            except Exception:
                # Any kernel failure -> fall back to the full sequential
                # head (preserves robustness).
                pass
        return self.head(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(
    model: CoordinationPretrainer,
    samples: List[GraphSample],
    order: np.ndarray,
    batch_size: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_cn = tot_bo = 0.0
    abs_cn = abs_bo = 0.0
    n_seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = collate([samples[i] for i in idx], device)
            y_z = (batch["y"] - mean) / std
            pred_z = model(batch)
            per_task = ((pred_z - y_z) ** 2).mean(dim=0)  # (2,)
            loss = per_task.mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = len(idx)
            n_seen += bs
            tot_loss += float(loss.item()) * bs
            tot_cn += float(per_task[0].item()) * bs
            tot_bo += float(per_task[1].item()) * bs
            raw = pred_z.detach() * std + mean
            err = (raw - batch["y"]).abs().mean(dim=0)
            abs_cn += float(err[0].item()) * bs
            abs_bo += float(err[1].item()) * bs
    return {
        "loss": tot_loss / max(n_seen, 1),
        "mse_coord_z": tot_cn / max(n_seen, 1),
        "mse_bo_z": tot_bo / max(n_seen, 1),
        "mae_coord": abs_cn / max(n_seen, 1),
        "mae_bo": abs_bo / max(n_seen, 1),
    }


def load_pretrained_encoder(
    path: Path | str = DEFAULT_CKPT,
    map_location: str = "cpu",
) -> Tuple[DirectedMPNN, Dict[str, object]]:
    """Rebuild the pre-trained D-MPNN encoder for downstream fine-tuning.

    Returns ``(encoder, metadata)``; the encoder's ``forward_per_atom`` /
    ``forward`` can be dropped straight into the MetalCytoToxDB models.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = MPNNConfig(**ckpt["mpnn_config"])
    encoder = DirectedMPNN(cfg)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return encoder, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train D-MPNN on tmQM.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--metals", nargs="*", default=list(PAPER_METALS))
    p.add_argument("--all-metals", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap n samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    metals = None if args.all_metals else list(args.metals)
    print(f"[pretrain] device={device} metals={metals or 'ALL'} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr}")

    df = load_tmqm(metals=metals, require_smiles=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"[pretrain] tmQM rows: {len(df):,}")

    t0 = time.time()
    samples, n_failed = featurize_rows(
        df["smiles"].tolist(),
        df["coord_number"].tolist(),
        df["metal_bo_total"].tolist(),
    )
    print(f"[pretrain] featurised {len(samples):,} graphs "
          f"({n_failed:,} skipped) in {time.time() - t0:.1f}s")
    if not samples:
        print("[pretrain] ERROR: no usable samples")
        return 1

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    n_val = int(len(samples) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[pretrain] train={len(train_idx):,}  val={len(val_idx):,}")

    y_train = np.array(
        [[samples[i].coord_number, samples[i].metal_bo] for i in train_idx],
        dtype=np.float32,
    )
    mean = torch.tensor(y_train.mean(0), device=device)
    std = torch.tensor(y_train.std(0).clip(min=1e-6), device=device)
    print(f"[pretrain] target mean={mean.tolist()} std={std.tolist()}")

    cfg = MPNNConfig(
        atom_feat_dim=samples[0].x.shape[1],
        edge_feat_dim=samples[0].edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = CoordinationPretrainer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params: {n_params:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        order = rng.permutation(train_idx)
        tr = run_epoch(model, samples, order, args.batch_size, device, mean, std, optimizer)
        va = (
            run_epoch(model, samples, val_idx, args.batch_size, device, mean, std, None)
            if len(val_idx) else dict.fromkeys(tr, float("nan"))
        )
        rec = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_mae_coord": tr["mae_coord"],
            "val_mae_coord": va["mae_coord"],
            "train_mae_bo": tr["mae_bo"],
            "val_mae_bo": va["mae_bo"],
            "seconds": time.time() - t_ep,
        }
        history.append(rec)
        best_val = min(best_val, va["loss"]) if va["loss"] == va["loss"] else best_val
        print(
            f"[pretrain] epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"MAE_CN(train/val)={tr['mae_coord']:.3f}/{va['mae_coord']:.3f}  "
            f"MAE_BO(train/val)={tr['mae_bo']:.3f}/{va['mae_bo']:.3f}  "
            f"({rec['seconds']:.1f}s)",
            flush=True,
        )

    meta = {
        "source": "tmQM (Balcells & Skjelstad, JCIM 2020) TPSSh-D3BJ/def2-SVP",
        "metals": metals or "ALL",
        "n_pretraining_samples": len(samples),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_failed_featurization": int(n_failed),
        "tasks": ["coord_number(MND)", "metal_wiberg_bo_total"],
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "AdamW",
        "n_params": int(n_params),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": best_val,
        "history": history,
    }
    out = Path(args.out)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "mpnn_config": asdict(cfg),
            "target_mean": mean.cpu(),
            "target_std": std.cpu(),
            "meta": meta,
        },
        out,
    )
    print(f"[pretrain] saved checkpoint -> {out}")

    json_out = REPORTS_DIR / "f2_pretrain_history.json"
    json_out.write_text(json.dumps(meta, indent=2))
    print(f"[pretrain] wrote {json_out}")
    print(
        f"[pretrain] FINAL train_loss={history[-1]['train_loss']:.4f} "
        f"val_loss={history[-1]['val_loss']:.4f} "
        f"val_MAE_CN={history[-1]['val_mae_coord']:.3f} "
        f"val_MAE_BO={history[-1]['val_mae_bo']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
